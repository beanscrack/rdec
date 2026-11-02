//! T2: the differential-execution gate.
//!
//! `rdec c` proves the emitted C *compiles*. That is a weak claim: C that
//! compiles can compute anything. T2 runs both sides — the original machine
//! code (the same `.obj` the decompiler read, linked into a driver) and the
//! decompiled C (the emitter's own text) — on the same argument vectors, and
//! compares the 64-bit value each side leaves in RAX.
//!
//! Four rules make the comparison mean something:
//!
//!  * **The same inputs, by construction.** Both harnesses embed one Rust array
//!    of argument vectors, so "same inputs" is a property of the generator, not
//!    a claim about two files that look alike.
//!  * **Nothing is compared that has no defined answer.** A function whose
//!    machine code reaches `ret` with RAX unwritten, reads memory through a
//!    parameter, calls a symbol we do not model, ends in `Exit`, or has no COFF
//!    symbol to link against is *not run* — counted, named, and kept out of the
//!    match column. Comparing an undefined value manufactures a mismatch that
//!    is noise, and a gate full of noise gets ignored.
//!  * **A crash is a bucket, not a verdict.** Each runner prints one flushed
//!    line per job; the parent restarts after the job that never reported and
//!    files it under faults. A divide-by-zero in one function cannot erase the
//!    other seventy measurements, and cannot masquerade as a match either.
//!  * **The comparison width is RAX, and the C side is widened from its own
//!    declared return type.** A 32-bit register write zeroes bits 63:32 of RAX
//!    architecturally, so a function whose answer reaches RAX through one has an
//!    answer that is exactly `zero_extend_32(value)` — the upper half is a fact,
//!    not undefined. The lifter records that width and the emitted C returns
//!    `uint32_t`; the harness calls it as `uint32_t` and widens the value in C,
//!    so both sides' 64 bits are compared with nothing guessed at. (The earlier
//!    claim here — "the Win64 ABI leaves the upper halves undefined for a 32-bit
//!    return, so reading the full RAX of a `uint64_t` call is the honest thing" —
//!    was the emitter's over-extension wearing the ABI's clothes: it made 39 of
//!    732 runs disagree on bits the machine code had already cleared.)
//!
//! Loop-shaped functions get small arguments only: a 2^31-trip loop is a
//! timeout, not a test. Loop-free functions additionally get the width and
//! signedness bit patterns (0x80000000, -1, the sign bit), which is where a
//! wrong sign extension shows up.
use crate::m4::{self, Subject};
use rdec_core::emit_c::{emit_c, PRELUDE};
use rdec_core::ir::{BinOp, CastKind, Function, Op, Terminator};
use rdec_core::ty::{Sx, Ty, Width};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Per-line patience: a runner that has not printed for this long is stuck.
const PER_LINE: Duration = Duration::from_secs(20);
/// Per-process patience, so a fast-printing loop cannot hold the gate open.
const RUN_BUDGET: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// The IR questions that decide whether a function has a comparable answer.
// ---------------------------------------------------------------------------

fn def_op(f: &Function, v: u32) -> Option<&Op> {
    let (bb, idx) = f.def(v)?;
    f.block(bb).instrs.get(idx).map(|i| &i.op)
}

/// Every value that flows into `root`, transitively.
///
/// A phi's incoming value from a block no path reaches contributes no value, so
/// following it would call a function "undefined" for a phi slot that can never
/// be taken. That refinement is the difference between naming a real gap and
/// filtering a working function: the first T2 run filtered `loop_sum` this way.
fn ancestors(f: &Function, root: u32) -> HashSet<u32> {
    let reach = reachable(f);
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        if let Some(op) = def_op(f, v) {
            match op {
                Op::Phi(pairs) => stack.extend(
                    pairs
                        .iter()
                        .filter(|(_, p)| reach.contains(p))
                        .map(|(v, _)| *v),
                ),
                _ => {
                    let mut u = Vec::new();
                    op.uses(&mut u);
                    stack.extend(u);
                }
            }
        }
    }
    seen
}

/// Blocks a path from the entry can reach.
fn reachable(f: &Function) -> HashSet<u32> {
    let mut reach = HashSet::new();
    let mut stack = vec![f.entry];
    let mut succ = Vec::new();
    while let Some(b) = stack.pop() {
        if !reach.insert(b) {
            continue;
        }
        f.block(b).term.successors(&mut succ);
        stack.extend(succ.iter().copied());
    }
    reach
}

/// An address the decompiler can name at compile time: a constant, or
/// arithmetic on constants. With the lifter's fixed entry RSP, every
/// `rsp + disp` frame slot of a `-O0` function folds to one of these.
///
/// Anything else — a pointer from a caller, a pointer reloaded from a frame
/// slot — names an address the two sides do not share: the C model masks into
/// `rdec_mem`, the machine code uses the real address space.
fn address_is_static(f: &Function, addr: u32) -> bool {
    ancestors(f, addr).iter().all(|&v| {
        matches!(
            def_op(f, v),
            Some(Op::Const(_)) | Some(Op::Bin(..)) | Some(Op::Cast(..)) | Some(Op::Un(..))
        )
    })
}

/// The function touches memory at an address that is not its modelled frame.
///
/// There is no address both sides agree on, so this subset is not run: the C
/// model masks every address into `rdec_mem`, the machine code dereferences the
/// real pointer. `carry_chain` is the case that proves the rule is needed — its
/// `*lo = l` stores through a pointer *reloaded from its own frame slot*, so a
/// taint check that stops at the load lets it through, and the native side then
/// takes an access fault on every argument vector.
fn touches_memory_outside_the_frame(f: &Function) -> bool {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| match &i.op {
            Op::Load(_, a) => !address_is_static(f, *a),
            Op::Store(_, a, _) => !address_is_static(f, *a),
            _ => false,
        })
}

/// A backward edge in the lifted CFG: the function can loop.
fn loop_shaped(f: &Function) -> bool {
    let mut succ = Vec::new();
    f.blocks.iter().any(|b| {
        b.term.successors(&mut succ);
        succ.iter().any(|&s| f.block(s).addr <= b.addr)
    })
}

/// The returned value is (partly) `Undef`: the machine code reaches `ret` with
/// RAX unwritten on some path. Whatever the real chip leaves there is not a
/// value, and the C model's `0` is not the same non-value.
fn returns_undef(f: &Function) -> bool {
    f.blocks.iter().any(|b| match &b.term {
        Terminator::Return(v) => ancestors(f, *v)
            .iter()
            .any(|&x| matches!(def_op(f, x), Some(Op::Undef))),
        _ => false,
    })
}

fn has_exit(f: &Function) -> bool {
    f.blocks.iter().any(|b| matches!(b.term, Terminator::Exit))
}

/// Discovery's synthetic name for code no symbol names. It is not a link name:
/// the object has nothing to resolve, so there is no machine-code side to run.
fn synthetic(sym: &str, start: u64) -> bool {
    sym == format!("fn_{start:x}")
}

// ---------------------------------------------------------------------------
// Inputs.
// ---------------------------------------------------------------------------

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Small arguments: every trip count stays a test rather than a timeout.
const SMALL: [i64; 10] = [0, 1, 2, 3, 7, 10, 17, 40, -1, -100];
/// The width/signedness corners, where a wrong sext/zext shows up.
const STRESS: [u64; 8] = [
    0xffff_ffff,
    0x8000_0000,
    0x7fff_ffff,
    u64::MAX,
    1 << 63,
    0x5555_5555_5555_5555,
    0x0123_4567_89ab_cdef,
    1 << 32,
];

/// The shared argument table. Both harnesses embed this array verbatim, which
/// is what "the same inputs" means here.
fn vectors(n_small: usize, n_stress: usize) -> Vec<[u64; 4]> {
    let mut v: Vec<[u64; 4]> = vec![[0, 1, 2, 3], [1, 0, 0, 0]];
    let mut s = 0x5eed_1234_5678_9abc_u64;
    while v.len() < n_small.max(2) {
        v.push([0; 4].map(|_| SMALL[(splitmix64(&mut s) as usize) % SMALL.len()] as u64));
    }
    v.push([0x8000_0000, 0x8000_0000, 1, 0]);
    v.push([1 << 63, u64::MAX, 1 << 32, 0x7fff_ffff]);
    while v.len() < n_small.max(2) + n_stress {
        v.push([0; 4].map(|_| STRESS[(splitmix64(&mut s) as usize) % STRESS.len()]));
    }
    v
}

fn c_array(vecs: &[[u64; 4]]) -> String {
    let mut s = String::from("static const uint64_t V[][4] = {\n");
    for v in vecs {
        s.push_str(&format!(
            "  {{ {:#018x}ULL, {:#018x}ULL, {:#018x}ULL, {:#018x}ULL }},\n",
            v[0], v[1], v[2], v[3]
        ));
    }
    s.push_str("};\n");
    s
}

/// The body every runner shares: run jobs `--start N ..`, one flushed line each.
///
/// The `fflush` *before* the call is the point: nothing may be pending when a
/// job can crash, or the parent blames the crash on the wrong function.
fn main_body(n_jobs: usize) -> String {
    format!(
        "int main(int argc, char **argv) {{\n\
         \x20 long long start = 0;\n\
         \x20 if (argc >= 3 && strcmp(argv[1], \"--start\") == 0) start = atoll(argv[2]);\n\
         \x20 for (long long j = start; j < (long long){n_jobs}; ++j) {{\n\
         \x20   fflush(stdout);\n\
         \x20   uint64_t r = FN[JOB_SYM[j]](V[JOB_VEC[j]][0], V[JOB_VEC[j]][1],\n\
         \x20                            V[JOB_VEC[j]][2], V[JOB_VEC[j]][3]);\n\
         \x20   printf(\"RDEC %lld %016llx\\n\", j, (unsigned long long)r);\n\
         \x20   fflush(stdout);\n\
         \x20 }}\n\
         \x20 printf(\"RDEC DONE\\n\");\n\
         \x20 fflush(stdout);\n\
         \x20 return 0;\n}}\n"
    )
}

/// The job tables. `jobs[j] = (function index in FN[], vector index)`; the
/// runner reports the LOCAL index `j`, and the parent maps it to a global job
/// id with the same list, so both sides of the gate are keyed the same way.
fn job_tables(jobs: &[(usize, usize)]) -> String {
    let n = jobs.len();
    let col = |k: usize| -> String {
        jobs.iter()
            .enumerate()
            .map(|(i, j)| {
                let v = if k == 0 { j.0 } else { j.1 };
                format!("{v}{}", if i + 1 < n { "," } else { "" })
            })
            .collect()
    };
    format!(
        "static const unsigned JOB_SYM[{n}] = {{{}}};\n\
         static const unsigned JOB_VEC[{n}] = {{{}}};\n",
        col(0),
        col(1)
    )
}

/// The decompiled side's call wrapper, one per function.
///
/// The `(uint64_t)` cast is applied to the CALL EXPRESSION, whose type is the
/// emitted function's own declared return type. That is what "compare the C side
/// widened from its declared return type" means, and it is plain C: a function
/// declared `uint32_t` called as `uint32_t` yields the 32-bit value the C
/// computed, and the cast widens it - which is the same bits a 32-bit machine
/// write leaves in RAX. Reading all 64 bits of a call made through a
/// `uint64_t`-returning pointer is the other claim, and the worse one: the ABI
/// leaves the callee's upper bits unspecified, so the gate would be comparing
/// bits nobody promised.
///
/// The arguments are the ABI's four, cast down to the emitted signature's
/// `int64_t` - spelled here rather than hidden in a function-pointer cast.
fn c_wrapper(idx: usize, cname: &str) -> String {
    format!(
        "static uint64_t w{idx}(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {{\n\
         \x20 return (uint64_t){cname}((int64_t)a, (int64_t)b, (int64_t)c, (int64_t)d);\n}}\n"
    )
}

/// Does this function's emitted C declare a 32-bit return, i.e. did the lifter
/// record that its answer reaches `RAX` through a 32-bit write? Counted so the
/// width modelling is a number the gate prints, not a claim in a comment.
fn declares_32_bit_return(f: &Function) -> bool {
    matches!(
        f.ret_ty,
        Ty::Int {
            w: Width::W32,
            sx: Sx::Unsigned
        }
    )
}

// ---------------------------------------------------------------------------
// Running a harness.
// ---------------------------------------------------------------------------

struct Run {
    /// Global job id -> the 64-bit value the side returned.
    vals: BTreeMap<usize, u64>,
    /// Global job ids that never reported, with why.
    faults: Vec<(usize, String)>,
    restarts: usize,
}

/// Run `exe`, restarting after any job that does not report.
///
/// The protocol is one flushed line per job, so a crash costs one job, not the
/// run: the parent sees the gap, files it, and asks for the rest.
fn drain(exe: &Path, ids: &[usize]) -> Run {
    let total = ids.len();
    let mut vals = BTreeMap::new();
    let mut faults: Vec<(usize, String)> = Vec::new();
    let mut next = 0usize;
    let mut restarts = 0usize;
    while next < total {
        let spawned = Command::new(exe)
            .arg("--start")
            .arg(next.to_string())
            .stdout(Stdio::piped())
            // To the void: a stderr pipe nobody reads can block the child.
            .stderr(Stdio::null())
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                eprintln!("t2: cannot run {}: {e}", exe.display());
                faults.extend((next..total).map(|i| (ids[i], "spawn failed".to_string())));
                break;
            }
        };
        let out = child.stdout.take().expect("stdout is piped");
        let (tx, rx) = mpsc::channel::<String>();
        let reader = thread::spawn(move || {
            let mut r = BufReader::new(out);
            let mut l = String::new();
            loop {
                l.clear();
                match r.read_line(&mut l) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(l.trim_end().to_string()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        let mut done = false;
        let mut timed_out = false;
        // Every job below `last` has reported; `last` is the one to blame if the
        // process stops answering.
        let mut last = next;
        let t0 = Instant::now();
        loop {
            if t0.elapsed() > RUN_BUDGET {
                timed_out = true;
                break;
            }
            match rx.recv_timeout(PER_LINE) {
                Ok(line) => {
                    let Some(rest) = line.strip_prefix("RDEC ") else {
                        continue; // printf from the machine code under test
                    };
                    let rest = rest.trim();
                    if rest == "DONE" {
                        done = true;
                        break;
                    }
                    let mut it = rest.split_whitespace();
                    let (Some(i), Some(v)) = (
                        it.next().and_then(|s| s.parse::<usize>().ok()),
                        it.next().and_then(|s| u64::from_str_radix(s, 16).ok()),
                    ) else {
                        continue;
                    };
                    if i < total {
                        vals.insert(ids[i], v);
                        last = last.max(i + 1);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    timed_out = true;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        if !done {
            let _ = child.kill();
        }
        let status = child.wait().ok();
        let _ = reader.join();
        if done {
            break;
        }
        let stuck = last.max(next);
        let why = if timed_out {
            "timeout".to_string()
        } else {
            match status.and_then(|s| s.code()) {
                // 0xc0000094 is a divide by zero, 0xc0000005 an access fault:
                // the machine code, driven with arguments its source never saw.
                Some(c) => format!("exit {:#010x}", c as u32),
                None => "signal".to_string(),
            }
        };
        if stuck < total {
            faults.push((ids[stuck], why));
        }
        next = stuck + 1;
        restarts += 1;
    }
    Run {
        vals,
        faults,
        restarts,
    }
}

// ---------------------------------------------------------------------------
// Building the harnesses.
// ---------------------------------------------------------------------------

struct Build {
    src: PathBuf,
    exe: PathBuf,
    /// The object whose machine code the driver links against. None is the
    /// decompiled side, which is pure C.
    obj: Option<String>,
}

fn compile(cc: &str, b: &Build) -> Result<(), String> {
    let mut cmd = Command::new(cc);
    cmd.arg("-std=c17")
        .arg("-O0")
        // The same warning set `rdec c` builds its TU with: a harness gets no
        // privileges for being a harness.
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wno-unused-parameter")
        .arg("-Werror")
        .arg(&b.src);
    if let Some(o) = &b.obj {
        cmd.arg(o);
    }
    cmd.arg("-o").arg(&b.exe);
    let out = cmd
        .output()
        .map_err(|e| format!("{}: could not run {cc}: {e}", b.src.display()))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let first = err
        .lines()
        .find(|l| l.contains("error"))
        .unwrap_or("<no error line>");
    Err(format!("{}: clang: {first}", b.src.display()))
}

/// Compile every harness with a small pool: one driver per object means dozens
/// of links, and a serial build is the slowest part of the gate.
fn compile_all(cc: &str, builds: &[Build]) -> Vec<Result<(), String>> {
    let n = builds.len();
    let next = AtomicUsize::new(0);
    let done: Mutex<Vec<Option<Result<(), String>>>> = Mutex::new((0..n).map(|_| None).collect());
    let workers = n
        .min(
            thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
        )
        .max(1);
    thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= n {
                    break;
                }
                let r = compile(cc, &builds[i]);
                done.lock().unwrap()[i] = Some(r);
            });
        }
    });
    let mut v = done.lock().unwrap();
    v.iter_mut()
        .map(|x| x.take().unwrap_or_else(|| Err("not built".to_string())))
        .collect()
}

fn stem_of(obj: &str) -> String {
    let p = Path::new(obj);
    let f = p.file_name().and_then(|s| s.to_str()).unwrap_or("obj");
    m4::c_ident(f.strip_suffix(".obj").unwrap_or(f))
}

/// `corpus\obj\arith_basic.O2.obj` -> `arith_basic`: the snippet, across all
/// four optimisation levels. One hundred mismatch lines are not a work item;
/// the six snippets they fall on are.
fn snippet_of(obj: &str) -> String {
    let f = Path::new(obj)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("obj");
    let s = f.strip_suffix(".obj").unwrap_or(f);
    s.split('.').next().unwrap_or(s).to_string()
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

/// Does this pair differ only in RAX's top 32 bits?
///
/// This is the signature of a WIDTH modelling error, and after the return-width
/// work it should read zero: the lifter records the width the answer reaches
/// RAX at, the emitted C returns that width, and the harness widens the C value
/// from its declared type. A pair that agrees in the low 32 bits and differs
/// above is a function whose recorded width is wrong (the emitter stretched a
/// 32-bit answer to 64, or clipped a 64-bit one), which is why the bucket is
/// kept: it names the class, and the count is the work list.
///
/// It is STILL a mismatch. The mask experiments (compare at 32 bits; infer the
/// width from the observed answers) all ended with a gate that could not fail —
/// a function that returned 0 on every vector read as "1 bit wide", and
/// saturation's `C = 0x7fffffffffffffff` against `machine = 0xffffffff` is not an
/// extension of anything, it is a different number. RAX is compared in full.
fn extension_only(x: u64, y: u64) -> bool {
    (x ^ y) >> 32 != 0 && (x as u32) == (y as u32)
}

/// The ops whose signedness a mismatch is usually about. This is a HINT for the
/// reader, not a proof: it names what the function contains, so 100 mismatch
/// lines become a work item ("signed shift on a value the IR types unsigned")
/// instead of 100 pairs of hex numbers.
fn suspects(f: &Function) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for b in &f.blocks {
        for i in &b.instrs {
            let s = match &i.op {
                Op::Bin(BinOp::Sar, _, _) => "sar",
                Op::Bin(BinOp::Shr, _, _) => "shr",
                Op::Bin(BinOp::SDiv, _, _) => "sdiv",
                Op::Bin(BinOp::SRem, _, _) => "srem",
                Op::Bin(BinOp::RotL, _, _) | Op::Bin(BinOp::RotR, _, _) => "rotate",
                Op::Cast(CastKind::SExt, ..) => "sext",
                _ => continue,
            };
            if !v.contains(&s) {
                v.push(s);
            }
        }
    }
    v.sort_unstable();
    v
}

/// One function T2 will run, on both sides.
struct Sub<'a> {
    s: &'a Subject,
    /// The emitted C text — `rdec c`'s text, same emitter, same names.
    text: String,
    /// Index into the decompiled side's function table.
    c_sym: usize,
    loops: bool,
}

/// Zero-extend every argument to 32 bits.
///
/// This is the shape `--args32` measures and is NOT a fix: the Win64 ABI leaves
/// the top halves of RCX/RDX/R8/R9 undefined for a caller passing `int`, and a
/// clang caller of `f(7, -1)` emits `mov ecx, 7`, which zeroes them. Running
/// both ways is the measurement — the difference between the two counts is the
/// number of functions whose decompilation depends on bits the machine code
/// never reads.
fn zero_extend_32(v: &[[u64; 4]]) -> Vec<[u64; 4]> {
    v.iter().map(|q| q.map(|x| x & 0xffff_ffff)).collect()
}

pub fn cmd_t2(args: &[String]) -> Result<i32, String> {
    let mut files: Vec<String> = Vec::new();
    let mut out_dir = "corpus/t2".to_string();
    let mut limit = usize::MAX;
    let mut n_vec = 12usize;
    let mut args32 = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                let man = std::fs::read_to_string("corpus/manifest.tsv")
                    .map_err(|e| format!("corpus/manifest.tsv: {e}"))?;
                for line in man.lines() {
                    if line.starts_with('#') || line.starts_with("name\t") {
                        continue;
                    }
                    if let Some(o) = line.split('\t').nth(2) {
                        files.push(o.to_string());
                    }
                }
            }
            "--out" if i + 1 < args.len() => {
                out_dir = args[i + 1].clone();
                i += 1;
            }
            "--limit" if i + 1 < args.len() => {
                limit = args[i + 1].parse().unwrap_or(usize::MAX);
                i += 1;
            }
            "--vectors" if i + 1 < args.len() => {
                n_vec = args[i + 1].parse().unwrap_or(12).max(4);
                i += 1;
            }
            "--args32" => args32 = true,
            other => {
                if !other.starts_with("--") {
                    files.push(other.to_string());
                }
            }
        }
        i += 1;
    }
    if files.is_empty() {
        return Err(
            "usage: rdec t2 <object.obj> [...] [--corpus] [--out DIR] [--limit N] [--vectors N]"
                .to_string(),
        );
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    // One pipeline, one naming scheme: this is `rdec c`'s own front half.
    let subjects = m4::lift_all(&files)?;
    let mut lifted = 0usize;
    let mut refused = 0usize;
    let mut broken = 0usize;
    let mut undef_dumps = 0usize;
    // reason -> the functions it names. A skipped function is counted AND
    // named, so a shrunk comparison set reads as a work item, not as a win.
    let mut skipped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut subs: Vec<Sub> = Vec::new();

    for s in &subjects {
        lifted += 1;
        let tag = format!("{} :: {}", s.obj, s.sym);
        let skip = |why: &str, skipped: &mut BTreeMap<String, Vec<String>>| {
            skipped
                .entry(why.to_string())
                .or_default()
                .push(tag.clone());
        };
        if rdec_core::verify::verify_ssa(&s.f).is_err() {
            broken += 1;
            continue;
        }
        let Ok(out) = emit_c(&s.f) else {
            // `rdec c` counts and names these; T2 does not re-count an emitter
            // refusal as a T2 gap.
            refused += 1;
            continue;
        };
        if !out.externs.is_empty() {
            skip("calls a symbol we do not model", &mut skipped);
            continue;
        }
        if has_exit(&s.f) {
            skip("unmodelled control flow (Exit)", &mut skipped);
            continue;
        }
        if synthetic(&s.sym, s.start) {
            skip("no COFF symbol to link (discovery target)", &mut skipped);
            continue;
        }
        if !s.external {
            // `static int h(int)` compiles to a file-static COFF symbol: the
            // decompiler lifts it, the linker refuses to name it, and the
            // driver build dies with `undefined symbol: h`.
            skip(
                "symbol is file-static (not linkable from a driver)",
                &mut skipped,
            );
            continue;
        }
        if touches_memory_outside_the_frame(&s.f) {
            skip("memory outside the modelled frame", &mut skipped);
            continue;
        }
        if returns_undef(&s.f) {
            // The post-pass IR is the diagnosis: this bucket is only trustworthy
            // if a reader can check that the undef really reaches the return, so
            // the first few are written out, exactly as `rdec c` dumps refusals.
            if undef_dumps < 3 {
                undef_dumps += 1;
                let dump = format!("{out_dir}/undef_{undef_dumps}.ir");
                let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&s.f));
                println!("  undef dump {dump}: {}", tag);
            }
            skip("returns RAX unwritten on some path", &mut skipped);
            continue;
        }
        if subs.len() >= limit {
            skip("--limit reached", &mut skipped);
            continue;
        }
        subs.push(Sub {
            s,
            // `rdec c`'s own emitter output, verbatim, in one TU with the
            // prelude's memory model.
            text: out.text.clone(),
            c_sym: subs.len(),
            loops: loop_shaped(&s.f),
        });
    }
    if subs.is_empty() {
        println!("t2: {lifted} lifted, {refused} refused by the emitter, 0 functions T2 can run");
        for (why, names) in &skipped {
            for n in names.iter().take(3) {
                println!("  not run [{why}] {n}");
            }
        }
        println!("FAIL: nothing comparable in {} objects", files.len());
        return Ok(1);
    }

    let raw = vectors(n_vec, n_vec);
    let vecs = if args32 {
        zero_extend_32(&raw)
    } else {
        raw.clone()
    };
    if args32 {
        println!("args: every argument zero-extended to 32 bits (--args32): the shape a clang caller of `f(int,int)` leaves in RCX/RDX/R8/R9");
    }
    // Jobs are (function, vector), grouped by function, so each object's jobs
    // are contiguous and one crash costs one restart.
    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for (k, sb) in subs.iter().enumerate() {
        // A loop-shaped function only takes the small vectors: see the header.
        let n = if sb.loops { n_vec } else { vecs.len() };
        for v in 0..n {
            jobs.push((k, v));
        }
    }
    let loop_free = subs.iter().filter(|s| !s.loops).count();
    let skipped_n: usize = skipped.values().map(|v| v.len()).sum();
    // The width modelling, as a number: how many of the functions about to be
    // compared declare a 32-bit return because their answer reaches RAX through
    // a 32-bit write (which zeroes bits 63:32), versus the 64-bit default.
    let ret32 = subs
        .iter()
        .filter(|s| declares_32_bit_return(&s.s.f))
        .count();

    // --- the decompiled side: the emitter's own text, plus a table ---
    let mut decomp = String::from(
        "/* rdec t2: the decompiled side. Every function below is the text\n\
         \x20  `rdec c` writes into corpus/c/all.c, from the same emitter. */\n\
         #include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n",
    );
    decomp.push_str(PRELUDE);
    decomp.push('\n');
    for sb in &subs {
        decomp.push_str(&sb.text);
        decomp.push('\n');
    }
    decomp.push_str("typedef uint64_t (*Fn)(uint64_t, uint64_t, uint64_t, uint64_t);\n");
    for sb in &subs {
        decomp.push_str(&c_wrapper(sb.c_sym, &sb.s.cname));
    }
    decomp.push_str("static const Fn FN[] = {\n");
    for sb in &subs {
        decomp.push_str(&format!("  w{},\n", sb.c_sym));
    }
    decomp.push_str("};\n");
    decomp.push_str(&c_array(&vecs));
    // The decompiled side runs every job in job order, so its job list is the
    // global one: that is what makes the two sides' job lists the same list.
    decomp.push_str(&job_tables(&jobs));
    decomp.push_str(&main_body(jobs.len()));
    let decomp_src = PathBuf::from(format!("{out_dir}/t2_decomp.c"));
    let decomp_exe = PathBuf::from(format!("{out_dir}/t2_decomp.exe"));
    std::fs::write(&decomp_src, &decomp).map_err(|e| format!("{}: {e}", decomp_src.display()))?;

    // --- the machine-code side: one driver per object ---
    // 120 objects all define `f`, so no single driver can link them: one
    // process per object is the price of running the real code.
    let mut order: Vec<String> = Vec::new();
    let mut by_obj: HashMap<String, Vec<usize>> = HashMap::new();
    for (k, sb) in subs.iter().enumerate() {
        if !by_obj.contains_key(&sb.s.obj) {
            order.push(sb.s.obj.clone());
        }
        by_obj.entry(sb.s.obj.clone()).or_default().push(k);
    }

    let mut builds = vec![Build {
        src: decomp_src.clone(),
        exe: decomp_exe.clone(),
        obj: None,
    }];
    let mut drivers: Vec<(String, PathBuf, Vec<usize>)> = Vec::new();
    for obj in &order {
        let ks = &by_obj[obj];
        let mut syms: Vec<String> = Vec::new();
        for &k in ks {
            let sym = subs[k].s.sym.clone();
            if !syms.contains(&sym) {
                syms.push(sym);
            }
        }
        let mut src = format!(
            "/* rdec t2: the machine-code side of {obj}.\n\
             \x20  The real .obj is linked in; the externs are its own symbols. */\n\
             #include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        );
        for sym in &syms {
            // The declared types are the ABI's, not the source's: the object
            // reads ECX of a Win64 first argument, and ECX is the low 32 bits of
            // the 64-bit register the driver passes — the same read the lifter
            // models when it lifts `mov dword ptr [rbp-4], ecx`.
            src.push_str(&format!(
                "extern uint64_t {sym}(uint64_t, uint64_t, uint64_t, uint64_t);\n"
            ));
        }
        src.push_str("typedef uint64_t (*Fn)(uint64_t, uint64_t, uint64_t, uint64_t);\n");
        src.push_str("static const Fn FN[] = {\n");
        for sym in &syms {
            src.push_str(&format!("  {sym},\n"));
        }
        src.push_str("};\n");
        // This driver's jobs, in global order: (symbol index, vector), plus the
        // global job id it reports, so the parent can merge the two sides.
        let mut local: Vec<(usize, usize)> = Vec::new();
        let mut ids: Vec<usize> = Vec::new();
        for (g, (k, v)) in jobs.iter().enumerate() {
            if ks.contains(k) {
                local.push((ks.iter().position(|kk| kk == k).unwrap(), *v));
                ids.push(g);
            }
        }
        src.push_str(&c_array(&vecs));
        src.push_str(&job_tables(&local));
        src.push_str(&main_body(local.len()));
        let sp = PathBuf::from(format!("{out_dir}/t2_native_{}.c", stem_of(obj)));
        let ep = PathBuf::from(format!("{out_dir}/t2_native_{}.exe", stem_of(obj)));
        std::fs::write(&sp, &src).map_err(|e| format!("{}: {e}", sp.display()))?;
        builds.push(Build {
            src: sp,
            exe: ep.clone(),
            obj: Some(obj.clone()),
        });
        drivers.push((obj.clone(), ep, ids));
    }

    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let t0 = Instant::now();
    let results = compile_all(&cc, &builds);
    let build_failed: Vec<String> = results
        .iter()
        .filter_map(|r| r.as_ref().err().cloned())
        .collect();
    let built = results.iter().filter(|r| r.is_ok()).count();

    println!(
        "t2: {lifted} lifted, {refused} refused by the emitter (not run), {broken} invalid after passes"
    );
    println!(
        "runnable: {} functions on both sides, {skipped_n} not runnable ({})",
        subs.len(),
        skipped
            .iter()
            .map(|(k, v)| format!("{} {k}", v.len()))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!(
        "inputs: {n_vec} small vectors for every function, {} for the {loop_free} loop-free ones -> {} comparisons",
        vecs.len(),
        jobs.len()
    );
    println!(
        "return width: {ret32}/{} of the runnable functions declare uint32_t (their answer reaches RAX through a 32-bit write, which zeroes bits 63:32), the rest int64_t",
        subs.len()
    );
    println!(
        "build: {built}/{} harnesses with {cc} -Werror in {:?}, {} failures",
        builds.len(),
        t0.elapsed(),
        build_failed.len()
    );
    for f in build_failed.iter().take(5) {
        println!("  BUILD FAIL {f}");
    }
    if !build_failed.is_empty() {
        println!("FAIL: a harness did not compile");
        return Ok(1);
    }

    // --- run both sides ---
    let all_ids: Vec<usize> = (0..jobs.len()).collect();
    let crun = drain(&decomp_exe, &all_ids);
    let mut native: BTreeMap<usize, u64> = BTreeMap::new();
    let mut native_faults: Vec<(usize, String)> = Vec::new();
    let mut native_restarts = 0usize;
    for (obj, exe, ids) in &drivers {
        let r = drain(exe, ids);
        native.extend(r.vals);
        native_restarts += r.restarts;
        for (g, why) in r.faults {
            native_faults.push((g, format!("{obj}: {why}")));
        }
    }

    println!(
        "compare: the full 64-bit RAX, every vector, on both sides; the C side widened from its declared return type"
    );

    let mut match_n = 0usize;
    let mut mismatch: Vec<String> = Vec::new();
    let mut not_compared = 0usize;
    let mut above_bit31 = 0usize;
    // snippet -> (mismatches, comparisons)
    let mut by_snippet: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    // IR shape signature -> how many mismatching functions carry it
    let mut opsig: BTreeMap<String, usize> = BTreeMap::new();
    let mut shaped: HashSet<usize> = HashSet::new();
    for (g, (k, v)) in jobs.iter().enumerate() {
        let row = by_snippet
            .entry(snippet_of(&subs[*k].s.obj))
            .or_insert([0, 0]);
        let (Some(&x), Some(&y)) = (native.get(&g), crun.vals.get(&g)) else {
            // One side faulted here: no verdict, and not a match either.
            not_compared += 1;
            continue;
        };
        row[1] += 1;
        if x == y {
            match_n += 1;
        } else {
            if extension_only(x, y) {
                // Its own bucket: a real mismatch, with a name.
                above_bit31 += 1;
            }
            row[0] += 1;
            if shaped.insert(*k) {
                *opsig.entry(suspects(&subs[*k].s.f).join("+")).or_default() += 1;
            }
            mismatch.push(format!(
                "{} :: {}  args=[{:#x},{:#x},{:#x},{:#x}]  machine={:#018x}  C={:#018x}",
                subs[*k].s.obj,
                subs[*k].s.sym,
                vecs[*v][0],
                vecs[*v][1],
                vecs[*v][2],
                vecs[*v][3],
                x,
                y
            ));
        }
    }
    let compared = match_n + mismatch.len();
    let fault_jobs: BTreeSet<usize> = crun
        .faults
        .iter()
        .map(|(g, _)| *g)
        .chain(native_faults.iter().map(|(g, _)| *g))
        .collect();

    println!(
        "runs: {} native drivers ({} restarts after faults), 1 decompiled runner ({} restarts)",
        drivers.len(),
        native_restarts,
        crun.restarts
    );
    println!(
        "compared: {compared} | match: {match_n} | mismatch: {} | not compared (one side faulted): \
         {not_compared} | of the mismatches, differing only above bit 31: {above_bit31}",
        mismatch.len()
    );
    if !by_snippet.is_empty() {
        // Grouped by snippet across all four -O levels: 100 mismatch lines are
        // noise, the six snippets they land on are the work list.
        let rows: Vec<String> = by_snippet
            .iter()
            .filter(|(_, r)| r[0] > 0)
            .map(|(k, r)| format!("{k} {}/{}", r[0], r[1]))
            .collect();
        if !rows.is_empty() {
            println!("mismatch by snippet: {}", rows.join(" | "));
        }
        // A hint, not a proof: the signedness-shaped ops each mismatching
        // function contains, so the first fix has a target.
        let mut ops: Vec<(String, usize)> = opsig.into_iter().collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let top: Vec<String> = ops
            .iter()
            .take(4)
            .map(|(k, n)| {
                format!(
                    "{n} [{}]",
                    if k.is_empty() {
                        "no sar/shr/sdiv/srem/rotate/sext"
                    } else {
                        k
                    }
                )
            })
            .collect();
        println!(
            "mismatching functions by signedness-shaped ops: {}",
            top.join(" | ")
        );
    }
    for m in mismatch.iter().take(10) {
        println!("  MISMATCH {m}");
    }
    if mismatch.len() > 10 {
        println!("  ... and {} more", mismatch.len() - 10);
    }
    if !fault_jobs.is_empty() {
        // Faults are their own bucket: a function that crashes under the
        // harness is a coverage gap, never a match and never a mismatch.
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for (g, why) in native_faults.iter().chain(crun.faults.iter()) {
            let (k, v) = jobs[*g];
            let who = format!("{} :: {} vec {v}", subs[k].s.obj, subs[k].s.sym);
            if seen.insert(who.clone()) {
                println!("  FAULT {who}: {why}");
            }
            if seen.len() >= 10 {
                break;
            }
        }
        println!(
            "note: {} job(s) faulted; they are NOT counted as matches",
            fault_jobs.len()
        );
    }
    for (why, names) in &skipped {
        for n in names.iter().take(3) {
            println!("  not run [{why}] {n}");
        }
    }

    if compared == 0 {
        println!("FAIL: nothing was compared");
        return Ok(1);
    }
    if !mismatch.is_empty() {
        println!(
            "FAIL: the decompiled C disagrees with the machine code on {} of {compared} runs",
            mismatch.len()
        );
        return Ok(1);
    }
    println!(
        "OK: the decompiled C computes the same 64-bit result as the machine code on all {compared} runs"
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdec_core::ir::{BinOp, Function, Op, Terminator};
    use rdec_core::ty::Ty;

    #[test]
    fn a_load_through_a_caller_pointer_is_not_comparable_and_a_frame_slot_is() {
        // The frame rule, on the two shapes that decide it: `*a` (an address
        // from the caller) and `*(STACK+16)` (a slot of the modelled frame).
        let mut f = Function::new("p", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(a);
        let v = f.emit(e, Op::Load(Ty::u64(), a), Ty::u64(), 0);
        f.block_mut(e).term = Terminator::Return(v);
        assert!(
            touches_memory_outside_the_frame(&f),
            "*a has no address both sides share"
        );

        let mut g = Function::new("s", 0);
        let e = g.entry;
        let sp = g.emit(e, Op::Const(1 << 19), Ty::u64(), 0);
        let d = g.emit(e, Op::Const(16), Ty::u64(), 0);
        let slot = g.emit(e, Op::Bin(BinOp::Add, sp, d), Ty::u64(), 0);
        let v = g.emit(e, Op::Load(Ty::u64(), slot), Ty::u64(), 0);
        g.block_mut(e).term = Terminator::Return(v);
        assert!(
            !touches_memory_outside_the_frame(&g),
            "a constant frame slot is the same bytes on both sides"
        );
    }

    #[test]
    fn a_pointer_reloaded_from_a_frame_slot_is_not_comparable() {
        // carry_chain's exact shape: `*p = x` where p was spilled and reloaded.
        // The store's address comes from a load, not from a parameter, so a
        // taint check that stops at the load lets it through — and the native
        // side then faults on every argument vector.
        let mut f = Function::new("cc", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(p);
        let slot = f.emit(e, Op::Const(1 << 19), Ty::u64(), 0);
        f.emit_void(e, Op::Store(Ty::u64(), slot, p), 0);
        let reload = f.emit(e, Op::Load(Ty::u64(), slot), Ty::u64(), 0);
        let one = f.emit(e, Op::Const(1), Ty::u64(), 0);
        f.emit_void(e, Op::Store(Ty::u64(), reload, one), 0);
        f.block_mut(e).term = Terminator::Return(one);
        assert!(
            touches_memory_outside_the_frame(&f),
            "the address of `*p = x` came out of memory, not out of the frame"
        );
    }

    #[test]
    fn a_backward_edge_is_a_loop_and_a_straight_line_is_not() {
        let mut f = Function::new("l", 0);
        let a = f.new_block(0x10);
        let b = f.new_block(0x20);
        f.block_mut(a).term = Terminator::Jump(b);
        f.block_mut(b).term = Terminator::Jump(a);
        assert!(loop_shaped(&f), "0x20 -> 0x10 is a backward edge");

        let mut g = Function::new("s", 0);
        let x = g.emit(g.entry, Op::Const(7), Ty::u64(), 0);
        g.block_mut(g.entry).term = Terminator::Return(x);
        assert!(!loop_shaped(&g), "no edges, no loop");
    }

    #[test]
    fn an_unwritten_return_value_is_not_comparable() {
        let mut f = Function::new("u", 0);
        let e = f.entry;
        let u = f.emit(e, Op::Undef, Ty::u64(), 0);
        f.block_mut(e).term = Terminator::Return(u);
        assert!(returns_undef(&f), "ret with RAX unwritten has no answer");

        let mut g = Function::new("w", 0);
        let e = g.entry;
        let c = g.emit(e, Op::Const(7), Ty::u64(), 0);
        g.block_mut(e).term = Terminator::Return(c);
        assert!(!returns_undef(&g), "a written value is comparable");
    }

    #[test]
    fn an_unreachable_phi_slot_does_not_make_a_return_undefined() {
        // The refinement that keeps `loop_sum` runnable: a phi with one real
        // incoming edge and one `undef` edge from a block no path reaches. The
        // undef cannot flow to the return, so the answer is defined.
        use rdec_core::ir::Instr;
        let mut f = Function::new("p", 0);
        let e = f.entry;
        let b = f.new_block(0x10);
        let dead = f.new_block(0x900);
        let c = f.emit(e, Op::Const(1), Ty::u64(), 0);
        let u = f.emit(e, Op::Undef, Ty::u64(), 0);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: b,
            else_bb: b,
        };
        let phi = f.new_value(Ty::u64());
        f.block_mut(b).instrs.insert(
            0,
            Instr::new(
                phi,
                Op::Phi(vec![(c, e), (u, dead)].into_boxed_slice()),
                Ty::u64(),
                0x10,
            ),
        );
        f.block_mut(b).term = Terminator::Return(phi);
        assert!(
            !returns_undef(&f),
            "an undef on an edge nothing reaches is not a gap"
        );

        // The same phi with the undef on a REACHED edge is a real gap.
        f.block_mut(b).term = Terminator::Return(phi);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: b,
            else_bb: dead,
        };
        f.block_mut(dead).term = Terminator::Jump(b);
        assert!(returns_undef(&f), "this edge is reachable, so it carries 0");
    }

    #[test]
    fn the_input_table_is_deterministic_and_covers_the_sign_bit() {
        // Both harnesses embed this array, so a changing generator changes both
        // sides at once — the one property the comparison cannot do without.
        assert_eq!(vectors(8, 8), vectors(8, 8));
        let v = vectors(8, 8);
        assert_eq!(v.len(), 16);
        assert!(
            v.iter().flatten().any(|&x| x == 1 << 63),
            "the sign bit must be in the table"
        );
        assert!(
            v[2..8]
                .iter()
                .all(|q| q.iter().all(|&x| (x as i64).abs() <= 100)),
            "the small vectors keep every trip count small"
        );
    }

    #[test]
    fn a_synthetic_discovery_name_is_not_a_link_name() {
        assert!(synthetic("fn_3b", 0x3b));
        assert!(!synthetic("f", 0));
        assert!(
            !synthetic("fn_3b", 0x4),
            "only the name discovery actually invented is synthetic"
        );
    }

    /// The widening is a C cast on the call expression, so the type that decides
    /// it is the emitted function's declared return type: a `uint32_t` function
    /// called as `uint32_t` yields the 32-bit value the C computed and the cast
    /// widens it, which is the same bits a 32-bit machine write leaves in RAX.
    /// The wrapper must not reach RAX through a `uint64_t`-returning pointer -
    /// that compares bits the ABI leaves unspecified.
    #[test]
    fn the_c_side_is_widened_from_its_declared_return_type() {
        let w = c_wrapper(3, "o7_f");
        assert!(w.contains("static uint64_t w3("), "{w}");
        assert!(w.contains("(uint64_t)o7_f(("), "{w}");
        assert!(!w.contains('*'), "no function-pointer cast: {w}");
    }

    /// Which functions the gate counts as 32-bit-returning, pinned on the two
    /// declarations the lifter emits.
    #[test]
    fn a_declared_uint32_return_is_the_counted_width() {
        let mut f = Function::new("w32", 0);
        let c = f.emit(f.entry, Op::Const(7), Ty::u32(), 0);
        f.block_mut(f.entry).term = Terminator::Return(c);
        f.ret_ty = Ty::u32();
        assert!(declares_32_bit_return(&f));

        let mut g = Function::new("w64", 0);
        let d = g.emit(g.entry, Op::Const(7), Ty::u64(), 0);
        g.block_mut(g.entry).term = Terminator::Return(d);
        g.ret_ty = Ty::i64();
        assert!(!declares_32_bit_return(&g));
    }
}
