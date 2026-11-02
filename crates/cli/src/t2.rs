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
use rdec_demangle::Demangled;
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
pub(crate) fn touches_memory_outside_the_frame(f: &Function) -> bool {
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
pub(crate) fn returns_undef(f: &Function) -> bool {
    f.blocks.iter().any(|b| match &b.term {
        Terminator::Return(v) => ancestors(f, *v)
            .iter()
            .any(|&x| matches!(def_op(f, x), Some(Op::Undef))),
        _ => false,
    })
}

pub(crate) fn has_exit(f: &Function) -> bool {
    f.blocks.iter().any(|b| matches!(b.term, Terminator::Exit))
}

/// Per-return-path RAX definedness: `(paths, defined paths, notes)`.
///
/// A path is undefined when the value it returns is (partly) `Undef`, and the
/// note names every instruction that reads the undefined value with the address
/// of the machine instruction it came from - the line a reader can find in
/// `llvm-objdump` and check. A bucket counted but not named is a comparison set
/// that shrank while reading as a win.
///
/// This is a REPORT. T2 runs a function only when EVERY path is defined, so the
/// comparison is per function and never per path: no vector is dropped, no bit
/// is masked, and the count of compared runs is untouched by this function.
pub(crate) fn rax_paths(f: &Function) -> (usize, usize, Vec<String>) {
    let reach = reachable(f);
    let mut paths = 0usize;
    let mut notes = Vec::new();
    for b in &f.blocks {
        if !reach.contains(&b.id) {
            continue;
        }
        let Terminator::Return(v) = b.term else {
            continue;
        };
        paths += 1;
        let anc = ancestors(f, v);
        let mut sites: Vec<String> = Vec::new();
        for &x in &anc {
            if !matches!(def_op(f, x), Some(Op::Undef)) {
                continue;
            }
            for u in f.uses_of(x) {
                sites.push(if u.instr == usize::MAX {
                    format!("0x{:x}: return", f.block(u.bb).addr)
                } else {
                    let i = &f.block(u.bb).instrs[u.instr];
                    format!("0x{:x}: {}", i.addr, rdec_core::fmt_ir::format_instr(f, i))
                });
            }
        }
        sites.sort();
        sites.dedup();
        if !sites.is_empty() {
            notes.push(format!(
                "ret bb{}@0x{:x} reads unwritten: {}",
                b.id,
                b.addr,
                sites.join(" ; ")
            ));
        }
    }
    (paths, paths - notes.len(), notes)
}

/// Discovery's synthetic name for code no symbol names. It is not a link name:
/// the object has nothing to resolve, so there is no machine-code side to run.
pub(crate) fn synthetic(sym: &str, start: u64) -> bool {
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
    // Per-return-path RAX definedness over every lifted function. A report, so
    // the "returns RAX unwritten" bucket can be read as paths, not just names.
    let mut path_tot = 0usize;
    let mut path_def = 0usize;
    let mut undef_notes: Vec<String> = Vec::new();
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
        let (n_paths, n_def, _) = rax_paths(&s.f);
        path_tot += n_paths;
        path_def += n_def;
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
            // Name the paths, not just the function: which return, at what
            // address, and which instruction reads the register that no modelled
            // path writes. The IR dump is the same evidence in full.
            let (p, d, notes) = rax_paths(&s.f);
            undef_notes.push(format!("{tag}: RAX defined on {d} of {p} return paths"));
            for n in notes {
                undef_notes.push(format!("      {n}"));
            }
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
        "paths: RAX defined on {path_def} of {path_tot} return paths over the {lifted} lifted functions; a function is runnable only when EVERY path is defined, so the comparison is per function, never per path"
    );
    for n in &undef_notes {
        println!("  RAX unwritten: {n}");
    }
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
    for m in mismatch.iter().take(200) {
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

// ---------------------------------------------------------------------------
// T2 over C++: `rdec t2 --cpp`.
// ---------------------------------------------------------------------------
//
// `rdec cpp --corpus` proves the emitted C++ C *compiles*. That is the same
// weak claim `rdec c` made before this harness existed: C that compiles can
// compute anything. This gate runs both sides — the real `.obj`, linked into a
// driver through its decorated COFF symbol (declared with an asm-label
// binding: clang refuses `?add@@YAHHH@Z` as an identifier), and the emitter's
// own C text, under `m8::cpp_pipeline`'s names — on the same vectors, and
// compares the full 64-bit RAX each side leaves behind. The rigour is
// `cmd_t2`'s, unchanged: same vectors, same predicates, same crash protocol,
// the C side widened from its declared return type, nothing masked.
//
// Comparison classes, named because they answer different questions:
//
//  * **RAX class** — free functions and static members (no `this`) with
//    builtin-integer args and a builtin-integer return. The Windows x64 ABI
//    puts up to four integer args in RCX/RDX/R8/R9 and answers in RAX, which
//    is exactly the shape the C harness already compares.
//  * **window class** — member functions: the machine side gets a real
//    4 KiB buffer, the C side gets the ADDRESS INTEGER of a window inside
//    `rdec_mem` (0x2000, clear of the frame at STACK_BASE = 1 << 19) preloaded
//    with the same bytes before every call. The RETURN value is what is
//    compared; memory equivalence is NOT claimed — each side mutates its own
//    window and nothing compares the windows afterwards.
//  * **XMM0 class** — the honest float/double comparison (bitwise on XMM0's
//    raw bits, so -0.0 vs 0.0 and NaN payloads are compared as bits, which is
//    what the machine answers). The count is 0 and stays 0 until the lifter
//    models it: ARG_REGS is RCX/RDX/R8/R9, so a float argument is an undef
//    live-in (refused), and `ret` with XMM0 written and RAX unwritten lands in
//    "returns RAX unwritten" — the emitted C returns the integer RAX. The
//    gate prints the class line with its real count so the gap is a number,
//    not a comment; a comparison over a set the lifter cannot express is a
//    gate over nothing.
//
// Accounting closes: every lifted function lands in exactly one bucket
// (compared+match, compared+mismatch, faulted, excluded-with-a-name, refused,
// broken), and the buckets' sum is printed against the lifted count.

/// The demangled text's signature: (return type, parameters). `"int add(int,
/// int)"` -> `("int", ["int", "int"])`, `"int& Buffer::slot(int)"` ->
/// `("int&", ["int"])`. The demangler spells parameters without names, so a
/// plain comma split is the whole parse for the supported set.
fn split_signature(text: &str) -> Option<(String, Vec<String>)> {
    let open = text.find('(')?;
    let close = text.rfind(')')?;
    if close < open {
        return None;
    }
    let head = text[..open].trim_end();
    let params_s = text[open + 1..close].trim();
    let qual = head.split_whitespace().last().unwrap_or(head);
    let ret = head[..head.len() - qual.len()].trim();
    let params = if params_s.is_empty() || params_s == "void" {
        Vec::new()
    } else {
        params_s.split(',').map(|p| p.trim().to_string()).collect()
    };
    Some((ret.to_string(), params))
}

/// The builtin integer types, as the demangler spells them (MSVC: `long` is
/// 32-bit, `__int64` is 64). Anything outside this list is the type-model
/// gap, and the exclusion names the type.
const INT_TYPES: &[&str] = &[
    "bool",
    "char",
    "signed char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "unsigned",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "__int64",
    "unsigned __int64",
    "wchar_t",
];

fn is_int_type(t: &str) -> bool {
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    INT_TYPES.contains(&t.as_str())
}

fn is_float_type(t: &str) -> bool {
    matches!(t.trim(), "float" | "double" | "long double")
}

fn takes_float(params: &[String]) -> Option<String> {
    params
        .iter()
        .find(|p| is_float_type(p))
        .cloned()
        .or_else(|| params.iter().find(|p| p.contains("...")).cloned())
}

fn takes_pointerish(params: &[String]) -> Option<String> {
    params
        .iter()
        .find(|p| {
            let p = p.trim();
            p.contains('*')
                || p.contains('&')
                || p.starts_with("struct ")
                || p.starts_with("class ")
                || p.starts_with("union ")
                || p.starts_with("enum ")
        })
        .cloned()
}

/// Why this signature is outside a class's comparison, `None` when it is
/// inside. `max_params` is the register budget: 4 for the RAX class
/// (RCX/RDX/R8/R9), 3 for the window class (`this` takes RCX).
///
/// The float exclusions name the XMM0 class explicitly: the honest class
/// exists (compare XMM0's raw bits), the lifter cannot express its inputs
/// yet, and the reader should see that sentence, not infer it.
fn signature_gap(text: &str, max_params: usize) -> Option<String> {
    let (ret, params) = match split_signature(text) {
        Some(s) => s,
        None => return Some("demangled text carries no parseable signature".to_string()),
    };
    let ret = ret.trim();
    if ret == "void" {
        return Some("returns void: no value to compare".to_string());
    }
    if ret.contains('*') || ret.contains('&') {
        return Some(
            "returns a pointer/reference (address identity: the two sides do not share an address space)"
                .to_string(),
        );
    }
    if is_float_type(ret) {
        return Some(
            "returns float/double: the XMM0 comparison class (bitwise on XMM0's raw bits) — the lifter models the answer in RAX only, so no emitted C spells this value"
                .to_string(),
        );
    }
    if !is_int_type(ret) {
        return Some(format!(
            "return type `{ret}` is not a builtin integer (the type model)"
        ));
    }
    if params.len() > max_params {
        return Some(format!(
            "{} parameters: beyond the {} argument registers the model has",
            params.len(),
            max_params
        ));
    }
    if let Some(p) = takes_float(&params) {
        return Some(format!(
            "takes a float/double parameter `{p}`: XMM0-3 are not modelled as arguments (ARG_REGS is RCX/RDX/R8/R9) — XMM0 class"
        ));
    }
    if let Some(p) = takes_pointerish(&params) {
        return Some(format!(
            "takes a pointer/reference/by-value parameter `{p}`"
        ));
    }
    for p in &params {
        if !is_int_type(p) {
            return Some(format!(
                "parameter `{p}` is not a builtin integer (the type model)"
            ));
        }
    }
    None
}

/// Evaluate a statically-named address: a constant, or arithmetic on
/// constants. With the lifter's fixed entry RSP, every `-O0` `rsp + disp` /
/// `rbp + disp` frame slot folds to one of these — and the number is the key
/// that makes a store and a reload the SAME slot (the lifter emits a fresh
/// `Add` node per instruction, so value ids do not match; constants do).
fn fold_addr(f: &Function, v: u32) -> Option<u64> {
    match def_op(f, v)? {
        Op::Const(c) => Some(*c),
        Op::Copy(a) => fold_addr(f, *a),
        Op::Bin(BinOp::Add, a, b) => Some(fold_addr(f, *a)?.wrapping_add(fold_addr(f, *b)?)),
        Op::Bin(BinOp::Sub, a, b) => Some(fold_addr(f, *a)?.wrapping_sub(fold_addr(f, *b)?)),
        Op::Cast(CastKind::Bitcast | CastKind::ZExt | CastKind::SExt, _, a) => fold_addr(f, *a),
        Op::Un(rdec_core::ir::UnOp::Neg, a) => Some(fold_addr(f, *a)?.wrapping_neg()),
        _ => None,
    }
}

/// Is this value built from `this` and constants by addition/subtraction
/// (`this + 4`, `this + this_slot_disp`), or is it a frame slot that carries
/// exactly such a value? That is the window rule, and it is the exact shape
/// the machine side can mirror: the harness hands the function a real buffer
/// on one side and the window's address integer on the other, and both sides
/// then read the same bytes at the same offsets.
///
/// The `Load` arm is what admits the `-O0` idiom `mov [rbp-8],rcx; mov
/// rax,[rbp-8]`: the slot's address folds to a constant both sides share,
/// and the value stored there is itself window-rooted, so the reload carries
/// OUR address (window on one side, buffer on the other) — dereferencing it
/// reads the window on both sides. `carry_chain`'s shape — a caller pointer
/// spilled and reloaded — fails exactly here: the stored value is a `Param`
/// that is not `this`, and the pointer it carries means nothing the two
/// sides share.
///
/// Scaled arithmetic (`this * 4`) is NOT rooted: the machine side multiplies
/// a real address and lands outside the buffer. So is a phi over `this`, a
/// frame slot whose stores are not rooted, and a load from a slot nothing
/// wrote (both sides would read their own stack garbage).
fn window_rooted(
    f: &Function,
    v: u32,
    this: u32,
    slots: &BTreeMap<u64, Vec<u32>>,
    memo: &mut HashMap<u32, bool>,
) -> bool {
    if let Some(&b) = memo.get(&v) {
        return b;
    }
    // The SSA graph is acyclic (one def per value), so the recursion ends.
    let b = match def_op(f, v) {
        Some(Op::Const(_)) => true,
        Some(Op::Param) => v == this,
        Some(Op::Copy(a)) => window_rooted(f, *a, this, slots, memo),
        // Only offset arithmetic: an address the other side can form from
        // its own base by the same additions.
        Some(Op::Bin(op, a, b2)) => {
            matches!(op, BinOp::Add | BinOp::Sub)
                && window_rooted(f, *a, this, slots, memo)
                && window_rooted(f, *b2, this, slots, memo)
        }
        Some(Op::Cast(kind, _, a)) => {
            matches!(kind, CastKind::Bitcast | CastKind::ZExt | CastKind::SExt)
                && window_rooted(f, *a, this, slots, memo)
        }
        Some(Op::Load(_, a)) => match fold_addr(f, *a) {
            Some(addr) if window_rooted(f, *a, this, slots, memo) => {
                slots.get(&addr).is_some_and(|vs| {
                    !vs.is_empty() && vs.iter().all(|&x| window_rooted(f, x, this, slots, memo))
                })
            }
            _ => false,
        },
        _ => false,
    };
    memo.insert(v, b);
    b
}

/// The window-class IR rule: every Load/Store address is window-rooted.
fn this_window_taint(f: &Function) -> Result<(), String> {
    let this = *f.params.first().ok_or_else(|| {
        "member function lifted with no parameters (`this` not modelled)".to_string()
    })?;
    let mut slots: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for b in &f.blocks {
        for i in &b.instrs {
            if let Op::Store(_, a, val) = &i.op {
                if let Some(addr) = fold_addr(f, *a) {
                    slots.entry(addr).or_default().push(*val);
                }
            }
        }
    }
    let mut memo: HashMap<u32, bool> = HashMap::new();
    for b in &f.blocks {
        for i in &b.instrs {
            let (addr, is_load) = match &i.op {
                Op::Load(_, a) => (*a, true),
                Op::Store(_, a, _) => (*a, false),
                _ => continue,
            };
            if !window_rooted(f, addr, this, &slots, &mut memo) {
                // Name the instruction and its address, the way the RAX
                // bucket names its sites: a bucket a reader cannot check is
                // a bucket that gets quietly believed.
                return Err(format!(
                    "0x{:x}: {} reaches memory at an address the window rule does not cover (not `this`, not a frame slot, not a slot carrying `this`)",
                    i.addr,
                    rdec_core::fmt_ir::format_instr(f, i)
                ));
            }
            // A load from a constant slot nothing ever wrote reads per-side
            // garbage: the machine's stack bytes, the C model's zeros. A load
            // from the window itself (a non-constant rooted address) reads
            // the bytes both harnesses preloaded, and is fine.
            if is_load {
                if let Some(ad) = fold_addr(f, addr) {
                    if !slots.contains_key(&ad) {
                        return Err(format!(
                            "0x{:x}: {} reads a frame slot no store ever wrote (the machine reads stack garbage, the C model reads zeros)",
                            i.addr,
                            rdec_core::fmt_ir::format_instr(f, i)
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// The window: 4 KiB at 0x2000 in the C model's address space. `RDEC_ADDR`
/// masks into 1 MiB, and 0x2000 + 4096 stays clear of the frame (STACK_BASE
/// = 1 << 19) and of any single struct these functions own.
const WIN_ADDR: u64 = 0x2000;
const WIN_WORDS: usize = 512; // 4 KiB

/// Window bytes: pure function of (vector, word), embedded verbatim in both
/// harnesses the way the argument table is, so "same bytes in the window" is
/// a property of the generator, not a claim about two files.
fn window_table(n_vec: usize) -> Vec<[u64; WIN_WORDS]> {
    (0..n_vec)
        .map(|v| {
            let mut w = [0u64; WIN_WORDS];
            let mut s = 0xc0ff_ee42_0000_0000_u64 ^ ((v as u64) << 32);
            for x in w.iter_mut() {
                *x = splitmix64(&mut s);
            }
            w
        })
        .collect()
}

fn window_c_array(ws: &[[u64; WIN_WORDS]]) -> String {
    use std::fmt::Write as _;
    let mut s = String::from("static const uint64_t W[][512] = {\n");
    for w in ws {
        s.push_str("  { ");
        for (i, x) in w.iter().enumerate() {
            let _ = write!(
                &mut s,
                "{:#018x}{}",
                x,
                if i + 1 < w.len() { ", " } else { "" }
            );
        }
        s.push_str(" },\n");
    }
    s.push_str("};\n");
    s
}

/// The window runner's body: same one-flushed-line protocol as `main_body`,
/// the wrapper takes the VECTOR index because it must reload the window
/// before the call (the callee may store into it; the next call must start
/// from the shared bytes on both sides).
fn main_body_win(n_jobs: usize) -> String {
    format!(
        "int main(int argc, char **argv) {{\n\
         \x20 long long start = 0;\n\
         \x20 if (argc >= 3 && strcmp(argv[1], \"--start\") == 0) start = atoll(argv[2]);\n\
         \x20 for (long long j = start; j < (long long){n_jobs}; ++j) {{\n\
         \x20   fflush(stdout);\n\
         \x20   uint64_t r = FN[JOB_SYM[j]](JOB_VEC[j]);\n\
         \x20   printf(\"RDEC %lld %016llx\\n\", j, (unsigned long long)r);\n\
         \x20   fflush(stdout);\n\
         \x20 }}\n\
         \x20 printf(\"RDEC DONE\\n\");\n\
         \x20 fflush(stdout);\n\
         \x20 return 0;\n}}\n"
    )
}

/// The decompiled side's window wrapper: preload the window, call with the
/// window's address integer, widen the return from the declared type.
fn c_wrapper_win(idx: usize, cname: &str, n_args: usize, sabotage: bool) -> String {
    // The C side's `this` is the window's ADDRESS INTEGER; the emitted
    // function reaches memory as `rdec_mem[RDEC_ADDR(this + off)]`. The call
    // passes exactly the declared arity: `this` plus the vector words the
    // signature has slots for (clang -Werror counts arguments).
    let flip = if sabotage && idx == 0 { "^ 1ULL" } else { "" };
    let mut args = vec![format!("(int64_t)RDEC_ADDR({WIN_ADDR}ULL)")];
    for k in 1..n_args.min(4) {
        args.push(format!("(int64_t)V[j][{k}]"));
    }
    format!(
        "static uint64_t v{idx}(unsigned j) {{\n\
         \x20 memcpy(&rdec_mem[RDEC_ADDR({WIN_ADDR}ULL)], W[j], sizeof(W[j]));\n\
         \x20 return (uint64_t){cname}({}) {flip};\n}}\n",
        args.join(", ")
    )
}

/// The machine side's window wrapper: preload a real 4 KiB buffer and pass
/// its address. The buffer is a local: the callee stores into it, the driver
/// reads only the return value, and every call starts from `W[j]`. The
/// machine side is never sabotaged — the sabotage hook corrupts the C side,
/// which is the side under test.
fn native_wrapper_win(idx: usize, sym_alias: &str) -> String {
    format!(
        "static uint64_t g{idx}(unsigned j) {{\n\
         \x20 uint64_t buf[512];\n\
         \x20 memcpy(buf, W[j], sizeof buf);\n\
         \x20 return (uint64_t){sym_alias}((uint64_t)(uintptr_t)buf, V[j][1], V[j][2], V[j][3]);\n}}\n"
    )
}

/// The machine side's extern, bound to the decorated COFF symbol by an asm
/// label: clang refuses `?add@@YAHHH@Z` as a C identifier, and the label is
/// the plain C name the rest of the driver uses. The declared types are the
/// ABI's, exactly as the C driver declares them.
fn native_extern(alias: &str, sym: &str) -> String {
    format!("extern uint64_t {alias}(uint64_t, uint64_t, uint64_t, uint64_t) __asm__(\"{sym}\");\n")
}

fn c_wrapper_cpp(idx: usize, cname: &str, sabotage: bool) -> String {
    let flip = if sabotage && idx == 0 { "^ 1ULL" } else { "" };
    format!(
        "static uint64_t w{idx}(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {{\n\
         \x20 return (uint64_t){cname}((int64_t)a, (int64_t)b, (int64_t)c, (int64_t)d) {flip};\n}}\n"
    )
}

/// What one T2-over-C++ run measured. The CLI prints it and owns the exit
/// code; the hermetic test asserts on it directly.
#[derive(Debug, Default)]
pub(crate) struct T2CppReport {
    pub lifted: usize,
    pub refused: usize,
    pub broken: usize,
    /// (functions, runs) per comparison class.
    pub rax_functions: usize,
    pub rax_runs: usize,
    pub win_functions: usize,
    pub win_runs: usize,
    pub compared: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub not_compared: usize,
    /// function -> its final bucket, the accounting that closes.
    pub fn_buckets: BTreeMap<String, String>,
    pub excluded: BTreeMap<String, Vec<String>>,
    /// Global job id -> the value the native side reported (the hermetic
    /// test pins a hand-assembled function's known answer through it).
    pub native_values: BTreeMap<usize, u64>,
    pub mismatches: Vec<String>,
}

pub fn cmd_t2_cpp(args: &[String]) -> Result<i32, String> {
    if !args[1..].iter().any(|a| a == "--cpp") {
        return Err("usage: rdec t2 --cpp [--out DIR] [--limit N] [--vectors N]".to_string());
    }
    let mut files: Vec<String> = Vec::new();
    let mut out_dir = "corpus/t2cpp".to_string();
    let mut limit = usize::MAX;
    let mut n_vec = 12usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            other => {
                if !other.starts_with("--") && other != "--cpp" {
                    files.push(other.to_string());
                }
            }
        }
        i += 1;
    }
    if files.is_empty() {
        files = crate::m8::cpp_corpus_files()?;
    }
    let rep = t2_cpp_run(&files, &out_dir, n_vec, limit, false)?;
    println!(
        "function buckets: compared+match {} | compared+mismatch {} | faulted(one side) {} | excluded {} | refused {} | broken(passes) {} | sum {} == lifted {}",
        rep.fn_buckets.values().filter(|b| **b == "compared+match").count(),
        rep.fn_buckets.values().filter(|b| **b == "compared+mismatch").count(),
        rep.fn_buckets.values().filter(|b| *b == "faulted (one side, no verdict)").count(),
        rep.fn_buckets.values().filter(|b| b.starts_with("excluded: ")).count(),
        rep.fn_buckets.values().filter(|b| **b == "refused by the emitter").count(),
        rep.fn_buckets.values().filter(|b| **b == "broken (passes)").count(),
        rep.fn_buckets.len(),
        rep.lifted
    );
    if rep.compared == 0 {
        println!("FAIL: nothing was compared - a gate that compares nothing proves nothing");
        return Ok(1);
    }
    if rep.mismatched > 0 {
        println!(
            "FAIL: the decompiled C++ C disagrees with the machine code on {} of {} runs",
            rep.mismatched, rep.compared
        );
        return Ok(1);
    }
    println!(
        "OK: the decompiled C computes the same 64-bit RAX as the C++ machine code on all {} runs, over {} RAX-class + {} window-class functions",
        rep.compared, rep.rax_functions, rep.win_functions
    );
    Ok(0)
}

/// The gate's measured core, shared by the CLI and the hermetic test.
/// `sabotage` flips one bit of the C side's FIRST RAX-class wrapper: a
/// comparison that can never fail is not a gate, and this is the proof it
/// can. It is never true from the command line.
pub(crate) fn t2_cpp_run(
    files: &[String],
    out_dir: &str,
    n_vec: usize,
    limit: usize,
    sabotage: bool,
) -> Result<T2CppReport, String> {
    let mut rep = T2CppReport::default();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let p = crate::m8::cpp_pipeline(files)?;

    // ---- 1. classify every lifted function into exactly one bucket ---------
    let mut subs: Vec<Sub> = Vec::new(); // RAX class
    let mut wsubs: Vec<Sub> = Vec::new(); // window class
    let mut taken_names: HashSet<String> = HashSet::new();
    let mut refusal_keys: BTreeMap<String, usize> = BTreeMap::new();
    let mut path_tot = 0usize;
    let mut path_def = 0usize;
    let mut undef_notes: Vec<String> = Vec::new();
    let bucket = |rep: &mut T2CppReport, tag: &str, b: &str| {
        rep.fn_buckets
            .entry(tag.to_string())
            .or_insert_with(|| b.to_string());
    };
    let skip = |rep: &mut T2CppReport, why: &str, tag: &str| {
        rep.excluded
            .entry(why.to_string())
            .or_default()
            .push(tag.to_string());
        bucket(rep, tag, &format!("excluded: {why}"));
    };

    for s in &p.subjects {
        rep.lifted += 1;
        let tag = format!("{} :: {}", s.obj, s.sym);
        if rdec_core::verify::verify_ssa(&s.f).is_err() {
            bucket(&mut rep, &tag, "broken (passes)");
            rep.broken += 1;
            continue;
        }
        if synthetic(&s.sym, s.start) {
            // Before the kind check: it has no demangled kind, and the C++
            // corpus has real ones (discovery reached code no symbol names).
            skip(&mut rep, "no COFF symbol to link (discovery target)", &tag);
            continue;
        }
        let out = match emit_c(&s.f) {
            Ok(o) => o,
            Err(rs) => {
                rep.refused += 1;
                bucket(&mut rep, &tag, "refused by the emitter");
                for r in &rs {
                    *refusal_keys.entry(crate::m4::key(r)).or_default() += 1;
                }
                continue;
            }
        };
        let (n_paths, n_def, _) = rax_paths(&s.f);
        path_tot += n_paths;
        path_def += n_def;
        if returns_undef(&s.f) {
            let (p_, d_, notes) = rax_paths(&s.f);
            undef_notes.push(format!("{tag}: RAX defined on {d_} of {p_} return paths"));
            for n in notes {
                undef_notes.push(format!("      {n}"));
            }
            skip(&mut rep, "returns RAX unwritten on some path", &tag);
            continue;
        }
        let dem = p.dem_of.get(&s.sym);
        let kind = match dem {
            Some(Demangled::Name { kind, .. }) => kind,
            Some(Demangled::Unsupported { .. }) => {
                skip(&mut rep, "demangler gap (Unsupported encoding)", &tag);
                continue;
            }
            None => {
                skip(&mut rep, "no C++ decoration (plain COFF symbol)", &tag);
                continue;
            }
        };
        let text = match dem {
            Some(Demangled::Name { text, .. }) => text.as_str(),
            _ => "",
        };
        // Runnability, `cmd_t2`'s predicates in `cmd_t2`'s order.
        let mut run_gap: Option<String> = None;
        if !out.externs.is_empty() {
            run_gap = Some("calls a symbol we do not model".to_string());
        } else if has_exit(&s.f) {
            run_gap = Some("unmodelled control flow (Exit)".to_string());
        } else if !s.external {
            run_gap = Some("symbol is file-static (not linkable from a driver)".to_string());
        }
        if let Some(g) = run_gap {
            skip(&mut rep, &g, &tag);
            continue;
        }
        // Class assignment: kind decides, the signature refines.
        let class: Result<&str, String> = match kind {
            rdec_demangle::SymKind::FreeFunction | rdec_demangle::SymKind::StaticMember { .. } => {
                match signature_gap(text, 4) {
                    None => Ok("rax"),
                    Some(g) => Err(g),
                }
            }
            rdec_demangle::SymKind::MemberFunction { thunk: true, .. } => {
                Err("vcall/displacement thunk (the offset adjustment is not modelled)".to_string())
            }
            rdec_demangle::SymKind::MemberFunction { .. } => match signature_gap(text, 3) {
                Some(g) => Err(g),
                None => match this_window_taint(&s.f) {
                    Ok(()) => Ok("win"),
                    Err(g) => Err(g),
                },
            },
            rdec_demangle::SymKind::Constructor { .. } => Err(
                "constructor (returns the `this` pointer: address identity the sides do not share)"
                    .to_string(),
            ),
            rdec_demangle::SymKind::Destructor { .. } => Err(
                "destructor (returns the `this` pointer / stack-variant hidden arg)".to_string(),
            ),
            rdec_demangle::SymKind::Vtable { .. } => Err("vtable data, not code".to_string()),
        };
        let class = match class {
            Ok(c) => c,
            Err(g) => {
                skip(&mut rep, &g, &tag);
                continue;
            }
        };
        if touches_memory_outside_the_frame(&s.f) && class == "rax" {
            skip(&mut rep, "memory outside the modelled frame", &tag);
            continue;
        }
        if !taken_names.insert(s.cname.clone()) {
            skip(
                &mut rep,
                "duplicate C name across objects (one TU, one definition)",
                &tag,
            );
            continue;
        }
        if subs.len() + wsubs.len() >= limit {
            skip(&mut rep, "--limit reached", &tag);
            continue;
        }
        let sub = Sub {
            s,
            text: out.text.clone(),
            c_sym: match class {
                "rax" => subs.len(),
                _ => wsubs.len(),
            },
            loops: loop_shaped(&s.f),
        };
        match class {
            "rax" => subs.push(sub),
            _ => wsubs.push(sub),
        }
    }

    // ---- 2. both sides of both classes, on the shared vectors --------------
    let vecs = vectors(n_vec, n_vec);
    let wins = window_table(vecs.len());

    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for (k, sb) in subs.iter().enumerate() {
        let n = if sb.loops { n_vec } else { vecs.len() };
        for v in 0..n {
            jobs.push((k, v));
        }
    }
    let mut wjobs: Vec<(usize, usize)> = Vec::new();
    for (k, sb) in wsubs.iter().enumerate() {
        let n = if sb.loops { n_vec } else { vecs.len() };
        for v in 0..n {
            wjobs.push((k, v));
        }
    }
    let ret32 = subs
        .iter()
        .chain(wsubs.iter())
        .filter(|s| declares_32_bit_return(&s.s.f))
        .count();

    // The decompiled side: two runners, one per class (the wrappers take
    // different shapes); every function's text is `rdec cpp`'s own emitter
    // output, verbatim, under `rdec cpp`'s own names.
    let mut decomp = String::from(
        "/* rdec t2 --cpp: the decompiled side. Every function is the text\n\
         \x20  `rdec cpp --corpus` emits, from the same pipeline and names. */\n\
         #include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n",
    );
    decomp.push_str(PRELUDE);
    decomp.push('\n');
    for sb in subs.iter().chain(wsubs.iter()) {
        decomp.push_str(&sb.text);
        decomp.push('\n');
    }
    // The base carries the function texts only: V is embedded in the RAX
    // runner and in the window runner (its wrappers read V[j][1..3]), W in the
    // window runner only. A table one TU never reads is an unused-static
    // warning, and a harness gets no privileges for being a harness.

    let mut builds = Vec::new();
    let mut drivers: Vec<(String, PathBuf, Vec<usize>)> = Vec::new();

    // --- RAX class ---
    let mut decomp_exe = PathBuf::new();
    if !subs.is_empty() {
        let mut src = decomp.clone();
        src.push_str(&c_array(&vecs));
        src.push_str("typedef uint64_t (*Fn)(uint64_t, uint64_t, uint64_t, uint64_t);\n");
        for sb in &subs {
            src.push_str(&c_wrapper_cpp(sb.c_sym, &sb.s.cname, sabotage));
        }
        src.push_str("static const Fn FN[] = {\n");
        for sb in &subs {
            src.push_str(&format!("  w{},\n", sb.c_sym));
        }
        src.push_str("};\n");
        src.push_str(&job_tables(&jobs));
        src.push_str(&main_body(jobs.len()));
        let sp = PathBuf::from(format!("{out_dir}/t2cpp_decomp.c"));
        let ep = PathBuf::from(format!("{out_dir}/t2cpp_decomp.exe"));
        std::fs::write(&sp, &src).map_err(|e| format!("{}: {e}", sp.display()))?;
        builds.push(Build {
            src: sp,
            exe: ep.clone(),
            obj: None,
        });
        decomp_exe = ep;
    }
    // --- window class ---
    let mut wdecomp_exe = PathBuf::new();
    if !wsubs.is_empty() {
        let mut src = decomp.clone();
        src.push_str(&c_array(&vecs));
        src.push_str(&window_c_array(&wins));
        src.push_str("typedef uint64_t (*FnW)(unsigned);\n");
        for sb in &wsubs {
            src.push_str(&c_wrapper_win(
                sb.c_sym,
                &sb.s.cname,
                sb.s.f.params.len(),
                false,
            ));
        }
        src.push_str("static const FnW FN[] = {\n");
        for sb in &wsubs {
            src.push_str(&format!("  v{},\n", sb.c_sym));
        }
        src.push_str("};\n");
        src.push_str(&job_tables(&wjobs));
        src.push_str(&main_body_win(wjobs.len()));
        let sp = PathBuf::from(format!("{out_dir}/t2cpp_wdecomp.c"));
        let ep = PathBuf::from(format!("{out_dir}/t2cpp_wdecomp.exe"));
        std::fs::write(&sp, &src).map_err(|e| format!("{}: {e}", sp.display()))?;
        builds.push(Build {
            src: sp,
            exe: ep.clone(),
            obj: None,
        });
        wdecomp_exe = ep;
    }

    // --- the machine side: one driver per object per class ---
    let mut order: Vec<String> = Vec::new();
    let mut by_obj: HashMap<String, Vec<usize>> = HashMap::new();
    for (k, sb) in subs.iter().enumerate() {
        if !by_obj.contains_key(&sb.s.obj) {
            order.push(sb.s.obj.clone());
        }
        by_obj.entry(sb.s.obj.clone()).or_default().push(k);
    }
    let mut wby_obj: HashMap<String, Vec<usize>> = HashMap::new();
    for (k, sb) in wsubs.iter().enumerate() {
        if !wby_obj.contains_key(&sb.s.obj) {
            order.push(sb.s.obj.clone());
        }
        wby_obj.entry(sb.s.obj.clone()).or_default().push(k);
    }
    for obj in &order {
        let ks = by_obj.get(obj).cloned().unwrap_or_default();
        let wks = wby_obj.get(obj).cloned().unwrap_or_default();
        let mut src = format!(
            "/* rdec t2 --cpp: the machine-code side of {obj}. The real .obj is\n\
             \x20  linked in; every decorated symbol is bound by an asm label. */\n\
             #include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        );
        let mut ids: Vec<usize> = Vec::new();
        let mut local: Vec<(usize, usize)> = Vec::new();
        for (g, (k, v)) in jobs.iter().enumerate() {
            if ks.contains(k) {
                // LOCAL index into this driver's FN[], like the C gate's
                // drivers: the runner reports its local j, and `ids[j]` is
                // the global job id the parent files the value under.
                local.push((ks.iter().position(|kk| kk == k).unwrap(), *v));
                ids.push(g);
            }
        }
        src.push_str(&c_array(&vecs));
        if !ks.is_empty() {
            let mut syms: Vec<(usize, String)> = Vec::new();
            for &k in &ks {
                let sym = subs[k].s.sym.clone();
                if !syms.iter().any(|(_, s)| *s == sym) {
                    syms.push((syms.len(), sym));
                }
            }
            for (i, sym) in &syms {
                src.push_str(&native_extern(&format!("n{i}"), sym));
            }
            src.push_str("typedef uint64_t (*Fn)(uint64_t, uint64_t, uint64_t, uint64_t);\n");
            src.push_str("static const Fn FN[] = {\n");
            for &k in &ks {
                let i = syms.iter().position(|(_, s)| *s == subs[k].s.sym).unwrap();
                src.push_str(&format!("  n{i},\n"));
            }
            src.push_str("};\n");
            src.push_str(&job_tables(&local));
            src.push_str(&main_body(local.len()));
        }
        let mut wid_ids: Vec<usize> = Vec::new();
        let mut wlocal: Vec<(usize, usize)> = Vec::new();
        for (g, (k, v)) in wjobs.iter().enumerate() {
            if wks.contains(k) {
                wlocal.push((wks.iter().position(|kk| kk == k).unwrap(), *v));
                wid_ids.push(jobs.len() + g);
            }
        }
        if !wks.is_empty() {
            // The window driver is a separate program (its main takes the
            // window wrappers); same protocol, W table, and job ids.
            let mut wsrc = format!(
                "/* rdec t2 --cpp: the machine-code window side of {obj}. */\n\
                 #include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
            );
            wsrc.push_str(&c_array(&vecs));
            wsrc.push_str(&window_c_array(&wins));
            let mut syms: Vec<String> = Vec::new();
            for &k in &wks {
                let sym = wsubs[k].s.sym.clone();
                if !syms.contains(&sym) {
                    syms.push(sym);
                }
            }
            for (i, sym) in syms.iter().enumerate() {
                wsrc.push_str(&native_extern(&format!("m{i}"), sym));
            }
            wsrc.push_str("typedef uint64_t (*FnW)(unsigned);\n");
            for i in 0..syms.len() {
                wsrc.push_str(&native_wrapper_win(i, &format!("m{i}")));
            }
            wsrc.push_str("static const FnW FN[] = {\n");
            for &k in &wks {
                let i = syms.iter().position(|s| *s == wsubs[k].s.sym).unwrap();
                wsrc.push_str(&format!("  g{i},\n"));
            }
            wsrc.push_str("};\n");
            wsrc.push_str(&job_tables(&wlocal));
            wsrc.push_str(&main_body_win(wlocal.len()));
            let wsp = PathBuf::from(format!("{out_dir}/t2cpp_wnative_{}.c", stem_of(obj)));
            let wep = PathBuf::from(format!("{out_dir}/t2cpp_wnative_{}.exe", stem_of(obj)));
            std::fs::write(&wsp, &wsrc).map_err(|e| format!("{}: {e}", wsp.display()))?;
            builds.push(Build {
                src: wsp,
                exe: wep.clone(),
                obj: Some(obj.clone()),
            });
            drivers.push((format!("{obj} [window]"), wep, wid_ids.clone()));
        }
        if !ks.is_empty() {
            let sp = PathBuf::from(format!("{out_dir}/t2cpp_native_{}.c", stem_of(obj)));
            let ep = PathBuf::from(format!("{out_dir}/t2cpp_native_{}.exe", stem_of(obj)));
            std::fs::write(&sp, &src).map_err(|e| format!("{}: {e}", sp.display()))?;
            builds.push(Build {
                src: sp,
                exe: ep.clone(),
                obj: Some(obj.clone()),
            });
            drivers.push((obj.clone(), ep, ids));
        }
    }

    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let t0 = Instant::now();
    let results = compile_all(&cc, &builds);
    let build_failed: Vec<String> = results
        .iter()
        .filter_map(|r| r.as_ref().err().cloned())
        .collect();
    let built = results.iter().filter(|r| r.is_ok()).count();

    // ---- 3. report: classification first, so a shrunk set reads as one -----
    println!(
        "t2 --cpp: {lifted} lifted, {refused} refused by the emitter (not run), {broken} invalid after passes",
        lifted = rep.lifted,
        refused = rep.refused,
        broken = rep.broken
    );
    if !refusal_keys.is_empty() {
        let mut ks: Vec<(String, usize)> = refusal_keys.into_iter().collect();
        ks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!(
            "refusal classes: {}",
            ks.iter()
                .take(6)
                .map(|(k, n)| format!("{n} {k}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    for n in &undef_notes {
        println!("  RAX unwritten: {n}");
    }
    println!(
        "paths: RAX defined on {path_def} of {path_tot} return paths over the lifted functions; runnable only when EVERY path is defined"
    );
    println!(
        "classes: RAX-compared: {rf} functions ({rs} runs) | WINDOW-compared: {wf} functions ({ws} runs) | XMM0-compared: 0 functions / 0 runs",
        rf = subs.len(),
        rs = jobs.len(),
        wf = wsubs.len(),
        ws = wjobs.len()
    );
    println!(
        "XMM0 class: 0 is the honest count while the lifter's ARG_REGS is RCX/RDX/R8/R9: a float argument is an undef live-in (refused) and a float answer reaches `ret` with RAX unwritten (excluded, named). The comparison itself is ready to wire (bitwise on XMM0's raw bits, so -0.0 vs 0.0 and NaN payloads compare as bits — the machine's answer); the class opens the day the lifter models XMM0-3 as arguments and an XMM0 return."
    );
    let skipped_n: usize = rep.excluded.values().map(|v| v.len()).sum();
    println!(
        "runnable: {} functions on both sides ({} RAX + {} window), {skipped_n} not runnable ({})",
        subs.len() + wsubs.len(),
        subs.len(),
        wsubs.len(),
        rep.excluded
            .iter()
            .map(|(k, v)| format!("{} {k}", v.len()))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!(
        "inputs: {} shared integer vectors, {} shared 4 KiB window byte images (regenerated before EVERY window call; the RETURN value is what is compared, memory equivalence is NOT claimed)",
        vecs.len(),
        wins.len()
    );
    println!(
        "return width: {ret32}/{} of the runnable functions declare uint32_t (their answer reaches RAX through a 32-bit write), the rest int64_t",
        subs.len() + wsubs.len()
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
        return Ok(rep);
    }

    // ---- 4. run and compare -------------------------------------------------
    let mut crun_vals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut c_faults: Vec<(usize, String)> = Vec::new();
    let mut restarts = 0usize;
    if !jobs.is_empty() {
        let r = drain(&decomp_exe, &(0..jobs.len()).collect::<Vec<_>>());
        restarts += r.restarts;
        crun_vals.extend(r.vals);
        c_faults.extend(r.faults);
    }
    if !wjobs.is_empty() {
        // Window jobs continue the global id space after the RAX jobs.
        let r = drain(
            &wdecomp_exe,
            &(0..wjobs.len()).map(|g| jobs.len() + g).collect::<Vec<_>>(),
        );
        restarts += r.restarts;
        crun_vals.extend(r.vals);
        c_faults.extend(r.faults);
    }
    let mut native: BTreeMap<usize, u64> = BTreeMap::new();
    let mut n_faults: Vec<(usize, String)> = Vec::new();
    for (obj, exe, ids) in &drivers {
        let r = drain(exe, ids);
        restarts += r.restarts;
        native.extend(r.vals);
        for (g, why) in r.faults {
            n_faults.push((g, format!("{obj}: {why}")));
        }
    }
    rep.native_values = native.clone();

    // Global job ids: [0, jobs.len()) are RAX-class, the rest window-class.
    let all_jobs: Vec<(bool, usize, usize)> = jobs
        .iter()
        .map(|(k, v)| (false, *k, *v))
        .chain(wjobs.iter().map(|(k, v)| (true, *k, *v)))
        .collect();

    println!(
        "compare: the full 64-bit RAX on every shared vector; the C side widened from its declared return type; nothing masked, nothing truncated"
    );
    let mut fn_runs: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new(); // tag -> (runs, mismatch, fault)
    for (g, (is_win, k, v)) in all_jobs.iter().enumerate() {
        let sb = if *is_win { &wsubs[*k] } else { &subs[*k] };
        let tag = format!("{} :: {}", sb.s.obj, sb.s.sym);
        let row = fn_runs.entry(tag.clone()).or_insert((0, 0, 0));
        let (Some(&x), Some(&y)) = (native.get(&g), crun_vals.get(&g)) else {
            row.2 += 1;
            rep.not_compared += 1;
            continue;
        };
        rep.compared += 1;
        row.0 += 1;
        if x == y {
            rep.matched += 1;
        } else {
            rep.mismatched += 1;
            row.1 += 1;
            rep.mismatches.push(format!(
                "{} :: {}  args=[{:#x},{:#x},{:#x},{:#x}]  machine={:#018x}  C={:#018x}",
                sb.s.obj, sb.s.sym, vecs[*v][0], vecs[*v][1], vecs[*v][2], vecs[*v][3], x, y
            ));
        }
    }
    rep.rax_functions = subs.len();
    rep.rax_runs = jobs.len();
    rep.win_functions = wsubs.len();
    rep.win_runs = wjobs.len();

    // Function-level buckets: the accounting the task demands, in priority
    // order: a mismatch outranks a fault, a fault outranks "compared+match".
    for (tag, (runs, mm, ft)) in &fn_runs {
        let b = if *mm > 0 {
            "compared+mismatch"
        } else if *runs == 0 {
            // Not one vector produced two values: every run faulted.
            "faulted (every vector, one side, no verdict)"
        } else if *ft > 0 {
            // Some vectors compared and matched; the rest faulted on one
            // side. Naming it says so — a bare "match" would overstate.
            "compared+match (some vectors faulted)"
        } else {
            "compared+match"
        };
        rep.fn_buckets
            .entry(tag.clone())
            .or_insert_with(|| b.to_string());
    }

    println!(
        "runs: {} native drivers, 2 decompiled runners (C gate protocol), {} restarts after faults in total",
        drivers.len(),
        restarts
    );
    println!(
        "compared: {} | match: {} | mismatch: {} | not compared (one side faulted): {}",
        rep.compared, rep.matched, rep.mismatched, rep.not_compared
    );
    for m in rep.mismatches.iter().take(50) {
        println!("  MISMATCH {m}");
    }
    if rep.mismatches.len() > 50 {
        println!("  ... and {} more", rep.mismatches.len() - 50);
    }
    {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for (g, why) in n_faults.iter().chain(c_faults.iter()) {
            if let Some((is_win, k, v)) = all_jobs.get(*g) {
                let sb = if *is_win { &wsubs[*k] } else { &subs[*k] };
                let who = format!("{} :: {} vec {v}", sb.s.obj, sb.s.sym);
                if seen.insert(who.clone()) {
                    println!("  FAULT {who}: {why}");
                }
                if seen.len() >= 12 {
                    break;
                }
            }
        }
    }
    for (why, names) in &rep.excluded {
        for n in names.iter().take(3) {
            println!("  not run [{why}] {n}");
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdec_core::ir::{BinOp, Function, Op, Terminator};
    use rdec_core::ty::Ty;

    /// The RAX-class signature filter, on the corpus' own demangled texts.
    #[test]
    fn the_signature_rule_admits_builtins_and_names_everything_else() {
        assert_eq!(signature_gap("int add(int, int)", 4), None);
        assert_eq!(signature_gap("__int64 ll(__int64)", 4), None);
        assert_eq!(
            signature_gap("unsigned long long ull(unsigned long long)", 4),
            None
        );
        assert_eq!(signature_gap("int window(int, int, int, int)", 4), None);
        assert!(signature_gap("double add(double, double)", 4)
            .unwrap()
            .contains("XMM0"));
        assert!(signature_gap("float fl(float)", 4)
            .unwrap()
            .contains("XMM0"));
        assert!(signature_gap("int chain(signed char, unsigned char, char, short, unsigned short, int, unsigned int, long, unsigned long, __int64, unsigned __int64, float, double, bool, wchar_t)", 4)
            .is_some());
        assert!(signature_gap("int byval(struct P2)", 4)
            .unwrap()
            .contains("by-value"));
        assert!(signature_gap(
            "int one_ptr(int *, int const *, int *const, int volatile *)",
            4
        )
        .unwrap()
        .contains("pointer"));
        assert!(signature_gap("int refs(int&, int const&)", 4)
            .unwrap()
            .contains("pointer"));
        assert!(signature_gap("Kind pick(Kind a, Kind b)", 4).is_some());
        assert!(signature_gap("void f(int)", 4).unwrap().contains("void"));
        assert!(signature_gap("int& Buffer::slot(int)", 4)
            .unwrap()
            .contains("pointer/reference"));
        // `this` spends RCX in the window class: 3 declared parameters max.
        assert_eq!(signature_gap("int bump(int)", 3), None);
        assert!(signature_gap("int use(int, int, int, int)", 3).is_some());
    }

    /// `this` rooted at parameter 0 is the window; a second pointer rooted at
    /// parameter 1 is not, and a reload through memory is not.
    #[test]
    fn the_window_rule_admits_this_and_names_the_other_shapes() {
        // `mov eax,[rcx]; ret` shape in IR: return *(param0).
        let mut f = Function::new("m", 0);
        let e = f.entry;
        let p0 = f.emit(e, Op::Param, Ty::u64(), 0);
        f.params.push(p0);
        let v = f.emit(e, Op::Load(Ty::u64(), p0), Ty::u64(), 0);
        f.block_mut(e).term = Terminator::Return(v);
        assert!(
            this_window_taint(&f).is_ok(),
            "reads through `this` are the window class"
        );

        // `ret x + *p1`: the load is rooted at parameter 1.
        let mut g = Function::new("n", 0);
        let e = g.entry;
        let p0 = g.emit(e, Op::Param, Ty::u64(), 0);
        let p1 = g.emit(e, Op::Param, Ty::u64(), 0);
        g.params.push(p0);
        g.params.push(p1);
        let v = g.emit(e, Op::Load(Ty::u64(), p1), Ty::u64(), 0);
        g.block_mut(e).term = Terminator::Return(v);
        assert!(this_window_taint(&g)
            .unwrap_err()
            .contains("window rule does not cover"));

        // The -O0 idiom: `this` spilled to a frame slot and reloaded, then
        // dereferenced. The slot's stores are `this`, so the reload carries
        // OUR address (window on the C side, buffer on the machine side) and
        // both sides read the same bytes at the same offsets.
        let mut h = Function::new("r", 0);
        let e = h.entry;
        let p0 = h.emit(e, Op::Param, Ty::u64(), 0);
        h.params.push(p0);
        let slot = h.emit(e, Op::Const(1 << 19), Ty::u64(), 0);
        h.emit_void(e, Op::Store(Ty::u64(), slot, p0), 0);
        let reload = h.emit(e, Op::Load(Ty::u64(), slot), Ty::u64(), 0);
        let v = h.emit(e, Op::Load(Ty::u64(), reload), Ty::u64(), 0);
        h.block_mut(e).term = Terminator::Return(v);
        assert!(
            this_window_taint(&h).is_ok(),
            "a slot carrying `this` carries the window"
        );

        // carry_chain's shape: a CALLER pointer spilled and reloaded, then
        // read through. The slot is a fine frame slot; the pointer it carries
        // means nothing the two sides share.
        let mut k = Function::new("c", 0);
        let e = k.entry;
        let p0 = k.emit(e, Op::Param, Ty::u64(), 0);
        let p1 = k.emit(e, Op::Param, Ty::u64(), 0);
        k.params.push(p0);
        k.params.push(p1);
        let slot = k.emit(e, Op::Const(1 << 19), Ty::u64(), 0);
        k.emit_void(e, Op::Store(Ty::u64(), slot, p1), 0);
        let reload = k.emit(e, Op::Load(Ty::u64(), slot), Ty::u64(), 0);
        let v = k.emit(e, Op::Load(Ty::u64(), reload), Ty::u64(), 0);
        k.block_mut(e).term = Terminator::Return(v);
        assert!(
            this_window_taint(&k).is_err(),
            "a slot carrying a caller pointer is not the window"
        );

        // A load from a constant address nothing wrote: the machine reads its
        // stack garbage, the C model reads its zeros. Not comparable.
        let mut u = Function::new("u", 0);
        let e = u.entry;
        let p0 = u.emit(e, Op::Param, Ty::u64(), 0);
        u.params.push(p0);
        let g = u.emit(e, Op::Const(0x1234), Ty::u64(), 0);
        let v = u.emit(e, Op::Load(Ty::u64(), g), Ty::u64(), 0);
        u.block_mut(e).term = Terminator::Return(v);
        assert!(
            this_window_taint(&u).is_err(),
            "a read of an unwritten slot reads per-side garbage"
        );
    }

    /// The asm-label binding is the one C can spell: a driver must be able to
    /// NAME `?add@@YAHHH@Z` at all. This checks the generator's spelling.
    #[test]
    fn the_native_extern_binds_the_decorated_symbol_by_asm_label() {
        let s = native_extern("n0", "?add@@YAHHH@Z");
        assert!(
            s.contains("extern uint64_t n0(uint64_t, uint64_t, uint64_t, uint64_t)"),
            "{s}"
        );
        assert!(s.contains("__asm__(\"?add@@YAHHH@Z\")"), "{s}");
        assert!(native_wrapper_win(1, "m1").contains("memcpy(buf, W[j], sizeof buf)"));
        let w = c_wrapper_win(0, "Point_sum", 1, false);
        assert!(w.contains("RDEC_ADDR(8192ULL)"), "{w}");
        assert!(
            !w.contains("Point_sum((int64_t)RDEC_ADDR(8192ULL)) ^ 1ULL"),
            "{w}"
        );
        // 3 declared slots (this + 2): exactly that many arguments.
        let w3 = c_wrapper_win(0, "Buffer_set", 3, false);
        assert_eq!(w3.matches(", (int64_t)V[j][").count(), 2, "{w3}");
        assert!(c_wrapper_win(0, "f", 4, true).contains("^ 1ULL"));
    }

    /// The sabotage hook exists and lands on wrapper 0 of the C side only.
    #[test]
    fn the_sabotage_hook_flips_exactly_one_bit_of_the_first_c_wrapper() {
        assert!(c_wrapper_cpp(0, "f", true).contains("^ 1ULL"));
        assert!(!c_wrapper_cpp(1, "f", true).contains('^'));
        assert!(!c_wrapper_cpp(0, "f", false).contains('^'));
    }

    /// The window table is deterministic (both harnesses embed it verbatim)
    /// and the window cannot collide with the frame.
    #[test]
    fn the_window_table_is_deterministic_and_the_window_clears_the_frame() {
        assert_eq!(window_table(3), window_table(3));
        assert_ne!(window_table(3)[0], window_table(3)[1]);
        assert!(
            WIN_ADDR + (WIN_WORDS as u64) * 8 <= (1 << 19),
            "the window must end where the frame begins"
        );
        assert!(window_c_array(&window_table(2)).contains("static const uint64_t W[][512]"));
    }

    /// A C++ free function with integer args, hand-assembled, whose machine
    /// result is known, run through the SAME code path the gate runs: lift ->
    /// emit -> drivers -> link -> run -> compare. `int t2_demo(int, int)` =
    /// `lea eax,[rcx+rcx*2]; add eax,edx; ret` = 3a + b (32-bit, zero-extended
    /// into RAX by the architecture). Vector 0 is [0,1,2,3], so the first
    /// answer is 3*0 + 1 = 1. The sabotage pass flips one bit of the C side
    /// and the gate must see it: a comparison that can never fail is a gate
    /// that can never be true.
    #[test]
    fn a_hand_assembled_cpp_free_function_is_value_verified_and_sabotage_is_caught() {
        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
        if Command::new(&cc).arg("--version").output().is_err() {
            eprintln!("SKIP: {cc} is not on PATH: the hermetic t2 --cpp test needs clang");
            return;
        }
        let dir = std::env::temp_dir().join("rdec_t2cpp_hermetic");
        std::fs::create_dir_all(&dir).expect("temp dir");
        // Hand-assembled: every byte written by hand, symbol decorated as
        // clang would: `int t2_demo(int, int)`.
        let asm = concat!(
            ".file \"t2_demo.s\"\n",
            ".text\n",
            ".globl \"?t2_demo@@YAHHH@Z\"\n",
            "\"?t2_demo@@YAHHH@Z\":\n",
            ".byte 0x8D, 0x04, 0x49\n", // lea  eax, [rcx+rcx*2]
            ".byte 0x01, 0xD0\n",       // add  eax, edx
            ".byte 0xC3\n",             // ret
        );
        let sp = dir.join("t2_demo.s");
        let op = dir.join("t2_demo.obj");
        std::fs::write(&sp, asm).expect("write .s");
        let out = Command::new(&cc)
            .arg("-target")
            .arg("x86_64-pc-windows-msvc")
            .arg("-c")
            .arg(&sp)
            .arg("-o")
            .arg(&op)
            .output()
            .expect("run clang");
        assert!(
            out.status.success(),
            "clang must assemble the hand-written object: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let obj = op.to_str().unwrap().to_string();
        let out_dir = dir.join("gen");
        let rep = t2_cpp_run(
            std::slice::from_ref(&obj),
            out_dir.to_str().unwrap(),
            12,
            usize::MAX,
            false,
        )
        .expect("the gate runs on the hermetic object");
        assert_eq!(
            rep.rax_functions, 1,
            "the demo function is RAX-class: {rep:?}"
        );
        assert_eq!(rep.win_functions, 0);
        assert_eq!(
            refuse_not_broken(&rep),
            0,
            "nothing refused, nothing broken"
        );
        assert!(
            rep.compared >= 12,
            "every vector compared: {}",
            rep.compared
        );
        assert_eq!(rep.mismatched, 0, "mismatches: {:?}", rep.mismatches);
        // The known answer: job 0 is vector 0 = [0,1,2,3] -> 3*0 + 1 = 1.
        assert_eq!(
            rep.native_values.get(&0).copied(),
            Some(1),
            "machine leaves 3a+b in RAX"
        );
        // ... and the machine's bit pattern, not a plausible one: vector
        // [1,0,0,0] is job-list position 1 -> 3.
        assert_eq!(rep.native_values.get(&1).copied(), Some(3));
        // The gate can fail: flip one bit on the C side, the comparison sees
        // it on EVERY vector of that function.
        let bad = t2_cpp_run(
            std::slice::from_ref(&obj),
            out_dir.to_str().unwrap(),
            12,
            usize::MAX,
            true,
        )
        .expect("the sabotage pass runs");
        assert_eq!(bad.compared, rep.compared, "sabotage runs the same jobs");
        assert!(
            bad.mismatched >= 12,
            "the flipped bit is a mismatch, not a match: {bad:?}"
        );
        assert!(bad.matched == 0, "one function, all vectors wrong");
    }

    fn refuse_not_broken(rep: &T2CppReport) -> usize {
        rep.refused + rep.broken
    }

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
