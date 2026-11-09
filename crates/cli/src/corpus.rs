//! The MSVC leg of the corpus matrix (roadmap section 3: `{clang, gcc, MSVC}`).
//!
//! The clang generator lives in `rdec_verify::corpus` and is untouched by this
//! file: `rdec corpus` without `--msvc` does byte-identically what it did before.
//! This module compiles the SAME `corpus/src/*.c` with `cl /c` into COFF objects
//! under `corpus/msvc/obj/`, writes a manifest with the same column shape, and
//! points the existing gates at it (`--corpus-dir corpus/msvc`) — one code path
//! per gate, a different denominator, not a second and weaker copy of each gate.
//!
//! What cl embeds, measured on cl 19.44.35207 (two builds of the same 32 sources
//! into the same directory, two seconds apart):
//!
//! * `TimeDateStamp` at object offset 4 — the build clock; all 32 objects moved.
//! * the 8-byte payload at `.chks64` + 8 — a checksum taken over the object, so
//!   it moves when the clock moves. `.chks64` bytes 0..8 (the file-identity hash)
//!   did NOT move.
//! * `.debug$S` carries the source path and the `.obj` path (`S_OBJNAME`), and the
//!   FILE symbol's aux record carries the `.obj` path. Those are environment, not
//!   clock: two builds into the SAME directory agree, which is exactly the shape
//!   the determinism gate uses. Building into a *different* directory moves bytes
//!   inside `.debug$S` — measured, and the reason the check is "twice into one
//!   dir", never "two dirs, compare".
//!
//! After zeroing the two clock-derived fields, two same-directory builds are
//! byte-identical: 0 of 32 objects differ (before normalization: 32 of 32).
//! `msvc_stamp_offsets_pin_the_real_fields` pins both offsets, and
//! `crates/cli/tests/msvc_corpus.rs::the_normalization_rule_burns_off_a_real_clock`
//! pins them against objects cl just built.
//!
//! Gate policy that follows: **same-toolchain determinism, never cross-compiler
//! byte equality**. The runner's clang 20.1.8 vs this box's 22.1.8 already moved a
//! gate from 1364 to 1412 comparisons; comparing cl objects to clang objects would
//! test the vendors, not rdec.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_core::emit_c::{emit_c_sig, Refusal};
use rdec_core::verify::verify_ssa;

/// The optimization legs of the MSVC corpus.
///
/// `Od`/`O1`/`O2` are the analogues of clang's `-O0/-O1/-O2`. `Os` is the size
/// leg, the analogue of clang's `-Oz`: measured on cl 19.44, `/Os` produces a
/// `.text` that differs from `/O1` on 30 of 32 snippets and from `/O2` on 29 of
/// 32, so it is a leg with shapes of its own.
///
/// `/Oxs` and `/Ox` are NOT legs, and that is a measurement, not an oversight:
/// their `.text` is byte-identical to `/O2` on 32 of 32 snippets (only
/// `.drectve`/debug records differ), so a leg for them would gate nothing.
///
/// `/GS` (the security cookie) is ON for the `O2` leg and OFF for the other three,
/// so the cookie shape is in the matrix. Recorded honestly: on THIS corpus `/GS`
/// is a no-op — 0 of 32 objects reference `__security_cookie`, 0 reference
/// `__chkstk`, and the `/O2 /GS` `.text` is byte-identical to `/O2 /GS-`. None of
/// the 32 snippets has the local-buffer shape that makes MSVC install a cookie, so
/// the lifter refuses nothing by those names. The roadmap WANTS those shapes: the
/// corpus needs a snippet with a stack buffer (`corpus/src/*.c` is not this lane's
/// file; recorded as the next step in the ROADMAP ledger).
pub const MSVC_OPTS: &[&str] = &["Od", "O1", "O2", "Os"];

/// Snippets cl cannot compile, with the measured reason. Loud and counted: a leg
/// that silently drops 2 of 34 sources is the silent-drop sin, and the manifest
/// header prints this list so a census number can be attributed to 32 snippets.
pub const MSVC_EXCLUDED: &[(&str, &str)] = &[
    (
        "call_helper",
        "clang-only spelling: `__attribute__((noinline))` - cl: error C2143: syntax error: missing ')' before '('",
    ),
    (
        "mul_wide",
        "clang-only type: `__int128` - cl: error C4235: '__int128' keyword not supported on this architecture",
    ),
    (
        // Added by the parent, not the lane: the M3 tail-call witness arrived after
        // this list was written, and the leg's completeness check caught the gap as
        // `built 132, wanted 140`. Measured on cl 19.44.35228, all four legs print
        // the same line-2 error, identical to call_helper's.
        "tailcall_local",
        "clang-only spelling: `__attribute__((noinline))` on line 2 - cl: error C2143: syntax error: missing ')' before '(' (measured on all four legs)",
    ),
    (
        // The M3 noreturn witness: same defect, same measured diagnostic. Its
        // tail-call sibling `tailcall_external.c` uses no attribute and DOES build
        // on all four legs, so the tail-call witness survives the toolchain switch;
        // only the noreturn declaration does not.
        "noreturn_call",
        "clang-only spelling: `__attribute__((noreturn))` on line 2 - cl: error C2143: syntax error: missing ')' before '(' (measured on all four legs)",
    ),
];

/// The flags of one leg, spelled out. The manifest records them per leg, so a
/// census number names the flag set that produced it.
///
/// `/Gw-` is explicit, not decorative: whole-global data optimization only does
/// anything under `/GL` (whole-program compilation), and `/GL` replaces the
/// per-function COFF code with an MSVC IL bitstream — there would be no `.text`
/// to decompile at all. So: `/c` per file, no `/GL`, `/Gw-`.
///
/// `/Gs` (the stack-probe threshold) is left at its default: `__chkstk` needs a
/// frame over 4096 bytes and no snippet has one (measured: 0 references).
pub fn msvc_flags(opt: &str) -> Vec<String> {
    let mut f = vec![
        String::from("/nologo"),
        String::from("/c"),
        String::from("/Gw-"),
    ];
    match opt {
        "Od" => f.extend(["/Od".to_string(), "/GS-".to_string()]),
        "O1" => f.extend(["/O1".to_string(), "/GS-".to_string()]),
        "O2" => f.extend(["/O2".to_string(), "/GS".to_string()]),
        "Os" => f.extend(["/O1".to_string(), "/Os".to_string(), "/GS-".to_string()]),
        other => f.extend([format!("/{other}"), "/GS-".to_string()]),
    }
    f
}

/// The cl driver. `RDEC_CL` wins, then `cl` on PATH.
///
/// `cl` is NOT on PATH on `windows-latest`, and not on PATH in a plain git-bash
/// shell on this box either: both need a vcvars64 environment. A Rust process
/// cannot `call` a batch file (`call` is a cmd builtin, not a program), so the CI
/// step writes a wrapper batch into `RUNNER_TEMP` — `call vcvars64.bat` then
/// `cl %*` — and points `RDEC_CL` at it. The LAST LINE matters: the driver is
/// invoked as `driver <flags> <src> /Fo<obj>`, so a wrapper ending in a bare `%*`
/// hands cl's switches to cmd.exe. Measured: every one of 128 spawns died with
/// `cl exit 1` and the banner probe printed no banner, because cmd ate `/nologo`.
/// With `cl %*` the driver is one executable that behaves exactly like `clang`.
///
/// Forward slashes in that path are normalized to backslashes: `Command::new`
/// launches a `.cmd` through cmd.exe, and a batch path spelled with `/` is not
/// found - measured as `cl exit 1` for all 128 objects and no banner, from a
/// git-bash `RDEC_CL=C:/Users/...` (PowerShell's `Join-Path` gives backslashes,
/// which is why the CI path worked and the shell path did not).
pub fn cl_path() -> String {
    let p = std::env::var("RDEC_CL").unwrap_or_else(|_| "cl".to_string());
    if cfg!(windows) && p.contains('/') {
        p.replace('/', "\\")
    } else {
        p
    }
}

/// The compiler's own banner, for the `# cl:` manifest line.
pub fn cl_stamp(cl: &str) -> String {
    let Some(o) = Command::new(cl).arg("/?").output().ok() else {
        return "unknown (driver not runnable: set RDEC_CL)".into();
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    text.lines()
        .map(str::trim)
        .find(|l| l.to_lowercase().contains("compiler version"))
        .filter(|l| !l.is_empty())
        .unwrap_or("unknown (no banner)")
        .to_string()
}

/// COFF: `Machine` at 0, `NumberOfSections` at 2, `TimeDateStamp` at 4,
/// `SizeOfOptionalHeader` at 16. An *object* has no image signature field and
/// `cl` writes `SizeOfOptionalHeader == 0`, so section headers start at 20.
fn coff_header(bytes: &[u8]) -> Option<(u16, usize, usize)> {
    if bytes.len() < 20 {
        return None;
    }
    let machine = u16::from_le_bytes([bytes[0], bytes[1]]);
    let nsec = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
    let sizeopt = u16::from_le_bytes([bytes[16], bytes[17]]) as usize;
    Some((machine, nsec, 20 + sizeopt))
}

/// One 4-byte field of a 40-byte section header, by field offset:
/// `VirtualSize` 8, `TimeDateStamp` 16, `PointerToRawData` 20,
/// `NumberOfRelocations` 32, `Characteristics` 36.
fn section_field(bytes: &[u8], base: usize, field: usize) -> Option<usize> {
    let i = base.checked_add(field)?;
    Some(u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?) as usize)
}

/// Does this 8-byte COFF name field spell `name`, NUL-padded? (A section name is
/// 8 bytes, never 9: `b".chks64\0"` is exactly 8 and `b".text\0\0\0\0"` is 9 —
/// the off-by-one that shows up as a section named `` `.text ``.)
fn name_is(field: &[u8], name: &str) -> bool {
    field.len() == 8
        && field[..name.len()] == *name.as_bytes()
        && field[name.len()..].iter().all(|&b| b == 0)
}

/// Zero the COFF `TimeDateStamp` (object offset 4) of an AMD64 object.
///
/// The clang leg's rule, byte-for-byte, and deliberately a *copy* rather than a
/// call: `crates/verify/**` is not this lane's file. The copies cannot drift in
/// silence — `the_two_stamp_rules_agree_on_the_offset` runs both on one object
/// and requires identical output.
pub fn normalize_coff_stamp(bytes: &mut [u8]) -> bool {
    let Some((machine, _, _)) = coff_header(bytes) else {
        return false;
    };
    if machine != 0x8664 || bytes.len() < 20 {
        return false;
    }
    if bytes[4..8] == [0, 0, 0, 0] {
        return false; // already normalized: report "unchanged"
    }
    bytes[4..8].copy_from_slice(&[0, 0, 0, 0]);
    true
}

/// Zero the 8-byte checksum at `.chks64` + 8 of an AMD64 object.
///
/// `.chks64` is 16 bytes: a file-identity hash (bytes 0..8, measured STABLE
/// across rebuilds) and a checksum (bytes 8..16, measured volatile in 32 of 32)
/// taken over the object *including the TimeDateStamp*. Normalizing the clock
/// without normalizing this leaves every manifest hash moving: the determinism
/// gate would then be gated on a checksum of a field it already erased.
pub fn normalize_chks64_checksum(bytes: &mut [u8]) -> bool {
    let Some((machine, nsec, secbase)) = coff_header(bytes) else {
        return false;
    };
    if machine != 0x8664 {
        return false;
    }
    let mut hit = false;
    for i in 0..nsec {
        let h = secbase + 40 * i;
        let Some(name) = bytes.get(h..h + 8) else {
            break;
        };
        if !name_is(name, ".chks64") {
            continue;
        }
        let Some(raw) = section_field(bytes, h, 20) else {
            continue;
        };
        if raw + 16 <= bytes.len() && bytes[raw + 8..raw + 16] != [0; 8] {
            bytes[raw + 8..raw + 16].copy_from_slice(&[0; 8]);
            hit = true;
        }
    }
    hit
}

/// The generator's report.
#[derive(Debug, Default)]
pub struct MsvcReport {
    pub built: usize,
    pub bytes: u64,
    pub normalized_stamp: usize,
    pub normalized_chks64: usize,
    pub failures: Vec<String>,
    pub entries: Vec<(String, String, PathBuf, u64, u64)>,
}

/// Build the MSVC corpus: every `src/*.c` cl accepts, at every leg.
pub fn generate_msvc(
    cl: &str,
    src_dir: &Path,
    out_dir: &Path,
    opts: &[&str],
) -> Result<MsvcReport, String> {
    let names = source_names(src_dir)?;
    if names.is_empty() {
        return Err(format!(
            "no .c sources in {} - refusing to write an empty MSVC corpus",
            src_dir.display()
        ));
    }
    let obj_dir = out_dir.join("obj");
    std::fs::create_dir_all(&obj_dir).map_err(|e| format!("{}: {e}", obj_dir.display()))?;

    let mut rep = MsvcReport::default();
    let mut manifest: BTreeMap<(String, String), (PathBuf, u64, u64)> = BTreeMap::new();

    // The exclusion list is a PIN, not an allowance. It rots in two directions and
    // both are silent, so both are checked here, one compile per declared name at
    // the first leg (3 compiles, ~4 s through the vcvars wrapper):
    //  - a declared exclusion that cl now COMPILES: its objects are gateable, and
    //    leaving it excluded shrinks the corpus by a leg's worth of coverage with
    //    every number in the ledger still self-consistent;
    //  - a declared exclusion with no source file: `want` is computed as
    //    (sources - |excluded|) x legs, so a phantom row shrinks the denominator of
    //    every census and the `# excluded:` manifest header becomes a lie.
    // Today's corpus grew past the list once already (the M3 tail-call/noreturn
    // witnesses use `__attribute__`, which cl rejects); the completeness check below
    // caught it as a loud `built 132, wanted 140`. This block is the same rot
    // caught on the side the completeness check cannot see.
    for (name, why) in MSVC_EXCLUDED {
        let src = src_dir.join(format!("{name}.c"));
        if !src.is_file() {
            return Err(format!(
                "MSVC_EXCLUDED names {name} (reason on the ledger: {why}) but {} does not exist - drop the row, it shrinks every census denominator by {} objects",
                src.display(),
                opts.len()
            ));
        }
        let probe = obj_dir.join(format!("_probe_{name}.obj"));
        let mut cmd = Command::new(cl);
        cmd.args(msvc_flags(opts[0]))
            .arg(&src)
            .arg(format!("/Fo{}", probe.display()));
        let built = matches!(cmd.output(), Ok(o) if o.status.success());
        let _ = std::fs::remove_file(&probe);
        if built {
            return Err(format!(
                "MSVC_EXCLUDED names {name} as uncompilable (reason on the ledger: {why}) but cl built it at /{} - its objects must be gated, not excluded; drop the row",
                opts[0]
            ));
        }
    }

    for name in &names {
        if MSVC_EXCLUDED.iter().any(|(n, _)| n == name) {
            continue;
        }
        let src = src_dir.join(format!("{name}.c"));
        for &opt in opts {
            let obj = obj_dir.join(format!("{name}.{opt}.obj"));
            let mut cmd = Command::new(cl);
            cmd.args(msvc_flags(opt))
                .arg(&src)
                .arg(format!("/Fo{}", obj.display()));
            // `.output()`, not `.status()`: a bare "cl exit 1" with the compiler's
            // own message swallowed is undebuggable, and the first cl line is what
            // names the cause (this cost a full 128-object spin to find once).
            let out = cmd.output();
            let (code, stdout, stderr) = match out {
                Ok(o) => (
                    o.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&o.stdout).to_string(),
                    String::from_utf8_lossy(&o.stderr).to_string(),
                ),
                Err(e) => {
                    rep.failures.push(format!("{name} /{opt}: spawn {cl}: {e}"));
                    continue;
                }
            };
            if code != 0 {
                // The FIRST non-empty cl line is the source echo (`noreturn_call.c`),
                // so taking it as "the cause" names the file and nothing else - which
                // is exactly what this line did to the 8 failures that started this
                // fix. Prefer the line that carries a diagnostic, then any line that
                // is not the echoed source name, then say so.
                let echo = format!("{name}.c");
                let lines = stderr
                    .lines()
                    .chain(stdout.lines())
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>();
                let msg = lines
                    .iter()
                    .find(|l| {
                        l.contains("error C")
                            || l.contains("fatal error")
                            || l.contains("error LNK")
                    })
                    .or_else(|| lines.iter().find(|l| **l != echo))
                    .cloned()
                    .unwrap_or("(cl printed nothing but the source echo)");
                rep.failures
                    .push(format!("{name} /{opt}: cl exit {code}: {msg}"));
                continue;
            }
            let Ok(mut bytes) = std::fs::read(&obj) else {
                rep.failures.push(format!(
                    "{name} /{opt}: cl exited 0 but wrote no {}",
                    obj.display()
                ));
                continue;
            };
            if normalize_coff_stamp(&mut bytes) {
                rep.normalized_stamp += 1;
            }
            if normalize_chks64_checksum(&mut bytes) {
                rep.normalized_chks64 += 1;
            }
            // Rewrite so the artifact on disk and the manifest hash agree: a byte
            // diff of two builds is then EMPTY, not merely equal modulo a rule.
            std::fs::write(&obj, &bytes).map_err(|e| format!("{}: {e}", obj.display()))?;
            let h = fnv1a64(&bytes);
            rep.built += 1;
            rep.bytes += bytes.len() as u64;
            rep.entries.push((
                name.clone(),
                opt.to_string(),
                obj.clone(),
                bytes.len() as u64,
                h,
            ));
            manifest.insert(
                (name.clone(), opt.to_string()),
                (obj, bytes.len() as u64, h),
            );
        }
    }

    let mut m = String::new();
    m.push_str(&format!("# cl: {}\n", cl_stamp(cl)));
    m.push_str(&format!("# sources: {}\n", src_dir.display()));
    for &(n, why) in MSVC_EXCLUDED {
        m.push_str(&format!("# excluded: {n}.c - {why}\n"));
    }
    for &opt in opts {
        m.push_str(&format!("# flags[{opt}]: {}\n", msvc_flags(opt).join(" ")));
    }
    m.push_str("# normalized: COFF TimeDateStamp zeroed at object offset 4\n");
    m.push_str("# normalized: .chks64 checksum (8 bytes at section offset 8) zeroed - it hashes the stamp\n");
    m.push_str("# determinism: same cl, twice, into the SAME dir (.debug$S and the FILE aux record embed the .obj path)\n");
    m.push_str("name\topt\tobj\tbytes\tfnv1a64\n");
    for ((name, opt), (path, bytes, hash)) in &manifest {
        // Forward slashes in the manifest: `out_dir` is a user string and
        // `Path::join` would append a backslash, giving the ugly middle form
        // `corpus/msvc\obj\x.obj`. Windows accepts `/` in every file API, and a
        // path that reads the same on every platform is the one a gate can print.
        let rel = path.display().to_string().replace('\\', "/");
        m.push_str(&format!("{name}\t{opt}\t{rel}\t{bytes}\t{hash:016x}\n"));
    }
    std::fs::write(out_dir.join("manifest.tsv"), &m)
        .map_err(|e| format!("{}: {e}", out_dir.join("manifest.tsv").display()))?;
    Ok(rep)
}

fn source_names(src_dir: &Path) -> Result<Vec<String>, String> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(src_dir).map_err(|e| format!("{}: {e}", src_dir.display()))? {
        let p = e.map_err(|e| format!("read_dir: {e}"))?.path();
        if p.extension().map(|x| x == "c").unwrap_or(false) {
            if let Some(s) = p.file_stem() {
                v.push(s.to_string_lossy().to_string());
            }
        }
    }
    v.sort();
    Ok(v)
}

/// `rdec corpus --msvc [--out DIR] [--src DIR] [--opt Od,O2]`
pub fn cmd_corpus_msvc(args: &[String]) -> i32 {
    let mut out = String::from("corpus/msvc");
    let mut src = String::from("corpus/src");
    let mut opts: Vec<String> = MSVC_OPTS.iter().map(|s| s.to_string()).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 2;
            }
            "--src" if i + 1 < args.len() => {
                src = args[i + 1].clone();
                i += 2;
            }
            "--opt" if i + 1 < args.len() => {
                opts = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                i += 2;
            }
            "--msvc" => {
                // The dispatch flag itself: `rdec corpus --msvc` hands us args[1..],
                // which starts with the flag that selected this command.
                i += 1;
            }
            other => {
                eprintln!("corpus --msvc: ignoring unknown arg {other}");
                i += 1;
            }
        }
    }
    let cl = cl_path();
    let opt_refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
    let n_src = source_names(Path::new(&src)).unwrap_or_default().len();
    let want = n_src.saturating_sub(MSVC_EXCLUDED.len()) * opt_refs.len();
    let rep = match generate_msvc(&cl, Path::new(&src), Path::new(&out), &opt_refs) {
        Ok(r) => r,
        Err(e) => {
            println!("FAIL msvc corpus: {e}");
            return 1;
        }
    };
    for f in &rep.failures {
        println!("FAIL  {f}");
    }
    println!("cl driver: {cl}");
    println!("cl version: {}", cl_stamp(&cl));
    println!(
        "{} legs x {} snippets ({} sources, {} excluded by name) -> {want} objects",
        opt_refs.len(),
        n_src.saturating_sub(MSVC_EXCLUDED.len()),
        n_src,
        MSVC_EXCLUDED.len(),
    );
    println!("built {} objects, {} bytes", rep.built, rep.bytes);
    println!(
        "normalized: {} stamps (object offset 4), {} .chks64 checksums (section offset 8)",
        rep.normalized_stamp, rep.normalized_chks64
    );
    println!("manifest: {out}/manifest.tsv");
    if rep.failures.is_empty() && rep.built == want && want > 0 {
        println!("msvc corpus OK");
        0
    } else {
        println!(
            "FAIL: msvc corpus incomplete ({} built, {want} wanted)",
            rep.built
        );
        1
    }
}

// ---------------------------------------------------------------------------
// msvc-gates: the MSVC leg's gate battery, vacuity-guarded at every stage.
// ---------------------------------------------------------------------------

/// `rdec msvc-gates [--corpus-dir DIR] [--limit N] [--full]`
///
/// Four stages over the MSVC objects. The first three ARE the gates the clang
/// corpus is gated by — `objdump-check`, `discover`, `lift` — run over this
/// manifest through `--corpus-dir`. The fourth is the refusal census by MNEMONIC,
/// the roadmap's "lifter priority comes from frequency" measurement for a new
/// toolchain.
///
/// This command does not adopt `rdec c`'s exit code: refusals are the honest
/// state of a toolchain on the day it is added, and a gate that ships red measures
/// nothing. What it DOES gate, because those must be zero on any compiler: an
/// empty corpus, a non-AMD64 object, a pass that breaks SSA, a refusal that is not
/// counted, a refusal that names nothing, and emitted C that clang rejects.
pub fn cmd_msvc_gates(args: &[String]) -> i32 {
    let dir = args
        .iter()
        .position(|a| a == "--corpus-dir")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "corpus/msvc".to_string());
    let limit = if args.iter().any(|a| a == "--full") {
        usize::MAX
    } else {
        args.iter()
            .position(|a| a == "--limit")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(40)
    };

    let mut fails: Vec<String> = Vec::new();
    println!("=== msvc-gates: {dir} (objdump limit {limit}) ===");

    // Stage 0: the corpus exists, is AMD64, and is not empty. A gate over zero
    // objects is the failure mode this repo has been burned by twice.
    let objects = match objects_of_manifest(&dir) {
        Ok(o) => o,
        Err(e) => {
            println!("FAIL vacuity: {e}");
            return 1;
        }
    };
    let n_objects = objects.len();
    let mut bytes = 0u64;
    let mut bad_machine = 0usize;
    let mut code_sections = 0usize;
    let mut code_symbols = 0usize;
    let mut missing = 0usize;
    for f in &objects {
        let Ok(b) = std::fs::read(f) else {
            missing += 1;
            continue;
        };
        bytes += b.len() as u64;
        if coff_header(&b).map(|(m, _, _)| m) != Some(0x8664) {
            bad_machine += 1;
        }
        if let Ok(obj) = rdec_arch_x86::loader::load(Path::new(f)) {
            code_sections += (0..obj.sections.len()).filter(|&i| obj.is_code(i)).count();
            code_symbols += obj.code_symbols().len();
        }
    }
    println!(
        "corpus: {} objects, {bytes} bytes, {code_sections} code sections, {code_symbols} code symbols; cl = {}",
        objects.len(),
        cl_stamp(&cl_path())
    );
    if missing > 0 {
        fails.push(format!(
            "{missing} manifest objects are not on disk (run `rdec corpus --msvc`)"
        ));
    }
    if bad_machine > 0 {
        fails.push(format!(
            "{bad_machine} objects are not AMD64 (Machine != 0x8664)"
        ));
    }
    if code_symbols == 0 {
        fails.push(
            "0 code symbols in the whole MSVC corpus - every later stage is vacuous".to_string(),
        );
    }

    // Stage a: instruction-boundary agreement vs llvm-objdump (the M1 gate).
    println!("\n--- a. objdump-check (boundaries vs llvm-objdump) ---");
    let rc = crate::m1::cmd_objdump_check(&gate_args("objdump-check", &dir, limit)).unwrap_or_else(
        |e| {
            eprintln!("objdump-check: {e}");
            1
        },
    );
    if rc != 0 {
        fails.push(format!("objdump-check exited {rc}"));
    }

    // Stage b: discovery - every symbol-table function reachable, coverage.
    //
    // Run as a subprocess of the SAME binary, so this stays the one discover
    // implementation and we are only reading its numbers. Its exit code is
    // MEASURED, not adopted, because the M1 floor (99.5% of non-padding code
    // bytes) is not a comparable denominator on this leg. Measured over the 132
    // objects (2026-11-08): 134 functions, 0 missing, 0 extra starts, coverage
    // 6979/7093 = 98.39%, 114 uncovered bytes in 7 runs, gaps: indirect branch 2.
    // (The lane's first leg, 128 objects, read 6932/7048 = 98.35%, 116 bytes in 8
    // runs.) The
    // uncovered bytes are inside `.text$mn` (NOT `.pdata`/`.xdata`: objdump -h on
    // switch_dense.O2.obj shows `.drectve .debug$S .text$mn .chks64`, and the
    // loader counts exactly one code section per object). Two shapes, both MSVC
    // spelling: a `switch` lowered into a table of `mov eax,k; jmp` thunks reached
    // through an INDIRECT branch - 56 bytes at 0x2c in one object, the gap the
    // reachability pass reports as `indirect branch` - and 2-byte `66 90`
    // alignment fills, which `m1::is_padding_run` NOW names (it decodes to
    // `xchg ax,ax`, so a `starts_with("nop")` predicate called them missed code).
    // Measured effect of naming them: 2 of the 116 bytes, one gap run of eight.
    // So this leg gates what is toolchain-independent: every symbol-table function
    // reachable (`missing == 0`) and a non-empty corpus; coverage is printed with
    // the reason. REMAINING owner for the 114 bytes: the indirect-branch jump-table
    // work (roadmap M2) claims the thunk table - after that, the M1 floor is
    // comparable here.
    println!("\n--- b. discover (reachability + coverage) ---");
    match run_and_capture(&["discover", "--corpus-dir", dir.as_str()]) {
        Ok(text) => {
            let missing = field_after(&text, "discover:", "missing");
            let funcs = field_after(&text, "discover:", "symbol-table");
            let extra = field_after(&text, "discover:", "extra");
            println!(
                "parsed: {funcs} symbol-table functions, {missing} missing, {extra} extra starts"
            );
            if funcs == 0 {
                fails.push(format!(
                    "discover saw 0 symbol-table functions over {n_objects} MSVC objects - vacuous"
                ));
            }
            if missing > 0 {
                fails.push(format!(
                    "{missing} symbol-table functions are not reachable"
                ));
            }
            if extra > 0 {
                println!(
                    "NOTE: {extra} reachable starts are not in the symbol table - MSVC's \
                     thunk table is reached by an indirect jump, so reachability finds it \
                     and the symbol table does not name it (measured, not gated)"
                );
            }
            for pfx in ["code bytes:", "coverage:", "gaps:"] {
                if let Some(line) = text.lines().find(|l| l.starts_with(pfx)) {
                    println!("{line}");
                }
            }
            println!(
                "coverage is measured, not gated here: MSVC lowers `switch` to a table of \
                 mov+jmp thunks inside .text$mn (reached through an indirect branch) and \
                 pads with 2-byte `66 90` fills, named as padding on 2026-11-08 and \
                 worth 2 of the leg's 116 uncovered bytes. The clang leg has neither \
                 shape (see the note above the call)"
            );
            if text.contains("discovery checked nothing") {
                fails.push("discover reported it checked nothing".to_string());
            }
        }
        Err(e) => fails.push(format!("could not run discover: {e}")),
    }

    // Stage c: lift + verify_ssa.
    println!("\n--- c. lift (IR validity) ---");
    // Unbounded: `--limit` exists to bound llvm-objdump spawns, and lift spawns
    // nothing. A lift leg over 40 of 128 objects is a sample calling itself a leg.
    let rc = crate::m1::cmd_lift(&gate_args("lift", &dir, usize::MAX)).unwrap_or_else(|e| {
        eprintln!("lift: {e}");
        1
    });
    if rc != 0 {
        fails.push(format!("lift exited {rc}"));
    }

    // Stage d: emitted C + the refusal census by mnemonic.
    println!("\n--- d. c: emitted C + REFUSAL CENSUS BY MNEMONIC ---");
    if let Err(e) = msvc_census(&objects, &format!("{dir}/c")) {
        println!("FAIL: {e}");
        fails.push("the MSVC C census stage failed".to_string());
    }

    println!();
    if fails.is_empty() {
        println!(
            "msvc-gates OK: {} objects, {code_symbols} code symbols, census printed, nothing vacuous",
            objects.len()
        );
        0
    } else {
        for f in &fails {
            println!("FAIL: {f}");
        }
        println!("msvc-gates: {} stages failing", fails.len());
        1
    }
}

/// Run this same binary with these args and hand back its stdout+stderr. The
/// gates stay ONE implementation (the clang leg runs the same commands); this is
/// how a battery reads a gate's numbers without forking its logic.
fn run_and_capture(args: &[&str]) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let out = Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| format!("spawn {}: {e}", args.join(" ")))?;
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// The number in front of `word` on a line starting with `prefix`; 0 if absent.
/// Parsing a printed line is fragile by construction, so a missing field reads as
/// 0, and every caller treats 0 as "vacuous / gate it", never as "fine".
fn field_after(text: &str, prefix: &str, word: &str) -> usize {
    for line in text.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        for (i, t) in toks.iter().enumerate() {
            if t.starts_with(word) && i > 0 {
                let digits: String = toks[i - 1].chars().filter(|c| c.is_ascii_digit()).collect();
                return digits.parse().unwrap_or(0);
            }
        }
    }
    0
}

fn gate_args(cmd: &str, dir: &str, limit: usize) -> Vec<String> {
    let mut v = vec![cmd.to_string(), "--corpus-dir".to_string(), dir.to_string()];
    if limit != usize::MAX {
        v.extend(["--limit".to_string(), limit.to_string()]);
    }
    v
}

fn objects_of_manifest(dir: &str) -> Result<Vec<String>, String> {
    let man = Path::new(dir).join("manifest.tsv");
    let text = std::fs::read_to_string(&man)
        .map_err(|e| format!("{}: {e} - run `rdec corpus --msvc` first", man.display()))?;
    let v: Vec<String> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\t"))
        .filter_map(|l| l.split('\t').nth(2).map(|s| s.to_string()))
        .collect();
    if v.is_empty() {
        return Err(format!("{} lists 0 objects", man.display()));
    }
    Ok(v)
}

/// Lift every MSVC function, run the passes, emit C, print the refusal census
/// keyed by MNEMONIC, and compile what was emitted. The census is the product of
/// this stage, so refusals are not a failure. What is: an empty census over
/// refused functions (a silent drop), a refusal that names nothing, IR a pass
/// broke, emitted C clang rejects, and 0 functions.
fn msvc_census(files: &[String], emit_dir: &str) -> Result<(), String> {
    let subjects = crate::m4::lift_all(files)?;
    if subjects.is_empty() {
        return Err(format!(
            "0 functions lifted from {} objects - the census would check nothing",
            files.len()
        ));
    }
    let protos = crate::m4::protos_of(&subjects);
    let (mut seen, mut emitted, mut refused, mut broken, mut keyless) = (0, 0, 0, 0, 0);
    let mut iso_bits: u32 = 0;
    let mut census: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_refusal: Option<String> = None;
    let mut text = String::new();
    text.push_str("#include <stdint.h>\n#include <string.h>\n\n");
    text.push_str(rdec_core::emit_c::PRELUDE);
    text.push('\n');

    for s in &subjects {
        seen += 1;
        iso_bits |= s.iso.verified;
        if let Some(b) = &s.iso.broke {
            println!(
                "pass-broke-ssa record: {} :: {}: after pass {} '{}': {}",
                s.obj,
                s.sym,
                b.index,
                b.name,
                b.errors.join("; ")
            );
        }
        if let Err(errs) = verify_ssa(&s.f) {
            broken += 1;
            if broken <= 3 {
                println!("PASS BREAK {} :: {}: {}", s.obj, s.sym, errs.join("; "));
            }
            continue;
        }
        match emit_c_sig(&s.f, &protos) {
            Ok(out) => {
                emitted += 1;
                text.push_str(&out.text);
                text.push('\n');
            }
            Err(rs) => {
                refused += 1;
                for r in &rs {
                    let k = census_key(r);
                    if k.is_empty() {
                        keyless += 1;
                    }
                    *census.entry(k).or_default() += 1;
                }
                if first_refusal.is_none() {
                    if let Some(r) = rs.first() {
                        first_refusal = Some(format!(
                            "{} :: {} @ {:#x}: {}",
                            s.obj, s.sym, r.at, r.reason
                        ));
                    }
                }
            }
        }
    }
    let records: usize = census.values().sum();
    println!(
        "emit-c: {seen} lifted functions, {emitted} emitted C, {refused} refused, {broken} invalid after passes"
    );
    // Same isolation summary as `rdec c --corpus` (see `cmd_c`): distinct
    // passes that ran and were SSA-verified, on the emitted count.
    println!(
        "pass isolation: {} passes verified on {emitted} emitted functions",
        iso_bits.count_ones()
    );
    println!("cookie shapes: MSVC leg runs /GS on at O2; the corpus emits no __security_cookie or __chkstk site (measured), so nothing is refused by those names");
    println!(
        "refusal records: {records} in {refused} refused functions, {} census keys",
        census.len()
    );
    println!("REFUSAL CENSUS BY MNEMONIC (MSVC objects; count = refusals naming it):");
    let mut ranked: Vec<(String, usize)> = census.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (k, n) in &ranked {
        println!("  {n:>5}  {k}");
    }
    if let Some(r) = &first_refusal {
        println!("first refusal: {r}");
    }
    if seen == 0 || emitted + refused == 0 {
        return Err(format!(
            "vacuity: {seen} functions seen, {emitted} emitted, {refused} refused"
        ));
    }
    if refused > 0 && records == 0 {
        return Err(format!(
            "{refused} refused functions produced 0 census records - a refusal is being dropped"
        ));
    }
    if keyless > 0 {
        return Err(format!(
            "{keyless} refusal records named nothing - the census cannot attribute them"
        ));
    }
    if broken > 0 {
        return Err(format!(
            "a pass broke SSA in {broken} MSVC-lifted functions"
        ));
    }
    std::fs::create_dir_all(emit_dir).map_err(|e| format!("{emit_dir}: {e}"))?;
    let path = format!("{emit_dir}/all.c");
    std::fs::write(&path, &text).map_err(|e| format!("{path}: {e}"))?;
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let obj = format!("{emit_dir}/all.obj");
    let out = Command::new(&cc)
        .args([
            "-std=c17",
            "-c",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
        ])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .output();
    match out {
        Ok(o) if o.status.success() => println!("clang -Werror: OK ({path} -> {obj})"),
        Ok(o) => {
            let e = String::from_utf8_lossy(&o.stderr);
            let first = e
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or("<no error line>");
            return Err(format!(
                "{cc} -Werror rejected the MSVC-emitted C ({emitted} functions emitted): {first}"
            ));
        }
        Err(e) => return Err(format!("could not run {cc}: {e} (set CC)")),
    }
    Ok(())
}

/// One census key per refusal, keyed by MNEMONIC where the reason names one.
///
/// `m4::key` cuts at the first `:` on purpose, so the clang gate's bucket for
/// every unmodelled instruction is one line, `unmodelled instruction`. On a new
/// toolchain the question is *which instruction*, so this cuts one level deeper:
/// `opaque: unmodelled instruction: movsd (memory operand)` becomes `movsd`. A
/// reason that names no instruction keeps `m4::key`'s shape, so the two gates'
/// class lists stay comparable line by line.
pub fn census_key(r: &Refusal) -> String {
    let s = r.reason.trim();
    let s = s.strip_prefix("opaque:").map(str::trim).unwrap_or(s);
    if let Some(rest) = s.strip_prefix("unmodelled instruction:") {
        let word: String = rest
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !word.is_empty() {
            return word.to_lowercase();
        }
    }
    let low = s.to_lowercase();
    let cut = low.find(':').unwrap_or(low.len());
    // One clip rule for the whole repo (crate::m4::clip): it announces a cut with a
    // visible ellipsis, breaks on a word, and slices on a char boundary - a raw
    // `&low[..46]` silently invented `__Imag`-style fragments and panics on a
    // multi-byte character straddling byte 46.
    crate::m4::clip(low[..cut].trim(), 46)
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Unit tests: the offsets, the flag table, the census key, the guards.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-section AMD64 COFF object with a `.chks64` section at raw 60, the
    /// field layout cl 19.44 writes (no optional header, section headers at 20).
    fn object() -> Vec<u8> {
        let mut b = vec![0u8; 20 + 40 + 512];
        b[0..2].copy_from_slice(&0x8664u16.to_le_bytes());
        b[2..4].copy_from_slice(&1u16.to_le_bytes()); // NumberOfSections
        b[4..8].copy_from_slice(&0x6a_f0_0e_12u32.to_le_bytes()); // TimeDateStamp
        b[16..18].copy_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
        let h = 20;
        b[h..h + 8].copy_from_slice(b".chks64\0");
        b[h + 8..h + 12].copy_from_slice(&16u32.to_le_bytes()); // VirtualSize
        b[h + 20..h + 24].copy_from_slice(&60u32.to_le_bytes()); // PointerToRawData
        b[h + 36..h + 40].copy_from_slice(&0x0400_0040u32.to_le_bytes());
        b[60..68].copy_from_slice(&[0x23, 0x07, 0x66, 0x15, 0x27, 0x1a, 0xbf, 0x1a]);
        b[68..76].copy_from_slice(&[0x0d, 0x4f, 0x12, 0x0b, 0x06, 0xda, 0x9b, 0xb8]);
        b
    }

    #[test]
    fn msvc_stamp_offsets_pin_the_real_fields() {
        // Offsets are the whole point, and both are measured facts about what cl
        // embeds: the clock at object offset 4, and the checksum over that clock
        // at `.chks64` + 8. A guard on the wrong offset silently disables the
        // normalization and the corpus stops being reproducible - the failure
        // that made the clang corpus un-gatable once (offset 2 is NumberOfSections).
        let mut b = object();
        assert!(normalize_coff_stamp(&mut b), "stamp at offset 4 is zeroed");
        assert_eq!(&b[4..8], &[0, 0, 0, 0]);
        assert_eq!(
            &b[0..4],
            &[0x64, 0x86, 1, 0],
            "Machine and NumberOfSections untouched"
        );
        assert!(
            normalize_chks64_checksum(&mut b),
            "the 8 checksum bytes at .chks64 + 8 are zeroed"
        );
        assert_eq!(&b[68..76], &[0; 8]);
        assert_eq!(
            &b[60..68],
            &[0x23, 0x07, 0x66, 0x15, 0x27, 0x1a, 0xbf, 0x1a],
            "the file-identity half of .chks64 is not a clock field: untouched"
        );
        assert!(
            !normalize_coff_stamp(&mut b),
            "already normalized reports false"
        );
        assert!(
            !normalize_chks64_checksum(&mut b),
            "already normalized reports false"
        );

        // A non-AMD64 object survives untouched: the guard reads Machine at 0.
        let mut other = object();
        other[0..2].copy_from_slice(&0x014cu16.to_le_bytes());
        assert!(!normalize_coff_stamp(&mut other));
        assert!(!normalize_chks64_checksum(&mut other));
        assert_eq!(&other[4..8], &0x6a_f0_0e_12u32.to_le_bytes());
        assert_ne!(&other[68..76], &[0; 8]);
    }

    #[test]
    fn a_chks64_name_field_is_eight_bytes_not_nine() {
        // A COFF section name is 8 bytes. `b".chks64\0"` is exactly 8;
        // `b".text\0\0\0\0"` is 9 - the off-by-one that shows up later as a
        // section named `` `.text `` and every field after it shifted by one.
        assert_eq!(b".chks64\0".len(), 8);
        assert_eq!(b".text\0\0\0\0".len(), 9);
        let b = object();
        let (_, nsec, secbase) = coff_header(&b).unwrap();
        assert_eq!((nsec, secbase), (1, 20));
        assert!(name_is(&b[secbase..secbase + 8], ".chks64"));
        assert!(!name_is(&b[secbase..secbase + 8], ".text"));
    }

    #[test]
    fn the_two_stamp_rules_agree_on_the_offset() {
        // The clang leg keeps its own copy of the rule (crates/verify is not this
        // lane's file). Both must read object offset 4, so run both on one object
        // and require identical bytes out.
        let mut a = object();
        let mut b = object();
        assert!(normalize_coff_stamp(&mut a));
        assert!(rdec_verify::corpus::normalize_coff_stamp(&mut b));
        assert_eq!(a, b);
    }

    #[test]
    fn every_leg_has_flags_and_the_cookie_leg_is_named() {
        // The manifest records one `# flags[leg]:` line per leg, so a census
        // number is attributable to a flag set. /GS is ON for exactly one leg.
        assert_eq!(MSVC_OPTS, &["Od", "O1", "O2", "Os"]);
        for opt in MSVC_OPTS {
            let f = msvc_flags(opt);
            assert!(f.iter().any(|x| x == "/c"), "{opt}: per-file compile");
            assert!(
                f.iter().any(|x| x == "/Gw-"),
                "{opt}: /Gw stated, not assumed"
            );
            let gs = f.iter().any(|x| x == "/GS");
            let gs_off = f.iter().any(|x| x == "/GS-");
            assert!(gs ^ gs_off, "{opt}: exactly one of /GS and /GS-");
            assert_eq!(gs, *opt == "O2", "{opt}: the cookie leg is O2");
        }
    }

    /// Does this text quote a cl diagnostic code - `C` + four digits, or `LNK` +
    /// four digits, at a word boundary? Written after the naive version
    /// (`contains("C2")`) failed the leg's own `__int128` row.
    fn has_diag_code(s: &str) -> bool {
        let b = s.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            let boundary = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
            if boundary
                && b[i] == b'C'
                && b.len() > i + 4
                && b[i + 1..i + 5].iter().all(u8::is_ascii_digit)
            {
                return true;
            }
            if boundary
                && i + 7 <= b.len()
                && &b[i..i + 3] == b"LNK"
                && b[i + 3..i + 7].iter().all(u8::is_ascii_digit)
            {
                return true;
            }
            i += 1;
        }
        false
    }

    #[test]
    fn the_excluded_snippets_are_clang_only_and_named() {
        // The exclusion list is a claim about the sources: if a snippet stops
        // being clang-only, the list is stale and the leg silently shrinks. The
        // count is pinned on purpose - a row added to buy coverage must be a row
        // with a reason, and the day the leg grows a 5th exclusion is a day this
        // test is edited in the same commit as the ROADMAP ledger.
        assert_eq!(MSVC_EXCLUDED.len(), 4);
        for (name, why) in MSVC_EXCLUDED {
            assert!(why.contains("cl"), "{name}: the reason names the compiler");
            // "quotes a cl diagnostic code" - and the code family is wider than the
            // two cases in front of me: cl numbers C1000..C8xxx plus LNK####, so a
            // prefix test written as `contains("C2")` fails on the leg's own
            // `__int128` row (measured: `error C4235`, caught by this very test on
            // the first run of the commit that added it). Scan for the SHAPE.
            assert!(
                has_diag_code(why),
                "{name}: the reason quotes no cl diagnostic code: {why}"
            );
            let src = Path::new("../../corpus/src").join(format!("{name}.c"));
            // A phantom row is the silent half of the rot: `want` is
            // (sources - |excluded|) x legs, so a name with no source file shrinks
            // every census denominator by a leg's worth of objects. Loud here (no
            // compiler needed); the build-time probe is its twin with cl.
            let text = std::fs::read_to_string(&src).unwrap_or_else(|e| {
                panic!("{name}: excluded, yet {} is unreadable: {e}", src.display())
            });
            assert!(
                text.contains("__int128") || text.contains("__attribute__"),
                "{name}: excluded, yet no clang-only construct in {}",
                src.display()
            );
        }
    }

    #[test]
    fn a_long_census_key_announces_its_cut_and_breaks_on_a_word() {
        // The leg's keys got LONGER when the refusal started naming the real
        // shortage ("flag setting against a cell whose relocation names no
        // address"), which put the 46-char clip under load. A key that is cut must
        // SAY it is cut and must not end mid-identifier: the clang gate invented
        // `__Imag` out of `__ImageBase` exactly this way.
        let k = census_key(&Refusal {
            at: 0,
            reason: "opaque: flag setting against a cell whose relocation names no address"
                .to_string(),
        });
        assert!(
            k.ends_with('\u{2026}'),
            "a cut key must announce the cut: {k:?}"
        );
        let body = k.trim_end_matches('…');
        assert!(
            body.ends_with("relocation"),
            "cut at a word boundary: {k:?}"
        );
        assert!(
            !body.ends_with(char::is_whitespace),
            "no trailing space before the cut: {k:?}"
        );
    }

    #[test]
    fn census_key_reaches_the_mnemonic_behind_the_class() {
        // `m4::key` cuts at the first ':' (that is its job). The MSVC census needs
        // the mnemonic one level deeper, or every refused instruction is one line.
        let mk = |reason: &str| Refusal {
            at: 0,
            reason: reason.to_string(),
        };
        assert_eq!(
            census_key(&mk("opaque: unmodelled instruction: movsd (memory)")),
            "movsd"
        );
        assert_eq!(
            census_key(&mk("opaque: unmodelled instruction: paddd")),
            "paddd"
        );
        assert_eq!(
            census_key(&mk("opaque: indirect call: no signature model")),
            "indirect call"
        );
        assert_eq!(
            census_key(&mk("flags value v7 survived flag elimination")),
            "flags value v7 survived flag elimination"
        );
        // A reason that names nothing yields the empty key, which the census
        // counts as `keyless` and FAILS on. That is the silent-drop guard.
        assert_eq!(census_key(&mk("opaque:")), "");
    }

    #[test]
    fn fnv_is_stable_and_sensitive() {
        assert_eq!(fnv1a64(b"abc"), fnv1a64(b"abc"));
        assert_ne!(fnv1a64(b"abc"), fnv1a64(b"abd"));
    }
}
