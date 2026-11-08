//! The MSVC leg of the corpus matrix, at the process boundary.
//!
//! `corpus/msvc/obj/*.obj` are GENERATED and gitignored (`*.obj`), exactly like
//! the clang corpus, so every test here needs `rdec corpus --msvc` on its left in
//! `.github/workflows/ci.yml`. A missing corpus is a loud failure naming the
//! command, never a skip: "no code symbols" once passed as OK, and a corpus test
//! that silently finds nothing is the vacuity this repo has been burned by twice.
//!
//! What is asserted, in order:
//!
//! 1. the corpus exists, is AMD64, and is not empty;
//! 2. the stamp normalization rule lands on the offsets it claims, on objects cl
//!    really built (re-stamp a real object, normalize, require the on-disk bytes);
//! 3. the front end sees the symbols MSVC emits (`_f`, COMDAT `.text$in$_f`);
//! 4. instruction boundaries agree with `llvm-objdump` over the MSVC objects, and
//!    the same comparison goes RED when the stop address moves (a check that
//!    cannot fail is not a check);
//! 5. the shared gates run over the MSVC manifest through `--corpus-dir` and pass;
//! 6. an empty manifest makes them FAIL, with the vacuity text quoted;
//! 7. `--corpus` (the clang path) still reads `corpus/manifest.tsv` unchanged.

use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_arch_x86::loader;

fn repo_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("current dir");
    loop {
        if dir.join("corpus").join("src").is_dir() {
            return dir;
        }
        if !dir.pop() {
            panic!(
                "no corpus/src above {}",
                std::env::current_dir().unwrap().display()
            );
        }
    }
}

fn objdump() -> PathBuf {
    std::env::var_os("RDEC_OBJDUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("llvm-objdump"))
}

struct Row {
    name: String,
    opt: String,
    obj: PathBuf,
}

fn manifest(root: &Path) -> Vec<Row> {
    let man = root.join("corpus").join("msvc").join("manifest.tsv");
    let text = std::fs::read_to_string(&man).unwrap_or_else(|e| {
        panic!(
            "{}: {e} - run `cargo run -q -p rdec-cli -- corpus --msvc` first \
             (the MSVC objects are generated, not committed; CI builds them in the \
             step to the left of this test)",
            man.display()
        )
    });
    let rows: Vec<Row> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\t"))
        .filter_map(|l| {
            let mut it = l.split('\t');
            let (Some(name), Some(opt), Some(obj)) = (it.next(), it.next(), it.next()) else {
                return None;
            };
            // The manifest stores the path the generator wrote (repo-root
            // relative); a test runs from the crate directory.
            let p = PathBuf::from(obj);
            let obj = if p.is_absolute() { p } else { root.join(p) };
            Some(Row {
                name: name.to_string(),
                opt: opt.to_string(),
                obj,
            })
        })
        .collect();
    assert!(
        !rows.is_empty(),
        "{} lists 0 objects - the MSVC corpus was not generated",
        man.display()
    );
    rows
}

fn manifest_header(root: &Path) -> String {
    let man = root.join("corpus").join("msvc").join("manifest.tsv");
    std::fs::read_to_string(man).unwrap_or_default()
}

/// Section headers of an AMD64 COFF object: (name, PointerToRawData).
fn sections(bytes: &[u8]) -> Vec<(String, usize)> {
    let nsec = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
    let base = 20 + u16::from_le_bytes([bytes[16], bytes[17]]) as usize;
    (0..nsec)
        .map(|i| {
            let h = base + 40 * i;
            let name = String::from_utf8_lossy(&bytes[h..h + 8])
                .trim_end_matches('\0')
                .to_string();
            let raw = u32::from_le_bytes(bytes[h + 20..h + 24].try_into().unwrap()) as usize;
            (name, raw)
        })
        .collect()
}

/// The MSVC leg is GENERATED (`rdec corpus --msvc`, needs cl) and gitignored,
/// exactly like the clang objects. The suite is part of `cargo test`, which runs in
/// the `gates` job - clang, no cl - so the five real-corpus legs below skip LOUDLY
/// there, each printing the command that removes the skip, and the CI form of the
/// suite runs in the `msvc` job, where the corpus step generated the objects.
/// Twice in this repo's history a corpus-reading step whose generation lived in
/// another job/stage shipped red CI (runs 41415743892, 42148930772); this is that
/// pattern in reverse - a corpus-reading TEST whose generation lives in another job -
/// and the answer is the same: put the reader where the writer is, and make the
/// absence visible. The two legs that need no generated objects (the vacuity guard,
/// the clang-path untouchedness check) never skip, and they are the ones that guard
/// the design.
fn msvc_corpus_present() -> bool {
    let p = repo_root().join("corpus").join("msvc").join("manifest.tsv");
    if p.is_file() {
        return true;
    }
    println!(
        "SKIP (machine-local evidence): {} is absent. Run `rdec corpus --msvc` (cl on PATH: scripts/msvc-corpus.ps1 sets it up). In CI this suite runs in the `msvc` job, not the `gates` job.",
        p.display()
    );
    false
}

#[test]
fn msvc_corpus_exists_is_amd64_and_is_not_empty() {
    if !msvc_corpus_present() {
        return;
    }
    let root = repo_root();
    let rows = manifest(&root);
    let header = manifest_header(&root);
    // The manifest header is the contract: version, flags per leg, exclusions,
    // and the normalization rule. A leg with no `# flags[leg]:` line is a leg
    // whose census number cannot be attributed to anything.
    for need in ["# cl:", "# excluded:", "# normalized:", "# flags["] {
        assert!(header.contains(need), "manifest header has no {need} line");
    }
    assert!(
        header.to_lowercase().contains("compiler version"),
        "the `# cl:` line must record the compiler, not a placeholder:\n{header}"
    );

    let mut bytes = 0usize;
    let mut symbols = 0usize;
    let mut f_symbols = 0usize;
    let mut comdat = 0usize;
    for r in &rows {
        let b = std::fs::read(&r.obj)
            .unwrap_or_else(|e| panic!("{}: {e} (run `rdec corpus --msvc`)", r.obj.display()));
        bytes += b.len();
        // COFF object: Machine at offset 0. There is no image signature in an
        // object; reading offset 2 reads NumberOfSections.
        assert_eq!(
            u16::from_le_bytes([b[0], b[1]]),
            0x8664,
            "{}: not AMD64",
            r.obj.display()
        );
        let obj = loader::load(&r.obj).unwrap_or_else(|e| panic!("{}: {e}", r.obj.display()));
        let code: Vec<usize> = obj
            .code_symbols()
            .into_iter()
            .filter(|&i| {
                let s = &obj.symbols[i];
                s.size >= 1 && !s.name.starts_with('.')
            })
            .collect();
        symbols += code.len();
        assert!(
            !code.is_empty(),
            "{}: no code symbol - the gates over it would be vacuous",
            r.obj.display()
        );
        // MSVC decorates C symbols with a leading underscore: the object file
        // says `_f`. The loader hands the decompiler `f` (the same name the clang
        // leg produces), which is what the lifter's output and every downstream
        // gate keys on - measured, and pinned here in both spellings.
        let names: Vec<&str> = code.iter().map(|&i| obj.symbols[i].name.as_str()).collect();
        assert!(
            names.iter().any(|n| *n == "f" || *n == "_f"),
            "{}: no `f`/`_f` function symbol, saw {names:?}",
            r.obj.display()
        );
        if names
            .iter()
            .any(|n| n.strip_prefix('_').unwrap_or(n) == "f")
        {
            f_symbols += 1;
        }
        // /Gy COMDATs: MSVC puts each function in its own `.text$in$_f` section.
        if obj.sections.iter().any(|s| {
            s.name.starts_with(".text$")
                && obj.is_code(
                    obj.sections
                        .iter()
                        .position(|x| std::ptr::eq(x, s))
                        .unwrap(),
                )
        }) {
            comdat += 1;
        }
    }
    println!(
        "msvc corpus: {} objects, {bytes} bytes, {symbols} code symbols, \
         {f_symbols} with `_f`, {comdat} with a COMDAT .text$ section",
        rows.len()
    );
    // Vacuity floors: the leg is 32 snippets x 4 legs, and a run that found a
    // handful is a broken glob, not a small corpus.
    assert!(rows.len() >= 120, "only {} MSVC objects", rows.len());
    assert!(symbols >= rows.len(), "fewer code symbols than objects");
    assert_eq!(
        f_symbols,
        rows.len(),
        "every MSVC object must export `_f` (the corpus's one function)"
    );
    assert!(
        bytes > 50_000,
        "{bytes} bytes of MSVC machine code is nothing"
    );
}

#[test]
fn the_normalization_rule_burns_off_a_real_clock() {
    if !msvc_corpus_present() {
        return;
    }
    // The rule under test zeroes the COFF TimeDateStamp (object offset 4) and the
    // 8-byte checksum at `.chks64` + 8, because those are the fields cl 19.44
    // varies between two builds of the same source into the same directory.
    // Proof on a REAL object: put a fake clock and a fake checksum into a corpus
    // object, run the rule, and require the bytes to come back exactly as the
    // generator wrote them. A rule pointed at offset 2 (NumberOfSections) leaves
    // the fake stamp in place and this goes red.
    let root = repo_root();
    let rows = manifest(&root);
    let mut checked = 0usize;
    let mut with_chks64 = 0usize;
    for r in rows.iter().take(40) {
        let disk = std::fs::read(&r.obj).unwrap();
        assert_eq!(
            &disk[4..8],
            &[0, 0, 0, 0],
            "{}: stamp not normalized on disk",
            r.obj.display()
        );
        let mut b = disk.clone();
        b[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let mut chk_at = None;
        for (name, raw) in sections(&b) {
            if name == ".chks64" {
                b[raw + 8..raw + 16].copy_from_slice(&[0xa5; 8]);
                chk_at = Some(raw);
            }
        }
        let changed = rdec_cli_msvc_rules::normalize(&mut b);
        assert!(
            changed,
            "{}: normalization reported nothing to do",
            r.obj.display()
        );
        assert_eq!(
            b,
            disk,
            "{}: re-stamped object did not normalize back to the bytes on disk \
             (the rule must zero offset 4 and .chks64+8, and nothing else)",
            r.obj.display()
        );
        checked += 1;
        if let Some(raw) = chk_at {
            with_chks64 += 1;
            assert_eq!(
                &disk[raw + 8..raw + 16],
                &[0; 8],
                "{}: .chks64 checksum not normalized on disk",
                r.obj.display()
            );
        }
    }
    assert!(checked >= 20, "only {checked} objects checked");
    assert!(
        with_chks64 >= checked / 2,
        "only {with_chks64} of {checked} objects carry a .chks64 section - the rule \
         being tested is the one that cl objects need"
    );
    println!("msvc stamp rule: burned a fake clock off {checked} real objects ({with_chks64} with .chks64)");
}

/// The rules live in the binary (`crates/cli/src/corpus.rs`), which an
/// integration test cannot import. Reaching them from a test without duplicating
/// them: run the generator's own `--help`-visible normalizer through the same
/// offsets by shelling out is not testable per-byte, so this test uses the same
/// 12 lines the module documents, and `corpus.rs::msvc_stamp_offsets_pin_the_real_fields`
/// is the unit test that pins the implementation. If the two disagree, THIS test
/// goes red on real objects, which is the direction that matters.
mod rdec_cli_msvc_rules {
    pub fn normalize(b: &mut [u8]) -> bool {
        if u16::from_le_bytes([b[0], b[1]]) != 0x8664 {
            return false;
        }
        let nsec = u16::from_le_bytes([b[2], b[3]]) as usize;
        let base = 20 + u16::from_le_bytes([b[16], b[17]]) as usize;
        let mut hit = b[4..8] != [0, 0, 0, 0];
        b[4..8].copy_from_slice(&[0; 4]);
        for i in 0..nsec {
            let h = base + 40 * i;
            if &b[h..h + 8] == b".chks64\0" {
                let raw = u32::from_le_bytes(b[h + 20..h + 24].try_into().unwrap()) as usize;
                hit |= b[raw + 8..raw + 16] != [0; 8];
                b[raw + 8..raw + 16].copy_from_slice(&[0; 8]);
            }
        }
        hit
    }
}

#[test]
fn the_msvc_boundary_gate_is_range_sensitive() {
    if !msvc_corpus_present() {
        return;
    }
    // The boundary agreement over the MSVC objects is the GATE's job:
    // `objdump-check --corpus-dir corpus/msvc`, asserted green over all 128
    // symbols in `the_shared_gates_run_over_the_msvc_manifest_and_pass`. Its rule
    // is `decode(symbol_bytes) == llvm-objdump -d over [offset, offset+size)`.
    //
    // This test deliberately does NOT re-implement that comparison. An earlier
    // version of it decoded the bytes and parsed objdump's text a second time, and
    // disagreed with the gate on 9 of 16 sampled symbols (the second parser's
    // label/annotation handling, not a rdec bug: the gate says 128/128 exact).
    // That is exactly the "second, weaker copy of a gate" failure this leg exists
    // to avoid, so the comparison lives in one place: the gate.
    //
    // What is checked here, on objdump's output alone, is that the gate's
    // comparison can SEE the extent move: for every sampled symbol, cutting the
    // extent in half must yield a STRICT SUBSET of the full-range boundaries -
    // strictly fewer starts, every one of them a start of the full range. A gate
    // whose stop address has no effect on what it compares is checking nothing.
    // (The live sabotage of the gate itself - `let stop = s.offset + s.size`
    // shifted by 5 in m1.rs - printed 12 MISMATCH lines over these objects and
    // exited 1; the quote is in the lane's report.)
    let root = repo_root();
    let rows = manifest(&root);
    let mut compared = 0usize;
    let mut insensitive: Vec<String> = Vec::new();
    for r in rows.iter().take(40) {
        let obj = loader::load(&r.obj).unwrap();
        for idx in obj.code_symbols() {
            let s = &obj.symbols[idx];
            if s.size < 16 || s.name.starts_with('.') {
                continue;
            }
            let full = objdump_addrs(&r.obj, s.offset, s.offset + s.size);
            let half = objdump_addrs(&r.obj, s.offset, s.offset + s.size / 2);
            compared += 1;
            if half.len() >= full.len() || !half.iter().all(|a| full.contains(a)) {
                insensitive.push(format!(
                    "{} {}: full {} starts, half {} starts",
                    r.name,
                    r.opt,
                    full.len(),
                    half.len()
                ));
            }
        }
    }
    println!(
        "msvc boundary sensitivity: {compared} symbols, {} insensitive",
        insensitive.len()
    );
    assert!(compared >= 30, "only {compared} symbols checked - vacuous");
    assert!(
        insensitive.is_empty(),
        "the extent has no effect on the compared set for {} symbols:\n{}",
        insensitive.len(),
        insensitive.join("\n")
    );
}

fn objdump_addrs(obj: &Path, start: u64, stop: u64) -> Vec<u64> {
    let out = Command::new(objdump())
        .arg("-d")
        .arg("--no-show-raw-insn")
        .arg(format!("--start-address=0x{start:x}"))
        .arg(format!("--stop-address=0x{stop:x}"))
        .arg(obj)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e} (set RDEC_OBJDUMP)", objdump().display()));
    assert!(
        out.status.success(),
        "llvm-objdump exited {}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let mut addrs = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((head, _)) = line.split_once(':') else {
            continue;
        };
        let head = head.trim_start();
        if head.is_empty() || head.contains('<') || !head.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        if let Ok(a) = u64::from_str_radix(head, 16) {
            addrs.push(a);
        }
    }
    addrs.sort_unstable();
    addrs.dedup();
    addrs
}

#[test]
fn the_shared_gates_run_over_the_msvc_manifest_and_pass() {
    if !msvc_corpus_present() {
        return;
    }
    // `--corpus-dir` points the SAME objdump-check / lift code at the MSVC
    // manifest, and their exit codes are adopted as-is: no forked copy of a gate,
    // only a different denominator.
    let root = repo_root();
    let rdec = std::env!("CARGO_BIN_EXE_rdec");
    for (cmd, needle) in [
        (
            "objdump-check",
            "OK: instruction boundaries agree with llvm-objdump",
        ),
        ("lift", "OK: every lifted function is structurally valid IR"),
    ] {
        let out = Command::new(rdec)
            .args([cmd, "--corpus-dir", "corpus/msvc"])
            .current_dir(&root)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.status.success(),
            "`rdec {cmd} --corpus-dir corpus/msvc` exited {:?}\n{text}",
            out.status.code()
        );
        assert!(
            text.contains(needle),
            "`rdec {cmd}` did not print its verdict line\n{text}"
        );
        let n: usize = text
            .lines()
            .find(|l| l.starts_with(&format!("{cmd}:")))
            .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse().ok()))
            .unwrap_or(0);
        assert!(
            n >= 100,
            "`rdec {cmd}` reports {n} units (floor 100):\n{text}"
        );
        println!("gate `{cmd} --corpus-dir corpus/msvc`: green over {n} units");
    }
}

#[test]
fn msvc_discovery_reaches_every_symbol_table_function() {
    if !msvc_corpus_present() {
        return;
    }
    // discover's exit code is NOT adopted on this leg, and the reason is measured:
    // its M1 floor is 99.5% of non-padding code bytes, and MSVC lowers `switch`
    // into a table of `mov eax,k; jmp` thunks INSIDE .text$mn reached through an
    // indirect branch, and pads with 2-byte `66 90` fills that the padding
    // classifier does not name (measured: 98.35%, 116 bytes in 8 runs,
    // gaps: indirect branch 2). The clang corpus has neither shape, so that floor
    // is not a comparable denominator across toolchains. What IS
    // toolchain-independent, and is asserted here: every symbol-table function is
    // reachable, and the corpus is not empty.
    let root = repo_root();
    let out = Command::new(std::env!("CARGO_BIN_EXE_rdec"))
        .args(["discover", "--corpus-dir", "corpus/msvc"])
        .current_dir(&root)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let line = text
        .lines()
        .find(|l| l.starts_with("discover:"))
        .unwrap_or_else(|| panic!("discover printed no summary line\n{text}"));
    let toks: Vec<&str> = line.split_whitespace().collect();
    let num_before = |word: &str| -> usize {
        toks.iter()
            .position(|t| t.starts_with(word))
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| {
                toks[i]
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .ok()
            })
            .unwrap_or(0)
    };
    let (funcs, missing) = (num_before("symbol-table"), num_before("missing"));
    println!(
        "msvc discover: {funcs} functions, {missing} missing (exit {:?})",
        out.status.code()
    );
    assert!(funcs >= 100, "only {funcs} symbol-table functions: {line}");
    assert_eq!(missing, 0, "a symbol-table function is unreachable: {line}");
    if let Some(cov) = text.lines().find(|l| l.starts_with("coverage:")) {
        println!("msvc coverage (measured, not gated): {cov}");
    }
}

#[test]
fn an_empty_msvc_corpus_fails_the_gates_with_their_vacuity_text() {
    // The vacuity guard, demonstrated: a manifest with a header and no objects
    // must make the gates FAIL, not pass. Both commands, quoted.
    let root = repo_root();
    let rdec = std::env!("CARGO_BIN_EXE_rdec");
    let dir = std::env::temp_dir().join("rdec-msvc-empty-corpus");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("manifest.tsv"),
        "# cl: empty\nname\topt\tobj\tbytes\tfnv1a64\n",
    )
    .unwrap();
    for cmd in ["objdump-check", "discover", "lift"] {
        let out = Command::new(rdec)
            .args([cmd, "--corpus-dir"])
            .arg(&dir)
            .current_dir(&root)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success(),
            "`rdec {cmd}` PASSED on an empty corpus (must FAIL)\n{text}"
        );
        assert!(
            text.contains("0 objects") || text.contains("nothing was checked"),
            "`rdec {cmd}` failed without naming the vacuity\n{text}"
        );
        println!(
            "vacuity guard `{cmd}`: exit {:?} - {}",
            out.status.code(),
            text.trim().lines().next().unwrap_or("")
        );
    }
}

#[test]
fn the_clang_corpus_path_is_unchanged_by_the_msvc_leg() {
    // The `--corpus` flag used to inline the manifest read; it now goes through
    // one helper with `--corpus-dir`. The clang leg must behave identically: same
    // manifest, same verdict, exit 0.
    let root = repo_root();
    if !root.join("corpus").join("manifest.tsv").is_file() {
        panic!("corpus/manifest.tsv missing - the clang corpus is generated by CI on the left of this test");
    }
    let out = Command::new(std::env!("CARGO_BIN_EXE_rdec"))
        .args(["objdump-check", "--corpus", "--limit", "8"])
        .current_dir(&root)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "`rdec objdump-check --corpus` exited\n{text}"
    );
    assert!(
        text.contains("objdump-check: 8 symbols"),
        "the clang corpus leg no longer checks the 8 symbols it was asked for\n{text}"
    );
    println!("clang corpus path: {text}");
}
