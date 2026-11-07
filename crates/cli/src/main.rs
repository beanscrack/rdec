//! rdec — command line driver.
//!
//! M0 gives the two things the phase needs: the T1 flag oracle, and a self-test
//! that proves the IR verifier actually rejects broken functions. `disasm`,
//! `ir`, `cfg` and `decompile` land with M1/M2.

use rdec_core::ir::*;
use rdec_core::ty::{Sx, Ty, Width};
use rdec_core::verify;
use rdec_verify::{hw, model_shim, Sabotage};

mod fuzz;
mod m1;
mod m4;
mod m6;
mod m8;
mod t2;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rc = match cmd {
        "flags" => cmd_flags(&args),
        "selftest" => cmd_selftest(),
        "corpus" => {
            if args.iter().any(|a| a == "--cpp") {
                m8::cmd_corpus_cpp(&args[1..])
            } else {
                cmd_corpus(&args[1..])
            }
        }
        "demangle" => match m8::cmd_demangle(&args) {
            Ok(rc) => rc,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        },
        "types" => m6::cmd_types(&args),
        "cpp" => match m8::cmd_cpp(&args) {
            Ok(rc) => rc,
            Err(e) => {
                eprintln!("cpp: {e}");
                1
            }
        },
        "disasm" => m1::cmd_disasm(&args),
        "c" => match m4::cmd_c(&args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("c: {e}");
                2
            }
        },
        "t2" => {
            let r = if args.iter().any(|a| a == "--cpp") {
                t2::cmd_t2_cpp(&args)
            } else {
                t2::cmd_t2(&args)
            };
            match r {
                Ok(code) => code,
                Err(e) => {
                    eprintln!("t2: {e}");
                    2
                }
            }
        }
        "lift" => match m1::cmd_lift(&args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("lift: {e}");
                2
            }
        },
        "discover" => match m1::cmd_discover(&args) {
            Ok(rc) => rc,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        },
        "objdump-check" => match m1::cmd_objdump_check(&args) {
            Ok(rc) => rc,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        },
        "pe-info" => cmd_pe_info(&args),
        "pe-functions" => cmd_pe_functions(&args),
        "fuzz" => match fuzz::cmd_fuzz(&args) {
            Ok(rc) => rc,
            Err(e) => {
                eprintln!("fuzz: {e}");
                2
            }
        },
        "status" => cmd_status(),
        "probe" => cmd_probe(&args),
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_help();
            2
        }
    };
    std::process::exit(rc);
}

fn print_help() {
    println!(
        "rdec — from-scratch x86-64 -> C decompiler\n\
         \n\
         commands:\n\
         \x20 flags [--full N] [--sabotage]   T1 oracle: flag model vs real EFLAGS\n\
         \x20 probe <op> <w> <a> <b>          print model+hardware flags for one case\n\
         \x20 selftest                        prove the IR verifier rejects broken IR\n\
         \x20 corpus [--out DIR] [--opt O0,O2]  build the clang corpus + hashed manifest\n\
         \x20 corpus --cpp [--out DIR]         build the C++ corpus (M8, clang++ MSVC ABI)\n\
         \x20 demangle --cpp                  M8 gate: MSVC C++ names vs llvm-nm\n\
         \x20 types --corpus                  M6 gate: recovered signatures vs corpus/truth.tsv;\n\
         \x20   [--explain NAME]                claims printed with their machine-fact witnesses\n\
         \x20 types --selftest                  the forcing drill: rule probes + register/admin gates\n\
         \x20 cpp --corpus                    M8 gate: the C++ corpus through the C\n\
         \x20                                   pipeline - demangled names, emitted C,\n\
         \x20                                   refusals and gaps by class\n\
         \x20 t2 [--corpus] [--limit N]        T2 differential gate: decompiled C vs\n\
         \x20                                   the machine code, same inputs, RAX\n\
         \x20 t2 --cpp [--limit N]             T2 over the C++ corpus: RAX class\n\
         \x20                                   (free/static integer functions),\n\
         \x20                                   window class (member functions,\n\
         \x20                                   `this` over a preloaded window)\n\
         \x20 fuzz [--iterations N] [--seed S]  hostile-input gate: generated objects and
         \x20      [--out DIR] [--max-seconds N]  byte streams through the real pipeline
         \x20                                   (load, demangle, discover, lift, passes,
         \x20                                   verify, emit C, interpreter). Per-stage
         \x20                                   census; exit 1 on any panic or hang
         \x20 pe-info <exe>                   M7: PE32+ image headers - image base,
         \x20                                   entry RVA, subsystem, section table
         \x20                                   (name/vaddr/vsize/rawptr/rawsize/flags);
         \x20                                   refuses non-PE by field, exit 2
         \x20 pe-functions <exe>            M7: .pdata SEH unwind table as the
         \x20                                   symbol-free function-start source:
         \x20                                   begin..end/unwind/flags/first_code
         \x20                                   per RUNTIME_FUNCTION, flags and
         \x20                                   first-code census, chaining refusals
         \x20                                   counted and named; exit 2 on refusal
         \x20 status                          roadmap section 9's table, from commands\n\
         \n\
         coming next (M1): disasm <obj> [symbol]"
    );
}

fn cmd_flags(args: &[String]) -> i32 {
    let full = args.iter().position(|a| a == "--full").map(|i| {
        args.get(i + 1)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(20_000)
    });
    let sabotage = args.iter().any(|a| a == "--sabotage");
    let sab = sabotage.then_some(Sabotage {
        op: "add",
        w: Width::W32,
        flag: rdec_core::ir::Flag::OF,
    });
    let t0 = std::time::Instant::now();
    let rep = rdec_verify::check(sab, full.unwrap_or(0));
    let dt = t0.elapsed();
    println!("{}", rep.summary());
    println!("elapsed {dt:?}");
    for m in rep.mismatches.iter().take(40) {
        println!("  MISMATCH {m}");
    }
    if rep.mismatches.len() > 40 {
        println!("  ... and {} more", rep.mismatches.len() - 40);
    }
    if rep.ok() {
        println!("OK: flag model matches hardware on every probe");
        0
    } else {
        println!("FAIL");
        1
    }
}

fn cmd_probe(args: &[String]) -> i32 {
    let get = |i: usize| args.get(i).cloned().unwrap_or_default();
    let op = get(1);
    let w: u32 = get(2).parse().unwrap_or(64);
    let a = u64::from_str_radix(get(3).trim_start_matches("0x"), 16).unwrap_or(0);
    let b = u64::from_str_radix(get(4).trim_start_matches("0x"), 16).unwrap_or(0);
    let Some(width) = Width::from_bits(w) else {
        eprintln!("width must be 8, 16, 32 or 64");
        return 2;
    };
    let seed = hw::ALL_SEED;
    let (mres, mdef, hres, hflags) = match op.as_str() {
        "add" => {
            let (m, d) = model_shim::add(width, a, b);
            let (h, f) = hw::add(width, a, b, seed);
            (m, d, h, f)
        }
        "sub" => {
            let (m, d) = model_shim::sub(width, a, b);
            let (h, f) = hw::sub(width, a, b, seed);
            (m, d, h, f)
        }
        "and" => {
            let (m, d) = model_shim::logic(width, a, b, rdec_arch_x86::flags::LogicOp::And);
            let (h, f) = hw::and(width, a, b, seed);
            (m, d, h, f)
        }
        _ => {
            eprintln!("probe supports add|sub|and");
            return 2;
        }
    };
    println!("{op}.{w} a={a:#x} b={b:#x}");
    println!("  result  model={mres:#x}  hw={hres:#x}");
    for fl in rdec_arch_x86::flags::FlagDef::ALL {
        let mask = rdec_verify::flag_mask(fl);
        let model = match mdef.get(fl) {
            Some(v) => format!("{v}"),
            None => "undef".into(),
        };
        println!("  {fl}      model={model:<5}  hw={}", hw::bit(hflags, mask));
    }
    0
}

/// M0 gate: the verifier must accept well-formed IR and reject each of a set of
/// hand-broken functions. An oracle that cannot fail is not an oracle.
fn cmd_selftest() -> i32 {
    let mut rc = 0;

    // 1. A valid function must verify.
    let f = tiny_fn();
    match verify::verify(&f) {
        Ok(()) => println!("ok    valid function verifies"),
        Err(e) => {
            println!("FAIL  valid function rejected: {e:?}");
            rc = 1;
        }
    }

    // 2. Each sabotage must be caught, by name.
    type BrokenCase = (&'static str, fn() -> Function);
    let cases: Vec<BrokenCase> = vec![
        ("use of an undefined value", broken_undef_use),
        ("value defined twice", broken_double_def),
        ("branch to a missing block", broken_bad_target),
        ("terminator branch on a non-bool", broken_bad_branch_ty),
    ];
    for (name, mk) in cases {
        let f = mk();
        match verify::verify(&f) {
            Ok(()) => {
                println!("FAIL  verifier accepted: {name}");
                rc = 1;
            }
            Err(errs) => {
                let hit = errs.iter().any(|e| !e.starts_with("warn:"));
                if hit {
                    println!(
                        "ok    verifier rejected: {name}  ({}: {})",
                        errs.len(),
                        errs[0]
                    );
                } else {
                    println!("FAIL  only warnings for: {name}: {errs:?}");
                    rc = 1;
                }
            }
        }
    }
    if rc == 0 {
        println!("selftest OK: the verifier can fail, on the right things");
    }
    rc
}

/// `rdec corpus [--out DIR] [--opt O0,O2]` — generate and compile the corpus.
/// M0's gate is "the corpus builds"; later gates read `manifest.tsv`.
fn cmd_corpus(args: &[String]) -> i32 {
    use std::path::Path;
    let mut out = "corpus".to_string();
    let mut opts: Vec<String> = rdec_verify::corpus::OPTS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 2;
            }
            "--opt" if i + 1 < args.len() => {
                opts = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                i += 2;
            }
            other => {
                eprintln!("corpus: ignoring unknown arg {other}");
                i += 1;
            }
        }
    }
    let opt_refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
    let clang = rdec_verify::corpus::clang_path();
    let rep = rdec_verify::corpus::generate(Path::new(&out), &opt_refs, &clang);
    let want = rdec_verify::corpus::SNIPPETS.len() * opt_refs.len();
    for f in &rep.failures {
        println!("FAIL  {f}");
    }
    let total: u64 = rep.entries.iter().map(|e| e.bytes).sum();
    println!(
        "{} snippets x {} opt levels -> {}/{} objects, {} bytes of machine code",
        rdec_verify::corpus::SNIPPETS.len(),
        opt_refs.len(),
        rep.built,
        want,
        total
    );
    println!("manifest: {out}/manifest.tsv");
    if rep.failures.is_empty() && rep.built == want {
        println!("corpus OK");
        0
    } else {
        1
    }
}

/// `int f(int a, int b) { return a + b; }` in IR.
fn tiny_fn() -> Function {
    let mut f = Function::new("f", 0x1000);
    let bb = f.entry;
    let a = f.emit(bb, Op::Param, Ty::int(Width::W32, Sx::Signed), 0x1000);
    let b = f.emit(bb, Op::Param, Ty::int(Width::W32, Sx::Signed), 0x1001);
    let s = f.emit(bb, Op::Bin(BinOp::Add, a, b), Ty::u32(), 0x1002);
    f.block_mut(bb).term = Terminator::Return(s);
    f.params = vec![a, b];
    f.ret_ty = Ty::int(Width::W32, Sx::Signed);
    f
}

fn broken_undef_use() -> Function {
    let mut f = tiny_fn();
    let bb = f.entry;
    // Reference a value id beyond the arena.
    let ghost = f.next_value + 7;
    f.block_mut(bb).term = Terminator::Return(ghost);
    f
}

fn broken_double_def() -> Function {
    let mut f = tiny_fn();
    let bb = f.entry;
    let (a, b) = (f.params[0], f.params[1]);
    let s = f.block(bb).instrs[2].dest;
    // Define %s a second time: not single-assignment.
    f.block_mut(bb)
        .push(Instr::new(s, Op::Bin(BinOp::Sub, a, b), Ty::u32(), 0x1003));
    f
}

fn broken_bad_target() -> Function {
    let mut f = tiny_fn();
    let bb = f.entry;
    f.block_mut(bb).term = Terminator::Jump(99);
    f
}

fn broken_bad_branch_ty() -> Function {
    let mut f = tiny_fn();
    let bb = f.entry;
    let b2 = f.new_block(0x1010);
    let s = f.block(bb).instrs[2].dest; // an int32, not a bool
    f.block_mut(bb).term = Terminator::Branch {
        cond: s,
        then_bb: b2,
        else_bb: bb,
    };
    f.block_mut(b2).term = Terminator::Return(s);
    f
}

// ---------------------------------------------------------------------------
// status: roadmap section 9's measured table, generated from commands.
// ---------------------------------------------------------------------------

/// Every row is the result of running something, so the state section of the
/// roadmap is never hand-written again (and a stale claim cannot survive a
/// commit). `rdec status` is what CI prints.
fn cmd_status() -> i32 {
    let mut rows: Vec<(String, String, bool)> = Vec::new();

    let git = sh("git", ["rev-parse", "--short", "HEAD"]);
    let dirty = sh("git", ["status", "--porcelain"]);
    rows.push((
        "tree".into(),
        format!(
            "{}{}",
            git.1.trim(),
            if dirty.1.trim().is_empty() {
                ""
            } else {
                " (dirty)"
            }
        ),
        git.0,
    ));

    let rustc = sh("rustc", ["-V"]);
    rows.push(("toolchain".into(), rustc.1.trim().into(), rustc.0));

    let fmt = sh("cargo", ["fmt", "--check"]);
    rows.push((
        "cargo fmt --check".into(),
        if fmt.0 {
            "clean".into()
        } else {
            "DIRTY".into()
        },
        fmt.0,
    ));

    let clp = sh("cargo", ["clippy", "--all-targets", "--", "-D", "warnings"]);
    rows.push((
        "clippy -D warnings".into(),
        if clp.0 {
            "clean".into()
        } else {
            "FINDINGS".into()
        },
        clp.0,
    ));

    let tst = sh("cargo", ["test", "--no-fail-fast"]);
    let mut passed = 0usize;
    let mut failed = 0usize;
    for l in tst.1.lines() {
        if let Some(rest) = l.trim().strip_prefix("test result: ") {
            // "test result: ok. 15 passed; 0 failed; ..." -> ["ok.","15","passed;","0",...]
            let a: Vec<&str> = rest.split_whitespace().collect();
            if a.len() >= 4 {
                passed += a[1].parse().unwrap_or(0);
                failed += a[3].parse().unwrap_or(0);
            }
        }
    }
    rows.push((
        "cargo test".into(),
        format!("{} passed, {} failed", passed, failed),
        tst.0 && failed == 0 && passed > 0,
    ));

    let exe = std::env::current_exe().unwrap_or_else(|_| "cargo".into());
    let self_cmd: Vec<String> = if exe.file_stem().map(|s| s == "cargo").unwrap_or(false) {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "selftest".into(),
        ]
    } else {
        vec!["selftest".into()]
    };
    let st = sh(&exe, &self_cmd);
    rows.push((
        "rdec selftest".into(),
        st.1.lines().last().unwrap_or("").trim().into(),
        st.0,
    ));

    let fl_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "flags".into(),
        ]
    } else {
        vec!["flags".into()]
    };
    let fl = sh(&exe, &fl_cmd);
    rows.push((
        "rdec flags (grid)".into(),
        fl.1.lines().last().unwrap_or("").trim().into(),
        fl.0,
    ));

    rows.push(corpus_row());
    let ob_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "objdump-check".into(),
            "--corpus".into(),
            "--limit".into(),
            "60".into(),
        ]
    } else {
        vec![
            "objdump-check".into(),
            "--corpus".into(),
            "--limit".into(),
            "60".into(),
        ]
    };
    let ob = sh(&exe, &ob_cmd);
    rows.push((
        "objdump-check (M1: boundaries vs llvm-objdump)".into(),
        ob.1.lines().last().unwrap_or("").trim().into(),
        ob.0,
    ));
    // M8's first dependency: our from-scratch MSVC demangler vs llvm-nm.
    // The gate owns the exit code; this row reports its verdict line.
    let dm_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "demangle".into(),
            "--cpp".into(),
        ]
    } else {
        vec!["demangle".into(), "--cpp".into()]
    };
    let dm = sh(&exe, &dm_cmd);
    rows.push((
        "demangle --cpp (M8: MSVC C++ names vs llvm-nm)".into(),
        dm.1.lines().last().unwrap_or("").trim().into(),
        dm.0,
    ));
    // M8 step 1: the C++ measurement. Like the t2 row, this REPORTS the gate's
    // verdict and does not own it: `rdec cpp --corpus` decides its own exit
    // code, and two owners on one gate is how a gate gets weakened.
    let cpp_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "cpp".into(),
            "--corpus".into(),
        ]
    } else {
        vec!["cpp".into(), "--corpus".into()]
    };
    let cpp = sh(&exe, &cpp_cmd);
    let cpp_line = cpp
        .1
        .lines()
        .find(|l| l.starts_with("compared:"))
        .unwrap_or("no measurement ran")
        .trim()
        .to_string();
    let cpp_verdict = cpp
        .1
        .lines()
        .find(|l| l.starts_with("FAIL"))
        .map(|l| format!("{cpp_line} | {l}"))
        .unwrap_or_else(|| format!("{cpp_line} | gate green"));
    rows.push((
        "cpp --corpus (M8: C++ machine code -> demangled C; gate: rdec cpp)".into(),
        cpp_verdict,
        // Informational row: enforced by `rdec cpp --corpus`, which exits 1.
        true,
    ));
    // T2 is the only row that RUNS both sides: `rdec c` proves the emitted C
    // compiles, this proves it computes the same values as the machine code.
    // It is a real verdict, so it is allowed to fail the scoreboard.
    let t2_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "t2".into(),
            "--corpus".into(),
        ]
    } else {
        vec!["t2".into(), "--corpus".into()]
    };
    let t2 = sh(&exe, &t2_cmd);
    // The verdict is printed verbatim, including the FAIL text, so the scoreboard
    // cannot hide it - but it does not decide this process's exit code. `rdec t2
    // --corpus` IS the gate and exits 1 while the decompiled C disagrees; the
    // scoreboard's job is to report where the project stands, the way the M1
    // coverage and M4 refusal gates are red in their own commands while `status`
    // stays a scoreboard. Moving enforcement here would put two owners on one gate.
    let t2_line =
        t2.1.lines()
            .find(|l| l.starts_with("compared:"))
            .unwrap_or("no comparison ran")
            .trim()
            .to_string();
    let t2_verdict =
        t2.1.lines()
            .find(|l| l.starts_with("FAIL"))
            .map(|l| format!("{t2_line} | {l}"))
            .unwrap_or_else(|| format!("{t2_line} | gate green"));
    rows.push((
        "t2 differential (C vs machine code, same inputs, RAX); gate: rdec t2".into(),
        t2_verdict,
        // Informational row: enforced by `rdec t2 --corpus`, which exits 1.
        true,
    ));
    // T2 over C++: the decompiled C++ is VALUE-verified, not just compiled.
    // Same rule as every gate row: the gate owns its exit code (`rdec t2
    // --cpp` exits 1 on a mismatch or on zero comparisons); the scoreboard
    // reports the verdict line and does not put a second owner on the gate.
    let t2c_cmd: Vec<String> = if self_cmd[0] == "run" {
        vec![
            "run".into(),
            "-q".into(),
            "-p".into(),
            "rdec-cli".into(),
            "--".into(),
            "t2".into(),
            "--cpp".into(),
        ]
    } else {
        vec!["t2".into(), "--cpp".into()]
    };
    let t2c = sh(&exe, &t2c_cmd);
    let t2c_line = t2c
        .1
        .lines()
        .find(|l| l.starts_with("compared:"))
        .unwrap_or("no comparison ran")
        .trim()
        .to_string();
    let t2c_verdict = t2c
        .1
        .lines()
        .find(|l| l.starts_with("FAIL"))
        .map(|l| format!("{t2c_line} | {l}"))
        .unwrap_or_else(|| format!("{t2c_line} | gate green"));
    rows.push((
        "t2 --cpp (C++ machine code -> C, values on RAX + `this` windows); gate: rdec t2 --cpp"
            .into(),
        t2c_verdict,
        // Informational row: enforced by `rdec t2 --cpp`, which exits 1.
        true,
    ));
    let local = rdec_verify::corpus::clang_stamp(&rdec_verify::corpus::clang_path()).0;
    let recorded = rdec_verify::corpus::manifest_clang(std::path::Path::new("corpus/manifest.tsv"));
    let same = recorded == local;
    rows.push((
        "corpus toolchain".into(),
        if same {
            format!("manifest written by this clang ({local})")
        } else {
            format!("recorded: {recorded} | local: {local} (hashes track the compiler)")
        },
        // Informational: object bytes are a function of the compiler version.
        true,
    ));

    println!("| check | result |");
    println!("|---|---|");
    let mut ok = true;
    for (k, v, good) in &rows {
        println!("| {k} | {}{} |", if *good { "" } else { "**FAIL** " }, v);
        ok &= *good;
    }
    println!(
        "\n{}",
        if ok {
            "status OK"
        } else {
            "status: FAILING ROWS ABOVE"
        }
    );
    if ok {
        0
    } else {
        1
    }
}

/// Re-read the corpus manifest and re-parse every object's COFF header: the row
/// says the objects are what the manifest claims, AMD64, and non-empty.
fn corpus_row() -> (String, String, bool) {
    let Ok(man) = std::fs::read_to_string("corpus/manifest.tsv") else {
        return (
            "corpus".into(),
            "no manifest (run `rdec corpus`)".into(),
            false,
        );
    };
    let mut n = 0usize;
    let mut bytes = 0usize;
    let mut bad = Vec::new();
    for line in man
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\topt"))
    {
        let mut it = line.split('\t');
        let (Some(name), Some(opt), Some(path)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let Ok(b) = std::fs::read(path) else {
            bad.push(format!("{name}.{opt}: missing"));
            continue;
        };
        n += 1;
        bytes += b.len();
        // A .obj starts with the COFF header, so Machine is at offset 0. Only
        // images (MZ) carry the PE\0\0 signature; claiming the header is where
        // an image's header is would read garbage for every object.
        let le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        let m = if b.len() > 0x40 && &b[0..2] == b"MZ" {
            let pe = u32::from_le_bytes([b[0x3c], b[0x3d], b[0x3e], b[0x3f]]) as usize;
            if b.len() > pe + 6 && &b[pe..pe + 4] == b"PE\0\0" {
                le(pe + 4)
            } else {
                0
            }
        } else if b.len() > 2 {
            le(0)
        } else {
            0
        };
        if m != 0x8664 {
            bad.push(format!("{name}.{opt}: machine {m:#x}"));
        }
    }
    let verdict = if bad.is_empty() {
        format!("{n} objects, {bytes} bytes, every Machine == 0x8664")
    } else {
        format!(
            "{n} objects, {bytes} bytes, {} bad: {}",
            bad.len(),
            bad[..bad.len().min(3)].join(", ")
        )
    };
    ("corpus".into(), verdict, bad.is_empty() && n > 0)
}

/// `rdec pe-info <exe>` — the M7 first slice made visible: what the PE32+ image
/// reader read, and what it refused.
///
/// Exit codes follow the convention already in this file: **0** on a parsed
/// image, **2** on a refusal — the `c` / `t2` / `lift` / `fuzz` arms use 2 for
/// "the input was refused", and `discover`'s 1 is a pipeline gap, a different
/// event. The refusal text is `PeImage::load`'s verbatim, which names the
/// field, the value found, the value expected, the offset and the file length,
/// so the same message is what `tests/pe_fixtures.rs` asserts on.
fn cmd_pe_info(args: &[String]) -> i32 {
    use rdec_arch_x86::pe::PeImage;
    let Some(path) = args.get(1) else {
        eprintln!("pe-info: usage: rdec pe-info <exe>");
        return 2;
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("pe-info: {path}: {e}");
            return 2;
        }
    };
    let img = match PeImage::load(&bytes, path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!(
        "{}: PE32+ image, {} bytes, machine {:#x} ({})",
        img.path,
        img.file_len,
        img.machine,
        if img.machine == 0x8664 {
            "AMD64"
        } else {
            "not AMD64"
        }
    );
    println!(
        "  image base      {:#x}   entry RVA {:#x} (VA {:#x})",
        img.image_base,
        img.entry_rva,
        img.entry_va()
    );
    println!(
        "  subsystem       {} ({})   characteristics {:#x}   DllCharacteristics {:#x}",
        img.subsystem,
        img.subsystem_name(),
        img.characteristics,
        img.dll_characteristics
    );
    println!(
        "  size of image   {:#x}   size of headers {:#x}   section align {:#x}   file align {:#x}",
        img.size_of_image, img.size_of_headers, img.section_alignment, img.file_alignment
    );
    println!(
        "  data directories {} declared (NumberOfRvaAndSizes)",
        img.number_of_rva_and_sizes
    );
    for d in img
        .data_directories
        .iter()
        .filter(|d| d.rva != 0 || d.size != 0)
    {
        println!("    {:<15} rva {:#x} size {:#x}", d.name, d.rva, d.size);
    }
    println!("  sections        {}", img.sections.len());
    println!(
        "  {:<3} {:<10} {:<10} {:<10} {:<10} {:<10} flags",
        "idx", "name", "vaddr", "vsize", "rawptr", "rawsize"
    );
    for s in &img.sections {
        println!(
            "  {:<3} {:<10} {:<10} {:<10} {:<10} {:<10} {}",
            s.index,
            s.name,
            format!("{:#x}", s.virtual_address),
            format!("{:#x}", s.virtual_size),
            format!("{:#x}", s.pointer_to_raw_data),
            format!("{:#x}", s.size_of_raw_data),
            s.flags_summary()
        );
    }
    // What the reader did NOT read, printed so "it loaded" is never mistaken
    // for "the image is understood": these are the next M7 slices.
    let mut later: Vec<&str> = Vec::new();
    if img.section_named(".pdata").is_some() {
        later.push(".pdata/.xdata SEH unwind (the function-start source)");
    }
    if img.section_named(".reloc").is_some() {
        later.push(".reloc base relocations");
    }
    if img
        .data_directories
        .iter()
        .any(|d| (d.name == "IMPORT" || d.name == "IAT") && d.rva != 0)
    {
        later.push("the import table (imports are calls to a boundary that stays LOUD)");
    }
    if !later.is_empty() {
        println!("  loaded, not yet read: {}", later.join("; "));
    }
    0
}

/// `rdec pe-functions <exe>` — M7 slice 2 made visible: the `.pdata` SEH
/// unwind table harvested as the symbol-free function-start source.
///
/// Exit codes follow `pe-info`: **0** on a read table, **2** on a refusal —
/// and the refusal text is `UnwindTable::read`'s verbatim, naming field,
/// entry (index + BeginAddress), got/expected, offset and file length, the
/// same text `tests/pdata_fixtures.rs` asserts on. Chaining refusals do NOT
/// change the exit code: the table as a whole parsed, and every chained
/// entry is printed LOUD on its own line and counted in the summary — the
/// count is the anti-silent-drop guarantee (functions + refusals == dir/12).
fn cmd_pe_functions(args: &[String]) -> i32 {
    use rdec_arch_x86::pdata::UnwindTable;
    use rdec_arch_x86::pe::PeImage;
    let Some(path) = args.get(1) else {
        eprintln!("pe-functions: usage: rdec pe-functions <exe>");
        return 2;
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("pe-functions: {path}: {e}");
            return 2;
        }
    };
    let img = match PeImage::load(&bytes, path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let t = match UnwindTable::read(&img) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!(
        "{}: EXCEPTION dir rva {:#x} size {:#x} = {} RUNTIME_FUNCTION entries",
        img.path,
        t.dir_rva,
        t.dir_size,
        t.entry_count()
    );
    println!(
        "  {:<28} {:<12} {:<7} first_code",
        "begin_va..end_va", "unwind_va", "flags"
    );
    for e in t.entries() {
        println!(
            "  {:#x}..{:#x}  {:#x}  {:#x}  {}",
            img.va(e.begin_rva),
            img.va(e.end_rva),
            img.va(e.unwind_rva),
            e.flags,
            e.first_code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
    }
    let flags: Vec<String> = t
        .flags_census()
        .iter()
        .map(|(f, n)| format!("{f:#x}:{n}"))
        .collect();
    println!("  census flags: {}", flags.join(" "));
    let codes: Vec<String> = t
        .first_code_census()
        .iter()
        .map(|(c, n)| match c {
            Some(c) => format!("code {c}:{n}"),
            None => format!("no-codes:{n}"),
        })
        .collect();
    println!("  census first-code: {}", codes.join(" "));
    // LOUD, per entry: a chained entry is refused as a function start, and
    // the print is what makes "N functions" auditable against M/12.
    for c in t.chaining_refusals() {
        println!("  CHAINED REFUSAL: {}", c.message());
    }
    for v in t.version_refusals() {
        println!("  VERSION REFUSAL: {}", v.message(&img));
    }
    println!(
        "{} functions, .pdata {}/12 entries, chaining refusals {}, version refusals {}",
        t.entries().len(),
        t.dir_size,
        t.chaining_refusals().len(),
        t.version_refusals().len()
    );
    0
}

/// Run a command, return (success, stdout+stderr).
fn sh<P, A, S>(program: P, args: A) -> (bool, String)
where
    P: AsRef<std::ffi::OsStr>,
    A: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut c = std::process::Command::new(program);
    c.args(args);
    match c.output() {
        Ok(o) => (
            o.status.success(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
        ),
        Err(e) => (false, format!("could not run: {e}")),
    }
}
