//! rdec — command line driver.
//!
//! M0 gives the two things the phase needs: the T1 flag oracle, and a self-test
//! that proves the IR verifier actually rejects broken functions. `disasm`,
//! `ir`, `cfg` and `decompile` land with M1/M2.

use rdec_core::ir::*;
use rdec_core::ty::{Sx, Ty, Width};
use rdec_core::verify;
use rdec_verify::{hw, model_shim, Sabotage};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rc = match cmd {
        "flags" => cmd_flags(&args),
        "selftest" => cmd_selftest(),
        "corpus" => cmd_corpus(&args[1..]),
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
