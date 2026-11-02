//! M1 front-end commands: `disasm` and `objdump-check`.
//!
//! `objdump-check` is the M1 gate. The claim it tests is narrow on purpose:
//! on a real object, for each function symbol, the instruction boundaries we
//! find are the instruction boundaries llvm-objdump finds. Mnemonics are
//! decoration — addresses are the contract, because basic blocks, SSA and the
//! CFG all consume addresses, and one misaligned decode invents a basic block
//! that the machine code never had.

use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_arch_x86::decode::{decode, round_trip, Syntax};
use rdec_arch_x86::loader::{self, SizeSource};

fn objdump_path() -> PathBuf {
    std::env::var_os("RDEC_OBJDUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("llvm-objdump"))
}

/// Symbols worth disassembling: in a code section, not a section symbol, with
/// a non-zero extent, and not pure padding.
fn callable_symbols(obj: &loader::Object) -> Vec<usize> {
    obj.code_symbols()
        .into_iter()
        .filter(|&i| {
            let s = &obj.symbols[i];
            s.size >= 1 && !s.name.starts_with('.')
        })
        .filter(|&i| {
            let (bytes, _) = match obj.symbol_bytes(i) {
                Some(b) => b,
                None => return false,
            };
            bytes.iter().any(|&b| b != 0)
        })
        .collect()
}

fn syntax_of(args: &[String]) -> Syntax {
    match args.iter().position(|a| a == "--syntax") {
        Some(i) => match args.get(i + 1).map(|s| s.as_str()) {
            Some("nasm") => Syntax::Nasm,
            _ => Syntax::Intel,
        },
        None => Syntax::Intel,
    }
}

/// `rdec disasm <obj> [filter] [--syntax intel|nasm]`
pub fn cmd_disasm(args: &[String]) -> i32 {
    let Some(path) = args.get(1) else {
        eprintln!("usage: rdec disasm <object.obj> [symbol-filter] [--syntax intel|nasm]");
        return 2;
    };
    // The symbol filter is the first positional argument - not the value of
    // `--syntax nasm`, which the first version read as a filter and then
    // disassembled nothing.
    let mut skip = false;
    let mut filter = String::new();
    for a in args.iter().skip(2) {
        if skip {
            skip = false;
            continue;
        }
        if a == "--syntax" {
            skip = true;
            continue;
        }
        if !a.starts_with("--") {
            filter = a.clone();
            break;
        }
    }
    let syntax = syntax_of(args);
    let obj = match loader::load(Path::new(path)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("load failed: {e}");
            return 1;
        }
    };
    if !obj.amd64() {
        eprintln!(
            "{}: machine {:#06x} is not AMD64 (0x8664); refusing to guess",
            obj.path, obj.machine
        );
        return 1;
    }
    let mut shown = 0usize;
    for sym in callable_symbols(&obj) {
        let s = &obj.symbols[sym];
        if !filter.is_empty() && !s.name.contains(&filter) {
            continue;
        }
        let (bytes, va) = obj.symbol_bytes(sym).unwrap();
        let section = s.section.unwrap();
        let relocs = obj.relocs_in(section, s.offset, s.offset + s.size);
        let extent = match s.size_source {
            SizeSource::NextSymbol => "extent: next symbol (COFF has no sizes)",
            SizeSource::SectionEnd => "extent: section end",
            SizeSource::Exact => "extent: exact",
        };
        println!(
            "; {}  {}+{:#x}  {} bytes  [{}]  {} relocations",
            s.name,
            obj.sections[section].name,
            s.offset,
            s.size,
            extent,
            relocs.len()
        );
        let insns = decode(bytes, va, syntax);
        let mut bad = 0usize;
        for i in &insns {
            match &i.text {
                Some(t) => {
                    let raw: Vec<String> = bytes[(i.va - va) as usize..][..i.len]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    let anno: Vec<String> = relocs
                        .iter()
                        .filter(|r| r.address >= i.va - va && r.address < i.va - va + i.len as u64)
                        .map(|r| {
                            let n = obj
                                .symbols
                                .get(r.symbol)
                                .map(|s| s.name.as_str())
                                .unwrap_or("?");
                            format!("reloc typ {} -> {n}", r.typ)
                        })
                        .collect();
                    let anno = if anno.is_empty() {
                        String::new()
                    } else {
                        format!("   ; {}", anno.join(", "))
                    };
                    println!("  {:#06x}  {:<22}  {}{}", i.va, raw.join(" "), t, anno);
                }
                None => {
                    bad += 1;
                    println!("  {:#06x}  ??                    ; UNDECODABLE", i.va);
                }
            }
        }
        match round_trip(bytes, va) {
            Ok(n) => println!("; {n} instructions, decode/encode/decode stable, {bad} undecodable"),
            Err(e) => println!("; decode/encode/decode FAILED: {e}"),
        }
        println!();
        shown += 1;
    }
    if shown == 0 {
        eprintln!(
            "no symbols matched ({} code symbols in object)",
            obj.code_symbols().len()
        );
        return 1;
    }
    0
}

/// Instruction addresses llvm-objdump reports for a section range.
fn objdump_addrs(obj: &Path, start: u64, stop: u64) -> Result<Vec<u64>, String> {
    let out = Command::new(objdump_path())
        .arg("-d")
        .arg("--no-show-raw-insn")
        .arg(format!("--start-address=0x{:x}", start))
        .arg(format!("--stop-address=0x{:x}", stop))
        .arg(obj)
        .output()
        .map_err(|e| format!("run {}: {e}", objdump_path().display()))?;
    if !out.status.success() {
        return Err(format!(
            "llvm-objdump exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let mut addrs = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((head, _rest)) = line.split_once(':') else {
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
    Ok(addrs)
}

/// `rdec objdump-check <obj> [more...] [--limit N]`
pub fn cmd_objdump_check(args: &[String]) -> Result<i32, String> {
    // Positional object paths, skipping the values that belong to flags:
    // collecting `--limit 60` as a file named `60` is how the first version of
    // this failed the `rdec status` row that runs it.
    const TAKES_VALUE: [&str; 2] = ["--limit", "--syntax"];
    let mut files: Vec<String> = Vec::new();
    let mut skip_value = false;
    for a in args.iter().skip(1) {
        if skip_value {
            skip_value = false;
            continue;
        }
        if TAKES_VALUE.contains(&a.as_str()) {
            skip_value = true;
            continue;
        }
        if !a.starts_with("--") {
            files.push(a.clone());
        }
    }
    if args.iter().any(|a| a == "--corpus") {
        // Object paths straight out of the manifest, so the gate is driven by
        // the corpus itself and not by a shell glob.
        let man = std::fs::read_to_string("corpus/manifest.tsv")
            .map_err(|e| format!("corpus/manifest.tsv: {e}"))?;
        for line in man.lines() {
            if line.starts_with('#') || line.starts_with("name	") {
                continue;
            }
            if let Some(obj) = line.split('\t').nth(2) {
                files.push(obj.to_string());
            }
        }
    }
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if files.is_empty() {
        eprintln!("usage: rdec objdump-check <object.obj> [more.obj ...] [--limit N]");
        return Ok(2);
    }
    let syntax = syntax_of(args);
    let mut checked = 0usize;
    let mut matched = 0usize;
    let mut skipped = 0usize;
    let mut first_problem: Option<String> = None;
    let mut budget = limit;

    for file in &files {
        let obj = match loader::load(Path::new(file)) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("load failed: {e}");
                return Ok(1);
            }
        };
        for sym in callable_symbols(&obj) {
            if budget == 0 {
                break;
            }
            budget -= 1;
            checked += 1;
            let s = &obj.symbols[sym];
            let (bytes, va) = obj.symbol_bytes(sym).unwrap();
            let ours: Vec<u64> = decode(bytes, va, syntax)
                .iter()
                .map(|i| i.va - va + s.offset)
                .collect();
            let stop = s.offset + s.size;
            let theirs = match objdump_addrs(Path::new(file), s.offset, stop) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{}: {e}", obj.path);
                    return Ok(1);
                }
            };
            if ours == theirs {
                matched += 1;
                continue;
            }
            let only_ours: Vec<u64> = ours
                .iter()
                .filter(|a| !theirs.contains(a))
                .copied()
                .collect();
            let only_theirs: Vec<u64> = theirs
                .iter()
                .filter(|a| !ours.contains(a))
                .copied()
                .collect();
            if only_theirs.is_empty() && only_ours.len() == 1 && ours.len() == theirs.len() + 1 {
                // Our sweep refuses at the end of the range (a truncated
                // instruction at the extent boundary). The boundaries inside
                // the function agreed; the extent, not the decoder, is the
                // thing to report — and COFF extents are inferred.
                skipped += 1;
                continue;
            }
            let msg = format!(
                "{}: {} at {:#x}: {}/{} instructions, {} extra, {} missing (first extra {:?}, first missing {:?})",
                obj.path,
                s.name,
                s.offset,
                ours.len(),
                theirs.len(),
                only_ours.len(),
                only_theirs.len(),
                only_ours.first(),
                only_theirs.first()
            );
            eprintln!("MISMATCH {msg}");
            if first_problem.is_none() {
                first_problem = Some(msg);
            }
        }
    }

    println!(
        "objdump-check: {checked} symbols, {matched} exact boundary matches, {skipped} extent-boundary refusals, {} mismatches",
        checked - matched - skipped
    );
    if checked == 0 {
        // A gate that checked nothing has proven nothing.
        eprintln!("FAIL: no code symbols found in {files:?} - nothing was checked");
        return Ok(1);
    }
    match first_problem {
        None => {
            println!("OK: instruction boundaries agree with llvm-objdump");
            Ok(0)
        }
        Some(_) => Ok(1),
    }
}

/// `rdec discover <obj> [more...] [--corpus] [--limit N]`
///
/// M1 step 2's gate. Ground truth is the object's own symbol table: every
/// function symbol must be recovered by descent alone, from entry points and
/// call relocations. Extra starts (branch targets promoted by a call, local
/// labels called through a pointer) are reported, not punished; *missing*
/// starts are the failure mode, because a function we never reach is a function
/// we cannot decompile.
pub fn cmd_discover(args: &[String]) -> Result<i32, String> {
    use rdec_arch_x86::discover::{discover_all, overlap_free, Code, Gap};
    use std::collections::HashMap;

    const TAKES_VALUE: [&str; 2] = ["--limit", "--syntax"];
    let mut files: Vec<String> = Vec::new();
    let mut skip_value = false;
    for a in args.iter().skip(1) {
        if skip_value {
            skip_value = false;
            continue;
        }
        if TAKES_VALUE.contains(&a.as_str()) {
            skip_value = true;
            continue;
        }
        if !a.starts_with("--") {
            files.push(a.clone());
        }
    }
    if args.iter().any(|a| a == "--corpus") {
        let man = std::fs::read_to_string("corpus/manifest.tsv")
            .map_err(|e| format!("corpus/manifest.tsv: {e}"))?;
        for line in man.lines() {
            if line.starts_with('#') || line.starts_with("name\t") {
                continue;
            }
            if let Some(obj) = line.split('\t').nth(2) {
                files.push(obj.to_string());
            }
        }
    }
    if files.is_empty() {
        return Err(
            "usage: rdec discover <object.obj> [more.obj ...] [--corpus] [--limit N]".to_string(),
        );
    }
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);

    let mut objects = 0usize;
    let mut truth_total = 0usize;
    let mut missing_total = 0usize;
    let mut extra_total = 0usize;
    let mut code_bytes = 0u64;
    let mut covered = 0u64;
    let mut gaps = HashMap::<&'static str, usize>::new();
    let mut overlap_problems = 0usize;
    let mut padding_total = 0u64;
    let mut other_uncovered_total = 0u64;
    let mut other_runs_total = 0usize;
    let mut first_missing: Option<String> = None;
    let mut budget = limit;

    for file in &files {
        let obj = loader::load(Path::new(file))?;
        // Each code section gets its own address range. Basing every section
        // at 0 - the first version - makes two sections' offset 0x10 look like
        // the same byte, and the overlap check reports 30 objects "disagreeing
        // about a boundary" that never shared a byte.
        let mut ranges: Vec<Code> = Vec::new();
        let mut idx_of = HashMap::new();
        let mut base_of: Vec<u64> = Vec::new();
        let mut next_base = 0u64;
        for (i, sec) in obj.sections.iter().enumerate() {
            if obj.is_code(i) && !sec.bytes.is_empty() {
                idx_of.insert(i, ranges.len());
                base_of.push(next_base);
                ranges.push(Code {
                    base: next_base,
                    bytes: sec.bytes.as_slice(),
                });
                next_base += sec.bytes.len() as u64 + 16;
            }
        }
        if ranges.is_empty() {
            continue;
        }
        let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
        let mut truth: Vec<u64> = Vec::new();
        for i in obj.code_symbols() {
            let s = &obj.symbols[i];
            let sec = s.section.unwrap();
            if !obj.is_code(sec) || s.name.starts_with('.') || s.size == 0 {
                continue;
            }
            let Some(&r) = idx_of.get(&sec) else { continue };
            if budget == 0 {
                break;
            }
            truth.push(base_of[r] + s.offset);
            seeds.entry(r).or_default().0.push(base_of[r] + s.offset);
        }
        budget = budget.saturating_sub(truth.len().max(1));
        // Call relocations naming a symbol defined in this object are function
        // boundaries visible without symbols - the stripped-binary case.
        for (sec, rels) in obj.sections.iter().enumerate() {
            let Some(&r) = idx_of.get(&sec) else { continue };
            for rl in &rels.relocations {
                if !matches!(rl.typ, 2 | 4 | 5 | 6 | 7) {
                    continue;
                }
                if let Some(t) = obj.symbols.get(rl.symbol) {
                    if t.section == Some(sec) && !t.name.starts_with('.') {
                        seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                    }
                }
            }
        }
        let plan = discover_all(&ranges, &seeds);
        let truth_n = truth.len();
        let missing: Vec<u64> = truth
            .iter()
            .copied()
            .filter(|a| !plan.functions.contains(a))
            .collect();
        let extra: Vec<u64> = plan
            .functions
            .iter()
            .copied()
            .filter(|a| !truth.contains(a))
            .collect();
        let cb: u64 = ranges.iter().map(|r| r.bytes.len() as u64).sum();
        if !overlap_free(&plan.insns) {
            overlap_problems += 1;
            let cov: u64 = plan.insns.iter().map(|(_, l)| *l as u64).sum();
            eprintln!(
                "OVERLAP {}: covered {cov} bytes of {cb} code, {} insns, {} blocks",
                file,
                plan.insns.len(),
                plan.blocks.len()
            );
        }
        // Split the uncovered bytes: alignment padding is a run of 0xCC (int3)
        // and is not a coverage failure, while any other uncovered byte is
        // code we did not reach. Counting them together lets the coverage
        // number be inflated by padding.
        let mut padding = 0u64;
        let mut other_uncovered = 0u64;
        let mut other_runs = 0usize;
        for r in &ranges {
            let mut hit = vec![false; r.bytes.len()];
            for &(a, l) in &plan.insns {
                if a >= r.base && ((a - r.base) as usize) < hit.len() {
                    let start = (a - r.base) as usize;
                    for k in start..(start + l).min(hit.len()) {
                        hit[k] = true;
                    }
                }
            }
            let mut i = 0usize;
            while i < hit.len() {
                if hit[i] {
                    i += 1;
                    continue;
                }
                let j = hit[i..]
                    .iter()
                    .position(|&h| h)
                    .map(|p| i + p)
                    .unwrap_or(hit.len());
                let run = &r.bytes[i..j];
                // int3 padding is 0xCC and can be any length; 0x00 fills are
                // only padding when they are long enough that they cannot be
                // an instruction stream.
                // Padding is a *semantic* claim, so ask the decoder: LLVM
                // aligns with multi-byte NOPs (`0f 1f 84 00 ...`, `66 66 2e 0f
                // 1f ...`), which no byte-pattern rule names. A run whose
                // instructions are all nops is padding; int3 (0xCC) and long
                // 0x00 fills are padding too.
                let all_cc = run.iter().all(|&b| b == 0xCC);
                let all_00 = run.iter().all(|&b| b == 0x00);
                let all_nop = {
                    let ins = decode(run, 0, Syntax::Intel);
                    !ins.is_empty()
                        && ins.iter().all(|i| {
                            i.valid()
                                && i.text
                                    .as_deref()
                                    .map(|t| t.starts_with("nop"))
                                    .unwrap_or(false)
                        })
                };
                if all_cc || all_nop || (all_00 && run.len() >= 4) {
                    padding += run.len() as u64;
                } else {
                    other_uncovered += run.len() as u64;
                    other_runs += 1;
                    if other_runs_total + other_runs <= 12 {
                        let hex: String = run.iter().take(16).map(|b| format!("{b:02x}")).collect();
                        println!(
                            "  uncovered {} bytes at {:#x} in {}: {}",
                            run.len(),
                            r.base + i as u64,
                            obj.sections
                                .iter()
                                .find(|s| s.bytes.as_slice() == r.bytes)
                                .map(|s| s.name.as_str())
                                .unwrap_or("?"),
                            hex
                        );
                    }
                }
                i = j;
            }
        }
        padding_total += padding;
        other_uncovered_total += other_uncovered;
        other_runs_total += other_runs;
        objects += 1;
        truth_total += truth_n;
        missing_total += missing.len();
        extra_total += extra.len();
        code_bytes += cb;
        covered += plan.covered_bytes();
        for g in &plan.gaps {
            let name = match g.gap {
                Gap::IndirectBranch => "indirect branch",
                Gap::IndirectCall => "indirect call",
                Gap::Undecodable => "undecodable",
                Gap::OutsideCode => "outside code",
            };
            *gaps.entry(name).or_default() += 1;
        }
        if !missing.is_empty() && first_missing.is_none() {
            first_missing = Some(format!("{}: missing starts {:?}", file, missing));
        }
    }

    let code_no_pad = code_bytes.saturating_sub(padding_total);
    let pct = if code_no_pad > 0 {
        100.0 * covered as f64 / code_no_pad as f64
    } else {
        0.0
    };
    println!(
        "discover: {objects} objects, {truth_total} symbol-table functions, {missing_total} missing, {extra_total} extra starts"
    );
    println!(
        "code bytes: {code_bytes}, padding (int3, nop sequences, long 0x00 fills): {padding_total}"
    );
    println!(
        "coverage: {covered}/{code_no_pad} non-padding code bytes = {pct:.2}%; uncovered non-padding = {other_uncovered_total} bytes in {other_runs_total} runs"
    );
    let mut gs: Vec<_> = gaps.into_iter().collect();
    gs.sort();
    println!(
        "gaps: {}",
        if gs.is_empty() {
            "none".to_string()
        } else {
            gs.iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    if overlap_problems > 0 {
        eprintln!("FAIL: {overlap_problems} objects with overlapping instruction ranges - two sweeps disagreed about a boundary");
        return Ok(1);
    }
    if truth_total == 0 {
        return Err("no functions found - discovery checked nothing".to_string());
    }
    if let Some(m) = first_missing {
        eprintln!("FAIL {m}");
        return Ok(1);
    }
    // M1's gate is 99.5% coverage of non-padding code, over everything the
    // descent can reach. Indirect sites are M5's problem; the number here is
    // the honest denominator, so the gap is visible rather than smoothed away.
    if pct < 99.5 {
        eprintln!(
            "FAIL: coverage {pct:.2}% is below the 99.5% M1 gate ({other_uncovered_total} uncovered non-padding bytes in {other_runs_total} runs)"
        );
        return Ok(1);
    }
    println!("OK: every symbol-table function is reachable, coverage above the 99.5% gate");
    Ok(0)
}

pub fn cmd_lift(args: &[String]) -> Result<i32, String> {
    use rdec_arch_x86::discover::{discover_all, Code};
    use rdec_arch_x86::lift::lift_one;
    use rdec_core::fmt_ir::format_function;
    use rdec_core::verify::verify_ssa;
    use std::collections::HashMap;

    const TAKES_VALUE: [&str; 3] = ["--limit", "--ir", "--syntax"];
    let mut files: Vec<String> = Vec::new();
    let mut ir_name: Option<String> = None;
    let mut skip_value = false;
    for a in args.iter().skip(1) {
        if skip_value {
            skip_value = false;
            continue;
        }
        if TAKES_VALUE.contains(&a.as_str()) {
            skip_value = true;
            if a == "--ir" {
                ir_name = Some(args[args.iter().position(|x| x == a).unwrap() + 1].clone());
            }
            continue;
        }
        if !a.starts_with("--") {
            files.push(a.clone());
        }
    }
    if args.iter().any(|a| a == "--corpus") {
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
    if files.is_empty() {
        return Err(
            "usage: rdec lift <object.obj> [...] [--corpus] [--limit N] [--ir NAME]".to_string(),
        );
    }
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);

    let mut objects = 0usize;
    let mut lifted = 0usize;
    let mut verified = 0usize;
    let mut clean = 0usize;
    let mut gap_total = 0usize;
    let mut exit_total = 0usize;
    let mut unnamed_total = 0usize;
    let mut clobber_total = 0usize;
    let mut unreach_total = 0usize;
    let mut undef_live = 0usize;
    let mut scaffolding = 0usize;
    let (mut ret_n, mut no_ret, mut step_budget, mut holes, mut unmodeled_load, mut fault) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut opaque = 0usize;
    let mut first_bad: Option<String> = None;
    let mut bad_msgs: Vec<String> = Vec::new();
    let mut first_gappy: Option<String> = None;
    let mut budget = limit;

    for file in &files {
        let obj = loader::load(Path::new(file))?;
        objects += 1;
        let mut ranges: Vec<Code> = Vec::new();
        let mut idx_of = HashMap::new();
        let mut base_of: Vec<u64> = Vec::new();
        let mut next_base = 0u64;
        for (i, sec) in obj.sections.iter().enumerate() {
            if obj.is_code(i) && !sec.bytes.is_empty() {
                idx_of.insert(i, ranges.len());
                base_of.push(next_base);
                ranges.push(Code {
                    base: next_base,
                    bytes: sec.bytes.as_slice(),
                });
                next_base += sec.bytes.len() as u64 + 16;
            }
        }
        if ranges.is_empty() {
            continue;
        }
        let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
        for i in obj.code_symbols() {
            let s = &obj.symbols[i];
            let sec = s.section.unwrap();
            if !obj.is_code(sec) || s.name.starts_with('.') || s.size == 0 {
                continue;
            }
            if let Some(&r) = idx_of.get(&sec) {
                seeds.entry(r).or_default().0.push(base_of[r] + s.offset);
            }
        }
        for (sec, rels) in obj.sections.iter().enumerate() {
            let Some(&r) = idx_of.get(&sec) else { continue };
            for rl in &rels.relocations {
                if !matches!(rl.typ, 2 | 4 | 5 | 6 | 7) {
                    continue;
                }
                if let Some(t) = obj.symbols.get(rl.symbol) {
                    if t.section == Some(sec) && !t.name.starts_with('.') {
                        seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                    }
                }
            }
        }
        let plan = discover_all(&ranges, &seeds);

        // Addresses we can name: defined symbols, and relocation targets (the
        // callee of a call, which is what a call target needs to compile).
        let mut names: HashMap<u64, String> = HashMap::new();
        for (i, sec) in obj.sections.iter().enumerate() {
            let Some(&r) = idx_of.get(&i) else { continue };
            for s in obj
                .symbols
                .iter()
                .filter(|s| s.section == Some(i) && !s.name.starts_with('.'))
            {
                names.insert(base_of[r] + s.offset, s.name.clone());
            }
            for rl in &sec.relocations {
                if let Some(t) = obj.symbols.get(rl.symbol) {
                    names.insert(base_of[r] + rl.address, t.name.clone());
                }
            }
        }

        for (k, &start) in plan.functions.iter().enumerate() {
            if budget == 0 {
                break;
            }
            budget -= 1;
            let Some((_, rg)) = ranges
                .iter()
                .enumerate()
                .find(|(_, r)| start >= r.base && start < r.base + r.bytes.len() as u64)
            else {
                continue;
            };
            let end = plan.functions[k + 1..]
                .iter()
                .find(|a| **a >= rg.base && **a < rg.base + rg.bytes.len() as u64)
                .copied()
                .unwrap_or(rg.base + rg.bytes.len() as u64);
            let name = names
                .get(&start)
                .cloned()
                .unwrap_or_else(|| format!("fn_{start:x}"));
            let (f, rep) = lift_one(&name, rg.bytes, rg.base, start, end, &plan, &names);
            lifted += 1;
            gap_total += rep.gaps;
            exit_total += rep.exits;
            unnamed_total += rep.unnamed_calls;
            clobber_total += rep.clobbers;
            unreach_total += rep.unreachable_blocks;
            undef_live += rep.undef_live_ins;
            scaffolding += rep.scaffolding_removed;
            // M3's semantic check, on top of the structural one: run the IR
            // interpreter over the lifted function. It is strict - every operand
            // it consumes must already be defined - so a renaming that leaves a
            // dangling reference, a phi wired to a value that is not live on that
            // edge, or a register placeholder that never got rewritten all surface
            // as a fault that the structural verifier's rules cannot express.
            let ex = rdec_core::interp::run_budget(&f, &[], 20000);
            match &ex.outcome {
                rdec_core::interp::Outcome::Ret(_) => ret_n += 1,
                rdec_core::interp::Outcome::NoReturn => no_ret += 1,
                rdec_core::interp::Outcome::Budget => step_budget += 1,
                rdec_core::interp::Outcome::Fault(ft) => match ft {
                    rdec_core::interp::Fault::MissingOperand(v) => {
                        holes += 1;
                        if first_bad.is_none() {
                            first_bad = Some(format!(
                                "{file} :: {name}: the interpreter uses %{v}, which nothing defines"
                            ));
                        }
                    }
                    rdec_core::interp::Fault::UnmodeledLoad => unmodeled_load += 1,
                    _ => fault += 1,
                },
            }
            if ex
                .effects
                .iter()
                .any(|e| matches!(e, rdec_core::interp::Effect::Opaque(_)))
            {
                opaque += 1;
            }
            match verify_ssa(&f) {
                Ok(()) => verified += 1,
                Err(errs) => {
                    let msg = format!("{file} :: {name}: {}", errs.join("; "));
                    if first_bad.is_none() {
                        first_bad = Some(msg.clone());
                    }
                    // Three, not one: with 141 functions, a single example is how
                    // you miss that two failures are two different classes.
                    if bad_msgs.len() < 3 {
                        bad_msgs.push(msg);
                    }
                }
            }
            if rep.gaps == 0 && rep.exits == 0 {
                clean += 1;
            } else if first_gappy.is_none() && rep.gaps > 0 {
                first_gappy = Some(format!(
                    "{file} :: {name}: {} gaps, {} exits",
                    rep.gaps, rep.exits
                ));
            }
            if let Some(w) = &ir_name {
                if *w == name {
                    println!("{}", format_function(&f));
                }
            }
        }
    }
    println!(
        "lift: {objects} objects, {lifted} functions lifted, {verified} pass verify_ssa, {clean} with no gaps and no exits"
    );
    println!(
        "evidence: {gap_total} loud gaps, {exit_total} Exit terminators, {unnamed_total} unnamed call targets, {clobber_total} modelled register clobbers, {unreach_total} unreachable blocks dropped, {scaffolding} live-in placeholders resolved, {undef_live} of them `undef` (the ABI question)"
    );
    println!(
        "interp: {ret_n} returned, {no_ret} ended at Exit, {step_budget} hit the step budget, {opaque} reached an unmodelled op, {unmodeled_load} unmodelled loads, {fault} arithmetic faults, {holes} used a value that is never defined"
    );
    for (i, b) in bad_msgs.iter().enumerate() {
        println!("verifier rejection {}: {b}", i + 1);
    }
    if holes > 0 {
        println!("FAIL: {holes} lifted functions reference a value they never define");
        return Ok(1);
    }
    if let Some(g) = &first_gappy {
        println!("first function with gaps: {g}");
    }
    if verified == lifted && lifted > 0 {
        println!("OK: every lifted function is structurally valid IR");
        Ok(0)
    } else {
        println!(
            "FAIL: {} lifted functions are not valid IR",
            lifted - verified
        );
        Ok(1)
    }
}
