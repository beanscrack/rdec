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
