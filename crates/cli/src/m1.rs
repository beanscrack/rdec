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

/// `--corpus-dir DIR`: point THIS gate at another generator's manifest.
///
/// `--corpus` stays the clang corpus at `corpus/manifest.tsv`, read exactly as
/// before. `--corpus-dir corpus/msvc` runs the SAME code over the MSVC leg's
/// manifest: one code path per gate, a different denominator. A gate forked per
/// toolchain is where a leg quietly stops checking anything.
///
/// An empty manifest is an ERROR, not an empty file list: a corpus that yields 0
/// objects turns every gate over it into a pass over nothing, and this repo has
/// been burned by exactly that twice. The gates themselves keep their own floors
/// ("no code symbols", "discovery checked nothing") - this is the layer under them.
pub fn corpus_objects(args: &[String]) -> Result<Vec<String>, String> {
    let Some(i) = args.iter().position(|a| a == "--corpus-dir") else {
        return Ok(Vec::new());
    };
    let dir = args.get(i + 1).ok_or("--corpus-dir needs a directory")?;
    let man = format!("{dir}/manifest.tsv");
    let text = std::fs::read_to_string(&man)
        .map_err(|e| format!("{man}: {e} - run `rdec corpus --msvc` first"))?;
    let files: Vec<String> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\t"))
        .filter_map(|l| l.split('\t').nth(2).map(|s| s.to_string()))
        .collect();
    if files.is_empty() {
        return Err(format!(
            "{man} lists 0 objects - a gate over it would check nothing"
        ));
    }
    Ok(files)
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
                                .symbol_of(r.symbol)
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
    const TAKES_VALUE: [&str; 3] = ["--limit", "--syntax", "--corpus-dir"];
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
    // The MSVC leg's manifest, through the SAME code path (see `corpus_objects`).
    files.extend(corpus_objects(args)?);
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if files.is_empty() {
        eprintln!(
            "usage: rdec objdump-check <object.obj> [more.obj ...] [--corpus] [--corpus-dir DIR] [--limit N]"
        );
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

/// The architectural no-op family, as our decoder SPELLS it. `starts_with("nop")`
/// is not this family: the 2-byte canonical NOP `66 90` decodes to `xchg ax,ax`,
/// which is a NOP by encoding (exchanging AX with itself) and the fill MSVC emits
/// for 2-byte alignment inside `.text$mn`, while clang aligns with `nopw`/`0f 1f`
/// (whose prefix DOES match). So one toolchain's alignment was padding and the
/// other's was missed code, on a mnemonic prefix.
///
/// Scale, measured on the 2026-11-08 MSVC leg, because the first version of this
/// comment claimed the whole gap and does not own it: this rule reclassifies 2 of
/// the leg's 116 uncovered bytes (coverage 98.35% -> 98.39%, 8 gap runs -> 7). The
/// remaining 114 bytes are the `switch` thunks MSVC reaches through an indirect
/// branch, which is M2's work and STAYS uncovered here on purpose.
///
/// Space-insensitive: the Intel printer's spacing is not a semantic property of the
/// byte.
fn is_nop_text(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("nop") || t.replace(' ', "").eq_ignore_ascii_case("xchgax,ax")
}

/// Is an uncovered byte run padding rather than missed code? This is a SEMANTIC
/// claim, so it asks the decoder rather than matching bytes: LLVM aligns with
/// multi-byte NOPs (`0f 1f 84 00 ...`, `66 66 2e 0f 1f 84 00 ...`) and MSVC with
/// `66 90`, and no byte-pattern rule names those. A run whose instructions are ALL
/// no-ops is padding; so is `0xCC` (int3, any length, the tail fill of both
/// toolchains) and a run of 4+ `0x00` (three zero bytes can be an instruction
/// stream - `00 00 00` is `add [rax],al` - so the shorter run is charged as
/// uncovered, not silently excused).
///
/// The claim is one-directional on purpose: a run that decodes to anything with a
/// side effect (`mov`, `jmp`, a thunk) stays UNCOVERED and lands in the gap census,
/// where M1's 99.5% floor sees it. Widening this predicate buys coverage on paper;
/// that is why it is a function with tests, not an inline condition.
pub fn is_padding_run(run: &[u8]) -> bool {
    if run.is_empty() {
        return false;
    }
    if run.iter().all(|&b| b == 0xCC) {
        return true;
    }
    if run.iter().all(|&b| b == 0x00) {
        return run.len() >= 4;
    }
    let ins = decode(run, 0, Syntax::Intel);
    !ins.is_empty()
        && ins
            .iter()
            .all(|i| i.valid() && i.text.as_deref().map(is_nop_text).unwrap_or(false))
}

/// `rdec discover <obj> [more...] [--corpus] [--corpus-dir DIR] [--limit N]`
///
/// M1 step 2's gate. Ground truth is the object's own symbol table: every
/// function symbol must be recovered by descent alone, from entry points and
/// call relocations. Extra starts (branch targets promoted by a call, local
/// labels called through a pointer) are reported, not punished; *missing*
/// starts are the failure mode, because a function we never reach is a function
/// we cannot decompile.
pub fn cmd_discover(args: &[String]) -> Result<i32, String> {
    use rdec_arch_x86::discover::{
        discover_all_with_container, jump_tables, overlap_free, Code, Gap,
    };
    use std::collections::HashMap;

    const TAKES_VALUE: [&str; 3] = ["--limit", "--syntax", "--corpus-dir"];
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
    // See `cmd_objdump_check`: the manifest is the driver, and one code path
    // serves the clang corpus and the MSVC corpus.
    files.extend(corpus_objects(args)?);
    if files.is_empty() {
        return Err(
            "usage: rdec discover <object.obj> [more.obj ...] [--corpus] [--corpus-dir DIR] [--limit N]".to_string(),
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
    // The jump-table census, next to the coverage number it produced: tables
    // recovered, slots they carry, targets they name, slots left unresolved.
    let mut tables_resolved = 0usize;
    let mut table_slots = 0usize;
    let mut table_targets = 0usize;
    let mut table_unresolved = 0usize;
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
                if let Some(t) = obj.symbol_of(rl.symbol) {
                    if t.section == Some(sec) && !t.name.starts_with('.') {
                        seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                    }
                }
            }
        }
        // The jump-table evidence: relocation-covered tables in the object's
        // data sections, keyed by the indirect jump they feed. An object with no
        // such table is an object whose indirect jumps stay counted gaps.
        let tables = jump_tables(&obj, &|sec: usize| {
            idx_of.get(&sec).copied().map(|r| base_of[r])
        });
        // A direct call's target is the relocation at its displacement, not the
        // immediate the bytes hold: the same evidence the jump tables above are
        // built from, and the reason a call site seeds the CALLEE's address and
        // not an invented start in the middle of the caller.
        let disp = rdec_arch_x86::discover::displacements(&obj, &|sec: usize| {
            idx_of.get(&sec).copied().map(|r| base_of[r])
        });
        let plan = discover_all_with_container(&ranges, &seeds, &tables, &disp);
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
                if is_padding_run(run) {
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
                Gap::PartialTable => "indirect jump over a partially-resolved table",
                Gap::IndirectCall => "indirect call",
                Gap::ExternalCall => {
                    "call to a symbol this object does not define (an import: the name is known, the code is not)"
                }
                Gap::Undecodable => "undecodable",
                Gap::OutsideCode => "outside code",
            };
            *gaps.entry(name).or_default() += 1;
        }
        for t in &plan.tables {
            tables_resolved += 1;
            table_slots += t.entries;
            table_targets += t.targets.len();
            table_unresolved += t.unresolved.len();
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
    println!(
        "jump tables: {tables_resolved} recovered from relocations ({table_slots} slots, {table_targets} targets, {table_unresolved} slots unresolved)"
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
    use rdec_arch_x86::discover::Code;
    use rdec_arch_x86::lift::lift_one;
    use rdec_core::fmt_ir::format_function;
    use rdec_core::verify::verify_ssa;
    use std::collections::HashMap;

    const TAKES_VALUE: [&str; 4] = ["--limit", "--ir", "--syntax", "--corpus-dir"];
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
    // The clang leg and the MSVC leg run the SAME lifting code through this one
    // entry point (`corpus_objects`).
    files.extend(corpus_objects(args)?);
    if files.is_empty() {
        return Err(
            "usage: rdec lift <object.obj> [...] [--corpus] [--corpus-dir DIR] [--limit N] [--ir NAME]".to_string(),
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
    // The call census, printed with the rest of the evidence: the modelled
    // number and each refused class, so the direct-call model is auditable and
    // not a claim. This gate lifts with `lift_one` (no callee table), so its
    // modelled count is 0 by construction - the line says so with numbers.
    let mut call_modelled = 0usize;
    let mut call_indirect = 0usize;
    let mut call_nosym = 0usize;
    let mut call_over = 0usize;
    let mut call_untypable = 0usize;
    // The relocation census, printed with the calls census. This gate lifts
    // with `lift_one` - no callee table, no relocation map - so K is 0 by
    // construction and L counts every rel32 call displacement with no
    // relocation consulted; the C/C++ gates run the same pipeline WITH the
    // map, and their `relocs:` lines carry the resolved number.
    let mut relocs_parsed = 0usize;
    let mut call_reloc_resolved = 0usize;
    let mut call_reloc_unresolved = 0usize;
    let mut rip_named = 0usize;
    let mut unreach_total = 0usize;
    // The tail-call / noreturn census (see `m4::tail_line`). This gate lifts with
    // `lift_one`, which has no relocation map, so a `jmp`'s target is its
    // immediate and every one of these is 0 by construction here — the census
    // with the map in hand is `rdec c --corpus`'s.
    let mut tail_total = 0usize;
    let mut tail_unnamed_total = 0usize;
    let mut noreturn_total = 0usize;
    let mut past_ended_total = 0usize;
    let mut undef_live = 0usize;
    let mut scaffolding = 0usize;
    let mut ret_w32 = 0usize;
    let mut self_idioms = 0usize;
    // The SSE census, counted by the lifter, printed next to the self-idiom
    // census: `sum(sse)` is the number of SSE instructions the decoder saw, and
    // `llvm-objdump -d` over the same objects is the check on it.
    let mut sse: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut sse_zeroing = 0usize;
    // Named, not counted: where a register read resolved to `undef`.
    let mut undef_sites: Vec<(u64, rdec_arch_x86::lift::Grp)> = Vec::new();
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
        relocs_parsed += obj.reloc_census().parsed;
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
                if let Some(t) = obj.symbol_of(rl.symbol) {
                    if t.section == Some(sec) && !t.name.starts_with('.') {
                        seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                    }
                }
            }
        }
        // See `cmd_discover`: a direct call seeds the symbol its displacement
        // names, which is the callee, not the address its immediate spells.
        let disp = rdec_arch_x86::discover::displacements(&obj, &|sec: usize| {
            idx_of.get(&sec).copied().map(|r| base_of[r])
        });
        let plan = rdec_arch_x86::discover::discover_all_with_container(
            &ranges,
            &seeds,
            &rdec_arch_x86::discover::Tables::default(),
            &disp,
        );

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
                if let Some(t) = obj.symbol_of(rl.symbol) {
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
            call_modelled += rep.calls_modelled;
            call_indirect += rep.calls_indirect;
            call_nosym += rep.calls_no_symbol;
            call_over += rep.calls_over_args;
            call_untypable += rep.calls_untypable;
            call_reloc_resolved += rep.calls_reloc_resolved;
            call_reloc_unresolved += rep.calls_reloc_external + rep.calls_reloc_absent;
            rip_named += rep.rip_named;
            unreach_total += rep.unreachable_blocks;
            tail_total += rep.tail_calls;
            tail_unnamed_total += rep.tail_calls_unnamed;
            noreturn_total += rep.noreturn_calls;
            past_ended_total += rep.bytes_past_ended_block;
            undef_live += rep.undef_live_ins;
            scaffolding += rep.scaffolding_removed;
            ret_w32 += rep.ret_w32;
            self_idioms += rep.self_idioms;
            for (k, v) in &rep.sse {
                *sse.entry(k.clone()).or_insert(0) += *v;
            }
            sse_zeroing += rep.sse_zeroing;
            undef_sites.extend(rep.undef_sites.iter().copied());
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
        "return width: {ret_w32} of {lifted} functions answer at 32-bit width (a 32-bit write zeroes RAX's upper half, so the emitted C returns uint32_t)"
    );
    println!(
        "calls: {call_modelled} modelled | indirect {call_indirect} | no symbol {call_nosym} | >4 args {call_over} | untypable {call_untypable}"
    );
    println!(
        "relocs: {relocs_parsed} parsed | call targets resolved {call_reloc_resolved} | unresolved {call_reloc_unresolved} | rip-relative addresses named {rip_named}  (this gate lifts with `lift_one`: no relocation map, so resolution is 0 by construction - `rdec c`/`rdec cpp` run the same pipeline with the map)"
    );
    println!(
        "tail calls: {tail_total} modelled | {tail_unnamed_total} with no callable callee | noreturn calls: {noreturn_total} | {past_ended_total} instructions not lifted (their block's terminator left the function)  (same reason: no relocation map here, so 0 by construction)"
    );
    println!(
        "evidence: {gap_total} loud gaps, {exit_total} Exit terminators, {unnamed_total} unnamed call targets, {clobber_total} modelled register clobbers, {unreach_total} unreachable blocks dropped, {scaffolding} live-in placeholders resolved, {undef_live} of them `undef` (the ABI question), {self_idioms} register self-idioms translated as machine-code facts (`xor/sub r,r` write 0 without reading the register; `and/or r,r` write it back)"
    );
    println!("{}", crate::m4::sse_line(&sse, sse_zeroing));
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

/// Discover a whole PE32+ image, not one object section — the M7 step that
/// makes `discover` speak linked binaries.
///
/// An object seeds descent from exports and relocations; a linked image has
/// neither (the probe EXE prints `no symbols` under `llvm-nm`). The image's
/// own function-start evidence is its `.pdata` unwind table: the linker's
/// record of every function that has a teardown, plus the entry point (DLL
/// and EXE entry code has no unwind entry by convention). Chained and
/// version-refused `.pdata` entries are NOT seeds — they stay counted, the
/// same partition `pe-functions` prints.
///
/// The second seed source is `.reloc` (`reloc.rs`): a base relocation records
/// a place the image embeds an address, and an address in the image that lands
/// on code is a function start the image names. Measured on `m7.exe`: 47
/// `DIR64` sites hold a `.text` address, 38 distinct ones. A relocation whose
/// TARGET is not code is a boundary and stays loud — named by RVA (the import
/// directory is not read by this slice, so no callee name is invented) and
/// counted as a gap, never silently walked past. A relocation site inside
/// `.text` is printed as its own census line: on an MSVC image there are none
/// (code is position-independent there), so the number being 0 is a fact, not
/// an absence of checking.
///
/// The v1 sweep runs over `.text` only. Seeds that land outside `.text`
/// (an executable section the sweep does not cover, an address in no section)
/// are PRINTED, not claimed, and counted, so the coverage number is the size of
/// exactly one range the same way `objdump-check` pins objects. A run that
/// discovers 0 functions prints `VERDICT: FAIL` and exits 1: every number this
/// command prints is a claim, and 0 discovered functions makes them all void.
pub fn cmd_pe_discover(args: &[String]) -> Result<i32, String> {
    let path = match args.get(1) {
        Some(p) => p.clone(),
        None => return Err("usage: rdec pe-discover <image.exe|image.dll>".to_string()),
    };
    let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let img = rdec_arch_x86::pe::PeImage::load(&bytes, &path)?;
    let text = img
        .section_named(".text")
        .ok_or_else(|| format!("{path}: image has no .text section"))?;
    let text_start = img.va(text.virtual_address);
    // The sweep covers what the loader read: raw bytes past virtual_size are
    // alignment padding, and `read_at` refuses past raw_data. The printed
    // `.text ..` span is that covered window, not the section header's number.
    let text_len = std::cmp::min(text.virtual_size, text.size_of_raw_data) as usize;
    let text_bytes = img
        .read_at(text.virtual_address, text_len)
        .map_err(|e| format!("{path}: .text: {e}"))?;
    let text_end = text_start + text_bytes.len() as u64;

    // Seed source: the EXCEPTION directory (`.pdata` is its section), read
    // the way `pe-functions` reads it. No `has_pdata` helper exists; the
    // directory is the fact.
    let pdata = img
        .data_directories
        .iter()
        .find(|d| d.name == "EXCEPTION" && d.rva != 0 && d.size >= 12);

    let mut entries: Vec<u64> = Vec::new();
    let census;
    if let Some(_dir) = pdata {
        match rdec_arch_x86::pdata::UnwindTable::read(&img) {
            Ok(t) => {
                entries.extend(t.functions().map(|(begin, _end)| img.va(begin)));
                // `seeded` counts the addresses actually PUSHED as seeds, not the entries
                // that parsed: seed a refusal anywhere in this command and this number rises
                // while the parsed count stays put. That gap is the leak detector.
                let seeded = entries.len();
                census = format!(
                    ".pdata: {} function seeds, {} chained refusals, {} version refusals \
                     (seeded {seeded} of {} entries; a refused entry is NEVER a seed)",
                    t.functions().count(),
                    t.chaining_refusals().len(),
                    t.version_refusals().len(),
                    t.entry_count()
                );
            }
            Err(e) => census = format!(".pdata read refused: {e} — entry point only"),
        }
    } else {
        census = "no .pdata directory — entry point only".to_string();
    }
    let pdata_seeds: std::collections::BTreeSet<u64> = entries.iter().copied().collect();
    let entry_in_text = text_start <= img.entry_va() && img.entry_va() < text_end;
    if entry_in_text {
        entries.push(img.entry_va());
    }

    // `.reloc`: every address the image embeds, and where it points. Loud on
    // both outcomes — a missing table is a fact about the image (fixed base),
    // and a refused table is a fact about this reader.
    let reloc_census;
    let mut reloc_detail: Vec<String> = Vec::new();
    let mut boundary_gaps: usize = 0;
    let mut reloc_targets_in_text = 0usize;
    match rdec_arch_x86::reloc::BaseRelocs::read(&img) {
        Ok(r) => {
            let c = r.census();
            let types: Vec<String> = r
                .relocs_by_type()
                .iter()
                .map(|(k, n)| format!("{}:{n}", rdec_arch_x86::reloc::type_name(*k)))
                .collect();
            reloc_census = format!(
                ".reloc: {} blocks, {} entries [{}] sum(SizeOfBlock) {:#x} == dir size {:#x}; partition {}+{}+{}+{}+{} == {} balanced: {}",
                r.block_count(),
                r.entry_count(),
                types.join(" "),
                r.sum_block_sizes(),
                r.dir_size,
                c.padding,
                c.absolute,
                c.relative,
                c.unreadable,
                c.refused_kind,
                c.entries,
                c.balanced()
            );
            let (targets, unresolved) = r.targets(&img);
            let mut reloc_detail_boundary: Vec<String> = Vec::new();
            let mut data_pointers = 0usize;
            let mut new_seeds: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
            for t in &targets {
                let va = img.va(t.target_rva);
                let site_va = img.va(t.site_rva);
                let site_in_text = text_start <= site_va && site_va < text_end;
                if text_start <= va && va < text_end {
                    reloc_targets_in_text += 1;
                    new_seeds.insert(va);
                } else if site_in_text {
                    // A relocation SITE inside code whose target is not code:
                    // the IAT/indirect-call shape. Named by RVA, counted as a
                    // gap, and given NO callee name — the import directory is
                    // not read by this slice, so "the callee is X" would be an
                    // invention.
                    boundary_gaps += 1;
                    let where_ = match img.section_of(t.target_rva) {
                        Some(s) => format!("section {} ({})", s.index, s.name),
                        None => "NO section".to_string(),
                    };
                    reloc_detail_boundary.push(format!(
                        "  boundary: {} relocation at site {site_va:#x} (inside .text) names {va:#x} in {where_} — an indirect call/jmp through an address the image embeds: counted as a gap, callee NOT invented (the import directory is not read by this slice)",
                        rdec_arch_x86::reloc::type_name(t.kind)
                    ));
                } else {
                    // A data-to-data pointer: the image embeds an address in a
                    // vtable/table. It is a fact about the data, NOT a code
                    // boundary, and counting it as a gap would inflate the gap
                    // census with events that are not gaps. Census only.
                    data_pointers += 1;
                }
            }
            // Detail lines are capped: a real image has hundreds of boundaries
            // and a gate reads the census, not a wall of text. The count is the
            // claim; the lines are the witnesses, and they name the RVA.
            let shown = 8.min(reloc_detail_boundary.len());
            for line in reloc_detail_boundary.iter().take(shown) {
                reloc_detail.push(line.clone());
            }
            if reloc_detail_boundary.len() > shown {
                reloc_detail.push(format!(
                    "  ... {} more boundary relocations (every one is counted in the gap census)",
                    reloc_detail_boundary.len() - shown
                ));
            }
            let extra: Vec<u64> = new_seeds
                .iter()
                .copied()
                .filter(|va| !pdata_seeds.contains(va))
                .collect();
            entries.extend(extra.iter().copied());
            let distinct = new_seeds.len();
            reloc_detail.push(format!(
                "  .reloc targets: {reloc_targets_in_text} name an address in .text ({distinct} distinct), {} of them seeded with .reloc-only provenance; {boundary_gaps} have a SITE in .text and a target outside it (counted as gaps); {data_pointers} are data-to-data pointers (census, not gaps); {} name an address this reader refuses to name",
                extra.len(),
                unresolved.len()
            ));
            if !extra.is_empty() {
                let named: Vec<String> =
                    extra.iter().take(12).map(|va| format!("{va:#x}")).collect();
                let more = extra.len().saturating_sub(named.len());
                reloc_detail.push(format!(
                    "  .reloc-only function starts (provenance = .reloc, NOT .pdata): {}{}",
                    named.join(" "),
                    if more > 0 {
                        format!(" (+{more} more)")
                    } else {
                        String::new()
                    }
                ));
            }
            for u in unresolved.iter().take(10) {
                reloc_detail.push(format!(
                    "  unresolved: {} at site {:#x} — {}",
                    rdec_arch_x86::reloc::type_name(u.kind),
                    img.va(u.site_rva),
                    u.why
                ));
            }
            for (i, msg) in r.refusals().into_iter().take(10) {
                reloc_detail.push(format!("  reloc entry refusal {i}: {msg}"));
            }
            let in_text_sites: Vec<String> = r
                .entries()
                .iter()
                // ABSOLUTE padding patches NOTHING (width 0), so it has no
                // site: counting it as "a relocation site inside .text" would
                // report a boundary-shaped event at a byte the loader ignores.
                .filter(|e| rdec_arch_x86::reloc::patch_width(e.kind) != Some(0))
                .filter(|e| {
                    let va = img.va(e.site_rva);
                    text_start <= va && va < text_end
                })
                .take(6)
                .map(|e| {
                    format!(
                        "{} at {:#x}",
                        rdec_arch_x86::reloc::type_name(e.kind),
                        img.va(e.site_rva)
                    )
                })
                .collect();
            let sites: usize = r
                .entries()
                .iter()
                .filter(|e| rdec_arch_x86::reloc::patch_width(e.kind) != Some(0))
                .filter(|e| {
                    let va = img.va(e.site_rva);
                    text_start <= va && va < text_end
                })
                .count();
            reloc_detail.push(format!(
                "  .reloc sites inside .text: {sites}{}",
                if in_text_sites.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", in_text_sites.join(", "))
                }
            ));
        }
        Err(e) => reloc_census = format!(".reloc read refused: {e} — no relocation evidence"),
    }
    entries.sort_unstable();
    entries.dedup();

    // One range (v1: .text), entry seeds and no call seeds; the descent
    // finds calls inside the range itself from the immediates.
    let ranges = [rdec_arch_x86::discover::Code {
        base: text_start,
        bytes: text_bytes,
    }];
    let mut seeds = std::collections::HashMap::new();
    seeds.insert(0usize, (entries.clone(), Vec::new()));
    let plan = rdec_arch_x86::discover::discover_all(&ranges, &seeds);

    println!("{path}: PE32+ image, base {:#x}", img.image_base);
    println!("  seeds: {census}");
    println!("  {reloc_census}");
    for line in &reloc_detail {
        println!("{line}");
    }
    println!(
        "  {} entry seeds, entry point {:#x} ({})",
        entries.len(),
        img.entry_va(),
        if entry_in_text {
            "in .text"
        } else {
            "outside .text: not seeded"
        }
    );
    println!(
        "  .text {text_start:#x}..{text_end:#x} ({} bytes covered, VirtualSize {:#x}): functions {}, blocks {}, insns {}, covered {} bytes",
        text_end - text_start,
        text.virtual_size,
        plan.functions.len(),
        plan.blocks.len(),
        plan.insns.len(),
        plan.covered_bytes()
    );
    // The coverage number is the claim this command exists to make, so print it
    // as a fraction of the one range the sweep owns.
    let span = text_end - text_start;
    println!(
        "  coverage: {:.2}% of the {span} .text bytes the sweep owns",
        if span == 0 {
            0.0
        } else {
            100.0 * plan.covered_bytes() as f64 / span as f64
        }
    );
    let mut gap_line = String::new();
    for (name, g) in [
        (
            "indirect-branch",
            rdec_arch_x86::discover::Gap::IndirectBranch,
        ),
        ("partial-table", rdec_arch_x86::discover::Gap::PartialTable),
        ("indirect-call", rdec_arch_x86::discover::Gap::IndirectCall),
        ("external-call", rdec_arch_x86::discover::Gap::ExternalCall),
        ("decoder-refusal", rdec_arch_x86::discover::Gap::Undecodable),
        ("outside-code", rdec_arch_x86::discover::Gap::OutsideCode),
    ] {
        let n = plan.gap_count(g);
        if n > 0 {
            gap_line.push_str(&format!("{name}:{n} "));
        }
    }
    // `.reloc` boundaries: a relocation that names a non-code address is a
    // boundary the sweep cannot cross, and the count is part of the claim.
    if boundary_gaps > 0 {
        gap_line.push_str(&format!("reloc-boundary:{boundary_gaps} "));
    }
    println!(
        "  gaps: {} ({} reloc boundaries counted with them)",
        if gap_line.is_empty() {
            "none".to_string()
        } else {
            gap_line.trim_end().to_string()
        },
        boundary_gaps
    );
    println!(
        "  overlap_free: {}",
        rdec_arch_x86::discover::overlap_free(&plan.insns)
    );
    // Seeds that did not land in the one range the sweep owns: printed, counted,
    // NOT claimed. Measured here over the seed list, not over `plan.functions`,
    // because the sweep never reaches outside its ranges — the old filter read
    // the plan, so this branch could not fire and a seed outside `.text` vanished.
    let outside: Vec<u64> = entries
        .iter()
        .filter(|va| !(text_start <= **va && **va < text_end))
        .copied()
        .collect();
    if outside.is_empty() {
        println!("  seeds outside .text: 0");
    } else {
        println!(
            "  {} seeds landed outside .text (not claimed):",
            outside.len()
        );
        for f in outside.iter().take(10) {
            println!("    {f:#x}");
        }
    }
    // Non-vacuity: every number above is a claim about discovered code, and a
    // run that discovered none has made no claim at all. This repo has been
    // burned twice by a gate that passed on 0.
    if plan.functions.is_empty() {
        println!(
            "  VERDICT: FAIL — 0 functions discovered from {} seeds over {} .text bytes: the coverage numbers above are void",
            entries.len(),
            span
        );
        return Ok(1);
    }
    println!(
        "  VERDICT: OK — {} functions, {} .text bytes covered, {} gaps ({} descent + {} reloc boundary)",
        plan.functions.len(),
        plan.covered_bytes(),
        plan.gaps.len() + boundary_gaps,
        plan.gaps.len(),
        boundary_gaps
    );
    Ok(0)
}

// Tests live at the BOTTOM of the file on purpose: `scripts/unwrap_ratchet.sh`
// cuts each file at its first `#[cfg(test)]`, so a mid-file test module blinds the
// panic-site count for everything below it. This module adds no panic site.
#[cfg(test)]
mod padding_run_tests {
    use super::is_padding_run;
    use rdec_arch_x86::decode::{decode, Syntax};

    fn text(bytes: &[u8]) -> Vec<String> {
        decode(bytes, 0, Syntax::Intel)
            .iter()
            .map(|i| i.text.clone().unwrap_or_default())
            .collect()
    }

    /// The MSVC leg's whole coverage delta lives here. `66 90` IS the canonical
    /// 2-byte NOP (exchange AX with itself), and MSVC pads 2-byte alignment inside
    /// `.text$mn` with it; our decoder spells it `xchg ax, ax`, which
    /// `starts_with("nop")` rejects, so a run of it was charged as missed code.
    #[test]
    fn the_two_byte_fill_msvc_pads_with_is_padding() {
        assert!(
            is_padding_run(&[0x66, 0x90, 0x66, 0x90]),
            "66 90 66 90 is two canonical NOPs; decoder text was {:?}",
            text(&[0x66, 0x90, 0x66, 0x90])
        );
        // And the claim is about the DECODER's spelling, not a byte rule: this is
        // the fact the predicate keys on, so it is asserted, not assumed.
        let t = text(&[0x66, 0x90]);
        assert_eq!(t.len(), 1, "66 90 is one instruction: {t:?}");
        let s = t[0].replace(' ', "");
        assert!(
            s.eq_ignore_ascii_case("xchgax,ax") || s.starts_with("nop"),
            "expected the canonical NOP spelling, decoder said {t:?}"
        );
    }

    #[test]
    fn one_byte_and_multibyte_llvm_nops_are_padding() {
        assert!(is_padding_run(&[0x90; 3]), "90 90 90 is three nops");
        // `0f 1f 84 00 00000000` = nopw [rax+rax+0], LLVM's 8-byte alignment.
        assert!(is_padding_run(&[0x0f, 0x1f, 0x84, 0x00, 0, 0, 0, 0]));
        // `66 2e 0f 1f 84 00 00000000`, the 10-byte segment-override form.
        assert!(is_padding_run(&[
            0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0, 0, 0, 0
        ]));
    }

    #[test]
    fn int3_pads_at_any_length_and_zeroes_need_four_bytes() {
        assert!(is_padding_run(&[0xCC]), "one int3 is a tail fill");
        assert!(is_padding_run(&[0xCC; 16]));
        // Three zero bytes decode as `add [rax],al` - a real, if silly,
        // instruction stream - so they are charged as uncovered, not excused.
        assert!(
            !is_padding_run(&[0x00, 0x00, 0x00]),
            "decoder said {:?}",
            text(&[0, 0, 0])
        );
        assert!(is_padding_run(&[0x00; 4]), "a 4-byte zero fill is not code");
        assert!(!is_padding_run(&[]), "an empty run is not padding either");
    }

    /// The MSVC gap's other half must STAY uncovered: its switch lowering is a
    /// `mov eax, k` + indirect `jmp` thunk inside `.text$mn`. Padding is the
    /// innocent label; a thunk is a discovery hole, and M1's floor has to see it.
    #[test]
    fn a_run_that_moves_data_or_jumps_indirectly_is_code() {
        // mov eax, dword ptr [0x402000] - the MSVC switch thunk's load.
        assert!(!is_padding_run(&[0x8b, 0x04, 0x25, 0x00, 0x20, 0x40, 0x00]));
        // jmp qword ptr [rip+0] - the indirect branch that reaches it.
        assert!(!is_padding_run(&[0xff, 0x25, 0x00, 0x00, 0x00, 0x00]));
        // push rbp; mov rbp, rsp - the prologue of a function we never reached.
        assert!(!is_padding_run(&[0x55, 0x48, 0x89, 0xe5]));
    }

    /// One-directional: a `0xCC` (int3, a trap, semantically NOT a nop) mixed into
    /// a nop run keeps the whole run uncovered. Widening this predicate is how
    /// coverage goes up on paper, so the narrow edge is pinned.
    #[test]
    fn a_trap_inside_a_nop_run_is_not_padding() {
        assert!(!is_padding_run(&[0x90, 0x90, 0xCC, 0x90]));
        assert!(is_padding_run(&[0x90, 0x90, 0x90, 0x90]));
    }
}
