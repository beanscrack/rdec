//! M4: lifted IR -> C text -> clang. The first end of the pipeline.
//!
//! The gate has three parts, and all three have to be reported:
//!
//!  1. every lifted function either emits C or is *refused* with a reason
//!     (a refusal is a real gap: the count is the number, the reason is the work
//!     item, and neither is allowed to be a comment inside the C);
//!  2. the emitted C compiles under clang `-Werror`, so a width, a signedness or
//!     an uninitialised-local problem is a build failure, not a shrug;
//!  3. the passes that run before emission must not break SSA - checked with the
//!     same verifier, not with trust.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::emit_c::{emit_c, Refusal, PRELUDE};
use rdec_core::ir::Function;
use rdec_core::verify::verify_ssa;
use std::collections::HashMap;
use std::path::Path;

/// The pre-emission pipeline. Flag elimination first: `Ty::Flags` has no C
/// spelling on purpose, so a surviving flags value is a refusal, and the pass
/// that removes it is the thing that has to run.
fn clean(f: &mut Function) {
    for _ in 0..16 {
        let mut ch = false;
        ch |= rdec_core::fuse::flag_elimination(f);
        ch |= rdec_core::passes::prune_phis(f);
        ch |= rdec_core::passes::const_fold(f);
        ch |= rdec_core::passes::copy_prop(f);
        ch |= rdec_core::passes::simplify_branches(f);
        ch |= rdec_core::passes::dead_store_elimination(f);
        ch |= rdec_core::passes::dead_code_elimination(f);
        if !ch {
            break;
        }
    }
}

/// One refusal key per class of gap, so the report says what to build next
/// instead of dumping 200 sentences.
fn key(r: &Refusal) -> String {
    let s = r.reason.trim();
    // "opaque: live-in R8" and "opaque: syscall" are different work items; the
    // category alone would report 112 of one thing and hide what to build.
    if let Some(rest) = s.strip_prefix("opaque:") {
        let rest = rest.trim();
        let cut = rest.find(" at ").unwrap_or(rest.len());
        return format!("opaque: {}", rest[..cut.min(40)].trim());
    }
    let low = s.to_lowercase();
    let cut = low.find(':').unwrap_or(low.len());
    low[..cut.min(46)].trim().to_string()
}

fn c_ident(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn cmd_c(args: &[String]) -> Result<i32, String> {
    let mut files: Vec<String> = Vec::new();
    for a in args.iter().skip(1) {
        if a == "--corpus" {
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
            continue;
        }
        if !a.starts_with("--") {
            files.push(a.clone());
        }
    }
    if files.is_empty() {
        return Err("usage: rdec c <object.obj> [...] [--corpus] [--emit DIR]".to_string());
    }
    let out_dir = args
        .iter()
        .position(|a| a == "--emit")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "corpus/c".to_string());
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let mut seen = 0usize;
    let mut emitted = 0usize;
    let mut refused = 0usize;
    let mut pass_broken = 0usize;
    let mut keys: HashMap<String, usize> = HashMap::new();
    let mut first_refusal: Option<String> = None;
    let mut first_pass_break: Option<String> = None;
    let mut all = String::new();
    // The per-function text carries its own includes (so a single emitted
    // function is a compilable translation unit on its own); the concatenated
    // file needs them before the shared prelude, which uses uint64_t.
    all.push_str("#include <stdint.h>\n#include <string.h>\n\n");
    all.push_str(PRELUDE);
    all.push('\n');

    for (oi, file) in files.iter().enumerate() {
        let obj = rdec_arch_x86::loader::load(Path::new(file))?;
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
        let mut names: HashMap<u64, String> = HashMap::new();
        for i in obj.code_symbols() {
            let s = &obj.symbols[i];
            let sec = s.section.unwrap();
            if obj.is_code(sec) && !s.name.starts_with('.') && s.size > 0 {
                names.insert(base_of[idx_of[&sec]] + s.offset, s.name.clone());
            }
        }

        for (k, &start) in plan.functions.iter().enumerate() {
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
            // Unique per object: the corpus has 120 objects that all define `f`,
            // and one C file with 120 definitions of `f` is a redefinition error.
            let raw = names
                .get(&start)
                .cloned()
                .unwrap_or_else(|| format!("fn_{start:x}"));
            let name = format!("o{oi}_{}", c_ident(&raw));
            let (mut f, _rep) = lift_one(&name, rg.bytes, rg.base, start, end, &plan, &names);
            seen += 1;
            clean(&mut f);
            if let Err(errs) = verify_ssa(&f) {
                pass_broken += 1;
                // The post-pass IR of a broken function is the only way to see
                // WHICH op a pass emitted at the wrong width.
                if pass_broken <= 3 {
                    let dump = format!("{out_dir}/passbroken_{pass_broken}.ir");
                    let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&f));
                }
                if first_pass_break.is_none() {
                    first_pass_break = Some(format!("{file} :: {raw}: {}", errs.join("; ")));
                }
                continue;
            }
            match emit_c(&f) {
                Ok(out) => {
                    emitted += 1;
                    all.push_str(&out.text);
                    all.push('\n');
                }
                Err(rs) => {
                    refused += 1;
                    // When a function is refused, the post-pass IR IS the
                    // diagnosis: the refusal names a value, and the only useful
                    // question is what the passes left around it.
                    if refused <= 3 {
                        let dump = format!("{out_dir}/refused_{refused}.ir");
                        let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&f));
                    }
                    for r in &rs {
                        *keys.entry(key(r)).or_default() += 1;
                    }
                    if let Some(r) = rs.first() {
                        if first_refusal.is_none() {
                            first_refusal =
                                Some(format!("{file} :: {raw} @ {:#x}: {}", r.at, r.reason));
                        }
                    }
                }
            }
        }
    }

    let path = format!("{out_dir}/all.c");
    std::fs::write(&path, &all).map_err(|e| format!("{path}: {e}"))?;

    println!("emit-c: {seen} lifted functions, {emitted} emitted C, {refused} refused, {pass_broken} invalid after passes");
    let mut ks: Vec<(String, usize)> = keys.into_iter().collect();
    ks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if !ks.is_empty() {
        let top: Vec<String> = ks.iter().take(6).map(|(k, n)| format!("{n} {k}")).collect();
        println!("refusal classes: {}", top.join(" | "));
    }
    if let Some(r) = &first_refusal {
        println!("first refusal: {r}");
    }
    // The compile: -Werror, so the C cannot compile its way past a warning.
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let obj_out = format!("{out_dir}/all.obj");
    let out = std::process::Command::new(&cc)
        .arg("-std=c17")
        .arg("-c")
        .arg("-O0")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wno-unused-parameter")
        .arg("-Werror")
        .arg(&path)
        .arg("-o")
        .arg(&obj_out)
        .output();
    let Ok(out) = out else {
        println!("FAIL: could not run {cc} - is it on PATH? (set CC)");
        return Ok(1);
    };
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let first = err
            .lines()
            .find(|l| l.contains("error"))
            .unwrap_or("<no error line>");
        println!("FAIL: {cc} -Werror rejected the generated C ({emitted} functions emitted)");
        println!("clang: {first}");
        return Ok(1);
    }
    println!("clang -Werror: OK ({path} -> {obj_out})");
    if let Some(p) = &first_pass_break {
        // A pass that breaks SSA is reported after the compile, not before it:
        // the number of functions whose C clang accepts is a real measurement,
        // and returning early used to hide it.
        println!("FAIL: a pass broke SSA in {pass_broken} functions: {p}");
        return Ok(1);
    }
    if refused > 0 {
        println!("FAIL: {refused} functions were refused, not emitted");
        return Ok(1);
    }
    println!("OK: every lifted function emitted C that clang accepts");
    Ok(0)
}
