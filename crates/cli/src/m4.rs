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
use rdec_arch_x86::discover::Code;
use rdec_arch_x86::lift::{lift_one_sigs, CalleeSig, CalleeTable, FnShape, RelocMap, RelocTarget};
use rdec_core::emit_c::{callee_decl, emit_c_sig, text_census, Protos, Refusal, PRELUDE};
use rdec_core::ir::Function;
use rdec_core::structure::{sum_reports, Report};
use rdec_core::verify::verify_ssa;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Where one object's non-code sections live in the emitter's flat
/// `rdec_mem[1 << 20]` space. Above the frame (`STACK_BASE = 1 << 19`, where
/// every rsp-relative load/store already lives) so a stack access can never
/// collide with a data access; `DATA_STRIDE` apart per object so the 120
/// objects that all define `f` cannot collide with each other. `RDEC_ADDR`
/// masks addresses into 20 bits, and `DATA_BASE0 + 120 * DATA_STRIDE` is
/// below `1 << 20`, so the whole table fits without aliasing.
///
/// One function computes the layout and the C-side seeding in `rdec t2`
/// calls it: a drift between the two is the address a constant reads AT, the
/// kind of bug that compares clean and answers wrong.
pub const DATA_BASE0: u64 = rdec_arch_x86::lift::STACK_BASE + 4096;
pub const DATA_STRIDE: u64 = 4096;

/// One data section's bytes, placed at `va` in the flat space.
pub struct DataChunk {
    pub va: u64,
    pub bytes: Vec<u8>,
}

/// The data sections of object `oi`, mapped in section order, plus the
/// section-index -> base table `reloc_map_of` consults, plus the count of
/// sections DECLINED (a block that would run off the object's stride or the
/// 1 MiB window): counted, never silently dropped.
pub fn data_map_of(
    obj: &rdec_arch_x86::loader::Object,
    oi: usize,
) -> (Vec<DataChunk>, HashMap<usize, u64>, usize) {
    let mut chunks = Vec::new();
    let mut sec_va = HashMap::new();
    let mut declined = 0usize;
    let block0 = DATA_BASE0 + oi as u64 * DATA_STRIDE;
    let block_end = DATA_BASE0 + (oi as u64 + 1) * DATA_STRIDE;
    let mut at = block0;
    let mut full = false;
    for (i, sec) in obj.sections.iter().enumerate() {
        if obj.is_code(i) {
            continue;
        }
        // A zero-raw-size section (.bss: a C++ static member, a zero-initialised
        // global) gets an address and no bytes: `rdec_mem` is zero, which is
        // exactly .bss's content rule, so there is nothing to seed.
        let len = sec.bytes.len() as u64;
        if full || at + len > block_end.min(1 << 20) {
            full = true;
            declined += 1;
            continue;
        }
        sec_va.insert(i, at);
        if len > 0 {
            chunks.push(DataChunk {
                va: at,
                bytes: sec.bytes.clone(),
            });
        }
        at += len + 16;
    }
    (chunks, sec_va, declined)
}

/// The object's relocations keyed by the address of the displacement field
/// they name, in the same per-object virtual space as `names` and the plan:
/// the table that turns `callq 0x15` (whose immediate is the UNRELOCATED
/// next-IP) into the symbol and address the `REL32` relocation at 0x11 names.
/// A symbol this object does not define keeps its name and a `None` address:
/// the call site refuses with the symbol named, not the gap swallowed.
pub fn reloc_map_of(
    obj: &rdec_arch_x86::loader::Object,
    oi: usize,
    idx_of: &HashMap<usize, usize>,
    base_of: &[u64],
) -> RelocMap {
    let (_, sec_va, _) = data_map_of(obj, oi);
    let mut map = RelocMap::new();
    for (sec, rels) in obj.sections.iter().enumerate() {
        let Some(&r) = idx_of.get(&sec) else {
            continue;
        };
        for rl in &rels.relocations {
            if !rl.kind().is_rel32() {
                continue;
            }
            let Some(sym) = obj.symbol_of(rl.symbol) else {
                continue;
            };
            let addr = match sym.section {
                // Defined by this object, in a section the pipeline has placed:
                // a code section in the plan's space, a data section in the
                // flat `rdec_mem` space.
                Some(s2) if s2 == sec => Some(base_of[r] + sym.offset),
                Some(s2) if idx_of.contains_key(&s2) => Some(base_of[idx_of[&s2]] + sym.offset),
                Some(s2) => sec_va.get(&s2).map(|b| b + sym.offset),
                // External / absolute: the name is the whole evidence.
                None => None,
            };
            map.insert(
                base_of[r] + rl.address,
                RelocTarget {
                    sym: sym.name.clone(),
                    addr,
                },
            );
        }
    }
    map
}

/// The structuring census, on the same footing as `calls_line` and `tail_line`:
/// counted by the emitter from the statements the structuring walk built, per
/// function, and summed here.
///
/// The numbers add up by construction: `N regions` is exactly
/// `if + if/else + while`, `M functions structured` counts the functions that own
/// at least one of them, and every number is cross-checked against the printed
/// text by [`census_drift`] - a figure in this line is a fact about the C, never a
/// claim about the code.
pub fn control_flow_line(reports: &[Report], emitted: usize) -> String {
    let rep = sum_reports(reports);
    let structured = reports.iter().filter(|r| r.structured()).count();
    let mut line = format!(
        "control flow: {structured}/{emitted} functions structured | {} regions: {} if | \
         {} if/else | {} while | {} short-circuit | {} blocks left as goto",
        rep.ifs + rep.if_else + rep.whiles,
        rep.ifs,
        rep.if_else,
        rep.whiles,
        rep.shorts,
        rep.flat
    );
    if !rep.reasons.is_empty() {
        let rs: Vec<String> = rep
            .reasons
            .iter()
            .map(|(k, n)| format!("{n} {k}"))
            .collect();
        line.push_str(&format!(" (reasons: {})", rs.join(", ")));
    }
    line
}

/// The readability census, printed and never thresholded: the `goto` statements
/// and `lN:` labels in the emitted C, counted from that C. A structuring change is
/// measured by this line moving down; the gate on the change is `t2`, which runs
/// the same C, and this line is what tells a reader whether it got easier to read.
pub fn readability_line(text: &str, emitted: usize) -> String {
    let t = text_census(text);
    format!(
        "readability: {emitted} emitted functions | {} goto statements | {} labels | {} structured if \
         | {} while",
        t.gotos, t.labels, t.ifs, t.whiles
    )
}

/// Where the emitter's structuring census and a scan of the emitted text disagree.
/// `None` is the healthy case: the census is then exactly what the C contains,
/// counted by two independent paths.
///
/// A structured `if` prints exactly one `if (...) {` line, an `else` arm exactly
/// one `} else {`, a `while` exactly one `while (...) {`, and a function with no
/// structured region prints none of those - so the two paths must land on the same
/// numbers. When they do not, the census describes a file that does not exist.
pub fn census_drift(reports: &[Report], text: &str) -> Option<String> {
    let rep = sum_reports(reports);
    let tc = text_census(text);
    let mut d: Vec<String> = Vec::new();
    if tc.ifs != rep.ifs + rep.if_else {
        d.push(format!(
            "the text has {} `if` statements, the walk built {}",
            tc.ifs,
            rep.ifs + rep.if_else
        ));
    }
    if tc.els != rep.if_else {
        d.push(format!(
            "the text has {} `else` arms, the walk built {} if/else regions",
            tc.els, rep.if_else
        ));
    }
    if tc.whiles != rep.whiles {
        d.push(format!(
            "the text has {} `while` statements, the walk built {}",
            tc.whiles, rep.whiles
        ));
    }
    if tc.gotos != rep.gotos {
        d.push(format!(
            "the text has {} `goto` statements, the walk counted {}",
            tc.gotos, rep.gotos
        ));
    }
    if tc.labels != rep.labels {
        d.push(format!(
            "the text has {} labels, the walk marked {}",
            tc.labels, rep.labels
        ));
    }
    if d.is_empty() {
        None
    } else {
        Some(d.join("; "))
    }
}

/// The pre-emission pipeline. Flag elimination first: `Ty::Flags` has no C
/// spelling on purpose, so a surviving flags value is a refusal, and the pass
/// that removes it is the thing that has to run.
pub(crate) fn clean(f: &mut Function) {
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
///
/// `pub` because `rdec cpp` reports the same classes for the C++ corpus: two
/// tallying schemes would mean the two gates' "refusal class" lines are not
/// comparable, which is the whole point of the class list.
pub fn key(r: &Refusal) -> String {
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

/// The one naming scheme for C identifiers built from symbol names. `rdec t2`
/// calls the emitted C by these names: a second scheme in the harness would
/// mean the gate runs a function that the `c` gate never emitted.
pub fn c_ident(s: &str) -> String {
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

/// One lifted function plus the identity every gate needs to talk about it.
///
/// `sym` is the name the *object* uses (so `rdec t2` can link a driver against
/// it) and `cname` is the name the *generated translation unit* uses. They are
/// derived from each other here, once, so the gate that emits C and the gate
/// that runs it cannot drift apart.
pub struct Subject {
    pub obj: String,
    /// The COFF symbol name (`f`, `h`), or the synthetic `fn_<addr>` that
    /// discovery gives code no symbol names. A synthetic name is not a link
    /// name: nothing in the object resolves it.
    pub sym: String,
    /// `o<object index>_<sym>` — unique across the 120 objects that all
    /// define `f`, which is why one C file can hold every function.
    pub cname: String,
    /// Entry address in the per-object virtual space the discovery plan uses.
    pub start: u64,
    /// The COFF symbol is `IMAGE_SYM_CLASS_EXTERNAL`, so another object can
    /// link against it. A file-static symbol (`h` in `call_helper`) is visible
    /// to the decompiler and invisible to a linker: `rdec t2`'s driver for it
    /// fails with `undefined symbol`, so the gate has to know which is which.
    pub external: bool,
    /// The lifted IR, after `clean`.
    pub f: Function,
    /// The declared shape this function was lifted under. Every C-corpus
    /// subject carries `FnAbi::integer()` (a C symbol names no types); the
    /// C++ pipeline fills it from the DEMANGLED SIGNATURE - the evidence is
    /// the decorated name, stated at the site that builds it, never inferred
    /// from observed answers.
    pub abi: rdec_arch_x86::lift::FnAbi,
    /// The lifter's own count report for this function. `clean` may drop
    /// instructions, so the SSE census has to come from the report and not from
    /// a scan of the surviving IR: the number is "what the decoder saw", and
    /// that is the only number `llvm-objdump -d` can check.
    pub rep: rdec_arch_x86::lift::Report,
    /// Relocations the LOADER parsed in this subject's object (all sections).
    /// Object-level, carried on every subject of the object; `relocs_line`
    /// de-duplicates by object path, so it is a census, not a per-function count.
    pub relocs_parsed: usize,
}

/// Load, discover, lift and clean every function in `files`, named by
/// `name_of(object index, COFF symbol)` and shaped by
/// `shape_of(object index, COFF symbol)`.
///
/// The naming scheme and the shape are parameters, not copies of the
/// pipeline: `rdec cpp` names the C++ functions from their DEMANGLED text and
/// declares their parameter kinds from the same evidence, and it must run the
/// same discovery, the same lifter and the same passes as `rdec c` - a C++
/// gate that re-implemented the front half would be measuring its own copy.
///
/// `shape_of` is asked for EVERY symbol of the object, not only for the
/// function being lifted, because the result is also the object's CALLEE TABLE:
/// a `call` to a sibling is only modelable when the front end can name the
/// target (the COFF symbol at that address) and say what its signature declares.
/// `FnShape::typed == false` is the loud case - a signature the demangler could
/// not fully type - and a call to such a callee stays refused.
pub fn lift_all_named_shapes(
    files: &[String],
    name_of: &dyn Fn(usize, &str) -> String,
    shape_of: &dyn Fn(usize, &str) -> FnShape,
) -> Result<Vec<Subject>, String> {
    let mut out: Vec<Subject> = Vec::new();
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
                if let Some(t) = obj.symbol_of(rl.symbol) {
                    if t.section == Some(sec) && !t.name.starts_with('.') {
                        seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                    }
                }
            }
        }
        // The jump tables this object's relocations name, keyed by dispatch site:
        // the evidence that turns `jmp *%rax` into N successors. An object with no
        // such table lifts exactly as it did.
        let tables = rdec_arch_x86::discover::jump_tables(&obj, &|sec: usize| {
            idx_of.get(&sec).copied().map(|r| base_of[r])
        });
        // The relocations this object carries on its displacement fields: the
        // evidence that a direct `call`'s target is the named symbol, not the
        // unrelocated immediate the bytes hold.
        let disp = rdec_arch_x86::discover::displacements(&obj, &|sec: usize| {
            idx_of.get(&sec).copied().map(|r| base_of[r])
        });
        let plan =
            rdec_arch_x86::discover::discover_all_with_container(&ranges, &seeds, &tables, &disp);
        // Say it out loud: the tables this object's relocations name, and how
        // many slots resolved. A table with a hole in it is visible here, not
        // only as a refusal of the one function that dispatches through it.
        let slots: usize = tables.values().map(|t| t.entries).sum();
        let unresolved: usize = tables.values().map(|t| t.unresolved.len()).sum();
        if !tables.is_empty() {
            println!(
                "  jump tables: {} recovered from relocations ({slots} slots, {unresolved} unresolved)",
                tables.len()
            );
        }
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut external: HashSet<u64> = HashSet::new();
        for i in obj.code_symbols() {
            let s = &obj.symbols[i];
            let sec = s.section.unwrap();
            if obj.is_code(sec) && !s.name.starts_with('.') && s.size > 0 {
                let at = base_of[idx_of[&sec]] + s.offset;
                names.insert(at, s.name.clone());
                if s.storage_class == rdec_arch_x86::loader::STORAGE_EXTERNAL {
                    external.insert(at);
                }
            }
        }

        // What every symbol of this object DECLARES, keyed by the COFF name: the
        // table a call site consults. Keyed by COFF name because that is what an
        // address resolves to; the value carries the name the generated C uses.
        let mut sigs: CalleeTable = CalleeTable::new();
        for raw in names.values() {
            if sigs.contains_key(raw.as_str()) {
                continue;
            }
            let sh = shape_of(oi, raw);
            sigs.insert(
                raw.clone(),
                CalleeSig {
                    cname: name_of(oi, raw),
                    abi: sh.abi,
                    typed: sh.typed,
                    // The C++ demangler knows which of these never come back; the
                    // C corpus declares none, and a name the front end does not
                    // declare is modelled as returning (the code after the call
                    // stays reachable) - the model never invents a noreturn from a
                    // guess about a name.
                    noreturn: rdec_arch_x86::lift::noreturn_symbol(raw),
                },
            );
        }

        // The object's relocations, keyed by the address of the displacement
        // field they name: the evidence a `call`'s immediate does not carry in
        // an object file, and the same table that names a RIP-relative data
        // address. Built per object, in the same per-object virtual space as
        // `names`, the plan, and `data_map_of`.
        let relocs = reloc_map_of(&obj, oi, &idx_of, &base_of);
        let relocs_parsed = obj.reloc_census().parsed;

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
            let name = name_of(oi, &raw);
            let abi = shape_of(oi, &raw).abi;
            let (mut f, rep) = lift_one_sigs(
                &name, rg.bytes, rg.base, start, end, &plan, &names, &abi, &sigs, &relocs, &tables,
            );
            clean(&mut f);
            out.push(Subject {
                obj: file.clone(),
                sym: raw,
                cname: name,
                start,
                external: external.contains(&start),
                relocs_parsed,
                f,
                abi,
                rep,
            });
        }
    }
    Ok(out)
}

/// Load, discover, lift and clean every function in `files`, named by
/// `name_of`, with every declared shape treated as fully typed. This is the
/// entry point for a front end whose shapes are all shapes it resolved; a
/// signature it cannot type goes through `lift_all_named_shapes` with
/// `FnShape::typed: false`, so its callers stay refused instead of guessing.
pub fn lift_all_named_sigs(
    files: &[String],
    name_of: &dyn Fn(usize, &str) -> String,
    abi_of: &dyn Fn(usize, &str) -> rdec_arch_x86::lift::FnAbi,
) -> Result<Vec<Subject>, String> {
    lift_all_named_shapes(files, name_of, &|oi, raw| FnShape {
        abi: abi_of(oi, raw),
        typed: true,
    })
}

/// The declared shape of every subject, keyed by the C name `Op::Call` carries.
/// One source for the whole translation unit: the prototype a caller writes for
/// `g` and the definition of `g` are both spelled from `g`'s own `Function`, so
/// `clang -Werror` can never see `conflicting types` between them.
pub fn protos_of(subjects: &[Subject]) -> Protos {
    subjects
        .iter()
        .map(|s| (s.cname.clone(), callee_decl(&s.f)))
        .collect()
}

/// The call census, on the same footing as the SSE census: counted by the lifter
/// at its own call site, so the modelled number and each refused class are
/// auditable, and the refused classes are the work list. Summed over the
/// lifter's per-function reports - `clean` deletes instructions, so a scan of
/// the surviving IR would count what the passes kept, not what the decoder saw.
pub fn calls_line(subjects: &[Subject]) -> String {
    let (mut m, mut ind, mut nosym, mut over, mut unty) = (0, 0, 0, 0, 0);
    for s in subjects {
        m += s.rep.calls_modelled;
        ind += s.rep.calls_indirect;
        nosym += s.rep.calls_no_symbol;
        over += s.rep.calls_over_args;
        unty += s.rep.calls_untypable;
    }
    format!("calls: {m} modelled | indirect {ind} | no symbol {nosym} | >4 args {over} | untypable {unty}")
}

/// The tail-call and noreturn census, on the same footing as `calls_line`:
/// the `jmp`s whose relocation named a defined function symbol (modelled as
/// `return callee(args);`), the `jmp`s whose relocation named a symbol with no
/// callable callee, the calls the front end DECLARED noreturn (BY NAME: a
/// function whose IR ends in a noreturn exit has no fall-through answer, so the
/// sites are the work list), and the instructions the lifter did not lift
/// because their block's own terminator had already left the function.
///
/// All four are 0 over the C and C++ corpora, and that is the measurement, not
/// an assumption: `llvm-objdump -d -r` over the corpus finds 65 `jmp` sites, none
/// carrying a relocation that names a defined function (the relocated ones name
/// `__guard_x86` and `_tail_call_i386`), and 31 call sites, none to the
/// termination family. The model is therefore pinned by
/// `crates/arch-x86/tests/tail_calls.rs`, whose fixtures quote their own
/// `llvm-objdump -d -r`, and a non-zero number here is the day a corpus witness
/// appears.
pub fn tail_line(subjects: &[Subject]) -> String {
    let (mut t, mut un, mut nr, mut past) = (0usize, 0usize, 0usize, 0usize);
    let mut sites: Vec<(u64, String)> = Vec::new();
    for s in subjects {
        t += s.rep.tail_calls;
        un += s.rep.tail_calls_unnamed;
        nr += s.rep.noreturn_calls;
        past += s.rep.bytes_past_ended_block;
        sites.extend(s.rep.noreturn_sites.iter().cloned());
    }
    let mut line = format!(
        "tail calls: {t} modelled | {un} with no callable callee | noreturn calls: {nr} | {past} instructions not lifted (their block's terminator left the function)"
    );
    for (a, sym) in sites {
        line.push_str(&format!(" | {a:#x} -> {sym}"));
    }
    line
}

/// The relocation census, on the same footing as `calls_line`: N is what the
/// loader parsed across the objects (counted once per object, not once per
/// function), K is direct-call displacements whose relocation resolved to a
/// defined symbol, L is the call sites that stayed loud - the relocation named
/// a symbol the object does not define, named nothing, or the call's
/// displacement carried no relocation at all. `llvm-objdump -r` over the same
/// objects checks N; the lifter's own counters check K and L.
pub fn relocs_line(subjects: &[Subject]) -> String {
    let mut by_obj: HashMap<&str, usize> = HashMap::new();
    let (mut resolved, mut unresolved) = (0usize, 0usize);
    for s in subjects {
        by_obj.insert(s.obj.as_str(), s.relocs_parsed);
        resolved += s.rep.calls_reloc_resolved;
        unresolved += s.rep.calls_reloc_external + s.rep.calls_reloc_absent;
    }
    let parsed: usize = by_obj.values().sum();
    format!("relocs: {parsed} parsed | call targets resolved {resolved} | unresolved {unresolved}")
}

/// Load, discover, lift and clean every function in `files`, named by
/// `name_of(object index, COFF symbol)`, under the integer model.
pub fn lift_all_named(
    files: &[String],
    name_of: &dyn Fn(usize, &str) -> String,
) -> Result<Vec<Subject>, String> {
    lift_all_named_sigs(files, name_of, &|_, _| {
        rdec_arch_x86::lift::FnAbi::integer()
    })
}

/// Load, discover, lift and clean every function in `files`, with the naming
/// scheme every C gate shares.
///
/// This is `rdec c`'s front half, split out so `rdec t2` runs the *same*
/// pipeline on the *same* names — a differential gate that re-derived the
/// function list its own way would be measuring its own copy, not the corpus.
pub fn lift_all(files: &[String]) -> Result<Vec<Subject>, String> {
    // The `o<index>_` prefix is what makes one TU legal: the C corpus has 120
    // objects that all define `f`.
    lift_all_named(files, &|oi, raw| format!("o{oi}_{}", c_ident(raw)))
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

    let subjects = lift_all(&files)?;
    println!("{}", calls_line(&subjects));
    println!("{}", tail_line(&subjects));
    println!("{}", relocs_line(&subjects));
    // The callee shapes of the whole translation unit, built once, before the
    // first function is emitted: a call to a sibling needs a prototype that
    // already exists, and every definition is in this file, so the set is known.
    let protos = protos_of(&subjects);

    let mut seen = 0usize;
    let mut emitted = 0usize;
    let mut refused = 0usize;
    let mut pass_broken = 0usize;
    let mut keys: HashMap<String, usize> = HashMap::new();
    // The structuring census, one report per EMITTED function (a refused
    // function has no C, so it has no control flow to count).
    let mut reports: Vec<Report> = Vec::new();
    let mut first_refusal: Option<String> = None;
    let mut first_pass_break: Option<String> = None;
    let mut all = String::new();
    // The per-function text carries its own includes (so a single emitted
    // function is a compilable translation unit on its own); the concatenated
    // file needs them before the shared prelude, which uses uint64_t.
    all.push_str("#include <stdint.h>\n#include <string.h>\n\n");
    all.push_str(PRELUDE);
    all.push('\n');

    for s in &subjects {
        seen += 1;
        if let Err(errs) = verify_ssa(&s.f) {
            pass_broken += 1;
            // The post-pass IR of a broken function is the only way to see
            // WHICH op a pass emitted at the wrong width.
            if pass_broken <= 3 {
                let dump = format!("{out_dir}/passbroken_{pass_broken}.ir");
                let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&s.f));
            }
            if first_pass_break.is_none() {
                first_pass_break = Some(format!("{} :: {}: {}", s.obj, s.sym, errs.join("; ")));
            }
            continue;
        }
        match emit_c_sig(&s.f, &protos) {
            Ok(out) => {
                emitted += 1;
                reports.push(out.structuring.clone());
                // `RDEC_C_GOTOFLOW=1` prints the per-function readability
                // exchange: what the flat label + `goto` spelling of THIS
                // function costs, against what the printed C costs. The flat
                // form is the before-picture computed from the same IR, in the
                // same units, so a reader can see which functions structuring
                // bought and which it did not, by name.
                if std::env::var_os("RDEC_C_GOTOFLOW").is_some() {
                    let flat = rdec_core::structure::flat_walk(&s.f).report;
                    let st = &out.structuring;
                    println!(
                        "goto/label {} :: {}  goto {} -> {}  label {} -> {}  {}",
                        s.obj,
                        s.sym,
                        flat.gotos,
                        st.gotos,
                        flat.labels,
                        st.labels,
                        if (flat.gotos, flat.labels) == (st.gotos, st.labels) {
                            "unchanged"
                        } else {
                            "CHANGED"
                        }
                    );
                }
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
                    let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&s.f));
                }
                for r in &rs {
                    *keys.entry(key(r)).or_default() += 1;
                }
                if let Some(r) = rs.first() {
                    if first_refusal.is_none() {
                        first_refusal = Some(format!(
                            "{} :: {} @ {:#x}: {}",
                            s.obj, s.sym, r.at, r.reason
                        ));
                    }
                }
            }
        }
    }
    let path = format!("{out_dir}/all.c");
    std::fs::write(&path, &all).map_err(|e| format!("{path}: {e}"))?;

    println!("emit-c: {seen} lifted functions, {emitted} emitted C, {refused} refused, {pass_broken} invalid after passes");
    println!("{}", control_flow_line(&reports, emitted));
    println!("{}", readability_line(&all, emitted));
    if let Some(d) = census_drift(&reports, &all) {
        // The census is a claim about the emitted C. When the emitter's own count
        // and a scan of the text are not the same number, the census is broken,
        // and a broken census cannot certify anything else in this report.
        println!("FAIL: the structuring census disagrees with the emitted C: {d}");
        return Ok(1);
    }
    let mut ks: Vec<(String, usize)> = keys.into_iter().collect();
    ks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if !ks.is_empty() {
        // The census has to be auditable against the refusal count: every
        // refusal is in exactly one class, so the sum is the number of refusal
        // records, and a class list that stops at six still says so.
        let total: usize = ks.iter().map(|(_, n)| n).sum();
        println!(
            "refusal records: {total} in {refused} refused functions, {} classes",
            ks.len()
        );
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

/// The SSE census, on the same footing as the self-idiom census: counted by the
/// lifter at its own dispatch gate, so `sum(sse)` is the number of SSE
/// instructions the decoder saw, which is what `llvm-objdump -d` over the same
/// objects prints. Auditable, not trusted.
pub fn sse_line(sse: &std::collections::BTreeMap<String, u32>, zeroing: usize) -> String {
    if sse.is_empty() {
        return "  sse mnemonics counted: 0 (scalar subset modelled, packed refused)".to_string();
    }
    let mut parts: Vec<String> = sse.iter().map(|(k, v)| format!("{k}={v}")).collect();
    parts.sort();
    let tot: u32 = sse.values().sum();
    format!(
        "  sse mnemonics counted: {tot} [{}] (scalar subset modelled, packed refused by name; zeroing idiom: {zeroing})",
        parts.join(" ")
    )
}
