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
use rdec_arch_x86::lift::{
    lift_one_sigs, CalleeSig, CalleeTable, FnShape, RelocMap, RelocTarget, STACK_BASE,
};
use rdec_core::emit_c::{callee_decl, emit_c_sig, text_census, Protos, Refusal, PRELUDE};
use rdec_core::ir::{BinOp, Callee, Function, Op, VId};
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

/// The M6 readability exchange, printed beside the readability line: the gotos
/// that became `return` statements, the return blocks whose text every edge made
/// dead, and the jumps the flat layout absorbed by printing the target next.
///
/// These are the three ways this pass removes a `goto` without touching the CFG,
/// and each has a proof behind it (a pure tail with a readable value, every edge
/// spelling the return, an edge that lands on the next line of text). The line
/// exists so `123 goto statements -> 61` is auditable: the drop is attributed to a
/// mechanism per edge, not claimed as a total.
pub fn readability_exchange_line(reports: &[Report]) -> String {
    let rep = sum_reports(reports);
    format!(
        "readability exchange: {} goto sites printed as return | {} return blocks absorbed          (text and label gone) | {} jumps absorbed by the flat layout",
        rep.early_returns, rep.absorbed, rep.fallthroughs
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
///
/// A thin alias for `rdec_core::passes::clean_isolated` (the one pass table),
/// and it RETURNS the isolation report: SSA verified after every pass, with the
/// FIRST breaking pass named (index, name, verifier errors). `cmd_c` prints a
/// `pass-broke-ssa record:` line per broken function and a `pass isolation:`
/// summary; a pass that breaks SSA is a bug in that pass, and a pipeline that
/// verifies only at the end cannot name its own bug.
pub(crate) fn clean(f: &mut Function) -> rdec_core::passes::Isolation {
    rdec_core::passes::clean_isolated(f)
}

/// Cap a census key WITHOUT inventing an identifier. The census key is the map
/// of what to build next, and a raw character slice produces labels like
/// `lea of an address we cannot name: __Imag` — measured, printed by this gate
/// on the MSVC leg, where the real target is `__ImageBase`. Two harms: the key
/// names a symbol that does not exist, and every distinct target whose first 40
/// characters agree lands in ONE bucket, so the count is a count of prefixes.
/// A clip that is announced with an ellipsis cannot do either silently.
pub fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let head: String = s.chars().take(n).collect();
    match head.rfind(char::is_whitespace) {
        Some(i) if i > n / 2 => format!("{}…", head[..i].trim_end()),
        _ => format!("{head}…"),
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
        return format!("opaque: {}", clip(rest[..cut].trim(), 64));
    }
    let low = s.to_lowercase();
    let cut = low.find(':').unwrap_or(low.len());
    clip(low[..cut].trim(), 72)
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
    /// The per-pass SSA isolation report for this subject's `clean` run: the
    /// set of passes verified after running, and the FIRST pass that broke SSA
    /// (the pipeline stops there, so `f` is exactly what that pass produced).
    /// `cmd_c` prints a `pass-broke-ssa record:` line for every broken subject
    /// and a `pass isolation:` summary whose count is the union of these
    /// `verified` sets.
    pub iso: rdec_core::passes::Isolation,
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
///
/// THE TWO-PASS. Pass 1 lifts every subject with the callee table as it is
/// (every callee at the default four pairs). For every symbol this object
/// DEFINES - one whose body pass 1 lifted - the body itself says how many
/// argument cells it reads (`Function::args_read`), and that number becomes the
/// callee's `nargs` in the table. Pass 2 re-lifts every function with the
/// enriched table, and the pass-2 subjects are what this returns: the census
/// counters, the emitted C, and the arity all come from the SAME pass.
/// Recursion is covered by the same rule: `f`'s own pass-1 body is the evidence
/// for `f`'s calls to itself (that is `fib_rec`). Callees the pipeline cannot
/// see - imports, relocations naming no symbol of this object, defined symbols
/// with no lifted body - keep `nargs = 4`, the loud default (see
/// `CalleeSig::nargs`). The `arity:` line printed at the end states the counts
/// of that enrichment; it is printed from the same counters that set the table.
pub fn lift_all_named_shapes(
    files: &[String],
    name_of: &dyn Fn(usize, &str) -> String,
    shape_of: &dyn Fn(usize, &str) -> FnShape,
) -> Result<Vec<Subject>, String> {
    let mut out: Vec<Subject> = Vec::new();
    // The arity census, summed over objects: N entries ENRICHED from a lifted
    // body, M defined-in-object callee symbols (the recovered denominator), K
    // table entries left at the default four pairs. The `arity:` line prints
    // exactly these, and the table the two-pass builds is built from the same
    // numbers - a printed count and its pass condition, from one source.
    let (mut arity_n, mut arity_m, mut arity_k) = (0usize, 0usize, 0usize);
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
                    // Start at the default four pairs; the two-pass below LOWERS
                    // this for a callee whose own lifted body reads fewer cells.
                    // Nothing here lowers it from a name, an ABI, or a guess.
                    nargs: 4,
                },
            );
        }

        // The object's relocations, keyed by the address of the displacement
        // field they name: the evidence a `call`'s immediate does not carry in
        // an object file, and the same table that names a RIP-relative data
        // address. Built per object, in the same per-object virtual space as
        // `names`, the plan, and `data_map_of`.
        let relocs = reloc_map_of(&obj, oi, &idx_of, &base_of);
        // THE TERMINATION FAMILY IS A FRONT-END DECLARATION, and in a linked object
        // the family states itself only as a NAME. `abort`, `exit`, `_exit`,
        // `__stack_chk_fail`, `__fastfail`, `ExitProcess` are `U` in `llvm-nm`:
        // imports, with no section, no size, no address, so they never reach
        // `names` and the loop above never declares them — which is precisely the
        // case M3's noreturn line was written for, because in a real binary the
        // termination family IS imported, and a block that falls through a
        // `call exit` sends CFG recovery into whatever code follows. So the
        // declaration is made here, from the name the relocation carries, for
        // IMPORTS ONLY (a defined symbol's entry above wins), and only for names
        // the front end declares: an import named `weird` gets nothing and keeps
        // the loud `Exit`. `tail_calls.rs` pins both directions — the table names
        // the family and nothing else, and the anti-invention test reddens if a
        // noreturn is inferred from IR shape instead of read from the name.
        for rt in relocs.values() {
            if rt.addr.is_some()
                || sigs.contains_key(rt.sym.as_str())
                || !rdec_arch_x86::lift::noreturn_symbol(rt.sym.as_str())
            {
                continue;
            }
            sigs.insert(
                rt.sym.clone(),
                CalleeSig {
                    cname: name_of(oi, rt.sym.as_str()),
                    noreturn: true,
                    // `typed: false`, and the default ABI: an import has no
                    // prototype in the object, so the call reads and clobbers the
                    // six argument registers and DEFINES NOTHING. Declaring
                    // noreturn is a statement about control, not about a return
                    // type the object never gave.
                    ..Default::default()
                },
            );
        }
        let relocs_parsed = obj.reloc_census().parsed;

        // THE TWO-PASS: pass 1 lifts every subject with the table as it is
        // (every callee at the default four pairs); the pass-1 bodies then
        // LOWER `nargs` for the callees this object defines; pass 2 re-lifts
        // with the enriched table and its subjects are what gets returned.
        // One closure run twice, not a copy of the loop: a second copy would
        // be a second pipeline, and the two passes would measure different
        // code.
        let lift_pass = |table: &CalleeTable| -> Vec<Subject> {
            let mut v: Vec<Subject> = Vec::new();
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
                    &name, rg.bytes, rg.base, start, end, &plan, &names, &abi, table, &relocs,
                    &tables,
                );
                let iso = clean(&mut f);
                v.push(Subject {
                    obj: file.clone(),
                    sym: raw,
                    cname: name,
                    start,
                    external: external.contains(&start),
                    relocs_parsed,
                    f,
                    abi,
                    rep,
                    iso,
                });
            }
            v
        };
        let pass1 = lift_pass(&sigs);

        // THE ARITY FACT, from the pass-1 body. A callee's lifted body names
        // the argument cells it reads (`Function::args_read`: max index of a
        // READ parameter plus one, terminator reads included - the soundness
        // direction is that over-estimating keeps today's loud refusal and
        // under-estimating would be the miscompile, so this is never a count
        // of leading reads). Only a symbol THIS OBJECT DEFINES qualifies:
        // `names` is the object's symbol table for code it defines (an import
        // has no address and never enters it, and a synthetic `fn_<addr>`
        // start is not a table name), so the anti-invention floor - a callee
        // with no body in this object keeps the default four pairs - is the
        // shape of the set, not a filter on it. `Subject::external` is NOT
        // this filter: in COFF, EXTERNAL storage on a code symbol with a
        // section means DEFINED-and-linkable (the `f` of `fib_rec`), and it is
        // the body that is the evidence, not the link name.
        let defined: HashSet<&str> = names.values().map(|n| n.as_str()).collect();
        let mut arity_of: HashMap<String, usize> = HashMap::new();
        for s in &pass1 {
            if !defined.contains(s.sym.as_str()) {
                continue;
            }
            if let Some(sg) = sigs.get_mut(&s.sym) {
                sg.nargs = s.f.args_read();
                arity_of.insert(s.sym.clone(), sg.nargs);
            }
        }
        arity_n += arity_of.len();
        arity_m += defined.len();
        arity_k += sigs.len() - arity_of.len();

        let pass2 = lift_pass(&sigs);
        // The definition spells the SAME count the call sites pass: min of the
        // recovered nargs and the pairs the callee's own shape seeds. One
        // number per function, in the Function, read by `c_param_tys`,
        // `callee_decl`, the emitter's entry-params loop, and the t2 drivers.
        for s in pass2 {
            let mut s = s;
            if let Some(&n) = arity_of.get(&s.sym) {
                s.f.nparams = Some(n.min(s.abi.seeded_pairs().len()));
            }
            out.push(s);
        }
    }
    println!(
        "arity: {arity_n} of {arity_m} in-object callees recovered from the callee's own body, \
         {arity_k} left at the 4-pair default"
    );
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

/// Width of the frame window inspected by [`frame_touched`]: the slots below
/// the model's entry RSP (`STACK_BASE = 1 << 19`) that an `-O0` frame uses.
/// The window matches the model's own geometry - a rsp-relative frame at
/// entry RSP = STACK_BASE lives strictly below STACK_BASE, and the corpus's
/// deepest frame is far under 4 KiB.
pub const FRAME_WINDOW: u64 = 0x1000;

/// Resolve an operand to a constant address. The lifter materialises frame
/// addresses as plain constants (`%9 = 0x7fffc ; store %9 = ...`), so `Const`
/// is the corpus shape; `Copy` and `add`/`sub` over constants cover the
/// shapes the passes leave behind (bounded recursion, depth 8). Anything
/// else - a `Param`, a computed address - is `None`: the check claims only
/// addresses it can resolve, and a caller-buffer access (Param-rooted, the
/// shape `t2`'s memory comparison covers) is by construction outside it.
fn const_addr(f: &Function, v: VId, depth: u8) -> Option<u64> {
    if depth > 8 {
        return None;
    }
    let ins = f
        .blocks
        .iter()
        .find_map(|b| b.instrs.iter().find(|i| i.dest == v))?;
    match &ins.op {
        Op::Const(c) => Some(*c),
        Op::Copy(x) => const_addr(f, *x, depth + 1),
        Op::Bin(BinOp::Add, a, b) => {
            Some(const_addr(f, *a, depth + 1)?.wrapping_add(const_addr(f, *b, depth + 1)?))
        }
        Op::Bin(BinOp::Sub, a, b) => {
            Some(const_addr(f, *a, depth + 1)?.wrapping_sub(const_addr(f, *b, depth + 1)?))
        }
        _ => None,
    }
}

/// Is `x` a frame-window address (the slots below the model's entry RSP)?
fn in_window(x: u64) -> bool {
    (STACK_BASE - FRAME_WINDOW..STACK_BASE).contains(&x)
}

/// Is the operand tree rooted at `v` FRAME-ROOTED: does it contain a constant
/// inside the frame window, anywhere in the tree? This is the question the
/// frame check has to ask of an address it CANNOT resolve, and it is deliberately
/// wider than [`const_addr`]: it walks every `Bin` arm (`Mul`, `Shl`, `Add`, ...),
/// `Copy`, `Cast`, `Phi` and `Select`, so the indexed-local shape clang spells
/// as `mov %edx, (%rsp,%rcx,4)` ->
/// `add 0x7fffc, mul(%rcx, 4)` IS frame-rooted, while the caller-buffer shape
/// `add %param, 0x2008` is NOT: a `Param`-rooted tree carries no in-window
/// constant, exactly as `const_addr`'s doc-comment promises about it. The
/// depth bound is the same 8 as `const_addr`'s: beyond it the answer is "not
/// frame-rooted", which is the CONSERVATIVE direction for over-fire only (a
/// tree that deep is a refusal of nothing today and no corpus shape reaches it).
fn frame_rooted(f: &Function, v: VId, depth: u8) -> bool {
    if depth > 8 {
        return false;
    }
    let Some(ins) = f
        .blocks
        .iter()
        .find_map(|b| b.instrs.iter().find(|i| i.dest == v))
    else {
        return false;
    };
    match &ins.op {
        Op::Const(c) => in_window(*c),
        Op::Copy(x) | Op::Cast(_, _, x) => frame_rooted(f, *x, depth + 1),
        Op::Bin(_, a, b) => frame_rooted(f, *a, depth + 1) || frame_rooted(f, *b, depth + 1),
        Op::Select(_, t, e) => frame_rooted(f, *t, depth + 1) || frame_rooted(f, *e, depth + 1),
        Op::Phi(pairs) => pairs.iter().any(|(x, _)| frame_rooted(f, *x, depth + 1)),
        _ => false,
    }
}

/// One function's frame traffic, split by whether the model can NAME the
/// addresses it touches.
pub struct FrameTouch {
    /// The constant-resolved in-window addresses it loads or stores, sorted
    /// and deduped (the refusal names the LOWEST, so the text is deterministic).
    pub slots: Vec<u64>,
    /// At least one `Load`/`Store` has a FRAME-ROOTED address that is NOT
    /// constant-resolvable - an indexed local, `add 0x7fffc, mul(%rcx, 4)`.
    /// Such an access can reach EVERY address in the window (index 0 hits the
    /// base itself), so the function's frame traffic is UNBOUNDED and the
    /// check may not reason about it as a slot set. `Param`-rooted traffic (the
    /// caller buffer) never sets this: it is not the function's frame.
    pub unbounded: bool,
}

/// The frame traffic of `f`: which frame slots it provably touches, and
/// whether it also has frame traffic the model cannot bound. See
/// [`frame_rooted`] for what counts as the function's own frame and
/// [`frame_alias_map`] for what the check does with the answer.
pub fn frame_touch(f: &Function) -> FrameTouch {
    let mut v: Vec<u64> = Vec::new();
    let mut unbounded = false;
    for b in &f.blocks {
        for ins in &b.instrs {
            let addr = match &ins.op {
                Op::Load(_, a) | Op::Store(_, a, _) => *a,
                _ => continue,
            };
            match const_addr(f, addr, 0) {
                Some(x) if in_window(x) => v.push(x),
                // Unresolved (or resolved outside the window): frame-rooted
                // traffic the model cannot name is the unbounded case.
                _ => unbounded |= frame_rooted(f, addr, 0),
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    FrameTouch {
        slots: v,
        unbounded,
    }
}

/// The model addresses this function touches inside the frame window
/// `[STACK_BASE - FRAME_WINDOW, STACK_BASE)`: the frame slots of its
/// activation, every `Load`/`Store` whose address resolves to a constant in
/// the window. Sorted and deduped on purpose: the shared address a refusal
/// names is the LOWEST one, so the refusal text is deterministic for one
/// input (the pipeline's determinism contract covers the census text too).
/// A thin wrapper over [`frame_touch`]: the slot set only, so a caller that
/// cannot answer "what about the traffic you cannot name?" keeps compiling
/// and stays honest about what it does not ask.
pub fn frame_touched(f: &Function) -> Vec<u64> {
    frame_touch(f).slots
}

/// Which side of a call cannot bound its own frame traffic. It decides both
/// the shared set and the sentence, because the sentence has to say WHICH
/// side the model is blind about.
enum FrameSide {
    /// Both sides are constant-bounded: the literal slot intersection.
    Slots,
    /// The callee's frame traffic is frame-rooted and unresolvable.
    Callee,
    /// The caller's is.
    Caller,
    /// Both are.
    Both,
}

/// THE FRAME-ACTIVATION CHECK. The model gives every function ONE frame:
/// entry RSP is the `STACK_BASE` constant, so a caller's frame slots and a
/// callee's frame slots are the SAME addresses in the model's flat memory.
/// The machine gives every ACTIVATION a frame - the model has one per
/// function, the machine has one per activation. Witness (measured,
/// `corpus/c/all.c` at `arity-wip` def1647, reproducible outside the
/// harness): `o8_f` (`call_helper.O0::f`) spills to `0x7fff8`/`0x7fffc`,
/// its callee `o8_h` spills to `0x7fffc`/`0x7fff8` - the same addresses -
/// and the -O0 leg answers `3*(a*a+1)` where the machine answers
/// `3*(a*a+b)`: h's frame clobbered f's slot for b. Recursion is the same
/// defect through a self-call: every activation past the first shares one
/// frame.
///
/// The rule, per subject `f` that touches at least one frame slot: for
/// every `Op::Call { callee: Callee::Symbol(sym), .. }` naming a body this
/// pipeline lifted, if `sym` is `f`'s own name (recursion) or `f`'s
/// touched set intersects the callee's touched set, `f` is REFUSED. A
/// callee with no lifted body - an import, a relocation naming no symbol -
/// is NOT refused: nothing is known about its frame, and over-refusing on
/// imports would punish every function that calls `memcpy`. The refusal is
/// the whole slice: a frame model (an activation-keyed frame, the frame as
/// an implicit parameter) is the named next step, not this change.
///
/// One reach refinement, MEASURED at corpus level (2026-11-08): the literal
/// set-intersection above also refused `tailcall_local.O0::f`, and t2
/// MATCHED that function at `arity-wip` def1647 - it is one of the base
/// 2452 comparisons the comparison set must keep. Its IR (dumped from this
/// very gate) stores `b`/`a` to `0x7fffc`/`0x7fff8`, RELOADS BOTH BEFORE the
/// call, calls `helper` at 0x34 and returns the call's value: the caller
/// reads no shared slot after the call, so the callee's clobber is
/// unobservable, and the refusal would cost what the defect's reach does
/// not cover (the PE-slice rule: a refusal may only cost what its defect's
/// reach covers, measured, not caution). So the rule is liveness-aware: a
/// call site refuses its caller only if some LOAD of a shared address
/// executes CONTROL-WISE AFTER the call - in the same block, at a later
/// instruction, or in any block forward-reachable from the call's
/// successors. That is exactly "the callee's activation overwrites a cell
/// the caller is live on across the call", which is the defect. A call
/// whose value is a `TailReturn` terminator (a modelled `jmp callee`:
/// tailcall_local O1/O2/Oz, where clang really does tail-call) is skipped:
/// the caller's activation dies at the jump, so nothing can be live across
/// it. The `tail_call_leg` / `pre_load_caller` fixtures and the corpus
/// census pin both directions.
///
/// Keyed by `Subject::cname`, which embeds the object index (`o8_f`), so a
/// callee name resolves within its own object by construction - the same
/// global map `protos_of` uses, and `Op::Call` carries that same name.
///
/// UNBOUNDED FRAME TRAFFIC (2026-11-08, the CI red this commit fixes). The
/// slot set above is an UNDER-APPROXIMATION: `const_addr` resolves a `Const`,
/// a `Copy`, or `add`/`sub` over constants, so a frame access clang spells
/// with an index register - `mov %edx, (%rsp,%rcx,4)` ->
/// `add 0x7fffc, mul(%rcx, 4)` - resolves to nothing and the function records
/// NO frame traffic. Measured: `c --corpus` on this commit printed `156
/// lifted, 135 emitted, 21 refused, 0 invalid` on this box's clang 21.1.8 and
/// `156 lifted, 136 emitted, 20 refused, 1 invalid` on the CI runner's clang
/// 20.1.8 - same lifted, same emitted-for-the-same-reasons, and one frame
/// refusal that did not fire because the object that triggers it was built by
/// a different codegen. The `invalid` was the decompiled C computing the wrong
/// answer: the callee's activation overwrote a slot the caller reads back. A
/// refusal keyed on a compiler's SPELLING is not a refusal, and `invalid`
/// catching it is the pipeline being honest, not the fix.
///
/// The rule, in the sound direction (over-approximating here REFUSES, which is
/// loud, counted and named; under-approximating miscompiles in silence):
/// [`frame_touch`] reports the resolvable slots AND an `unbounded` flag for
/// frame-rooted traffic the model cannot name. A callee with `unbounded` may
/// write ANY window address, so it shares the caller's WHOLE known frame; a
/// caller with `unbounded` has a live set covering the whole window, so it
/// shares it with any callee that touches the frame at all; both bounded is
/// the literal intersection as before. Liveness is NOT relaxed by the
/// wildcard - the refusal still requires a shared frame LOAD executing after
/// the call, which is what keeps `tailcall_local.O0::f` emitting. Imports stay
/// exempt (nothing is known about their frame, and over-refusing on imports
/// punishes every function that calls `memcpy`), and `Terminator::TailReturn`
/// calls stay exempt (the activation dies at the jump).
pub fn frame_alias_map(subjects: &[Subject]) -> HashMap<String, Refusal> {
    let touch: HashMap<&str, FrameTouch> = subjects
        .iter()
        .map(|s| (s.cname.as_str(), frame_touch(&s.f)))
        .collect();
    let mut out: HashMap<String, Refusal> = HashMap::new();
    for s in subjects {
        let mine = &touch[s.cname.as_str()];
        // No frame traffic at all: nothing of this activation's can be aliased
        // (the call_helper O1/O2/Oz legs - calls, no frame - keep emitting).
        if mine.slots.is_empty() && !mine.unbounded {
            continue;
        }
        // Frame-address LOADS with their control position. `Some(x)` is a
        // resolved slot; `None` is a frame-rooted load the model cannot name -
        // a candidate shared load only when the CALLER's traffic is unbounded
        // (then its live set is the whole window), never a licence to skip
        // liveness: see `live_load_after`.
        let mut loads: Vec<(usize, usize, Option<u64>)> = Vec::new();
        for (bi, b) in s.f.blocks.iter().enumerate() {
            for (ii, ins) in b.instrs.iter().enumerate() {
                if let Op::Load(_, a) = &ins.op {
                    match const_addr(&s.f, *a, 0) {
                        Some(x) if in_window(x) => loads.push((bi, ii, Some(x))),
                        _ => {
                            if frame_rooted(&s.f, *a, 0) {
                                loads.push((bi, ii, None));
                            }
                        }
                    }
                }
            }
        }
        let succ: Vec<Vec<rdec_core::ir::BbId>> =
            s.f.blocks
                .iter()
                .map(|b| {
                    let mut v = Vec::new();
                    b.term.successors(&mut v);
                    v
                })
                .collect();
        'calls: for (bi, b) in s.f.blocks.iter().enumerate() {
            // The call whose value the block's TAIL RETURN returns: the
            // caller's activation dies at the machine's `jmp callee`, so
            // clobbering its slots is unobservable - see the reach note.
            let tail = match &b.term {
                rdec_core::ir::Terminator::TailReturn(v) => *v,
                _ => u32::MAX,
            };
            for (ii, ins) in b.instrs.iter().enumerate() {
                let Op::Call {
                    callee: Callee::Symbol(sym),
                    ..
                } = &ins.op
                else {
                    continue;
                };
                if ins.dest == tail {
                    continue;
                }
                // Shared slots: self-recursion shares EVERY slot the
                // function touches (one frame, two activations); an ordinary
                // callee shares the intersection, WIDENED to a whole-window
                // wildcard on whichever side cannot bound its own frame
                // traffic; an import (no lifted body) shares NOTHING - the
                // model knows no frame for it, and over-refusing on imports
                // would punish every function that calls `memcpy` (the
                // anti-over-fire case).
                let self_call = sym == &s.cname;
                match touch.get(sym.as_str()) {
                    Some(theirs) => {
                        let (shared, wildcard, side) = match (mine.unbounded, theirs.unbounded) {
                            (false, false) => {
                                let inter: Vec<u64> = mine
                                    .slots
                                    .iter()
                                    .filter(|a| theirs.slots.contains(a))
                                    .copied()
                                    .collect();
                                if inter.is_empty() {
                                    continue;
                                }
                                (inter, false, FrameSide::Slots)
                            }
                            // An unbounded callee may write ANY window address,
                            // so it shares the caller's whole known frame -
                            // non-empty by construction: a caller with no slots
                            // here has no live set for the callee to clobber.
                            (false, true) => {
                                if mine.slots.is_empty() {
                                    continue;
                                }
                                (mine.slots.clone(), false, FrameSide::Callee)
                            }
                            // An unbounded caller's live set IS the whole
                            // window, so it shares it with any callee that
                            // touches the frame at all.
                            (true, false) => {
                                if theirs.slots.is_empty() {
                                    continue;
                                }
                                (theirs.slots.clone(), true, FrameSide::Caller)
                            }
                            (true, true) => (Vec::new(), true, FrameSide::Both),
                        };
                        if !live_load_after(&loads, &succ, bi, ii, &shared, wildcard) {
                            continue;
                        }
                        let at = shared
                            .first()
                            .map(|a| format!("{a:#x}"))
                            .unwrap_or_else(|| {
                                "the whole frame window, unbounded (indexed access)".to_string()
                            });
                        // A self-call is in this arm too (the subject's own
                        // body is keyed by its own cname) - it gets the
                        // recursion text, which names the self-call.
                        let reason = if self_call {
                            format!(
                                "frame slots alias the callee's: {cname} calls itself and both \
                                 frames use {at} - the model has one frame per function, the \
                                 machine has one per activation (recursion: activation 2 \
                                 overwrites activation 1's slots)",
                                cname = s.cname,
                            )
                        } else {
                            match side {
                                FrameSide::Slots => format!(
                                    "frame slots alias the callee's: {} and {} both use {at} - \
                                     the model has one frame per function, the machine has one \
                                     per activation",
                                    s.cname, sym,
                                ),
                                FrameSide::Callee => format!(
                                    "frame slots alias the callee's: {} and {} share the frame: \
                                     the callee's frame traffic is unbounded - not \
                                     constant-bounded (indexed access) - so it may write any slot \
                                     the caller lives across (its lowest is {at}) - the model has \
                                     one frame per function, the machine has one per activation",
                                    s.cname, sym,
                                ),
                                FrameSide::Caller => format!(
                                    "frame slots alias the callee's: {} and {} share the frame: \
                                     the caller's frame traffic is unbounded - not \
                                     constant-bounded (indexed access) - so its live set covers \
                                     the whole window and the callee writes {at} in it - the \
                                     model has one frame per function, the machine has one per \
                                     activation",
                                    s.cname, sym,
                                ),
                                FrameSide::Both => format!(
                                    "frame slots alias the callee's: {} and {} share the frame: \
                                     both sides' frame traffic is unbounded - not \
                                     constant-bounded (indexed access) - so both cover the whole \
                                     window - the model has one frame per function, the machine \
                                     has one per activation",
                                    s.cname, sym,
                                ),
                            }
                        };
                        out.entry(s.cname.clone()).or_insert(Refusal {
                            at: ins.addr,
                            reason,
                        });
                        break 'calls;
                    }
                    None => {
                        if !self_call {
                            continue;
                        }
                        // Recursion: the second activation's frame IS the
                        // first one. Same liveness bound: the recursive
                        // call's spills overwrite the slots the caller reads
                        // back on the way (the fib_rec shape), and the same
                        // wildcard when the function's own traffic is indexed.
                        if !live_load_after(&loads, &succ, bi, ii, &mine.slots, mine.unbounded) {
                            continue;
                        }
                        out.entry(s.cname.clone()).or_insert(Refusal {
                            at: ins.addr,
                            reason: format!(
                                "frame slots alias the callee's: {cname} calls itself and both \
                                 frames use {at:#x} - the model has one frame per function, the \
                                 machine has one per activation (recursion: activation 2 \
                                 overwrites activation 1's slots)",
                                cname = s.cname,
                                at = mine.slots[0],
                            ),
                        });
                        break 'calls;
                    }
                }
            }
        }
    }
    out
}

/// Does a LOAD of any address in `shared` execute CONTROL-WISE AFTER the
/// call at (block `bi`, instruction `ii`)? Same block at a later
/// instruction, or any block forward-reachable from the call's successors.
/// Forward reachability over-approximates "executes after" by the block
/// (a load in a reachable block is a candidate; loops make it reach more,
/// which is the sound direction for a refusal), and the within-block bound
/// is what keeps `tailcall_local.O0::f` - all its loads before the call,
/// no successor blocks - out of the refusal.
///
/// `wildcard` is the UNBOUNDED CALLER: its live set covers the whole window,
/// so a frame-rooted load the model cannot name (`None`) is a shared load.
/// It widens WHICH load counts, never WHETHER liveness is required: a
/// function that reloads its frame cells before the call and then tail-jumps
/// keeps emitting with `wildcard` set, which is what the `pre_load_caller`
/// and `tail_call_leg` fixtures pin against the sabotage.
fn live_load_after(
    loads: &[(usize, usize, Option<u64>)],
    succ: &[Vec<rdec_core::ir::BbId>],
    bi: usize,
    ii: usize,
    shared: &[u64],
    wildcard: bool,
) -> bool {
    let mut reach: Vec<bool> = vec![false; succ.len()];
    let mut stack: Vec<rdec_core::ir::BbId> = succ.get(bi).cloned().unwrap_or_default();
    while let Some(b) = stack.pop() {
        let b = b as usize;
        if b < reach.len() && !reach[b] {
            reach[b] = true;
            stack.extend(succ[b].iter().copied());
        }
    }
    loads.iter().any(|(lb, li, a)| {
        // Every entry is a frame-rooted load (a resolved window slot, or an
        // address rooted in one that the model cannot name). With `wildcard`
        // the caller's live set IS the window, so any of them is shared;
        // without it, only the addresses on the shared list are.
        let is_shared = wildcard || matches!(a, Some(x) if shared.contains(x));
        is_shared
            && ((lb == &bi && li > &ii) || (lb != &bi && reach.get(*lb).copied().unwrap_or(false)))
    })
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
/// For the corpus's whole first life all four read 0, and that was the
/// measurement, not an assumption: `llvm-objdump -d -r` over the objects found
/// 65 `jmp` sites, none carrying a relocation naming a defined function, and 31
/// call sites, none to the termination family. The model was pinned by
/// `crates/arch-x86/tests/tail_calls.rs`, whose fixtures quote their own
/// `llvm-objdump -d -r`. `tailcall_local`, `tailcall_external` and
/// `noreturn_call` are the corpus witnesses that ended that era (2026-11-08),
/// and `cmd_c` now FAILS if the counts return to 0 — an unexercised feature is
/// an unmeasured one. The C++ corpus is still all-zero on this line: its
/// functions are leaf arithmetic, and the demangler's noreturn knowledge has no
/// corpus site to land on either.
///
/// The four tail-call/noreturn counters, in one place, so the census LINE and
/// the gate's floor assertion cannot disagree about what they measured. A gate
/// that re-derives a number it also prints is a gate that can print 3 and fail
/// on 2.
pub fn tail_counts(subjects: &[Subject]) -> (usize, usize, usize, usize) {
    let (mut t, mut un, mut nr, mut past) = (0usize, 0usize, 0usize, 0usize);
    for s in subjects {
        t += s.rep.tail_calls;
        un += s.rep.tail_calls_unnamed;
        nr += s.rep.noreturn_calls;
        past += s.rep.bytes_past_ended_block;
    }
    (t, un, nr, past)
}

pub fn tail_line(subjects: &[Subject]) -> String {
    let (t, un, nr, past) = tail_counts(subjects);
    let mut sites: Vec<(u64, String)> = Vec::new();
    for s in subjects {
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

/// The frame-traffic census, on the same footing as `calls_line`: how many
/// lifted functions carry frame traffic the model can NAME (constant slots),
/// how many carry frame-rooted traffic it CANNOT name (indexed access), and
/// how many touch no frame at all. `c --corpus` prints it, because the middle
/// number is the quantity the frame-activation refusal is calibrated on: it is
/// the count that MOVES with the compiler that built the objects (this box's
/// clang 21.1.8 and the CI runner's 20.1.8 build the same sources into
/// different spellings), and a refusal keyed on a spelling is only catchable
/// where the spelling is visible. `frame_touched` is the bounded arm's question
/// and `frame_touch`'s flag the wildcard arm's, so both are computed here.
pub fn frame_traffic_line(subjects: &[Subject]) -> String {
    let (mut named, mut unbounded, mut none) = (0usize, 0usize, 0usize);
    let mut who: Vec<&str> = Vec::new();
    for s in subjects {
        let slots = frame_touched(&s.f);
        let t = frame_touch(&s.f);
        if t.unbounded {
            unbounded += 1;
            who.push(&s.cname);
        } else if !slots.is_empty() {
            named += 1;
        } else {
            none += 1;
        }
    }
    // The record dump names WHICH functions: the middle number is a census, and
    // on the machine whose clang spells a frame access with an index register it
    // is the list that tells you which leg the refusal has to stop.
    let names = if std::env::var_os("RDEC_C_CENSUS_ALL").is_some() {
        format!(" ({}: {})", who.len(), who.join(", "))
    } else {
        String::new()
    };
    format!(
        "frame traffic: {named} functions on named slots | {unbounded} with unbounded (indexed) \
         frame traffic{names} | {none} with no frame traffic"
    )
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
    let mut skip_value = false;
    for a in args.iter().skip(1) {
        if skip_value {
            skip_value = false;
            continue;
        }
        if a == "--corpus" {
            // The clang corpus, read EXACTLY as before this leg existed: the MSVC
            // leg is additive, and `rdec c --corpus` (a repo floor: 98 emitted /
            // 42 refused / 0 invalid) must not feel it. This block was dropped
            // once while the `--corpus-dir` plumbing went in, and `c --corpus`
            // answered with the usage line and exit 2 - caught by running the
            // clang floors after the change, not before it.
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
        if a == "--emit" || a == "--corpus-dir" {
            // The value belongs to the flag. Reading `--emit DIR`'s directory as
            // an object path is how `c --corpus --emit DIR` died loading "DIR".
            skip_value = true;
            continue;
        }
        if !a.starts_with("--") {
            files.push(a.clone());
        }
    }
    // `--corpus-dir DIR` (see `crate::m1::corpus_objects`) points this gate at
    // another generator's manifest — the MSVC leg — through the SAME pipeline,
    // the same refusal classes and the same clang -Werror compile of the C.
    files.extend(crate::m1::corpus_objects(args)?);
    if files.is_empty() {
        return Err(
            "usage: rdec c <object.obj> [...] [--corpus] [--corpus-dir DIR] [--emit DIR]"
                .to_string(),
        );
    }
    let out_dir = args
        .iter()
        .position(|a| a == "--emit")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "corpus/c".to_string());
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let subjects = lift_all(&files)?;
    // The frame-activation refusals, computed ONCE over the whole subject
    // set (a callee's body has to be lifted before its caller can be judged)
    // and consumed in the loop below, so the refused count, the class census
    // and the record dump are the gate's own counters, not a second tally.
    let frame_alias = frame_alias_map(&subjects);
    println!("{}", calls_line(&subjects));
    println!("{}", tail_line(&subjects));
    println!("{}", frame_traffic_line(&subjects));
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
    // Read once, at the top: the record dump inside the loop and the class
    // list after it are the same auditable census, on one switch.
    let all_classes = std::env::var_os("RDEC_C_CENSUS_ALL").is_some();
    // The structuring census, one report per EMITTED function (a refused
    // function has no C, so it has no control flow to count).
    let mut reports: Vec<Report> = Vec::new();
    let mut first_refusal: Option<String> = None;
    let mut first_pass_break: Option<String> = None;
    // Union of the per-pass verification bits every subject's `clean` run
    // proved (see `Subject::iso`). The `pass isolation:` summary prints its
    // popcount, and a pinning test asserts that count equals the pipeline's
    // own table length - a driver that iterated an empty table prints 0 and
    // the test reddens, so the summary line cannot pass vacuously.
    let mut iso_bits: u32 = 0;
    let mut all = String::new();
    // The per-function text carries its own includes (so a single emitted
    // function is a compilable translation unit on its own); the concatenated
    // file needs them before the shared prelude, which uses uint64_t.
    all.push_str("#include <stdint.h>\n#include <string.h>\n\n");
    all.push_str(PRELUDE);
    all.push('\n');

    for s in &subjects {
        seen += 1;
        iso_bits |= s.iso.verified;
        // Name the culprit, not just the victim: when a pass leaves the
        // function with invalid SSA, say WHICH pass (index + name from the
        // pipeline's own table). The `invalid record:` below names the
        // function and reproduces these same errors, because the pipeline
        // STOPS at the breaking pass and the IR is exactly what it produced.
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
            pass_broken += 1;
            // The post-pass IR of a broken function is the only way to see
            // WHICH op a pass emitted at the wrong width.
            if pass_broken <= 3 {
                let dump = format!("{out_dir}/passbroken_{pass_broken}.ir");
                let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&s.f));
            }
            // UNCONDITIONAL, unlike the refusal records: the `invalid` bucket is
            // expected to be EMPTY, so every member is a bug, and a count with no
            // name is not a diagnosis. Measured 2026-11-08: CI printed `1 invalid`
            // on two successive commits and named nothing - the name exists here
            // (the `first_pass_break` string) but is printed only on the FAIL line
            // at the end, which the corpus floor tests capture and discard. Two
            // fixes were written to that count blind as a result.
            println!(
                "invalid record: {} :: {}: {}",
                s.obj,
                s.sym,
                errs.join("; ")
            );
            if first_pass_break.is_none() {
                first_pass_break = Some(format!("{} :: {}: {}", s.obj, s.sym, errs.join("; ")));
            }
            continue;
        }
        // The frame-activation refusal (see `frame_alias_map`): a function
        // whose frame slots alias its callee's - or alias its own recursion -
        // emits C that computes the WRONG answer (one frame per function in
        // the model, one per activation on the machine), so it must not reach
        // the emitted set, and `rdec t2` (same rule, same map) keeps it out
        // of the comparison. Refused, with the caller, the callee and a
        // shared address named - never emitted, never silent.
        if let Some(r) = frame_alias.get(&s.cname) {
            refused += 1;
            // Same record dump as an emitter refusal: the post-pass IR is the
            // diagnosis of the aliasing, slot by slot.
            if refused <= 3 {
                let dump = format!("{out_dir}/refused_{refused}.ir");
                let _ = std::fs::write(&dump, rdec_core::fmt_ir::format_function(&s.f));
            }
            *keys.entry(key(r)).or_default() += 1;
            if all_classes {
                println!(
                    "refusal record: {} :: {} @ {:#x}: {}",
                    s.obj, s.sym, r.at, r.reason
                );
            }
            if first_refusal.is_none() {
                first_refusal = Some(format!(
                    "{} :: {} @ {:#x}: {}",
                    s.obj, s.sym, r.at, r.reason
                ));
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
                    // The two "before" columns are the M5 numbers: the flat form's
                    // gotos BEFORE the early return and the layout took any of them
                    // (`gotos + early + fall`), its labels before an absorbed block took
                    // one (`labels + absorbed`). The "after" columns are the printed C.
                    // The `early/absorbed/fall` triple is printed form's, so a reader can
                    // attribute the drop of a function that the tree also structured.
                    let _ = &st;
                    println!(
                        "goto/label {} :: {}  goto {} -> {}  label {} -> {}  early {} absorbed {} \
                     fall {}  {}",
                        s.obj,
                        s.sym,
                        flat.gotos + flat.early_returns + flat.fallthroughs,
                        st.gotos,
                        flat.labels + flat.absorbed,
                        st.labels,
                        st.early_returns,
                        st.absorbed,
                        st.fallthroughs,
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
                    // `RDEC_C_CENSUS_ALL=1` doubles as the record dump: the
                    // census tells you a class grew, the record tells you
                    // WHICH object grew it, and a class name without an owner
                    // is a census you cannot audit.
                    if all_classes {
                        println!(
                            "refusal record: {} :: {} @ {:#x}: {}",
                            s.obj, s.sym, r.at, r.reason
                        );
                    }
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
    // The per-pass verification summary, the silent-path half of the isolation
    // tool: when no pass breaks SSA this is the ONLY extra line (records print
    // per broken function above). The count is the number of DISTINCT passes
    // that ran AND were verified after running, from the driver's own
    // iteration, and `crates/cli/tests/pass_isolation.rs` pins it against the
    // pipeline's table length - the loop-that-iterates-nothing failure mode
    // prints 0 and reddens that test.
    println!(
        "pass isolation: {} passes verified on {emitted} emitted functions",
        iso_bits.count_ones()
    );
    println!("{}", control_flow_line(&reports, emitted));
    println!("{}", readability_line(&all, emitted));
    println!("{}", readability_exchange_line(&reports));
    // The epilogue lane's own tally, printed so the win is measured and not
    // assumed: how many pure return tails moved onto their edges (the mechanism
    // behind the `goto -> return` conversions in the line above), how many were
    // refused and for what, and the two numbers this lane promises are zeros:
    // a frame store is never deleted, and the teardown writes DCE removed.
    println!("{}", rdec_core::passes::epilogue_line());
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
        // `RDEC_C_CENSUS_ALL=1` prints every class, not the top six: the M8
        // packed-SIMD slice needs the full census verbatim (each remaining
        // mnemonic names the next lane's work item), and a truncation that
        // hides `paddd` hides the map.
        let all_classes = std::env::var_os("RDEC_C_CENSUS_ALL").is_some();
        let top: Vec<String> = ks
            .iter()
            .take(if all_classes { ks.len() } else { 6 })
            .map(|(k, n)| format!("{n} {k}"))
            .collect();
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
    // THE EXERCISE FLOOR. A feature the corpus never reaches is a feature no
    // gate measures, and an unmeasured feature rots: the tail-call and noreturn
    // model shipped pinned by hand-built bytes, with `tail calls: 0 | noreturn
    // calls: 0` on every floor since. This floor makes that state fail. The
    // numbers come from `tail_counts`, the same function that produced the
    // census line printed above, so the verdict and the print cannot disagree.
    let (tail, tail_unnamed, noreturn, _past) = tail_counts(&subjects);
    if tail + tail_unnamed == 0 {
        println!(
            "FAIL: exercise floor - the corpus contains {tail} modelled tail calls and {tail_unnamed} tail calls with no callable callee, so nothing in the corpus exercises the tail-call model."
        );
        println!("      A corpus snippet must really sibling-call: `return callee(a, b);` where the definition is visible and the callee is `__attribute__((noinline))`, which clang compiles to `jmp callee` with a relocation naming a DEFINED symbol, and one external target (`jmp` to a symbol the object does not define) for the refusal bucket.");
        return Ok(1);
    }
    if noreturn == 0 {
        println!(
            "FAIL: exercise floor - the corpus contains {noreturn} noreturn calls, so nothing in the corpus exercises the noreturn CFG rule (a call whose callee is declared noreturn ends the block; without it, CFG recovery runs off the end of one function into the next)."
        );
        println!("      A corpus snippet must really call one: `if (x > 100) abort();` keeps the abort leg off the vectors the differential drives and puts the `call abort` + its relocation in every -O level.");
        return Ok(1);
    }
    // THE ANTI-INVENTION FLOOR, at corpus level. `tail_calls.rs` pins that
    // noreturn is READ from the callee's declared name and never inferred from
    // IR shape — and a sabotage that declared EVERY import noreturn (the
    // invention direction, the dangerous one) moved exactly one corpus function
    // and passed every gate above this line. That is the hole this closes: every
    // site the noreturn census names must be a name the front-end table
    // declares. Naming a callee that does come back as one that does not ends a
    // live block with `__builtin_unreachable()`, which is a silent miscompile —
    // the one failure mode this pipeline refuses to have.
    let mut invented: Vec<String> = subjects
        .iter()
        .flat_map(|s| s.rep.noreturn_sites.iter().map(|(_, sym)| sym.clone()))
        .filter(|sym| !rdec_arch_x86::lift::noreturn_symbol(sym.as_str()))
        .collect();
    invented.sort();
    invented.dedup();
    if !invented.is_empty() {
        println!(
            "FAIL: anti-invention floor - the noreturn census names {invented:?}, and the front end does not declare those noreturn."
        );
        println!("      The declared family is `abort exit _exit __stack_chk_fail __fastfail ExitProcess` (lift::noreturn_symbol). Termination is READ from that declaration, never inferred from the bytes or from a block that happens to have no successor.");
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
        return "  sse mnemonics counted: 0 (scalar + packed data-movement modelled, rest refused)"
            .to_string();
    }
    let mut parts: Vec<String> = sse.iter().map(|(k, v)| format!("{k}={v}")).collect();
    parts.sort();
    let tot: u32 = sse.values().sum();
    format!(
        "  sse mnemonics counted: {tot} [{}] (scalar subset and packed data movement / lane-independent bitwise modelled, the rest refused by name; zeroing idiom: {zeroing})",
        parts.join(" ")
    )
}

#[cfg(test)]
mod frame_alias_tests {
    use super::{frame_alias_map, frame_touch, frame_touched, key, Subject};
    use rdec_arch_x86::lift::{FnAbi, Report, STACK_BASE};
    use rdec_core::emit_c::Refusal;
    use rdec_core::ir::{BinOp, Callee, Function, Op, Terminator};
    use rdec_core::ty::Ty;
    use std::collections::HashMap;

    // The corpus's own slot addresses (measured in `corpus/c/all.c`:
    // `call_helper.O0::f` spills to 0x7fff8/0x7fffc, `h` to the same pair).
    const SLOT_A: u64 = STACK_BASE - 4; // 0x7fffc
    const SLOT_B: u64 = STACK_BASE - 8; // 0x7fff8

    fn subject(cname: &str, f: Function) -> Subject {
        Subject {
            obj: "fixture.obj".to_string(),
            sym: cname.rsplit('_').next().unwrap_or(cname).to_string(),
            cname: cname.to_string(),
            start: 0x1000,
            external: true,
            f,
            abi: FnAbi::integer(),
            rep: Report::default(),
            relocs_parsed: 0,
            iso: Default::default(),
        }
    }

    /// The `-O0` spill shape, minimal: store the parameter into the frame
    /// slot `slot`, call each name in `callees` (result read, so the call
    /// survives DCE-shaped passes), reload the slot and return it. Exactly
    /// the shape `rdec c` emitted for `call_helper.O0::f`.
    fn caller(cname: &str, slot: u64, callees: &[&str]) -> Function {
        let mut f = Function::new(cname, 0x1000);
        f.ret_ty = Ty::u64();
        let bb = f.entry;
        let p = f.emit(bb, Op::Param, Ty::u64(), 0x1000);
        f.params.push(p);
        f.param_tys.push(Ty::u64());
        let a = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1001);
        f.emit_void(bb, Op::Store(Ty::u64(), a, p), 0x1002);
        for (i, c) in callees.iter().enumerate() {
            f.emit(
                bb,
                Op::Call {
                    callee: Callee::Symbol(c.to_string()),
                    ret_ty: Ty::u64(),
                    args: Box::new([p]),
                    pure: false,
                },
                Ty::u64(),
                0x1010 + i as u64,
            );
        }
        let a2 = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1020);
        let v = f.emit(bb, Op::Load(Ty::u64(), a2), Ty::u64(), 0x1021);
        f.block_mut(bb).term = Terminator::Return(v);
        f
    }

    /// A leaf: spills its parameter into each of `slots`, reads them back,
    /// returns the last read. `&[]` is the no-frame leg (call_helper O1/O2/Oz).
    fn leaf(cname: &str, slots: &[u64]) -> Function {
        let mut f = Function::new(cname, 0x2000);
        f.ret_ty = Ty::u64();
        let bb = f.entry;
        let p = f.emit(bb, Op::Param, Ty::u64(), 0x2000);
        f.params.push(p);
        f.param_tys.push(Ty::u64());
        let mut last = p;
        for s in slots {
            let a = f.emit(bb, Op::Const(*s), Ty::u64(), 0x2001);
            f.emit_void(bb, Op::Store(Ty::u64(), a, p), 0x2002);
            let a2 = f.emit(bb, Op::Const(*s), Ty::u64(), 0x2003);
            last = f.emit(bb, Op::Load(Ty::u64(), a2), Ty::u64(), 0x2004);
        }
        f.block_mut(bb).term = Terminator::Return(last);
        f
    }

    /// A MODELLED TAIL CALL: the block ends in `TailReturn(the call)`, so
    /// the machine's `jmp callee` kills the caller's activation at the call.
    /// `tailcall_local.O0::f` is this shape (t2 matched it at def1647).
    fn tail_caller(cname: &str, slot: u64, callee: &str) -> Function {
        let mut f = Function::new(cname, 0x1000);
        f.ret_ty = Ty::u64();
        let bb = f.entry;
        let p = f.emit(bb, Op::Param, Ty::u64(), 0x1000);
        f.params.push(p);
        f.param_tys.push(Ty::u64());
        let a = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1001);
        f.emit_void(bb, Op::Store(Ty::u64(), a, p), 0x1002);
        let c = f.emit(
            bb,
            Op::Call {
                callee: Callee::Symbol(callee.to_string()),
                ret_ty: Ty::u64(),
                args: Box::new([p]),
                pure: false,
            },
            Ty::u64(),
            0x1010,
        );
        f.block_mut(bb).term = Terminator::TailReturn(c);
        f
    }

    /// The MEASURED corpus leg, hermetic twin: a PLAIN call (clang at -O0
    /// does not tail-call) whose caller reloads the slot BEFORE the call and
    /// returns the callee's answer - `tailcall_local.O0::f`. The callee
    /// clobbers the slot, and nobody reads it after.
    fn pre_load_caller(cname: &str, slot: u64, callee: &str) -> Function {
        let mut f = Function::new(cname, 0x1000);
        f.ret_ty = Ty::u64();
        let bb = f.entry;
        let p = f.emit(bb, Op::Param, Ty::u64(), 0x1000);
        f.params.push(p);
        f.param_tys.push(Ty::u64());
        let a = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1001);
        f.emit_void(bb, Op::Store(Ty::u64(), a, p), 0x1002);
        let a2 = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1003);
        f.emit(bb, Op::Load(Ty::u64(), a2), Ty::u64(), 0x1004);
        let c = f.emit(
            bb,
            Op::Call {
                callee: Callee::Symbol(callee.to_string()),
                ret_ty: Ty::u64(),
                args: Box::new([p]),
                pure: false,
            },
            Ty::u64(),
            0x1010,
        );
        f.block_mut(bb).term = Terminator::Return(c);
        f
    }

    /// (a) The aliasing direction: a caller with a frame slot at
    /// STACK_BASE-4 calling a callee that touches the same address is
    /// refused, and the refusal names the callee and the shared address -
    /// the text is the map, "which function, which address" is the whole
    /// question a frame-activation refusal has to answer.
    #[test]
    fn a_caller_whose_frame_slot_aliases_its_callees_is_refused_naming_both() {
        let subjects = vec![
            subject("o8_f", caller("o8_f", SLOT_A, &["o8_h"])),
            subject("o8_h", leaf("o8_h", &[SLOT_A, SLOT_B])),
        ];
        let m = frame_alias_map(&subjects);
        let r: &Refusal = m
            .get("o8_f")
            .expect("the caller must be refused: frame slots alias the callee's");
        assert_eq!(key(r), "frame slots alias the callee's");
        assert!(
            r.reason.contains("o8_f and o8_h"),
            "the refusal names caller and callee: {}",
            r.reason
        );
        assert!(
            r.reason.contains(&format!("{SLOT_A:#x}")),
            "the refusal names a shared address: {}",
            r.reason
        );
        // The callee calls nothing: it is not refused for being aliased INTO.
        assert!(!m.contains_key("o8_h"));
        // The touched sets the check runs on, in the window's sorted order.
        assert_eq!(frame_touched(&subjects[0].f), vec![SLOT_A]);
        assert_eq!(frame_touched(&subjects[1].f), vec![SLOT_B, SLOT_A]);
    }

    /// (b) The anti-over-fire direction, all three shapes: a callee whose
    /// lifted body touches NO frame slot, a callee that is an import (no
    /// lifted body at all - nothing is known about its frame), and disjoint
    /// frames. None of these may cost the caller an emission; a check that
    /// refuses every function with frame slots reddens this test, and the
    /// corpus's `memcpy` callers are the shape it would punish.
    #[test]
    fn a_caller_is_not_refused_when_the_callee_touches_no_slot_or_has_no_body() {
        let no_frame_callee = vec![
            subject("o2_f", caller("o2_f", SLOT_A, &["o2_g"])),
            subject("o2_g", leaf("o2_g", &[])),
        ];
        assert!(
            !frame_alias_map(&no_frame_callee).contains_key("o2_f"),
            "a callee with no frame slots aliases nothing"
        );
        let import_callee = vec![subject("o3_f", caller("o3_f", SLOT_A, &["printf"]))];
        assert!(
            !frame_alias_map(&import_callee).contains_key("o3_f"),
            "an import callee (no lifted body) must not cause a refusal"
        );
        let disjoint = vec![
            subject("o5_f", caller("o5_f", SLOT_A, &["o5_g"])),
            subject("o5_g", leaf("o5_g", &[SLOT_B])),
        ];
        assert!(
            !frame_alias_map(&disjoint).contains_key("o5_f"),
            "disjoint frames do not alias"
        );
        // The reach refinement, hermetic twin: a MODELLED TAIL CALL into the
        // same slot. The caller's activation is dead at the machine's
        // `jmp callee`, the emitted C (`return o7_g(...)`) has the same
        // property, and `tailcall_local.O0::f` is a BASE t2 comparison - the
        // refusal may not cost what the defect's reach does not cover.
        let tail = vec![
            subject("o7_f", tail_caller("o7_f", SLOT_A, "o7_g")),
            subject("o7_g", leaf("o7_g", &[SLOT_A])),
        ];
        assert!(
            !frame_alias_map(&tail).contains_key("o7_f"),
            "a modelled tail call shares slots the caller cannot live to read"
        );
        // The -O0 measured leg: a PLAIN call whose caller loaded the shared
        // slot BEFORE the call. `tailcall_local.O0::f` is exactly this IR
        // (dumped from the gate): all its frame loads precede the call, so
        // the callee's clobber is unobservable and the emission stands.
        let pre_load = vec![
            subject("o7_h", pre_load_caller("o7_h", SLOT_A, "o7_i")),
            subject("o7_i", leaf("o7_i", &[SLOT_A])),
        ];
        assert!(
            !frame_alias_map(&pre_load).contains_key("o7_h"),
            "a shared slot the caller reads only BEFORE the call is not a live clobber"
        );
        // The TailReturn exemption's own leg: a frame load whose INDEX follows
        // a modelled tail call. The liveness gate is index-based and would
        // refuse it; the exemption - the activation is dead at the `jmp` - is
        // what keeps the emission standing. Deleting the exemption was measured
        // to leave every other leg green, so this one exists.
        let tail_post = vec![
            subject("o7_j", tail_caller_post_load("o7_j", SLOT_A, "o7_k")),
            subject("o7_k", leaf("o7_k", &[SLOT_A])),
        ];
        assert!(
            !frame_alias_map(&tail_post).contains_key("o7_j"),
            "a frame load whose index follows a TAIL call is not live across it: the \
             TailReturn exemption is what keeps this function emitting"
        );
    }

    /// (c) The recursion leg: `f` calling `f` with frame slots shares one
    /// frame between activations, and the text must say so (self), not name
    /// a second function. A recursion WITHOUT frame slots aliases no slot
    /// and must keep emitting (the fib_rec non-frame legs are that shape).
    #[test]
    fn a_self_recursive_function_with_frame_slots_is_refused_naming_the_self_call() {
        let subjects = vec![subject("o9_f", caller("o9_f", SLOT_B, &["o9_f"]))];
        let m = frame_alias_map(&subjects);
        let r = m
            .get("o9_f")
            .expect("recursion aliases the caller's own frame");
        assert_eq!(key(r), "frame slots alias the callee's");
        assert!(
            r.reason.contains("calls itself"),
            "the recursion text must name the self call: {}",
            r.reason
        );
        assert!(
            r.reason.contains(&format!("{SLOT_B:#x}")),
            "the recursion text names a shared slot: {}",
            r.reason
        );
        // Recursion with no frame slots: nothing of the activation's to alias.
        // The store sits ABOVE the window (the data region), so `f` touches
        // no frame slot at all.
        let no_frame = vec![subject(
            "o9_g",
            caller("o9_g", STACK_BASE + 0x1000, &["o9_g"]),
        )];
        let m2: HashMap<String, Refusal> = frame_alias_map(&no_frame);
        assert!(
            !m2.contains_key("o9_g"),
            "recursion without frame slots aliases no slot: {m2:?}"
        );
    }

    /// A callee whose ONLY frame traffic is a COMPUTED (indexed) address -
    /// the machine shape of an indexed local, `mov %edx, (%rsp,%rcx,4)`,
    /// which lifts to `store %addr = %p` with
    /// `%addr = add 0x7fffc, mul(%param, 4)`. The base constant is inside the
    /// frame window; the address is NOT constant-resolvable, so the model
    /// cannot name the slot the callee writes. Hand-built IR, no compiler:
    /// this is the shape that made the CI `c --corpus` leg print `1 invalid`
    /// (see `crates/cli/tests/frame_activation.rs`), and it is deliberately
    /// spelled with only `Const`/`Param`/`Mul`/`Add` so no one clang version's
    /// lowering is load-bearing.
    fn indexed_leaf(cname: &str, slot: u64) -> Function {
        let mut f = Function::new(cname, 0x2000);
        f.ret_ty = Ty::u64();
        let bb = f.entry;
        let p = f.emit(bb, Op::Param, Ty::u64(), 0x2000);
        f.params.push(p);
        f.param_tys.push(Ty::u64());
        // The index register (a register value; a Param is what a lifted
        // register read is after SSA construction).
        let k = f.emit(bb, Op::Param, Ty::u64(), 0x2001);
        f.params.push(k);
        f.param_tys.push(Ty::u64());
        let scale = f.emit(bb, Op::Const(4), Ty::u64(), 0x2002);
        let off = f.emit(bb, Op::Bin(BinOp::Mul, k, scale), Ty::u64(), 0x2003);
        let base = f.emit(bb, Op::Const(slot), Ty::u64(), 0x2004);
        let addr = f.emit(bb, Op::Bin(BinOp::Add, base, off), Ty::u64(), 0x2005);
        f.emit_void(bb, Op::Store(Ty::u64(), addr, p), 0x2006);
        f.block_mut(bb).term = Terminator::Return(p);
        f
    }

    /// The `Terminator::TailReturn` exemption's OWN witness: a modelled tail
    /// call whose block ALSO contains a frame load at a LATER index than the
    /// call. The machine cannot have one (the block ends in `jmp callee`, so
    /// such a load is dead by construction), and the liveness gate is
    /// index-based, so this is precisely the shape the exemption exists for.
    /// It is pinned because DELETING the exemption was measured (this commit)
    /// to leave every other fixture green: an unexercised guard is a guard no
    /// test can miss.
    fn tail_caller_post_load(cname: &str, slot: u64, callee: &str) -> Function {
        let mut f = tail_caller(cname, slot, callee);
        let bb = f.entry;
        let c = match f.block_mut(bb).term {
            Terminator::TailReturn(c) => c,
            _ => panic!("tail_caller builds a TailReturn"),
        };
        let a = f.emit(bb, Op::Const(slot), Ty::u64(), 0x1020);
        f.emit(bb, Op::Load(Ty::u64(), a), Ty::u64(), 0x1021);
        f.block_mut(bb).term = Terminator::TailReturn(c);
        f
    }

    /// (d) THE COMPILER-INDEPENDENT DIRECTION. The caller spills a constant
    /// displacement (`0x7fffc`) and READS IT BACK AFTER the call - the exact
    /// liveness the rule needs - and its callee's only frame store is the
    /// indexed address above. The callee may write the caller's slot (index 0
    /// writes `0x7fffc` itself); the model cannot bound where it writes, so
    /// the caller must be REFUSED and must not emit. Under-approximating here
    /// is the silent direction: the caller emits, the callee's activation
    /// overwrites the caller's slot, the answer is wrong, and only a
    /// differential pass with the luck of a covering vector sees it.
    #[test]
    fn a_callee_whose_only_frame_store_is_an_indexed_address_refuses_its_caller() {
        let subjects = vec![
            subject("o6_f", caller("o6_f", SLOT_A, &["o6_h"])),
            subject("o6_h", indexed_leaf("o6_h", SLOT_A)),
        ];
        let m = frame_alias_map(&subjects);
        let r: &Refusal = m.get("o6_f").unwrap_or_else(|| {
            panic!(
                "the caller EMITTED: its callee's ONLY frame store is an indexed address \
                 (add {SLOT_A:#x}, mul(param, 4)), which the model cannot resolve, so it must \
                 count as a whole-window touch. refusal map: {:?}",
                m.keys().collect::<Vec<_>>()
            )
        });
        assert_eq!(key(r), "frame slots alias the callee's");
        assert!(
            r.reason.contains("o6_f and o6_h"),
            "the refusal names caller and callee: {}",
            r.reason
        );
        assert!(
            r.reason.contains("unbounded"),
            "the refusal must say WHICH side is unbounded: {}",
            r.reason
        );
        // The traffic the rule reads: the callee names NO slot and is flagged
        // unbounded; the caller keeps its one resolved slot and stays bounded.
        let callee_touch = frame_touch(&subjects[1].f);
        assert!(
            callee_touch.slots.is_empty() && callee_touch.unbounded,
            "the indexed store names no slot and must set the unbounded flag: {:?} {}",
            callee_touch.slots,
            callee_touch.unbounded
        );
        let caller_touch = frame_touch(&subjects[0].f);
        assert!(
            caller_touch.slots == vec![SLOT_A] && !caller_touch.unbounded,
            "a constant-disp caller stays bounded: {:?} {}",
            caller_touch.slots,
            caller_touch.unbounded
        );
        // The callee itself calls nothing: not refused for being aliased into.
        assert!(!m.contains_key("o6_h"));
    }

    /// (e) The same unbounded traffic on the CALLER side: its live set covers
    /// the whole window, so a callee that touches the frame at all shares it.
    /// The callee's store is a constant slot the caller's indexed access can
    /// name no address for - the intersection is empty and yet the clobber is
    /// real, so the caller (which reads a frame cell back after the call) is
    /// refused naming the CALLER as the unbounded side.
    #[test]
    fn a_caller_with_an_indexed_frame_access_is_refused_by_any_frame_touching_callee() {
        let mut caller = Function::new("o6_k", 0x1000);
        caller.ret_ty = Ty::u64();
        let bb = caller.entry;
        let p = caller.emit(bb, Op::Param, Ty::u64(), 0x1000);
        caller.params.push(p);
        caller.param_tys.push(Ty::u64());
        let k = caller.emit(bb, Op::Param, Ty::u64(), 0x1001);
        caller.params.push(k);
        caller.param_tys.push(Ty::u64());
        // store into the indexed slot (unbounded frame traffic)
        let scale = caller.emit(bb, Op::Const(4), Ty::u64(), 0x1002);
        let off = caller.emit(bb, Op::Bin(BinOp::Mul, k, scale), Ty::u64(), 0x1003);
        let base = caller.emit(bb, Op::Const(SLOT_B), Ty::u64(), 0x1004);
        let addr = caller.emit(bb, Op::Bin(BinOp::Add, base, off), Ty::u64(), 0x1005);
        caller.emit_void(bb, Op::Store(Ty::u64(), addr, p), 0x1006);
        // the call, then a load of a CONSTANT slot in a slot the callee does
        // not touch - the intersection is empty; the wildcard is the caller.
        let c = caller.emit(
            bb,
            Op::Call {
                callee: Callee::Symbol("o6_l".to_string()),
                ret_ty: Ty::u64(),
                args: Box::new([p]),
                pure: false,
            },
            Ty::u64(),
            0x1010,
        );
        let a2 = caller.emit(bb, Op::Const(SLOT_A), Ty::u64(), 0x1020);
        let v = caller.emit(bb, Op::Load(Ty::u64(), a2), Ty::u64(), 0x1021);
        let _ = c;
        caller.block_mut(bb).term = Terminator::Return(v);
        let subjects = vec![
            subject("o6_k", caller),
            subject("o6_l", leaf("o6_l", &[SLOT_B])),
        ];
        let m = frame_alias_map(&subjects);
        let r: &Refusal = m.get("o6_k").expect(
            "a caller whose live set covers the whole window shares it with any callee \
             that touches the frame at all",
        );
        assert_eq!(key(r), "frame slots alias the callee's");
        assert!(
            r.reason.contains("o6_k and o6_l") && r.reason.contains("caller's"),
            "the refusal names both sides and says the CALLER is the unbounded one: {}",
            r.reason
        );
    }

    /// (f) The anti-over-fire direction for the new rule: a Param-rooted
    /// computed address - the seeded caller-buffer shape `t2`'s memory
    /// comparison covers - is NOT frame-rooted. A callee whose only memory
    /// traffic is `store [param + 8] = p` touches no frame, so its caller
    /// keeps emitting; make frame-rooted fire on a Param base and the
    /// caller-buffer legs die here first.
    #[test]
    fn a_param_rooted_computed_address_is_not_frame_rooted() {
        let mut callee = Function::new("o4_m", 0x2000);
        callee.ret_ty = Ty::u64();
        let bb = callee.entry;
        let p = callee.emit(bb, Op::Param, Ty::u64(), 0x2000);
        callee.params.push(p);
        callee.param_tys.push(Ty::u64());
        let buf = callee.emit(bb, Op::Param, Ty::u64(), 0x2001);
        callee.params.push(buf);
        callee.param_tys.push(Ty::u64());
        // buf + 0x2008: the caller buffer's own cell (STACK_BASE + 0x8 is the
        // window `t2` seeds at param 0). Out of the frame window by geometry.
        let off = callee.emit(bb, Op::Const(0x2008), Ty::u64(), 0x2002);
        let addr = callee.emit(bb, Op::Bin(BinOp::Add, buf, off), Ty::u64(), 0x2003);
        callee.emit_void(bb, Op::Store(Ty::u64(), addr, p), 0x2004);
        callee.block_mut(bb).term = Terminator::Return(p);
        let subjects = vec![
            subject("o4_f", caller("o4_f", SLOT_A, &["o4_m"])),
            subject("o4_m", callee),
        ];
        let m = frame_alias_map(&subjects);
        assert!(
            !m.contains_key("o4_f"),
            "a Param-rooted computed address is not frame traffic: {:?}",
            m.keys().collect::<Vec<_>>()
        );
        assert!(frame_touched(&subjects[1].f).is_empty());
        assert!(
            !frame_touch(&subjects[1].f).unbounded,
            "a caller-buffer store is not unbounded FRAME traffic"
        );
    }

    /// (g) The wildcard is not a licence to skip LIVENESS. An UNBOUNDED caller
    /// (indexed frame store) that reloads its frame cell BEFORE the call and
    /// then makes a plain call, and the same shape as a modelled TAIL CALL,
    /// must both keep emitting: the callee's clobber is unobservable, exactly
    /// as in `tailcall_local.O0::f` - the reach rule is a measured bound, not
    /// caution. Make the unbounded side refuse on ANY frame-rooted access and
    /// these two legs go red; the `unbounded` assertions below are what prove
    /// the legs are live (a fixture leg that never exercises the rule agrees
    /// with any bug).
    #[test]
    fn an_unbounded_caller_that_reads_its_frame_only_before_the_call_keeps_emitting() {
        for tail in [false, true] {
            let mut f = Function::new("o6_p", 0x1000);
            f.ret_ty = Ty::u64();
            let bb = f.entry;
            let p = f.emit(bb, Op::Param, Ty::u64(), 0x1000);
            f.params.push(p);
            f.param_tys.push(Ty::u64());
            let k = f.emit(bb, Op::Param, Ty::u64(), 0x1001);
            f.params.push(k);
            f.param_tys.push(Ty::u64());
            // indexed frame store: the caller's frame traffic is unbounded
            let scale = f.emit(bb, Op::Const(4), Ty::u64(), 0x1002);
            let off = f.emit(bb, Op::Bin(BinOp::Mul, k, scale), Ty::u64(), 0x1003);
            let base = f.emit(bb, Op::Const(SLOT_B), Ty::u64(), 0x1004);
            let addr = f.emit(bb, Op::Bin(BinOp::Add, base, off), Ty::u64(), 0x1005);
            f.emit_void(bb, Op::Store(Ty::u64(), addr, p), 0x1006);
            // the reload, BEFORE the call (the measured tailcall_local leg)
            let a2 = f.emit(bb, Op::Const(SLOT_A), Ty::u64(), 0x1007);
            let v = f.emit(bb, Op::Load(Ty::u64(), a2), Ty::u64(), 0x1008);
            let c = f.emit(
                bb,
                Op::Call {
                    callee: Callee::Symbol("o6_q".to_string()),
                    ret_ty: Ty::u64(),
                    args: Box::new([p]),
                    pure: false,
                },
                Ty::u64(),
                0x1010,
            );
            f.block_mut(bb).term = if tail {
                Terminator::TailReturn(c)
            } else {
                Terminator::Return(v)
            };
            let touch = frame_touch(&f);
            assert!(
                touch.unbounded,
                "the leg must exercise the wildcard: the caller's own frame traffic is bounded"
            );
            let subjects = vec![
                subject("o6_p", f),
                subject("o6_q", leaf("o6_q", &[SLOT_A, SLOT_B])),
            ];
            let m = frame_alias_map(&subjects);
            assert!(
                !m.contains_key("o6_p"),
                "an unbounded caller with no frame read AFTER the call ({}, {}) is refused: {:?} - \
                 the wildcard widens WHICH load is shared, it does not delete the liveness bound",
                if tail { "tail call" } else { "plain call" },
                SLOT_A,
                m.get("o6_p").map(|r| r.reason.clone()),
            );
        }
    }
}

#[cfg(test)]
mod census_key_tests {
    use super::{clip, key};
    use rdec_core::emit_c::Refusal;

    fn r(reason: &str) -> Refusal {
        Refusal {
            at: 0x2f,
            reason: reason.to_string(),
        }
    }

    /// The clip is a class label, not a symbol. Measured, printed by this very
    /// gate on the MSVC leg: `opaque: lea of an address we cannot name: __Imag`,
    /// where the target is `__ImageBase`. A raw slice invents an identifier and
    /// merges every target sharing a prefix, so the census counts prefixes
    /// instead of work items.
    #[test]
    fn a_clipped_key_announces_the_clip_and_never_looks_like_a_symbol() {
        let long = "lea of an address we cannot name: __ImageBase_and_a_tail_that_keeps_going";
        let k = clip(long, 40);
        assert!(k.ends_with('…'), "the clip must be visible: {k}");
        assert!(
            !k.contains("__Imag"),
            "a clipped key must not print a partial identifier: {k}"
        );
        assert!(
            clip("lea of an address we cannot name: __ImageBase", 64).ends_with("__ImageBase"),
            "the shipped cap must not clip the target it exists to name"
        );
    }

    /// `key()` is shared with `rdec cpp`, so both gates' class lines are the same
    /// map. This pins the MSVC leg's real refusal text end to end.
    #[test]
    fn the_lea_key_names_the_target_it_cannot_name() {
        let k = key(&r("opaque: lea of an address we cannot name: __ImageBase"));
        assert!(
            k.ends_with("__ImageBase"),
            "the census key names the full target, got {k}"
        );
        assert!(!k.contains(" at "), "the site sentence is cut: {k}");
    }
}
