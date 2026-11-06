//! Jump tables: the two corpus switch objects, and the shape that is NOT one.
//!
//! Ground truth is `llvm-objdump -r` and `-d` on the fixtures, quoted below
//! verbatim so the expected numbers are auditable and not a restatement of
//! whatever the code happens to compute.
//!
//! `corpus/obj/switch_dense.O0.obj`, `llvm-objdump -r`:
//!
//! ```text
//! RELOCATION RECORDS FOR [.text]:
//! 000000000000001d IMAGE_REL_AMD64_REL32    .rdata
//! RELOCATION RECORDS FOR [.rdata]:
//! 0000000000000000 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000004 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000008 IMAGE_REL_AMD64_REL32    .text
//! 000000000000000c IMAGE_REL_AMD64_REL32    .text
//! 0000000000000010 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000014 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000018 IMAGE_REL_AMD64_REL32    .text
//! 000000000000001c IMAGE_REL_AMD64_REL32    .text
//! ```
//!
//! `llvm-objdump -s -j .rdata` gives the eight 4-byte entries, and `-d` gives
//! the dispatch and the case bodies:
//!
//! ```text
//!  0000 2e000000 3c000000 4a000000 58000000
//!  0010 66000000 74000000 82000000 90000000
//!
//!       1a: 48 8d 05 00 00 00 00   leaq (%rip), %rax   # 0x21  (REL32 -> .rdata)
//!       21: 48 63 0c 88            movslq (%rax,%rcx,4), %rcx
//!       25: 48 01 c8               addq %rcx, %rax
//!       28: ff e0                  jmpq *%rax
//!       2a: c7 44 24 0c 07 00 00 00  movl $0x7, 0xc(%rsp)
//!       32: eb 4e                  jmp 0x82
//!       34: ...                     (and so on: 0x3e, 0x48, 0x52, 0x5c, 0x66, 0x70)
//! ```
//!
//! So the eight case bodies are `.text+0x2a, 0x34, 0x3e, 0x48, 0x52, 0x5c, 0x66,
//! 0x70` - the last one ends at 0x7a, and 0x2a is exactly the first byte of the
//! 80-byte run `rdec discover` reports as uncovered.
//!
//! `switch_fallthrough.O0.obj` is the same shape with four entries
//! (`.rdata`: `36 00 00 00 45 00 00 00 54 00 00 00 65 00 00 00`, dispatch
//! `jmpq *%rax` at 0x30) and case bodies at 0x32, 0x3d, 0x48, 0x55.
//!
//! COFF `REL32` semantics give the arithmetic: the bytes at the site are the
//! addend, and the value the linker writes is `symbol + addend - (site + 4)`.
//! The dispatch computes `table_base + entry` at runtime (`movslq` then `addq`),
//! so a target's offset in the symbol's section is
//! `addend - (site_offset + 4 - table_base_offset)`. Entry 0 of the dense table
//! checks by hand: `0x2e - (0 + 4 - 0) = 0x2a`, the first case body; entry 7:
//! `0x90 - (28 + 4) = 0x70`. The `.text` relocation at 0x1d (`leaq (%rip)`,
//! addend 0) is what names the table base, and it is also what ties the table to
//! a *dispatch site* - a table with no `REL32` naming it in code is data, and an
//! indirect jump with no table is still a gap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rdec_arch_x86::discover::{
    discover_all_with_tables, jump_tables, overlap_free, Code, Gap, Plan, Tables,
};
use rdec_arch_x86::loader::{load, Object};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn obj(rel: &str) -> PathBuf {
    root().join(rel)
}

/// The address space the gates use: every non-empty code section gets its own
/// base, so two sections' offset 0x10 never look like the same byte.
struct Space {
    base_of: Vec<u64>,
    idx_of: HashMap<usize, usize>,
}

fn code_ranges(o: &Object) -> (Vec<Code<'_>>, Space) {
    let mut ranges = Vec::new();
    let mut idx_of = HashMap::new();
    let mut base_of = Vec::new();
    let mut next = 0u64;
    for (i, sec) in o.sections.iter().enumerate() {
        if o.is_code(i) && !sec.bytes.is_empty() {
            idx_of.insert(i, ranges.len());
            base_of.push(next);
            ranges.push(Code {
                base: next,
                bytes: sec.bytes.as_slice(),
            });
            next += sec.bytes.len() as u64 + 16;
        }
    }
    (ranges, Space { base_of, idx_of })
}

/// Discovery exactly as `rdec discover` drives it: symbol entry points plus the
/// call relocations that name a symbol defined in this object.
fn discover_obj(path: &Path) -> (Object, Plan, Tables) {
    let o = load(path).expect("object loads");
    let (ranges, sp) = code_ranges(&o);
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    for i in o.code_symbols() {
        let s = &o.symbols[i];
        let sec = s.section.unwrap();
        if !o.is_code(sec) || s.name.starts_with('.') || s.size == 0 {
            continue;
        }
        if let Some(&r) = sp.idx_of.get(&sec) {
            seeds.entry(r).or_default().0.push(sp.base_of[r] + s.offset);
        }
    }
    for (sec, rels) in o.sections.iter().enumerate() {
        let Some(&r) = sp.idx_of.get(&sec) else {
            continue;
        };
        for rl in &rels.relocations {
            if !matches!(rl.typ, 2 | 4 | 5 | 6 | 7) {
                continue;
            }
            if let Some(t) = o.symbol_of(rl.symbol) {
                if t.section == Some(sec) && !t.name.starts_with('.') {
                    seeds.entry(r).or_default().1.push(sp.base_of[r] + t.offset);
                }
            }
        }
    }
    // Only CODE sections can be addressed: a table entry naming a symbol in a
    // data section has no address in this space, which is a fact to report, not
    // a base to invent.
    let base_of = |sec: usize| -> Option<u64> { sp.idx_of.get(&sec).map(|&r| sp.base_of[r]) };
    let tables = jump_tables(&o, &base_of);
    let plan = discover_all_with_tables(&ranges, &seeds, &tables);
    (o, plan, tables)
}

/// Bytes of the section that no claimed instruction covers, as (offset, len).
/// Every object these tests use has exactly one code section, and the address
/// space built above bases it at 0.
fn uncovered(o: &Object, plan: &Plan, section: &str) -> Vec<(u64, usize)> {
    let sec = o
        .sections
        .iter()
        .find(|s| s.name == section)
        .expect("section exists");
    let mut hit = vec![false; sec.bytes.len()];
    for &(a, l) in &plan.insns {
        let off = a;
        if (off as usize) < hit.len() {
            for k in off as usize..(off as usize + l).min(hit.len()) {
                hit[k] = true;
            }
        }
    }
    let mut out = Vec::new();
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
        out.push((i as u64, j - i));
        i = j;
    }
    out
}

#[test]
fn switch_dense_indirect_jump_resolves_to_the_eight_case_bodies() {
    let (_o, plan, tables) = discover_obj(&obj("corpus/obj/switch_dense.O0.obj"));
    // The dispatch site is `jmpq *%rax` at .text+0x28 (see the header).
    let t = tables
        .iter()
        .find(|(site, _)| **site == 0x28)
        .unwrap_or_else(|| {
            panic!(
                "no jump table at 0x28; the sites are {:?}",
                tables.keys().collect::<Vec<_>>()
            )
        });
    assert_eq!(
        t.1.entries, 8,
        "the .rdata object has 8 REL32-covered slots"
    );
    assert!(
        t.1.unresolved.is_empty(),
        "every entry resolves: {:?}",
        t.1.unresolved
    );
    assert_eq!(
        t.1.targets,
        vec![0x2a, 0x34, 0x3e, 0x48, 0x52, 0x5c, 0x66, 0x70],
        "the eight case bodies, per `llvm-objdump -d` in the header comment"
    );
    // Claimed, not just listed: the bodies are blocks, and the indirect branch
    // is no longer an unresolved gap.
    for a in [0x2a, 0x34, 0x3e, 0x48, 0x52, 0x5c, 0x66, 0x70] {
        assert!(plan.blocks.contains(&a), "case body {a:#x} is not a block");
    }
    assert_eq!(plan.gap_count(Gap::IndirectBranch), 0, "{:?}", plan.gaps);
    assert!(
        overlap_free(&plan.insns),
        "sweeps overlapped: {:?}",
        plan.insns
    );
}

#[test]
fn switch_fallthrough_indirect_jump_resolves_to_four_case_bodies() {
    let (_o, plan, tables) = discover_obj(&obj("corpus/obj/switch_fallthrough.O0.obj"));
    let t = tables
        .iter()
        .find(|(site, _)| **site == 0x30)
        .expect("the dispatch is `jmpq *%rax` at .text+0x30");
    assert_eq!(t.1.entries, 4, "16 bytes of .rdata / 4 = 4 entries");
    assert!(t.1.unresolved.is_empty(), "{:?}", t.1.unresolved);
    assert_eq!(t.1.targets, vec![0x32, 0x3d, 0x48, 0x55]);
    assert_eq!(plan.gap_count(Gap::IndirectBranch), 0, "{:?}", plan.gaps);
}

#[test]
fn resolved_switch_bodies_leave_no_uncovered_code_in_their_objects() {
    // The coverage claim, per object: the whole .text of each switch fixture is
    // instructions the descent reaches.
    for rel in [
        "corpus/obj/switch_dense.O0.obj",
        "corpus/obj/switch_fallthrough.O0.obj",
    ] {
        let (o, plan, _) = discover_obj(&obj(rel));
        let gap = uncovered(&o, &plan, ".text");
        assert!(gap.is_empty(), "{rel}: uncovered bytes {gap:?}");
    }
}

#[test]
fn an_indirect_jump_with_no_relocation_behind_it_stays_a_counted_gap() {
    // No REL32 in code names a data base, so there is no table: the bucket is
    // the one the gate already prints. `jmp *%rax` = ff e0.
    let bytes = vec![0x48, 0x8b, 0x01, 0xff, 0xe0];
    let o = fixture(&[
        (".text", 0x6000_0020, &bytes),
        (".rdata", 0x4050_0040, &[0u8; 8]),
    ]);
    let (_o, plan, tables) = discover_fixture(&o);
    assert!(
        tables.is_empty(),
        "no relocation names a table base: {tables:?}"
    );
    assert_eq!(plan.gap_count(Gap::IndirectBranch), 1, "{:?}", plan.gaps);
}

#[test]
fn a_vtable_is_not_a_jump_table() {
    // The vtable shape: 8-byte slots, `IMAGE_REL_AMD64_ADDR64` (type 1) per
    // slot - `corpus/cpp/obj`'s virtual dispatch. A `jmp *%rax` that reads a
    // slot of it is a thunk/vtable dispatch, NOT a computed jump over a
    // relocation-covered 4-byte table, and must stay in the counted bucket.
    let text = vec![0x48, 0x8b, 0x01, 0xff, 0xe0]; // mov (%rcx),%rax; jmp *%rax
    let rdata = vec![0u8; 16]; // two 8-byte slots
    let o = fixture_with_relocs(
        &[
            (".text", 0x6000_0020, &text),
            (".rdata", 0x4050_0040, &rdata),
        ],
        &[(1, 0x0, 1u16, 0), (1, 0x8, 1, 0)], // ADDR64 -> .text per slot
    );
    let (_o, plan, tables) = discover_fixture(&o);
    assert!(
        tables.is_empty(),
        "an 8-byte ADDR64 table is not a 4-byte REL32 jump table: {tables:?}"
    );
    assert_eq!(plan.gap_count(Gap::IndirectBranch), 1, "{:?}", plan.gaps);
}

#[test]
fn cpp_virtual_dispatch_objects_yield_no_jump_tables() {
    // Real C++ objects: the indirect sites are vtable CALLS, the data is
    // 8-byte/ADDR64, and nothing here is a jump table.
    let dir = root().join("corpus/cpp/obj");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("corpus/cpp/obj")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "obj").unwrap_or(false))
        .collect();
    files.sort();
    assert!(files.len() >= 10, "the C++ corpus is present");
    let mut calls = 0usize;
    let mut tables = 0usize;
    for f in &files {
        let (_o, plan, t) = discover_obj(f);
        tables += t.len();
        calls += plan.gap_count(Gap::IndirectCall);
        assert!(
            plan.tables.is_empty(),
            "{}: a jump table was resolved in a C++ object: {:?}",
            f.display(),
            plan.tables
        );
    }
    assert_eq!(tables, 0, "no C++ object in the corpus has a jump table");
    assert!(calls > 0, "the C++ objects do have indirect vtable calls");
}

#[test]
fn a_partially_resolved_table_is_reported_partially_resolved() {
    // Three 4-byte entries, all covered by REL32 -> .text, but entry 1's symbol
    // is UNDEFINED (section 0): no address to name. The two good targets are
    // discovered, and the site is a counted, named gap - not a silent success.
    //   0: leaq (%rip),%rax   (48 8d 05 <rel32 -> .rdata>)
    //   7: movslq (%rax,%rcx,4),%rcx  (48 63 0c 88)
    //   b: addq %rcx,%rax             (48 01 c8)
    //   e: jmpq *%rax                 (ff e0)
    //  10: c3                         ret
    //  11: c3                         ret
    //  12: c3                         ret
    let text: Vec<u8> = vec![
        0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // leaq (%rip),%rax
        0x48, 0x63, 0x0c, 0x88, // movslq (%rax,%rcx,4),%rcx
        0x48, 0x01, 0xc8, // addq %rcx,%rax
        0xff, 0xe0, // jmpq *%rax
        0xc3, 0xc3, 0xc3, // three case bodies, 1 byte each
    ];
    // addends: COFF REL32 in a table gives target = sym_off + addend - slot - 4,
    // so entry 0 -> 0x10, entry 2 -> 0x12. Entry 1 names `far`, which this
    // object does not define: no address exists to name.
    let rdata: Vec<u8> = vec![
        0x14, 0x00, 0x00, 0x00, // -> .text+0x10
        0x15, 0x00, 0x00, 0x00, // -> `far`: undefined here
        0x1e, 0x00, 0x00, 0x00, // -> .text+0x12
    ];
    let o = fixture_with_relocs_full(
        &[
            (".text", 0x6000_0020, &text),
            (".rdata", 0x4050_0040, &rdata),
        ],
        &[
            (0, 0x3, 4u16, 1), // .text+0x3: the leaq's displacement -> .rdata
            (1, 0x0, 4, 2),    // entry 0 -> `f` (.text+0)
            (1, 0x4, 4, 3),    // entry 1 -> `far` (undefined)
            (1, 0x8, 4, 2),    // entry 2 -> `f`
        ],
        &[
            (".text", 1, 0, 3),
            (".rdata", 2, 0, 3),
            ("f", 1, 0, 2),
            ("far", 0, 0, 2),
        ],
    );
    let (_o, plan, tables) = discover_fixture(&o);
    let t = tables.get(&0xe).unwrap_or_else(|| {
        panic!(
            "no table at the dispatch 0xe: {:?}",
            tables.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(t.entries, 3, "three consecutive REL32 slots");
    assert_eq!(t.targets, vec![0x10, 0x12], "the two entries that resolve");
    assert_eq!(
        t.unresolved.len(),
        1,
        "one slot named no address: {:?}",
        t.unresolved
    );
    assert_eq!(
        plan.gap_count(Gap::PartialTable),
        1,
        "a partially-resolved table is a counted gap: {:?}",
        plan.gaps
    );
    assert_eq!(
        plan.gap_count(Gap::IndirectBranch),
        0,
        "the site is reported as the partial table, not as an unknown indirect jump"
    );
    // The resolved bodies are still claimed, because coverage is coverage.
    assert!(
        plan.blocks.contains(&0x10) && plan.blocks.contains(&0x12),
        "{:?}",
        plan.blocks
    );
}

// ---------------------------------------------------------------------------
// Hand-built COFF objects. `loader.rs`'s own test builder makes one .text
// section with no relocations; these fixtures need a data section, a
// per-section relocation table, and an undefined symbol, so they build the
// whole container: 20 + 40n + (18 per symbol entry) + string table + data.
// ---------------------------------------------------------------------------

struct RelocRow {
    /// Index into `Section`.
    section: usize,
    address: u32,
    typ: u16,
    /// Index into the symbol rows; the raw COFF index is that plus two entries
    /// per section (every section symbol carries one aux record).
    symbol: u32,
}

struct SymRow {
    name: &'static str,
    /// 1-based COFF section number; 0 = undefined.
    secnum: i16,
    value: u32,
    storage: u8,
}

fn build_object(
    sections: &[(&'static str, u32, &[u8])],
    relocs: &[RelocRow],
    syms: &[SymRow],
) -> Vec<u8> {
    let nsec = sections.len();
    // One section-definition symbol (STATIC, named after the section, 1 aux
    // record) per section, then the caller's symbols.
    let entries = 2 * nsec + syms.len();
    let symoff = 20 + 40 * nsec;
    let strtab = symoff + 18 * entries;
    let mut b: Vec<u8> = Vec::new();
    // Section data is laid out after the string table.
    b.extend_from_slice(&0x8664u16.to_le_bytes()); // MACHINE_AMD64
    b.extend_from_slice(&(nsec as u16).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
    b.extend_from_slice(&(symoff as u32).to_le_bytes());
    b.extend_from_slice(&(entries as u32).to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    b.extend_from_slice(&0u16.to_le_bytes()); // Characteristics

    // Section data laid out after the string table, 4-byte aligned.
    let mut data_at: Vec<u32> = Vec::new();
    let mut off = strtab + 4;
    for (_, _, bytes) in sections {
        data_at.push(off as u32);
        off += bytes.len();
        off = (off + 3) & !3;
    }
    let rel_at: Vec<u32> = (0..nsec).map(|_| 0u32).collect();
    let mut rel_off = off;
    let mut rel_pos: Vec<u32> = Vec::new();
    for i in 0..nsec {
        let n = relocs.iter().filter(|r| r.section == i).count();
        rel_pos.push(rel_off as u32);
        rel_off += 10 * n;
    }
    let _ = rel_at;

    for (i, (name, chars, bytes)) in sections.iter().enumerate() {
        let mut n = [0u8; 8];
        n[..name.len().min(8)].copy_from_slice(&name.as_bytes()[..name.len().min(8)]);
        b.extend_from_slice(&n);
        b.extend_from_slice(&(bytes.len() as u32).to_le_bytes()); // VirtualSize
        b.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
        b.extend_from_slice(&(bytes.len() as u32).to_le_bytes()); // SizeOfRawData
        b.extend_from_slice(&data_at[i].to_le_bytes()); // PointerToRawData
        b.extend_from_slice(&rel_pos[i].to_le_bytes()); // PointerToRelocations
        b.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
        let nrel = relocs.iter().filter(|r| r.section == i).count();
        b.extend_from_slice(&(nrel as u16).to_le_bytes()); // NumberOfRelocations
        b.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
        b.extend_from_slice(&chars.to_le_bytes());
    }

    for (i, (name, _, _)) in sections.iter().enumerate() {
        put_sym(&mut b, name, 0, (i + 1) as i16, 0, 3, 1); // section definition
        b.extend_from_slice(&[0u8; 18]); // its aux record
    }
    for s in syms {
        put_sym(&mut b, s.name, s.value, s.secnum, 0x20, s.storage, 0);
    }
    b.extend_from_slice(&4u32.to_le_bytes()); // string table: size field only

    for (_, _, bytes) in sections {
        b.extend_from_slice(bytes);
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
    }
    for i in 0..nsec {
        for r in relocs.iter().filter(|r| r.section == i) {
            b.extend_from_slice(&r.address.to_le_bytes());
            // Raw index: two entries per section header symbol, then the row.
            b.extend_from_slice(&(r.symbol + 2 * nsec as u32).to_le_bytes());
            b.extend_from_slice(&r.typ.to_le_bytes());
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Slice 2: the resolved table reaches the IR, the CFG, the verifier, the
// interpreter and the emitted C. Ground truth for the expected answers is the
// C source the fixtures were built from, as `llvm-objdump -d` spells it out:
// switch_dense stores 7/11/13/17/19/23/29/31 for the eight case bodies and
// `-1` for the out-of-range arm; switch_fallthrough accumulates 1+2+4 (falling
// through), 2+4, 4, and 8.
// ---------------------------------------------------------------------------

use rdec_arch_x86::lift::{lift_one_sigs, no_callees, no_relocs, Report};
use rdec_core::cfg::Cfg;
use rdec_core::interp::{run, Fault, Outcome, Val};
use rdec_core::ir::{Function, Op, Terminator};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

/// Lift `f` out of a switch fixture, in the same space discovery used.
fn lift_fixture(rel: &str) -> (rdec_core::ir::Function, Report) {
    let o = load(&obj(rel)).expect("object loads");
    let (ranges, sp) = code_ranges(&o);
    let (_text_sec, text_base) = sp
        .idx_of
        .iter()
        .find(|(sec, _)| o.sections[**sec].name == ".text")
        .map(|(sec, r)| (*sec, sp.base_of[*r]))
        .expect(".text has a range");
    let rg = ranges
        .iter()
        .find(|r| r.base == text_base)
        .expect("the .text range");
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    for i in o.code_symbols() {
        let s = &o.symbols[i];
        if s.name.starts_with('.') || s.size == 0 {
            continue;
        }
        if let Some(&r) = sp.idx_of.get(&s.section.unwrap()) {
            seeds.entry(r).or_default().0.push(sp.base_of[r] + s.offset);
        }
    }
    let base_of = |sec: usize| -> Option<u64> { sp.idx_of.get(&sec).map(|&r| sp.base_of[r]) };
    let tables = jump_tables(&o, &base_of);
    let plan = discover_all_with_tables(&ranges, &seeds, &tables);
    let start = *plan.functions.first().expect("f is discovered");
    let end = rg.base + rg.bytes.len() as u64;
    lift_one_sigs(
        "f",
        rg.bytes,
        rg.base,
        start,
        end,
        &plan,
        &Default::default(),
        &rdec_arch_x86::lift::FnAbi::integer(),
        &no_callees(),
        &no_relocs(),
        &tables,
    )
}

#[test]
fn the_switch_dispatch_becomes_cfg_edges_that_reach_every_case_body() {
    let (f, rep) = lift_fixture("corpus/obj/switch_dense.O0.obj");
    // The dispatch is no longer an `indirect jmp` refusal.
    let indirect = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(&i.op, rdec_core::ir::Op::Opaque(t) if t.as_ref() == "indirect jmp"))
        .count();
    assert_eq!(indirect, 0, "the computed jump is modelled, not refused");
    // Every case body is a block, and every one of them is reachable from the
    // entry through REAL successor edges.
    let cfg = Cfg::build(&f);
    let bodies = [0x2au64, 0x34, 0x3e, 0x48, 0x52, 0x5c, 0x66, 0x70];
    for a in bodies {
        let bb = f.blocks.iter().find(|b| b.addr == a).unwrap_or_else(|| {
            panic!(
                "case body {a:#x} is not a block: {:?}",
                f.blocks.iter().map(|b| b.addr).collect::<Vec<_>>()
            )
        });
        assert!(
            cfg.is_reachable(bb.id),
            "case body {a:#x} (bb{}) is unreachable",
            bb.id
        );
    }
    // The CFG's edges are the terminator's successors: every target named by a
    // terminator is a block of this function, which is what `verify_ssa` checks.
    verify_ssa(&f).expect("the switch function is valid IR");
    // The dispatch is a `Terminator::Switch`, and its target list IS the table:
    // eight destinations in slot order, each one a block of this function. There
    // is no second target list to keep in step - the CFG carries the only one -
    // so slot order is what makes index k reach the body for case k.
    let switches: Vec<&Terminator> = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Switch { .. }))
        .map(|b| &b.term)
        .collect();
    assert_eq!(switches.len(), 1, "one dispatch, exactly");
    let Terminator::Switch { index, targets } = switches[0] else {
        unreachable!()
    };
    let got: Vec<u64> = targets.iter().map(|&b| f.block(b).addr).collect::<Vec<_>>();
    assert_eq!(got, bodies, "the switch's successors, in table slot order");
    assert_ne!(*index, rdec_core::ir::NO_VAL, "the dispatch has a selector");
    // Counted, at the `jmp` where the evidence is: one table, eight edges,
    // nothing missing and nothing half-resolved.
    assert_eq!(rep.jump_tables, 1);
    assert_eq!(rep.jump_table_edges, 8);
    assert_eq!(rep.jump_table_missing, 0);
    assert_eq!(rep.jump_table_partial, 0);
}

#[test]
fn the_interpreter_runs_the_switch_and_the_case_bodies_answer() {
    let (f, _rep) = lift_fixture("corpus/obj/switch_dense.O0.obj");
    // `corpus/src/switch_dense.c`, verbatim:
    //   int f(int k) { switch (k) { case 0: return 7; case 1: return 11;
    //     case 2: return 13; case 3: return 17; case 4: return 19;
    //     case 5: return 23; case 6: return 29; case 7: return 31;
    //     default: return -1; } }
    // `subq $0x7,%rax; ja` gates the index (CF is set below 0, so `ja` is not
    // taken there), so slot k answers for k in 0..=7 and -1 - the 32-bit store
    // writes 0xffffffff - outside.
    let want: [(u64, u64); 10] = [
        (0, 7),
        (1, 11),
        (2, 13),
        (3, 17),
        (4, 19),
        (5, 23),
        (6, 29),
        (7, 31),
        (8, 0xffff_ffff),
        (99, 0xffff_ffff),
    ];
    for (input, answer) in want {
        let ex = run(&f, &[input]);
        match ex.outcome {
            Outcome::Ret(v) => assert_eq!(v, Val::Int(answer), "f({input}) via the jump table"),
            other => panic!("f({input}) did not return: {other:?}"),
        }
    }
}

#[test]
fn the_fallthrough_switch_answers_through_the_same_edges() {
    let (f, _rep) = lift_fixture("corpus/obj/switch_fallthrough.O0.obj");
    verify_ssa(&f).expect("valid IR");
    // `corpus/src/switch_fallthrough.c`: case 0 falls through 1 and 2 (+1+2+4),
    // case 1 gives +2+4, case 2 gives +4, `break` leaves the chain, case 3 is
    // +8, and anything outside the table leaves `r` at 0.
    for (input, answer) in [(0u64, 7u64), (1, 6), (2, 4), (3, 8), (4, 0), (9, 0)] {
        let ex = run(&f, &[input]);
        match ex.outcome {
            Outcome::Ret(v) => assert_eq!(v, Val::Int(answer), "f({input})"),
            other => panic!("f({input}) did not return: {other:?}"),
        }
    }
}

#[test]
fn an_index_the_table_covers_nothing_for_faults_and_never_guesses() {
    // A machine code jump table covers exactly the slots it has. An index at or
    // past the end names no destination this function owns, so the model stops
    // the run on a fault - never a fallthrough, never a picked block.
    //
    // This is IR built by hand, three blocks, so the rule is pinned with no
    // bounds guard in the way: `switch_dense`'s own `subq $0x7,%rax; ja` hides
    // the case by routing out-of-range inputs to `.Ldefault` before the `jmp`.
    let mut f = Function::new("sw", 0);
    let e = f.entry;
    let idx = f.emit(e, Op::Param, Ty::i64(), 0);
    // `Op::Param` alone does not bind an input; the interpreter binds what
    // `params` lists, in order.
    f.params.push(idx);
    let one = f.new_block(1);
    let two = f.new_block(2);
    let three = f.new_block(3);
    let a = f.emit(one, Op::Const(101), Ty::i64(), 1);
    f.block_mut(one).term = Terminator::Return(a);
    let b = f.emit(two, Op::Const(202), Ty::i64(), 2);
    f.block_mut(two).term = Terminator::Return(b);
    f.block_mut(e).term = Terminator::Switch {
        index: idx,
        targets: vec![one, two],
    };
    f.block_mut(three).term = Terminator::Unreachable;
    verify_ssa(&f).expect("a switch over real blocks verifies");
    assert_eq!(run(&f, &[0]).outcome, Outcome::Ret(Val::Int(101)));
    assert_eq!(run(&f, &[1]).outcome, Outcome::Ret(Val::Int(202)));
    assert_eq!(
        run(&f, &[2]).outcome,
        Outcome::Fault(Fault::SwitchOutOfRange { index: 2, len: 2 }),
        "slot 2 of a 2-slot table faults, it does not reach bb3"
    );
    assert_eq!(
        run(&f, &[u64::MAX]).outcome,
        Outcome::Fault(Fault::SwitchOutOfRange {
            index: u64::MAX,
            len: 2
        })
    );
}

#[test]
fn a_switch_target_that_is_not_a_block_of_the_function_fails_verification() {
    // The claim "every declared target exists in the function's block set" is
    // the verifier's, not the lifter's promise: a target outside the block set -
    // an address the table named that lifting never turned into a block - must
    // stop the function, not ride through as a gap.
    let mut f = Function::new("sw", 0);
    let e = f.entry;
    let idx = f.emit(e, Op::Param, Ty::i64(), 0);
    f.params.push(idx);
    let one = f.new_block(1);
    f.block_mut(one).term = Terminator::Return(idx);
    // Slot 0 names a real block, slot 1 names block id 9 of a 2-block function.
    f.block_mut(e).term = Terminator::Switch {
        index: idx,
        targets: vec![one, 9],
    };
    let err = verify_ssa(&f)
        .expect_err("a switch target outside the block set")
        .join("\n");
    assert!(
        err.contains("not a block of this function"),
        "the verifier names the slot: {err}"
    );
    // And a switch over nothing at all - the shape a table with no resolved slot
    // leaves behind - has no successor, so it is not control flow.
    let mut g = Function::new("empty", 0);
    let ge = g.entry;
    let gi = g.emit(ge, Op::Param, Ty::i64(), 0);
    g.params.push(gi);
    g.block_mut(ge).term = Terminator::Switch {
        index: gi,
        targets: vec![],
    };
    let err = verify_ssa(&g)
        .expect_err("an empty target list is not control flow")
        .join("\n");
    assert!(
        err.contains("empty target list"),
        "the verifier says why: {err}"
    );
}

#[test]
fn the_emitted_c_names_every_case_body_and_traps_past_the_table() {
    // The emitter's `switch` is the CFG's edge list in C spelling: one `case`
    // per slot, each `goto` naming the same label the rest of the file uses for
    // that block, and a `default` that traps - the model's standing answer for a
    // destination it does not have, and what `clang -Werror` accepts (the corpus
    // gate compiles every emitted TU, and these two are among them).
    let (mut f, _rep) = lift_fixture("corpus/obj/switch_dense.O0.obj");
    // The same pre-emission pipeline `rdec c` runs. `Ty::Flags` has no C spelling
    // on purpose, so a function whose flags value flag elimination cannot fuse is
    // a refusal, and the pass that removes it has to run before emission.
    for _ in 0..16 {
        let mut ch = false;
        ch |= rdec_core::fuse::flag_elimination(&mut f);
        ch |= rdec_core::passes::prune_phis(&mut f);
        ch |= rdec_core::passes::const_fold(&mut f);
        ch |= rdec_core::passes::copy_prop(&mut f);
        ch |= rdec_core::passes::simplify_branches(&mut f);
        ch |= rdec_core::passes::dead_store_elimination(&mut f);
        ch |= rdec_core::passes::dead_code_elimination(&mut f);
        if !ch {
            break;
        }
    }
    let out = rdec_core::emit_c::emit_c_sig(&f, &rdec_core::emit_c::Protos::new())
        .expect("the switch function emits C");
    assert!(out.text.contains("switch ("), "a switch statement");
    assert_eq!(
        out.text.matches("case ").count(),
        8,
        "one case per table slot:\n{}",
        out.text
    );
    assert!(
        out.text.contains("default: __builtin_trap();"),
        "the out-of-table arm traps:\n{}",
        out.text
    );
    for k in 0..8 {
        assert!(
            out.text.contains(&format!("case {k}: goto l")),
            "slot {k} jumps to a block label"
        );
    }
}

fn put_sym(b: &mut Vec<u8>, name: &str, value: u32, secnum: i16, typ: u16, storage: u8, naux: u8) {
    let mut n = [0u8; 8];
    n[..name.len().min(8)].copy_from_slice(&name.as_bytes()[..name.len().min(8)]);
    b.extend_from_slice(&n);
    b.extend_from_slice(&value.to_le_bytes());
    b.extend_from_slice(&secnum.to_le_bytes());
    b.extend_from_slice(&typ.to_le_bytes());
    b.push(storage);
    b.push(naux);
}

fn fixture(sections: &[(&'static str, u32, &[u8])]) -> Vec<u8> {
    let syms = [
        SymRow {
            name: ".text",
            secnum: 1,
            value: 0,
            storage: 3,
        },
        SymRow {
            name: ".rdata",
            secnum: 2,
            value: 0,
            storage: 3,
        },
        SymRow {
            name: "f",
            secnum: 1,
            value: 0,
            storage: 2,
        },
    ];
    build_object(sections, &[], &syms)
}

fn fixture_with_relocs(
    sections: &[(&'static str, u32, &[u8])],
    rows: &[(usize, u32, u16, u32)],
) -> Vec<u8> {
    let relocs: Vec<RelocRow> = rows
        .iter()
        .map(|(section, address, typ, symbol)| RelocRow {
            section: *section,
            address: *address,
            typ: *typ,
            symbol: *symbol,
        })
        .collect();
    build_object(
        sections,
        &relocs,
        &[
            SymRow {
                name: ".text",
                secnum: 1,
                value: 0,
                storage: 3,
            },
            SymRow {
                name: ".rdata",
                secnum: 2,
                value: 0,
                storage: 3,
            },
            SymRow {
                name: "f",
                secnum: 1,
                value: 0,
                storage: 2,
            },
        ],
    )
}

fn fixture_with_relocs_full(
    sections: &[(&'static str, u32, &[u8])],
    rows: &[(usize, u32, u16, u32)],
    syms: &[(&'static str, i16, u32, u8)],
) -> Vec<u8> {
    let relocs: Vec<RelocRow> = rows
        .iter()
        .map(|(section, address, typ, symbol)| RelocRow {
            section: *section,
            address: *address,
            typ: *typ,
            symbol: *symbol,
        })
        .collect();
    let owned: Vec<SymRow> = syms
        .iter()
        .map(|(name, secnum, value, storage)| SymRow {
            name,
            secnum: *secnum,
            value: *value,
            storage: *storage,
        })
        .collect();
    build_object(sections, &relocs, &owned)
}

/// Discovery over a hand-built object: the entry point is `f`, the only code
/// symbol.
fn discover_fixture(bytes: &[u8]) -> (Object, Plan, Tables) {
    let o = rdec_arch_x86::loader::load_bytes(bytes, "fixture.obj").expect("fixture parses");
    let (ranges, sp) = code_ranges(&o);
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    for i in o.code_symbols() {
        let s = &o.symbols[i];
        let sec = s.section.unwrap();
        if let Some(&r) = sp.idx_of.get(&sec) {
            seeds.entry(r).or_default().0.push(sp.base_of[r] + s.offset);
        }
    }
    let base_of =
        |sec: usize| -> Option<u64> { sp.idx_of.get(&sec).map(|&r| sp.base_of[r]).or(Some(0)) };
    let tables = jump_tables(&o, &base_of);
    let plan = discover_all_with_tables(&ranges, &seeds, &tables);
    (o, plan, tables)
}
