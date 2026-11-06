//! Shared machinery for the jump-table tests.
//!
//! Two things live here, and both are shared by `jump_tables.rs` (the corpus
//! objects, the shapes that are NOT a jump table) and `jump_table_fixtures.rs`
//! (the embedded COFF fixtures whose bytes ARE the ground truth):
//!
//! * the address space every gate builds — one base per non-empty code section,
//!   so two sections' offset `0x10` never look like the same byte;
//! * a hand-built COFF container: header, section headers, section-definition
//!   symbols with their aux records, a per-section relocation table, a string
//!   table, section data. This is the technique `loader.rs`'s own `.bss` test
//!   uses (see its `PointerToRawData` fixtures) — the fixtures here need what
//!   that test does not: a data section, relocations, and a defined code symbol
//!   to name the entry point.
//!
//! `#![allow(dead_code)]`: one module, two test binaries, each using a subset.
//! Without it `cargo clippy --all-targets -- -D warnings` fails on the copy that
//! does not call every helper.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use rdec_arch_x86::discover::{discover_all_with_tables, jump_tables, Code, Plan, Tables};
use rdec_arch_x86::loader::{load, Object};

pub fn obj_path(manifest_dir: &str, rel: &str) -> std::path::PathBuf {
    Path::new(manifest_dir).join("../..").join(rel)
}

pub fn load_obj(manifest_dir: &str, rel: &str) -> Object {
    load(&obj_path(manifest_dir, rel)).expect("object loads")
}

/// The address space the gates use: every non-empty code section gets its own
/// base, so two sections' offset 0x10 never look like the same byte.
pub struct Space {
    pub base_of: Vec<u64>,
    pub idx_of: HashMap<usize, usize>,
}

pub fn code_ranges(o: &Object) -> (Vec<Code<'_>>, Space) {
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
pub fn discover_obj(path: &Path) -> (Object, Plan, Tables) {
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

/// Discovery over a hand-built object in memory: the entry points are the code
/// symbols, exactly as the corpus driver takes them from the container.
pub fn discover_fixture(bytes: &[u8]) -> (Object, Plan, Tables) {
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

/// Bytes of the section that no claimed instruction covers, as (offset, len).
/// Every object these tests use has exactly one code section, and the address
/// space built above bases it at 0.
pub fn uncovered(o: &Object, plan: &Plan, section: &str) -> Vec<(u64, usize)> {
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

/// Every `IMAGE_REL_AMD64_REL32` slot in a data section that names a symbol in a
/// CODE section, as (slot offset in the section, the target the COFF arithmetic
/// states). This is the object's OWN list of computed-jump destinations — the
/// version-independent ground truth: a clang that lowers a switch a different way
/// has a different list, and every entry its list contains must be discovered.
///
/// The arithmetic is `discover::table_target`, spelled here independently
/// because a test that computes its expectation with the function under test
/// asserts nothing: `sym_offset + addend - (slot - table_base) - 4`, with the
/// addend being the slot's own 4 bytes in an unlinked object.
pub fn relocated_table_targets(o: &Object, table_section: &str) -> Vec<(u64, u64)> {
    let dsi = o
        .sections
        .iter()
        .position(|s| s.name == table_section)
        .expect("table section exists");
    let dsec = &o.sections[dsi];
    let mut out = Vec::new();
    for rl in &dsec.relocations {
        // IMAGE_REL_AMD64_REL32 == 4.
        if rl.typ != 4 {
            continue;
        }
        let Some(sym) = o.symbol_of(rl.symbol) else {
            continue;
        };
        let Some(tsi) = sym.section else {
            continue;
        };
        if !o.is_code(tsi) {
            continue;
        }
        let slot = rl.address;
        let w = &dsec.bytes[slot as usize..slot as usize + 4];
        let addend = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        // Every corpus table starts at its section's origin (`llvm-objdump -r`
        // shows the run beginning at 0), and so does every fixture here.
        let target = sym.offset as i64 + addend as i64 - slot as i64 - 4;
        out.push((slot, target as u64));
    }
    out.sort_by_key(|(s, _)| *s);
    out
}

// ---------------------------------------------------------------------------
// The hand-built COFF container.
// ---------------------------------------------------------------------------

pub struct RelocRow {
    /// Index into `Section`.
    pub section: usize,
    pub address: u32,
    pub typ: u16,
    /// Index into the symbol rows; the raw COFF index is that plus two entries
    /// per section (every section symbol carries one aux record).
    pub symbol: u32,
}

pub struct SymRow {
    pub name: &'static str,
    /// 1-based COFF section number; 0 = undefined.
    pub secnum: i16,
    pub value: u32,
    pub storage: u8,
}

pub fn build_object(
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

pub fn put_sym(
    b: &mut Vec<u8>,
    name: &str,
    value: u32,
    secnum: i16,
    typ: u16,
    storage: u8,
    naux: u8,
) {
    let mut n = [0u8; 8];
    n[..name.len().min(8)].copy_from_slice(&name.as_bytes()[..name.len().min(8)]);
    b.extend_from_slice(&n);
    b.extend_from_slice(&value.to_le_bytes());
    b.extend_from_slice(&secnum.to_le_bytes());
    b.extend_from_slice(&typ.to_le_bytes());
    b.push(storage);
    b.push(naux);
}

pub fn fixture(sections: &[(&'static str, u32, &[u8])]) -> Vec<u8> {
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

pub fn fixture_with_relocs(
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

pub fn fixture_with_relocs_full(
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
