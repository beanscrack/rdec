//! The relocation table as an evidence source, checked against `llvm-objdump -r`.
//!
//! Two claims, both measured, never trusted:
//!
//! 1. `call_helper.O0.obj` carries its call targets in the RELOCATION table, at
//!    the address of each instruction's displacement field: `callq` at 0x10 has
//!    the unrelocated next-IP as its immediate, and the real target (`h`) lives
//!    in the relocation at 0x11. A call-target resolver that reads only the
//!    immediate sees 0x15 and never 0x50 - this is why `calls: 0 modelled`.
//!    These offsets and types are the field-offset guard for the relocation
//!    parser: read `address` from the wrong file offset and the set stops
//!    matching objdump's, which the corpus-wide test below fires on.
//!
//! 2. the parser's per-section relocation set equals `llvm-objdump -r`'s, over
//!    EVERY object in the fixed corpus - offset, type and symbol name, long
//!    (string-table) names included. Where `llvm-objdump` is not on PATH the
//!    test degrades to the counted census the gate prints, so the number stays
//!    auditable instead of the test going silently vacuous.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_arch_x86::loader::{load, reloc_display, RelocKind, REL_AMD64_ADDR32NB, REL_AMD64_REL32};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn corpus_objects() -> Vec<PathBuf> {
    let man = std::fs::read_to_string(root().join("corpus/manifest.tsv")).expect("manifest");
    man.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name\t"))
        .filter_map(|l| l.split('\t').nth(2).map(|o| root().join(o)))
        .collect()
}

/// `(offset, type-name, symbol)` exactly as `llvm-objdump -r` prints it.
fn objdump_relocs(obj: &Path) -> Option<Vec<(String, u64, String, String)>> {
    let out = Command::new("llvm-objdump")
        .arg("-r")
        .arg(obj)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut sec = String::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RELOCATION RECORDS FOR [") {
            sec = rest.split(']').next().unwrap_or("").to_string();
            continue;
        }
        let mut w = line.split_whitespace();
        let (Some(off), Some(typ), Some(sym)) = (w.next(), w.next(), w.next()) else {
            continue;
        };
        if !typ.starts_with("IMAGE_REL_") {
            continue; // the header lines, and the OFFSET/TYPE/VALUE column row
        }
        let Ok(off) = u64::from_str_radix(off, 16) else {
            continue;
        };
        rows.push((sec.clone(), off, typ.to_string(), sym.to_string()));
    }
    Some(rows)
}

/// The loader's set for one object, in objdump's spelling.
fn loader_relocs(obj: &Path) -> Vec<(String, u64, String, String)> {
    let o = load(obj).expect("loads");
    let mut rows = Vec::new();
    for s in &o.sections {
        for r in &s.relocations {
            rows.push((
                s.name.clone(),
                r.address,
                format!("IMAGE_REL_AMD64_{}", reloc_display(r.typ)),
                r.symbol_name(&o).to_string(),
            ));
        }
    }
    rows.sort();
    rows
}

#[test]
fn call_helper_text_relocs_are_the_call_targets() {
    // The red claim, straight from `llvm-objdump -r` on this object:
    //   0000000000000011 IMAGE_REL_AMD64_REL32  h
    //   000000000000001e IMAGE_REL_AMD64_REL32  h
    //   0000000000000037 IMAGE_REL_AMD64_REL32  h
    // and the disassembly shows the `callq` at 0x10/0x1d/0x36 whose displacement
    // field starts at exactly those addresses.
    let o = load(&root().join("corpus/obj/call_helper.O0.obj")).expect("loads");
    let text = (0..o.sections.len())
        .find(|&i| o.sections[i].name == ".text")
        .expect(".text");
    let set: BTreeMap<u64, (RelocKind, String)> = o.sections[text]
        .relocations
        .iter()
        .map(|r| (r.address, (r.kind(), r.symbol_name(&o).to_string())))
        .collect();
    assert_eq!(set.len(), 3, ".text has exactly three relocations: {set:?}");
    for off in [0x11u64, 0x1e, 0x37] {
        let (kind, sym) = set
            .get(&off)
            .unwrap_or_else(|| panic!("no relocation at {off:#x}; the set is {set:?}"));
        assert_eq!(*kind, RelocKind::Rel32, "reloc at {off:#x} must be REL32");
        assert_eq!(sym, "h", "the call at {:#x} targets h", off - 1);
    }
    // The type numbers are the field-offset guard: REL32 is 4 and ADDR32NB is
    // 3 in the AMD64 COFF spec. Swapping the 2/3 pair (as the first table did)
    // makes the .pdata rows below come out as ADDR32 and this assertion fires.
    assert_eq!(REL_AMD64_REL32, 4);
    assert_eq!(REL_AMD64_ADDR32NB, 3);
    let pdata = (0..o.sections.len())
        .find(|&i| o.sections[i].name == ".pdata")
        .expect(".pdata");
    assert_eq!(
        o.sections[pdata].relocations.len(),
        6,
        ".pdata has six ADDR32NB entries in this object"
    );
    for r in &o.sections[pdata].relocations {
        assert_eq!(
            r.kind(),
            RelocKind::Addr32NB,
            "pdata reloc type read from the wrong offset?"
        );
    }
}

#[test]
fn every_relocation_the_loader_parses_is_the_one_objdump_prints() {
    let objects = corpus_objects();
    assert!(objects.len() > 100, "the corpus is fixed at 120 objects");
    let mut cross_checked = 0usize;
    let mut skipped = 0usize;
    for obj in &objects {
        let mine = loader_relocs(obj);
        match objdump_relocs(obj) {
            Some(mut theirs) => {
                theirs.sort();
                assert_eq!(
                    theirs,
                    mine,
                    "{}: the parsed relocation set is not llvm-objdump's",
                    obj.display()
                );
                cross_checked += 1;
            }
            None => skipped += 1,
        }
    }
    // The counted census, printed so the number is auditable without the test:
    // every relocation in the corpus is parsed, none lands in `other`.
    let mut census = rdec_arch_x86::loader::RelocCensus::default();
    let mut longs = 0usize;
    for obj in &objects {
        let o = load(obj).expect("loads");
        census += &o.reloc_census();
        longs += o
            .sections
            .iter()
            .flat_map(|s| s.relocations.iter())
            .filter(|r| r.symbol_name(&o).starts_with("__xmm@"))
            .count();
    }
    println!(
        "relocs census over {} objects: {census:?} (long string-table names: {longs}; \
         cross-checked against llvm-objdump on {cross_checked}, tool absent on {skipped})",
        objects.len()
    );
    assert_eq!(
        census.other.values().sum::<usize>(),
        0,
        "an unparsed relocation type appeared: {:?}",
        census.other
    );
    assert!(census.parsed > 0, "the corpus has relocations to parse");
    // Long names through the string table: `__xmm@...` constants are named by
    // relocations in the paddd/movsd objects, and a 37-character name never
    // fits the 8-byte short field.
    assert!(
        longs > 0,
        "no long symbol name is reachable from a relocation"
    );
    if cross_checked == 0 {
        println!("NOTE: llvm-objdump absent - the census above is the only cross-check");
    }
}
