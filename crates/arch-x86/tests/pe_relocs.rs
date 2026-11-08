//! `.reloc` base relocations, hand-built: every address the image embeds, counted.
//!
//! WHY THIS FILE EXISTS. `IMAGE_DIRECTORY_ENTRY_BASERELOC` is the linker's list
//! of places where the file contains an address, so it is the *image-side* twin
//! of the COFF `REL32` records `relocs.rs` reads for objects — and the same
//! lesson applies: the target is in the relocation, not in the bytes. A reader
//! that gets the block walk wrong does not crash, it returns a plausible entry
//! list offset by a few entries, which is the failure mode this suite is built
//! to make impossible: every field is read back by index, page, offset and
//! value, and the entry census is asserted as a PARTITION (`Census::balanced`)
//! so a dropped padding entry is a red test, not a smaller number.
//!
//! THE BYTES ARE HERMETIC (the `pe_fixtures.rs` / `pdata_fixtures.rs` rule: a
//! test that reads a file CI builds asserts which toolchain ran). The good
//! fixture is laid out exactly as the table below quotes it, so every malformed
//! sibling is ONE patch of ONE offset.
//!
//! THE GOOD FIXTURE — (file offset, size, value, field); `pe` = 0x40:
//!
//! ```text
//! 0x44  2  0x8664          Machine              0x46  2  3   NumberOfSections
//! 0x54  2  240             SizeOfOptionalHeader 0x58  2  0x20b  Magic (PE32+)
//! 0x68  4  0x1000          AddressOfEntryPoint
//! 0x70  8  0x140000000     ImageBase
//! 0x90  4  0x4000          SizeOfImage          0xc4  4  16  NumberOfRvaAndSizes
//! 0xf0  4  0x3000          DataDirectory[5].VirtualAddress  (BASERELOC)
//! 0xf4  4  38              DataDirectory[5].Size  (14 + 14 + 10, NOT a multiple
//!                          of 8 — the shape `m7.exe` really has, see below)
//! --- section table at 0x148 -------------------------------------------------
//! 0x148 .text   va 0x1000 vsize 0x100 rawptr 0x200 rawsize 0x200 CODE|EXECUTE|READ
//! 0x170 .rdata  va 0x2000 vsize 0x200 rawptr 0x400 rawsize 0x200 INIT_DATA|READ
//! 0x198 .reloc  va 0x3000 vsize 0x200 rawptr 0x600 rawsize 0x200 INIT_DATA|READ
//! --- 0x200: .text (rva 0x1000) ----------------------------------------------
//! 0x200  e8 0b 00 00 00   call 0x1010   (next 0x1005 + 0xb)
//! 0x205  c3               ret
//! 0x210  b8 2a 00 00 00   mov eax,42
//! 0x215  c3               ret
//! 0x240  ff 25 00 00 00 00   jmp *qword ptr [rip+0]  — reads the 8 bytes at 0x246
//! 0x246  10 10 01 40 00 00 00 00  the address 0x140001010: a CODE target whose
//!                                 DIR64 site is INSIDE .text (the IAT-call shape)
//! 0x250  ff 15 00 00 00 00   call *qword ptr [rip+0] — reads the 8 bytes at 0x256
//! 0x256  40 20 00 40 00 00 00 00  the address 0x140002040: a DATA target (.rdata)
//! --- 0x400: .rdata (rva 0x2000) ---------------------------------------------
//! 0x410  0c f0 ff ff      4-byte addend 0xfffff00c: the REL32 site 0x2010, whose
//!                          target is 0x2010 + 4 - 0xff4 = 0x1010 (code)
//! 0x430  10 10 40 00      4-byte value 0x00401010 at the HIGHLOW site 0x2030: not
//!                          ImageBase + RVA, so the address is refused, not guessed
//! --- 0x600: .reloc (rva 0x3000) = 3 IMAGE_BASE_RELOCATION blocks -------------
//! 0x600  00 10 00 00  0e 00 00 00   block 0  PageRVA 0x1000 SizeOfBlock 14
//! 0x608  46 a0        entry: type 10 (DIR64) offset 0x046  -> site 0x1046 (in .text)
//! 0x60a  56 a0        entry: type 10 (DIR64) offset 0x056  -> site 0x1056 (in .text)
//! 0x60c  00 00        entry: type  0 (ABSOLUTE) offset 0   -> padding, COUNTED
//! 0x60e  00 20 00 00  0e 00 00 00   block 1  PageRVA 0x2000 SizeOfBlock 14
//! 0x616  10 70        entry: type  7 (REL32)  offset 0x010 -> site 0x2010
//! 0x618  20 d0        entry: type 13 (RELOC64) offset 0x020 -> refused BY NAME
//! 0x61a  30 30        entry: type  3 (HIGHLOW) offset 0x030 -> site 0x2030
//! 0x61c  00 30 00 00  0a 00 00 00   block 2  PageRVA 0x3000 SizeOfBlock 10
//! 0x624  f8 af        entry: type 10 (DIR64) offset 0xff8  -> site 0x3ff8, in NO
//!                          section: refused AS AN ENTRY, the table stays readable
//! total 0x800 (2048) bytes of file
//! ```
//!
//! The census this fixture must produce, by construction: 3 blocks, 7 entries =
//! 1 padding + 3 absolute + 1 relative + 1 unreadable-site + 1 refused-kind, and
//! `Census::balanced()` is the assertion that says the reader counted them all.
//!
//! THE DIRECTORY SIZE IS 38, NOT A MULTIPLE OF 8, on purpose. `m7.exe` states
//! BASERELOC size 0x64c = 1612 (remainder 4 mod 8) because its last block is
//! SizeOfBlock 0x4c = 8 + 34 entries; the first cut of the reader required the
//! directory size to be a multiple of the 8-byte header and REFUSED that real,
//! healthy image. The stride a base-relocation table must respect is the
//! 2-byte ENTRY; the 8-byte multiple is a property individual blocks usually
//! have and are not required to have. This fixture carries the real shape so a
//! regression to `% 8` reddens a test instead of a target image.

use rdec_arch_x86::reloc::{
    classify, patch_width, type_name, BaseRelocs, Census, Entry, Kind, REL_BASED_ABSOLUTE,
    REL_BASED_DIR64, REL_BASED_HIGHADJ, REL_BASED_HIGHLOW, REL_BASED_LOW, REL_BASED_REL32,
    REL_BASED_RELOC64,
};
use rdec_arch_x86::{pe::PeImage, reloc};

// --- the fixture's own offsets, so a malformed sibling is ONE named patch ----
const OFF_ENTRY_RVA: usize = 0x68;
const OFF_DIR5_RVA: usize = 0xf0;
const OFF_DIR5_SIZE: usize = 0xf4;
const BLK0: usize = 0x600; // block 0 header (PageRVA, SizeOfBlock)
const BLK0_ENTRIES: usize = 0x608;
const BLK1: usize = 0x60e;
const BLK1_ENTRIES: usize = 0x616;
const BLK2: usize = 0x61c;
const TEXT_RVA: u32 = 0x1000;
const RELOC_RVA: u32 = 0x3000;
const FILE_LEN: usize = 0x800;
const DIR5_SIZE: usize = 38;

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// `(type << 12) | offset`, the entry encoding.
fn ent(kind: u16, offset: u16) -> u16 {
    ((kind & 0xF) << 12) | (offset & 0xFFF)
}

/// The good fixture, laid out exactly as the table above quotes it.
fn pe32plus_with_reloc() -> Vec<u8> {
    let mut b = vec![0u8; FILE_LEN];
    put16(&mut b, 0x00, 0x5a4d); // "MZ"
    put32(&mut b, 0x3c, 0x40); // e_lfanew
    b[0x40..0x44].copy_from_slice(b"PE\0\0");
    put16(&mut b, 0x44, 0x8664); // Machine
    put16(&mut b, 0x46, 3); // NumberOfSections
    put16(&mut b, 0x54, 240); // SizeOfOptionalHeader
    put16(&mut b, 0x56, 0x0022); // Characteristics
    put16(&mut b, 0x58, 0x20b); // Magic: PE32+
    put32(&mut b, OFF_ENTRY_RVA, 0x1000); // AddressOfEntryPoint
    put64(&mut b, 0x70, 0x0001_4000_0000); // ImageBase
    put32(&mut b, 0x78, 0x1000); // SectionAlignment
    put32(&mut b, 0x7c, 0x200); // FileAlignment
    put32(&mut b, 0x90, 0x4000); // SizeOfImage
    put32(&mut b, 0x94, 0x200); // SizeOfHeaders
    put16(&mut b, 0x9c, 3); // Subsystem: Windows CUI
    put16(&mut b, 0x9e, 0x0160); // DllCharacteristics
    put32(&mut b, 0xc4, 16); // NumberOfRvaAndSizes
    put32(&mut b, OFF_DIR5_RVA, RELOC_RVA); // DataDirectory[5] = BASERELOC
    put32(&mut b, OFF_DIR5_SIZE, DIR5_SIZE as u32);

    // Section 0: .text
    b[0x148..0x148 + 5].copy_from_slice(b".text");
    put32(&mut b, 0x150, 0x100); // VirtualSize
    put32(&mut b, 0x154, TEXT_RVA); // VirtualAddress
    put32(&mut b, 0x158, 0x200); // SizeOfRawData
    put32(&mut b, 0x15c, 0x200); // PointerToRawData
    put32(&mut b, 0x16c, 0x6000_0020); // CODE | EXECUTE | READ
                                       // Section 1: .rdata
    b[0x170..0x170 + 6].copy_from_slice(b".rdata");
    put32(&mut b, 0x178, 0x200);
    put32(&mut b, 0x17c, 0x2000);
    put32(&mut b, 0x180, 0x200);
    put32(&mut b, 0x184, 0x400);
    put32(&mut b, 0x194, 0x4000_0040); // INIT_DATA | READ
                                       // Section 2: .reloc
    b[0x198..0x198 + 6].copy_from_slice(b".reloc");
    put32(&mut b, 0x1a0, 0x200);
    put32(&mut b, 0x1a4, RELOC_RVA);
    put32(&mut b, 0x1a8, 0x200);
    put32(&mut b, 0x1ac, 0x600);
    put32(&mut b, 0x1bc, 0x4200_0040); // DISCARDABLE | INIT_DATA | READ

    // .text: a call, a leaf, an IAT-shaped jmp and an IAT-shaped call.
    b[0x200..0x206].copy_from_slice(&[0xe8, 0x0b, 0x00, 0x00, 0x00, 0xc3]);
    b[0x210..0x216].copy_from_slice(&[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]);
    b[0x240..0x246].copy_from_slice(&[0xff, 0x25, 0x00, 0x00, 0x00, 0x00]);
    put64(&mut b, 0x246, 0x0000_0001_4000_1010); // the code address DIR64 names
    b[0x250..0x256].copy_from_slice(&[0xff, 0x15, 0x00, 0x00, 0x00, 0x00]);
    put64(&mut b, 0x256, 0x0000_0001_4000_2040); // the data address DIR64 names

    // .rdata: the REL32 addend and the un-nameable HIGHLOW value.
    put32(&mut b, 0x410, 0xffff_f00c); // 0x2010 + 4 + (-0xff4) = 0x1010
    put32(&mut b, 0x430, 0x0040_1010); // not ImageBase + a 32-bit RVA

    // .reloc: the three blocks, byte-exact per the table above.
    put32(&mut b, BLK0, 0x1000);
    put32(&mut b, BLK0 + 4, 14);
    put16(&mut b, BLK0_ENTRIES, ent(REL_BASED_DIR64, 0x046));
    put16(&mut b, BLK0_ENTRIES + 2, ent(REL_BASED_DIR64, 0x056));
    put16(&mut b, BLK0_ENTRIES + 4, ent(REL_BASED_ABSOLUTE, 0x000));
    put32(&mut b, BLK1, 0x2000);
    put32(&mut b, BLK1 + 4, 14);
    put16(&mut b, BLK1_ENTRIES, ent(REL_BASED_REL32, 0x010));
    put16(&mut b, BLK1_ENTRIES + 2, ent(REL_BASED_RELOC64, 0x020));
    put16(&mut b, BLK1_ENTRIES + 4, ent(REL_BASED_HIGHLOW, 0x030));
    put32(&mut b, BLK2, 0x3000);
    put32(&mut b, BLK2 + 4, 10);
    put16(&mut b, BLK2 + 8, ent(REL_BASED_DIR64, 0xff8));
    b
}

fn table(b: &[u8]) -> BaseRelocs {
    let img = PeImage::load(b, "fixture.exe").expect("the good image parses");
    BaseRelocs::read(&img).expect("the good .reloc parses")
}

fn refusal(b: &[u8]) -> String {
    let img = PeImage::load(b, "fixture.exe").expect("the header patch kept the image loadable");
    BaseRelocs::read(&img).expect_err("the .reloc patch must be refused")
}

// ---------------------------------------------------------------------------
// THE GOOD FIXTURE: every field the reader claims to read, read back.
// ---------------------------------------------------------------------------

#[test]
fn a_three_block_table_reads_back_every_block_and_entry() {
    let r = table(&pe32plus_with_reloc());
    assert_eq!(r.dir_rva, RELOC_RVA, "the BASERELOC directory RVA");
    assert_eq!(r.dir_size, DIR5_SIZE as u32, "the BASERELOC directory size");
    assert_eq!(r.block_count(), 3, "three IMAGE_BASE_RELOCATION blocks");
    assert_eq!(
        r.entry_count(),
        7,
        "every entry, refused and padding included"
    );
    // The walk-coverage fact: the blocks exactly tile the directory.
    assert_eq!(
        r.sum_block_sizes(),
        DIR5_SIZE as u64,
        "sum(SizeOfBlock) equals the declared directory size"
    );
    assert_eq!(
        r.entries_per_block(),
        vec![(0x1000, 3), (0x2000, 3), (0x3000, 1)],
        "per-page entry counts, in page order"
    );

    let b = r.blocks();
    assert_eq!(b[0].page_rva, 0x1000);
    assert_eq!(b[0].size_of_block, 14);
    assert_eq!(b[0].header_rva, 0x3000, "the block's RVA inside .reloc");
    assert_eq!(b[0].file_off, Some(0x600), "the block's file offset");
    assert_eq!(b[0].first_entry, 0);
    assert_eq!(b[0].entry_count, 3);
    assert_eq!(b[1].page_rva, 0x2000);
    assert_eq!(b[1].first_entry, 3);
    assert_eq!(b[2].page_rva, 0x3000);
    assert_eq!(b[2].first_entry, 6);

    let e = r.entries();
    // Entry 0: DIR64 whose site is INSIDE .text — the IAT-call shape, the site
    // the COFF-object reader has no equivalent of.
    assert_eq!(e[0].index, 0);
    assert_eq!(e[0].block, 0);
    assert_eq!(e[0].pos_in_block, 0);
    assert_eq!(e[0].kind, REL_BASED_DIR64);
    assert_eq!(e[0].offset, 0x046, "the raw 12-bit offset field");
    assert_eq!(e[0].site_rva, 0x1046, "PageRVA + offset");
    match &e[0].entry {
        Entry::Absolute { bytes } => assert_eq!(
            bytes.as_slice(),
            &[0x10, 0x10, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00],
            "the 8 bytes DIR64 patches, read from .text (little-endian 0x140001010)"
        ),
        other => panic!("entry 0 must be a readable absolute, got {other:?}"),
    }
    // Entry 1: DIR64 in .text pointing at DATA (the IAT slot the call reads).
    assert_eq!(e[1].site_rva, 0x1056);
    // Entry 2: padding. COUNTED, and named — a census that skipped it would
    // report a 6-entry table for a 7-entry file.
    assert_eq!(e[2].kind, REL_BASED_ABSOLUTE);
    assert!(matches!(e[2].entry, Entry::Padding), "type 0 is padding");
    // Entry 3: REL32, addend read as the file states it.
    assert_eq!(e[3].kind, REL_BASED_REL32);
    assert_eq!(e[3].site_rva, 0x2010);
    match &e[3].entry {
        Entry::Relative { addend } => assert_eq!(*addend, -0xff4, "the signed addend"),
        other => panic!("entry 3 must be relative, got {other:?}"),
    }
    // Entry 4: a type this reader has not read the header of. Refused, named,
    // counted, and NOT applied — the site bytes are never read as this layout.
    assert_eq!(e[4].kind, REL_BASED_RELOC64);
    match &e[4].entry {
        Entry::Refused { reason } => assert!(
            reason.contains("16-byte header"),
            "the refusal says why: {reason}"
        ),
        other => panic!("entry 4 must be refused by name, got {other:?}"),
    }
    // Entry 5: HIGHLOW, readable, 4 bytes.
    assert_eq!(e[5].kind, REL_BASED_HIGHLOW);
    assert_eq!(e[5].site_rva, 0x2030);
    // Entry 6: a site in NO section — refused as an entry, table stays readable.
    assert_eq!(e[6].site_rva, 0x3ff8);
    match &e[6].entry {
        Entry::Unreadable { reason } => assert!(
            reason.contains("0x3ff8"),
            "the refusal names the site it could not read: {reason}"
        ),
        other => panic!("entry 6 must be an unreadable site, got {other:?}"),
    }
}

#[test]
fn the_entry_census_is_a_partition_and_counts_padding() {
    let r = table(&pe32plus_with_reloc());
    let c = r.census();
    assert_eq!(
        c,
        Census {
            entries: 7,
            padding: 1,
            absolute: 3,
            relative: 1,
            unreadable: 1,
            refused_kind: 1,
        },
        "five buckets over seven entries"
    );
    assert!(c.balanced(), "the buckets sum to the entry count: {c:?}");
    assert_eq!(
        r.relocs_by_type(),
        vec![
            (REL_BASED_ABSOLUTE, 1),
            (REL_BASED_HIGHLOW, 1),
            (REL_BASED_REL32, 1),
            (REL_BASED_DIR64, 3),
            (REL_BASED_RELOC64, 1),
        ],
        "the type census, ascending by type, padding INCLUDED"
    );
    // The two per-entry refusals are the loud half of the partition, and the
    // table itself is still returned — one bad entry is not a table defect.
    let refs = r.refusals();
    assert_eq!(refs.len(), 2, "one refused kind, one unreadable site");
    assert!(refs[0].1.contains("RELOC64"), "{}", refs[0].1);
    assert!(refs[1].1.contains("0x3ff8"), "{}", refs[1].1);
}

#[test]
fn a_relocation_inside_code_names_its_target_and_a_half_address_refuses() {
    let b = pe32plus_with_reloc();
    let img = PeImage::load(&b, "fixture.exe").expect("the good image parses");
    let r = table(&b);
    let (targets, unresolved) = r.targets(&img);

    // DIR64 at 0x1046 holds 0x140001010 -> RVA 0x1010, and REL32 at 0x2010 with
    // addend -0xff4 computes 0x2010 + 4 - 0xff4 = 0x1010. Two kinds, one target.
    let mut got: Vec<(u32, u16, u32)> = targets
        .iter()
        .map(|t| (t.site_rva, t.kind, t.target_rva))
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![
            (0x1046, REL_BASED_DIR64, 0x1010),
            (0x1056, REL_BASED_DIR64, 0x2040),
            // 0x2010 + 4 + (-0xff4): the addend is relative to the SITE + 4
            (0x2010, REL_BASED_REL32, 0x1020),
        ],
        "every nameable address, with the site that carried it"
    );
    // HIGHLOW is refused as a FAMILY, before its value is even looked at: this
    // reader does not pair HIGH with LOW, and a half address is not a name.
    assert_eq!(unresolved.len(), 1, "the HIGHLOW entry is the one refusal");
    assert_eq!(unresolved[0].site_rva, 0x2030);
    assert_eq!(unresolved[0].kind, REL_BASED_HIGHLOW);
    assert_eq!(unresolved[0].value, 0x0040_1010);
    assert!(
        unresolved[0].why.contains("does not pair HIGH with LOW"),
        "the refusal says why: {}",
        unresolved[0].why
    );
}

/// The OTHER half of the "address is not named" rule: a `DIR64` whose bytes are
/// readable but whose value is not `ImageBase + a 32-bit RVA inside the image`.
/// A pointer into a foreign module, a heap address, a sentinel — the reader
/// prints the value and refuses the name, because inventing an RVA for a value
/// that is not one is exactly the fiction this repo refuses.
#[test]
fn a_readable_site_whose_value_is_not_an_address_in_this_image_is_refused_by_value() {
    let mut b = pe32plus_with_reloc();
    put64(&mut b, 0x246, 0x0000_7ff0_dead_beef); // a heap-shaped pointer at 0x1046
    let img = PeImage::load(&b, "fixture.exe").expect("the image is still loadable");
    let r = BaseRelocs::read(&img).expect("the table is well formed; only a value moved");
    let (targets, unresolved) = r.targets(&img);
    assert_eq!(
        targets.iter().map(|t| t.site_rva).collect::<Vec<_>>(),
        vec![0x1056, 0x2010],
        "the other targets survive; a bad value costs one entry, not its block"
    );
    let mine = unresolved
        .iter()
        .find(|u| u.site_rva == 0x1046)
        .expect("0x1046 must be refused by value");
    assert_eq!(mine.kind, REL_BASED_DIR64);
    assert_eq!(mine.value, 0x0000_7ff0_dead_beef);
    assert!(
        mine.why
            .contains("not ImageBase 0x140000000 plus a 32-bit RVA"),
        "the refusal names the arithmetic it checked: {}",
        mine.why
    );
}

#[test]
fn every_type_nibble_names_itself_and_the_absolute_types_are_named() {
    // The names are the contract the CLI census prints. `DIR64` (10) is the
    // x64 base-relocation type; the 16-bit family is named because a report
    // that calls HIGH "unknown" hides the half-address it refused to pair.
    for (kind, name) in [
        (REL_BASED_ABSOLUTE, "ABSOLUTE"),
        (REL_BASED_LOW, "LOW"),
        (REL_BASED_HIGHLOW, "HIGHLOW"),
        (REL_BASED_HIGHADJ, "HIGHADJ"),
        (REL_BASED_REL32, "REL32"),
        (REL_BASED_DIR64, "DIR64"),
        (REL_BASED_RELOC64, "RELOC64"),
    ] {
        assert_eq!(type_name(kind), name);
    }
    // 0..=15 are all named: the type field is a nibble, so an unnamed nibble
    // is impossible, and `TYPE?` is a bug if this test ever trips.
    for k in 0..=15u16 {
        assert_ne!(type_name(k), "TYPE?", "nibble {k} prints as a name");
    }
    // Classify is total over the nibble, and every class has a width rule.
    for k in 0..=15u16 {
        let c = classify(k);
        match c {
            Kind::Padding => assert_eq!(patch_width(k), Some(0), "padding patches nothing"),
            Kind::Absolute => assert!(
                patch_width(k).is_some_and(|w| w > 0),
                "absolute type {k} patches a real width"
            ),
            Kind::Relative => assert_eq!(patch_width(k), Some(4), "REL32 patches 4 bytes"),
            // RELOC64 and the ARM64X function-table pair are the refused kinds
            // that DO have a width — and the width is exactly WHY they are
            // refused (a 16-byte header under an 8-byte reader; a 128-bit table
            // entry under a 32/64-bit address model). Every other refused kind
            // has no width at all.
            Kind::Refused(why) => {
                assert!(!why.is_empty(), "refused type {k} carries a reason");
                assert!(
                    patch_width(k).is_none() || matches!(k, 13..=15),
                    "refused type {k} has a patch width this reader would use"
                );
            }
        }
    }
    assert_eq!(classify(REL_BASED_DIR64), Kind::Absolute);
    assert_eq!(classify(REL_BASED_REL32), Kind::Relative);
    assert_eq!(classify(REL_BASED_ABSOLUTE), Kind::Padding);
    // RELOC64 is refused AND has a width: the width is the reason it is refused
    // (a 16-byte header under an 8-byte reader), which the loop above pins.
    assert!(matches!(classify(REL_BASED_RELOC64), Kind::Refused(_)));
    assert_eq!(patch_width(REL_BASED_RELOC64), Some(16));
}

#[test]
fn a_half_address_entry_is_a_target_this_reader_refuses_to_pair() {
    // HIGH (1) and LOW (2) each carry 16 bits of an address. Naming the pair
    // would reconstruct a 64-bit address the file never states as one, so both
    // are counted and refused, and the partition stays whole.
    let mut b = pe32plus_with_reloc();
    put16(&mut b, BLK1_ENTRIES + 4, ent(REL_BASED_LOW, 0x030));
    put16(&mut b, BLK1_ENTRIES + 2, ent(reloc::REL_BASED_HIGH, 0x02e));
    let img = PeImage::load(&b, "fixture.exe").expect("image");
    let r = table(&b);
    let (_, unresolved) = r.targets(&img);
    assert_eq!(
        r.census(),
        Census {
            entries: 7,
            padding: 1,
            absolute: 4,
            relative: 1,
            unreadable: 1,
            refused_kind: 0,
        },
        "the two halves are readable absolutes, so the partition shifts"
    );
    assert!(
        unresolved
            .iter()
            .any(|u| u.kind == REL_BASED_LOW && u.why.contains("16-bit half")),
        "LOW is refused as a half: {:?}",
        unresolved
    );
    assert!(
        unresolved
            .iter()
            .any(|u| u.why.contains("16-bit half") && u.why.contains("HIGH")),
        "HIGH is refused as a half too: {:?}",
        unresolved
    );
}

// ---------------------------------------------------------------------------
// EVERY STRUCTURAL REFUSAL: one patch of the good fixture, one named field.
// ---------------------------------------------------------------------------

#[test]
fn a_block_size_below_its_own_header_is_refused_naming_the_block() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK0 + 4, 6); // SizeOfBlock < 8
    let e = refusal(&b);
    assert!(e.contains("block 0"), "{e}");
    assert!(e.contains("SizeOfBlock is 6"), "{e}");
    assert!(e.contains("8-byte header"), "{e}");
    assert!(e.contains("2048"), "the refusal names the file length: {e}");
}

#[test]
fn a_block_size_that_is_not_a_whole_number_of_entries_is_refused() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK0 + 4, 13); // 13 - 8 = 5: not 2*k
    let e = refusal(&b);
    assert!(e.contains("block 0"), "{e}");
    assert!(e.contains("SizeOfBlock is 13"), "{e}");
    assert!(e.contains("5 bytes of entries"), "{e}");
    assert!(e.contains("2-byte entries"), "{e}");
}

#[test]
fn a_block_running_past_the_directory_end_is_refused_naming_the_directory() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK2 + 4, 10 + 8); // the last block claims 8 bytes of table it has
    let e = refusal(&b);
    assert!(e.contains("block 2"), "{e}");
    assert!(
        e.contains("past the BASERELOC directory's end"),
        "names the directory it crossed: {e}"
    );
    assert!(e.contains("0x3026"), "the table RVA it reached: {e}");
}

#[test]
fn blocks_that_do_not_ascend_by_page_are_refused_naming_the_pair() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK2, 0x2000); // equal to block 1's page: one page patched twice
    let e = refusal(&b);
    assert!(e.contains("not sorted by PageRVA"), "{e}");
    assert!(e.contains("block 2"), "{e}");
    assert!(e.contains("block 1"), "the refusal names BOTH sides: {e}");
    assert!(e.contains("apply that page's patches twice"), "{e}");
}

#[test]
fn an_unaligned_page_is_refused_because_its_offsets_leave_the_page() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK1, 0x2008); // PageRVA not a multiple of 0x1000
    let e = refusal(&b);
    assert!(e.contains("block 1"), "{e}");
    assert!(
        e.contains("not a multiple of 0x1000"),
        "names the alignment rule: {e}"
    );
    assert!(
        e.contains("12 bits WITHIN that page"),
        "says what an unaligned page breaks: {e}"
    );
}

#[test]
fn a_directory_size_that_is_not_a_whole_number_of_entries_is_refused() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, OFF_DIR5_SIZE, (DIR5_SIZE + 1) as u32); // 39: an odd byte
    let e = refusal(&b);
    assert!(e.contains("size 39"), "{e}");
    assert!(e.contains("2-byte entry stride"), "{e}");
    assert!(e.contains("1 byte of remainder"), "{e}");
}

#[test]
fn a_directory_size_below_one_block_header_is_refused() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, OFF_DIR5_SIZE, 6);
    let e = refusal(&b);
    assert!(e.contains("size 6"), "{e}");
    assert!(e.contains("IMAGE_BASE_RELOCATION header"), "{e}");
}

#[test]
fn a_directory_whose_end_overflows_the_rva_space_is_refused() {
    let mut b = pe32plus_with_reloc();
    // 0x3000 + 0xFFFFD000 = 0x1_0000_0000: the table's end is not an RVA.
    put32(&mut b, OFF_DIR5_SIZE, 0xffff_d000);
    let e = refusal(&b);
    assert!(
        e.contains("past the 32-bit RVA space"),
        "names the overflow, not a file bound: {e}"
    );
    assert!(e.contains("0xffffd000"), "{e}");
}

#[test]
fn a_directory_that_promises_more_table_than_the_file_has_is_refused() {
    let mut b = pe32plus_with_reloc();
    put32(&mut b, OFF_DIR5_SIZE, 0x1000); // .reloc carries 0x200 bytes
    let e = refusal(&b);
    assert!(e.contains("declares 4096 bytes of blocks"), "{e}");
    assert!(e.contains("its last byte 0x3fff is not in the file"), "{e}");
    assert!(e.contains("2048-byte file"), "{e}");
}

#[test]
fn an_empty_basereloc_directory_is_a_named_loud_gap() {
    // A fixed-base image legitimately has no .reloc. The reader refuses, and
    // the refusal says what the image declared, so the CLI's loud line is the
    // fact "this image names no embedded addresses", not a silent 0.
    let mut b = pe32plus_with_reloc();
    put32(&mut b, OFF_DIR5_RVA, 0);
    put32(&mut b, OFF_DIR5_SIZE, 0);
    let e = refusal(&b);
    assert!(e.contains("no .reloc"), "{e}");
    assert!(e.contains("index 5"), "names the directory index: {e}");
    assert!(
        e.contains("no base relocations"),
        "says what the image declared: {e}"
    );
}

#[test]
fn a_refused_entry_does_not_cost_the_entries_around_it() {
    // The `.pdata` v2 lesson, re-proved on this table: one RELOC64 entry is
    // that entry's property. The other 6 entries must survive it, because the
    // addresses they name are true addresses of the image.
    let r = table(&pe32plus_with_reloc());
    assert_eq!(
        r.entry_count(),
        7,
        "the table is read past the refused entry"
    );
    assert_eq!(r.blocks().len(), 3, "and past the unreadable site");
    assert_eq!(r.relocs_by_type().iter().map(|(_, n)| n).sum::<usize>(), 7);
    // Remove the refused entry's block entirely: the rest is unchanged.
    let mut b = pe32plus_with_reloc();
    put16(&mut b, BLK1_ENTRIES + 2, ent(REL_BASED_DIR64, 0x020)); // a DIR64 at 0x2020
    put64(&mut b, 0x420, 0x0000_0001_4000_1000); // the address it patches
    let img = PeImage::load(&b, "fixture.exe").expect("image");
    let r2 = table(&b);
    assert_eq!(r2.census().refused_kind, 0, "no refused kinds left");
    assert_eq!(r2.census().entries, 7, "same seven entries");
    let (t, _) = r2.targets(&img);
    assert_eq!(t.len(), 4, "the new DIR64 names an address too");
    assert!(r2.census().balanced());
}

// ---------------------------------------------------------------------------
// THE REAL IMAGE: the probe artifact, cross-checked against a walk that shares
// no code with the reader. Loud SKIP when the artifact is absent.
// ---------------------------------------------------------------------------

/// An INDEPENDENT raw-byte walk of the same table, written from `winnt.h`
/// offsets only and sharing no function with `reloc.rs`. If the two agree, the
/// number is a fact about the file; if the reader drifts by one entry, this
/// disagrees.
/// The independent walk's shape, named because clippy is right that four
/// nested tuples are not a signature a reader checks.
type PythonWalk = (Vec<(u32, usize)>, Vec<(u16, usize)>, usize);

fn python_walk(b: &[u8]) -> PythonWalk {
    let u16 = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32 = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let pe = u32(0x3c) as usize;
    let sizeopt = u16(pe + 20) as usize;
    let opt = pe + 24;
    let dir_rva = u32(opt + 112 + 8 * 5) as usize;
    let dir_size = u32(opt + 112 + 8 * 5 + 4) as usize;
    // .reloc's file offset, from the section table (the walk needs the mapping
    // the loader's own fields give it, computed here independently).
    let nsec = u16(pe + 6) as usize;
    let table = opt + sizeopt;
    let mut off = None;
    for i in 0..nsec {
        let h = table + 40 * i;
        let va = u32(h + 12) as usize;
        let vs = {
            let v = u32(h + 8);
            let r = u32(h + 16);
            if v > 0 {
                v as usize
            } else {
                r as usize
            }
        };
        let ptr = u32(h + 20) as usize;
        if va <= dir_rva && dir_rva < va + vs {
            off = Some(ptr + (dir_rva - va));
        }
    }
    let start = off.expect("the independent walk finds the .reloc section");
    let mut blocks = Vec::new();
    let mut types: Vec<(u16, usize)> = Vec::new();
    let mut total = 0usize;
    let mut pos = start;
    while pos - start < dir_size {
        let page = u32(pos);
        let size = u32(pos + 4) as usize;
        let n = (size - 8) / 2;
        blocks.push((page, n));
        for k in 0..n {
            let e = u16(pos + 8 + 2 * k);
            let t = e >> 12;
            total += 1;
            match types.iter_mut().find(|(x, _)| *x == t) {
                Some((_, c)) => *c += 1,
                None => types.push((t, 1)),
            }
        }
        pos += size;
    }
    types.sort_unstable();
    (blocks, types, total)
}

fn scratch_m7() -> Option<std::path::PathBuf> {
    let root = std::env::var("LOCALAPPDATA")
        .ok()
        .map(std::path::PathBuf::from)?;
    let p = root.join("hermes/cache/scratch/m7/m7.exe");
    p.is_file().then_some(p)
}

#[test]
fn the_real_linked_image_partitions_exactly_as_an_independent_walk_reads_it() {
    let Some(path) = scratch_m7() else {
        println!(
            "SKIP(pe-relocs real-image): m7.exe is not present at \
             %LOCALAPPDATA%/hermes/cache/scratch/m7/m7.exe — the real-image \
             base-relocation leg did not run; rebuild it from the ROADMAP.md recipe"
        );
        return;
    };
    let bytes = std::fs::read(&path).expect("the probe artifact reads");
    let img = PeImage::load(&bytes, &path.display().to_string()).expect("the probe is PE32+");
    let r = BaseRelocs::read(&img).expect("the probe's .reloc parses");
    let (blocks, types, total) = python_walk(&bytes);

    assert_eq!(
        r.entries_per_block(),
        blocks,
        "per-page block census vs the independent walk"
    );
    assert_eq!(
        r.relocs_by_type(),
        types,
        "the type census vs the independent walk"
    );
    assert_eq!(r.entry_count(), total, "the entry count vs the walk");
    assert_eq!(
        r.sum_block_sizes(),
        u64::from(r.dir_size),
        "the blocks exactly tile the directory the header declares"
    );
    let c = r.census();
    assert!(c.balanced(), "the real image partitions: {c:?}");
    // The measured probe census, from `dumpbin /RELOCATIONS m7.exe` and the
    // walk above: 782 entries = 779 DIR64 + 3 ABSOLUTE padding, in 6 blocks,
    // and the directory size 0x64c is NOT a multiple of 8 (last block 0x4c).
    assert_eq!(c.entries, 782, "{c:?}");
    assert_eq!(c.absolute, 779, "DIR64: {c:?}");
    assert_eq!(c.padding, 3, "ABSOLUTE padding is COUNTED: {c:?}");
    assert_eq!(c.relative, 0, "MSVC links .text position-independent");
    assert_eq!(c.unreadable, 0, "every real site is in the file");
    assert_eq!(c.refused_kind, 0, "no exotic type in an x64 MSVC image");
    assert_eq!(r.block_count(), 6, "six blocks");
    assert_eq!(r.dir_size, 0x64c, "the directory size the header states");

    // The 47 `DIR64` sites that hold a `.text` address (38 distinct) are
    // function starts a symbol-free EXE names nowhere else — the reason this
    // reader feeds the discovery seeder at all.
    let (targets, unresolved) = r.targets(&img);
    let text = img.section_named(".text").expect("the probe has .text");
    let text_end = u64::from(text.virtual_address) + text.span();
    let in_text: Vec<u32> = targets
        .iter()
        .filter(|t| {
            let r = u64::from(t.target_rva);
            u64::from(text.virtual_address) <= r && r < text_end
        })
        .map(|t| t.target_rva)
        .collect();
    let mut distinct = in_text.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(in_text.len(), 47, "47 relocations name a .text address");
    assert_eq!(distinct.len(), 38, "38 distinct ones");
    assert_eq!(unresolved.len(), 0, "the probe names every address it has");
    // MEASURED, and the interesting part: 7 of the 38 are NOT `.pdata` starts.
    // `.reloc` is the only source that names them, so the CLI seeds them and
    // prints that provenance rather than blending them into the `.pdata` count.
    let t = rdec_arch_x86::pdata::UnwindTable::read(&img).expect("the probe's .pdata parses");
    let starts: std::collections::BTreeSet<u32> = t.functions().map(|(b, _)| b).collect();
    let reloc_only: Vec<u32> = distinct
        .iter()
        .copied()
        .filter(|r| !starts.contains(r))
        .collect();
    assert_eq!(
        reloc_only,
        vec![0x15a8, 0x15ac, 0x1f20, 0x1fa8, 0x3e40, 0x8c14, 0xa9d8],
        "the code addresses only .reloc names (measured against an independent walk)"
    );
    // 31 of the 38 ARE .pdata starts: on the overlap the two tables agree,
    // which is the cross-check the census is for.
    assert_eq!(distinct.len() - reloc_only.len(), 31);
    // The boundary census the CLI prints, measured: no `DIR64` site is in
    // .text, so the indirect-call-via-relocation class is empty here and the
    // 732 non-code targets are the loud boundaries.
    let sites_in_text = r
        .entries()
        .iter()
        .filter(|e| {
            let r = u64::from(e.site_rva);
            u64::from(text.virtual_address) <= r && r < text_end
        })
        .count();
    assert_eq!(sites_in_text, 0, "the probe patches no code byte");
    assert_eq!(
        targets.len() - in_text.len(),
        732,
        "non-code targets = the named boundaries"
    );
}

#[test]
fn a_corrupt_block_size_cannot_make_the_reader_allocate() {
    // `dir_size` and `SizeOfBlock` are hostile u32s. The reader bounds the
    // directory against the file BEFORE allocating and reads each block through
    // `read_at`, so a table that claims 4 GB of entries refuses, names the
    // field, and does not reserve anything.
    let mut b = pe32plus_with_reloc();
    put32(&mut b, BLK0 + 4, 0x7fff_fff6); // a block claiming ~2 GB
    let e = refusal(&b);
    assert!(e.contains("block 0"), "{e}");
    assert!(e.contains("past the BASERELOC directory's end"), "{e}");
    assert!(e.contains("2147483638"), "the size it claimed: {e}");
}
