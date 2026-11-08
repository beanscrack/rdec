//! `rdec pe-discover` as a GATE: the seed census of a PE32+ image, asserted on a
//! hand-built image whose every number this file laid down (the `pe_fixtures.rs`
//! / `pdata_fixtures.rs` rule — a test that reads a file CI builds asserts which
//! toolchain ran, and skips when the file is absent, which is how a gate
//! silently stops being a gate).
//!
//! THE SEED SOURCES, and the rule each test exists to prove:
//!   * `.pdata` starts that PARSE become seeds. The entries REFUSED (chained
//!     unwind, unknown unwind version) must NOT: those functions are reachable
//!     only through a runtime-linked list or an unwind version this reader does
//!     not implement. The fixture lays DECODABLE code at the refused addresses,
//!     so a leak into the seeds moves the function count and is caught.
//!   * The entry point is a seed when it is in `.text`, and is named-and-refused
//!     when it is not.
//!   * A `.reloc` DIR64 target inside `.text` is a seed, with its PROVENANCE
//!     printed: on `m7.exe` 7 of the 38 code addresses `.reloc` names are NOT
//!     `.pdata` starts, and blending them into the `.pdata` census would hide
//!     exactly that.
//!   * A `.reloc` SITE inside `.text` whose target is NOT in `.text` is a
//!     boundary — the IAT / indirect-call shape. It is named by site and target
//!     RVA, counted as a gap, and given NO callee name: the import directory is
//!     out of scope for this slice, so naming a callee would be an invention.
//!   * A data-to-data pointer is NOT a gap. Counting the 732 vtable-shaped
//!     pointers of `m7.exe` as gaps would bury the 160 indirect calls that ARE
//!     boundaries.
//!   * No `.pdata` prints a loud line. `pe-discover` exits nonzero ONLY when it
//!     finds zero functions, which is the vacuity that has burned this repo
//!     twice.
//!
//! THE GOOD FIXTURE — (file offset, size, value, field):
//! ```text
//! 0x00  2  "MZ"           0x3c  4  0x40   e_lfanew        0x40  4  "PE\0\0"
//! 0x44  4  0x0004_8664    Machine 0x8664 | NumberOfSections 4
//! 0x54  4  0x0022_00f0    SizeOfOptionalHeader 240 | Characteristics
//! 0x58  4  0x0001_020b    Magic 0x20b (PE32+) | linker 1.1
//! 0x68  4  0x1000          AddressOfEntryPoint
//! 0x70  8  0x140000000     ImageBase
//! 0x90  4  0x7000          SizeOfImage
//! 0xc4  4  16              NumberOfRvaAndSizes
//! 0xe0  4  0x4000          DataDirectory[3] (EXCEPTION / .pdata)
//! 0xe4  4  0x30              4 RUNTIME_FUNCTIONs, no padding
//! 0xf0  4  0x6000          DataDirectory[5] (BASERELOC)
//! 0xf4  4  24              14 + 10: NOT a multiple of 8 (the m7.exe shape)
//! --- section table at 0x148, IMAGE_SECTION_HEADER = 40 bytes each ----------
//! 0x148 .text   va 0x1000 vsize 0x1000 raw 0x400 ptr 0x200 CODE|EXECUTE|READ
//! 0x170 .pdata  va 0x4000 vsize 0x40  raw 0x40  ptr 0x400 INIT_DATA|READ
//! 0x198 .reloc  va 0x6000 vsize 24    raw 0x40  ptr 0x600 DISCARDABLE
//! 0x1c0 .data   va 0x3000 vsize 0x40  raw 0x40  ptr 0x800 INIT_DATA|READ
//! --- .text (file 0x200..0x600), the rest of the section is 0xcc ------------
//! 0x200  e8 0b 00 00 00 90 90 c3  rva 0x1000: call 0x1010, nop, nop,ret ENTRY
//! 0x206  10 10 00 40 01 00 00 00  the 8 bytes the DIR64 at site 0x1006 patches
//! 0x210  48 83 c0 01 c3           rva 0x1010: add rax,1; ret              .pdata
//! 0x220  90 90 90 90 90 90 90 90 c3  rva 0x1020                           .pdata
//! 0x230  48 8b c1 c3              rva 0x1030  CHAINED .pdata entry: REFUSED
//! 0x240  48 8b c1 c3              rva 0x1040  version 1 .pdata entry: REFUSED
//! 0x260  00 30 00 40 01 00 00 00  the 8 bytes the DIR64 at site 0x1060 patches
//! --- .pdata (file 0x400..0x440) --------------------------------------------
//! 0x400  10 10 00 00 16 10 00 00 30 40 00 00  0x1010..0x1016 info 0x4030
//! 0x40c  20 10 00 00 29 10 00 00 34 40 00 00  0x1020..0x1029 info 0x4034
//! 0x418  30 10 00 00 31 10 00 00 38 40 00 00  0x1030 info 0x4038: CHAINING flag
//! 0x424  40 10 00 00 41 10 00 00 3c 40 00 00  0x1040 info 0x403c: version 2
//! 0x430  01 02 00 00                UNWIND_INFO ver 1, flags 0, prolog 2, codes 0
//! 0x434  01 02 00 00                the next block: 4 bytes later, no overlap
//! 0x438  21 02 00 00                ver 1 + UNW_FLAG_CHAINING (bit 2 of flags)
//! 0x43c  02 02 00 00                Version 2: a layout this reader has not read
//! --- .reloc (file 0x600..0x620) --------------------------------------------
//! 0x600  00 10 00 00 0e 00 00 00  block 0: page 0x1000, SizeOfBlock 14
//! 0x608  06 70                     DIR64  site 0x1006 -> 0x140001010  code seed
//! 0x60a  60 70                     DIR64  site 0x1060 -> 0x140003000  BOUNDARY
//! 0x60c  00 00                     ABSOLUTE padding (counted, patches nothing)
//! 0x60e  00 30 00 00 0a 00 00 00  block 1: page 0x3000, SizeOfBlock 10
//! 0x616  00 70                     DIR64  site 0x3000 -> 0x140003000  data ptr
//! --- .data (file 0x800..0x840) ---------------------------------------------
//! 0x800  00 30 00 40 01 00 00 00  the 8 bytes block 1 patches
//! ```

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rdec"))
}

// --- the fixture's own offsets, so a mutant is ONE named patch --------------
const PE: usize = 0x40;
const OFF_ENTRY_RVA: usize = PE + 0x28; // 0x68 AddressOfEntryPoint
const OFF_IMAGE_BASE: usize = PE + 0x30; // 0x70 ImageBase (PE32+)
const OFF_SIZE_OF_IMAGE: usize = PE + 0x50; // 0x90 SizeOfImage
const DIR_PDATA: usize = 0xe0; // DataDirectory[3] (pdata_fixtures.rs)
const DIR_BASERELOC: usize = 0xf0; // DataDirectory[5]
const SEC_TEXT: usize = 0x148;
const SEC_PDATA: usize = 0x170;
const SEC_RELOC: usize = 0x198;
const SEC_DATA: usize = 0x1c0;
const TEXT_FILE: usize = 0x200;
const PDATA_FILE: usize = 0x400;
const RELOC_FILE: usize = 0x600;
const BLK0: usize = RELOC_FILE;
const BLK0_ENTRIES: usize = RELOC_FILE + 8;
const BLK1: usize = RELOC_FILE + 14;
const DATA_FILE: usize = 0x800;
const FILE_LEN: usize = 0x1000;
const DIR5_SIZE: usize = 24;

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// One base relocation entry: `type << 12 | offset`, the loader's packing.
fn ent(ty: u16, offset: u16) -> u16 {
    (ty << 12) | offset
}

/// The good image, byte for byte as the table above quotes it.
fn good() -> Vec<u8> {
    let mut b = vec![0u8; FILE_LEN];
    b[0x00..0x02].copy_from_slice(&0x5a4du16.to_le_bytes()); // "MZ"
    put32(&mut b, 0x3c, PE as u32); // e_lfanew
    b[PE..PE + 4].copy_from_slice(b"PE\0\0");
    put32(&mut b, 0x44, 0x0004_8664); // Machine | NumberOfSections = 4
    put32(&mut b, 0x54, 0x0022_00f0); // SizeOfOptionalHeader 240 | Characteristics
    put32(&mut b, 0x58, 0x0001_020b); // Magic 0x20b PE32+ | linker 1.1
    put32(&mut b, OFF_ENTRY_RVA, 0x1000); // AddressOfEntryPoint
    put64(&mut b, OFF_IMAGE_BASE, 0x0001_4000_0000);
    put32(&mut b, OFF_SIZE_OF_IMAGE, 0x7000);
    put32(&mut b, 0xc4, 16); // NumberOfRvaAndSizes
                             // pe.rs refuses an image whose alignments are 0 (it cannot map VA -> file
                             // offset without them), so the fixture carries the real linker pair.
    put32(&mut b, 0x78, 0x1000); // SectionAlignment
    put32(&mut b, 0x7c, 0x200); // FileAlignment
    put32(&mut b, 0x94, 0x200); // SizeOfHeaders
    put32(&mut b, DIR_PDATA, 0x4000);
    put32(&mut b, DIR_PDATA + 4, 0x30); // 4 * 12, no padding
    put32(&mut b, DIR_BASERELOC, 0x6000);
    put32(&mut b, DIR_BASERELOC + 4, DIR5_SIZE as u32);

    // .text
    b[SEC_TEXT..SEC_TEXT + 5].copy_from_slice(b".text");
    put32(&mut b, SEC_TEXT + 8, 0x1000); // VirtualSize
    put32(&mut b, SEC_TEXT + 12, 0x1000); // VirtualAddress
    put32(&mut b, SEC_TEXT + 16, 0x400); // SizeOfRawData
    put32(&mut b, SEC_TEXT + 20, TEXT_FILE as u32); // PointerToRawData
    put32(&mut b, SEC_TEXT + 36, 0x6000_0020); // CODE|EXECUTE|READ
                                               // .pdata
    b[SEC_PDATA..SEC_PDATA + 6].copy_from_slice(b".pdata");
    put32(&mut b, SEC_PDATA + 8, 0x40);
    put32(&mut b, SEC_PDATA + 12, 0x4000);
    put32(&mut b, SEC_PDATA + 16, 0x40);
    put32(&mut b, SEC_PDATA + 20, PDATA_FILE as u32);
    put32(&mut b, SEC_PDATA + 36, 0x4000_0040);
    // .reloc
    b[SEC_RELOC..SEC_RELOC + 6].copy_from_slice(b".reloc");
    put32(&mut b, SEC_RELOC + 8, DIR5_SIZE as u32);
    put32(&mut b, SEC_RELOC + 12, 0x6000);
    put32(&mut b, SEC_RELOC + 16, 0x40);
    put32(&mut b, SEC_RELOC + 20, RELOC_FILE as u32);
    put32(&mut b, SEC_RELOC + 36, 0x0200_0040); // DISCARDABLE | INIT_DATA | READ
                                                // .data
    b[SEC_DATA..SEC_DATA + 5].copy_from_slice(b".data");
    put32(&mut b, SEC_DATA + 8, 0x40);
    put32(&mut b, SEC_DATA + 12, 0x3000);
    put32(&mut b, SEC_DATA + 16, 0x40);
    put32(&mut b, SEC_DATA + 20, DATA_FILE as u32);
    put32(&mut b, SEC_DATA + 36, 0xc000_0040);

    // .text: int3 padding, then the five code bodies the table quotes.
    b[TEXT_FILE..TEXT_FILE + 0x400].fill(0xcc);
    b[0x200..0x208].copy_from_slice(&[0xe8, 0x0b, 0x00, 0x00, 0x00, 0x90, 0x90, 0xc3]);
    b[0x210..0x215].copy_from_slice(&[0x48, 0x83, 0xc0, 0x01, 0xc3]);
    b[0x220..0x229].copy_from_slice(&[0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3]);
    b[0x230..0x234].copy_from_slice(&[0x48, 0x8b, 0xc1, 0xc3]); // 0x1030 chained
    b[0x240..0x244].copy_from_slice(&[0x48, 0x8b, 0xc1, 0xc3]); // 0x1040 version 1
                                                                // The 8 bytes each .text DIR64 patches, as the loader would find them.
    put64(&mut b, 0x206, 0x0000_0001_4000_1010); // site 0x1006 -> code 0x1010
    put64(&mut b, 0x260, 0x0000_0001_4000_3000); // site 0x1060 -> .data 0x3000

    // .pdata: 4 RUNTIME_FUNCTIONs, then the four UNWIND_INFO blocks they name.
    // The encodings are `pdata.rs`'s own: version is `w & 7` and must be 1;
    // chained is the UNW_FLAG_CHAINING bit of `flags = (w >> 3) & 0x1F`, i.e.
    // bit 5 of the header word, NOT an odd UnwindData RVA.
    for (at, bg, en, ud) in [
        (PDATA_FILE, 0x1010u32, 0x1016u32, 0x4030u32),
        (PDATA_FILE + 12, 0x1020, 0x1029, 0x4034),
        (PDATA_FILE + 24, 0x1030, 0x1031, 0x4038), // UNW_FLAG_CHAINING => refused
        (PDATA_FILE + 36, 0x1040, 0x1041, 0x403c), // Version 2 => refused
    ] {
        put32(&mut b, at, bg);
        put32(&mut b, at + 4, en);
        put32(&mut b, at + 8, ud);
    }
    put32(&mut b, PDATA_FILE + 48, 0x0000_0201); // ver 1, flags 0, prolog 2, codes 0
    put32(&mut b, PDATA_FILE + 52, 0x0000_0201); // next block: 4 bytes later
    put32(&mut b, PDATA_FILE + 56, 0x0000_0221); // + UNW_FLAG_CHAINING (flags bit 2)
    put32(&mut b, PDATA_FILE + 60, 0x0000_0202); // Version 2

    // .reloc block 0: page 0x1000, 3 entries.
    put32(&mut b, BLK0, 0x1000);
    put32(&mut b, BLK0 + 4, 14);
    put16(&mut b, BLK0_ENTRIES, ent(10, 0x006)); // DIR64 -> code 0x1010
    put16(&mut b, BLK0_ENTRIES + 2, ent(10, 0x060)); // DIR64 -> the boundary
    put16(&mut b, BLK0_ENTRIES + 4, ent(0, 0)); // ABSOLUTE padding
                                                // .reloc block 1: page 0x3000, 1 entry.
    put32(&mut b, BLK1, 0x3000);
    put32(&mut b, BLK1 + 4, 10);
    put16(&mut b, BLK1 + 8, ent(10, 0x000)); // data-to-data pointer
                                             // .data: the 8 bytes block 1 patches.
    put64(&mut b, DATA_FILE, 0x0000_0001_4000_3000);
    b
}

fn run(bytes: &[u8]) -> String {
    let (out, ok) = run_status(bytes);
    assert!(
        ok,
        "pe-discover must exit 0 on an image that yields functions; output:\n{out}"
    );
    out
}

fn run_status(bytes: &[u8]) -> (String, bool) {
    let dir = std::env::temp_dir().join("rdec-m7b-pe-discover");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("img-{}-{:x}.exe", std::process::id(), {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            h ^= u64::from(*byte);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }));
    std::fs::write(&path, bytes).expect("write image");
    let out = Command::new(bin())
        .arg("pe-discover")
        .arg(&path)
        .output()
        .expect("run rdec pe-discover");
    let mut s = String::from_utf8(out.stdout).expect("utf-8 stdout");
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (s, out.status.success())
}

fn run_file(path: &str) -> String {
    let out = Command::new(bin())
        .arg("pe-discover")
        .arg(path)
        .output()
        .expect("run rdec pe-discover");
    assert!(
        out.status.success(),
        "pe-discover exited nonzero on {path}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut s = String::from_utf8(out.stdout).expect("utf-8 stdout");
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

/// Read back `functions 3,` / `covered 22 bytes` from the census lines.
fn count(word: &str, out: &str) -> u64 {
    let mut it = out.split_whitespace();
    while let Some(t) = it.next() {
        if t == word {
            let raw = it.next().unwrap_or("");
            let digits: String = raw.chars().take_while(|c| c.is_ascii_digit()).collect();
            assert!(!digits.is_empty(), "no number after `{word}` in:\n{out}");
            return digits.parse().expect("digits parse");
        }
    }
    panic!("no `{word}` in output:\n{out}")
}

/// The gate, positively. `.pdata` starts and the entry point become functions;
/// the refused `.pdata` entries — whose addresses hold decodable code — do not.
#[test]
fn pdata_seeds_and_the_entry_point_become_functions_and_refusals_do_not() {
    let out = run(&good());
    println!("{out}");
    // NON-VACUITY, stated as loudly as the command states it: a build of this
    // command that discovers 0 functions fails HERE, not in a log review.
    let functions = count("functions", &out);
    assert!(
        functions > 0,
        "pe-discover found ZERO functions on an image whose .text is decoded \
         code at known RVAs — a vacuous pass:\n{out}"
    );
    assert!(
        out.contains(".pdata: 2 function seeds, 1 chained refusals, 1 version refusals"),
        "the seed census must name the refused classes separately:\n{out}"
    );
    // The seed census — not the function count — is where a refusal leak shows. The linear
    // sweep decodes with the SAME engine a seed uses, so a refused address that happens to
    // decode cleanly becomes a function with or without the leak; the census counts the
    // addresses actually pushed from `.pdata`, and that number moves the instant a refusal
    // is pushed.
    assert!(
        out.contains("(seeded 2 of 4 entries; a refused entry is NEVER a seed)"),
        ".pdata must seed exactly the 2 parsed entries of 4; seeding a refusal reads \
         'seeded 3 of 4':\n{out}"
    );
    assert!(
        out.contains("entry seeds, entry point 0x140001000 (in .text)"),
        "the entry point is a seed and the output must say so:\n{out}"
    );
    // Entry (0x1000) + the two PARSED .pdata starts (0x1010, 0x1020). The code
    // at 0x1030/0x1040 is decodable: a refusal that leaked into the seeds makes
    // this 5, which is exactly why that code was laid down.
    assert_eq!(functions, 3, "entry + the two parsed .pdata starts:\n{out}");
    assert!(
        count("covered", &out) >= 22,
        "the three code bodies are 22 instruction bytes:\n{out}"
    );
    assert!(
        out.contains("overlap_free: true"),
        "byte-claiming is the invariant:\n{out}"
    );
    assert!(
        out.contains("VERDICT: OK"),
        "a real run must say OK:\n{out}"
    );
}

/// The `.reloc` seed path with PROVENANCE, and the boundary shape: the DIR64 at
/// 0x1006 names code `.pdata` already names (so the function count must NOT
/// move), and the DIR64 at 0x1060 has its SITE in code and its target in `.data`
/// — the IAT/indirect-call shape, named by RVA, counted as a gap, given no
/// callee.
#[test]
fn reloc_seeds_code_with_provenance_and_names_a_boundary_without_inventing_a_callee() {
    let out = run(&good());
    assert!(
        out.contains(".reloc: 2 blocks, 4 entries [ABSOLUTE:1 DIR64:3]"),
        "the .reloc census must print, padding counted:\n{out}"
    );
    assert!(
        out.contains("sum(SizeOfBlock) 0x18 == dir size 0x18"),
        "the block sizes must add up to the directory, and say so:\n{out}"
    );
    assert!(
        out.contains("1+3+0+0+0 == 4 balanced: true"),
        "the entry census is a partition and must print as one:\n{out}"
    );
    assert!(
        out.contains(
            "1 name an address in .text (1 distinct), 0 of them seeded with .reloc-only provenance"
        ),
        "a .reloc code target that .pdata already names must be counted, and \
         must NOT be dressed up as a new function start:\n{out}"
    );
    assert!(
        out.contains("boundary: DIR64 relocation at site 0x140001060 (inside .text) names 0x140003000 in section 3 (.data)"),
        "the IAT/indirect-call shape must be NAMED by site, target and section:\n{out}"
    );
    assert!(
        out.contains("callee NOT invented"),
        "the boundary line must say what it refuses to do:\n{out}"
    );
    assert!(
        out.contains("1 are data-to-data pointers (census, not gaps)"),
        "a vtable-shaped pointer is not a code boundary and must not be counted \
         as one — the gap census has to stay small enough to read:\n{out}"
    );
    assert!(
        out.contains("reloc-boundary:1"),
        "the boundary must be in the gap census:\n{out}"
    );
    assert!(
        out.contains(".reloc sites inside .text: 2"),
        "the sites inside code are a census of their own:\n{out}"
    );
    // Seeds unchanged by .reloc here (its code target was already a .pdata
    // start), and the boundary is NOT a function: 3, as the seed test says.
    assert_eq!(count("functions", &out), 3);
    // The VERDICT must not report a quieter gap count than the census does. This
    // command shipped a run where the census said 213 gaps and the verdict said 0
    // (it counted only the reloc boundaries); the total is now spelled out.
    assert!(
        out.contains("VERDICT: OK — 3 functions, 30 .text bytes covered, 1 gaps (0 descent + 1 reloc boundary)"),
        "the verdict's gap total must be the census total, split into its parts:\n{out}"
    );
}

/// Seeds outside the swept range are named and not claimed. Point the entry
/// point at `.data`: the loud line fires, and the functions left are the two
/// `.pdata` starts.
#[test]
fn a_seed_outside_text_is_named_and_not_claimed() {
    let mut b = good();
    put32(&mut b, OFF_ENTRY_RVA, 0x3000); // .data
    let out = run(&b);
    assert!(
        out.contains("entry point 0x140003000 (outside .text: not seeded)"),
        "an entry point outside .text must be named and refused as a seed:\n{out}"
    );
    assert_eq!(
        count("functions", &out),
        2,
        "the entry point in .data is not a function; the two .pdata starts are:\n{out}"
    );
}

/// No `.pdata` at all must be LOUD, not a silent one-function run — and the
/// function count is the proof that nothing leaked in from the refusals.
#[test]
fn a_pe_with_no_pdata_prints_the_loud_line_and_still_finds_the_entry_point() {
    let mut b = good();
    put32(&mut b, DIR_PDATA, 0);
    put32(&mut b, DIR_PDATA + 4, 0);
    let out = run(&b);
    assert!(
        out.contains("no .pdata"),
        "an image with no .pdata must say so, not quietly find one function:\n{out}"
    );
    // The seeds left are the entry point (0x1000, padding, no body) and the one code
    // address .reloc names (0x1010). NOT 4: the chained and version refusals of the
    // .pdata table are gone with the table, and if the command had cached them, this
    // is where the leak would show.
    assert_eq!(
        count("functions", &out),
        2,
        "with .pdata gone the seeds are the entry point and the one code address .reloc \
         names; more than 2 means a .pdata refusal leaked into the seeds:\n{out}"
    );
}

/// The boundary census must survive a `.reloc` table that refuses: the refusal
/// is its own labelled line naming field and value, and it costs only itself.
#[test]
fn a_broken_reloc_table_is_a_loud_line_that_costs_only_itself() {
    let mut b = good();
    put32(&mut b, BLK0 + 4, 6); // SizeOfBlock below the 8-byte header
    let out = run(&b);
    assert!(
        out.contains(".reloc read refused:"),
        "the refusal must be on its own labelled line:\n{out}"
    );
    assert!(
        out.contains("SizeOfBlock is 6"),
        "the refusal must name the field and the value:\n{out}"
    );
    assert!(
        out.contains(".pdata: 2 function seeds"),
        "a broken .reloc must not cost the .pdata seeds:\n{out}"
    );
    assert_eq!(count("functions", &out), 3);
    assert!(
        !out.contains("reloc-boundary"),
        "a table that was never read has no boundaries to report, and must not \
         print a census for it:\n{out}"
    );
}

/// Vacuity is a FAILURE in the command itself, not only in this file: strip
/// every seed source (no `.pdata`, entry point in `.data`) and the run that
/// finds 0 functions must exit nonzero and say its numbers are void.
#[test]
fn zero_functions_is_a_failing_verdict_not_a_quiet_zero() {
    let mut b = good();
    put32(&mut b, DIR_PDATA, 0);
    put32(&mut b, DIR_PDATA + 4, 0);
    put32(&mut b, DIR_BASERELOC, 0); // .reloc is the third seed source: strip it too
    put32(&mut b, DIR_BASERELOC + 4, 0);
    put32(&mut b, OFF_ENTRY_RVA, 0x3000); // .data: not a code seed
    let (out, ok) = run_status(&b);
    assert!(!ok, "a 0-function discovery must exit nonzero:\n{out}");
    assert!(
        out.contains("VERDICT: FAIL"),
        "the output must say FAIL and why:\n{out}"
    );
    assert!(
        out.contains("the coverage numbers above are void"),
        "the FAIL must name what it voids:\n{out}"
    );
}

/// The probe artifact, if a build of it is on this machine: the real-image
/// census, asserted against numbers measured from the bytes by a walk that
/// shares no code with this reader. The skip branch is LOUD, not silent.
#[test]
fn the_real_image_seed_and_reloc_census_is_the_measured_one() {
    let path = "C:/Users/fortn/AppData/Local/hermes/cache/scratch/m7/m7.exe";
    if !PathBuf::from(path).exists() {
        eprintln!(
            "SKIP the_real_image_seed_and_reloc_census_is_the_measured_one: {path} is absent \
             — rebuild it from the ROADMAP.md recipe (clang -O1 main.c + corpus/src/arith_basic.c, \
             linked through MSVC-link); this test is not silently vacuous."
        );
        return;
    }
    let out = run_file(path);
    println!("{out}");
    for needle in [
        ".pdata: 328 function seeds, 4 chained refusals, 4 version refusals",
        ".reloc: 6 blocks, 782 entries [ABSOLUTE:3 DIR64:779]",
        "sum(SizeOfBlock) 0x64c == dir size 0x64c",
        "3+779+0+0+0 == 782 balanced: true",
        "47 name an address in .text (38 distinct), 7 of them seeded with .reloc-only provenance",
        "0 have a SITE in .text and a target outside it",
        "732 are data-to-data pointers (census, not gaps)",
        ".reloc sites inside .text: 0",
        "coverage:",
        "overlap_free: true",
        "VERDICT: OK",
    ] {
        assert!(
            out.contains(needle),
            "output must contain {needle:?}:\n{out}"
        );
    }
    // The 7 `.reloc`-only starts are NAMED, not counted, and they are the ones
    // the independent byte walk found.
    for va in [
        "0x1400015a8",
        "0x1400015ac",
        "0x140001f20",
        "0x140001fa8",
        "0x140003e40",
        "0x140008c14",
        "0x14000a9d8",
    ] {
        assert!(
            out.contains(va),
            "the .reloc-only function start {va} must be named:\n{out}"
        );
    }
    // Non-vacuity on the real image, at the measured number.
    assert_eq!(count("functions", &out), 418, "output:\n{out}");
    assert!(
        count("covered", &out) > 50_000,
        "56266 .text bytes are covered:\n{out}"
    );
}
