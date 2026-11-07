//! `.pdata` SEH unwind tables, hand-built: the RUNTIME_FUNCTION array IS the claim.
//!
//! WHY THIS FILE EXISTS. A stripped, symbol-free EXE still carries its function
//! starts: the Windows x64 loader binary-searches the `IMAGE_DIRECTORY_ENTRY_EXCEPTION`
//! array of RUNTIME_FUNCTIONs on every thrown exception, so the linker must have
//! written one entry per function, sorted, bounded, and pointing at parseable
//! UNWIND_INFO. That makes `.pdata` the *symbol-free function-start source* for
//! whole binaries (M7 slice 2): the array is a census of the image's code, and
//! every claim it makes (`BeginAddress` in a code section, `EndAddress > Begin`,
//! 4-aligned `UnwindData` in readable bytes, version 1) is a byte-level fact this
//! reader checks and refuses BY NAME, entry and field, when the file lies.
//!
//! THE BYTES ARE HERMETIC (the `pe_fixtures.rs` / `jump_table_fixtures.rs` rule:
//! a test that reads a file CI builds asserts which toolchain ran, not what the
//! code does). `pe32plus_with_pdata()` lays the fixture out exactly as the table
//! below quotes it, so every malformed sibling is ONE patch of one offset, and
//! the refusal under test is attributable to exactly one field.
//!
//! THE GOOD FIXTURE — (file offset, size, value, field), all little-endian;
//! `pe` = `e_lfanew` = 0x40. It extends the `pe_fixtures.rs` layout (same DOS +
//! COFF + optional-header bytes) to three sections and a filled EXCEPTION
//! directory:
//!
//! ```text
//! 0x46  2  3             NumberOfSections   .text, .pdata, .xdata
//! 0xc4  4  16            NumberOfRvaAndSizes
//! 0xe0  4  0x3000        DataDirectory[3].VirtualAddress   (EXCEPTION = .pdata)
//! 0xe4  4  36            DataDirectory[3].Size             (3 entries * 12)
//! 0x90  4  0x5000        SizeOfImage
//! --- section table at 0x148: .text, .pdata, .xdata --------------------------
//! 0x148 .text   vaddr 0x1000 vsize 0x40 rawptr 0x200 rawsize 0x200  CODE|EXECUTE|READ
//! 0x170 .pdata  vaddr 0x3000 vsize 36   rawptr 0x400 rawsize 0x200  INIT_DATA|READ
//! 0x198 .xdata  vaddr 0x4000 vsize 0x210 rawptr 0x600 rawsize 0x200  INIT_DATA|READ
//!               (VirtualSize runs 0x10 past the raw data: virtual bytes with
//!                no file behind them, the m7.exe `.data` shape)
//! --- 0x400: .pdata = 3 RUNTIME_FUNCTIONs (12 bytes each) --------------------
//! 0x400  0x1000 0x1010 0x4000   RF0  begin end unwind   (UWOP_PUSH_NONVOL + ALLOC_SMALL)
//! 0x40c  0x1010 0x1020 0x4008   RF1  begin end unwind   (zero codes: no prolog)
//! 0x418  0x1020 0x1040 0x4010   RF2  begin end unwind   (EHANDLER, frame RBP*16, SET_FPREG)
//! --- 0x600: .xdata = 3 UNWIND_INFO blocks (each starts where the last ENDED:
//!     U0's 2 codes occupy 0x4004..0x4008, so U1 starts at 0x4008, not 0x4004) --
//! 0x600  01 04 02 00 | 00 30 | 05 52   U0: ver 1, flags 0, prolog 4, 2 codes:
//!                                      (off 0, 0x30: PUSH_NONVOL rbx) (off 5, 0x52: ALLOC_SMALL 48)
//! 0x608  01 00 00 00                   U1: ver 1, flags 0, prolog 0, 0 codes
//! 0x610  09 08 01 25 | 08 44           U2: ver 1, flags EHANDLER, prolog 8, 1 code
//!                                      frame reg 5 (RBP), frame offset 2 (x16 bytes);
//!                                      (off 8, 0x44: SET_FPREG)
//! total 0x800 (2048) bytes of file
//! ```
//!
//! The UNWIND_INFO header is a u32: Version bits 0-2, Flags bits 3-7,
//! SizeOfProlog, CountOfCodes, FrameRegister bits 24-27, FrameOffset bits 28-31
//! (`winnt.h`, x64). The `Flags` here is the 5-bit FIELD, so
//! `UNW_FLAG_CHAINING` is 0x4 of that field (bit 0x20 of the u32) — the codes
//! array is stored raw, never interpreted: modelling it is the NEXT slice.

use rdec_arch_x86::pdata::UnwindTable;
use rdec_arch_x86::pe::PeImage;

// --- the fixture's own offsets, so a malformed sibling is ONE named patch ----
const OFF_SIZEOF_IMAGE: usize = 0x90;
const OFF_DIR3_RVA: usize = 0xe0;
const OFF_DIR3_SIZE: usize = 0xe4;
const RF0_UNWIND: usize = 0x408;
const RF1_BEGIN: usize = 0x40c;
const RF2_BEGIN: usize = 0x418;
const RF2_END: usize = 0x41c;
const RF2_UNWIND: usize = 0x420;
const U0: usize = 0x600; // UNWIND_INFO header of entry 0 (u32 at 0x600)
const U0_CODES: usize = 0x604; // first code of entry 0: U0 has 2 codes, so its
                               // array runs 0x604..0x608 and entry 1 starts at 0x608
const U1: usize = 0x608; // UNWIND_INFO header of entry 1
const U2: usize = 0x610; // UNWIND_INFO header of entry 2
const TEXT_RVA: u32 = 0x1000;
const PDATA_RVA: u32 = 0x3000;
const XDATA_RVA: u32 = 0x4000;
const FILE_LEN: usize = 0x800;

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// The good fixture, laid out exactly as the table above quotes it.
fn pe32plus_with_pdata() -> Vec<u8> {
    let mut b = vec![0u8; FILE_LEN];
    b[0x00..0x02].copy_from_slice(&0x5a4du16.to_le_bytes()); // "MZ"
    put32(&mut b, 0x3c, 0x40); // e_lfanew
    b[0x40..0x44].copy_from_slice(b"PE\0\0");
    // One u32 covers Machine (low 16) and NumberOfSections (high 16) = 3.
    put32(&mut b, 0x44, 0x0003_8664);
    put32(&mut b, 0x54, 0x0022_00f0); // SizeOfOptionalHeader 240 | Characteristics 0x0022
    put32(&mut b, 0x58, 0x0001_020b); // Magic 0x20b PE32+ | linker 1.1
    put32(&mut b, 0x5c, 0x1000); // SizeOfCode
    put32(&mut b, 0x60, 0x200); // SizeOfInitializedData
    put32(&mut b, 0x68, 0x1030); // AddressOfEntryPoint
    put32(&mut b, 0x6c, 0x1000); // BaseOfCode
    put32(&mut b, 0x70, 0x4000_0000); // ImageBase low 4: 0x140000000 is
    put32(&mut b, 0x74, 0x0000_0001); // ImageBase high 4: 0x1 in the high half
    put32(&mut b, 0x78, 0x1000); // SectionAlignment
    put32(&mut b, 0x7c, 0x200); // FileAlignment
    put32(&mut b, 0x80, 0x0000_0006); // MajorOperatingSystemVersion 6
    put32(&mut b, 0x88, 0x0000_0006); // MajorSubsystemVersion 6
    put32(&mut b, OFF_SIZEOF_IMAGE, 0x5000); // SizeOfImage
    put32(&mut b, 0x94, 0x200); // SizeOfHeaders
    put32(&mut b, 0x9c, 0x0160_0003); // Subsystem 3 (CUI) | DllCharacteristics 0x0160
    put32(&mut b, 0xa0, 0x0010_0000); // SizeOfStackReserve
    put32(&mut b, 0xa8, 0x0000_1000); // SizeOfStackCommit
    put32(&mut b, 0xb0, 0x0010_0000); // SizeOfHeapReserve
    put32(&mut b, 0xb8, 0x0000_1000); // SizeOfHeapCommit
    put32(&mut b, 0xc4, 16); // NumberOfRvaAndSizes
    put32(&mut b, OFF_DIR3_RVA, PDATA_RVA); // DataDirectory[3] = EXCEPTION
    put32(&mut b, OFF_DIR3_SIZE, 36); // 3 RUNTIME_FUNCTIONs * 12 bytes

    // Section 0: .text
    b[0x148..0x148 + 5].copy_from_slice(b".text");
    put32(&mut b, 0x150, 0x40); // VirtualSize
    put32(&mut b, 0x154, TEXT_RVA); // VirtualAddress
    put32(&mut b, 0x158, 0x200); // SizeOfRawData
    put32(&mut b, 0x15c, 0x200); // PointerToRawData
    put32(&mut b, 0x16c, 0x6000_0020); // CODE | EXECUTE | READ
                                       // Section 1: .pdata
    b[0x170..0x170 + 6].copy_from_slice(b".pdata");
    put32(&mut b, 0x178, 36); // VirtualSize
    put32(&mut b, 0x17c, PDATA_RVA);
    put32(&mut b, 0x180, 0x200);
    put32(&mut b, 0x184, 0x400);
    put32(&mut b, 0x194, 0x4000_0040); // INIT_DATA | READ
                                       // Section 2: .xdata — VirtualSize 0x210 OVER
                                       // SizeOfRawData 0x200: 16 virtual bytes with no
                                       // file behind them, the m7.exe `.data` shape. The
                                       // past-the-file refusal aims at that tail.
    b[0x198..0x198 + 6].copy_from_slice(b".xdata");
    put32(&mut b, 0x1a0, 0x210);
    put32(&mut b, 0x1a4, XDATA_RVA);
    put32(&mut b, 0x1a8, 0x200);
    put32(&mut b, 0x1ac, 0x600);
    put32(&mut b, 0x1bc, 0x4000_0040); // INIT_DATA | READ

    // The .text content: int3 padding and the entry-point bytes, as in pe_fixtures.
    for i in 0..0x40usize {
        b[0x200 + i] = 0xcc;
    }
    b[0x230..0x236].copy_from_slice(&[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]);

    // The .pdata RUNTIME_FUNCTION array (unwind RVAs per the xdata note below).
    for (at, bg, en, ud) in [
        (0x400usize, 0x1000u32, 0x1010u32, 0x4000u32),
        (0x40c, 0x1010, 0x1020, 0x4008),
        (0x418, 0x1020, 0x1040, 0x4010),
    ] {
        put32(&mut b, at, bg);
        put32(&mut b, at + 4, en);
        put32(&mut b, at + 8, ud);
    }

    // The .xdata UNWIND_INFO blocks. Each block starts where the previous
    // block's codes END: U0 is 4 + 2*2 bytes, so U1 lives at 0x4008 and U2
    // at 0x4010 — the same chaining-without-overlap the real linkers do.
    put32(&mut b, U0, 0x0002_0401); // ver 1, flags 0, prolog 4, codes 2, frame 0
    b[U0_CODES] = 0x00; // code 0: offset 0
    b[U0_CODES + 1] = 0x30; // PUSH_NONVOL, info rbx(3)
    b[U0_CODES + 2] = 0x05; // code 1: offset 5
    b[U0_CODES + 3] = 0x52; // ALLOC_SMALL(2), info 5 -> 48 bytes
    put32(&mut b, U1, 0x0000_0001); // ver 1, flags 0, prolog 0, codes 0
    put32(&mut b, U2, 0x2501_0809); // ver 1, flags EHANDLER(1), prolog 8, codes 1,
                                    // frame reg 5 (RBP), frame offset 2 (x16)
    b[U2 + 4] = 0x08; // code 0: offset 8
    b[U2 + 5] = 0x44; // SET_FPREG(4), info 4
    b
}

fn table(b: &[u8]) -> UnwindTable {
    let img = PeImage::load(b, "fixture.exe").expect("the good image parses");
    UnwindTable::read(&img).expect("the good .pdata parses")
}

fn refusal(b: &[u8]) -> String {
    let img = PeImage::load(b, "fixture.exe").expect("the header patch kept the image loadable");
    UnwindTable::read(&img).expect_err("the unwind-table patch must be refused")
}

// ---------------------------------------------------------------------------
// THE GOOD FIXTURE: every field the reader claims to read, read back.
// ---------------------------------------------------------------------------

#[test]
fn a_three_entry_unwind_table_reads_back_every_field() {
    let t = table(&pe32plus_with_pdata());
    assert_eq!(t.dir_rva, PDATA_RVA, "the EXCEPTION directory RVA");
    assert_eq!(t.dir_size, 36, "the EXCEPTION directory size");
    assert_eq!(t.entry_count(), 3, "dir size / 12");
    let e = t.entries();
    assert_eq!(e.len(), 3, "three RUNTIME_FUNCTIONs");

    // Entry 0: PUSH_NONVOL + ALLOC_SMALL, codes stored RAW (not interpreted).
    assert_eq!(e[0].index, 0);
    assert_eq!(e[0].begin_rva, 0x1000);
    assert_eq!(e[0].end_rva, 0x1010);
    assert_eq!(e[0].unwind_rva, 0x4000);
    assert_eq!(e[0].version, 1, "UNWIND_INFO Version");
    assert_eq!(e[0].flags, 0, "UNW_FLAG_NHANDLER");
    assert_eq!(e[0].size_of_prolog, 4, "SizeOfProlog");
    assert_eq!(e[0].count_of_codes, 2, "CountOfCodes");
    assert_eq!(e[0].frame_register, 0, "no frame register");
    assert_eq!(e[0].frame_offset, 0);
    assert_eq!(e[0].codes, [0x3000, 0x5205], "the code array, raw u16");
    assert_eq!(e[0].first_code(), Some(0), "UWOP_PUSH_NONVOL");

    // Entry 1: zero codes is legal (a function with no prolog to describe).
    assert_eq!(e[1].begin_rva, 0x1010);
    assert_eq!(e[1].codes, [], "CountOfCodes 0 stores no codes");
    assert_eq!(e[1].first_code(), None);

    // Entry 2: EHANDLER + frame pointer + SET_FPREG. The handler RVA behind
    // EHANDLER is NOT followed by this slice, and the reader says nothing
    // about it; the flags byte records that it exists.
    assert_eq!(e[2].flags, 1, "UNW_FLAG_EHANDLER of the 5-bit Flags field");
    assert_eq!(e[2].frame_register, 5, "RBP");
    assert_eq!(e[2].frame_offset, 2, "2 * 16 bytes");
    assert_eq!(e[2].codes, [0x4408]);
    assert_eq!(e[2].first_code(), Some(4), "UWOP_SET_FPREG");

    // The discovery source: (begin, end) pairs, in array order.
    let fns: Vec<(u32, u32)> = t.functions().collect();
    assert_eq!(
        fns,
        [(0x1000u32, 0x1010u32), (0x1010, 0x1020), (0x1020, 0x1040)],
        "functions() is the function-start source"
    );
    assert!(t.chaining_refusals().is_empty(), "nothing chained here");

    // The census: flags and first opcode, the slice's whole output — the
    // codes are counted, never interpreted.
    assert_eq!(t.flags_census(), vec![(0u8, 2), (1, 1)]);
    assert_eq!(
        t.first_code_census(),
        vec![(Some(0u8), 1), (Some(4), 1), (None, 1)],
        "None = CountOfCodes 0, counted, not dropped"
    );
}

// ---------------------------------------------------------------------------
// REFUSALS. Each is ONE patch of the good fixture at one quoted offset, and
// each asserts the refusal NAMES the field and the entry: a refusal that only
// says "bad .pdata" is a refusal nobody can act on with a hex editor.
// ---------------------------------------------------------------------------

/// The loader binary-searches this array; an unsorted array breaks the search,
/// and the refusal names the PAIR that is out of order, indices and values.
#[test]
fn an_unsorted_runtime_function_array_is_refused_naming_the_pair() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, RF1_BEGIN, 0x1030); // entry 1 now starts AFTER entry 2
    let e = refusal(&b);
    assert!(e.contains("sorted"), "the refusal says sortedness: {e}");
    assert!(
        e.contains("BeginAddress"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains("0x1030"), "and the late entry's begin: {e}");
    assert!(e.contains("0x1020"), "and the entry it jumped over: {e}");
}

#[test]
fn a_directory_size_that_is_not_a_multiple_of_12_is_refused() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, OFF_DIR3_SIZE, 37);
    let e = refusal(&b);
    assert!(
        e.contains("EXCEPTION"),
        "the refusal names the directory: {e}"
    );
    assert!(e.contains("12"), "and the entry stride: {e}");
    assert!(
        e.contains("37"),
        "and the size that leaves a remainder: {e}"
    );
}

#[test]
fn an_empty_exception_directory_is_refused_by_name() {
    // The symbol-free EXE that HAS .pdata is the point of this slice; the
    // image that does NOT (a DLL built without SEH data) refuses, naming
    // the directory, so "no functions found" is never silent.
    let mut b = pe32plus_with_pdata();
    put32(&mut b, OFF_DIR3_RVA, 0);
    put32(&mut b, OFF_DIR3_SIZE, 0);
    let e = refusal(&b);
    assert!(
        e.contains("EXCEPTION"),
        "the refusal names the directory: {e}"
    );
    assert!(
        e.contains("pdata"),
        "and the section it is the table of: {e}"
    );
}

/// A BeginAddress that names readable data, not code, means the array is not
/// describing functions: the refusal says which section it landed in.
#[test]
fn a_begin_address_outside_the_executable_sections_is_refused() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, RF2_BEGIN, 0x3000); // inside .pdata itself: readable, not executable
    let e = refusal(&b);
    assert!(
        e.contains("BeginAddress"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains(".pdata"), "and the section it landed in: {e}");
    assert!(
        e.contains("executab") || e.contains("code"),
        "and says it is not executable: {e}"
    );
}

#[test]
fn an_end_address_that_does_not_follow_its_begin_is_refused() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, RF2_END, 0x1020); // EndAddress == BeginAddress
    let e = refusal(&b);
    assert!(e.contains("EndAddress"), "the refusal names the field: {e}");
    assert!(
        e.contains("BeginAddress"),
        "and the field it is measured against: {e}"
    );
    assert!(e.contains("0x1020"), "and the value: {e}");
}

#[test]
fn an_unaligned_unwind_data_rva_is_refused() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, RF0_UNWIND, 0x4001);
    let e = refusal(&b);
    assert!(e.contains("UnwindData"), "the refusal names the field: {e}");
    assert!(e.contains("align"), "and the rule it broke: {e}");
    assert!(e.contains("0x1000"), "and the entry's BeginAddress: {e}");
}

/// 0x4200 is the raw end of `.xdata` (rawptr 0x600 + rawsize 0x200 = 0x800 =
/// the file's last byte), and `.xdata`'s VirtualSize 0x210 PROMISES 16 bytes
/// past it — the m7.exe `.data` shape, virtual bytes with no file behind
/// them. The 4-byte header read at 0x4200 is inside the declared extent and
/// inside no bytes the file has; the reader must refuse naming SizeOfRawData
/// and the file length, not slice short.
#[test]
fn an_unwind_data_rva_whose_header_runs_past_the_file_is_refused() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, RF2_UNWIND, 0x4200);
    let e = refusal(&b);
    assert!(e.contains("UnwindData"), "the refusal names the field: {e}");
    assert!(e.contains("past"), "and says the read runs off: {e}");
    assert!(
        e.contains("SizeOfRawData"),
        "and names the field that fell short: {e}"
    );
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the length of the file: {e}"
    );
}

/// CountOfCodes 255 on the LAST entry promises 510 code bytes starting at
/// 0x4014 — 4 bytes inside a section whose raw data has 508, whose VirtualSize
/// ends at 0x4210, and whose raw end IS the file end. The code array must be
/// BOUNDED, and the refusal names the field that promised the bytes — this is
/// the hostile-header case the bounds check exists for.
#[test]
fn a_code_array_that_outruns_its_section_is_refused() {
    let mut b = pe32plus_with_pdata();
    b[U2 + 2] = 255; // U2 CountOfCodes (byte 2 of the header): 1 -> 255
    let e = refusal(&b);
    assert!(
        e.contains("CountOfCodes"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains("255"), "and the value it found: {e}");
}

/// Version is a 3-bit field, `UNWIND_INFO_VERSION` is 1, and the Windows
/// loader's x64 walker is written against that layout. A version this reader
/// has not read the layout of is refused AS A FUNCTION, PER ENTRY: reading a
/// 2 as a 1 does not fail, it shifts every code byte into a plausible-looking
/// code array — so the entry's code array is never read. But the refusal is
/// that entry's property, not the table's: the RUNTIME_FUNCTION record
/// (Begin/End/UnwindData) is version-independent, and the siblings parse.
#[test]
fn an_unwind_info_of_version_two_is_refused_per_entry_counted_not_dropped() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, U1, 0x0000_0002); // entry 1's header: Version 2
    let img = PeImage::load(&b, "fixture.exe").expect("the fixture image parses");
    let t = UnwindTable::read(&img).expect("one v2 entry refuses itself, not the table");
    let e = t.entries();
    assert_eq!(e.len(), 2, "the v2 entry is not a function start");
    assert_eq!(
        t.functions().count(),
        2,
        "functions() must not yield the v2 entry"
    );
    let v = t.version_refusals();
    assert_eq!(v.len(), 1, "counted, not silently dropped, not table-wide");
    assert_eq!(v[0].begin_rva, 0x1010, "named by BeginAddress");
    assert_eq!(v[0].index, 1, "named by array index");
    assert_eq!(
        v[0].version, 2,
        "the version found is carried, not flattened"
    );
    let m = v[0].message(&img);
    assert!(m.contains("Version"), "the refusal names the field: {m}");
    assert!(m.contains("0x1010"), "and the entry's BeginAddress: {m}");
    assert!(m.contains("2"), "the version found: {m}");
    assert!(m.contains("1"), "the version wanted: {m}");
    // The census cannot lie in either direction: functions + chaining +
    // version refusals == the directory's declared count.
    assert_eq!(
        t.functions().count() + t.chaining_refusals().len() + t.version_refusals().len(),
        t.entry_count(),
        "the census partitions the directory exactly"
    );
}

/// UNW_FLAG_CHAINING is the per-entry REFUSAL that must be COUNTED, loudly.
/// A chained entry's BeginAddress is not a function start — it is a nested
/// region of the outer function (a /EHsc scope), so handing it to the
/// discovery source invents functions, and silently dropping it invents
/// gaps. It is refused, named, counted, and never dropped.
#[test]
fn a_chained_entry_is_refused_by_name_counted_not_dropped() {
    let mut b = pe32plus_with_pdata();
    put32(&mut b, U2, 0x2501_0821); // entry 2's flags: EHANDLER -> CHAINING (field bit 0x4)
    let t = table(&b); // the TABLE parses: a chained entry refuses ITSELF, not the image
    let e = t.entries();
    assert_eq!(e.len(), 2, "the chained entry is not a function");
    assert_eq!(
        t.functions().count(),
        2,
        "functions() must not yield the chained entry"
    );
    let c = t.chaining_refusals();
    assert_eq!(c.len(), 1, "counted, not silently dropped");
    assert_eq!(c[0].begin_rva, 0x1020, "named by BeginAddress");
    assert_eq!(c[0].index, 2, "named by array index");
    let m = c[0].message();
    assert!(
        m.contains("UNW_FLAG_CHAINING"),
        "the refusal names the flag: {m}"
    );
    assert!(m.contains("0x1020"), "and the entry: {m}");
    // The census sees the two parsed entries only — and the refusal count is
    // reported beside it, so 'N functions' can never quietly shrink.
    assert_eq!(t.flags_census(), vec![(0u8, 2)]);
}

// ---------------------------------------------------------------------------
// THE REAL IMAGE. m7.exe is the M7 probe artifact (clang -O1 + MSVC link,
// 111104 bytes, NOT in the repo): this leg SKIPS LOUDLY when absent, like
// pe_fixtures.rs does. What it measures is QUOTED, not aspired to, and the
// two facts that broke the slice's assumptions are pinned in the open:
// (1) the linked image carries NO COFF symbol table — `llvm-nm m7.exe`
//     prints `no symbols` — so .pdata is not a supplement to symbols, it is
//     the ONLY function-start source, which is the stripped case this slice
//     exists for; and (2) FOUR entries of the 336 carry UNWIND_INFO
//     Version 2 (the newer CET-hardened CRT's 16-byte guard-check thunks,
//     sharing unwind info in .rdata), which the version rule refuses by name
//     at the first one. The parity leg therefore pins the refusal VERBATIM
//     and cross-checks every claim of it with an independent walk: the
//     refusal is only honest if entry 290's byte really is 0x02.
// ---------------------------------------------------------------------------

fn scratch_m7() -> Option<std::path::PathBuf> {
    let root = std::env::var("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .ok()?;
    let p = root.join("hermes/cache/scratch/m7/m7.exe");
    p.is_file().then_some(p)
}

fn u32at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The independent walk the pin is checked against: raw offsets, no reader.
/// It duplicates the reading ON PURPOSE — a parity leg that calls the code
/// under test verifies nothing about the refusal's truthfulness.
fn m7_pdata_facts(bytes: &[u8]) -> (Vec<(u32, u32, u32)>, Vec<u32>) {
    let pe = u32at(bytes, 0x3c) as usize;
    let sizeopt = u16_at(bytes, pe + 20) as usize;
    let opt = pe + 24;
    let dir_rva = u32at(bytes, opt + 112 + 3 * 8);
    let dir_size = u32at(bytes, opt + 112 + 3 * 8 + 4);
    let nsec = u16_at(bytes, pe + 6) as usize;
    let tbl = opt + sizeopt;
    let mut secs: Vec<(u32, u32, u32, u32)> = Vec::new(); // (va, vsize, rawsize, rawptr)
    for i in 0..nsec {
        let o = tbl + 40 * i;
        secs.push((
            u32at(bytes, o + 12),
            u32at(bytes, o + 8),
            u32at(bytes, o + 16),
            u32at(bytes, o + 20),
        ));
    }
    let n = (dir_size / 12) as usize;
    let mut rf = Vec::with_capacity(n);
    let mut versions = Vec::with_capacity(n);
    for i in 0..n {
        let at = va2off_secs(&secs, dir_rva) + 12 * i;
        let (bg, en, ud) = (u32at(bytes, at), u32at(bytes, at + 4), u32at(bytes, at + 8));
        let ud_off = va2off_secs(&secs, ud);
        rf.push((bg, en, ud));
        versions.push(u32at(bytes, ud_off) & 7);
    }
    (rf, versions)
}

fn va2off_secs(secs: &[(u32, u32, u32, u32)], rva: u32) -> usize {
    for &(va, vs, rs, rp) in secs {
        let span = std::cmp::max(vs, rs) as u64;
        if u64::from(rva) >= u64::from(va) && u64::from(rva) < u64::from(va) + span {
            return rp as usize + (rva - va) as usize;
        }
    }
    panic!("walk: rva {rva:#x} unmapped")
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

#[test]
fn the_real_exe_pdata_matches_the_refusal_the_reader_prints() {
    let Some(path) = scratch_m7() else {
        println!("SKIP(real-pdata): m7.exe is not present at %LOCALAPPDATA%/hermes/cache/scratch/m7/m7.exe — the real-image .pdata parity leg did not run");
        return;
    };
    let bytes = std::fs::read(&path).expect("m7.exe reads");
    let img = PeImage::load(&bytes, "m7.exe").expect("the real linked EXE parses as PE32+");

    // Ground facts of the table, from the independent walk.
    let (rf, versions) = m7_pdata_facts(&bytes);
    assert_eq!(rf.len(), 336, "EXCEPTION dir 0xfc0 / 12 == 336 entries");
    assert!(
        rf.windows(2).all(|w| w[0].0 < w[1].0),
        "the real table is sorted (the loader binary-searches it)"
    );
    let text = img.section_named(".text").expect(".text");
    assert!(
        rf.iter().all(|(bg, _, _)| text.contains(*bg)),
        "every BeginAddress lies inside .text"
    );
    assert!(
        rf.iter().all(|(bg, en, _)| en > bg),
        "every EndAddress follows its BeginAddress"
    );
    assert!(
        rf.iter().all(|(_, _, ud)| ud % 4 == 0),
        "every UnwindData is 4-byte aligned"
    );
    let v2: Vec<usize> = versions
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 1)
        .map(|(i, _)| i)
        .collect();
    let chained: Vec<usize> = rf
        .iter()
        .enumerate()
        .filter(|(_, (_, _, ud))| {
            // Chaining = Flags field (bits 3-7) bit 0x4 -> u32 bit 0x20.
            let o = img.va_to_off(*ud).expect("unwind rva maps");
            bytes[o] & 0x20 != 0
        })
        .map(|(i, _)| i)
        .collect();
    println!(
        "REAL m7.exe .pdata: {} entries, sorted, all begins in .text, versions: {} x v1, {} x v2 (entries {:?}), chained: {} (entries {:?})",
        rf.len(),
        rf.len() - v2.len() - chained.len(),
        v2.len(),
        v2.iter().map(|&i| format!("{:#x}", rf[i].0)).collect::<Vec<_>>(),
        chained.len(),
        chained.iter().map(|&i| format!("{:#x}", rf[i].0)).collect::<Vec<_>>(),
    );
    assert_eq!(
        v2.len(),
        4,
        "the pinned finding: FOUR version-2 entries (CET guard-thunk CRT objects)"
    );
    assert_eq!(v2[0], 290, "the first is array index 290");
    assert_eq!(rf[290].0, 0xe8e0, "at BeginAddress 0xe8e0");

    // The reader's answer: the table READS. Its per-entry v2 refusals must be
    // exactly the independent byte-walk's v2 index set, its chaining refusals
    // the walk's chained set, and the census must partition the directory —
    // the parse may drop precisely the entries the raw bytes mark, no others.
    let t = UnwindTable::read(&img)
        .expect("m7.exe .pdata reads: four v2 entries refuse themselves, not the image");
    let refs: Vec<usize> = t.version_refusals().iter().map(|v| v.index).collect();
    assert_eq!(
        refs, v2,
        "the reader's v2 refusals == the independent walk's v2 indices, in order"
    );
    let crefs: Vec<usize> = t.chaining_refusals().iter().map(|c| c.index).collect();
    assert_eq!(crefs, chained, "chaining refusals == the walk's");
    assert_eq!(
        t.entry_count(),
        rf.len(),
        "declared directory count == the walk's entry count"
    );
    assert_eq!(
        t.functions().count() + refs.len() + crefs.len(),
        t.entry_count(),
        "the census partitions the directory exactly: no entry double-counted, none lost"
    );
    for v in t.version_refusals() {
        let m = v.message(&img);
        assert!(m.contains("Version"), "names the field: {m}");
        assert!(m.contains("2"), "names the version found: {m}");
    }
    println!(
        "REAL m7.exe pe-functions floor: {} functions, {} chaining refusals, {} version refusals (indices {:?})",
        t.functions().count(),
        crefs.len(),
        refs.len(),
        refs
    );

    // The linked image is SYMBOL-FREE: `llvm-nm m7.exe` says `no symbols`.
    // Pinned, because it is the premise of the slice: there is no symbol
    // table to cross-check, .pdata IS the function-start source.
    let out = std::process::Command::new("llvm-nm").arg(&path).output();
    let Ok(o) = out else {
        println!(
            "SKIP(real-pdata): llvm-nm is not on PATH — the symbol-free premise leg did not run"
        );
        return;
    };
    let text_nm = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    println!("REAL m7.exe llvm-nm, verbatim: {}", text_nm.trim());
    assert!(
        text_nm.contains("no symbols"),
        "the probe artifact's premise: the linked image carries no COFF symbol table. llvm-nm said: {text_nm}"
    );

    // The one named-address parity available to a symbol-free image: its own
    // entry point (mainCRTStartup, RVA 0x1280 from the pe_fixtures parity leg)
    // lies INSIDE a table entry, and that entry's BeginAddress is the RVA
    // itself — clang+MSVC-link keeps function starts exact.
    let covering: Vec<(u32, u32)> = rf
        .iter()
        .filter(|(bg, en, _)| *bg <= img.entry_rva && img.entry_rva < *en)
        .map(|(bg, en, _)| (*bg, *en))
        .collect();
    assert_eq!(
        covering,
        vec![(0x1280, 0x1292)],
        "the entry point 0x1280 is covered by a table entry whose BeginAddress is exactly 0x1280"
    );
}
