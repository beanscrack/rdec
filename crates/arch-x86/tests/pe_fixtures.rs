//! PE32+ IMAGE bytes, hand-built: the header layout IS the claim.
//!
//! WHY THIS FILE EXISTS. `rdec discover m7.exe` — a real linked console EXE —
//! prints `section 1 (?) has SizeOfRawData 1750344141 at PointerToRawData
//! 1881174889, past the end of a 111104-byte file`: the COFF *object* reader in
//! `loader.rs` reads the PE image header as a section table and refuses with
//! exact bytes. That refusal is correct — there was no image support — and the
//! fix is an image reader beside it, not a looser object reader. The
//! object reader's own API is untouched by everything below.
//!
//! THE BYTES ARE HERMETIC, and that is the point (the lesson of
//! `jump_table_fixtures.rs`: a test that reads a file the CI runner builds is
//! asserting which toolchain ran, not what the code does). Every field of the
//! good fixture is written by `pe32plus()` at the offset quoted in the table
//! below, so each malformed sibling is ONE patch of one known offset — the
//! refusal under test is then attributable to that one field and nothing else.
//!
//! THE GOOD FIXTURE — every row is (file offset, size, value, field). All
//! little-endian. Offsets are absolute file offsets; `pe` = `e_lfanew` = 0x40.
//!
//! ```text
//! 0x00  2  0x5a4d        e_magic            "MZ"
//! 0x3c  4  0x40          e_lfanew           -> the PE header
//! --- pe + 0x00 ---------------------------------------------------------------
//! 0x40  4  "PE\0\0"      PE signature
//! 0x44  2  0x8664        Machine            IMAGE_FILE_MACHINE_AMD64
//! 0x46  2  2             NumberOfSections
//! 0x48  4  0             TimeDateStamp
//! 0x4c  4  0             PointerToSymbolTable   (images: unused)
//! 0x50  4  0             NumberOfSymbols        (images: unused)
//! 0x54  2  240           SizeOfOptionalHeader   (0xf0 = 112 + 16*8)
//! 0x56  2  0x0022        Characteristics        EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
//! --- pe + 0x18: the optional header, magic 0x20b = PE32+ ---------------------
//! 0x58  2  0x20b         Magic              PE32+ (0x10b would be PE32, 32-bit)
//! 0x5a  1  14            MajorLinkerVersion     0x5b 1 MinorLinkerVersion
//! 0x5c  4  0x1000        SizeOfCode
//! 0x60  4  0x200         SizeOfInitializedData
//! 0x64  4  0             SizeOfUninitializedData
//! 0x68  4  0x1030        AddressOfEntryPoint    (an RVA: .text + 0x30)
//! 0x6c  4  0x1000        BaseOfCode
//!                            (PE32+ has NO BaseOfData — that field is PE32-only,
//!                             which is one reason the two magics are not
//!                             interchangeable and 0x10b is refused by name)
//! 0x70  8  0x140000000   ImageBase
//! 0x78  4  0x1000        SectionAlignment
//! 0x7c  4  0x200         FileAlignment
//! 0x80  2  6             MajorOperatingSystemVersion
//! 0x82  2  0             MinorOperatingSystemVersion
//! 0x84  2  0             MajorImageVersion
//! 0x86  2  0             MinorImageVersion
//! 0x88  2  6             MajorSubsystemVersion
//! 0x8a  2  0             MinorSubsystemVersion
//! 0x8c  4  0             Win32Version
//! 0x90  4  0x3000        SizeOfImage
//! 0x94  4  0x200         SizeOfHeaders
//! 0x98  4  0             CheckSum
//! 0x9c  2  3             Subsystem            IMAGE_SUBSYSTEM_WINDOWS_CUI
//! 0x9e  2  0x0160        DllCharacteristics   HIGH_ENTROPY_VA | DYNAMIC_BASE | NX_COMPAT
//! 0xa0  8  0x100000      SizeOfStackReserve
//! 0xa8  8  0x1000        SizeOfStackCommit
//! 0xb0  8  0x100000      SizeOfHeapReserve
//! 0xb8  8  0x1000        SizeOfHeapCommit
//! 0xc0  4  0             LoaderFlags
//! 0xc4  4  16            NumberOfRvaAndSizes
//! 0xc8  16*8 zero        DataDirectory[0..16]  (ends at 0x148)
//! --- 0x148: the section table, 40 bytes per entry ----------------------------
//! 0x148 8  ".text\0\0\0" Name
//! 0x150 4  0x40          VirtualSize          (bytes of the section that are alive)
//! 0x154 4  0x1000        VirtualAddress       (RVA)
//! 0x158 4  0x200         SizeOfRawData        (FileAlignment-rounded)
//! 0x15c 4  0x200         PointerToRawData     (file offset)
//! 0x160 4  0             PointerToRelocations (must be 0 in an image)
//! 0x164 4  0             PointerToLinenumbers
//! 0x168 2  0             NumberOfRelocations
//! 0x16a 2  0             NumberOfLinenumbers
//! 0x16c 4  0x60000020    Characteristics      CNT_CODE | MEM_EXECUTE | MEM_READ
//! 0x170 8  ".rdata\0\0\0" Name
//! 0x178 4  0x200         VirtualSize          (== SizeOfRawData: fully backed)
//! 0x17c 4  0x2000        VirtualAddress
//! 0x180 4  0x200         SizeOfRawData
//! 0x184 4  0x400         PointerToRawData     (the section ends AT the last byte
//!                                             of the file: 0x400+0x200 == 0x600)
//! 0x194 4  0x40000040    Characteristics      CNT_INITIALIZED_DATA | MEM_READ
//! --- 0x200: .text content. 0x30 of 0xcc, then the entry point, then 0xcc -----
//! 0x230 6  b8 2a 00 00 00 c3   mov $0x2a,%eax ; ret   (AddressOfEntryPoint)
//! --- 0x400: .rdata content ---------------------------------------------------
//! 0x400 4  de ad be ef   the first four bytes of read-only data
//! total 0x600 (1536) bytes of file
//! ```
//!
//! The `.rdata` tail is at the file's last byte on purpose: `read_at` crossing
//! the end of the file is the one edge a section-header check cannot catch on
//! its own, and it has to refuse rather than slice.

use rdec_arch_x86::pe::PeImage;

// --- the fixture's own offsets, so a malformed sibling is ONE named patch ----
const OFF_E_MAGIC: usize = 0x00;
const OFF_E_LFANEW: usize = 0x3c;
const OFF_PE_SIG: usize = 0x40;
const OFF_MACHINE: usize = 0x44;
const OFF_NSECTIONS: usize = 0x46;
const OFF_SIZEOF_OPT_HDR: usize = 0x54;
const OFF_OPT_MAGIC: usize = 0x58;
const OFF_ENTRY: usize = 0x68;
const OFF_IMAGE_BASE: usize = 0x70;
const OFF_SIZEOF_IMAGE: usize = 0x90;
const OFF_SUBSYSTEM: usize = 0x9c;
const TEXT_VSIZE: usize = 0x150;
const TEXT_VADDR: usize = 0x154;
const TEXT_RAW_SIZE: usize = 0x158;
const TEXT_RAW_PTR: usize = 0x15c;
const RDATA_RAW_PTR: usize = 0x184;
const TEXT_RVA: u32 = 0x1000;
const RDATA_RVA: u32 = 0x2000;
const ENTRY_RVA: u32 = 0x1030;
const FILE_LEN: usize = 0x600;
/// `mov $0x2a,%eax ; ret` at AddressOfEntryPoint.
const ENTRY_BYTES: [u8; 6] = [0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3];

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// The good fixture, laid out exactly as the table above quotes it.
fn pe32plus() -> Vec<u8> {
    let mut b = vec![0u8; FILE_LEN];
    put16(&mut b, OFF_E_MAGIC, 0x5a4d); // "MZ"
    put32(&mut b, OFF_E_LFANEW, 0x40); // e_lfanew
    b[OFF_PE_SIG..OFF_PE_SIG + 4].copy_from_slice(b"PE\0\0");
    put16(&mut b, OFF_MACHINE, 0x8664);
    put16(&mut b, OFF_NSECTIONS, 2);
    put16(&mut b, OFF_SIZEOF_OPT_HDR, 240);
    put16(&mut b, 0x56, 0x0022); // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
    put16(&mut b, OFF_OPT_MAGIC, 0x20b); // PE32+
    b[0x5a] = 14; // MajorLinkerVersion
    put32(&mut b, 0x5c, 0x1000); // SizeOfCode
    put32(&mut b, 0x60, 0x200); // SizeOfInitializedData
    put32(&mut b, OFF_ENTRY, ENTRY_RVA); // AddressOfEntryPoint
    put32(&mut b, 0x6c, 0x1000); // BaseOfCode
    put64(&mut b, OFF_IMAGE_BASE, 0x0001_4000_0000);
    put32(&mut b, 0x78, 0x1000); // SectionAlignment
    put32(&mut b, 0x7c, 0x200); // FileAlignment
    put16(&mut b, 0x80, 6); // MajorOperatingSystemVersion
    put16(&mut b, 0x88, 6); // MajorSubsystemVersion
    put32(&mut b, OFF_SIZEOF_IMAGE, 0x3000);
    put32(&mut b, 0x94, 0x200); // SizeOfHeaders
    put16(&mut b, OFF_SUBSYSTEM, 3); // Windows CUI
    put16(&mut b, 0x9e, 0x0160); // DllCharacteristics
    put64(&mut b, 0xa0, 0x100000); // SizeOfStackReserve
    put64(&mut b, 0xa8, 0x1000); // SizeOfStackCommit
    put64(&mut b, 0xb0, 0x100000); // SizeOfHeapReserve
    put64(&mut b, 0xb8, 0x1000); // SizeOfHeapCommit
    put32(&mut b, 0xc4, 16); // NumberOfRvaAndSizes
                             // DataDirectory[0..16] stays zero: no imports, no exports, no reloc dir.
                             // Section 0: .text
    b[0x148..0x148 + 5].copy_from_slice(b".text");
    put32(&mut b, TEXT_VSIZE, 0x40);
    put32(&mut b, TEXT_VADDR, TEXT_RVA);
    put32(&mut b, TEXT_RAW_SIZE, 0x200);
    put32(&mut b, TEXT_RAW_PTR, 0x200);
    put32(&mut b, 0x16c, 0x6000_0020); // CODE | EXECUTE | READ
                                       // Section 1: .rdata, ending exactly at the last byte of the file
    b[0x170..0x170 + 6].copy_from_slice(b".rdata");
    put32(&mut b, 0x178, 0x200); // VirtualSize
    put32(&mut b, 0x17c, RDATA_RVA);
    put32(&mut b, 0x180, 0x200); // SizeOfRawData
    put32(&mut b, RDATA_RAW_PTR, 0x400); // PointerToRawData
    put32(&mut b, 0x194, 0x4000_0040); // Characteristics: INIT_DATA | READ
                                       // Content
    for i in 0..0x40usize {
        b[0x200 + i] = 0xcc;
    }
    b[0x200 + 0x30..0x200 + 0x30 + ENTRY_BYTES.len()].copy_from_slice(&ENTRY_BYTES);
    b[0x400..0x404].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    b
}

fn load(b: &[u8]) -> Result<PeImage<'_>, String> {
    PeImage::load(b, "fixture.exe")
}

fn refusal(b: &[u8]) -> String {
    load(b).expect_err("the fixture is malformed and must be refused")
}

// ---------------------------------------------------------------------------
// THE GOOD FIXTURE: every field the loader claims to read, read back.
// ---------------------------------------------------------------------------

#[test]
fn a_pe32_plus_image_reads_back_every_header_field() {
    let b = pe32plus();
    let img = load(&b).expect("the good fixture parses");
    assert_eq!(img.machine, 0x8664, "Machine");
    assert_eq!(img.image_base, 0x0001_4000_0000, "ImageBase");
    assert_eq!(img.entry_rva, ENTRY_RVA, "AddressOfEntryPoint");
    assert_eq!(img.entry_va(), 0x0001_4000_1030, "entry RVA + ImageBase");
    assert_eq!(img.size_of_image, 0x3000, "SizeOfImage");
    assert_eq!(img.subsystem, 3, "Subsystem");
    assert_eq!(img.subsystem_name(), "Windows CUI", "Subsystem by name");
    assert_eq!(img.sections.len(), 2, "NumberOfSections");

    let text = img.section_named(".text").expect("the .text section");
    assert_eq!(text.virtual_address, TEXT_RVA);
    assert_eq!(text.virtual_size, 0x40);
    assert_eq!(text.size_of_raw_data, 0x200);
    assert_eq!(text.pointer_to_raw_data, 0x200);
    assert_eq!(text.characteristics, 0x6000_0020);
    assert!(text.is_code(), "CNT_CODE | MEM_EXECUTE is code");
    // The flags summary is what `rdec pe-info` prints: raw 0x60000020 in,
    // names out, so the section table is readable without the spec open.
    assert_eq!(text.flags_summary(), "CODE|EXECUTE|READ");

    let rdata = img.section_named(".rdata").expect("the .rdata section");
    assert_eq!(rdata.virtual_address, RDATA_RVA);
    assert_eq!(rdata.virtual_size, 0x200);
    assert_eq!(rdata.pointer_to_raw_data, 0x400);
    assert_eq!(rdata.flags_summary(), "INIT_DATA|READ");
    assert!(!rdata.is_code(), "INIT_DATA is not CODE");
    assert!(
        !rdata.flags_summary().contains("EXECUTE"),
        "a read-only data section must not print EXECUTE: {}",
        rdata.flags_summary()
    );
    // The data directories: NumberOfRvaAndSizes 16 means sixteen slots exist in
    // the header, and this fixture fills none of them. The list is what M7's
    // import slice will read, so it is exposed as parsed entries, counted.
    assert_eq!(img.data_directories.len(), 16, "NumberOfRvaAndSizes 16");
    assert_eq!(
        img.data_directories[1].name, "IMPORT",
        "directory 1 is the imports"
    );
    assert_eq!(
        img.data_directories[1].rva, 0,
        "and this image declares none"
    );
}

/// COFF relocation tables are an OBJECT concept: a linked image is already
/// based, and its relocations live in the `.reloc` SECTION (m7.exe has one,
/// 0x64c bytes). An image carrying per-section COFF relocation tables is
/// lying about one of the two containers, and the loader says which field.
#[test]
fn an_image_with_a_coff_relocation_table_is_refused() {
    let mut b = pe32plus();
    put16(&mut b, 0x168, 1); // .text NumberOfRelocations
    let e = refusal(&b);
    assert!(
        e.contains("NumberOfRelocations"),
        "the refusal names the field: {e}"
    );
    assert!(
        e.contains(".reloc"),
        "and points at where images keep them: {e}"
    );
}

#[test]
fn an_image_with_no_sections_is_refused() {
    let mut b = pe32plus();
    put16(&mut b, OFF_NSECTIONS, 0);
    let e = refusal(&b);
    assert!(
        e.contains("NumberOfSections"),
        "the refusal names the field: {e}"
    );
}

#[test]
fn va_to_off_round_trips_through_the_section_table() {
    let b = pe32plus();
    let img = load(&b).expect("the good fixture parses");
    // Section starts, and the middle of each.
    assert_eq!(img.va_to_off(TEXT_RVA).expect("text start"), 0x200);
    assert_eq!(img.va_to_off(ENTRY_RVA).expect("entry"), 0x230);
    assert_eq!(img.va_to_off(RDATA_RVA).expect("rdata start"), 0x400);
    assert_eq!(img.va_to_off(RDATA_RVA + 0x1f).expect("rdata +0x1f"), 0x41f);
    // The last byte the section table says exists, and the first byte past it:
    // `.text` is VirtualSize 0x40, so RVA 0x103f is inside and 0x1040 is not.
    assert_eq!(img.va_to_off(TEXT_RVA + 0x3f).expect("text last"), 0x23f);
    let e = img
        .va_to_off(TEXT_RVA + 0x40)
        .expect_err("past .text's VirtualSize");
    assert!(e.contains("0x1040"), "the refusal names the RVA: {e}");
    assert!(
        e.contains("section"),
        "the refusal says what is missing: {e}"
    );
    // RVA 0 is the header space, not a section.
    let e = img.va_to_off(0).expect_err("the headers are not a section");
    assert!(e.contains("0x0"), "the refusal names the RVA: {e}");
}

#[test]
fn read_at_returns_the_backing_bytes_and_refuses_past_them() {
    let b = pe32plus();
    let img = load(&b).expect("the good fixture parses");
    assert_eq!(
        img.read_at(ENTRY_RVA, 6).expect("entry bytes"),
        &ENTRY_BYTES
    );
    assert_eq!(
        img.read_at(RDATA_RVA, 4).expect("rdata bytes"),
        &[0xde, 0xad, 0xbe, 0xef]
    );
    // .rdata's last raw byte is the file's last byte (0x400 + 0x200 == 0x600).
    // One byte AT the end is fine; two is a refusal, not a short slice, and not
    // a panic — this is the edge a hostile SizeOfRawData aims at.
    let last = RDATA_RVA + 0x1ff;
    assert_eq!(img.read_at(last, 1).expect("the last byte"), &[0u8]);
    let e = img.read_at(last, 2).expect_err("one byte past the file");
    assert!(
        e.contains("SizeOfRawData") || e.contains("PointerToRawData"),
        "the refusal names the field that promised the bytes: {e}"
    );
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the file length: {e}"
    );
    // A read that outruns the section's VirtualSize refuses on the size, and
    // says so, even when the raw bytes are sitting there (.text has 0x200 raw
    // bytes and 0x40 live ones).
    let e = img
        .read_at(ENTRY_RVA, 0x40)
        .expect_err("a read that outruns .text's VirtualSize");
    assert!(
        e.contains("VirtualSize"),
        "the refusal names the field: {e}"
    );
}

// ---------------------------------------------------------------------------
// MALFORMED SIBLINGS. Each is ONE patch of the good fixture at one quoted
// offset, and each asserts that the refusal NAMES the field: a refusal that
// only says "bad file" is a refusal nobody can act on.
// ---------------------------------------------------------------------------

#[test]
fn bytes_that_are_not_mz_are_refused_by_name() {
    let mut b = pe32plus();
    b[OFF_E_MAGIC] = b'N'; // "NZ"
    let e = refusal(&b);
    assert!(
        e.contains("MZ"),
        "the refusal names the magic it wanted: {e}"
    );
    assert!(e.contains("0x00"), "and the offset: {e}");
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the length of what it got: {e}"
    );
}

/// The probe's own result, inverted: `rdec discover m7.exe` reads a PE image
/// with the object reader and refuses; this reads an OBJECT with the image
/// reader and refuses the same way, naming the same field. The two readers
/// meeting in the middle is what stops a container being silently misparsed.
#[test]
fn a_coff_object_is_refused_by_the_image_reader() {
    // 64 bytes of a COFF object (file header + one section header): long
    // enough that the DOS-header length check passes, so the refusal under
    // test is the MZ magic and not the size.
    let mut o = vec![0u8; 64];
    o[0..2].copy_from_slice(&0x8664u16.to_le_bytes());
    o[2..4].copy_from_slice(&1u16.to_le_bytes());
    let e = refusal(&o);
    assert!(e.contains("MZ"), "the refusal says it is not an image: {e}");
    assert!(e.contains("64 86"), "and quotes the bytes it got: {e}");
}

#[test]
fn a_file_shorter_than_the_dos_header_is_refused_with_its_length() {
    let b = vec![b'M', b'Z'];
    let e = refusal(&b);
    assert!(
        e.contains("64"),
        "the refusal says how many bytes a DOS header needs: {e}"
    );
    assert!(e.contains("2"), "and how many were there: {e}");
}

#[test]
fn a_pe_signature_that_is_not_pe_nul_nul_is_refused_at_its_offset() {
    let mut b = pe32plus();
    b[OFF_PE_SIG + 2] = b'Q'; // "PEQ\0"
    let e = refusal(&b);
    assert!(
        e.contains("signature"),
        "the refusal names the PE signature: {e}"
    );
    assert!(e.contains("0x40"), "and its offset: {e}");
}

#[test]
fn a_machine_that_is_not_amd64_is_refused_by_number() {
    let mut b = pe32plus();
    put16(&mut b, OFF_MACHINE, 0x014c); // IMAGE_FILE_MACHINE_I386
    let e = refusal(&b);
    assert!(e.contains("Machine"), "the refusal names the field: {e}");
    assert!(e.contains("0x14c"), "the value it got: {e}");
    assert!(e.contains("0x8664"), "the value it wanted: {e}");
}

/// The 32-bit image, refused BY NAME. Treating 0x10b as 0x20b is not a
/// simplification: PE32 has no 8-byte ImageBase and HAS a BaseOfData, so every
/// field from 0x70 onward is 4 bytes off, and the loader would read a
/// plausible-looking image base out of SectionAlignment and FileAlignment.
#[test]
fn a_pe32_image_is_refused_by_name_not_read_as_pe32_plus() {
    let mut b = pe32plus();
    put16(&mut b, OFF_OPT_MAGIC, 0x10b); // PE32, 32-bit
    let e = refusal(&b);
    assert!(
        e.contains("0x10b"),
        "the refusal names the magic found: {e}"
    );
    assert!(e.contains("PE32"), "and says which format that is: {e}");
    assert!(e.contains("0x20b"), "and the one this reader handles: {e}");
}

#[test]
fn an_e_lfanew_past_the_end_of_the_file_is_refused_with_both_numbers() {
    let mut b = pe32plus();
    put32(&mut b, OFF_E_LFANEW, 0x1000_0000);
    let e = refusal(&b);
    assert!(
        e.contains("e_lfanew"),
        "the refusal names the DOS-header field: {e}"
    );
    assert!(e.contains("0x10000000"), "the value it got: {e}");
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "the length the value has to fit: {e}"
    );
}

#[test]
fn an_e_lfanew_inside_the_dos_header_is_refused() {
    let mut b = pe32plus();
    put32(&mut b, OFF_E_LFANEW, 0x10);
    let e = refusal(&b);
    assert!(e.contains("e_lfanew"), "the refusal names the field: {e}");
    assert!(
        e.contains("DOS"),
        "and says where the value points (inside the header that stores it): {e}"
    );
}

#[test]
fn a_section_table_that_runs_past_the_file_is_refused() {
    // One patch: a bigger optional header pushes the section table off the end.
    let mut b = pe32plus();
    put16(&mut b, OFF_SIZEOF_OPT_HDR, 0x2000);
    let e = refusal(&b);
    assert!(
        e.contains("SizeOfOptionalHeader"),
        "the refusal names the field that moved the table: {e}"
    );
    assert!(e.contains("section table"), "and what no longer fits: {e}");
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the file length it is measured against: {e}"
    );
}

/// `NumberOfSections` is a u16, so 0xFFFF is the widest table a legal field
/// value can ask for: 4,194,300 bytes. Checked arithmetic, not a slice index —
/// `40 * nsec` in a usize on a 64-bit target cannot overflow, and the
/// allocation in front of the loop must not be sized by the field either.
#[test]
fn a_section_count_of_0xffff_is_refused_before_anything_is_allocated() {
    let mut b = pe32plus();
    put16(&mut b, OFF_NSECTIONS, 0xffff);
    let e = refusal(&b);
    assert!(
        e.contains("NumberOfSections"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains("65535"), "the count it got: {e}");
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the file that cannot hold it: {e}"
    );
}

#[test]
fn raw_data_promised_past_the_end_of_the_file_is_refused() {
    let mut b = pe32plus();
    put32(&mut b, TEXT_RAW_PTR, 0x1000); // 0x1000 + 0x200 > 0x600
    let e = refusal(&b);
    assert!(
        e.contains("PointerToRawData"),
        "the refusal names the field: {e}"
    );
    assert!(
        e.contains("SizeOfRawData"),
        "and the size that made it run off: {e}"
    );
    assert!(
        e.contains(&FILE_LEN.to_string()),
        "and the file length: {e}"
    );
    assert!(e.contains(".text"), "and which section: {e}");
}

#[test]
fn a_pointer_to_raw_data_of_zero_with_bytes_to_read_is_refused() {
    // PE's "no raw data" convention is SizeOfRawData == 0 AND PointerToRawData
    // == 0 (an all-zero .bss). A nonzero size with a zero pointer promises
    // bytes at file offset 0 — the DOS header — which is the exact misread
    // `loader.rs`'s `.bss` test documents for objects.
    let mut b = pe32plus();
    put32(&mut b, TEXT_RAW_PTR, 0);
    let e = refusal(&b);
    assert!(
        e.contains("PointerToRawData"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains(".text"), "and the section: {e}");
}

/// VirtualSize 0xFFFFFFFF at a high VirtualAddress: `rva + size` does not fit
/// in the 32-bit RVA space at all, so the section's extent is not a number this
/// format can express. Checked arithmetic, and the refusal says "overflow".
#[test]
fn a_section_extent_that_overflows_the_rva_space_is_refused() {
    let mut b = pe32plus();
    put32(&mut b, TEXT_VSIZE, 0xffff_ffff);
    let e = refusal(&b);
    assert!(
        e.contains("VirtualSize"),
        "the refusal names the field: {e}"
    );
    assert!(
        e.contains("verflow"),
        "and says the arithmetic overflowed: {e}"
    );
    assert!(e.contains(".text"), "and the section: {e}");
}

#[test]
fn a_section_past_size_of_image_is_refused() {
    // No overflow, just a section the declared image size does not cover:
    // 0x1000 + 0x8000 > SizeOfImage 0x3000.
    let mut b = pe32plus();
    put32(&mut b, TEXT_VSIZE, 0x8000);
    let e = refusal(&b);
    assert!(
        e.contains("SizeOfImage"),
        "the refusal names the field the section is measured against: {e}"
    );
    assert!(e.contains(".text"), "and the section: {e}");
}

#[test]
fn an_optional_header_too_small_to_hold_the_fields_is_refused() {
    let mut b = pe32plus();
    put16(&mut b, OFF_SIZEOF_OPT_HDR, 96);
    let e = refusal(&b);
    assert!(
        e.contains("SizeOfOptionalHeader"),
        "the refusal names the field: {e}"
    );
    assert!(e.contains("112"), "and the minimum it needs: {e}");
}

#[test]
fn overlapping_sections_are_refused_by_name() {
    // .rdata moved on top of .text: two sections claiming one RVA means
    // `va_to_off` has two answers, and the loader picking one silently is how a
    // decompiler reads a byte that the image does not have at that address.
    let mut b = pe32plus();
    put32(&mut b, 0x17c, TEXT_RVA);
    let e = refusal(&b);
    assert!(e.contains(".rdata"), "the refusal names the section: {e}");
    assert!(e.contains(".text"), "and the one it overlaps: {e}");
}

// ---------------------------------------------------------------------------
// THE REAL IMAGE. `m7.exe` is the M7 probe's artifact: `clang -O1 -c main.c` +
// `corpus/src/arith_basic.c`, linked with the MSVC link (ROADMAP, "M7 probe").
// It is NOT in the repo, so this leg SKIPS LOUDLY — a printed line, never a
// silent pass — when the file is not on the machine, and cross-checks the
// section table against `llvm-objdump -h` when it is.
// ---------------------------------------------------------------------------

fn scratch_m7() -> Option<std::path::PathBuf> {
    let root = std::env::var("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .ok()?;
    let p = root.join("hermes/cache/scratch/m7/m7.exe");
    p.is_file().then_some(p)
}

#[test]
fn the_real_linked_exe_matches_llvm_objdump() {
    let Some(path) = scratch_m7() else {
        println!("SKIP(real-pe): m7.exe is not present at %LOCALAPPDATA%/hermes/cache/scratch/m7/m7.exe — the real-image parity leg did not run");
        return;
    };
    let bytes = std::fs::read(&path).expect("m7.exe reads");
    let img = PeImage::load(&bytes, "m7.exe").expect("the real linked EXE parses as PE32+");

    // Ground truth, quoted from the M7 probe and re-measured here.
    assert_eq!(bytes.len(), 111_104, "the probe artifact is 111104 bytes");
    assert_eq!(img.image_base, 0x0001_4000_0000, "ImageBase");
    assert_eq!(img.entry_rva, 0x1280, "AddressOfEntryPoint");
    assert_eq!(img.size_of_image, 0x20000, "SizeOfImage");
    assert_eq!(img.subsystem, 3, "Subsystem 3 == Windows CUI");
    for want in [".text", ".pdata", ".reloc"] {
        assert!(
            img.section_named(want).is_some(),
            "the probe lists {want}: sections {:?}",
            img.sections
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
        );
    }
    // The .text VMA the probe measured, derived, not hardcoded twice.
    assert_eq!(
        img.image_base + img.section_named(".text").expect(".text").virtual_address as u64,
        0x0001_4000_1000,
        ".text at VMA 0x140001000"
    );

    // Cross-check against the tool, not against this test's own expectations.
    let out = std::process::Command::new("llvm-objdump")
        .args(["-h", &path.display().to_string()])
        .output();
    let Ok(o) = out else {
        println!("SKIP(real-pe): llvm-objdump is not on PATH — the section table was checked against the probe's numbers only");
        return;
    };
    assert!(o.status.success(), "llvm-objdump -h ran: {o:?}");
    let text = String::from_utf8_lossy(&o.stdout);
    let mut tool: Vec<(String, u64, u64)> = Vec::new();
    for line in text.lines() {
        // "  0 .text         0000eff7 0000000140001000 TEXT"
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() == 5 && f[0].parse::<usize>().is_ok() {
            tool.push((
                f[1].to_string(),
                u64::from_str_radix(f[2], 16).expect("objdump size"),
                u64::from_str_radix(f[3], 16).expect("objdump VMA"),
            ));
        }
    }
    assert!(
        !tool.is_empty(),
        "llvm-objdump -h printed no section rows:\n{text}"
    );
    assert_eq!(
        tool.len(),
        img.sections.len(),
        "llvm-objdump lists {} sections, the loader lists {}",
        tool.len(),
        img.sections.len()
    );
    for (i, (name, size, vma)) in tool.iter().enumerate() {
        let s = &img.sections[i];
        assert_eq!(&s.name, name, "section {i} name");
        // `llvm-objdump -h`'s `Size` column is NOT `VirtualSize`: LLVM's COFF
        // reader reports `min(VirtualSize, SizeOfRawData)` (its `getSize`).
        // Measured on m7.exe: `.text` prints 0xeff7 (its VirtualSize; raw is
        // 0xf000) and `.data` prints 0xc00 (its SizeOfRawData; VirtualSize is
        // 0x1b60 - the BSS tail has no bytes in the file). Comparing against
        // VirtualSize alone would call the tool's own convention a loader bug.
        let tool_size = std::cmp::min(s.virtual_size, s.size_of_raw_data) as u64;
        assert_eq!(
            tool_size, *size,
            "{name} min(VirtualSize, SizeOfRawData) vs objdump Size (vsize {:#x}, rawsize {:#x})",
            s.virtual_size, s.size_of_raw_data
        );
        assert_eq!(
            img.image_base + s.virtual_address as u64,
            *vma,
            "{name} VMA = ImageBase + VirtualAddress"
        );
        // Every byte the tool prints as the section's range has to be reachable.
        let off = img
            .va_to_off(s.virtual_address)
            .expect("section start maps");
        assert_eq!(
            off, s.pointer_to_raw_data as usize,
            "{name} maps to its raw start"
        );
    }
    // The entry point is a real byte of the image, reached through the mapping.
    let e = img.read_at(img.entry_rva, 1).expect("the entry byte maps");
    assert!(!e.is_empty(), "entry RVA 0x{:x}", img.entry_rva);
}
