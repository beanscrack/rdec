//! Crashes the fuzz surface found, each minimised to the SMALLEST object that
//! reproduces it - bytes as constants, never compiled at test time (CI's clang
//! is 20.1.8, local is 22.1.8, and this repo has been burned by
//! lowering-specific tests).
//!
//! Every fixture here was found by `rdec fuzz` / the bounded CI fuzz in
//! `tests/fuzz.rs`, minimised by hand, and pinned. The RED DIRECTION of each is
//! proven the same way every time: put the bug back (delete the guard the
//! comment names), watch the named test fail with the named message, restore,
//! watch it pass. The sabotage table lives in the commit message.
//!
//! The crash each fixture pins, as a list (a markdown table in a doc comment
//! trips `clippy::doc_lazy_continuation` under `-D warnings`):
//!
//! - `bss_pointer_to_raw_data_zero` - SILENT MISREAD: `.bss` handed back the
//!   file header's bytes as its content (82c285f's family, live again).
//! - `reloc_table_past_end_of_file` - range panic inside `loader::load_bytes`,
//!   reached from every call site in the repo.
//! - `symbol_table_inside_the_headers` - the loader read section headers as
//!   symbol entries. The count-driven scan is bounded by the file, so a
//!   `NumberOfSymbols` of 0xFFFFFFF0 on a 64-byte file terminates in ~13
//!   probes: the guard is about the misread, the bound is what stops a spin.
//! - `string_table_size_past_end_of_file` - the loader clamped silently, so a
//!   truncated name became a `?` with nothing recorded.
//! - `string_table_size_smaller_than_its_own_field` - `data[4..1]` ->
//!   `start > end` panic in the range code, on a 300-byte file.
//! - `absolute_symbol_in_code_symbol_sort` - `Option::unwrap()` on a sort key:
//!   a panic on any object carrying an absolute symbol.
//!
//! The demangler's family (truncated decorated names, unbounded type nesting)
//! is pinned in `crates/demangle/tests/fuzz_findings.rs`; the interpreter's
//! `cf.unwrap()` in `crates/core/tests/interp_missing_flag.rs`.

/// Where a built object put its tables, so a fixture edits a field by name
/// instead of by a magic offset that rots when the builder changes.
pub struct Layout {
    pub symoff: usize,
    pub stroff: usize,
    pub dataoff: usize,
    /// Section-header table start (the file header is 20 bytes, no optional
    /// header, so section `i`'s header is at `shdr + 40*i`).
    pub shdr: usize,
}

/// Build a COFF file: one `.text` section, the caller's extra section-header
/// entries, a symbol table of raw 18-byte records (so a fixture can place a
/// record the loader has never seen: `SectionNumber == 0`, aux counts, a name
/// offset past the string table), a string table, then `.text`.
pub fn coff(
    code: &[u8],
    extra_sections: &[[u8; 40]],
    raw_symbols: &[[u8; 18]],
) -> (Vec<u8>, Layout) {
    let nsec = 1 + extra_sections.len();
    let symoff = 20 + 40 * nsec;
    let stroff = symoff + 18 * raw_symbols.len();
    let str_body: &[u8] = b".text\0"; // Size counts itself, so: 4 + 6
    let dataoff = stroff + 4 + str_body.len();
    let mut v: Vec<u8> = Vec::new();
    v.extend_from_slice(&0x8664u16.to_le_bytes()); // Machine
    v.extend_from_slice(&(nsec as u16).to_le_bytes()); // NumberOfSections
    v.extend_from_slice(&[0u8; 4]); // TimeDateStamp
    v.extend_from_slice(&(symoff as u32).to_le_bytes()); // PointerToSymbolTable
    v.extend_from_slice(&(raw_symbols.len() as u32).to_le_bytes()); // NumberOfSymbols
    v.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    v.extend_from_slice(&0x002Eu16.to_le_bytes()); // Characteristics
    debug_assert_eq!(
        v.len(),
        20,
        "COFF header is 20 bytes with no optional header"
    );
    // .text: VA 0x1000, raw data at `dataoff`.
    let mut sh = [0u8; 40];
    sh[..5].copy_from_slice(b".text");
    sh[8..12].copy_from_slice(&(code.len() as u32).to_le_bytes()); // VirtualSize
    sh[12..16].copy_from_slice(&0x1000u32.to_le_bytes()); // VirtualAddress
    sh[16..20].copy_from_slice(&(code.len() as u32).to_le_bytes()); // SizeOfRawData
    sh[20..24].copy_from_slice(&(dataoff as u32).to_le_bytes()); // PointerToRawData
    sh[36..40].copy_from_slice(&0x6050_0020u32.to_le_bytes()); // code, exec, read
    v.extend_from_slice(&sh);
    for s in extra_sections {
        v.extend_from_slice(s);
    }
    for s in raw_symbols {
        v.extend_from_slice(s);
    }
    v.extend_from_slice(&((4 + str_body.len()) as u32).to_le_bytes());
    v.extend_from_slice(str_body);
    debug_assert_eq!(v.len(), dataoff, "the string table ends where .text starts");
    v.extend_from_slice(code);
    (
        v,
        Layout {
            symoff,
            stroff,
            dataoff,
            shdr: 20,
        },
    )
}

/// A raw 18-byte symbol record, COFF field offsets: `Name` 0..8, `Value` 8..12,
/// `SectionNumber` 12..14, `Type` 14..16, `StorageClass` 16,
/// `NumberOfAuxiliarySymbols` 17. `SectionNumber` is the field whose 0 means
/// ABSOLUTE; `NumberOfAuxiliarySymbols` is the one that makes a raw table index
/// differ from a main-symbol index.
///
/// `name` is the 8-byte short-name field: a NUL-padded inline name, or
/// `[0,0,0,0, lo..hi]` to address the string table.
pub fn sym(name: &[u8; 8], section: u16, value: u32, storage: u8, aux: u8) -> [u8; 18] {
    let mut s = [0u8; 18];
    s[0..8].copy_from_slice(name);
    s[8..12].copy_from_slice(&value.to_le_bytes());
    s[12..14].copy_from_slice(&section.to_le_bytes());
    s[16] = storage; // 3 = STORAGE_STATIC
    s[17] = aux;
    s
}

/// A short inline name in the 8-byte `Name` field.
fn nm(s: &str) -> [u8; 8] {
    let mut n = [0u8; 8];
    n[..s.len().min(8)].copy_from_slice(&s.as_bytes()[..s.len().min(8)]);
    n
}

fn code4() -> Vec<u8> {
    vec![0x48, 0x89, 0xC8, 0xC3] // mov rax, rcx ; ret
}

#[test]
fn bss_pointer_to_raw_data_zero() {
    // The 82c285f family, live again. A `.bss`-style section has
    // `PointerToRawData == 0` BY DEFINITION - no file bytes - while
    // `SizeOfRawData` may be non-zero (clang writes 16 there; the PE loader
    // zeroes the VAs). A loader that slices `file[0..16]` hands out the first
    // 16 bytes of the file - the COFF header, whose first bytes are
    // `64 86 ...` - as the section's content. Discovery then disassembles the
    // object's own header, and the decompiler prints C for machine code that is
    // not in the binary. That is the cardinal sin: a silent misread, not a
    // crash, which is why the fuzz gate cannot be the only defence and a
    // byte-exact fixture is.
    let (mut v, _l) = coff(
        &code4(),
        &[{
            let mut sh = [0u8; 40];
            sh[..4].copy_from_slice(b".bss");
            sh[8..12].copy_from_slice(&16u32.to_le_bytes()); // VirtualSize
            sh[12..16].copy_from_slice(&0x2000u32.to_le_bytes()); // VirtualAddress
            sh[16..20].copy_from_slice(&16u32.to_le_bytes()); // SizeOfRawData: non-zero
            sh[20..24].copy_from_slice(&0u32.to_le_bytes()); // PointerToRawData: ZERO
            sh[24..28].copy_from_slice(&0xC000_0080u32.to_le_bytes()); // uninit, rw
            sh
        }],
        &[sym(&nm("f"), 1, 0, 3, 0)],
    );
    // 32 bytes of file behind the headers, so a `file[0..16]` read is possible.
    v.resize(v.len() + 32, 0xAA);
    let o = rdec_arch_x86::loader::load_bytes(&v, "bss_zero_ptr.obj").expect("loads");
    let bss = o.sections.iter().find(|s| s.name == ".bss").expect(".bss");
    // No file content means an EMPTY section, never the file header's bytes.
    // The guard is the `raw_size == 0 || raw_off == 0` branch in
    // `loader::load_bytes`: drop it and this section reads `64 86 01 00 ...`,
    // the COFF header, as `.bss` content - the 82c285f miscompile.
    assert!(
        bss.bytes.is_empty(),
        ".bss has no bytes in the file: the loader must model that as an empty \
         section, not as the file's first SizeOfRawData bytes (the 82c285f \
         miscompile). Got: {:?} - the file starts {:?} and a `file[0..16]` read \
         is exactly that",
        &bss.bytes[..bss.bytes.len().min(8)],
        &v[0..8]
    );
    assert_eq!(&v[0..2], &[0x64, 0x86]); // the bytes a header read would return
                                         // The code section is unaffected: a real PointerToRawData still maps.
    let text = o
        .sections
        .iter()
        .find(|s| s.name == ".text")
        .expect(".text");
    assert_eq!(
        text.bytes,
        code4(),
        "the section that HAS file bytes keeps them"
    );
}

#[test]
fn reloc_table_past_end_of_file() {
    // `PointerToRelocations` inside the file with a `NumberOfRelocations` whose
    // table runs off the end. The parser used to trust the count: the slice
    // `data[start..start + 10 * n]` panics in the range code, from inside
    // `load_bytes` - so every gate that loads objects dies on one hostile file.
    // Found by the byte-level field mutator: corpus seed, seed 20261106,
    // iteration 145 (the first landing on a section-header field; pure random
    // bytes take far longer to produce a valid COFF plus an in-range pointer).
    let (mut v, l) = coff(&code4(), &[], &[sym(&nm("f"), 1, 0, 3, 0)]);
    // Section 0's relocation table: starts inside the file (at `.text`) and
    // claims 100 records - 1000 bytes - which the file does not have.
    v[l.shdr + 24..l.shdr + 28].copy_from_slice(&(l.dataoff as u32).to_le_bytes());
    v[l.shdr + 32..l.shdr + 36].copy_from_slice(&100u32.to_le_bytes());
    let e = rdec_arch_x86::loader::load_bytes(&v, "relocs_past_end.obj")
        .expect_err("the table is truncated, so the object must be refused");
    assert!(
        e.contains("relocation table runs off the end")
            && e.contains("100 relocations at PointerToRelocations")
            && e.contains("past the end of a"),
        "the refusal must name the field, the count and the file length, got: {e}"
    );
    // Sabotage: replace the `checked_add(...).filter(|&e| e <= len)` with the
    // old `start + 10 * n` slice and this test panics in the range code.
}

#[test]
fn symbol_table_inside_the_headers() {
    // `NumberOfSymbols` huge with `PointerToSymbolTable` pointing at the
    // section headers: the loader would read header bytes as symbol entries -
    // names, section numbers, aux counts, all fabricated from the header - and
    // discovery would run on symbols that do not exist. The refusal names the
    // field and the header end; the count-driven scan is separately bounded by
    // the file (`start + 18 <= len`), which is what makes a 4-billion-entry
    // table terminate.
    // One section, so the headers end at 60; `PointerToSymbolTable = 20` points
    // the symbol table AT the section-header table.
    let (mut v, _l) = coff(&code4(), &[], &[]);
    v[8..12].copy_from_slice(&20u32.to_le_bytes()); // PointerToSymbolTable: in headers
    v[12..16].copy_from_slice(&4u32.to_le_bytes()); // NumberOfSymbols: 4 records
    let e = rdec_arch_x86::loader::load_bytes(&v, "syms_in_headers.obj")
        .expect_err("a symbol table inside the headers must be refused");
    assert!(
        e.contains("inside the headers"),
        "the refusal must name PointerToSymbolTable and the header end, got: {e}"
    );

    // The second half of the same field pair, the unbounded-allocation / spin
    // case: a valid `PointerToSymbolTable` with `NumberOfSymbols` = 0xFFFFFFF0
    // (4.29e9 entries, 77 GB of table). The scan is bounded by the FILE
    // (`start + 18 <= len`), and the refusal is the loud one naming the offset
    // - a count-driven loop over 4.29e9 iterations is a CI timeout, and a
    // `Vec::with_capacity(nsyms)` is an out-of-memory kill, on a 92-byte file.
    let (mut v, l) = coff(&code4(), &[], &[]);
    v[8..12].copy_from_slice(&(l.symoff as u32).to_le_bytes());
    v[12..16].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
    let t = std::time::Instant::now();
    let e = rdec_arch_x86::loader::load_bytes(&v, "nsyms_4e9.obj")
        .expect_err("a 4-billion-entry symbol table must be refused, not allocated");
    let dt = t.elapsed();
    assert!(e.contains("symbol table"), "got: {e}");
    assert!(
        dt.as_millis() < 1000,
        "a 4-billion-entry table on a {}-byte file took {dt:?}: the scan is \
         not bounded by the file length",
        v.len()
    );
}

#[test]
fn string_table_size_past_end_of_file() {
    // `StringTableSize` (the first u32 of the string table) claiming more bytes
    // than the file holds. The loader used to clamp silently, so a symbol whose
    // string has no NUL before EOF produced a name read as `?` with nothing
    // recorded. Now the file is refused, naming the offset and the field.
    // The loader's answer to a string table that lies about its size is LOUD
    // AND SOFT: the object loads (its code is intact, and refusing a file whose
    // only damage is in the name table would refuse real objects too), the
    // table is clamped to the file, and `string_table_truncated` records that
    // long names near its end resolved to `?`. The census counts the flag
    // (`load.string_table_truncated`), so the loss is a number in the gate.
    let (mut v, l) = coff(&code4(), &[], &[sym(&nm("f"), 1, 0, 3, 0)]);
    v[l.stroff..l.stroff + 4].copy_from_slice(&100_000u32.to_le_bytes()); // the lie
    let o = rdec_arch_x86::loader::load_bytes(&v, "st_past_end.obj").expect("loads");
    assert!(
        o.string_table_truncated,
        "a string table claiming 100000 bytes in a {}-byte file must set the \
         truncation flag - a silent clamp is how a name loss disappears",
        v.len()
    );
    // The SAME builder with a well-formed size loads clean and the truncation
    // flag is false - the guard must not refuse good files (CI's 124 objects
    // depend on that; `rdec discover --corpus` coverage is the regression).
    let (good, _l) = coff(&code4(), &[], &[sym(&nm("f"), 1, 0, 3, 0)]);
    let o = rdec_arch_x86::loader::load_bytes(&good, "fine.obj").expect("loads");
    assert!(!o.string_table_truncated);
}

#[test]
fn string_table_size_smaller_than_its_own_field() {
    // The string table's `Size` field counts itself (COFF: >= 4). `Size == 1`
    // made the slice `data[4..1]`, which panics `start > end` in the range
    // code. Found by the random-bytes family: seed 1234, iteration 2114, a
    // 300-byte input. The guard is the `st_size.max(4)` in `loader::load_bytes`
    // - a size below its own field is a clamp, not a refusal, because the file
    // is otherwise readable and the clamp is the smallest sound reading.
    let (mut v, l) = coff(&code4(), &[], &[sym(&nm("f"), 1, 0, 3, 0)]);
    v[l.stroff..l.stroff + 4].copy_from_slice(&1u32.to_le_bytes());
    // Either refused, or loaded with the size clamped - never a panic, which is
    // all this fixture pins, and the reason it does not name a message.
    let _ = rdec_arch_x86::loader::load_bytes(&v, "st_size_one.obj");
}

#[test]
fn absolute_symbol_in_code_symbol_sort() {
    // A symbol record with `SectionNumber == 0` is ABSOLUTE (no section):
    // `Symbol::section` is `None`. `Object::code_symbols` sorted by
    // `section.unwrap()`, so the sort key panicked - on the loader's public
    // API, reached from discovery, on the exact class of object production
    // hands a decompiler (linked objects carry absolute symbols; `.debug$S`
    // contributes a wall of them). The key is now
    // `section.unwrap_or(usize::MAX)` - a total order - in
    // `loader::Object::code_symbols`.
    let (v, _l) = coff(
        &code4(),
        &[],
        &[
            sym(&nm("fa"), 1, 0, 3, 0),      // .text function
            sym(&nm("ab"), 0, 0x1234, 3, 0), // ABSOLUTE: SectionNumber 0 -> None
            sym(&nm("fb"), 1, 4, 3, 0),      // second .text function
            sym(&nm("ud"), 0xFFFF, 0, 2, 0), // undefined: section None
        ],
    );
    let o = rdec_arch_x86::loader::load_bytes(&v, "absolute_syms.obj").expect("loads");
    assert_eq!(
        o.symbols.iter().filter(|s| s.section.is_none()).count(),
        2,
        "the absolute and the undefined symbol both have no section"
    );
    assert_eq!(
        o.code_symbols(),
        vec![0, 2],
        "the two .text symbols in address order, absolute/undefined filtered out"
    );
    // The fixture's teeth, proven: today's filter excludes `None`, so the old
    // `unwrap()` needs ONE plausible filter change to fire. Sabotage the filter
    // (`s.section.map(|x| self.is_code(x)).unwrap_or(true)` - treating a symbol
    // with no section as code) and this test panics with `Option::unwrap()` on a
    // `None` value, which is the rot a total-order key exists to prevent.
    let mut v2 = v.clone();
    v2[8..12].copy_from_slice(&0u32.to_le_bytes()); // PointerToSymbolTable = 0
    assert!(
        rdec_arch_x86::loader::load_bytes(&v2, "symoff_zero.obj").is_err(),
        "a non-zero NumberOfSymbols with PointerToSymbolTable == 0 is refused: \
         the loader must not read the file header as symbol entries (82c285f)"
    );
}

// ---------------------------------------------------------------------------
// 7. The 65-second hang: the interpreter's step budget charged instructions
//    only, so a cycle of EMPTY blocks was free.
// ---------------------------------------------------------------------------

/// The fuzzer's seed 1, iteration 1992 (generator `insn`, 270 bytes, detail
/// "152 random bytes as .text"): a one-section object whose `.text` is 152
/// random bytes at `PointerToRawData` 0x76.
///
/// Lifted, the discovered function is
///
/// ```text
/// bb0(at 0x0):  ... instructions ...   br bb1
/// bb1(at 0x24):                        br bb1
/// ```
///
/// `bb1` is EMPTY - its `jmp` arrived as the block's terminator and nothing
/// else, the `bytes_past_ended_block` shape the lifter already counts. The
/// interpreter charged one step per INSTRUCTION, so every lap of `bb1` cost
/// zero and `self.steps > self.budget` could never become true.
///
/// Measured at HEAD (instruction-only budget): `rdec fuzz --repro` on these
/// bytes ran past a 300-second timeout and was killed, and the census line it
/// left behind said only `slowest 65594 ms`, which names nothing. With the
/// per-block-entry charge in `crates/core/src/interp.rs::exec_bb`, the same
/// input completes in single-digit milliseconds.
///
/// WHY THESE EXACT BYTES AND NOT A SMALLER HAND-BUILT OBJECT (measured, not
/// assumed): byte-level delta-minimisation of this object cannot shrink it -
/// every 1-byte deletion moves `PointerToSymbolTable`, `PointerToRawData` or
/// `SizeOfRawData` off the value the header claims, the loader then REFUSES the
/// file, and a refusal is not a hang (270 bytes in, 270 bytes out, zero
/// deletable chunks). A container-built object with a hand-written self-jump
/// also does not hang: it lifts to a block that DOES contain the `jmp`, so the
/// instruction charge covers it. The empty-block cycle needs this object's
/// symbol extents and block plan. The hermetic minimal shape therefore lives one
/// level down, hand-built in IR, at
/// `crates/core/tests/interp_block_transfer.rs::an_empty_self_loop_block_spends_budget`.
const SEED1_ITER1992: &[u8] = &[
    0x64, 0x86, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x50, 0x60, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x01, 0x98, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x20, 0x00,
    0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0xb4, 0x8f, 0xba, 0x6b, 0x2e, 0x2a, 0xab, 0xfb, 0x44, 0x3c,
    0x44, 0x20, 0xba, 0x65, 0xf5, 0x1c, 0x58, 0x41, 0xdf, 0xc8, 0x6e, 0x95, 0x55, 0x3a, 0xfa, 0xba,
    0x21, 0xe1, 0xa7, 0x3a, 0xdb, 0xae, 0xcc, 0x2a, 0xed, 0x65, 0xeb, 0xfe, 0xac, 0x86, 0xfc, 0xef,
    0xb2, 0xc8, 0x53, 0x0f, 0xd5, 0xcd, 0x21, 0x70, 0xbc, 0x4d, 0x58, 0xa3, 0x01, 0xe1, 0x02, 0x78,
    0x3a, 0xf2, 0xe1, 0xe7, 0x43, 0xc9, 0x61, 0xe2, 0xae, 0x4f, 0xa1, 0x51, 0x5d, 0xf7, 0x7b, 0x9f,
    0xbd, 0x9f, 0x82, 0xc1, 0xe9, 0xe2, 0xf2, 0x45, 0x56, 0x28, 0x0a, 0x90, 0x12, 0x49, 0x11, 0x3b,
    0xcf, 0xa6, 0x72, 0x89, 0x88, 0xe4, 0xe6, 0xc0, 0xcd, 0xe0, 0x92, 0x5d, 0x35, 0x02, 0x34, 0x62,
    0x10, 0xd4, 0xc8, 0x2f, 0x9a, 0x96, 0xa6, 0x6d, 0x74, 0xf4, 0x05, 0x89, 0x18, 0x5f, 0xf8, 0xc3,
    0x85, 0xeb, 0x2c, 0xc0, 0xb9, 0x68, 0x6d, 0x1b, 0xa3, 0x3e, 0x35, 0x4c, 0x07, 0x70, 0xd8, 0x14,
    0x81, 0xf6, 0x44, 0xbc, 0xc1, 0x0d, 0xdc, 0xe0, 0xfa, 0xb4, 0x09, 0x56, 0xd4, 0xef,
];

#[test]
fn the_seed1_hang_input_terminates_under_a_step_budget() {
    use std::time::Instant;
    // The fuzzer's own surface, called exactly the way `tests/fuzz.rs` calls
    // it: decoder -> container -> seeds -> discovery -> lift -> passes ->
    // verify -> emit C -> interpreter with the production 20 000-step budget.
    let input = rdec_arch_x86::fuzz::Input {
        seed: 1,
        iter: 1992,
        gen: "insn",
        detail: "152 random bytes as .text".to_string(),
        bytes: SEED1_ITER1992.to_vec(),
    };
    let mut st = rdec_arch_x86::fuzz::Stats::default();
    let t = Instant::now();
    rdec_arch_x86::fuzz::front_half(&input, &mut st);
    let ms = t.elapsed().as_millis();
    // Anti-invention: the fixture is only worth having if the input still
    // REACHES the stage that used to spin. If a loader/lift change makes this
    // object stop reaching the interpreter, the fixture pins nothing, and this
    // assert says so out loud instead of passing on a vacuous green.
    assert!(
        st.get("interp.runs") > 0,
        "the witness must still reach the interpreter (stages: {:?}); if it \
         no longer does, this fixture pins nothing and must be re-found",
        st.stages
    );
    // The assert is TERMINATION. Pre-fix this call never returned: the budget
    // charged one step per INSTRUCTION, the lifted function's `bb1` is empty
    // and jumps to itself, and every lap of it charged zero.
    //
    // Generous (the real run is single-digit ms) and four orders under the
    // 65 s this input cost at HEAD, so the assert cannot pass by accident.
    assert!(
        ms < 5_000,
        "one 270-byte object took {ms} ms through the fuzz front half; the \
         instruction-only step budget let the same input run for 65 s and \
         never finish (fuzz seed 1, iteration 1992)"
    );
}
