//! The `.reloc` base-relocation table of a PE32+ *image*.
//!
//! WHY THIS FILE EXISTS. An image is linked at a *preferred* address and is
//! loaded at some other one half the time, so the linker records every place
//! in the file where an address is embedded: the
//! `IMAGE_DIRECTORY_ENTRY_BASERELOC` array of `IMAGE_BASE_RELOCATION` blocks.
//! That makes `.reloc` not just the loader's patch list but a *census of
//! addresses the image embeds* — the image-side twin of the COFF `REL32`
//! records `relocs.rs` reads for objects, which is where this front end learned
//! that "the target of a call is not in the instruction; it is in the
//! relocation". Measured on the M7 probe artifact (`m7.exe`, recipe in
//! ROADMAP.md): BASERELOC rva 0x1f000 size 0x64c, 6 blocks, 782 entries =
//! 779 `DIR64` plus 3 `ABSOLUTE` padding entries, and 47 of the `DIR64` sites
//! hold an address that lands inside `.text` (38 distinct addresses) — function
//! starts a symbol-free EXE names nowhere else. Zero entries have their *site*
//! inside `.text` (MSVC puts the patch sites in `.rdata`/`.data` and keeps
//! `.text` position-independent), so the "relocation site inside code" class is
//! witnessed by a synthetic fixture, not by the real image.
//!
//! THE LAYOUT, from `winnt.h`, and every bound this reader puts on it:
//!
//! ```text
//! IMAGE_BASE_RELOCATION  { PageRVA: u32, SizeOfBlock: u32 }        // 8 bytes
//! (SizeOfBlock - 8) / 2 entries of u16: type = e >> 12, offset = e & 0xFFF
//! site RVA = PageRVA + offset                                      // 12-bit offset
//! ```
//!
//! * the directory's `RVA + Size` must not overflow the RVA space, and its last
//!   byte must be in the file (checked BEFORE any allocation, so a hostile
//!   `Size` cannot size a `Vec`);
//! * the directory size is a multiple of the 8-byte block header;
//! * `SizeOfBlock >= 8`, `SizeOfBlock` is 8 plus a whole number of 2-byte
//!   entries, and the block ends inside the directory (a block that runs past
//!   the directory end is a block that runs past the table, and the refusal
//!   names the directory whose end it crossed);
//! * `PageRVA` is 4096-aligned: the entry offset is 12 bits *within that page*,
//!   so an unaligned page declares a window straddling two pages and its
//!   offsets name bytes outside the page the block is about;
//! * `PageRVA` strictly ascends the table — the loader walks blocks in file
//!   order applying patches to pages, and two blocks for one page apply that
//!   page twice; the refusal names the pair, the same shape as `.pdata`'s
//!   sortedness check;
//! * every entry's site (and the whole width its type patches) is readable
//!   bytes of the file — refused PER ENTRY, named, counted.
//!
//! TOTALITY IS A PARTITION, and the API states it as one:
//! [`BaseRelocs::census`] buckets every entry as padding / absolute / relative /
//! unreadable-site / refused-kind, and `balanced()` says the five buckets sum to
//! the entry count. A census that dropped the 3 padding entries of `m7.exe`
//! would report "779 entries" for a 782-entry table, which is the same sin as
//! the silent `.pdata` drop M7 slice 2 refused: the count is the claim.
//!
//! TYPES ARE NAMED, never reinterpreted. `IMAGE_REL_BASED_ABSOLUTE` (0) is
//! padding: counted, never applied. The absolute family (1, 2, 3, 4, 10)
//! patches an address into the site; `REL32` (7) is relative and its target is
//! `site + 4 + addend`. A type from another architecture, and
//! `IMAGE_REL_BASED_BASE_RELOC64` (13 — a 16-byte block header, so reading the
//! block as 8 bytes would not fail, it would slide every entry by 8 and hand
//! back a plausible-looking entry list from a table this reader has not read
//! the header of), is refused PER ENTRY BY NAME. One entry of a type this
//! reader does not know is that entry's property, not the table's: the
//! per-entry granularity is the measured lesson of `.pdata` v2, where a
//! whole-table refusal threw away 328 true function starts.
//!
//! WHAT IS NOT MODELLED, deliberately, and it is loud: no relocation is
//! *applied* (nothing is patched, no bytes are rewritten — the image on disk is
//! the image we decode), the HIGH/LOW 16-bit family is never *paired* into a
//! 64-bit address (a half is reported as unresolvable, not guessed), and the
//! import directory is not read, so a relocation site inside code whose target
//! is an IAT slot is named **by RVA**, never by the callee's name.

use crate::pe::PeImage;
use std::collections::BTreeMap;

/// The data-directory index of the base relocation table.
pub const BASERELOC_DIR: usize = 5;
/// `sizeof(IMAGE_BASE_RELOCATION)`: the block header, `PageRVA` + `SizeOfBlock`.
pub const BLOCK_HEADER_LEN: usize = 8;
/// The stride of one relocation entry (`type:4 | offset:12` in a `u16`).
pub const ENTRY_LEN: usize = 2;
/// The page `PageRVA` names, and the exclusive bound of the 12-bit offset.
pub const PAGE_SIZE: u32 = 0x1000;

// `IMAGE_REL_BASED_*` (`winnt.h`), the type nibble of a base-relocation entry.
pub const REL_BASED_ABSOLUTE: u16 = 0;
pub const REL_BASED_HIGH: u16 = 1;
pub const REL_BASED_LOW: u16 = 2;
pub const REL_BASED_HIGHLOW: u16 = 3;
pub const REL_BASED_HIGHADJ: u16 = 4;
pub const REL_BASED_MIPS_JMPADDR: u16 = 5;
pub const REL_BASED_SECTION: u16 = 6;
pub const REL_BASED_REL32: u16 = 7;
pub const REL_BASED_MIPS_JMPADDR16: u16 = 9;
pub const REL_BASED_DIR64: u16 = 10;
pub const REL_BASED_ARM32: u16 = 11;
pub const REL_BASED_ARM_THUMB2: u16 = 12;
pub const REL_BASED_RELOC64: u16 = 13;
pub const REL_BASED_HIGHADJ_1: u16 = 14;
pub const REL_BASED_HIGHADJ_2: u16 = 15;

/// The `winnt.h` name of a type nibble. An unknown nibble prints as
/// `TYPE{n}`, never as the name of a neighbouring type: a census that renames
/// a type it does not know has stopped reporting what the file says.
pub fn type_name(kind: u16) -> &'static str {
    match kind {
        REL_BASED_ABSOLUTE => "ABSOLUTE",
        REL_BASED_HIGH => "HIGH",
        REL_BASED_LOW => "LOW",
        REL_BASED_HIGHLOW => "HIGHLOW",
        REL_BASED_HIGHADJ => "HIGHADJ",
        REL_BASED_MIPS_JMPADDR => "MIPS_JMPADDR",
        REL_BASED_SECTION => "SECTION",
        REL_BASED_REL32 => "REL32",
        // winnt.h assigns no name to 8 — the gap between REL32 (7) and
        // MIPS_JMPADDR16 (9). It prints as itself, never as a neighbour's name.
        8 => "UNASSIGNED",
        REL_BASED_MIPS_JMPADDR16 => "MIPS_JMPADDR16",
        REL_BASED_DIR64 => "DIR64",
        REL_BASED_ARM32 => "ARM32",
        REL_BASED_ARM_THUMB2 => "ARM_THUMB2",
        REL_BASED_RELOC64 => "RELOC64",
        REL_BASED_HIGHADJ_1 => "HIGHADJ_1",
        REL_BASED_HIGHADJ_2 => "HIGHADJ_2",
        // The type field is a nibble, so a parsed entry can never reach this
        // arm; the API takes a u16, and an out-of-range one prints as itself.
        _ => "TYPE?",
    }
}

/// What a type nibble means to this reader, and what it costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Type 0: padding, present so blocks end on a page. Counted, never applied.
    Padding,
    /// HIGH / LOW / HIGHLOW / HIGHADJ / DIR64: the bytes at the site *hold an
    /// address*, so the site is a place the image embeds one.
    Absolute,
    /// REL32: the 4 bytes at the site are `target - (site + 4)`.
    Relative,
    /// A type whose layout this reader has not read: the site is not read as
    /// this layout and the entry is refused AS AN ENTRY, naming the type.
    Refused(&'static str),
}

/// Classify a type nibble. Every nibble lands in exactly one bucket, which is
/// what makes [`BaseRelocs::census`] a partition and not a filter.
pub fn classify(kind: u16) -> Kind {
    match kind {
        REL_BASED_ABSOLUTE => Kind::Padding,
        REL_BASED_HIGH | REL_BASED_LOW | REL_BASED_HIGHLOW | REL_BASED_HIGHADJ
        | REL_BASED_DIR64 => Kind::Absolute,
        REL_BASED_REL32 => Kind::Relative,
        REL_BASED_MIPS_JMPADDR => Kind::Refused("MIPS_JMPADDR: a MIPS instruction address"),
        REL_BASED_SECTION => Kind::Refused("SECTION: a section-index relocation, not a base patch"),
        REL_BASED_MIPS_JMPADDR16 => Kind::Refused("MIPS_JMPADDR16: a MIPS16 instruction address"),
        REL_BASED_ARM32 => Kind::Refused("ARM32: an ARM32 base relocation"),
        REL_BASED_ARM_THUMB2 => Kind::Refused("ARM_THUMB2: a Thumb-2 base relocation"),
        REL_BASED_RELOC64 => Kind::Refused(
            "RELOC64: IMAGE_BASE_RELOCATION64 has a 16-byte header, so this 8-byte block layout \
             would not fail, it would slide every entry and return a plausible entry list",
        ),
        REL_BASED_HIGHADJ_1 => Kind::Refused("HIGHADJ_1: an ARM64X function-table relocation"),
        REL_BASED_HIGHADJ_2 => Kind::Refused("HIGHADJ_2: an ARM64X function-table relocation"),
        // Unreachable from a parsed entry (the type field is a nibble); the
        // function takes a u16, so the arm exists and says so.
        _ => Kind::Refused("a type nibble winnt.h does not define"),
    }
}

/// How many bytes the type patches at its site: the width the site read is
/// bounded by. `None` for a refused kind (its bytes are never read as this
/// layout) and for padding (a padding entry's site means nothing; the loader
/// skips it, so binding it to the file would refuse real images for a field
/// nobody reads).
pub fn patch_width(kind: u16) -> Option<usize> {
    match kind {
        REL_BASED_ABSOLUTE => Some(0),
        REL_BASED_HIGH | REL_BASED_LOW | REL_BASED_HIGHADJ_1 | REL_BASED_HIGHADJ_2 => Some(2),
        REL_BASED_HIGHLOW | REL_BASED_HIGHADJ | REL_BASED_REL32 => Some(4),
        REL_BASED_DIR64 => Some(8),
        REL_BASED_RELOC64 => Some(16),
        _ => None,
    }
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn u64le(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

/// One `IMAGE_BASE_RELOCATION` block, read exactly.
#[derive(Clone, Debug)]
pub struct RelocBlock {
    /// Position in the block walk — what a refusal names.
    pub index: usize,
    /// `PageRVA`: the page its entry offsets live in.
    pub page_rva: u32,
    /// `SizeOfBlock`, as the file states it (its own bounds are checked).
    pub size_of_block: u32,
    /// RVA of the block header.
    pub header_rva: u32,
    /// File offset of the block header, when the mapping gave one.
    pub file_off: Option<usize>,
    /// Index of this block's first entry in [`BaseRelocs::entries`].
    pub first_entry: usize,
    /// `(SizeOfBlock - 8) / 2`: entries in this block. Every one of them is in
    /// `entries`, refused or not — a block's count is not a count of survivors.
    pub entry_count: usize,
}

/// The per-entry outcome, and the bucket of the partition it belongs to.
#[derive(Clone, Debug)]
pub enum Entry {
    /// Type 0 padding. Counted; its site is not read.
    Padding,
    /// An absolute-family entry whose patched bytes are in the file.
    Absolute { bytes: Vec<u8> },
    /// A `REL32` entry whose 4 addend bytes are in the file.
    Relative { addend: i32 },
    /// The site (plus its patched width) is not in the image's readable bytes:
    /// refused AS AN ENTRY, named, counted, never dropped.
    Unreadable { reason: String },
    /// A type this reader does not read. Refused AS AN ENTRY, by name.
    Refused { reason: &'static str },
}

impl Entry {
    /// The partition bucket's name, as the census prints it.
    pub fn bucket(&self) -> &'static str {
        match self {
            Entry::Padding => "padding",
            Entry::Absolute { .. } => "absolute",
            Entry::Relative { .. } => "relative",
            Entry::Unreadable { .. } => "unreadable-site",
            Entry::Refused { .. } => "refused-kind",
        }
    }
}

/// One parsed relocation entry: the entry, its block, and the site it names.
#[derive(Clone, Debug)]
pub struct RelocEntry {
    /// Index across the whole table (every block's entries, in file order).
    pub index: usize,
    /// Index of the block that carries it.
    pub block: usize,
    /// Its position inside that block.
    pub pos_in_block: usize,
    /// The raw type nibble, printed with [`type_name`].
    pub kind: u16,
    /// The raw 12-bit offset field.
    pub offset: u16,
    /// `PageRVA + offset`: the RVA of the patched byte. Checked to be a
    /// 4096-aligned-page offset of a page that the block declares.
    pub site_rva: u32,
    /// The outcome the reader reached for this entry.
    pub entry: Entry,
}

/// The five-bucket partition of the table. Every entry lands in exactly one,
/// and the sum equals the entry count — asserted on a synthetic fixture AND on
/// the real linked image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Every entry the table carries.
    pub entries: usize,
    pub padding: usize,
    pub absolute: usize,
    pub relative: usize,
    pub unreadable: usize,
    pub refused_kind: usize,
}

impl Census {
    /// The invariant: the buckets are a partition of `entries`, not a filter.
    pub fn balanced(&self) -> bool {
        self.padding + self.absolute + self.relative + self.unreadable + self.refused_kind
            == self.entries
    }
}

/// A relocation that names an address, resolved to an RVA.
#[derive(Clone, Debug)]
pub struct Target {
    /// The site the address sits in.
    pub site_rva: u32,
    /// The type nibble that said so.
    pub kind: u16,
    /// The address the site holds, as an RVA (`value - ImageBase`).
    pub target_rva: u32,
    /// The address as the file states it.
    pub target_va: u64,
}

/// A relocation whose address this reader will not name, and why. Loud and
/// counted: a half-address is not a guess at an address.
#[derive(Clone, Debug)]
pub struct Unresolved {
    pub site_rva: u32,
    pub kind: u16,
    /// The bytes at the site, as a number (0 for the 16-bit halves).
    pub value: u64,
    pub why: String,
}

/// The whole BASERELOC directory: blocks, every entry, and the partition.
#[derive(Clone, Debug)]
pub struct BaseRelocs {
    /// The image's path, carried into every refusal.
    pub path: String,
    /// The BASERELOC directory's RVA as the header states it.
    pub dir_rva: u32,
    /// Its size; the sum of every `SizeOfBlock` equals it, checked.
    pub dir_size: u32,
    blocks: Vec<RelocBlock>,
    entries: Vec<RelocEntry>,
}

impl BaseRelocs {
    /// Read the `IMAGE_DIRECTORY_ENTRY_BASERELOC` array. Every refusal names
    /// the field, the block/entry, the value found, the value expected, the
    /// offset, and the length of the file (`pe.rs` convention).
    pub fn read(img: &PeImage) -> Result<BaseRelocs, String> {
        let dir = img
            .data_directories
            .get(BASERELOC_DIR)
            .filter(|d| d.rva != 0 && d.size != 0)
            .ok_or_else(|| {
                let (r, s) = img
                    .data_directories
                    .get(BASERELOC_DIR)
                    .map(|d| (d.rva, d.size))
                    .unwrap_or((0, 0));
                format!(
                    "{}: no .reloc: the BASERELOC data directory (index {BASERELOC_DIR}) is empty (rva {r:#x}, size {s:#x}). The image declares no base relocations — a fixed-base image has none — so this front end has no record of where the image embeds an address, in a {}-byte file",
                    img.path, img.file_len
                )
            })?;
        if dir.size < BLOCK_HEADER_LEN as u32 {
            return Err(format!(
                "{}: BASERELOC directory at rva {:#x} has size {}, below the {BLOCK_HEADER_LEN}-byte IMAGE_BASE_RELOCATION header: the table cannot contain one block, in a {}-byte file",
                img.path, dir.rva, dir.size, img.file_len
            ));
        }
        if dir.size % ENTRY_LEN as u32 != 0 {
            // The stride bound is the 2-byte ENTRY, not the 8-byte header, and
            // that is a MEASURED correction, not a loosening: `m7.exe` states
            // BASERELOC size 0x64c = 1612, whose remainder mod 8 is 4, because
            // its last block is SizeOfBlock 0x4c (8 header + 34 entries). A
            // `% 8` rule refuses that real, healthy image at its directory.
            return Err(format!(
                "{}: BASERELOC directory at rva {:#x} has size {}, which is not a multiple of the {ENTRY_LEN}-byte entry stride the blocks are made of ({} byte{} of remainder): the table boundary and the directory disagree about where .reloc ends, in a {}-byte file",
                img.path,
                dir.rva,
                dir.size,
                dir.size % ENTRY_LEN as u32,
                if dir.size % ENTRY_LEN as u32 == 1 { "" } else { "s" },
                img.file_len
            ));
        }
        // RVA-space bound, then the file bound, BEFORE any allocation: `dir.size`
        // is a hostile u32 and a `Vec::with_capacity(size / 10)` on it is the
        // allocation trap from `pe.rs` in a new costume.
        let dir_end = u64::from(dir.rva) + u64::from(dir.size);
        if dir_end > u64::from(u32::MAX) {
            return Err(format!(
                "{}: BASERELOC directory at rva {:#x} with size {:#x} ends past the 32-bit RVA space: the table's end is not an address this format can name, in a {}-byte file",
                img.path, dir.rva, dir.size, img.file_len
            ));
        }
        let last_rva = (dir_end - 1) as u32;
        img.read_at(last_rva, 1).map_err(|e| {
            format!(
                "{}: BASERELOC directory at rva {:#x} size {:#x} declares {} bytes of blocks, but its last byte {last_rva:#x} is not in the file, in a {}-byte file: {e}",
                img.path,
                dir.rva,
                dir.size,
                dir.size,
                img.file_len
            )
        })?;

        let mut blocks: Vec<RelocBlock> = Vec::new();
        let mut entries: Vec<RelocEntry> = Vec::new();
        let mut cursor: u64 = u64::from(dir.rva);
        let mut prev: Option<(usize, u32)> = None;
        while cursor < dir_end {
            let here = cursor as u32;
            let idx = blocks.len();
            let hdr = img.read_at(here, BLOCK_HEADER_LEN).map_err(|e| {
                format!(
                    "{}: block {idx} of the BASERELOC table at rva {here:#x}: reading its {BLOCK_HEADER_LEN}-byte header runs past the bytes the image has, in a {}-byte file: {e}",
                    img.path, img.file_len
                )
            })?;
            let page_rva = u32le(hdr, 0);
            let size = u32le(hdr, 4);
            let file_off = img.va_to_off(here).ok();
            let named = |what: String| {
                format!(
                    "{}: block {idx} at rva {here:#x} (PageRVA {page_rva:#x}{})",
                    what,
                    file_off
                        .map(|o| format!(", file offset {o:#x}"))
                        .unwrap_or_default()
                )
            };
            if size < BLOCK_HEADER_LEN as u32 {
                return Err(format!(
                    "{}: SizeOfBlock is {size}, below the {BLOCK_HEADER_LEN}-byte header it is the size of: the block cannot hold its own header, in a {}-byte file",
                    named(String::new()),
                    img.file_len
                ));
            }
            if !(size - BLOCK_HEADER_LEN as u32).is_multiple_of(ENTRY_LEN as u32) {
                return Err(format!(
                    "{}: SizeOfBlock is {size}: {} bytes of entries after the {BLOCK_HEADER_LEN}-byte header is not a whole number of {ENTRY_LEN}-byte entries, in a {}-byte file",
                    named(String::new()),
                    size - BLOCK_HEADER_LEN as u32,
                    img.file_len
                ));
            }
            // The block's extent in the TABLE is measured from where the block
            // sits, not from PageRVA: PageRVA is the page it PATCHES, and the
            // two have nothing to do with each other.
            let block_end = cursor + u64::from(size);
            if block_end > dir_end {
                return Err(format!(
                    "{}: SizeOfBlock {size} carries the block to table rva {:#x}, past the BASERELOC directory's end {dir_end:#x} (rva {:#x} + size {:#x}): the block runs out of its own table, in a {}-byte file",
                    named(String::new()),
                    block_end,
                    dir.rva,
                    dir.size,
                    img.file_len
                ));
            }
            if !page_rva.is_multiple_of(PAGE_SIZE) {
                return Err(format!(
                    "{}: PageRVA {page_rva:#x} is not a multiple of {PAGE_SIZE:#x}: a block's entry offsets are 12 bits WITHIN that page, so an unaligned page declares a window straddling two pages and its offsets name bytes outside the page the block is about, in a {}-byte file",
                    named(String::new()),
                    img.file_len
                ));
            }
            if let Some((pi, pp)) = prev {
                if page_rva <= pp {
                    return Err(format!(
                        "{}: the base-relocation blocks are not sorted by PageRVA: block {idx} PageRVA {page_rva:#x} does not follow block {pi} PageRVA {pp:#x} (the loader walks blocks in file order, so two blocks for one page apply that page's patches twice), in a {}-byte file",
                        named(String::new()),
                        img.file_len
                    ));
                }
            }
            prev = Some((idx, page_rva));

            let entry_count = (size as usize - BLOCK_HEADER_LEN) / ENTRY_LEN;
            let block = RelocBlock {
                index: idx,
                page_rva,
                size_of_block: size,
                header_rva: here,
                file_off,
                first_entry: entries.len(),
                entry_count,
            };
            // The block's entry array, bounded to the bytes the image has.
            let raw = if entry_count > 0 {
                let at = here + BLOCK_HEADER_LEN as u32;
                img.read_at(at, ENTRY_LEN * entry_count).map_err(|e| {
                    format!(
                        "{}: its {entry_count} entries ({} bytes) at header+8 ({:#x}) do not fit the bytes the image has, in a {}-byte file: {e}",
                        named(String::new()),
                        ENTRY_LEN * entry_count,
                        at,
                        img.file_len
                    )
                })?
            } else {
                &[]
            };
            blocks.push(block);
            let (bidx, brva, bcount) = (idx, page_rva, entry_count);
            for (k, w) in raw.as_chunks::<ENTRY_LEN>().0.iter().enumerate() {
                let e = u16le(w, 0);
                let kind = e >> 12;
                let offset = e & 0x0FFF;
                let site_rva = brva + u32::from(offset);
                debug_assert!(k < bcount);
                let entry = match classify(kind) {
                    Kind::Padding => Entry::Padding,
                    Kind::Refused(reason) => Entry::Refused { reason },
                    Kind::Absolute | Kind::Relative => {
                        let width = patch_width(kind).unwrap_or(0);
                        match img.read_at(site_rva, width) {
                            Ok(bytes) => {
                                if kind == REL_BASED_REL32 {
                                    Entry::Relative {
                                        addend: i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                                    }
                                } else {
                                    Entry::Absolute {
                                        bytes: bytes.to_vec(),
                                    }
                                }
                            }
                            Err(e) => Entry::Unreadable {
                                reason: format!(
                                    "site rva {site_rva:#x} (PageRVA {brva:#x} + offset {offset:#x}) with a {}-byte {} patch: {e}",
                                    width,
                                    type_name(kind)
                                ),
                            },
                        }
                    }
                };
                entries.push(RelocEntry {
                    index: entries.len(),
                    block: bidx,
                    pos_in_block: k,
                    kind,
                    offset,
                    site_rva,
                    entry,
                });
            }
            cursor += u64::from(size);
        }

        Ok(BaseRelocs {
            path: img.path.clone(),
            dir_rva: dir.rva,
            dir_size: dir.size,
            blocks,
            entries,
        })
    }

    /// Every block, in table order.
    pub fn blocks(&self) -> &[RelocBlock] {
        &self.blocks
    }

    /// Every entry, in table order — refused ones included. The count is the
    /// table's, not the survivors'.
    pub fn entries(&self) -> &[RelocEntry] {
        &self.entries
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// THE PARTITION: the five buckets over every entry.
    pub fn census(&self) -> Census {
        let mut c = Census {
            entries: self.entries.len(),
            ..Default::default()
        };
        for e in &self.entries {
            match &e.entry {
                Entry::Padding => c.padding += 1,
                Entry::Absolute { .. } => c.absolute += 1,
                Entry::Relative { .. } => c.relative += 1,
                Entry::Unreadable { .. } => c.unreadable += 1,
                Entry::Refused { .. } => c.refused_kind += 1,
            }
        }
        c
    }

    /// Type nibble -> count, ascending, EVERY entry counted (padding included).
    /// This is the census the CLI prints and the fixture asserts.
    pub fn relocs_by_type(&self) -> Vec<(u16, usize)> {
        let mut map: BTreeMap<u16, usize> = BTreeMap::new();
        for e in &self.entries {
            *map.entry(e.kind).or_insert(0) += 1;
        }
        map.into_iter().collect()
    }

    /// Per-page entry counts, in page order: the cross-check an independent
    /// walker (dumpbin, a Python loop) produces block by block.
    pub fn entries_per_block(&self) -> Vec<(u32, usize)> {
        self.blocks
            .iter()
            .map(|b| (b.page_rva, b.entry_count))
            .collect()
    }

    /// `sum(SizeOfBlock)`: must equal `dir_size`, the walk-coverage fact.
    pub fn sum_block_sizes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|b| u64::from(b.size_of_block))
            .sum::<u64>()
    }

    /// Every relocation that names an address, split into the ones this reader
    /// can name (in table order) and the ones it refuses with a reason.
    ///
    /// `DIR64`: the 8 bytes at the site ARE the address at the image's
    /// preferred base, so `value - ImageBase` is its RVA. `REL32`: the target is
    /// `site + 4 + addend`. A 16-bit half (HIGH/LOW) is refused — pairing two
    /// halves into a 64-bit address is a reconstruction this reader will not
    /// guess at, and an address assembled from a pair the file never pairs is
    /// exactly the "plausible invention" the roadmap forbids.
    pub fn targets(&self, img: &PeImage) -> (Vec<Target>, Vec<Unresolved>) {
        let mut ok: Vec<Target> = Vec::new();
        let mut no: Vec<Unresolved> = Vec::new();
        for e in &self.entries {
            match &e.entry {
                Entry::Absolute { bytes } => {
                    if bytes.len() == 8 {
                        let va = u64le(bytes, 0);
                        match va.checked_sub(img.image_base) {
                            Some(rva) if rva <= u64::from(u32::MAX) => ok.push(Target {
                                site_rva: e.site_rva,
                                kind: e.kind,
                                target_rva: rva as u32,
                                target_va: va,
                            }),
                            _ => no.push(Unresolved {
                                site_rva: e.site_rva,
                                kind: e.kind,
                                value: va,
                                why: format!(
                                    "the 8 bytes at the site are {va:#x}, which is not ImageBase {:#x} plus a 32-bit RVA: the address points outside this image, so it names nothing inside it",
                                    img.image_base
                                ),
                            }),
                        }
                    } else {
                        let mut v = 0u64;
                        for (i, &byte) in bytes.iter().enumerate() {
                            v |= u64::from(byte) << (8 * i);
                        }
                        no.push(Unresolved {
                            site_rva: e.site_rva,
                            kind: e.kind,
                            value: v,
                            why: format!(
                                "{} is a {}-bit half of an address: this reader does not pair HIGH with LOW into a wider address, so the address is not named",
                                type_name(e.kind),
                                bytes.len() * 8
                            ),
                        });
                    }
                }
                Entry::Relative { addend } => {
                    let from = u64::from(e.site_rva) + 4;
                    let target = i64::from(*addend).checked_add(from as i64);
                    match target {
                        Some(t) if (0..=u64::from(u32::MAX) as i64).contains(&t) => ok.push(Target {
                            site_rva: e.site_rva,
                            kind: e.kind,
                            target_rva: t as u32,
                            target_va: img.va(t as u32),
                        }),
                        _ => no.push(Unresolved {
                            site_rva: e.site_rva,
                            kind: e.kind,
                            value: u64::from(*addend as u32),
                            why: format!(
                                "REL32 at site {:#x} with addend {addend} computes a target outside the RVA space",
                                e.site_rva
                            ),
                        }),
                    }
                }
                _ => {}
            }
        }
        (ok, no)
    }

    /// The entries refused for a reason, with the text the CLI prints: the
    /// loud half of the partition.
    pub fn refusals(&self) -> Vec<(usize, String)> {
        self.entries
            .iter()
            .filter_map(|e| match &e.entry {
                Entry::Unreadable { reason } => {
                    Some((e.index, format!("entry {}: {reason}", e.index)))
                }
                Entry::Refused { reason } => Some((
                    e.index,
                    format!(
                        "entry {} ({} at site {:#x}, block {}): {reason}",
                        e.index,
                        type_name(e.kind),
                        e.site_rva,
                        e.block
                    ),
                )),
                _ => None,
            })
            .collect()
    }
}
