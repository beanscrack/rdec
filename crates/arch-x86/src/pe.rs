//! Loading PE32+ *images* - hand-rolled, beside the COFF object reader.
//!
//! WHY THIS IS ITS OWN READER. `loader.rs` reads COFF **objects**: no image
//! signature, no base relocation, no section VA, and no symbol size. A linked
//! `.exe` has all four. Feeding one to the other is the measured behavior the
//! M7 probe records: `rdec discover m7.exe` printed `section 1 (?) has
//! SizeOfRawData 1750344141 at PointerToRawData 1881174889, past the end of a
//! 111104-byte file` - the object reader reading `IMAGE_OPTIONAL_HEADER` bytes
//! as a section table. The refusal was right; the missing thing is a reader
//! that knows the container. The object reader's API is untouched by this file,
//! and `tests/pe_fixtures.rs` asserts the meeting point in both directions: an
//! image through the object reader refuses, an object through this reader
//! refuses, and each names the field it tripped on.
//!
//! SCOPE, and it is deliberate: this is a LOAD-only reader. Headers, section
//! table, the VA<->file mapping, the entry point, the data directories as
//! parsed numbers. `.pdata`/`.xdata` SEH unwind, `.reloc` application, import
//! thunks and CRT library signatures are later M7 slices and are NOT modelled
//! here; what this file guarantees is that everything it does read is read
//! exactly, and that every field it cannot read refuses with the field, the
//! value it found, the value it wanted, the offset, and the length of the file.

/// `IMAGE_DOS_HEADER.e_magic`.
pub const DOS_MAGIC: u16 = 0x5a4d;
/// Offset of `e_lfanew` inside the DOS header.
pub const OFF_E_LFANEW: usize = 0x3c;
/// The DOS header is a fixed 64 bytes; `e_lfanew` lives at its end, so nothing
/// can be trusted before that many bytes exist.
pub const DOS_HEADER_LEN: usize = 64;
/// The smallest DOS header the PE header may point at: everything below 0x40
/// aliases the DOS header that stores the pointer itself.
pub const MIN_E_LFANEW: usize = 0x40;
/// `IMAGE_OPTIONAL_HEADER64.Magic` - PE32+, the only format this reads.
pub const PE32_PLUS_MAGIC: u16 = 0x20b;
/// `IMAGE_OPTIONAL_HEADER32.Magic` - PE32. Named in the refusal that rejects
/// it: PE32 has no 8-byte `ImageBase` and HAS a `BaseOfData`, so every field
/// from `pe+0x30` onward sits 4 bytes off. Reading it as PE32+ does not fail,
/// it produces a plausible image base out of `SectionAlignment` and
/// `FileAlignment` - the silent misread this whole file exists to prevent.
pub const PE32_MAGIC: u16 = 0x10b;
/// The fixed part of the PE32+ optional header, before the data directories:
/// 112 bytes. Below that, a field this loader reads does not exist.
pub const MIN_OPTIONAL_HEADER: usize = 112;
/// `IMAGE_FILE_HEADER` size, and the stride of a section header.
pub const COFF_FILE_HEADER_LEN: usize = 20;
pub const SECTION_HEADER_LEN: usize = 40;

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

/// A section header, read exactly. `index` is its position in the section
/// table, which is what a refusal names alongside the name (two sections can
/// share a name in a hostile file; the index cannot).
#[derive(Clone, Debug)]
pub struct PeSection {
    /// Position in the section table.
    pub index: usize,
    /// `Name`, NUL-trimmed. Images have no string table, so a name is at most
    /// the 8 header bytes - `/123` long-name indirection is a COFF object
    /// feature and does not apply here.
    pub name: String,
    pub virtual_address: u32,
    pub virtual_size: u32,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub characteristics: u32,
}

impl PeSection {
    /// The section's virtual extent. `VirtualSize == 0` is a real linker
    /// shortcut (MSVC does it for `.reloc`-shaped sections), and the bytes
    /// declared in the file are then the only extent the image states.
    pub fn span(&self) -> u64 {
        if self.virtual_size > 0 {
            self.virtual_size as u64
        } else {
            self.size_of_raw_data as u64
        }
    }

    /// True if `rva` is inside the section's declared virtual extent.
    pub fn contains(&self, rva: u32) -> bool {
        match u64::from(self.virtual_address).checked_add(self.span()) {
            Some(end) => u64::from(rva) >= u64::from(self.virtual_address) && u64::from(rva) < end,
            None => false,
        }
    }

    /// `CNT_CODE` with `MEM_EXECUTE` - the same test `Object::is_code` applies
    /// to an object, so "is this code" means one thing in both readers.
    pub fn is_code(&self) -> bool {
        (self.characteristics & 0x0000_0020) != 0 && (self.characteristics & 0x2000_0000) != 0
    }

    /// The characteristics as names, for `rdec pe-info`. Uncovered bits are
    /// appended as a hex tail rather than dropped: a flags summary that
    /// silently omits a flag is a summary that can lie about the section.
    pub fn flags_summary(&self) -> String {
        const NAMED: [(u32, &str); 10] = [
            (0x0000_0020, "CODE"),
            (0x0000_0040, "INIT_DATA"),
            (0x0000_0080, "UNINIT_DATA"),
            (0x0000_0100, "LNK_INFO"),
            (0x0200_0000, "DISCARDABLE"),
            (0x0800_0000, "NOT_PAGED"),
            (0x1000_0000, "SHARED"),
            (0x2000_0000, "EXECUTE"),
            (0x4000_0000, "READ"),
            (0x8000_0000, "WRITE"),
        ];
        let mut out: Vec<&str> = Vec::new();
        let mut rest = self.characteristics;
        for (bit, name) in NAMED {
            if self.characteristics & bit != 0 {
                out.push(name);
                rest &= !bit;
            }
        }
        let mut s = out.join("|");
        if rest != 0 {
            // Every characteristic bit is named, so an unnamed bit is outside
            // the set above. Printed, never dropped.
            if !s.is_empty() {
                s.push('|');
            }
            s.push_str(&format!("CHAR{rest:#010x}"));
        }
        s
    }
}

/// One `IMAGE_DATA_DIRECTORY` entry, parsed as a number. Nothing here follows
/// the RVA: the import table, the exception table (`.pdata`) and the base
/// relocation table (`.reloc`) are the next M7 slices, and a directory that is
/// printed as `IMPORT 0x1000/0x98` is the honest state of "we know where it is
/// and have not read it".
#[derive(Clone, Debug)]
pub struct DataDir {
    pub index: usize,
    pub name: &'static str,
    pub rva: u32,
    pub size: u32,
}

/// The names `winnt.h` gives the sixteen directories, by index.
pub fn data_dir_name(i: usize) -> &'static str {
    [
        "EXPORT",
        "IMPORT",
        "RESOURCE",
        "EXCEPTION",
        "SECURITY",
        "BASERELOC",
        "DEBUG",
        "ARCHITECTURE",
        "GLOBALPTR",
        "TLS",
        "LOAD_CONFIG",
        "BOUND_IMPORT",
        "IAT",
        "DELAY_IMPORT",
        "COM_DESCRIPTOR",
        "RESERVED16",
    ]
    .get(i)
    .copied()
    .unwrap_or("UNKNOWN")
}

/// A parsed PE32+ image. `bytes` is the whole file, borrowed: `read_at` hands
/// out slices of it, so no section content is copied and the loader cannot
/// serve bytes that are not in the file.
#[derive(Clone, Debug)]
pub struct PeImage<'a> {
    pub path: String,
    /// File length, kept next to the fields every bound is measured against.
    pub file_len: usize,
    pub time_date_stamp: u32,
    pub machine: u16,
    /// `Characteristics` of the COFF file header.
    pub characteristics: u16,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub image_base: u64,
    /// `AddressOfEntryPoint`, an RVA. A DLL may legitimately declare 0.
    pub entry_rva: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub section_alignment: u32,
    pub file_alignment: u32,
    /// `NumberOfRvaAndSizes` as the file states it: the count of directories
    /// this reader parses, bounded against `SizeOfOptionalHeader` (the
    /// directories live inside that header). Measured on the M7 probe
    /// artifact: `m7.exe` declares 16 and fills five — IMPORT 0x17b18/0x28,
    /// EXCEPTION 0x1c000/0xfc0 (the `.pdata` the probe counted), BASERELOC
    /// 0x1f000/0x64c (`.reloc`), LOAD_CONFIG, IAT. The RVAs are read; nothing
    /// behind them is followed yet — that is the next slice.
    pub number_of_rva_and_sizes: u32,
    pub data_directories: Vec<DataDir>,
    pub sections: Vec<PeSection>,
    /// Section indexes in ascending `virtual_address` order - the order
    /// `va_to_off` scans, so the mapping never depends on table order.
    by_va: Vec<usize>,
    bytes: &'a [u8],
}

impl<'a> PeImage<'a> {
    /// Parse a PE32+ image from the whole file's bytes. Every refusal names the
    /// field, the value found, the value expected, the file offset, and the
    /// length of the file - the `loader.rs` convention, so a refusal is
    /// actionable against a hex editor.
    pub fn load(bytes: &'a [u8], path: &str) -> Result<PeImage<'a>, String> {
        let len = bytes.len();
        if len < DOS_HEADER_LEN {
            return Err(format!(
                "{}: not a PE image: the file is {len} bytes and a DOS header needs {DOS_HEADER_LEN} (e_lfanew lives at 0x{:x})",
                path, OFF_E_LFANEW
            ));
        }
        if u16le(bytes, 0) != DOS_MAGIC {
            return Err(format!(
                "{}: not a PE image: bytes 0x00..0x02 are {:02x} {:02x}, expected `MZ` ({:#06x}) in a {len}-byte file",
                path,
                bytes[0],
                bytes[1],
                DOS_MAGIC
            ));
        }
        let pe = u32le(bytes, OFF_E_LFANEW) as usize;
        if pe < MIN_E_LFANEW {
            return Err(format!(
                "{}: e_lfanew at 0x{:x} is {pe:#x}, inside the {MIN_E_LFANEW:#x}-byte DOS header that stores it - the PE header cannot start there in a {len}-byte file",
                path, OFF_E_LFANEW
            ));
        }
        // Every later bound is measured from `pe`, so each is one addition
        // checked against the file - no unchecked `bytes[pe + k]` below.
        let need = |end: usize, what: &str| -> Result<(), String> {
            if end > len {
                Err(format!(
                    "{}: e_lfanew {pe:#x} + {} for {what} runs {} bytes past the end of a {len}-byte file",
                    path,
                    end - pe,
                    end - len
                ))
            } else {
                Ok(())
            }
        };
        need(pe + 4, "the PE signature")?;
        if &bytes[pe..pe + 4] != b"PE\0\0" {
            return Err(format!(
                "{}: PE signature at {pe:#x} is {:02x} {:02x} {:02x} {:02x}, expected `PE\\0\\0` (50 45 00 00) in a {len}-byte file",
                path,
                bytes[pe],
                bytes[pe + 1],
                bytes[pe + 2],
                bytes[pe + 3]
            ));
        }
        let coff = pe + 4;
        need(coff + COFF_FILE_HEADER_LEN, "the COFF file header")?;
        let machine = u16le(bytes, coff);
        let nsec = u16le(bytes, coff + 2) as usize;
        let time_date_stamp = u32le(bytes, coff + 4);
        let sizeopt = u16le(bytes, coff + 16) as usize;
        let characteristics = u16le(bytes, coff + 18);
        if machine != crate::loader::MACHINE_AMD64 {
            return Err(format!(
                "{}: Machine at {:#x} is {:#x}, expected {:#x} (IMAGE_FILE_MACHINE_AMD64): this front end decodes x86-64 and nothing else",
                path,
                coff,
                machine,
                crate::loader::MACHINE_AMD64
            ));
        }
        if sizeopt < MIN_OPTIONAL_HEADER {
            return Err(format!(
                "{}: SizeOfOptionalHeader at {:#x} is {sizeopt}, below the {MIN_OPTIONAL_HEADER} bytes of fixed PE32+ fields this loader reads, in a {len}-byte file",
                path,
                coff + 16
            ));
        }
        let opt = coff + COFF_FILE_HEADER_LEN;
        if opt + sizeopt > len {
            // The same event seen from two fields: a too-large optional header
            // pushes the section table off the file too, so the refusal names
            // both - whichever one the reader has to fix in the file.
            return Err(format!(
                "{}: SizeOfOptionalHeader at {:#x} is {sizeopt}: the optional header at {opt:#x} would end at {:#x}, {} bytes past the end of a {len}-byte file, and the section table at {:#x} with it",
                path,
                coff + 16,
                opt + sizeopt,
                opt + sizeopt - len,
                opt + sizeopt
            ));
        }
        let magic = u16le(bytes, opt);
        if magic != PE32_PLUS_MAGIC {
            let which = if magic == PE32_MAGIC {
                "PE32 (32-bit)"
            } else {
                "not PE32+"
            };
            return Err(format!(
                "{}: optional header Magic at {:#x} is {magic:#x} = {which}; expected {PE32_PLUS_MAGIC:#x} (PE32+). The two layouts differ from this field on (PE32 has a 4-byte ImageBase and a BaseOfData), so the 32-bit header is refused, not reinterpreted",
                path, opt
            ));
        }
        let entry_rva = u32le(bytes, opt + 16);
        let image_base = u64le(bytes, opt + 24);
        let section_alignment = u32le(bytes, opt + 32);
        let file_alignment = u32le(bytes, opt + 36);
        let size_of_image = u32le(bytes, opt + 56);
        let size_of_headers = u32le(bytes, opt + 60);
        let subsystem = u16le(bytes, opt + 68);
        let dll_characteristics = u16le(bytes, opt + 70);
        let nrva = u32le(bytes, opt + 108);
        if section_alignment == 0 || file_alignment == 0 {
            // Alignment 0 makes every "is this RVA aligned" reasoning below a
            // division by zero, and no linker emits it. Loud, naming both.
            return Err(format!(
                "{}: SectionAlignment at {:#x} is {section_alignment} and FileAlignment at {:#x} is {file_alignment}; both must be non-zero",
                path,
                opt + 32,
                opt + 36
            ));
        }
        if size_of_image == 0 {
            return Err(format!(
                "{}: SizeOfImage at {:#x} is 0, so no section extent can be measured against it in a {len}-byte file",
                path,
                opt + 56
            ));
        }
        // The directories live INSIDE the optional header: 112 fixed bytes plus
        // 8 per declared directory. A `NumberOfRvaAndSizes` the header has no
        // room for is a field/header disagreement, and reading past the header
        // would land in the section table.
        let dirs_end = opt
            + MIN_OPTIONAL_HEADER
            + usize::try_from(nrva)
                .ok()
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| {
                    format!(
                        "{}: NumberOfRvaAndSizes at {:#x} is {nrva}, too large for its 8-byte entries in a {len}-byte file",
                        path,
                        opt + 108
                    )
                })?;
        if dirs_end > opt + sizeopt {
            return Err(format!(
                "{}: NumberOfRvaAndSizes at {:#x} is {nrva}, needing {} bytes of directories; SizeOfOptionalHeader {sizeopt} leaves {} - the count and the header disagree in a {len}-byte file",
                path,
                opt + 108,
                nrva as usize * 8,
                sizeopt - MIN_OPTIONAL_HEADER
            ));
        }
        let mut data_directories = Vec::new();
        for i in 0..nrva as usize {
            let at = opt + MIN_OPTIONAL_HEADER + 8 * i;
            data_directories.push(DataDir {
                index: i,
                name: data_dir_name(i),
                rva: u32le(bytes, at),
                size: u32le(bytes, at + 4),
            });
        }

        // The section table: sized by the declared count, bounded by the file
        // before any section is pushed. `40 * nsec` cannot overflow on a
        // 64-bit target, and the count is a u16, so the widest ask is
        // 4,194,300 bytes - still refused against the file, not allocated.
        let table = opt + sizeopt;
        let table_end = table + SECTION_HEADER_LEN * nsec;
        if table_end > len {
            return Err(format!(
                "{}: section table at {table:#x} for NumberOfSections {nsec} needs {} bytes, running {} bytes past the end of a {len}-byte file (SizeOfOptionalHeader is {sizeopt})",
                path,
                SECTION_HEADER_LEN * nsec,
                table_end - len
            ));
        }
        if nsec == 0 {
            return Err(format!(
                "{}: NumberOfSections at {:#x} is 0: an image with no sections has no bytes to load, in a {len}-byte file",
                path,
                coff + 2
            ));
        }

        let mut sections: Vec<PeSection> = Vec::with_capacity(nsec);
        for (i, hdr) in bytes[table..table_end]
            .as_chunks::<SECTION_HEADER_LEN>()
            .0
            .iter()
            .enumerate()
        {
            // `hdr` is 40 bytes by construction (the range is exactly
            // `40 * nsec`), so these offsets are closed.
            let name = section_name(&hdr[..8]);
            let h = table + SECTION_HEADER_LEN * i;
            let virtual_size = u32le(hdr, 8);
            let virtual_address = u32le(hdr, 12);
            let size_of_raw_data = u32le(hdr, 16);
            let pointer_to_raw_data = u32le(hdr, 20);
            let ptr_relocs = u32le(hdr, 24);
            let nrelocs = u16le(hdr, 32);
            let characteristics = u32le(hdr, 36);
            // The MSVC spec: both relocation fields "must be zero for images".
            // An image that carries them claims an object's relocation model
            // AND an image's base-relocation block; the second one is `.reloc`
            // (m7.exe has 0x64c bytes of it), and neither is consumed yet. Loud.
            if nrelocs != 0 || ptr_relocs != 0 {
                return Err(format!(
                    "{}: section {i} ({name}) at {h:#x} declares NumberOfRelocations {nrelocs} at PointerToRelocations {ptr_relocs:#x}: a linked image carries base relocations in its .reloc section, not in per-section COFF tables, in a {len}-byte file",
                    path
                ));
            }
            // Virtual extent: overflow first (a section whose end is not a u32
            // RVA has no address in this format at all), then SizeOfImage.
            let end_va = match u64::from(virtual_address).checked_add(
                if virtual_size > 0 {
                    virtual_size as u64
                } else {
                    size_of_raw_data as u64
                },
            ) {
                Some(e) if e <= u64::from(u32::MAX) => e,
                _ => {
                    return Err(format!(
                        "{}: section {i} ({name}) VirtualAddress {virtual_address:#x} + VirtualSize {virtual_size:#x} overflows the 32-bit RVA space at {h:#x}: the section's end is not an address this format can name, in a {len}-byte file",
                        path
                    ))
                }
            };
            if end_va > u64::from(size_of_image) {
                return Err(format!(
                    "{}: section {i} ({name}) VirtualAddress {virtual_address:#x} + VirtualSize {virtual_size:#x} reaches {end_va:#x}, past SizeOfImage {size_of_image:#x} (declared at {:#x}), in a {len}-byte file",
                    path,
                    opt + 56
                ));
            }
            // Raw extent. `PointerToRawData == 0` means "no raw data" ONLY with
            // `SizeOfRawData == 0` (an all-zero .bss): a nonzero size with a
            // zero pointer promises bytes at file offset 0, which is the DOS
            // header. `loader.rs`'s `.bss` test documents the same misread for
            // objects, where it made a C++ static member read the machine word.
            if size_of_raw_data > 0 && pointer_to_raw_data == 0 {
                return Err(format!(
                    "{}: section {i} ({name}) at {h:#x} has SizeOfRawData {size_of_raw_data} with PointerToRawData 0: the bytes it promises are the DOS header, in a {len}-byte file",
                    path
                ));
            }
            let raw_end = u64::from(pointer_to_raw_data) + u64::from(size_of_raw_data);
            if raw_end > len as u64 {
                return Err(format!(
                    "{}: section {i} ({name}) SizeOfRawData {size_of_raw_data} at PointerToRawData {pointer_to_raw_data:#x} (field at {h:#x}) runs {} bytes past the end of a {len}-byte file",
                    path,
                    raw_end - len as u64
                ));
            }
            sections.push(PeSection {
                index: i,
                name,
                virtual_address,
                virtual_size,
                size_of_raw_data,
                pointer_to_raw_data,
                characteristics,
            });
        }

        // Two sections claiming one RVA means `va_to_off` has two answers, and
        // picking one silently is how a decompiler reads a byte the image does
        // not have at that address. Checked in VA order, named by both sections.
        let mut by_va: Vec<usize> = (0..sections.len()).collect();
        by_va.sort_by_key(|&i| sections[i].virtual_address);
        for w in by_va.windows(2) {
            let (a, b) = (&sections[w[0]], &sections[w[1]]);
            let a_end = u64::from(a.virtual_address) + a.span();
            if a_end > u64::from(b.virtual_address) {
                return Err(format!(
                    "{}: section {} ({}) at {:#x} spans {} bytes and overlaps section {} ({}) at {:#x}: one RVA would have two answers, in a {len}-byte file",
                    path, a.index, a.name, a.virtual_address, a.span(), b.index, b.name,
                    b.virtual_address
                ));
            }
        }

        Ok(PeImage {
            path: path.to_string(),
            file_len: len,
            time_date_stamp,
            machine,
            characteristics,
            subsystem,
            dll_characteristics,
            image_base,
            entry_rva,
            size_of_image,
            size_of_headers,
            section_alignment,
            file_alignment,
            number_of_rva_and_sizes: nrva,
            data_directories,
            sections,
            by_va,
            bytes,
        })
    }

    pub fn section_named(&self, name: &str) -> Option<&PeSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// The section whose declared virtual extent holds `rva`, in VA order.
    pub fn section_of(&self, rva: u32) -> Option<&PeSection> {
        self.by_va
            .iter()
            .find(|&&i| self.sections[i].contains(rva))
            .map(|&i| &self.sections[i])
    }

    /// RVA -> file offset. Containment first, then the file-backed bound: a
    /// section can declare more virtual bytes than it has raw bytes (`.bss`,
    /// and m7.exe's `.data`: VirtualSize 0x1b60 over SizeOfRawData 0xc00). The
    /// bytes past the raw data exist at run time as zeros and NOWHERE IN THIS
    /// FILE, so the mapping refuses them and says which field is short.
    pub fn va_to_off(&self, rva: u32) -> Result<usize, String> {
        let s = self.section_of(rva).ok_or_else(|| {
            format!(
                "{}: RVA {:#x} is in no section: ImageBase {:#x}, SizeOfImage {:#x}, {} sections{}",
                self.path,
                rva,
                self.image_base,
                self.size_of_image,
                self.sections.len(),
                self.section_ranges()
            )
        })?;
        let in_sec = u64::from(rva) - u64::from(s.virtual_address);
        let raw_end = u64::from(s.pointer_to_raw_data) + u64::from(s.size_of_raw_data);
        if u64::from(s.pointer_to_raw_data) + in_sec >= raw_end {
            return Err(format!(
                "{}: RVA {:#x} is in section {} ({}) at VirtualOffset {:#x}, past its SizeOfRawData {} (PointerToRawData {:#x}): the virtual bytes beyond the raw data are zero at run time and are not in this {}-byte file",
                self.path,
                rva,
                s.index,
                s.name,
                in_sec,
                s.size_of_raw_data,
                s.pointer_to_raw_data,
                self.file_len
            ));
        }
        Ok(s.pointer_to_raw_data as usize + in_sec as usize)
    }

    /// `n` bytes at `rva`, from the file. Refuses on any bound: containment,
    /// the section's declared virtual size, its raw size, the file length.
    /// A short read is never silent, and never panics - a hostile
    /// `SizeOfRawData` is the input this function exists to survive.
    pub fn read_at(&self, rva: u32, n: usize) -> Result<&'a [u8], String> {
        let s = self.section_of(rva).ok_or_else(|| {
            format!(
                "{}: read of {n} bytes at RVA {:#x}: no section contains it (ImageBase {:#x}, SizeOfImage {:#x}, {} sections{})",
                self.path,
                rva,
                self.image_base,
                self.size_of_image,
                self.sections.len(),
                self.section_ranges()
            )
        })?;
        let in_sec = u64::from(rva) - u64::from(s.virtual_address);
        let want = n as u64;
        // The raw/file bounds FIRST. At a section's last byte, "past the
        // declared virtual size" and "past the bytes the file has" are both
        // true; the second is the fact the caller can act on (it names the
        // field that promised the bytes), so it is the one that prints.
        let Some(have_raw) = u64::from(s.size_of_raw_data).checked_sub(in_sec) else {
            return Err(format!(
                "{}: read of {n} bytes at RVA {:#x}: section {} ({}) has SizeOfRawData {} at PointerToRawData {:#x} and this RVA is {} bytes into its VirtualAddress {:#x} - past the raw data entirely, in a {}-byte file",
                self.path,
                rva,
                s.index,
                s.name,
                s.size_of_raw_data,
                s.pointer_to_raw_data,
                in_sec,
                s.virtual_address,
                self.file_len
            ));
        };
        if want > have_raw {
            return Err(format!(
                "{}: read of {n} bytes at RVA {:#x}: section {} ({}) has SizeOfRawData {} at PointerToRawData {:#x}, {} of them after this offset, in a {}-byte file",
                self.path,
                rva,
                s.index,
                s.name,
                s.size_of_raw_data,
                s.pointer_to_raw_data,
                have_raw,
                self.file_len
            ));
        }
        let off = s.pointer_to_raw_data as usize + in_sec as usize;
        if want > (self.file_len - off) as u64 {
            return Err(format!(
                "{}: read of {n} bytes at file offset {off:#x} (RVA {:#x}) runs {} bytes past the end of a {}-byte file",
                self.path,
                rva,
                want - (self.file_len - off) as u64,
                self.file_len
            ));
        }
        // Then the section's own declared extent: the raw bytes can be there
        // (.text here has 0x200 of them) and the section still not declare
        // them as live, and a read of dead bytes is not a read of the image.
        if in_sec.checked_add(want).is_none_or(|end| end > s.span()) {
            return Err(format!(
                "{}: read of {n} bytes at RVA {:#x} runs past section {} ({}): VirtualAddress {:#x} + VirtualSize {:#x} ends at {:#x}",
                self.path,
                rva,
                s.index,
                s.name,
                s.virtual_address,
                s.virtual_size,
                u64::from(s.virtual_address) + s.span()
            ));
        }
        Ok(&self.bytes[off..off + n])
    }

    /// `ImageBase + rva`, the address the section table's VMAs are stated in.
    pub fn va(&self, rva: u32) -> u64 {
        self.image_base + u64::from(rva)
    }

    pub fn entry_va(&self) -> u64 {
        self.va(self.entry_rva)
    }

    pub fn subsystem_name(&self) -> &'static str {
        match self.subsystem {
            0 => "UNKNOWN",
            1 => "Native",
            2 => "Windows GUI",
            3 => "Windows CUI",
            5 => "OS/2 CUI",
            7 => "POSIX CUI",
            8 => "Windows Native",
            9 => "Windows GUI driver",
            10 => "Windows console driver",
            11 => "Windows NT driver",
            12 => "Windows CE GUI",
            13 => "OS/2 CUI subsystem",
            14 => "OS/2 character subsystem",
            15 => "Windows-based POSIX",
            16 => "EFI application",
            17 => "EFI boot service driver",
            18 => "EFI runtime driver",
            19 => "EFI ROM",
            20 => "XBOX",
            _ => "UNKNOWN",
        }
    }

    /// The section ranges, for the refusals that need to say what IS mapped.
    fn section_ranges(&self) -> String {
        let mut parts: Vec<String> = self
            .by_va
            .iter()
            .map(|&i| {
                let s = &self.sections[i];
                format!(" {} [{:#x},{:#x})", s.name, s.virtual_address, s.span())
            })
            .collect();
        if parts.len() > 8 {
            let extra = parts.len() - 8;
            parts.truncate(8);
            parts.push(format!(" ... {extra} more"));
        }
        parts.concat()
    }
}

/// `Name[8]`, NUL-trimmed. An 8-char name has no NUL and is kept whole.
fn section_name(raw: &[u8]) -> String {
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).to_string()
}
