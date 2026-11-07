//! The `.pdata` SEH unwind table as a SYMBOL-FREE function-start source.
//!
//! WHY THIS FILE EXISTS. A linked x64 Windows image carries an
//! `IMAGE_DIRECTORY_ENTRY_EXCEPTION` array of `RUNTIME_FUNCTION` entries —
//! one per function — because the OS exception dispatcher binary-searches it
//! on every throw. The linker MUST write it for the image to function, even
//! when every symbol has been stripped. Measured on the M7 probe artifact:
//! `llvm-nm m7.exe` prints `no symbols` (MSVC link ships the EXE without a
//! COFF symbol table), while its `.pdata` lists 336 entries covering every
//! byte of executable code. For whole-binary discovery this table is not a
//! supplement to symbols; it is the census of what functions exist, and this
//! reader is the mechanism that harvests it.
//!
//! WHAT IS CHECKED, and why each check is loud (the `pe.rs` convention:
//! refusal names field, entry, got/expected, offset, file length):
//! - the directory size is an exact multiple of the 12-byte entry stride — a
//!   remainder means the array boundary and the file disagree about where the
//!   table ends;
//! - `BeginAddress` strictly ascends the array — the loader BINARY-SEARCHES
//!   it, so an unsorted array is a broken table, and the refusal names the
//!   out-of-order PAIR;
//! - every entry: `BeginAddress` in an executable section, `EndAddress >
//!   BeginAddress` inside the same section, `UnwindData` 4-aligned and inside
//!   readable bytes, and the `UNWIND_INFO` header (and its whole code array)
//!   inside the file — the checks the Windows loader skips, because a
//!   decompiler's "function at VA X" is only worth printing if the bytes say
//!   so.
//! - `UNWIND_INFO.Version` must be 1. Version 2 exists in the wild —
//!   measured on `m7.exe`: four CET-hardened CRT guard-thunk entries at
//!   BeginAddress 0xe8e0/0xe9f0/0xefb0/0xf3f0 — and its code-array layout is
//!   NOT this layout, so reading it as version 1 would not fail, it would
//!   shift every code byte into a plausible-looking code array. A version
//!   this reader has not read the layout of is refused AS A FUNCTION,
//!   PER ENTRY, named and COUNTED (the chaining shape): the refusal is
//!   about the UNWIND_INFO, and the code array is never read — but the
//!   whole table is not the property of one entry. The RUNTIME_FUNCTION
//!   record itself (Begin/End/UnwindData, sortedness, bounds) is version-
//!   independent and stays checked table-wide; one v2 CRT thunk must not
//!   throw away the 332 v1 function starts a stripped EXE's only
//!   start-source carries. Modelling v2 code arrays is a later slice; the
//!   refused entries stay named until then.
//!
//! WHAT IS PARSED AND NOT MODELLED, deliberately: the code array
//! (`CountOfCodes` x 2 bytes: PushNonvol, AllocLarge/Small, SaveXmm, SetFp,
//! SaveNonvol, Epilog, EpilogDce, ...) is stored RAW — never interpreted,
//! never refused-for-being-unknown. The frame model built from these codes is
//! the NEXT slice; this slice's output is the function census plus a flags /
//! first-opcode census, and a census must count everything it sees.
//! `UNW_FLAG_CHAINING` is the census's cardinal case: a chained entry's
//! BeginAddress is a nested region of the outer function (a /EHsc scope), not
//! a function start — so it is REFUSED as a function, LOUDLY, named, and
//! counted. Dropping it silently would invent gaps in the function map;
//! keeping it would invent functions. Both are the same sin.
//!
//! THE FLAGS ARE THE 5-BIT FIELD (bits 3-7 of the header u32):
//! `UNW_FLAG_CHAINING` is `0x4` of that field, i.e. bit `0x20` of the u32.

use crate::pe::PeImage;

/// The data-directory index of the exception (`.pdata`) table.
pub const EXCEPTION_DIR: usize = 3;
/// `sizeof(RUNTIME_FUNCTION)`, the stride of the array.
pub const RUNTIME_FUNCTION_LEN: usize = 12;
/// The only `UNWIND_INFO.Version` whose layout this reader knows.
pub const UNWIND_INFO_VERSION: u8 = 1;
/// `UNW_FLAG_EHANDLER` (of the 5-bit Flags field): an exception-handler RVA
/// follows the code array. Recorded, not followed — the handler is slice N+1.
pub const UNW_FLAG_EHANDLER: u8 = 0x1;
/// `UNW_FLAG_UHANDLER`: a termination-handler RVA follows. Recorded, not followed.
pub const UNW_FLAG_UHANDLER: u8 = 0x2;
/// `UNW_FLAG_CHAINING`: the entry is a nested scope of an outer function.
/// Refused as a function start, loudly and by count.
pub const UNW_FLAG_CHAINING: u8 = 0x4;
/// `MEM_READ`: the section characteristic that makes `UnwindData` legal.
const MEM_READ: u32 = 0x4000_0000;
/// `MEM_EXECUTE`: chained with READ by the check, so an `.xdata` merged into
/// an executable section is accepted the way the loader accepts it.
const MEM_EXECUTE: u32 = 0x2000_0000;

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// One parsed `RUNTIME_FUNCTION` + `UNWIND_INFO` header, all fields read back
/// from the file, code array stored raw.
#[derive(Clone, Debug)]
pub struct UnwindEntry {
    /// Position in the (sorted) array — what a refusal and the CLI name.
    pub index: usize,
    /// `BeginAddress`, an RVA: a function start. The discovery source.
    pub begin_rva: u32,
    /// `EndAddress`, an RVA, exclusive.
    pub end_rva: u32,
    /// `UnwindData`, an RVA into readable bytes: the `UNWIND_INFO`.
    pub unwind_rva: u32,
    /// Version, bits 0-2. Checked: only [`UNWIND_INFO_VERSION`] parses.
    pub version: u8,
    /// Flags, bits 3-7: [`UNW_FLAG_EHANDLER`] / [`UNW_FLAG_UHANDLER`] bits
    /// say a handler RVA follows the codes; this slice records, never follows.
    pub flags: u8,
    /// `SizeOfProlog`. Stored; the prolog it counts is modelled by later slices.
    pub size_of_prolog: u8,
    /// `CountOfCodes`: the length of `codes`, whose bytes are in the file.
    pub count_of_codes: u8,
    /// `FrameRegister`, bits 24-27 (4 bits, x64): 0 = none, 5 = RBP.
    pub frame_register: u8,
    /// `FrameOffset`, bits 28-31 (4 bits): frame = offset * 16 bytes.
    pub frame_offset: u8,
    /// The raw `UnwindCode` array: each u16 is (offset byte, opcode byte),
    /// opcode = low nibble, info = high nibble. Stored, never interpreted.
    pub codes: Vec<u16>,
}

impl UnwindEntry {
    /// The first code's OPCODE nibble — the census key that answers "how do
    /// this image's functions open their frames" (0 = PushNonvol, 4 = SetFp,
    /// ...) without interpreting any of them. `None` = `CountOfCodes` 0,
    /// counted as a distinct census bucket, never dropped.
    pub fn first_code(&self) -> Option<u8> {
        self.codes.first().map(|&c| ((c >> 8) & 0xF) as u8)
    }
}

/// A `RUNTIME_FUNCTION` refused as a function start because its
/// `UNWIND_INFO` sets `UNW_FLAG_CHAINING`. It is NOT a function: its
/// BeginAddress names a nested region of an outer function. It is refused,
/// named, and COUNTED — a chained entry that vanished from the census would
/// turn "N functions" into a number that quietly lies.
#[derive(Clone, Debug)]
pub struct ChainedEntry {
    /// Position in the array.
    pub index: usize,
    /// The entry's `BeginAddress` RVA — a region start, not a function start.
    pub begin_rva: u32,
    /// The entry's `EndAddress` RVA.
    pub end_rva: u32,
    /// The chained `UNWIND_INFO` RVA.
    pub unwind_rva: u32,
    /// The Flags field that refused it (bit [`UNW_FLAG_CHAINING`] set).
    pub flags: u8,
}

impl ChainedEntry {
    /// The refusal, naming the flag and the entry — the same text the CLI
    /// prints and the fixture asserts on.
    pub fn message(&self) -> String {
        format!(
            "entry {} (BeginAddress {:#x}, EndAddress {:#x}): UNW_FLAG_CHAINING is set in its UNWIND_INFO at {:#x} — a chained entry names a nested scope of an outer function, not a function start; refused as a function, counted, not dropped",
            self.index, self.begin_rva, self.end_rva, self.unwind_rva
        )
    }
}

/// A `RUNTIME_FUNCTION` whose `UNWIND_INFO` carries a `Version` this reader
/// has not read the layout of. The code array is NOT parsed (reading a 2 as
/// a 1 shifts every code byte into a plausible-looking code array), so the
/// entry is refused AS A FUNCTION, named, COUNTED — the chaining shape. It
/// is not dropped: four v2 CRT thunks hidden inside "332 functions" would
/// turn the census into a quiet lie, and four v2 entries thrown away with
/// the whole table would throw out 332 true function starts with them.
#[derive(Clone, Debug)]
pub struct VersionEntry {
    /// Position in the array.
    pub index: usize,
    /// The entry's `BeginAddress` RVA.
    pub begin_rva: u32,
    /// The entry's `EndAddress` RVA.
    pub end_rva: u32,
    /// The `UNWIND_INFO` RVA whose version refused it.
    pub unwind_rva: u32,
    /// The version that was found.
    pub version: u8,
}

impl VersionEntry {
    /// The refusal, naming the version, the entry, and the offset — the text
    /// the CLI prints and the fixture asserts on.
    pub fn message(&self, img: &PeImage) -> String {
        let off = img
            .va_to_off(self.unwind_rva)
            .map(|o| format!("{o:#x}"))
            .unwrap_or_else(|_| "unmapped".to_string());
        format!(
            "entry {} (BeginAddress {:#x}, EndAddress {:#x}): UNWIND_INFO Version at rva {:#x} (file offset {off}) is {}, expected 1 (1): the code-array layout is defined for version 1 and is NOT read for {} — a 2 read as a 1 shifts every code byte into a plausible-looking code array; refused as a function, counted, not dropped",
            self.index, self.begin_rva, self.end_rva, self.unwind_rva, self.version, self.version
        )
    }
}

/// The whole EXCEPTION-directory table: the parsed functions, the chained
/// refusals beside them, and the census of what the codes look like.
#[derive(Clone, Debug)]
pub struct UnwindTable {
    /// The image's path, carried into every refusal.
    pub path: String,
    /// The EXCEPTION directory's RVA as the header states it.
    pub dir_rva: u32,
    /// Its size; `entry_count()` is this divided by the stride, exactly.
    pub dir_size: u32,
    entries: Vec<UnwindEntry>,
    chained: Vec<ChainedEntry>,
    versioned: Vec<VersionEntry>,
}

impl UnwindTable {
    /// Read the `IMAGE_DIRECTORY_ENTRY_EXCEPTION` array. Every refusal names
    /// the field, the entry (index + BeginAddress), the value found, the
    /// value expected, the offset, and the file length.
    pub fn read(img: &PeImage) -> Result<UnwindTable, String> {
        let dir = img
            .data_directories
            .get(EXCEPTION_DIR)
            .filter(|d| d.rva != 0 && d.size != 0)
            .ok_or_else(|| {
                let (r, s) = img
                    .data_directories
                    .get(EXCEPTION_DIR)
                    .map(|d| (d.rva, d.size))
                    .unwrap_or((0, 0));
                format!(
                    "{}: no .pdata: the EXCEPTION data directory (index {EXCEPTION_DIR}) is empty (rva {r:#x}, size {s:#x}). The image declares no SEH unwind table, so this front end has no symbol-free function-start source in a {}-byte file",
                    img.path, img.file_len
                )
            })?;
        if dir.size % RUNTIME_FUNCTION_LEN as u32 != 0 {
            return Err(format!(
                "{}: EXCEPTION directory at rva {:#x} has size {}, which is not a multiple of the {RUNTIME_FUNCTION_LEN}-byte RUNTIME_FUNCTION stride ({} byte{} of remainder): the array boundary and the directory disagree about where .pdata ends, in a {}-byte file",
                img.path,
                dir.rva,
                dir.size,
                dir.size % RUNTIME_FUNCTION_LEN as u32,
                if dir.size % RUNTIME_FUNCTION_LEN as u32 == 1 { "" } else { "s" },
                img.file_len
            ));
        }
        // The last declared table byte must exist in the file BEFORE the
        // arrays are sized: `dir_size` is a hostile u32, and reserving
        // `dir_size / 12` entries up front is the 0xFFFF-section-count
        // allocation trap from `pe.rs`, in a new costume. One byte-read at
        // the end of the directory bounds every later allocation by the file.
        let last = dir.rva.checked_add(dir.size - 1).ok_or_else(|| {
            format!(
                "{}: EXCEPTION directory at rva {:#x} has size {:#x}, which overflows the RVA space, in a {}-byte file",
                img.path, dir.rva, dir.size, img.file_len
            )
        })?;
        img.read_at(last, 1).map_err(|e| {
            format!(
                "{}: EXCEPTION directory at rva {:#x} size {:#x} declares {} entries of {RUNTIME_FUNCTION_LEN} bytes, but its last byte {last:#x} is not in the file, in a {}-byte file: {e}",
                img.path,
                dir.rva,
                dir.size,
                dir.size as usize / RUNTIME_FUNCTION_LEN,
                img.file_len
            )
        })?;
        let count = dir.size as usize / RUNTIME_FUNCTION_LEN;

        // Pass 1: the 12-byte entries. `read_at` bounds each against the
        // section's raw data and the file, so a directory that promises more
        // table than the bytes carry refuses naming the field that promised.
        let mut rfs: Vec<(u32, u32, u32)> = Vec::with_capacity(count);
        for i in 0..count {
            let at = dir
                .rva
                .checked_add((i * RUNTIME_FUNCTION_LEN) as u32)
                .ok_or_else(|| {
                    format!(
                        "{}: entry {i} of the EXCEPTION array at rva {:#x} overflows the RVA space in a {}-byte file",
                        img.path, dir.rva, img.file_len
                    )
                })?;
            let raw = img.read_at(at, RUNTIME_FUNCTION_LEN).map_err(|e| {
                format!(
                    "{}: RUNTIME_FUNCTION entry {i} at rva {at:#x}: {e}",
                    img.path
                )
            })?;
            rfs.push((u32le(raw, 0), u32le(raw, 4), u32le(raw, 8)));
        }

        // Sortedness FIRST: the loader binary-searches this array, so one
        // backward pair poisons every lookup, and the pair is the fixable
        // fact — the refusal names both entries, indices and values.
        for (i, w) in rfs.windows(2).enumerate() {
            if w[1].0 <= w[0].0 {
                return Err(format!(
                    "{}: the .pdata RUNTIME_FUNCTION array is not sorted by BeginAddress: entry {i} BeginAddress {:#x} does not precede entry {} BeginAddress {:#x} (the loader binary-searches this array), in a {}-byte file",
                    img.path,
                    w[0].0,
                    i + 1,
                    w[1].0,
                    img.file_len
                ));
            }
        }

        // Pass 2: per-entry, in array order — the first refusal is the first
        // liar, and its text names it.
        let mut entries: Vec<UnwindEntry> = Vec::with_capacity(count);
        let mut chained: Vec<ChainedEntry> = Vec::new();
        let mut versioned: Vec<VersionEntry> = Vec::new();
        for (i, &(begin, end, unwind)) in rfs.iter().enumerate() {
            let sec = img.section_of(begin).ok_or_else(|| {
                format!(
                    "{}: RUNTIME_FUNCTION entry {i} BeginAddress {begin:#x} is in no section: it names a function at an address the image does not have, in a {}-byte file",
                    img.path, img.file_len
                )
            })?;
            if !sec.is_code() {
                return Err(format!(
                    "{}: RUNTIME_FUNCTION entry {i} BeginAddress {begin:#x} lies in section {} ({}), which is not executable (characteristics {:#x}): .pdata entries describe functions, and functions live in executable sections, in a {}-byte file",
                    img.path,
                    sec.index,
                    sec.name,
                    sec.characteristics,
                    img.file_len
                ));
            }
            if end <= begin {
                return Err(format!(
                    "{}: RUNTIME_FUNCTION entry {i}: EndAddress {end:#x} <= BeginAddress {begin:#x}: the function's extent is empty or inverted, in a {}-byte file",
                    img.path, img.file_len
                ));
            }
            let sec_end = u64::from(sec.virtual_address) + sec.span();
            if u64::from(end) > sec_end {
                return Err(format!(
                    "{}: RUNTIME_FUNCTION entry {i}: EndAddress {end:#x} runs past section {} ({}), which ends at {sec_end:#x}: the function's extent leaves the section its BeginAddress is in, in a {}-byte file",
                    img.path, sec.index, sec.name, img.file_len
                ));
            }
            if unwind % 4 != 0 {
                return Err(format!(
                    "{}: RUNTIME_FUNCTION entry {i} (BeginAddress {begin:#x}): UnwindData {unwind:#x} is not 4-byte aligned: UNWIND_INFO begins on a DWORD boundary, in a {}-byte file",
                    img.path, img.file_len
                ));
            }
            let usec = img.section_of(unwind).ok_or_else(|| {
                format!(
                    "{}: RUNTIME_FUNCTION entry {i} (BeginAddress {begin:#x}): UnwindData {unwind:#x} is in no section, in a {}-byte file",
                    img.path, img.file_len
                )
            })?;
            if usec.characteristics & (MEM_READ | MEM_EXECUTE) == 0 {
                return Err(format!(
                    "{}: RUNTIME_FUNCTION entry {i} (BeginAddress {begin:#x}): UnwindData {unwind:#x} lies in section {} ({}), which is not readable (characteristics {:#x}): the unwind info has to be readable to be walked, in a {}-byte file",
                    img.path,
                    usec.index,
                    usec.name,
                    usec.characteristics,
                    img.file_len
                ));
            }
            let hdr = img.read_at(unwind, 4).map_err(|e| {
                format!(
                    "{}: entry {i} (BeginAddress {begin:#x}): UnwindData {unwind:#x} — reading the 4-byte UNWIND_INFO header runs past the bytes the image has, in a {}-byte file: {e}",
                    img.path, img.file_len
                )
            })?;
            let w = u32le(hdr, 0);
            let version = (w & 7) as u8;
            if version != UNWIND_INFO_VERSION {
                // Refused AS A FUNCTION, named, counted, never dropped — and
                // never read. One v2 CRT thunk is that entry's property, not
                // the table's: the remaining starts are true starts.
                versioned.push(VersionEntry {
                    index: i,
                    begin_rva: begin,
                    end_rva: end,
                    unwind_rva: unwind,
                    version,
                });
                continue;
            }
            let flags = ((w >> 3) & 0x1F) as u8;
            let size_of_prolog = ((w >> 8) & 0xFF) as u8;
            let count_of_codes = ((w >> 16) & 0xFF) as u8;
            let frame_register = ((w >> 24) & 0xF) as u8;
            let frame_offset = ((w >> 28) & 0xF) as u8;
            if flags & UNW_FLAG_CHAINING != 0 {
                // Refused AS A FUNCTION, named, counted, never dropped.
                chained.push(ChainedEntry {
                    index: i,
                    begin_rva: begin,
                    end_rva: end,
                    unwind_rva: unwind,
                    flags,
                });
                continue;
            }
            // The code array, bounded to the bytes the image has. Counted
            // and stored raw; NO code is interpreted, and no code the codes
            // use is refused — modelling them is the next slice.
            let mut codes: Vec<u16> = Vec::with_capacity(count_of_codes as usize);
            if count_of_codes > 0 {
                let codes_rva = unwind
                    .checked_add(4)
                    .ok_or_else(|| format!(
                        "{}: entry {i} (BeginAddress {begin:#x}): UnwindData {unwind:#x} + 4 for CountOfCodes overflows the RVA space, in a {}-byte file",
                        img.path, img.file_len
                    ))?;
                let raw = img.read_at(codes_rva, 2 * count_of_codes as usize).map_err(|e| {
                    format!(
                        "{}: entry {i} (BeginAddress {begin:#x}): CountOfCodes {count_of_codes} code entries ({} bytes) at UnwindData+4 ({codes_rva:#x}) do not fit the bytes the image has, in a {}-byte file: {e}",
                        img.path,
                        2 * count_of_codes as usize,
                        img.file_len
                    )
                })?;
                for c in raw.as_chunks::<2>().0 {
                    codes.push(u16::from_le_bytes([c[0], c[1]]));
                }
            }
            entries.push(UnwindEntry {
                index: i,
                begin_rva: begin,
                end_rva: end,
                unwind_rva: unwind,
                version,
                flags,
                size_of_prolog,
                count_of_codes,
                frame_register,
                frame_offset,
                codes,
            });
        }

        Ok(UnwindTable {
            path: img.path.clone(),
            dir_rva: dir.rva,
            dir_size: dir.size,
            entries,
            chained,
            versioned,
        })
    }

    /// `dir_size / 12`: every entry the DIRECTORY declares, parsed or refused.
    pub fn entry_count(&self) -> usize {
        self.dir_size as usize / RUNTIME_FUNCTION_LEN
    }

    /// The parsed entries — the census's numerator. Chained refusals are NOT
    /// in here; they are in [`UnwindTable::chaining_refusals`], counted.
    pub fn entries(&self) -> &[UnwindEntry] {
        &self.entries
    }

    /// THE DISCOVERY SOURCE: `(begin_rva, end_rva)` for every function the
    /// table proves exists. Whole-binary discovery enumerates these.
    pub fn functions(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.entries.iter().map(|e| (e.begin_rva, e.end_rva))
    }

    /// The chained entries, refused as function starts. The count is part of
    /// the API contract: `functions().count() + chaining_refusals().len()
    /// + version_refusals().len()` equals `entry_count()` for every image
    /// this reader accepts.
    pub fn chaining_refusals(&self) -> &[ChainedEntry] {
        &self.chained
    }

    /// Entries whose `UNWIND_INFO` version this reader has not read the
    /// layout of: refused as function starts, named, counted, never read.
    pub fn version_refusals(&self) -> &[VersionEntry] {
        &self.versioned
    }

    /// `Flags` field -> count, ascending by flags: the census line the CLI
    /// prints, and the map of what later slices must model (EHANDLER/UHANDLER
    /// entries carry handler RVAs; CHAINING lives in the refusal list).
    pub fn flags_census(&self) -> Vec<(u8, usize)> {
        let mut map: std::collections::BTreeMap<u8, usize> = std::collections::BTreeMap::new();
        for e in &self.entries {
            *map.entry(e.flags).or_insert(0) += 1;
        }
        map.into_iter().collect()
    }

    /// First-code opcode -> count, `None` (zero codes) sorted LAST so it is
    /// read as "and this many have none", never as a code 0.
    pub fn first_code_census(&self) -> Vec<(Option<u8>, usize)> {
        let mut map: std::collections::BTreeMap<Option<u8>, usize> =
            std::collections::BTreeMap::new();
        for e in &self.entries {
            *map.entry(e.first_code()).or_insert(0) += 1;
        }
        let mut v: Vec<(Option<u8>, usize)> = map.into_iter().collect();
        v.sort_by_key(|(k, _)| (k.is_none(), k.unwrap_or(0)));
        v
    }
}
