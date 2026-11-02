//! Loading COFF objects - hand-rolled, deliberately.
//!
//! goblin 0.10 ships no COFF *object* reader (`src/read/coff` does not exist;
//! only images are covered), and a decompiler whose thesis is "we parse the
//! container you target" cannot outsource that to a crate that does not read
//! the container. The format is 20 + 40n + 18m + 10r bytes; the interesting
//! parts are long names and symbol extents.
//!
//! The extent part matters more than it looks: a COFF `.obj` records **no
//! symbol size**. The only extent derivable from the object is "up to the next
//! symbol in this section", and `SizeSource` exists so that no downstream code
//! can mistake that inference for a declaration.

use std::path::Path;

/// Where a function's byte extent came from. A guessed extent must never be
/// readable as a declared one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeSource {
    /// Derived from the next symbol in the section - the only option for `.obj`.
    NextSymbol,
    /// The section ends the symbol (last symbol in the section).
    SectionEnd,
    /// Known exactly (PE export table, debug info). Not used for `.obj`.
    Exact,
}

#[derive(Clone, Debug)]
pub struct Section {
    pub name: String,
    pub bytes: Vec<u8>,
    /// File offset of `bytes`.
    pub pointer_to_raw_data: u32,
    pub characteristics: u32,
    pub relocations: Vec<Reloc>,
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    /// Index into `Object::sections`, `None` for absolute/undefined/debug.
    pub section: Option<usize>,
    /// Offset within the section.
    pub offset: u64,
    pub size: u64,
    pub size_source: SizeSource,
    pub typ: u16,
    pub storage_class: u8,
    /// Symbol-table index, for relocation lookups.
    pub index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reloc {
    /// Offset within the section.
    pub address: u64,
    pub typ: u16,
    /// Index into `Object::symbols`.
    pub symbol: usize,
}

#[derive(Clone, Debug)]
pub struct Object {
    pub path: String,
    pub machine: u16,
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub time_date_stamp: u32,
}

/// `IMAGE_FILE_MACHINE_AMD64`.
pub const MACHINE_AMD64: u16 = 0x8664;
/// `IMAGE_SYM_CLASS_EXTERNAL`.
pub const STORAGE_EXTERNAL: u8 = 2;
/// `IMAGE_SYM_CLASS_STATIC`.
pub const STORAGE_STATIC: u8 = 3;
/// `IMAGE_REL_AMD64_REL32` - the call/jump relocation.
pub const REL_AMD64_REL32: u16 = 4;
/// `IMAGE_REL_AMD64_ADDR32NB`.
pub const REL_AMD64_ADDR32NB: u16 = 2;

pub fn reloc_type_name(t: u16) -> &'static str {
    match t {
        0 => "ABSOLUTE",
        1 => "ADDR64",
        2 => "ADDR32NB",
        3 => "ADDR32",
        4..=7 => "REL32",
        9 => "SECREL",
        20 => "ADDR32NB",
        _ => "?",
    }
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn short_z(raw: &[u8]) -> String {
    let s = std::str::from_utf8(raw).unwrap_or("?");
    s.split('\0').next().unwrap_or("").to_string()
}

fn name_from_short(raw: &[u8], strings: &[u8]) -> String {
    if raw[..4].iter().all(|&x| x == 0) {
        // Long name: a 4-byte offset into the string table. COFF offsets are
        // relative to the table START (the 4-byte size field occupies offset
        // 0..4), while `strings` is the table BODY - subtract 4.
        // (Measured on clang MSVC-ABI objects: the first entry lives at
        // offset 4; without the subtraction every long name printed
        // 4 characters short, e.g. `ircle@@QEAA@N@Z`.)
        let off = u32le(raw, 4) as usize;
        return strings
            .get(off.saturating_sub(4)..)
            .map(short_z)
            .unwrap_or_else(|| "?".to_string());
    }
    short_z(&raw[..8])
}

fn section_name(raw: &[u8], strings: &[u8]) -> String {
    let s = short_z(raw);
    if let Some(off) = s.strip_prefix('/') {
        // `/123` - offset into the string table.
        if let Ok(o) = off.parse::<usize>() {
            return strings
                .get(o..)
                .map(short_z)
                .unwrap_or_else(|| "?".to_string());
        }
    }
    s
}

impl Object {
    pub fn amd64(&self) -> bool {
        self.machine == MACHINE_AMD64
    }

    pub fn is_code(&self, section: usize) -> bool {
        // CODE (0x20) with EXECUTE (0x20000000).
        let c = self.sections[section].characteristics;
        (c & 0x0000_0020) != 0 && (c & 0x2000_0000) != 0
    }

    /// Symbols that name a location in executable code, in address order.
    pub fn code_symbols(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .symbols
            .iter()
            .filter(|s| {
                s.section.map(|x| self.is_code(x)).unwrap_or(false)
                    && !s.name.starts_with('.')
                    && (s.storage_class == STORAGE_EXTERNAL || s.storage_class == STORAGE_STATIC)
            })
            .map(|s| s.index)
            .collect();
        v.sort_by_key(|&i| (self.symbols[i].section.unwrap(), self.symbols[i].offset, i));
        v
    }

    pub fn symbol_bytes(&self, sym: usize) -> Option<(&[u8], u64)> {
        let s = self.symbols.get(sym)?;
        let sec = self.sections.get(s.section?)?;
        let start = s.offset as usize;
        let end = start + s.size as usize;
        Some((sec.bytes.get(start..end)?, s.offset))
    }

    pub fn relocs_in(&self, section: usize, start: u64, end: u64) -> Vec<Reloc> {
        self.sections
            .get(section)
            .map(|s| {
                s.relocations
                    .iter()
                    .filter(|r| r.address >= start && r.address < end)
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total relocations across all sections - the number behind the roadmap's
    /// "no relocation means no call target" argument.
    pub fn total_relocs(&self) -> usize {
        self.sections.iter().map(|s| s.relocations.len()).sum()
    }
}

/// Read a COFF object file from disk.
pub fn load(path: &Path) -> Result<Object, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    load_bytes(&bytes, &path.display().to_string())
}

/// Read a COFF object from memory.
pub fn load_bytes(bytes: &[u8], path: &str) -> Result<Object, String> {
    let need = |n: usize, what: &str| -> Result<(), String> {
        if bytes.len() < n {
            Err(format!(
                "{}: truncated COFF ({} bytes, need {n} for {what})",
                path,
                bytes.len()
            ))
        } else {
            Ok(())
        }
    };
    need(20, "the file header")?;
    // Machine is at offset 0 in an object header: there is no image signature
    // field, so `PE\0\0`/e_lfanew does not exist here.
    let machine = u16le(bytes, 0);
    let nsec = u16le(bytes, 2) as usize;
    let stamp = u32le(bytes, 4);
    let symoff = u32le(bytes, 8) as usize;
    let nsyms = u32le(bytes, 12) as usize;
    let sizeopt = u16le(bytes, 16) as usize;
    if sizeopt != 0 {
        return Err(format!(
            "{}: object with a {sizeopt}-byte optional header",
            path
        ));
    }
    need(20 + 40 * nsec, "the section headers")?;

    // The string table sits after the symbol table.
    let strings_off = symoff + nsyms * 18;
    let strings: &[u8] = if strings_off + 4 <= bytes.len() {
        let len = u32le(bytes, strings_off) as usize;
        let end = (strings_off + len).min(bytes.len());
        &bytes[strings_off + 4..end]
    } else {
        &[]
    };

    let mut sections = Vec::with_capacity(nsec);
    for i in 0..nsec {
        let h = 20 + 40 * i;
        let name = section_name(&bytes[h..h + 8], strings);
        let raw_size = u32le(bytes, h + 16) as usize;
        let raw_off = u32le(bytes, h + 20) as usize;
        let rel_off = u32le(bytes, h + 24) as usize;
        let nrel = u32le(bytes, h + 32) as usize;
        let chars = u32le(bytes, h + 36);
        let data = if raw_size == 0 || raw_off >= bytes.len() {
            Vec::new()
        } else {
            bytes[raw_off..(raw_off + raw_size).min(bytes.len())].to_vec()
        };
        let mut relocations = Vec::with_capacity(nrel);
        for r in 0..nrel {
            let rpos = rel_off + 10 * r;
            if rpos + 10 > bytes.len() {
                return Err(format!(
                    "{}: section {i} ({name}) relocation table runs off the end",
                    path
                ));
            }
            relocations.push(Reloc {
                address: u32le(bytes, rpos) as u64,
                symbol: u32le(bytes, rpos + 4) as usize,
                typ: u16le(bytes, rpos + 8),
            });
        }
        sections.push(Section {
            name,
            bytes: data,
            pointer_to_raw_data: raw_off as u32,
            characteristics: chars,
            relocations,
        });
    }

    // Symbols, skipping auxiliary entries.
    let mut symbols = Vec::new();
    let mut pos = symoff;
    // `NumberOfSymbols` counts EVERY entry, auxiliary records included, so the
    // loop bound is entries consumed - not main symbols emitted.
    let mut entries = 0usize;
    while entries < nsyms {
        if pos + 18 > bytes.len() {
            return Err(format!("{}: symbol table runs off the end", path));
        }
        let name = name_from_short(&bytes[pos..pos + 8], strings);
        let value = u32le(bytes, pos + 8);
        let secnum = u16le(bytes, pos + 12) as i16;
        let typ = u16le(bytes, pos + 14);
        let storage = bytes[pos + 16];
        let naux = bytes[pos + 17] as usize;
        let section = if (1..=nsec as i16).contains(&secnum) {
            Some((secnum - 1) as usize)
        } else {
            None
        };
        symbols.push(Symbol {
            name,
            section,
            offset: value as u64,
            size: 0,
            size_source: SizeSource::NextSymbol,
            typ,
            storage_class: storage,
            index: symbols.len(),
        });
        entries += 1 + naux;
        pos += 18 * (1 + naux);
    }

    // Extents: the next symbol in the section, or the section end.
    //
    // Section-definition symbols (one per section, named after it, at offset 0)
    // must be excluded: they sit at the same address as the first real
    // function, so an offset-ordered walk that keeps them drops that function
    // and leaves it with size 0. That is exactly how the first version of this
    // loader found "no symbols to disassemble" in a corpus of real objects.
    let definition: Vec<bool> = symbols
        .iter()
        .map(|s| {
            s.storage_class == STORAGE_STATIC
                && s.section.map(|x| sections[x].name == s.name) == Some(true)
        })
        .collect();
    for (si, sec) in sections.iter().enumerate() {
        let mut in_sec: Vec<usize> = (0..symbols.len())
            .filter(|&i| symbols[i].section == Some(si) && !definition[i])
            .collect();
        in_sec.sort_by_key(|&i| symbols[i].offset);
        in_sec.dedup_by_key(|i| symbols[*i].offset);
        for (k, &i) in in_sec.iter().enumerate() {
            match in_sec.get(k + 1) {
                Some(&n) => {
                    symbols[i].size = symbols[n].offset.saturating_sub(symbols[i].offset);
                    symbols[i].size_source = SizeSource::NextSymbol;
                }
                None => {
                    symbols[i].size = (sec.bytes.len() as u64).saturating_sub(symbols[i].offset);
                    symbols[i].size_source = SizeSource::SectionEnd;
                }
            }
        }
    }

    Ok(Object {
        path: path.to_string(),
        machine,
        sections,
        symbols,
        time_date_stamp: stamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-section `.text` object: a section-definition symbol at offset 0 plus
    /// the given functions, section data last.
    fn text_object(text: &[u8], fns: &[(&str, u32)]) -> Vec<u8> {
        let nsec = 1usize;
        // COFF counts EVERY table entry, so the section symbol plus its aux
        // record is two entries, not one.
        let nsyms = 2 + fns.len();
        let symoff = 20 + 40 * nsec;
        let strtab = symoff + 18 * nsyms;
        let data = strtab + 4;
        let mut b = Vec::new();
        b.extend_from_slice(&MACHINE_AMD64.to_le_bytes());
        b.extend_from_slice(&(nsec as u16).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
        b.extend_from_slice(&(symoff as u32).to_le_bytes());
        b.extend_from_slice(&(nsyms as u32).to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
        b.extend_from_slice(&0u16.to_le_bytes()); // Characteristics
        b.extend_from_slice(b".text\0\0\0"); // Name[8]: 5 + 3 NULs
        b.extend_from_slice(&(text.len() as u32).to_le_bytes()); // VirtualSize
        b.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
        b.extend_from_slice(&(text.len() as u32).to_le_bytes()); // SizeOfRawData
        b.extend_from_slice(&(data as u32).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // PointerToRelocations
        b.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
        b.extend_from_slice(&0u16.to_le_bytes()); // NumberOfRelocations
        b.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
        b.extend_from_slice(&0x6050_0020u32.to_le_bytes()); // CODE | EXECUTE

        fn sym(b: &mut Vec<u8>, name: &str, value: u32, typ: u16, storage: u8, naux: u8) {
            let mut n = [0u8; 8];
            n[..name.len()].copy_from_slice(name.as_bytes());
            b.extend_from_slice(&n);
            b.extend_from_slice(&value.to_le_bytes());
            b.extend_from_slice(&1i16.to_le_bytes()); // SectionNumber (1-based)
            b.extend_from_slice(&typ.to_le_bytes());
            b.push(storage);
            b.push(naux);
        }

        sym(&mut b, ".text", 0, 0, STORAGE_STATIC, 1);
        b.extend_from_slice(&[0u8; 18]); // the aux record
        for (name, value) in fns {
            sym(&mut b, name, *value, 0x20, STORAGE_EXTERNAL, 0);
        }
        b.extend_from_slice(&4u32.to_le_bytes()); // string table, size only
        b.extend_from_slice(text);
        b
    }

    #[test]
    fn a_function_sharing_offset_zero_with_its_section_symbol_gets_an_extent() {
        // Every real object has this shape: the section-definition symbol and
        // the first function are both at offset 0. Let the definition win and
        // the function gets size 0 - the first version of this loader then
        // found "no code symbols" in a corpus of real objects and the gate
        // called that OK.
        let text = [0x90u8; 6];
        let o = load_bytes(&text_object(&text, &[("f", 0)]), "one.obj").expect("parses");
        assert!(o.is_code(0), ".text must be code");
        let codes = o.code_symbols();
        assert_eq!(codes.len(), 1, "only the function, not the section symbol");
        let f = &o.symbols[codes[0]];
        assert_eq!(f.name, "f");
        assert_eq!(f.size, 6, "extent must reach the section end, not zero");
        assert_eq!(f.size_source, SizeSource::SectionEnd);
        let (bytes, va) = o.symbol_bytes(codes[0]).expect("bytes");
        assert_eq!(bytes, &text);
        assert_eq!(va, 0);
    }

    #[test]
    fn two_functions_in_one_section_split_at_the_next_symbol() {
        let mut text = vec![0x90u8; 4];
        text.extend_from_slice(&[0xC3u8; 3]);
        let o = load_bytes(&text_object(&text, &[("f", 0), ("g", 4)]), "two.obj").expect("parses");
        let codes = o.code_symbols();
        assert_eq!(codes.len(), 2, "f and g");
        assert_eq!(o.symbols[codes[0]].name, "f");
        assert_eq!(o.symbols[codes[0]].size, 4, "f ends where g starts");
        assert_eq!(o.symbols[codes[0]].size_source, SizeSource::NextSymbol);
        assert_eq!(o.symbols[codes[1]].offset, 4);
        assert_eq!(o.symbols[codes[1]].size, 3);
    }

    #[test]
    fn a_truncated_object_is_a_loud_error_not_a_panic() {
        let mut v = vec![0u8; 8];
        v[0..2].copy_from_slice(&MACHINE_AMD64.to_le_bytes());
        let e = load_bytes(&v, "tiny.obj").unwrap_err();
        assert!(e.contains("truncated"), "{e}");
        v.resize(20, 0);
        v[2..4].copy_from_slice(&9u16.to_le_bytes());
        let e = load_bytes(&v, "tiny.obj").unwrap_err();
        assert!(e.contains("section headers"), "{e}");
    }

    #[test]
    fn an_optional_header_in_an_object_is_refused() {
        let mut v = vec![0u8; 20];
        v[0..2].copy_from_slice(&MACHINE_AMD64.to_le_bytes());
        v[16..18].copy_from_slice(&240u16.to_le_bytes());
        let e = load_bytes(&v, "weird.obj").unwrap_err();
        assert!(e.contains("optional header"), "{e}");
    }
}
