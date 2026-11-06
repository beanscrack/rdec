//! The fuzz surface: generators, a panic-catching driver, and a census.
//!
//! Why this lives in the front-end crate: `rdec fuzz` (the gate, in
//! `crates/cli/src/fuzz.rs`) and `crates/arch-x86/tests/fuzz.rs` (the bounded
//! CI run) must exercise the SAME generator and the SAME panic-catching driver.
//! Two generator implementations means the gate finds crashes the CI test
//! cannot represent, which is the worst possible split: the reproducer in
//! `tests/fuzz_findings.rs` is written from the gate's output and has to be
//! runnable from the test.
//!
//! The two generators, both seeded from one `--seed` so any run is reproducible:
//!
//! * **byte-level** (`gen == "bytes"`): pure random bytes, and mutations of
//!   REAL corpus objects - bit flips, byte splices, truncations, and edits of
//!   the exact header fields the COFF parser trusts: `NumberOfSections`,
//!   `SizeOfOptionalHeader`, `PointerToSymbolTable`, `NumberOfSymbols`,
//!   `StringTableSize`, and per section `VirtualSize`, `SizeOfRawData`,
//!   `PointerToRawData`, `PointerToRelocations`, `NumberOfRelocations`,
//!   `Characteristics`, plus per symbol `SectionNumber` and `NumberOfAuxiliary`.
//!   Random bytes reach a parser's front door; field edits reach the thin part
//!   behind it, where a value is trusted as a count or an offset.
//! * **instruction-level** (`gen == "insn"`): random byte streams handed
//!   straight to the decoder, plus mutated copies of real corpus `.text`,
//!   embedded in a well-formed object so the whole pipeline sees them.
//!   The decoder's boundary contract is asserted with `debug_assert!` in
//!   `decode.rs`, and `cargo test` runs debug - so the CI test fires them.
//!   A release build compiles them out: `FuzzStats::debug_checks` says which
//!   build the census came from, because a green release fuzz run is a weaker
//!   claim than a green debug one and must not be reported as the same thing.
//!
//! Panics are caught, per input, with the input retained (`Crash`). The driver
//! never aborts on the first crash: the census wants the population, not the
//! first specimen. What it CANNOT catch is said plainly - a stack overflow, an
//! allocation failure, or an unbounded loop escape `catch_unwind`, so every
//! loop the generators can steer must be bounded by the input length (the
//! production rule the loader/discover follow), and `--max-seconds` is the
//! between-iteration guard that turns a hang into a non-zero exit.

use std::collections::BTreeMap;
use std::panic;
use std::time::{Duration, Instant};

/// Deterministic, seedable, no dependencies: splitmix64. The fuzzer's
/// reproducibility claim rests on this being a pure function of the seed, so
/// the seed and the iteration number together name every input.
#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` for the small `n` the generators use (a byte, an
    /// offset, a mutation kind). Modulo bias at n < 2^32 is irrelevant here:
    /// what matters is that the sequence is reproducible.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % (n as u64)) as usize
        }
    }
    pub fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
}

/// The values a "hostile field editor" writes. Every one of these is a value a
/// real field can hold and a parser must survive: zero, the maximum, the
/// maximum-minus-one (the off-by-one that beats an unsigned `>` guard), a
/// value past the end of the file, and a value that overflows when multiplied
/// by a record size.
pub const NASTY: [u64; 12] = [
    0,
    1,
    2,
    0x0F,
    0xFF,
    0xFFFF,
    0xFFFF_FFFF,
    0x7FFF_FFFF,
    0x8000_0000,
    0x00FF_FFFF,
    0x4000_0000,
    0xFFFF_FFFF,
];

/// What the per-input ceiling does when it fires: it MEASURES a hang, it
/// cannot interrupt one. Every message about an over-budget input says so,
/// because the tempting reading of a green `slow.over_budget = 0` is "hangs are
/// fixed", and they are not - the algorithms are bounded, which is the fix.
pub const OVER_BUDGET_NOTE: &str =
    "a decompiler that needs this long for one object is hung for every user of
     it; the per-input ceiling only measures it, so the algorithm it landed in must
     be bounded (and the ceiling cannot abort a running input)";

/// One generated input. `gen` names the generator so the census can say which
/// family found a crash, and `iter`/`seed` make the input addressable: the
/// report can print "seed 7, iteration 4121, gen bytes" and a reader can
/// regenerate the exact bytes.
#[derive(Clone, Debug)]
pub struct Input {
    pub seed: u64,
    pub iter: u64,
    pub gen: &'static str,
    pub detail: String,
    pub bytes: Vec<u8>,
}

impl Input {
    /// The comment a fixture gets: the provenance a byte array cannot carry.
    pub fn provenance(&self) -> String {
        format!(
            "seed {}, iteration {}, generator {} ({})",
            self.seed, self.iter, self.gen, self.detail
        )
    }
}

/// What one pipeline stage counts. Everything is a named counter, because the
/// census is the deliverable: "0 panics" is only meaningful next to "312
/// inputs reached the interpreter".
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub iterations: u64,
    pub inputs: BTreeMap<&'static str, u64>,
    /// Stage -> count. `load.ok`, `load.err`, `demangle.name`,
    /// `demangle.unsupported`, `discover.functions`, `lift.functions`,
    /// `verify.ok`, `verify.reject`, `emit.c`, `emit.refused`, `interp.run`,
    /// `decode.insn`, `decode.refused` - the per-stage census.
    pub stages: BTreeMap<String, u64>,
    /// Load errors by class, named from the loader's own message.
    pub load_errors: BTreeMap<String, u64>,
    /// Counted gaps (discovery/lifter) by class.
    pub gaps: BTreeMap<String, u64>,
    /// Stage -> slowest single input, in ms. A census that counts instructions
    /// cannot tell you where 65 seconds went; this can.
    pub stage_ms: BTreeMap<String, u64>,
    /// The slowest input of the run, kept so a `wall.slowest_input_ms` census
    /// line is a pointer to bytes and not a shrug.
    pub slowest: Option<Input>,
    /// The slowest per-stage time, as words: the pipeline stage that ate the
    /// 65 seconds, named.
    pub slow_note: String,
    pub crashes: Vec<Crash>,
    /// Bytes generated in total: the coverage number for a run that found
    /// nothing.
    pub bytes_total: u64,
    pub max_input: usize,
    /// True when `debug_assert!`s were live for this run (a debug build).
    pub debug_checks: bool,
    pub elapsed: Duration,
    /// Iterations that took the longest, wall-clock: the number that makes a
    /// quadratic-in-file-size parser visible before it is a CI timeout.
    pub slowest_ms: u128,
}

impl Stats {
    /// Record how long a stage took on this input. A census that counts
    /// instructions cannot tell you where 65 seconds went; a census that also
    /// carries the slowest input per stage can.
    pub fn time(&mut self, stage: &str, ms: u64) {
        let e = self.stage_ms.entry(stage.to_string()).or_insert(0);
        *e = (*e).max(ms);
    }

    pub fn bump(&mut self, stage: &str, n: u64) {
        *self.stages.entry(stage.to_string()).or_insert(0) += n;
    }
    pub fn get(&self, stage: &str) -> u64 {
        self.stages.get(stage).copied().unwrap_or(0)
    }
    pub fn gap(&mut self, kind: &str, n: u64) {
        *self.gaps.entry(kind.to_string()).or_insert(0) += n;
    }
    /// Total crash count, including ones beyond the retained cap.
    pub fn crash_count(&self) -> usize {
        self.crashes.len()
    }
}

/// One caught panic, with the input that caused it.
#[derive(Clone, Debug)]
pub struct Crash {
    pub input: Input,
    pub stage: String,
    pub message: String,
}

impl Crash {
    /// The byte literal a fixture embeds, formatted for Rust source.
    pub fn bytes_as_rust(&self) -> String {
        let mut s = String::new();
        for (i, b) in self.input.bytes.iter().enumerate() {
            if i % 16 == 0 {
                s.push_str("\n    ");
            }
            s.push_str(&format!("0x{b:02x}, "));
        }
        s
    }
}

/// The config for one run.
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub iterations: u64,
    /// Between-iteration wall-clock guard: a hang inside one input is a
    /// production bug, and this turns it into a non-zero exit plus the last
    /// input, not into a 60-minute CI timeout.
    ///
    /// It is checked BETWEEN inputs, so it cannot interrupt a running one. That
    /// is why `max_input_ms` exists as well: the input that takes minutes is
    /// caught by the per-input ceiling, named, and its bytes kept.
    pub max_seconds: u64,
    /// Per-input ceiling. An input over it is a FINDING (`slow.over_budget`),
    /// recorded with its bytes - a decompiler that needs minutes for a 4 KB
    /// object is hung for every purpose that matters.
    pub max_input_ms: u64,
    /// One line per input, written and flushed BEFORE it runs, so the tail of
    /// stderr names the input a stalled run is stuck on.
    pub trace: bool,
    /// Cap on retained crashes (each one carries its bytes).
    pub keep_crashes: usize,
    /// Max generated object size, so a corpus object cannot be mutated into a
    /// 4 GiB allocation by a size field.
    pub max_input_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 0x5DEE_71CE,
            iterations: 3000,
            max_seconds: 240,
            // 2 s. The census's own slowest debug input is ~60 ms, so this is
            // ~30x the observed worst: a 4 KB object has no honest work to do
            // that takes 2 s, and an input over it is a finding, not a flake.
            max_input_ms: 2_000,
            trace: false,
            keep_crashes: 64,
            max_input_bytes: 1 << 20,
        }
    }
}

/// A corpus seed: the object bytes, and the `.text` bytes pulled out of it.
pub struct Seed {
    pub name: String,
    pub obj: Vec<u8>,
    pub text: Vec<u8>,
    /// Decorated (`?`-prefixed) symbol names the object carries. The name
    /// generator seeds itself with the shapes real clang emits: the mutation
    /// space around a real decorated name - truncate it, flip a byte in it -
    /// is the space a hostile `.obj`'s string table actually explores, and
    /// random bytes over a 64-letter alphabet rarely land in it.
    pub names: Vec<String>,
}

/// The `.text` of a loaded object (the first executable section), for the
/// instruction-level generator.
pub fn text_of(obj: &crate::loader::Object) -> Vec<u8> {
    obj.sections
        .iter()
        .enumerate()
        .find(|(i, _)| obj.is_code(*i))
        .map(|(_, s)| s.bytes.clone())
        .unwrap_or_default()
}

/// Build seeds from raw object bytes, loading each to pull out `.text`.
/// A corpus object that will not load is not a seed: the byte-level generator
/// mutates the RAW bytes, so an unparseable seed would silently shrink the
/// corpus, and `skipped` is the loud count of that.
pub fn seeds_from_raw(raw: &[(String, Vec<u8>)]) -> (Vec<Seed>, Vec<String>) {
    let mut seeds = Vec::new();
    let mut skipped = Vec::new();
    for (name, bytes) in raw {
        match crate::loader::load_bytes(bytes, name) {
            Ok(o) => seeds.push(Seed {
                name: name.clone(),
                obj: bytes.clone(),
                text: text_of(&o),
                names: o
                    .symbols
                    .iter()
                    .map(|s| s.name.clone())
                    .filter(|n| n.starts_with('?'))
                    .collect(),
            }),
            Err(e) => skipped.push(format!("{name}: {e}")),
        }
    }
    (seeds, skipped)
}

/// A `u32` at an offset, 0 if the file is too short to have it. The mutator
/// reads COFF header fields before it edits them, and a mutator that panics on
/// a truncated seed stops the census.
fn u32_at(b: &[u8], at: usize) -> u32 {
    if at + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The `nasty` value for a field of `bytes` width, truncated to that width.
fn nasty_for(rng: &mut Rng, width: usize) -> u64 {
    let v = NASTY[rng.below(NASTY.len())];
    match width {
        1 => v & 0xFF,
        2 => v & 0xFFFF,
        4 => v & 0xFFFF_FFFF,
        _ => v,
    }
}

fn put(b: &mut [u8], at: usize, width: usize, val: u64) {
    // Every write is bounds-checked: a generator that panics is a generator
    // that stops the census, and the generators are not the code under test.
    let end = at + width;
    if at >= b.len() || end > b.len() {
        return;
    }
    for (k, slot) in b[at..end].iter_mut().enumerate() {
        *slot = (val >> (8 * k)) as u8;
    }
}

/// One header field the COFF parser reads as a count, an offset, or a size.
/// `Copy`: the mutator copies the field out before it writes, so a borrow of
/// the buffer it is about to edit cannot overlap the edit.
#[derive(Clone, Copy)]
pub struct Field {
    pub name: &'static str,
    pub at: usize,
    pub width: usize,
}

/// Every file/section/symbol field the loader trusts, at its real offset in
/// these bytes. Empty for input too short to have headers - the random-bytes
/// generator does not need it, the mutator does.
pub fn coff_fields(b: &[u8]) -> Vec<Field> {
    let mut v = Vec::new();
    let rd16 = |o: usize| -> u64 {
        b.get(o..o + 2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]) as u64)
            .unwrap_or(0)
    };
    let rd32 = |o: usize| -> u64 {
        b.get(o..o + 4)
            .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as u64)
            .unwrap_or(0)
    };
    if b.len() < 20 {
        return v;
    }
    for (name, at, w) in [
        ("NumberOfSections", 2usize, 2usize),
        ("PointerToSymbolTable", 8, 4),
        ("NumberOfSymbols", 12, 4),
        ("SizeOfOptionalHeader", 16, 2),
    ] {
        v.push(Field { name, at, width: w });
    }
    let nsec = (rd16(2) as usize).min(512);
    for i in 0..nsec {
        let h = 20 + 40 * i;
        if h + 40 > b.len() {
            break;
        }
        for (name, off, w) in [
            ("VirtualSize", 8usize, 4usize),
            ("VirtualAddress", 12, 4),
            ("SizeOfRawData", 16, 4),
            ("PointerToRawData", 20, 4),
            ("PointerToRelocations", 24, 4),
            ("NumberOfRelocations", 32, 2),
            ("NumberOfLinenumbers", 34, 2),
            ("Characteristics", 36, 4),
        ] {
            v.push(Field {
                name,
                at: h + off,
                width: w,
            });
        }
    }
    let symoff = rd32(8) as usize;
    let nsyms = (rd32(12) as usize).min(2048);
    for i in 0..nsyms {
        let p = symoff + 18 * i;
        if p + 18 > b.len() {
            break;
        }
        for (name, off, w) in [
            ("SymbolValue", 8usize, 4usize),
            ("SectionNumber", 12, 2),
            ("SymbolType", 14, 2),
            ("StorageClass", 16, 1),
            ("NumberOfAuxiliary", 17, 1),
        ] {
            v.push(Field {
                name,
                at: p + off,
                width: w,
            });
        }
    }
    // The string table's own size field: it sits after the symbol table, and
    // the loader reads it to slice the table.
    let str_off = symoff + 18 * nsyms;
    if str_off + 4 <= b.len() {
        v.push(Field {
            name: "StringTableSize",
            at: str_off,
            width: 4,
        });
    }
    v
}

/// Mutate `src`: one of bit flip / byte splice / truncation / length edit /
/// COFF field edit. `detail` names the mutation for the provenance line.
pub fn mutate(rng: &mut Rng, src: &[u8], max_len: usize, detail: &mut String) -> Vec<u8> {
    let mut b = src.to_vec();
    if b.is_empty() {
        b.resize(16, 0);
    }
    let kinds = 7;
    match rng.below(kinds) {
        6 => {
            // Absolute-symbol injection. A symbol record with
            // `SectionNumber == 0` is ABSOLUTE: `Symbol::section` is `None`.
            // Every linked object and every object with `.debug$S` carries
            // them, and random bytes never produce a *valid* COFF record with
            // that field zero at a table offset the loader reads - so the one
            // class of symbol record that production guarantees is a class the
            // byte mutator has to synthesize on purpose. `Object::code_symbols`
            // used to `unwrap()` the section in a sort key: a panic on this
            // record, on the loader's public API.
            let nsyms = u32_at(src, 12) as usize;
            let symoff = u32_at(src, 8) as usize;
            let n = 1 + rng.below(3);
            // Append `n` records with SectionNumber = 0, Type = 0, random Value,
            // static storage, and bump NumberOfSymbols to count them.
            for _ in 0..n {
                let mut rec = [0u8; 18];
                let v = rng.below(1 << 24) as u32;
                rec[4..8].copy_from_slice(&v.to_le_bytes());
                rec[8..10].copy_from_slice(&0u16.to_le_bytes()); // SectionNumber 0
                rec[12] = 3; // STORAGE_STATIC
                b.extend_from_slice(&rec);
            }
            if symoff > 0 && nsyms > 0 {
                put(&mut b, 12, 4, (nsyms + n) as u64);
            }
            *detail = format!("injected {n} absolute symbol record(s) past a {nsyms}-entry table");
        }
        0 => {
            // Bit flip.
            let n = 1 + rng.below(4);
            for _ in 0..n {
                let at = rng.below(b.len());
                let bit = rng.below(8);
                b[at] ^= 1 << bit;
            }
            *detail = format!("{} bit flip(s) in {} bytes", n, src.len());
        }
        1 => {
            // Byte splice: copy a run from one place to another.
            let len = 1 + rng.below(32);
            let from = rng.below(b.len());
            let to = rng.below(b.len());
            let run: Vec<u8> = b[from..(from + len).min(b.len())].to_vec();
            if !run.is_empty() {
                for (i, x) in run.iter().enumerate() {
                    if to + i < b.len() {
                        b[to + i] = *x;
                    }
                }
            }
            *detail = format!("spliced {len} bytes {from:#x}->{to:#x}");
        }
        2 => {
            // Truncation: the file ends mid-table.
            let cut = 1 + rng.below(b.len().saturating_sub(1).max(1));
            b.truncate(cut);
            *detail = format!("truncated to {cut} bytes");
        }
        3 => {
            // Length edit: append (the "size field claims more than the file
            // holds" case, with the bytes absent).
            let add = 1 + rng.below(64);
            if b.len() + add <= max_len {
                for _ in 0..add {
                    let x = rng.byte();
                    b.push(x);
                }
            }
            *detail = format!("appended {add} bytes");
        }
        4 => {
            // Field edit: a count/offset/size the parser reads.
            let fields = coff_fields(&b);
            if let Some(f) = fields.get(rng.below(fields.len())).copied() {
                let v = nasty_for(rng, f.width);
                put(&mut b, f.at, f.width, v);
                *detail = format!("{} := {v:#x}", f.name);
            } else {
                let at = rng.below(b.len());
                b[at] = rng.byte();
                *detail = format!("no header fields; byte {at} := random");
            }
        }
        _ => {
            // Two field edits at once: the shape where one field's value is
            // only a gap once a second field is believed too (a relocation
            // count past the end, a symbol offset past the section).
            let fields = coff_fields(&b);
            for _ in 0..2 {
                if let Some(f) = fields.get(rng.below(fields.len())).copied() {
                    let v = nasty_for(rng, f.width);
                    put(&mut b, f.at, f.width, v);
                }
            }
            *detail = "two field edits".to_string();
        }
    }
    if b.len() > max_len {
        b.truncate(max_len);
    }
    b
}

/// A well-formed one-section COFF object whose `.text` is `text`, with one
/// external symbol `f` at offset 0. This is the host the instruction-level
/// generator puts its streams in: the decoder is not the only consumer of the
/// bytes, and a stream that only ever reaches `decode()` fuzzes 5% of the
/// pipeline.
pub fn text_object(text: &[u8]) -> Vec<u8> {
    let nsec = 1usize;
    // section symbol + aux record + `f`
    let nsyms = 3usize;
    let symoff = 20 + 40 * nsec;
    let strtab = symoff + 18 * nsyms;
    let data = strtab + 4;
    let mut b = Vec::new();
    b.extend_from_slice(&crate::loader::MACHINE_AMD64.to_le_bytes());
    b.extend_from_slice(&(nsec as u16).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&(symoff as u32).to_le_bytes());
    b.extend_from_slice(&(nsyms as u32).to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(b".text\0\0\0");
    b.extend_from_slice(&(text.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&(text.len() as u32).to_le_bytes());
    b.extend_from_slice(&(data as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0x6050_0020u32.to_le_bytes());
    // section symbol
    b.extend_from_slice(&[0u8; 8]);
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&1i16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.push(3); // STATIC
    b.push(1); // one aux record
    b.extend_from_slice(&(text.len() as u32).to_le_bytes());
    b.extend_from_slice(&[0u8; 14]);
    // `f`
    b.extend_from_slice(b"f\0\0\0\0\0\0\0");
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&1i16.to_le_bytes());
    b.extend_from_slice(&0x20u16.to_le_bytes());
    b.push(2); // EXTERNAL
    b.push(0);
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(text);
    b
}

/// The nth input of the run. The generator alternates the two families and
/// picks within a family from the same seeded stream, so iteration `i` is
/// reproducible for a given seed and corpus, whatever the other flags say.
/// A string in MSVC decorated-name SHAPE: `?` then a run of the alphabet the
/// grammar uses. Random ASCII rarely produces this shape, and the demangler is
/// the one stage in the pipeline that consumes raw symbol bytes, so the family
/// exists to feed it. Seeded with the corpus's real decorated names by
/// `name_corpus`; this function supplies the shapes the corpus lacks.
/// A name in the grammar the demangler actually accepts, then cut.
///
/// The old version was `'?'` + random characters from the MSVC alphabet, and
/// that is not a fuzzer for this input space: the shapes the demangler
/// special-cases (`??0` constructors, `??_7` vftables, `??_` operator tags,
/// `?__CxxFrameHandler3@` guard names) appear by chance about once in 4000
/// draws, so `rdec fuzz --iterations 800` measured ZERO of them — and an
/// injected `split_at(1)` panic on the truncated `??_` name went unfound. A
/// generator that cannot reach the parser's grammar reaches no bugs.
///
/// Two changes, both aimed at the class the bugs live in:
///
/// - a PREFIX LIBRARY of the tokens the demangler branches on, so the grammar
///   is walked and not won at roulette;
/// - TRUNCATION: a decoration cut off mid-token. A hostile `.obj` carries
///   symbol names that stop in the middle (the `peek().unwrap` family and the
///   c660ff3 `split_at(1)` panic are all "shape tested at index i, bytes read
///   at index i" on a cut name), and the exhaustive prefix sweep in
///   `sweep_names` covers corpus names only — generated shapes need their own
///   cuts.
///
/// The tokens the demangler BRANCHES on. Public because the generator uses
/// them two ways: as the prefix library of a random shape, and as the table a
/// deterministic enumeration walks, cutting every token at every length
/// (`truncated_token`). The second use is the one that matters: the
/// `split_at(1)`/`peek().unwrap` bug class lives at "sigil with nothing after
/// it", and at a 1-in-350 draw a 400-iteration CI run finds it about a third of
/// the time, which is not a gate. Enumerated, every bounded run covers it.
pub const NAME_TOKENS: &[&str] = &[
    "?",
    "??",
    "??0",
    "??1",
    "??_",
    "??$",
    "??_7",
    "??_8",
    "??_9",
    "??_B",
    "??_F",
    "??_R",
    "??_D",
    "??_E",
    "?__CxxFrameHandler3@",
    "?f@@YAH",
    "??_7X@@6B@",
    "?m@X@@UAEXXZ",
    "@",
    "????2",
    "???",
    "??@",
];

/// The `idx`-th (token, cut) pair, in order: every token cut at every length
/// from 0 to its full length, token by token, each one CLOSED with the `@` that
/// ends an MSVC identifier. The `@` is not decoration - `Parser::segment` reads
/// the identifier up to `@`, so a bare `??_` is rejected as an unterminated
/// identifier and never reaches the branch that panics; `??_@` is the shape
/// that does (measured: with the `split_at(1)` bug restored, `??_` is a clean
/// `Unsupported` and `??_@` is the panic). `None` once the table is exhausted,
/// which is where the random shapes take over. Deterministic, so a
/// 400-iteration CI run covers the same prefix of the table every time.
pub fn truncated_token(idx: u64) -> Option<String> {
    let mut i = idx;
    for t in NAME_TOKENS {
        let cuts = t.len() as u64 + 1;
        if i < cuts {
            // Every token is ASCII, so `i` is always a char boundary.
            return Some(format!("{}@", &t[..i as usize]));
        }
        i -= cuts;
    }
    None
}

pub fn shaped_name(rng: &mut Rng) -> String {
    const ALPHA: &[u8] = b"?@$_0123456789ABCDEFYXWVUTSRQPONMLKJIHGFEDCBAabcdefghij";
    let mut s = String::from(NAME_TOKENS[rng.below(NAME_TOKENS.len())]);
    let len = rng.below(24);
    for _ in 0..len {
        // ASCII alphabet: the truncation below cannot split a code point.
        s.push(ALPHA[rng.below(ALPHA.len())] as char);
    }
    if rng.below(3) == 0 {
        s.truncate(rng.below(s.len() + 1));
    }
    s
}

/// Every decorated name the seeds carry (real clang shapes), deduped.
pub fn name_corpus(seeds: &[Seed]) -> Vec<String> {
    let mut v: Vec<String> = seeds.iter().flat_map(|s| s.names.clone()).collect();
    v.sort();
    v.dedup();
    v
}

/// The truncation-at-every-length sweep: for each name, every prefix
/// `name[..1..=len]`, fed to `f` (production: `rdec_demangle::demangle`).
///
/// This is the c660ff3 bug class - "shape tested at index i, bytes read at
/// index i" - attacked exhaustively instead of probabilistically: a truncated
/// decoration is a prefix, and every prefix of every real name is 100% of that
/// input space at O(len) per name. Per-name panics are counted apart from the
/// iteration census (`name.panic` vs the pipeline's `crash`), which is what
/// makes the per-generator numbers in the report honest.
pub fn sweep_names(cfg: &Config, names: &[String], st: &mut Stats, f: &mut dyn FnMut(&str)) {
    let mut kept = 0usize;
    for n in names {
        st.bump("name.names", 1);
        for k in 1..=n.len() {
            let cut = &n[..k];
            st.bump("name.truncations", 1);
            let owned = cut.to_string();
            let r = catch_panic(|| f(&owned));
            if let Err(msg) = r {
                st.bump("crash", 1);
                st.bump("name.panic", 1);
                if kept < cfg.keep_crashes {
                    kept += 1;
                    st.crashes.push(Crash {
                        input: Input {
                            bytes: owned.clone().into_bytes(),
                            gen: "name",
                            detail: format!("name sweep truncation: {owned}"),
                            seed: cfg.seed,
                            iter: st.get("name.truncations"),
                        },
                        stage: "demangle(name sweep)".to_string(),
                        message: msg,
                    });
                }
            }
        }
    }
}

pub fn generate(cfg: &Config, seeds: &[Seed], iter: u64) -> Input {
    let mut rng = Rng::new(cfg.seed.wrapping_mul(0x1000_0001).wrapping_add(iter));
    let mut detail = String::new();
    let bytes: Vec<u8>;
    let gen: &'static str;
    let half = cfg.iterations / 2;
    if iter % 8 == 7 {
        // The name family: an MSVC-shaped string straight for the demangler.
        // Random bytes (the `bytes` family) almost never produce the shape, and
        // the demangler consumes raw symbol bytes in production, so the family
        // exists to feed it - the exhaustive truncation sweep is `sweep_names`.
        //
        // The first `NAME_TOKENS` x cuts inputs are the DETERMINISTIC
        // enumeration (`truncated_token`): every token the demangler branches
        // on, cut at every length. That is what turns "the fuzzer sometimes
        // stumbles into a truncated sigil" into "every bounded run walks them",
        // which is the difference between a gate and a lottery.
        gen = "name";
        let n = match truncated_token(iter / 8) {
            Some(t) => t,
            None => shaped_name(&mut rng),
        };
        detail = format!("name: {n}");
        bytes = n.into_bytes();
    } else if iter < half || seeds.is_empty() {
        gen = "bytes";
        // Byte-level: 1/4 pure random, 3/4 mutations of a real object.
        if rng.below(4) == 0 || seeds.is_empty() {
            let n = 8 + rng.below(4096);
            let mut v = vec![0u8; n.min(cfg.max_input_bytes)];
            for x in v.iter_mut() {
                *x = rng.byte();
            }
            detail = format!("{} random bytes", v.len());
            bytes = v;
        } else {
            let s = &seeds[rng.below(seeds.len())];
            bytes = mutate(&mut rng, &s.obj, cfg.max_input_bytes, &mut detail);
            detail = format!("{} of {}", detail, s.name);
        }
    } else {
        gen = "insn";
        // Instruction-level: a random stream, or a mutated real .text, both
        // embedded in a well-formed object (and the stream is ALSO handed to
        // the decoder directly by the pipeline, which is what fires the
        // decoder's debug asserts on raw noise).
        let text: Vec<u8> = if rng.below(3) == 0 || seeds.iter().all(|s| s.text.is_empty()) {
            let n = 4 + rng.below(256);
            let mut v = vec![0u8; n];
            for x in v.iter_mut() {
                *x = rng.byte();
            }
            detail = format!("{} random bytes as .text", n);
            v
        } else {
            let s = &seeds[rng.below(seeds.len())];
            if s.text.is_empty() {
                let n = 4 + rng.below(128);
                let mut v = vec![0u8; n];
                for x in v.iter_mut() {
                    *x = rng.byte();
                }
                detail = format!("{n} random bytes ({} has no .text)", s.name);
                v
            } else {
                let mut d = String::new();
                let v = mutate(&mut rng, &s.text, 8192, &mut d);
                detail = format!("{} of {}.text", d, s.name);
                v
            }
        };
        bytes = text_object(&text);
    }
    Input {
        seed: cfg.seed,
        iter,
        gen,
        detail,
        bytes,
    }
}

/// The front-half pipeline, in the crate that owns the front half.
///
/// What it runs: the decoder on the raw bytes, the COFF loader, discovery with
/// the object's own relocation evidence (jump tables + displacement map), the
/// lifter, the pass pipeline (`rdec_core::passes::clean`, the same list the
/// gates run), the IR verifier, the C emitter, and the interpreter on the
/// lifted IR.
///
/// What it does NOT run, and why: the MSVC demangler (in `rdec-demangle`, which
/// depends on this crate - a real dependency the other way is a cycle) and the
/// emitter-side data-section placement of `m4::data_map_of` (a property of the
/// generated C's memory model, not of the front end). `rdec fuzz` runs the
/// production pipeline with both. This function exists so
/// `crates/arch-x86/tests/fuzz.rs` can run a bounded fuzz in `cargo test`,
/// where the CLI binary is not a dependency - and so the subset it covers is
/// named in one place, not guessed from a diff.
pub fn front_half(input: &Input, st: &mut Stats) {
    use crate::decode::{decode, Syntax};
    use crate::discover::{discover_all_with_container, displacements, jump_tables, Code};
    use crate::lift::{lift_one_sigs, CalleeSig, CalleeTable, FnAbi, RelocMap, RelocTarget};
    use std::collections::HashMap;

    // Stage 0: the decoder, on the raw bytes. `decode.rs`'s boundary asserts
    // fire here in a debug build, which is the whole reason the CI test is a
    // debug test.
    let ins = decode(&input.bytes, 0x1000, Syntax::Intel);
    st.bump("decode.calls", 1);
    st.bump("decode.insns", ins.len() as u64);
    st.bump(
        "decode.refusals",
        ins.iter().filter(|i| !i.valid()).count() as u64,
    );

    // Stage 1: the container.
    let obj = match crate::loader::load_bytes(&input.bytes, "fuzz.obj") {
        Ok(o) => {
            st.bump("load.ok", 1);
            o
        }
        Err(e) => {
            st.bump("load.err", 1);
            *st.load_errors.entry(classify_load_error(&e)).or_insert(0) += 1;
            return;
        }
    };
    if obj.string_table_truncated {
        st.bump("load.string_table_truncated", 1);
    }
    st.bump("symbols", obj.symbols.len() as u64);
    st.bump("sections", obj.sections.len() as u64);
    st.bump("relocs", obj.total_relocs() as u64);
    st.bump("demangle.names", obj.symbols.len() as u64);

    // Stage 2: the code ranges, in the flat per-object space every gate uses.
    let mut ranges: Vec<Code> = Vec::new();
    let mut idx_of: HashMap<usize, usize> = HashMap::new();
    let mut base_of: Vec<u64> = Vec::new();
    let mut next_base = 0u64;
    for (i, sec) in obj.sections.iter().enumerate() {
        if obj.is_code(i) && !sec.bytes.is_empty() {
            idx_of.insert(i, ranges.len());
            base_of.push(next_base);
            ranges.push(Code {
                base: next_base,
                bytes: sec.bytes.as_slice(),
            });
            next_base += sec.bytes.len() as u64 + 16;
        }
    }
    if ranges.is_empty() {
        st.bump("discover.no-code", 1);
        return;
    }
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    for i in obj.code_symbols() {
        let s = &obj.symbols[i];
        let sec = match s.section {
            Some(x) => x,
            None => continue,
        };
        if !obj.is_code(sec) || s.name.starts_with('.') || s.size == 0 {
            continue;
        }
        if let Some(&r) = idx_of.get(&sec) {
            seeds.entry(r).or_default().0.push(base_of[r] + s.offset);
        }
    }
    for (sec, rels) in obj.sections.iter().enumerate() {
        let &r = match idx_of.get(&sec) {
            Some(x) => x,
            None => continue,
        };
        for rl in &rels.relocations {
            if !matches!(rl.typ, 2 | 4 | 5 | 6 | 7) {
                continue;
            }
            if let Some(t) = obj.symbol_of(rl.symbol) {
                if t.section == Some(sec) && !t.name.starts_with('.') {
                    seeds.entry(r).or_default().1.push(base_of[r] + t.offset);
                }
            }
        }
    }
    let base_fn = |sec: usize| idx_of.get(&sec).copied().map(|r| base_of[r]);
    let tables = jump_tables(&obj, &base_fn);
    let disp = displacements(&obj, &base_fn);
    let plan = discover_all_with_container(&ranges, &seeds, &tables, &disp);
    st.bump("discover.functions", plan.functions.len() as u64);
    st.bump("discover.blocks", plan.blocks.len() as u64);
    st.bump("discover.bytes", plan.covered_bytes());
    st.bump("discover.tables", plan.tables.len() as u64);
    for g in &plan.gaps {
        st.gap(&format!("discover.{:?}", g.gap), 1);
    }

    // Stage 3: names, callee shapes, relocations - the evidence the lifter
    // reads. The data sections are placed above the stack window, bounded: a
    // section that would run past the window is DECLINED and counted, never
    // silently placed on top of the code.
    let mut names: HashMap<u64, String> = HashMap::new();
    let mut sigs: CalleeTable = CalleeTable::new();
    for i in obj.code_symbols() {
        let s = &obj.symbols[i];
        let sec = match s.section {
            Some(x) => x,
            None => continue,
        };
        if obj.is_code(sec) && !s.name.starts_with('.') && s.size > 0 {
            if let Some(&r) = idx_of.get(&sec) {
                let at = base_of[r] + s.offset;
                names.insert(at, s.name.clone());
                if !sigs.contains_key(s.name.as_str()) {
                    sigs.insert(
                        s.name.clone(),
                        CalleeSig {
                            cname: crate::fuzz::c_ident(&s.name),
                            abi: FnAbi::integer(),
                            typed: true,
                            noreturn: crate::lift::noreturn_symbol(&s.name),
                        },
                    );
                }
            }
        }
    }
    let mut relocs: RelocMap = RelocMap::new();
    let mut data_at = 1u64 << 24;
    let mut sec_va: HashMap<usize, u64> = HashMap::new();
    for (i, sec) in obj.sections.iter().enumerate() {
        if obj.is_code(i) {
            continue;
        }
        if data_at + sec.bytes.len() as u64 >= 1u64 << 26 {
            st.gap("data-window-declined", 1);
            continue;
        }
        sec_va.insert(i, data_at);
        data_at += sec.bytes.len() as u64 + 16;
    }
    for (sec, rels) in obj.sections.iter().enumerate() {
        let &r = match idx_of.get(&sec) {
            Some(x) => x,
            None => continue,
        };
        for rl in &rels.relocations {
            if !rl.kind().is_rel32() {
                continue;
            }
            let Some(sym) = obj.symbol_of(rl.symbol) else {
                continue;
            };
            let addr = match sym.section {
                Some(s2) if s2 == sec => Some(base_of[r] + sym.offset),
                Some(s2) if idx_of.contains_key(&s2) => Some(base_of[idx_of[&s2]] + sym.offset),
                Some(s2) => sec_va.get(&s2).map(|b| b + sym.offset),
                None => None,
            };
            relocs.insert(
                base_of[r] + rl.address,
                RelocTarget {
                    sym: sym.name.clone(),
                    addr,
                },
            );
        }
    }

    // Stage 4: lift, passes, verify, emit, interpret.
    for (k, &start) in plan.functions.iter().enumerate() {
        let rg = match ranges
            .iter()
            .find(|r| start >= r.base && start < r.base + r.bytes.len() as u64)
        {
            Some(x) => *x,
            None => continue,
        };
        let end = plan.functions[k + 1..]
            .iter()
            .find(|a| **a >= rg.base && **a < rg.base + rg.bytes.len() as u64)
            .copied()
            .unwrap_or(rg.base + rg.bytes.len() as u64);
        let raw = names
            .get(&start)
            .cloned()
            .unwrap_or_else(|| format!("fn_{start:x}"));
        let abi = FnAbi::integer();
        let (mut f, rep) = lift_one_sigs(
            &c_ident(&raw),
            rg.bytes,
            rg.base,
            start,
            end,
            &plan,
            &names,
            &abi,
            &sigs,
            &relocs,
            &tables,
        );
        st.bump("lift.functions", 1);
        st.bump("lift.gaps", rep.gaps as u64);
        st.bump("lift.exits", rep.exits as u64);
        rdec_core::passes::clean(&mut f);
        match rdec_core::verify::verify_ssa(&f) {
            Ok(()) => st.bump("verify.ok", 1),
            Err(errs) => {
                st.bump("verify.reject", 1);
                st.gap("verify-broke-ssa", 1);
                let _ = errs;
            }
        }
        match rdec_core::emit_c::emit_c(&f) {
            Ok(out) => {
                st.bump("emit.c", 1);
                st.bump("emit.c_bytes", out.text.len() as u64);
            }
            Err(rs) => {
                st.bump("emit.refused", 1);
                for r in &rs {
                    st.gap(&format!("emit: {}", refusal_key(&r.reason)), 1);
                }
            }
        }
        let exec = rdec_core::interp::run_budget(&f, &[1, 2, 3, 4], 20_000);
        st.bump("interp.runs", 1);
        match &exec.outcome {
            rdec_core::interp::Outcome::Ret(_) => st.bump("interp.ret", 1),
            rdec_core::interp::Outcome::NoReturn => st.bump("interp.noreturn", 1),
            rdec_core::interp::Outcome::Budget => st.bump("interp.budget", 1),
            rdec_core::interp::Outcome::Fault(f) => st.gap(&format!("interp.fault {f:?}"), 1),
        }
    }
}

/// The C-identifier spelling of a symbol name, shared with the CLI's naming so
/// the two pipelines call the same function by the same name.
pub fn c_ident(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// One refusal class, keyed the way `m4::key` keys them: the leading category
/// of the reason, so the census's refusal classes are the gates' refusal
/// classes and a count means the same thing in both reports.
pub fn refusal_key(reason: &str) -> String {
    let s = reason.trim();
    if let Some(rest) = s.strip_prefix("opaque:") {
        let rest = rest.trim();
        let cut = rest.find(" at ").unwrap_or(rest.len());
        return format!("opaque: {}", rest[..cut.min(40)].trim());
    }
    let low = s.to_lowercase();
    let cut = low.find(':').unwrap_or(low.len());
    low[..cut.min(46)].trim().to_string()
}

/// Classify a loader refusal into the census's named kinds. The classes are
/// the loader's own failure modes, so the census says which guard fired and a
/// new guard with a new message shows up as a new class - not folded into
/// "other" where a regression could hide.
pub fn classify_load_error(msg: &str) -> String {
    // The message is `path: reason`; classify the reason.
    let m = msg.split_once(": ").map(|x| x.1).unwrap_or(msg);
    let kind: &str = if m.contains("truncated COFF") {
        "truncated-header"
    } else if m.contains("optional header") {
        "optional-header"
    } else if m.contains("inside the headers") {
        "symbol-table-in-headers"
    } else if m.contains("symbol table runs off the end") {
        "symbol-table-off-end"
    } else if m.contains("auxiliary records") {
        "aux-records-past-table"
    } else if m.contains("relocation table runs off the end")
        || m.contains("relocations at PointerToRelocations")
    {
        "reloc-table-off-end"
    } else if m.contains("past the end of a") || m.contains("bytes past the end") {
        "section-data-off-end"
    } else if m.starts_with("io:") || m.contains("os error") {
        "io"
    } else {
        // Unclassified refusals keep their own first clause: a class the
        // census cannot name is a class nobody will ever fix. It is kept as
        // `other: <first clause>` so the census still groups it, and the
        // literal is in the census line, not only in the file.
        return format!(
            "other: {}",
            m.split(',')
                .next()
                .unwrap_or(m)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join("-")
        );
    };
    kind.to_string()
}

/// The panic payload, as a single line.
fn payload(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Run `body` with panics caught and captured (the default hook prints a
/// backtrace to stderr, which is noise in a 3000-iteration census).
pub fn catch_panic<R>(body: impl FnOnce() -> R) -> Result<R, String> {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let r = panic::catch_unwind(panic::AssertUnwindSafe(body));
    panic::set_hook(prev);
    match r {
        Ok(v) => Ok(v),
        Err(p) => Err(payload(p.as_ref())),
    }
}

/// The driver: generate, run the caller's pipeline, catch panics, count.
///
/// `pipeline` is the code under test: it gets one input's bytes, runs the real
/// stages, and bumps its own `Stats`. It must not print (the census owns
/// stdout). A caught panic is a `Crash` record and the driver keeps going, so
/// the census counts the population of crashes, not just the first one.
pub fn run(
    cfg: &Config,
    seeds: &[Seed],
    st: &mut Stats,
    pipeline: &mut dyn FnMut(&Input, &mut Stats),
) {
    let started = Instant::now();
    st.debug_checks = cfg!(debug_assertions);
    let deadline = started + Duration::from_secs(cfg.max_seconds.max(1));
    // Every input that panicked, and how many times each distinct panic
    // message appeared: 3000 iterations of one bug are one finding, not 3000.
    let mut seen: BTreeMap<String, u64> = BTreeMap::new();
    for iter in 0..cfg.iterations {
        let input = generate(cfg, seeds, iter);
        st.iterations += 1;
        *st.inputs.entry(input.gen).or_insert(0) += 1;
        st.bytes_total += input.bytes.len() as u64;
        st.max_input = st.max_input.max(input.bytes.len());
        if cfg.trace {
            // Written BEFORE the input runs: the tail of stderr then names the
            // input a stalled process is stuck inside, which is how a hang gets
            // attributed without a debugger.
            eprintln!(
                "fuzz: iter {iter} gen={} bytes={} :: {}",
                input.gen,
                input.bytes.len(),
                input.detail
            );
        }
        let t0 = Instant::now();
        // A panic in the GENERATOR is a bug in the fuzz surface and lands in
        // the same census, at stage "generate".
        let r = catch_panic(|| {
            // The pipeline gets a scratch Stats: a panic mid-stage leaves
            // partial counts OUT of the census rather than half-merged into it,
            // so the per-stage totals always describe inputs that completed.
            let mut sub = Stats::default();
            pipeline(&input, &mut sub);
            sub
        });
        match r {
            Ok(sub) => {
                // Note the slowest input FIRST: the merge below consumes
                // `sub`'s maps, and the note is about this input's stages.
                let ms = t0.elapsed().as_millis() as u64;
                let top = sub
                    .stage_ms
                    .iter()
                    .max_by_key(|(_, v)| **v)
                    .map(|(k, v)| format!("{k} {v} ms"))
                    .unwrap_or_else(|| "no stage recorded a time".to_string());
                for (k, v) in sub.stages {
                    st.bump(&k, v);
                }
                for (k, v) in sub.load_errors {
                    *st.load_errors.entry(k).or_insert(0) += v;
                }
                for (k, v) in sub.gaps {
                    st.gap(&k, v);
                }
                for (k, v) in &sub.stage_ms {
                    st.time(k, *v);
                }
                let per_stage = sub
                    .stage_ms
                    .iter()
                    .map(|(k, v)| format!("{k}={v}ms"))
                    .collect::<Vec<_>>()
                    .join(" ");
                // The slowest input, with the stage that ate it named. A census
                // that says "slowest 65594 ms" is a shrug; this line points at
                // bytes and at one stage, which is what makes it fixable.
                if ms > st.slowest_ms as u64 {
                    st.slowest_ms = ms as u128;
                    st.slow_note = format!(
                        "slowest input: {ms} ms total, {top} of it in one stage; {}",
                        input.provenance()
                    );
                    st.slowest = Some(input.clone());
                }
                if ms > cfg.max_input_ms {
                    // An input over the per-input ceiling is a FINDING, kept
                    // with its bytes. It is not a crash and it is not OK.
                    st.bump("slow.over_budget", 1);
                    if st.crashes.len() < cfg.keep_crashes {
                        let ceiling = cfg.max_input_ms;
                        let note = OVER_BUDGET_NOTE;
                        let stages = per_stage;
                        st.crashes.push(Crash {
                            input: input.clone(),
                            stage: format!("slow: over {ms} ms budget"),
                            message: format!(
                                "input took {ms} ms, over the {ceiling} ms per-input ceiling: {stages}\n{note}"
                            ),
                        });
                    }
                }
            }
            Err(msg) => {
                let key = format!(
                    "{} :: {}",
                    msg.split('\n').next().unwrap_or("").trim(),
                    input.gen
                );
                let n = seen.entry(key.clone()).or_insert(0);
                *n += 1;
                if *n == 1 && st.crashes.len() < cfg.keep_crashes {
                    st.crashes.push(Crash {
                        input: input.clone(),
                        stage: "pipeline".to_string(),
                        message: msg,
                    });
                }
            }
        }
        st.slowest_ms = st.slowest_ms.max(t0.elapsed().as_millis());
        if Instant::now() > deadline {
            st.bump("deadline.hit", 1);
            break;
        }
    }
    for (k, v) in seen {
        st.bump("crash", v);
        st.gap(&format!("crash: {k}"), v);
    }
    st.elapsed = started.elapsed();
}
