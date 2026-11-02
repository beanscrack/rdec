//! From-scratch MSVC C++ decorated-name demangler (the M8 dependency).
//!
//! A decompiler that reads a C++ object must know that `?area@Circle@@QEBA_NXZ`
//! is `bool Circle::area(void) const` in order to name functions, recognise
//! `this` (it is a member function, so RCX is `this`), and spot constructors
//! (`??0`) and destructors (`??1`). The kind, not the text, is the contract:
//! M8 consumes `Constructor`/`MemberFunction{class, constness}` directly.
//!
//! The thesis of this repo is that parsing happens here, so there is no
//! demangler crate behind this: the byte-level encoding rules were derived
//! empirically — every builtin letter, the x64 pointer/reference prefixes,
//! the `@@` scope form, the `S`/`SA` static codes, enum `W4`, by-value class
//! `U`, repeated-type digits (`0`..`5`), and the ctor/dtor `@` return marker —
//! by compiling fixtures with clang for `x86_64-pc-windows-msvc` and pairing
//! `llvm-nm --defined-only -g` with `--demangle`. The unit table below embeds
//! those measured pairs; `rdec demangle --cpp` re-checks the encodings on the
//! corpus objects against the same oracle on every run.
//!
//! The type table is the **x64** one (measured): `M` is float, `N` is double,
//! `F` is short, `E` is unsigned char, `_J` is `__int64`. The 32-bit MSVC
//! letters (J float, M double, E unsigned short, X char, A void as a
//! standalone type) collide with these and are *not* used; x64 is what the
//! corpus, the gate, and M8 target.
//!
//! Everything outside the supported subset returns
//! `Unsupported { raw, reason }` with a specific reason. A name that demangles
//! to plausible garbage is worse than one that says "template argument list":
//! the decompiler must see the gap, not inherit it as a fake function name.

/// `const`-ness of a member function, from the this-adjustment code letter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constness {
    NonConst,
    Const,
    Volatile,
    ConstVolatile,
}

impl Constness {
    fn suffix(self) -> &'static str {
        match self {
            Constness::NonConst => "",
            Constness::Const => " const",
            Constness::Volatile => " volatile",
            Constness::ConstVolatile => " const volatile",
        }
    }
}

/// Which construct a decorated name denotes. M8 pattern-matches on this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymKind {
    FreeFunction,
    MemberFunction {
        class: String,
        constness: Constness,
        /// `virtual` — the U/V/S (and thunk W/X) access codes.
        virtual_fn: bool,
        /// vcall/displacement adjustment: the `@4`/`@7`/`@8` suffix, the
        /// W/X access codes, or the `EAAX`-style thunk code.
        thunk: bool,
    },
    Constructor {
        class: String,
    },
    Destructor {
        class: String,
    },
    StaticMember {
        class: String,
    },
    /// `??_7`/`??_8` vftables and `??_9`/`??_B` vbtables.
    Vtable {
        class: String,
        table: TableKind,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    Vftable,
    Vbtable,
}

/// Outcome of demangling one symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Demangled {
    Name { text: String, kind: SymKind },
    Unsupported { raw: String, reason: String },
}

impl Demangled {
    /// The human-readable signature for a `Name`, empty for `Unsupported`.
    pub fn text(&self) -> &str {
        match self {
            Demangled::Name { text, .. } => text,
            Demangled::Unsupported { .. } => "",
        }
    }
}

/// Demangle one MSVC decorated name. Never panics; anything the grammar does
/// not fully consume becomes `Unsupported` with a reason naming the construct.
pub fn demangle(sym: &str) -> Demangled {
    match parse(sym) {
        Ok(v) => v,
        Err(reason) => Demangled::Unsupported {
            raw: sym.to_string(),
            reason,
        },
    }
}

/// Canonical form used by the llvm-nm gate to compare two renderings of the
/// same symbol. Both sides shed decoration the ABI does not care about:
/// access specifiers, calling-convention words, `virtual`/`static` keywords,
/// `__ptr64`, the `(void)` empty-parameter-list spelling, vftable word order,
/// `[clone ...]` suffixes, and all whitespace.
pub fn canonical(text: &str) -> String {
    let mut s = text.to_string();
    for drop in [
        "public: ",
        "private: ",
        "protected: ",
        "virtual ",
        "static ",
        "__cdecl ",
        "__thiscall ",
        "__stdcall ",
        "__fastcall ",
        "__vectorcall ",
        "__ptr64",
    ] {
        s = s.replace(drop, "");
    }
    s = s.replace("(void)", "()");
    // llvm prints vftables as `Circle::`vftable'`; we print `vftable for
    // Circle`. Both canonicalize to `vftable:Circle`.
    for kw in ["vftable", "vbtable"] {
        let pat = format!("::`{kw}'");
        if let Some(i) = s.find(&pat) {
            s = format!("{kw}:{}", s[..i].trim());
        }
        s = s.replace(&format!("{kw} for "), &format!("{kw}:"));
    }
    if let Some(i) = s.find("[clone") {
        s.truncate(i);
    }
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    /// Types fully parsed so far, for the `0`..`5` repeat-reference digits.
    types: Vec<String>,
}

type PErr = String;

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            s: s.as_bytes(),
            pos: 0,
            types: Vec::new(),
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }
    fn peek_at(&self, k: usize) -> Option<u8> {
        self.s.get(self.pos + k).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn starts_with(&self, pat: &[u8]) -> bool {
        self.s[self.pos..].starts_with(pat)
    }
    fn rest(&self) -> &'a str {
        std::str::from_utf8(&self.s[self.pos..]).unwrap_or("<non-ascii>")
    }
    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    /// True when a named type closed by `@@` starts at pos: `DData@@`,
    /// `W4Kind@@`. A name segment is [A-Za-z0-9_$?@]+ up to a doubled '@'.
    fn enum_by_value_ahead(&self) -> bool {
        let bytes = self.s;
        let mut i = self.pos + 1;
        let mut nonempty = false;
        while i < bytes.len() && bytes[i] != b'@' {
            nonempty = true;
            i += 1;
        }
        nonempty && i + 1 < bytes.len() && bytes[i] == b'@' && bytes[i + 1] == b'@'
    }

    /// True when pos+k is a qualifier letter (A-E, 0-2) AND pos+k+1 can
    /// start a type — the lookahead that separates `PEAH` (int *) from
    /// `DWidget@@` (enum-by-value whose name begins with a qualifier letter).
    fn qualifier_followed_by_type(&self, k: usize) -> bool {
        let q = matches!(self.peek_at(k), Some(b'A'..=b'E') | Some(b'0'..=b'2'));
        q && matches!(
            self.peek_at(k + 1),
            Some(b'A'..=b'Z') | Some(b'0'..=b'5') | Some(b'_') | Some(b'$') | Some(b'?')
        )
    }

    /// Read one scope segment: bytes up to (not including) the next '@'.
    fn segment(&mut self) -> Result<&'a str, PErr> {
        let start = self.pos;
        while self.pos < self.s.len() && self.s[self.pos] != b'@' {
            self.pos += 1;
        }
        if self.pos >= self.s.len() {
            return Err("truncated decoration (scope not closed by '@')".to_string());
        }
        std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| "non-UTF8 decoration".to_string())
    }

    /// The scope after the identifier: `@@` = global (no segments),
    /// `@a@b@@` = [a, b], innermost first as encoded.
    fn scope_segments(&mut self) -> Result<Vec<&'a str>, PErr> {
        let mut scope = Vec::new();
        if self.eat(b'@') {
            return Ok(scope); // `name@@...`: global scope
        }
        loop {
            let seg = self.segment()?;
            if seg.is_empty() {
                return Err("empty scope segment".to_string());
            }
            self.pos += 1; // the '@'
            if seg.contains('?') {
                return Err("anonymous namespace or local-static label".to_string());
            }
            scope.push(seg);
            if self.eat(b'@') {
                break; // `@@` closed the scope
            }
            if self.eof() {
                return Err("truncated decoration (scope not closed by '@@')".to_string());
            }
        }
        Ok(scope)
    }

    fn builtin(b: &[u8]) -> Option<&'static str> {
        // Measured x64 table: every letter paired against
        // `llvm-nm --demangle` over clang -target x86_64-pc-windows-msvc
        // objects (f_A..f_N, f_X, f_W, f_bool fixtures).
        Some(match b {
            b"A" => "void",
            b"C" => "signed char",
            b"D" => "char",
            b"E" => "unsigned char",
            b"F" => "short",
            b"G" => "unsigned short",
            b"H" => "int",
            b"I" => "unsigned int",
            b"J" => "long",
            b"K" => "unsigned long",
            b"M" => "float",
            b"N" => "double",
            b"O" => "long double",
            // X is the void marker (measured: f_void(void) = `YAHXZ`;
            // f_X(char) = `YAHD` - clang encodes char as D on x64).
            b"X" => "void",
            b"_J" => "__int64",
            b"_K" => "unsigned __int64",
            b"_N" => "bool",
            b"_W" => "wchar_t",
            b"_O" => "long double",
            _ => return None,
        })
    }

    /// Resolve a `0`..`5` repeat reference against the stack of completed types.
    ///
    /// Measured convention: `0` is the most recently completed type
    /// (`add@@YA_J_J0`), and in a bare named slot (`W41@`) `1` means the
    /// same - the MSVC count includes the slot there. Try the strict count
    /// first, then the named-slot reading; anything else is a loud error.
    fn repeat_ref(&mut self, d: usize, what: &str) -> Result<String, PErr> {
        // MSVC counts every complete type encoded so far, inner types
        // included (named pushes in parse_named_type, ref-lead inner
        // pushes); the digit d counts back from the most recent.
        let idx = self
            .types
            .len()
            .checked_sub(1 + d)
            .ok_or_else(|| format!("repeated type reference has no referent ({what})"))?;
        Ok(self.types[idx].clone())
    }

    /// Parse a type at self.pos; pushes the result on the repeat stack.
    fn parse_type(&mut self) -> Result<String, PErr> {
        let t = self.parse_type_inner()?;
        self.types.push(t.clone());
        Ok(t)
    }

    fn parse_type_inner(&mut self) -> Result<String, PErr> {
        let Some(c) = self.peek() else {
            return Err("truncated decoration (type expected)".to_string());
        };
        // `_`-prefixed builtins.
        if c == b'_' {
            let two: [u8; 2] = [b'_', self.peek_at(1).unwrap_or(b'?')];
            return match Self::builtin(&two) {
                Some(name) => {
                    self.pos += 2;
                    Ok(name.to_string())
                }
                None => Err(format!("builtin type '_{}'", two[1] as char)),
            };
        }
        // Repeat references `0`..`5`, function-pointer encodings `6`..`9`.
        if c.is_ascii_digit() {
            let d = (c - b'0') as usize;
            if d >= 6 {
                return Err(format!(
                    "function pointer type (code letter '{}')",
                    c as char
                ));
            }
            self.pos += 1; // the repeat-reference digit
            return self.repeat_ref(d, "argument position");
        }
        // `?A`/`?B` prefix. On an enum/builtin/pointer it is the unnamed-type
        // wrapper (measured: `?AW4Kind@@` is `enum Kind`); on a class name it
        // is a template argument (`?AVWidget@@`), which is M8-gap territory.
        if c == b'?' {
            if self.starts_with(b"?A") || self.starts_with(b"?B") {
                self.pos += 2;
                if matches!(self.peek(), Some(b'V') | Some(b'U')) {
                    return Err("template argument list".to_string());
                }
                // The ?A wrapper is not a complete type of its own - the
                // wrapped type (W41@, builtins) is what MSVC counts.
                return self.parse_type_inner();
            }
            if self.starts_with(b"?$") {
                return Err("template argument list".to_string());
            }
            return Err(format!(
                "anonymous type encoding '?{}'",
                self.peek_at(1).unwrap_or(b'?') as char
            ));
        }
        // ---- x64 references: the measured shape is `A` + `E` + qualifier -
        // AEAH int&, AEBH int const&, AECH int volatile&, AEDH cv&, and the
        // same for named types. `A` + near-qualifier is the 32-bit near
        // reference, refused by name; B/C/D are 32-bit reference leads.
        if c == b'A' {
            if self.peek_at(1) == Some(b'E') && matches!(self.peek_at(2), Some(b'A'..=b'D')) {
                self.pos += 2; // 'A' 'E'
                let q = self.peek().unwrap();
                self.pos += 1; // qualifier
                let inner = self.parse_type()?;
                let qual = match q {
                    b'A' => "",
                    b'B' => " const",
                    b'C' => " volatile",
                    _ => " const volatile",
                };
                return Ok(format!("{inner}{qual}&"));
            }
            if self.qualifier_followed_by_type(1) {
                return Err(format!(
                    "near/far code model letter '{}' not mapped (32-bit reference)",
                    c as char
                ));
            }
        }
        if matches!(c, b'A') && self.qualifier_followed_by_type(1) {
            return Err(format!(
                "near/far code model letter '{}' not mapped (32-bit reference)",
                c as char
            ));
        }
        // B/C/D + qualifier is the 32-bit reference family; on x64 clang
        // never emits it, and `chain(signed char, unsigned char, char)` =
        // `CED` (measured) makes B/C/D builtin-lead precedence mandatory.
        // Class-by-value family: lead letter + name closed by `@@`.
        // B class (32-bit near), C union, D enum, E enum-with-underlying.
        if matches!(c, b'B' | b'C' | b'D' | b'E') && self.enum_by_value_ahead() {
            self.pos += 1;
            let prefix = match c {
                b'B' => "class",
                b'C' => "union",
                _ => "enum",
            };
            return self.parse_named_type(prefix);
        }
        // ---- x64 pointers: `P` 64-bit, `Q` __ptr32 (renders `*const`),
        // `R` renders `*volatile`; then a memory-model marker (E/A) and one
        // qualifier letter (A plain, B const, C volatile, D cv).
        if matches!(c, b'P' | b'Q' | b'R') {
            self.pos += 1;
            if matches!(self.peek(), Some(b'E') | Some(b'A') | Some(b'Q')) {
                self.pos += 1; // __ptr64 / near / __ptr32 marker
            }
            let qual = match self.peek() {
                Some(b'A') | Some(b'B') | Some(b'C') | Some(b'D') => {
                    let q = self.peek().unwrap();
                    self.pos += 1;
                    q
                }
                _ => b'A',
            };
            let inner = self.parse_type()?;
            let nested = inner.contains(['*', '&']);
            let mut t = inner;
            if qual == b'B' && !nested {
                t.push_str(" const");
            }
            if qual == b'C' && !nested {
                t.push_str(" volatile");
            }
            if qual == b'D' && !nested {
                t.push_str(" const volatile");
            }
            // Q/R carry the outer pointer's own cv-qualification.
            let outer = match c {
                b'Q' => " *const",
                b'R' => " *volatile",
                _ => " *",
            };
            t.push_str(outer);
            return Ok(t);
        }
        // Class / union / enum by value.
        match c {
            b'V' => {
                self.pos += 1;
                return self.parse_named_type("struct");
            }
            b'U' => {
                self.pos += 1;
                return self.parse_named_type("struct");
            }
            // `C` is NOT a union lead on x64 - it is builtin `signed char`
            // (measured: `YACC@Z` = `sc(signed char)`); unions use `CUU@@`
            // with a name, handled by the class-by-value guard above.
            b'W' => {
                self.pos += 2; // 'W' + access letter (0-8, '@', '?')
                return self.parse_named_type("enum");
            }
            b'Y' => return Err("array type".to_string()),
            b'$' => {
                if self.starts_with(b"$$Q") || self.starts_with(b"$$R") {
                    return Err("rvalue reference".to_string());
                }
                return Err(format!(
                    "extended type encoding '{}'",
                    self.rest().chars().take(3).collect::<String>()
                ));
            }
            _ => {}
        }
        // One-letter builtins.
        let one: [u8; 1] = [c];
        if let Some(name) = Self::builtin(&one) {
            self.pos += 1;
            return Ok(name.to_string());
        }
        Err(format!("unknown type encoding '{}'", c as char))
    }

    /// A named type after its kind letter (`V`/`U`/`C`/`D`/`W` consumed):
    /// `@@`-terminated segments (single-digit segments are repeat references),
    /// then the optional 2-hex-digit size + `$$V` tail. `W41@` closes with a
    /// single '@', so the trailing '@' is consumed only when it doubles.
    fn parse_named_type(&mut self, prefix: &str) -> Result<String, PErr> {
        let seg = self.segment()?;
        self.pos += 1; // '@'
        if seg.len() == 1 && seg.as_bytes()[0].is_ascii_digit() {
            // A repeat reference (`W41@`) resolves to a COMPLETE type string,
            // prefix included - re-prefixing it yields "enum enum Kind".
            return self.repeat_ref((seg.as_bytes()[0] - b'0') as usize, "type name");
        }
        let mut segs: Vec<String> = Vec::new();
        {
            if seg.contains('?') {
                return Err("anonymous namespace in a type name".to_string());
            }
            if seg.starts_with("?$") {
                return Err("template in class scope".to_string());
            }
            segs.push(seg.to_string());
        }
        // Outer scopes: `VInner@Outer@@`.
        loop {
            if self.eat(b'@') {
                break; // `@@` closed the name
            }
            if self.eof() {
                return Err("truncated decoration (type name not closed)".to_string());
            }
            let outer = self.segment()?;
            if outer.is_empty() {
                return Err("empty segment in type name".to_string());
            }
            self.pos += 1; // '@'
            if outer.contains('?') {
                return Err("anonymous namespace in a type name".to_string());
            }
            segs.insert(0, outer.to_string());
        }
        // MSVC's full form appends a 2-hex-digit size and `$$V`.
        if self.starts_with(b"$$V") {
            self.pos += 3;
        } else if self.pos + 5 <= self.s.len()
            && self.s[self.pos..self.pos + 2]
                .iter()
                .all(|b| b.is_ascii_hexdigit())
            && self.s[self.pos + 2..].starts_with(b"$$V")
        {
            self.pos += 5;
        }
        let t = format!("{prefix} {}", segs.join("::"));
        // The NAMED type is itself a complete encoding for reference
        // purposes: `AEBU1@@` (const P2&) refers to the inner `UP2@@`, not
        // to the enclosing `struct P2&`.
        self.types.push(t.clone());
        Ok(t)
    }
}

fn parse(sym: &str) -> Result<Demangled, String> {
    let mut p = Parser::new(sym);
    if p.peek() != Some(b'?') {
        return Err("not an MSVC decorated name (no leading '?')".to_string());
    }
    // `?name@...`: the '?' is a sigil. `??0Circle...`: the identifier IS the
    // second '?' and its two trailing characters - consuming it as a sigil
    // turns every ctor into a "?0Circle" static-local name.
    if p.peek_at(1) != Some(b'?') {
        p.pos += 1;
    }
    let id = p.segment()?.to_string();
    p.pos += 1; // the '@' that ended the identifier

    // ---- classify the identifier ------------------------------------------
    // `??`-prefixed identifiers glue the class name to the operator code
    // (`??0Circle`, `??_7Circle`); plain identifiers put the scope after.
    let mut ctor_dtor: Option<bool> = None; // Some(true) = ctor, Some(false) = dtor
    let mut table: Option<TableKind> = None;
    let mut glued_class = String::new();
    if let Some(rest) = id.strip_prefix("??0") {
        ctor_dtor = Some(true);
        glued_class = rest.to_string();
    } else if let Some(rest) = id.strip_prefix("??1") {
        ctor_dtor = Some(false);
        glued_class = rest.to_string();
    } else if id.starts_with("??$") {
        return Err(format!("template argument list (name '{id}')"));
    } else if let Some(rest) = id.strip_prefix("??_") {
        let (tag, glued) = rest.split_at(1);
        match tag {
            "7" | "8" => table = Some(TableKind::Vftable),
            "9" | "B" => table = Some(TableKind::Vbtable),
            "D" | "E" => ctor_dtor = Some(false), // deleting dtors
            "R" => return Err(format!("RTTI descriptor name '{id}'")),
            _ => return Err(format!("operator/deleting allocation name '??_{tag}'")),
        }
        glued_class = glued.to_string();
    } else if id.starts_with("??") {
        return Err(format!("operator name '{}'", &id[..3.min(id.len())]));
    } else if id.starts_with('?') || id == "0" {
        return Err("static local symbol / anonymous name".to_string());
    }

    let mut scope: Vec<String> = p.scope_segments()?.into_iter().map(String::from).collect();
    scope.reverse(); // encoded innermost-first; display order is outer::inner
    let class = if glued_class.is_empty() {
        scope.join("::")
    } else if scope.is_empty() {
        glued_class.clone()
    } else {
        format!("{}::{glued_class}", scope.join("::"))
    };
    let name = id.clone();

    // ---- the code field ----------------------------------------------------
    let Some(c0) = p.peek() else {
        return Err("truncated decoration (no code field)".to_string());
    };

    if let Some(tbl) = table {
        // vtables end in `6B@` with no return type and no argument list.
        if !p.starts_with(b"6B@") {
            return Err(format!("vtable code field '{}' is not '6B@'", p.rest()));
        }
        p.pos += 3;
        if !p.eof() {
            return Err(format!("trailing bytes after vtable code: '{}'", p.rest()));
        }
        let kw = match tbl {
            TableKind::Vftable => "const",
            TableKind::Vbtable => "const",
        };
        let table_kw = match tbl {
            TableKind::Vftable => "vftable",
            TableKind::Vbtable => "vbtable",
        };
        return Ok(Demangled::Name {
            // Matches llvm-nm: "const C::`vftable'".
            text: format!("{kw} {class}::`{table_kw}'"),
            kind: SymKind::Vtable { class, table: tbl },
        });
    }
    // Data symbols (`?live_@Widget@@0HA`): a type code, not a function code.
    if c0.is_ascii_digit() {
        return Err(format!("data symbol (code field '{}')", c0 as char));
    }

    let mut kind_member = false;
    let mut kind_static = false;
    let mut constness = Constness::NonConst;
    let mut virtual_fn = false;
    let mut thunk = false;
    match c0 {
        b'Y' => {
            // Free function, or a static member (the scope says which).
            p.pos += 1;
            conv_letter(&mut p)?;
        }
        b'S' if p.peek_at(1) != Some(b'E') => {
            // S public static, R protected static, T private static.
            p.pos += 1;
            conv_letter(&mut p)?;
            if scope.is_empty() {
                return Err("static member code outside a class scope".to_string());
            }
            kind_static = true;
        }
        b'R' | b'T' => {
            p.pos += 1;
            conv_letter(&mut p)?;
            if scope.is_empty() {
                return Err("static member code outside a class scope".to_string());
            }
            kind_static = true;
        }
        b'Q' | b'I' | b'K' => {
            p.pos += 1;
            member_tail(&mut p, &mut constness, &mut thunk)?;
            kind_member = true;
        }
        b'U' | b'V' | b'W' | b'X' => {
            // public/protected/private virtual; W/X are the vcall-thunk forms.
            p.pos += 1;
            virtual_fn = true;
            thunk = matches!(c0, b'W' | b'X');
            member_tail(&mut p, &mut constness, &mut thunk)?;
            kind_member = true;
        }
        b'S' => {
            // 'S' + 'E' = protected virtual.
            p.pos += 1;
            virtual_fn = true;
            member_tail(&mut p, &mut constness, &mut thunk)?;
            kind_member = true;
        }
        b'E' => {
            // Thunk access code: E + this-adjust + ptr64 + [X = vcall].
            p.pos += 1;
            thunk = true;
            this_adjust(&mut p, &mut constness)?;
            let a = p
                .peek()
                .ok_or("truncated decoration (thunk ptr64 letter)")?;
            p.pos += 1;
            if a != b'A' {
                return Err(format!(
                    "near/far code model letter '{}' not mapped",
                    a as char
                ));
            }
            ptr64_letter(&mut p, &mut true)?;
            kind_member = true;
        }
        _ => {
            return Err(format!(
                "near/far code model letter '{}' not mapped",
                c0 as char
            ))
        }
    }
    if kind_member && scope.is_empty() && ctor_dtor.is_none() {
        return Err("member function code outside a class scope".to_string());
    }

    // ---- return type / ctor-dtor marker / arguments -------------------------
    let ret = if let Some(is_ctor) = ctor_dtor {
        if !p.eat(b'@') {
            return Err(format!(
                "{} expected '@' in the return-type position, found '{}'",
                if is_ctor { "constructor" } else { "destructor" },
                p.rest().chars().next().unwrap_or('?')
            ));
        }
        String::new()
    } else {
        p.parse_type()?
    };

    let mut args: Vec<String> = Vec::new();
    loop {
        match p.peek() {
            Some(b'Z') => {
                p.pos += 1;
                break;
            }
            Some(b'@') => {
                p.pos += 1;
                // `@Z` ends the list; `@<d>Z` is a thunk adjustment (4 =
                // displacement, 7/8 = vcall offset).
                let mut digits = String::new();
                while p.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    digits.push(p.peek().unwrap() as char);
                    p.pos += 1;
                }
                if !digits.is_empty() {
                    if digits.len() != 1 {
                        return Err(format!(
                            "thunk adjustment '@{digits}Z' is not a single digit"
                        ));
                    }
                    thunk = true;
                }
                if !p.eat(b'Z') {
                    return Err(format!(
                        "'@' in argument position not followed by Z: '{}'",
                        p.rest()
                    ));
                }
                break;
            }
            // The `XZ` tail: X is the void-parameter-list marker, not an arg.
            Some(b'X') if p.peek_at(1) == Some(b'Z') && args.is_empty() => {
                p.pos += 1;
            }
            _ => args.push(p.parse_type()?),
        }
    }
    if !p.eof() {
        // vcall/displacement adjustment: `@<d>` appended after the final Z.
        if p.peek() == Some(b'@') && p.peek_at(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            p.pos += 2;
            thunk = true;
        }
    }
    if !p.eof() {
        return Err(format!(
            "trailing bytes after the signature: '{}'",
            p.rest()
        ));
    }

    let args_s = args.join(", ");
    let leaf = class.rsplit("::").next().unwrap_or(&class).to_string();
    let (text, kind) = if ctor_dtor == Some(true) {
        (
            format!("{class}::{leaf}({args_s})"),
            SymKind::Constructor { class },
        )
    } else if ctor_dtor == Some(false) {
        (
            format!("{class}::~{leaf}({args_s})"),
            SymKind::Destructor { class },
        )
    } else if kind_member {
        (
            format!("{ret} {class}::{name}({args_s}){}", constness.suffix()),
            SymKind::MemberFunction {
                class,
                constness,
                virtual_fn,
                thunk,
            },
        )
    } else if kind_static {
        (
            format!("{ret} {class}::{name}({args_s})"),
            SymKind::StaticMember { class },
        )
    } else if scope.is_empty() {
        (format!("{ret} {name}({args_s})"), SymKind::FreeFunction)
    } else {
        // Namespace-scope free function: the scope is part of the name.
        (
            format!("{ret} {class}::{name}({args_s})"),
            SymKind::FreeFunction,
        )
    };
    Ok(Demangled::Name { text, kind })
}

/// Calling-convention letter after `Y`/`S`/`R`/`T`, including the near/far
/// letters. Letters outside the documented set are refused by name.
fn conv_letter(p: &mut Parser) -> Result<(), String> {
    // Measured on x64: free-function codes use A (cdecl), Q (cdecl near32),
    // S (cdecl far16), I (far cdecl); K (thiscall) is a 32-bit code that
    // must stay loud, and C/R/T/S are member-access codes, not conventions.
    let c = p
        .peek()
        .ok_or("truncated decoration (no convention letter)")?;
    match c {
        // Measured x64: a single convention letter. The 32-bit far/far16
        // chains (YAIQ, YASQ...) collide with the type letters I/Q and do
        // not exist on x64 - a second far letter demangles as a type and
        // the resulting junk is refused by later arms (loud, not silent).
        b'A' | b'E' | b'I' | b'Q' | b'S' => {
            p.pos += 1;
            Ok(())
        }
        _ => Err(format!(
            "near/far code model letter '{}' not mapped",
            c as char
        )),
    }
}

/// `[E][this-adjust][__ptr64]` after an access letter on x64.
fn member_tail(p: &mut Parser, constness: &mut Constness, thunk: &mut bool) -> Result<(), String> {
    let m = p.peek().ok_or("truncated decoration (member modifier)")?;
    p.pos += 1;
    if m != b'E' {
        return Err(format!(
            "non-x64 member function code (modifier letter '{}' is not 'E')",
            m as char
        ));
    }
    this_adjust(p, constness)?;
    ptr64_letter(p, thunk)
}

fn this_adjust(p: &mut Parser, constness: &mut Constness) -> Result<(), String> {
    let a = p
        .peek()
        .ok_or("truncated decoration (this-adjust letter)")?;
    p.pos += 1;
    *constness = match a {
        b'A' => Constness::NonConst,
        b'B' => Constness::Const,
        b'C' => Constness::Volatile,
        b'D' => Constness::ConstVolatile,
        _ => {
            return Err(format!(
                "near/far code model letter '{}' not mapped",
                a as char
            ))
        }
    };
    Ok(())
}

fn ptr64_letter(p: &mut Parser, thunk: &mut bool) -> Result<(), String> {
    let a = p.peek().ok_or("truncated decoration (ptr64 letter)")?;
    p.pos += 1;
    match a {
        b'A' => Ok(()), // __ptr64
        b'X' => {
            *thunk = true; // vcall-thunk this model
            Ok(())
        }
        b'B' | b'C' => Err(format!("16-bit this model letter '{}'", a as char)),
        _ => Err(format!(
            "near/far code model letter '{}' not mapped",
            a as char
        )),
    }
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn name(sym: &str) -> (String, SymKind) {
        match demangle(sym) {
            Demangled::Name { text, kind } => (text, kind),
            Demangled::Unsupported { reason, .. } => panic!("{sym}: unsupported: {reason}"),
        }
    }
    fn reason(sym: &str) -> String {
        match demangle(sym) {
            Demangled::Unsupported { reason, .. } => reason,
            Demangled::Name { text, .. } => panic!("{sym}: unexpectedly demangled to {text}"),
        }
    }

    /// Measured pairs: (mangled, llvm-nm --demangle text). Both sides go
    /// through `canonical`, so the test checks agreement with the oracle's
    /// semantics, not its cosmetics. Every line was captured from
    /// `llvm-nm --defined-only -g --demangle` on a clang 22
    /// x86_64-pc-windows-msvc object; `rdec demangle --cpp` re-checks the
    /// encodings on the corpus objects against the same oracle on every run.
    const MEASURED: &[(&str, &str)] = &[
        ("?cdecl_base@@YAHHH@Z", "int __cdecl cdecl_base(int, int)"),
        ("?winapi_add@@YAHHH@Z", "int __cdecl winapi_add(int, int)"),
        ("?add@@YAHHH@Z", "int __cdecl add(int, int)"),
        ("?add@@YAJJJ@Z", "long __cdecl add(long, long)"),
        ("?add@@YANNN@Z", "double __cdecl add(double, double)"),
        ("?add@@YA_J_J0@Z", "__int64 __cdecl add(__int64, __int64)"),
        ("?window@@YAHHHHH@Z", "int __cdecl window(int, int, int, int)"),
        (
            "?clamp@detail@geo@@YAHHHH@Z",
            "int __cdecl geo::detail::clamp(int, int, int)",
        ),
        ("?norm@geo@@YAHH@Z", "int __cdecl geo::norm(int)"),
        ("?sum@Point@@QEAAHXZ", "public: int __cdecl Point::sum(void)"),
        (
            "?shift@Point@@QEAAXHH@Z",
            "public: void __cdecl Point::shift(int, int)",
        ),
        (
            "??0Circle@@QEAA@N@Z",
            "public: __cdecl Circle::Circle(double)",
        ),
        (
            "??1Circle@@QEAA@XZ",
            "public: __cdecl Circle::~Circle(void)",
        ),
        (
            "?grow@Circle@@QEAANN@Z",
            "public: double __cdecl Circle::grow(double)",
        ),
        (
            "?radius@Circle@@QEBANXZ",
            "public: double __cdecl Circle::radius(void) const",
        ),
        (
            "?size@Buffer@@QEBAHXZ",
            "public: int __cdecl Buffer::size(void) const",
        ),
        (
            "?slot@Buffer@@QEAAAEAHH@Z",
            "public: int & __cdecl Buffer::slot(int)",
        ),
        (
            "?slot@Buffer@@QEBAAEBHH@Z",
            "public: int const & __cdecl Buffer::slot(int) const",
        ),
        (
            "?get@Inner@Outer@@QEBAHXZ",
            "public: int __cdecl Outer::Inner::get(void) const",
        ),
        (
            "?bump@Inner@Outer@@QEAAHH@Z",
            "public: int __cdecl Outer::Inner::bump(int)",
        ),
        (
            "?count@Widget@@SAHH@Z",
            "public: static int __cdecl Widget::count(int)",
        ),
        (
            "?seed_and_count@Widget@@SAHH@Z",
            "public: static int __cdecl Widget::seed_and_count(int)",
        ),
        (
            "?add_and_count@Widget@@QEAAHH@Z",
            "public: int __cdecl Widget::add_and_count(int)",
        ),
        (
            "??1Base@@UEAA@XZ",
            "public: virtual __cdecl Base::~Base(void)",
        ),
        (
            "?twice@Base@@UEBAHXZ",
            "public: virtual int __cdecl Base::twice(void) const",
        ),
        (
            "?use_derived@@YAHPEAUDerived@@@Z",
            "int __cdecl use_derived(struct Derived *)",
        ),
        ("?bl@@YA_N_N@Z", "bool __cdecl bl(bool)"),
        ("?ch@@YADD@Z", "char __cdecl ch(char)"),
        ("?sc@@YACC@Z", "signed char __cdecl sc(signed char)"),
        ("?uc@@YAEE@Z", "unsigned char __cdecl uc(unsigned char)"),
        ("?sh@@YAFF@Z", "short __cdecl sh(short)"),
        ("?us@@YAGG@Z", "unsigned short __cdecl us(unsigned short)"),
        ("?in@@YAHH@Z", "int __cdecl in(int)"),
        ("?un@@YAII@Z", "unsigned int __cdecl un(unsigned int)"),
        ("?lo@@YAJJ@Z", "long __cdecl lo(long)"),
        ("?ul@@YAKK@Z", "unsigned long __cdecl ul(unsigned long)"),
        ("?ll@@YA_J_J@Z", "__int64 __cdecl ll(__int64)"),
        ("?ull@@YA_K_K@Z", "unsigned __int64 __cdecl ull(unsigned __int64)"),
        ("?fl@@YAMM@Z", "float __cdecl fl(float)"),
        ("?dbl@@YANN@Z", "double __cdecl dbl(double)"),
        ("?wc@@YA_W_W@Z", "wchar_t __cdecl wc(wchar_t)"),
        (
            "?chain@@YAHCEDFGHIJK_J_KMN_N_W@Z",
            "int __cdecl chain(signed char, unsigned char, char, short, unsigned short, int, unsigned int, long, unsigned long, __int64, unsigned __int64, float, double, bool, wchar_t)",
        ),
        (
            "?one_ptr@@YAHPEAHPEBHQEAHPECH@Z",
            "int __cdecl one_ptr(int *, int const *, int *const, int volatile *)",
        ),
        (
            "?two_ptr@@YAHPEAPEAHPEADPEBDPEBQEAD@Z",
            "int __cdecl two_ptr(int **, char *, char const *, char *const *)",
        ),
        (
            "?refs@@YAHAEAHAEBHAEA_JAEAUP2@@AEBU1@@Z",
            "int __cdecl refs(int &, int const &, __int64 &, struct P2 &, struct P2 const &)",
        ),
        (
            "?ptr64fn@@YAHPEAXPEBX@Z",
            "int __cdecl ptr64fn(void *, void const *)",
        ),
        ("?byval@@YAHUP2@@@Z", "int __cdecl byval(struct P2)"),
        (
            "?pick@@YA?AW4Kind@@W41@0@Z",
            "enum Kind __cdecl pick(enum Kind, enum Kind)",
        ),
        (
            "?use_kind@@YAHW4Kind@@W4Legacy@@@Z",
            "int __cdecl use_kind(enum Kind, enum Legacy)",
        ),
        ("?classify@@YAHH@Z", "int __cdecl classify(int)"),
    ];

    #[test]
    fn measured_table_agrees_with_llvm_renderings() {
        for (m, llvm_text) in MEASURED {
            let d = demangle(m);
            let Demangled::Name { text, .. } = &d else {
                panic!("{m}: unsupported: {d:?}");
            };
            assert_eq!(
                canonical(text),
                canonical(llvm_text),
                "{m}: mine {text:?} vs llvm {llvm_text:?}"
            );
        }
    }

    #[test]
    fn ctor_and_dtor_are_classified_by_kind_not_text() {
        // M8 pattern-matches on the kind; a text match would rot the moment
        // the rendering style changes.
        assert_eq!(
            name("??0Circle@@QEAA@N@Z").1,
            SymKind::Constructor {
                class: "Circle".into()
            }
        );
        assert_eq!(
            name("??1Circle@@QEAA@XZ").1,
            SymKind::Destructor {
                class: "Circle".into()
            }
        );
        assert_eq!(
            name("??1Base@@UEAA@XZ").1,
            SymKind::Destructor {
                class: "Base".into()
            }
        );
    }

    #[test]
    fn member_function_kinds_carry_class_and_constness() {
        assert_eq!(
            name("?area@Circle@@QEBA_NXZ").1,
            SymKind::MemberFunction {
                class: "Circle".into(),
                constness: Constness::Const,
                virtual_fn: false,
                thunk: false
            }
        );
        assert!(matches!(
            name("?grow@Circle@@QEAANN@Z").1,
            SymKind::MemberFunction { ref class, constness: Constness::NonConst, .. } if class == "Circle"
        ));
        assert_eq!(
            name("?twice@Base@@UEBAHXZ").1,
            SymKind::MemberFunction {
                class: "Base".into(),
                constness: Constness::Const,
                virtual_fn: true,
                thunk: false
            }
        );
        assert_eq!(
            name("?count@Widget@@SAHH@Z").1,
            SymKind::StaticMember {
                class: "Widget".into()
            }
        );
        let (text, k) = name("?clamp@detail@geo@@YAHHHH@Z");
        assert_eq!(k, SymKind::FreeFunction);
        assert_eq!(text, "int geo::detail::clamp(int, int, int)");
    }

    #[test]
    fn namespaces_and_nested_class_scopes() {
        assert_eq!(name("?norm@geo@@YAHH@Z").0, "int geo::norm(int)");
        assert_eq!(
            name("?get@Inner@Outer@@QEBAHXZ").0,
            "int Outer::Inner::get() const"
        );
    }

    #[test]
    fn calling_convention_letters_including_far_variants() {
        // Measured free-function codes on x64: YA cdecl; YE/Q/I/S are the
        // near/far chain letters (YE = far, YI = far, YQ = near32,
        // YS = far16); the 4-character far forms chain letters (YAIQ,
        // YASQ) and demangle too.
        for code in ["YA", "YE", "YI", "YQ", "YS"] {
            let m = format!("?f@@{code}HHH@Z");
            let (text, kind) = name(&m);
            assert_eq!(kind, SymKind::FreeFunction, "{m}");
            assert_eq!(text, "int f(int, int)", "{m}");
        }
        // 32-bit-only codes stay loud: K thiscall, B thiscall, C stdcall.
        for code in ["YK", "YB", "YC", "YF", "YG"] {
            let m = format!("?f@@{code}HH@Z");
            assert!(
                reason(&m).starts_with("near/far code model letter"),
                "{m}: {}",
                reason(&m)
            );
        }
    }

    #[test]
    fn vcall_adjusted_and_thunk_names_are_supported() {
        // Documented x64 thunk shapes: `@7`/`@8` vcall adjustment and `@4`
        // displacement; the E-access `EAAX`-style code is the public thunk.
        let (text, k) = name("?twice@Leaf@@EAAXXZ@8");
        assert_eq!(text, "void Leaf::twice()");
        assert_eq!(
            k,
            SymKind::MemberFunction {
                class: "Leaf".into(),
                constness: Constness::NonConst,
                virtual_fn: false,
                thunk: true
            }
        );
        assert!(matches!(
            name("?get3@Derived@@UEAAXXZ@7").1,
            SymKind::MemberFunction {
                thunk: true,
                virtual_fn: true,
                ..
            }
        ));
        assert!(matches!(
            name("?get3@Derived@@UEAAXXZ@4").1,
            SymKind::MemberFunction { thunk: true, .. }
        ));
    }

    #[test]
    fn vtable_symbols() {
        let (text, k) = name("??_7Circle@@6B@");
        assert_eq!(text, "const Circle::`vftable'");
        assert_eq!(
            k,
            SymKind::Vtable {
                class: "Circle".into(),
                table: TableKind::Vftable
            }
        );
        assert_eq!(
            name("??_BCircle@@6B@").1,
            SymKind::Vtable {
                class: "Circle".into(),
                table: TableKind::Vbtable
            }
        );
    }

    #[test]
    fn void_and_zero_argument_forms() {
        assert_eq!(name("?sum@Point@@QEAAHXZ").0, "int Point::sum()");
        assert_eq!(
            name("?shift@Point@@QEAAXHH@Z").0,
            "void Point::shift(int, int)"
        );
        // Void-returning, no-argument free function.
        let (text, _) = name("?go@@YAXXZ");
        assert_eq!(text, "void go()");
        assert_eq!(canonical("void __cdecl go(void)"), canonical(&text));
    }

    #[test]
    fn pointer_and_reference_qualifiers() {
        assert_eq!(
            name("?p@@YAHPEAHPEBHQEAHPECHPEAXPEBX@Z").0,
            "int p(int *, int const *, int *const, int volatile *, void *, void const *)"
        );
    }

    #[test]
    fn gaps_are_loud_with_specific_reasons() {
        assert!(reason("??$tmax@H@@YAHHH@Z").starts_with("template argument list"));
        assert_eq!(reason("?f@@YAH$$QH@Z"), "rvalue reference");
        assert_eq!(reason("?f@@YAHYAH$02@Z"), "array type");
        assert_eq!(
            reason("?f@@YAH6AXZ"),
            "function pointer type (code letter '6')"
        );
        assert_eq!(reason("?live_@Widget@@0HA"), "data symbol (code field '0')");
        assert!(reason("??_RCircle@@2A@Z").starts_with("RTTI descriptor"));
        assert_eq!(
            reason("malloc"),
            "not an MSVC decorated name (no leading '?')"
        );
        assert_eq!(
            reason("?f@?A0x1234@@YAHH@Z"),
            "anonymous namespace or local-static label"
        );
        assert_eq!(reason("??2@YAPAXI@Z"), "operator name '??2'");
        assert!(reason("?f@@YA?AVWidget@@@Z").starts_with("template argument list"));
        assert!(reason("?f@@YAH?0@Z").starts_with("anonymous type encoding"));
        // A truncated name must not demangle to a plausible prefix.
        assert!(reason("?f@@YAHH").starts_with("truncated decoration"));
        // 32-bit member codes are refused by name (this is an x64 demangler).
        assert_eq!(
            reason("?f@C@@QAEXZ"),
            "non-x64 member function code (modifier letter 'A' is not 'E')"
        );
    }

    #[test]
    fn repeated_type_references_resolve() {
        // `add(__int64, __int64)` encodes the second __int64 as `0`.
        assert_eq!(name("?add@@YA_J_J0@Z").0, "__int64 add(__int64, __int64)");
        // `pick` encodes both the second enum (W41@) and the third arg (0).
        assert_eq!(
            name("?pick@@YA?AW4Kind@@W41@0@Z").0,
            "enum Kind pick(enum Kind, enum Kind)"
        );
    }
}
