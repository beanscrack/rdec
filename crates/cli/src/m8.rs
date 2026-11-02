//! M8 front door: the C++ corpus, and the demangler gate.
//!
//! `rdec corpus --cpp` compiles `CPP_SNIPPETS` with clang++ for
//! `x86_64-pc-windows-msvc` and writes `corpus/cpp/{src,obj,manifest.tsv}`
//! in the same shape as the C corpus (`name<TAB>opt<TAB>obj<TAB>bytes<TAB>
//! fnv1a64`, clang stamp header, COFF TimeDateStamp zeroed at offset 4).
//!
//! `rdec demangle --cpp` is the M8 gate. For every defined *function* symbol
//! in the C++ corpus it demangles with our from-scratch demangler and with
//! `llvm-nm --demangle`, canonicalizes both, and compares. The failure modes
//! it refuses, mirroring the M1 rule that a gate which checked nothing has
//! proven nothing: any disagreement, zero compared symbols, and a supported
//! count below the stated floor. `Unsupported` demangles are NOT failures —
//! they are reported, grouped by reason, so the gaps (templates first among
//! them) stay visible between milestones.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_demangle::{canonical, demangle, Demangled};

/// One C++ corpus file: (name, what it forces the demangler to handle, code).
/// No standard-library includes: the corpus must compile on a bare host with
/// no MSVC SDK headers on PATH, exactly like the C corpus.
pub const CPP_SNIPPETS: &[(&str, &str, &str)] = &[
    (
        "struct_methods",
        "struct with methods: member codes QEAA/QEA, thiscall, the `@@` scope form",
        "struct Point { int x, y; int sum(); void shift(int dx, int dy); };
int Point::sum() { return x + y; }
void Point::shift(int dx, int dy) { x += dx; y += dy; }
",
    ),
    (
        "class_ctor_dtor",
        "ctor ??0 / dtor ??1 with the @ return marker, private data, const method",
        "class Circle { double r; public: Circle(double radius); ~Circle(); double radius() const; double grow(double by); };
Circle::Circle(double radius) : r(radius) {}
Circle::~Circle() {}
double Circle::radius() const { return r; }
double Circle::grow(double by) { r += by; return r; }
",
    ),
    (
        "const_overloads",
        "const overload pair (A vs B this-adjust) and reference-returning members",
        "class Buffer { int n_; public: int size() const; int &slot(int i); const int &slot(int i) const; };
int Buffer::size() const { return n_; }
int &Buffer::slot(int i) { (void)i; return n_; }
const int &Buffer::slot(int i) const { (void)i; return n_; }
",
    ),
    (
        "namespaces",
        "namespace and nested namespace scopes (?f@N@@, ?g@inner@outer@@)",
        "namespace geo { namespace detail { int clamp(int v, int lo, int hi); } int norm(int v); }
int geo::detail::clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
int geo::norm(int v) { return detail::clamp(v, 0, 10); }
",
    ),
    (
        "nested_class",
        "class nested in a class: ?get@Inner@Outer@@ scope chains",
        "struct Outer { int a; struct Inner { int b; int get() const; int bump(int d); }; Inner in; int use(int v); };
int Outer::Inner::get() const { return b; }
int Outer::Inner::bump(int d) { b += d; return b; }
int Outer::use(int v) { in.bump(v); return in.get() + a; }
",
    ),
    (
        "free_overloads",
        "overloaded free functions: same name, argument-list-distinguishing encodings",
        "int add(int a, int b); double add(double a, double b); long add(long a, long b); long long add(long long a, long long b);
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
long add(long a, long b) { return a + b; }
long long add(long long a, long long b) { return a + b; }
",
    ),
    (
        "static_members",
        "static member functions (SA codes) + a static data symbol to filter out",
        "class Widget { static int live_; public: static int count(int seed); static int seed_and_count(int seed); int add_and_count(int seed); };
int Widget::live_ = 0;
int Widget::count(int seed) { return live_ + seed; }
int Widget::seed_and_count(int seed) { live_ = seed; return live_; }
int Widget::add_and_count(int seed) { live_ += seed; return live_; }
",
    ),
    (
        "builtin_params",
        "every x64 builtin letter, in return and argument position (C E D F G H I J K _J _K M N _N _W)",
        "signed char sc(signed char v); unsigned char uc(unsigned char v); char ch(char v); short sh(short v);
unsigned short us(unsigned short v); int in(int v); unsigned int un(unsigned int v); long lo(long v);
unsigned long ul(unsigned long v); long long ll(long long v); unsigned long long ull(unsigned long long v);
float fl(float v); double dbl(double v); bool bl(bool v); wchar_t wc(wchar_t v);
signed char sc(signed char v) { return v; }
unsigned char uc(unsigned char v) { return v; }
char ch(char v) { return v; }
short sh(short v) { return v; }
unsigned short us(unsigned short v) { return v; }
int in(int v) { return v; }
unsigned int un(unsigned int v) { return v; }
long lo(long v) { return v; }
unsigned long ul(unsigned long v) { return v; }
long long ll(long long v) { return v; }
unsigned long long ull(unsigned long long v) { return v; }
float fl(float v) { return v; }
double dbl(double v) { return v; }
bool bl(bool v) { return v; }
wchar_t wc(wchar_t v) { return v; }
int chain(signed char a, unsigned char b, char c, short d, unsigned short e, int f, unsigned int g, long h,
          unsigned long i, long long j, unsigned long long k, float l, double m, bool n, wchar_t o);
int chain(signed char a, unsigned char b, char c, short d, unsigned short e, int f, unsigned int g, long h,
          unsigned long i, long long j, unsigned long long k, float l, double m, bool n, wchar_t o)
{ return (int)(a + b + c + d + e + f + g + h + i + j + k + (long long)l + (long long)m + (int)n + (int)o); }
",
    ),
    (
        "ref_ptr_params",
        "x64 pointer/reference prefixes (PEA, PEB, QEA, PEC, AEA, AEB), volatile, **",
        "struct P2 { int a; int b; };
int byval(P2 p) { return p.a + p.b; }
int one_ptr(int *p, const int *q, int *const r, volatile int *v) { return *p + *q + *r + *v; }
int two_ptr(int **pp, char *s, const char *c, char *const *cc) { return **pp + *s + *c + **cc; }
int refs(int &a, const int &b, long long &c, P2 &p, const P2 &q) { return a + b + (int)c + p.a + q.b; }
int ptr64fn(void *a, const void *b) { return (int)(long long)a + (int)(long long)b; }
",
    ),
    (
        "default_args",
        "default arguments: mangled away; exercises the @Z multi-arg terminator",
        "int window(int x, int y = 8, int w = 640, int h = 480);
int window(int x, int y, int w, int h) { return x * 1 + y * 2 + w * 3 + h * 4; }
",
    ),
    (
        "calling_conventions",
        "__cdecl/__stdcall/__fastcall free functions (x64 unifies them; the @YA letter set must map all)",
        "__stdcall int winapi_add(int a, int b);
__fastcall int fast_scale(int a, int b);
__cdecl int cdecl_base(int a, int b);
int winapi_add(int a, int b) { return a + b; }
int fast_scale(int a, int b) { return a * b; }
int cdecl_base(int a, int b) { return a - b; }
",
    ),
    (
        "virtual_inherit",
        "virtual methods (UEAA/UEBA), virtual dtor, override chain, multiple inheritance",
        "struct Base { virtual int value() const; virtual int twice() const; virtual ~Base(); int seed; };
struct Mid : Base { int value() const override; int twice() const override; ~Mid() override; int extra; };
struct Leaf : Mid { int value() const override; int probe() const; int offset; };
int Base::value() const { return seed; }
int Base::twice() const { return value() * 2; }
Base::~Base() {}
int Mid::value() const { return Base::value() + extra; }
int Mid::twice() const { return Base::twice() + 1; }
Mid::~Mid() {}
int Leaf::value() const { return Mid::value() + offset; }
int Leaf::probe() const { return twice(); }
struct Alpha { virtual int get() const; int a; };
struct Beta { virtual int get3() const; int b; };
struct Derived : Alpha, Beta { int get() const override; int get3() const override; int d; };
int Alpha::get() const { return a; }
int Beta::get3() const { return b; }
int Derived::get() const { return d; }
int Derived::get3() const { return d + 1; }
int use_derived(Derived *p) { return p->get() + p->get3(); }
",
    ),
    (
        "vcall_thunk",
        "virtual base + override across the vbtable (the vcall-offset territory)",
        "struct VBase { virtual int vf() const; virtual int vg() const; int v; };
struct VMid : virtual VBase { int vf() const override; int via_base() const; int m; };
struct VLeaf : VMid { int vg() const override; int leaf; };
int VBase::vf() const { return v; }
int VBase::vg() const { return v + 1; }
int VMid::vf() const { return v + m; }
int VMid::via_base() const { const VBase *b = this; return b->vg(); }
int VLeaf::vg() const { return leaf; }
",
    ),
    (
        "scoped_enum",
        "scoped enum (W4 + the ?A wrapper) and unscoped enum in return/argument position",
        "enum class Kind : int { kA = 1, kB = 2 };
enum Legacy { kLow, kHigh };
int use_kind(Kind k, Legacy l);
Kind pick(Kind a, Kind b);
int classify(int v);
int use_kind(Kind k, Legacy l) { return (int)k + (int)l; }
Kind pick(Kind a, Kind b) { return a > b ? a : b; }
int classify(int v) { return v > 0 ? 1 : 0; }
",
    ),
    (
        "template_gap",
        "instantiated template function: the loud-gap fixture, must land in Unsupported",
        "template <typename T> T tmax(T a, T b) { return a > b ? a : b; }
int use_template(int a, int b) { return tmax(a, b) + tmax(a * 2, b * 2); }
",
    ),
];

pub fn clangxx_path() -> String {
    std::env::var("RDEC_CLANGXX").unwrap_or_else(|_| "clang++".to_string())
}

pub fn nm_path() -> PathBuf {
    std::env::var_os("RDEC_NM")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("llvm-nm"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    // Same FNV-1a the C corpus manifest uses, so the two manifests are the
    // same kind of document.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `rdec corpus --cpp [--out DIR]` — build corpus/cpp: sources, objects,
/// manifest.tsv (deterministic: COFF TimeDateStamp normalized at offset 4).
pub fn cmd_corpus_cpp(args: &[String]) -> i32 {
    let mut out = PathBuf::from("corpus/cpp");
    if let Some(i) = args.iter().position(|a| a == "--out") {
        if let Some(p) = args.get(i + 1) {
            out = PathBuf::from(p);
        }
    }
    let src_dir = out.join("src");
    let obj_dir = out.join("obj");
    let _ = std::fs::create_dir_all(&src_dir);
    let _ = std::fs::create_dir_all(&obj_dir);

    let cxx = clangxx_path();
    let mut manifest =
        String::from("# clang: (see stamp line below)\nname\topt\tobj\tbytes\tfnv1a64\n");
    let mut built = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (name, exercises, code) in CPP_SNIPPETS {
        let src = src_dir.join(format!("{name}.cpp"));
        let body = format!("/* rdec cpp corpus: {exercises} */\n{code}");
        if let Err(e) = std::fs::write(&src, body.replace("\r\n", "\n")) {
            failures.push(format!("{name}: write source: {e}"));
            continue;
        }
        let obj = obj_dir.join(format!("{name}.O0.obj"));
        let status = Command::new(&cxx)
            .arg("-target")
            .arg("x86_64-pc-windows-msvc")
            .arg("-std=c++17")
            .arg("-O0")
            .arg("-g0")
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                failures.push(format!("{name}: clang++ exit {s}"));
                continue;
            }
            Err(e) => {
                failures.push(format!("{name}: spawn {cxx}: {e}"));
                continue;
            }
        }
        match std::fs::read(&obj) {
            Ok(mut bytes) => {
                if rdec_verify::corpus::normalize_coff_stamp(&mut bytes) {
                    let _ = std::fs::write(&obj, &bytes);
                }
                let h = fnv1a64(&bytes);
                manifest.push_str(&format!(
                    "{name}\tO0\t{}\t{}\t{h:016x}\n",
                    obj.display(),
                    bytes.len()
                ));
                built += 1;
            }
            Err(e) => failures.push(format!("{name}: read obj: {e}")),
        }
    }
    let (ver, tgt) = rdec_verify::corpus::clang_stamp(&clang_path_side());
    manifest = manifest.replace(
        "# clang: (see stamp line below)\n",
        &format!(
            "# clang: {ver}\n# target: {tgt} (objects built -target x86_64-pc-windows-msvc)\n# normalized: COFF TimeDateStamp zeroed at object offset 4\n"
        ),
    );
    let _ = std::fs::write(out.join("manifest.tsv"), manifest);
    for f in &failures {
        println!("FAIL  {f}");
    }
    println!(
        "{} cpp snippets -> {built}/{} objects",
        CPP_SNIPPETS.len(),
        CPP_SNIPPETS.len()
    );
    println!("manifest: {}/manifest.tsv", out.display());
    if failures.is_empty() && built == CPP_SNIPPETS.len() {
        println!("corpus --cpp OK");
        0
    } else {
        1
    }
}

/// The C driver is enough for stamping; clang++ shares its version.
fn clang_path_side() -> String {
    std::env::var("RDEC_CLANG").unwrap_or_else(|_| "clang".to_string())
}

/// A defined function symbol: in a section, external/static, typ has the
/// FUNCTION bit (0x20), and it is not the section-definition symbol.
fn defined_function_symbols(obj: &rdec_arch_x86::loader::Object) -> Vec<String> {
    // Same predicate the loader uses for code_symbols (clang leaves the
    // COFF typ field 0, so the typ&0x20 "function" bit is NOT usable):
    // a symbol that names a location in an executable section.
    let mut names: Vec<String> = obj
        .symbols
        .iter()
        .filter(|s| {
            s.section.map(|x| obj.is_code(x)).unwrap_or(false)
                && !s.name.starts_with('.')
                && (s.storage_class == rdec_arch_x86::loader::STORAGE_EXTERNAL
                    || s.storage_class == rdec_arch_x86::loader::STORAGE_STATIC)
                && s.name.starts_with('?')
        })
        .map(|s| s.name.clone())
        .collect();
    names.sort();
    names.dedup();
    names
}

fn nm_symbols(obj: &Path, demangle_names: bool) -> Result<Vec<(String, String, String)>, String> {
    // `llvm-nm --defined-only -g`: one line per symbol,
    // `ADDRESS TYPE NAME` (NAME = mangled, or demangled with --demangle; the
    // address/type columns are stable between the two invocations, which is
    // how the pairs are matched). Verified by measurement, not assumption.
    let mut cmd = Command::new(nm_path());
    cmd.arg("--defined-only").arg("-g");
    if demangle_names {
        cmd.arg("--demangle");
    }
    cmd.arg(obj);
    let out = cmd
        .output()
        .map_err(|e| format!("run {}: {e}", nm_path().display()))?;
    if !out.status.success() {
        return Err(format!(
            "llvm-nm exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let mut v = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mut it = line.splitn(3, ' ');
        let (Some(addr), Some(typ), Some(name)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if !addr.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue; // `a @feat.00` and friends
        }
        v.push((addr.to_string(), typ.to_string(), name.to_string()));
    }
    Ok(v)
}

/// The M8 gate. Floor: `SUPPORTED_FLOOR` supported symbols on the corpus —
/// set from the measured count minus headroom for codegen drift, so a corpus
/// that silently shrinks (or a demangler that regresses a whole family of
/// encodings) fails loudly instead of comparing an empty set.
pub const SUPPORTED_FLOOR: usize = 60;

pub fn cmd_demangle(args: &[String]) -> Result<i32, String> {
    if !args[1..].iter().any(|a| a == "--cpp") {
        return Err(
            "usage: rdec demangle --cpp   (MSVC C++ names vs llvm-nm, on corpus/cpp)".to_string(),
        );
    }
    let man = std::fs::read_to_string("corpus/cpp/manifest.tsv")
        .map_err(|e| format!("corpus/cpp/manifest.tsv: {e} (run `rdec corpus --cpp` first)"))?;
    let mut files: Vec<String> = Vec::new();
    for line in man.lines() {
        if line.starts_with('#') || line.starts_with("name\t") {
            continue;
        }
        if let Some(obj) = line.split('\t').nth(2) {
            files.push(obj.to_string());
        }
    }
    if files.is_empty() {
        return Err("corpus/cpp/manifest.tsv lists no objects".to_string());
    }

    let mut total = 0usize;
    let mut supported = 0usize;
    let mut agreed = 0usize;
    let mut disagreements: Vec<(String, String, String)> = Vec::new();
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut no_oracle = 0usize;

    for file in &files {
        let obj = rdec_arch_x86::loader::load(Path::new(file))?;
        if !obj.amd64() {
            return Err(format!(
                "{file}: machine {:#06x} is not AMD64 (0x8664)",
                obj.machine
            ));
        }
        let mangled = nm_symbols(Path::new(file), false).map_err(|e| format!("{file}: {e}"))?;
        let demangled = nm_symbols(Path::new(file), true).map_err(|e| format!("{file}: {e}"))?;
        if mangled.len() != demangled.len() {
            return Err(format!(
                "{file}: llvm-nm printed {} symbols plain, {} demangled",
                mangled.len(),
                demangled.len()
            ));
        }
        let oracle: BTreeMap<(String, String, String), String> = mangled
            .iter()
            .zip(demangled.iter())
            .map(|((a, t, m), (a2, t2, d))| {
                if a != a2 || t != t2 {
                    // Pairing is by (addr, type) identity; a mismatch means
                    // llvm-nm reordered between runs, so the pairing itself
                    // is untrustworthy - say so, loudly.
                    (
                        (a.clone(), t.clone(), m.clone()),
                        format!("\u{0}PAIRING-BROKEN {d}"),
                    )
                } else {
                    ((a.clone(), t.clone(), m.clone()), d.clone())
                }
            })
            .collect();

        for sym in defined_function_symbols(&obj) {
            total += 1;
            let ours = demangle(&sym);
            let Demangled::Name { text, .. } = &ours else {
                let Demangled::Unsupported { reason, .. } = &ours else {
                    unreachable!()
                };
                *unsupported.entry(reason.clone()).or_default() += 1;
                continue;
            };
            supported += 1;
            let key = mangled.iter().find(|(_, _, m)| *m == sym).cloned();
            let Some(k) = key else {
                no_oracle += 1;
                continue;
            };
            match oracle.get(&k) {
                None => {
                    no_oracle += 1;
                }
                Some(theirs) if theirs.starts_with('\u{0}') => {
                    disagreements.push((file.clone(), sym.clone(), theirs[1..].to_string()));
                }
                Some(theirs) => {
                    if canonical(text) == canonical(theirs) {
                        agreed += 1;
                    } else {
                        disagreements.push((
                            file.clone(),
                            format!("{sym} => {text}"),
                            theirs.clone(),
                        ));
                    }
                }
            }
        }
    }

    println!(
        "demangle: {total} defined MSVC-decorated function symbols in {} objects (llvm-nm: {})",
        files.len(),
        nm_path().display()
    );
    {
        let dis_n = disagreements.len();
        let uns_n: usize = unsupported.values().sum();
        println!(
            "supported by ours: {supported}, agreed: {agreed}, disagreed: {dis_n}, unsupported-by-mine: {uns_n}, no llvm-nm counterpart: {no_oracle}"
        );
    }
    if unsupported.is_empty() {
        println!("unsupported reasons: none");
    } else {
        let mut v: Vec<(&String, &usize)> = unsupported.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!(
            "unsupported reasons: {}",
            v.iter()
                .map(|(r, n)| format!("{n} x {r}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    for (file, m, theirs) in &disagreements {
        println!("DISAGREED {file}: {m}");
        println!("  llvm-nm: {theirs}");
    }
    if total == 0 || agreed == 0 {
        println!("FAIL: {total} symbols found, {agreed} compared - a gate that compares nothing proves nothing");
        return Ok(1);
    }
    if supported < SUPPORTED_FLOOR {
        println!(
            "FAIL: supported count {supported} is below the stated floor {SUPPORTED_FLOOR} (the corpus, the demangler, or both regressed)"
        );
        return Ok(1);
    }
    if !disagreements.is_empty() {
        println!("FAIL: {} disagreements with llvm-nm", disagreements.len());
        return Ok(1);
    }
    println!(
        "OK: {agreed}/{supported} supported symbols demangle identically to llvm-nm (canonical: no access/convention/void/whitespace decoration)"
    );
    Ok(0)
}
