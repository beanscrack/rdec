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

// ---------------------------------------------------------------------------
// M8 step 1: `rdec cpp --corpus` — the first real C++ measurement.
// ---------------------------------------------------------------------------
//
// This command is a MEASUREMENT with a floor, not a green light to be earned
// by emitting less C. It runs the existing pipeline (`m4::lift_all_named`: the
// same discovery, the same lifter, the same passes, the same emitter) over
// `corpus/cpp/obj/*.obj`, names every emitted function from its DEMANGLED
// text, and prints what fraction of C++ codegen the decompiler covers:
//
//  * the `SymKind` census — the honest "what is C++ codegen" count;
//  * `compared | lifted | valid IR | emitted C | refused`;
//  * refusals by class, plus the C++ gap classes named from the IR (indirect
//    calls, vcall thunks, vtable/static addresses, `__security_cookie`,
//    by-value struct args, templates), each with its exact reason string;
//  * the member functions whose `this` in RCX is modelled as the first
//    parameter, and the ctors/dtors identified by kind.
//
// It FAILS on: zero symbols compared, clang rejecting the generated TU, a pass
// breaking SSA, an emitted name that does not round-trip into a legal C
// identifier, a duplicated C name (one TU, one definition), and any stated
// floor being missed. A low emitted-C fraction is the point of the step.

use rdec_core::emit_c::PRELUDE;
use rdec_core::ir::{Callee, Function, Op};
use rdec_core::verify::verify_ssa;
use rdec_demangle::SymKind;

/// Stated floors. Each is measured from the current corpus with headroom, so a
/// corpus that shrinks, a discovery step that stops seeding, or an emitter that
/// starts refusing a family it used to handle all fail loudly — and so the gate
/// never passes by comparing nothing.
pub const CPP_SYMBOL_FLOOR: usize = 60;
pub const CPP_LIFT_FLOOR: usize = 40;
pub const CPP_EMIT_FLOOR: usize = 25;
pub const CPP_THIS_FLOOR: usize = 12;
pub const CPP_CDTOR_FLOOR: usize = 2;

/// The census bucket for a demangled kind. The member-function split is the
/// point: plain, virtual and thunk are three different amounts of work.
/// `pub(crate)` because `rdec t2 --cpp` names its exclusions with the same
/// buckets: two naming schemes would make the two gates' lines incomparable.
pub(crate) fn label(k: &SymKind) -> &'static str {
    match k {
        SymKind::FreeFunction => "FreeFunction",
        SymKind::StaticMember { .. } => "StaticMember",
        SymKind::Constructor { .. } => "Constructor",
        SymKind::Destructor { .. } => "Destructor",
        SymKind::Vtable { .. } => "Vtable",
        SymKind::MemberFunction {
            thunk, virtual_fn, ..
        } => match (thunk, virtual_fn) {
            (true, _) => "MemberFunction(thunk)",
            (false, true) => "MemberFunction(virtual)",
            (false, false) => "MemberFunction",
        },
    }
}

/// One line saying what a kind MEANS for the calling convention, printed above
/// every emitted function so a reader can check `this` was accounted for.
fn kind_note(k: &SymKind) -> String {
    match k {
        SymKind::FreeFunction => "free function".to_string(),
        SymKind::StaticMember { class } => format!("static member of {class} (no `this`)"),
        SymKind::Constructor { class } => format!("constructor of {class} (`this` in RCX)"),
        SymKind::Destructor { class } => format!("destructor of {class} (`this` in RCX)"),
        SymKind::Vtable { class, table } => format!("{table:?} for {class} (data, not code)"),
        SymKind::MemberFunction {
            class,
            constness,
            virtual_fn,
            thunk,
        } => {
            let v = if *thunk {
                "vcall/displacement thunk"
            } else if *virtual_fn {
                "virtual member"
            } else {
                "member"
            };
            format!("{v} function of {class} (this-adjustment {:?})", constness)
        }
    }
}

/// True when the demangled signature says the function receives a pointer:
/// every member function (`this`), plus any parameter spelled with `*` or `&`.
/// `rdec t2` runs integer vectors only, so these are the functions its harness
/// excludes — the count is reported, never padded to look like a win.
fn takes_pointer(d: Option<&Demangled>) -> bool {
    let Some(d) = d else { return false };
    let Demangled::Name { text, kind } = d else {
        return false;
    };
    if matches!(
        kind,
        SymKind::MemberFunction { .. } | SymKind::Constructor { .. } | SymKind::Destructor { .. }
    ) {
        return true;
    }
    text.split_once('(')
        .map(|(_, rest)| rest.contains('*') || rest.contains('&'))
        .unwrap_or(false)
}

/// A C17 keyword cannot be a function name; a decompiler that emits `switch`
/// has written a syntax error, so the naming scheme has to know the list.
const C_KEYWORDS: &[&str] = &[
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "_Complex",
    "_Imaginary",
];

/// Collapse decoration into a legal C identifier fragment: every run of
/// non-`[A-Za-z0-9]` (`::`, `*`, `, `, `&`, `` ` ``) becomes exactly one `_`,
/// and the ends are trimmed.
fn ident_fragment(s: &str) -> String {
    let mut out = String::new();
    let mut pending = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            if pending {
                out.push('_');
                pending = false;
            }
            out.push(c);
        } else {
            pending = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// The demangled text -> C identifier map, in one place so the gate can prove
/// it round-trips. `int Point::sum()` becomes `Point_sum`, an overload
/// `void Point::shift(int, int)` becomes `Point_shift_int_int`, and the
/// const-qualified half of a const overload pair gets a `_const` tail so the
/// pair cannot collide in one translation unit.
///
/// The result is always a legal C identifier: the caller checks it against
/// `crate::m4::c_ident` (the identity the C gates call emitted C by), so this function
/// and that one cannot disagree about what a name is.
pub fn cpp_ident(text: &str) -> String {
    let (head, rest) = match text.find('(') {
        Some(i) => (&text[..i], &text[i..]),
        None => (text, ""),
    };
    // The qualified name is the last whitespace-separated token before the
    // parameter list: `int Point::sum` -> `Point::sum`, `Circle::Circle` ->
    // `Circle::Circle`. The return type is deliberately NOT in the name; it is
    // in the emitted signature, where C can spell it.
    let qual = head.split_whitespace().last().unwrap_or(head.trim());
    let mut parts: Vec<String> = Vec::new();
    let q = ident_fragment(qual);
    if !q.is_empty() {
        parts.push(q);
    }
    let params = rest
        .strip_prefix('(')
        .and_then(|r| r.split(')').next())
        .unwrap_or("")
        .trim();
    if !params.is_empty() && params != "void" {
        for a in params.split(',') {
            let frag = ident_fragment(a.trim());
            if !frag.is_empty() {
                parts.push(frag);
            }
        }
    }
    let trimmed = text.trim_end();
    if trimmed.ends_with(" const volatile") || trimmed.ends_with(" volatile const") {
        parts.push("const_volatile".to_string());
    } else if trimmed.ends_with(" const") {
        parts.push("const".to_string());
    } else if trimmed.ends_with(" volatile") {
        parts.push("volatile".to_string());
    }
    let mut name = parts.join("_");
    if name.is_empty() {
        name = "fn".to_string();
    }
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    if C_KEYWORDS.contains(&name.as_str()) {
        name = format!("fn_{name}");
    }
    name
}

/// Create the output directory, tolerating this host's filter driver: a fresh
/// `create_dir_all` returns `Access is denied (os error 5)` here AFTER the
/// directory has been created (the real-time scan holds the new directory for
/// the instant `create_dir_all` re-checks it), so a plain `?` refuses a
/// directory the same process just made. The directory is verified to exist;
/// a genuine failure is still an error.
fn ensure_dir(dir: &Path) -> Result<(), String> {
    for _ in 0..8 {
        if dir.is_dir() {
            return Ok(());
        }
        let _ = std::fs::create_dir_all(dir);
        if dir.is_dir() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    if dir.is_dir() {
        Ok(())
    } else {
        Err(format!(
            "{}: cannot create the output directory",
            dir.display()
        ))
    }
}

/// The opaque-reason class the report tallies, named from the lifter's own
/// text. The reason string is the work item, so it is not paraphrased.
fn opaque_class(why: &str) -> String {
    let w = why.trim();
    if w.starts_with("clobbered by call") {
        return "opaque: clobbered by call (caller-saved registers)".to_string();
    }
    if w.starts_with("indirect call") {
        return "indirect call: no signature model (virtual dispatch, thunk)".to_string();
    }
    if w.starts_with("indirect jmp") {
        return "indirect jmp: jump table not recovered".to_string();
    }
    if w.starts_with("lea of an address we cannot name") {
        return "lea of an unnamed address (vtable, static data, RTTI)".to_string();
    }
    if w.starts_with("live-in ") {
        return "opaque: live-in register (the ABI question)".to_string();
    }
    if let Some(rest) = w.strip_prefix("unmodelled instruction: ") {
        return format!("unmodelled instruction: {rest}");
    }
    let cut = w.find(" at ").unwrap_or(w.len());
    format!("opaque: {}", w[..cut.min(46)].trim())
}

/// Every gap class visible in one lifted function, counted from the IR itself -
/// not from a guess about what C++ "usually" contains.
fn scan_gaps(f: &Function, gap: &mut BTreeMap<String, usize>) {
    for b in &f.blocks {
        for i in &b.instrs {
            match &i.op {
                Op::Opaque(why) => {
                    *gap.entry(opaque_class(why)).or_default() += 1;
                }
                Op::Call { callee, .. } => match callee {
                    Callee::Indirect(_) => {
                        *gap.entry(
                            "indirect call: no signature model (virtual dispatch, thunk)"
                                .to_string(),
                        )
                        .or_default() += 1;
                    }
                    Callee::Symbol(s) => {
                        if s.contains("security_cookie") || s.contains("security_check_cookie") {
                            *gap.entry(
                                "call to the MSVC /GS __security_cookie check".to_string(),
                            )
                            .or_default() += 1;
                        }
                        if s.starts_with('?') {
                            *gap.entry(
                                "call to a decorated C++ symbol (extern under its sanitized name)"
                                    .to_string(),
                            )
                            .or_default() += 1;
                        }
                    }
                    Callee::Addr(_) => {}
                },
                _ => {}
            }
        }
    }
}

/// `int byval(P2)` - a by-value class parameter. The decompiler sees a
/// 64-bit register, the C side needs a struct: a type-model gap, named here
/// from the demangled signature because the machine code has no shape for it.
///
/// The demangler spells a by-value class `struct P2`, an enum `enum Kind`, and
/// a reference `int&`, so the test is "a `struct`/`class`/`union` token that is
/// not behind a `*` or `&`" - not "not a builtin", which counted enums.
fn byval_struct(d: Option<&Demangled>) -> bool {
    let Some(Demangled::Name { text, .. }) = d else {
        return false;
    };
    let Some((_, rest)) = text.split_once('(') else {
        return false;
    };
    let params = rest.split(')').next().unwrap_or("").trim().to_string();
    params.split(',').any(|p| {
        let p = p.trim();
        (p.starts_with("struct ") || p.starts_with("class ") || p.starts_with("union "))
            && !p.contains('*')
            && !p.contains('&')
    })
}

/// The C++ census, the demangled names, and the lifted subjects, in one
/// value: `rdec cpp --corpus` and `rdec t2 --cpp` run THIS, so the emission
/// gate and the value gate measure the same subjects under the same names.
/// A second copy of discovery + naming in the value gate would be measuring
/// its own copy, the exact thing `lift_all_named` exists to prevent.
pub struct CppPipeline {
    pub census: BTreeMap<&'static str, usize>,
    pub unsupported: BTreeMap<String, usize>,
    /// (object, raw COFF symbol) in manifest order, one entry per object.
    pub symbols: Vec<(String, String)>,
    pub dem_of: BTreeMap<String, Demangled>,
    /// raw symbol -> the C identifier the emitted C is defined under.
    pub c_name_of: BTreeMap<String, String>,
    pub unique_names: usize,
    pub named_from_decoration: usize,
    pub collisions: Vec<(String, String, String)>,
    pub data_tables: usize,
    pub eh_sections: usize,
    pub eh_symbols: usize,
    pub subjects: Vec<crate::m4::Subject>,
}

pub fn cpp_pipeline(files: &[String]) -> Result<CppPipeline, String> {
    // ---- 1. the census: what C++ codegen is the decompiler facing? ---------
    let mut census: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut symbols: Vec<(String, String)> = Vec::new(); // (obj, raw) in manifest order
    let mut dem_of: BTreeMap<String, Demangled> = BTreeMap::new();
    let mut data_tables = 0usize;
    let mut eh_sections = 0usize;
    let mut eh_symbols = 0usize;
    for file in files {
        let obj = rdec_arch_x86::loader::load(Path::new(file))?;
        if !obj.amd64() {
            return Err(format!(
                "{file}: machine {:#06x} is not AMD64 (0x8664)",
                obj.machine
            ));
        }
        // Vftables/vbtables are the demangler's `Vtable` kind and the C
        // pipeline has nothing to lift: they are data, in a non-executable
        // section. Counted here so the census says they exist.
        data_tables += obj
            .symbols
            .iter()
            .filter(|s| s.name.starts_with("??_7") || s.name.starts_with("??_8"))
            .count();
        // Exception handling, measured from the object and not from a guess: the
        // xdata/pdata run sections, and the runtime symbols an EH codelet calls.
        eh_sections += obj
            .sections
            .iter()
            .filter(|s| s.name.starts_with(".xdata") || s.name.starts_with(".pdata"))
            .count();
        eh_symbols += obj
            .symbols
            .iter()
            .filter(|s| {
                s.name.contains("CxxThrow")
                    || s.name.contains("UnregisterException")
                    || s.name.contains("__C_specific_handler")
            })
            .count();
        for sym in defined_function_symbols(&obj) {
            let d = demangle(&sym);
            match &d {
                Demangled::Name { kind, .. } => {
                    *census.entry(label(kind)).or_default() += 1;
                }
                Demangled::Unsupported { reason, .. } => {
                    *census.entry("Unsupported").or_default() += 1;
                    *unsupported.entry(reason.clone()).or_default() += 1;
                }
            }
            dem_of.entry(sym.clone()).or_insert_with(|| d.clone());
            symbols.push((file.clone(), sym));
        }
    }

    // ---- 2. names: the demangled text is what a human is shown --------------
    // One TU holds every emitted function, so a name has to be unique across
    // the corpus. Overloads share a demangled qualified name, so the parameter
    // list is folded in and a genuine collision gets a counted numeric tail.
    let mut c_name_of: BTreeMap<String, String> = BTreeMap::new();
    let mut taken: BTreeMap<String, String> = BTreeMap::new();
    // How many names could NOT come from demangled text - the demangler's own
    // gaps, named by sanitized decoration instead. A gap that is counted is a
    // gap someone can go and fix.
    let mut named_from_decoration = 0usize;
    let mut collisions: Vec<(String, String, String)> = Vec::new();
    for (_, raw) in &symbols {
        let base = match dem_of.get(raw) {
            Some(Demangled::Name { text, .. }) => cpp_ident(text),
            _ => crate::m4::c_ident(raw),
        };
        if !matches!(dem_of.get(raw), Some(Demangled::Name { .. })) {
            named_from_decoration += 1;
        }
        let mut name = base.clone();
        let mut n = 2usize;
        while let Some(prev) = taken.get(&name) {
            if prev == raw {
                break; // the same symbol in two objects: one name, one meaning
            }
            name = format!("{base}_{n}");
            n += 1;
        }
        if name != base {
            collisions.push((raw.clone(), base, name.clone()));
        }
        taken.insert(name.clone(), raw.clone());
        c_name_of.insert(raw.clone(), name);
    }

    // ---- 3. the existing pipeline, with the demangled names ----------------
    let subjects = crate::m4::lift_all_named(files, &|oi: usize, raw: &str| {
        c_name_of
            .get(raw)
            .cloned()
            .unwrap_or_else(|| format!("o{oi}_{}", crate::m4::c_ident(raw)))
    })?;

    let unique_names = c_name_of.len();
    Ok(CppPipeline {
        census,
        unsupported,
        symbols,
        dem_of,
        c_name_of,
        unique_names,
        named_from_decoration,
        collisions,
        data_tables,
        eh_sections,
        eh_symbols,
        subjects,
    })
}

/// The manifest of `corpus/cpp`, the same document both C++ gates read.
pub fn cpp_corpus_files() -> Result<Vec<String>, String> {
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
    Ok(files)
}

pub fn cmd_cpp(args: &[String]) -> Result<i32, String> {
    if !args[1..].iter().any(|a| a == "--corpus") {
        return Err(
            "usage: rdec cpp --corpus [--emit DIR]   (M8: the C++ corpus through the C pipeline)"
                .to_string(),
        );
    }
    let out_dir = args
        .iter()
        .position(|a| a == "--emit")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "corpus/cpp/gen".to_string());
    let files = cpp_corpus_files()?;
    ensure_dir(Path::new(&out_dir))?;

    let p = cpp_pipeline(&files)?;
    let compared = p.symbols.len();
    let census = p.census;
    let unsupported = p.unsupported;
    let symbols = p.symbols;
    let dem_of = p.dem_of;
    let c_name_of = p.c_name_of;
    let named_from_decoration = p.named_from_decoration;
    let collisions = p.collisions;
    let data_tables = p.data_tables;
    let eh_sections = p.eh_sections;
    let eh_symbols = p.eh_symbols;
    let subjects = p.subjects;
    let unique_names = p.unique_names;
    // The name round-trip line counts distinct C names.
    let mut names: Vec<&String> = c_name_of.values().collect();
    names.sort();

    let mut lifted = 0usize;
    let mut pass_broken = 0usize;
    let mut emitted = 0usize;
    let mut refused = 0usize;
    let mut keys: BTreeMap<String, usize> = BTreeMap::new();
    let mut gap: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_refusal: Option<String> = None;
    let mut first_pass_break: Option<String> = None;
    // raw symbol -> (C name, verdict). `synthetic` collects the lifted
    // functions no C++ symbol names, so they are reported, not hidden.
    let mut verdicts: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut synthetic: Vec<(String, String, String)> = Vec::new();
    let mut this_lifted = 0usize;
    let mut this_emitted = 0usize;
    let mut ctors: Vec<String> = Vec::new();
    let mut dtors: Vec<String> = Vec::new();
    let mut byval_n = 0usize;
    // T2: the SAME criteria `rdec t2` applies, applied to C++ functions.
    let mut t2_ok = 0usize;
    let mut t2_ok_ptr = 0usize;
    let mut t2_names: Vec<String> = Vec::new();
    let mut t2_skip: BTreeMap<String, usize> = BTreeMap::new();
    let mut t2_ptr = 0usize;

    let mut all = String::new();
    all.push_str("#include <stdint.h>\n#include <string.h>\n\n");
    all.push_str(PRELUDE);
    all.push('\n');

    for s in &subjects {
        lifted += 1;
        scan_gaps(&s.f, &mut gap);
        let dem = dem_of.get(&s.sym);
        let kind_line = match dem {
            Some(Demangled::Name { kind, .. }) => kind_note(kind),
            Some(Demangled::Unsupported { reason, .. }) => format!("unsupported: {reason}"),
            None => "no C++ decoration (plain COFF symbol)".to_string(),
        };
        if byval_struct(dem) {
            byval_n += 1;
            *gap.entry("by-value struct argument (needs the type model)".to_string())
                .or_default() += 1;
        }
        let member = matches!(
            dem,
            Some(Demangled::Name {
                kind: SymKind::MemberFunction { .. }
                    | SymKind::Constructor { .. }
                    | SymKind::Destructor { .. },
                ..
            })
        );
        // `this` is modelled: ARG_REGS is the Windows x64 integer convention,
        // so RCX is parameter 0 of every lifted function, and a member
        // function's first IR parameter is the class pointer.
        let this_modelled = member && !s.f.params.is_empty();
        if this_modelled {
            this_lifted += 1;
        }
        match dem {
            Some(Demangled::Name {
                kind: SymKind::Constructor { class },
                ..
            }) => ctors.push(format!("{} [{class}]", s.sym)),
            Some(Demangled::Name {
                kind: SymKind::Destructor { class },
                ..
            }) => dtors.push(format!("{} [{class}]", s.sym)),
            _ => {}
        }
        if let Err(errs) = verify_ssa(&s.f) {
            pass_broken += 1;
            if first_pass_break.is_none() {
                first_pass_break = Some(format!("{} :: {}: {}", s.obj, s.sym, errs.join("; ")));
            }
            verdicts.insert(
                s.sym.clone(),
                (s.cname.clone(), "FAIL passes broke SSA".to_string()),
            );
            continue;
        }
        match rdec_core::emit_c::emit_c(&s.f) {
            Ok(out) => {
                emitted += 1;
                if this_modelled {
                    this_emitted += 1;
                }
                let text = match dem {
                    Some(Demangled::Name { text, .. }) => text.clone(),
                    _ => s.sym.clone(),
                };
                // The raw decorated name stays in the file: the demangled name
                // is for the human, the decoration is the ground truth.
                let this_note = if !member {
                    "n/a (not a member function)".to_string()
                } else if s.f.params.is_empty() {
                    "member function with NO parameter - `this` was NOT modelled".to_string()
                } else {
                    "`this` in RCX = parameter a0 (a W64 value; its pointer TYPE is the type-model gap)"
                        .to_string()
                };
                all.push_str(&format!(
                    "/* rdec cpp: emitted from C++ machine code\n   COFF symbol : {}\n   demangled   : {}\n   C name      : {}\n   kind        : {}\n   this        : {}\n*/\n",
                    s.sym, text, s.cname, kind_line, this_note
                ));
                all.push_str(&out.text);
                all.push('\n');
                verdicts.insert(s.sym.clone(), (s.cname.clone(), "emit".to_string()));
                // T2 runnability, with `rdec t2`'s own predicates, in its own
                // order. Nothing here is weaker than the gate.
                let skip: Option<String> = if !out.externs.is_empty() {
                    Some("calls a symbol we do not model".to_string())
                } else if crate::t2::has_exit(&s.f) {
                    Some("unmodelled control flow (Exit)".to_string())
                } else if crate::t2::synthetic(&s.sym, s.start) {
                    Some("no COFF symbol to link (discovery target)".to_string())
                } else if !s.external {
                    Some("symbol is file-static (not linkable from a driver)".to_string())
                } else if crate::t2::touches_memory_outside_the_frame(&s.f) {
                    Some(
                        "dereferences a pointer argument: no address the two sides share"
                            .to_string(),
                    )
                } else if crate::t2::returns_undef(&s.f) {
                    Some("returns RAX unwritten on some path".to_string())
                } else {
                    None
                };
                match skip {
                    None => {
                        t2_ok += 1;
                        if takes_pointer(dem) {
                            t2_ok_ptr += 1;
                        }
                        t2_names.push(format!(
                            "{}{}",
                            s.cname,
                            if takes_pointer(dem) {
                                " (takes a pointer)"
                            } else {
                                ""
                            }
                        ));
                    }
                    Some(w) => {
                        *t2_skip.entry(w).or_default() += 1;
                        if takes_pointer(dem) {
                            t2_ptr += 1;
                        }
                    }
                }
            }
            Err(rs) => {
                refused += 1;
                for r in &rs {
                    *keys.entry(crate::m4::key(r)).or_default() += 1;
                }
                let first = rs
                    .first()
                    .map(|r| format!("{} @ {:#x}", r.reason, r.at))
                    .unwrap_or_else(|| "no reason".to_string());
                if first_refusal.is_none() {
                    first_refusal = Some(format!("{} :: {}: {}", s.obj, s.sym, first));
                }
                verdicts.insert(
                    s.sym.clone(),
                    (s.cname.clone(), format!("refuse: [{first}]")),
                );
            }
        }
    }
    // Every lifted function the census has no symbol for: discovery reached
    // code no COFF symbol names. Listed, not folded into the totals.
    for s in &subjects {
        if !c_name_of.contains_key(&s.sym) {
            let verdict = verdicts
                .get(&s.sym)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| "no verdict".to_string());
            synthetic.push((s.sym.clone(), s.cname.clone(), verdict));
        }
    }

    // ---- 4. the translation unit, compiled the way `rdec c` compiles -------
    let path = format!("{out_dir}/all_cpp.c");
    std::fs::write(&path, &all).map_err(|e| format!("{path}: {e}"))?;
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let obj_out = format!("{out_dir}/all_cpp.obj");
    let compiled = std::process::Command::new(&cc)
        .arg("-std=c17")
        .arg("-c")
        .arg("-O0")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wno-unused-parameter")
        .arg("-Werror")
        .arg(&path)
        .arg("-o")
        .arg(&obj_out)
        .output();

    // ---- 5. the report ------------------------------------------------------
    println!(
        "cpp: {} objects (corpus/cpp, clang++ -std=c++17 -O0 -target x86_64-pc-windows-msvc)",
        files.len()
    );
    println!("census: {compared} defined MSVC-decorated function symbols");
    let mut order: Vec<(&'static str, usize)> = census.iter().map(|(k, v)| (*k, *v)).collect();
    // Census order is the roadmap's: what a C decompiler already does, then
    // what C++ adds.
    order.sort_by_key(|(k, _)| {
        (
            match *k {
                "FreeFunction" => 0,
                "StaticMember" => 1,
                "MemberFunction" => 2,
                "MemberFunction(virtual)" => 3,
                "MemberFunction(thunk)" => 4,
                "Constructor" => 5,
                "Destructor" => 6,
                "Vtable" => 7,
                _ => 8,
            },
            *k,
        )
    });
    println!(
        "kind census: {}",
        order
            .iter()
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    // Measured, not assumed: this clang build leaves the vftables unnamed (a
    // `.rdata` section symbol plus REL32 relocations), so the census of vtable
    // SYMBOLS is 0 and the visible evidence of a vtable is the load site.
    println!(
        "vftable/vbtable symbols named in these objects: {data_tables}; vtable/static/RTTI address sites seen in the IR: {} (lea of an unnamed address), {} (mov with an unmodelled source), {} (mov to an address we cannot name)",
        gap
            .get("lea of an unnamed address (vtable, static data, RTTI)")
            .cloned()
            .unwrap_or(0),
        gap
            .get("opaque: mov with an unmodelled source")
            .cloned()
            .unwrap_or(0),
        gap
            .get("opaque: mov to an address we cannot name (rip-re")
            .cloned()
            .unwrap_or(0)
    );
    if unsupported.is_empty() {
        println!("unsupported by the demangler: none");
    } else {
        println!(
            "unsupported by the demangler: {}",
            unsupported
                .iter()
                .map(|(r, n)| format!("{n} x {r}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    println!("per function (verdict | C name | COFF symbol | demangled / reason):");
    for (_obj, raw) in &symbols {
        let d = dem_of.get(raw);
        let text = match d {
            Some(Demangled::Name { text, .. }) => text.clone(),
            Some(Demangled::Unsupported { reason, .. }) => format!("[demangler] {reason}"),
            None => String::new(),
        };
        let (cname, verdict) = verdicts.get(raw).cloned().unwrap_or_else(|| {
            (
                "-".to_string(),
                "not lifted: discovery never reached it".into(),
            )
        });
        println!(
            "  {:<11} {:<30} {:<34} {}  {}",
            verdict.split(':').next().unwrap_or(&verdict),
            cname,
            raw,
            text,
            if verdict.contains(':') {
                verdict.split_once(": ").map(|(_, r)| r).unwrap_or("")
            } else {
                ""
            }
        );
    }
    for (sym, cname, verdict) in &synthetic {
        println!(
            "  {:<11} {cname:<30} {sym:<34} {}  (no COFF symbol: a discovery target, not a link name)",
            verdict.split(':').next().unwrap_or(verdict),
            if verdict.contains(':') {
                verdict.split_once(": ").map(|(_, r)| r).unwrap_or("")
            } else {
                ""
            }
        );
    }
    if !synthetic.is_empty() {
        println!(
            "lifted functions with no COFF symbol: {} (the {compared} census symbols plus these = the {lifted} lifted)",
            synthetic.len()
        );
    }
    println!("compared: {compared} | lifted: {lifted} | valid IR: {}/{lifted} | emitted C: {emitted} | refused: {refused}", lifted - pass_broken);
    println!(
        "this-in-RCX: {this_lifted} member/ctor/dtor functions lifted with the class pointer modelled as parameter 0; {this_emitted} of them emitted C"
    );
    // The SSE census, counted by the lifter over the same subjects, so the
    // bracketed per-mnemonic numbers can be checked against
    // `llvm-objdump -d corpus/cpp/obj/*.obj` the way the self-idiom count can.
    let mut sse: BTreeMap<String, u32> = BTreeMap::new();
    let mut sse_zeroing = 0usize;
    let mut ret_xmm0 = 0usize;
    for s in &subjects {
        for (k, v) in &s.rep.sse {
            *sse.entry(k.clone()).or_insert(0) += *v;
        }
        sse_zeroing += s.rep.sse_zeroing;
        ret_xmm0 += s.rep.ret_xmm0;
    }
    println!("{}", crate::m4::sse_line(&sse, sse_zeroing));
    println!(
        "ret-float-in-XMM0: {ret_xmm0} returns reach a written XMM0 with RAX undefined: the Windows x64 ABI answers a float/double function in XMM0, and the emitted C returns the integer RAX. Their VALUE is therefore not in T2's compared set (T2 compares RAX, and its rule is unchanged)."
    );
    println!(
        "ctors identified: {} {} | dtors identified: {} {}",
        ctors.len(),
        if ctors.is_empty() {
            "(-)".to_string()
        } else {
            ctors.join(", ")
        },
        dtors.len(),
        if dtors.is_empty() {
            "(-)".to_string()
        } else {
            dtors.join(", ")
        }
    );
    let mut ks: Vec<(String, usize)> = keys.into_iter().collect();
    ks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if ks.is_empty() {
        println!("refusal classes: none");
    } else {
        println!(
            "refusal classes: {}",
            ks.iter()
                .map(|(k, n)| format!("{n} {k}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    if let Some(r) = &first_refusal {
        println!("first refusal: {r}");
    }
    let mut gs: Vec<(&String, &usize)> = gap.iter().collect();
    gs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!(
        "gap classes in the lifted IR: {}",
        if gs.is_empty() {
            "none".to_string()
        } else {
            gs.iter()
                .map(|(k, n)| format!("{n} {k}"))
                .collect::<Vec<_>>()
                .join(" | ")
        }
    );
    let thunk_n = *census.get("MemberFunction(thunk)").unwrap_or(&0);
    println!(
        "named C++ gaps, counted: vcall/displacement thunk symbols {thunk_n} | vtable symbols {data_tables} | indirect call sites {} | by-value struct args {byval_n} | template symbols {} | EH sections {eh_sections}, EH runtime symbols {eh_symbols}, int3 sites {} | __security_cookie calls {}",
        gap
            .get("indirect call: no signature model (virtual dispatch, thunk)")
            .cloned()
            .unwrap_or(0),
        unsupported
            .iter()
            .filter(|(r, _)| r.contains("template"))
            .map(|(_, n)| n)
            .sum::<usize>(),
        gap.get("unmodelled instruction: int3")
            .cloned()
            .unwrap_or(0),
        gap.get("call to the MSVC /GS __security_cookie check")
            .cloned()
            .unwrap_or(0)
    );
    println!(
        "EH is measured, not assumed: .xdata/.pdata sections and the runtime symbols a codelet calls come from the objects, and the int3 padding that ends a codelet boundary shows up as an unmodelled instruction - a function that reaches one is refused with that instruction named, never skipped"
    );
    println!(
        "name round-trip: {unique_names} names for {compared} symbols - {} built from demangled text, {} from the sanitized decoration (the demangler's gaps, named out loud); {} legal C identifiers under `rdec c`'s own c_ident; {} collisions disambiguated",
        unique_names - named_from_decoration,
        named_from_decoration,
        names
            .iter()
            .filter(|n| crate::m4::c_ident(n) == ***n)
            .count(),
        collisions.len()
    );
    for (raw, base, name) in &collisions {
        println!("  collision: {raw}: {base} -> {name}");
    }
    println!(
        "t2 for C++: {t2_ok} of {lifted} lifted functions meet `rdec t2`'s existing criteria ({t2_ok_ptr} of those take a pointer, and they pass only because they return the pointer they were handed unchanged); nothing weakened - the harness's own predicates decide"
    );
    for (w, n) in &t2_skip {
        println!("  t2 skips: {n} x {w}");
    }
    for nm in t2_names.iter().take(12) {
        println!("  t2 candidate: {nm}");
    }
    if t2_names.len() > 12 {
        println!("  ... and {} more", t2_names.len() - 12);
    }
    println!(
        "  of the skipped functions, {t2_ptr} take a pointer argument (`this`, or a `*`/`&` parameter): the harness runs integer vectors only, and a `this` dereference reads at an address the two sides do not share"
    );
    let compiled = match compiled {
        Ok(o) => o,
        Err(e) => {
            println!("FAIL: could not run {cc} - is it on PATH? (set CC): {e}");
            return Ok(1);
        }
    };
    if !compiled.status.success() {
        let err = String::from_utf8_lossy(&compiled.stderr);
        let first = err
            .lines()
            .find(|l| l.contains("error"))
            .unwrap_or("<no error line>");
        println!("FAIL: {cc} -Werror rejected the generated C++ decompilation ({emitted} functions emitted)");
        println!("clang: {first}");
        return Ok(1);
    }
    println!("clang -Werror: OK ({path} -> {obj_out})");

    let mut fails: Vec<String> = Vec::new();
    if compared == 0 {
        fails.push("no C++ symbols compared - a gate that compares nothing proves nothing".into());
    }
    if pass_broken > 0 {
        fails.push(format!(
            "a pass broke SSA in {pass_broken} functions: {}",
            first_pass_break.unwrap_or_default()
        ));
    }
    let bad_names: Vec<&&String> = names
        .iter()
        .filter(|n| crate::m4::c_ident(n) != ***n)
        .collect();
    if !bad_names.is_empty() {
        fails.push(format!(
            "{} emitted names are not legal C identifiers: {:?}",
            bad_names.len(),
            &bad_names[..bad_names.len().min(5)]
        ));
    }
    if let Some(dup) = names.windows(2).find(|w| w[0] == w[1]) {
        fails.push(format!("duplicate C name in one TU: {}", dup[0]));
    }
    for (what, got, floor) in [
        ("symbols compared", compared, CPP_SYMBOL_FLOOR),
        ("lifted", lifted, CPP_LIFT_FLOOR),
        ("emitted C", emitted, CPP_EMIT_FLOOR),
        ("this-in-RCX lifted", this_lifted, CPP_THIS_FLOOR),
        (
            "ctors+dtors identified",
            ctors.len() + dtors.len(),
            CPP_CDTOR_FLOOR,
        ),
    ] {
        if got < floor {
            fails.push(format!("floor: {what} {got} < stated floor {floor}"));
        }
    }
    println!(
        "floor (stated in m8.rs, measured here): compared>={CPP_SYMBOL_FLOOR} lifted>={CPP_LIFT_FLOOR} emitted>={CPP_EMIT_FLOOR} this-in-RCX>={CPP_THIS_FLOOR} ctors+dtors>={CPP_CDTOR_FLOOR} clang -Werror accepts the TU, and every emitted name is a legal C identifier"
    );
    println!(
        "measured: compared={compared} lifted={lifted} valid IR={}/emitted C={emitted} refused={refused} this-in-RCX={this_lifted} ctors+dtors={}",
        lifted - pass_broken,
        ctors.len() + dtors.len()
    );
    for f in &fails {
        println!("FAIL: {f}");
    }
    if fails.is_empty() {
        println!(
            "OK: {compared} C++ symbols measured, {emitted} emitted C that clang accepts; the rest are named gaps, not silent losses"
        );
        Ok(0)
    } else {
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdec_arch_x86::discover::{discover, Code};
    use rdec_arch_x86::lift::lift_one;
    use rdec_core::emit_c::emit_c;
    use rdec_core::ir::Op;
    use std::collections::HashMap;

    /// Lift hand-assembled bytes the way the pipeline does: one code range, one
    /// entry, one function spanning the bytes. No clang, no corpus: the fixtures
    /// are bytes, so these tests run on a bare checkout.
    fn lift(name: &str, bytes: &[u8]) -> rdec_core::ir::Function {
        let plan = discover(Code { base: 0, bytes }, &[0], &[]);
        let (f, _rep) = lift_one(
            name,
            bytes,
            0,
            0,
            bytes.len() as u64,
            &plan,
            &HashMap::new(),
        );
        f
    }

    #[test]
    fn a_member_functions_this_in_rcx_is_the_first_parameter_and_the_body_dereferences_it() {
        // `int Point::sum() { return x + y; }` at -O0: mov eax,[rcx]; add
        // eax,[rcx+4]; ret. The demangled kind says the first parameter is a
        // `Point *`; the lifter's ARG_REGS convention has to agree with it.
        const BODY: &[u8] = &[0x8B, 0x01, 0x03, 0x41, 0x04, 0xC3];
        let d = demangle("?sum@Point@@QEAAHXZ");
        let Demangled::Name { text, kind } = &d else {
            panic!("the corpus' own member symbol must demangle: {d:?}");
        };
        assert_eq!(
            kind,
            &SymKind::MemberFunction {
                class: "Point".into(),
                constness: rdec_demangle::Constness::NonConst,
                virtual_fn: false,
                thunk: false,
            }
        );
        assert!(takes_pointer(Some(&d)), "a member function takes `this`");

        let name = cpp_ident(text);
        assert_eq!(name, "Point_sum");
        assert_eq!(
            crate::m4::c_ident(&name),
            name,
            "the demangled name round-trips into a legal C identifier"
        );

        // The gate's own passes, not a hand-picked subset: a surviving flags
        // value is a refusal, and the pass that removes it is part of the shape.
        let mut f = lift(&name, BODY);
        crate::m4::clean(&mut f);
        let f = f;
        assert!(
            !f.params.is_empty(),
            "a lifted function has the four argument registers as parameters"
        );
        let this = f.params[0];
        // `this` is modelled: RCX is parameter 0, and the loads read through it.
        let derefs_this = f
            .blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .any(|i| matches!(&i.op, Op::Load(_, a) if *a == this));
        assert!(
            derefs_this,
            "the body must read `x` through parameter 0, not through a mystery value"
        );
        let out = emit_c(&f).expect("this shape emits C");
        assert!(
            out.text.contains(&format!("RDEC_ADDR(x{this})")),
            "the emitted C dereferences the first parameter:\n{}",
            out.text
        );
        // The named gap, said out loud rather than papered over: a register
        // group is one W64 SSA value, so `this` reaches C as an integer address,
        // not as `Point *`. Modelling: yes. Pointer TYPE: the type-recovery gap.
        assert!(
            matches!(
                f.ty_of(this),
                Some(rdec_core::ty::Ty::Int {
                    w: rdec_core::ty::Width::W64,
                    ..
                })
            ),
            "the IR models `this` as a 64-bit register value: {:?}",
            f.ty_of(this)
        );
    }

    #[test]
    fn a_ctor_symbol_is_emitted_under_a_name_that_identifies_the_class() {
        let d = demangle("??0Circle@@QEAA@N@Z");
        let Demangled::Name { text, kind } = &d else {
            panic!("??0 is the constructor encoding: {d:?}");
        };
        assert_eq!(
            kind,
            &SymKind::Constructor {
                class: "Circle".into()
            }
        );
        let name = cpp_ident(text);
        assert_eq!(name, "Circle_Circle_double");
        assert!(name.starts_with("Circle"), "{name} names the class");
        assert_eq!(crate::m4::c_ident(&name), name);
        // The destructor of the same class is a different C name, in one TU.
        let dd = demangle("??1Circle@@QEAA@XZ").text().to_string();
        let dname = cpp_ident(&dd);
        assert_eq!(dname, "Circle_Circle");
        assert_ne!(name, dname, "ctor and dtor cannot share one definition");
        assert_eq!(
            label(&SymKind::Constructor { class: "C".into() }),
            "Constructor"
        );
        assert_eq!(
            label(&SymKind::Destructor { class: "C".into() }),
            "Destructor"
        );
    }

    #[test]
    fn a_virtual_dispatch_body_lands_in_a_named_refusal_and_not_in_the_emitted_set() {
        // The thunk/vcall shape: load the vftable slot from `this`, call
        // through it. `mov rax,[rcx]; call qword ptr [rax]; ret`.
        const BODY: &[u8] = &[0x48, 0x8B, 0x01, 0xFF, 0x10, 0xC3];
        let mut f = lift("Derived_get3_const", BODY);
        crate::m4::clean(&mut f);
        let err = match emit_c(&f) {
            Ok(out) => panic!(
                "a vcall body must NOT be emitted, it emitted:\n{}",
                out.text
            ),
            Err(rs) => rs,
        };
        let reasons: String = err
            .iter()
            .map(|r| format!("{};", r.reason))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            reasons.contains("indirect call"),
            "the refusal must name the indirect call, got: {reasons}"
        );
        assert!(
            opaque_class("indirect call").starts_with("indirect call: no signature model"),
            "the gap class is the work item: {}",
            opaque_class("indirect call")
        );
        // And the demangler's own thunk encoding is a census bucket of its own,
        // so a corpus that starts emitting thunks shows up as a number.
        let t = demangle("?get3@Derived@@UEAAXXZ@7");
        let Demangled::Name { kind, .. } = &t else {
            panic!("@7 is the vcall-adjusted thunk: {t:?}");
        };
        assert_eq!(label(kind), "MemberFunction(thunk)");
    }

    #[test]
    fn a_const_overload_pair_becomes_two_distinct_legal_c_names() {
        // `int &Buffer::slot(int)` and `const int &Buffer::slot(int) const`:
        // one TU, one definition each, and both derived from demangled text.
        let a = cpp_ident(demangle("?slot@Buffer@@QEAAAEAHH@Z").text());
        let b = cpp_ident(demangle("?slot@Buffer@@QEBAAEBHH@Z").text());
        assert_eq!(a, "Buffer_slot_int");
        assert_eq!(b, "Buffer_slot_int_const");
        assert_ne!(a, b);
        for n in [&a, &b] {
            assert_eq!(crate::m4::c_ident(n), *n, "{n} is not a legal C name");
            assert!(!C_KEYWORDS.contains(&n.as_str()), "{n} would be a keyword");
        }
    }
}
