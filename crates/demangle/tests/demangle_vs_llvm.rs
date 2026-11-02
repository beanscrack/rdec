//! The demangler vs llvm-nm, as a cargo test.
//!
//! Same comparison as the `rdec demangle --cpp` gate, over the C++ corpus.
//! llvm-nm is not guaranteed on every machine (CI installs it; laptops vary),
//! so a missing oracle prints SKIP **loudly** and the test passes — the gate
//! command is the enforcing artifact and exits non-zero when the tool is
//! absent. What may never happen silently is a disagreement.
//!
//! Symbol enumeration uses the repo's own COFF loader (dev-dependency on
//! rdec-arch-x86: this crate has no production dependencies, and the
//! dependency graph stays a DAG).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rdec_demangle::{canonical, demangle, Demangled};

fn nm_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RDEC_NM") {
        let p = PathBuf::from(p);
        return if p.exists() { Some(p) } else { None };
    }
    let probe = Command::new("llvm-nm").arg("--version").output();
    probe.is_ok().then(|| PathBuf::from("llvm-nm"))
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/demangle
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .expect("crate lives in the workspace")
}

fn nm_lines(nm: &Path, obj: &Path, dem: bool) -> Vec<(String, String, String)> {
    let mut cmd = Command::new(nm);
    cmd.arg("--defined-only").arg("-g");
    if dem {
        cmd.arg("--demangle");
    }
    cmd.arg(obj);
    let out = cmd.output().expect("llvm-nm runs");
    assert!(out.status.success(), "llvm-nm failed on {}", obj.display());
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.splitn(3, ' ');
            let (Some(a), Some(t), Some(n)) = (it.next(), it.next(), it.next()) else {
                return None;
            };
            a.bytes()
                .all(|b| b.is_ascii_hexdigit())
                .then(|| (a.to_string(), t.to_string(), n.to_string()))
        })
        .collect()
}

#[test]
fn corpus_cpp_names_agree_with_llvm_nm() {
    let Some(nm) = nm_path() else {
        eprintln!("SKIP: llvm-nm not found (set RDEC_NM) - the enforcing artifact is `rdec demangle --cpp`, which exits non-zero without it");
        return;
    };
    let root = repo_root();
    let obj_dir = root.join("corpus/cpp/obj");
    let mut objs: Vec<PathBuf> = match std::fs::read_dir(&obj_dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "obj").unwrap_or(false))
            .collect(),
        Err(e) => panic!("corpus/cpp/obj: {e} (run `rdec corpus --cpp`)"),
    };
    objs.sort();
    assert!(!objs.is_empty(), "no .obj files in corpus/cpp/obj");

    let mut compared = 0usize;
    let mut supported = 0usize;
    let mut unsupported = 0usize;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for obj in &objs {
        let object = rdec_arch_x86::loader::load(obj).expect("corpus object loads");
        assert!(object.amd64(), "{}: not AMD64", obj.display());
        let plain = nm_lines(&nm, obj, false);
        let dem = nm_lines(&nm, obj, true);
        assert_eq!(
            plain.len(),
            dem.len(),
            "{}: llvm-nm run pair",
            obj.display()
        );
        let oracle: BTreeMap<String, String> = plain
            .iter()
            .zip(dem.iter())
            .map(|((a, t, m), (a2, t2, d))| {
                assert_eq!((a, t), (a2, t2), "{}: llvm-nm reordered", obj.display());
                (m.clone(), d.clone())
            })
            .collect();
        for sym in object
            .symbols
            .iter()
            .filter(|s| {
                s.section.map(|x| object.is_code(x)).unwrap_or(false)
                    && !s.name.starts_with('.')
                    && (s.storage_class == rdec_arch_x86::loader::STORAGE_EXTERNAL
                        || s.storage_class == rdec_arch_x86::loader::STORAGE_STATIC)
                    && s.name.starts_with('?')
            })
            .map(|s| s.name.clone())
        {
            let ours = demangle(&sym);
            let text = match &ours {
                Demangled::Name { text, .. } => text,
                Demangled::Unsupported { reason, .. } => {
                    unsupported += 1;
                    *reasons.entry(reason.clone()).or_default() += 1;
                    continue;
                }
            };
            supported += 1;
            let Some(theirs) = oracle.get(&sym) else {
                continue; // no llvm-nm counterpart for this symbol
            };
            compared += 1;
            assert_eq!(
                canonical(text),
                canonical(theirs),
                "{}: {sym}\n  mine: {text}\n  llvm: {theirs}",
                obj.display()
            );
        }
    }
    println!(
        "demangle_vs_llvm: {compared} compared, {supported} supported, {unsupported} unsupported ({})",
        reasons
            .iter()
            .map(|(r, n)| format!("{n} x {r}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    assert!(compared > 0, "zero symbols compared - the corpus is empty");
}

/// The classification M8 depends on, re-checked against corpus *symbols*
/// (not fixtures): the corpus defines a ctor, a dtor, a const member, a
/// static, a namespace function, and a template gap.
#[test]
fn corpus_kinds_are_classified() {
    let root = repo_root();
    let obj = root.join("corpus/cpp/obj/class_ctor_dtor.O0.obj");
    let object = rdec_arch_x86::loader::load(&obj).expect("ctor/dtor object loads");
    let mut saw_ctor = false;
    let mut saw_dtor = false;
    let mut saw_const_member = false;
    for s in object.symbols.iter().filter(|s| s.name.starts_with('?')) {
        match demangle(&s.name) {
            Demangled::Name { kind, .. } => match kind {
                rdec_demangle::SymKind::Constructor { class } => {
                    assert_eq!(class, "Circle");
                    saw_ctor = true;
                }
                rdec_demangle::SymKind::Destructor { class } => {
                    assert_eq!(class, "Circle");
                    saw_dtor = true;
                }
                rdec_demangle::SymKind::MemberFunction {
                    class, constness, ..
                } => {
                    assert_eq!(class, "Circle");
                    if constness == rdec_demangle::Constness::Const {
                        saw_const_member = true;
                    }
                }
                other => panic!("{}: unexpected kind {other:?}", s.name),
            },
            Demangled::Unsupported { reason, .. } => panic!("{}: {reason}", s.name),
        }
    }
    assert!(
        saw_ctor && saw_dtor && saw_const_member,
        "ctor/dtor/const all present"
    );
}

#[test]
fn corpus_template_gap_is_loud() {
    let root = repo_root();
    let obj = root.join("corpus/cpp/obj/template_gap.O0.obj");
    let object = rdec_arch_x86::loader::load(&obj).expect("template object loads");
    let gap: Vec<String> = object
        .symbols
        .iter()
        .filter(|s| s.name.starts_with("??$"))
        .map(|s| match demangle(&s.name) {
            Demangled::Unsupported { reason, .. } => reason,
            Demangled::Name { text, .. } => panic!("template demangled to {text}: {}", s.name),
        })
        .collect();
    assert!(
        !gap.is_empty(),
        "template_gap.cpp must produce a ??$ symbol"
    );
    assert!(
        gap.iter().all(|r| r.starts_with("template argument list")),
        "{gap:?}"
    );
}
