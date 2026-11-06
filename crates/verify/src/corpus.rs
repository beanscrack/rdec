//! Test-corpus generator (roadmap M0).
//!
//! The corpus is the ground truth for every later gate: M2 op coverage, M4
//! differential equivalence, M5 structuring/switch recovery. It is generated,
//! never hand-written as assembly, because the only honest corpus for a
//! decompiler is *real compiler output* over known source.
//!
//! Layout under the output dir:
//!   src/<name>.c          the snippet (one exported function `f`)
//!   obj/<name>.<opt>.obj  compiled object
//!   manifest.tsv          name, opt, obj path, size, fnv1a-64 of object bytes
//!
//! The manifest is the contract: gates consume it, so a corpus change is
//! visible as a manifest diff instead of a silent behavior change.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One corpus entry: a C snippet whose observable behavior is fully defined by
/// its return value (and the memory it writes through pointer params).
pub struct Snippet {
    pub name: &'static str,
    /// What this snippet is meant to force the pipeline to handle.
    pub exercises: &'static str,
    pub code: &'static str,
}

/// Every snippet exports `f`, so the decompiler's entry selection is uniform.
pub const SNIPPETS: &[Snippet] = &[
    Snippet {
        name: "arith_basic",
        exercises: "add/sub/imul widths; the bread and butter of expression recovery",
        code: "int f(int a, int b) { int t = a * 3 + b; return (t ^ a) - (b >> 2); }",
    },
    Snippet {
        name: "shift_all",
        exercises: "shl/shr/sar over signed+unsigned, incl. counts >= width",
        code: "long f(long a, unsigned n) {\
              long x = a << (n & 63u);\
              long y = (unsigned long)a >> (n & 63u);\
              long z = a >> (n & 63u);\
              return x + y + z; }",
    },
    Snippet {
        name: "cmp_signed",
        exercises: "signed jcc: jl/jle/jg/jge flag fusion",
        code: "int f(int a, int b) { int r = 0; if (a < b) r += 1; if (a <= b) r += 2; if (a > b) r += 4; if (a >= b) r += 8; return r; }",
    },
    Snippet {
        name: "cmp_unsigned",
        exercises: "unsigned jcc: below/above-or-equal (the CF side)",
        code: "unsigned f(unsigned a, unsigned b) { unsigned r = 0; if (a < b) r += 1; if (a >= b) r += 2; if (a > b) r += 4; return r; }",
    },
    Snippet {
        name: "setcc",
        exercises: "setcc + movzx: boolean materialized into an integer",
        code: "int f(int a, int b) { return (a < b) + (a == b) * 2 + (a > b) * 4; }",
    },
    Snippet {
        name: "cmov",
        exercises: "branchless select: cmov must become a ternary, not an if",
        code: "int f(int a, int b) { int m = a > b ? a : b; int n = a < b ? a : b; return m * 2 - n; }",
    },
    Snippet {
        name: "loop_sum",
        exercises: "reducible while-loop, induction variable, loop-carried phi",
        code: "int f(int n) { int s = 0; for (int i = 0; i < n; i++) s += i * 3; return s; }",
    },
    Snippet {
        name: "loop_do",
        exercises: "do/while (bottom-tested) vs while (top-tested)",
        code: "int f(int n) { int s = 0; do { s ^= n; n >>= 1; } while (n != 0); return s; }",
    },
    Snippet {
        name: "loop_nested_break",
        exercises: "nested loops with early exit: multi-level control flow",
        code: "int f(int n) { int s = 0; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { if (j > i) break; s += j; } if (s > 1000) break; } return s; }",
    },
    Snippet {
        name: "switch_dense",
        exercises: "dense switch: jump-table recovery with base/index/bounds",
        code: "int f(int k) { switch (k) { case 0: return 7; case 1: return 11; case 2: return 13; case 3: return 17; case 4: return 19; case 5: return 23; case 6: return 29; case 7: return 31; default: return -1; } }",
    },
    Snippet {
        name: "switch_sparse",
        exercises: "sparse switch: binary-search chain, not a table",
        code: "int f(int k) { switch (k) { case 2: return 1; case 97: return 2; case 131: return 3; case 911: return 4; case 104729: return 5; default: return 0; } }",
    },
    Snippet {
        name: "switch_fallthrough",
        exercises: "fallthrough edges: CFG shape that structuring gets wrong",
        code: "int f(int k) { int r = 0; switch (k) { case 0: r += 1; case 1: r += 2; case 2: r += 4; break; case 3: r += 8; } return r; }",
    },
    Snippet {
        name: "divmod_signed",
        exercises: "idiv: rdx:rax pairing, sign correction, div-by-power-of-2 strength reduction",
        code: "int f(int a, int b) { return a / b + a % b; }",
    },
    Snippet {
        name: "div_unsigned",
        exercises: "div on unsigned: zero-extension of rdx",
        code: "unsigned f(unsigned a, unsigned b) { return a / b - a % b; }",
    },
    Snippet {
        name: "div_pow2",
        exercises: "signed division by constant: sar + add magic sequence",
        code: "int f(int a) { return a / 6 + a / 16 + a / 7; }",
    },
    Snippet {
        name: "mul_wide",
        exercises: "mul/imul high half: 128-bit product, rdx as data not flag",
        code: "long f(long a, long b) { return (long)(((__int128)a * (__int128)b) >> 64); }",
    },
    Snippet {
        name: "carry_chain",
        exercises: "add/adc carry threading through 128-bit add (CF as data)",
        code: "void f(unsigned long a, unsigned long b, unsigned long *lo, unsigned long *hi) { unsigned long l = a + b; unsigned long h = (unsigned long)(l < a); *lo = l; *hi = h; }",
    },
    Snippet {
        name: "logic_flags",
        exercises: "and/or/xor/test: ZF/SF set, CF/OF cleared, PF",
        code: "int f(int a, int b) { int r = 0; if ((a & b) == 0) r |= 1; if ((a | b) < 0) r |= 2; if ((a ^ b) == 0) r |= 4; return r; }",
    },
    Snippet {
        name: "narrow_ext",
        exercises: "movzx/movsx: signedness of sub-word loads",
        code: "int f(char a, unsigned char b, short c, unsigned short d) { return a + b + c + d; }",
    },
    Snippet {
        name: "narrow_store",
        exercises: "sub-register writes (al/ax): partial-flag merge semantics",
        code: "int f(int a) { char c = (char)a; short s = (short)(a >> 8); return (int)c * 3 + (int)s; }",
    },
    Snippet {
        name: "locals_addressed",
        exercises: "address-taken locals: must stay memory, never promoted to a register var",
        code: "int f(int n) { int a = n, b = n + 1; int *p = &a; for (int i = 0; i < n; i++) { *p += i; p = (p == &a) ? &b : &a; } return a + b; }",
    },
    Snippet {
        name: "array_walk",
        exercises: "pointer indexing, [base+index*scale+disp] addressing",
        code: "long f(const long *a, int n) { long s = 0; for (int i = 0; i < n; i++) s += a[i] * (i + 1); return s; }",
    },
    Snippet {
        name: "struct_access",
        exercises: "struct field offsets: the raw material for struct recovery",
        code: "struct S { int a; char b; long c; short d; }; int f(struct S *s) { s->a += 1; s->c = s->a * 2; return s->a + s->b + (int)s->c + s->d; }",
    },
    Snippet {
        name: "nested_if_tree",
        exercises: "deep if/else nesting, else-if chains",
        code: "int f(int a, int b, int c) { if (a > 0) { if (b > 0) { if (c > 0) return 1; else return 2; } else if (c > 0) return 3; else return 4; } else { if (b + c > 10) return 5; return 6; } }",
    },
    Snippet {
        name: "early_returns",
        exercises: "multiple exits: goto-reduction pressure",
        code: "int f(int a, int b) { if (a < 0) return -1; if (b == 0) return 0; if (a > 100 && b > 100) return 1; for (int i = 0; i < b; i++) { if (a == i) return i; } return 42; }",
    },
    Snippet {
        name: "gcd_loop",
        exercises: "loop with modulo, no induction variable (not a for)",
        code: "int f(int a, int b) { while (b != 0) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }",
    },
    Snippet {
        name: "fib_rec",
        exercises: "recursive call, stack frame, ABI argument shuffling",
        code: "long f(int n) { if (n < 2) return n; return f(n - 1) + f(n - 2); }",
    },
    Snippet {
        name: "call_helper",
        exercises: "non-inlined call: call/ret, Win64 arg registers, callee-saved preservation",
        code: "__attribute__((noinline)) static int h(int x) { return x * x + 1; } int f(int a, int b) { return h(a) + h(b) + h(a + b); }",
    },
    Snippet {
        name: "saturation",
        exercises: "saturating arithmetic: cmp/cmov/setcc idiom cluster",
        code: "int f(int a, int b) { long t = (long)a + (long)b; if (t > 2147483647L) return 2147483647; if (t < -2147483648L) return (-2147483647 - 1); return (int)t; }",
    },
    Snippet {
        name: "irreducible_shape",
        exercises: "computed-goto-ish shape: two-entry loop, goto fallback must not crash",
        code: "int f(int n) { int s = 0, i = 0; goto mid; top: s += i; i++; mid: if (i < n) { if (s < 500) goto top; else goto top; } return s; }",
    },
    // Caller-buffer fixtures: the ground truth is the buffer's bytes, which the
    // return value cannot see. `store_to_buffer` is the shape that proves a
    // memory-effect comparison bites: return the right `b[0]` while scribbling
    // `b[3]` anywhere but its own slot, and a return-value gate stays green on
    // a broken decompiler. All three take a single pointer parameter, so the
    // Win64 driver delivers the buffer in RCX (no stack-passed arguments).
    Snippet {
        name: "store_to_buffer",
        exercises: "writes through a caller buffer: b[3] is a scribble the return value cannot see",
        code: "long long f(long long *b) { b[0] = b[1] + b[2]; b[3] = b[0] * 2; return b[0]; }",
    },
    Snippet {
        name: "copy_words",
        exercises: "fixed-count store loop through a caller buffer: 16 words written, 16 returned",
        code: "long long f(long long *b) { int n = 16; for (int i = 0; i < n; i++) b[i] = b[i] + (long long)i; return n; }",
    },
    Snippet {
        name: "read_only_words",
        exercises: "read-mostly over a caller buffer: MUST NOT write; a spurious store is a miscompile",
        code: "long long f(const long long *b) { long long s = b[0] + b[1] * 2 + b[2] * 3 + b[3] * 4; return s; }",
    },
];

pub const OPTS: &[&str] = &["O0", "O1", "O2", "Oz"];

#[derive(Debug, Default)]
pub struct Report {
    pub built: usize,
    pub entries: Vec<Entry>,
    pub failures: Vec<String>,
}

#[derive(Debug)]
pub struct Entry {
    pub name: String,
    pub opt: String,
    pub obj: PathBuf,
    pub bytes: u64,
    pub hash: u64,
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn clang_path() -> String {
    std::env::var("RDEC_CLANG").unwrap_or_else(|_| "clang".to_string())
}

/// The compiler that produced the corpus, as `(version, target triple)`.
///
/// Object hashes are a function of the compiler, so a manifest that moves
/// between clang 20 and clang 22 is *expected*, not a regression: recording the
/// version turns "the hashes changed" into "different compiler", and a gate that
/// wants stability compares two runs on the same toolchain (see
/// `.github/workflows/ci.yml`). Asserting byte-identical codegen across compiler
/// versions tests clang, not RDec.
pub fn clang_stamp(clang: &str) -> (String, String) {
    let ver = Command::new(clang)
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());
    let tgt = Command::new(clang)
        .arg("-dumpmachine")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    (ver, tgt)
}

/// The `# clang:` line a manifest was written with, if present.
pub fn manifest_clang(path: &Path) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("# clang:"))
                .map(|l| l["# clang:".len()..].trim().to_string())
        })
        .unwrap_or_else(|| "unrecorded".into())
}

/// Zero the COFF `TimeDateStamp` (offset 4) of an AMD64 object.
///
/// COFF objects embed the build clock, so an unnormalized corpus is not a
/// fingerprint of anything: measured locally, two runs of the same clang over
/// the same sources produced 120/120 different hashes at *identical sizes* --
/// the signature of a fixed-width volatile field, not a codegen change. The
/// manifest is the contract the gates consume, so the stamp is normalized and
/// that fact is recorded in the manifest header. The check is guarded on the
/// Machine field, so this never rewrites something that is not an AMD64 object.
pub fn normalize_coff_stamp(bytes: &mut [u8]) -> bool {
    // A COFF *object* header starts at offset 0 with Machine (there is no image
    // signature field), and TimeDateStamp follows at offset 4. Checking offset 2
    // reads NumberOfSections and silently disables the normalization.
    if bytes.len() < 20 || u16::from_le_bytes([bytes[0], bytes[1]]) != 0x8664 {
        return false;
    }
    if bytes[4..8] == [0, 0, 0, 0] {
        return false; // already normalized; report "unchanged"
    }
    bytes[4..8].copy_from_slice(&[0, 0, 0, 0]);
    true
}

/// Compile every snippet at every requested `-O` level.
pub fn generate(out: &Path, opts: &[&str], clang: &str) -> Report {
    let src_dir = out.join("src");
    let obj_dir = out.join("obj");
    let _ = fs::create_dir_all(&src_dir);
    let _ = fs::create_dir_all(&obj_dir);

    let mut rep = Report::default();
    // (name, opt) -> (path, size, hash), sorted for a deterministic manifest.
    let mut manifest: BTreeMap<(String, String), (PathBuf, u64, u64)> = BTreeMap::new();

    for sn in SNIPPETS {
        let src = src_dir.join(format!("{}.c", sn.name));
        // LF endings: clang is fine with either, and a stable source file keeps
        // the object hash stable across checkouts on Windows.
        let body = format!("/* rdec corpus: {} */\n{}\n", sn.exercises, sn.code);
        if let Err(e) = fs::write(&src, body.replace("\r\n", "\n")) {
            rep.failures.push(format!("{}: write source: {e}", sn.name));
            continue;
        }
        for &opt in opts {
            let obj = obj_dir.join(format!("{}.{}.obj", sn.name, opt));
            let status = Command::new(clang)
                .arg(format!("-{opt}"))
                .arg("-std=c17")
                // No CodeView debug sections: they embed the source path, which
                // is a reproducibility hazard and not machine code.
                // (Reproducible *stamps* are normalized afterwards; clang cannot
                // be told from this driver -- `/Brepro` is clang-cl only, and
                // `-Xclang -brepro` is unknown in clang 22. Both measured.)
                .arg("-g0")
                .arg("-c")
                .arg(&src)
                .arg("-o")
                .arg(&obj)
                .status();
            match status {
                Ok(s) if s.success() => {}
                Ok(s) => {
                    rep.failures
                        .push(format!("{} -{}: clang exit {}", sn.name, opt, s));
                    continue;
                }
                Err(e) => {
                    rep.failures
                        .push(format!("{} -{}: spawn {clang}: {e}", sn.name, opt));
                    continue;
                }
            }
            match fs::read(&obj) {
                Ok(mut bytes) => {
                    if normalize_coff_stamp(&mut bytes) {
                        // Write the normalized object back, so the artifact on
                        // disk and the manifest hash agree, and a downstream
                        // byte diff of two builds is empty.
                        let _ = fs::write(&obj, &bytes);
                    }
                    let h = fnv1a64(&bytes);
                    rep.built += 1;
                    rep.entries.push(Entry {
                        name: sn.name.to_string(),
                        opt: opt.to_string(),
                        obj: obj.clone(),
                        bytes: bytes.len() as u64,
                        hash: h,
                    });
                    manifest.insert(
                        (sn.name.to_string(), opt.to_string()),
                        (obj, bytes.len() as u64, h),
                    );
                }
                Err(e) => rep
                    .failures
                    .push(format!("{} -{}: read obj: {e}", sn.name, opt)),
            }
        }
    }

    // manifest.tsv: gates consume this, so write it even when some builds failed
    // (a partial manifest with a failure list is more debuggable than none).
    let (ver, tgt) = clang_stamp(clang);
    let mut m = format!(
        "# clang: {ver}\n# target: {tgt}\n# normalized: COFF TimeDateStamp zeroed at object offset 4\nname\topt\tobj\tbytes\tfnv1a64\n"
    );
    for ((name, opt), (path, bytes, hash)) in &manifest {
        m.push_str(&format!(
            "{name}\t{opt}\t{}\t{bytes}\t{hash:016x}\n",
            path.display()
        ));
    }
    let _ = fs::write(out.join("manifest.tsv"), m);
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coff_stamp_normalization_targets_the_real_field() {
        // Offsets are the whole point: Machine at 0, TimeDateStamp at 4 in a
        // COFF object. A guard on the wrong offset silently disables the
        // normalization and the corpus stops being reproducible.
        let mut obj = vec![0u8; 20];
        obj[0..2].copy_from_slice(&0x8664u16.to_le_bytes());
        obj[2..4].copy_from_slice(&6u16.to_le_bytes()); // NumberOfSections
        obj[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        assert!(normalize_coff_stamp(&mut obj));
        assert_eq!(&obj[4..8], &[0, 0, 0, 0]);
        assert!(!normalize_coff_stamp(&mut obj), "already normalized");

        // Anything that is not an AMD64 object must survive untouched.
        let mut other = vec![0u8; 20];
        other[0..2].copy_from_slice(&0x014cu16.to_le_bytes()); // i386
        other[4..8].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert!(!normalize_coff_stamp(&mut other));
        assert_eq!(&other[4..8], &0xdead_beefu32.to_le_bytes());
    }

    /// Corpus integrity checks that must hold without invoking a compiler:
    /// unique names, every snippet exports `f`, no libc dependency (the corpus
    /// has to compile on a bare host and must not depend on CRT behavior).
    #[test]
    fn snippets_are_wellformed() {
        let mut seen = std::collections::HashSet::new();
        for sn in SNIPPETS {
            assert!(seen.insert(sn.name), "duplicate snippet name {}", sn.name);
            assert!(
                sn.code.contains(" f(") || sn.code.contains("f ("),
                "{} must define f",
                sn.name
            );
            for banned in ["printf", "stdlib", "malloc", "include", "assert"] {
                assert!(
                    !sn.code.contains(banned),
                    "{} must not depend on {banned}",
                    sn.name
                );
            }
        }
        assert!(SNIPPETS.len() >= 24, "corpus too small to gate anything");
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a64(b"abc"), fnv1a64(b"abc"));
        assert_ne!(fnv1a64(b"abc"), fnv1a64(b"abd"));
    }
}
