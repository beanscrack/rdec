//! T2 memory effects: the caller-buffer comparison must PROVE it bites.
//!
//! The contract this file pins (roadmap: "T2 compares returned value only;
//! memory effects not proven" - the gap the caller-buffer class closes):
//!
//!  1. GREEN on its own merits: on the caller-buffer fixtures the gate runs
//!     machine code, decompiled C, and the IR interpreter on the same seed
//!     bytes, and all three leave the same buffer.
//!  2. RED on a scribble the return cannot see (`RDEC_T2_SABOTAGE=store`,
//!     the emitter sabotage: window stores land 8 bytes late): the RETURN
//!     tuple keeps `mismatch: 0` and the MEMORY tuple goes red, naming the
//!     function, the vector, and the byte offset. That asymmetry is the
//!     entire point of the lane - a decompiler that returns the right int
//!     while scribbling the caller's buffer is a broken decompiler, and a
//!     return-only gate cannot see it.
//!  3. RED on a spurious write (`RDEC_T2_SABOTAGE=prologue`, one injected
//!     `store_8` per caller-buffer function): it fires even on
//!     `read_only_words`, whose source writes nothing at all.
//!  4. The red lives in the BYTES, not the bookkeeping: adding the
//!     comparator sabotage (`store,cmp` - "lengths match" for equality)
//!     turns the same difference green. If flipping the comparator does
//!     NOT change the verdict, the memory axis is counting, not comparing.
//!
//! The sabotage hooks live in `t2.rs` (`sabotage_modes`) and corrupt the
//! DECOMPILED side only - the machine code is the oracle, never the suspect.
//!
//! The tests spawn the real `rdec` binary (`CARGO_BIN_EXE_rdec`): the gate
//! under test is the gate CI runs, in a process of its own, and stdout is
//! asserted verbatim. The corpus objects are GENERATED (`rdec corpus` runs
//! before the tests in `ci.yml`); a missing object panics loudly here - a
//! corpus-dependent test that silently skips is the exact trap `ci.yml`
//! documents twice.
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `corpus/obj/store_to_buffer.O0.obj` - the fixture, generated, asserted
/// to exist (no silent skip).
fn fixture(name: &str, opt: &str) -> PathBuf {
    let p = root().join(format!("corpus/obj/{name}.{opt}.obj"));
    assert!(
        p.exists(),
        "{} missing: run `cargo run -p rdec-cli -- corpus` first (CI generates the corpus before the tests)",
        p.display()
    );
    p
}

fn run_t2(objs: &[PathBuf], out: &Path, sabotage: Option<&str>) -> (i32, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rdec"));
    cmd.arg("t2").arg("--out").arg(out);
    for o in objs {
        cmd.arg(o);
    }
    if let Some(s) = sabotage {
        cmd.env("RDEC_T2_SABOTAGE", s);
    }
    let out = cmd.output().expect("rdec t2 runs");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

fn field(text: &str, prefix: &str) -> String {
    text.lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting with {prefix:?} in:\n{text}"))
        .to_string()
}

fn bucket(text: &str, prefix: &str, key: &str) -> usize {
    let line = field(text, prefix);
    let parts: Vec<&str> = line.split('|').map(|p| p.trim()).collect();
    let p = parts
        .iter()
        .find(|p| p.starts_with(key))
        .unwrap_or_else(|| panic!("no {key:?} in {line:?}"));
    // The value is the first token AFTER the key ("ir leg: 216", "match: 7").
    p[key.len()..]
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("no value after {key:?} in {p:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("{key:?} value not a number in {p:?}"))
}

fn tmp(tag: &str) -> PathBuf {
    let p = root().join(format!("target/t2-mem-{tag}"));
    std::fs::create_dir_all(&p).expect("tmp out dir");
    p
}

/// The class this lane exists for: on the pristine tree the caller-buffer
/// gate runs all three legs green, and the fixtures named in the task are
/// all in the comparison set (not silently excluded).
#[test]
fn caller_buffer_fixtures_run_green_on_all_three_legs() {
    let objs: Vec<PathBuf> = ["O0", "O1", "O2", "Oz"]
        .iter()
        .flat_map(|o| [fixture("store_to_buffer", o), fixture("read_only_words", o)])
        .chain(["O0", "O1"].iter().map(|o| fixture("copy_words", o)))
        .collect();
    let (code, text) = run_t2(&objs, &tmp("green"), None);
    assert_eq!(code, 0, "gate output:\n{text}");
    assert!(
        text.contains("caller-buffer: 10 functions"),
        "the 11 emitted caller-buffer fixtures (copy_words.O2 is refused by the emitter for movdqu, loudly) must enter the window class, saw:\n{text}"
    );
    let mem = field(&text, "memory:");
    assert!(mem.contains("mismatch: 0"), "memory tuple: {mem}");
    // 10 functions x vectors at the opts that emit: store_to_buffer and
    // read_only_words are loop-free (24 vectors each, 4 opts), copy_words is
    // loop-shaped (12 vectors, 2 opts - O2/Oz are refused for movdqu,
    // loudly by the emitter's own rule): 24*8 + 12*2.
    let mem_cmp = bucket(&text, "memory:", "memory:");
    assert_eq!(mem_cmp, 216, "memory comparison count moved: {mem}");
    assert_eq!(bucket(&text, "memory:", "match:"), mem_cmp);
    let ir = field(&text, "ir leg:");
    assert_eq!(
        bucket(&text, "ir leg:", "match:"),
        bucket(&text, "ir leg:", "ir leg:"),
        "IR leg not fully green: {ir}"
    );
    assert_eq!(bucket(&text, "ir leg:", "mismatch:"), 0);
    // The return tuple counts the new vectors too: every job here is a
    // window job, so the return axis runs exactly the 216 comparisons.
    assert_eq!(bucket(&text, "compared:", "compared:"), 216);
    assert_eq!(bucket(&text, "compared:", "mismatch:"), 0);
}

/// Sabotage (a): the emitter scribbles the buffer (window stores shift +8)
/// and NOTHING else. The gate must go red naming the function, the vector,
/// and the byte offset - while the return tuple keeps `mismatch: 0`. That
/// asymmetry is the whole claim of this lane.
#[test]
fn emitter_sabotage_goes_red_on_memory_and_stays_green_on_returns() {
    // The copy loop at O0/O1: the decompiled return is the constant 16, so
    // a store-address shift cannot move it, while every store the loop
    // performs lands 8 bytes late. (store_to_buffer at O0 reloads b[0] in
    // its return path, so its return moves under this sabotage too - the
    // prologue test carries the return-green half of the claim structurally.)
    let objs: Vec<PathBuf> = ["O0", "O1"]
        .iter()
        .map(|o| fixture("copy_words", o))
        .collect();
    let (code, text) = run_t2(&objs, &tmp("sab-store"), Some("store"));
    assert_eq!(
        code, 1,
        "the emitter sabotage MUST go the gate red:\n{text}"
    );
    assert!(text.contains("SABOTAGE"), "sabotage is LOUD:\n{text}");
    // The return axis: untouched by a window-address shift.
    assert_eq!(
        bucket(&text, "compared:", "mismatch:"),
        0,
        "the return comparison stayed green, as designed:\n{text}"
    );
    // The memory axis: red, with a name.
    let mem = field(&text, "memory:");
    assert!(
        bucket(&text, "memory:", "mismatch:") > 0,
        "the memory tuple must go red: {mem}"
    );
    let line = text
        .lines()
        .find(|l| l.contains("MEMORY MISMATCH") && l.contains("copy_words"))
        .expect("a MEMORY MISMATCH line must name copy_words");
    // The task's evidence bar: function, vector, byte offset, both values.
    assert!(line.contains("vector"), "line names the vector: {line}");
    assert!(
        line.contains("byte offset"),
        "line names the byte offset: {line}"
    );
    assert!(
        line.contains("word 1") && line.contains("byte offset 0x8"),
        "a shifted store moves word k to word k+1, so word 1 (byte offset 0x8) is the first difference: {line}"
    );
    assert!(
        text.contains("the return values of those runs matched"),
        "the FAIL text says the returns matched (the asymmetry): {text}"
    );
}

/// Sabotage (a') on the read-mostly fixture: a SPURIOUS store. The source
/// writes NOTHING, so the only way the buffers can differ is a write the
/// decompiler invented - exactly what "memory comparison must catch
/// spurious stores by the C" demands.
#[test]
fn prologue_sabotage_catches_a_spurious_store_in_a_read_only_function() {
    let objs: Vec<PathBuf> = ["O0", "O1", "O2", "Oz"]
        .iter()
        .map(|o| fixture("read_only_words", o))
        .collect();
    let (code, text) = run_t2(&objs, &tmp("sab-prologue"), Some("prologue"));
    assert_eq!(code, 1, "a spurious store must go the gate red:\n{text}");
    // The scribble lands at window word 48, OUTSIDE the fixture's footprint:
    // the return value cannot depend on it, so the return axis stays green
    // while the memory axis burns. This is the task's asymmetry, made
    // structural: the ONLY way this red exists is the buffer comparison.
    assert_eq!(
        bucket(&text, "compared:", "mismatch:"),
        0,
        "a write at word 48 cannot move the return:\n{text}"
    );
    assert!(
        bucket(&text, "memory:", "mismatch:") > 0,
        "read_only_words source writes nothing; a store by the C is a MISCOMPILE the bytes catch:\n{text}"
    );
    let line = text
        .lines()
        .find(|l| l.contains("MEMORY MISMATCH") && l.contains("read_only_words"))
        .unwrap_or_else(|| panic!("a MEMORY MISMATCH line must name read_only_words:\n{text}"));
    assert!(
        line.contains("word 48") && line.contains("byte offset 0x180"),
        "the line names the scribbled word: {line}"
    );
    assert!(
        text.contains("the return values of those runs matched"),
        "the FAIL text names the asymmetry:\n{text}"
    );
}

/// Sabotage (b): the COMPARATOR. `store` scribbles the C-side buffer;
/// `cmp` ("lengths match") must turn the SAME real difference green. If the
/// verdict does not change, the memory axis is bookkeeping, not a byte
/// comparison - the red's cause must be the comparator, provably.
#[test]
fn comparator_sabotage_is_what_makes_the_scribble_red() {
    // The copy loop (constant return): the ONLY axis the store sabotage can
    // move is memory, so the pair isolates the comparator exactly.
    let objs: Vec<PathBuf> = ["O0", "O1"]
        .iter()
        .map(|o| fixture("copy_words", o))
        .collect();
    // (a) store alone: red on memory.
    let (code, text) = run_t2(&objs, &tmp("sab-pair-a"), Some("store"));
    assert_eq!(code, 1);
    assert!(bucket(&text, "memory:", "mismatch:") > 0);
    // (b) store + cmp: the byte compare is gone, the SAME difference is
    // green - the sabotage run itself asserts this (it is the red that
    // disappears, so a comparator that never looked at bytes cannot be
    // distinguished from the shipped one).
    let (code, text) = run_t2(&objs, &tmp("sab-pair-b"), Some("store,cmp"));
    assert_eq!(
        code, 0,
        "with the comparator sabotaged the scribble reads green - this \
         proves the shipped red lives in the byte comparison:\n{text}"
    );
    assert_eq!(bucket(&text, "memory:", "mismatch:"), 0);
    assert!(field(&text, "memory:").contains("mismatch: 0"));
}

/// The copy-loop fixture: 16 words written, 16 returned. Its vector is a
/// buffer the loop walks, so the emitter sabotage lands on the loop's own
/// stores; the return (the constant 16) cannot see any of it.
#[test]
fn copy_loop_memory_red_return_green() {
    let objs: Vec<PathBuf> = ["O0", "O1", "Oz"]
        .iter()
        .map(|o| fixture("copy_words", o))
        .collect();
    let (code, text) = run_t2(&objs, &tmp("sab-copy"), Some("store"));
    assert_eq!(
        code, 1,
        "the copy loop's shifted stores must go red:\n{text}"
    );
    assert_eq!(bucket(&text, "compared:", "mismatch:"), 0);
    assert!(bucket(&text, "memory:", "mismatch:") > 0);
    assert!(
        text.lines()
            .any(|l| l.contains("MEMORY MISMATCH") && l.contains("copy_words")),
        "the line names copy_words:\n{text}"
    );
}
