//! The ARITY slice at the corpus gate, on the process boundary the CI job
//! uses (`rdec c --corpus`), asserting the eight named sites and the numbers
//! the gate itself prints.
//!
//! `corpus/obj/*.obj` are GENERATED (`rdec corpus`), not committed: a missing
//! corpus SKIPS LOUDLY - it names the generator command and exits 0 with the
//! word SKIP in the output, never a silent pass.
//!
//! The rule this lane follows: a gate's printed count and its pass condition
//! come from ONE parse of the gate's own line, never from a re-derived
//! number. So every number asserted below is parsed out of the `emit-c:` /
//! `refusal classes:` lines the gate printed in this run.
use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The eight sites the slice names: `call_helper`'s and `fib_rec`'s `f` in
/// each of the four optimisation levels, the ones `c --corpus` refused with
/// `opaque: clobbered by call` at 57747c0.
const NAMED: [&str; 8] = [
    "call_helper.O0",
    "call_helper.O1",
    "call_helper.O2",
    "call_helper.Oz",
    "fib_rec.O0",
    "fib_rec.O1",
    "fib_rec.O2",
    "fib_rec.Oz",
];

fn gate() -> String {
    let root = root();
    let out = Command::new(std::env!("CARGO_BIN_EXE_rdec"))
        .args(["c", "--corpus"])
        .env("RDEC_C_CENSUS_ALL", "1")
        .current_dir(&root)
        .output()
        .expect("rdec c --corpus");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    // The gate exits 1 while ANY refusal remains (the lea/Rsp/ext/shufps
    // classes are real gaps this slice does not close); the assertions below
    // read the numbers, not the exit code.
    text
}

#[test]
fn the_named_clobbered_by_call_sites_are_gone_and_the_counts_move_by_eight() {
    let root = root();
    let mut missing = Vec::new();
    for name in NAMED {
        let obj = root.join("corpus").join("obj").join(format!("{name}.obj"));
        if !obj.is_file() {
            missing.push(obj.display().to_string());
        }
    }
    if !missing.is_empty() {
        eprintln!(
            "SKIP (loud): the corpus objects are GENERATED, not committed - {} {} missing. \
             Run `cargo run -q -p rdec-cli -- corpus` (and `corpus --cpp`) first; \
             in CI the build step to the left of this test does it.",
            missing.len(),
            missing[0]
        );
        return;
    }
    let text = gate();

    // ONE parse of the gate's own line: (lifted, emitted, refused, invalid).
    let emit_line = text
        .lines()
        .find(|l| l.starts_with("emit-c:"))
        .expect("the gate prints its emit-c line");
    let nums: Vec<usize> = emit_line
        .split_whitespace()
        .filter_map(|w| w.trim_end_matches(',').parse::<usize>().ok())
        .collect();
    assert_eq!(nums.len(), 4, "four counts in `{emit_line}`");
    let (lifted, emitted, refused, invalid) = (nums[0], nums[1], nums[2], nums[3]);
    assert_eq!(
        invalid, 0,
        "`0 invalid` after passes, and `invalid` is the witness of the bug this lane fixes: \
         the SAME commit on the CI runner (clang 20.1.8) printed `136 emitted, 20 refused, 1 \
         invalid` while this tree (clang 22.1.8, `clang --version` measured here, the same \
         version the corpus manifest records) printed `135 emitted, 21 refused, 0 invalid` - \
         one frame refusal keyed on a spelling, so the function EMITTED and the emitted C \
         computed a wrong answer. A non-zero `invalid` here means a frame-aliasing caller is \
         back in the emitted set: {emit_line}"
    );
    // Floors moved 137/19 -> 135/21 on 2026-11-08 (branch `frame-activation`,
    // this tree): the frame-activation refusal took `call_helper.O0::f` and
    // `fib_rec.O0::f` OUT of the emitted set - their C computes a wrong
    // answer when a callee/recursion clobbers a frame slot the caller is
    // live on (one frame per function in the model, one per activation on
    // the machine; `frame_activation.rs` beside this file names the
    // witnesses, the census class and the t2 confirmation). The eight
    // clobber closures this row owns stay closed: the refusal class they
    // name stays gone, asserted below on the record dump.
    //
    // RE-MEASURED 2026-11-08 on branch `frame2` at 156/135/21/0 - UNMOVED: the
    // unbounded-frame-traffic rule (indexed frame access = whole-window touch)
    // adds no refusal on this box's clang, because the discovery that moved
    // the numbers is a CI fact - the runner's clang 20.1.8 spells one frame
    // store with an index register, the old bounded-only check saw no slot
    // there, that leg emitted, and its `invalid` was 1.
    assert_eq!(emitted, 135, "137 - 2 frame refusals: {emit_line}");
    assert_eq!(refused, 21, "19 + 2 frame refusals: {emit_line}");
    assert_eq!(lifted, 156, "the lift count is untouched: {emit_line}");

    // The class census, same run: `clobbered by call` is GONE and the class
    // counts partition the refused count printed above (one source).
    let class_line = text
        .lines()
        .find(|l| l.starts_with("refusal classes:"))
        .expect("the gate prints its refusal classes");
    assert!(
        !class_line.contains("clobbered by call"),
        "the class the slice closes is still in the census: {class_line}"
    );
    let class_total: usize = class_line
        .trim_start_matches("refusal classes:")
        .split('|')
        .filter_map(|c| c.split_whitespace().next()?.parse::<usize>().ok())
        .sum();
    assert_eq!(
        class_total, refused,
        "the class census and the refused count are one fact: {class_line}"
    );

    // The eight named sites appear in NO refusal record with the closed class.
    for line in text.lines().filter(|l| l.starts_with("refusal record:")) {
        for name in NAMED {
            if line.contains(name) {
                assert!(
                    !line.contains("clobbered by call"),
                    "{name} still refuses `clobbered by call`: {line}"
                );
            }
        }
    }

    // The two-pass states itself, once, in numbers.
    let arity_line = text
        .lines()
        .find(|l| l.starts_with("arity:"))
        .expect("the gate prints the `arity:` line");
    assert!(
        arity_line.contains("in-object callees recovered")
            && arity_line.contains("left at the 4-pair default"),
        "the arity census names both sides: {arity_line}"
    );

    // The newly-emitted C compiles under the gate's own flags.
    assert!(
        text.lines().any(|l| l.contains("clang -Werror: OK")),
        "clang -Werror must accept the generated C:\n{}",
        text.lines()
            .filter(|l| l.contains("clang") || l.contains("error"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
