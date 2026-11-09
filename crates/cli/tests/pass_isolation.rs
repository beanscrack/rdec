//! Pins the pass-isolation contract on the corpus - compiler-independently
//! pinned against the pipeline's OWN table, on this box.
//!
//! `rdec c --corpus` must print
//!
//! ```text
//! pass isolation: K passes verified on N emitted functions
//! ```
//!
//! with K EQUAL TO `PRE_EMIT_PASSES.len()`. This is the anti-vacuity pin: a
//! driver that iterates an empty table, skips a pass, or counts a pass that
//! never ran prints a smaller K and reddens here. The per-pass contract on
//! hand-built IR (every pass runs, every pass's post-state is verified, a
//! permanent breakage names the pass) lives in
//! `crates/core/tests/pass_isolation_fixtures.rs`; this test pins that the
//! CLI's summary line reports the same K on the full corpus, and that a clean
//! run is SILENT of breakage records (the floors say the corpus is clean; the
//! isolation line must agree).
//!
//! Floor discipline: `N` is the emitted count, asserted equal to the emit
//! floor the other corpus tests pin (135), so the summary cannot quietly
//! count a shrinking denominator.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn gate() -> String {
    let root = root();
    let sentinel = root
        .join("corpus")
        .join("obj")
        .join("locals_addressed.O2.obj");
    if !sentinel.is_file() {
        eprintln!(
            "SKIP (loud): the corpus objects are GENERATED, not committed - {} missing. \
             Run `cargo run -q -p rdec-cli -- corpus` (and `corpus --cpp`) first; \
             in CI the build step to the left of this test does it.",
            sentinel.display()
        );
        return String::new();
    }
    let out = Command::new(std::env!("CARGO_BIN_EXE_rdec"))
        .args(["c", "--corpus"])
        .current_dir(&root)
        .output()
        .expect("rdec c --corpus");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    // The gate exits 1 while ANY refusal remains; the assertions below read
    // the numbers, not the exit code (same discipline as arity_corpus).
    text
}

#[test]
fn c_corpus_reports_every_pass_verified_and_stays_silent() {
    let out = gate();
    if out.is_empty() {
        return; // skipped loudly above
    }

    // 1. the summary line, counted against the pipeline's own table.
    let want_k = rdec_core::passes::PRE_EMIT_PASSES.len();
    let line = out
        .lines()
        .find(|l| l.starts_with("pass isolation: "))
        .expect("c --corpus must print the `pass isolation: K passes verified on N emitted functions` summary");
    let rest = line.strip_prefix("pass isolation: ").unwrap();
    let mut it = rest.split_whitespace();
    let k: usize = it
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("unparsable isolation line: {line}"));
    assert!(
        it.next() == Some("passes") && it.next() == Some("verified"),
        "isolation line lost its shape: {line}"
    );
    let on = it.next().unwrap_or("");
    let n: usize = it
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("isolation line must name the emitted denominator: {line}"));
    assert_eq!(on, "on", "isolation line lost its shape: {line}");
    assert_eq!(
        k, want_k,
        "the summary names {k} verified passes; the pipeline's own table has {want_k} - a driver that iterates fewer passes verifies fewer, and that is exactly the vacuity this pin defends"
    );

    // 2. the denominator is the emit floor, not a shrinking set.
    // `emit-c: 156 lifted functions, 135 emitted C, ...` - the SECOND number.
    let emitted = out
        .lines()
        .find(|l| l.starts_with("emit-c: "))
        .and_then(|l| {
            let nums: Vec<usize> = l
                .split_whitespace()
                .filter_map(|s| s.trim_end_matches(',').parse::<usize>().ok())
                .collect();
            nums.get(1).copied()
        })
        .expect("emit-c: summary line");
    assert_eq!(
        emitted, n,
        "the isolation denominator must be the emitted count of the same run"
    );
    assert_eq!(
        n, 135,
        "emit floor: 135 functions emit C (pinned by arity_corpus/frame_activation)"
    );

    // 3. a clean corpus run carries no breakage records.
    let records: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("pass-broke-ssa record:"))
        .collect();
    assert!(
        records.is_empty(),
        "a clean corpus run must be silent of breakage records, got:\n{}",
        records.join("\n")
    );
}
