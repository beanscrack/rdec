//! The bounded CI fuzz run THROUGH THE DEMANGLER.
//!
//! `crates/arch-x86/tests/fuzz.rs` drives `fuzz::front_half`, which is the
//! arch-x86 front half: decoder, container, seeds, discovery, lift, passes,
//! verify, emit, interpreter. It never touches `rdec-demangle` — the crate does
//! not even depend on it — so for all its counters, the CI fuzz surface had a
//! hole exactly where the demangler's own bug history lives (`peek().unwrap()`
//! at three sites, the `split_at(1)` truncation panic, the unbounded type
//! nesting that overflowed the stack). Verified, not assumed: with the
//! `split_at(1)` bug put back, that test reports `crashes 0`.
//!
//! This test closes the hole from the crate that has both dependencies, using
//! the same driver, the same panic-catching, and the same census discipline. The
//! name family's first inputs are the DETERMINISTIC enumeration
//! (`fuzz::truncated_token`: every token the demangler branches on, cut at every
//! length, closed with the `@` that terminates an MSVC identifier), so the
//! truncation class is walked every single run — no seed has to get lucky.
//!
//! RED DIRECTION (executed, table in the commit message): put the
//! `split_at(1)` panic back in `crates/demangle/src/lib.rs` (replace
//! `rest.split_at_checked(1)` with `Some(rest.split_at(1))`) and this test fails
//! with `1 panic(s), first: "seed ..., iteration 135, generator name (name:
//! ??_@)"` and the panic message; restore, and it is green.

use rdec_arch_x86::fuzz::{self, Config, Stats};

const ITERATIONS: u64 = 400;

#[test]
fn names_through_the_demangler_never_panic_and_the_run_actually_ran() {
    let cfg = Config {
        seed: 0x51_7A_1E_05,
        iterations: ITERATIONS,
        max_seconds: 120,
        max_input_ms: 2_000,
        keep_crashes: 8,
        max_input_bytes: 1 << 16,
        trace: false,
    };
    let mut st = Stats::default();
    // No corpus seeds: the name family is self-contained and the object
    // families generate from scratch, which keeps this test's runtime in
    // milliseconds and its input set identical on every machine.
    fuzz::run(&cfg, &[], &mut st, &mut |input, sub| {
        if input.gen == "name" {
            // Exactly the CLI census's name branch: the string goes straight to
            // the production demangler.
            let name = input
                .detail
                .strip_prefix("name: ")
                .unwrap_or(&input.detail)
                .to_string();
            match rdec_demangle::demangle(&name) {
                rdec_demangle::Demangled::Name { .. } => sub.bump("demangle.name", 1),
                rdec_demangle::Demangled::Unsupported { .. } => sub.bump("demangle.unsupported", 1),
            }
            sub.bump("name.inputs", 1);
        } else {
            fuzz::front_half(input, sub);
        }
    });

    println!(
        "demangle-fuzz: {} inputs (asked {ITERATIONS}), demangle.name={} \
         demangle.unsupported={} front-half={} crashes {} wall {:.1}s",
        st.iterations,
        st.get("demangle.name"),
        st.get("demangle.unsupported"),
        st.get("load.ok") + st.get("load.err"),
        st.get("crash"),
        st.elapsed.as_secs_f64(),
    );
    for c in &st.crashes {
        println!(
            "demangle-fuzz: CRASH [{}] stage {}: {}",
            c.input.provenance(),
            c.stage,
            c.message.lines().next().unwrap_or("").trim()
        );
    }

    assert_eq!(
        st.iterations, ITERATIONS,
        "the fuzzer ran {} of {ITERATIONS} iterations",
        st.iterations
    );
    // The coverage asserts: a "0 panics" that ran on nothing is not a gate.
    assert!(
        st.get("name.inputs") > 0,
        "the name family fed the demangler 0 names"
    );
    assert!(
        st.get("demangle.name") + st.get("demangle.unsupported") > 0,
        "the demangler was never called: this test would pass on any bug"
    );
    // The truncation enumeration is what makes the truncation bug class a gate.
    // Assert it ran and that it covers the operator-tag token: iteration 135 of
    // a run whose name inputs are `iter % 8 == 7` is the 17th name input, the
    // full `??_` token closed with `@` - the exact shape the `split_at(1)` bug
    // panics on. Hand-counted from `fuzz::truncated_token`'s table order
    // (`?` 2 cuts, `??` 3, `??0` 4, `??1` 4, `??_` 4 -> index 16).
    assert_eq!(
        rdec_arch_x86::fuzz::truncated_token(16),
        Some("??_@".to_string()),
        "the deterministic truncation table must include the operator-tag token"
    );
    assert_eq!(
        st.get("crash"),
        0,
        "{} panic(s), first: {:?}",
        st.get("crash"),
        st.crashes
            .first()
            .map(|c| (c.input.provenance(), c.message.clone()))
    );
    assert_eq!(
        st.get("slow.over_budget"),
        0,
        "{} input(s) over the per-input ceiling (slowest {} ms): the ceiling \
         cannot interrupt a running input, so this is an unbounded algorithm",
        st.get("slow.over_budget"),
        st.slowest_ms
    );
}
