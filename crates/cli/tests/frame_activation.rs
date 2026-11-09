//! THE FRAME-ACTIVATION REFUSAL at the corpus gate, on the process boundary
//! the CI job uses (`rdec c --corpus`), asserting the named sites, the class
//! key, and the census/refused-count invariant - all parsed from the lines
//! the gate printed in this run, never re-derived.
//!
//! The defect this slice refuses (measured at `arity-wip` def1647, reproduced
//! outside the harness): the model gives every function ONE frame (entry RSP
//! is the `STACK_BASE` constant), so `call_helper.O0::f` (`o8_f`) spills to
//! `0x7fff8`/`0x7fffc` and its callee `o8_h` spills to the SAME addresses -
//! the -O0 leg answers `3*(a*a+1)` where the machine answers `3*(a*a+b)`.
//! `fib_rec` is the same defect through recursion: activation 2 overwrites
//! activation 1's slots.
//!
//! `corpus/obj/*.obj` are GENERATED (`rdec corpus`), not committed: a missing
//! corpus SKIPS LOUDLY - it names the generator command and exits 0 with the
//! word SKIP in the output, never a silent pass. The test rides the `gates`
//! job, whose steps to the LEFT generate the C corpus (the precedent is
//! `arity_corpus.rs` in this directory); in a leg that only ran `corpus
//! --msvc` this is the skip proving the guard, not a pass.
use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../..")
}

const KEY: &str = "frame slots alias the callee's";

fn gate() -> String {
    let root = root();
    let out = Command::new(std::env!("CARGO_BIN_EXE_rdec"))
        .args(["c", "--corpus"])
        // The record dump: the class census tells you a class exists, the
        // records tell you WHICH function grew it.
        .env("RDEC_C_CENSUS_ALL", "1")
        .current_dir(&root)
        .output()
        .expect("rdec c --corpus");
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    text
}

#[test]
fn the_frame_aliasing_sites_refuse_with_the_key_and_the_census_partitions_refused() {
    let root = root();
    // Guard the OBJECTS, never the manifest: `corpus/manifest.tsv` is
    // committed, so an is_file() on it is a green light with zero inputs.
    let obj_dir = root.join("corpus").join("obj");
    let have_objects = std::fs::read_dir(&obj_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|x| x == "obj"))
        })
        .unwrap_or(false);
    if !have_objects {
        eprintln!(
            "SKIP (loud): no .obj under {} - the corpus objects are GENERATED, not \
             committed. Run `cargo run -q -p rdec-cli -- corpus` first; in CI the \
             build step to the left of this test does it.",
            obj_dir.display()
        );
        return;
    }
    let text = gate();

    // ONE parse of the gate's own `emit-c:` line: (lifted, emitted, refused, invalid).
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
    // THE `0 invalid` ASSERTION IS THE POINT OF THIS FILE. 2026-11-08, CI run of
    // `ab64a4d`, `gates` job, verbatim:
    //     assertion `left == right` failed: `0 invalid` after passes: emit-c: 156
    //     lifted functions, 135 emitted C, 20 refused, 1 invalid after passes
    //     left: 1   right: 0
    // with this box at `135 emitted, 21 refused, 0 invalid` on the same commit.
    // Same lifted, same emitted, a DIFFERENT BUCKET: the runner's clang (20.1.8,
    // per `.github/workflows/ci.yml`; this box's is 22.1.8) built that function's
    // frame store as an INDEXED access, the bounded-only check resolved no slot
    // there, the refusal did not fire, the caller EMITTED, and the differential
    // pass caught the emitted C as `invalid`. The `invalid` is the pipeline being
    // honest about a miscompile; the fix is the rule, and this line is the floor
    // that says the miscompile is not in the emitted set.
    assert_eq!(invalid, 0, "`0 invalid` after passes: {emit_line}");
    assert_eq!(
        lifted, 156,
        "the lift count is untouched by a pre-emit refusal: {emit_line}"
    );
    // The measured floor at this commit (see ROADMAP): the frame refusals
    // took 2 functions out of the 137 the arity slice emitted -
    // call_helper.O0::f (its callee h clobbers slots f reads after the call)
    // and fib_rec.O0::f (recursion); the liveness bound keeps the fib_rec
    // O1/O2/Oz legs (calls, no live read of a shared slot after the call)
    // and tailcall_local.O0::f (all frame loads before the call) emitting,
    // and t2 confirms: 2560 compared = the base 2452 + 72 + 36, all matching.
    //
    // RE-MEASURED 2026-11-08 on branch `frame2` (unbounded frame traffic =
    // whole-window touch): 156 lifted / 135 emitted / 21 refused / 0 invalid,
    // UNMOVED on this box's clang 22.1.8. The rule fires on 3 corpus functions
    // (`frame traffic:` line below: 49 named / 3 indexed / 104 none, all three
    // `locals_addressed`, which calls nothing), so the widening costs no
    // emission here - what it buys is that the refusal no longer depends on
    // which compiler built the objects, which is exactly what the CI numbers
    // above measured.
    assert_eq!(
        emitted, 135,
        "measured floor, branch frame2 (clang-version-independent rule): {emit_line}"
    );

    // The class-sum == refused-count invariant, on the gate's OWN lines: one
    // refusal record per refused function, in exactly one class.
    let class_line = text
        .lines()
        .find(|l| l.starts_with("refusal classes:"))
        .expect("the gate prints its refusal classes");
    let class_total: usize = class_line
        .trim_start_matches("refusal classes:")
        .split('|')
        .filter_map(|c| c.split_whitespace().next()?.parse::<usize>().ok())
        .sum();
    assert_eq!(
        class_total, refused,
        "the class census and the refused count are one fact: {class_line} vs `{emit_line}`"
    );
    let frame_count = class_line
        .split('|')
        .filter_map(|c| {
            let c = c.trim();
            c.contains(KEY)
                .then(|| c.split_whitespace().next()?.parse::<usize>().ok())
                .flatten()
        })
        .sum::<usize>();
    assert!(
        frame_count > 0,
        "the frame-activation class is absent from the census: {class_line}"
    );

    let records: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("refusal record:"))
        .collect();

    // The named sites. call_helper.O0::f: the refusal names the caller, the
    // callee AND the shared address - the sentence is the map.
    let f_rec = records
        .iter()
        .find(|l| l.contains("call_helper.O0") && l.contains(KEY))
        .expect("call_helper.O0::f must refuse with the frame key");
    assert!(
        f_rec.contains("o8_f and o8_h") && f_rec.contains("0x7fff8"),
        "the record names caller, callee and a shared address: {f_rec}"
    );
    // fib_rec: recursion, named as a self-call.
    let fib: Vec<&&str> = records
        .iter()
        .filter(|l| l.contains("fib_rec") && l.contains(KEY))
        .collect();
    assert!(
        !fib.is_empty(),
        "fib_rec's recursive legs must refuse with the frame key"
    );
    for l in &fib {
        assert!(
            l.contains("calls itself"),
            "the recursion text must name the self call: {l}"
        );
    }

    // The anti-over-fire leg, at corpus level, MEASURED against t2: the
    // no-frame legs of call_helper (O1/O2/Oz - calls, no frame slots), the
    // fib_rec O1/O2/Oz legs (recursive, but every read of a shared slot
    // happens BEFORE the call - the C's clobber is unobservable, and t2
    // matches their vectors), and tailcall_local.O0::f (a plain -O0 call
    // whose frame loads all precede it; O1/O2/Oz are modelled TAIL CALLS
    // whose caller activation dies at the `jmp`). None of these may carry a
    // frame-class record - the refusal may cost only what the defect's
    // reach covers, and every one of them is a MATCHING t2 comparison that
    // the refusal would delete.
    for name in [
        "call_helper.O1",
        "call_helper.O2",
        "call_helper.Oz",
        "tailcall_local.O0",
        "fib_rec.O1",
        "fib_rec.O2",
        "fib_rec.Oz",
    ] {
        for l in &records {
            if l.contains(name) {
                assert!(
                    !l.contains(KEY),
                    "{name} refuses with the frame key, but its leg has no frame slots: {l}"
                );
            }
        }
    }

    // Every frame-class record carries the evidence sentence: the census class
    // cannot be reached by a text that does not name the shared frame - a
    // window address when both sides are constant-bounded, the word `unbounded`
    // when one side's frame traffic is indexed and the model cannot name one.
    for l in &records {
        if l.contains(KEY) {
            assert!(
                l.contains("0x7ff") || l.contains("0x7f") || l.contains("unbounded"),
                "a frame-key record must name a frame-window address or say unbounded: {l}"
            );
        }
    }
    // The frame-traffic census PARTITIONS the lifted set. The buckets move with
    // the compiler that built the objects (this box's clang 22.1.8: 49 named
    // slots / 3 unbounded / 104 none; an indexed-spelling codegen moves the
    // middle number up), the TOTAL does not, and the refusal rule reads all
    // three buckets - so this is the version-independent half of the floor: the
    // day a corpus rebuild shifts traffic between buckets, this line says the
    // census still accounts for every function.
    let traffic = text
        .lines()
        .find(|l| l.starts_with("frame traffic:"))
        .expect("the gate prints its `frame traffic:` census");
    let buckets: Vec<usize> = traffic
        .split_whitespace()
        .filter_map(|w| w.parse::<usize>().ok())
        .collect();
    assert_eq!(
        buckets.len(),
        3,
        "three buckets in `{traffic}` (named / unbounded / no frame traffic)"
    );
    assert_eq!(
        buckets.iter().sum::<usize>(),
        lifted,
        "frame traffic must account for every lifted function: `{traffic}` vs `{emit_line}`"
    );
    assert!(
        buckets[0] + buckets[1] > 0,
        "no function touches the frame at all, so the frame check is measuring nothing: {traffic}"
    );
    // The refused count moves by exactly the frame class: 19 (measured at
    // def1647) + frame_count, one source for both numbers.
    assert_eq!(
        refused,
        19 + frame_count,
        "the refusal total is def1647's 19 plus the frame class: `{emit_line}` ({frame_count} frame)"
    );
}
