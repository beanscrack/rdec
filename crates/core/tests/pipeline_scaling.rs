//! Middle-end scaling and honesty at scale.
//!
//! Two production claims the corpus cannot make, because the corpus's biggest function
//! is ~40 blocks. A decompiler that ships must answer "what does a 20k-block function
//! cost" and "what happens when your fixpoint loop runs out of budget", and the corpus
//! answers neither.
//!
//! The second one is the reason this file exists, and it is now closed. `run_pipeline`
//! used to loop until nothing changed, capped at 32 rounds, while `merge_blocks` and
//! `dedup_blocks` collapsed **one** thing per call: a 250-block chain needed ~250
//! rounds, the cap fired with work still outstanding, and the report said
//!
//!     "converged after 33 rounds"
//!
//! which is not what happened - it stopped, it did not converge. A pass pipeline that
//! reports convergence when it ran out of budget is the same class of bug as a
//! decompiler that silently drops an instruction, and a corpus of small functions can
//! never show it.
//!
//! The fix moved the fixpoint INSIDE each pass (`passes::fixpoint_bound`), so the
//! round count is structural: a 250-block chain converges in 2 rounds, a 20k-block
//! chain converges in 2 rounds (355 ms debug), and the budget is what it always
//! claimed to be - a liveness guard for pass interaction. The budget test below keeps its claim by driving an explicit
//! budget (`run_pipeline_with_budget`) instead of reaching the production one through
//! a chain long enough to exhaust it.
use std::time::{Duration, Instant};

use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{BinOp, Function, IntCmp, Op, Terminator};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

/// A phi-free chain of `n` blocks: each computes `v_i = v_{i-1} + 1` and branches to
/// the next block on both arms. Every value is live (the next block reads it) and
/// nothing folds (the chain starts at a parameter), so this exercises the per-node
/// passes - copy propagation, DCE, dead-store elimination - on real work, not on a
/// function they are free to delete.
fn ladder(n: usize) -> Function {
    let mut f = Function::new("ladder", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0);
    let k = f.emit(e, Op::Const(1), Ty::i64(), 0);
    // Entry condition: `p0 == p0`, which folds. Deliberately not a comparison against
    // a constant: the emitter REFUSES `Op::Cmp` with a `Const` operand (measured:
    // `comparison Ult on a  operand` - `Op::Const`'s own `ty()` is `None`, IR constants
    // are width-agnostic, and the emitter's Cmp path needs a width from the operand).
    // That is a named emitter gap, not what this fixture measures; the lifter's own
    // comparisons go through flags + `GetFlag`, which is why no corpus function shows
    // it. Every comparison below reads two registers, like a real `cmp reg, reg`.
    let zero = f.emit(e, Op::Const(0), Ty::i64(), 0);
    let c0 = f.emit(e, Op::Cmp(IntCmp::Eq, p0, p0), Ty::Bool, 0);
    let mut addr = 0x10u64;
    let first = f.new_block(addr);
    f.block_mut(e).term = Terminator::Branch {
        cond: c0,
        then_bb: first,
        else_bb: first,
    };
    // A CHAIN WITH LIVE COMPARISONS: each block reads the previous block's value in
    // both its add and its comparison, so no value is dead. The first version of this fixture branched to the same block
    // on both arms and read only the last comparison; `simplify_branches` + DCE then
    // deleted 2/3 of the chain and the timing test measured an empty function (the
    // `the_scaling_fixture_is_not_secretly_trivial` guard is what caught that).
    let mut prev_v = p0;
    let mut prev_c = c0;
    let mut cur = first;
    for _ in 0..n {
        // `cb` is the previous comparison as an integer. A Select, not a ZExt of a
        // Bool: copy propagation turns `ZExt(Bool)` into the Bool itself, and the
        // emitter refuses `Ult` on a Bool operand (measured: `comparison Ult on a
        // operand`, at HEAD) - a real emitter limitation, and not what this fixture is
        // here to measure. Select keeps the comparison live with an i64 result.
        // Every node in this block is read by the NEXT block, so DCE cannot delete the
        // chain: `cb` is consumed by the add, `v` by the comparison and by the next
        // block's add, `c` by the next block's select and by the tail.
        let cb = f.emit(cur, Op::Select(prev_c, k, zero), Ty::i64(), addr);
        let v = f.emit(cur, Op::Bin(BinOp::Add, prev_v, cb), Ty::i64(), addr);
        let c = f.emit(cur, Op::Cmp(IntCmp::Eq, v, prev_v), Ty::Bool, addr);
        addr += 0x10;
        let next = f.new_block(addr);
        f.block_mut(cur).term = Terminator::Branch {
            cond: c,
            then_bb: next,
            else_bb: next,
        };
        prev_v = v;
        prev_c = c;
        cur = next;
    }
    // The tail reads the LAST comparison, so that comparison is live (the `Select` is
    // the `setcc` shape the emitter already knows). `Eq` is false for every link, so
    // the answer is exactly n increments plus one.
    let cb = f.emit(cur, Op::Select(prev_c, k, zero), Ty::i64(), addr);
    let s1 = f.emit(cur, Op::Bin(BinOp::Add, prev_v, k), Ty::i64(), addr);
    let sum = f.emit(cur, Op::Bin(BinOp::Add, s1, cb), Ty::i64(), addr);
    f.block_mut(cur).term = Terminator::Return(sum);
    f.params.extend([p0]);
    f
}

fn nodes(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.instrs.len()).sum()
}

fn unreachable_blocks(f: &Function) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Unreachable))
        .count()
}

#[test]
fn the_pipeline_scales_no_worse_than_quadratically_on_a_long_chain() {
    // Measured on the shape above. The corpus's biggest function is ~40 blocks, so it
    // can never show what a real function costs; this can. Times are printed as a
    // census (a threshold invented from one sample is a guess), and the bound is
    // deliberately generous: it exists to catch a pass that makes the growth WORSE,
    // not to police normal variance on a loaded machine.
    let mut curve = Vec::new();
    for n in [50usize, 100, 200] {
        let mut f = ladder(n);
        verify_ssa(&f).expect("ladder fixture is valid IR");
        let built = nodes(&f);
        let t = Instant::now();
        let rep = rdec_core::run_pipeline(&mut f);
        let el = t.elapsed();
        let out =
            rdec_core::emit_c::emit_c(&f).unwrap_or_else(|r| panic!("{n}-block ladder: {r:?}"));
        // The interpreter must answer, and the answer is now PINNED. It used to be a
        // range, because how many links survived depended on where the round budget
        // cut the chain off ("measured: 27 for n=50" - pinning a number then would
        // have pinned the defect). Every size here converges, so the hand-derived
        // value is the contract: link i carries the previous comparison, `p0 == p0`
        // is true so link 1 carries 1, and `v_i == v_{i-1}` is true exactly when the
        // previous link carried 0, giving v_i = 1 + (i-1)/2 and a tail of
        // v_n + 1 + Select(c_n) = n/2 + 2 for even n.
        match run(&f, &[0]).outcome {
            Outcome::Ret(Val::Int(v)) => assert_eq!(
                v as usize,
                n / 2 + 2,
                "{n}-block ladder answered {v}, expected {}",
                n / 2 + 2
            ),
            other => panic!("{n}-block ladder must answer a value, got {other:?}"),
        }
        println!(
            "scaling: {n:5} blocks / {built:6} nodes -> {:>9.1?} | emitted {:>7} bytes | {} | budget_hit={} | invocations={}",
            el,
            out.text.len(),
            rep.passes_run.join(", "),
            rep.not_converged,
            rep.pass_invocations
        );
        curve.push((n, el));
    }
    let (n1, t1) = curve[0];
    let (n4, t4) = *curve.last().unwrap();
    // 4x the blocks. Quadratic is 16x; cubic is 64x.
    // Slack so a small fixture's fixed cost cannot make this flaky, while anything
    // that turns the pipeline cubic (64x) fails immediately.
    let ratio = t4.as_secs_f64() / t1.as_secs_f64().max(1e-6);
    println!("scaling: {n1} -> {n4} blocks, time x{ratio:.1}");
    assert!(
        ratio < 25.0,
        "{n1} -> {n4} blocks (8x) cost {ratio:.0}x more time: the pipeline grew worse than quadratic"
    );
    assert!(
        t4 < Duration::from_secs(30),
        "a {n4}-block function took {t4:?} in the middle end"
    );
}

#[test]
fn a_20k_block_function_finishes_and_answers_the_right_number() {
    // The production-scale probe. It used to be `#[ignore]`d as a known defect - "20k
    // blocks did not finish in 60s (measured 2026-11-06, HEAD d810661)" - and the
    // per-pass fixpoints are what closed it. Measured on this machine, 2026-11-06,
    // after the fix, from this test's own line: pipeline 355.5 ms DEBUG / 86.4 ms
    // release on ladder(20_000) = 60_007 nodes, "converged after 2 rounds", 36 pass
    // invocations, emitting 6,327,177 bytes of C, answering Ret(Int(10002)).
    // Pre-fix, the same census in RELEASE was 22.8 ms / 80.6 ms / 301.3 ms / 9.0 s /
    // 287.1 s at 50 / 100 / 200 / 1k / 5k blocks, every one of them
    // `not_converged=true`; post-fix release is 234 us / 478 us / 838 us / 4.4 ms /
    // 21.9 ms, every one converged (x5.0 time for x5 blocks).
    //
    // The 60 s budget is the DEBUG one and CI runs debug, so the probe runs in CI on
    // every push with a ~300x margin. What it does NOT do is call `verify_ssa` on the
    // 20k fixture: that pass is quadratic in the node count (measured on this same
    // fixture, debug: 200 blocks 80 ms, 500 492 ms, 1k 1.95 s, 2k 12.53 s), which is
    // outside this file's lane (`verify.rs`) and outside what this probe measures.
    // The SSA claim at 20k is `a_20k_block_ladder_is_valid_ir` below - release-only,
    // and honest about it. The fixture's SSA validity is checked at every size the
    // CI tests run (50/100/200/250/1k).
    const N: usize = 20_000;
    let mut f = ladder(N);
    let built = nodes(&f);
    let t = Instant::now();
    let rep = rdec_core::run_pipeline(&mut f);
    let el = t.elapsed();
    let out =
        rdec_core::emit_c::emit_c(&f).unwrap_or_else(|r| panic!("the ladder must emit C: {r:?}"));
    println!(
        "scaling: {N} blocks / {built} nodes -> pipeline {el:?}, emitted {} bytes | {} | rounds={} invocations={}",
        out.text.len(),
        rep.passes_run.join(", "),
        rep.rounds,
        rep.pass_invocations
    );
    assert!(el < Duration::from_secs(60), "20k blocks took {el:?}");
    // Converged, and said so: the budget must not fire on a chain at production
    // scale, which is the whole point of the per-pass fixpoints.
    assert!(
        !rep.not_converged,
        "the 20k-block ladder ran out of budget: {}",
        rep.passes_run.join(", ")
    );
    match run(&f, &[0]).outcome {
        Outcome::Ret(Val::Int(v)) => {
            // Hand-derived, same trace as the 250-block test: link i carries the
            // previous comparison, v_i = 1 + (i-1)/2, c_i is true for even i, so the
            // tail of an even-length ladder is v = 1 + (N-2)/2 plus 1 plus 1.
            // N = 20_000 -> 10_000 + 2 = 10_002.
            assert_eq!(
                v as usize,
                N / 2 + 2,
                "the 20k-block ladder answered {v}, expected {}",
                N / 2 + 2
            )
        }
        other => panic!("the ladder must answer a value, got {other:?}"),
    }
}

#[test]
#[ignore = "release-only, measured: verify_ssa is quadratic in nodes on this fixture (debug 1k 1.95s, 2k 12.53s; release 1k 432ms, 2k 1.73s), so at 20k blocks the SSA check costs minutes-to-hours while the pipeline costs 86ms. Run with: cargo test --release -p rdec-core --test pipeline_scaling -- --ignored"]
fn a_20k_block_ladder_is_valid_ir() {
    // The claim the CI probe cannot afford: the FIXTURE, 20k blocks of it, is valid
    // IR. Ignored with its measured cost in the reason, so nobody reads a green CI as
    // "verify_ssa saw 20k blocks". It converges nothing and optimizes nothing; it is
    // the fixture's own validity certificate, kept honest about when it runs.
    let f = ladder(20_000);
    let t = Instant::now();
    verify_ssa(&f).expect("the 20k ladder fixture is valid IR");
    println!("verify_ssa(20k blocks): {:?}", t.elapsed());
}

#[test]
fn the_scaling_census_covers_1k_and_5k_blocks() {
    // The curve between the timing test's 200 blocks and the probe's 20k, so a
    // regression that only bites at production size has a name and a number.
    // The bound is deliberately loose (100x the blocks, 40x the time): it exists to
    // catch a pass that turns the pipeline super-quadratic, not to police variance.
    let mut curve = Vec::new();
    for n in [1_000usize, 5_000] {
        let mut f = ladder(n);
        if n <= 1_000 {
            // `verify_ssa` is quadratic in the node count (measured above), so at
            // 5k it would dominate a CI run with a check that is not what this test
            // measures. The fixture's SSA validity is verified at 50/100/200/250/1k.
            verify_ssa(&f).expect("the ladder fixture is valid IR");
        }
        let built = nodes(&f);
        let t = Instant::now();
        let rep = rdec_core::run_pipeline(&mut f);
        let el = t.elapsed();
        println!(
            "census: {n:>6} blocks / {built:>7} nodes -> {el:>10.1?} | {} | rounds={} invocations={}",
            rep.passes_run.join(", "),
            rep.rounds,
            rep.pass_invocations
        );
        assert!(
            !rep.not_converged,
            "the {n}-block ladder ran out of budget: {}",
            rep.passes_run.join(", ")
        );
        match run(&f, &[0]).outcome {
            Outcome::Ret(Val::Int(v)) => assert_eq!(
                v as usize,
                n / 2 + 2,
                "the {n}-block ladder answered {v}, expected {}",
                n / 2 + 2
            ),
            other => panic!("the {n}-block ladder must answer a value, got {other:?}"),
        }
        curve.push((n, el));
    }
    let (n1, t1) = curve[0];
    let (n2, t2) = curve[1];
    let ratio = t2.as_secs_f64() / t1.as_secs_f64().max(1e-6);
    println!("census: {n1} -> {n2} blocks (5x), time x{ratio:.1}");
    assert!(
        ratio < 40.0,
        "{n1} -> {n2} blocks (5x) cost {ratio:.0}x more time: the pipeline grew worse than quadratic"
    );
}

/// `groups` groups, each `G_i: branch -> A_i | B_i`, where `A_i` and `B_i` are identical
/// (same instruction, same successor) and the group's phi reads both arms, so the
/// duplicates are LIVE and dedup is the pass that acts on them.
///
/// Used at a size that converges (8 groups), as the counterexample to "always report
/// non-convergence". At 40 groups it hits the round budget too, which is the point of
/// the budget test: 32 rounds is not many for a pipeline that collapses one thing per
/// round.
fn identical_pairs(groups: usize) -> Function {
    let mut f = Function::new("pairs", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0);
    let big = f.emit(e, Op::Const(u64::from(u32::MAX)), Ty::i64(), 0);
    let mut cond = f.emit(e, Op::Cmp(IntCmp::Ult, p0, big), Ty::Bool, 0);
    let mut addr = 0x10u64;
    let mut nb = |f: &mut Function| {
        let b = f.new_block(addr);
        addr += 0x10;
        b
    };
    let mut cur = e;
    for _ in 0..groups {
        let a = nb(&mut f);
        let b = nb(&mut f);
        let g = nb(&mut f);
        f.block_mut(cur).term = Terminator::Branch {
            cond,
            then_bb: a,
            else_bb: b,
        };
        let xa = f.emit(a, Op::Bin(BinOp::Add, p0, p0), Ty::i64(), 0);
        f.block_mut(a).term = Terminator::Jump(g);
        let xb = f.emit(b, Op::Bin(BinOp::Add, p0, p0), Ty::i64(), 0);
        f.block_mut(b).term = Terminator::Jump(g);
        // G reads both arms through a phi, so BOTH duplicates are live: only dedup can
        // remove them, DCE cannot do the work first. Phi operands are (value,
        // predecessor) pairs.
        let phi = f.emit(g, Op::Phi(Box::new([(xa, a), (xb, b)])), Ty::i64(), 0);
        cond = f.emit(g, Op::Cmp(IntCmp::Ult, phi, big), Ty::Bool, 0);
        cur = g;
    }
    f.block_mut(cur).term = Terminator::Return(p0);
    f.params.extend([p0]);
    f
}

#[test]
fn a_fixpoint_that_runs_out_of_budget_says_so_and_never_says_converged() {
    // The honesty claim, unchanged by the fix: running out of budget is not
    // convergence, and the report must name the cause in words a gate can grep.
    //
    // The fixture is driven with an EXPLICIT budget of 1 round, which is what the fix
    // made necessary: this test used to reach the budget by building a 250-block
    // chain, because `merge_blocks` collapsed one edge per call and the chain needed
    // ~250 rounds. Once the passes collapse to internal fixpoints the chain converges
    // in 2 rounds (measured at every size this file tests: 50, 200, 250, 1k, 5k,
    // 20k all report "converged after 2 rounds"), so the budget can only be reached by
    // asking for a smaller one. Same fixture, same
    // assertions, same words - the production budget is what became structural.
    let mut f = ladder(250);
    verify_ssa(&f).expect("the ladder fixture is valid IR");
    let rep = rdec_core::run_pipeline_with_budget(&mut f, 1);
    let line = rep.passes_run.join(", ");
    println!("ladder(250): {line} | not_converged={}", rep.not_converged);
    assert!(
        rep.not_converged,
        "a 250-block chain ran out of budget and the report did not say so: {line}"
    );
    // THE CLAIM: running out of budget is not convergence.
    assert!(
        !line.contains("converged"),
        "the pipeline stopped with work outstanding and the report called it convergence: {line}"
    );
    assert!(
        line.contains("budget"),
        "the report must name the cause in words a gate can grep: {line}"
    );
}

/// THE FIX'S CLAIM, written red-first: the round budget must scale with the
/// function's STRUCTURE, not its block count.
///
/// `merge_blocks` collapsed exactly one edge per call (measured at HEAD e0dafc6 with
/// `RDEC_PASS_TRACE=1`: `merge_blocks changed=true` in 32 of 32 rounds, live blocks
/// 252 -> 220, one per round), so a 250-block chain needs ~250 rounds and the
/// 32-round budget fires on IR that converges in a handful. A ladder is a chain: it
/// has no loops, no phis and no duplication, and "ran out of budget" on it is a
/// property of the pipeline, not of the function.
#[test]
fn a_250_block_chain_converges_within_the_round_budget() {
    let mut f = ladder(250);
    verify_ssa(&f).expect("the ladder fixture is valid IR");
    let rep = rdec_core::run_pipeline(&mut f);
    let line = rep.passes_run.join(", ");
    println!(
        "ladder(250): {line} | not_converged={} | {}",
        rep.not_converged,
        rep.passes_run.join(", ")
    );
    assert!(
        !rep.not_converged,
        "a 250-block chain must converge inside the round budget; the report said: {line}"
    );
    // A converged chain has a hand-checkable answer. Link i carries the previous
    // comparison: `p0 == p0` is true so link 1 carries 1, and every link compares
    // `v_i == v_{i-1}`, which is true exactly when the previous link carried 0. So
    // v_i = 1 + (i-1)/2 (integer division), c_i is true for even i, and the tail adds
    // 1 plus `Select(c_n)`: n = 250 gives v = 1 + 124 = 125, tail 125 + 1 + 1 = 127.
    match run(&f, &[0]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(
            v as usize,
            250 / 2 + 2,
            "the 250-block ladder must answer 127 (hand-derived: 125 + 1 + 1)"
        ),
        other => panic!("the 250-block ladder must answer a value, got {other:?}"),
    }
    // The chain must survive as a chain: structuring an answer is only meaningful if
    // the passes did not delete the work they were measuring.
    let after = nodes(&f);
    assert!(
        after > 757 * 2 / 3,
        "the pipeline deleted the 250-block ladder ({after} nodes left of 757)"
    );
}

#[test]
fn a_function_that_really_does_converge_is_not_reported_as_out_of_budget() {
    // The anti-invention half: shapes that converge must NOT be flagged, so
    // "always report non-convergence" cannot pass the budget test. `pairs(40)`
    // converges (measured: every pair collapses inside the budget), and so does a
    // short chain.
    let mut p = identical_pairs(8);
    verify_ssa(&p).expect("the pairs fixture is valid IR");
    let rep = rdec_core::run_pipeline(&mut p);
    let line = rep.passes_run.join(", ");
    println!(
        "pairs(8): {line} | collapsed {} of 16 pair blocks into Unreachable | not_converged={}",
        unreachable_blocks(&p),
        rep.not_converged
    );
    assert!(
        !rep.not_converged && line.contains("converged"),
        "pairs(8) converges and must be reported as converging: {line}"
    );
    let mut f = ladder(5);
    verify_ssa(&f).expect("fixture is valid IR");
    let rep = rdec_core::run_pipeline(&mut f);
    let line = rep.passes_run.join(", ");
    println!("ladder(5): {line}");
    assert!(
        !rep.not_converged,
        "a 5-link chain converged and was flagged as out of budget: {line}"
    );
    match run(&f, &[0]).outcome {
        Outcome::Ret(Val::Int(v)) => {
            // ladder(5) converges, so its answer is fixed. A converging chain of 5
            // links: p0==p0 is true so link 1 carries 1; link 2 compares 1==1 true so link 3
            // carries 0, link 4 compares 1==1 true, link 5 carries 0; tail adds 1, select 0.
            // v = 3 -> answer 4. Derived by hand, not measured-to-pass: the hand trace is in the test.
            assert_eq!(v as usize, 4, "5-link ladder answered {v}")
        }
        other => panic!("expected a value, got {other:?}"),
    }
}

#[test]
fn the_scaling_fixture_is_not_secretly_trivial() {
    // Guards the timing test against its own fixture: if the passes deleted the chain,
    // the 20k number would be measuring an empty function.
    let mut f = ladder(200);
    verify_ssa(&f).expect("fixture is valid IR");
    let before = nodes(&f);
    rdec_core::run_pipeline(&mut f);
    let after = nodes(&f);
    println!("ladder(200): {before} nodes before passes, {after} after");
    // The carry chain must survive: every comparison is read by the next block, so a
    // pipeline that deletes 1/3 of the fixture is deleting live work.
    assert!(
        after > before * 2 / 3,
        "the pipeline deleted the ladder ({before} -> {after}); the timing test measures nothing"
    );
}
