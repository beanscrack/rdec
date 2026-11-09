//! Per-pass SSA isolation on HAND-BUILT IR - no clang, no corpus, no compiler
//! version in the loop. These fixtures bisect the pass pipeline itself: a
//! valid-SSA input, fed to the real pipeline, where a pipeline transform is
//! asked to create the CI defect shape (defs a live edge no longer supports).
//!
//! The rule these pins: **a pass that breaks SSA is a bug in that pass.**
//! `passes::clean_isolated` verifies SSA after every pass; a breakage a later
//! pass REPAIRS is a transient (the pipeline's round-boundary contract, which
//! the corpus floors measure) and stays silent; a breakage that survives to
//! the end lands in `Isolation::broke` naming the pass (1-based index + the
//! name from `PRE_EMIT_PASSES`). `rdec c` prints that as a
//! `pass-broke-ssa record:` line naming function AND pass.
//!
//! Red-first history (measured on branch `ssa-bisect`, base `6523cdc`): with
//! `simplify_branches` at base, BOTH arm-retirement fixtures here are RED,
//! printing e.g.
//!
//! ```text
//! pass-broke-ssa record: fixture :: f: after pass 5 'simplify_branches': bb2: %4 is defined in unreachable bb2
//! ```
//!
//! and stay green after the fix (the fold retires the dead arm: drops the
//! phi pairs keyed on the removed edge and clears blocks that lose their
//! last predecessor). `dead_arm_fold_keeps_ssa` is the red-first pin for the
//! fix; `the_isolation_tool_verifies_every_pass` is the anti-vacuity pin for
//! the tool (a driver iterating an empty table verifies nothing).

use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{BinOp, Function, IntCmp, Op, Terminator};
use rdec_core::passes::{clean_isolated, Isolation, PRE_EMIT_PASSES};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

/// One bit per pass in the table: every pass ran and its post-state was
/// verified. 7 passes today, computed - never a parallel hardcoded 7.
const ALL_VERIFIED: u32 = (1u32 << PRE_EMIT_PASSES.len()) - 1;

// The anti-vacuity floor the bitmask cannot express alone: a table emptied to
// `[]` makes the mask `0` and the mask assertion vacuously true. The clean
// contract (fuse, prune, fold, prop, simplify, dse, dce) is SEVEN passes;
// shrinking it below that is a pipeline change, and the pipeline change must
// be a decision, not a sabotage accident. Compile error is the loudest form.
const _: () = assert!(PRE_EMIT_PASSES.len() >= 7);

/// Print a `Isolation` exactly as the `c` gate prints it, so the RED text of
/// these fixtures quotes the production record line, not a test-local format.
fn report(b: &str, iso: &Isolation) -> String {
    match &iso.broke {
        Some(k) => format!(
            "pass-broke-ssa record: {b}: after pass {} '{}': {}",
            k.index,
            k.name,
            k.errors.join("; ")
        ),
        None => "pass isolation: no pass broke SSA".to_string(),
    }
}

fn no_break(iso: &Isolation, fixture: &str) {
    let line = report("fixture :: f", iso);
    assert!(
        iso.broke.is_none(),
        "{fixture}: a pass broke SSA and the pipeline could not repair it. gate line: {line}"
    );
}

fn verifies(f: &Function, fixture: &str) {
    if let Err(errs) = verify_ssa(f) {
        panic!(
            "{fixture}: post-pipeline IR must verify, got: {}",
            errs.join("; ")
        );
    }
}

/// The anti-vacuity pin for the TOOL: on a function nothing breaks, the
/// isolation run must have VERIFIED EVERY PASS IN THE TABLE, and named none.
/// A driver that iterates an empty table, or skips a pass, prints fewer bits,
/// and this reddens. `rdec c --corpus` prints the same count as
/// `pass isolation: K passes verified on N emitted functions`, pinned against
/// `PRE_EMIT_PASSES.len()` by `crates/cli/tests/pass_isolation.rs`.
#[test]
fn the_isolation_tool_verifies_every_pass() {
    // Straight-line and valid: p0 + 1, returned.
    let mut f = Function::new("f", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0x10);
    let one = f.emit(e, Op::Const(1), Ty::i64(), 0x10);
    let s = f.emit(e, Op::Bin(BinOp::Add, p0, one), Ty::i64(), 0x10);
    f.params.push(p0);
    f.block_mut(e).term = Terminator::Return(s);
    verify_ssa(&f).expect("fixture is valid IR");

    let iso = clean_isolated(&mut f);
    assert_eq!(
        iso.verified, ALL_VERIFIED,
        "the isolation driver verified {} of {} passes (bitmask {:#b}); every pass in PRE_EMIT_PASSES must run AND be verified",
        iso.verified.count_ones(),
        PRE_EMIT_PASSES.len(),
        iso.verified
    );
    assert!(
        iso.broke.is_none(),
        "nothing may break a straight-line fn: {}",
        report("fixture :: f", &iso)
    );
    verify_ssa(&f).expect("post-pipeline IR verifies");
    match run(&f, &[41]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v as i64, 42),
        other => panic!("fixture must answer, got {other:?}"),
    }
}

/// HAND-BUILT DEFECT SHAPE, red-first for the `simplify_branches` fix: a
/// constant-condition branch (the shape flag fusion + const folding leave on
/// real corpus functions - `if (constant)` after inlining), whose DEAD arm is
/// a join consuming an entry-local value through a phi pair. The fold turns
/// `bb0` into `jmp bb1`; `bb2` loses its last predecessor.
///
/// Input (valid SSA - every use dominated, the pair (%x, bb0) available on
/// its edge):
///
/// ```text
/// bb0: %p = param; %c = const 1; %x = add %p, %p; br %c -> bb1 | bb2
/// bb1: return %p
/// bb2: %q = phi [%x, bb0]; %r = mul %q, %q; return %r
/// ```
///
/// Pre-fix, `simplify_branches` folds the branch and RETIRES NOTHING: `bb2`
/// keeps live defs in a block that is now unreachable, and the per-pass
/// verifier names it:
///
/// ```text
/// pass-broke-ssa record: fixture :: f: after pass 5 'simplify_branches': bb2: %4 is defined in unreachable bb2
/// ```
///
/// Nothing in the seven-pass pipeline repairs that shape, so the breakage
/// survives the fixpoint and the isolation report names the pass (the repair
/// probe runs passes 6..7 and the final verify still fails). Post-fix, the
/// fold drops the phi pairs keyed on the removed edge and clears the arm that
/// lost its last predecessor, the answer is the taken arm's, and the run is
/// silent.
#[test]
fn dead_arm_fold_keeps_ssa() {
    let mut f = Function::new("f", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0x10);
    let one = f.emit(e, Op::Const(1), Ty::i64(), 0x10);
    let big = f.emit(e, Op::Const(1u64 << 40), Ty::i64(), 0x10);
    // Bool-typed comparison of two constants: `const_fold` turns it into
    // `Const(1) [Bool]` in the same round, which is exactly what
    // `simplify_branches` folds the branch on. An integer branch condition
    // would be a HARD verifier error (int branch condition) - the fixture
    // must start valid for the pass to be the one that breaks it.
    let c = f.emit(e, Op::Cmp(IntCmp::Ult, one, big), Ty::Bool, 0x10);
    let x = f.emit(e, Op::Bin(BinOp::Add, p0, p0), Ty::i64(), 0x10);
    f.params.push(p0);
    let bb1 = f.new_block(0x20);
    let bb2 = f.new_block(0x30);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: bb1,
        else_bb: bb2,
    };
    f.block_mut(bb1).term = Terminator::Return(p0);
    let q = f.emit(bb2, Op::Phi(Box::new([(x, e)])), Ty::i64(), 0x30);
    let r = f.emit(bb2, Op::Bin(BinOp::Mul, q, q), Ty::i64(), 0x30);
    f.block_mut(bb2).term = Terminator::Return(r);
    verify_ssa(&f).expect("fixture is valid IR (checked by hand: see doc comment)");

    let iso = clean_isolated(&mut f);
    no_break(&iso, "dead-arm fold");
    verifies(&f, "dead-arm fold");
    assert_eq!(
        iso.verified, ALL_VERIFIED,
        "every pass must run and be verified (bitmask {:#b})",
        iso.verified
    );
    // The answer is the taken arm's, for every vector.
    for v in [0i64, 7, -3] {
        match run(&f, &[v as u64]).outcome {
            Outcome::Ret(Val::Int(a)) => assert_eq!(a as i64, v, "vector {v}"),
            other => panic!("vector {v}: must answer, got {other:?}"),
        }
    }
}

/// The SAFE fold, pinned so the fix cannot over-retire: a `then == else`
/// branch (both arms name the same join) whose fold removes no edge.
/// `Cfg::build` records predecessors DEDUPLICATED per block, so a branch
/// naming one target twice is ONE edge, the phi's pair list is keyed once,
/// and folding to `jmp` retires nothing. This fixture is green before AND
/// after the fix - its job is to prove the fix does not start retiring
/// pairs/blocks on folds that remove no edge (the over-retirement twin of the
/// bug in `dead_arm_fold_keeps_ssa`).
#[test]
fn both_arms_fold_keeps_ssa() {
    let mut f = Function::new("f", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0x10);
    let one = f.emit(e, Op::Const(1), Ty::i64(), 0x10);
    let x = f.emit(e, Op::Bin(BinOp::Add, p0, p0), Ty::i64(), 0x10);
    let z = f.emit(e, Op::Bin(BinOp::Add, x, one), Ty::i64(), 0x10);
    f.params.push(p0);
    let bb2 = f.new_block(0x20);
    // Branch on a NON-constant so only the `then == else` rule can fold it.
    let big = f.emit(e, Op::Const(1u64 << 40), Ty::i64(), 0x10);
    let cond = f.emit(e, Op::Cmp(IntCmp::Ult, p0, big), Ty::Bool, 0x10);
    f.block_mut(e).term = Terminator::Branch {
        cond,
        then_bb: bb2,
        else_bb: bb2,
    };
    let q = f.emit(bb2, Op::Phi(Box::new([(x, e)])), Ty::i64(), 0x20);
    let w = f.emit(bb2, Op::Bin(BinOp::Add, q, z), Ty::i64(), 0x20);
    f.block_mut(bb2).term = Terminator::Return(w);
    verify_ssa(&f).expect("fixture is valid IR");

    let iso = clean_isolated(&mut f);
    no_break(&iso, "both-arms fold");
    verifies(&f, "both-arms fold");
    assert_eq!(
        iso.verified, ALL_VERIFIED,
        "every pass must run and be verified (bitmask {:#b})",
        iso.verified
    );
    // f(p0) = (p0+p0) + ((p0+p0) + 1) on the single surviving path.
    for v in [0i64, 7, 1000] {
        match run(&f, &[v as u64]).outcome {
            Outcome::Ret(Val::Int(a)) => assert_eq!(a as i64, 2 * v + (2 * v + 1), "vector {v}"),
            other => panic!("vector {v}: must answer, got {other:?}"),
        }
    }
}

/// The tool must name the pass, not just the victim, when a breakage is
/// PERMANENT: feed the post-fold defect shape DIRECTLY (hand-built IR that is
/// exactly what the pre-fix fold leaves behind) through the pipeline's own
/// table via a forcing driver, and assert the record carries the pass index
/// and the table's own name. This is the compiler-independent proof that
/// `pass-broke-ssa record: <fn>: after pass <N> '<name>': <errors>` fires on
/// the defect class the CI run of 2026-11-08 showed. It stays green with the
/// fix, because it feeds the BROKEN shape, not the fold.
#[test]
fn a_permanent_break_is_named_with_index_and_table_name() {
    // What `simplify_branches` pre-fix left for `locals_addressed`-class
    // functions: a block that is no longer reachable, still defining values.
    // bb0: param, jmp bb1. bb1: return %p. bb2 (preds: NONE): defines %x and
    // reads it - a def in an unreachable block is a hard verifier error the
    // pipeline cannot repair.
    let mut f = Function::new("f", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::i64(), 0x10);
    f.params.push(p0);
    let bb1 = f.new_block(0x20);
    f.block_mut(e).term = Terminator::Jump(bb1);
    f.block_mut(bb1).term = Terminator::Return(p0);
    let bb2 = f.new_block(0x30);
    let x = f.emit(bb2, Op::Bin(BinOp::Add, p0, p0), Ty::i64(), 0x30);
    f.block_mut(bb2).term = Terminator::Return(x);
    assert!(verify_ssa(&f).is_err(), "the fed shape must be invalid IR to begin with: a def in an unreachable block is exactly the surviving-breakage class");

    // `clean_isolated` verifies before any pass runs its repair schedule...
    // the shape is broken at entry, so the pipeline must report, not hang,
    // and the report's name must come from the table it iterated. The first
    // pass whose post-state is verified still carries the fed defect:
    let iso = clean_isolated(&mut f);
    let Some(b) = &iso.broke else {
        // A pass REPAIRED the fed shape (e.g. DCE clearing the dead block).
        // That is the repair-probe contract, and it is silent by design -
        // prove the naming leg on a shape no pass may repair.
        let line = report("fixture :: f", &iso);
        eprintln!("(fed shape was repaired by the pipeline: {line})");
        // `bb2: %2 is defined in unreachable` is repaired by DCE only if the
        // def is pure AND unread - here `%x` is returned by bb2, and bb2 is
        // unreachable, so this branch must NOT be taken.
        unreachable!("the fed def-in-unreachable must survive: {line}");
    };
    let line = report("fixture :: f", &iso);
    println!("{line}");
    assert!(
        b.index >= 1 && b.index <= PRE_EMIT_PASSES.len(),
        "index {0} is outside the table",
        b.index
    );
    assert!(
        PRE_EMIT_PASSES.iter().any(|p| p.name == b.name),
        "the record names '{}' which is not in the pass table it iterated",
        b.name
    );
    assert!(
        line.contains("pass-broke-ssa record:") && line.contains("after pass"),
        "the record line must be the gate's format: {line}"
    );
}
