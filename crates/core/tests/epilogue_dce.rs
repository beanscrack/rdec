//! The M6 epilogue lane: what the middle end may do to a **pure return tail**.
//!
//! The gap this file pins is the census line `rdec c --corpus` prints at base
//! 47b9076:
//!
//! ```text
//! control flow: 37/96 functions structured | ... | 195 blocks left as goto
//!   (reasons: 29 early return refused: target computes, ...)
//! ```
//!
//! The printer's rule (`structure.rs::tail_of`) is that a jump site may spell
//! `return` only when the target block computes NOTHING: every instruction a phi,
//! the terminator a Return, and the value readable at the site. The 29 sites are
//! measured (a census over every emitted corpus function, this lane, base
//! 47b9076) to be these shapes and NO OTHER:
//!
//! ```text
//!   8  Const                                       (mov $5,%eax; ret)
//!   5  Const, Load(uint32_t), Cast(ZExt)           (mov -0x8(%rbp),%eax; ret)
//!   1  Const, Load, Cast, Cast, Const, Load, Add, Cast
//! ```
//!
//! i.e. the tail block materializes the ANSWER - a constant, or a load of a frame
//! slot plus the zext a 32-bit write carries - and the printer refuses because that
//! work needs a statement and a `return` has no room for one.
//!
//! What is NOT in those 29 blocks, measured: not one RSP/RBP teardown write, and
//! not one `Op::Store`. Frame teardown is *already* deleted by today's DCE, because
//! a register write in this IR is an SSA value and a `Return` names only its value:
//! `mov rsp,rbp` / `pop rbp` lower to pure nodes nobody reads (`Op::is_pure`,
//! ir.rs), so `use_count == 0` deletes them. The test
//! `frame_teardown_on_a_return_only_path_is_already_gone` pins that so it cannot
//! rot, and `a_frame_slot_store_in_a_tail_is_kept_and_says_so` pins the half this
//! lane refuses to do: a STORE is not deletable. `interp.rs` pushes `Effect::Write`
//! for every store and `Exec::diff` compares the ordered effect trace AND final
//! memory, so deleting a store changes the IR's own meaning even though T2's memory
//! axis compares caller buffers, not the callee frame.
//!
//! The transform that does the work is therefore not a deletion: it is
//! `passes::materialize_return_tails`, which moves a pure tail's chain onto the
//! edges that reach it - the end of a predecessor is the same program point as the
//! start of a tail block with nothing in between, so the machine ran the work there
//! too - leaving the tail a pure tail, which the existing M6 rule spells `return`
//! with no change to `structure.rs`. That is what the two "becomes a pure tail"
//! tests are red for at base.
//!
//! The sabotage guards, named at the tests that carry them:
//!  * the value a `Return` reads stays live (`the_value_a_return_reads_is_live_at_the_return`),
//!  * a store in a tail is never moved, copied, or deleted
//!    (`a_frame_slot_store_in_a_tail_is_kept_and_says_so`),
//!  * a tail op that can FAULT is never duplicated onto an arm that did not run it
//!    (`a_faulting_tail_op_is_not_duplicated_onto_an_arm_that_did_not_run_it`),
//!  * a write that a fall-through reads stays (`a_write_the_fall_through_reads_survives`).

use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{BbId, BinOp, CastKind, Function, IntCmp, Op, Terminator, VId, NO_VAL};
use rdec_core::passes;
use rdec_core::structure;
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

const REFUSAL: &str = "early return refused: target computes";

/// The corpus's own frame-slot address: `STACK_BASE` (1<<19) minus 8, the constant
/// the lifter gives an entry-RSP-relative `mov -0x8(%rbp),%eax`.
const FRAME_SLOT: u64 = (1 << 19) - 8;

fn refusals(f: &Function) -> usize {
    structure::structure(f)
        .report
        .reasons
        .get(REFUSAL)
        .copied()
        .unwrap_or(0)
}

fn early_returns(f: &Function) -> usize {
    structure::structure(f).report.early_returns
}

fn non_phi(f: &Function, block: BbId) -> usize {
    f.blocks[block as usize]
        .instrs
        .iter()
        .filter(|i| !matches!(i.op, Op::Phi(_)))
        .count()
}

fn count_op(f: &Function, want: fn(&Op) -> bool) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| want(&i.op))
        .count()
}

/// The corpus's shared-epilogue shape, which is what makes a jump site a decision
/// the printer has to make. A jump whose target has not been printed yet is not a
/// decision at all: the region walks into it, and the readability exchange line
/// counts that as a jump the flat layout absorbed. A jump into a block the walk has
/// ALREADY emitted is a `goto` it has to spell, and that is where
/// `early return refused: target computes` comes from - every one of the corpus's
/// 29 is a jump into a block whose text is already on the page.
///
/// ```text
///   bb0 (entry): switch (p0) { 0 -> E, 1 -> X, 2 -> E }   slot 0 prints E's text
///   bb1 (X):     return p0                                inside the switch; slot 2
///                                                        then jumps BACK to it
///   bb2 (E):     <tail work>; return <value>              the shared epilogue
/// ```
///
/// At base that is one refusal (slot 2's arm), and the machine's answers are
/// `p0 == 0 -> 5`, `p0 == 1 -> p0`, `p0 == 2 -> 5` for the constant tail.
fn shared_tail(name: &str, tail: impl FnOnce(&mut Function, BbId) -> VId) -> Function {
    let mut f = Function::new(name, 0);
    let e = f.entry;
    let x = f.new_block(0x10);
    let t = f.new_block(0x20);
    let p0 = f.emit(e, Op::Param, Ty::u64(), 0x0);
    f.block_mut(e).term = Terminator::Switch {
        index: p0,
        targets: vec![t, x, t],
    };
    f.block_mut(x).term = Terminator::Return(p0);
    let v = tail(&mut f, t);
    f.block_mut(t).term = Terminator::Return(v);
    f.params.extend([p0]);
    f
}

/// The entry's parameter, for tails that need one (`shared_tail` builds it).
fn p0_of(f: &Function) -> VId {
    f.blocks[f.entry as usize].instrs[0].dest
}

/// `mov $5,%eax; ret` in a shared epilogue: the `switch_sparse.O1` shape.
fn const_tail() -> Function {
    shared_tail("const_tail", |f, t| {
        f.emit(t, Op::Const(5), Ty::u64(), 0x10)
    })
}

/// The `locals_addressed.O0` / `switch_dense.O0` shape: the answer is a load of a
/// frame slot widened by the 32-bit-write zext, and that work sits in the return
/// block. The STORE that fed the slot is in the entry, where it stays.
fn frame_load_tail() -> Function {
    shared_tail("frame_load_tail", |f, t| {
        let a = f.emit(f.entry, Op::Const(FRAME_SLOT), Ty::u64(), 0x3);
        let v32 = f.emit(
            f.entry,
            Op::Cast(CastKind::Trunc, Ty::u32(), p0_of(f)),
            Ty::u32(),
            0x4,
        );
        f.emit_void(f.entry, Op::Store(Ty::u32(), a, v32), 0x5);
        let addr = f.emit(t, Op::Const(FRAME_SLOT), Ty::u64(), 0x10);
        let load = f.emit(t, Op::Load(Ty::u32(), addr), Ty::u32(), 0x10);
        f.emit(
            t,
            Op::Cast(CastKind::ZExt, Ty::u64(), load),
            Ty::u64(),
            0x12,
        )
    })
}

/// The store-in-tail negative, in the same shared-epilogue shape.
fn store_in_tail() -> Function {
    shared_tail("store_in_tail", |f, t| {
        let slot = f.emit(t, Op::Const(FRAME_SLOT), Ty::u64(), 0x10);
        let v32 = f.emit(
            t,
            Op::Cast(CastKind::Trunc, Ty::u32(), p0_of(f)),
            Ty::u32(),
            0x11,
        );
        f.emit_void(t, Op::Store(Ty::u32(), slot, v32), 0x12);
        v32
    })
}

/// The faulting-tail negative: one epilogue, one `Switch` predecessor with three
/// arms, and the epilogue divides.
///
/// ```text
///   bb0 (entry): switch (p0) { 0 -> X, 1 -> E, 2 -> E }
///   bb1 (X):     return 0
///   bb2 (E):     return p1 / p0        (p0 is 1 or 2 on both edges into E)
/// ```
///
/// Copying `UDiv(p1, p0)` to the end of the switch block would run it on arm 0 as
/// well, and arm 0 is `p0 == 0`: the machine returns 0 there, the copy raises
/// `Fault::DivZero`. The pass allows the full pure set only on a `Jump` edge - one
/// successor, the machine's own edge - and a cannot-fault subset elsewhere.
fn div_tail() -> Function {
    let mut f = Function::new("div_tail", 0);
    let e = f.entry;
    let x = f.new_block(0x10);
    let t = f.new_block(0x20);
    let p0 = f.emit(e, Op::Param, Ty::u64(), 0x0);
    let p1 = f.emit(e, Op::Param, Ty::u64(), 0x1);
    let z = f.emit(e, Op::Const(0), Ty::u64(), 0x2);
    f.block_mut(e).term = Terminator::Switch {
        index: p0,
        targets: vec![x, t, t],
    };
    f.block_mut(x).term = Terminator::Return(z);
    let q = f.emit(t, Op::Bin(BinOp::UDiv, p1, p0), Ty::u64(), 0x20);
    f.block_mut(t).term = Terminator::Return(q);
    f.params.extend([p0, p1]);
    f
}

// ---------------------------------------------------------------------------
// (A) frame teardown: the premise the lane was handed, measured and pinned
// ---------------------------------------------------------------------------

/// `mov rsp,rbp; pop rbp; ret` as `lift.rs` lowers it: `Copy(rbp)` for the
/// frame-pointer move, and `Load(sp)` + `Add(sp,8)` + a write of the loaded value
/// for the pop. The `Return` names the ANSWER, so every one of those values is
/// unread, and DCE's `use_count == 0 && is_pure` rule deletes them.
///
/// This is the lane's (A) finding, pinned so it cannot rot: the 29 refusals are NOT
/// teardown writes, and the fix is not a liveness rule about RSP/RBP. The census
/// counter is checked here too, so the number it prints is wired to a deletion and
/// not to a wish (two of the five nodes are `Op::Copy`, which `copy_prop` claims
/// before DCE's sweep runs, hence `>= 4` and not `== 5`).
#[test]
fn frame_teardown_on_a_return_only_path_is_already_gone() {
    passes::reset_epilogue_counts();
    let mut f = Function::new("teardown", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::u64(), 0x0);
    let one = f.emit(e, Op::Const(1), Ty::u64(), 0x1);
    let ans = f.emit(e, Op::Bin(BinOp::Add, p0, one), Ty::u64(), 0x2);
    let rbp = f.emit(e, Op::Const(FRAME_SLOT), Ty::u64(), 0x3);
    let t = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(t);
    // mov rsp,rbp
    let sp1 = f.emit(t, Op::Copy(rbp), Ty::u64(), 0x10);
    // pop rbp = load the word, advance the stack, write it to the RBP cell
    let popped = f.emit(t, Op::Load(Ty::u64(), sp1), Ty::u64(), 0x11);
    let eight = f.emit(t, Op::Const(8), Ty::u64(), 0x11);
    f.emit(t, Op::Bin(BinOp::Add, sp1, eight), Ty::u64(), 0x11);
    f.emit(t, Op::Copy(popped), Ty::u64(), 0x11);
    assert_eq!(non_phi(&f, t), 5, "fixture: five teardown nodes");
    f.block_mut(t).term = Terminator::Return(ans);
    f.params.extend([p0]);
    verify_ssa(&f).expect("teardown fixture is valid IR");

    passes::clean(&mut f);
    verify_ssa(&f).expect("the cleaned teardown function is valid IR");
    assert_eq!(
        non_phi(&f, t),
        0,
        "teardown writes survived DCE on a return-only path:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    let counted = passes::epilogue_counts().teardown_writes_deleted;
    assert!(
        counted >= 4,
        "the census's `teardown writes deleted` count is not wired to the deletions \
         it names: the return block was emptied and the counter saw {counted} of 5 \
         nodes (>= 4, since copy_prop claims the two `Op::Copy` nodes first)"
    );
    // The answer is untouched: `p0 + 1`, not a frame value and not 0.
    match run(&f, &[41]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v, 42, "the teardown path lost the answer"),
        other => panic!("expected a value, got {other:?}"),
    }
    // Nothing a store does is involved: this function has no store, and deleting
    // the teardown deleted no memory effect (the interpreter records writes only,
    // so an unread load leaves no trace).
    assert_eq!(count_op(&f, |o| matches!(o, Op::Store(..))), 0);
}

/// The negative of that, and the liveness rule it guards: a write whose value the
/// fall-through READS is live, and DCE must keep it. This is the fixture the
/// sabotage "every register write unread at a Return is deletable" reddens: it is
/// liveness at the *fall-through*, not a per-terminator register rule, that keeps
/// the value.
#[test]
fn a_write_the_fall_through_reads_survives() {
    let mut f = Function::new("fallthrough_reads_sp", 0);
    let e = f.entry;
    let _p0 = f.emit(e, Op::Param, Ty::u64(), 0x0);
    let rsp0 = f.emit(e, Op::Const(FRAME_SLOT), Ty::u64(), 0x1);
    let eight = f.emit(e, Op::Const(8), Ty::u64(), 0x2);
    let b1 = f.new_block(0x10);
    let b2 = f.new_block(0x20);
    f.block_mut(e).term = Terminator::Jump(b1);
    // bb1 writes the RSP cell and falls through; bb2 RETURNS that value, so the
    // write is live at the fall-through and DCE cannot take it.
    let sp1 = f.emit(b1, Op::Bin(BinOp::Add, rsp0, eight), Ty::u64(), 0x10);
    f.block_mut(b1).term = Terminator::Jump(b2);
    f.block_mut(b2).term = Terminator::Return(sp1);
    f.params.extend([_p0]);
    verify_ssa(&f).expect("fixture is valid IR");
    passes::clean(&mut f);
    verify_ssa(&f).expect("cleaned fixture is valid IR");
    match run(&f, &[7]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(
            v,
            FRAME_SLOT + 8,
            "DCE deleted a register write the fall-through path reads: the returned \
             RSP cell must be the written value, not the entry one"
        ),
        other => panic!("expected a value, got {other:?}"),
    }
}

/// The over-deletion guard the brief asks for: the value a `Return` reads is live
/// at the Return. Delete terminator operands from DCE's use counts (the roadmap's
/// B1 bug) and this fixture goes red: the returned value is deleted and the
/// function answers from nothing.
#[test]
fn the_value_a_return_reads_is_live_at_the_return() {
    let mut f = Function::new("return_is_a_use", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::u64(), 0);
    let p1 = f.emit(e, Op::Param, Ty::u64(), 1);
    let s = f.emit(e, Op::Bin(BinOp::Mul, p0, p1), Ty::u64(), 2);
    f.block_mut(e).term = Terminator::Return(s);
    f.params.extend([p0, p1]);
    verify_ssa(&f).expect("fixture is valid IR");
    passes::dead_code_elimination(&mut f);
    assert_eq!(
        count_op(&f, |o| matches!(o, Op::Bin(..))),
        1,
        "DCE deleted the expression the Return reads (B1: terminator operands are \
         uses):\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    match run(&f, &[6, 7]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v, 42),
        other => panic!("expected a value, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The win: a pure return tail moves onto the edges that reach it
// ---------------------------------------------------------------------------

/// `mov $5,%eax; ret` in a shared epilogue: the tail block's only work is a
/// constant. After the pass the target computes nothing, and the M6 printer turns
/// the jump site into `return` with no change to `structure.rs`.
///
/// RED at base: the target keeps its `Const` node, so `tail_of` refuses (`early
/// return refused: target computes`) and the site stays a `goto`.
#[test]
fn a_constant_return_tail_becomes_a_pure_tail_the_printer_spells_return() {
    let mut f = const_tail();
    verify_ssa(&f).expect("fixture is valid IR");
    assert_eq!(
        refusals(&f),
        2,
        "the fixture must show the refusals this lane is about (one per switch arm \
         into the emitted epilogue):\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    let a0 = run(&f, &[0]).outcome;
    let a1 = run(&f, &[1]).outcome;
    let a2 = run(&f, &[2]).outcome;
    assert_eq!(
        a0,
        Outcome::Ret(Val::Int(5)),
        "fixture: arm 0 lands in the tail"
    );
    assert_eq!(a1, Outcome::Ret(Val::Int(1)), "fixture: arm 1 returns p0");
    assert_eq!(
        a2,
        Outcome::Ret(Val::Int(5)),
        "fixture: arm 2 jumps to the emitted tail"
    );

    passes::clean(&mut f);
    verify_ssa(&f).expect("the materialized function is valid IR");
    let t = 2; // the epilogue: `shared_tail`'s last `new_block`
    assert_eq!(
        refusals(&f),
        0,
        "the constant tail still computes, so the site is still a goto:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(
        non_phi(&f, t),
        0,
        "the return block must compute nothing (phis are allowed):\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(
        early_returns(&f) >= 1,
        "the printer did not turn the jump sites into returns:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(run(&f, &[0]).outcome, a0, "arm 0's answer moved");
    assert_eq!(run(&f, &[1]).outcome, a1, "arm 1's answer moved");
    assert_eq!(
        run(&f, &[2]).outcome,
        a2,
        "the site edge's answer moved: materializing the tail must not"
    );
    rdec_core::emit_c::emit_c(&f)
        .unwrap_or_else(|r| panic!("the materialized tail must emit C: {r:?}"));
}

/// The `switch_dense` / `locals_addressed` shape: `mov -0x8(%rbp),%eax; ret`. The
/// LOAD is pure in this IR (`Op::is_pure`), and a predecessor's terminator and a
/// block with no instructions in between are the same program point - so the load
/// runs at the end of the predecessor, exactly where the machine ran it. The frame
/// STORE that feeds the slot stays where it is, so the effect trace and the final
/// memory are byte-identical.
#[test]
fn a_frame_slot_load_tail_moves_to_the_edge_and_the_store_stays() {
    let mut f = frame_load_tail();
    verify_ssa(&f).expect("fixture is valid IR");
    assert_eq!(refusals(&f), 2, "the fixture must show the refusals");
    let before = run(&f, &[2]);
    assert_eq!(
        before.outcome,
        Outcome::Ret(Val::Int(2)),
        "fixture: the site edge answers from the frame slot, widened by the zext"
    );
    assert_eq!(
        before.effects.len(),
        1,
        "fixture: the frame store is one recorded Write effect"
    );
    let before_arm0 = run(&f, &[0]);
    let before_arm1 = run(&f, &[1]);

    passes::clean(&mut f);
    verify_ssa(&f).expect("the materialized function is valid IR");
    assert_eq!(
        refusals(&f),
        0,
        "the frame-slot load tail still computes:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(
        count_op(&f, |o| matches!(o, Op::Store(..))),
        1,
        "the frame store was moved or deleted: a store is NOT deletable - interp.rs \
         pushes Effect::Write for it and Exec::diff compares final memory, so this \
         lane keeps every store exactly where it is:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // TV's own comparison, not a hand-rolled one: `Exec::diff` is the repo's
    // outcome/effect-trace/memory oracle (it normalizes the memory snapshot, whose
    // order is not a claim, and keeps the effect trace ordered, where it is).
    let after = run(&f, &[2]);
    assert_eq!(
        before.diff(&after),
        Vec::<String>::new(),
        "the site edge's execution changed"
    );
    assert_eq!(
        before_arm0.diff(&run(&f, &[0])),
        Vec::<String>::new(),
        "the switch's first edge changed its execution"
    );
    assert_eq!(
        before_arm1.diff(&run(&f, &[1])),
        Vec::<String>::new(),
        "the exit arm changed its execution"
    );
    // The width-carrying work survived the move: the u32 load and the zext are
    // still in the function, now at the end of the edges that reach the tail.
    assert_eq!(
        count_op(&f, |o| matches!(o, Op::Load(t, _) if *t == Ty::u32())),
        1,
        "the frame-slot load did not survive the move:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(
        count_op(
            &f,
            |o| matches!(o, Op::Cast(CastKind::ZExt, t, _) if *t == Ty::u64())
        ),
        1,
        "the zext the 32-bit write carries did not survive the move:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(early_returns(&f) >= 1);
    rdec_core::emit_c::emit_c(&f)
        .unwrap_or_else(|r| panic!("the materialized tail must emit C: {r:?}"));
}

/// A tail that STORES is not a pure tail: the store needs a statement, a `return`
/// has no room for one, and the store is not deletable. This is the (B) decision,
/// pinned: `29 -> fewer`, not `29 -> 0 at any cost`.
///
/// Sabotage this fixture carries: make `Op::Store` tail-replicable (movable, and so
/// deletable) and it goes red - the store leaves the tail, the effect trace loses
/// its `Write`, and the memory axis T2 compares (the caller-buffer vectors in
/// `corpus/obj/store_to_buffer.*` and `copy_words.*`, gate tuple
/// `memory: 288/288/0`) stops being the machine's.
#[test]
fn a_frame_slot_store_in_a_tail_is_kept_and_says_so() {
    let mut f = store_in_tail();
    verify_ssa(&f).expect("fixture is valid IR");
    assert_eq!(
        refusals(&f),
        2,
        "fixture: a tail that stores is a refusal per arm into it:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );

    passes::clean(&mut f);
    verify_ssa(&f).expect("cleaned fixture is valid IR");
    assert_eq!(
        count_op(&f, |o| matches!(o, Op::Store(..))),
        1,
        "a store in a return tail was deleted: the callee frame is memory in this \
         IR, and the interpreter's effect trace plus final memory make that write \
         observable - it stays:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(
        refusals(&f) >= 2,
        "a tail that stores must stay a refusal (the printer needs a statement for \
         it), not be silently spelled as a return:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    let ex = run(&f, &[2]);
    assert_eq!(ex.outcome, Outcome::Ret(Val::Int(2)));
    assert_eq!(ex.effects.len(), 1, "the store's Write effect vanished");
}

/// A tail op that can FAULT (`UDiv` on zero -> `Fault::DivZero`, `interp.rs`) may
/// not be copied onto the sibling arm of the switch that reaches the tail: the copy
/// runs on the arm that never ran the tail, turning a returning execution into a
/// faulting one.
///
/// Sabotage this fixture carries: let a division be tail-replicable on a
/// multi-successor predecessor and it goes red - at `p0 == 0` the machine returns
/// 0, and the copied `UDiv(p1, 0)` faults.
#[test]
fn a_faulting_tail_op_is_not_duplicated_onto_an_arm_that_did_not_run_it() {
    let mut f = div_tail();
    verify_ssa(&f).expect("fixture is valid IR");
    let side = run(&f, &[0, 5]).outcome;
    assert_eq!(
        side,
        Outcome::Ret(Val::Int(0)),
        "fixture: arm 0 never reaches the dividing tail"
    );
    assert_eq!(
        run(&f, &[1, 10]).outcome,
        Outcome::Ret(Val::Int(10)),
        "fixture: arm 1 divides, p1 / 1"
    );
    assert_eq!(
        refusals(&f),
        2,
        "fixture: the dividing tail is a refusal per arm into it:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );

    passes::clean(&mut f);
    verify_ssa(&f).expect("cleaned fixture is valid IR");
    assert_eq!(
        run(&f, &[0, 5]).outcome,
        side,
        "a dividing tail op was copied onto the arm that never ran the tail: the \
         function now faults where the machine code returns 0"
    );
    assert_eq!(
        run(&f, &[1, 10]).outcome,
        Outcome::Ret(Val::Int(10)),
        "the dividing path must keep answering"
    );
    assert!(
        refusals(&f) >= 2,
        "a tail whose work faults on an arm the machine never ran must stay a \
         refusal, not be spelled as a return"
    );
}

/// TV, the oracle the repo owns: rewriting the tail must be invisible to outcome,
/// effect trace and final memory on sampled states, through the pass label the
/// pipeline reports.
#[test]
fn materializing_a_return_tail_preserves_meaning() {
    for (name, mut f) in [
        ("const_tail", const_tail()),
        ("frame_load_tail", frame_load_tail()),
        ("store_in_tail", store_in_tail()),
        ("div_tail", div_tail()),
    ] {
        let ss = rdec_core::tv::states(f.params.len(), 64, 0x5eed);
        match rdec_core::tv::run_validated(
            &mut f,
            ("dead_code_elimination", passes::dead_code_elimination),
            &ss,
            10_000,
        ) {
            Ok(_) => {}
            Err(fails) => panic!(
                "{name}: DCE changed meaning: {:?}",
                fails.first().map(|x| (&x.pass, &x.inputs, &x.diffs))
            ),
        }
    }
}

/// The census counter: the win is measured, not assumed. A function with no return
/// tail contributes nothing, so a counter that counts the fixture set rather than
/// the work is visible here.
#[test]
fn the_census_names_the_work_it_did() {
    passes::reset_epilogue_counts();
    let mut plain = Function::new("plain", 0);
    let e = plain.entry;
    let p0 = plain.emit(e, Op::Param, Ty::u64(), 0);
    plain.block_mut(e).term = Terminator::Return(p0);
    plain.params.extend([p0]);
    passes::clean(&mut plain);
    let empty = passes::epilogue_counts();
    assert_eq!(
        (empty.tails_materialized, empty.edges_materialized),
        (0, 0),
        "a function with no return tail must not move the counter"
    );

    let mut f = frame_load_tail();
    passes::clean(&mut f);
    let after = passes::epilogue_counts();
    assert!(
        after.tails_materialized > 0 && after.edges_materialized >= 1,
        "the frame-slot tail moved the counter to {} tails on {} edges: the win is \
         not measured",
        after.tails_materialized,
        after.edges_materialized
    );
    let line = passes::epilogue_line();
    assert!(
        line.contains("return tails materialized")
            && line.contains("frame stores deleted")
            && line.contains("teardown writes deleted"),
        "the census line must name the counts: {line}"
    );
    assert!(
        line.contains("frame stores deleted: 0"),
        "a pass in this lane that deleted a frame store would have to change this \
         assertion; the measured value is 0 and the census prints it: {line}"
    );
    passes::reset_epilogue_counts();
}

/// NO_VAL and SSA guards: a `return;` (void) tail with pure work in it must not
/// gain a phi, and a block nothing reaches must not gain one either.
#[test]
fn a_void_tail_and_an_unreachable_pred_are_left_alone() {
    let mut f = Function::new("void_tail", 0);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, Ty::u64(), 0);
    let z = f.emit(e, Op::Const(0), Ty::u64(), 1);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p0, z), Ty::Bool, 2);
    let t = f.new_block(0x10);
    let x = f.new_block(0x20);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: t,
        else_bb: x,
    };
    let _k = f.emit(t, Op::Bin(BinOp::Add, p0, z), Ty::u64(), 0x10);
    f.block_mut(t).term = Terminator::Return(NO_VAL);
    f.block_mut(x).term = Terminator::Return(z);
    f.params.extend([p0]);
    verify_ssa(&f).expect("fixture is valid IR");
    passes::clean(&mut f);
    verify_ssa(&f).expect("a void tail gained an illegal phi");

    // An unreachable block cannot donate a predecessor: `Cfg::preds` drops
    // unreachable predecessors, so the tail below keeps its own shape and no phi is
    // invented that would name a block `verify_ssa` calls unreachable.
    let mut g = Function::new("unreachable_tail", 0);
    let ge = g.entry;
    let q0 = g.emit(ge, Op::Param, Ty::u64(), 0);
    let dead = g.new_block(0x10);
    g.block_mut(dead).term = Terminator::Unreachable;
    g.block_mut(ge).term = Terminator::Return(q0);
    g.params.extend([q0]);
    verify_ssa(&g).expect("unreachable block fixture is valid IR");
    passes::clean(&mut g);
    verify_ssa(&g).expect("the pass invented a phi from an unreachable predecessor");
}
// ---------------------------------------------------------------------------
// The cascade: one DCE call runs a reverse-order deletion chain to the end
// ---------------------------------------------------------------------------

/// A chain of `Jump` blocks in which block `i` computes
///
/// ```text
///   v_i  = Add(v_{i-1}, 1)            live: feeds v_{i+1}, and the last one is returned
///   cb_i = ZExt(c_{i-1})              dead: its only reader is c_i
///   c_i  = Cmp(Eq, v_i, cb_i)         dead: its only reader is cb_{i+1}, ...
/// ```
///
/// and ends in `Jump`. Every `c_i` is unread at the terminator, so the whole
/// `cb`/`c` chain is dead - and it dies from the LAST block backwards, against the
/// block order a sweep walks: block `n`'s `c` dies, which frees block `n`'s `cb`,
/// which frees block `n-1`'s `c`, and so on up the chain.
///
/// This is the shape the sweep used to lose. A sweep that visits each node once in
/// block order advances the cascade one level per sweep, and the fixpoint around it
/// pays one O(nodes) sweep per level - quadratic. It is latent at base and it is
/// not academic: this lane's tail materialization produced exactly this shape on
/// `pipeline_scaling`'s ladder, and the census measured DCE quadratic in the block
/// count (debug, this machine, 2026-11-06, block-order sweep: 338 ms / 1.40 s /
/// 7.22 s at 500 / 1k / 2k blocks, `clean` 31.5 s at 4k, x3.9 per x2; the same
/// fixtures with the worklist sweep: 1.4 ms / 2.7 ms / 5.3 ms, linear).
///
/// The assertion is functional, not a stopwatch: ONE `dead_code_elimination` call
/// must leave the live `v` chain and nothing else. A sweep that cascades one node
/// per sweep leaves most of the dead chain behind and goes red here.
#[test]
fn one_dce_call_runs_the_whole_reverse_cascade() {
    fn chain(n: usize) -> Function {
        let mut f = Function::new("cascade", 0);
        let e = f.entry;
        let p0 = f.emit(e, Op::Param, Ty::u64(), 0);
        let one = f.emit(e, Op::Const(1), Ty::u64(), 1);
        // A comparison in the entry so block 1 has a `c_0` to read. Its result is
        // read by block 1's `cb`, exactly like the rest of the chain.
        let c0 = f.emit(e, Op::Cmp(IntCmp::Eq, p0, one), Ty::Bool, 2);
        let mut prev_v = p0;
        let mut prev_c = c0;
        let mut cur = e;
        for i in 0..n {
            let b = f.new_block(0x10 + 0x10 * i as u64);
            f.block_mut(cur).term = Terminator::Jump(b);
            cur = b;
            let v = f.emit(b, Op::Bin(BinOp::Add, prev_v, one), Ty::u64(), 0x20);
            let cb = f.emit(
                b,
                Op::Cast(CastKind::ZExt, Ty::u64(), prev_c),
                Ty::u64(),
                0x21,
            );
            let c = f.emit(b, Op::Cmp(IntCmp::Eq, v, cb), Ty::Bool, 0x22);
            prev_v = v;
            prev_c = c;
        }
        f.block_mut(cur).term = Terminator::Return(prev_v);
        f.params.extend([p0]);
        f
    }
    let n = 4000;
    let mut f = chain(n);
    verify_ssa(&f).expect("cascade fixture is valid IR");
    let live = |f: &Function| f.blocks.iter().map(|b| b.instrs.len()).sum::<usize>();
    assert_eq!(live(&f), 3 + 3 * n, "fixture: entry 3 + {n} blocks of 3");

    let t = std::time::Instant::now();
    passes::dead_code_elimination(&mut f);
    let el = t.elapsed();
    println!("cascade: {n} blocks / {} nodes -> DCE {:?}", 3 + 3 * n, el);
    verify_ssa(&f).expect("the cascaded function is valid IR");
    // Live after the sweep: the entry's 3 nodes and the `n` adds. Every `cb`/`c`
    // pair is dead, and the last block's `c`/`cb` are dead first.
    assert_eq!(
        live(&f),
        3 + n,
        "one dead_code_elimination call left {} nodes; the live set is {} (the dead          cb/c chain dies from the last block backwards, against the block order). A          sweep that cascades one node per sweep leaves the chain behind and makes          DCE quadratic in the block count - see the doc comment.",
        live(&f),
        3 + n
    );
    assert_eq!(
        run(&f, &[41]).outcome,
        Outcome::Ret(Val::Int(41 + n as u64)),
        "the cascade deleted a live node: the chain adds 1 {n} times to p0"
    );
    // The timing bound is for the half of this the end state cannot see: a sweep
    // that cascades one level per sweep reaches the SAME end state (the fixpoint
    // around it runs `blocks` sweeps of O(nodes) work), so the functional assert
    // above passes either way and only this bound separates the two implementations.
    // Measured on this machine, debug, 2026-11-06, this fixture: the worklist sweep
    // takes 12.8 ms; the same sweep with its in-sweep cascade removed takes 23.0 s
    // (1800x). The 3 s bound sits between them - 230x over the measured value, 7.7x
    // under the regression, so it is not a stopwatch the machine can trip by being
    // busy, and it is a behaviour change the machine cannot hide.
    assert!(
        el < std::time::Duration::from_secs(3),
        "the {n}-block reverse cascade took {el:?}: DCE is quadratic in the block \
         count again (a sweep that deletes one level per sweep, run to a fixpoint, \
         is that shape - see the doc comment)"
    );
}
