//! The interpreter's step budget must bound CONTROL FLOW, not only instructions.
//!
//! `Exec::steps` charges one step per block ENTRY plus one per INSTRUCTION, so
//! `budget` is a bound on every run the machine can be handed. It used to charge
//! instructions only, which is not a bound at all: a block with no instructions
//! that transfers control consumes nothing, and a cycle of such blocks runs
//! forever with `self.steps` frozen at zero - the check `steps > budget` can
//! never fire.
//!
//! The shape is lifter-produced, not exotic. A hostile `.text` of random bytes
//! lifts to
//!
//! ```text
//! bb0(at 0x0):  ... 16 instructions ...   br bb1
//! bb1(at 0x24):                            br bb1
//! ```
//!
//! (`bb1` is a block the plan created whose own instruction was never lifted -
//! the `jmp` arrived as a terminator and nothing else, exactly the `bytes_past_
//! ended_block` shape the lifter counts). The witness is the fuzzer's seed 1,
//! iteration 1992: a 270-byte object, `.text` = 152 random bytes, pinned in
//! `crates/arch-x86/tests/fuzz_findings.rs`. Pre-fix, `rdec fuzz` on that ONE
//! object ran for minutes and never finished, and `rdec t2` / TV / `rdec c` are
//! all reachable from the same call (`run_budget`): one 4 KB object is a DoS on
//! the tool's own users. A per-input wall-clock ceiling in a driver cannot
//! interrupt a running input, so the bound has to be in the algorithm.
//!
//! RED DIRECTIONS (both executed, table in the commit message):
//!
//! - Delete the per-block-entry charge in `exec_bb` (keep the instruction
//!   charge, which is what HEAD had): `the_step_account_is_blocks_plus_instructions`
//!   fails in milliseconds with `left: 3, right: 5` - the run charges
//!   entry(bb0) + Const + Const and nothing for the transfer into bb1 - and
//!   `an_empty_self_loop_block_spends_budget` never returns at all, which is the
//!   65-second production hang in test form (measured: `timeout 90 cargo test`
//!   exits 124).
//! - Charge per block entry one transfer late: the step account moves by one
//!   and `the_step_account_is_blocks_plus_instructions` is the test that says so.

use rdec_core::interp::{run_budget, Outcome, Val};
use rdec_core::ir::{BinOp, Function, Instr, IntCmp, Op, Terminator};
use rdec_core::ty::Ty;

/// `bb0: %k = const 7; br bb1` / `bb1: br bb1` - the second block is EMPTY and
/// jumps to itself, which is the shape a hostile `.text` lifts to.
fn empty_self_loop() -> Function {
    let mut f = Function::new("spin", 0);
    let e = f.entry;
    let k = f.emit(e, Op::Const(7), Ty::u64(), 0);
    let head = f.new_block(0x24);
    f.block_mut(e).term = Terminator::Jump(head);
    // No instruction in `head` at all: the block exists, transfers control, and
    // does nothing. Every lap of it must cost a step.
    f.block_mut(head).term = Terminator::Jump(head);
    let _ = k;
    f
}

/// Two straight blocks, one instruction each: `steps` must be exactly 2 blocks
/// + 2 instructions. Hand-computed, not measured-to-pass.
fn straight_line() -> Function {
    let mut f = Function::new("two", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Const(2), Ty::u64(), 0);
    let b = f.emit(e, Op::Const(3), Ty::u64(), 0);
    let n = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(n);
    let s = f.emit(n, Op::Bin(BinOp::Add, a, b), Ty::u64(), 0x10);
    f.block_mut(n).term = Terminator::Return(s);
    f
}

/// A real counted loop: `i = p; while (i < 4) i += 1; return i`. Hand-computed
/// answer 4 for p = 0. This is the anti-invention half: the bound must stop the
/// runaway WITHOUT turning ordinary finite loops into `Budget`.
fn counted_loop() -> Function {
    let mut f = Function::new("count", 0);
    let e = f.entry;
    let p = f.emit(e, Op::Param, Ty::u64(), 0);
    f.params.extend([p]);
    let head = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(head);
    let body = f.new_block(0x20);
    let tail = f.new_block(0x30);
    let four = f.emit(head, Op::Const(4), Ty::u64(), 0x10);
    // The loop-carried value `%i = phi [%p, entry], [%i1, body]` is written
    // once `i1` exists, then inserted as the block's PREFIX instruction, which
    // is the placement the SSA construction pass guarantees.
    let i = f.new_value(Ty::u64());
    let c = f.emit(head, Op::Cmp(IntCmp::Ult, i, four), Ty::Bool, 0x10);
    f.block_mut(head).term = Terminator::Branch {
        cond: c,
        then_bb: body,
        else_bb: tail,
    };
    let one = f.emit(body, Op::Const(1), Ty::u64(), 0x20);
    let i1 = f.emit(body, Op::Bin(BinOp::Add, i, one), Ty::u64(), 0x20);
    f.block_mut(body).term = Terminator::Jump(head);
    f.block_mut(head).instrs.insert(
        0,
        Instr::new(i, Op::Phi(Box::new([(p, e), (i1, body)])), Ty::u64(), 0x10),
    );
    f.block_mut(tail).term = Terminator::Return(i);
    f
}

#[test]
fn an_empty_self_loop_block_spends_budget() {
    let f = empty_self_loop();
    let e = run_budget(&f, &[], 500);
    // The run stops, and says why: the budget, not a hang, not a silent answer.
    assert_eq!(
        e.outcome,
        Outcome::Budget,
        "a cycle of empty blocks must run out of budget, not run forever"
    );
    // The bound is a COUNT the machine can check, and it is the budget: one
    // step per lap (block entry), zero instructions charged in `bb1`.
    assert_eq!(
        e.steps, 501,
        "the run must stop at exactly budget+1 steps, one per empty-block lap"
    );
}

#[test]
fn the_step_account_is_blocks_plus_instructions() {
    let f = straight_line();
    let e = run_budget(&f, &[], 1000);
    assert!(
        matches!(e.outcome, Outcome::Ret(Val::Int(5))),
        "{:?}",
        e.outcome
    );
    // 2 block entries + 3 instructions. `Return` is a terminator, not an
    // instruction, so it is not charged a second time.
    assert_eq!(
        e.steps, 5,
        "steps must count every block entry AND every instruction: \
         entry(bb0) + Const + Const + entry(bb1) + Add = 5"
    );
}

#[test]
fn a_finite_loop_still_answers_and_is_not_reported_as_budget() {
    let f = counted_loop();
    let e = run_budget(&f, &[0], 10_000);
    // Hand-computed: i goes 0,1,2,3,4 and the guard `i < 4` fails at 4.
    assert!(
        matches!(e.outcome, Outcome::Ret(Val::Int(4))),
        "the bound must not eat an ordinary finite loop: {:?}",
        e.outcome
    );
    // Hand-counted, block by block, and the assert is the pin:
    //
    // entry bb0: 1 step, 1 instruction (Param)                  ->  2
    // head x5:   5 steps, 3 instructions each (Phi,Const,Cmp)   -> 20
    // body x4:   4 steps, 2 instructions each (Const, Add)      -> 12
    // tail x1:   1 step, 0 instructions                         ->  1
    // total = 2 + 20 + 12 + 1 = 35
    assert_eq!(e.steps, 35, "step accounting on a real loop, hand-counted");
    // And it is nowhere near the budget, which is the point of the fixture.
    assert!(e.steps < 10_000);
}

#[test]
fn a_tight_budget_stops_a_finite_loop_and_says_budget() {
    // The same finite loop with a budget of 10: the bound fires mid-run. It is
    // `Budget`, named and counted - never a truncated answer wearing a `Ret`.
    let f = counted_loop();
    let e = run_budget(&f, &[0], 10);
    assert_eq!(e.outcome, Outcome::Budget);
    assert_eq!(e.steps, 11, "a run stops at budget+1 steps");
}
