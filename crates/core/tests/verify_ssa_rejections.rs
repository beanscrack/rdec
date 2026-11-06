//! THE REJECTION CONTRACT of `crates/core/src/verify.rs`.
//!
//! `verify_ssa` is the oracle every `rdec lift`, TV and T2 leg leans on, so its
//! contract is not only "valid IR is accepted" — it is *the set of shapes it
//! refuses*. A rewrite that gets faster by losing one `e.hard(...)` is a
//! decompiler that ships a wrong C expression, which is the cardinal sin this
//! repo names. This file is the red-first half of the SSA-verification lane:
//! one fixture per check the verifier implements, each asserting the words that
//! check prints, plus the control (the same fixture with the defect removed) so
//! a fixture cannot pass by being rejected for an unrelated reason.
//!
//! Written BEFORE the table-driven rewrite of `verify.rs` and run UNCHANGED
//! through it: any check that disappears during the rewrite turns a test here
//! red, by name.
//!
//! Named and UNWITNESSED (the `aa == ab` precedent — a check this file cannot
//! reach, kept honest by naming it):
//!
//!  * `verify.rs`: `"bad use-site for %{v} in bb{}"` — the use list is built by
//!    the same sweep that reads it (`use-site.instr` is by construction an index
//!    into the block it names), so no `Function` a caller can build through the
//!    public IR API makes it fire. It is a guard on the use-list builder itself,
//!    not on IR, and no fixture reaches it.
//!  * `verify_ssa`: `"%{v} is a phi operand from unreachable bb{pred}"` IS
//!    witnessed (`a_phi_operand_from_an_unreachable_predecessor_is_refused`),
//!    and so is the earlier `verify_ssa` phi check for the same shape; both
//!    names are asserted by name in that one fixture.

use rdec_core::ir::{
    BinOp, Callee, CastKind, FBinOp, FloatCmp, Function, Instr, IntCmp, Op, Terminator, NO_BB,
    NO_VAL,
};
use rdec_core::ty::Ty;
use rdec_core::verify::{verify, verify_ssa};

// ---------------------------------------------------------------- scaffolding

/// Every check prints words; the test asserts those words, so deleting the
/// check cannot be papered over by a neighbouring error message.
fn rejections(f: &Function) -> Vec<String> {
    match verify_ssa(f) {
        Ok(()) => panic!(
            "fixture {:?} was ACCEPTED by verify_ssa: the check under test must reject it",
            f.name
        ),
        Err(e) => {
            assert!(
                !e.is_empty(),
                "fixture {:?} was rejected with NO reason (B7: a rejection always names itself)",
                f.name
            );
            e
        }
    }
}

fn assert_rejects(f: &Function, needle: &str) {
    let errs = rejections(f);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "fixture {:?} must be rejected with a message containing {needle:?}; got {errs:?}",
        f.name
    );
}

fn assert_rejects_where(f: &Function, pred: impl Fn(&String) -> bool, words: &str) {
    let errs = rejections(f);
    assert!(
        errs.iter().any(pred),
        "fixture {:?} must be rejected with {words}; got {errs:?}",
        f.name
    );
}

fn assert_accepts(f: &Function) {
    if let Err(e) = verify_ssa(f) {
        panic!(
            "control fixture {:?} is legal IR and must be ACCEPTED, got {e:?}",
            f.name
        );
    }
}

/// A one-block function with one i64 parameter; the block's terminator is the
/// caller's to set. Every fixture starts here so the only difference between a
/// control and its defective twin is the defect.
fn one_block() -> (Function, rdec_core::ir::BbId, rdec_core::ir::VId) {
    let mut f = Function::new("fixture", 0);
    let e = f.entry;
    let p = f.emit(e, Op::Param, Ty::i64(), 0);
    f.params.push(p);
    (f, e, p)
}

/// The diamond every phi fixture is built on:
/// `e -> {t1, t2} -> g`, `g` merges `t1`'s and `t2`'s constant.
struct Diamond {
    f: Function,
    t1: rdec_core::ir::BbId,
    t2: rdec_core::ir::BbId,
    g: rdec_core::ir::BbId,
    x1: rdec_core::ir::VId,
    x2: rdec_core::ir::VId,
}

fn diamond() -> Diamond {
    let (mut f, e, p) = one_block();
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    let t1 = f.new_block(0x10);
    let t2 = f.new_block(0x20);
    let g = f.new_block(0x30);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: t1,
        else_bb: t2,
    };
    let x1 = f.emit(t1, Op::Const(1), Ty::i64(), 0x10);
    f.block_mut(t1).term = Terminator::Jump(g);
    let x2 = f.emit(t2, Op::Const(2), Ty::i64(), 0x20);
    f.block_mut(t2).term = Terminator::Jump(g);
    f.block_mut(g).term = Terminator::Return(p);
    Diamond {
        f,
        t1,
        t2,
        g,
        x1,
        x2,
    }
}

// ============================ `verify`: function shape ============================

#[test]
fn a_function_with_no_blocks_is_refused() {
    let (mut f, _, _) = one_block();
    f.blocks.clear();
    assert_rejects(&f, "function has no blocks");
}

#[test]
fn an_entry_id_past_the_block_list_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.entry = 99;
    assert_rejects(&f, "entry block id out of range");
}

#[test]
fn a_block_id_that_is_not_its_own_index_is_reported() {
    let (mut f, e, _) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(b);
    f.block_mut(b).term = Terminator::Unreachable;
    assert_accepts(&f);
    // The block at index 1 now claims id 7: the verifier's rule is that a
    // block's id is its index, and the id space is what `preds`/`succs`,
    // `f.block()` and every use site are keyed on.
    f.blocks[1].id = 7;
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("block id 7 is not at index 7"),
        "a warning naming the block id / index disagreement",
    );
}

#[test]
fn a_block_with_the_no_bb_id_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.blocks[0].id = NO_BB;
    assert_rejects(&f, "a block has the NO_BB id");
}

// ============================ `verify`: the value space ============================

#[test]
fn an_instruction_with_no_destination_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.block_mut(e)
        .push(Instr::new(NO_VAL, Op::Const(1), Ty::i64(), 0));
    assert_rejects(&f, "instruction with no destination");
}

#[test]
fn a_destination_at_or_past_next_value_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let ghost = f.next_value + 3;
    f.block_mut(e)
        .push(Instr::new(ghost, Op::Const(1), Ty::i64(), 0));
    assert_rejects(&f, ">= next_value");
}

#[test]
fn a_recorded_type_that_lies_about_the_op_is_a_warning() {
    let (mut f, e, p) = one_block();
    let a = f.emit(e, Op::Const(1), Ty::i64(), 0);
    let good = f.emit(e, Op::Bin(BinOp::Add, a, a), Ty::i64(), 0);
    f.block_mut(e).term = Terminator::Return(good);
    assert_accepts(&f);
    // The same add, recorded as int32: `Op::ty` says int64, the record says
    // int32, and the emitter prints from the record.
    let lying = f.new_value(Ty::i32());
    f.block_mut(e)
        .push(Instr::new(lying, Op::Bin(BinOp::Add, a, a), Ty::i32(), 0));
    f.block_mut(e).term = Terminator::Return(p);
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("op type"),
        "a warning that the op type and the recorded type disagree",
    );
}

#[test]
fn a_flags_type_from_a_non_flags_op_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Const(1), Ty::Flags, 0);
    assert_rejects(&f, "has Flags type from a non-Flags op");
}

#[test]
fn a_bin_op_that_mixes_widths_is_refused() {
    let (mut f, e, p) = one_block();
    let w32 = f.emit(e, Op::Const(1), Ty::u32(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Bin(BinOp::Add, p, w32), Ty::i64(), 0);
    assert_rejects(&f, "mixes widths");
}

#[test]
fn a_cmp_that_mixes_widths_is_refused() {
    let (mut f, e, p) = one_block();
    let w32 = f.emit(e, Op::Const(1), Ty::u32(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Cmp(IntCmp::Eq, p, w32), Ty::Bool, 0);
    assert_rejects(&f, "mixes widths");
}

#[test]
fn a_select_needs_a_bool_condition() {
    let (mut f, e, p) = one_block();
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    let good = f.emit(e, Op::Select(c, p, p), Ty::i64(), 0);
    f.block_mut(e).term = Terminator::Return(good);
    assert_accepts(&f);
    f.emit(e, Op::Select(p, p, p), Ty::i64(), 0);
    assert_rejects(&f, "select condition");
}

#[test]
fn a_selects_arms_must_share_a_type() {
    let (mut f, e, p) = one_block();
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    let w32 = f.emit(e, Op::Const(1), Ty::u32(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Select(c, p, w32), Ty::i64(), 0);
    assert_rejects(&f, "select arms disagree");
}

#[test]
fn a_float_op_refuses_an_integer_operand() {
    let (mut f, e, p) = one_block();
    let fl = f.emit(e, Op::ConstF(1.0), Ty::f64(), 0);
    let ok = f.emit(e, Op::FBin(FBinOp::Add, fl, fl), Ty::f64(), 0);
    let cmp = f.emit(e, Op::FCmp(FloatCmp::Lt, fl, fl), Ty::Bool, 0);
    let _ = cmp;
    f.block_mut(e).term = Terminator::Return(ok);
    assert_accepts(&f);
    f.emit(e, Op::FBin(FBinOp::Mul, p, fl), Ty::f64(), 0);
    assert_rejects(&f, "float op reads");
}

#[test]
fn a_float_comparison_refuses_an_integer_operand() {
    let (mut f, e, p) = one_block();
    let fl = f.emit(e, Op::ConstF(1.0), Ty::f64(), 0);
    let c = f.emit(e, Op::FCmp(FloatCmp::Lt, fl, fl), Ty::Bool, 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::FCmp(FloatCmp::Lt, p, fl), Ty::Bool, 0);
    assert_rejects(&f, "float op reads");
    let _ = c;
}

// ============================ `verify`: calls ============================

fn call(callee: Callee, ret_ty: Ty, args: Vec<rdec_core::ir::VId>) -> Op {
    Op::Call {
        callee,
        ret_ty,
        args: args.into_boxed_slice(),
        pure: false,
    }
}

#[test]
fn a_call_must_name_its_callee() {
    let (mut f, e, p) = one_block();
    f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(
        e,
        call(Callee::Symbol("   ".into()), Ty::u64(), vec![]),
        Ty::u64(),
        0,
    );
    assert_rejects(&f, "call with an empty callee name");
}

#[test]
fn an_indirect_call_through_an_undefined_value_is_refused() {
    let (mut f, e, p) = one_block();
    let target = f.emit(e, Op::Param, Ty::void_ptr(), 0);
    f.emit(
        e,
        call(Callee::Indirect(target), Ty::u64(), vec![]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let ghost = f.new_value(Ty::void_ptr());
    f.emit(
        e,
        call(Callee::Indirect(ghost), Ty::u64(), vec![]),
        Ty::u64(),
        0,
    );
    assert_rejects(&f, "indirect call through undefined");
}

#[test]
fn a_call_argument_must_be_a_register_cell() {
    let (mut f, e, p) = one_block();
    f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![c]),
        Ty::u64(),
        0,
    );
    assert_rejects(&f, "a call's arguments are register cells");
}

#[test]
fn a_call_must_declare_a_return_type_c_can_spell() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::Flags, vec![]),
        Ty::Flags,
        0,
    );
    assert_rejects(&f, "unspellable return type");
}

#[test]
fn a_clobber_must_stay_a_register_cell() {
    let (mut f, e, p) = one_block();
    f.emit(e, Op::Clobber, Ty::u64(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    // `Function::emit` takes the type from the op, so the lying type has to be
    // pushed by hand: a clobber recorded as a Bool.
    let bad = f.new_value(Ty::Bool);
    f.block_mut(e)
        .push(Instr::new(bad, Op::Clobber, Ty::Bool, 0));
    assert_rejects(&f, "a clobbered register is a W64 cell");
}

// ============================ `verify`: casts ============================

#[test]
fn a_float_to_int_cast_refuses_a_non_float_source() {
    let (mut f, e, p) = one_block();
    let fl = f.emit(e, Op::ConstF(1.0), Ty::f64(), 0);
    f.emit(e, Op::Cast(CastKind::FpToSI, Ty::i64(), fl), Ty::i64(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Cast(CastKind::FpToSI, Ty::i64(), p), Ty::i64(), 0);
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("FpToSI/FpToUI"),
        "a warning naming the FpToSI/FpToUI kind",
    );
}

#[test]
fn an_int_to_float_cast_refuses_a_non_int_source() {
    let (mut f, e, p) = one_block();
    f.emit(e, Op::Cast(CastKind::SIToFp, Ty::f64(), p), Ty::f64(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let fl = f.emit(e, Op::ConstF(1.0), Ty::f64(), 0);
    f.emit(e, Op::Cast(CastKind::SIToFp, Ty::f64(), fl), Ty::f64(), 0);
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("SIToFp/UIToFp"),
        "a warning naming the SIToFp/UIToFp kind",
    );
}

#[test]
fn an_int_to_float_cast_refuses_a_non_float_result() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    // The kind says float out; the declared result type says int64.
    f.emit(e, Op::Cast(CastKind::SIToFp, Ty::i64(), p), Ty::i64(), 0);
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("SIToFp/UIToFp"),
        "a warning naming the SIToFp/UIToFp kind",
    );
}

#[test]
fn a_float_widening_refuses_an_integer_source() {
    let (mut f, e, p) = one_block();
    let f32v = f.emit(e, Op::ConstF(1.0), Ty::f32(), 0);
    f.emit(e, Op::Cast(CastKind::FPExt, Ty::f64(), f32v), Ty::f64(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Cast(CastKind::FPExt, Ty::f64(), p), Ty::f64(), 0);
    assert_rejects_where(
        &f,
        |e| e.starts_with("warn:") && e.contains("FPExt/FPTrunc"),
        "a warning naming the FPExt/FPTrunc kind",
    );
}

// ============================ `verify`: terminators ============================

#[test]
fn a_terminator_that_names_a_missing_block_is_refused() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    f.block_mut(e).term = Terminator::Jump(999);
    assert_rejects(&f, "terminator to missing bb999");
}

#[test]
fn a_return_of_an_undefined_value_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let ghost = f.new_value(Ty::i64());
    f.block_mut(e).term = Terminator::Return(ghost);
    assert_rejects(&f, "ret of undefined");
}

#[test]
fn a_return_must_fit_the_declared_return_type() {
    let (mut f, e, p) = one_block();
    f.ret_ty = Ty::u64();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.ret_ty = Ty::u32();
    assert_rejects(&f, "more than the declared");
}

#[test]
fn a_tail_return_of_an_undefined_value_is_refused() {
    let (mut f, e, _) = one_block();
    let ghost = f.new_value(Ty::u64());
    f.block_mut(e).term = Terminator::TailReturn(ghost);
    assert_rejects(&f, "tail return of undefined");
}

#[test]
fn a_tail_return_must_be_defined_by_a_call() {
    let (mut f, e, p) = one_block();
    let g = f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::TailReturn(g);
    assert_accepts(&f);
    let c = f.emit(e, Op::Const(7), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::TailReturn(c);
    assert_rejects(&f, "not a call: a jmp computes no value");
}

#[test]
fn a_tail_calls_callee_must_have_a_name() {
    let (mut f, e, p) = one_block();
    let g = f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::TailReturn(g);
    assert_accepts(&f);
    let a = f.emit(
        e,
        call(Callee::Addr(0x1000), Ty::u64(), vec![]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::TailReturn(a);
    assert_rejects(&f, "tail call's callee has no name");
}

#[test]
fn a_tail_call_and_its_jump_must_be_one_block() {
    let (mut f, e, p) = one_block();
    let g = f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    let b = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(b);
    f.block_mut(b).term = Terminator::TailReturn(g);
    assert_rejects(
        &f,
        "the call and the jmp that leaves with it are one instruction",
    );
}

#[test]
fn a_tail_return_must_fit_the_declared_return_type() {
    let (mut f, e, p) = one_block();
    let g = f.emit(
        e,
        call(Callee::Symbol("g".into()), Ty::u64(), vec![p]),
        Ty::u64(),
        0,
    );
    f.ret_ty = Ty::u64();
    f.block_mut(e).term = Terminator::TailReturn(g);
    assert_accepts(&f);
    f.ret_ty = Ty::u32();
    assert_rejects(&f, "tail return");
    let errs = rejections(&f);
    assert!(
        errs.iter()
            .any(|e| e.contains("tail return") && e.contains("more than the declared")),
        "the width claim for a tail return must name the declared type; got {errs:?}"
    );
}

#[test]
fn a_noreturn_exit_must_follow_a_call() {
    let (mut f, e, _) = one_block();
    f.emit(
        e,
        call(Callee::Symbol("abort".into()), Ty::u64(), vec![]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::NoReturn;
    assert_accepts(&f);
    f.emit(e, Op::Const(1), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::NoReturn;
    assert_rejects(&f, "noreturn exit after");
}

#[test]
fn a_noreturn_exit_needs_an_instruction_at_all() {
    let mut f = Function::new("noreturn_empty", 0);
    let e = f.entry;
    f.block_mut(e).term = Terminator::NoReturn;
    assert_rejects(&f, "noreturn exit with no instruction at all");
}

#[test]
fn a_switch_must_select_on_a_defined_index() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    f.block_mut(e).term = Terminator::Switch {
        index: p,
        targets: vec![b],
    };
    assert_accepts(&f);
    let ghost = f.new_value(Ty::u64());
    f.block_mut(e).term = Terminator::Switch {
        index: ghost,
        targets: vec![b],
    };
    assert_rejects(&f, "switch on undefined index");
}

#[test]
fn a_switch_with_no_targets_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Switch {
        index: p,
        targets: vec![],
    };
    assert_rejects(&f, "switch with an empty target list");
}

#[test]
fn a_switch_target_must_be_a_block_of_this_function() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    f.block_mut(e).term = Terminator::Switch {
        index: p,
        targets: vec![b, 999],
    };
    assert_rejects(&f, "not a block of this function");
}

#[test]
fn a_switch_slot_may_not_name_its_own_block() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    f.block_mut(e).term = Terminator::Switch {
        index: p,
        targets: vec![b, e],
    };
    assert_rejects(&f, "switch slot names its own block");
}

#[test]
fn a_branch_condition_must_be_defined() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: b,
        else_bb: b,
    };
    assert_accepts(&f);
    let ghost = f.new_value(Ty::Bool);
    f.block_mut(e).term = Terminator::Branch {
        cond: ghost,
        then_bb: b,
        else_bb: b,
    };
    assert_rejects(&f, "branch on undefined");
}

#[test]
fn a_branch_condition_must_be_bool() {
    let (mut f, e, p) = one_block();
    let b = f.new_block(0x10);
    f.block_mut(b).term = Terminator::Return(p);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: b,
        else_bb: b,
    };
    assert_accepts(&f);
    f.block_mut(e).term = Terminator::Branch {
        cond: p,
        then_bb: b,
        else_bb: b,
    };
    assert_rejects(&f, "branch condition");
    let errs = rejections(&f);
    assert!(
        errs.iter()
            .any(|e| e.contains("branch condition") && e.contains("is not bool")),
        "the branch check must name the bool requirement; got {errs:?}"
    );
}

// ============================ `verify`: the def/use census ============================

#[test]
fn a_use_of_a_value_nobody_defines_is_refused() {
    let (mut f, e, _) = one_block();
    let ghost = f.new_value(Ty::i64());
    let s = f.emit(e, Op::Bin(BinOp::Add, ghost, ghost), Ty::i64(), 0);
    f.block_mut(e).term = Terminator::Return(s);
    assert_rejects(&f, "is used but never defined");
}

#[test]
fn a_value_defined_twice_is_refused() {
    let (mut f, e, p) = one_block();
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    let dup = f.new_value(Ty::i64());
    f.block_mut(e)
        .push(Instr::new(dup, Op::Const(1), Ty::i64(), 0));
    f.block_mut(e)
        .push(Instr::new(dup, Op::Const(2), Ty::i64(), 0));
    assert_rejects(&f, "is defined 2 times (not single-assignment)");
}

// ============================ `verify_ssa`: SSA-only shape ============================

#[test]
fn a_phi_must_come_before_every_non_phi_in_its_block() {
    let mut d = diamond();
    let phi = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(d.x1, d.t1), (d.x2, d.t2)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_accepts(&d.f);
    // A second phi, placed AFTER a non-phi: give `g` a real non-phi and append
    // the phi behind it.
    let c9 = f_new_value(&mut d.f, Ty::i64());
    let phi2 = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g)
        .push(Instr::new(c9, Op::Const(9), Ty::i64(), 0x30));
    d.f.block_mut(d.g).push(Instr::new(
        phi2,
        Op::Phi(vec![(d.x1, d.t1), (d.x2, d.t2)].into_boxed_slice()),
        Ty::i64(),
        0x30,
    ));
    assert_rejects(&d.f, "after a non-phi");
    // `verify` has no phi-prefix rule: the check lives in `verify_ssa`.
    assert!(
        verify(&d.f).is_ok(),
        "the phi-prefix rule is SSA-only; `verify` said {:?}",
        verify(&d.f)
    );
}

fn f_new_value(f: &mut Function, ty: Ty) -> rdec_core::ir::VId {
    f.new_value(ty)
}

#[test]
fn a_const_with_no_integer_type_is_refused() {
    let (mut f, e, p) = one_block();
    f.emit(e, Op::Const(1), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Const(1), Ty::Void, 0);
    assert_rejects(&f, "has no integer type");
}

#[test]
fn a_const_must_fit_the_width_it_claims() {
    let (mut f, e, p) = one_block();
    f.emit(e, Op::Const(0xffff_ffff), Ty::u32(), 0);
    f.block_mut(e).term = Terminator::Return(p);
    assert_accepts(&f);
    f.emit(e, Op::Const(0x1_0000_0000), Ty::u32(), 0);
    assert_rejects(&f, "does not fit 32-bit type");
}

// ============================ `verify_ssa`: phis ============================

#[test]
fn a_valid_diamond_phi_is_accepted() {
    let mut d = diamond();
    let phi = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(d.x1, d.t1), (d.x2, d.t2)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_accepts(&d.f);
}

#[test]
fn a_phi_whose_incoming_blocks_are_not_its_predecessors_is_refused() {
    let mut d = diamond();
    let phi = f_new_value(&mut d.f, Ty::i64());
    // Both pairs name t1: the multiset is [t1, t1], the predecessors are [t1, t2].
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(d.x1, d.t1), (d.x2, d.t1)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_rejects(&d.f, "!= predecessors");
    assert!(
        verify(&d.f).is_ok(),
        "the predecessor-multiset rule is SSA-only; `verify` said {:?}",
        verify(&d.f)
    );
}

#[test]
fn a_phi_missing_an_incoming_edge_is_refused() {
    let mut d = diamond();
    let phi = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(d.x1, d.t1)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_rejects(&d.f, "!= predecessors");
}

#[test]
fn a_phi_operand_that_is_not_defined_is_refused() {
    let mut d = diamond();
    let ghost = f_new_value(&mut d.f, Ty::i64());
    let phi = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(ghost, d.t1), (d.x2, d.t2)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_rejects(&d.f, "takes undefined");
}

#[test]
fn a_phi_operand_must_be_available_on_its_edge() {
    // The multisets match, so the predecessor check passes; the VALUES are on
    // the wrong edges: `x1` is defined in t1 and claims the t2 edge.
    let mut d = diamond();
    let phi = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.g).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(d.x1, d.t2), (d.x2, d.t1)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    assert_rejects(&d.f, "is not available on the edge");
    assert!(
        verify(&d.f).is_ok(),
        "edge availability is SSA-only; `verify` said {:?}",
        verify(&d.f)
    );
}

/// An unreachable block `u` jumps to an unreachable block `v`, whose phi reads
/// `u`. One fixture witnesses the three checks that name unreachability:
/// a definition in an unreachable block, a phi from an unreachable block, and a
/// phi operand available on an unreachable edge.
fn unreachable_phi() -> Function {
    let (mut f, e, p) = one_block();
    let a = f.new_block(0x10);
    f.block_mut(e).term = Terminator::Jump(a);
    f.block_mut(a).term = Terminator::Return(p);
    let u = f.new_block(0x20);
    let y = f.emit(u, Op::Const(3), Ty::i64(), 0x20);
    let v = f.new_block(0x30);
    f.block_mut(u).term = Terminator::Jump(v);
    let phi = f.new_value(Ty::i64());
    f.block_mut(v).instrs.insert(
        0,
        Instr::new(
            phi,
            Op::Phi(vec![(y, u)].into_boxed_slice()),
            Ty::i64(),
            0x30,
        ),
    );
    f.block_mut(v).term = Terminator::Return(phi);
    Function {
        name: "unreachable_phi".into(),
        ..f
    }
}

#[test]
fn a_definition_in_an_unreachable_block_is_refused() {
    assert_rejects(&unreachable_phi(), "is defined in unreachable bb");
}

#[test]
fn a_phi_from_an_unreachable_block_is_refused() {
    // The needle excludes the phi-EDGE check's own words, so this test is red
    // only for the phi-block reachability check, not for its neighbour.
    assert_rejects_where(
        &unreachable_phi(),
        |e| e.contains("from unreachable bb") && !e.contains("phi operand"),
        "a phi from an unreachable block, named without the phi-edge wording",
    );
}

#[test]
fn a_phi_operand_from_an_unreachable_predecessor_is_refused() {
    let errs = rejections(&unreachable_phi());
    assert!(
        errs.iter()
            .any(|e| e.contains("is a phi operand from unreachable bb")),
        "the phi-edge reachability check must print its own words; got {errs:?}"
    );
}

// ============================ `verify_ssa`: dominance ============================

#[test]
fn a_use_before_its_definition_in_the_same_block_is_refused() {
    // Control: the add is the second instruction in the block, so its reads
    // are program-order later than the parameter's def.
    let (mut f, e, _) = one_block();
    let p = f.params[0];
    let s = f.new_value(Ty::i64());
    f.block_mut(e)
        .push(Instr::new(s, Op::Bin(BinOp::Add, p, p), Ty::i64(), 0));
    f.block_mut(e).term = Terminator::Return(s);
    assert_accepts(&f);

    // The defect: the add at #0 reads the value defined at #1.
    let (mut g, ge, _) = one_block();
    let v = g.new_value(Ty::i64());
    let s = g.new_value(Ty::i64());
    g.block_mut(ge)
        .push(Instr::new(s, Op::Bin(BinOp::Add, v, v), Ty::i64(), 0));
    g.block_mut(ge)
        .push(Instr::new(v, Op::Const(4), Ty::i64(), 0));
    g.block_mut(ge).term = Terminator::Return(s);
    assert_rejects(&g, "use before def");
    assert!(
        verify(&g).is_ok(),
        "in-block order is SSA-only; `verify` said {:?}",
        verify(&g)
    );
}

#[test]
fn a_use_outside_the_dominance_scope_of_its_def_is_refused() {
    let mut d = diamond();
    // `x1` is defined in t1 and read in t2: the two arms do not dominate
    // each other, so the read is not available.
    let bad = f_new_value(&mut d.f, Ty::i64());
    d.f.block_mut(d.t2).instrs.insert(
        0,
        Instr::new(bad, Op::Bin(BinOp::Add, d.x1, d.x1), Ty::i64(), 0x20),
    );
    d.f.block_mut(d.t2).term = Terminator::Return(bad);
    assert_rejects(&d.f, "does not dominate use in bb");
    assert!(
        verify(&d.f).is_ok(),
        "dominance is SSA-only; `verify` said {:?}",
        verify(&d.f)
    );
}
