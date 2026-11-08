//! `Function::args_read` - the arity fact the two-pass front end stamps on a
//! callee's signature (`CalleeSig::nargs` in the lifter, the C emitter's
//! declared parameter count, and the m4 two-pass in crates/cli).
//!
//! The rule: `max(index of a READ parameter) + 1`, 0 when no parameter is
//! read. Over-estimating is the SAFE direction (the call site reads more
//! cells = today's loud `clobbered by call` refusal); under-estimating is
//! UNSAFE (the callee reads a cell the caller never passed). Every corner
//! below is the direction a measured bug broke: each one must fail loud if
//! the rule drifts.
use rdec_core::ir::{BinOp, Callee, Function, Instr, IntCmp, Op, Terminator};
use rdec_core::ty::Ty;

/// A function with the four seeded integer cells: RCX, RDX, R8, R9.
fn f4() -> Function {
    let mut f = Function::new("f", 0);
    for _ in 0..4 {
        let v = f.new_value(Ty::u64());
        f.params.push(v);
    }
    let e = f.entry;
    let ps: Vec<_> = f.params.clone();
    for p in ps {
        f.block_mut(e).push(Instr::new(p, Op::Param, Ty::u64(), 0));
    }
    f
}

#[test]
fn no_read_is_zero_arity() {
    let mut f = f4();
    let e = f.entry;
    let one = f.emit(e, Op::Const(1), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::Return(one);
    assert_eq!(f.args_read(), 0, "a body that reads no cell passes none");
}

#[test]
fn max_index_wins_never_a_leading_count() {
    // Reads cell 1 and NOT cell 0: two cells are needed, and a count of
    // LEADING reads would answer ONE - the unsafe direction. This pins the
    // design rule "max-index-plus-one over all reads".
    let mut f = f4();
    let e = f.entry;
    let s = f.emit(
        e,
        Op::Bin(BinOp::Add, f.params[1], f.params[1]),
        Ty::u64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(s);
    assert_eq!(f.args_read(), 2, "max read index 1, plus one");
}

#[test]
fn a_terminator_only_read_still_counts() {
    // The known trap: a parameter that reaches ONLY `Branch.cond` is a READ.
    // `uses_of` counts terminator operands for exactly this.
    let mut f = f4();
    let e = f.entry;
    let t = f.new_block(4);
    let c = f.emit(
        e,
        Op::Cmp(IntCmp::Ne, f.params[2], f.params[2]),
        Ty::Bool,
        0,
    );
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: t,
        else_bb: t,
    };
    let r = f.emit(t, Op::Const(7), Ty::u64(), 0);
    f.block_mut(t).term = Terminator::Return(r);
    assert_eq!(
        f.args_read(),
        3,
        "cell 2 read only through Branch.cond: three cells"
    );
}

#[test]
fn a_loop_carried_phi_read_counts_transitively() {
    // The `fib_rec` shape: the parameter reaches the loop header as
    // `phi(param, latch)`. Stopping the walk AT the phi would call the cell
    // UNREAD and under-estimate - the unsafe direction.
    let mut f = f4();
    let e = f.entry;
    let h = f.new_block(4);
    f.block_mut(e).term = Terminator::Jump(h);
    let p = f.params[1];
    let z = f.emit(
        h,
        Op::Phi(vec![(p, e), (p, h)].into_boxed_slice()),
        Ty::u64(),
        0,
    );
    f.block_mut(h).term = Terminator::Return(z);
    assert_eq!(f.args_read(), 2, "phi-transitive: cell 1 IS read");
}

#[test]
fn forwarding_to_self_is_not_reading_forwarding_to_h_is() {
    // SELF: `f` passes cells to a call to `f` and nothing else reads them.
    // A callee of arity k reads only pairs 0..k, so a cell forwarded to the
    // very callee whose arity is being computed is never consumed - NOT a
    // read. This is what lets `fib_rec`'s recursion drop to one cell in the
    // two-pass (measured: the loop-carried RDX reaches only self-call
    // argument slots, and 4 -> 1 is the whole refusal).
    let mut f = f4();
    let e = f.entry;
    let args: Vec<_> = vec![f.params[0], f.params[1]];
    let c = f.emit(
        e,
        Op::Call {
            callee: Callee::Symbol("f".into()),
            ret_ty: Ty::i64(),
            args: args.into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(c);
    assert_eq!(
        f.args_read(),
        0,
        "both cells reach only SELF-call argument slots: self-forwarding is not reading"
    );
    // A DIFFERENT named callee: passing cell 1 to `g` reads cell 1 FOR `g`.
    let mut g = f4();
    let e = g.entry;
    let args: Vec<_> = vec![g.params[0], g.params[1]];
    let c = g.emit(
        e,
        Op::Call {
            callee: Callee::Symbol("g".into()),
            ret_ty: Ty::i64(),
            args: args.into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    g.block_mut(e).term = Terminator::Return(c);
    assert_eq!(g.args_read(), 2, "forwarding x to h is reading x");
}

#[test]
fn an_indirect_call_reading_a_cell_is_a_read() {
    // The self-forwarding exception is ONLY for calls to this function's own
    // name. An indirect call can land anywhere: its argument reads count.
    let mut f = f4();
    let e = f.entry;
    let t = f.emit(e, Op::Const(0), Ty::u64(), 0);
    let args: Vec<_> = vec![f.params[0], f.params[1]];
    let c = f.emit(
        e,
        Op::Call {
            callee: Callee::Indirect(t),
            ret_ty: Ty::i64(),
            args: args.into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(c);
    assert_eq!(
        f.args_read(),
        2,
        "an indirect callee consumes what it is handed"
    );
}
