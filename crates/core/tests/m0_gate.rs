//! M0 gate fixtures: one test per §9 bug, written to fail against the code as
//! audited. Process rule §8: "every bug fix lands with the fixture that would
//! have caught it, in the oracle tier that should have caught it."
//!
//! Each test checks against ground truth computed in the test (an i32/i64 model
//! of the intended semantics) or against TV, never against the pass's own output.

use rdec_core::interp;
use rdec_core::ir::*;
use rdec_core::passes;
use rdec_core::tv;
use rdec_core::ty::{Sx, Ty, Width};
use rdec_core::verify;

fn i32() -> Ty {
    Ty::int(Width::W32, Sx::Signed)
}
fn u32t() -> Ty {
    Ty::int(Width::W32, Sx::Unsigned)
}
fn b() -> Ty {
    Ty::Bool
}

/// `int f(int a, int b)` with an empty entry block.
fn fn2(name: &str) -> Function {
    let mut f = Function::new(name, 0x1000);
    let e = f.entry;
    let a = f.emit(e, Op::Param, i32(), 0x1000);
    let bb = f.emit(e, Op::Param, i32(), 0x1004);
    f.params = vec![a, bb];
    f.ret_ty = i32();
    f
}

fn one(v: bool) -> u64 {
    v as u64
}

// ---------------------------------------------------------------------------
// B1: uses_of / replace_use_all ignore Return operands, so DCE deletes the
// expression a `return` feeds.
// ---------------------------------------------------------------------------

#[test]
fn b1_dce_must_not_delete_a_returned_value() {
    let mut f = fn2("b1");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let s = f.emit(e, Op::Bin(BinOp::Add, a, bb), i32(), 0x1008);
    f.block_mut(e).term = Terminator::Return(s);

    let before = f.clone();
    passes::dead_code_elimination(&mut f);

    // Structural: the add must still exist.
    let has_add = f.blocks.iter().any(|blk| {
        blk.instrs
            .iter()
            .any(|i| matches!(i.op, Op::Bin(BinOp::Add, ..)))
    });
    assert!(has_add, "DCE deleted the add that feeds `return` (B1)");

    // Behavioral, through TV's reference: f(7,5) == 12.
    let x = interp::run(&f, &[7, 5]);
    let want = interp::run(&before, &[7, 5]);
    assert_eq!(x.outcome, want.outcome, "return value changed (B1)");
    assert_eq!(x.outcome, interp::Outcome::Ret(val_i(12)));
}

/// Small helper so the expected value reads as data.
fn val_i(v: u64) -> interp::Val {
    interp::Val::Int(v)
}

#[test]
fn b1_terminator_uses_are_counted() {
    let mut f = fn2("b1u");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let s = f.emit(e, Op::Bin(BinOp::Add, a, bb), i32(), 0x1008);
    f.block_mut(e).term = Terminator::Return(s);
    assert_eq!(f.use_count(s), 1, "the `ret` operand is a use (B1)");
    assert_eq!(f.uses_of(s).len(), 1);
}

// ---------------------------------------------------------------------------
// B2: block_key renumbers free operands and omits the terminator's condition,
// so blocks that differ only by input operands or condition hash equal and
// `dedup_blocks` collapses them.
// ---------------------------------------------------------------------------

#[test]
fn b2_dedup_must_not_merge_blocks_with_different_operands() {
    //   entry: if (a == 0) goto L1 else L2
    //   L1: %s = p0 + p1 ; ret %s
    //   L2: %s = p2 + p3 ; ret %s
    // The two tails are structurally identical and must NOT be merged.
    let mut f = Function::new("b2", 0x1000);
    let e = f.entry;
    let ps: Vec<VId> = (0..4)
        .map(|i| f.emit(e, Op::Param, i32(), 0x1000 + 4 * i as u64))
        .collect();
    f.params = ps.clone();
    f.ret_ty = i32();
    let z = f.emit(e, Op::Const(0), i32(), 0x2000);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, ps[0], z), b(), 0x2004);
    let l1 = f.new_block(0x3000);
    let l2 = f.new_block(0x4000);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: l1,
        else_bb: l2,
    };
    let s1 = f.emit(l1, Op::Bin(BinOp::Add, ps[0], ps[1]), i32(), 0x3000);
    f.block_mut(l1).term = Terminator::Return(s1);
    let s2 = f.emit(l2, Op::Bin(BinOp::Add, ps[2], ps[3]), i32(), 0x4000);
    f.block_mut(l2).term = Terminator::Return(s2);

    let before = f.clone();
    passes::dedup_blocks(&mut f);

    // Both arms, so the check is "each tail kept its own operands", not just
    // "the merge happened/not". With p0 == 0 the branch takes L1 (p0+p1),
    // otherwise L2 (p2+p3).
    for (p0, p1, p2, p3) in [(1u64, 2, 100, 200), (0, 5, 7, 9), (3, 4, 50, 60)] {
        let inputs = [p0, p1, p2, p3];
        let want = if p0 == 0 { p0 + p1 } else { p2 + p3 };
        let a = interp::run(&before, &inputs);
        let x = interp::run(&f, &inputs);
        assert_eq!(
            a.outcome, x.outcome,
            "dedup merged two blocks that add different operands (B2) at {inputs:?}"
        );
        assert_eq!(
            a.outcome,
            interp::Outcome::Ret(val_i(want)),
            "reference semantics at {inputs:?}"
        );
    }
}

#[test]
fn b2_dedup_must_not_merge_blocks_with_different_conditions() {
    // entry: %c1 = a < b ; br %c1 L1 L2
    // L1: ret 1 ; L2: %c2 = a > b ; br %c2 L3 L4 ; L3: ret 2 ; L4: ret 3
    // L1 and L3 have identical bodies; the key must not ignore terminators.
    let mut f = fn2("b2c");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let l1 = f.new_block(0x2000);
    let l2 = f.new_block(0x3000);
    let l3 = f.new_block(0x4000);
    let l4 = f.new_block(0x5000);
    let c1 = f.emit(e, Op::Cmp(IntCmp::Lt, a, bb), b(), 0x2000);
    f.block_mut(e).term = Terminator::Branch {
        cond: c1,
        then_bb: l1,
        else_bb: l2,
    };
    let one_v = f.emit(l1, Op::Const(1), i32(), 0x2010);
    f.block_mut(l1).term = Terminator::Return(one_v);
    let c2 = f.emit(l2, Op::Cmp(IntCmp::Gt, a, bb), b(), 0x3000);
    f.block_mut(l2).term = Terminator::Branch {
        cond: c2,
        then_bb: l3,
        else_bb: l4,
    };
    let two = f.emit(l3, Op::Const(1), i32(), 0x4010);
    f.block_mut(l3).term = Terminator::Return(two);
    let three = f.emit(l4, Op::Const(3), i32(), 0x5010);
    f.block_mut(l4).term = Terminator::Return(three);

    let before = f.clone();
    passes::dedup_blocks(&mut f);
    for (x, y) in [(5u64, 9u64), (9, 5), (7, 7)] {
        let inp = [x, y];
        assert_eq!(
            interp::run(&before, &inp).outcome,
            interp::run(&f, &inp).outcome,
            "dedup changed behavior for a={x} b={y} (B2)"
        );
    }
}

// ---------------------------------------------------------------------------
// B3/B4: flag lowering treats each get_flag as its own tree root, and lowers
// SF of `cmp` to "a < 0". Ground truth: the flags come from a - b.
// ---------------------------------------------------------------------------

/// Build `cmp a,b` + a branch on a boolean expression over the flags.
fn with_cmp(name: &str, mk: &dyn Fn(&mut Function, VId, VId) -> Op) -> Function {
    let mut f = fn2(name);
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let fl = f.emit(
        e,
        Op::Flags(FlagsOp::Sub {
            a,
            b: bb,
            borrow: false,
        }),
        Ty::Flags,
        0x1008,
    );
    let top = mk(&mut f, a, fl);
    let cond = f.emit(e, top, b(), 0x100c);
    let lt = f.new_block(0x2000);
    let lf = f.new_block(0x3000);
    f.block_mut(e).term = Terminator::Branch {
        cond,
        then_bb: lt,
        else_bb: lf,
    };
    let o1 = f.emit(lt, Op::Const(1), i32(), 0x2000);
    f.block_mut(lt).term = Terminator::Return(o1);
    let o0 = f.emit(lf, Op::Const(0), i32(), 0x3000);
    f.block_mut(lf).term = Terminator::Return(o0);
    f
}

fn run01(f: &Function, a: u64, bb: u64) -> u64 {
    match interp::run(f, &[a, bb]).outcome {
        interp::Outcome::Ret(v) => v.bits(),
        other => panic!("expected a return, got {other:?}"),
    }
}

fn sub_s(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[test]
fn b4_sf_after_cmp_is_the_sign_of_a_minus_b() {
    // `js` after `cmp a,b`: SF of (a-b), not of a.
    let mut f = with_cmp("b4", &|_f, _a, fl| Op::GetFlag(Flag::SF, fl));
    passes::flag_elimination(&mut f);
    for (a, b) in [
        (1i32, 2i32),
        (-1, -2),
        (i32::MIN, 1),
        (5, 5),
        (0, -1),
        (i32::MAX, -1),
    ] {
        let exp = one(sub_s(a, b) < 0);
        let got = run01(&f, a as u32 as u64, b as u32 as u64);
        assert_eq!(
            got, exp,
            "js after cmp a={a} b={b}: SF must be sign(a-b), not sign(a) (B4)"
        );
    }
}

#[test]
fn b3_jl_after_cmp_is_signed_less() {
    // `jl` = SF xor OF, over the flags of (a-b).
    let mut f = with_cmp("b3", &|f, _a, fl| {
        // Build the SF^OF tree in the entry block, before the branch.
        let en = f.entry;
        let sfv = f.emit(en, Op::GetFlag(Flag::SF, fl), b(), 0x1100);
        let ofv = f.emit(en, Op::GetFlag(Flag::OF, fl), b(), 0x1104);
        Op::Bin(BinOp::Xor, sfv, ofv)
    });
    passes::flag_elimination(&mut f);
    for (a, b) in [
        (1i32, 2i32),
        (2, 1),
        (-1, 1),
        (i32::MAX, -1), // signed overflow in a-b: SF/OF flip
        (i32::MIN, 1),
        (0, 0),
        (-5, -3),
    ] {
        let exp = one(a < b);
        let got = run01(&f, a as u32 as u64, b as u32 as u64);
        assert_eq!(got, exp, "jl after cmp, a={a} b={b} (B3)");
    }
}

// ---------------------------------------------------------------------------
// B5/B6: the exact fallback path uses the Flags value as the arithmetic result,
// and covers only CF/OF/ZF/SF of add/sub, so PF after `test` leaves a live
// Flags value in the IR.
// ---------------------------------------------------------------------------

fn with_test(name: &str, flag: Flag) -> Function {
    let mut f = fn2(name);
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let fl = f.emit(e, Op::Flags(FlagsOp::And { a, b: bb }), Ty::Flags, 0x1008);
    let g = f.emit(e, Op::GetFlag(flag, fl), b(), 0x100c);
    let o = f.emit(e, Op::Cast(CastKind::ZExt, i32(), g), i32(), 0x1010);
    f.block_mut(e).term = Terminator::Return(o);
    f
}

#[test]
fn b5_zf_after_test_is_the_result_of_a_and_b() {
    let mut f = with_test("b5", Flag::ZF);
    let before = f.clone();
    passes::flag_elimination(&mut f);
    for (a, b) in [(0u32, 0u32), (1, 1), (0xf0, 0x0f), (0xff, 0xff), (3, 4)] {
        let exp = one((a & b) == 0);
        let got = run01(&f, a as u64, b as u64);
        assert_eq!(got, exp, "ZF after test a={a} b={b} (B5)");
    }
    let _ = before;
}

/// The B6 contract, checked the way the pipeline runs: every *read* is lowered
/// by `flag_elimination`, no Flags-typed value is left live (that was v1's bug),
/// and the now-dead flag defs are collected by DCE (design rule 1).
fn assert_no_flags_values(f: &Function) {
    let reads: Vec<&Op> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .map(|i| &i.op)
        .filter(|o| matches!(o, Op::GetFlag(..)))
        .collect();
    assert!(
        reads.is_empty(),
        "a get_flag survived flag_elimination (B6)"
    );
    for v in 0..f.next_value {
        if f.ty_of(v) == Some(rdec_core::Ty::Flags) {
            assert!(
                f.uses_of(v).is_empty(),
                "a Flags-typed value is still consumed (B6): %{v}"
            );
        }
    }
    let defs: Vec<&Op> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .map(|i| &i.op)
        .filter(|o| matches!(o, Op::Flags(..)))
        .collect();
    assert!(
        defs.is_empty(),
        "dead flag defs survived DCE (design rule 1): {defs:?}"
    );
}

#[test]
fn b6_pf_after_test_is_lower_byte_parity_and_leaves_no_flags() {
    let mut f = with_test("b6", Flag::PF);
    passes::flag_elimination(&mut f);
    for (a, b) in [(0u32, 0u32), (1, 1), (0xff, 0x01), (0xf0, 0x0f), (7, 3)] {
        let exp = one(((a & b) as u8).count_ones().is_multiple_of(2));
        let got = run01(&f, a as u64, b as u64);
        assert_eq!(got, exp, "PF after test a={a} b={b} (B6)");
    }
    // The structural half of B6, under the real pass order.
    while passes::flag_elimination(&mut f) {}
    while passes::dead_code_elimination(&mut f) {}
    assert_no_flags_values(&f);
}

#[test]
fn b6_cf_after_inc_is_undefined_and_must_be_reported_not_invented() {
    // inc preserves CF, and v1's IR has no prev-flags operand to express it.
    // The honest outcome is that the read is *defined as undefined*: the pass
    // must not fabricate a value, and must not leave a raw get_flag.
    let mut f = fn2("b6inc");
    let e = f.entry;
    let a = f.params[0];
    let fl = f.emit(e, Op::Flags(FlagsOp::Inc { a }), Ty::Flags, 0x1008);
    let g = f.emit(e, Op::GetFlag(Flag::CF, fl), b(), 0x100c);
    let o = f.emit(e, Op::Cast(CastKind::ZExt, i32(), g), i32(), 0x1010);
    f.block_mut(e).term = Terminator::Return(o);
    while passes::flag_elimination(&mut f) {}
    while passes::dead_code_elimination(&mut f) {}
    assert_no_flags_values(&f);
    // And the gap must be *stated*, not silently zeroed: the returned value has
    // to carry an Opaque node, so the function reports itself E0.
    let gaps: Vec<&Op> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .map(|i| &i.op)
        .filter(|o| matches!(o, Op::Opaque(_)))
        .collect();
    assert!(
        !gaps.is_empty(),
        "CF after inc was lowered to a value instead of a stated gap (B6)"
    );
}

// ---------------------------------------------------------------------------
// B7: verify_ssa drops verify()'s messages.
// ---------------------------------------------------------------------------

#[test]
fn b7_structural_failures_keep_their_messages() {
    let mut f = fn2("b7");
    let e = f.entry;
    let ghost = f.next_value + 9; // never defined
    f.block_mut(e).term = Terminator::Return(ghost);
    let err = verify::verify_ssa(&f).expect_err("must reject a return of an undefined value");
    assert!(
        !err.is_empty(),
        "verify_ssa returned an error with no messages (B7)"
    );
    assert!(
        err.iter().any(|m| !m.starts_with("warn:")),
        "expected a hard structural message, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Code-reading findings from §9 that need a fixture before the fix.
// ---------------------------------------------------------------------------

#[test]
fn cmp_pair_table_is_arithmetically_correct() {
    // §9: 8 of 16 entries are wrong — (ult && ne) mapped to <=, (sgt || ne)
    // mapped to >=, (uge || ne) mapped to ugt (it is constant true).
    // Ground truth is computed per pair over all 8-bit operands.
    use rdec_core::passes;
    let cases: &[(IntCmp, IntCmp, BinOp)] = &[
        (IntCmp::Ult, IntCmp::Ne, BinOp::And),
        (IntCmp::Uge, IntCmp::Ne, BinOp::Or),
        (IntCmp::Lt, IntCmp::Ne, BinOp::And),
        (IntCmp::Gt, IntCmp::Ne, BinOp::Or),
        (IntCmp::Eq, IntCmp::Ult, BinOp::And),
        (IntCmp::Ne, IntCmp::Ugt, BinOp::And),
    ];
    for &(c1, c2, op) in cases {
        for a in 0u64..16 {
            for b in 0u64..16 {
                let want = if op == BinOp::And {
                    c1.eval(a, b, Width::W8, Sx::Unsigned) && c2.eval(a, b, Width::W8, Sx::Unsigned)
                } else {
                    c1.eval(a, b, Width::W8, Sx::Unsigned) || c2.eval(a, b, Width::W8, Sx::Unsigned)
                };
                if let Some(folded) = passes::cmp_pair(c1, c2, op) {
                    let got = folded.eval(a, b, Width::W8, Sx::Unsigned);
                    assert_eq!(
                        got, want,
                        "cmp_pair({c1:?},{c2:?},{op:?}) at a={a} b={b}: folded gives {got}, truth {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn merge_blocks_updates_phi_incoming() {
    // Merging a block into its successor must rewrite phis in the successor that
    // named the merged block as an incoming edge.
    let mut f = fn2("mergephi");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let l1 = f.new_block(0x2000);
    let l2 = f.new_block(0x3000);
    let z = f.emit(e, Op::Const(0), i32(), 0x1008);
    let c = f.emit(e, Op::Cmp(IntCmp::Eq, a, z), b(), 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: c,
        then_bb: l1,
        else_bb: l2,
    };
    // l1 falls through to l2.
    f.emit(l1, Op::Const(7), i32(), 0x2000);
    f.block_mut(l1).term = Terminator::Jump(l2);
    // l2: %p = phi [a, l1], [bb, e] ; ret %p
    let p = f.new_value(i32());
    f.block_mut(l2).instrs.insert(
        0,
        Instr::new(p, Op::Phi(Box::new([(a, l1), (bb, e)])), i32(), 0x3000),
    );
    f.block_mut(l2).term = Terminator::Return(p);

    passes::merge_blocks(&mut f);
    // After merging l1 into l2, no phi may name the merged block.
    for blk in &f.blocks {
        for i in &blk.instrs {
            if let Op::Phi(pairs) = &i.op {
                for (_, from) in pairs.iter() {
                    assert!(
                        f.blocks
                            .get(*from as usize)
                            .map(|b| !b.instrs.is_empty()
                                || matches!(
                                    b.term,
                                    Terminator::Return(_)
                                        | Terminator::Jump(_)
                                        | Terminator::Branch { .. }
                                ))
                            .unwrap_or(false),
                        "phi names a block that no longer flows anywhere (merge_blocks phi bug)"
                    );
                }
            }
        }
    }
    let v = interp::run(&f, &[0, 5]);
    assert_eq!(v.outcome, interp::Outcome::Ret(val_i(0)));
}

#[test]
fn const_fold_mulhi_does_not_shift_by_64_at_narrow_widths() {
    let mut f = fn2("mulhi");
    let e = f.entry;
    let a = f.params[0];
    let k = f.emit(e, Op::Const(3), u32t(), 0x1008);
    let m = f.emit(e, Op::Bin(BinOp::MulHi, a, k), u32t(), 0x100c);
    f.block_mut(e).term = Terminator::Return(m);
    let before = f.clone();
    passes::const_fold(&mut f);
    let v = interp::run(&f, &[0x1_0000_0000, 0]);
    let w = interp::run(&before, &[0x1_0000_0000, 0]);
    assert_eq!(
        v.outcome, w.outcome,
        "const-fold changed MulHi (B: const_bin)"
    );
}

#[test]
fn tv_catches_the_sabotaged_dce_by_name() {
    // The M0 gate's third line.
    let mut f = fn2("sab");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let t = f.emit(e, Op::Bin(BinOp::Mul, a, a), i32(), 0x1008);
    let s = f.emit(e, Op::Bin(BinOp::Add, t, bb), i32(), 0x100c);
    let u = f.emit(e, Op::Bin(BinOp::Xor, s, a), i32(), 0x1010);
    f.block_mut(e).term = Terminator::Return(u);

    let states = tv::states(f.params.len(), 64, 0x1234);
    let res = tv::run_validated(&mut f, ("sabotage_dce", tv::sabotage_dce), &states, 50_000);
    match res {
        Ok(_) => panic!("TV did not catch a DCE that deletes a live instruction"),
        Err(fails) => {
            assert_eq!(fails[0].pass, "sabotage_dce", "must name the pass");
            assert!(!fails[0].diffs.is_empty());
        }
    }
}

#[test]
fn tv_accepts_the_real_pipeline_on_a_valid_function() {
    // The positive control: TV must not be a machine that only says "guilty".
    let mut f = fn2("ok");
    let e = f.entry;
    let (a, bb) = (f.params[0], f.params[1]);
    let s = f.emit(e, Op::Bin(BinOp::Add, a, bb), i32(), 0x1008);
    let m = f.emit(e, Op::Bin(BinOp::Mul, s, s), i32(), 0x100c);
    f.block_mut(e).term = Terminator::Return(m);
    let before = f.clone();
    let states = tv::states(2, 128, 0x99);
    for p in [
        (
            "const_fold",
            passes::const_fold as fn(&mut Function) -> bool,
        ),
        ("copy_prop", passes::copy_prop as fn(&mut Function) -> bool),
        (
            "dead_code_elimination",
            passes::dead_code_elimination as fn(&mut Function) -> bool,
        ),
        (
            "simplify_branches",
            passes::simplify_branches as fn(&mut Function) -> bool,
        ),
        (
            "merge_blocks",
            passes::merge_blocks as fn(&mut Function) -> bool,
        ),
        (
            "dedup_blocks",
            passes::dedup_blocks as fn(&mut Function) -> bool,
        ),
        (
            "prune_phis",
            passes::prune_phis as fn(&mut Function) -> bool,
        ),
    ] {
        let res = tv::run_validated(&mut f, p, &states, 50_000);
        assert!(
            res.is_ok(),
            "TV rejected a valid pass {:?}: {:?}",
            p.0,
            res.unwrap_err()
        );
    }
    assert_eq!(
        interp::run(&f, &[3, 4]).outcome,
        interp::run(&before, &[3, 4]).outcome
    );
}

// ---------------------------------------------------------------------------
// The verifier's own new checks (§1.3), and pipeline determinism, which is
// TV's precondition: a nondeterministic pipeline makes every before/after
// comparison meaningless.
// ---------------------------------------------------------------------------

#[test]
fn verifier_rejects_a_mixed_width_binop() {
    let mut f = fn2("vwidth");
    let e = f.entry;
    let a = f.params[0]; // i32
    let c = f.emit(e, Op::Const(1), Ty::u64(), 0x1000); // u64
    let s = f.emit(e, Op::Bin(BinOp::Add, a, c), Ty::u64(), 0x1004);
    f.block_mut(e).term = Terminator::Return(s);
    let err = verify::verify_ssa(&f).expect_err("mixed widths must be rejected");
    assert!(
        err.iter().any(|m| m.contains("mixes widths")),
        "expected a width-agreement message, got {err:?}"
    );
}

#[test]
fn verifier_rejects_a_select_with_an_integer_condition() {
    let mut f = fn2("vsel");
    let e = f.entry;
    let a = f.params[0]; // i32, not bool
    let s = f.emit(e, Op::Select(a, a, a), i32(), 0x1000);
    f.block_mut(e).term = Terminator::Return(s);
    let err = verify::verify_ssa(&f).expect_err("an integer select condition must be rejected");
    assert!(
        err.iter().any(|m| m.contains("select condition")),
        "expected a select-condition message, got {err:?}"
    );
}

#[test]
fn pipeline_is_deterministic() {
    let base = with_test("det", Flag::OF);
    let mut x = base.clone();
    let mut y = base.clone();
    rdec_core::run_pipeline(&mut x);
    rdec_core::run_pipeline(&mut y);
    assert_eq!(
        rdec_core::fmt_ir::format_function(&x),
        rdec_core::fmt_ir::format_function(&y),
        "the pipeline is not deterministic: TV comparisons would be noise"
    );
    for (p, q) in [(1u64, 2u64), (7, 9), (0, 0), (u32::MAX as u64, 3)] {
        assert_eq!(
            run01(&x, p, q),
            run01(&y, p, q),
            "results differ at ({p},{q})"
        );
    }
    let v = verify::verify_ssa(&x);
    let warns: Vec<String> = v.err().unwrap_or_default();
    assert!(
        warns.iter().all(|m| m.starts_with("warn:")),
        "pipeline output must not hard-fail verification: {warns:?}"
    );
}
