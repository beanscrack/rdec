//! P-track seed: the ORACLE gets its own oracle.
//!
//! Every T2 IR-leg comparison, every `run` in every fixture, trusts
//! `interp::binop`. When the interpreter drifts, the gates do not go red —
//! they agree with the drift, because the interpreter IS one side of the
//! comparison. This suite is the interpreter's check against machine
//! semantics WRITTEN FROM THE x86 FACTS in this file (a second, independent
//! implementation: wrapping width arithmetic, bitwise identity, x86 shift
//! masking, sign-extension for `sar`), over deterministic pseudo-random
//! values and the boundary set (0, 1, MAX, MAX-1, alternating bits, the sign
//! bit and its predecessor) at every width the IR carries.
//!
//! Scope is the set the LIFTER CAN EMIT (measured: `Add Sub Mul And Or Xor
//! Shl Shr Sar` appear in lift.rs; `MulHi/MulSmul/UDiv/URem/SDiv/SRem/RotL/
//! RotR` do not — they sit in the IR for a front end that has not needed
//! them, and testing them here would test code no gate reaches; the day an
//! `imul` high form lands, its lane extends this table).
//!
//! Shift counts are fed RAW (0..2^64) on purpose: the lifter masks with
//! `count_for`, but the interpreter is public — passes-fused and fuzz-built
//! IR present counts directly — and the suite's contract is that the model
//! masks exactly the way the machine does (5 bits for 8/16/32-bit operands,
//! 6 for 64). This is the rule `docs/ir-semantics.md` calls "hardware, not
//! taste". Feeding raw counts is what found the first P-track defect: the
//! old `Shr`/`Sar` arms guarded with `y.min(128)`, which is UB on a u64/i64
//! shift for counts in 64..=128 — debug panics, RELEASE WRAPS — and the old
//! `shift_left` answered 0 for raw counts 32..63 at W32 where the masked
//! machine shifts by 0..31.
//!
//! Signedness (`Sx`) is a DISPLAY concern (ir.rs, §1.3); every case is run
//! under BOTH displays and the reference ignores `Sx` — an interpreter that
//! consulted it would redden here, which is the same rule `emit_cmp_
//! signedness.rs` enforces at the emitter.
//!
//! SABOTAGE PROVEN (2026-11-06): `BinOp::Sar` rendered as a LOGICAL shift
//! (the exact bug class the T2 first run caught at the EMITTER) reddens
//! `sar_is_arithmetic_at_every_width`; `Shl` emitted without the width
//! truncation reddens the W8/W16/W32 legs. Restored green. The suite's
//! value is that a mutation in `interp::binop` cannot ship quietly.

use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{BinOp, Function, Op, Terminator};
use rdec_core::ty::{Sx, Ty, Width};

fn mask(w: Width) -> u64 {
    match w {
        Width::W1 => 1, // the suite never runs W1; the compiler requires the arm
        Width::W8 => 0xff,
        Width::W16 => 0xffff,
        Width::W32 => 0xffff_ffff,
        Width::W64 => u64::MAX,
    }
}

fn trunc(w: Width, x: u64) -> u64 {
    x & mask(w)
}

fn sign_bit(w: Width) -> u64 {
    1 << (w.bits() - 1)
}

/// x86 shift-count masking: 5 bits for 8/16/32-bit operands, 6 for 64.
fn mask_count(w: Width, y: u64) -> u64 {
    if w == Width::W64 {
        y & 63
    } else {
        y & 31
    }
}

/// THE REFERENCE: machine semantics, independent of `interp::binop`.
fn reference(o: BinOp, w: Width, x: u64, y: u64) -> u64 {
    let (x, y) = (trunc(w, x), trunc(w, y));
    match o {
        BinOp::Add => trunc(w, x.wrapping_add(y)),
        BinOp::Sub => trunc(w, x.wrapping_sub(y)),
        BinOp::Mul => trunc(w, x.wrapping_mul(y)),
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl => {
            let n = mask_count(w, y) as u32;
            trunc(w, ((x as u128) << n) as u64)
        }
        BinOp::Shr => {
            let n = mask_count(w, y) as u32;
            x >> n
        }
        BinOp::Sar => {
            let n = mask_count(w, y) as u32;
            let sx = if x & sign_bit(w) != 0 {
                // sign-extend the width-w value into i64
                (x | !mask(w)) as i64
            } else {
                x as i64
            };
            trunc(w, (sx >> n) as u64)
        }
        _ => unreachable!("suite is scoped to the liftable integer set"),
    }
}

fn model(o: BinOp, w: Width, sx: Sx, x: u64, y: u64) -> u64 {
    let mut f = Function::new("prop", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Param, Ty::Int { sx, w }, 0);
    let b = f.emit(e, Op::Param, Ty::Int { sx, w }, 0);
    let r = f.emit(e, Op::Bin(o, a, b), Ty::Int { sx, w }, 0);
    f.block_mut(e).term = Terminator::Return(r);
    f.params.extend([a, b]);
    f.ret_ty = Ty::Int { sx, w };
    match run(&f, &[x, y, 0, 0]).outcome {
        Outcome::Ret(Val::Int(v)) => v,
        other => panic!("{o:?} {w:?} x={x:#x} y={y:#x}: interpreter said {other:?}"),
    }
}

fn check(o: BinOp, w: Width) {
    // Boundary set: the values that find off-by-one width bugs.
    let m = mask(w);
    let edges: Vec<u64> = vec![
        0,
        1,
        2,
        m,
        m - 1,
        0x55,
        0xaa_55,
        m / 2,
        m / 2 + 1,
        0x8000_0001,
    ];
    for &x in &edges {
        for &y in &edges {
            for sx in [Sx::Unsigned, Sx::Signed] {
                let got = model(o, w, sx, x, y);
                let want = reference(o, w, x, y);
                assert_eq!(
                    got, want,
                    "{o:?} {w:?} ({sx:?}) x={x:#x} y={y:#x}: model {got:#x}, machine {want:#x}"
                );
            }
        }
    }
    // Deterministic pseudo-random sweep (xorshift64*: fixed seed, no deps).
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for _ in 0..3000 {
        let (x, y) = (next(), next());
        for sx in [Sx::Unsigned, Sx::Signed] {
            let got = model(o, w, sx, x, y);
            let want = reference(o, w, x, y);
            assert_eq!(
                got, want,
                "{o:?} {w:?} ({sx:?}) x={x:#x} y={y:#x}: model {got:#x}, machine {want:#x}"
            );
        }
    }
}

#[test]
fn add_sub_mul_wrap_at_every_width() {
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        for o in [BinOp::Add, BinOp::Sub, BinOp::Mul] {
            check(o, w);
        }
    }
}

#[test]
fn bitwise_ops_are_bitwise_at_every_width() {
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        for o in [BinOp::And, BinOp::Or, BinOp::Xor] {
            check(o, w);
        }
    }
}

#[test]
fn shl_and_shr_follow_the_machine_at_every_width() {
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        for o in [BinOp::Shl, BinOp::Shr] {
            check(o, w);
        }
    }
}

#[test]
fn sar_is_arithmetic_at_every_width() {
    // The register is modelled UNSIGNED (decision §1.3); `sar` is the op
    // that must read the width-w SIGN anyway. A logical rendering is the
    // exact bug T2's first run caught at the emitter - this test is the
    // same claim aimed at the interpreter.
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        check(BinOp::Sar, w);
    }
}

#[test]
fn sar_of_a_negative_never_loses_its_sign() {
    // A hand-named case, for the reader: 0x80 >> 1 at W8 is 0xC0, not 0x40.
    assert_eq!(model(BinOp::Sar, Width::W8, Sx::Unsigned, 0x80, 1), 0xc0);
    // SAR of INT_MIN by 31 is -1 — the 32-bit pattern 0xFFFFFFFF, not 1.
    // (My first expectation said 1; the machine overruled it. A hand-named
    // case earns its place by being derived, and the derivation said -1.)
    assert_eq!(
        model(BinOp::Sar, Width::W32, Sx::Unsigned, 0x8000_0000, 31),
        0xffff_ffff
    );
    assert_eq!(
        model(BinOp::Sar, Width::W64, Sx::Unsigned, u64::MAX, 63),
        u64::MAX
    );
    // `shr` of the same values is zero-fill - the pair must not be swapped.
    assert_eq!(model(BinOp::Shr, Width::W8, Sx::Unsigned, 0x80, 1), 0x40);
}

#[test]
fn shift_counts_are_machine_masked_not_saturated() {
    // The P-track defect, pinned by name. A count of 64..=128 in the old
    // arms was UB on the underlying u64/i64 shift: debug panicked, release
    // WRAPPED. The machine masks; the interpreter now masks the same way.
    // 32-bit operands mask to 5 bits:
    assert_eq!(model(BinOp::Shl, Width::W32, Sx::Unsigned, 1, 32), 1); // 32 & 31 = 0
    assert_eq!(model(BinOp::Shl, Width::W32, Sx::Unsigned, 1, 33), 2); // 33 & 31 = 1
    assert_eq!(
        model(BinOp::Shr, Width::W32, Sx::Unsigned, 0x8000_0000, 32),
        0x8000_0000
    );
    assert_eq!(
        model(BinOp::Shr, Width::W32, Sx::Unsigned, 0x8000_0000, 33),
        0x4000_0000
    );
    assert_eq!(
        model(BinOp::Sar, Width::W32, Sx::Unsigned, 0x8000_0000, 64),
        0x8000_0000
    );
    assert_eq!(
        model(BinOp::Sar, Width::W32, Sx::Unsigned, 0x8000_0000, 65),
        0xc000_0000
    );
    // 64-bit operands mask to 6 bits:
    assert_eq!(model(BinOp::Shl, Width::W64, Sx::Unsigned, 1, 64), 1); // 64 & 63 = 0
    assert_eq!(
        model(BinOp::Shr, Width::W64, Sx::Unsigned, u64::MAX, 64),
        u64::MAX
    );
    assert_eq!(
        model(BinOp::Sar, Width::W64, Sx::Unsigned, u64::MAX, 128),
        u64::MAX
    );
    // 8/16-bit operands mask to 5 bits like 32 (the x86 convention):
    assert_eq!(model(BinOp::Shl, Width::W8, Sx::Unsigned, 1, 8), 0); // 8 & 31 = 8
    assert_eq!(model(BinOp::Shl, Width::W8, Sx::Unsigned, 1, 32), 1); // 32 & 31 = 0
}

// ============================================================================
// THE ACCUMULATOR OPS — the extension this file's header asked for.
//
// `cdq`+`idiv`, `xor`+`div`, and one-operand `mul`/`imul` landed in
// `lift.rs` (fixtures, and the silicon measurements behind every expectation,
// in `crates/arch-x86/tests/acc_forms.rs`). The header scoped this suite to
// "the set the lifter CAN emit" and said the day an `imul` high form lands,
// this table extends. It did, so it does.
//
// Scope, width by width, is the set the lifter can actually produce:
//
//   MulHi / MulSmul   W64 ONLY. The IR contract is bits 64..127 of the
//                     128-bit product; for a 32-bit `mul` that is the wrong
//                     question (the machine's EDX is bits 32..63), so the
//                     lifter spells the 32-bit pair form as a W64 product plus
//                     a shift, and `emit_c` REFUSES a multiply-high below 64
//                     rather than emitting a zero nobody asked for. The signed
//                     limb identity below is a W64 identity (its correction
//                     terms are 2^64-based), which is the same scope, arrived
//                     at from the other direction.
//   UDiv / URem /
//   SDiv / SRem       every width, divisor non-zero. A zero divisor is a
//                     FAULT, pinned by name below — the machine #DEs, the
//                     interpreter faults, and the emitted C answers 0. That
//                     last leg is a deliberate, named divergence (the C has to
//                     return something to compare), not an accident.
//
// The multiply-high reference is a LIMB DECOMPOSITION in Rust — the same shape
// as the PRELUDE's C helper. That is the point: the interpreter computes
// multiply-high with `u128`/`i128`, this table computes it with limbs, and
// `crates/core/tests/mulhi_helpers.rs` checks the C helper against `__int128`.
// Three independent implementations, one answer, and no `__int128` in the C
// that ships (MSVC has no 128-bit integer type).
// ============================================================================

/// High 64 bits of a 64x64 product, by 32-bit limbs — deliberately NOT the
/// interpreter's `u128` route. Each limb product is < 2^64, so every term is
/// exact and the only wrap is the sum, which is the answer modulo 2^64.
fn mulhi_limbs(x: u64, y: u64) -> u64 {
    let (x0, x1) = (x & 0xFFFF_FFFF, x >> 32);
    let (y0, y1) = (y & 0xFFFF_FFFF, y >> 32);
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 32) + (p01 & 0xFFFF_FFFF) + (p10 & 0xFFFF_FFFF);
    p11.wrapping_add(p01 >> 32)
        .wrapping_add(p10 >> 32)
        .wrapping_add(mid >> 32)
}

fn sext(w: Width, x: u64) -> i64 {
    let x = trunc(w, x);
    if x & sign_bit(w) != 0 {
        (x | !mask(w)) as i64
    } else {
        x as i64
    }
}

/// THE REFERENCE for the accumulator ops, written from the x86 facts.
fn reference_acc(o: BinOp, w: Width, x: u64, y: u64) -> u64 {
    let (x, y) = (trunc(w, x), trunc(w, y));
    match o {
        BinOp::MulHi => trunc(w, mulhi_limbs(x, y)),
        // The signed high half of the same bits. Reading the signed product
        // through unsigned limbs: (a - 2^64)(b - 2^64) = ab - 2^64*b - 2^64*a
        // + 2^128, and the 2^128 term is out of reach of the high 64 bits, so
        // each negative operand costs the other one. Exact at W64, which is the
        // only width this op is emitted at.
        BinOp::MulSmul => {
            let mut h = mulhi_limbs(x, y);
            if x & sign_bit(w) != 0 {
                h = h.wrapping_sub(y);
            }
            if y & sign_bit(w) != 0 {
                h = h.wrapping_sub(x);
            }
            h
        }
        // `div`: the machine's unsigned divide, truncating toward zero — which
        // for unsigned operands is C's `/` exactly.
        BinOp::UDiv => trunc(w, x / y),
        BinOp::URem => trunc(w, x % y),
        // `idiv`: signed, and x86 truncates TOWARD ZERO, the C rule, not the
        // floor rule: -7/2 = -3 remainder -1, never -4 remainder +1.
        BinOp::SDiv => trunc(w, sext(w, x).wrapping_div(sext(w, y)) as u64),
        BinOp::SRem => trunc(w, sext(w, x).wrapping_rem(sext(w, y)) as u64),
        _ => unreachable!("accumulator scope: the ops named above"),
    }
}

fn check_acc(o: BinOp, w: Width) {
    // The domain where the model returns a VALUE. Outside it the model FAULTS,
    // and a fault is pinned by name below, not compared as a number:
    //   y == 0                    -> Fault::DivZero (the machine's #DE)
    //   xs == i64::MIN, ys == -1  -> Fault::IntMinDivMinusOne (also #DE)
    let faulted = |x: u64, y: u64| trunc(w, y) == 0 || (sext(w, x) == i64::MIN && sext(w, y) == -1);
    let m = mask(w);
    let edges: Vec<u64> = vec![
        0,
        1,
        2,
        3,
        m,
        m - 1,
        m / 2,
        m / 2 + 1,
        0x55,
        0xaa_55,
        0x1_0000_0000,
        0xFFFF_FFFF_FFFF_FF01,
    ];
    for &x in &edges {
        for &y in &edges {
            // The guard is on the TRUNCATED divisor: at W8 the edge value
            // 0x1_0000_0000 truncates to 0, and the interpreter faults on that
            // (pinned by `a_zero_divisor_faults_naming_the_divide` below), which
            // is exactly the machine's #DE.
            if faulted(x, y) {
                continue;
            }
            for sx in [Sx::Unsigned, Sx::Signed] {
                let got = model(o, w, sx, x, y);
                let want = reference_acc(o, w, x, y);
                assert_eq!(
                    got, want,
                    "{o:?} {w:?} ({sx:?}) x={x:#x} y={y:#x}: model {got:#x}, machine {want:#x}"
                );
            }
        }
    }
    let mut s: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut next = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for _ in 0..3000 {
        let (x, y) = (next(), next());
        // Same rule as the edges: the op sees the TRUNCATED operands, and a
        // value that is a multiple of 2^w at W8/W16/W32 is a zero divisor.
        if faulted(x, y) {
            continue;
        }
        for sx in [Sx::Unsigned, Sx::Signed] {
            let got = model(o, w, sx, x, y);
            let want = reference_acc(o, w, x, y);
            assert_eq!(
                got, want,
                "{o:?} {w:?} ({sx:?}) x={x:#x} y={y:#x}: model {got:#x}, machine {want:#x}"
            );
        }
    }
}

#[test]
fn multiply_high_matches_a_limb_decomposition_at_64_bits() {
    // W64 only, for the reason in the header: the lifter never emits a
    // multiply-high below 64, and the emitter refuses one if it did.
    for o in [BinOp::MulHi, BinOp::MulSmul] {
        check_acc(o, Width::W64);
    }
}

#[test]
fn division_and_remainder_match_the_machine_at_every_width() {
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        for o in [BinOp::UDiv, BinOp::URem, BinOp::SDiv, BinOp::SRem] {
            check_acc(o, w);
        }
    }
}

#[test]
fn signed_division_truncates_toward_zero_like_idiv() {
    // The hand-named cases, for the reader. `cdq` + `idiv` of -7 by 2 is -3
    // remainder -1 on the machine (measured, in acc_forms.rs); a floor-dividing
    // model answers -4 and +1, and the remainder's SIGN is where the two
    // implementations are told apart.
    assert_eq!(
        model(BinOp::SDiv, Width::W32, Sx::Unsigned, 0xFFFF_FFF9, 2),
        0xFFFF_FFFD
    );
    assert_eq!(
        model(BinOp::SRem, Width::W32, Sx::Unsigned, 0xFFFF_FFF9, 2),
        0xFFFF_FFFF
    );
    // The remainder's sign follows the DIVIDEND (the C99 rule), not the divisor:
    assert_eq!(
        model(BinOp::SRem, Width::W32, Sx::Unsigned, 7, 0xFFFF_FFFE),
        1
    );
    assert_eq!(
        model(BinOp::SDiv, Width::W32, Sx::Unsigned, 7, 0xFFFF_FFFE),
        0xFFFF_FFFD
    );
    // Unsigned on the same bytes is a different answer, and that difference is
    // the ONLY difference between `div` and `idiv`: 0xFFFFFFF9/2 = 0x7FFFFFFC
    // remainder 1 (measured).
    assert_eq!(
        model(BinOp::UDiv, Width::W32, Sx::Unsigned, 0xFFFF_FFF9, 2),
        0x7FFF_FFFC
    );
    assert_eq!(
        model(BinOp::URem, Width::W32, Sx::Unsigned, 0xFFFF_FFF9, 2),
        1
    );
    // INT_MIN / -1: the machine raises #DE at every width. At W64 the model
    // FAULTS (`int_min_over_minus_one_faults_at_64_and_wraps_below_it`), and
    // below 64 it answers the wrapped quotient — the value the emitted C's
    // guard spells. Pinned in both directions there, so a future "fix" that
    // faults at the narrow widths is a decision with a red test to turn green.
    assert_eq!(
        model(
            BinOp::SDiv,
            Width::W32,
            Sx::Unsigned,
            0x8000_0000,
            0xFFFF_FFFF
        ),
        0x8000_0000
    );
    assert_eq!(
        model(
            BinOp::SRem,
            Width::W32,
            Sx::Unsigned,
            0x8000_0000,
            0xFFFF_FFFF
        ),
        0
    );
}

#[test]
fn a_zero_divisor_faults_naming_the_divide() {
    // The machine raises #DE. The model's answer is a FAULT, not a number: an
    // interpreter that answered 0 here would make every guarded division in the
    // corpus agree with the emitted C's `y == 0 ? 0` and agree with the machine
    // on nothing.
    use rdec_core::interp::{run, Fault};
    for o in [BinOp::UDiv, BinOp::URem, BinOp::SDiv, BinOp::SRem] {
        for w in [Width::W32, Width::W64] {
            let mut f = Function::new("prop", 0);
            let e = f.entry;
            let a = f.emit(
                e,
                Op::Param,
                Ty::Int {
                    sx: Sx::Unsigned,
                    w,
                },
                0,
            );
            let b = f.emit(
                e,
                Op::Param,
                Ty::Int {
                    sx: Sx::Unsigned,
                    w,
                },
                0,
            );
            let r = f.emit(
                e,
                Op::Bin(o, a, b),
                Ty::Int {
                    sx: Sx::Unsigned,
                    w,
                },
                0,
            );
            f.block_mut(e).term = Terminator::Return(r);
            f.params.extend([a, b]);
            f.ret_ty = Ty::Int {
                sx: Sx::Unsigned,
                w,
            };
            match run(&f, &[7, 0, 0, 0]).outcome {
                Outcome::Fault(Fault::DivZero) => {}
                other => panic!("{o:?} {w:?} x=7 y=0: interpreter said {other:?}"),
            }
        }
    }
}

#[test]
fn int_min_over_minus_one_faults_at_64_and_wraps_below_it() {
    // The machine raises #DE at EVERY width here: the quotient +2^(w-1) does
    // not fit the destination. The model's guard is
    // `xs == i64::MIN && ys == -1` — a W64-shaped test — so:
    //
    //   W64: the model FAULTS (the honest answer, and the emitted C's
    //        `y == -1 ? 0 - x` guard spells the same wrapped value the C
    //        returns, which the differential never reaches because the IR leg
    //        faults and the pair is not compared).
    //   W8/W16/W32: the model answers the WRAPPED quotient. That agrees with the
    //        emitted C's guard and DISAGREES with the machine, which traps.
    //
    // Pinned in both directions so "quotient overflow is not modelled" stays a
    // named M4 gap with a red test to turn green, not a quiet asymmetry.
    use rdec_core::interp::{run, Fault};
    for o in [BinOp::SDiv, BinOp::SRem] {
        let mut f = Function::new("prop", 0);
        let e = f.entry;
        let ty = Ty::Int {
            sx: Sx::Unsigned,
            w: Width::W64,
        };
        let a = f.emit(e, Op::Param, ty.clone(), 0);
        let b = f.emit(e, Op::Param, ty.clone(), 0);
        let r = f.emit(e, Op::Bin(o, a, b), ty.clone(), 0);
        f.block_mut(e).term = Terminator::Return(r);
        f.params.extend([a, b]);
        f.ret_ty = ty;
        match run(&f, &[1 << 63, u64::MAX, 0, 0]).outcome {
            Outcome::Fault(Fault::IntMinDivMinusOne) => {}
            other => panic!("{o:?} W64 INT_MIN/-1: interpreter said {other:?}"),
        }
    }
    // The narrow leg: the model answers the wrapped quotient (the machine traps).
    assert_eq!(
        model(
            BinOp::SDiv,
            Width::W32,
            Sx::Unsigned,
            0x8000_0000,
            0xFFFF_FFFF
        ),
        0x8000_0000
    );
    assert_eq!(
        model(BinOp::SDiv, Width::W16, Sx::Unsigned, 0x8000, 0xFFFF),
        0x8000
    );
    assert_eq!(
        model(BinOp::SDiv, Width::W8, Sx::Unsigned, 0x80, 0xFF),
        0x80
    );
}

#[test]
fn multiply_high_agrees_with_the_prelude_helpers_and_int128() {
    // The triangle's third corner lives in
    // crates/core/tests/mulhi_helpers.rs (the C helper vs `__int128`, compiled
    // and run). This is the cheap leg: the table's limb reference must agree
    // with the `u128` route on the boundary set, so the two Rust readings cannot
    // drift apart silently while the C leg is green.
    let edges = [
        0u64,
        1,
        2,
        u64::from(u32::MAX),
        1 << 32,
        1 << 63,
        (1 << 63) - 1,
        u64::MAX,
        0x00FF_00FF_00FF_00FF,
    ];
    for &x in &edges {
        for &y in &edges {
            assert_eq!(
                mulhi_limbs(x, y),
                ((x as u128 * y as u128) >> 64) as u64,
                "limbs vs u128 at x={x:#x} y={y:#x}"
            );
            let sm = {
                let mut h = mulhi_limbs(x, y);
                if (x as i64) < 0 {
                    h = h.wrapping_sub(y);
                }
                if (y as i64) < 0 {
                    h = h.wrapping_sub(x);
                }
                h
            };
            assert_eq!(
                sm,
                (((x as i64) as i128 * (y as i64) as i128) >> 64) as u64,
                "signed limbs vs i128 at x={x:#x} y={y:#x}"
            );
        }
    }
}
