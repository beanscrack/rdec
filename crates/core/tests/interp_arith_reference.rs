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
