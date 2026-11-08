//! FOLDING MUST NOT CHANGE MEANING.
//!
//! Every other test in the repo that calls `const_fold` calls it as a *step of a
//! pipeline* — lift, fold, run, check the answer against a hand-named
//! expectation. That shape cannot see a folder that is wrong in the same way as
//! the rest of the pipeline: when the folder rewrites `Const` through `Const`,
//! the folded program is what the expectation was written against, and the
//! unfolded program that the machine implements is never run.
//!
//! This file runs BOTH programs on the same inputs and requires the same answer,
//! over every cast shape the IR can name at every width pair, on vectors chosen
//! at the width boundaries (0x80, 0x8000, 0x80000000, 0x8000000000000000) where a
//! sign bit is about to be read from the wrong place.
//!
//! It earned its place on a real defect: `const_cast` sign-extended from the
//! DESTINATION width. An SExt destination is wider by construction, so bit
//! `dst-1` of a value living in the low `src` bits is always 0, and every folded
//! sign-extension of a constant was a zero-extension. `mov eax,0x12348123 ; cwde`
//! folded to `0x8123`; the machine answers `0xFFFF8123`; the same function with
//! the value arriving in ECX — unfolded — was always correct. Absolute values
//! against silicon live in `crates/arch-x86/tests/acc_forms.rs`
//! (`cwde_sign_extends_ax_alone_and_zeroes_the_upper_half`): an agreement test
//! between two of our own programs is satisfied by both of them being wrong, so
//! this file is the breadth and that file is the truth.
use rdec_core::interp::{run_budget, Outcome};
use rdec_core::ir::CastKind;
use rdec_core::passes::const_fold;
use rdec_core::ty::{Sx, Ty, Width};
use rdec_core::verify::verify_ssa;

mod common;
use common::{cast, cells, k, ret, trunc};

fn si(w: Width) -> Ty {
    Ty::int(w, Sx::Signed)
}

fn unsigned(w: Width) -> Ty {
    Ty::int(w, Sx::Unsigned)
}

/// Every cast shape the folder touches: widening SExt/ZExt, narrowing Trunc, and
/// same-width Bitcast (the signedness relabel `movsx` emits). Shapes are
/// enumerated, not filtered by a verifier, so a rejected shape is a loud test
/// failure and not a silently skipped cell.
fn shapes() -> Vec<(&'static str, CastKind, Width, Width)> {
    use Width::*;
    let mut v = Vec::new();
    for sw in [W8, W16, W32] {
        for dw in [W16, W32, W64] {
            if dw.bits() > sw.bits() {
                v.push(("sext", CastKind::SExt, sw, dw));
                v.push(("zext", CastKind::ZExt, sw, dw));
            }
        }
    }
    for sw in [W16, W32, W64] {
        for dw in [W8, W16, W32] {
            if dw.bits() < sw.bits() {
                v.push(("trunc", CastKind::Trunc, sw, dw));
            }
        }
    }
    for w in [W8, W16, W32, W64] {
        v.push(("bitcast", CastKind::Bitcast, w, w));
    }
    v
}

/// Values at and around every sign bit the sweep can ask about.
fn vectors() -> Vec<u64> {
    vec![
        0,
        1,
        0x7f,
        0x80,
        0xff,
        0x7fff,
        0x8000,
        0xffff,
        0x7fffffff,
        0x8000_0000,
        0xffff_ffff,
        0x1234_8123,
        0xffff_0005,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        0xffff_ffff_ffff_ffff,
    ]
}

/// `%c = Const(x) ; %s = Trunc(%c -> si(sw)) ; %m = Kind(%s -> T(dw)) ;
///  %r = Trunc(%m -> u64) ; ret %r` — one cast, one answer, no registers, so the
/// ONLY thing that can move the answer is the folder.
fn program(x: u64, kind: CastKind, sw: Width, dw: Width) -> rdec_core::ir::Function {
    let (mut f, _cells) = cells(1);
    let cv = k(&mut f, x, Width::W64);
    let src = cast(&mut f, CastKind::Trunc, si(sw), cv);
    let dst = if kind == CastKind::SExt {
        si(dw)
    } else {
        unsigned(dw)
    };
    let mid = cast(&mut f, kind, dst, src);
    let out = trunc(&mut f, mid, Width::W64);
    ret(&mut f, out);
    f
}

#[test]
fn a_folded_cast_answers_what_the_unfolded_cast_answers() {
    let mut cells_checked = 0usize;
    let mut folded_away = 0usize;
    for (name, kind, sw, dw) in shapes() {
        for x in vectors() {
            let f = program(x, kind, sw, dw);
            verify_ssa(&f).unwrap_or_else(|e| panic!("{name} {sw:?}->{dw:?} of {x:#x}: {e:?}"));
            let unfolded = run_budget(&f, &[0, 0, 0, 0], 20000).outcome;

            let mut g = program(x, kind, sw, dw);
            const_fold(&mut g);
            let folded = run_budget(&g, &[0, 0, 0, 0], 20000).outcome;

            // The sweep must actually fold: if the folder leaves the cast in
            // place, "agreement" is the vacuous equality of a program with
            // itself, which is the failure mode this file exists to avoid.
            let still_cast = g.blocks.iter().any(|b| {
                b.instrs
                    .iter()
                    .any(|i| matches!(i.op, rdec_core::ir::Op::Cast(k, _, _) if k == kind))
            });
            if !still_cast {
                folded_away += 1;
            }

            cells_checked += 1;
            match (unfolded, folded) {
                (Outcome::Ret(a), Outcome::Ret(b)) => assert_eq!(
                    a.bits(),
                    b.bits(),
                    "{name} {sw:?}->{dw:?} of {x:#x}: the folder changed the answer"
                ),
                (u, g) => panic!("{name} {sw:?}->{dw:?} of {x:#x}: unfolded {u:?}, folded {g:?}"),
            }
        }
    }
    // Anti-vacuity: the sweep is 22 shapes x 16 vectors. If the shape list is
    // ever narrowed to nothing, this file measures nothing and must say so.
    assert!(cells_checked >= 300, "only {cells_checked} cells swept");
    assert!(
        folded_away >= cells_checked * 9 / 10,
        "the folder rewrote only {folded_away}/{cells_checked} programs — it is \
         not being exercised, so agreement proves nothing"
    );
}
