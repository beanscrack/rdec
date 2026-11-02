//! How many bits a value can need, read off the IR.
//!
//! The question this answers is one the return type has to answer: when a
//! function's answer reaches `RAX`, which bits of `RAX` can be set? On x86-64 a
//! 32-bit register write ZEROES bits 63:32 architecturally (the lifter models
//! that as `zext`), so a function whose answer reaches `RAX` through such a
//! write has an answer that is exactly `zero_extend_32(value)` - bits 63:32 are
//! zero on every path, not "undefined", not "the caller must not look". Emitting
//! `int64_t` for that function and returning a sign-extended 32-bit value is a
//! real over-extension in the OUTPUT, and T2 measures it: saturation's
//! `f(1,-100)` gives machine `RAX = 0xffffff9d` against the emitted C's
//! `0xffffffffffffff9d`, 39 of 732 runs.
//!
//! So the width is derived from the machine code, not from the observed answers.
//! (An earlier attempt inferred the compare width from the answers themselves
//! and collapsed 19 functions to a 1-bit gate that can never fail - a gate that
//! cannot fail is not a gate.)
//!
//! [`bits_needed`] is a *sound under-approximation*: the answer is the fewest
//! bits that provably hold the value, and the analysis answers 64 ("any u64")
//! the moment it cannot prove less. Every rule is one-directional, so the only
//! failure mode is "less knowledge than possible", which costs a wider return
//! type and can never claim a bit is zero that the machine sets.
//!
//! Where a value is cyclic (a loop-carried phi), the answer is 64: proving
//! anything about a value that flows through itself needs a real invariant, not
//! an optimistic first pass.

use crate::ir::{BinOp, CastKind, Function, Op, Terminator, UnOp, VId};
use std::collections::{HashMap, HashSet};

/// The fewest bits that provably hold `%v`'s value: `0` = provably zero,
/// `64` = any 64-bit pattern is possible. See the module comment.
pub fn bits_needed(f: &Function, v: VId) -> u32 {
    let mut memo: HashMap<VId, u32> = HashMap::new();
    let mut path: HashSet<VId> = HashSet::new();
    rec(f, v, &mut memo, &mut path)
}

/// The width at which this function's answer reaches `RAX`, as a machine fact:
/// `32` when every returned value provably fits in 32 bits - the observable
/// consequence of RAX having been written at 32-bit width, which zeroes the
/// upper half - and `64` otherwise. `64` is also the answer for a function with
/// no `Return` at all: no width was ever established, so nothing is claimed.
pub fn return_width(f: &Function) -> u32 {
    let mut need = 0u32;
    for b in &f.blocks {
        if let Terminator::Return(v) = &b.term {
            if *v != crate::ir::NO_VAL {
                need = need.max(bits_needed(f, *v));
            }
        }
    }
    if need > 0 && need <= 32 {
        32
    } else {
        64
    }
}

fn rec(f: &Function, v: VId, memo: &mut HashMap<VId, u32>, path: &mut HashSet<VId>) -> u32 {
    if let Some(&m) = memo.get(&v) {
        return m;
    }
    if path.contains(&v) {
        // A cycle: a loop-carried phi, whose incoming value flows back through
        // itself. No fixed point is computed, so no knowledge is claimed.
        return 64;
    }
    path.insert(v);
    let m = one(f, v, memo, path);
    path.remove(&v);
    memo.insert(v, m);
    m
}

/// The width of `%v`'s own type: a value cannot need more bits than it has.
fn cap(f: &Function, v: VId) -> u32 {
    f.ty_of(v)
        .and_then(|t| t.width())
        .map(|w| w.bits())
        .unwrap_or(64)
}

fn bits_of_ty(t: &crate::ty::Ty) -> u32 {
    t.width().map(|w| w.bits()).unwrap_or(64)
}

fn one(f: &Function, v: VId, memo: &mut HashMap<VId, u32>, path: &mut HashSet<VId>) -> u32 {
    let mut go = |x: VId| rec(f, x, memo, path);
    let Some((bb, ix)) = f.def(v) else {
        return 64; // not defined in this function: no claim
    };
    let Some(ins) = f.block(bb).instrs.get(ix) else {
        return 64;
    };
    let w = cap(f, v);
    let raw: u32 = match &ins.op {
        // A constant's own bit pattern is the whole truth about it.
        Op::Const(c) => 64 - c.leading_zeros(),
        Op::Copy(x) => go(*x),
        // Anything read from outside the IR - a parameter, memory, a call, an
        // unmodelled op, an undefined register - is as wide as its type.
        Op::Param | Op::Undef | Op::Opaque(_) | Op::Load(..) | Op::Call { .. } => w,
        Op::ConstF(_) | Op::Flags(_) => w,
        // A float op's result is as wide as its type says (64 for F64, 32 for
        // F32, 1 for a comparison). It is NOT the width of the integer the cast
        // to int will produce -- that question is the cast's own arm below.
        Op::FBin(..) | Op::FCmp(..) => w,
        Op::Cmp(..) | Op::GetFlag(..) => w, // Bool: 1 bit, via the cap
        Op::Cast(k, to, x) => {
            let m = go(*x);
            let tw = bits_of_ty(to);
            match k {
                // Truncation and same-width bitcasts cannot need more bits than
                // the destination has. `Bitcast` is same-width by construction
                // (see `CastKind`), so the min is a guard, not a behaviour.
                CastKind::Trunc | CastKind::Bitcast => m.min(tw),
                // A zero extension keeps the unsigned value: the bits it adds
                // are the zeros this analysis is about.
                CastKind::ZExt => m,
                // A sign extension duplicates the sign bit upward, so it needs
                // every bit UNLESS the sign bit is provably zero - and a value
                // that fits in fewer bits than the source width has exactly
                // that property.
                CastKind::SExt => {
                    let sw = f.ty_of(*x).map(|t| bits_of_ty(&t)).unwrap_or(64);
                    if m < sw {
                        m
                    } else {
                        64
                    }
                }
                // Floats: the bit pattern is data, and a float<->int conversion
                // has a range the IR does not model.
                CastKind::SIToFp | CastKind::UIToFp | CastKind::FPExt | CastKind::FPTrunc => 64,
                CastKind::FpToSI | CastKind::FpToUI => tw,
            }
        }
        Op::Bin(op, a, b) => {
            let (ma, mb) = (go(*a), go(*b));
            match op {
                // One bit for the carry.
                BinOp::Add => ma.max(mb).saturating_add(1),
                // `0 - 1` is all ones: a borrow fills the top.
                BinOp::Sub => 64,
                BinOp::Mul => ma.saturating_add(mb),
                BinOp::MulHi | BinOp::MulSmul => w,
                // Unsigned division: the quotient is not larger than the
                // dividend, the remainder is smaller than the divisor.
                BinOp::UDiv => ma,
                BinOp::URem => ma.min(mb),
                // Signed results can be negative, i.e. every top bit set.
                BinOp::SDiv | BinOp::SRem => w,
                BinOp::And => ma.min(mb),
                BinOp::Or | BinOp::Xor => ma.max(mb),
                // The shift count is the x86-masked one (width mod 64/32...), so
                // at most w-1 bits can be shifted up.
                BinOp::Shl => ma.saturating_add(w.saturating_sub(1)),
                // A logical right shift only moves bits down.
                BinOp::Shr => ma,
                // An arithmetic shift copies the sign bit up.
                BinOp::Sar => w,
                // A rotate can move the low bit to the top.
                BinOp::RotL | BinOp::RotR => w,
            }
        }
        Op::Un(op, _) => match op {
            // `~x` and `-x` can set every top bit; a count of at most `w` things
            // needs log2(w) bits; a (float) sqrt is data. All of it is bounded by
            // the result's own width, so no operand knowledge is needed.
            UnOp::Not | UnOp::Neg | UnOp::Sqrt => w,
            UnOp::Popcount | UnOp::Lzcnt | UnOp::Tzcnt => 64 - w.max(1).leading_zeros(),
        },
        // Both arms share a type (verify_ssa checks), and the result is one of
        // them, so the wider claim is the sound one.
        Op::Select(_, t, e) => go(*t).max(go(*e)),
        // A merge is only as narrow as its WIDEST incoming edge: every path has
        // to hold, and one path that can set bit 63 forces a 64-bit answer.
        // Cyclic inputs answer 64 (see `rec`).
        Op::Phi(pairs) => pairs.iter().map(|(x, _)| go(*x)).max().unwrap_or(0),
        Op::Store(..) => 0,
    };
    raw.min(w).min(64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Instr, IntCmp, Terminator};
    use crate::ty::{Sx, Ty, Width};

    fn int32() -> Ty {
        Ty::uint(Width::W32)
    }
    fn int64() -> Ty {
        Ty::uint(Width::W64)
    }

    /// A 32-bit add, zero-extended by the 32-bit register write the lifter
    /// models: bits 63:32 are zero on every path, so the answer is 32 bits.
    #[test]
    fn a_32_bit_write_zeroes_the_upper_half_and_that_is_a_width() {
        let mut f = Function::new("add32", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, int32(), 0);
        let b = f.emit(e, Op::Param, int32(), 4);
        f.params = vec![a, b];
        let s = f.emit(e, Op::Bin(BinOp::Add, a, b), int32(), 8);
        let w = f.emit(e, Op::Cast(CastKind::ZExt, int64(), s), int64(), 12);
        f.block_mut(e).term = Terminator::Return(w);
        assert_eq!(bits_needed(&f, s), 32, "an int32 sum fits in 32 bits");
        assert_eq!(bits_needed(&f, w), 32, "zero extension adds zeros");
        assert_eq!(return_width(&f), 32);
    }

    /// The same shape at 64 bits: nothing is known about the top half, so the
    /// return type stays 64 bits wide.
    #[test]
    fn a_64_bit_subtraction_needs_every_bit() {
        let mut f = Function::new("sub64", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, int64(), 0);
        let b = f.emit(e, Op::Param, int64(), 8);
        f.params = vec![a, b];
        let s = f.emit(e, Op::Bin(BinOp::Sub, a, b), int64(), 16);
        f.block_mut(e).term = Terminator::Return(s);
        assert_eq!(bits_needed(&f, s), 64, "0 - 1 is all ones");
        assert_eq!(return_width(&f), 64);
    }

    /// The saturation case: a cmov-style select between a 32-bit clamp
    /// constant and a 32-bit sum merges at 32 bits on both edges.
    #[test]
    fn a_merge_of_two_32_bit_edges_is_32_bits() {
        let mut f = Function::new("sat", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, int32(), 0);
        f.params = vec![a];
        let one = f.emit(e, Op::Const(1), int32(), 4);
        let s = f.emit(e, Op::Bin(BinOp::Add, a, one), int32(), 8);
        let hi = f.emit(e, Op::Const(0x7fff_ffff), int32(), 12);
        let c = f.emit(e, Op::Cmp(IntCmp::Ult, s, hi), Ty::Bool, 16);
        let sel = f.emit(e, Op::Select(c, s, hi), int32(), 20);
        let w = f.emit(e, Op::Cast(CastKind::ZExt, int64(), sel), int64(), 24);
        f.block_mut(e).term = Terminator::Return(w);
        assert_eq!(bits_needed(&f, w), 32);
        assert_eq!(return_width(&f), 32);

        // A phi merges at its WIDEST incoming edge, not its narrowest: one path
        // that can set bit 63 forces a 64-bit answer.
        let mut g = Function::new("sat64", 0);
        let e = g.entry;
        let a = g.emit(e, Op::Param, int64(), 0);
        g.params = vec![a];
        let c = g.emit(e, Op::Cmp(IntCmp::Ult, a, a), Ty::Bool, 4);
        let _ = c;
        let w32 = g.emit(e, Op::Cast(CastKind::Trunc, int32(), a), int32(), 8);
        let up = g.emit(e, Op::Cast(CastKind::ZExt, int64(), w32), int64(), 12);
        let b = g.new_block(16);
        g.block_mut(e).term = Terminator::Jump(b);
        let phi = g.new_value(int64());
        g.block_mut(b).instrs.insert(
            0,
            Instr::new(
                phi,
                Op::Phi(vec![(up, e), (a, e)].into_boxed_slice()),
                int64(),
                16,
            ),
        );
        g.block_mut(b).term = Terminator::Return(phi);
        assert_eq!(bits_needed(&g, phi), 64, "the 64-bit edge is the answer");
        assert_eq!(return_width(&g), 64);
    }

    /// A loop-carried phi is a cycle: no fixed point is computed, so nothing is
    /// claimed. A first-visit-wins analysis that read the phi as narrow would
    /// hand back a 32-bit return type for a counter loop that overflows, and a
    /// naive recursion on this shape does not terminate at all.
    #[test]
    fn a_cycle_claims_nothing() {
        let mut f = Function::new("loop", 0);
        let e = f.entry;
        let z = f.emit(e, Op::Const(7), int64(), 0);
        let b = f.new_block(8);
        f.block_mut(e).term = Terminator::Jump(b);
        // The register this block reads is its own live-in: the back edge
        // carries the phi to itself (the shape a loop header with no def in the
        // latch produces).
        let phi = f.new_value(int64());
        f.block_mut(b).instrs.insert(
            0,
            Instr::new(
                phi,
                Op::Phi(vec![(z, e), (phi, b)].into_boxed_slice()),
                int64(),
                8,
            ),
        );
        f.block_mut(b).term = Terminator::Return(phi);
        assert_eq!(bits_needed(&f, phi), 64, "a cycle is not proved narrow");
        assert_eq!(return_width(&f), 64);
    }

    /// Sign extension is the case the analysis must not get optimistic about:
    /// a value with a proven-zero sign bit extends like a zext, and one whose
    /// sign bit is data needs all 64 bits.
    #[test]
    fn sign_extension_is_only_free_when_the_sign_bit_is_provably_zero() {
        let mut f = Function::new("sx", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, int32(), 0);
        f.params = vec![a];
        // (a & 0xffff) has bit 31 provably zero.
        let m = f.emit(e, Op::Const(0xffff), int32(), 4);
        let masked = f.emit(e, Op::Bin(BinOp::And, a, m), int32(), 8);
        let signed = f.emit(
            e,
            Op::Cast(CastKind::Bitcast, Ty::int(Width::W32, Sx::Signed), masked),
            Ty::int(Width::W32, Sx::Signed),
            12,
        );
        let up = f.emit(e, Op::Cast(CastKind::SExt, int64(), signed), int64(), 16);
        f.block_mut(e).term = Terminator::Return(up);
        assert_eq!(bits_needed(&f, masked), 16, "an AND mask is a width");
        assert_eq!(bits_needed(&f, up), 16, "sext of a proven non-negative");

        // The same shape on an unmasked value: the sign bit is data.
        let mut g = Function::new("sx2", 0);
        let e = g.entry;
        let a = g.emit(e, Op::Param, int32(), 0);
        g.params = vec![a];
        let signed = g.emit(
            e,
            Op::Cast(CastKind::Bitcast, Ty::int(Width::W32, Sx::Signed), a),
            Ty::int(Width::W32, Sx::Signed),
            4,
        );
        let up = g.emit(e, Op::Cast(CastKind::SExt, int64(), signed), int64(), 8);
        g.block_mut(e).term = Terminator::Return(up);
        assert_eq!(bits_needed(&g, up), 64, "a signed 32-bit value is any i64");
    }

    /// A function that never returns establishes no width at all.
    #[test]
    fn no_return_establishes_no_width() {
        let f = Function::new("noreturn", 0);
        assert_eq!(return_width(&f), 64);
    }
}
