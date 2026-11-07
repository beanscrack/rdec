//! Shared fixture builders for the M6 type-recovery rules tests.
//!
//! Hand-built IR, on the `width.rs` builder pattern: parameters are the W64
//! register cells the lifter produces, the body reshapes them with casts (the
//! `movsx`/`movzx` shape), and stack slots are addressed off a `%rsp`-style
//! constant base (the frame model: the slot base is not a parameter cell).

#![allow(dead_code)]

use rdec_core::ir::{BinOp, CastKind, FBinOp, Function, IntCmp, Op, Terminator, VId, NO_VAL};
use rdec_core::ty::{FWidth, Ty, Width};

pub fn u32() -> Ty {
    Ty::uint(Width::W32)
}
pub fn u64() -> Ty {
    Ty::uint(Width::W64)
}
pub fn f32() -> Ty {
    Ty::Float { w: FWidth::F32 }
}
pub fn f64() -> Ty {
    Ty::Float { w: FWidth::F64 }
}

/// A function with `n` W64 parameter cells, declared as parameters.
pub fn cells(n: usize) -> (Function, Vec<VId>) {
    let mut f = Function::new("probe", 0);
    let e = f.entry;
    let mut cs = Vec::new();
    for i in 0..n {
        cs.push(f.emit(e, Op::Param, u64(), (i * 8) as u64));
    }
    f.params = cs.clone();
    (f, cs)
}

/// A `%rsp`-style frame base: a W64 constant, an address root that is not a
/// parameter (the point of the slot fixtures).
pub fn rsp(f: &mut Function) -> VId {
    f.emit(f.entry, Op::Const(0x1000), u64(), 0)
}

pub fn ret(f: &mut Function, v: VId) {
    f.block_mut(f.entry).term = Terminator::Return(v);
}
pub fn ret_void(f: &mut Function) {
    f.block_mut(f.entry).term = Terminator::Return(NO_VAL);
}

pub fn trunc(f: &mut Function, x: VId, w: Width) -> VId {
    let t = Ty::uint(w);
    f.emit(f.entry, Op::Cast(CastKind::Trunc, t.clone(), x), t, 0)
}
pub fn sext(f: &mut Function, x: VId) -> VId {
    f.emit(f.entry, Op::Cast(CastKind::SExt, u64(), x), u64(), 0)
}
pub fn zext(f: &mut Function, x: VId) -> VId {
    f.emit(f.entry, Op::Cast(CastKind::ZExt, u64(), x), u64(), 0)
}
pub fn cast(f: &mut Function, k: CastKind, to: Ty, x: VId) -> VId {
    f.emit(f.entry, Op::Cast(k, to.clone(), x), to, 0)
}
pub fn bin(f: &mut Function, op: BinOp, w: Width, x: VId, y: VId) -> VId {
    f.emit(f.entry, Op::Bin(op, x, y), Ty::uint(w), 0)
}
pub fn cmp(f: &mut Function, c: IntCmp, x: VId, y: VId) -> VId {
    f.emit(f.entry, Op::Cmp(c, x, y), Ty::Bool, 0)
}
pub fn fbin(f: &mut Function, w: FWidth, x: VId, y: VId) -> VId {
    let ty = Ty::Float { w };
    f.emit(f.entry, Op::FBin(FBinOp::Add, x, y), ty, 0)
}
pub fn konst_f32(f: &mut Function, v: f64) -> VId {
    f.emit(f.entry, Op::ConstF(v), f32(), 0)
}
pub fn k(f: &mut Function, v: u64, w: Width) -> VId {
    f.emit(f.entry, Op::Const(v), Ty::uint(w), 0)
}
pub fn load(f: &mut Function, ty: Ty, at: VId) -> VId {
    f.emit(f.entry, Op::Load(ty.clone(), at), ty, 0)
}
pub fn store(f: &mut Function, ty: Ty, at: VId, v: VId) {
    f.emit_void(f.entry, Op::Store(ty, at, v), 0);
}
pub fn phi(f: &mut Function, ty: Ty, pairs: Vec<(VId, rdec_core::ir::BbId)>) -> VId {
    f.emit(f.entry, Op::Phi(pairs.into_boxed_slice()), ty, 0)
}
pub fn call(f: &mut Function, args: Vec<VId>) -> VId {
    use rdec_core::ir::Callee;
    f.emit(
        f.entry,
        Op::Call {
            callee: Callee::Addr(0x4000),
            ret_ty: u64(),
            args: args.into_boxed_slice(),
            pure: false,
        },
        u64(),
        0,
    )
}
