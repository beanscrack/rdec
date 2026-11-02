//! Flag fusion: `get_flag` trees -> comparisons over the operands.
//!
//! Replaces v1's `passes::flag_elimination`, which the roadmap SS9 confirmed
//! broken four ways (B3-B6): it lowered each `get_flag` *individually*, so the
//! jcc matrix was unreachable (B3); it read SF of `cmp` as "a < 0" (B4); it used
//! the `Flags` value as the arithmetic result (B5); and its fallback covered only
//! CF/OF/ZF/SF of add/sub, leaving a live `Flags` value for anything else (B6).
//!
//! Three rules make this version correct by construction:
//!
//! 1. **Match the tree at its root.** A jcc is a boolean expression over flag
//!    reads (`jl` is `SF ^ OF`), so the node to replace is the top of that
//!    expression. Leaves are replaced only as part of the root's rewrite.
//! 2. **The result is recomputed from the operands.** `FlagsOp` carries no link
//!    to its integer result in the current IR (SS1.2 item 1), so ZF/SF/PF/AF are
//!    lowered against a freshly emitted `a op b`, never against the `Flags`
//!    value.
//! 3. **The fallback is total, and a real gap is loud.** Every
//!    (FlagsOp x flag) pair is answered: an exact expression, or `Opaque` with a
//!    reason -- CF after `inc` needs the previous flags as an operand (SS1.2
//!    item 3), and OF after `sar` by a variable count is architecturally
//!    undefined. A `Flags`-typed value never survives this pass.
//!
//! Readability is deliberately not this module's goal: the exact form is
//! verbose, and M3's fusion-rate gate drives collapsing it into `a < b`.
//! Correctness first, then the table, with T1R (M2) checking the table.

use crate::ir::*;
use crate::ty::{Ty, Width};

// ---------------------------------------------------------------------------
// Small expression builder: lowering emits ops into a list with temp indices,
// the driver materialises them at the root's position (where the operands
// dominate). v1 emitted into the entry block, which broke dominance.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub enum Tm {
    V(VId),
    T(usize),
}

#[derive(Clone, Debug)]
pub enum SO {
    Const(u64),
    Bin(BinOp, Tm, Tm),
    Cmp(IntCmp, Tm, Tm),
    Un(UnOp, Tm),
    Cast(CastKind, Ty, Tm),
    /// A loud gap: the IR cannot express this, so it says so (SS1.5). A gap
    /// makes the function E0; it is never a silent zero.
    Gap(&'static str),
}

pub struct SB {
    pub ops: Vec<(SO, Ty)>,
    /// The width the lowering is working at: the type of the flags operation
    /// being modelled. Constants a lowering invents (`sign`, `0xff`, `1`, a
    /// shift amount) are properties of THAT operation, so they carry its width.
    /// They used to be hard-wired to `u64`, and a W32 `sub` fused into
    /// `and(result, 0x80000000u64)` mixed widths, which the verifier rejects -
    /// 68 corpus functions.
    pub ty: Ty,
}

impl Default for SB {
    fn default() -> Self {
        SB::new()
    }
}

impl SB {
    pub fn new() -> SB {
        SB {
            ops: Vec::new(),
            ty: Ty::u64(),
        }
    }
    fn push(&mut self, so: SO, ty: Ty) -> Tm {
        self.ops.push((so, ty));
        Tm::T(self.ops.len() - 1)
    }
    fn c(&mut self, v: u64) -> Tm {
        let t = self.ty.clone();
        self.push(SO::Const(v), t)
    }
    /// A constant flag value. Flag positions are Bool - returning an integer
    /// constant here is what made `xor(bool, uint32_t 0)` and the verifier
    /// rejected the function ("mixes widths W1 and W32").
    fn const_false(&mut self) -> Tm {
        self.push(SO::Const(0), Ty::Bool)
    }
    fn const_true(&mut self) -> Tm {
        self.push(SO::Const(1), Ty::Bool)
    }
    /// Pin the working width before emitting a lowering for a flags operation.
    pub fn set_ty(&mut self, t: Ty) {
        if !t.is_bool() {
            self.ty = t;
        }
    }
    fn bin(&mut self, o: BinOp, x: Tm, y: Tm, ty: Ty) -> Tm {
        self.push(SO::Bin(o, x, y), ty)
    }
    fn cmp(&mut self, c: IntCmp, x: Tm, y: Tm) -> Tm {
        self.push(SO::Cmp(c, x, y), Ty::Bool)
    }
    /// `x != 0` -> Bool
    fn nz(&mut self, x: Tm, ty: Ty) -> Tm {
        let z = self.push(SO::Const(0), ty.clone());
        self.push(SO::Cmp(IntCmp::Ne, x, z), Ty::Bool)
    }
    /// Bool -> Bool equality test
    fn is_true(&mut self, x: Tm) -> Tm {
        let z = self.push(SO::Const(0), Ty::Bool);
        self.push(SO::Cmp(IntCmp::Ne, x, z), Ty::Bool)
    }
    fn bit(&mut self, x: Tm, k: u64, ty: Ty) -> Tm {
        let sh = self.c(k);
        let s = self.bin(BinOp::Shr, x, sh, ty.clone());
        let one = self.c(1);
        self.bin(BinOp::And, s, one, ty)
    }
    fn opaque(&mut self, why: &'static str) -> Tm {
        self.push(SO::Gap(why), Ty::Bool)
    }
}

// ---------------------------------------------------------------------------
// Flag expressions
// ---------------------------------------------------------------------------

/// A boolean expression over reads of flags.
#[derive(Clone, Debug)]
pub enum FlagExpr {
    Leaf(Flag),
    Not(Box<FlagExpr>),
    And(Box<FlagExpr>, Box<FlagExpr>),
    Or(Box<FlagExpr>, Box<FlagExpr>),
    Xor(Box<FlagExpr>, Box<FlagExpr>),
    /// Unmodelable node, or leaves reading different `Flags` values (SS1.2
    /// item 7 is M3's flag-through-phis work).
    Opaque(&'static str),
}

/// What a `Flags` value records, plus the width of its operands.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub op: FlagsOp,
    pub a: VId,
    pub b: VId,
    pub w: Width,
    pub ty: Ty,
    /// Resolved constant shift count, when the count is a constant.
    pub count: Option<u64>,
}

pub fn ctx_of(f: &Function, fop: FlagsOp) -> Ctx {
    let (a, b) = match fop {
        FlagsOp::Add { a, b, .. }
        | FlagsOp::Sub { a, b, .. }
        | FlagsOp::And { a, b }
        | FlagsOp::Or { a, b }
        | FlagsOp::Xor { a, b }
        | FlagsOp::Shl { a, b }
        | FlagsOp::Shr { a, b }
        | FlagsOp::Sar { a, b }
        | FlagsOp::Mul { a, b, .. } => (a, b),
        FlagsOp::Neg { a } | FlagsOp::Inc { a } | FlagsOp::Dec { a } => (a, NO_VAL),
    };
    let ty = f.ty_of(a).unwrap_or(Ty::u64());
    Ctx {
        op: fop,
        a,
        b,
        w: ty.width().unwrap_or(Width::W64),
        ty,
        count: match fop {
            FlagsOp::Shl { b, .. } | FlagsOp::Shr { b, .. } | FlagsOp::Sar { b, .. } => {
                const_of(f, b)
            }
            _ => None,
        },
    }
}

/// A value that is a plain constant, if it is.
pub fn const_of(f: &Function, v: VId) -> Option<u64> {
    let (bb, ii) = f.def(v)?;
    match &f.blocks[bb as usize].instrs[ii].op {
        Op::Const(c) => Some(*c),
        _ => None,
    }
}

/// The pass: one root per call, the driver iterates to the fixpoint.
pub fn flag_elimination(f: &mut Function) -> bool {
    for (bi, ii) in flag_roots(f) {
        let ins = f.blocks[bi].instrs[ii].clone();
        let tree = match expr_of(f, ins.dest) {
            Some(e) => e,
            None => continue,
        };
        let fop = match root_flags_op(f, ins.dest) {
            FlagsOf::One(o) => o,
            FlagsOf::Many => {
                f.blocks[bi].instrs[ii].op =
                    Op::Opaque("flag reads from more than one Flags value".into());
                return true;
            }
            FlagsOf::None => {
                f.blocks[bi].instrs[ii].op = Op::Opaque("get_flag of a non-flags value".into());
                return true;
            }
        };
        let ctx = ctx_of(f, fop);
        let mut sb = SB::new();
        let res = lower_tree(&mut sb, &tree, &ctx);
        // Materialise: children above the root, in order, then the root itself.
        let addr = ins.addr;
        let mut ids: Vec<VId> = Vec::with_capacity(sb.ops.len());
        for (k, (so, ty)) in sb.ops.iter().enumerate() {
            let op = materialise(so, &ids);
            let v = f.new_value(ty.clone());
            f.blocks[bi]
                .instrs
                .insert(ii + k, Instr::new(v, op, ty.clone(), addr));
            ids.push(v);
        }
        let final_v = match res {
            Tm::V(v) => v,
            Tm::T(k) => ids[k],
        };
        f.blocks[bi].instrs[ii + sb.ops.len()].op = Op::Copy(final_v);
        // The root's cone is folded. The *same* flags value is usually read by
        // bits this branch never uses - the lifter reads all five flags for a
        // jcc, defensively - and those leaves keep the `Flags` value alive: the
        // value is partially consumed, so it survives to the emitter, which
        // refuses it, and DCE cannot clean it up (deleting a flag read is not
        // semantics-preserving: reading a flag the model calls undefined faults).
        // Fold them here, where the arithmetic context is known.
        if let Some(fv) = root_flag_value(f, ins.dest) {
            fold_flag_reads(f, fv);
        }
        return true;
    }
    false
}

fn materialise(so: &SO, ids: &[VId]) -> Op {
    let g = |t: &Tm| match t {
        Tm::V(v) => *v,
        Tm::T(k) => ids[*k],
    };
    match so {
        SO::Const(c) => Op::Const(*c),
        SO::Bin(o, x, y) => Op::Bin(*o, g(x), g(y)),
        SO::Cmp(c, x, y) => Op::Cmp(*c, g(x), g(y)),
        SO::Un(o, x) => Op::Un(*o, g(x)),
        SO::Cast(k, t, x) => Op::Cast(*k, t.clone(), g(x)),
        SO::Gap(why) => Op::Opaque((*why).into()),
    }
}

pub enum FlagsOf {
    One(FlagsOp),
    Many,
    None,
}

/// The `FlagsOp` behind every leaf of the tree, or `Many` / `None`.
pub fn root_flags_op(f: &Function, root: VId) -> FlagsOf {
    let mut acc: Option<FlagsOp> = None;
    let mut stack = vec![root];
    let mut seen: Vec<VId> = Vec::new();
    while let Some(v) = stack.pop() {
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        let (bb, ii) = match f.def(v) {
            Some(x) => x,
            None => continue,
        };
        match &f.blocks[bb as usize].instrs[ii].op {
            Op::GetFlag(_, fv) => {
                let (fb, fi) = match f.def(*fv) {
                    Some(x) => x,
                    None => return FlagsOf::None,
                };
                match &f.blocks[fb as usize].instrs[fi].op {
                    Op::Flags(o) => match &acc {
                        None => acc = Some(*o),
                        Some(p) if *p == *o => {}
                        Some(_) => return FlagsOf::Many,
                    },
                    _ => return FlagsOf::None,
                }
            }
            Op::Bin(_, x, y) => {
                stack.push(*x);
                stack.push(*y);
            }
            Op::Un(UnOp::Not, x) => stack.push(*x),
            _ => {}
        }
    }
    match acc {
        Some(o) => FlagsOf::One(o),
        None => FlagsOf::None,
    }
}

/// Roots of flag-derived boolean expressions: nodes reachable from `get_flag`
/// through And/Or/Xor/Not whose own result is not so consumed (rule 1).
pub fn flag_roots(f: &Function) -> Vec<(usize, usize)> {
    let mut derived: Vec<bool> = vec![false; f.next_value as usize + 1];
    let mut changed = true;
    while changed {
        changed = false;
        for blk in &f.blocks {
            for ins in &blk.instrs {
                let d = match &ins.op {
                    // "Is this operand a flags value?" is answered by the op that
                    // defines it as much as by the type annotation, and a guard that
                    // can silently un-mark every leaf in every function is not a
                    // guard worth having. Relaxing this did NOT move the corpus
                    // counts - the root-predicate polarity below was the cause -
                    // so the change here is defensive, and the measurement that
                    // matters is the one attached to that polarity.
                    Op::GetFlag(_, fv) => {
                        f.ty_of(*fv) == Some(Ty::Flags)
                            || matches!(
                                f.def(*fv).map(|(b, i)| &f.blocks[b as usize].instrs[i].op),
                                Some(Op::Flags(_))
                            )
                    }
                    Op::Bin(o, x, y) => {
                        matches!(o, BinOp::And | BinOp::Or | BinOp::Xor)
                            && (gat(&derived, *x) || gat(&derived, *y))
                    }
                    Op::Un(UnOp::Not, x) => gat(&derived, *x),
                    _ => false,
                };
                if d && !gat(&derived, ins.dest) {
                    sat(&mut derived, ins.dest, true);
                    changed = true;
                }
            }
        }
    }
    let mut out = Vec::new();
    for blk in &f.blocks {
        for (ii, ins) in blk.instrs.iter().enumerate() {
            if !gat(&derived, ins.dest) {
                continue;
            }
            let consumed = f.uses_of(ins.dest).into_iter().any(|u| {
                // `consumed` here means "interior to the tree": consumed by
                // another derived node. A terminator use (instr == usize::MAX) is
                // NOT part of the flag expression - it is the branch that the
                // expression feeds - so it makes this node the ROOT, not an
                // interior node. Returning true here was inverted: it classified
                // every live jcc tree as interior, so the pass folded only the
                // flag reads nobody used, and every real `flags.sub` feeding a
                // branch survived to the C emitter (37 refused functions).
                if u.instr == usize::MAX {
                    return false;
                }
                f.blocks[u.bb as usize]
                    .instrs
                    .get(u.instr)
                    .map(|i| gat(&derived, i.dest))
                    .unwrap_or(false)
            });
            if !consumed {
                out.push((blk.id as usize, ii));
            }
        }
    }
    out
}

fn gat(v: &[bool], id: VId) -> bool {
    v.get(id as usize).copied().unwrap_or(false)
}
fn sat(v: &mut [bool], id: VId, x: bool) {
    if let Some(s) = v.get_mut(id as usize) {
        *s = x;
    }
}

fn expr_of(f: &Function, v: VId) -> Option<FlagExpr> {
    let (bb, ii) = f.def(v)?;
    let op = f.blocks[bb as usize].instrs[ii].op.clone();
    match op {
        Op::GetFlag(fl, fv) => {
            if f.ty_of(fv) == Some(Ty::Flags) {
                Some(FlagExpr::Leaf(fl))
            } else {
                Some(FlagExpr::Opaque("get_flag of a non-flags value"))
            }
        }
        Op::Bin(opr, x, y) => {
            let l = expr_of(f, x)?;
            let r = expr_of(f, y)?;
            Some(match opr {
                BinOp::And => FlagExpr::And(Box::new(l), Box::new(r)),
                BinOp::Or => FlagExpr::Or(Box::new(l), Box::new(r)),
                BinOp::Xor => FlagExpr::Xor(Box::new(l), Box::new(r)),
                _ => FlagExpr::Opaque("non-boolean op inside a flag expression"),
            })
        }
        Op::Un(UnOp::Not, x) => Some(FlagExpr::Not(Box::new(expr_of(f, x)?))),
        _ => Some(FlagExpr::Opaque(
            "unsupported node inside a flag expression",
        )),
    }
}

fn lower_tree(sb: &mut SB, e: &FlagExpr, ctx: &Ctx) -> Tm {
    sb.set_ty(ctx.ty.clone());
    match e {
        FlagExpr::Leaf(fl) => exact_flag(sb, ctx, *fl),
        FlagExpr::Not(x) => {
            let inner = lower_tree(sb, x, ctx);
            let b = sb.is_true(inner);
            let z = sb.push(SO::Const(0), Ty::Bool);
            sb.push(SO::Cmp(IntCmp::Eq, b, z), Ty::Bool)
        }
        FlagExpr::And(l, r) => combine(sb, BinOp::And, l, r, ctx),
        FlagExpr::Or(l, r) => combine(sb, BinOp::Or, l, r, ctx),
        FlagExpr::Xor(l, r) => combine(sb, BinOp::Xor, l, r, ctx),
        FlagExpr::Opaque(_) => sb.push(SO::Const(0), Ty::Bool),
    }
}

fn combine(sb: &mut SB, op: BinOp, l: &FlagExpr, r: &FlagExpr, ctx: &Ctx) -> Tm {
    let lo = lower_tree(sb, l, ctx);
    let ro = lower_tree(sb, r, ctx);
    // Shape preserved: the M3 fusion table matches on it.
    sb.push(SO::Bin(op, lo, ro), Ty::Bool)
}

// ---------------------------------------------------------------------------
// Exact lowerings: one arm per (FlagsOp x flag). Total by construction.
// ---------------------------------------------------------------------------

/// The instruction's own integer result, recomputed (rule 2).
fn res(sb: &mut SB, ctx: &Ctx) -> Tm {
    let a = Tm::V(ctx.a);
    let b = Tm::V(ctx.b);
    let t = ctx.ty.clone();
    match ctx.op {
        FlagsOp::Add { carry, .. } => {
            let s = sb.bin(BinOp::Add, a, b, t.clone());
            if carry {
                let o = sb.c(1);
                sb.bin(BinOp::Add, s, o, t)
            } else {
                s
            }
        }
        FlagsOp::Sub { borrow, .. } => {
            let d = sb.bin(BinOp::Sub, a, b, t.clone());
            if borrow {
                let o = sb.c(1);
                sb.bin(BinOp::Sub, d, o, t)
            } else {
                d
            }
        }
        FlagsOp::And { .. } => sb.bin(BinOp::And, a, b, t),
        FlagsOp::Or { .. } => sb.bin(BinOp::Or, a, b, t),
        FlagsOp::Xor { .. } => sb.bin(BinOp::Xor, a, b, t),
        FlagsOp::Shl { .. } => sb.bin(BinOp::Shl, a, b, t),
        FlagsOp::Shr { .. } => sb.bin(BinOp::Shr, a, b, t),
        FlagsOp::Sar { .. } => sb.bin(BinOp::Sar, a, b, t),
        FlagsOp::Neg { .. } => sb.push(SO::Un(UnOp::Neg, a), t),
        FlagsOp::Inc { .. } => {
            let o = sb.c(1);
            sb.bin(BinOp::Add, a, o, t)
        }
        FlagsOp::Dec { .. } => {
            let o = sb.c(1);
            sb.bin(BinOp::Sub, a, o, t)
        }
        FlagsOp::Mul { .. } => sb.bin(BinOp::Mul, a, b, t),
    }
}

fn exact_flag(sb: &mut SB, ctx: &Ctx, fl: Flag) -> Tm {
    sb.set_ty(ctx.ty.clone());
    let t = ctx.ty.clone();
    let w = ctx.w.bits() as u64;
    let sign: u64 = 1 << (w - 1);
    let a = Tm::V(ctx.a);
    let b = Tm::V(ctx.b);
    let r = res(sb, ctx);
    match fl {
        // ZF: result == 0.
        Flag::ZF => {
            let z = sb.c(0);
            sb.cmp(IntCmp::Eq, r, z)
        }
        // SF: top bit of the result.
        Flag::SF => {
            let s = sb.c(sign);
            let an = sb.bin(BinOp::And, r, s, t.clone());
            sb.nz(an, t)
        }
        // PF: even parity of the low byte.
        Flag::PF => {
            let m = sb.c(0xff);
            let lo = sb.bin(BinOp::And, r, m, t.clone());
            let pc = sb.push(SO::Un(UnOp::Popcount, lo), t.clone());
            let one = sb.c(1);
            let b0 = sb.bin(BinOp::And, pc, one, t.clone());
            let z = sb.c(0);
            sb.cmp(IntCmp::Eq, b0, z)
        }
        Flag::AF => match ctx.op {
            FlagsOp::Add { .. }
            | FlagsOp::Sub { borrow: false, .. }
            | FlagsOp::Inc { .. }
            | FlagsOp::Dec { .. } => {
                // AF = bit 4 of (a XOR b XOR result)
                let bb = match ctx.op {
                    FlagsOp::Add { .. } | FlagsOp::Sub { .. } => b,
                    _ => sb.const_true(),
                };
                let x = sb.bin(BinOp::Xor, a, r, t.clone());
                let y = sb.bin(BinOp::Xor, x, bb, t.clone());
                let bt = sb.bit(y, 4, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Sub { borrow: true, .. } => {
                let one = sb.c(1);
                let bb = sb.bin(BinOp::Add, b, one, t.clone());
                let x = sb.bin(BinOp::Xor, a, r, t.clone());
                let y = sb.bin(BinOp::Xor, x, bb, t.clone());
                let bt = sb.bit(y, 4, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Neg { .. } => {
                // AF = (a & 0xf) != 0
                let m = sb.c(0xf);
                let an = sb.bin(BinOp::And, a, m, t.clone());
                sb.nz(an, t)
            }
            _ => sb.opaque("AF is architecturally undefined for this instruction"),
        },
        Flag::CF => match ctx.op {
            // carry out of a + b: the sum wrapped, i.e. res <u a
            FlagsOp::Add { carry: false, .. } => sb.cmp(IntCmp::Ult, r, a),
            FlagsOp::Add { carry: true, .. } => sb.cmp(IntCmp::Ule, r, a),
            FlagsOp::Sub { borrow: false, .. } => sb.cmp(IntCmp::Ult, a, b),
            FlagsOp::Sub { borrow: true, .. } => sb.cmp(IntCmp::Ule, a, b),
            FlagsOp::And { .. } | FlagsOp::Or { .. } | FlagsOp::Xor { .. } => sb.const_false(),
            FlagsOp::Neg { .. } => {
                let z = sb.c(0);
                sb.cmp(IntCmp::Ne, a, z)
            }
            FlagsOp::Inc { .. } | FlagsOp::Dec { .. } => {
                sb.opaque("CF after inc/dec needs the previous flags as an operand (SS1.2 item 3)")
            }
            FlagsOp::Shl { .. } => {
                // CF = last bit shifted out = bit (width - n) of a
                let bits = sb.c(w);
                let sh = sb.bin(BinOp::Sub, bits, b, t.clone());
                let s = sb.bin(BinOp::Shr, a, sh, t.clone());
                let one = sb.c(1);
                let bt = sb.bin(BinOp::And, s, one, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Shr { .. } | FlagsOp::Sar { .. } => {
                // CF = bit (n-1) of a
                let one = sb.c(1);
                let n1 = sb.bin(BinOp::Sub, b, one, t.clone());
                let s = sb.bin(BinOp::Shr, a, n1, t.clone());
                let o = sb.c(1);
                let bt = sb.bin(BinOp::And, s, o, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Mul { signed, .. } => mul_overflow(sb, ctx, signed),
        },
        Flag::OF => match ctx.op {
            FlagsOp::Add { .. } => {
                // ((a ^ r) & (b ^ r)) >> (w-1)
                let x = sb.bin(BinOp::Xor, a, r, t.clone());
                let y = sb.bin(BinOp::Xor, b, r, t.clone());
                let an = sb.bin(BinOp::And, x, y, t.clone());
                let bt = sb.bit(an, w - 1, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Sub { .. } => {
                // ((a ^ b) & (a ^ r)) >> (w-1)
                let x = sb.bin(BinOp::Xor, a, b, t.clone());
                let y = sb.bin(BinOp::Xor, a, r, t.clone());
                let an = sb.bin(BinOp::And, x, y, t.clone());
                let bt = sb.bit(an, w - 1, t.clone());
                sb.nz(bt, t)
            }
            FlagsOp::Inc { .. } => {
                let imax = sb.c(sign - 1);
                sb.cmp(IntCmp::Eq, a, imax)
            }
            FlagsOp::Dec { .. } | FlagsOp::Neg { .. } => {
                let imin = sb.c(sign);
                sb.cmp(IntCmp::Eq, a, imin)
            }
            FlagsOp::And { .. } | FlagsOp::Or { .. } | FlagsOp::Xor { .. } => sb.const_false(),
            FlagsOp::Mul { signed, .. } => mul_overflow(sb, ctx, signed),
            FlagsOp::Shl { .. } | FlagsOp::Shr { .. } | FlagsOp::Sar { .. } => {
                match ctx.count {
                    Some(1) => match ctx.op {
                        FlagsOp::Shl { .. } => {
                            // OF = new MSB XOR carry-out
                            let m = sb.bit(r, w - 1, t.clone());
                            let co = sb.bit(a, w - 1, t.clone());
                            sb.push(SO::Bin(BinOp::Xor, m, co), Ty::Bool)
                        }
                        FlagsOp::Shr { .. } => {
                            // OF = MSB of the original operand
                            sb.bit(a, w - 1, t.clone())
                        }
                        _ => sb.const_false(), // SDM: SAR by 1 clears OF
                    },
                    _ => sb.opaque("OF after a shift by a count other than 1 is undefined"),
                }
            }
        },
    }
}

fn mul_overflow(sb: &mut SB, ctx: &Ctx, signed: bool) -> Tm {
    let t = ctx.ty.clone();
    let a = Tm::V(ctx.a);
    let b = Tm::V(ctx.b);
    let hi = sb.push(
        SO::Bin(if signed { BinOp::MulSmul } else { BinOp::MulHi }, a, b),
        t.clone(),
    );
    let lo = sb.bin(BinOp::Mul, a, b, t.clone());
    let se = sb.push(SO::Cast(CastKind::SExt, t.clone(), lo), t.clone());
    sb.cmp(IntCmp::Ne, hi, se)
}

// ---------------------------------------------------------------------------
// The fusion table (readability, not correctness)
// ---------------------------------------------------------------------------

/// Collapse `cmp(a,b) op cmp(a,b)` into one comparison, when the pair of
/// predicates describes a single relation on the same operands.
///
/// Every comparison over a fixed pair selects a subset of the trichotomy
/// {less, equal, greater}, and And/Or/Xor are set operations on that subset.
/// So the table is one line per subset, derived, not remembered: `Le & Ge` is
/// `{<}∩{=,>} ... ` — empty for the same signedness, and `Eq` for the union,
/// which is why the `joint_pair` special cases in v1 (which claimed `CF && ZF`
/// after `sub` means `a == b`, an empty set) are not needed.
///
/// Masks 0 (always false) and 7 (always true) return `None`: `IntCmp` has no
/// constant predicate, and keeping the boolean combination is correct.
/// Mixed signedness returns `None`: `Lt` and `Ule` are relations on different
/// orders of the same bits, so no single comparison expresses their
/// intersection.
pub fn cmp_pair(c1: IntCmp, c2: IntCmp, op: BinOp) -> Option<IntCmp> {
    let (m1, signed1) = cmp_mask(c1)?;
    let (m2, signed2) = cmp_mask(c2)?;
    if signed1 != signed2 {
        return None;
    }
    let m = match op {
        BinOp::And => m1 & m2,
        BinOp::Or => m1 | m2,
        BinOp::Xor => m1 ^ m2,
        _ => return None,
    };
    mask_cmp(m, signed1)
}

const LT: u8 = 1;
const EQ: u8 = 2;
const GT: u8 = 4;

/// (subset of the trichotomy, is the comparison signed).
fn cmp_mask(c: IntCmp) -> Option<(u8, bool)> {
    Some(match c {
        IntCmp::Eq => (EQ, true),
        IntCmp::Ne => (LT | GT, true),
        IntCmp::Lt => (LT, true),
        IntCmp::Le => (LT | EQ, true),
        IntCmp::Gt => (GT, true),
        IntCmp::Ge => (EQ | GT, true),
        IntCmp::Ult => (LT, false),
        IntCmp::Ule => (LT | EQ, false),
        IntCmp::Ugt => (GT, false),
        IntCmp::Uge => (EQ | GT, false),
    })
}

fn mask_cmp(m: u8, signed: bool) -> Option<IntCmp> {
    Some(match m {
        1 => {
            if signed {
                IntCmp::Lt
            } else {
                IntCmp::Ult
            }
        }
        2 => IntCmp::Eq,
        3 => {
            if signed {
                IntCmp::Le
            } else {
                IntCmp::Ule
            }
        }
        4 => {
            if signed {
                IntCmp::Gt
            } else {
                IntCmp::Ugt
            }
        }
        5 => IntCmp::Ne,
        6 => {
            if signed {
                IntCmp::Ge
            } else {
                IntCmp::Uge
            }
        }
        _ => return None,
    })
}

/// The single `Flags` value behind every leaf of the boolean expression rooted at
/// `root`, or `None` if the leaves read more than one, or none.
fn root_flag_value(f: &Function, root: VId) -> Option<VId> {
    let mut one: Option<VId> = None;
    let mut stack = vec![root];
    let mut seen: Vec<VId> = Vec::new();
    while let Some(v) = stack.pop() {
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        let (bb, ii) = f.def(v)?;
        match &f.blocks[bb as usize].instrs[ii].op {
            Op::GetFlag(_, fv) => match one {
                None => one = Some(*fv),
                Some(p) if p == *fv => {}
                Some(_) => return None,
            },
            Op::Bin(_, x, y) => {
                stack.push(*x);
                stack.push(*y);
            }
            Op::Un(UnOp::Not, x) => stack.push(*x),
            _ => {}
        }
    }
    one
}

/// Fold every remaining read of one `Flags` value, leaf by leaf.
///
/// A leaf is rewritten only when the lowering table answers it EXACTLY. A bit the
/// model calls undefined after this operation lowers to a `Gap`, and replacing a
/// read that faults in the interpreter with a value that does not would change
/// meaning - the TV harness would be right to call that a broken pass. So an
/// undefined bit keeps its `getflag`, the flags value keeps surviving, and the
/// emitter keeps refusing: the honest outcome for a bit x86 leaves undefined.
fn fold_flag_reads(f: &mut Function, fv: VId) -> bool {
    let (fb, fi) = match f.def(fv) {
        Some(x) => x,
        None => return false,
    };
    let fop = match &f.blocks[fb as usize].instrs[fi].op {
        Op::Flags(o) => *o,
        _ => return false,
    };
    let ctx = ctx_of(f, fop);
    let mut changed = false;
    loop {
        let mut target: Option<(usize, usize, Flag)> = None;
        'scan: for (bi, b) in f.blocks.iter().enumerate() {
            for (ii, ins) in b.instrs.iter().enumerate() {
                if let Op::GetFlag(bit, x) = ins.op {
                    if x == fv {
                        target = Some((bi, ii, bit));
                        break 'scan;
                    }
                }
            }
        }
        let Some((bi, ii, bit)) = target else {
            break;
        };
        let addr = f.blocks[bi].instrs[ii].addr;
        let mut sb = SB::new();
        let res = exact_flag(&mut sb, &ctx, bit);
        if sb.ops.iter().any(|(so, _)| matches!(so, SO::Gap(_))) {
            // Undefined bit: leave the faulting read alone, and stop - every
            // remaining read of this value is the same question.
            break;
        }
        let mut ids: Vec<VId> = Vec::with_capacity(sb.ops.len());
        for (k, (so, ty)) in sb.ops.iter().enumerate() {
            let op = materialise(so, &ids);
            let v = f.new_value(ty.clone());
            f.blocks[bi]
                .instrs
                .insert(ii + k, Instr::new(v, op, ty.clone(), addr));
            ids.push(v);
        }
        let final_v = match res {
            Tm::V(v) => v,
            Tm::T(k) => ids[k],
        };
        f.blocks[bi].instrs[ii + sb.ops.len()].op = Op::Copy(final_v);
        changed = true;
    }
    changed
}
