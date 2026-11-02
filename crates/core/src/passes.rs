//! Middle-end passes. Each pass is "rewrite to an equivalent function, report
//! whether anything changed", so the driver runs them to a fixpoint. That shape
//! is what makes a partial pass list safe: no pass relies on a later one
//! cleaning up after it.

use std::collections::HashMap;

use crate::cfg::Cfg;
use crate::ir::*;
use crate::ty::{Sx, Ty, Width};

// ---------------------------------------------------------------------------
// phi pruning: SSAn -> SSA
// ---------------------------------------------------------------------------

/// Remove phi nodes that carry no information:
///  * all incoming values equal (the common "not assigned on this path" case,
///    where a block feeds its own value back),
///  * self-only phis (reads before any write),
///  * dead phis.
///
/// Runs to fixpoint: collapsing one phi can degenerate another.
pub fn prune_phis(f: &mut Function) -> bool {
    let mut changed_any = false;
    loop {
        let mut changed = false;
        'outer: for bi in 0..f.blocks.len() {
            let n = f.blocks[bi].instrs.len();
            for ii in 0..n {
                let dest = f.blocks[bi].instrs[ii].dest;
                let ty = f.blocks[bi].instrs[ii].ty.clone();
                let Op::Phi(pairs) = f.blocks[bi].instrs[ii].op.clone() else {
                    continue;
                };
                let pruned: Vec<(VId, BbId)> =
                    pairs.iter().copied().filter(|(v, _)| *v != dest).collect();

                if f.use_count(dest) == 0 {
                    f.blocks[bi].instrs.remove(ii);
                    changed = true;
                    break 'outer;
                }

                if pruned.is_empty() {
                    // Only self-references: the variable is read before written.
                    f.blocks[bi].instrs.remove(ii);
                    let u = new_undef(f, ty);
                    f.replace_use_all(dest, u);
                    changed = true;
                    break 'outer;
                }

                let mut distinct: Vec<VId> = Vec::new();
                for (v, _) in pairs.iter() {
                    if !distinct.contains(v) {
                        distinct.push(*v);
                    }
                }
                if distinct.len() == 1 {
                    f.blocks[bi].instrs[ii].op = Op::Copy(distinct[0]);
                    f.replace_use_all(dest, distinct[0]);
                    changed = true;
                    break 'outer;
                }

                if pruned.len() < pairs.len() {
                    // Drop self-refs: a phi that returns its own value on one
                    // path is only live for the others.
                    f.blocks[bi].instrs[ii].op = Op::Phi(pruned.into_boxed_slice());
                    changed = true;
                    break 'outer;
                }
            }
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
}

fn new_undef(f: &mut Function, ty: Ty) -> VId {
    let v = f.new_value(ty.clone());
    f.blocks[f.entry as usize].push(Instr::new(v, Op::Undef, ty, 0));
    v
}

// ---------------------------------------------------------------------------
// flag elimination
// ---------------------------------------------------------------------------

/// The heart of x86 decompilation.
///
/// A `Ty::Flags` value is an opaque, local record of one instruction's flag
/// definition; the only reader is `get_flag`. Consequences, all mechanical:
///  1. dead flags vanish under plain DCE,
///  2. a branch condition is a boolean tree over reads of *one* flags value, so
///     it can be matched against the exact jcc matrix (see `FlagExpr`),
///  3. anything the table misses lowers to a width-exact helper expression
///     (`__rdec_add_cf(a, b)`), never to a guess. Helpers are emitted on demand,
///     so output always compiles and an unfused case is visible in the source.
pub fn flag_elimination(f: &mut Function) -> bool {
    // Rewriting one getflag can enable another, so do one rewrite per call and
    // let the driver iterate to the fixpoint.
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            if !matches!(f.blocks[bi].instrs[ii].op, Op::GetFlag(..)) {
                continue;
            }
            let bb = f.blocks[bi].id;
            if let Some(rep) = simplify_getflag(f, bb, ii) {
                f.blocks[bi].instrs[ii].op = rep;
                return true;
            }
        }
    }
    false
}

/// What a flags value records, in terms of its operands.
#[derive(Copy, Clone, Debug)]
struct FlagCtx {
    kind: FlagClass,
    a: VId,
    b: VId,
    /// The defining op's integer result, when it has one.
    res: VId,
    w: Width,
    signed: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum FlagClass {
    /// a - b, no borrow-in: the `cmp a, b` case.
    Cmp,
    /// a & b / a | b / a ^ b: CF = OF = 0.
    Test,
    /// a + b: flags need the carry chain, so only ZF/SF fuse here.
    Add,
}

fn ctx_of(f: &Function, fop: FlagsOp, res: VId) -> Option<FlagCtx> {
    let (kind, a, b) = match fop {
        FlagsOp::Sub {
            a,
            b,
            borrow: false,
        } => (FlagClass::Cmp, a, b),
        FlagsOp::And { a, b } | FlagsOp::Or { a, b } | FlagsOp::Xor { a, b } => {
            (FlagClass::Test, a, b)
        }
        FlagsOp::Add { a, b, carry: false } => (FlagClass::Add, a, b),
        _ => return None,
    };
    let ty = f.ty_of(a).unwrap_or(Ty::u64());
    let w = ty.width().unwrap_or(Width::W64);
    let signed = ty.sx() == Some(Sx::Signed);
    Some(FlagCtx {
        kind,
        a,
        b,
        res,
        w,
        signed,
    })
}

/// Rewrite the `get_flag` at (`bb`, `ii`) - as the root of its boolean tree -
/// into comparisons. Only at the tree root, so inner reads are replaced by the
/// same lowering instead of racing it.
fn simplify_getflag(f: &mut Function, bb: BbId, ii: usize) -> Option<Op> {
    let ins = f.blocks[bb as usize].instrs[ii].clone();
    let Op::GetFlag(_, fv) = ins.op else {
        return None;
    };
    let addr = ins.addr;
    let root = ins.dest;
    if f.def(root)? != (bb, ii) {
        return None;
    }
    let (def_bb, def_ii) = f.def(fv)?;
    let flags_ins = f.blocks[def_bb as usize].instrs[def_ii].clone();
    let Op::Flags(fop) = flags_ins.op.clone() else {
        return None;
    };
    let tree = collect_flag_expr(f, root, fv);
    if let Some(ctx) = ctx_of(f, fop, flags_ins.dest) {
        if let Some(op) = lower_flag_expr(f, &tree, ctx, addr) {
            return Some(op);
        }
    }
    // No table hit: exact per-flag helpers, so no Ty::Flags survives.
    lower_flag_expr_exact(f, &tree, &flags_ins, addr)
}

/// Lower a boolean tree over one flags value using the comparison table.
fn lower_flag_expr(f: &mut Function, e: &FlagExpr, ctx: FlagCtx, addr: u64) -> Option<Op> {
    if let Some(op) = joint_pair(f, e, ctx) {
        return Some(op);
    }
    match e {
        FlagExpr::One(fl) => atom(*fl, ctx, f, addr),
        FlagExpr::Not(x) => {
            let inner = lower_flag_expr(f, x, ctx, addr)?;
            Some(negate_bool(f, inner, addr))
        }
        FlagExpr::And(l, r) => bool_combine(f, BinOp::And, l, r, ctx, addr),
        FlagExpr::Or(l, r) => bool_combine(f, BinOp::Or, l, r, ctx, addr),
        FlagExpr::Xor(l, r) => bool_combine(f, BinOp::Xor, l, r, ctx, addr),
        FlagExpr::Opaque => None,
    }
}

fn bool_combine(
    f: &mut Function,
    op: BinOp,
    l: &FlagExpr,
    r: &FlagExpr,
    ctx: FlagCtx,
    addr: u64,
) -> Option<Op> {
    let a = lower_flag_expr(f, l, ctx, addr)?;
    let b = lower_flag_expr(f, r, ctx, addr)?;
    // Constant folding of the boolean algebra: this is what turns `SF ^ OF`
    // into `SF` for a `test`, where OF is known 0.
    match (a.as_const(), b.as_const()) {
        (Some(x), Some(y)) => {
            let v = match op {
                BinOp::And => (x & 1) & (y & 1),
                BinOp::Or => (x & 1) | (y & 1),
                BinOp::Xor => (x & 1) ^ (y & 1),
                _ => return None,
            };
            return Some(Op::Const(v));
        }
        (Some(x), None) => {
            if let Some(o) = unit_fold(op, x, b.clone(), f, addr) {
                return Some(o);
            }
        }
        (None, Some(y)) => {
            if let Some(o) = unit_fold(op, y, a.clone(), f, addr) {
                return Some(o);
            }
        }
        _ => {}
    }
    // Two comparisons of the same pair fold into one when the combination names
    // a single comparison (a<b && a!=b -> a<=b, etc).
    if let (Op::Cmp(c1, x1, y1), Op::Cmp(c2, x2, y2)) = (&a, &b) {
        if x1 == x2 && y1 == y2 {
            if let Some(c) = cmp_pair(*c1, *c2, op) {
                return Some(Op::Cmp(c, *x1, *y1));
            }
        }
    }
    let x = f.add_value_op(a, Ty::Bool, addr);
    let y = f.add_value_op(b, Ty::Bool, addr);
    Some(Op::Bin(op, x, y))
}

/// `x` is a boolean constant: fold it against `other`.
fn unit_fold(op: BinOp, x: u64, other: Op, f: &mut Function, addr: u64) -> Option<Op> {
    match (op, x & 1) {
        (BinOp::And, 0) => Some(Op::Const(0)),
        (BinOp::Or, 1) => Some(Op::Const(1)),
        (BinOp::And, 1) | (BinOp::Or, 0) => Some(other),
        (BinOp::Xor, 0) => Some(other),
        (BinOp::Xor, 1) => Some(negate_bool(f, other, addr)),
        _ => None,
    }
}

/// `(c1 op c2)` as a single comparison, when it is one.
fn cmp_pair(c1: IntCmp, c2: IntCmp, op: BinOp) -> Option<IntCmp> {
    use IntCmp::*;
    let tab: &[((IntCmp, IntCmp), IntCmp)] = match op {
        BinOp::Or => &[
            ((Ult, Eq), Ule),
            ((Eq, Ult), Ule),
            ((Uge, Ne), Ugt),
            ((Ne, Uge), Ugt),
            ((Lt, Eq), Le),
            ((Eq, Lt), Le),
            ((Gt, Ne), Ge),
            ((Ne, Gt), Ge),
        ],
        BinOp::And => &[
            ((Uge, Ne), Ugt),
            ((Ne, Uge), Ugt),
            ((Ult, Ne), Ule),
            ((Ne, Ult), Ule),
            ((Ge, Ne), Gt),
            ((Ne, Ge), Gt),
            ((Lt, Ne), Le),
            ((Ne, Lt), Le),
        ],
        _ => return None,
    };
    tab.iter()
        .find(|((l, r), _)| *l == c1 && *r == c2)
        .map(|(_, c)| *c)
}

fn negate_bool(f: &mut Function, op: Op, addr: u64) -> Op {
    match op {
        Op::Cmp(c, a, b) => Op::Cmp(c.negate(), a, b),
        Op::Const(v) => Op::Const((v & 1) ^ 1),
        other => {
            let v = f.add_value_op(other, Ty::Bool, addr);
            Op::Un(UnOp::Not, v)
        }
    }
}

/// One flag bit -> a C-visible op, from the flag's exact definition.
fn atom(fl: Flag, ctx: FlagCtx, f: &mut Function, addr: u64) -> Option<Op> {
    let zero = f.add_const(Ty::uint(ctx.w), 0, addr);
    match ctx.kind {
        FlagClass::Cmp => match fl {
            Flag::ZF => Some(Op::Cmp(IntCmp::Eq, ctx.a, ctx.b)),
            Flag::CF => Some(Op::Cmp(IntCmp::Ult, ctx.a, ctx.b)),
            Flag::SF => Some(Op::Cmp(IntCmp::Lt, ctx.a, zero)),
            Flag::OF => Some(Op::Opaque(
                format!("signed overflow of a={:#x} - b={:#x}", ctx.a, ctx.b).into_boxed_str(),
            )),
            Flag::PF | Flag::AF => None,
        },
        FlagClass::Test => match fl {
            Flag::ZF => Some(Op::Cmp(IntCmp::Eq, ctx.res, zero)),
            Flag::SF => Some(Op::Cmp(IntCmp::Lt, ctx.res, zero)),
            // and/or/xor clear CF and OF.
            Flag::CF | Flag::OF => Some(Op::Const(0)),
            Flag::PF | Flag::AF => None,
        },
        FlagClass::Add => match fl {
            Flag::ZF => Some(Op::Cmp(IntCmp::Eq, ctx.res, zero)),
            Flag::SF => Some(Op::Cmp(IntCmp::Lt, ctx.res, zero)),
            _ => None,
        },
    }
}

/// The classic jcc matrix: flag-read pairs that name one comparison.
fn joint_pair(f: &mut Function, e: &FlagExpr, ctx: FlagCtx) -> Option<Op> {
    let (op, l, r) = match e {
        FlagExpr::And(l, r) => (BinOp::And, l, r),
        FlagExpr::Or(l, r) => (BinOp::Or, l, r),
        FlagExpr::Xor(l, r) => (BinOp::Xor, l, r),
        _ => return None,
    };
    let (a, b) = (as_flag(l)?, as_flag(r)?);
    let (neg_l, neg_r) = (is_not(l), is_not(r));

    if ctx.kind == FlagClass::Cmp {
        // CF = a<b (unsigned), ZF = a==b, SF^OF = a<b (signed).
        let c = match (a, b, neg_l, neg_r, op) {
            (Flag::CF, Flag::ZF, false, false, BinOp::Or)
            | (Flag::ZF, Flag::CF, false, false, BinOp::Or) => Some(IntCmp::Ule),
            (Flag::CF, Flag::ZF, true, true, BinOp::And)
            | (Flag::ZF, Flag::CF, true, true, BinOp::And) => Some(IntCmp::Ugt),
            (Flag::SF, Flag::OF, false, false, BinOp::Xor)
            | (Flag::OF, Flag::SF, false, false, BinOp::Xor) => {
                Some(if ctx.signed { IntCmp::Lt } else { IntCmp::Ult })
            }
            (Flag::SF, Flag::OF, true, true, BinOp::And)
            | (Flag::OF, Flag::SF, true, true, BinOp::And) => {
                Some(if ctx.signed { IntCmp::Ge } else { IntCmp::Uge })
            }
            _ => None,
        };
        if let Some(c) = c {
            return Some(Op::Cmp(c, ctx.a, ctx.b));
        }
        // CF && ZF is unsatisfiable for a subtraction (a<b and a==b at once).
        if op == BinOp::And
            && !neg_l
            && !neg_r
            && matches!((a, b), (Flag::CF, Flag::ZF) | (Flag::ZF, Flag::CF))
        {
            return Some(Op::Const(0));
        }
    }

    if ctx.kind == FlagClass::Test {
        // CF and OF are constant 0 here, so CF|ZF collapses to ZF and SF^OF to
        // SF. lower_flag_expr does the constant folding; this shortcut keeps the
        // common shapes one lookup deep.
        let keep = match (a, b, neg_l, neg_r, op) {
            (Flag::CF, Flag::ZF, _, true, BinOp::And)
            | (Flag::CF, Flag::ZF, false, false, BinOp::Or)
            | (Flag::ZF, Flag::CF, false, false, BinOp::Or) => FlagExpr::One(Flag::ZF),
            (Flag::SF, Flag::OF, false, false, BinOp::Xor)
            | (Flag::OF, Flag::SF, false, false, BinOp::Xor)
            | (Flag::OF, Flag::SF, _, true, BinOp::And)
            | (Flag::SF, Flag::OF, _, true, BinOp::And) => FlagExpr::One(Flag::SF),
            _ => return None,
        };
        return lower_flag_expr(f, &keep, ctx, 0);
    }
    None
}

fn is_not(e: &FlagExpr) -> bool {
    matches!(e, FlagExpr::Not(_))
}

fn as_flag(e: &FlagExpr) -> Option<Flag> {
    match e {
        FlagExpr::One(fl) => Some(*fl),
        FlagExpr::Not(x) => match &**x {
            FlagExpr::One(fl) => Some(*fl),
            _ => None,
        },
        _ => None,
    }
}

/// Exact lowering for flag definitions the table does not cover. These are
/// *semantically exact* at the operand width, which is why leaving a fusion off
/// the table costs readability only, never correctness.
fn lower_flag_expr_exact(
    f: &mut Function,
    e: &FlagExpr,
    flags_ins: &Instr,
    addr: u64,
) -> Option<Op> {
    let (a, b, chain) = match flags_ins.op {
        Op::Flags(FlagsOp::Add { a, b, carry }) => (a, b, carry),
        Op::Flags(FlagsOp::Sub { a, b, borrow }) => (a, b, borrow),
        _ => return None,
    };
    let is_add = matches!(flags_ins.op, Op::Flags(FlagsOp::Add { .. }));
    let w = f.ty_of(a).and_then(|t| t.width()).unwrap_or(Width::W64);
    let cf_name = match (is_add, chain) {
        (true, true) => "__rdec_adc_cf",
        (true, false) => "__rdec_add_cf",
        (false, true) => "__rdec_sbb_cf",
        (false, false) => "__rdec_sub_cf",
    };
    let of_name = if is_add {
        "__rdec_add_of"
    } else {
        "__rdec_sub_of"
    };
    let helper = |n: &str, args: Vec<VId>| Op::Call {
        callee: Callee::Symbol(n.to_string()),
        ret_ty: Ty::Bool,
        args: args.into_boxed_slice(),
        pure: true,
    };
    let mut one = |fl: Flag| -> Option<Op> {
        match fl {
            Flag::CF => Some(helper(cf_name, vec![a, b])),
            Flag::OF => Some(helper(of_name, vec![a, b])),
            Flag::ZF => Some(Op::Cmp(
                IntCmp::Eq,
                flags_ins.dest,
                f.add_const(Ty::uint(w), 0, addr),
            )),
            Flag::SF => Some(Op::Cmp(
                IntCmp::Lt,
                flags_ins.dest,
                f.add_const(Ty::uint(w), 0, addr),
            )),
            _ => None,
        }
    };
    let mut leaf = |e: &FlagExpr| -> Option<Op> {
        match e {
            FlagExpr::One(fl) => one(*fl),
            FlagExpr::Not(x) => match &**x {
                FlagExpr::One(fl) => one(*fl).map(|o| match o {
                    Op::Cmp(c, a, b) => Op::Cmp(c.negate(), a, b),
                    Op::Const(v) => Op::Const((v & 1) ^ 1),
                    other => other,
                }),
                _ => None,
            },
            _ => None,
        }
    };
    match e {
        FlagExpr::One(_) | FlagExpr::Not(_) => leaf(e),
        FlagExpr::Or(l, r) | FlagExpr::And(l, r) | FlagExpr::Xor(l, r) => {
            let (x, y) = (leaf(l)?, leaf(r)?);
            let bin = match e {
                FlagExpr::Or(..) => BinOp::Or,
                FlagExpr::And(..) => BinOp::And,
                _ => BinOp::Xor,
            };
            let x = f.add_value_op(x, Ty::Bool, addr);
            let y = f.add_value_op(y, Ty::Bool, addr);
            Some(Op::Bin(bin, x, y))
        }
        FlagExpr::Opaque => None,
    }
}

/// A boolean expression over reads of one flags value.
#[derive(Clone, Debug)]
pub enum FlagExpr {
    One(Flag),
    Not(Box<FlagExpr>),
    And(Box<FlagExpr>, Box<FlagExpr>),
    Or(Box<FlagExpr>, Box<FlagExpr>),
    Xor(Box<FlagExpr>, Box<FlagExpr>),
    /// Anything not modeled (a phi of flags, a flag stored to memory, ...).
    Opaque,
}

/// Walk uses downward from a flags value to the boolean tree consuming it.
pub fn collect_flag_expr(f: &Function, root: VId, flags_val: VId) -> FlagExpr {
    let Some((bb, ii)) = f.def(root) else {
        return FlagExpr::Opaque;
    };
    match f.blocks[bb as usize].instrs[ii].op.clone() {
        Op::GetFlag(g, fv) => {
            if fv == flags_val {
                FlagExpr::One(g)
            } else {
                FlagExpr::Opaque
            }
        }
        Op::Bin(BinOp::And, a, b) => FlagExpr::And(
            Box::new(collect_flag_expr(f, a, flags_val)),
            Box::new(collect_flag_expr(f, b, flags_val)),
        ),
        Op::Bin(BinOp::Or, a, b) => FlagExpr::Or(
            Box::new(collect_flag_expr(f, a, flags_val)),
            Box::new(collect_flag_expr(f, b, flags_val)),
        ),
        Op::Bin(BinOp::Xor, a, b) => FlagExpr::Xor(
            Box::new(collect_flag_expr(f, a, flags_val)),
            Box::new(collect_flag_expr(f, b, flags_val)),
        ),
        Op::Un(UnOp::Not, a) => FlagExpr::Not(Box::new(collect_flag_expr(f, a, flags_val))),
        Op::Copy(a) => collect_flag_expr(f, a, flags_val),
        _ => FlagExpr::Opaque,
    }
}

// ---------------------------------------------------------------------------
// constant folding
// ---------------------------------------------------------------------------

pub fn const_fold(f: &mut Function) -> bool {
    let mut changed = false;
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let ins = f.blocks[bi].instrs[ii].clone();
            let rep: Option<Op> = match &ins.op {
                Op::Bin(op, a, b) => match (f.as_const(*a), f.as_const(*b)) {
                    (Some(x), Some(y)) => const_bin(*op, x, y, &ins),
                    _ => None,
                },
                Op::Un(op, a) => f
                    .as_const(*a)
                    .map(|x| Op::Const(const_un(*op, x, ins.ty.width().unwrap_or(Width::W64)))),
                Op::Cast(k, t, a) => f.as_const(*a).map(|x| Op::Const(const_cast(*k, t, x))),
                Op::Cmp(c, a, b) => match (f.as_const(*a), f.as_const(*b)) {
                    (Some(x), Some(y)) => {
                        let ta = f.ty_of(*a).unwrap_or(Ty::u64());
                        let (w, sx) = int_parts(&ta);
                        Some(Op::Const(c.eval(x, y, w, sx) as u64))
                    }
                    _ => None,
                },
                Op::Select(c, t, e) => {
                    if let Some(cc) = f.as_const(*c) {
                        Some(Op::Copy(if cc & 1 == 1 { *t } else { *e }))
                    } else if f.as_const(*t).is_some() && f.as_const(*t) == f.as_const(*e) {
                        f.as_const(*t).map(Op::Const)
                    } else {
                        None
                    }
                }
                Op::Copy(v) => f.as_const(*v).map(Op::Const),
                _ => None,
            };
            if let Some(op) = rep {
                let dest = ins.dest;
                f.blocks[bi].instrs[ii].op = op.clone();
                if let Op::Copy(src) = op {
                    f.replace_use_all(dest, src);
                }
                changed = true;
            }
        }
    }
    changed
}

fn int_parts(t: &Ty) -> (Width, Sx) {
    match t {
        Ty::Int { w, sx } => (*w, *sx),
        _ => (Width::W64, Sx::Unsigned),
    }
}

fn const_bin(op: BinOp, x: u64, y: u64, ins: &Instr) -> Option<Op> {
    let w = ins.ty.width()?;
    let r: u64 = match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::MulHi => ((x as u128 * y as u128) >> 64) as u64,
        BinOp::MulSmul => {
            let a = w.trunc(x, true) as i64 as i128;
            let b = w.trunc(y, true) as i64 as i128;
            ((a * b) >> 64) as u64
        }
        BinOp::UDiv => {
            if y == 0 {
                return None;
            }
            x / y
        }
        BinOp::URem => {
            if y == 0 {
                return None;
            }
            x % y
        }
        BinOp::SDiv => {
            let (a, b) = (w.trunc(x, true) as i64, w.trunc(y, true) as i64);
            if b == 0 {
                return None;
            }
            a.wrapping_div(b) as u64
        }
        BinOp::SRem => {
            let (a, b) = (w.trunc(x, true) as i64, w.trunc(y, true) as i64);
            if b == 0 {
                return None;
            }
            a.wrapping_rem(b) as u64
        }
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl => x.wrapping_shl((y & 63) as u32),
        BinOp::Shr => x.wrapping_shr((y & 63) as u32),
        BinOp::Sar => (w.trunc(x, true) as i64).wrapping_shr((y & 63) as u32) as u64,
        BinOp::RotL | BinOp::RotR => {
            let bits = w.bits().max(1);
            let m = w.mask();
            let v = x & m;
            let sh = ((y % bits as u64) as u32) % bits;
            match (bits, matches!(op, BinOp::RotL)) {
                (64, true) => v.rotate_left(sh),
                (64, false) => v.rotate_right(sh),
                (_, true) => ((v << sh) | (v >> (bits - sh).max(1))) & m,
                (_, false) => ((v >> sh) | (v << (bits - sh).max(1))) & m,
            }
        }
    };
    Some(Op::Const(w.trunc(r, false)))
}

fn const_un(op: UnOp, x: u64, w: Width) -> u64 {
    let v = w.trunc(x, false);
    match op {
        UnOp::Not => w.trunc(!x, false),
        UnOp::Neg => w.trunc(x.wrapping_neg(), false),
        UnOp::Popcount => v.count_ones() as u64,
        UnOp::Lzcnt => v.leading_zeros() as u64 - (64 - w.bits() as u64),
        UnOp::Tzcnt => {
            if v == 0 {
                w.bits() as u64
            } else {
                v.trailing_zeros() as u64
            }
        }
        UnOp::Sqrt => x,
    }
}

fn const_cast(k: CastKind, t: &Ty, x: u64) -> u64 {
    let w = t.width().unwrap_or(Width::W64);
    match k {
        CastKind::Trunc | CastKind::ZExt | CastKind::Bitcast => w.trunc(x, false),
        CastKind::SExt => w.trunc(x, true),
        _ => x,
    }
}

// ---------------------------------------------------------------------------
// copy propagation
// ---------------------------------------------------------------------------

/// `dest = copy src` is transparent in SSA: `src` is a single assignment that
/// dominates every use of `dest`, so propagation is always legal.
pub fn copy_prop(f: &mut Function) -> bool {
    let mut changed = false;
    for bi in 0..f.blocks.len() {
        let mut map: Vec<(VId, VId)> = Vec::new();
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let ins = f.blocks[bi].instrs[ii].clone();
            if let Op::Copy(src) = ins.op {
                if src != ins.dest {
                    map.push((ins.dest, src));
                }
                continue;
            }
            let mut op = ins.op.clone();
            let mut did = false;
            op.map_uses(&mut |u: &mut VId| {
                for (d, s) in map.iter() {
                    if *u == *d {
                        *u = *s;
                        did = true;
                        break;
                    }
                }
            });
            if did {
                f.blocks[bi].instrs[ii].op = op;
                changed = true;
            }
        }
    }
    for bi in 0..f.blocks.len() {
        let mut k = 0;
        while k < f.blocks[bi].instrs.len() {
            let ins = f.blocks[bi].instrs[k].clone();
            if matches!(ins.op, Op::Copy(_)) && f.use_count(ins.dest) == 0 {
                f.blocks[bi].instrs.remove(k);
                changed = true;
            } else {
                k += 1;
            }
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// DCE
// ---------------------------------------------------------------------------

pub fn dead_code_elimination(f: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let mut did = false;
        for bi in 0..f.blocks.len() {
            let mut k = 0;
            while k < f.blocks[bi].instrs.len() {
                let ins = f.blocks[bi].instrs[k].clone();
                let dead = f.use_count(ins.dest) == 0
                    && ins.op.is_pure()
                    && !matches!(ins.op, Op::Param | Op::Undef);
                if dead {
                    f.blocks[bi].instrs.remove(k);
                    did = true;
                    changed = true;
                } else {
                    k += 1;
                }
            }
        }
        if !did {
            break;
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// branch / block simplification
// ---------------------------------------------------------------------------

pub fn simplify_branches(f: &mut Function) -> bool {
    let mut changed = false;
    for bi in 0..f.blocks.len() {
        let b = f.blocks[bi].clone();
        if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = b.term
        {
            if then_bb == else_bb {
                f.blocks[bi].term = Terminator::Jump(then_bb);
                changed = true;
                continue;
            }
            if let Some(c) = f.as_const(cond) {
                f.blocks[bi].term = Terminator::Jump(if c & 1 == 1 { then_bb } else { else_bb });
                changed = true;
            }
        }
    }
    changed
}

/// Merge `a -> b` when `b`'s only predecessor is `a` and `b` has no phis.
pub fn merge_blocks(f: &mut Function) -> bool {
    let cfg = Cfg::build(f);
    for b in f.blocks.clone() {
        let succs = &cfg.succs[b.id as usize];
        if succs.len() != 1 || !matches!(b.term, Terminator::Jump(_)) || b.id == f.entry {
            continue;
        }
        let t = succs[0];
        if cfg.preds[t as usize].len() != 1 || !cfg.is_reachable(t) {
            continue;
        }
        if f.blocks[t as usize]
            .instrs
            .iter()
            .any(|i| matches!(i.op, Op::Phi(_)))
        {
            continue;
        }
        let tail = f.blocks[t as usize].instrs.clone();
        let term = f.blocks[t as usize].term.clone();
        f.blocks[b.id as usize].instrs.extend(tail);
        f.blocks[b.id as usize].term = term;
        for blk in f.blocks.iter_mut() {
            redirect(&mut blk.term, t, b.id);
        }
        f.blocks[t as usize].instrs.clear();
        f.blocks[t as usize].term = Terminator::Unreachable;
        return true;
    }
    false
}

pub(crate) fn redirect(t: &mut Terminator, from: BbId, to: BbId) {
    match t {
        Terminator::Jump(b) => {
            if *b == from {
                *b = to;
            }
        }
        Terminator::Branch {
            then_bb, else_bb, ..
        } => {
            if *then_bb == from {
                *then_bb = to;
            }
            if *else_bb == from {
                *else_bb = to;
            }
        }
        _ => {}
    }
}

/// Fold blocks with identical instruction streams (the duplicated join tails
/// that if/else lowering produces).
pub fn dedup_blocks(f: &mut Function) -> bool {
    let mut seen: HashMap<String, BbId> = HashMap::new();
    for b in f.blocks.clone() {
        if b.instrs.is_empty() || matches!(b.term, Terminator::Unreachable) {
            continue;
        }
        let key = block_key(f, &b);
        if let Some(&other) = seen.get(&key) {
            for blk in f.blocks.iter_mut() {
                redirect(&mut blk.term, b.id, other);
            }
            f.blocks[b.id as usize].instrs.clear();
            f.blocks[b.id as usize].term = Terminator::Unreachable;
            return true;
        }
        seen.insert(key, b.id);
    }
    false
}

/// Structural key: op shape with operands and dests renumbered by order of
/// appearance, so two copies of the same tail hash identically.
fn block_key(f: &Function, b: &Block) -> String {
    let mut ids: Vec<u32> = Vec::new();
    let enc = |v: u32, ids: &mut Vec<u32>| -> String {
        if let Some(i) = ids.iter().position(|x| *x == v) {
            format!("%{i}")
        } else {
            ids.push(v);
            format!("%{}", ids.len() - 1)
        }
    };
    let mut s = String::new();
    for i in &b.instrs {
        let mut ops = Vec::new();
        i.op.uses(&mut ops);
        let shape = op_shape(&i.op);
        let args: Vec<String> = ops
            .iter()
            .copied()
            .map(|o| match f.as_const(o) {
                Some(c) => format!("#{c:x}"),
                None => enc(o, &mut ids),
            })
            .collect();
        s.push_str(&format!(
            "{shape}({}) -> {}\n",
            args.join(","),
            enc(i.dest, &mut ids)
        ));
    }
    let mut succ = Vec::new();
    b.term.successors(&mut succ);
    let succ: Vec<String> = succ.iter().map(|x| enc(*x, &mut ids)).collect();
    s.push_str(&format!(
        "term {:?} {succ:?}",
        std::mem::discriminant(&b.term)
    ));
    s
}

fn op_shape(op: &Op) -> String {
    match op {
        Op::Const(c) => format!("const({c:x})"),
        Op::ConstF(c) => format!("constf({c})"),
        Op::Undef => "undef".into(),
        Op::Param => "param".into(),
        Op::Copy(_) => "copy".into(),
        Op::Bin(o, _, _) => format!("bin({o:?})"),
        Op::Un(o, _) => format!("un({o:?})"),
        Op::Cmp(c, _, _) => format!("cmp({c:?})"),
        Op::Flags(_) => "flags".into(),
        Op::GetFlag(g, _) => format!("getflag({g})"),
        Op::Cast(k, t, _) => format!("cast({k:?},{t})"),
        Op::Load(t, _) => format!("load({t})"),
        Op::Store(t, _, _) => format!("store({t})"),
        Op::Select(..) => "select".into(),
        Op::Phi(p) => format!("phi(len={})", p.len()),
        Op::Call { callee, ret_ty, .. } => format!("call({callee:?},{ret_ty})"),
        Op::Opaque(s) => format!("opaque({s})"),
    }
}

// ---------------------------------------------------------------------------
// dead store elimination
// ---------------------------------------------------------------------------

/// A store to a constant address that is overwritten before any read, with no
/// call in between, cannot be observed.
pub fn dead_store_elimination(f: &mut Function) -> bool {
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let Op::Store(_, addr_v, _) = f.blocks[bi].instrs[ii].op.clone() else {
                continue;
            };
            let Some(base) = f.as_const(addr_v) else {
                continue;
            };
            let mut kill = None;
            let mut alive = false;
            for k in ii + 1..n {
                match f.blocks[bi].instrs[k].op.clone() {
                    Op::Store(_, a2, _) => match f.as_const(a2) {
                        Some(x) if x == base => {
                            kill = Some(k);
                            break;
                        }
                        Some(_) => {}
                        None => {
                            alive = true;
                            break;
                        }
                    },
                    Op::Load(_, a2) => {
                        // Conservative: a load whose address is not a provably
                        // different constant may alias.
                        let may_alias = f.as_const(a2).map(|x| x == base).unwrap_or(true);
                        if may_alias {
                            alive = true;
                            break;
                        }
                    }
                    Op::Call { .. } => {
                        alive = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !alive {
                if let Some(k) = kill {
                    if k > ii {
                        f.blocks[bi].instrs.remove(ii);
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

pub fn notes(f: &Function) -> Vec<String> {
    let mut notes = Vec::new();
    for b in &f.blocks {
        for i in &b.instrs {
            match &i.op {
                Op::Flags(_) => {
                    if f.use_count(i.dest) > 0 {
                        notes.push(format!(
                            "bb{}@{:#x}: un-fused flag computation survived",
                            b.id, b.addr
                        ));
                    }
                }
                Op::Opaque(s) => notes.push(format!("bb{}@{:#x}: {s}", b.id, b.addr)),
                Op::Call {
                    callee: Callee::Indirect(v),
                    ..
                } => notes.push(format!(
                    "bb{}@{:#x}: indirect call through {}",
                    b.id,
                    b.addr,
                    crate::fmt_ir::vname(f, *v)
                )),
                Op::Store(Ty::Void, ..) => {
                    notes.push(format!("bb{}@{:#x}: store of unknown width", b.id, b.addr))
                }
                _ => {}
            }
        }
    }
    notes.sort();
    notes.dedup();
    notes
}

impl Function {
    /// Reuse an existing integer constant when there is one.
    pub fn add_const(&mut self, ty: Ty, v: u64, _addr: u64) -> VId {
        for b in self.blocks.iter() {
            for i in &b.instrs {
                if let Op::Const(c) = i.op {
                    if c == v && i.ty == ty {
                        return i.dest;
                    }
                }
            }
        }
        let bb = self.entry;
        self.emit(bb, Op::Const(v), ty, 0)
    }

    /// Place an op in the entry block and return its value. Used by flag
    /// lowering, which builds comparisons out of thin air.
    pub fn add_value_op(&mut self, op: Op, ty: Ty, addr: u64) -> VId {
        let bb = self.entry;
        self.emit(bb, op, ty, addr)
    }

    pub fn as_const(&self, v: VId) -> Option<u64> {
        let (bb, ii) = self.def(v)?;
        self.blocks[bb as usize].instrs[ii].op.as_const()
    }
}

impl Op {
    pub fn as_const(&self) -> Option<u64> {
        match self {
            Op::Const(c) => Some(*c),
            _ => None,
        }
    }

    /// Visit every operand slot so passes can rewrite uses in place.
    pub fn map_uses(&mut self, mut g: impl FnMut(&mut VId)) {
        match self {
            Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Opaque(_) | Op::Param => {}
            Op::Copy(a) => g(a),
            Op::Bin(_, a, b) => {
                g(a);
                g(b);
            }
            Op::Un(_, a) | Op::Cast(_, _, a) => g(a),
            Op::Cmp(_, a, b) => {
                g(a);
                g(b);
            }
            Op::Flags(fl) => {
                let (a, b) = match fl {
                    FlagsOp::Add { a, b, .. }
                    | FlagsOp::Sub { a, b, .. }
                    | FlagsOp::And { a, b }
                    | FlagsOp::Or { a, b }
                    | FlagsOp::Xor { a, b }
                    | FlagsOp::Shl { a, b }
                    | FlagsOp::Shr { a, b }
                    | FlagsOp::Sar { a, b }
                    | FlagsOp::Mul { a, b, .. } => (Some(a), Some(b)),
                    FlagsOp::Neg { a } | FlagsOp::Inc { a } | FlagsOp::Dec { a } => (Some(a), None),
                };
                if let Some(a) = a {
                    g(a);
                }
                if let Some(b) = b {
                    g(b);
                }
            }
            Op::GetFlag(_, v) => g(v),
            Op::Load(_, a) => g(a),
            Op::Store(_, a, v) => {
                g(a);
                g(v);
            }
            Op::Select(c, t, e) => {
                g(c);
                g(t);
                g(e);
            }
            Op::Phi(pairs) => {
                for (v, _) in pairs.iter_mut() {
                    g(v);
                }
            }
            Op::Call { args, .. } => {
                for a in args.iter_mut() {
                    g(a);
                }
            }
        }
    }
}
