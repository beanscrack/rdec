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
//
// v1's implementation lived here and roadmap SS9 confirmed it broken four ways
// (B3-B6): per-leaf lowering made the jcc matrix unreachable, SF of a `cmp` was
// read as "a < 0", the `Flags` value was used as the instruction's integer
// result, and the exact fallback covered only CF/OF/ZF/SF of add/sub so a
// `Flags`-typed value could survive the pass. It is deleted, not patched: the
// replacement lives in `crate::fuse` and is correct by construction (match the
// tree at its root, recompute the result from the operands, total fallback with
// loud `Opaque` gaps).

pub use crate::fuse::cmp_pair;

/// Lower flag reads into comparisons over the operands. See `crate::fuse`.
pub fn flag_elimination(f: &mut Function) -> bool {
    crate::fuse::flag_elimination(f)
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
///
/// Two blocks are mergeable when they are isomorphic *modulo the positional
/// mapping of their local values*, which is exactly what `block_key` computes.
/// The merge is therefore only sound if the uses are rewritten along that
/// mapping: v1 redirected the edges and dropped the block but left every use of
/// its locals pointing at values that no longer existed in a live block, so
/// `x = a + 1` / `return x` and `y = b + 1` / `return y` collapsed to whichever
/// `%1` hashed first and the function returned a constant 3 for every input
/// (§9 B2).
pub fn dedup_blocks(f: &mut Function) -> bool {
    let mut seen: HashMap<String, (BbId, Vec<VId>)> = HashMap::new();
    for bi in 0..f.blocks.len() {
        let bid = f.blocks[bi].id;
        if f.blocks[bi].instrs.is_empty() || matches!(f.blocks[bi].term, Terminator::Unreachable) {
            continue;
        }
        // A phi's incoming values come from its predecessors, so two phi blocks
        // can share a canonical shape while reading different values. Positional
        // canonicalisation cannot see that, so phi blocks are not deduped here;
        // phi-aware value numbering is M3's GVN.
        if f.blocks[bi]
            .instrs
            .iter()
            .any(|i| matches!(i.op, Op::Phi(_)))
        {
            continue;
        }
        let (key, map) = block_key_map(f, &f.blocks[bi]);
        let hit = seen.get(&key).cloned();
        if let Some((other, map_o)) = hit {
            for (i, v) in map.iter().enumerate() {
                if map_o[i] != *v {
                    f.replace_use_all(*v, map_o[i]);
                }
            }
            for blk in f.blocks.iter_mut() {
                redirect(&mut blk.term, bid, other);
            }
            f.blocks[bi].instrs.clear();
            f.blocks[bi].term = Terminator::Unreachable;
            return true;
        }
        seen.insert(key, (bid, map));
    }
    false
}

/// Structural key plus the canonical mapping (index -> value) it implies.
///
/// Only values **defined inside the block** may be renumbered positionally: two
/// blocks that compute the same expression from *different* inputs are not the
/// same computation, whatever the shape looks like after renumbering. So free
/// operands (params, anything defined in a dominating block) and successor
/// targets are encoded by identity, exactly like constants. v1 renumbered those
/// too, which is how `add p0,p1` and `add p2,p3` hashed equal (§9 B2).
pub fn block_key_map(f: &Function, b: &Block) -> (String, Vec<VId>) {
    let locals: Vec<VId> = b.instrs.iter().map(|i| i.dest).collect();
    let mut ids: Vec<u32> = Vec::new();
    let enc = |v: u32, ids: &mut Vec<u32>| -> String {
        if !locals.contains(&v) {
            // Free operand: identity, not position.
            return format!("v{v}");
        }
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
    let succ: Vec<String> = succ.iter().map(|x| format!("bb{x}")).collect();
    s.push_str(&format!(
        "term {:?} {succ:?}",
        std::mem::discriminant(&b.term)
    ));
    (s, ids)
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
