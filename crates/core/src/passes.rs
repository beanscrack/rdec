//! Middle-end passes. Each pass is "rewrite to an equivalent function, report
//! whether anything changed", and each one now collapses to its OWN fixpoint
//! before returning (`fixpoint_bound`), so the driver's round count tracks the
//! function's structure instead of its block count. That shape is what makes a
//! partial pass list safe: no pass relies on a later one cleaning up after it.

use std::collections::HashMap;

use crate::cfg::Cfg;
use crate::ir::*;
use crate::ty::{Sx, Ty, Width};

// ---------------------------------------------------------------------------
// the pass pipeline
// ---------------------------------------------------------------------------

/// THE pre-emission pass pipeline, run to a fixpoint (bounded, not to
/// convergence: a pass that oscillates must stop and be reported, not spin).
///
/// This is one function on purpose. Flag elimination has to run here because
/// `Ty::Flags` has no C spelling, so the order of these passes IS the semantics
/// of every emitted function, and a second copy of the list - in a gate, in a
/// fuzzer - would measure its own ordering instead of the product's.
/// `rdec c`, `rdec cpp`, `rdec t2` and `rdec fuzz` all reach the IR through
/// here (`m4::clean` is a thin alias), so "the passes broke SSA" means the same
/// thing in every one of them.
pub fn clean(f: &mut Function) {
    for _ in 0..16 {
        let mut ch = false;
        ch |= crate::fuse::flag_elimination(f);
        ch |= prune_phis(f);
        ch |= const_fold(f);
        ch |= copy_prop(f);
        ch |= simplify_branches(f);
        ch |= dead_store_elimination(f);
        ch |= dead_code_elimination(f);
        if !ch {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// scale: per-pass fixpoints, and the tables that make them cheap
// ---------------------------------------------------------------------------
//
// `Function::def` / `uses_of` / `use_count` / `as_const` scan every node per query,
// which `ir.rs` documents as deliberate: the corpus's biggest function is ~40
// blocks. A pass that asks one of them per node is therefore quadratic, and the
// census says what that costs on a chain: `dead_code_elimination` was 59.9 ms of a
// 67 ms round at 757 nodes - 89% of the entire 250-block pipeline
// (`RDEC_PASS_TRACE=1`, 2026-11-06, HEAD e0dafc6) - because DCE asked `use_count`
// once per instruction and every answer walked all 757.
//
// The fix is structural and it is in two parts, both of which are needed:
//  * every pass collapses to an INTERNAL fixpoint (`fixpoint_bound` below), so the
//    driver's round count tracks the function's structure, not its block count;
//  * a pass answers its per-node questions from a table built once per sweep, so a
//    sweep costs O(nodes) instead of O(nodes^2).

/// The anti-oscillation bound for a pass's internal fixpoint.
///
/// A pass that makes progress can, in the worst case, advance one node or one block
/// per iteration, so `nodes + blocks` iterations bound any monotone pass. The
/// bound exists for the non-monotone case: a pass that oscillates must stop and be
/// reported, not spin. When it fires inside a pass, the pass returns
/// "changed" and `run_pipeline` calls it again under the same bound - so the
/// number that gets *reported* as exhausted is the driver's, never a silent
/// in-pass stop.
pub fn fixpoint_bound(f: &Function) -> usize {
    let nodes: usize = f.blocks.iter().map(|b| b.instrs.len()).sum();
    nodes
        .saturating_add(f.blocks.len())
        .saturating_add(16)
        .max(32)
}

/// Run a sweep-style pass to its internal fixpoint, bounded.
fn to_fixpoint(f: &mut Function, mut sweep: impl FnMut(&mut Function) -> bool) -> bool {
    let bound = fixpoint_bound(f);
    let mut changed = false;
    for _ in 0..bound {
        if !sweep(f) {
            break;
        }
        changed = true;
    }
    changed
}

/// `v` -> the constant it is defined as, in ONE pass over the nodes.
/// `Function::as_const` is a `def` scan per query; asking it per operand is what
/// made folding quadratic. First definition wins, matching `def`'s scan order.
fn const_table(f: &Function) -> Vec<Option<u64>> {
    let mut t = vec![None; f.values.len()];
    for b in &f.blocks {
        for i in &b.instrs {
            if let Some(c) = i.op.as_const() {
                let v = i.dest as usize;
                if v < t.len() && t[v].is_none() {
                    t[v] = Some(c);
                }
            }
        }
    }
    t
}

/// `v` -> the number of operand slots and terminator slots that read it, in ONE
/// pass. Mirrors `Function::uses_of`: every operand slot counts (a value used twice
/// by one instruction counts twice) and terminator operands count, which is the
/// rule that saved `Return` from DCE (B1).
fn use_counts(f: &Function) -> Vec<u32> {
    let mut c = vec![0u32; f.values.len()];
    let mut ops = Vec::new();
    for b in &f.blocks {
        for i in &b.instrs {
            i.op.uses(&mut ops);
            for u in ops.iter().copied() {
                if let Some(n) = c.get_mut(u as usize) {
                    *n += 1;
                }
            }
        }
        let top: Option<VId> = match b.term {
            Terminator::Branch { cond, .. } => Some(cond),
            Terminator::Return(x) | Terminator::TailReturn(x) if x != NO_VAL => Some(x),
            Terminator::Switch { index, .. } => Some(index),
            _ => None,
        };
        if let Some(v) = top {
            if let Some(n) = c.get_mut(v as usize) {
                *n += 1;
            }
        }
    }
    c
}

/// Apply a value-rewrite table in ONE pass over the nodes: a use of `v` becomes a
/// use of `v`'s target, chains resolved (`a -> b`, `b -> c` lands on `c`), with
/// terminators included, exactly like `Function::replace_use_all`. The batched form
/// is what lets a sweep fold a thousand copies without a thousand scans; a cycle
/// resolves to where the walk started, i.e. no rewrite, the conservative answer.
fn apply_rewrites(f: &mut Function, rw: &mut [VId]) {
    fn resolve(rw: &mut [VId], v: VId) -> VId {
        let mut cur = v;
        let mut hops = 0usize;
        while let Some(&next) = rw.get(cur as usize) {
            if next == cur {
                return cur;
            }
            hops += 1;
            if hops > rw.len() {
                return cur;
            }
            cur = next;
        }
        cur
    }
    for b in f.blocks.iter_mut() {
        for i in b.instrs.iter_mut() {
            i.op.map_uses(&mut |u: &mut VId| {
                let r = resolve(rw, *u);
                if r != *u {
                    *u = r;
                }
            });
        }
        match &mut b.term {
            Terminator::Branch { cond, .. } => {
                let r = resolve(rw, *cond);
                if r != *cond {
                    *cond = r;
                }
            }
            Terminator::Return(x) | Terminator::TailReturn(x) => {
                let r = resolve(rw, *x);
                if r != *x {
                    *x = r;
                }
            }
            Terminator::Switch { index, .. } => {
                let r = resolve(rw, *index);
                if r != *index {
                    *index = r;
                }
            }
            _ => {}
        }
    }
}

/// Relabel a phi's incoming edge `from -> to` (phis are the block's prefix, so the
/// walk stops at the first non-phi). A pass that redirects an EDGE must relabel the
/// phis that name that edge: `verify_ssa` requires a phi's incoming blocks to equal
/// its predecessors as a multiset, so leaving them is a verifier-visible bug, not a
/// cosmetic one.
fn rewrite_phi_from(b: &mut Block, from: BbId, to: BbId) {
    for i in b.instrs.iter_mut() {
        if !matches!(i.op, Op::Phi(_)) {
            break;
        }
        if let Op::Phi(pairs) = &mut i.op {
            for (_, bb) in pairs.iter_mut() {
                if *bb == from {
                    *bb = to;
                }
            }
        }
    }
}

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

/// Fold every foldable node, to a fixpoint. `as_const` per operand is a full scan,
/// so the sweep answers from a `const_table` and updates it as it folds; the
/// fixpoint loop is what used to be the driver's job.
pub fn const_fold(f: &mut Function) -> bool {
    to_fixpoint(f, const_fold_sweep)
}

fn const_fold_sweep(f: &mut Function) -> bool {
    let consts = const_table(f);
    let kc = |v: VId| *consts.get(v as usize)?;
    // Copy folds are collected and applied in one pass at the end of the sweep:
    // `replace_use_all` per fold is a full pass per fold, which is the quadratic
    // half of this pass. Folding decisions read `consts`, not the use lists, so
    // deferring the rewrite cannot change what this sweep folds.
    let mut rw: Vec<VId> = (0..f.values.len() as VId).collect();
    let mut folded = false;
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let ins = f.blocks[bi].instrs[ii].clone();
            let rep: Option<Op> = match &ins.op {
                Op::Bin(op, a, b) => match (kc(*a), kc(*b)) {
                    (Some(x), Some(y)) => const_bin(*op, x, y, &ins),
                    _ => None,
                },
                Op::Un(op, a) => kc(*a)
                    .map(|x| Op::Const(const_un(*op, x, ins.ty.width().unwrap_or(Width::W64)))),
                // A cast whose SOURCE is a float is not foldable by this
                // integer folder: `as_const` answers for `Op::Const` only, so a
                // The integer folder folds the BIT-level casts (Trunc/ZExt/
                // SExt/Bitcast), where the answer is the same bits read at the
                // destination width. It does NOT touch a cast that crosses the
                // int/float boundary in either direction: SIToFp/UIToFp/FPExt/
                // FPTrunc convert a VALUE, FpToSI/FpToUI answer x86's magic
                // constant where C is undefined, and `const_cast` is a bit
                // truncation for them -- folding `cvtsi2sd(-1)` through it is how
                // `-1.0` silently becomes the bit pattern 0xFFFFFFFF read as a
                // double. The conversion stays an op: the interpreter computes it
                // and `emit_c` emits the guarded C.
                Op::Cast(k, t, a)
                    if !f.ty_of(*a).map(|ty| ty.is_float()).unwrap_or(false) && !t.is_float() =>
                {
                    kc(*a).map(|x| Op::Const(const_cast(*k, t, x)))
                }
                Op::Cmp(c, a, b) => match (kc(*a), kc(*b)) {
                    (Some(x), Some(y)) => {
                        let ta = f.ty_of(*a).unwrap_or(Ty::u64());
                        let (w, sx) = int_parts(&ta);
                        Some(Op::Const(c.eval(x, y, w, sx) as u64))
                    }
                    _ => None,
                },
                Op::Select(c, t, e) => match kc(*c) {
                    Some(cc) => Some(Op::Copy(if cc & 1 == 1 { *t } else { *e })),
                    None => {
                        let (kt, ke) = (kc(*t), kc(*e));
                        match kt {
                            Some(c) if kt.is_some() && kt == ke => Some(Op::Const(c)),
                            _ => None,
                        }
                    }
                },
                Op::Copy(v) => kc(*v).map(Op::Const),
                _ => None,
            };
            if let Some(op) = rep {
                let dest = ins.dest;
                if let Op::Copy(src) = op {
                    rw[dest as usize] = src;
                }
                f.blocks[bi].instrs[ii].op = op;
                folded = true;
            }
        }
    }
    if folded {
        apply_rewrites(f, &mut rw);
    }
    folded
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
///
/// Propagates to a fixpoint inside one call. The per-block lookup is a table keyed
/// by value, not a linear scan of a pair list: after `merge_blocks` does its job a
/// block can be the whole function, and scanning a pair list per operand is
/// quadratic in that block.
pub fn copy_prop(f: &mut Function) -> bool {
    to_fixpoint(f, copy_prop_sweep)
}

fn copy_prop_sweep(f: &mut Function) -> bool {
    let mut changed = false;
    let mut prop: Vec<VId> = (0..f.values.len() as VId).collect();
    let mut touched: Vec<VId> = Vec::new();
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let ins = f.blocks[bi].instrs[ii].clone();
            if let Op::Copy(src) = ins.op {
                if src != ins.dest {
                    prop[ins.dest as usize] = src;
                    touched.push(ins.dest);
                }
                continue;
            }
            let mut op = ins.op.clone();
            let mut did = false;
            op.map_uses(&mut |u: &mut VId| {
                if let Some(&s) = prop.get(*u as usize) {
                    if s != *u {
                        *u = s;
                        did = true;
                    }
                }
            });
            if did {
                f.blocks[bi].instrs[ii].op = op;
                changed = true;
            }
        }
        for v in touched.drain(..) {
            prop[v as usize] = v;
        }
    }
    // Drop copies nobody reads. One count table per sweep answers every copy;
    // `use_count` per copy was a full scan each.
    let counts = use_counts(f);
    for bi in 0..f.blocks.len() {
        f.blocks[bi].instrs.retain(|ins| {
            let dead = matches!(ins.op, Op::Copy(_))
                && counts.get(ins.dest as usize).copied().unwrap_or(0) == 0;
            if dead {
                changed = true;
            }
            !dead
        });
    }
    changed
}

// ---------------------------------------------------------------------------
// DCE
// ---------------------------------------------------------------------------

/// Delete pure, unread nodes, cascading to a fixpoint in one call.
///
/// The predicate is unchanged (`use_count == 0 && is_pure && not a binding`), the
/// accounting is not: `Function::use_count` is a scan of every node per query, and
/// asking it per node is what made this pass 89% of the pre-fix pipeline (see the
/// census note above). One table per sweep answers every node, and a removal
/// decrements its operands' counts, so the cascade the old loop-and-rescan did
/// still happens - in the same sweep, for everything the sweep visits later.
pub fn dead_code_elimination(f: &mut Function) -> bool {
    to_fixpoint(f, dce_sweep)
}

fn dce_sweep(f: &mut Function) -> bool {
    let mut counts = use_counts(f);
    let mut did = false;
    for bi in 0..f.blocks.len() {
        let counts = &mut counts;
        f.blocks[bi].instrs.retain(|ins| {
            let dead = counts.get(ins.dest as usize).copied().unwrap_or(0) == 0
                && ins.op.is_pure()
                && !matches!(ins.op, Op::Param | Op::Undef);
            if !dead {
                return true;
            }
            let mut ops = Vec::new();
            ins.op.uses(&mut ops);
            for u in ops.iter().copied() {
                if let Some(n) = counts.get_mut(u as usize) {
                    *n -= 1;
                }
            }
            did = true;
            false
        });
    }
    did
}

// ---------------------------------------------------------------------------
// branch / block simplification
// ---------------------------------------------------------------------------

pub fn simplify_branches(f: &mut Function) -> bool {
    to_fixpoint(f, simplify_branches_sweep)
}

fn simplify_branches_sweep(f: &mut Function) -> bool {
    let mut changed = false;
    let consts = const_table(f);
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
            if let Some(Some(c)) = consts.get(cond as usize) {
                f.blocks[bi].term = Terminator::Jump(if c & 1 == 1 { then_bb } else { else_bb });
                changed = true;
            }
        }
    }
    changed
}

/// Merge `a -> b` when `b`'s only predecessor is `a` and `b` has no phis.
///
/// One call collapses EVERY eligible edge, sweeping until none is left: a chain
/// halves per sweep, so a 20k-block chain settles in ~15 sweeps and the driver's
/// round budget stops being a function of the block count. The old shape returned
/// after the first merge, which is what made a 250-block chain "run out of budget"
/// (measured at HEAD e0dafc6: `merge_blocks changed=true` in 32 of 32 rounds, live
/// blocks 252 -> 220, one per round, ~220 merges still owed).
pub fn merge_blocks(f: &mut Function) -> bool {
    merge_blocks_counted(f, fixpoint_bound(f)).0
}

/// The counted form `run_pipeline` drives: (changed, sweeps, bound_hit).
pub(crate) fn merge_blocks_counted(f: &mut Function, bound: usize) -> (bool, usize, bool) {
    let mut cfg = Cfg::build(f);
    let mut changed = false;
    let mut sweeps = 0;
    for _ in 0..bound.max(1) {
        sweeps += 1;
        let mut did = false;
        for bi in 0..f.blocks.len() {
            if merge_one_edge(f, &mut cfg, bi as BbId) {
                did = true;
                changed = true;
            }
        }
        if !did {
            return (changed, sweeps, false);
        }
    }
    (changed, sweeps, true)
}

/// Collapse `b -> t` into `b`. `preds[t] == [b]`, so `b`'s own terminator is the
/// only terminator that can name `t` - the CFG tables are maintained in place
/// instead of rescanning every block per merge, which is what made the old
/// per-merge cost a full pass over the function.
fn merge_one_edge(f: &mut Function, cfg: &mut Cfg, b: BbId) -> bool {
    if b == f.entry || !matches!(f.blocks[b as usize].term, Terminator::Jump(_)) {
        return false;
    }
    if cfg.succs[b as usize].len() != 1 {
        return false;
    }
    let t = cfg.succs[b as usize][0];
    if cfg.preds[t as usize].len() != 1 || !cfg.is_reachable(t) {
        return false;
    }
    if matches!(f.blocks[t as usize].term, Terminator::Unreachable) {
        return false;
    }
    if f.blocks[t as usize]
        .instrs
        .iter()
        .any(|i| matches!(i.op, Op::Phi(_)))
    {
        return false;
    }
    // A successor whose phi names BOTH b and t would end up with two incoming pairs
    // for one predecessor (`verify_ssa`: pairs == preds as a multiset). Refuse the
    // merge there: the old single-merge pass had the same shape, and naming it keeps
    // the multi-merge sweep from making it more common.
    for s in cfg.succs[t as usize].iter().copied() {
        let p = &cfg.preds[s as usize];
        if p.contains(&b) && p.contains(&t) {
            return false;
        }
    }
    let tail = std::mem::take(&mut f.blocks[t as usize].instrs);
    f.blocks[b as usize].instrs.extend(tail);
    f.blocks[b as usize].term = f.blocks[t as usize].term.clone();
    f.blocks[t as usize].term = Terminator::Unreachable;
    // `b`'s new terminator may name `t` (the 2-cycle `b -> t -> b`): redirect it, as
    // the old "redirect every block" loop did.
    redirect(&mut f.blocks[b as usize].term, t, b);
    let tsucc = std::mem::take(&mut cfg.succs[t as usize]);
    for &s in tsucc.iter() {
        rewrite_phi_from(&mut f.blocks[s as usize], t, b);
        let ps = &mut cfg.preds[s as usize];
        ps.retain(|p| *p != t);
        if !ps.contains(&b) {
            ps.push(b);
        }
    }
    cfg.preds[t as usize].clear();
    cfg.succs[b as usize] = tsucc.iter().map(|&s| if s == t { b } else { s }).collect();
    true
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
        // A table slot redirected at a merged block names the survivor, exactly
        // as a branch target does. Skipping this would leave the slot pointing
        // at an emptied block.
        Terminator::Switch { targets, .. } => {
            for t in targets.iter_mut() {
                if *t == from {
                    *t = to;
                }
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
    to_fixpoint(f, dedup_one_pair)
}

/// Collapse one pair of structurally identical blocks. `dedup_blocks` is the
/// fixpoint over this; keeping the pair-collapse itself small and single-purpose is
/// what makes the B2 sabotage (merge blocks with different operands) visible.
fn dedup_one_pair(f: &mut Function) -> bool {
    let consts = const_table(f);
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
        let (key, map) = block_key_map_with(&f.blocks[bi], &consts);
        let hit = seen.get(&key).cloned();
        if let Some((other, map_o)) = hit {
            // The dying block's successors keep reading the survivor: their phis
            // name the dying edge, and the arm that arrives through it carries one
            // of the dying block's locals. `verify_ssa` requires a phi's incoming
            // blocks to equal its predecessors as a multiset, so the merge must
            // rewrite value AND label, and the pair count must still line up.
            let mut succ = Vec::new();
            f.blocks[bi].term.successors(&mut succ);
            let mut value_map: Vec<(VId, VId)> = Vec::new();
            for (i, v) in map.iter().enumerate() {
                if map_o[i] != *v {
                    value_map.push((*v, map_o[i]));
                }
            }
            if !phi_merge_ok(f, &succ, bid, other, &value_map) {
                // Skip THIS pair, keep looking: returning here would tell the
                // fixpoint loop there is nothing left to do.
                continue;
            }
            for (from, to) in value_map.iter().copied() {
                f.replace_use_all(from, to);
            }
            for &s in succ.iter() {
                if (s as usize) < f.blocks.len() {
                    collapse_phi_edge(&mut f.blocks[s as usize], bid, other);
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

/// Would the phis of `succ` still satisfy `verify_ssa`'s rule - incoming blocks ==
/// predecessors as a multiset - after `bid` is redirected at `other` and its locals
/// are rewritten by `value_map`?
///
/// The dedup's own value rewrite makes the two arms of the merged pair IDENTICAL
/// (that is what "structurally identical blocks" means), and identical pairs
/// collapse to one, which is the pair `verify_ssa` expects for the one surviving
/// predecessor. Anything that does not line up - two arms naming one block with
/// DIFFERENT values, e.g. a block with two edges into the same successor - is
/// refused rather than silently collapsed: a phi whose arms disagree is a value,
/// and dropping one is the silent-loss class of bug.
fn phi_merge_ok(
    f: &Function,
    succ: &[BbId],
    bid: BbId,
    other: BbId,
    value_map: &[(VId, VId)],
) -> bool {
    let map_v = |v: VId| -> VId {
        value_map
            .iter()
            .find(|(from, _)| *from == v)
            .map(|(_, to)| *to)
            .unwrap_or(v)
    };
    for &s in succ {
        // The predecessors `Cfg::build` will report: one entry per source block,
        // with the dying edge relabelled.
        let mut preds: Vec<BbId> = Vec::new();
        for blk in &f.blocks {
            let mut ss = Vec::new();
            blk.term.successors(&mut ss);
            let to = if blk.id == bid { other } else { blk.id };
            if ss.contains(&s) && !preds.contains(&to) {
                preds.push(to);
            }
        }
        for i in &f.blocks[s as usize].instrs {
            let Op::Phi(pairs) = &i.op else { break };
            let mut got: Vec<(VId, BbId)> = Vec::new();
            for (v, from) in pairs.iter() {
                let p = (map_v(*v), if *from == bid { other } else { *from });
                if !got.contains(&p) {
                    got.push(p);
                }
            }
            let mut got_bb: Vec<BbId> = got.iter().map(|(_, b)| *b).collect();
            got_bb.sort();
            let mut want = preds.clone();
            want.sort();
            if got_bb != want {
                return false;
            }
        }
    }
    true
}

/// Relabel a phi's dying edge and drop the pairs the relabel made identical
/// (`verify_ssa`: pairs == predecessors as a multiset, one entry per source block).
/// Only identical `(value, block)` pairs collapse, so a phi whose arms disagree is
/// never truncated - `phi_merge_ok` refuses that merge before it happens.
fn collapse_phi_edge(b: &mut Block, bid: BbId, other: BbId) {
    for i in b.instrs.iter_mut() {
        if !matches!(i.op, Op::Phi(_)) {
            break;
        }
        if let Op::Phi(pairs) = &mut i.op {
            let mut seen: Vec<(VId, BbId)> = Vec::new();
            let mut keep: Vec<(VId, BbId)> = Vec::new();
            for (v, from) in pairs.iter().copied() {
                let p = (v, if from == bid { other } else { from });
                if !seen.contains(&p) {
                    seen.push(p);
                    keep.push(p);
                }
            }
            *pairs = keep.into_boxed_slice();
        }
    }
}

/// Structural key plus the canonical mapping (index -> value) it implies.
pub fn block_key_map(f: &Function, b: &Block) -> (String, Vec<VId>) {
    let consts = const_table(f);
    block_key_map_with(b, &consts)
}

/// The keyed form used inside `dedup_blocks`, where the constant table is built
/// once per call rather than once per block.
fn block_key_map_with(b: &Block, consts: &[Option<u64>]) -> (String, Vec<VId>) {
    // Only values **defined inside the block** may be renumbered positionally: two
    // blocks that compute the same expression from *different* inputs are not the
    // same computation, whatever the shape looks like after renumbering. So free
    // operands (params, anything defined in a dominating block) and successor
    // targets are encoded by identity, exactly like constants. v1 renumbered those
    // too, which is how `add p0,p1` and `add p2,p3` hashed equal (§9 B2).
    //
    // Membership and position are answered from tables, not from linear scans of
    // the local list: the old `contains`/`position` per operand is quadratic in the
    // block's size, and a merged block can be the whole function.
    let mut is_local: HashMap<VId, ()> = HashMap::new();
    for i in &b.instrs {
        is_local.insert(i.dest, ());
    }
    let mut ids: Vec<u32> = Vec::new();
    let mut pos: HashMap<VId, usize> = HashMap::new();
    let mut enc = |v: u32| -> String {
        if !is_local.contains_key(&v) {
            // Free operand: identity, not position.
            return format!("v{v}");
        }
        if let Some(i) = pos.get(&v) {
            return format!("%{i}");
        }
        pos.insert(v, ids.len());
        ids.push(v);
        format!("%{}", ids.len() - 1)
    };
    let mut s = String::new();
    for i in &b.instrs {
        let mut ops = Vec::new();
        i.op.uses(&mut ops);
        let shape = op_shape(&i.op);
        let args: Vec<String> = ops
            .iter()
            .copied()
            .map(|o| match consts.get(o as usize).copied().flatten() {
                Some(c) => format!("#{c:x}"),
                None => enc(o),
            })
            .collect();
        s.push_str(&format!("{shape}({}) -> {}\n", args.join(","), enc(i.dest)));
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
        // The float ops keep their own identity: a diff that says
        // "bin(Add) -> fbin(Add)" is the decompiler's own bug report, and
        // collapsing them into one shape would hide it.
        Op::FBin(o, _, _) => format!("fbin({o:?})"),
        Op::FCmp(c, _, _) => format!("fcmp({c:?})"),
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
        // Its own shape: two blocks that clobber at the same point may be
        // merged, and a clobber must never key-match an opaque gap.
        Op::Clobber => "clobber".into(),
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
            Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Opaque(_) | Op::Clobber | Op::Param => {}
            Op::Copy(a) => g(a),
            Op::Bin(_, a, b) => {
                g(a);
                g(b);
            }
            // A float op's operands are VIds like any other: a pass that
            // rewrites uses in place and skips these leaves a value whose users
            // point at the OLD definition, which is the dangling-reference class
            // `interp::fetch` faults on. Register promotion hit exactly this
            // before the arm existed.
            Op::FBin(_, a, b) | Op::FCmp(_, a, b) => {
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
