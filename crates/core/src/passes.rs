//! Middle-end passes. Each pass is "rewrite to an equivalent function, report
//! whether anything changed", and each one now collapses to its OWN fixpoint
//! before returning (`fixpoint_bound`), so the driver's round count tracks the
//! function's structure instead of its block count. That shape is what makes a
//! partial pass list safe: no pass relies on a later one cleaning up after it.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::cfg::Cfg;
use crate::ir::*;
use crate::structure::value_readable;
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
/// No definition: a value id no instruction in this function writes.
const NO_DEF: (usize, usize) = (usize::MAX, usize::MAX);

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
                // SExt/Bitcast), where the answer is the same bits re-read at the
                // widths the cast names — for SExt the SOURCE width decides the
                // sign, so the source type has to ride into `const_cast`.
                // It does NOT touch a cast that crosses the
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
                    // The source type is an INPUT to the fold, not decoration:
                    // SExt reads its sign from the source width, and `t` — the
                    // destination, always the wider one — cannot answer that
                    // question. `Ty::u64()` is the conservative stand-in for a
                    // source with no recorded type: at W64 a sign-extension is
                    // the identity, which is what an untyped cell means here.
                    kc(*a).map(|x| {
                        let st = f.ty_of(*a).unwrap_or(Ty::u64());
                        Op::Const(const_cast(*k, t, &st, x))
                    })
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

fn const_cast(k: CastKind, t: &Ty, src: &Ty, x: u64) -> u64 {
    let w = t.width().unwrap_or(Width::W64);
    match k {
        CastKind::Trunc | CastKind::ZExt | CastKind::Bitcast => w.trunc(x, false),
        CastKind::SExt => {
            // Sign extension is a statement about the SOURCE's top bit. The
            // destination of an SExt is wider by construction, so reading the
            // sign from the DESTINATION width asks "is bit `dst-1` set?" of a
            // value that lives in the low `src` bits — always 0 — and every
            // folded sign-extension silently became a zero-extension. Measured
            // before the fix: `mov eax,0x12348123 ; cwde` folded to
            // Const(0x8123) where the machine answers 0xFFFF8123 (and the same
            // function with the value in ECX, unfolded, was already correct —
            // the miscompile appeared only once the constant folded).
            // Sign from the source, then keep the destination's bits: a W32
            // cell holds 0xFFFF8123, not a sign-extended 64-bit value, because
            // a 32-bit write zeroes bits 63:32.
            let sw = src.width().unwrap_or(w);
            w.trunc(sw.trunc(x, true), false)
        }
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
// the epilogue lane: a pure return tail moves onto the edges that reach it
// ---------------------------------------------------------------------------
//
// `structure.rs::tail_of` may spell a jump site `return` only when the target
// block computes NOTHING - every instruction a phi, the terminator a Return, the
// value readable at the site - because a `return` has no room for a statement.
// The census at base 47b9076 prints 29 refusals of exactly that shape, and the
// measurement over every emitted corpus function (this lane, base 47b9076) says
// what is inside those 29 blocks:
//
// ```text
//   8  Const                              (mov $5,%eax; ret)
//   5  Const, Load(u32), Cast(ZExt)       (mov -0x8(%rbp),%eax; ret)
//   1  Const, Load, Cast, Cast, Const, Load, Add, Cast
// ```
//
// and NOT what this lane was handed as its premise: no RSP/RBP teardown write
// survives in a corpus return-target block (every `mov rsp,rbp` / `pop rbp` the
// lifter emits is a pure SSA node the `Return` does not name, so `use_count == 0
// && is_pure` already deletes it - pinned by
// `tests/epilogue_dce.rs::frame_teardown_on_a_return_only_path_is_already_gone`),
// and no `Op::Store` sits in one of the 29 blocks either.
//
// So the win is not a deletion, it is a MOVEMENT. A tail block whose work is the
// whole distance between a predecessor's terminator and the `return` runs that
// work at the end of the predecessor too - same program point, same values.
// Copying a pure tail chain there, once per incoming edge, with a phi standing in
// the tail for the copy that edge ran, turns the target into the pure tail the
// printer already accepts, and M6 spells `return` with no change to
// `structure.rs`. The copies are new SSA nodes; the originals become unread and
// the DCE sweep in the same fixpoint deletes them, so the net is a move.
//
// Three lines this pass does not cross, each with the oracle that would see it:
//  * **a store is not deletable, movable, or copyable.** `interp.rs` pushes
//    `Effect::Write` for every `Op::Store` and `Exec::diff` compares the ordered
//    effect trace and the final memory, so a store that vanishes changes the IR's
//    own meaning - even though T2's memory axis compares CALLER BUFFERS at
//    `RDEC_WINDOW_ADDR` and would not see a frame write disappear. T2 not seeing
//    it is not the same as it being dead: the frame-slot STORE that materializes
//    a return value stays exactly where it is, which is why the goal is "29 ->
//    fewer", not "29 -> 0 at any cost".
//  * **work moves onto an edge only if it cannot fault on that edge.**
//    `UDiv`/`URem`/`SDiv`/`SRem` raise `Fault::DivZero` (and `SDiv`
//    `Fault::IntMinDivMinusOne`), `GetFlag` raises `Fault::UndefFlag`. A `Jump`
//    predecessor has the tail as its ONLY successor, so a copy runs on exactly
//    the paths the tail ran and the full pure set may move; a `Branch`/`Switch`
//    predecessor reaches arms the tail never ran, so only cannot-fault work moves.
//  * **a free operand of the chain must be readable at the predecessor**
//    (`structure::value_readable`: defined, and its def block dominates the
//    consumer). A chain that reads a value computed in a sibling has work that
//    does not exist at the edge, and this pass does not speculate.

/// What the epilogue lane did, measured rather than assumed: `m4.rs`'s census
/// prints these on one line, because a win that is not printed is assumed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EpilogueCounts {
    /// Return blocks that ended the sweep computing nothing (the M6 win).
    pub tails_materialized: usize,
    /// Edges that received a copy of a tail chain.
    pub edges_materialized: usize,
    /// Chains copied onto a `Jump` edge, where the tail is the only successor.
    pub chains_on_jump_edges: usize,
    /// Chains copied onto a `Branch`/`Switch` edge, restricted to the
    /// cannot-fault subset.
    pub chains_on_branch_edges: usize,
    /// Tails kept because they store, call, read a flag, or otherwise are not
    /// pure work.
    pub kept_effectful: usize,
    /// Tails kept because a faulting op (an integer division, a flag read) would
    /// have run on an arm that never ran the tail.
    pub kept_faulting: usize,
    /// Tails kept because a chain operand is not readable on an incoming edge.
    pub kept_unreadable: usize,
    /// Tails kept because their value is free (not computed in the tail), so the
    /// ordinary DCE sweep - not this pass - owns them.
    pub kept_free_value: usize,
    /// Frame-slot stores deleted. No code path can raise this: a store is never
    /// deletable in this IR (see the section comment), and the census prints the
    /// zero so the number is on the record.
    pub frame_stores_deleted: usize,
    /// Pure, unread nodes DCE deleted out of a block that ends in `Return` - the
    /// IR proxy for a Win64 frame-teardown write on a return-only path, because a
    /// `mov rsp,rbp` / `pop rbp` lowers to nodes the `Return` does not name.
    ///
    /// Scope, so the printed number is not read as more than it is: this is a
    /// per-process total of the deletions DCE performed, counted in every sweep,
    /// and the validated pipeline runs passes on a copy of the function as well as
    /// on the function itself. At base 47b9076 the number that matters - teardown
    /// writes surviving inside the 29 refused return-target blocks - is 0, and
    /// `tests/epilogue_dce.rs::frame_teardown_on_a_return_only_path_is_already_gone`
    /// pins why.
    pub teardown_writes_deleted: usize,
}

thread_local! {
    static EPILOGUE: RefCell<EpilogueCounts> = const {
        RefCell::new(EpilogueCounts {
            tails_materialized: 0,
            edges_materialized: 0,
            chains_on_jump_edges: 0,
            chains_on_branch_edges: 0,
            kept_effectful: 0,
            kept_faulting: 0,
            kept_unreadable: 0,
            kept_free_value: 0,
            frame_stores_deleted: 0,
            teardown_writes_deleted: 0,
        })
    };
}

fn bump_epilogue(what: impl FnOnce(&mut EpilogueCounts)) {
    EPILOGUE.with(|c| what(&mut c.borrow_mut()));
}

/// The epilogue counters as they stand.
pub fn epilogue_counts() -> EpilogueCounts {
    EPILOGUE.with(|c| *c.borrow())
}

/// Zero the epilogue counters, so a command that prints the census reports this
/// run's work and not an earlier command's.
pub fn reset_epilogue_counts() {
    EPILOGUE.with(|c| *c.borrow_mut() = EpilogueCounts::default());
}

/// The census line for the epilogue lane, drained (a second call reports zeros).
pub fn epilogue_line() -> String {
    let c = epilogue_counts();
    reset_epilogue_counts();
    format!(
        "epilogue: {} return tails materialized on {} edges ({} jump, {} branch) \
         | {} kept ({} store/call/flag, {} faults on an arm it did not run, {} unreadable \
         at the edge, {} value computed outside the tail) \
         | frame stores deleted: {} | teardown writes deleted: {} \
         (tallies count every DCE sweep, including the validated pipeline's copies)",
        c.tails_materialized,
        c.edges_materialized,
        c.chains_on_jump_edges,
        c.chains_on_branch_edges,
        c.kept_effectful + c.kept_faulting + c.kept_unreadable + c.kept_free_value,
        c.kept_effectful,
        c.kept_faulting,
        c.kept_unreadable,
        c.kept_free_value,
        c.frame_stores_deleted,
        c.teardown_writes_deleted,
    )
}

/// Can copying `op` onto an arm that never ran it change an execution? In this IR
/// a `Load` cannot fault (`Memory::load` answers an out-of-range read as 0, and
/// the edge-legality check preserves the address operand's definedness),
/// `FpToSI` returns the x86 magic constant, and a float division returns an
/// infinity - so the faulting set is the integer divisions and the flag read,
/// exactly the `Fault` variants `interp.rs` raises for them.
fn op_may_fault(op: &Op) -> bool {
    match op {
        Op::Bin(o, ..) => matches!(o, BinOp::UDiv | BinOp::URem | BinOp::SDiv | BinOp::SRem),
        Op::GetFlag(..) => true,
        _ => false,
    }
}

/// Copying a tail node onto an edge is legal exactly when it is pure
/// (`Op::is_pure`) AND it is a plain value node: a `Phi` is SSA's join and by the
/// prefix rule can never sit in a chain, and a `Flags` producer feeds `GetFlag`
/// reads the emitter refuses to print, so neither is work this pass will move.
fn tail_replicable(op: &Op) -> bool {
    op.is_pure() && !matches!(op, Op::Phi(_) | Op::Flags(_))
}

/// Rewrite a copied node's operands through the per-edge map: a chain node maps
/// to the copy made for this edge, a tail phi maps to the pair that arrives on
/// the edge, and a free operand names the same value at the edge, so it passes
/// through.
fn map_op(op: &Op, map: &HashMap<VId, VId>) -> Op {
    let m = |u: VId| map.get(&u).copied().unwrap_or(u);
    match op {
        Op::Undef => Op::Undef,
        Op::Const(n) => Op::Const(*n),
        Op::ConstF(b) => Op::ConstF(*b),
        Op::Param => Op::Param,
        Op::Copy(x) => Op::Copy(m(*x)),
        Op::GetFlag(k, v) => Op::GetFlag(*k, m(*v)),
        Op::Bin(o, x, y) => Op::Bin(*o, m(*x), m(*y)),
        Op::Un(o, x) => Op::Un(*o, m(*x)),
        Op::Cmp(c, x, y) => Op::Cmp(*c, m(*x), m(*y)),
        Op::FCmp(c, x, y) => Op::FCmp(*c, m(*x), m(*y)),
        Op::FBin(o, x, y) => Op::FBin(*o, m(*x), m(*y)),
        Op::Cast(k, t, x) => Op::Cast(*k, t.clone(), m(*x)),
        Op::Load(t, a) => Op::Load(t.clone(), m(*a)),
        Op::Select(c, x, y) => Op::Select(m(*c), m(*x), m(*y)),
        // Unreachable: a node outside `tail_replicable` is refused before planning.
        Op::Store(..)
        | Op::Call { .. }
        | Op::Phi(_)
        | Op::Flags(_)
        | Op::Opaque(_)
        | Op::Clobber => op.clone(),
    }
}

/// Move a pure return tail onto the edges that reach it, so the return block
/// computes nothing and `structure.rs`'s existing rule turns the jump site into
/// `return`.
///
/// Per candidate: the target ends in `Return(v)` naming a value THIS block
/// computes; every non-phi instruction in it is pure; every operand of that
/// chain is a chain node, one of the block's own phis (replaced on an edge by the
/// pair that arrives on it), or a value readable at every predecessor. Each
/// predecessor then gets its own copy of the chain at its end, and the target
/// keeps its phis plus one new phi for `v`, so `verify_ssa`'s two phi rules hold:
/// pairs equal the predecessor multiset, and the phi stays inside the phi prefix.
///
/// The number of non-phi instructions inside `Return` blocks strictly decreases
/// per rewrite - copies land in blocks that end in a `Jump`/`Branch`/`Switch`,
/// never in a `Return` block - so the enclosing fixpoint terminates.
pub fn materialize_return_tails(f: &mut Function) -> bool {
    let cfg = Cfg::build(f);
    let mut changed = false;
    for ti in 0..f.blocks.len() {
        let tail = f.blocks[ti].id;
        // Cloned, because building the copies allocates new values in `f` and the
        // tail's nodes are the read side of that work. Tail blocks are small; the
        // corpus's biggest is 8 instructions.
        let blk_instrs: Vec<Instr> = f.blocks[ti].instrs.clone();
        let blk_term: Terminator = f.blocks[ti].term.clone();
        let v = match blk_term {
            Terminator::Return(v) if v != NO_VAL => v,
            _ => continue,
        };
        if tail == f.entry {
            continue;
        }
        let preds: Vec<BbId> = cfg.preds[ti].clone();
        if preds.is_empty() {
            continue; // unreachable tail: its unread pure nodes are plain DCE's
        }
        let phi_len = blk_instrs
            .iter()
            .take_while(|i| matches!(i.op, Op::Phi(_)))
            .count();
        if phi_len == blk_instrs.len() {
            continue; // already a pure tail: the printer needs nothing from us
        }
        let chain = &blk_instrs[phi_len..];
        if !chain.iter().any(|i| i.dest == v) {
            // The returned value is free here, so it is already readable at every
            // pred and the tail's own nodes are pure-and-unread: plain DCE's work.
            bump_epilogue(|c| c.kept_free_value += 1);
            continue;
        }
        let mut effectful = false;
        let mut fault_free = true;
        for ins in chain {
            if !tail_replicable(&ins.op) {
                effectful = true;
            } else if op_may_fault(&ins.op) {
                fault_free = false;
            }
        }
        if effectful {
            bump_epilogue(|c| c.kept_effectful += 1);
            continue;
        }
        let chain_dest: HashSet<VId> = chain
            .iter()
            .filter(|i| i.dest != NO_VAL)
            .map(|i| i.dest)
            .collect();
        let mut phi_dests: HashSet<VId> = HashSet::new();
        for ins in &blk_instrs[..phi_len] {
            if ins.dest != NO_VAL {
                phi_dests.insert(ins.dest);
            }
        }
        let mut unreadable = false;
        let mut ops = Vec::new();
        for ins in chain {
            ins.op.uses(&mut ops);
            for u in ops.iter().copied() {
                if chain_dest.contains(&u) || phi_dests.contains(&u) {
                    continue;
                }
                if !preds.iter().all(|p| value_readable(f, &cfg, u, *p)) {
                    unreadable = true;
                }
            }
        }
        if unreadable {
            bump_epilogue(|c| c.kept_unreadable += 1);
            continue;
        }
        // A `Jump` predecessor has this tail as its only successor, so a copy
        // runs on exactly the paths the tail ran. Anything else reaches arms the
        // tail never ran, and only cannot-fault work may go there.
        let jump_edge = preds.iter().all(|p| {
            matches!(f.blocks[*p as usize].term,
                Terminator::Jump(t) if t == tail)
        });
        if !jump_edge && !fault_free {
            bump_epilogue(|c| c.kept_faulting += 1);
            continue;
        }
        // Build this tail's per-edge copies. Every chain node has a dest (the
        // pure set never includes a store, a clobber, or a call), and the chain is
        // in definition order, so a forward map is enough.
        let mut per_edge: Vec<(BbId, (VId, Vec<Instr>))> = Vec::with_capacity(preds.len());
        let mut ok = true;
        for p in &preds {
            let mut map: HashMap<VId, VId> = HashMap::new();
            for ins in &blk_instrs[..phi_len] {
                if let Op::Phi(pairs) = &ins.op {
                    match pairs.iter().find(|(_, b)| *b == *p) {
                        Some((pv, _)) => {
                            map.insert(ins.dest, *pv);
                        }
                        // A phi with no pair for a predecessor is a verify_ssa
                        // error already; do not invent an arm for it.
                        None => ok = false,
                    }
                }
            }
            if !ok {
                break;
            }
            let mut copies: Vec<Instr> = Vec::with_capacity(chain.len());
            for ins in chain {
                let op = map_op(&ins.op, &map);
                let d = if ins.dest == NO_VAL {
                    NO_VAL
                } else {
                    let d = f.new_value(ins.ty.clone());
                    map.insert(ins.dest, d);
                    d
                };
                copies.push(Instr::new(d, op, ins.ty.clone(), ins.addr));
            }
            // The copy of the value the tail returns. `v` is a chain node (checked
            // above), so its copy is in the map.
            let mapped_v = match map.get(&v) {
                Some(d) => *d,
                None => {
                    ok = false;
                    break;
                }
            };
            per_edge.push((*p, (mapped_v, copies)));
        }
        if !ok {
            bump_epilogue(|c| c.kept_unreadable += 1);
            continue;
        }
        let phi_ty = match f.ty_of(v) {
            Some(t) => t,
            None => continue,
        };
        let phi_dest = f.new_value(phi_ty.clone());
        let phi_addr = chain
            .iter()
            .find(|i| i.dest == v)
            .map(|i| i.addr)
            .unwrap_or(0);
        for (p, (_, copies)) in &per_edge {
            f.blocks[*p as usize].instrs.extend(copies.iter().cloned());
        }
        let tail_blk = &mut f.blocks[ti];
        tail_blk.instrs.splice(
            phi_len..phi_len,
            [Instr::new(
                phi_dest,
                Op::Phi(
                    per_edge
                        .iter()
                        .map(|(p, (mv, _))| (*mv, *p))
                        .collect::<Vec<_>>()
                        .into(),
                ),
                phi_ty,
                phi_addr,
            )],
        );
        tail_blk.term = Terminator::Return(phi_dest);
        bump_epilogue(|c| {
            c.tails_materialized += 1;
            c.edges_materialized += per_edge.len();
            if jump_edge {
                c.chains_on_jump_edges += 1;
            } else {
                c.chains_on_branch_edges += 1;
            }
        });
        changed = true;
    }
    changed
}

// ---------------------------------------------------------------------------
// DCE
// ---------------------------------------------------------------------------

/// Delete pure, unread nodes, cascading to a fixpoint in one call, and - in the
/// same fixpoint - move pure return tails onto the edges that reach them (see the
/// epilogue lane above). The two belong together: the materialization makes the
/// tail's nodes unread, and this sweep is what deletes them.
pub fn dead_code_elimination(f: &mut Function) -> bool {
    to_fixpoint(f, |f| {
        let moved = materialize_return_tails(f);
        moved || dce_sweep(f)
    })
}

/// Is this node deletable once its use count is 0? The rule is unchanged from the
/// sweep's original: pure, and not a `Param` (a function's parameter is its
/// signature, not a computation) and not an `Undef` (a hole the emitter prints).
fn dce_deletable(ins: &Instr, counts: &[u32]) -> bool {
    counts.get(ins.dest as usize).copied().unwrap_or(0) == 0
        && ins.op.is_pure()
        && !matches!(ins.op, Op::Param | Op::Undef)
}

/// Delete the pure, unread nodes - the whole cascade, in one sweep.
///
/// The cascade is the point. Deleting a node can make its operands unread, and
/// their operands unread, so a sweep that visits each node once in block order
/// advances a deletion chain by one level per sweep when the chain runs against
/// the block order. That is not academic: materializing a return tail leaves the
/// tail's own nodes unread, and the value chain feeding the tail then dies from
/// the far end backwards - one node per sweep on a chain-shaped function, which
/// the fixpoint around this sweep turned into `blocks` sweeps of O(nodes) work
/// (measured on `pipeline_scaling`'s ladder, debug, before the worklist: DCE 338 ms
/// at 500 blocks, 1.40 s at 1k, 7.22 s at 2k, `clean` 31.5 s at 4k, x3.9 per x2 =
/// quadratic; the same sweep after: linear). The worklist is the doctrine at the
/// top of this file applied literally: build the tables once per sweep, answer
/// every question from them, and let the cascade run to its own fixpoint inside
/// the sweep.
///
/// A pure, unread node in a block that ends in `Return` is the IR form of a Win64
/// frame-teardown write on a return-only path: `mov rsp,rbp` lowers to
/// `Copy(rbp)`, `pop rbp` to a `Load`, an `Add` and a write of the loaded value,
/// and the `Return` names the ANSWER, not the frame. Those deletions are counted
/// so the census prints the number the lane's premise expected to be the win
/// (measured 0 surviving inside the 29 refused target blocks on the corpus at base
/// 47b9076 - the count this prints is the teardown DCE removes everywhere, which
/// the epilogue pass makes visible in the tails it moves).
fn dce_sweep(f: &mut Function) -> bool {
    let mut counts = use_counts(f);
    // Per-sweep table: which instruction defines each value (`def_of`), built once
    // so the cascade can jump from a dying value to the node that produced it.
    let mut def_of: Vec<(usize, usize)> = vec![NO_DEF; f.values.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, ins) in b.instrs.iter().enumerate() {
            let d = ins.dest as usize;
            if d < def_of.len() && def_of[d] == NO_DEF {
                def_of[d] = (bi, ii);
            }
        }
    }
    // The cascade: every deletable node seeds the worklist, and each deletion
    // decrements its operands, enqueuing the operand whose last reader just died.
    // `queued` keeps a def in the list once, so a node is processed at most once
    // and the sweep costs O(nodes + uses).
    let mut dead: Vec<(usize, usize)> = Vec::new();
    let mut queued = vec![false; f.values.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, ins) in b.instrs.iter().enumerate() {
            if dce_deletable(ins, &counts)
                && !queued.get(ins.dest as usize).copied().unwrap_or(true)
            {
                if let Some(q) = queued.get_mut(ins.dest as usize) {
                    *q = true;
                }
                dead.push((bi, ii));
            }
        }
    }
    let mut head = 0;
    while head < dead.len() {
        let (bi, ii) = dead[head];
        head += 1;
        let mut ops = Vec::new();
        f.blocks[bi].instrs[ii].op.uses(&mut ops);
        for u in ops.iter().copied() {
            let Some(n) = counts.get_mut(u as usize) else {
                continue;
            };
            if *n == 0 {
                continue;
            }
            *n -= 1;
            if *n != 0 {
                continue;
            }
            let (db, di) = match def_of.get(u as usize) {
                Some(&d) if d != NO_DEF => d,
                _ => continue,
            };
            if queued[u as usize] {
                continue;
            }
            if dce_deletable(&f.blocks[db].instrs[di], &counts) {
                queued[u as usize] = true;
                dead.push((db, di));
            }
        }
    }
    if dead.is_empty() {
        return false;
    }
    let mut kill: Vec<HashSet<usize>> = vec![HashSet::new(); f.blocks.len()];
    for (bi, ii) in &dead {
        kill[*bi].insert(*ii);
    }
    let mut teardown = 0usize;
    for (bi, b) in f.blocks.iter_mut().enumerate() {
        let k = &kill[bi];
        if k.is_empty() {
            continue;
        }
        if matches!(b.term, Terminator::Return(_)) {
            teardown += k.len();
        }
        let mut ii = 0usize;
        b.instrs.retain(|_| {
            let d = k.contains(&ii);
            ii += 1;
            !d
        });
    }
    if teardown > 0 {
        bump_epilogue(|c| c.teardown_writes_deleted += teardown);
    }
    true
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
