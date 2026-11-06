//! Structural IR and SSA verifier. Runs after *every* pass (T4 in ROADMAP.md).
//!
//! The middle-end is a long chain of rewrites over a graph with use-def edges;
//! a single dangling `VId` becomes a wrong C expression three passes later, far
//! from the bug. Everything is reported as a list of human-readable errors, so
//! a failure prints the exact offending value/block and a pass can be bisected.
//!
//! THE COST MODEL. Every check here is O(1) per node, backed by one sweep that
//! builds the tables the checks ask about: the def site of every value, the use
//! sites of every value, the first index at which every block id appears, and
//! the dominator-tree intervals that answer "does this def dominate that use" in
//! constant time. The first version asked those questions by scanning
//! `f.blocks` per query — the shape the passes had before their own per-sweep
//! tables — and it cost O(nodes) *per query*: measured on the 20k-block ladder
//! (60,007 nodes) in release, `verify_ssa` was 274.7 s while the whole pass
//! pipeline was 86 ms.
//!
//! THE REFUSALS ARE THE CONTRACT, NOT THE SPEED. Every `e.hard(...)`/`e.soft(...)`
//! below has a fixture in `crates/core/tests/verify_ssa_rejections.rs` that turns
//! red if the check is deleted, so a faster verifier that misses a rejection
//! cannot land. The checks no fixture can reach are named there.

use crate::cfg::Cfg;
use crate::ir::*;
use crate::ty::Ty;
use std::collections::HashMap;

pub type Verdict = Result<(), Vec<String>>;

/// One sweep over the function, and the tables every query below reads.
///
/// Built in a single pass over blocks, instructions and terminators, so the
/// whole verifier is O(nodes + edges). The tables are keyed by value id and
/// block id — exactly the space `Function::def` / `Function::uses_of` scan — and
/// they reproduce those scans:
///
///  * `def` holds the FIRST definition site in block order / instruction order,
///    which is what `Function::def` returns when a value is defined twice (the
///    duplicate is a separate hard failure, reported by the census below).
///  * `uses` holds the sites in the order `Function::uses_of` emits them
///    (instructions in order, operand slots in `Op::uses` order, then the
///    terminator), over the same value range: an id at or past `next_value` has
///    no entry, exactly as the scan reports nothing for it.
///  * `def_count` is the census `verify` used to build inline.
///  * `first_pos` maps a block id to the first index at which it appears — the
///    `position(|x| x.id == b.id)` the id/index rule asked for. A block list
///    where two blocks claim one id keeps the old behaviour exactly: the FIRST
///    occurrence is the one that satisfies `id == index`.
#[derive(Default)]
struct Sweep {
    def: Vec<Option<(BbId, usize)>>,
    def_count: Vec<u32>,
    uses: Vec<Vec<UseSite>>,
    first_pos: HashMap<BbId, usize>,
}

impl Sweep {
    fn build(f: &Function) -> Sweep {
        let nv = f.next_value as usize;
        let mut t = Sweep {
            def: vec![None; nv],
            def_count: vec![0u32; nv],
            uses: vec![Vec::new(); nv],
            first_pos: HashMap::new(),
        };
        let mut ops: Vec<VId> = Vec::new();
        for (idx, b) in f.blocks.iter().enumerate() {
            t.first_pos.entry(b.id).or_insert(idx);
            for (ii, ins) in b.instrs.iter().enumerate() {
                if ins.dest != NO_VAL && (ins.dest as usize) < nv {
                    let d = ins.dest as usize;
                    if t.def[d].is_none() {
                        t.def[d] = Some((b.id, ii));
                    }
                    t.def_count[d] += 1;
                }
                ins.op.uses(&mut ops);
                for (slot, u) in ops.iter().enumerate() {
                    if (*u as usize) < nv {
                        t.uses[*u as usize].push(UseSite {
                            bb: b.id,
                            instr: ii,
                            slot,
                        });
                    }
                }
            }
            // Terminator operands are uses, exactly as `Function::uses_of` says
            // (including the `usize::MAX` instruction marker and the `NO_VAL`
            // guard: a `Return(NO_VAL)` names no value and is no use).
            let top: Option<VId> = match b.term {
                Terminator::Branch { cond, .. } => Some(cond),
                Terminator::Return(x) | Terminator::TailReturn(x) if x != NO_VAL => Some(x),
                Terminator::Switch { index, .. } => Some(index),
                _ => None,
            };
            if let Some(v) = top {
                if (v as usize) < nv {
                    t.uses[v as usize].push(UseSite {
                        bb: b.id,
                        instr: usize::MAX,
                        slot: 0,
                    });
                }
            }
        }
        t
    }

    /// The def site `Function::def` would return.
    fn def_of(&self, v: VId) -> Option<(BbId, usize)> {
        self.def.get(v as usize).copied().flatten()
    }

    fn defined(&self, v: VId) -> bool {
        self.def_of(v).is_some()
    }

    /// The use sites `Function::uses_of` would return, as a slice.
    fn uses_of(&self, v: VId) -> &[UseSite] {
        self.uses
            .get(v as usize)
            .map(|u| u.as_slice())
            .unwrap_or(&[])
    }

    fn is_used(&self, v: VId) -> bool {
        !self.uses_of(v).is_empty()
    }
}

/// Structural checks that must hold in every IR state (SSA form not required).
pub fn verify(f: &Function) -> Verdict {
    verify_swept(f, &Sweep::build(f))
}

fn verify_swept(f: &Function, t: &Sweep) -> Verdict {
    let mut e = Errs::default();

    if f.blocks.is_empty() {
        e.hard("function has no blocks");
        return e.finish();
    }
    if f.entry as usize >= f.blocks.len() {
        e.hard("entry block id out of range");
    }
    for b in &f.blocks {
        if b.id as usize != t.first_pos.get(&b.id).copied().unwrap_or(usize::MAX) {
            e.soft(format!("block id {} is not at index {}", b.id, b.id));
        }
    }
    if f.blocks.iter().any(|b| b.id == NO_BB) {
        e.hard("a block has the NO_BB id");
    }

    // Value space: every def is unique and every operand is defined.
    for b in &f.blocks {
        for i in &b.instrs {
            if i.dest == NO_VAL {
                e.hard(format!("bb{}: instruction with no destination", b.id));
                continue;
            }
            if i.dest as usize >= f.next_value as usize {
                e.hard(format!(
                    "bb{}: dest {} >= next_value {}",
                    b.id, i.dest, f.next_value
                ));
                continue;
            }
            // Op-declared type must agree with the recorded type.
            if let Some(ty) = i.op.ty(f) {
                if ty != i.ty {
                    e.soft(format!(
                        "bb{}: %{} op type {ty} != recorded {}",
                        b.id, i.dest, i.ty
                    ));
                }
            }
            if matches!(i.ty, Ty::Flags) && !matches!(i.op, Op::Flags(_)) {
                e.hard(format!(
                    "bb{}: %{} has Flags type from a non-Flags op",
                    b.id, i.dest
                ));
            }
            // Operand width agreement (§1.3). `Bin`/`Cmp` over mixed widths is
            // the signature of a missing extend, and the C emitter will happily
            // print `(int)a + b` — legal C, wrong arithmetic, so: hard.
            let pair = match &i.op {
                Op::Bin(_, x, y) | Op::Cmp(_, x, y) => Some((*x, *y)),
                _ => None,
            };
            if let Some((x, y)) = pair {
                if let (Some(wx), Some(wy)) = (
                    f.ty_of(x).and_then(|t| t.width()),
                    f.ty_of(y).and_then(|t| t.width()),
                ) {
                    if wx != wy {
                        e.hard(format!(
                            "bb{}: %{} mixes widths {wx:?} and {wy:?}",
                            b.id, i.dest
                        ));
                    }
                }
            }
            // A `Select` condition is Bool and its arms share one type. Hard for
            // the same reason as the branch check: C's ternary accepts an integer
            // condition, so a soft warning ships a pass bug as legal C.
            if let Op::Select(c, x, y) = &i.op {
                if !f.ty_of(*c).map(|t| t.is_bool()).unwrap_or(false) {
                    e.hard(format!(
                        "bb{}: %{} select condition %{} is not bool",
                        b.id, i.dest, c
                    ));
                }
                if let (Some(tx), Some(ty2)) = (f.ty_of(*x), f.ty_of(*y)) {
                    if tx != ty2 {
                        e.hard(format!(
                            "bb{}: %{} select arms disagree: {tx} vs {ty2}",
                            b.id, i.dest
                        ));
                    }
                }
            }
            // Float arithmetic and float comparison are typed ops, not width
            // ops. The width pair-check above compares `Ty::Float`'s width
            // (F32 -> W32, F64 -> W64), which catches mixing precisions; this
            // catches the other bug, an integer wired into a float op, which is
            // invisible to a width rule because `int` and `float` share widths.
            // C prints either -- `(double)a + b` compiles -- so only a check
            // here keeps a pass bug from becoming silently-wrong C.
            if let Op::FBin(_, x, y) | Op::FCmp(_, x, y) = &i.op {
                for v in [*x, *y] {
                    match f.ty_of(v) {
                        Some(ty) if ty.is_float() => {}
                        other => e.hard(format!(
                            "bb{}: %{} float op reads %{} with type {} (needs a float)",
                            b.id,
                            i.dest,
                            v,
                            other
                                .map(|t| t.to_string())
                                .unwrap_or_else(|| "<no type>".to_string())
                        )),
                    }
                }
            }
            // A call is the one op whose meaning lives OUTSIDE this function, so
            // it is the one op the verifier has to check against its own claim.
            // Three rules, all hard, because all three produce C that compiles:
            //
            //  * the target's identity must be there. `Symbol("")` names nobody,
            //    and the emitter would print a call to nothing.
            //  * an argument is a register cell (int, float, pointer). A `Bool`
            //    or a `Flags` value wired into an argument list has no C
            //    spelling as an argument, and `Ty::Flags` reaching a call means
            //    flag elimination did not run.
            //  * the declared return type must be one C can spell, because the
            //    prototype the emitter writes for the callee is spelled from it.
            if let Op::Call {
                callee,
                args,
                ret_ty,
                ..
            } = &i.op
            {
                match callee {
                    Callee::Symbol(s) if s.trim().is_empty() => e.hard(format!(
                        "bb{}: %{} call with an empty callee name",
                        b.id, i.dest
                    )),
                    Callee::Indirect(v) if !t.defined(*v) => e.hard(format!(
                        "bb{}: %{} indirect call through undefined %{}",
                        b.id, i.dest, v
                    )),
                    _ => {}
                }
                for a in args.iter() {
                    match f.ty_of(*a) {
                        Some(ref ty)
                            if !matches!(ty, Ty::Int { .. } | Ty::Float { .. } | Ty::Ptr(_)) =>
                        {
                            e.hard(format!(
                                "bb{}: %{} call argument %{} is {ty} (a call's arguments are register cells: int, float or pointer)",
                                b.id, i.dest, a
                            ));
                        }
                        _ => {}
                    }
                }
                if matches!(ret_ty, Ty::Flags | Ty::Func(..) | Ty::Struct { .. }) {
                    e.hard(format!(
                        "bb{}: %{} call declares an unspellable return type {ret_ty}",
                        b.id, i.dest
                    ));
                }
            }
            // A clobber is a definition with no operands: the one thing to check
            // is that it stayed a register cell, because the register file is one
            // unsigned W64 word per register and the SSA transfer function keys on
            // that, not on a second table.
            if matches!(i.op, Op::Clobber) && !matches!(i.ty, Ty::Int { .. }) {
                e.hard(format!(
                    "bb{}: %{} clobber is {} (a clobbered register is a W64 cell)",
                    b.id, i.dest, i.ty
                ));
            }
            // A float conversion whose SOURCE is not a float (or whose result
            // is not the float kind the kind names) is a pass bug: the C cast
            // syntax accepts all of them, and the answers differ.
            if let Op::Cast(k, ct, a) = &i.op {
                let src = f.ty_of(*a);
                // The string, not the Option<Ty>: Ty is Clone but not Copy, and
                // the closure below is called from several arms.
                let src_name = src
                    .as_ref()
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "<no type>".to_string());
                let mut named = |k: &'static str| {
                    e.soft(format!(
                        "bb{}: %{} {k} from %{} with type {}",
                        b.id, i.dest, a, src_name
                    ))
                };
                match k {
                    CastKind::FpToSI | CastKind::FpToUI => {
                        if !src.map(|t| t.is_float()).unwrap_or(false) {
                            named("FpToSI/FpToUI");
                        }
                    }
                    CastKind::SIToFp | CastKind::UIToFp => {
                        if !src.map(|t| t.is_int()).unwrap_or(false) || !ct.is_float() {
                            named("SIToFp/UIToFp");
                        }
                    }
                    CastKind::FPExt | CastKind::FPTrunc
                        if !(src.as_ref().map(|t| t.is_float()).unwrap_or(false)
                            && ct.is_float()) =>
                    {
                        named("FPExt/FPTrunc");
                    }
                    _ => {}
                }
            }
        }
        // Terminator targets in range.
        let mut succ = Vec::new();
        b.term.successors(&mut succ);
        for s in succ {
            if s as usize >= f.blocks.len() {
                e.hard(format!("bb{}: terminator to missing bb{}", b.id, s));
            }
        }
        match &b.term {
            Terminator::Return(v) => {
                if *v != NO_VAL && !t.defined(*v) {
                    e.hard(format!("bb{}: ret of undefined %{}", b.id, v));
                }
                // The declared return type is a WIDTH CLAIM, and it is the one
                // type claim in the IR that changes the bits the caller sees:
                // `uint32_t` zero-extends on the way out, `int64_t` sign-extends,
                // and the machine's `RAX` has exactly the bits the width says
                // (a 32-bit write cleared the upper half). A value that needs
                // more bits than the return type has is emitted C that drops
                // them, and a pass that widens a returned value - folding a
                // constant, replacing a `zext` chain, fusing a `select` - breaks
                // the claim the lifter made from the machine code. Hard, because
                // the C still compiles: the same reason the branch and Select
                // checks are hard.
                if *v != NO_VAL {
                    if let Some(w) = f.ret_ty.width() {
                        let bits = w.bits();
                        if bits < 64 {
                            let need = crate::width::bits_needed(f, *v);
                            if need > bits {
                                e.hard(format!(
                                    "bb{}: ret %{} needs {need} bits, more than the declared {} return type holds",
                                    b.id, v, f.ret_ty
                                ));
                            }
                        }
                    }
                }
            }
            // A TAIL CALL IS A CLAIM ABOUT THE MACHINE CODE: the block left the
            // function through a `jmp` whose relocation named a function symbol.
            // So the returned value must be defined by a CALL in this block with a
            // NAMED callee. A `jmp` computes nothing, so a `TailReturn` of any other
            // value means the IR invented a tail call; an unnamed callee (`Addr`,
            // `Indirect`, an empty name) means the function's whole answer belongs
            // to nobody, which is the hole the C would paper over with a prototype.
            Terminator::TailReturn(v) => {
                if *v == NO_VAL || !t.defined(*v) {
                    e.hard(format!("bb{}: tail return of undefined %{}", b.id, v));
                    continue;
                }
                let Some((dbb, di)) = t.def_of(*v) else {
                    e.hard(format!(
                        "bb{}: tail return of %{} has no definition",
                        b.id, v
                    ));
                    continue;
                };
                match &f.block(dbb).instrs[di].op {
                    Op::Call { callee, .. } => match callee {
                        Callee::Symbol(n) if !n.is_empty() => {}
                        _ => e.hard(format!(
                            "bb{}: tail call's callee has no name ({}), so whose answer is bb{}'s return?",
                            b.id,
                            match callee {
                                Callee::Addr(a) => format!("an address, {a:#x}"),
                                Callee::Indirect(_) => "a register".to_string(),
                                _ => "an empty symbol name".to_string(),
                            },
                            b.id
                        )),
                    },
                    other => e.hard(format!(
                        "bb{}: tail return of %{} is defined by {other:?}, not a call: a jmp computes no value",
                        b.id, v
                    )),
                }
                if dbb != b.id {
                    e.hard(format!(
                        "bb{}: tail return of %{} defined in bb{dbb}: the call and the jmp that leaves with it are one instruction",
                        b.id, v
                    ));
                }
                // The same width claim `Return` makes: the callee answers in RAX
                // and the caller's declared return type is the width of that cell.
                if let Some(w) = f.ret_ty.width() {
                    let bits = w.bits();
                    if bits < 64 {
                        let need = crate::width::bits_needed(f, *v);
                        if need > bits {
                            e.hard(format!(
                                "bb{}: tail return %{} needs {need} bits, more than the declared {} return type holds",
                                b.id, v, f.ret_ty
                            ));
                        }
                    }
                }
            }
            // A noreturn exit is a claim that the block's last act was a call to a
            // callee that never returns. If the block's last instruction is not a
            // call, the terminator invented the shape: refuse to certify it.
            Terminator::NoReturn => match b.instrs.last() {
                Some(i) if matches!(i.op, Op::Call { .. }) => {}
                Some(i) => e.hard(format!(
                    "bb{}: noreturn exit after {:?}, not after a call",
                    b.id, i.op
                )),
                None => e.hard(format!(
                    "bb{}: noreturn exit with no instruction at all",
                    b.id
                )),
            },
            Terminator::Switch { index, targets } => {
                // The index is read by the dispatch, so it must be a value this
                // function defines (or a parameter): an undefined selector means
                // the lifter pointed the switch at nothing.
                if !t.defined(*index) {
                    e.hard(format!("bb{}: switch on undefined index %{}", b.id, index));
                }
                // A switch over no target at all is the shape a half-resolved
                // table leaves behind. It has no successor, so it is not control
                // flow, and a function whose dispatch reaches it has lost the
                // case bodies: refuse to certify it.
                if targets.is_empty() {
                    e.hard(format!(
                        "bb{}: switch with an empty target list (a partially resolved table)",
                        b.id
                    ));
                }
                // EVERY DECLARED TARGET MUST BE A BLOCK OF THIS FUNCTION. The
                // successors above came from the same list, so this is the check
                // that makes "every target is a real CFG edge" a property of the
                // IR: a target the lifter invented, outside the block set, is a
                // hard failure, not a note.
                for (k, &t) in targets.iter().enumerate() {
                    if t as usize >= f.blocks.len() {
                        e.hard(format!(
                            "bb{}: switch slot {k} names bb{t}, not a block of this function",
                            b.id
                        ));
                    }
                }
                // Duplicate slots are legal (two table entries may name one body,
                // which is what a C `switch` with shared case labels compiles to),
                // so they are not checked. A slot that names the dispatch block
                // itself is not: it is a self-loop with no condition, and no
                // machine code table a `switch` came from says that.
                for &t in targets.iter() {
                    if t == b.id {
                        e.hard(format!("bb{}: switch slot names its own block", b.id));
                    }
                }
            }
            Terminator::Branch { cond, .. } => {
                if !t.defined(*cond) {
                    e.hard(format!("bb{}: branch on undefined %{}", b.id, cond));
                }
                // Hard, not a warning: `br` requires a bool. If the condition is
                // an integer, emitted C (`if (x)` on an int32) still *compiles*,
                // so the type model has lied and a pass bug hides behind legal C.
                if !f.ty_of(*cond).map(|t| t.is_bool()).unwrap_or(false) {
                    e.hard(format!(
                        "bb{}: branch condition %{} is not bool",
                        b.id, cond
                    ));
                }
            }
            _ => {}
        }
    }
    for (v, c) in t.def_count.iter().enumerate() {
        if *c == 0 && v < f.next_value as usize && t.is_used(v as VId) {
            e.hard(format!("value %{v} is used but never defined"));
        }
        if *c > 1 {
            e.hard(format!(
                "value %{v} is defined {c} times (not single-assignment)"
            ));
        }
    }
    for v in 0..f.next_value {
        for u in t.uses_of(v) {
            if u.instr != usize::MAX && u.instr >= f.blocks[u.bb as usize].instrs.len() {
                e.hard(format!("bad use-site for %{v} in bb{}", u.bb));
            }
        }
    }
    e.finish()
}

/// SSA-specific checks.
/// The predecessor a use site belongs to, when the use is a phi operand: the
/// use's operand slot indexes the phi's (value, predecessor) pairs, so the edge
/// is recoverable and the check can be made about availability, not position.
fn phi_pred(f: &Function, u: UseSite) -> Option<BbId> {
    if u.instr == usize::MAX {
        return None;
    }
    let ins = f.block(u.bb).instrs.get(u.instr)?;
    match &ins.op {
        Op::Phi(pairs) => pairs.get(u.slot).map(|(_, bb)| *bb),
        _ => None,
    }
}

pub fn verify_ssa(f: &Function) -> Verdict {
    // ONE sweep for the whole function, shared by the structural checks and the
    // SSA checks below. Every question the checks ask — `where is %v defined`,
    // `where is %v read`, `does bb{a} dominate bb{b}` — is a table read, so the
    // cost is O(nodes + edges) and not O(nodes) per query.
    let t = Sweep::build(f);
    let mut e = Errs::default();
    let cfg = Cfg::build(f);
    // B7: structural failures used to be routed into a local `Errs` and then
    // discarded, so callers saw `Err(vec![])` -- a rejection with no reason.
    if let Err(errs) = verify_swept(f, &t) {
        for x in errs {
            e.hard(x);
        }
    }
    let dom = DomIndex::build(&cfg);

    for b in &f.blocks {
        // Phis form a contiguous prefix.
        let mut seen_non_phi = false;
        for i in &b.instrs {
            if matches!(i.op, Op::Phi(_)) {
                if seen_non_phi {
                    e.hard(format!("bb{}: phi %{} after a non-phi", b.id, i.dest));
                }
            } else {
                seen_non_phi = true;
            }
        }
        // A constant must fit the width it was declared at. Constants carry no
        // width of their own (`Op::ty` deliberately returns None for them), so
        // the declared type is the only claim about size - and a `uint32_t`
        // constant holding 0x1_0000_0000 is a lie about the value.
        for i in &b.instrs {
            if let Op::Const(v) = i.op {
                match f.ty_of(i.dest).and_then(|t| t.width()) {
                    None => e.hard(format!("bb{}: const %{} has no integer type", b.id, i.dest)),
                    Some(w) => {
                        let bits = w.bits();
                        if bits < 64 && (v >> bits) != 0 {
                            e.hard(format!(
                                "bb{}: const %{} = {v:#x} does not fit {bits}-bit type",
                                b.id, i.dest
                            ));
                        }
                    }
                }
            }
        }

        // phi incoming blocks == predecessors, as a multiset. The sorted
        // predecessor list is the same for every phi in the block, so it is
        // built once per block, not cloned and sorted once per phi.
        let mut want: Option<Vec<BbId>> = None;
        for i in &b.instrs {
            if let Op::Phi(pairs) = &i.op {
                let mut got: Vec<BbId> = pairs.iter().map(|(_, bb)| *bb).collect();
                got.sort();
                let want = want.get_or_insert_with(|| {
                    let mut w = cfg.preds[b.id as usize].clone();
                    w.sort();
                    w
                });
                if got != *want {
                    e.hard(format!(
                        "bb{}: phi %{} incoming {got:?} != predecessors {want:?}",
                        b.id, i.dest
                    ));
                }
                for (v, from) in pairs.iter() {
                    if !t.defined(*v) {
                        e.hard(format!("bb{}: phi %{} takes undefined %{v}", b.id, i.dest));
                    } else if !cfg.is_reachable(*from) {
                        e.hard(format!(
                            "bb{}: phi %{} from unreachable bb{from}",
                            b.id, i.dest
                        ));
                    }
                }
            }
        }
    }

    // Def must dominate every use; in-block uses must be program-order later.
    for v in 0..f.next_value {
        let Some((dbb, dii)) = t.def_of(v) else {
            continue;
        };
        if !cfg.is_reachable(dbb) {
            e.hard(format!("%{v} is defined in unreachable bb{dbb}"));
        }
        for u in t.uses_of(v) {
            // A phi operand is a claim about its EDGE, not about the phi's
            // position: the value has to be live at the END of its predecessor.
            // Ordinary dominance (def dominates the phi's block, and is earlier
            // in the same block) is the wrong rule here, and rejects the
            // standard loop-carried form `%v = phi [%init, %head], [%next, %loop]`
            // where `%next` is defined in the latch - the case a self edge makes
            // into "defined later in this very block".
            if let Some(pred) = phi_pred(f, *u) {
                if dbb != pred && !dom.dominates(&cfg, dbb, pred) {
                    e.hard(format!(
                        "%{v} defined in bb{dbb} is not available on the edge bb{pred} -> bb{}",
                        u.bb
                    ));
                }
                if !cfg.is_reachable(pred) {
                    e.hard(format!("%{v} is a phi operand from unreachable bb{pred}"));
                }
                continue;
            }
            if u.bb == dbb {
                if u.instr != usize::MAX && u.instr <= dii {
                    e.hard(format!(
                        "%{v} used at bb{}#{}, defined at #{dii} (use before def)",
                        u.bb, u.instr
                    ));
                }
            } else if !dom.dominates(&cfg, dbb, u.bb) {
                e.hard(format!(
                    "%{v} defined in bb{dbb} does not dominate use in bb{}",
                    u.bb
                ));
            }
        }
    }
    e.finish()
}

/// Constant-time dominance queries over a `Cfg`, by DFS number.
///
/// `Cfg::dominates` walks `b`'s idom chain upward: O(depth) per query. On the
/// 20k-block ladder the dominator tree is a 20k-deep chain and the
/// def-dominates-use loop asks it 60,007 questions. A block's dominator-tree
/// subtree is exactly the interval `[pre, post]` a DFS gives it, so one
/// traversal answers every query in O(1).
///
/// The answer matches `Cfg::dominates` on every input, including the two
/// degenerate classes that one treats specially: a block whose `idom` is `NO_BB`
/// dominates nothing (the entry self-dominates; unreachable blocks have no idom),
/// and a block with no DFS number is dominated by nobody. The
/// `dominance_index_agrees_with_the_walk` test checks EVERY block pair of ten
/// shapes against the walk.
struct DomIndex {
    pre: Vec<u32>,
    post: Vec<u32>,
}

impl DomIndex {
    fn build(cfg: &Cfg) -> DomIndex {
        let n = cfg.reachable.len();
        // `u32::MAX` = "no interval", the same role `NO_BB` plays for `idom`.
        let mut pre = vec![u32::MAX; n];
        let mut post = vec![u32::MAX; n];
        let mut clock = 0u32;
        // Iterative: the dominator tree of a long chain is 20k deep and the
        // native stack is not that patient.
        for root in 0..n {
            // Dom-tree roots: a block that dominates itself (the entry) and a
            // reachable block the idom fixpoint never reached (defensive; the
            // walk answers "no" for every pair that goes through it, and so
            // does an interval-less node).
            let is_root = pre[root] == u32::MAX
                && (cfg.idom[root] as usize == root
                    || (cfg.reachable[root] && cfg.idom[root] == NO_BB));
            if !is_root {
                continue;
            }
            pre[root] = clock;
            clock += 1;
            let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
            // The DFS walks with the live `last_mut` borrow: the increment goes
            // through `top` itself. The copied-index + re-borrowed-`last_mut`
            // shape (the cfg.rs pattern) is provably guarded, but an
            // explicit-panic call there is a PANIC SITE, and the ratchet counts
            // it — a new guarded site in a rewrite is fixed by not needing the
            // guard, not by raising the baseline.
            while let Some(top) = stack.last_mut() {
                let (node, idx) = *top;
                if idx < cfg.dom_tree[node].len() {
                    top.1 += 1;
                    let c = cfg.dom_tree[node][idx] as usize;
                    if pre[c] == u32::MAX {
                        pre[c] = clock;
                        clock += 1;
                        stack.push((c, 0));
                    }
                } else {
                    stack.pop();
                    post[node] = clock;
                    clock += 1;
                }
            }
        }
        DomIndex { pre, post }
    }

    fn dominates(&self, cfg: &Cfg, a: BbId, b: BbId) -> bool {
        let (ai, bi) = (a as usize, b as usize);
        // The walk's own first guard: a block with no idom dominates nothing.
        match cfg.idom.get(ai) {
            Some(&NO_BB) | None => return false,
            Some(_) => {}
        }
        let (pa, pb, qa, qb) = match (
            self.pre.get(ai),
            self.pre.get(bi),
            self.post.get(ai),
            self.post.get(bi),
        ) {
            (Some(&pa), Some(&pb), Some(&qa), Some(&qb)) => (pa, pb, qa, qb),
            _ => return false,
        };
        // No number = outside the dominator tree = dominated by nobody, which is
        // the walk's answer for an unreachable `b`: its idom chain is `NO_BB`
        // and never reaches `a`.
        if pa == u32::MAX || pb == u32::MAX {
            return false;
        }
        pa <= pb && qb <= qa
    }
}

#[derive(Default)]
struct Errs {
    errs: Vec<String>,
}

impl Errs {
    fn hard(&mut self, msg: impl Into<String>) {
        self.errs.push(msg.into());
    }
    /// Warnings that are legal mid-pipeline (e.g. a pass that temporarily
    /// leaves a phi type stale) but that the driver escalates at phase
    /// boundaries.
    fn soft(&mut self, msg: impl Into<String>) {
        self.errs.push(format!("warn: {}", msg.into()));
    }
    fn finish(self) -> Verdict {
        if self.errs.is_empty() {
            Ok(())
        } else {
            Err(self.errs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IntCmp;

    /// A diamond: entry -> {a, b} -> join -> exit.
    fn diamond() -> Function {
        let mut f = Function::new("diamond", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(p);
        let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
        let a = f.new_block(0x10);
        let b = f.new_block(0x20);
        let j = f.new_block(0x30);
        let x = f.new_block(0x40);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: a,
            else_bb: b,
        };
        f.block_mut(a).term = Terminator::Jump(j);
        f.block_mut(b).term = Terminator::Jump(j);
        f.block_mut(j).term = Terminator::Jump(x);
        f.block_mut(x).term = Terminator::Return(p);
        f
    }

    /// A loop: entry -> head, head -> {tail, exit}, tail -> head.
    fn looped() -> Function {
        let mut f = Function::new("looped", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(p);
        let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
        let head = f.new_block(0x10);
        let tail = f.new_block(0x20);
        let exit = f.new_block(0x30);
        f.block_mut(e).term = Terminator::Jump(head);
        f.block_mut(head).term = Terminator::Branch {
            cond: c,
            then_bb: tail,
            else_bb: exit,
        };
        f.block_mut(tail).term = Terminator::Jump(head);
        f.block_mut(exit).term = Terminator::Return(p);
        f
    }

    /// An irreducible pair: two blocks that jump to each other, both entered
    /// from the entry. The dominator tree here is as flat as it gets.
    fn irreducible() -> Function {
        let mut f = Function::new("irreducible", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(p);
        let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
        let a = f.new_block(0x10);
        let b = f.new_block(0x20);
        let z = f.new_block(0x30);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: a,
            else_bb: b,
        };
        f.block_mut(a).term = Terminator::Branch {
            cond: c,
            then_bb: b,
            else_bb: z,
        };
        f.block_mut(b).term = Terminator::Jump(a);
        f.block_mut(z).term = Terminator::Return(p);
        f
    }

    /// A chain: the shape whose dominator tree is n deep.
    fn chain(n: usize) -> Function {
        let mut f = Function::new("chain", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::i64(), 0);
        f.params.push(p);
        let mut cur = e;
        for i in 1..n {
            let nx = f.new_block(0x10 * i as u64);
            f.block_mut(cur).term = Terminator::Jump(nx);
            cur = nx;
        }
        f.block_mut(cur).term = Terminator::Return(p);
        f
    }

    /// Blocks nothing reaches: the dominator tree does not contain them.
    fn with_unreachable() -> Function {
        let mut f = diamond();
        let o1 = f.new_block(0x900);
        let o2 = f.new_block(0x910);
        f.block_mut(o1).term = Terminator::Jump(o2);
        f.block_mut(o2).term = Terminator::Jump(o1);
        f
    }

    /// A switch fan-out with a duplicated table slot.
    fn switchy() -> Function {
        let mut f = Function::new("switchy", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::u64(), 0);
        f.params.push(p);
        let a = f.new_block(0x10);
        let b = f.new_block(0x20);
        let c = f.new_block(0x30);
        f.block_mut(e).term = Terminator::Switch {
            index: p,
            targets: vec![a, b, c, a],
        };
        f.block_mut(a).term = Terminator::Jump(c);
        f.block_mut(b).term = Terminator::Jump(c);
        f.block_mut(c).term = Terminator::Return(p);
        f
    }

    /// A loop with two entry edges (a rotated loop): the header's idom is the
    /// entry and its latch is inside the loop.
    fn two_entry_loop() -> Function {
        let mut f = Function::new("two_entry_loop", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::u64(), 0);
        f.params.push(p);
        let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
        let head = f.new_block(0x10);
        let mid = f.new_block(0x20);
        let latch = f.new_block(0x30);
        let exit = f.new_block(0x40);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: head,
            else_bb: mid,
        };
        f.block_mut(head).term = Terminator::Jump(mid);
        f.block_mut(mid).term = Terminator::Branch {
            cond: c,
            then_bb: latch,
            else_bb: exit,
        };
        f.block_mut(latch).term = Terminator::Jump(head);
        f.block_mut(exit).term = Terminator::Return(p);
        f
    }

    /// A self-loop block: the degenerate back edge.
    fn self_loop() -> Function {
        let mut f = Function::new("self_loop", 0);
        let e = f.entry;
        let p = f.emit(e, Op::Param, Ty::u64(), 0);
        f.params.push(p);
        let c = f.emit(e, Op::Cmp(IntCmp::Eq, p, p), Ty::Bool, 0);
        let s = f.new_block(0x10);
        let x = f.new_block(0x20);
        f.block_mut(e).term = Terminator::Branch {
            cond: c,
            then_bb: s,
            else_bb: x,
        };
        f.block_mut(s).term = Terminator::Branch {
            cond: c,
            then_bb: s,
            else_bb: x,
        };
        f.block_mut(x).term = Terminator::Return(p);
        f
    }

    /// THE EQUIVALENCE THE REWRITE DEPENDS ON: the interval index answers every
    /// (a, b) pair the way the O(depth) idom walk does, for every shape. Break
    /// either one and this goes red.
    #[test]
    fn dominance_index_agrees_with_the_walk() {
        let shapes = [
            diamond(),
            looped(),
            irreducible(),
            chain(1),
            chain(2),
            chain(40),
            with_unreachable(),
            switchy(),
            two_entry_loop(),
            self_loop(),
        ];
        for f in shapes {
            let cfg = Cfg::build(&f);
            let idx = DomIndex::build(&cfg);
            let n = f.blocks.len();
            for a in 0..n {
                for b in 0..n {
                    let (a, b) = (a as BbId, b as BbId);
                    let walk = cfg.dominates(a, b);
                    let fast = idx.dominates(&cfg, a, b);
                    assert_eq!(
                        walk, fast,
                        "{}: dominates(bb{a}, bb{b}): the idom walk says {walk}, the interval index says {fast} (idom[bb{a}]={}, reachable[bb{a}]={})",
                        f.name,
                        cfg.idom[a as usize],
                        cfg.reachable[a as usize]
                    );
                }
            }
        }
    }

    /// The sweep is the other half of the rewrite: the tables must answer what
    /// the scans answer, on IR that is deliberately messy (a duplicate def,
    /// terminator uses, phi operands, a use in a block that is not the def's).
    #[test]
    fn the_sweep_reproduces_def_and_uses_of() {
        let mut f = diamond();
        let p = f.params[0];
        // A phi with two operand uses (slot 0 and slot 1, on different edges).
        let phi = f.new_value(Ty::i64());
        f.block_mut(2).instrs.insert(
            0,
            Instr::new(
                phi,
                Op::Phi(vec![(p, 0), (p, 1)].into_boxed_slice()),
                Ty::i64(),
                0x30,
            ),
        );
        // A value defined twice: the table keeps the FIRST site, as the scan does.
        let dup = f.new_value(Ty::i64());
        f.block_mut(1)
            .push(Instr::new(dup, Op::Const(1), Ty::i64(), 0x20));
        f.block_mut(1)
            .push(Instr::new(dup, Op::Const(2), Ty::i64(), 0x20));
        // A terminator use.
        f.block_mut(3).term = Terminator::Return(dup);
        let t = Sweep::build(&f);
        for v in 0..f.next_value {
            assert_eq!(
                t.def_of(v),
                f.def(v),
                "the def table disagrees with the scan for %{v}"
            );
            assert_eq!(
                t.uses_of(v),
                f.uses_of(v).as_slice(),
                "the use table disagrees with the scan for %{v}"
            );
        }
        assert_eq!(t.def_count[dup as usize], 2, "the census counts both defs");
        assert_eq!(t.uses_of(phi).len(), 0, "the phi's result is never read");
        // %p is read by the entry's comparison (slots 0 and 1), by the phi
        // (slots 0 and 1) and by the exit's terminator.
        assert_eq!(
            t.uses_of(p).len(),
            5,
            "the operand slots and the terminator use are all counted"
        );
        assert_eq!(
            t.uses_of(p)[4],
            UseSite {
                bb: 4,
                instr: usize::MAX,
                slot: 0
            },
            "the terminator use carries the usize::MAX marker, as the scan emits it"
        );
    }
}
