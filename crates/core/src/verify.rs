//! Structural IR and SSA verifier. Runs after *every* pass (T4 in ROADMAP.md).
//!
//! The middle-end is a long chain of rewrites over a graph with use-def edges;
//! a single dangling `VId` becomes a wrong C expression three passes later, far
//! from the bug. Everything is reported as a list of human-readable errors, so
//! a failure prints the exact offending value/block and a pass can be bisected.

use crate::cfg::Cfg;
use crate::ir::*;
use crate::ty::Ty;

pub type Verdict = Result<(), Vec<String>>;

/// Structural checks that must hold in every IR state (SSA form not required).
pub fn verify(f: &Function) -> Verdict {
    let mut e = Errs::default();

    if f.blocks.is_empty() {
        e.hard("function has no blocks");
        return e.finish();
    }
    if f.entry as usize >= f.blocks.len() {
        e.hard("entry block id out of range");
    }
    for b in &f.blocks {
        if b.id as usize
            != f.blocks
                .iter()
                .position(|x| x.id == b.id)
                .unwrap_or(usize::MAX)
        {
            e.soft(format!("block id {} is not at index {}", b.id, b.id));
        }
    }
    if f.blocks.iter().any(|b| b.id == NO_BB) {
        e.hard("a block has the NO_BB id");
    }

    // Value space: every def is unique and every operand is defined.
    let mut def_count = vec![0u32; f.next_value as usize];
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
            def_count[i.dest as usize] += 1;
            // Op-declared type must agree with the recorded type.
            if let Some(t) = i.op.ty(f) {
                if t != i.ty {
                    e.soft(format!(
                        "bb{}: %{} op type {t} != recorded {}",
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
                        Some(t) if t.is_float() => {}
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
                    Callee::Indirect(v) if !defined(f, *v) => e.hard(format!(
                        "bb{}: %{} indirect call through undefined %{}",
                        b.id, i.dest, v
                    )),
                    _ => {}
                }
                for a in args.iter() {
                    match f.ty_of(*a) {
                        Some(ref t)
                            if !matches!(t, Ty::Int { .. } | Ty::Float { .. } | Ty::Ptr(_)) =>
                        {
                            e.hard(format!(
                                "bb{}: %{} call argument %{} is {t} (a call's arguments are register cells: int, float or pointer)",
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
            if let Op::Cast(k, t, a) = &i.op {
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
                        if !src.map(|t| t.is_int()).unwrap_or(false) || !t.is_float() {
                            named("SIToFp/UIToFp");
                        }
                    }
                    CastKind::FPExt | CastKind::FPTrunc
                        if !(src.as_ref().map(|t| t.is_float()).unwrap_or(false)
                            && t.is_float()) =>
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
                if *v != NO_VAL && !defined(f, *v) {
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
            Terminator::Switch { index, targets } => {
                // The index is read by the dispatch, so it must be a value this
                // function defines (or a parameter): an undefined selector means
                // the lifter pointed the switch at nothing.
                if !defined(f, *index) {
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
                if !defined(f, *cond) {
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
    for (v, c) in def_count.iter().enumerate() {
        if *c == 0 && v < f.next_value as usize && is_used(f, v as VId) {
            e.hard(format!("value %{v} is used but never defined"));
        }
        if *c > 1 {
            e.hard(format!(
                "value %{v} is defined {c} times (not single-assignment)"
            ));
        }
    }
    for v in 0..f.next_value {
        for u in f.uses_of(v) {
            if u.instr != usize::MAX && u.instr >= f.blocks[u.bb as usize].instrs.len() {
                e.hard(format!("bad use-site for %{v} in bb{}", u.bb));
            }
            let _ = u;
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
    let mut e = Errs::default();
    let cfg = Cfg::build(f);
    // B7: structural failures used to be routed into a local `Errs` and then
    // discarded, so callers saw `Err(vec![])` -- a rejection with no reason.
    if let Err(errs) = verify(f) {
        for x in errs {
            e.hard(x);
        }
    }

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

        // phi incoming blocks == predecessors, as a multiset.
        for i in &b.instrs {
            if let Op::Phi(pairs) = &i.op {
                let mut got: Vec<BbId> = pairs.iter().map(|(_, bb)| *bb).collect();
                let mut want: Vec<BbId> = cfg.preds[b.id as usize].clone();
                got.sort();
                want.sort();
                if got != want {
                    e.hard(format!(
                        "bb{}: phi %{} incoming {got:?} != predecessors {want:?}",
                        b.id, i.dest
                    ));
                }
                for (v, from) in pairs.iter() {
                    if !defined(f, *v) {
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
        let Some((dbb, dii)) = f.def(v) else {
            continue;
        };
        if !cfg.is_reachable(dbb) {
            e.hard(format!("%{v} is defined in unreachable bb{dbb}"));
        }
        for u in f.uses_of(v) {
            // A phi operand is a claim about its EDGE, not about the phi's
            // position: the value has to be live at the END of its predecessor.
            // Ordinary dominance (def dominates the phi's block, and is earlier
            // in the same block) is the wrong rule here, and rejects the
            // standard loop-carried form `%v = phi [%init, %head], [%next, %loop]`
            // where `%next` is defined in the latch - the case a self edge makes
            // into "defined later in this very block".
            if let Some(pred) = phi_pred(f, u) {
                if dbb != pred && !cfg.dominates(dbb, pred) {
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
            } else if !cfg.dominates(dbb, u.bb) {
                e.hard(format!(
                    "%{v} defined in bb{dbb} does not dominate use in bb{}",
                    u.bb
                ));
            }
        }
    }
    e.finish()
}

fn defined(f: &Function, v: VId) -> bool {
    f.def(v).is_some()
}

fn is_used(f: &Function, v: VId) -> bool {
    !f.uses_of(v).is_empty()
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
