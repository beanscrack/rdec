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
