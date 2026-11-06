//! Control-flow structuring: the CFG of an SSA function, turned into the C
//! statements the graph PROVES exist.
//!
//! The emitter used to spell every block as a label and every edge as a `goto`,
//! which is always correct and never readable. This module recognises the shapes
//! machine code actually produces and hands the emitter a statement tree instead:
//!
//! * `if (c) { body }` - a guard whose skipped arm is the join,
//! * `if (c) { then } else { else }` - two arms that share only their join,
//! * `while (c) { body }` - a pre-test loop with one back edge,
//! * `if (a && b) { body }` - the short-circuit chain of nested guards.
//!
//! Everything else - irreducible loops, multi-entry regions, a block with a
//! predecessor the pattern cannot account for, a region nesting deeper than
//! [`MAX_DEPTH`] - stays a label and a `goto`. That is the point: a decompiler
//! that invents an `if` the graph does not support is a miscompiler. Each region
//! is accepted only after checking (a) forward closure, (b) a single exit that is
//! the region's join, (c) no unaccounted entry edge, (d) for a short-circuit
//! link, that its block computes nothing except the compared value and carries no
//! SSA edge copy. The [`Report`] counts what was recognised and what was left
//! flat, with a reason per flat region, and `rdec c --corpus` prints it.
//!
//! The condition of a structured `if` is the value the machine code tested. Where
//! the shape needs the opposite sense (a `while` whose exit is the `then` arm, an
//! early-exit guard in the `else` arm) the condition prints negated, `!(x)`.
//! Never by swapping the arms, and never by inverting the compared operands: that
//! is the classic structuring miscompile and the polarity test in
//! `crates/core/tests/structuring.rs` pins it.
//!
//! SSA edge copies stay on their edge: the copy an edge writes is printed as the
//! last statement of the source block's statement (see [`StmtKind::Plain`]'s
//! consumer in `emit_c`), exactly as the flat emitter spells it, so the rules
//! `passes.rs` documents for loops are untouched by structuring.

use crate::ir::{BbId, Function, Op, Terminator, VId, NO_BB, NO_VAL};
use std::collections::BTreeMap;

/// The virtual "out of the function" region exit.
const EXIT: BbId = NO_BB;
/// Regions deeper than this stay labels and gotos. The graph proofs do not
/// weaken with depth; this is the ceiling that keeps the walk's recursion cost
/// and the emitted C's indentation bounded.
const MAX_DEPTH: usize = 6;
/// The longest short-circuit chain merged into one condition.
const MAX_LINKS: usize = 4;

/// One tested value inside a merged condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cond {
    /// The value the machine code tested: a lifted compare's destination.
    pub cond: VId,
    /// Print it negated, `!(x)`. True only where the shape needs the opposite
    /// sense of the tested value, never as a way to move work around.
    pub negated: bool,
}

/// What a folded chain of guards proves: its conditions, in order and with their
/// polarity; the blocks that carry them; the exit every one of them skips to; and
/// the arm each guard runs.
type Chain = (Vec<Cond>, Vec<BbId>, BbId, Vec<BbId>);

/// How a statement's text hands control to the rest of the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cont {
    /// The block's own terminator, spelled as the flat emitter spells it.
    Term,
    /// `goto lX;` - control leaves the structured text here.
    Goto(BbId),
    /// Control falls out of this statement into the statement that follows it in
    /// the enclosing list.
    Exit,
    /// Control falls into the next statement, which the analysis guarantees
    /// starts at this block. Only legal when the walk really continues there.
    Next(BbId),
    /// The block's own terminator, EXCEPT that the one edge `x` names lands on
    /// the statement printed immediately after this one: the text prints nothing
    /// and the machine's jump is the text's fall-through. The flat layout earns
    /// this by printing `x` next; a statement list that does not print `x` next
    /// must never carry it, which is why only the flat layout builds it.
    Fall(BbId),
}

/// What a statement is.
#[derive(Debug, Clone)]
pub enum StmtKind {
    /// The entry's `Op::Param` assignments, alone. They have to sit above the
    /// walk so the entry's edge copy has a statement to be emitted after.
    Params,
    /// A straight-line block.
    Block,
    /// `if (c) { then } [else { els }]`.
    If {
        /// The tested values, `&&`-merged, in the machine code's order.
        conds: Vec<Cond>,
        then: Vec<Stmt>,
        els: Option<Vec<Stmt>>,
    },
    /// `while (c) { body }` - the guard's block is this statement's block, so the
    /// test runs before the body.
    While { conds: Vec<Cond>, body: Vec<Stmt> },
    /// `goto lX;` on its own.
    Goto,
}

/// A statement: a block of work, a control structure, and where control goes.
#[derive(Debug, Clone)]
pub struct Stmt {
    /// The block this statement is entered at.
    pub bb: BbId,
    pub kind: StmtKind,
    pub cont: Cont,
    /// Emit `l{bb}:` before this statement (a `goto` lands here).
    pub label: bool,
    /// The successors of this statement's block whose `goto` the emitter prints as
    /// `return`: `(target, the value that edge carries)`. A target is listed only
    /// when [`An::tail_of`] proved it a pure tail AND that the value is readable
    /// at this site, and the walk did not count the goto it replaced.
    pub tails: Vec<(BbId, VId)>,
    /// Why this is a flat region, when it is one. Printed in the census, never in
    /// the C: the C gets a `goto`, not a comment.
    pub reason: Option<String>,
}

/// What the structuring walk built for one function.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Regions emitted as `if` with no `else`.
    pub ifs: usize,
    /// Regions emitted as `if`/`else`.
    pub if_else: usize,
    /// Regions emitted as `while`.
    pub whiles: usize,
    /// Conditions built by merging k>=2 guards with `&&`.
    pub shorts: usize,
    /// Blocks left as a label and a `goto`, by reason.
    pub reasons: BTreeMap<String, usize>,
    /// Blocks emitted flat.
    pub flat: usize,
    /// `goto` statements in the emitted C.
    pub gotos: usize,
    /// `lN:` labels in the emitted C.
    pub labels: usize,
    /// `goto lX;` sites the emitter prints as `return` (M6 early return): the
    /// target block computes nothing, returns, and the value that edge carries is
    /// readable here. Counted INSTEAD of a goto, never in addition to one, which
    /// is why the `gotos` total goes down by exactly this number.
    pub early_returns: usize,
    /// Return blocks whose text the early returns made dead: every edge into them
    /// spells the return, so no `goto` names their label and the block prints
    /// nothing. Their `lN:` label is gone from the text, counted here.
    pub absorbed: usize,
    /// `goto lN;` edges the flat layout absorbed by printing `N` next (M6
    /// fall-through): the jump is the text's fall-through, so the text prints
    /// nothing. The block's `lN:` label stays: others may jump to it.
    pub fallthroughs: usize,
}

impl Report {
    /// Any `if`, `if`/`else` or `while` at all.
    pub fn structured(&self) -> bool {
        self.ifs + self.if_else + self.whiles > 0
    }

    /// Record why something stayed flat. Public so the emitter can add the
    /// fallback's own reason to the report it is about to print.
    pub fn reason(&mut self, r: &str) {
        *self.reasons.entry(r.to_string()).or_insert(0) += 1;
    }
}

/// The corpus total of per-function reports.
pub fn sum_reports(v: &[Report]) -> Report {
    let mut out = Report::default();
    for r in v {
        out.ifs += r.ifs;
        out.if_else += r.if_else;
        out.whiles += r.whiles;
        out.shorts += r.shorts;
        out.flat += r.flat;
        out.gotos += r.gotos;
        out.labels += r.labels;
        out.early_returns += r.early_returns;
        out.absorbed += r.absorbed;
        out.fallthroughs += r.fallthroughs;
        for (k, n) in &r.reasons {
            *out.reasons.entry(k.clone()).or_insert(0) += *n;
        }
    }
    out
}

/// Walk the structuring pass over `f`'s CFG.
pub fn structure(f: &Function) -> Structured {
    structure_with_params(f, true)
}

/// The same walk, with the entry's `Op::Param` assignments optionally hoisted
/// into a statement of their own. The C emitter needs that (the entry's SSA edge
/// copy must be emitted after them); the interpreter's copy of the walk does not,
/// because it executes `Op::Param` as an instruction like any other. Both spell
/// the SAME statements out of the SAME analysis - see the tests in
/// `crates/core/tests/structuring.rs`.
pub fn structure_with_params(f: &Function, hoist_params: bool) -> Structured {
    let mut an = An::new(f);
    let mut top: Vec<Stmt> = Vec::new();
    if hoist_params {
        let p = an.param_stmt();
        top.push(p);
    }
    let entry = f.entry;
    if an.in_bounds(entry) {
        top.extend(an.seq(entry, EXIT, 0));
    }
    an.finish(top)
}

/// The label + `goto` spelling: one flat statement per reachable block, in
/// reverse postorder - exactly what this emitter printed before M5, kept as the
/// fallback a structured tree is compared against.
///
/// It is the safe answer for every shape the patterns cannot prove, and it has one
/// property the structured walk has to earn: it prints EVERY reachable block, so
/// every label its `goto`s name is a label its text defines ([`Structured::
/// labels_defined`]). `rdec c` prints `blocks left as goto` from the census, so
/// falling back is visible, never silent.
pub fn flat_walk(f: &Function) -> Structured {
    let mut an = An::new(f);
    let mut top: Vec<Stmt> = Vec::new();
    top.push(an.param_stmt());
    for b in layout_order(f) {
        if !an.in_bounds(b) || an.claimed(b) {
            continue;
        }
        let r = term_reason(&f.blocks[b as usize].term);
        top.push(an.flat_stmt(b, &r));
    }

    // 2. Absorb and prune, to a fixed point: dropping a block makes its
    //    predecessors and its successor adjacent, which can earn a fall-through,
    //    which can in turn make another return block dead. Each round removes at
    //    least one statement, so this terminates.
    loop {
        let mut round = false;

        // Fall-through: a `jmp` whose target is the statement printed next is a jump
        // to the next line of text. The text prints nothing; the target's label
        // stays, because other edges may jump to it. A jump that goes BACKWARD is
        // never absorbed - dropping it would run the loop body one extra time - and
        // a layout that puts a back edge forward is not used at all.
        let mut i = 0usize;
        while i + 1 < top.len() {
            if let Cont::Term = top[i].cont {
                if let Terminator::Jump(x) = f.blocks[top[i].bb as usize].term {
                    // Not an edge the early return already spells: that statement
                    // prints `return`, and a fall-through would print nothing and
                    // lose the return.
                    let spelled_as_return = top[i].tails.iter().any(|(t, _)| *t == x);
                    if top[i + 1].bb == x
                        && matches!(top[i].kind, StmtKind::Block)
                        && !spelled_as_return
                    {
                        top[i].cont = Cont::Fall(x);
                        an.rep.gotos -= 1;
                        an.rep.fallthroughs += 1;
                        // The block's text stays (body and label: others may jump
                        // here); only its census entry as "a block left as a goto"
                        // goes away.
                        an.uncount("fall-through");
                        round = true;
                    }
                }
            }
            i += 1;
        }

        // Absorption, and never of a fall-through's target: that edge is spelled by
        // the layout ("the statement printed next IS the target"), so it relies on
        // the target's text being there.
        let fall_target: Vec<BbId> = top
            .iter()
            .filter_map(|st| match st.cont {
                Cont::Fall(x) => Some(x),
                _ => None,
            })
            .collect();
        let dead: Vec<BbId> = top
            .iter()
            .filter(|st| matches!(st.kind, StmtKind::Block))
            .map(|st| st.bb)
            .filter(|b| an.absorbable(*b) && !fall_target.contains(b))
            .collect();
        for b in dead {
            top.retain(|st| !(matches!(st.kind, StmtKind::Block) && st.bb == b));
            an.rep.absorbed += 1;
            an.unprint(b, "return");
            round = true;
        }
        if !round {
            break;
        }
    }
    an.finish(top)
}

/// The order the flat form prints its blocks: a depth-first walk from the entry
/// that follows each block's own successors, so a `jmp` chain lands on adjacent
/// text and the `goto` that ends it becomes the text's fall-through.
///
/// DFS preorder is not a cosmetic choice; it buys the two properties the layout
/// test in `crates/core/tests/structuring.rs` pins for every fixture:
///
/// * every path the DFS takes from the entry to `b` passes through `b`'s
///   dominators, so a dominator prints BEFORE the block it dominates. That is what
///   keeps [`value_readable`] true of the TEXT and not only of the graph, which is
///   what the early return's `return x7;` depends on;
/// * a back edge (its head dominates its tail) is an edge to a DFS ancestor, so it
///   prints BACKWARD: a loop still reads as a loop, and the flat form's backward
///   jumps stay visible as what they are.
///
/// A layout that fails either property is not used, and RPO is printed instead.
fn layout_order(f: &Function) -> Vec<BbId> {
    let cfg = crate::cfg::Cfg::build(f);
    let mut seen = vec![false; f.blocks.len()];
    let mut out: Vec<BbId> = Vec::new();
    let mut stack: Vec<BbId> = vec![f.entry];
    while let Some(b) = stack.pop() {
        if (b as usize) >= f.blocks.len() || seen[b as usize] {
            continue;
        }
        seen[b as usize] = true;
        out.push(b);
        // Natural order on the stack, so the LAST successor is popped first: for a
        // conditional jump the lifter records the taken target first and the
        // fall-through second, and the fall-through is the edge the text can absorb.
        // Preferring it is what makes a `jcc` chain print as straight-line text.
        for s in cfg.succs[b as usize].iter() {
            if !seen[*s as usize] {
                stack.push(*s);
            }
        }
    }
    let rpo = cfg.rpo.clone();
    if out.len() != rpo.len() || !out.iter().all(|b| rpo.contains(b)) {
        // The DFS reached a different set than the CFG's own reachability: print
        // the order the fallback has always printed.
        return rpo;
    }
    for b in &out {
        for s in &cfg.succs[*b as usize] {
            if cfg.dominates(*s, *b) {
                let from = out.iter().position(|x| x == b).unwrap_or(0);
                let to = out.iter().position(|x| x == s).unwrap_or(usize::MAX);
                if to > from {
                    return rpo;
                }
            }
        }
    }
    out
}

/// The statement tree plus the census.
#[derive(Debug, Clone)]
pub struct Structured {
    pub stmts: Vec<Stmt>,
    pub report: Report,
    /// Indexed by block id: this block's `lN:` label is printed, so the emitter's
    /// `(void)&&lN;` uses must name it - and must name nothing else, or
    /// `-Werror -Wunused-label` refuses the file.
    pub label_blocks: Vec<bool>,
    /// Indexed by block id: a `goto` in the emitted text names this block's label.
    /// [`Structured::labels_defined`] is the statement tree's own compile check.
    pub marks: Vec<bool>,
}

impl Structured {
    /// Every label a `goto` names is a label the text defines.
    ///
    /// A `goto lN;` whose block the walk never printed does not compile - clang
    /// says `use of undeclared label` - and no C statement can stand in for the
    /// block's work. A tree that fails this check is not a spelling of the
    /// function, so the emitter drops it and prints the flat form, which prints
    /// every reachable block and therefore names only labels it defines. This is
    /// the conservative exit of the whole pass: an unstructured region is honest,
    /// a structured one the graph does not support is a miscompile.
    pub fn labels_defined(&self) -> bool {
        fn walk(v: &[Stmt], printed: &mut [bool]) {
            for st in v {
                if (st.bb as usize) < printed.len() {
                    printed[st.bb as usize] = true;
                }
                match &st.kind {
                    StmtKind::If { then, els, .. } => {
                        walk(then, printed);
                        if let Some(e) = els {
                            walk(e, printed);
                        }
                    }
                    StmtKind::While { body, .. } => walk(body, printed),
                    _ => {}
                }
            }
        }
        let mut printed = vec![false; self.marks.len()];
        walk(&self.stmts, &mut printed);
        self.marks.iter().zip(printed.iter()).all(|(m, p)| !m || *p)
    }
}

struct An<'a> {
    f: &'a Function,
    taken: Vec<bool>,
    marked: Vec<bool>,
    /// Reverse postorder index of each block from the entry: the order the emitted
    /// text is in, and the order a join is found in (the FIRST block two arms
    /// share is their join). `usize::MAX` for a block the entry cannot reach.
    rpo: Vec<usize>,
    /// Dominators: what the early return checks before it spells a `return` at a
    /// jump site (the value must be readable on the edge the text takes).
    cfg: crate::cfg::Cfg,
    /// `(from, to)` for every jump site the text spells as `return`. The flat
    /// layout reads this set to decide which return blocks no `goto` names any
    /// more, and drops their text.
    tail_edges: Vec<(BbId, BbId)>,
    rep: Report,
    /// The entry's leading `Op::Param` instructions, already emitted as their own
    /// statement, so a walk through the entry prints the instructions after them.
    params: usize,
}

impl<'a> An<'a> {
    fn new(f: &'a Function) -> Self {
        An {
            f,
            taken: vec![false; f.blocks.len()],
            rpo: rpo_index(f),
            marked: vec![false; f.blocks.len() + 1],
            cfg: crate::cfg::Cfg::build(f),
            tail_edges: Vec::new(),
            rep: Report::default(),
            params: 0,
        }
    }

    /// Stamp `lN:` on exactly the statements a marked block starts, and count
    /// them into the report: the number of labels the emitter is about to print.
    fn finish(self, mut stmts: Vec<Stmt>) -> Structured {
        fn walk(v: &mut [Stmt], marked: &[bool], flag: &mut Vec<bool>, labels: &mut usize) {
            for st in v.iter_mut() {
                // Every marked block gets its label line, whatever statement it
                // starts: a `goto` lands on the BLOCK, and a fall-through
                // statement (`Cont::Next`) is still that block's only text. The
                // first corpus run spelled this exactly - `use of undeclared label
                // 'l7'` - by labelling only the statements that looked like label
                // sites and then jumping to one that was not.
                if marked.get(st.bb as usize).copied().unwrap_or(false)
                    && !matches!(st.kind, StmtKind::Params)
                {
                    // `Params` is the entry's argument loads, printed above the entry's
                    // own statement: the block gets ONE label line, from the statement
                    // that runs its terminator.
                    st.label = true;
                    flag[st.bb as usize] = true;
                    *labels += 1;
                }
                match &mut st.kind {
                    StmtKind::If { then, els, .. } => {
                        walk(then, marked, flag, labels);
                        if let Some(e) = els {
                            walk(e, marked, flag, labels);
                        }
                    }
                    StmtKind::While { body, .. } => walk(body, marked, flag, labels),
                    _ => {}
                }
            }
        }
        let mut flag = vec![false; self.marked.len()];
        let mut n = 0usize;
        walk(&mut stmts, &self.marked, &mut flag, &mut n);
        flag.truncate(self.f.blocks.len());
        let mut rep = self.rep;
        rep.labels = n;
        let mut marks = self.marked.clone();
        marks.truncate(self.f.blocks.len());
        Structured {
            stmts,
            report: rep,
            label_blocks: flag,
            marks,
        }
    }

    fn in_bounds(&self, b: BbId) -> bool {
        b != EXIT && (b as usize) < self.f.blocks.len()
    }

    fn succs(&self, b: BbId) -> Vec<BbId> {
        let mut v = Vec::new();
        if self.in_bounds(b) {
            self.f.blocks[b as usize].term.successors(&mut v);
        }
        v
    }

    /// The guard of a conditional block: (tested value, then arm, else arm).
    fn br(&self, b: BbId) -> Option<(VId, BbId, BbId)> {
        if !self.in_bounds(b) {
            return None;
        }
        match self.f.blocks[b as usize].term {
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => Some((cond, then_bb, else_bb)),
            _ => None,
        }
    }

    fn mark(&mut self, b: BbId) {
        let i = if self.in_bounds(b) {
            b as usize
        } else {
            self.f.blocks.len()
        };
        self.marked[i] = true;
    }

    fn reason(&mut self, r: &str) {
        *self.rep.reasons.entry(r.to_string()).or_insert(0) += 1;
    }

    fn param_stmt(&mut self) -> Stmt {
        let entry = self.f.entry;
        let n = self.f.blocks[entry as usize]
            .instrs
            .iter()
            .take_while(|i| matches!(i.op, Op::Param))
            .count();
        self.params = n;
        if n > 0 {
            self.mark(entry);
        }
        Stmt {
            bb: entry,
            kind: StmtKind::Params,
            cont: Cont::Next(NO_BB),
            label: false,
            tails: Vec::new(),
            reason: None,
        }
    }

    /// A block an enclosing statement already printed. Control reaching it
    /// from here can only be spelled `goto`.
    fn claimed(&self, b: BbId) -> bool {
        self.in_bounds(b) && self.taken[b as usize]
    }

    /// True when this block's terminator leaves the function: `return`, tail
    /// call, noreturn call, trap, `Exit`, unreachable. Such a block is where
    /// control ENDS, so it can serve as the join of two arms.
    fn exit_like(&self, b: BbId) -> bool {
        if !self.in_bounds(b) {
            return false;
        }
        matches!(
            self.f.blocks[b as usize].term,
            Terminator::Return(_)
                | Terminator::TailReturn(_)
                | Terminator::NoReturn
                | Terminator::Unreachable
                | Terminator::Exit
        )
    }

    /// Forward closure of `entry`, stopping at `stop` (the join, NOT part of the
    /// region). `(nodes, exits)`: the blocks a node branches to from outside the
    /// node set. Running into a block an enclosing statement claimed is no
    /// region at all: `None`.
    fn closure(&self, entry: BbId, stop: BbId) -> Option<(Vec<BbId>, Vec<BbId>)> {
        if stop == entry || self.claimed(entry) || !self.in_bounds(entry) {
            return None;
        }
        let mut nodes: Vec<BbId> = Vec::new();
        let mut on = vec![false; self.f.blocks.len()];
        let mut exits: Vec<BbId> = Vec::new();
        let mut stack = vec![entry];
        on[entry as usize] = true;
        while let Some(b) = stack.pop() {
            nodes.push(b);
            for s in self.succs(b) {
                if !self.in_bounds(s) {
                    return None;
                }
                if s == stop {
                    if !exits.contains(&s) {
                        exits.push(s);
                    }
                } else if self.claimed(s) {
                    return None;
                } else if !on[s as usize] {
                    on[s as usize] = true;
                    stack.push(s);
                }
            }
        }
        Some((nodes, exits))
    }

    /// The full forward closure of `entry` plus its boundary: the blocks a node
    /// branches to that are not in the closure. Two arms whose boundaries are the
    /// same single block converge on it - the join of an `if`/`else`.
    fn frontier(&self, entry: BbId) -> Option<(Vec<BbId>, Vec<BbId>)> {
        if self.claimed(entry) || !self.in_bounds(entry) {
            return None;
        }
        let mut on = vec![false; self.f.blocks.len()];
        let mut nodes: Vec<BbId> = Vec::new();
        let mut stack = vec![entry];
        on[entry as usize] = true;
        while let Some(b) = stack.pop() {
            nodes.push(b);
            for s in self.succs(b) {
                if !self.in_bounds(s) {
                    return None;
                }
                if !on[s as usize] && !self.claimed(s) {
                    on[s as usize] = true;
                    stack.push(s);
                }
            }
        }
        let mut bnd: Vec<BbId> = Vec::new();
        for n in &nodes {
            for s in self.succs(*n) {
                if !on[s as usize] && !bnd.contains(&s) {
                    bnd.push(s);
                }
            }
        }
        Some((nodes, bnd))
    }

    /// The region entered at `entry` with join `stop`: forward closed, every edge
    /// that leaves it lands on `stop`, the join is reached, and every edge that
    /// ENTERS it is accounted for (from the region, from its entry, or from one
    /// of `allow` - the header blocks a pattern names).
    ///
    /// An unaccounted predecessor is a multi-entry region. No `if`, `while` or
    /// `for` in C has a second door, so this answers `None` and the caller leaves
    /// those blocks as labels and gotos.
    fn region(&mut self, entry: BbId, stop: BbId, allow: &[BbId]) -> Option<Vec<BbId>> {
        let (nodes, exits) = self.closure(entry, stop)?;
        if nodes.is_empty() || exits.is_empty() || !exits.iter().all(|x| *x == stop) {
            return None;
        }
        for n in &nodes {
            for p in preds(self.f, *n) {
                if p != entry && !nodes.contains(&p) && !allow.contains(&p) {
                    return None;
                }
            }
        }
        Some(nodes)
    }

    /// A closed region that leaves the function: forward closed, no edge leaves
    /// it, every entry edge is accounted for, and a block in it returns, traps or
    /// tail-calls. This is the body of an early-exit guard - the shape machine
    /// code makes when a C function checks an argument and returns.
    fn closed_exit_region(&self, entry: BbId, allow: &[BbId]) -> Option<Vec<BbId>> {
        let (nodes, exits) = self.closure(entry, EXIT)?;
        if nodes.is_empty() || !exits.is_empty() {
            return None;
        }
        for n in &nodes {
            for p in preds(self.f, *n) {
                if p != entry && !nodes.contains(&p) && !allow.contains(&p) {
                    return None;
                }
            }
        }
        nodes.iter().any(|m| self.exit_like(*m)).then_some(nodes)
    }

    /// The short-circuit chain: `h` is a guard whose skipped arm is the join and
    /// whose tested arm is a guard that skips to the same join, `k` deep.
    ///
    /// A link merges into the condition only when its block computes NOTHING
    /// except the compared value (the machine code that built `&&` computes
    /// nothing on the skipped edge, so merged C must not run a statement the
    /// machine code skips) and when the edge into it carries no SSA edge copy (an
    /// edge copy is an assignment; folding its block into a condition would run
    /// it on the paths the machine code skips).
    ///
    /// Returns (conditions, link blocks, body entry, body region).
    /// The short-circuit chain: `h` is a guard that skips to `exit`, and the arm
    /// it runs is itself a guard that skips to the same `exit`, `k` deep. The
    /// conditions fold into one C expression, `a && b`, with the body of the last
    /// guard as the `if` body.
    ///
    /// Folding a guard into a condition is only the same program when NOTHING the
    /// machine code runs on one arm gets moved onto the other:
    ///
    /// - every link after the first computes nothing except its compared value
    ///   ([`An::chain_link_ok`]) - the machine code evaluates `b` on one path and
    ///   the merged C must not run a statement on the path it skips;
    /// - every edge into a link carries no SSA edge copy - an edge copy is an
    ///   assignment, and the link's entry edge is inside a condition in the C;
    /// - the join's phi copies agree across ALL the skip edges: the merged C leaves
    ///   the condition at the join from several places, and one copy before the
    ///   `if` can only be right if all of them write the same value ([`An::
    ///   skip_edges_agree`]).
    ///
    /// Returns (conditions, the guard blocks folded in, the body's entry block,
    /// the body's region).
    fn chain(&mut self, h: BbId, exit: BbId) -> Option<Chain> {
        if exit == EXIT {
            return None;
        }
        let (_c0, t0, e0) = self.br(h)?;
        if t0 == e0 {
            return None;
        }
        let mut conds: Vec<Cond> = Vec::new();
        let mut links: Vec<BbId> = Vec::new();
        let mut allow: Vec<BbId> = Vec::new();
        let (mut cur, mut cond) = (h, _c0);
        for _ in 0..MAX_LINKS {
            let (c2, t2, e2) = self.br(cur)?;
            let _ = c2;
            // The guard skips to the join on one arm and runs the next thing on
            // the other. `negated` is the machine's polarity, never an arm swap.
            let (neg, arm) = if e2 == exit {
                (false, t2)
            } else if t2 == exit {
                (true, e2)
            } else {
                break;
            };
            if arm == exit || self.claimed(arm) {
                break;
            }
            if !links.is_empty() {
                // A folded link: nothing but the compare, and no assignment on
                // the edge that leads into it.
                if !self.chain_link_ok(cur, cond) || self.edge_copy_from(cur) {
                    break;
                }
                let entered = preds(self.f, cur)
                    .iter()
                    .all(|p| allow.contains(p) || links.contains(p));
                if !entered {
                    break;
                }
            }
            links.push(cur);
            allow.push(cur);
            conds.push(Cond { cond, negated: neg });
            match self.br(arm) {
                // The next link: a guard that skips to the same join, computing
                // nothing else, with all the skip edges in agreement.
                Some((c3, t3, e3))
                    if (t3 == exit || e3 == exit)
                        && self.chain_link_ok(arm, c3)
                        && !self.edge_copy_from(arm)
                        && self.skip_edges_agree(&links, exit)
                        && links.len() < MAX_LINKS =>
                {
                    cur = arm;
                    cond = c3;
                }
                // Anything else is the body.
                _ => {
                    let reg = self.region(arm, exit, &allow)?;
                    return Some((conds, links, arm, reg));
                }
            }
        }
        None
    }

    /// Do the join's phi copies agree across the skip edges of the guards folded
    /// into one condition?
    ///
    /// `if (a && b) body` leaves the condition for the join in two places (`a`
    /// false, and `a` true with `b` false) and the C text can carry the join's
    /// edge copies in only one of them - before the `if`, where they run on every
    /// path. That is the same program exactly when every skip edge writes the same
    /// value into the join, which is what this checks. No phi at the join (the
    /// common case, and every fixture that exercises the merge) is vacuously true.
    fn skip_edges_agree(&self, links: &[BbId], exit: BbId) -> bool {
        let first = match links.first() {
            Some(f) => *f,
            None => return true,
        };
        let phi = match &self.f.blocks[exit as usize].instrs.first() {
            Some(i) => match &i.op {
                Op::Phi(pairs) => pairs,
                _ => return true,
            },
            None => return true,
        };
        for l in links.iter().skip(1) {
            for (v, from) in phi.iter() {
                if *from == *l {
                    // The value this skip edge writes must be the one the first
                    // guard's skip edge writes: the single copy in the text is
                    // the copy of THAT edge.
                    match phi.iter().find(|(_, f)| *f == first) {
                        Some((v0, _)) if v0 == v => {}
                        _ => return false,
                    }
                }
            }
        }
        true
    }

    /// Can the guard block `g`, whose tested value is `cond`, fold into a
    /// condition? See `chain` for what the two checks mean.
    fn chain_link_ok(&self, g: BbId, cond: VId) -> bool {
        if cond == NO_VAL {
            return false;
        }
        let b = &self.f.blocks[g as usize];
        // The block computes the compared value and nothing else: no instruction
        // with another destination, no phi with more than one incoming edge.
        if b.instrs.iter().any(|i| i.dest != cond) {
            return false;
        }
        if phi_pairs(b).iter().filter(|(_, p)| *p == g).count() > 1 {
            return false;
        }
        if self.f.use_count(cond) != 1 {
            return false;
        }
        !self.edge_copy_from(g)
    }

    /// True when an edge leaving `b` writes an SSA edge copy, i.e. `b` is a
    /// predecessor of a phi block.
    fn edge_copy_from(&self, b: BbId) -> bool {
        self.succs(b).into_iter().any(|s| {
            self.in_bounds(s)
                && phi_pairs(&self.f.blocks[s as usize])
                    .iter()
                    .any(|(_, p)| *p == b)
        })
    }

    /// The arms of an `if`/`else`: two regions that share only their join.
    fn if_arms(&mut self, h: BbId, exit: BbId) -> Option<(Vec<BbId>, Vec<BbId>)> {
        let (_c, t, e) = self.br(h)?;
        if exit == EXIT || t == e || t == exit || e == exit {
            return None;
        }
        let a = match self.region(t, exit, &[h]) {
            Some(a) => a,
            None => {
                return None;
            }
        };
        let b = match self.region(e, exit, &[h]) {
            Some(b) => b,
            None => {
                return None;
            }
        };
        if a.iter().any(|m| b.contains(m)) {
            return None;
        }
        // The same test the early-exit guard makes, for both arms: an arm's text
        // must hand control to the join (or to a block inside the arm), never to a
        // block outside the region that the walk will never print.
        for (entry, arm) in [(t, &a), (e, &b)] {
            let stmts = self.probe(|an| an.seq(entry, exit, 1));
            // (a) every edge the arm's text can take stays inside the arm or lands
            //     on the join; (b) every label a `goto` inside the arm names is a
            //     block the arm prints (the join is printed by the parent, after
            //     the statement, so it counts as defined).
            if !stmts
                .iter()
                .all(|st| self.stmt_closes(st, &|x| x == exit || arm.contains(&x)))
            {
                return None;
            }
            if !self.probe(|an| {
                let base = an.marked.clone();
                let st = an.seq(entry, exit, 1);
                an.labels_defined_from(&st, &base, &|x| x == exit)
            }) {
                return None;
            }
        }
        Some((a, b))
    }

    /// A pre-test loop: a guard that branches to a body which (directly, or
    /// through blocks below the guard) jumps back to it, and to an exit arm that
    /// is NOT below the guard - an arm that lives inside the loop is a nested
    /// region, not the loop's exit.
    ///
    /// Returns (body entry, exit arm, negated): `negated` is true exactly when the
    /// BODY is the `else` arm, so the C condition is the tested value spelled `!`.
    /// The arms are never swapped - swapping them is the other way to make this
    /// bug, and it is the one a reader of the C would not see.
    fn pre_loop(&mut self, h: BbId) -> Option<(BbId, BbId, bool)> {
        let (_c, t, e) = self.br(h)?;
        if t == e {
            return None;
        }
        let mut below: Vec<BbId> = Vec::new();
        forward_from(self.f, h, &mut below);
        let back = |a: BbId| a == h || (below.contains(&a) && reaches(self.f, a, h));
        // The polarity is the machine's, and it is decided by WHICH arm runs the
        // body: a `then` arm body runs while the tested value is true, so the C
        // condition is the value itself; an `else` arm body runs while it is
        // false, spelled `!`. Getting this backwards is a `while` that runs the
        // body on the iteration the machine code skips, and it is exactly what
        // the zero-iteration case in `crates/core/tests/structuring.rs` fails.
        let (a, x, neg) = if back(t) && !back(e) {
            (t, e, false)
        } else if back(e) && !back(t) {
            (e, t, true)
        } else {
            return None;
        };
        if self.claimed(a) || self.claimed(x) || x == EXIT {
            return None;
        }
        let mut body: Vec<BbId> = below
            .iter()
            .copied()
            .filter(|m| *m != h && self.in_bounds(*m) && reaches(self.f, *m, a))
            .collect();
        body.push(a);
        for n in &body {
            for p in preds(self.f, *n) {
                if p != h && p != a && !body.contains(&p) && !below.contains(&p) {
                    return None;
                }
            }
        }
        Some((a, x, neg))
    }

    /// The block two arms of a branch converge on: the first block, in the order
    /// the emitted text runs, that both arms' forward closures reach. That is a
    /// CANDIDATE only - `region` is what proves a set of blocks ends there and
    /// nowhere else, and a candidate that fails that proof leaves the branch as
    /// labels and gotos. Picking the first shared block can only miss a shape,
    /// never licence one the graph does not support.
    fn join_of(&self, t: BbId, e: BbId) -> Option<BbId> {
        let (n_t, _) = self.frontier(t)?;
        let (n_e, _) = self.frontier(e)?;
        let mut best: Option<(usize, BbId)> = None;
        for b in &n_t {
            if n_e.contains(b) {
                let r = self.rpo.get(*b as usize).copied().unwrap_or(usize::MAX);
                if best.map(|(r0, _)| r < r0).unwrap_or(true) {
                    best = Some((r, *b));
                }
            }
        }
        best.map(|(_, b)| b)
    }

    /// The join of the single-if / short-circuit patterns. Inside a region the
    /// join is the region's exit, given. At the top level the exit is the end of
    /// the function, so the join has to be FOUND: the arm that runs must land on
    /// the other arm's block and nowhere else - `if (c) { body } rest`, where
    /// `rest` is the block the skipped arm names. `None` is the honest answer
    /// whenever the running arm escapes to more than one place.
    fn chain_join(&self, h: BbId, exit: BbId) -> Option<BbId> {
        if exit != EXIT {
            return Some(exit);
        }
        let (_c, t, e) = self.br(h)?;
        if t == e {
            return None;
        }
        // The skipped arm's block IS the join: `if (c) { body } rest`, where `rest`
        // is the block the guard jumps over. Anything else is not this pattern.
        let j = self.join_of(t, e)?;
        if j == e || j == t {
            Some(j)
        } else {
            None
        }
    }

    /// The join of an `if`/`else`: the one block both arms converge on. At the top
    /// level the arms' boundaries must be the same single block, and the arms may
    /// not share a block below it (a shared block below the join is a multi-entry
    /// region, which no C statement expresses).
    fn if_join(&self, h: BbId, exit: BbId) -> Option<BbId> {
        if exit != EXIT {
            return Some(exit);
        }
        let (_c, t, e) = self.br(h)?;
        if t == e {
            return None;
        }
        let j = self.join_of(t, e)?;
        if j == t || j == e {
            return None;
        }
        Some(j)
    }

    /// The early-exit guard: one arm is a closed region that leaves the function
    /// (the `if (bad) return;` shape machine code loves) and the other arm is
    /// simply what comes next. No join is involved - the guard's body ENDS the
    /// function. Returns (exit arm, continuation arm, negated); the polarity is
    /// the machine's, so a `then` arm that exits is tested as `if (c)` and an
    /// `else` arm that exits as `if (!(c))`.
    fn guard_exit(&mut self, h: BbId) -> Option<(BbId, BbId, bool)> {
        let (_c, t, e) = self.br(h)?;
        if t == e {
            return None;
        }
        for (arm, next, neg) in [(t, e, false), (e, t, true)] {
            if !self.in_bounds(next) || self.claimed(next) || self.claimed(arm) {
                continue;
            }
            let nodes = match self.closed_exit_region(arm, &[h]) {
                Some(n) => n,
                None => continue,
            };
            // The body leaves by its terminators only, and the continuation is a
            // block the walk can keep walking. A `Jump` in the body whose target is
            // in the body is the fall-through of the text; one that leaves is the
            // `o40_f` shape: a flat block inside the arm that jumps out of the
            // statement, whose target then never gets a label line (the first
            // corpus run spelled it: `use of undeclared label`).
            let ok = self
                .probe(|a| a.seq(arm, EXIT, 1))
                .iter()
                .all(|st| self.stmt_closes(st, &|x| nodes.contains(&x)));
            if ok {
                return Some((arm, next, neg));
            }
        }
        None
    }

    /// Run a walk as a PREDICATE: the answer is kept, the bookkeeping is not.
    /// The arm probes below ask "what would the walk do here", and a probe that
    /// left its marks, claims and counters behind would inflate the census with
    /// shapes it only considered.
    fn probe<T>(&mut self, run: impl FnOnce(&mut Self) -> T) -> T {
        let (taken, marked, rep, params, tails) = (
            self.taken.clone(),
            self.marked.clone(),
            self.rep.clone(),
            self.params,
            self.tail_edges.clone(),
        );
        let out = run(self);
        self.taken = taken;
        self.marked = marked;
        self.rep = rep;
        self.params = params;
        self.tail_edges = tails;
        out
    }

    /// Would printing `stmts` leave a label that a `goto` names undefined?
    ///
    /// A `goto` (a flat terminator's spelling, a `Cont::Goto`, the target of a
    /// fall-through that had to become a jump) marks the block whose `lN:` the text
    /// names, and the emitter prints the label lines from exactly those marks. A
    /// marked block that no statement of this statement list prints - because the
    /// walk ended before reaching it, or because an enclosing region took it - has
    /// no label line, and the file will not compile: clang's
    /// `use of undeclared label 'l7'` in `o40_f` is this, and it is the reason a
    /// region's proof includes "the blocks my arms' gotos name are blocks my arms
    /// print".
    /// `base` is the mark set as it stood BEFORE this region's walk: marks the
    /// surrounding text already needed are defined by that text, not by this
    /// region, so they are not this region's promise. A probe that checks the
    /// global mark set fails every arm of every `if` in a function with
    /// parameters - the entry is marked by its own argument loads.
    fn labels_defined_from(
        &self,
        stmts: &[Stmt],
        base: &[bool],
        extra: &dyn Fn(BbId) -> bool,
    ) -> bool {
        let mut printed: Vec<bool> = vec![false; self.f.blocks.len()];
        fn walk(v: &[Stmt], printed: &mut Vec<bool>) {
            for st in v {
                if (st.bb as usize) < printed.len() {
                    printed[st.bb as usize] = true;
                }
                match &st.kind {
                    StmtKind::If { then, els, .. } => {
                        walk(then, printed);
                        if let Some(e) = els {
                            walk(e, printed);
                        }
                    }
                    StmtKind::While { body, .. } => walk(body, printed),
                    _ => {}
                }
            }
        }
        walk(stmts, &mut printed);
        self.marked
            .iter()
            .enumerate()
            .take(self.f.blocks.len())
            .all(|(b, m)| {
                !m || base.get(b).copied().unwrap_or(false) || printed[b] || extra(b as BbId)
            })
    }

    /// Does this statement's text account for every edge it can reach?
    ///
    /// A flat block's C ends in its own terminator, and the flat spelling names
    /// EVERY successor (`if (c) goto lA; else goto lB;`), so each of them must be
    /// inside the region - the labels the text defines are the labels a `goto` may
    /// name. A fall-through names only the block after it, so only that one needs
    /// covering. `inside` is the region's own membership test: the arm's blocks,
    /// plus (for an arm of a region with a join) the join itself.
    ///
    /// `o40_f` is the shape this catches: inside `if (x44)`, the arm block l4
    /// branches to l5 and l7, both outside the arm. The arm's text ends in
    /// `if (x62) goto l7; else goto l5;` and neither label is ever defined - clang
    /// says `use of undeclared label 'l7'`, and the census's own goto count (which
    /// counts flat terminators by their spelling, 2 for a branch) is what makes
    /// the disagreement visible from the outside.
    fn stmt_closes(&self, st: &Stmt, inside: &dyn Fn(BbId) -> bool) -> bool {
        match st.cont {
            Cont::Exit => true,
            Cont::Next(x) => inside(x),
            Cont::Fall(x) => inside(x),
            Cont::Goto(x) => inside(x),
            Cont::Term => successors_of(&self.f.blocks[st.bb as usize].term)
                .iter()
                .all(|x| inside(*x)),
        }
    }

    /// Where a statement's text hands control to `x`. `Cont::Next` is the
    /// fall-through, which is only legal because the walk then emits `x`'s
    /// statement next; a claimed, out-of-bounds, or entry `x` becomes a `goto`
    /// here and nowhere else. `from` is the block whose text this is: the early
    /// return needs the edge, not just the target.
    fn cont_to(&mut self, from: BbId, x: BbId, exit: BbId) -> (Cont, Option<String>, Option<VId>) {
        if x == exit {
            return (Cont::Exit, None, None);
        }
        if x == self.f.entry {
            self.mark(x);
            self.rep.gotos += 1;
            let r = "back edge to the function entry";
            self.reason(r);
            self.rep.flat += 1;
            return (Cont::Goto(x), Some(r.to_string()), None);
        }
        if self.in_bounds(x) && !self.claimed(x) {
            return (Cont::Next(x), None, None);
        }
        // The jump is real (the target is claimed, out of the region, or already
        // emitted). Its text is a `return` when the target is a pure tail and the
        // value is readable here: the block then ends the function on this edge, so
        // it is not a block "left as goto" - no reason, no `flat`, and no label
        // marked, because no `goto` names it from here.
        if let Some(v) = self.tail_of(from, x) {
            self.tail_edges.push((from, x));
            self.rep.early_returns += 1;
            return (Cont::Goto(x), None, Some(v));
        }
        self.mark(x);
        self.rep.gotos += 1;
        let r = "join is not this region's exit (fall-through kept as goto)";
        self.reason(r);
        self.rep.flat += 1;
        (Cont::Goto(x), Some(r.to_string()), None)
    }

    /// The `return` a jump site `from -> to` may print instead of `goto l{to};`,
    /// as the value that edge carries (`NO_VAL` for a `return;`).
    ///
    /// Two proofs, both of which the C text needs and the CFG alone does not give:
    ///
    /// * **the target is a pure tail.** Its block computes NOTHING: every
    ///   instruction is a phi (whose value arrives by the edge copy the emitter
    ///   already prints at the end of `from`'s text), and its terminator is a
    ///   `return`. A block with a store, a call, or an arithmetic instruction needs
    ///   a STATEMENT, and a `return` spelling has no room for one: the machine
    ///   would run that work on the edge into the target and the C would run
    ///   nothing. That case is refused and counted out loud
    ///   (`early return refused: target computes`).
    /// * **the value is readable HERE.** The emitted text runs statements, not the
    ///   graph, so a `return x7;` is only the same program if `x7` was written on
    ///   the path that got here. Where the target returns a phi, the text spells
    ///   the phi operand that arrives on THIS edge (the edge copy is the value the
    ///   machine returns on this edge, and the emitter prints it as the last
    ///   statement of `from`), never the phi itself. Where the target returns a
    ///   value defined elsewhere, [`value_readable`] is the dominance check.
    fn tail_of(&mut self, from: BbId, to: BbId) -> Option<VId> {
        if !self.in_bounds(to) || to == self.f.entry {
            return None;
        }
        let b = &self.f.blocks[to as usize];
        let v = match b.term {
            Terminator::Return(v) => v,
            _ => return None,
        };
        if b.instrs.iter().any(|i| !matches!(i.op, Op::Phi(_))) {
            // Loud, not silent: this is a goto the pass chose not to spell.
            self.reason("early return refused: target computes");
            return None;
        }
        let mut val = v;
        for i in &b.instrs {
            if let Op::Phi(pairs) = &i.op {
                if i.dest == v {
                    // The value this edge hands the phi. No pair for `from` means
                    // the graph and the edge disagree; keep the goto.
                    val = pairs.iter().find(|(_, p)| *p == from)?.0;
                }
            }
        }
        if val != NO_VAL && !value_readable(self.f, &self.cfg, val, from) {
            self.reason("early return refused: value not readable at the site");
            return None;
        }
        Some(val)
    }

    /// The tail spellings for every successor of `from`, recorded as this
    /// statement's `tails` and as `(from, to)` edges for the layout's absorption
    /// check. Each accepted tail is counted in `early_returns` where the caller
    /// would otherwise count a `goto`, so the census and the text stay one number.
    fn tails_for(&mut self, from: BbId, succs: &[BbId]) -> Vec<(BbId, VId)> {
        let mut out: Vec<(BbId, VId)> = Vec::new();
        for s in succs {
            if let Some(v) = self.tail_of(from, *s) {
                self.tail_edges.push((from, *s));
                out.push((*s, v));
            }
        }
        out
    }

    /// The statements of the region entered at `entry`, whose fall-out is `exit`.
    fn seq(&mut self, entry: BbId, exit: BbId, depth: usize) -> Vec<Stmt> {
        let mut out: Vec<Stmt> = Vec::new();
        let mut cur = entry;
        while cur != exit {
            if depth >= MAX_DEPTH {
                let from = out.last().map(|st| st.bb).unwrap_or(NO_BB);
                out.push(self.goto_stmt(from, cur, "region nesting deeper than the ceiling"));
                break;
            }
            if !self.in_bounds(cur) || self.claimed(cur) {
                let from = out.last().map(|st| st.bb).unwrap_or(NO_BB);
                out.push(self.goto_stmt(from, cur, "join already emitted by an enclosing region"));
                break;
            }
            // 1. Pre-test loop: the guard lives in this block, so `while` is the
            //    only spelling that tests it before the body runs.
            if let Some((a, x, neg)) = self.pre_loop(cur) {
                let cond = match self.br(cur) {
                    Some((c, _, _)) => c,
                    None => break,
                };
                self.rep.whiles += 1;
                self.taken[cur as usize] = true;
                let body = self.seq(a, cur, depth + 1);
                let (cont, why, tail) = self.cont_to(cur, x, exit);
                let tails = tail.map(|v| vec![(x, v)]).unwrap_or_default();
                out.push(Stmt {
                    bb: cur,
                    kind: StmtKind::While {
                        conds: vec![Cond { cond, negated: neg }],
                        body,
                    },
                    cont,
                    label: false,
                    tails,
                    reason: why,
                });
                match cont {
                    Cont::Next(y) => cur = y,
                    _ => break,
                }
                continue;
            }
            // 2. Single if / short-circuit: the skipped arm is the join, the
            //    tested arm is a region that runs and arrives there.
            if let Some(stop) = self.chain_join(cur, exit) {
                if let Some((conds, links, body_entry, _reg)) = self.chain(cur, stop) {
                    for l in &links {
                        self.taken[*l as usize] = true;
                    }
                    if links.len() >= 2 {
                        self.rep.shorts += 1;
                    }
                    self.rep.ifs += 1;
                    let then = self.seq(body_entry, stop, depth + 1);
                    let (cont, why, tail) = self.cont_to(cur, stop, exit);
                    let tails = tail.map(|v| vec![(stop, v)]).unwrap_or_default();
                    out.push(Stmt {
                        bb: cur,
                        kind: StmtKind::If {
                            conds,
                            then,
                            els: None,
                        },
                        cont,
                        label: false,
                        tails,
                        reason: why,
                    });
                    match cont {
                        Cont::Next(y) => cur = y,
                        _ => break,
                    }
                    continue;
                }
            }
            // 3. If/else: two regions to one join. The condition is the value the
            //    machine code tested, un-negated.
            if let Some(stop) = self.if_join(cur, exit) {
                if let (Some((cond, t, e)), Some(_arms)) = (self.br(cur), self.if_arms(cur, stop)) {
                    self.rep.if_else += 1;
                    self.taken[cur as usize] = true;
                    let then = self.seq(t, stop, depth + 1);
                    let els = self.seq(e, stop, depth + 1);
                    let (cont, why, tail) = self.cont_to(cur, stop, exit);
                    let tails = tail.map(|v| vec![(stop, v)]).unwrap_or_default();
                    out.push(Stmt {
                        bb: cur,
                        kind: StmtKind::If {
                            conds: vec![Cond {
                                cond,
                                negated: false,
                            }],
                            then,
                            els: Some(els),
                        },
                        cont,
                        label: false,
                        tails,
                        reason: why,
                    });
                    match cont {
                        Cont::Next(y) => cur = y,
                        _ => break,
                    }
                    continue;
                }
            }
            // 4. Early-exit guard: the body ends the function, the skip arm is
            //    what comes next.
            if let Some((arm, next_arm, neg)) = self.guard_exit(cur) {
                let cond = match self.br(cur) {
                    Some((c, _, _)) => c,
                    None => break,
                };
                self.rep.ifs += 1;
                self.taken[cur as usize] = true;
                let then = self.seq(arm, EXIT, depth + 1);
                let (cont, why, tail) = self.cont_to(cur, next_arm, exit);
                let tails = tail.map(|v| vec![(next_arm, v)]).unwrap_or_default();
                out.push(Stmt {
                    bb: cur,
                    kind: StmtKind::If {
                        conds: vec![Cond { cond, negated: neg }],
                        then,
                        els: None,
                    },
                    cont,
                    label: false,
                    tails,
                    reason: why,
                });
                match cont {
                    Cont::Next(y) => cur = y,
                    _ => break,
                }
                continue;
            }
            // 5. An ordinary block: its statements, then the edge it takes.
            let term = self.f.blocks[cur as usize].term.clone();
            self.taken[cur as usize] = true;
            match term {
                Terminator::Jump(s) => {
                    let (cont, why, tail) = self.cont_to(cur, s, exit);
                    let tails = tail.map(|v| vec![(s, v)]).unwrap_or_default();
                    out.push(self.plain(cur, cont, why, tails));
                    match cont {
                        Cont::Next(y) => cur = y,
                        _ => break,
                    }
                }
                other => {
                    let r = term_reason(&other);
                    out.push(self.flat_stmt(cur, &r));
                    break;
                }
            }
        }
        out
    }

    fn goto_stmt(&mut self, from: BbId, to: BbId, reason: &str) -> Stmt {
        // The same two proofs as `cont_to`: a `goto` at this site becomes a
        // `return` when the target computes nothing and the value is readable HERE
        // (`from` is the block whose text ends with this jump). NO_BB from a region
        // with no printed site block names no edge, so it earns no tail.
        if let Some(v) = self.tail_of(from, to) {
            self.tail_edges.push((from, to));
            self.rep.early_returns += 1;
            return Stmt {
                bb: to,
                kind: StmtKind::Goto,
                cont: Cont::Exit,
                label: false,
                tails: vec![(to, v)],
                reason: None,
            };
        }
        self.reason(reason);
        self.rep.flat += 1;
        self.mark(to);
        self.rep.gotos += 1;
        Stmt {
            bb: to,
            kind: StmtKind::Goto,
            cont: Cont::Exit,
            label: false,
            tails: Vec::new(),
            reason: Some(reason.to_string()),
        }
    }

    fn plain(
        &mut self,
        bb: BbId,
        cont: Cont,
        reason: Option<String>,
        tails: Vec<(BbId, VId)>,
    ) -> Stmt {
        if let Some(r) = &reason {
            self.reason(r);
            self.rep.flat += 1;
        }
        Stmt {
            bb,
            kind: StmtKind::Block,
            cont,
            label: false,
            tails,
            reason,
        }
    }

    /// The label + `goto` form: the block's statements, then its terminator
    /// spelled exactly as the flat emitter spells it.
    fn flat_stmt(&mut self, bb: BbId, reason: &str) -> Stmt {
        self.reason(reason);
        self.rep.flat += 1;
        self.mark(bb);
        self.taken[bb as usize] = true;
        let succ = successors_of(&self.f.blocks[bb as usize].term);
        // The edges this block's text may print as `return` instead of `goto`.
        let tails = self.tails_for(bb, &succ);
        self.rep.gotos += term_gotos(&self.f.blocks[bb as usize].term, &tails);
        self.rep.early_returns += tails.len();
        for s in &succ {
            // A `return` names no label, so a tail edge marks nothing: that is how
            // a pure-return block ends up with no label to drop when every edge
            // into it returns.
            if !tails.iter().any(|(t, _)| t == s) {
                self.mark(*s);
            }
        }
        Stmt {
            bb,
            kind: StmtKind::Block,
            cont: Cont::Term,
            label: false,
            tails,
            reason: Some(reason.to_string()),
        }
    }

    /// May the flat form print nothing for this block? A pure-return block whose
    /// every incoming edge the text spells `return` has no work left to run: its
    /// phis are written by the predecessors' edge copies, and its `return` by each
    /// of those edges. The phi destinations must not be read outside the block:
    /// their values are the last edge copy that ran in TEXT order, which is a
    /// claim about the block's own `return` only.
    fn absorbable(&self, b: BbId) -> bool {
        if !self.in_bounds(b) || b == self.f.entry {
            return false;
        }
        let blk = &self.f.blocks[b as usize];
        if !matches!(blk.term, Terminator::Return(_)) {
            return false;
        }
        if blk.instrs.iter().any(|i| !matches!(i.op, Op::Phi(_))) {
            return false;
        }
        for i in &blk.instrs {
            if self.f.uses_of(i.dest).iter().any(|u| u.bb != b) {
                return false;
            }
        }
        self.cfg.preds[b as usize]
            .iter()
            .all(|p| self.tail_edges.contains(&(*p, b)))
    }

    /// Take back the census of a block the flat text no longer prints: its label,
    /// its `flat` count and the reason line that named it. The block's work is in
    /// the text - spelled on the edges that replaced it - and the census must say
    /// so, not keep a count of a statement that is gone.
    fn unprint(&mut self, b: BbId, reason: &str) {
        self.marked[b as usize] = false;
        self.uncount(reason);
    }

    /// The census of a block that no longer ends in a `goto` leaves the "blocks
    /// left as goto" total and the reason histogram that named it.
    fn uncount(&mut self, reason: &str) {
        self.rep.flat = self.rep.flat.saturating_sub(1);
        if let Some(n) = self.rep.reasons.get_mut(reason) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.rep.reasons.remove(reason);
            }
        }
    }
}

/// Every block forward-reachable from `from`, `from` included.
fn forward_from(f: &Function, from: BbId, out: &mut Vec<BbId>) {
    if from >= f.blocks.len() as u32 || out.contains(&from) {
        return;
    }
    out.push(from);
    let mut succ = Vec::new();
    f.blocks[from as usize].term.successors(&mut succ);
    for s in succ {
        forward_from(f, s, out);
    }
}

/// Is `to` reachable from `from`?
fn reaches(f: &Function, from: BbId, to: BbId) -> bool {
    let mut seen = Vec::new();
    forward_from(f, from, &mut seen);
    seen.contains(&to)
}

/// The `goto` statements the flat spelling of this terminator prints: one per
/// edge, minus the edges the early return spells as `return` (`tails`, which the
/// emitter prints from this same list). The census that does not match the
/// emitter's spelling is the one that failed the first corpus run: 51 gotos in the
/// text against 30 in the report.
///
/// The other discount - the `goto` the LAYOUT absorbs by printing the target next
/// - is not a property of the terminator, so it is not taken here. `flat_walk`
///   turns that statement's `Cont::Term` into `Cont::Fall` and decrements its own
///   count where it makes the change.
pub fn term_gotos(t: &Terminator, tails: &[(BbId, VId)]) -> usize {
    let named = |b: BbId| tails.iter().any(|(x, _)| *x == b);
    match t {
        Terminator::Jump(b) => usize::from(!named(*b)),
        Terminator::Branch {
            then_bb, else_bb, ..
        } => usize::from(!named(*then_bb)) + usize::from(!named(*else_bb)),
        // A `switch` spells one `goto` per table slot and traps past the end.
        Terminator::Switch { targets, .. } => targets.iter().filter(|b| !named(**b)).count(),
        _ => 0,
    }
}

/// Is `v` readable in the printed text at the END of block `at`'s statement - the
/// point where its terminator runs, which is where an early `return` is spelled?
///
/// The emitter declares a local for every value at the top of the function, so
/// scope is never the question; the question is whether the WRITE has run. `v` is
/// readable when the block that defines it is `at` itself (the emitter prints a
/// block's instructions, and its phi edge copies, before its terminator), or when
/// it dominates `at`: every path to `at` runs that block's text first, and the flat
/// layout prints a dominator before the block it dominates.
///
/// **This check is defensive on valid IR, and the named gap is that no emitted C
/// can witness it failing.** For a target that returns a phi, SSA validity is
/// stronger than this check: `verify_ssa` requires a phi operand's def to dominate
/// the predecessor of the phi's block (`crates/core/src/verify.rs`), which is
/// exactly "readable on the edge". For a target that returns a plain value, if the
/// def dominates the target's block then it dominates every predecessor of the
/// target (every path through a predecessor continues into the target, and the def
/// is on it), so the check cannot fire on a function `verify_ssa` accepted. It
/// guards the pass against the graph it was handed, not against a shape the
/// pipeline produces: `crates/core/tests/structuring.rs` witnesses it on a
/// hand-built graph that `verify_ssa` rejects, so the rule is proven live, and the
/// gap - "no corpus function can show it refusing" - stays on the record.
pub fn value_readable(f: &Function, cfg: &crate::cfg::Cfg, v: VId, at: BbId) -> bool {
    if (at as usize) >= f.blocks.len() {
        return false;
    }
    let (d, _) = match f.def(v) {
        Some(x) => x,
        None => return false,
    };
    if d == at {
        return true;
    }
    if (d as usize) >= f.blocks.len() {
        return false;
    }
    if !cfg.reachable[d as usize] || !cfg.reachable[at as usize] {
        return false;
    }
    cfg.dominates(d, at)
}

/// Why a terminator is a flat region in the emitted C. These keys are the census
/// vocabulary; `4 unstructured conditional branch` is the pre-existing refusal
/// census key for the same shape, so a reader can line the two censuses up.
fn term_reason(t: &Terminator) -> String {
    match t {
        Terminator::Branch { .. } => "unstructured conditional branch".to_string(),
        Terminator::Return(_) => "return".to_string(),
        Terminator::TailReturn(_) => "tail call".to_string(),
        Terminator::NoReturn => "noreturn call".to_string(),
        Terminator::Exit => "exit".to_string(),
        Terminator::Unreachable => "unreachable".to_string(),
        Terminator::Switch { .. } => "switch".to_string(),
        Terminator::Jump(_) => "fall-through".to_string(),
    }
}

/// The blocks that branch to `b`.
fn preds(f: &Function, b: BbId) -> Vec<BbId> {
    let mut v = Vec::new();
    for (i, blk) in f.blocks.iter().enumerate() {
        let mut s = Vec::new();
        blk.term.successors(&mut s);
        if s.contains(&b) {
            v.push(i as BbId);
        }
    }
    v
}

/// A terminator's successors, as a vector.
fn successors_of(t: &Terminator) -> Vec<BbId> {
    let mut v = Vec::new();
    t.successors(&mut v);
    v
}

/// A block's phi pairs, read out of the `Op::Phi` instruction that carries them
/// (the IR keeps a phi as an instruction, not as a block field).
fn phi_pairs(b: &crate::ir::Block) -> Vec<(VId, BbId)> {
    b.instrs
        .iter()
        .find_map(|i| match &i.op {
            Op::Phi(pairs) => Some(pairs.to_vec()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Reverse postorder index from the entry, the order the emitted text runs.
fn rpo_index(f: &Function) -> Vec<usize> {
    let mut v = vec![usize::MAX; f.blocks.len()];
    let cfg = crate::cfg::Cfg::build(f);
    for (i, b) in cfg.rpo.iter().enumerate() {
        if (*b as usize) < f.blocks.len() {
            v[*b as usize] = i;
        }
    }
    v
}
