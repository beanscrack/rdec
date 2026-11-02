//! Control-flow graph analysis: reachability, dominators (Cooper-Harvey-Kennedy),
//! dominance frontiers, natural loops, RPO.

use crate::ir::{BbId, Block, Function, NO_BB};

#[derive(Clone, Debug, Default)]
pub struct Cfg {
    /// Indexed by BbId.
    pub succs: Vec<Vec<BbId>>,
    pub preds: Vec<Vec<BbId>>,
    /// Blocks reachable from entry, in reverse postorder.
    pub rpo: Vec<BbId>,
    pub reachable: Vec<bool>,
    /// Immediate dominator; `NO_BB` for entry and for unreachable blocks.
    pub idom: Vec<BbId>,
    /// idom children.
    pub dom_tree: Vec<Vec<BbId>>,
    /// Dominance frontier.
    pub df: Vec<Vec<BbId>>,
    pub postorder: Vec<BbId>,
}

impl Cfg {
    pub fn build(f: &Function) -> Cfg {
        let n = f.blocks.len();
        let mut c = Cfg {
            succs: vec![Vec::new(); n],
            preds: vec![Vec::new(); n],
            rpo: Vec::new(),
            reachable: vec![false; n],
            idom: vec![NO_BB; n],
            dom_tree: vec![Vec::new(); n],
            df: vec![Vec::new(); n],
            postorder: Vec::new(),
        };
        let mut succ = Vec::new();
        for b in &f.blocks {
            b.term.successors(&mut succ);
            for s in succ.iter().copied() {
                if (s as usize) < n && !c.succs[b.id as usize].contains(&s) {
                    c.succs[b.id as usize].push(s);
                    c.preds[s as usize].push(b.id);
                }
            }
        }
        c.dfs_postorder(f);
        c.compute_idom();
        c.build_dom_tree();
        c.compute_df();
        c
    }

    /// Iterative DFS to avoid blowing the native stack on big functions.
    fn dfs_postorder(&mut self, f: &Function) {
        let entry = f.entry as usize;
        if entry >= self.succs.len() {
            return;
        }
        // (node, next succ index)
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        let mut seen = vec![false; self.succs.len()];
        seen[entry] = true;
        self.reachable[entry] = true;
        while let Some((node, idx)) = stack.last_mut().copied() {
            if idx < self.succs[node].len() {
                stack.last_mut().unwrap().1 += 1;
                let s = self.succs[node][idx] as usize;
                if !seen[s] {
                    seen[s] = true;
                    self.reachable[s] = true;
                    stack.push((s, 0));
                }
            } else {
                stack.pop();
                self.postorder.push(node as BbId);
            }
        }
        self.rpo = self.postorder.iter().rev().copied().collect();
    }

    /// Cooper, Harvey, Kennedy 2001: "A Simple, Fast Dominance Algorithm".
    fn compute_idom(&mut self) {
        let n = self.succs.len();
        let mut rpo_num = vec![u32::MAX; n];
        for (i, b) in self.rpo.iter().enumerate() {
            rpo_num[*b as usize] = i as u32;
        }
        let inter = |a: BbId, b: BbId, idom: &[BbId], rpo_num: &[u32]| -> BbId {
            let mut f1 = a;
            let mut f2 = b;
            while f1 != f2 {
                while rpo_num[f1 as usize] > rpo_num[f2 as usize] {
                    f1 = idom[f1 as usize];
                }
                while rpo_num[f2 as usize] > rpo_num[f1 as usize] {
                    f2 = idom[f2 as usize];
                }
            }
            f1
        };
        let entry = self.rpo.first().copied();
        let Some(entry) = entry else { return };
        self.idom[entry as usize] = entry;
        let mut changed = true;
        let mut scratch = Vec::new();
        while changed {
            changed = false;
            for &b in &self.rpo {
                if b == entry {
                    continue;
                }
                // new_idom = first processed pred
                let mut new_idom = NO_BB;
                scratch.clear();
                scratch.extend(self.preds[b as usize].iter().copied());
                for p in scratch.iter().copied() {
                    if self.reachable[p as usize] && self.idom[p as usize] != NO_BB {
                        new_idom = if new_idom == NO_BB {
                            p
                        } else {
                            inter(p, new_idom, &self.idom, &rpo_num)
                        };
                    }
                }
                if new_idom != NO_BB && self.idom[b as usize] != new_idom {
                    self.idom[b as usize] = new_idom;
                    changed = true;
                }
            }
        }
    }

    fn build_dom_tree(&mut self) {
        for b in 0..self.succs.len() {
            let d = self.idom[b];
            if d != NO_BB && d as usize != b && self.reachable[b] {
                self.dom_tree[d as usize].push(b as BbId);
            }
        }
    }

    /// Standard construction (Cytron et al.): for every join node b with >= 2
    /// predecessors, walk each predecessor up the dominator tree to idom(b),
    /// adding b to the DF of every node on that walk.
    fn compute_df(&mut self) {
        for b in 0..self.succs.len() {
            let b = b as BbId;
            if !self.is_reachable(b) || self.idom[b as usize] == NO_BB {
                continue;
            }
            if self.preds[b as usize].len() < 2 {
                continue;
            }
            let stop = self.idom[b as usize];
            for i in 0..self.preds[b as usize].len() {
                let p = self.preds[b as usize][i];
                let mut runner = p;
                loop {
                    if !self.df[runner as usize].contains(&b) {
                        self.df[runner as usize].push(b);
                    }
                    if runner == stop {
                        break;
                    }
                    let d = self.idom[runner as usize];
                    if d == NO_BB {
                        break;
                    }
                    runner = d;
                }
            }
        }
    }

    pub fn dominates(&self, a: BbId, b: BbId) -> bool {
        if self.idom[a as usize] == NO_BB {
            return false;
        }
        let mut x = b as usize;
        loop {
            if x == a as usize {
                return true;
            }
            let d = self.idom[x];
            if d == NO_BB || d as usize == x {
                return false;
            }
            x = d as usize;
        }
    }

    pub fn is_reachable(&self, b: BbId) -> bool {
        self.reachable.get(b as usize).copied().unwrap_or(false)
    }

    pub fn unreachable(&self) -> Vec<BbId> {
        (0..self.reachable.len() as BbId)
            .filter(|b| !self.reachable[*b as usize])
            .collect()
    }

    /// Natural loops found from back edges (head dominates tail).
    pub fn loops(&self, f: &Function) -> Vec<Loop> {
        let mut loops: Vec<Loop> = Vec::new();
        for &h in &self.rpo {
            for &t in &self.preds[h as usize] {
                if self.dominates(h, t) {
                    let mut body: Vec<BbId> = vec![h];
                    let mut work = vec![t];
                    while let Some(x) = work.pop() {
                        if body.contains(&x) {
                            continue;
                        }
                        body.push(x);
                        for &p in &self.preds[x as usize] {
                            if !body.contains(&p) {
                                work.push(p);
                            }
                        }
                    }
                    body.sort();
                    let back_edges: Vec<(BbId, BbId)> = self.preds[h as usize]
                        .iter()
                        .filter(|p| self.dominates(h, **p))
                        .map(|p| (*p, h))
                        .collect();
                    let header_entry = self.preds[h as usize]
                        .iter()
                        .copied()
                        .filter(|p| !self.dominates(h, *p))
                        .collect();
                    loops.push(Loop {
                        header: h,
                        header_addr: f.block(h).addr,
                        body,
                        back_edges,
                        entry_blocks: header_entry,
                        parent: usize::MAX,
                    });
                }
            }
        }
        // Nesting: a loop's parent is the smallest other loop containing its header.
        for i in 0..loops.len() {
            let h = loops[i].header;
            let mut best: Option<(usize, usize)> = None;
            for (j, lp) in loops.iter().enumerate() {
                if i == j {
                    continue;
                }
                if lp.body.contains(&h) {
                    let sz = lp.body.len();
                    if best.map(|(bs, _)| sz < bs).unwrap_or(true) {
                        best = Some((sz, j));
                    }
                }
            }
            if let Some((_, j)) = best {
                loops[i].parent = j;
            }
        }
        loops
    }

    pub fn to_dot(&self, f: &Function) -> String {
        let mut s = String::new();
        s.push_str("digraph cfg {\n  rankdir=TB;\n  node [shape=box, fontname=\"Consolas\", fontsize=10];\n");
        for b in &f.blocks {
            let mut body = format!("bb{} @ {:#x}\n", b.id, b.addr);
            for i in &b.instrs {
                body.push_str(&format!("  {}\n", crate::ir::format_instr(f, i)));
            }
            body.push_str(&format!("  {}", crate::ir::format_term(f, &b.term)));
            s.push_str(&format!(
                "  bb{} [label=\"{}\"]; \n",
                b.id,
                dot_escape(&body)
            ));
            let mut succ = Vec::new();
            b.term.successors(&mut succ);
            for t in succ {
                let label = match b.term {
                    crate::ir::Terminator::Branch { then_bb, .. } if then_bb == t => "T",
                    crate::ir::Terminator::Branch { .. } => "F",
                    _ => "",
                };
                s.push_str(&format!("  bb{} -> bb{} [label=\"{label}\"];\n", b.id, t));
            }
        }
        s.push_str("}\n");
        s
    }
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[derive(Clone, Debug)]
pub struct Loop {
    pub header: BbId,
    pub header_addr: u64,
    pub body: Vec<BbId>,
    /// (tail, head) edges.
    pub back_edges: Vec<(BbId, BbId)>,
    /// Predecessors of the header outside the loop.
    pub entry_blocks: Vec<BbId>,
    /// Index of the enclosing loop in the same Vec, NO_BB for outermost.
    pub parent: usize,
}

impl Loop {
    pub fn in_body(&self, b: BbId) -> bool {
        self.body.contains(&b)
    }

    /// Blocks in the loop body that have a predecessor outside the loop.
    pub fn exits_to(&self, cfg: &Cfg) -> Vec<(BbId, BbId)> {
        let mut out = Vec::new();
        for &b in &self.body {
            for &s in &cfg.succs[b as usize] {
                if !self.in_body(s) {
                    out.push((b, s));
                }
            }
        }
        out
    }
}

pub fn block_list(f: &Function) -> Vec<&Block> {
    f.blocks.iter().collect()
}
