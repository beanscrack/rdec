//! Translation validation (TV): the pass oracle.
//!
//! A pass is a function on IR. TV's claim is that it preserves the function's
//! observable behavior, evaluated on sampled states:
//!
//! ```text
//!   for each state s:  run(f, s) == run(pass(f), s)
//! ```
//!
//! "==", per docs/ir-semantics.md, is `Exec::diff`: outcome, the ordered effect
//! trace (calls, writes, undefined-flag reads, exits, opaques) and final memory.
//! A difference names the pass, the state and the difference, which is the whole
//! product of this module — a failing TV run has to be a pointer into a function,
//! not a green/red light.
//!
//! This is the oracle §9 shows was missing: 1.1 kLOC of passes were written with
//! nothing able to see them, and four confirmed bugs produced wrong logic while
//! every check passed.
//!
//! Sound by construction, not by inspection:
//! - The interpreter is the reference, and it shares no code with the passes.
//! - A pass that panics is a failure, not a skip.
//! - `Budget` on one side only is a difference.
//! - Sampling is a fixed seed plus a boundary grid, so a failure reproduces.

use crate::interp::{self, Exec, DEFAULT_BUDGET};
use crate::ir::{Function, Op};

/// One sampled input vector.
#[derive(Clone, Debug)]
pub struct State {
    pub inputs: Vec<u64>,
}

/// Deterministic PRNG (xorshift64*), so a TV failure replays.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

/// Boundary values that catch width/signedness bugs, then random words.
pub fn states(n_params: usize, k: usize, seed: u64) -> Vec<State> {
    let boundary: [u64; 8] = [
        0,
        1,
        u64::MAX,
        0x8000_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000,
        0x7fff_ffff,
        0xff,
    ];
    let n = n_params.max(1);
    let mut out = Vec::with_capacity(k);
    let mut rng = Rng::new(seed);
    // Grid over boundaries for the first states: every pair for 2 params, and a
    // rotating pattern beyond that.
    let bi = boundary.len();
    for i in 0..k {
        let inputs = if i < bi * bi {
            // Digit-expansion over the boundary values: every pair for two
            // params, no repeated rows.
            (0..n)
                .map(|j| {
                    let mut idx = i;
                    for _ in 0..j {
                        idx /= bi;
                    }
                    boundary[idx % bi]
                })
                .collect::<Vec<_>>()
        } else {
            (0..n).map(|_| rng.next_u64()).collect::<Vec<_>>()
        };
        out.push(State { inputs });
    }
    out
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub pass: String,
    pub state_index: usize,
    pub inputs: Vec<u64>,
    pub diffs: Vec<String>,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pass `{}` changed behavior on state #{} over inputs {:?}:\n  {}",
            self.pass,
            self.state_index + 1,
            self.inputs,
            self.diffs.join("\n  ")
        )
    }
}

/// Compare `before` and `after` across `ss`. Returns every failure found (up to
/// `max_report`), so one run shows a pass's whole failure class, not one case.
pub fn compare(
    before: &Function,
    after: &Function,
    pass: &str,
    ss: &[State],
    budget: usize,
    max_report: usize,
) -> Vec<Failure> {
    let mut out = Vec::new();
    for (i, s) in ss.iter().enumerate() {
        let a = interp::run_budget(before, &s.inputs, budget);
        let b = interp::run_budget(after, &s.inputs, budget);
        let diffs = a.diff(&b);
        if !diffs.is_empty() {
            out.push(Failure {
                pass: pass.to_string(),
                state_index: i,
                inputs: s.inputs.clone(),
                diffs,
            });
            if out.len() >= max_report {
                break;
            }
        }
    }
    out
}

/// A named pass.
pub type Pass = (&'static str, fn(&mut Function) -> bool);

/// Run `pass` on `f`, validating it. On failure, `f` is left transformed (the
/// caller decides whether to adopt it) and the failures are returned.
pub fn run_validated(
    f: &mut Function,
    pass: Pass,
    ss: &[State],
    budget: usize,
) -> Result<bool, Vec<Failure>> {
    let before = f.clone();
    let changed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pass.1(f)));
    let changed = match changed {
        Ok(c) => c,
        Err(_) => {
            return Err(vec![Failure {
                pass: pass.0.to_string(),
                state_index: usize::MAX,
                inputs: vec![],
                diffs: vec!["pass panicked".to_string()],
            }])
        }
    };
    if !changed {
        return Ok(false);
    }
    let fails = compare(&before, f, pass.0, ss, budget, 3);
    if fails.is_empty() {
        Ok(true)
    } else {
        Err(fails)
    }
}

/// Run the whole pipeline with every pass validated. Stops at the first pass that
/// breaks behavior: a pipeline that keeps going after a miscompile produces an
/// IR whose provenance is unknowable.
pub fn run_pipeline_validated(
    f: &mut Function,
    passes: &[Pass],
    k: usize,
    seed: u64,
    budget: usize,
) -> Result<Vec<String>, Failure> {
    let ss = states(f.params.len(), k, seed);
    let mut ran = Vec::new();
    for &p in passes {
        match run_validated(f, p, &ss, budget) {
            Ok(_) => ran.push(p.0.to_string()),
            Err(mut fails) => return Err(fails.remove(0)),
        }
    }
    Ok(ran)
}

/// A green TV run says nothing unless it can go red. This is the roadmap's
/// example: a DCE that drops one live instruction. It must be caught, by name.
pub fn sabotage_dce(f: &mut Function) -> bool {
    // Find the first instruction whose result is live (used by an instruction or
    // by the terminator) and is not already trivially required, then delete it.
    for bi in 0..f.blocks.len() {
        let n = f.blocks[bi].instrs.len();
        for ii in 0..n {
            let dest = f.blocks[bi].instrs[ii].dest;
            let op = f.blocks[bi].instrs[ii].op.clone();
            // Bindings and constants are not what the gate is about: delete a
            // real computation, which is the mutation that matters.
            if matches!(op, Op::Param | Op::Const(_) | Op::Undef | Op::ConstF(_)) {
                continue;
            }
            let used = live_use(f, dest);
            if used {
                f.blocks[bi].instrs.remove(ii);
                return true;
            }
        }
    }
    false
}

/// Liveness that includes terminator operands — the exact thing `uses_of` gets
/// wrong in B1, computed independently here.
fn live_use(f: &Function, v: u32) -> bool {
    for b in &f.blocks {
        for ins in &b.instrs {
            let mut u = Vec::new();
            ins.op.uses(&mut u);
            if u.contains(&v) {
                return true;
            }
        }
        let mut t = Vec::new();
        match &b.term {
            crate::ir::Terminator::Return(x) => t.push(*x),
            crate::ir::Terminator::Branch { cond, .. } => t.push(*cond),
            crate::ir::Terminator::Switch { index, .. } => t.push(*index),
            _ => {}
        }
        if t.contains(&v) {
            return true;
        }
    }
    false
}

/// Convenience: run one pass unvalidated, for fixtures that only need the IR.
pub fn run(f: &Function, inputs: &[u64]) -> Exec {
    interp::run_budget(f, inputs, DEFAULT_BUDGET)
}
