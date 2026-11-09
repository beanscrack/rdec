//! RDec core: a target-agnostic decompiler middle-end.
//!
//! Pipeline, in the order the CLI runs it:
//!
//! ```text
//!   machine code
//!     -> [front]  disassembly (iced)
//!     -> [front]  per-instruction lift to IR, versioned phi  =  SSAn form
//!     -> [core]   phi pruning + degenerate-copy removal
//!     -> [core]   flag elimination (GetFlag trees -> IntCmp), constant folding,
//!                 copy propagation, DCE, dead-store elimination
//!     -> [core]   CFG structuring (patterns: if/then/else, while, do-while, goto)
//!     -> [back]   C emission with expression inlining + precedence
//! ```

pub mod cfg;
pub mod emit_c;
pub mod fmt_ir;
pub mod fuse;
pub mod interp;
pub mod ir;
pub mod passes;
pub mod structure;
pub mod tv;
pub mod ty;
pub mod tyinfer;
pub mod verify;
pub mod width;

pub use ir::{BbId, Function, Op, Terminator, VId, NO_BB, NO_VAL};
pub use ty::{Sx, Ty, Width};

/// Result of running the whole middle-end.
#[derive(Clone, Debug, Default)]
pub struct PipelineReport {
    pub passes_run: Vec<String>,
    pub notes: Vec<String>,
    /// True when the pipeline STOPPED with work outstanding: the outer round
    /// budget was exhausted while passes were still changing, or a pass was still
    /// changing after its internal fixpoint bound (`passes_run` names which one).
    ///
    /// It is NOT a function of size. A 250-block chain and a 20k-block chain both
    /// converge and report `false`; a tiny function with a pass that oscillates
    /// reports `true`. Reporting convergence when the pipeline stopped is the same
    /// class of bug as silently dropping an instruction, so the only way to clear
    /// this field is to actually run out of things to change.
    pub not_converged: bool,
    /// Outer rounds used (the number in the `converged after N rounds` line).
    pub rounds: usize,
    /// Pass invocations: one per outer-round call PLUS the iterations each pass's
    /// internal fixpoint spent. Added with the per-pass fixpoints, because "how
    /// many times did the middle end work" is the question the round count used to
    /// pretend to answer - it is the number that scales with the function's
    /// structure, and `tests/pipeline_scaling.rs` prints it.
    pub pass_invocations: usize,
    /// Per-pass SSA verification, from the same driver the callers run:
    /// `verified` is the set of passes that ran and were verified, `broke`
    /// names the FIRST pass that left invalid SSA. A pass that breaks SSA is a
    /// bug in that pass, and this driver stops there (`passes_run` carries the
    /// record and `not_converged` is true) so no later pass launders the
    /// evidence.
    pub iso: passes::Isolation,
}

/// The production outer budget. It is a liveness guard for pass interaction, not a
/// work allowance: no pass needs a round per block (see `passes::fixpoint_bound`),
/// so a chain of any length converges inside it.
pub const DEFAULT_ROUNDS: usize = 32;

/// Run every middle-end pass to fixpoint. `flag_elim` must run before DCE so
/// dead flag defs disappear, and before structuring so branches are comparisons.
pub fn run_pipeline(f: &mut Function) -> PipelineReport {
    run_pipeline_with_budget(f, DEFAULT_ROUNDS)
}

/// The pipeline with an explicit outer round budget.
///
/// The budget is a parameter so the reporting claim stays testable at a size a
/// test can build: after the fix, an ordinary function cannot exhaust the
/// production budget, and "running out of budget is not convergence" must be
/// provable without a 10k-block fixture.
pub fn run_pipeline_with_budget(f: &mut Function, max_rounds: usize) -> PipelineReport {
    let mut rep = PipelineReport::default();
    let mut iter = 0;
    // Each pass fixpoints internally; the driver re-calls it under the same bound
    // as a net, so a pass that returns "changed" at its own bound keeps making
    // progress and the number that gets REPORTED as exhausted is this one.
    let bound = passes::fixpoint_bound(f);
    loop {
        iter += 1;
        let mut changed = false;
        let mut stopped: Option<String> = None;
        macro_rules! run {
            ($name:expr, $call:expr) => {{
                let mut ch;
                let mut any = false;
                let mut iters = 0;
                loop {
                    iters += 1;
                    rep.pass_invocations += 1;
                    ch = $call;
                    // The loop's last call is the one that reports "no change", so
                    // what the round must see is whether ANY call changed.
                    any |= ch;
                    if !ch {
                        break;
                    }
                    if iters >= bound {
                        stopped = Some(format!(
                            "stopped: pass fixpoint bound ({bound} iterations) exhausted in {}",
                            $name
                        ));
                        break;
                    }
                }
                any
            }};
        }
        // SSA after EVERY pass, on this driver's own pass list: a pass that
        // breaks SSA is a bug in that pass, and the record names it (index in
        // this driver's order, name from `passes::NAME_*`). A breakage is
        // judged by the END of the pipeline (the round-boundary contract
        // `clean_isolated` documents: a pass whose breakage a later pass
        // repairs is a transient, and this driver must not change the IR the
        // m0/tv/scaling tests measure); a breakage that survives to the end
        // lands in `passes_run` naming its pass, with `not_converged` true.
        macro_rules! iso {
            ($idx:expr, $name:expr) => {{
                rep.iso.verified |= 1u32 << $idx;
                if rep.iso.broke.is_none() {
                    if let Err(errs) = crate::verify::verify_ssa(f) {
                        rep.iso.broke = Some(passes::PassBreak {
                            index: $idx + 1,
                            name: $name,
                            pipeline: "run_pipeline",
                            errors: errs,
                            resume: usize::MAX,
                        });
                    }
                }
            }};
        }
        changed |= run!(passes::NAME_PRUNE_PHIS, passes::prune_phis(f));
        iso!(0, passes::NAME_PRUNE_PHIS);
        changed |= run!(passes::NAME_FLAG_ELIMINATION, passes::flag_elimination(f));
        iso!(1, passes::NAME_FLAG_ELIMINATION);
        changed |= run!(passes::NAME_CONST_FOLD, passes::const_fold(f));
        iso!(2, passes::NAME_CONST_FOLD);
        changed |= run!(passes::NAME_COPY_PROP, passes::copy_prop(f));
        iso!(3, passes::NAME_COPY_PROP);
        changed |= run!(
            passes::NAME_DEAD_CODE_ELIM,
            passes::dead_code_elimination(f)
        );
        iso!(4, passes::NAME_DEAD_CODE_ELIM);
        changed |= run!(passes::NAME_SIMPLIFY_BRANCHES, passes::simplify_branches(f));
        iso!(5, passes::NAME_SIMPLIFY_BRANCHES);
        changed |= run!(passes::NAME_MERGE_BLOCKS, {
            // The counted form: its collapse sweeps are fixpoint iterations too, so
            // they belong in the invocation count.
            let (ch, sweeps, _bound_hit) = passes::merge_blocks_counted(f, bound);
            rep.pass_invocations += sweeps;
            ch
        });
        iso!(6, passes::NAME_MERGE_BLOCKS);
        changed |= run!(passes::NAME_DEDUP_BLOCKS, passes::dedup_blocks(f));
        iso!(7, passes::NAME_DEDUP_BLOCKS);
        if let Some(why) = stopped {
            rep.not_converged = true;
            rep.rounds = iter;
            rep.passes_run.push(why);
            break;
        }
        if !changed {
            rep.rounds = iter;
            rep.passes_run
                .push(format!("converged after {iter} rounds"));
            break;
        }
        if iter >= max_rounds {
            // Running out of budget is NOT convergence. The report is what the
            // gates print, and a function with more work than rounds used to be
            // announced as "converged after 33 rounds".
            rep.not_converged = true;
            rep.rounds = iter;
            rep.passes_run.push(format!(
                "stopped: pass budget ({iter} rounds) exhausted, passes still changing"
            ));
            break;
        }
    }
    if let Some(b) = rep.iso.broke.take() {
        // The first failing pass is judged by the END of the pipeline (the
        // round-boundary contract, as in `passes::clean_isolated`): a
        // breakage a later pass repaired is a transient and stays silent; a
        // breakage that survives names its pass in words a gate can grep.
        if let Err(errs) = crate::verify::verify_ssa(f) {
            let line = format!(
                "pass-broke-ssa: after pass {} '{}' in the {} pipeline: {}",
                b.index,
                b.name,
                b.pipeline,
                errs.join("; ")
            );
            rep.not_converged = true;
            rep.rounds = iter;
            rep.passes_run.push(line);
            rep.iso.broke = Some(passes::PassBreak {
                index: b.index,
                name: b.name,
                pipeline: b.pipeline,
                errors: errs,
                resume: b.resume,
            });
        }
    }
    rep.notes.extend(passes::notes(f));
    rep
}
