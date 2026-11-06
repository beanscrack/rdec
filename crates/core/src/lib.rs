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
pub mod tv;
pub mod ty;
pub mod verify;
pub mod width;

pub use ir::{BbId, Function, Op, Terminator, VId, NO_BB, NO_VAL};
pub use ty::{Sx, Ty, Width};

/// Result of running the whole middle-end.
#[derive(Clone, Debug, Default)]
pub struct PipelineReport {
    pub passes_run: Vec<String>,
    pub notes: Vec<String>,
    /// True when the round budget ran out with passes still changing - i.e. the
    /// pipeline STOPPED without converging. Stub for the red-first witness in
    /// `tests/pipeline_scaling.rs`: the field exists and is false here, which is
    /// exactly the bug the test is written to catch.
    pub not_converged: bool,
}

/// Run every middle-end pass to fixpoint. `flag_elim` must run before DCE so
/// dead flag defs disappear, and before structuring so branches are comparisons.
pub fn run_pipeline(f: &mut Function) -> PipelineReport {
    let mut rep = PipelineReport::default();
    let mut iter = 0;
    loop {
        iter += 1;
        let mut changed = false;
        changed |= passes::prune_phis(f);
        changed |= passes::flag_elimination(f);
        changed |= passes::const_fold(f);
        changed |= passes::copy_prop(f);
        changed |= passes::dead_code_elimination(f);
        changed |= passes::simplify_branches(f);
        changed |= passes::merge_blocks(f);
        changed |= passes::dedup_blocks(f);
        if !changed {
            rep.passes_run
                .push(format!("converged after {iter} rounds"));
            break;
        }
        if iter >= 32 {
            // Running out of budget is NOT convergence. `dedup_blocks` collapses one
            // duplicate per call, so a function with more duplicates than rounds gets
            // here with work outstanding - and this report is what the gates print.
            rep.not_converged = true;
            rep.passes_run.push(format!(
                "stopped: pass budget ({iter} rounds) exhausted, passes still changing"
            ));
            break;
        }
    }
    rep.notes.extend(passes::notes(f));
    rep
}
