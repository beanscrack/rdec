// M3: does lifting a *join* and a *loop* produce valid SSA?
//
// The corpus gate (`rdec lift --corpus`) covers 141 real functions, and CI does not
// run it. These three functions are hand-assembled so the check is hermetic, and
// each encodes one SSA question:
//
//   join   - two paths write eax and merge into one block that returns it: the
//            merge needs a phi, and a phi at a 2-predecessor block is exactly what
//            the verifier's dominance and phi/edge rules check.
//   loop   - a header with an entry edge and a back edge: the loop-carried value is
//            defined in the latch, which the *edge-availability* rule (not ordinary
//            dominance) is what makes legal.
//   straight - no branch at all: a phi here would mean the pass is inserting them
//            where nothing merges.
//
// The interpreter run is the semantic half: it consumes every operand strictly, so a
// dangling rename or an unrewritten register placeholder faults as MissingOperand.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::interp::{run_budget, Fault, Outcome, Val};
use rdec_core::ir::{Function, Op};
use rdec_core::verify::verify_ssa;

fn lift_one_fn(bytes: &[u8]) -> Function {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    // Discovery is seed-driven: in the corpus, seeds come from symbol-table
    // function symbols and REL32 call relocations. Hand-assembled bytes have no
    // symbol table, so seed the entry address ourselves.
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(
        plan.functions.len(),
        1,
        "expected exactly one discovered function"
    );
    // `Plan::functions` is a list of start addresses; a function's extent runs to
    // the next start, which is how the CLI lifts it too.
    let start = plan.functions[0];
    let end = owned.len() as u64;
    let (f, _rep) = lift_one("f", &owned, 0, start, end, &plan, &Default::default());
    f
}

fn phi_count(f: &Function) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i.op, Op::Phi(_)))
        .count()
}

fn assert_clean(f: &Function, what: &str) {
    if let Err(errs) = verify_ssa(f) {
        panic!("{what}: verifier rejected the lifted function: {errs:?}");
    }
    let ex = run_budget(f, &[], 20000);
    if let Outcome::Fault(Fault::MissingOperand(v)) = ex.outcome {
        panic!("{what}: the interpreter uses %{v}, which nothing defines");
    }
    if matches!(ex.outcome, Outcome::Budget) {
        panic!("{what}: the interpreter never terminated - a broken back edge?");
    }
}

/// if (ecx == 0) eax = 2; else eax = 1; ... both merge into a block that rets.
const JOIN: &[u8] = &[
    0xb8, 0x01, 0x00, 0x00, 0x00, //  0: mov eax, 1
    0x85, 0xc9, //  5: test ecx, ecx
    0x74, 0x0b, //  7: je 0x14
    0xb8, 0x02, 0x00, 0x00, 0x00, //  9: mov eax, 2
    0xeb, 0x07, // 14: jmp 0x17
    0x90, 0x90, 0x90, 0x90, // 16: alignment padding
    0xb8, 0x03, 0x00, 0x00, 0x00, // 20: mov eax, 3   (the `je` target)
    0xeb, 0x03, // 25: jmp 0x1e
    0x90, 0x90, 0x90, // 27: alignment padding
    0xc3, // 30: ret
];

/// eax = 0; while (ecx > 0) { eax -= 1; ecx -= 1; } ret
const LOOP: &[u8] = &[
    0xb8, 0x00, 0x00, 0x00, 0x00, //  0: mov eax, 0
    0x83, 0xf9, 0x00, //  5: cmp ecx, 0
    0x7e, 0x0a, //  8: jle 0x14
    0x83, 0xe8, 0x01, // 10: sub eax, 1
    0x83, 0xe9, 0x01, // 13: sub ecx, 1
    0xeb, 0xf3, // 16: jmp 0x5   (back edge)
    0x90, 0x90, // 18: alignment padding
    0xc3, // 20: ret
];

/// ret, with nothing to merge.
const STRAIGHT: &[u8] = &[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]; // mov eax, 42; ret

#[test]
fn a_machine_code_join_lifts_into_valid_ssa() {
    let f = lift_one_fn(JOIN);
    assert!(
        phi_count(&f) >= 1,
        "a join that merges eax into the return needs a phi; the IR has none"
    );
    assert_clean(&f, "join");
}

#[test]
fn a_machine_code_loop_lifts_into_valid_ssa() {
    let f = lift_one_fn(LOOP);
    if phi_count(&f) == 0 {
        panic!(
            "a loop header needs a phi for its loop-carried registers; the IR has none:\n{}",
            rdec_core::fmt_ir::format_function(&f)
        );
    }
    assert_clean(&f, "loop");
}

#[test]
fn straight_line_code_gets_no_phi() {
    let f = lift_one_fn(STRAIGHT);
    assert_eq!(
        phi_count(&f),
        0,
        "phi insertion fired where no control flow merges"
    );
    assert_clean(&f, "straight line");
}

/// THE CI DEFECT CLASS, reproduced on hand-assembled bytes - no clang, no
/// corpus, no compiler version in the loop.
///
/// CI run 42544451810 (commit c2123ee, clang 20.1.8) printed:
///
/// ```text
/// pass-broke-ssa record: corpus\obj\locals_addressed.O2.obj :: f: after pass 1
/// 'flag_elimination': %50 defined in bb2 is not available on the edge
/// bb6 -> bb7; %52 defined in bb2 is not available on the edge bb6 -> bb7
/// ```
///
/// `flag_elimination` is the FIRST pass in `PRE_EMIT_PASSES`, so the per-pass
/// verifier's first run sees the lifter's IR with one pass applied. That pass
/// has no CFG surgery to its name (`fuse.rs` writes `.op` fields and inserts
/// instructions within one block - no `.term`, no `.blocks`, no phi writes),
/// and the verifier's edge-availability verdict for an existing value depends
/// only on def placement, phi pairs, the CFG, and dominance - all outside that
/// pass's mutation set. The break is therefore in the pass's INPUT: the
/// lifter's phi-placement gate was "read at the head of this block, or at
/// the head of a SUCCESSOR" - a ONE-step approximation of liveness.
///
/// The shape (ids as lifted, 9 blocks): bb1 (entry fall) defines Rax/RSI
/// (`mov rax,1; mov rsi,2`) and jumps to the join bb5@34 (a nop); bb2 ->
/// bb3 -> bb4 (ecx == 0, then eax == 0) and bb2 -> bb4 reach bb5 WITHOUT
/// bb1, so bb1 does not dominate bb5. The value is read only TWO hops down
/// (bb5 -> bb6, bb6 -> bb8 `add rax,rsi`), so the one-step gate places NO
/// phi at bb5, `out[bb5] = out[preds[0]] = out[bb1]`, and bb6 (preds
/// {bb5, bb7}, a bare jmp) DOES get a phi (its successor bb8 head-reads) -
/// whose pair keyed on bb5 carries bb1's defs across the edge bb5 -> bb6,
/// which bb5's def block does not dominate. Pre-fix (measured on base
/// c2123ee, raw lift - the CI message verbatim in form):
///
/// ```text
/// %13 defined in bb1 is not available on the edge bb5 -> bb6;
/// %14 defined in bb1 is not available on the edge bb5 -> bb6
/// ```
///
/// Post-fix, placement runs on FULL liveness: bb5 gets its phi (keyed bb1 ->
/// bb1's defs; keyed bb3/bb4 -> the entry `undef` those paths genuinely
/// see), every pair is available on its edge, and the answers become the
/// honest machine values. The interpreter IS the IR's ground truth;
/// `Undef` reads as the fixed 0 (`interp.rs`, documented convention):
/// (ecx, edx, eax) = (1, ., .) takes bb1 and answers 1 + 2 = 3; (0, 0, .)
/// takes the `add` block bb7 from unseeded RAX/RSI and answers
/// (0+1) + (0+1) = 2; (0, 1, .) takes bb2 -> bb3 -> bb5, which never touches
/// RAX/RSI, and answers 0. Pre-fix the (0, 1, 0) vector WRONGLY answers 3 -
/// the pair forwards bb1's defs across the bypass - so the answer
/// assertions below are part of the proof, not decoration; and the mask
/// pins (every pass verified) mean no "fix" that merely stops running the
/// passes can pass either.
const FLOW_THROUGH_JOIN: &[u8] = &[
    0x85, 0xC9, //  0: test ecx,ecx  (bb0)
    0x74, 0x10, //  2: je 20  (ecx == 0 -> bb2: the bypass)
    0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, //  4: mov rax,1  (bb1: the def)
    0x48, 0xC7, 0xC6, 0x02, 0x00, 0x00, 0x00, // 11: mov rsi,2
    0xEB, 0x0E, // 18: jmp 34 (bb1 -> bb5: bb1's edge into the join)
    0x85, 0xD2, // 20: test edx,edx (bb2)
    0x74, 0x10, // 22: je 40  (edx == 0 -> bb7: the `add` block)
    0x85, 0xC0, // 24: test eax,eax (bb3: Rax unseeded = dynamic)
    0x74, 0x06, // 26: je 34  (Rax == 0 -> bb5: the edge that BYPASSES bb1)
    0xEB, 0x04, // 28: jmp 34 (bb4 -> bb5: the second bypass edge)
    0xCC, 0xCC, 0xCC, 0xCC, // 30: padding, never decoded (no leader)
    0x90, // 34: nop (bb5: preds {bb1, bb3, bb4}; phi-less pre-fix: read 2 hops down)
    0xEB, 0x00, // 35: jmp 37 (bb5 -> bb6)
    0xEB, 0x0A, // 37: jmp 49 (bb6: EMPTY, preds {bb5, bb7}, gets the phi)
    0x48, 0x83, 0xC0, 0x01, // 39: add rax,1  (bb7, from the unseeded undef)
    0x48, 0x83, 0xC6, 0x01, // 43: add rsi,1
    0xEB, 0xF4, // 47: jmp 37 (bb7 -> bb6)
    0x48, 0x01, 0xF0, // 49: add rax,rsi (bb8: head-reads BOTH groups)
    0xC3, // 52: ret
];

#[test]
fn liveness_phi_placement_keeps_phi_pairs_available() {
    let f = lift_one_fn(FLOW_THROUGH_JOIN);
    assert_eq!(f.blocks.len(), 9, "the hand-built CFG has 9 blocks");
    // The raw lift must VERIFY: this is the leg that is red on base c2123ee
    // with exactly the CI class of error.
    if let Err(errs) = verify_ssa(&f) {
        panic!(
            "liveness placement: the raw lift breaks phi-pair availability: {}",
            errs.join("; ")
        );
    }
    // bb5 AND bb6 each carry a phi for both groups (Rax, Rsi).
    assert_eq!(
        phi_count(&f),
        4,
        "liveness placement must phi the flow-through join too:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );

    // Feed it through the real pipeline: every pass verified, no pass named.
    let mut g = f.clone();
    let iso = rdec_core::passes::clean_isolated(&mut g);
    let all = (1u32 << rdec_core::passes::PRE_EMIT_PASSES.len()) - 1;
    assert_eq!(
        iso.verified,
        all,
        "the isolation driver verified {} of {} passes (bitmask {:#b})",
        iso.verified.count_ones(),
        rdec_core::passes::PRE_EMIT_PASSES.len(),
        iso.verified
    );
    if let Some(b) = &iso.broke {
        panic!(
            "pass-broke-ssa record: fixture :: f: after pass {} '{}': {}",
            b.index,
            b.name,
            b.errors.join("; ")
        );
    }
    // Same program: the flag ops the pipeline must eliminate are gone...
    for blk in &g.blocks {
        for i in &blk.instrs {
            assert!(
                !matches!(i.op, Op::Flags(_) | Op::GetFlag(..)),
                "flags survived the pipeline: {}",
                rdec_core::fmt_ir::format_instr(&g, i)
            );
        }
    }
    // ...and the values every path must keep are kept (the table in the doc
    // comment above; the pre-fix IR answers (0, 1, 0) = 3, which is the bug,
    // not the truth).
    for (ecx, edx, want) in [(1u64, 1, 3), (0, 0, 2), (0, 1, 0)] {
        match run_budget(&g, &[ecx, edx], 20000).outcome {
            Outcome::Ret(Val::Int(v)) => {
                assert_eq!(v, want, "ecx={ecx} edx={edx}")
            }
            other => panic!("ecx={ecx} edx={edx}: must answer {want}, got {other:?}"),
        }
    }
}
