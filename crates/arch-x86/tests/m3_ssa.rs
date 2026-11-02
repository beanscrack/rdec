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
use rdec_core::ir::{Function, Op};
use rdec_core::interp::{run_budget, Fault, Outcome};
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
