// The registers the ABI and the frame model hand to a function, pinned on
// hand-assembled machine code.
//
// A register is one unsigned W64 SSA value, and a block's OUTGOING register map
// is the transfer function SSA construction solves. Two facts follow, and both
// used to be missing:
//
//   * The ENTRY block defines RCX/RDX/R8/R9 (the four integer arguments) and
//     RSP (the modelled stack base). They are seeded, not written by an
//     instruction, so nothing marked them - and an unmarked register is
//     invisible to the transfer function: every block after the entry read the
//     argument as `undef`. `mov eax, edx` two instructions into a function is
//     the source's `return b`, and the decompiler had no value for it.
//   * A call's RAX and its clobbered caller-saved registers are definitions
//     too. Unmarked, a successor of a call read the *pre-call* value of a
//     register the call had destroyed - a silent miscompile, the kind that
//     ships.
//
// And the SSA rule they expose: an incoming value that IS the phi is a real
// fact (the block does not write the register, so it carries its own value),
// not a gap. `passes::prune_phis` collapses `phi(x, phi)` to `copy x`, which is
// the loop-invariant register the machine code has. Inventing `undef` there is
// what made every loop that reads an argument "return RAX unwritten".
//
// Each test below pins the fact AND the counterfactual, so a regression cannot
// quietly reappear as "44 functions" again.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::interp::{run_budget, Outcome, Val};
use rdec_core::ir::{Function, Instr, Op, Terminator};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

fn lift(bytes: &[u8]) -> (Function, rdec_arch_x86::lift::Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 1, "one discovered function");
    let start = plan.functions[0];
    let end = owned.len() as u64;
    lift_one("f", &owned, 0, start, end, &plan, &Default::default())
}

/// The pre-emission pipeline, as `rdec c` runs it.
fn clean(f: &mut Function) {
    for _ in 0..16 {
        let mut ch = false;
        ch |= rdec_core::fuse::flag_elimination(f);
        ch |= rdec_core::passes::prune_phis(f);
        ch |= rdec_core::passes::const_fold(f);
        ch |= rdec_core::passes::copy_prop(f);
        ch |= rdec_core::passes::simplify_branches(f);
        ch |= rdec_core::passes::dead_store_elimination(f);
        ch |= rdec_core::passes::dead_code_elimination(f);
        if !ch {
            break;
        }
    }
}

fn answer(f: &Function, inputs: &[u64]) -> Outcome {
    run_budget(f, inputs, 20000).outcome
}

fn opaque_texts(f: &Function) -> Vec<String> {
    let mut v: Vec<String> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| match &i.op {
            Op::Opaque(t) => Some(t.to_string()),
            _ => None,
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Every value that flows into `root`, transitively, following only phi edges
/// whose predecessor a path from the entry reaches (T2's own reachability
/// refinement, so these tests measure the gap the gate would see).
fn ancestors(f: &Function, root: u32) -> Vec<u32> {
    let mut reach = std::collections::HashSet::new();
    let mut stack = vec![f.entry];
    let mut succ = Vec::new();
    while let Some(b) = stack.pop() {
        if !reach.insert(b) {
            continue;
        }
        f.block(b).term.successors(&mut succ);
        stack.extend(succ.iter().copied());
    }
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        let Some((bb, ix)) = f.def(v) else { continue };
        let op = f.block(bb).instrs[ix].op.clone();
        match op {
            Op::Phi(pairs) => stack.extend(
                pairs
                    .iter()
                    .filter(|(_, p)| reach.contains(p))
                    .map(|(v, _)| *v),
            ),
            _ => {
                let mut u = Vec::new();
                op.uses(&mut u);
                stack.extend(u);
            }
        }
    }
    seen.into_iter().collect()
}

fn reads_undef(f: &Function, root: u32) -> bool {
    ancestors(f, root).iter().any(|&v| {
        matches!(
            f.def(v).map(|(bb, ix)| &f.block(bb).instrs[ix].op),
            Some(Op::Undef)
        )
    })
}

/// The single returned value of a function that has exactly one.
fn returned(f: &Function) -> u32 {
    let mut v = None;
    for b in &f.blocks {
        if let Terminator::Return(x) = b.term {
            assert!(v.replace(x).is_none(), "one return expected");
        }
    }
    v.expect("a return")
}

// --- the four argument registers are the entry block's, for every block ------
//
//   0: 85 c9           test ecx, ecx
//   2: 74 03           je   0x7
//   4: 89 d0           mov  eax, edx     <- the SECOND argument, in block 2
//   6: c3              ret
//   7: b8 01 00 00 00  mov  eax, 1
//   c: c3              ret
const ARG_IN_A_LATER_BLOCK: &[u8] = &[
    0x85, 0xC9, 0x74, 0x03, 0x89, 0xD0, 0xC3, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3,
];

#[test]
fn an_argument_read_after_the_entry_block_is_the_incoming_parameter() {
    let (f, rep) = lift(ARG_IN_A_LATER_BLOCK);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(rep.gaps, 0, "nothing here is unmodelled");
    assert_eq!(
        rep.undef_live_ins, 0,
        "RDX is an argument, not a register read on a path that never writes it"
    );
    // The counterfactual this replaces: `mov eax, edx` at 0x4 returned `undef`.
    let mut paths = 0;
    for b in &f.blocks {
        if let Terminator::Return(v) = b.term {
            paths += 1;
            assert!(
                !reads_undef(&f, v),
                "the return of RAX is defined on every path\n{}",
                rdec_core::fmt_ir::format_function(&f)
            );
        }
    }
    assert_eq!(paths, 2, "two returns, both defined");

    // The value, measured: a == 0 takes `je` (ZF=1) and returns 1; anything else
    // falls through to `mov eax, edx`, which returns b. Only an argument that
    // really reached the block can produce 0x1234.
    assert_eq!(answer(&f, &[0, 0x1234, 0, 0]), Outcome::Ret(Val::Int(1)));
    assert_eq!(
        answer(&f, &[7, 0x1234, 0, 0]),
        Outcome::Ret(Val::Int(0x1234)),
        "the second argument reaches a block after the entry\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );

    // The answer arrives through a 32-bit write (`mov eax, edx`), so bits 63:32
    // are zero: the width the lifter records, and the C signature that matches.
    assert_eq!(rdec_core::width::return_width(&f), 32);
    let mut g = f.clone();
    clean(&mut g);
    assert_eq!(verify_ssa(&g), Ok(()), "the passes keep it valid");
    let out = rdec_core::emit_c::emit_c(&g).expect("it emits C, not a refusal");
    assert!(out.text.contains("return"), "{}", out.text);
}

#[test]
fn the_argument_registers_are_unsigned_register_values() {
    // Registers are one unsigned W64 word; the arguments ARE registers, so the
    // entry's params are too. Typed `int64_t`, they were the only signed values
    // in the register model, and every op a later block built from one reported
    // "op type int64_t != recorded uint64_t" to the verifier.
    let (f, _) = lift(ARG_IN_A_LATER_BLOCK);
    assert_eq!(f.params.len(), 4, "the four integer arguments");
    for p in &f.params {
        assert_eq!(
            f.ty_of(*p),
            Some(Ty::u64()),
            "argument %{p} is a register value\n{}",
            rdec_core::fmt_ir::format_function(&f)
        );
    }
    assert_eq!(verify_ssa(&f), Ok(()));
    let mut g = f.clone();
    clean(&mut g);
    assert_eq!(
        verify_ssa(&g),
        Ok(()),
        "and the passes keep that true: {}\n{}",
        rdec_core::fmt_ir::format_function(&g),
        g.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .map(|i| format!("%{} {:?}", i.dest, i.op))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// --- the modelled stack base is the entry block's, for every block ----------
//
//   0: 55                push rbp              <- RSP := RSP-8, in the ENTRY
//   1: 85 c9             test ecx, ecx
//   3: 74 09             je   0xe
//   5: 48 8b 04 24       mov  rax, [rsp]       <- a frame slot, in a LATER block
//   9: 48 83 c0 01       add  rax, 1
//   d: c3                ret
//   e: b8 07 00 00 00    mov  eax, 7
//  13: c3                ret
const PUSH_THEN_FRAME_SLOT: &[u8] = &[
    0x55, 0x85, 0xC9, 0x74, 0x09, 0x48, 0x8B, 0x04, 0x24, 0x48, 0x83, 0xC0, 0x01, 0xC3, 0xB8, 0x07,
    0x00, 0x00, 0x00, 0xC3,
];

#[test]
fn a_push_in_the_entry_block_leaves_rsp_defined_for_later_blocks() {
    let (f, rep) = lift(PUSH_THEN_FRAME_SLOT);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // The load's address is the frame: constant-foldable, so T2's frame rule can
    // name the same bytes on both sides. Unmarked, RSP left the entry as
    // `undef`, the address was not static, and the whole function was filed
    // under "memory outside the modelled frame".
    let load = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Load(..)))
        .expect("the frame-slot load");
    let addr = match load.op {
        Op::Load(_, a) => a,
        _ => unreachable!(),
    };
    assert!(
        !reads_undef(&f, addr),
        "the load's address is RSP-relative and RSP is defined\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(rep.exits, 0, "no unmodelled tail");
    // The push stored RBP (undefined at entry - the caller's frame pointer, not
    // ours), so reading it back is reading the C model's 0: the answer on the
    // fallthrough path is 0 + 1.
    assert_eq!(
        answer(&f, &[7, 0, 0, 0]),
        Outcome::Ret(Val::Int(1)),
        "the frame slot round-trips\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(answer(&f, &[0, 0, 0, 0]), Outcome::Ret(Val::Int(7)));
}

// --- a call is a definition of RAX and of every caller-saved register -------
//
//   0: 85 c9            test ecx, ecx
//   2: 74 14            je   0x18
//   4: e8 f7 0f 00 00   call 0x1000            (outside: the object has no such byte)
//   9: 85 f6            test esi, esi
//   b: 75 07            jne  0x14
//   d: b8 00 00 00 00   mov  eax, 0
//  12: eb 04            jmp  0x18
//  14: 89 c8            mov  eax, ecx          <- ECX in a SUCCESSOR of the call
//  16: c3               ret
//  18: c3               ret
const CALL_THEN_READ_ARG: &[u8] = &[
    0x85, 0xC9, 0x74, 0x14, 0xE8, 0xF7, 0x0F, 0x00, 0x00, 0x85, 0xF6, 0x75, 0x07, 0xB8, 0x00, 0x00,
    0x00, 0x00, 0xEB, 0x04, 0x89, 0xC8, 0xC3, 0x90, 0xC3,
];

#[test]
fn a_successor_of_a_call_reads_the_clobber_not_the_pre_call_argument() {
    let (f, rep) = lift(CALL_THEN_READ_ARG);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(
        rep.clobbers >= 9,
        "the nine caller-saved registers, counted"
    );
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i.op, Op::Clobber)),
        "the clobber is in the IR: {:?}\n{}",
        opaque_texts(&f),
        rdec_core::fmt_ir::format_function(&f)
    );
    // The claim: the `mov eax, ecx` after the call consumes the clobber, so the
    // function has NO defined answer on that path. Before the mark, ECX flowed
    // across the call unchanged and the emitted C returned the caller's `a` -
    // a value the call had destroyed, with nothing loud about it.
    let after_call = f
        .blocks
        .iter()
        .find(|b| b.addr == 0x14)
        .expect("the block after the call");
    let v = match after_call.term {
        Terminator::Return(v) => v,
        ref t => panic!("expected a return, got {t:?}"),
    };
    let has_clobber = ancestors(&f, v).iter().any(|&x| {
        matches!(
            f.def(x).map(|(bb, ix)| &f.block(bb).instrs[ix].op),
            Some(Op::Clobber)
        )
    });
    assert!(
        has_clobber,
        "the value returned after a call comes from the clobber\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
}

// --- a loop that reads a register it never writes: the machine's fact -------
//
//   0: 31 c0           xor  eax, eax           ; sum = 0
//   2: 85 c9           test ecx, ecx
//   4: 7e 0c           jle  0x12
//   6: 83 fa 03        cmp  edx, 3             <- loop header reads EDX ...
//   9: 7d 07           jge  0x12
//   b: 01 c8           add  eax, ecx           ; sum += a
//   d: 83 c1 ff        add  ecx, -1            ; a -= 1
//  10: 75 f4           jne  0x6                ; ... and the back edge writes NOT
//  12: c3              ret
const LOOP_READS_ARG: &[u8] = &[
    0x31, 0xC0, 0x85, 0xC9, 0x7E, 0x0C, 0x83, 0xFA, 0x03, 0x7D, 0x07, 0x01, 0xC8, 0x83, 0xC1, 0xFF,
    0x75, 0xF4, 0xC3,
];

#[test]
fn a_register_the_loop_never_writes_carries_its_own_value_around_the_back_edge() {
    let (f, rep) = lift(LOOP_READS_ARG);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(rep.gaps, 0);
    // The header's phi for RDX has the parameter on the entry edge and ITSELF on
    // the back edge - `phi(x, phi)`, the standard form for a loop-invariant
    // register. `prune_phis` collapses it, so after the pipeline the loop reads
    // the argument, exactly as the machine does.
    let mut g = f.clone();
    clean(&mut g);
    assert_eq!(verify_ssa(&g), Ok(()), "the passes keep it valid");
    let phi_left = g.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(&i.op, Op::Phi(pairs) if pairs.iter().any(|(v, _)| *v == i.dest)))
    });
    assert!(
        !phi_left,
        "the self-referential phi collapsed to the invariant value\n{}",
        rdec_core::fmt_ir::format_function(&g)
    );

    // Measured, and the two cases are what distinguishes "the argument" from
    // "undef": b = 2 (< 3) runs the loop to a == 0, b = 5 (>= 3) leaves at once.
    // An undefined RDX reads as 0 in the strict model, which would run the loop
    // for b = 5 as well - the answer the decompiler must NOT give.
    assert_eq!(
        answer(&g, &[3, 2, 0, 0]),
        Outcome::Ret(Val::Int(6)),
        "3 + 2 + 1"
    );
    assert_eq!(
        answer(&g, &[3, 5, 0, 0]),
        Outcome::Ret(Val::Int(0)),
        "b >= 3 leaves the loop before it runs\n{}",
        rdec_core::fmt_ir::format_function(&g)
    );
    let r = returned(&g);
    assert!(
        !reads_undef(&g, r),
        "the loop's answer is defined on every path\n{}",
        rdec_core::fmt_ir::format_function(&g)
    );
}

// --- the real gap, kept loud: a loop-carried register nothing defines -------
//
//   0: 85 c9           test ecx, ecx
//   2: 7e 07           je   0xb
//   4: 01 d8           add  eax, ebx     <- RAX and RBX: read, never written
//   6: 83 c1 ff        add  ecx, -1
//   9: eb f9           jmp  0x4
//   b: c3              ret
const LOOP_WITH_NO_BASE: &[u8] = &[
    0x85, 0xC9, 0x7E, 0x07, 0x01, 0xD8, 0x83, 0xC1, 0xFF, 0xEB, 0xF9, 0xC3,
];

#[test]
fn a_register_no_path_ever_writes_stays_undefined_and_loud() {
    let (f, rep) = lift(LOOP_WITH_NO_BASE);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // RAX and RBX are not arguments: the caller owns them. The loop header's phi
    // has no base case, so it is the genuine gap - the value the machine leaves
    // there is the caller's leftover, and the C model's 0 is not that.
    assert!(
        rep.undef_live_ins > 0,
        "the read-before-write is counted, not hidden"
    );
    let mut g = f.clone();
    clean(&mut g);
    assert_eq!(verify_ssa(&g), Ok(()));
    let r = returned(&g);
    assert!(
        reads_undef(&g, r),
        "a self-referential phi with no base case is `undef`, and T2 must see it\n{}",
        rdec_core::fmt_ir::format_function(&g)
    );
}

#[test]
fn a_self_referential_phi_is_valid_ir_and_the_verifier_accepts_it() {
    // The phi form the SSA pass now produces for a loop-invariant register,
    // built directly: the header's `%p = phi [entry]1, [latch]%p`, in a NATURAL
    // loop (the header dominates the latch), which is the case the verifier's
    // phi-edge rule was written for - "%v has to be live at the END of its
    // predecessor", and a header's own value is exactly that on its back edge.
    // `prune_phis` then collapses it to `copy 1`.
    use rdec_core::ty::Ty as T;
    let mut f = Function::new("inv", 0);
    let e = f.entry;
    let h = f.new_block(0x10);
    let l = f.new_block(0x20);
    let x = f.emit(e, Op::Const(1), T::u64(), 0);
    f.block_mut(e).term = Terminator::Jump(h);
    let p = f.new_value(T::u64());
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            p,
            Op::Phi(vec![(x, e), (p, l)].into_boxed_slice()),
            T::u64(),
            0x10,
        ),
    );
    let cond = f.emit(h, Op::Const(1), T::Bool, 0);
    let done = f.new_block(0x30);
    f.block_mut(h).term = Terminator::Branch {
        cond,
        then_bb: l,
        else_bb: done,
    };
    f.block_mut(l).term = Terminator::Jump(h);
    f.block_mut(done).term = Terminator::Return(p);
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "a self edge in a natural loop is SSA, not an error"
    );
    assert!(
        rdec_core::passes::prune_phis(&mut f),
        "the self edge collapses"
    );
    let after = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| i.dest == p)
        .map(|i| i.op.clone());
    assert_eq!(after, Some(Op::Copy(x)), "the invariant value, not undef");
    assert_eq!(verify_ssa(&f), Ok(()), "and the result still verifies");
}
