//! TAIL CALLS: a `jmp` whose relocation names a defined function symbol is
//! `return callee(args);` — the callee gets the caller's argument registers
//! (the pair rule already put them there), its answer is the caller's answer,
//! and control NEVER returns to the caller's code.
//!
//! GROUND TRUTH — `clang -O2 -c tc.c` (clang 22.1.8, x86_64-pc-windows-msvc),
//! `llvm-objdump -d -r tc.O2.obj`, quoted verbatim:
//!
//! ```text
//! 0000000000000000 <g>:
//!        0: 8d 04 49                     leal (%rcx,%rcx,2), %eax
//!        3: 01 d0                        addl %edx, %eax
//!        5: c3                           retq
//!        6: 66 2e 0f 1f 84 00 00 00 00 00    nopw %cs:(%rax,%rax)
//!
//! 0000000000000010 <tail>:
//!       10: ff c1                        incl %ecx
//!       12: e9 00 00 00 00               jmp  0x17 <tail+0x7>
//!               0000000000000013:  IMAGE_REL_AMD64_REL32  g
//!       17: 66 0f 1f 84 00 00 00 00 00   nopw (%rax,%rax)
//!
//! 0000000000000020 <noident>:
//!       20: 48 83 ec 28                  subq $0x28, %rsp
//!       24: e8 00 00 00 00               callq 0x29 <noident+0x9>
//!               0000000000000025:  IMAGE_REL_AMD64_REL32  g
//!       29: ff c0                        incl %eax
//!       2b: 48 83 c4 28                  addq $0x28, %rsp
//!       2f: c3                           retq
//! ```
//!
//! Three facts, all of them the object's, none of them a reading of what the
//! code does:
//!
//! * the `jmp` at 0x12 carries `IMAGE_REL_AMD64_REL32 -> g` at its displacement
//!   (0x13), and `g` is a DEFINED function symbol (`llvm-nm`: `T g` at 0x0). The
//!   printed immediate `0x17` is the UNRELOCATED next-IP — the same trap that
//!   made `calls: 0 modelled` before `82c285f`. The relocation is the target.
//! * the `jmp` is the last thing in `tail`: what follows it is `nopw` padding,
//!   and nothing in the object branches to it. Code after a tail call is not
//!   reachable from it.
//! * the sibling form above (`noident`) is the SAME relocation on a `call`. The
//!   only difference between a call and a tail call is the opcode and the fact
//!   that the `jmp` has no fall-through successor.
//!
//! NOT tail calls, each with its own fixture below:
//!
//! * `jmp *%rax` — a thunk, a vtable slot, a jump-table dispatch. No target in
//!   the bytes and no relocation, so there is no callee to name. The corpus's
//!   only two `jmp`-with-no-fallthrough-in-a-switch shapes are exactly this
//!   (`corpus/obj/switch_dense.O0.obj` 0x28, `switch_fallthrough.O0.obj` 0x30).
//! * a direct `jmp` INSIDE one function — a loop back edge. `llvm-objdump -d -r`
//!   over all 135 corpus objects: 65 `jmp` sites, 0 of them carry a relocation,
//!   and every one of them targets an address inside the function that holds it.
//!   That is why the corpus has no tail-call witness and this file's fixtures are
//!   hand-assembled to the clang shape quoted above.
//! * a `jmp` whose relocation names a symbol the object does not define (an
//!   external `jmp abort`): the name is evidence, the address is not there, and
//!   the C has no callee to call. Loud refusal, counted.
//!
//! The fixtures bind `f.params` through the real `FnAbi` seed (`Lifter::new_sig`
//! seeds RCX/RDX/R8/R9 as the parameters), so every argument the test reads is a
//! parameter the pipeline declared, never an unbound `Op::Param`.
use rdec_arch_x86::discover::{
    discover_all, discover_all_with_container, Code, Displacements, Tables,
};
use rdec_arch_x86::lift::{
    lift_one_sigs, no_relocs, no_tables, CalleeSig, FnAbi, Lifter, RelocMap, RelocTarget,
};
use rdec_core::emit_c::{callee_decl, emit_c_sig, Protos};
use rdec_core::interp::{run_program_budget, Fault, Outcome};
use rdec_core::ir::{Callee, Function, Op, Terminator, NO_VAL};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;
use std::collections::HashMap;

/// The bytes of the clang object above, with one addition: where clang emitted
/// `nopw` padding after the tail call, this fixture emits
/// `mov eax, 0x190; ret` (0x17..0x1d). It is UNREACHABLE — nothing branches to
/// it, the `jmp` has no fall-through — and every test below asserts it never
/// reaches the IR, the C, or an answer.
fn bytes() -> Vec<u8> {
    vec![
        // 0x00  g(int a, int b): lea eax,[rcx+rcx*2]; add eax,edx; ret
        0x8D, 0x04, 0x49, 0x01, 0xD0, 0xC3, //
        // 0x06  three padding bytes so `tail` starts at 0x10 like the object
        0x90, 0x90, 0x90, //
        // 0x09  slack to 0x10
        0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
        // 0x10  tail(int a, int b): incl ecx; jmp g
        0xFF, 0xC1, //           0x10  incl %ecx
        0xE9, 0x00, 0x00, 0x00, 0x00, //  0x12  jmp rel32 (reloc at 0x13 -> g)
        // 0x17  unreachable: the code after a tail call
        0xB8, 0x90, 0x01, 0x00, 0x00, //  0x17  mov eax, 0x190
        0xC3, //                   0x1c  ret
        // 0x1d  slack to 0x20
        0x90, 0x90, 0x90, //
        // 0x20  noident(int a, int b): sub rsp,0x28; call g; inc eax; add rsp,0x28; ret
        0x48, 0x83, 0xEC, 0x28, //   0x20
        0xE8, 0x00, 0x00, 0x00, 0x00, //  0x24  call rel32 (reloc at 0x25 -> g)
        0xFF, 0xC0, //          0x29  inc eax
        0x48, 0x83, 0xC4, 0x28, //   0x2b
        0xC3, //                0x2f  ret
    ]
}

/// One relocation this fixture's object carries: the address of the DISPLACEMENT
/// FIELD, the symbol it names, and that symbol's address when the object DEFINES
/// it (`None` for a symbol it does not — an import, `U` in `llvm-nm`).
///
/// The fixture keeps ONE table of this evidence and spells it the two ways the
/// pipeline consumes it, exactly as `m4::lift_all_named_shapes` does: a
/// `RelocMap` for the lifter (what `m4::reloc_map_of` builds) and a
/// `Displacements` map for discovery (what `discover::displacements(obj, ..)`
/// builds). Discovery has to be given the same evidence the lifter is given, or
/// it reads a relocated branch's UNRELOCATED immediate as a branch target and
/// records a block at `0x17` — the phantom `82c285f` fixed for calls, and a
/// block the emitted C then has to account for.
type RelocRow = (u64, &'static str, Option<u64>);

/// The two `REL32` relocations of `bytes()`: the `jmp` at 0x12 and the `call` at
/// 0x24, both naming `g`, which the object defines at 0x0 (`T g`).
fn tail_rows() -> Vec<RelocRow> {
    vec![(0x13, "g", Some(0)), (0x25, "g", Some(0))]
}

/// The one relocation of `nr_bytes()`: the `call` at 0x16 names `abort`, which
/// the object does not define (`U abort`), so it has no address here.
fn nr_rows() -> Vec<RelocRow> {
    vec![(0x17, "abort", None)]
}

/// The relocation table in the shape `m4::reloc_map_of` builds: keyed by the
/// address of the DISPLACEMENT FIELD.
fn relocs_of(rows: &[RelocRow]) -> RelocMap {
    rows.iter()
        .map(|(a, sym, addr)| {
            (
                *a,
                RelocTarget {
                    sym: (*sym).to_string(),
                    addr: *addr,
                },
            )
        })
        .collect()
}

/// The same rows in the shape `discover::displacements` builds: the field's
/// address to the named symbol's address, `None` for a symbol not defined here.
fn disp_of(rows: &[RelocRow]) -> Displacements {
    rows.iter().map(|(a, _, addr)| (*a, *addr)).collect()
}

fn names() -> HashMap<u64, String> {
    [("g", 0u64), ("tail", 0x10u64), ("noident", 0x20u64)]
        .into_iter()
        .map(|(n, a)| (a, n.to_string()))
        .collect()
}

fn g_table() -> HashMap<String, CalleeSig> {
    [(
        "g".to_string(),
        CalleeSig {
            cname: "g".to_string(),
            abi: FnAbi::integer(),
            typed: true,
            ..Default::default()
        },
    )]
    .into_iter()
    .collect()
}

/// Lift one of the fixture's three functions through the same entry point every
/// gate uses, with the relocation evidence in hand on BOTH sides of the pipeline:
/// the discovery plan (`discover_all_with_container`) and the lifter
/// (`lift_one_sigs`'s `RelocMap`).
fn lift_at(
    start: u64,
    end: u64,
    name: &str,
    rows: &[RelocRow],
) -> (Function, rdec_arch_x86::lift::Report) {
    let owned = bytes();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    seeds.insert(0, (vec![0u64, 0x10, 0x20], vec![]));
    let plan = discover_all_with_container(&ranges, &seeds, &Tables::default(), &disp_of(rows));
    let (f, rep) = lift_one_sigs(
        name,
        &owned,
        0,
        start,
        end,
        &plan,
        &names(),
        &FnAbi::integer(),
        &g_table(),
        &relocs_of(rows),
        &no_tables(),
    );
    (f, rep)
}

fn lift_tail() -> (Function, rdec_arch_x86::lift::Report) {
    lift_at(0x10, 0x20, "tail", &tail_rows())
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

fn g_protos() -> Protos {
    let (mut g, _) = lift_at(0, 0x10, "g", &[]);
    clean(&mut g);
    [("g".to_string(), callee_decl(&g))].into_iter().collect()
}

// ---------------------------------------------------------------------------
// THE REQUIREMENT. A tail call is `return callee(args);`.
// ---------------------------------------------------------------------------
#[test]
fn a_jmp_to_a_named_function_symbol_is_a_tail_call_that_returns_the_callee_answer() {
    let (mut f, rep) = lift_tail();
    clean(&mut f);
    verify_ssa(&f).expect("a tail-calling function is valid SSA");

    // (1) the machine fact is counted, at the jmp, where the evidence is.
    assert_eq!(
        rep.tail_calls, 1,
        "one tail call modelled (the jmp at 0x12, REL32 -> g)"
    );
    assert_eq!(rep.tail_calls_unnamed, 0);

    // (2) the IR: a Call that defines the answer, and a terminator that exits
    // with it. The CFG's own successors are the control flow: a tail call has
    // NO fall-through successor.
    let term_bb = f
        .blocks
        .iter()
        .find(|b| matches!(b.term, Terminator::TailReturn(_)))
        .expect("the tail call's block ends in TailReturn, not a fall-through");
    let mut succ = Vec::new();
    term_bb.term.successors(&mut succ);
    assert!(
        succ.is_empty(),
        "a tail call has no successor: {:?} at bb{}",
        succ,
        term_bb.id
    );
    let value = match term_bb.term {
        Terminator::TailReturn(v) => v,
        _ => unreachable!(),
    };
    let (def_bb, def_i) = f.def(value).expect("the returned value is defined");
    assert_eq!(
        def_bb, term_bb.id,
        "the tail call and its terminator are in one block"
    );
    match &f.blocks[def_bb as usize].instrs[def_i].op {
        Op::Call { callee, args, .. } => {
            assert_eq!(
                format!("{callee:?}"),
                "Symbol(\"g\")",
                "the callee's identity is the symbol the relocation named"
            );
            assert!(
                !args.is_empty(),
                "the callee gets the caller's argument registers"
            );
        }
        other => panic!("a tail call returns a Call's value, not {other:?}"),
    }
    // `Block::addr` counts from the function's own start (instruction addresses
    // are absolute), so the entry block is block 0 at offset 0.
    assert_eq!(
        term_bb.id, 0,
        "the block that tail-calls is the function's entry block"
    );
    // The `jmp`'s own address is on the call, and it is the `jmp`: an assertion
    // on the instruction, not on the block, because the block starts earlier.
    let jmp_addr = f.blocks[def_bb as usize].instrs[def_i].addr;
    assert_eq!(
        jmp_addr, 0x12,
        "the tail call is the `jmp` at 0x12, not the instruction before it"
    );

    // (3) the code after the tail call is NOT reachable: `mov eax,0x190; ret`
    // at 0x17 is bytes the function can never execute, so 0x190 appears nowhere.
    let has_0x190 = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i.op, Op::Const(c) if c == 0x190));
    assert!(
        !has_0x190,
        "code after a tail call reached the IR:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .all(|i| i.addr < 0x17),
        "an IR node records an address at or past the tail call"
    );

    // (4) the C: `return callee(args);`, with the callee's prototype.
    let protos = g_protos();
    let out = match emit_c_sig(&f, &protos) {
        Ok(o) => o,
        Err(rs) => panic!(
            "a tail call must emit C, refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        ),
    };
    assert!(
        out.text
            .lines()
            .any(|l| l.starts_with("extern uint32_t g(uint64_t, uint64_t, uint64_t, uint64_t);")),
        "(a) no prototype for g:\n{}",
        out.text
    );
    assert!(
        out.text.lines().any(|l| {
            let t = l.trim();
            t.starts_with("return ") && t.contains("g(") && t.ends_with(';')
        }),
        "(b) the emitter must say `return g(...);`:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("__builtin_trap"),
        "(c) a modelled tail call must not trap:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("0x190"),
        "(d) the unreachable tail leaked into the C:\n{}",
        out.text
    );

    // (5) the interpreter answers with the callee's value: tail(5,7) =
    // g(6,7) = 6*3+7 = 25, computed by running g's own body.
    let exec = run_program_budget(
        &f,
        &[5, 7, 0, 0],
        &[{
            let (mut g, _) = lift_at(0, 0x10, "g", &[]);
            clean(&mut g);
            g
        }],
        20000,
    );
    match exec.outcome {
        Outcome::Ret(v) => assert_eq!(v.bits(), 25, "tail(5,7) = g(6,7)"),
        other => panic!("a tail call returns the callee's answer, got {other:?}"),
    }
}

#[test]
fn an_absent_tail_callee_is_a_fault_never_a_guessed_value() {
    let (mut f, _) = lift_tail();
    clean(&mut f);
    let ex = run_program_budget(&f, &[5, 7, 0, 0], &[], 20000);
    match ex.outcome {
        Outcome::Fault(Fault::UnknownCallee) => {}
        other => {
            panic!("a tail call to a callee that is not in the model must fault, got {other:?}")
        }
    }
}

#[test]
fn the_verifier_refuses_a_tail_call_whose_callee_has_no_name() {
    // The identity IS the point: a tail call has no fall-through, so an
    // unnamed callee means the function's whole answer names nobody.
    let mut f = Function::new("addr_tail", 0);
    let e = f.entry;
    let p = f.emit(e, Op::Param, Ty::u64(), 0);
    f.params.push(p);
    let v = f.emit(
        e,
        Op::Call {
            callee: Callee::Addr(0x40),
            ret_ty: Ty::i64(),
            args: vec![p].into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    f.block_mut(e).term = Terminator::TailReturn(v);
    f.ret_ty = Ty::i64();
    let errs = verify_ssa(&f).expect_err("an unnamed tail callee is a hole in the IR");
    assert!(
        errs.iter().any(|s| s.contains("callee")),
        "the verifier must name the callee: {errs:?}"
    );

    // An empty symbol name is the same hole.
    let mut g = Function::new("empty_tail", 0);
    let e = g.entry;
    let p = g.emit(e, Op::Param, Ty::u64(), 0);
    g.params.push(p);
    let v = g.emit(
        e,
        Op::Call {
            callee: Callee::Symbol(String::new()),
            ret_ty: Ty::i64(),
            args: vec![p].into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    g.block_mut(e).term = Terminator::TailReturn(v);
    g.ret_ty = Ty::i64();
    let errs = verify_ssa(&g).expect_err("an empty callee name is a hole in the IR");
    assert!(
        errs.iter().any(|s| s.contains("callee")),
        "the verifier must name the callee: {errs:?}"
    );

    // A tail terminator whose value is NOT a call is the IR lying about a
    // machine shape: no `jmp` computes a value.
    let mut h = Function::new("not_a_call", 0);
    let e = h.entry;
    let c = h.emit(e, Op::Const(7), Ty::u64(), 0);
    h.block_mut(e).term = Terminator::TailReturn(c);
    h.ret_ty = Ty::u64();
    let errs = verify_ssa(&h).expect_err("TailReturn of a non-call is not a tail call");
    assert!(
        errs.iter().any(|s| s.contains("tail")),
        "the verifier must name the tail call: {errs:?}"
    );

    // The well-formed shape verifies.
    let mut ok = Function::new("ok", 0);
    let e = ok.entry;
    let p = ok.emit(e, Op::Param, Ty::u64(), 0);
    ok.params.push(p);
    let v = ok.emit(
        e,
        Op::Call {
            callee: Callee::Symbol("g".into()),
            ret_ty: Ty::i64(),
            args: vec![p].into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    ok.block_mut(e).term = Terminator::TailReturn(v);
    ok.ret_ty = Ty::i64();
    verify_ssa(&ok).expect("a named tail call verifies");
}

// ---------------------------------------------------------------------------
// NOT tail calls.
// ---------------------------------------------------------------------------
#[test]
fn an_indirect_jmp_is_not_a_tail_call() {
    // `jmp *%rax` at 0x0 (FF E0), then `ret`. A thunk, a vtable slot, a
    // jump-table dispatch: no callee in the bytes and no relocation naming one.
    let owned = vec![0xFF, 0xE0, 0xC3];
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let (f, rep) = lift_one_sigs(
        "disp",
        &owned,
        0,
        0,
        owned.len() as u64,
        &plan,
        &names(),
        &FnAbi::integer(),
        &g_table(),
        &no_relocs(),
        &no_tables(),
    );
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.tail_calls, 0, "an indirect jmp names no callee");
    assert_eq!(rep.tail_calls_unnamed, 0);
    assert_eq!(rep.exits, 1, "it keeps its Exit");
    assert!(
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::TailReturn(_))),
        "no TailReturn from an indirect jmp"
    );
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .any(|i| matches!(&i.op, Op::Opaque(t) if &**t == "indirect jmp")),
        "the indirect jmp keeps its own reason string"
    );
}

#[test]
fn a_direct_jmp_inside_one_function_is_not_a_tail_call() {
    // A loop back edge: `mov eax,ecx; add eax,1; jmp 0x0`, i.e. the direct
    // `jmp rel8` whose target is a block of THIS function (no relocation at the
    // displacement, and every corpus `jmp` is this shape).
    let owned = vec![
        0x89, 0xC8, // 0x00 mov eax, ecx
        0x83, 0xC0, 0x01, // 0x02 add eax, 1
        0xEB, 0xFB, // 0x05 jmp 0x2  (back edge, rel8: 0x07-10 = 0x2)
    ];
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let (f, rep) = lift_one_sigs(
        "loop_",
        &owned,
        0,
        0,
        owned.len() as u64,
        &plan,
        &names(),
        &FnAbi::integer(),
        &g_table(),
        &no_relocs(),
        &no_tables(),
    );
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(
        rep.tail_calls, 0,
        "a jump inside a function is control flow"
    );
    assert_eq!(rep.tail_calls_unnamed, 0);
    let jumps = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Jump(_)))
        .count();
    assert!(
        jumps >= 1,
        "the back edge is a block edge:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
}

#[test]
fn a_jmp_to_a_symbol_the_object_does_not_define_stays_loud() {
    // The `jmp` at 0x12 names `abort`, and `abort` is not defined here:
    // `RelocTarget{ addr: None }`. The name is evidence; there is no callee in
    // the translation unit to call, and no address to call. Loud, counted.
    // The same relocation evidence, spelled for both halves of the pipeline.
    let rows: Vec<RelocRow> = vec![(0x13, "abort", None), (0x25, "g", Some(0))];
    let (f, rep) = lift_at(0x10, 0x20, "tail", &rows);
    assert_eq!(rep.tail_calls, 0, "no callee, no tail call");
    assert_eq!(
        rep.tail_calls_unnamed, 1,
        "the refused tail call is counted, not swallowed"
    );
    assert_eq!(rep.exits, 1, "and the block keeps its Exit");
    let has_reason = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(&i.op, Op::Opaque(t) if t.contains("abort")));
    assert!(
        has_reason,
        "the refusal names the symbol the relocation carried:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
}

#[test]
fn a_call_is_not_a_tail_call_and_keeps_its_fall_through() {
    // `noident` is the same relocation on a `call` (0x25 -> g): the call
    // returns, `inc eax` runs, `ret` answers. Counted as a call, never as a
    // tail call, and the block after the call is reachable.
    let (mut f, rep) = lift_at(0x20, 0x40, "noident", &tail_rows());
    clean(&mut f);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.tail_calls, 0, "a call is not a tail call");
    assert_eq!(rep.calls_modelled, 1, "it is the call model that owns it");
    assert!(
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::TailReturn(_))),
        "the caller's own `ret` answers, not a tail terminator"
    );
    let term = &f
        .blocks
        .iter()
        .find(|b| matches!(b.term, Terminator::Return(_)))
        .map(|b| b.term.clone());
    assert!(
        matches!(term, Some(Terminator::Return(_))),
        "the ret is the exit"
    );
    // 5,7 -> g(5,7)+1 = (5*3+7)+1 = 23.
    let exec = run_program_budget(
        &f,
        &[5, 7, 0, 0],
        &[{
            let (mut g, _) = lift_at(0, 0x10, "g", &[]);
            clean(&mut g);
            g
        }],
        20000,
    );
    match exec.outcome {
        Outcome::Ret(v) => assert_eq!(v.bits(), 23, "g(5,7)+1"),
        other => panic!("the call returns, got {other:?}"),
    }
    let _ = NO_VAL;
}

/// The seeding window is unchanged by all of this: a tail call is a statement
/// about registers, not about parameters. (A fixture that built `Op::Param`
/// nodes without assigning `f.params` binds no argument and passes on broken
/// code, which is why the fixtures above lift real bytes.)
#[test]
fn lifting_a_tail_caller_keeps_the_abi_seeding() {
    let n = names();
    let l = Lifter::new_abi("tail", 0x10, &n, &FnAbi::integer());
    assert_eq!(
        l.entry_regs().iter().filter(|(_, v)| v.is_some()).count(),
        4,
        "the four integer argument registers"
    );
    let (f, _) = lift_tail();
    assert_eq!(f.params.len(), 4, "the lifted caller binds four parameters");
    for p in &f.params {
        assert_eq!(f.ty_of(*p), Some(Ty::u64()));
    }
    let _ = NO_VAL;
}

// ===========================================================================
// SLICE 2: a call to a callee that NEVER RETURNS.
//
// GROUND TRUTH - `clang -O0 -c nr.c` (nr.c: `int noreturn_tail(int a,int b){
// if (a>b) return g(a,b); abort(); }`), `llvm-objdump -d -r nr.O0.obj`, verbatim:
//
// ```text
// 0000000000000049 <noreturn_tail+0x29>:
//       49: e8 00 00 00 00                   callq 0x4e <noreturn_tail+0x2e>
//               000000000000004a:  IMAGE_REL_AMD64_REL32  abort
//       4e: cc                               int3
//       4f: 90                               nop
//
// 000000000000006d <direct_exit+0x1d>:
//       6d: e8 00 00 00 00                   callq 0x72 <direct_exit+0x22>
//               000000000000006e:  IMAGE_REL_AMD64_REL32  exit
//       72: cc                               int3
// ```
//
// and `llvm-nm nr.O0.obj`:
//
// ```text
//          U abort
//          U exit
// 00000050 T direct_exit
// 00000000 T g
// 00000020 T noreturn_tail
// ```
//
// Three facts, and the fixture below encodes each of them:
//
// * the callee is `U` - UNDEFINED in the object, an import. The NAME is the only
//   evidence, and it comes from the relocation at the displacement field exactly
//   as a call's does; the printed `callq 0x4e` is the unrelocated next-IP.
// * the block ENDS at the call. What follows is `int3` then `nop`: interrupt and
//   padding, not the caller's code. Control after a noreturn call is not
//   reachable from it, and the function has no fall-through answer.
// * `noreturn` is a DECLARATION, not a reading of the bytes. Nothing in that
//   encoding says `abort` does not come back; the front end says it. So the
//   lifter models the shape only when the callee table declares it
//   (`CalleeSig::noreturn`, built by the front end from `lift::noreturn_symbol`),
//   and the same bytes WITHOUT that declaration keep the loud call shape they
//   always had - the last test pins exactly that.
//
// THE CORPUS HAS NO WITNESS, and the number is measured, not assumed: over all
// 135 objects `llvm-objdump -d -r` finds 31 call sites, and the census of their
// relocation names is 12 `h`, 5 `f`, 2 `??$tmax@H@@YAHHH@Z`, one each of
// `?clamp@detail@geo@@YAHHHH@Z`, `?bump@Inner@Outer@@QEAAHH@Z`,
// `?get@Inner@Outer@@QEBAHXZ`, `?value@Base@@UEBAHXZ`, `?twice@Base@@UEBAHXZ`,
// `??1Base@@UEAA@XZ`, `?value@Mid@@UEBAHXZ`, plus 5 calls with NO relocation at
// all (`3 * (%rax)`, `2 * 0x8(%rax)` - thunks and vtable slots). Not one is
// `abort`, `exit`, `_exit`, `__fastfail`, a throw, or `operator new`, so
// `rdec lift --corpus` reports `noreturn calls: 0` and these fixtures are the
// only witnesses the model has. Tail calls, same story: 65 `jmp` sites, 0 of
// them carrying a relocation.
// ===========================================================================
/// The `abort()` tail of `noreturn_tail`, to the shape quoted above. Where clang
/// put `int3` + `nop` after the call, this fixture puts `mov eax,0x190; ret`:
/// unreachable code, and every test below asserts it never reaches the IR, the
/// C, or an answer.
fn nr_bytes() -> Vec<u8> {
    vec![
        // 0x00..0x10 slack: the buffer's byte index IS the address (`Code` has
        // base 0), so the body has to sit at the addresses the reloc map names.
        0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, //
        0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, //
        0x89, 0xC8, //                 0x10  mov eax, ecx  (a real body: the block has one)
        0x48, 0x83, 0xEC, 0x28, //     0x12  sub rsp, 0x28 (the shadow frame)
        0xE8, 0x00, 0x00, 0x00, 0x00, // 0x16  call rel32 (reloc at 0x17 -> abort)
        0xB8, 0x90, 0x01, 0x00, 0x00, // 0x1b  mov eax, 0x190 - unreachable
        0xC3, //                       0x20  ret           - unreachable
    ]
}

fn nr_relocs() -> RelocMap {
    // One relocation: the `call` at 0x16 names `abort`, and `U abort` is the
    // object's own statement that it has no address for that symbol.
    relocs_of(&nr_rows())
}

fn table_with(sig: CalleeSig, key: &str) -> HashMap<String, CalleeSig> {
    [(key.to_string(), sig)].into_iter().collect()
}

fn lift_nr(table: &HashMap<String, CalleeSig>) -> (Function, rdec_arch_x86::lift::Report) {
    let owned = nr_bytes();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: HashMap<usize, (Vec<u64>, Vec<u64>)> = HashMap::new();
    seeds.insert(0, (vec![0x10], vec![]));
    // The container's relocations go to discovery as well: the site is recorded
    // as an EXTERNAL call (the name is known, the code is not) and the sweep
    // keeps going, because a `call` returns as far as the bytes say. It is the
    // LIFTER's declared model of the callee that ends the BLOCK, not the sweep.
    let plan =
        discover_all_with_container(&ranges, &seeds, &Tables::default(), &disp_of(&nr_rows()));
    lift_one_sigs(
        "nr",
        &owned,
        0,
        0x10,
        owned.len() as u64,
        &plan,
        &names(),
        &FnAbi::integer(),
        table,
        &nr_relocs(),
        &no_tables(),
    )
}

fn abort_declared() -> CalleeSig {
    CalleeSig {
        cname: "abort".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: true,
    }
}

#[test]
fn a_call_to_a_declared_noreturn_callee_ends_the_block_and_defines_nothing() {
    let (mut f, rep) = lift_nr(&table_with(abort_declared(), "abort"));
    clean(&mut f);
    verify_ssa(&f).expect("a function ending in a noreturn call is valid SSA");

    // (1) counted, and named: `noreturn calls: N`, with the site and the callee.
    assert_eq!(rep.noreturn_calls, 1, "one noreturn call modelled");
    assert_eq!(
        rep.noreturn_sites,
        vec![(0x16u64, "abort".to_string())],
        "the site address and the symbol the relocation named"
    );
    assert_eq!(rep.tail_calls, 0, "a call is not a tail call");

    // (2) the terminator: no successor, and the block's last act is the call.
    let bb = f
        .blocks
        .iter()
        .find(|b| matches!(b.term, Terminator::NoReturn))
        .expect("the block ends in NoReturn, not a fall-through into an Exit");
    let mut succ = Vec::new();
    bb.term.successors(&mut succ);
    assert!(succ.is_empty(), "a noreturn exit has no successor");
    assert_eq!(bb.id, 0, "the noreturn call is in the entry block");
    assert_eq!(rep.exits, 0, "no Exit: this control flow is modelled");
    assert!(
        matches!(bb.instrs.last().map(|i| &i.op), Some(Op::Call { .. })),
        "the block's last instruction is the call:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );

    // (3) the call defines NOTHING: void type, so a statement, and there is no
    // answer value for anything to read.
    let call = bb.instrs.last().unwrap();
    match &call.op {
        Op::Call { ret_ty, callee, .. } => {
            assert_eq!(format!("{ret_ty:?}"), "Void", "no answer defined");
            assert_eq!(
                format!("{callee:?}"),
                "Symbol(\"abort\")",
                "the callee is the symbol the relocation named"
            );
        }
        other => panic!("the last op is a call, not {other:?}"),
    }
    assert_eq!(f.ty_of(call.dest), Some(Ty::Void));

    // (4) the code after the call is unreachable.
    assert!(
        !f.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i.op, Op::Const(c) if c == 0x190)),
        "code after a noreturn call reached the IR:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .all(|i| i.addr <= 0x16),
        "an IR node records an address past the noreturn call"
    );

    // (5) the C: the call is a statement, the end is `__builtin_unreachable()`,
    // and there is NO `return` - the function has no fall-through answer.
    let out = match rdec_core::emit_c::emit_c_sig(&f, &Protos::new()) {
        Ok(o) => o,
        Err(rs) => panic!(
            "a noreturn-ending function must emit C, refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        ),
    };
    assert!(
        out.text
            .lines()
            .any(|l| l.trim().starts_with("extern void abort(")),
        "(a) the prototype says void:\n{}",
        out.text
    );
    assert!(
        out.text.lines().any(|l| {
            let t = l.trim();
            t.contains("abort(") && t.ends_with(';') && !t.starts_with("extern")
        }),
        "(b) the call is a statement:\n{}",
        out.text
    );
    assert!(
        out.text.contains("__builtin_unreachable()"),
        "(c) the end is unreachable, not a value:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("return "),
        "(d) a noreturn function has no return:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("0x190"),
        "(e) the unreachable tail leaked into the C:\n{}",
        out.text
    );

    // (f) `-Werror`, with every warning the floor gate uses.
    // `__builtin_unreachable` is what makes a body with no `return`
    // warning-free, and the corpus has no noreturn call, so no gate would ever
    // have compiled this shape.
    let dir = std::env::temp_dir().join("rdec_tail_calls_nr");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let c = dir.join("nr.c");
    std::fs::write(&c, format!("{}\n{}", rdec_core::emit_c::PRELUDE, out.text)).expect("write C");
    // The flag set is the floor gate's (`m4::compile`), spelled out here so the
    // check cannot drift from what CI compiles the emitted C with.
    let build = std::process::Command::new("clang")
        .args([
            "-std=c17",
            "-c",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
            &c.display().to_string(),
            "-o",
            &dir.join("nr.o").display().to_string(),
        ])
        .output()
        .expect("clang runs");
    assert!(
        build.status.success(),
        "clang -Werror rejects the noreturn TU:\n{}\n{}",
        String::from_utf8_lossy(&build.stderr),
        out.text
    );
}

#[test]
fn a_noreturn_call_answers_with_a_fault_naming_the_cause_never_a_value() {
    let (mut f, _) = lift_nr(&table_with(abort_declared(), "abort"));
    clean(&mut f);
    let ex = run_program_budget(&f, &[5, 7, 0, 0], &[], 20000);
    match ex.outcome {
        Outcome::Fault(Fault::NoReturn) => {}
        other => panic!(
            "a function that ends in abort() has no answer; got {other:?} from\n{}",
            rdec_core::fmt_ir::format_function(&f)
        ),
    }
    // The call itself is an observable effect, so a pass that drops it is caught.
    assert!(
        ex.effects
            .iter()
            .any(|e| matches!(e, rdec_core::interp::Effect::Call(n, _) if n == "abort")),
        "the abort call is in the effects: {:?}",
        ex.effects
    );
}

#[test]
fn the_verifier_refuses_a_noreturn_terminator_that_no_call_earned() {
    // `Terminator::NoReturn` claims the block's last act was a call to a callee
    // that never returns. An `add` earns nothing.
    let mut f = Function::new("fake_noreturn", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Const(1), Ty::u64(), 0);
    let b = f.emit(e, Op::Const(2), Ty::u64(), 0);
    f.emit(e, Op::Bin(rdec_core::ir::BinOp::Add, a, b), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::NoReturn;
    f.ret_ty = Ty::i64();
    let errs = verify_ssa(&f).expect_err("a noreturn exit after an add is the IR lying");
    assert!(
        errs.iter().any(|s| s.contains("noreturn")),
        "the verifier must name the shape: {errs:?}"
    );

    // An empty block ending in NoReturn: the same lie, shorter.
    let mut g = Function::new("empty_noreturn", 0);
    g.block_mut(g.entry).term = Terminator::NoReturn;
    g.ret_ty = Ty::i64();
    let errs = verify_ssa(&g).expect_err("a noreturn exit with no instruction is a lie");
    assert!(
        errs.iter().any(|s| s.contains("noreturn")),
        "the verifier must name the shape: {errs:?}"
    );

    // The well-formed shape verifies: a call, then NoReturn.
    let mut h = Function::new("ok_noreturn", 0);
    let e = h.entry;
    let p = h.emit(e, Op::Param, Ty::u64(), 0);
    h.params.push(p);
    h.emit(
        e,
        Op::Call {
            callee: Callee::Symbol("abort".into()),
            ret_ty: Ty::Void,
            args: vec![p].into_boxed_slice(),
            pure: false,
        },
        Ty::Void,
        0,
    );
    h.block_mut(e).term = Terminator::NoReturn;
    h.ret_ty = Ty::i64();
    verify_ssa(&h).expect("a real noreturn call verifies");
}

#[test]
fn the_same_bytes_without_a_declared_noreturn_callee_keep_the_loud_call_shape() {
    // `noreturn` is a declaration, not a reading of the code. Undeclared, this
    // call keeps exactly the shape it always had: a relocation naming a symbol
    // the object does not define, so no address, so the loud bucket, so the
    // block's Exit. Nothing is invented to make it comparable.
    let undeclared = CalleeSig {
        cname: "abort".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: false,
    };
    let (f, rep) = lift_nr(&table_with(undeclared, "abort"));
    assert_eq!(rep.noreturn_calls, 0, "no declaration, no noreturn model");
    assert!(rep.noreturn_sites.is_empty());
    assert_eq!(rep.tail_calls, 0);
    assert_eq!(
        rep.calls_reloc_external, 1,
        "the import census still counts it"
    );
    assert_eq!(rep.exits, 1, "and the block keeps its Exit");
    assert!(
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::NoReturn)),
        "no NoReturn invented:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
}

#[test]
fn the_noreturn_table_names_the_termination_family_and_nothing_else() {
    // The table is the whole front-end claim, so it is tested on both sides: the
    // CRT/C++ termination family in, an ordinary callee out. A callee wrongly
    // marked noreturn loses its caller's answer; one wrongly left out keeps an
    // Exit, which is loud.
    for n in [
        "abort",
        "_abort",
        "exit",
        "_exit",
        "_Exit",
        "quick_exit",
        "__fastfail",
        "_CxxThrowException",
        "?_Xbad_alloc@std@@YAXXZ",
        "?_Xlength_error@std@@YAXXZ",
        "?terminate@std@@YAXXZ",
    ] {
        assert!(rdec_arch_x86::lift::noreturn_symbol(n), "{n} never returns");
    }
    for n in ["h", "f", "printf", "g", "malloc", "_alloca", "memcpy"] {
        assert!(
            !rdec_arch_x86::lift::noreturn_symbol(n),
            "{n} returns: marking it noreturn would delete its caller's answer"
        );
    }
}
