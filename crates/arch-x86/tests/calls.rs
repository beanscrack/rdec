// A DIRECT CALL to a target whose identity we know, modelled as a call — not as
// a pile of register clobbers.
//
// The old model was honest and useless: a `call` produced the call plus one
// `Op::Opaque("clobbered by call")` per caller-saved register INCLUDING RAX, so
// every function that called a sibling and used the answer was refused. That
// refusal class is the biggest one in both corpora, and it is the reason no
// function that calls a decompiled sibling emits C today.
//
// The model here: the call DEFINES its outputs (RAX for a declared integer
// return, the XMM0 cell for a declared float return — the CALLEE's declared
// `RetKind`, never a reading of what the code did) and CLOBBERS the rest of the
// caller-saved set. Not one clobber disappears: a caller-saved register read
// after the call still resolves to a clobber, and the emitter still refuses it.
// The difference is that the call's own answer is now a value, so a function
// that uses it compiles.
//
// Everything that is NOT a direct, known, typable call stays exactly as loud as
// it was: indirect call, target with no COFF symbol, a callee whose declared
// shape the front end could not type. The counts are in the Report so the gate
// prints them.
//
// Two functions in one buffer, hand-assembled bytes, no corpus dependency.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::{
    lift_one_sigs, no_relocs, no_tables, ArgKind, CalleeSig, FnAbi, Lifter, RetKind,
};
use rdec_core::emit_c::{callee_decl, emit_c_sig, Protos};
use rdec_core::interp::{run_program_budget, Fault, Outcome, Val};
use rdec_core::ir::{Function, Op, Terminator};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

/// `g(int)` = `lea eax,[rcx+1]; ret` and `f(int,int)` = spill `b`, `call g`,
/// `add eax, [rsp+8]`, `ret` — the shape a compiler emits for
/// `int f(int a, int b) { return g(a) + b; }` at -O0: `b` is spilled BECAUSE the
/// call destroys the register that held it, which is the fact the clobber model
/// exists to record.
fn bytes_int_pair() -> Vec<u8> {
    vec![
        // 0x00  g: lea eax, [rcx+1] ; ret
        0x8D, 0x41, 0x01, 0xC3, //
        // 0x04  f:
        0x89, 0x54, 0x24, 0x08, //   mov  [rsp+8], edx      ; spill b
        0xE8, 0xF3, 0xFF, 0xFF, 0xFF, // call 0x0            ; g(a)  (next = 13, target 0)
        0x03, 0x44, 0x24, 0x08, //   add  eax, [rsp+8]      ; + b
        0xC3, //                     ret
    ]
}

fn lift_pair(
    which: usize,
    abi: &FnAbi,
    sigs: &std::collections::HashMap<String, CalleeSig>,
) -> (Function, rdec_arch_x86::lift::Report) {
    let owned = bytes_int_pair();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64, 4u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 2, "discovery finds g and f");
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "g".to_string());
    names.insert(4, "f".to_string());
    let start = plan.functions[which];
    let end = plan
        .functions
        .get(which + 1)
        .copied()
        .unwrap_or(owned.len() as u64);
    lift_one_sigs(
        if which == 0 { "g" } else { "f" },
        &owned,
        0,
        start,
        end,
        &plan,
        &names,
        abi,
        sigs,
        &no_relocs(),
        &no_tables(),
    )
}

/// The callee table the pipeline hands the lifter: g is a sibling of this object
/// whose declared shape is the integer model, and whose C name in the generated
/// translation unit is `g`.
fn g_table() -> std::collections::HashMap<String, CalleeSig> {
    [(
        "g".to_string(),
        CalleeSig {
            cname: "g".to_string(),
            abi: FnAbi::integer(),
            typed: true,
            // `g` RETURNS: its caller reads the answer. A callee wrongly marked
            // noreturn loses its caller's answer, so the flag is spelled out at
            // every declaration site rather than defaulted.
            noreturn: false,
        },
    )]
    .into_iter()
    .collect()
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

fn refusals_of(f: &Function) -> Vec<String> {
    refusals_with(f, &Protos::new())
}

fn refusals_with(f: &Function, protos: &Protos) -> Vec<String> {
    match emit_c_sig(f, protos) {
        Ok(_) => vec![],
        Err(rs) => rs.iter().map(|r| r.reason.clone()).collect(),
    }
}

/// The pipeline's `m4::protos_of` for this two-function object: the generated
/// translation unit declares `g`, because `g`'s definition is in the same file.
/// A refusal test that hands the emitter an EMPTY table is measuring "no
/// prototype was built", not the gap it names — the clobber tests use this.
fn g_protos() -> Protos {
    let (mut g, _) = lift_pair(0, &FnAbi::integer(), &g_table());
    clean(&mut g);
    [("g".to_string(), callee_decl(&g))].into_iter().collect()
}

// ---------------------------------------------------------------------------
// THE REQUIREMENT, as one test: f's TU declares g, calls it, and says nothing
// about a clobber.
// ---------------------------------------------------------------------------
#[test]
fn a_direct_call_to_a_sibling_declares_prototypes_calls_and_does_not_refuse() {
    let (mut g, _) = lift_pair(0, &FnAbi::integer(), &g_table());
    clean(&mut g);
    let (mut f, rep) = lift_pair(1, &FnAbi::integer(), &g_table());
    clean(&mut f);
    verify_ssa(&f).expect("the calling function is valid SSA");
    assert_eq!(rep.calls_modelled, 1, "one direct call modelled");
    assert_eq!(rep.calls_indirect, 0);
    assert_eq!(rep.calls_no_symbol, 0);
    assert_eq!(rep.calls_untypable, 0);

    // The IR: a call op whose target identity is the callee's NAME.
    let call = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Call { .. }))
        .expect("the call survives the passes (it is not pure)");
    match &call.op {
        Op::Call {
            callee,
            args,
            ret_ty,
            ..
        } => {
            assert_eq!(
                format!("{callee:?}"),
                "Symbol(\"g\")",
                "the callee's identity is its name, not an address"
            );
            assert_eq!(args.len(), 4, "the declared shape's four integer args");
            assert_eq!(
                format!("{ret_ty}"),
                "int64_t",
                "the callee's integer answer"
            );
        }
        _ => unreachable!(),
    }

    // The C, as the pipeline concatenates it: prototypes for every callee, then
    // the functions.
    let mut protos: Protos = Protos::new();
    protos.insert("g".to_string(), callee_decl(&g));
    let out_f = match emit_c_sig(&f, &protos) {
        Ok(o) => o,
        Err(rs) => panic!(
            "f must emit C, refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        ),
    };
    let text = out_f.text;
    // (a) a PROTOTYPE for g, spelled from the same shape as its definition:
    // g's return is `uint32_t` because that is the type the lifter gave the
    // value g returns, and the prototype comes from that same `Function`.
    assert!(
        text.lines()
            .any(|l| l.starts_with("extern uint32_t g(uint64_t, uint64_t, uint64_t, uint64_t);")),
        "(a) no prototype for g in:\n{text}"
    );
    // (b) a REAL CALL, its result carried at the call's declared type.
    assert!(
        text.lines()
            .any(|l| l.contains("= (int64_t)g(") || l.contains("= g(")),
        "(b) no call to g in:\n{text}"
    );
    // (c) the refusal class that used to own this function is gone.
    assert!(
        !text.contains("clobbered by call"),
        "(c) the clobber refusal text leaked into the C:\n{text}"
    );
    // The call's arguments are the callee's declared register PAIRS, in declared
    // order: f's four parameter cells.
    let call_line = text
        .lines()
        .find(|l| l.contains("g(") && l.contains("= "))
        .unwrap_or("");
    assert!(
        call_line.contains("x0") && call_line.contains("x1"),
        "the args are the caller's cells for RCX/RDX: {call_line}"
    );

    // g's own definition is compatible with the prototype the caller declared:
    // the same spelling, from the same function.
    let out_g = emit_c_sig(&g, &Protos::new()).expect("g emits");
    assert!(
        out_g.text.contains("uint32_t g(uint64_t a0 "),
        "g's definition: {}",
        out_g.text
    );

    // The interpreter runs the call against the SAME IR: g is not a guessed
    // answer, it is the callee's own body. f(5, 7) = g(5) + 7 = 13.
    let exec = run_program_budget(&f, &[5, 7, 0, 0], &[g.clone()], 20000);
    match exec.outcome {
        Outcome::Ret(v) => assert_eq!(v.bits(), 13, "g(5) + 7"),
        other => panic!("f must return, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The clobbers do not vanish: a caller-saved register LIVE after the call is
// still a refusal, because leaving it live is the silent miscompile.
// ---------------------------------------------------------------------------
#[test]
fn a_register_live_across_the_call_is_still_a_refusal() {
    // g, then `mov eax, ecx; ret` AFTER the call: ECX is caller-saved, so the
    // answer is a value the call destroyed.
    let owned: Vec<u8> = vec![
        0x8D, 0x41, 0x01, 0xC3, // 0x00 g: lea eax,[rcx+1]; ret
        0xE8, 0xF7, 0xFF, 0xFF, 0xFF, // 0x04 f: call 0x0
        0x89, 0xC8, //       0x09    mov eax, ecx
        0xC3, //             0x0B    ret
    ];
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64, 4u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "g".to_string());
    names.insert(4, "f".to_string());
    let (mut f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        4,
        owned.len() as u64,
        &plan,
        &names,
        &FnAbi::integer(),
        &g_table(),
        &no_relocs(),
        &no_tables(),
    );
    clean(&mut f);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.calls_modelled, 1, "the call is modelled");
    assert!(rep.clobbers >= 1, "the clobbers are still counted");
    let r = refusals_with(&f, &g_protos());
    assert!(
        r.iter().any(|s| s == "opaque: clobbered by call"),
        "reading ECX after the call must refuse with the class it always used, got {r:?}\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
}

// ---------------------------------------------------------------------------
// Targets that are NOT direct, known, typable calls keep their loud shape.
// ---------------------------------------------------------------------------
#[test]
fn a_call_target_the_pipeline_gave_no_shape_for_stays_loud() {
    // The shape table is built from the object's symbol table, so in the
    // pipeline an entry there means "no COFF symbol at the target": an address,
    // not an identity. An empty table measures the same emitter path.
    let (mut f, rep) = lift_pair(1, &FnAbi::integer(), &std::collections::HashMap::new());
    clean(&mut f);
    assert_eq!(rep.calls_no_symbol, 1, "no COFF symbol at the target");
    assert_eq!(rep.calls_modelled, 0);
    // The callee is invisible, so RAX is clobbered, so the answer is refused.
    let r = refusals_of(&f);
    assert!(
        r.iter().any(|s| s == "opaque: clobbered by call"),
        "a call to an unnamed target must stay refused, got {r:?}"
    );
}

#[test]
fn a_callee_whose_shape_is_not_typed_stays_loud() {
    let mut t = g_table();
    t.get_mut("g").unwrap().typed = false;
    let (mut f, rep) = lift_pair(1, &FnAbi::integer(), &t);
    clean(&mut f);
    assert_eq!(rep.calls_untypable, 1, "the callee's shape is not typed");
    assert_eq!(rep.calls_modelled, 0);
    let r = refusals_with(&f, &g_protos());
    assert!(
        r.iter().any(|s| s == "opaque: clobbered by call"),
        "an untypable callee keeps the clobber refusal, got {r:?}"
    );
}

#[test]
fn an_indirect_call_stays_loud() {
    // `call *%rax` (FF D0), then `ret`.
    let owned: Vec<u8> = vec![0xFF, 0xD0, 0xC3];
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let (f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        0,
        owned.len() as u64,
        &plan,
        &std::collections::HashMap::new(),
        &FnAbi::integer(),
        &std::collections::HashMap::new(),
        &no_relocs(),
        &no_tables(),
    );
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.calls_indirect, 1);
    assert_eq!(rep.calls_modelled, 0);
    let has_indirect = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(&i.op, Op::Opaque(t) if t.starts_with("indirect call")));
    assert!(
        has_indirect,
        "the indirect call keeps its own reason string"
    );
}

// ---------------------------------------------------------------------------
// A declared float return: the answer is XMM0's cell, moved as BITS, and the
// prototype declares `double`.
// ---------------------------------------------------------------------------
#[test]
fn a_double_returning_call_passes_bits_and_declares_double() {
    // g: `addsd xmm0, xmm0; ret` = `double g(double d) { return d + d; }`
    // f: spill the argument, call g, `addsd xmm0, [rsp]`, ret.
    let owned: Vec<u8> = vec![
        0xF2, 0x0F, 0x58, 0xC0, 0xC3, // 0x00 g
        0xF2, 0x0F, 0x11, 0x04, 0x24, // 0x05 f: movsd [rsp], xmm0
        0xE8, 0xF1, 0xFF, 0xFF, 0xFF, // 0x0A    call 0x0 (next = 15, -15 -> 0)
        0xF2, 0x0F, 0x58, 0x04, 0x24, // 0x0F    addsd xmm0, [rsp]
        0xC3, //       0x14            ret
    ];
    let abi_g = FnAbi {
        args: vec![ArgKind::F64],
        ret: RetKind::F64,
        member: false,
    };
    let abi_f = FnAbi {
        args: vec![ArgKind::F64, ArgKind::F64],
        ret: RetKind::F64,
        member: false,
    };
    let sig_g = CalleeSig {
        cname: "g".to_string(),
        abi: abi_g.clone(),
        typed: true,
        noreturn: false, // a float-returning sibling: the caller reads XMM0
    };
    let sigs: std::collections::HashMap<String, CalleeSig> =
        [("g".to_string(), sig_g)].into_iter().collect();

    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64, 5u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "g".to_string());
    names.insert(5, "f".to_string());
    let (mut g, _) = lift_one_sigs(
        "g",
        &owned,
        0,
        0,
        5,
        &plan,
        &names,
        &abi_g,
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    let (mut f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        5,
        owned.len() as u64,
        &plan,
        &names,
        &abi_f,
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    clean(&mut g);
    clean(&mut f);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.calls_modelled, 1);

    let mut protos: Protos = Protos::new();
    protos.insert("g".to_string(), callee_decl(&g));
    let out = emit_c_sig(&f, &protos).unwrap_or_else(|rs| {
        panic!(
            "the float call emits, refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        )
    });
    assert!(
        out.text.contains("extern double g(double);"),
        "the prototype declares the DECLARED shape:\n{}",
        out.text
    );
    assert!(
        out.text.contains("rdec_bits_f64("),
        "a W64 cell feeding a double parameter moves BITS, never a truncating cast:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("clobbered by call"),
        "no clobber refusal text:\n{}",
        out.text
    );
    // The spill is f's OWN argument: `movsd [rsp], xmm0` saved the 3.0 it came
    // in, so f answers g(3.0) + 3.0 = 9.0 - the callee's body ran (3.0 + 3.0),
    // and the answer came back through the XMM0 cell, not through memory.
    let exec = run_program_budget(
        &f,
        &[3.0f64.to_bits(), 10.0f64.to_bits(), 0, 0],
        &[g.clone()],
        20000,
    );
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => assert_eq!(v, 9.0),
        other => panic!("expected a double answer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The interpreter answers from the callee's body, and says so when the callee
// is not in the model: a counted fault, never a guess.
// ---------------------------------------------------------------------------
#[test]
fn a_call_to_a_target_that_is_not_in_the_model_is_a_fault_not_a_guess() {
    let (mut g, _) = lift_pair(0, &FnAbi::integer(), &g_table());
    clean(&mut g);
    let (mut f, _) = lift_pair(1, &FnAbi::integer(), &g_table());
    clean(&mut f);
    // With g in the program, f returns.
    let with = run_program_budget(&f, &[5, 7, 0, 0], &[g.clone()], 20000);
    assert!(
        matches!(with.outcome, Outcome::Ret(_)),
        "{:?}",
        with.outcome
    );
    // Without it, the interpreter FAULTS at the call: it does not invent an
    // answer for a callee it cannot see.
    let without = run_program_budget(&f, &[5, 7, 0, 0], &[], 20000);
    match without.outcome {
        Outcome::Fault(Fault::UnknownCallee) => {}
        other => panic!("a missing callee must fault, got {other:?}"),
    }
    // And a register clobbered by the call is a fault, not a zero.
    let mut h = Function::new("h", 0);
    let e = h.entry;
    let v = h.emit(e, Op::Clobber, Ty::u64(), 0);
    h.block_mut(e).term = Terminator::Return(v);
    h.ret_ty = Ty::u64();
    let ex = run_program_budget(&h, &[], &[], 20000);
    match ex.outcome {
        Outcome::Fault(Fault::Clobbered(_)) => {}
        other => panic!("reading a clobbered register must fault, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The verifier validates the new op: it does not wave an ill-typed call through.
// ---------------------------------------------------------------------------
#[test]
fn the_verifier_checks_a_call_operands_and_returns() {
    // A Bool wired into a call's argument list: no C spelling, and the IR
    // claiming it is an argument.
    let mut f = Function::new("bad", 0);
    let e = f.entry;
    let c = f.emit(e, Op::Const(1), Ty::Bool, 0);
    f.emit(
        e,
        Op::Call {
            callee: rdec_core::ir::Callee::Symbol("g".into()),
            ret_ty: Ty::i64(),
            args: vec![c].into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    f.block_mut(e).term = Terminator::Return(rdec_core::ir::NO_VAL);
    let errs = verify_ssa(&f).expect_err("a Bool argument is not a register cell");
    assert!(
        errs.iter().any(|s| s.contains("call")),
        "the verifier must name the call: {errs:?}"
    );

    // A callee with no name: the identity IS the point of the op.
    let mut g = Function::new("noname", 0);
    let e = g.entry;
    g.emit(
        e,
        Op::Call {
            callee: rdec_core::ir::Callee::Symbol(String::new()),
            ret_ty: Ty::i64(),
            args: Box::new([]),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    g.block_mut(e).term = Terminator::Return(rdec_core::ir::NO_VAL);
    let errs = verify_ssa(&g).expect_err("an unnamed callee is a hole in the IR");
    assert!(
        errs.iter().any(|s| s.contains("callee")),
        "the verifier must name the callee: {errs:?}"
    );

    // A well-formed call verifies.
    let mut ok = Function::new("ok", 0);
    let e = ok.entry;
    let p = ok.emit(e, Op::Param, Ty::u64(), 0);
    ok.params.push(p);
    let v = ok.emit(
        e,
        Op::Call {
            callee: rdec_core::ir::Callee::Symbol("g".into()),
            ret_ty: Ty::i64(),
            args: vec![p].into_boxed_slice(),
            pure: false,
        },
        Ty::i64(),
        0,
    );
    let bits = ok.emit(
        e,
        Op::Cast(rdec_core::ir::CastKind::Bitcast, Ty::u64(), v),
        Ty::u64(),
        0,
    );
    ok.block_mut(e).term = Terminator::Return(bits);
    ok.ret_ty = Ty::u64();
    verify_ssa(&ok).expect("a well-formed call verifies");
}

// ---------------------------------------------------------------------------
// The lifter's own claim, checked at the IR: the call defines RAX and clobbers
// the rest, and `Ty::Flags` is never a call's output.
// ---------------------------------------------------------------------------
#[test]
fn the_call_defines_its_answer_and_clobbers_the_rest() {
    let (f, rep) = lift_pair(1, &FnAbi::integer(), &g_table());
    verify_ssa(&f).expect("valid SSA");
    let mut calls = 0;
    let mut clobbers = 0;
    for b in &f.blocks {
        for i in &b.instrs {
            match &i.op {
                Op::Call { ret_ty, .. } => {
                    calls += 1;
                    assert_eq!(format!("{ret_ty}"), "int64_t");
                }
                Op::Clobber => {
                    clobbers += 1;
                    assert_eq!(format!("{}", i.ty), "uint64_t", "a register cell");
                }
                Op::Opaque(t) => assert!(
                    !t.starts_with("clobbered by call"),
                    "the clobber is a named op, not an opaque: {t}"
                ),
                _ => {}
            }
        }
    }
    assert_eq!(calls, 1, "one call op");
    assert_eq!(
        clobbers, rep.clobbers,
        "every clobber the report counts is in the IR"
    );
    // RAX is defined BY THE CALL, so it is not among the clobbers: 25
    // caller-saved registers minus the one the call defines.
    assert_eq!(clobbers, 24, "the defined output is not clobbered");
}

/// The entry-seeding window is unchanged by all of this: a call is a statement
/// about registers, not about parameters.
#[test]
fn lifting_a_caller_keeps_the_abi_seeding() {
    let names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    let l = Lifter::new_abi("f", 4, &names, &FnAbi::integer());
    let seeded = l.entry_regs().iter().filter(|(_, v)| v.is_some()).count();
    assert_eq!(seeded, 4, "the four integer argument registers");
}
