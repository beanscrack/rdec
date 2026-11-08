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
            // The default four pairs: this hand-built table carries no body
            // evidence, so no arity is recovered (see `CalleeSig::nargs`).
            nargs: 4,
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
        nargs: 4,
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
    // RAX is defined BY THE CALL, so it is not among the clobbers. 2026-11-06
    // root-cause move 24 -> 40: the pair model (M8 packed slice) made each XMM
    // register TWO caller-saved cells (lo+hi) — an IR node defines one value,
    // so the IR count is cells: 9 GPR + 32 XMM cells - the defined output.
    // 2026-11-08 Win64-arity slice moves 40 -> 38: the volatile integer set is
    // the Windows x64 ABI's SEVEN (RAX, RCX, RDX, R8-R11) — RSI and RDI are
    // nonvolatile there (the 9-GPR list was SysV's answer), so 7 GPR + 32 XMM
    // cells - the defined output. The register-level ABI fact is what the
    // comment above states; clobbering the high lane is the point of the
    // model, not a side effect.
    assert_eq!(clobbers, 38, "the defined output is not clobbered");
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

// ---------------------------------------------------------------------------
// ARITY, fixture (a): the `call_helper` shape. The ground truth is the corpus
// source `corpus/src/call_helper.c`, quoted:
//
//     int h(int x) { return x * x + 1; }
//     int f(int a, int b) { return h(a) + h(b); }
//
// compiled `clang -O1`, which keeps `b` in RSI (a Win64 NONVOLATILE register:
// the callee preserves it) and rewrites only RCX between the two calls:
// RDX, the second argument cell, is NOT rewritten between them. The 4-pair
// model made the second call site read RDX to "pass" a fourth argument the
// callee never reads; RDX is destroyed by the FIRST call, so the emitter
// refused the function with `opaque: clobbered by call` - a refusal on a
// machine-dead argument cell. Hand-assembled -O1 shape:
fn bytes_call_helper() -> Vec<u8> {
    vec![
        // 0x00  h: imul ecx, ecx ; lea eax, [rcx+1] ; ret   =  x*x + 1
        0x0F, 0xAF, 0xC9, 0x8D, 0x41, 0x01, 0xC3, 0x90, //
        // 0x08  f(a, b):
        0x53, // 0x08  push rbx
        0x56, // 0x09  push rsi
        0x48, 0x83, 0xEC, 0x20, // 0x0A  sub rsp, 0x20
        0x89, 0xD6, // 0x0E  mov  esi, edx      ; b -> RSI (RDX read #1)
        0xE8, 0xEB, 0xFF, 0xFF, 0xFF, // 0x10  call 0x0 ; h(a)  (next 0x15)
        0x89, 0xC3, // 0x15  mov  ebx, eax      ; h(a) -> RBX (nonvolatile)
        0x89, 0xF1, // 0x17  mov  ecx, esi      ; RCX = RSI (RSI read AFTER call #1)
        0xE8, 0xE2, 0xFF, 0xFF, 0xFF, // 0x19  call 0x0 ; h(b)  (next 0x1E; RDX NOT rewritten)
        0x01, 0xD8, // 0x1E  add  eax, ebx      ; h(b) + h(a)
        0x48, 0x83, 0xC4, 0x20, // 0x20  add rsp, 0x20
        0x5B, // 0x24  pop rbx
        0x5E, // 0x25  pop rsi
        0xC3, // 0x26  ret
    ]
}

#[test]
fn a_one_argument_callee_makes_both_call_sites_read_one_cell() {
    let owned = bytes_call_helper();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64, 8u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "h".to_string());
    names.insert(8, "f".to_string());
    // The callee table the TWO-PASS front end hands the lifter: `h`'s own
    // lifted body reads ONE argument cell (`x * x + 1` reads x), so its
    // signature carries `nargs: 1`, the value `Function::args_read` computes
    // from the body lifted below - the fixture asserts that equality, so the
    // hand-built table cannot drift from the rule the pipeline recovers.
    let sig_h = CalleeSig {
        cname: "h".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: false,
        nargs: 1,
    };
    let sigs: std::collections::HashMap<String, CalleeSig> =
        [("h".to_string(), sig_h)].into_iter().collect();
    let (mut h, _) = lift_one_sigs(
        "h",
        &owned,
        0,
        0,
        8,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    let (mut f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        8,
        owned.len() as u64,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    assert_eq!(h.args_read(), 1, "h reads only its first argument cell");
    assert_eq!(rep.calls_modelled, 2, "both calls are modelled");
    clean(&mut h);
    clean(&mut f);
    // f reads a (to pass to h) and b (the `mov esi, edx`) - and nothing of
    // R8/R9: the arity the two-pass stamps on f's OWN signature.
    assert_eq!(f.args_read(), 2);
    // The two-pass stamps these (crates/cli/src/m4.rs); the fixture mirrors
    // the stamped state so the emitter and the C harness see what pass 2 emits.
    h.nparams = Some(1);
    f.nparams = Some(2);
    verify_ssa(&f).expect("valid SSA");
    let mut protos: Protos = Protos::new();
    protos.insert("h".to_string(), callee_decl(&h));
    let out = emit_c_sig(&f, &protos).unwrap_or_else(|rs| {
        panic!(
            "the second call site reads ONE cell, so f emits; refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        )
    });
    assert!(
        out.text.contains("extern uint32_t h(uint64_t);"),
        "the prototype declares the RECOVERED arity (one parameter):\n{}",
        out.text
    );
    assert!(
        !out.text.contains(", 0, 0, 0)"),
        "the call passes ONE argument, not the 4-pair spelling:\n{}",
        out.text
    );
    assert!(
        !out.text.contains("clobbered by call"),
        "no clobber refusal text:\n{}",
        out.text
    );
    // Ground truth, from the C source quoted above:
    // f(3, 4) = h(3) + h(4) = (3*3+1) + (4*4+1) = 10 + 17 = 27.
    let exec = run_program_budget(&f, &[3, 4, 0, 0], &[h.clone()], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v, 27, "f(3,4) = h(3) + h(4) = 27"),
        other => panic!("expected the answer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// ARITY, fixture (b): the `fib_rec` shape. Ground truth, the corpus source
// `corpus/src/fib_rec.c` (`long f(int n)` computing Fibonacci), compiled
// -O1: the recursion carries `n` in RSI across the self-call and accumulates
// in RDI - both NONVOLATILE - and the self-call passes ONE cell. Pre-fix the
// loop-carried RDX/R8/R9 cells (destroyed by the previous iteration's call)
// sat in the call's argument list and the function refused. Hand-assembled
// from the -O1 disassembly:
//
//     f(5):  acc=0, n=5 >= 2: acc += f(4)=3, n=3
//            n=3 >= 2:       acc += f(2)=1 -> 4, n=1
//            n=1 <  2: return acc + n = 4 + 1 = 5
//     f(4):  acc += f(3)=2, n=2; acc += f(1)=1 -> 3, n=0; return 3.
//     f(3):  acc += f(2)=1, n=1; return 1+1 = 2.
//     f(2):  acc += f(1)=1, n=0; return 1.   f(1)=1, f(0)=0.
fn bytes_fib_rec() -> Vec<u8> {
    vec![
        0x56, // 0x00  push rsi
        0x57, // 0x01  push rdi
        0x48, 0x83, 0xEC, 0x28, // 0x02  sub rsp, 0x28
        0x89, 0xCE, // 0x06  mov esi, ecx       ; n -> RSI (RSI read)
        0x31, 0xFF, // 0x08  xor edi, edi       ; acc = 0
        0x83, 0xFE, 0x02, // 0x0A  cmp esi, 2
        0x7C, 0x13, // 0x0D  jl  0x22
        0x90, // 0x0F  nop
        0x8D, 0x4E, 0xFF, // 0x10  lea ecx, [rsi-1]   ; n-1 -> RCX
        0xE8, 0xE8, 0xFF, 0xFF, 0xFF, // 0x13  call 0x0 ; f(n-1)  (next 0x18; self)
        0x01, 0xC7, // 0x18  add edi, eax       ; acc += answer (RAX defined BY the call)
        0x83, 0xC6, 0xFE, // 0x1A  add esi, -2        ; n -= 2 (RSI read AFTER the call)
        0x83, 0xFE, 0x02, // 0x1D  cmp esi, 2
        0x7D, 0xEE, // 0x20  jge 0x10
        0x01, 0xFE, // 0x22  add edi, esi       ; acc + n
        0x89, 0xF0, // 0x24  mov eax, esi       ; answer = acc + n
        0x48, 0x83, 0xC4, 0x28, // 0x26  add rsp, 0x28
        0x5F, // 0x2A  pop rdi
        0x5E, // 0x2B  pop rsi
        0xC3, // 0x2C  ret
    ]
}

#[test]
fn a_recursive_callee_reads_its_own_first_cell_only() {
    let owned = bytes_fib_rec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "f".to_string());
    // RECURSION IS COVERED BY THE CALLEE'S OWN BODY: `f`'s pass-1 lift is the
    // evidence for `f`'s calls to itself, and a self-forwarded cell is not a
    // read (`Function::args_read`), so the arity drops to the ONE cell the
    // body reads. The hand-built table carries that recovered value.
    let sig_f = CalleeSig {
        cname: "f".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: false,
        nargs: 1,
    };
    let sigs: std::collections::HashMap<String, CalleeSig> =
        [("f".to_string(), sig_f)].into_iter().collect();
    let (mut f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        0,
        owned.len() as u64,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    assert_eq!(rep.calls_modelled, 1, "the self-call is modelled");
    clean(&mut f);
    assert_eq!(
        f.args_read(),
        1,
        "the loop-carried RDX reaches only SELF-call argument slots: forwarding to self is not reading"
    );
    f.nparams = Some(1);
    verify_ssa(&f).expect("valid SSA");
    let mut protos: Protos = Protos::new();
    protos.insert("f".to_string(), callee_decl(&f));
    let out = emit_c_sig(&f, &protos).unwrap_or_else(|rs| {
        panic!(
            "the loop-carried call reads ONE cell, so f emits; refused with {:?}\n{}",
            rs.iter().map(|r| &r.reason).collect::<Vec<_>>(),
            rdec_core::fmt_ir::format_function(&f)
        )
    });
    assert!(
        !out.text.contains("clobbered by call"),
        "no clobber refusal text:\n{}",
        out.text
    );
    // The interpreter runs the recursion; the derivation is in the comment on
    // `bytes_fib_rec`: f(5)=5, f(4)=3, f(2)=1, f(0)=0 (Fibonacci).
    let callee = f.clone();
    for (n, want) in [(5u64, 5u64), (4, 3), (3, 2), (2, 1), (1, 1), (0, 0)] {
        let exec = run_program_budget(&f, &[n, 0, 0, 0], std::slice::from_ref(&callee), 20000);
        match exec.outcome {
            Outcome::Ret(Val::Int(v)) => assert_eq!(v, want, "f({n})"),
            other => panic!("f({n}) must answer, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ANTI-INVENTION: a callee with NO visible body keeps the four pairs. The
// arity comes from a lifted body and from nothing else: an import, a thunk,
// any name the pipeline cannot see into, stays at the loud 4-pair default
// (over-estimating refuses loudly; under-estimating reads a cell the caller
// never passed). Asserted on the IR the lifter emits, not on prose.
#[test]
fn a_callee_with_no_recovered_arity_keeps_four_argument_reads() {
    let owned = bytes_int_pair();
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
    // `g` is TYPED (the integer model) with NO body evidence: nargs stays 4.
    let sig = CalleeSig {
        cname: "g".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: false,
        nargs: 4,
    };
    let sigs: std::collections::HashMap<String, CalleeSig> =
        [("g".to_string(), sig)].into_iter().collect();
    let (g, _) = lift_one_sigs(
        "g",
        &owned,
        0,
        0,
        4,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    let (f, _) = lift_one_sigs(
        "f",
        &owned,
        0,
        4,
        owned.len() as u64,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    let call = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Call { .. }))
        .expect("f has a call");
    match &call.op {
        Op::Call { args, .. } => assert_eq!(args.len(), 4, "the 4-pair default stands"),
        _ => unreachable!(),
    }
    let decl = callee_decl(&g);
    assert_eq!(
        decl.params.len(),
        4,
        "the declaration keeps the four seeded cells: {} params",
        decl.params.len()
    );
    assert!(
        decl.params.iter().all(|p| *p == Ty::u64()),
        "the four cells are the integer model's, undeclared-arity intact: {:?}",
        decl.params
    );
}

// ---------------------------------------------------------------------------
// The MAX-INDEX rule at the call site: a callee that reads TWO argument cells
// gets TWO arguments at every call site - never one (a leading-reads count
// would give 1 if the body's reads were sparse), never four. A body that
// reads cell 1 and NOT cell 0 needs two cells: `max(read index) + 1`.
#[test]
fn a_two_cell_callee_gets_exactly_two_arguments() {
    // g(a, b): `imul ecx, edx; lea eax, [rcx+1]; ret` = x*y + 1 - reads
    // RCX (cell 0) and RDX (cell 1). f(a, b, c, d): `call g; ret` passes
    // exactly the two cells g reads.
    let owned: Vec<u8> = vec![
        0x0F, 0xAF, 0xCA, // 0x00  g: imul ecx, edx
        0x8D, 0x41, 0x01, // 0x03       lea eax, [rcx+1]
        0xC3, // 0x06       ret
        // 0x07  f:
        0xE8, 0xF4, 0xFF, 0xFF, 0xFF, // 0x07  call 0x0 (next 0x0C, target 0)
        0xC3, // 0x0C       ret
    ];
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64, 7u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    names.insert(0, "g".to_string());
    names.insert(7, "f".to_string());
    let sig = CalleeSig {
        cname: "g".to_string(),
        abi: FnAbi::integer(),
        typed: true,
        noreturn: false,
        nargs: 2,
    };
    let sigs: std::collections::HashMap<String, CalleeSig> =
        [("g".to_string(), sig)].into_iter().collect();
    let (g, _) = lift_one_sigs(
        "g",
        &owned,
        0,
        0,
        7,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    let (mut f, _) = lift_one_sigs(
        "f",
        &owned,
        0,
        7,
        owned.len() as u64,
        &plan,
        &names,
        &FnAbi::integer(),
        &sigs,
        &no_relocs(),
        &no_tables(),
    );
    assert_eq!(g.args_read(), 2, "imul reads cells 0 and 1");
    clean(&mut f);
    let call = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Call { .. }))
        .expect("f calls g");
    match &call.op {
        Op::Call { args, .. } => {
            assert_eq!(args.len(), 2, "TWO arguments - max index 1 plus one");
            assert_eq!(args[0], f.params[0], "argument 0 is the caller's RCX cell");
            assert_eq!(
                args[1], f.params[1],
                "argument 1 is the caller's RDX cell, not an undef"
            );
        }
        _ => unreachable!(),
    }
}
