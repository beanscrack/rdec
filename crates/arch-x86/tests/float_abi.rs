// The Windows x64 FLOAT ABI, modelled from the DECLARED shape, pinned on
// hand-assembled bytes.
//
// The integer model seeds RCX/RDX/R8/R9 as the four parameters and answers
// `ret` from RAX. The float ABI splits the arguments by register class:
// integer parameters consume RCX/RDX/R8/R9 and float parameters consume
// XMM0-3, IN PARALLEL — `int f(int a, double b)` arrives with `a` in RCX and
// `b` in XMM0, and `b` does NOT consume RDX. The kinds arrive per function as
// a `FnAbi` (the C++ pipeline builds it from the demangled signature; the
// lifter never reads the code to guess it), and a declared float return
// answers `ret` from XMM0's cell.
//
// A float parameter's IR value is the register's W64 CELL (the register file
// is one unsigned word per register); its DECLARED type (`param_tys`) is what
// the emitted C spells, and the assignment moves bits through the memcpy
// helpers — a `(uint64_t)d` conversion would truncate the answer to zero and
// make every float decompilation a lie.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::{lift_one, lift_one_sig, ArgKind, FnAbi, Grp, Lifter, RetKind};
use rdec_core::emit_c::emit_c;
use rdec_core::interp::{run_budget, Outcome, Val};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

fn lift_abi(bytes: &[u8], abi: &FnAbi) -> (rdec_core::ir::Function, rdec_arch_x86::lift::Report) {
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
    lift_one_sig("f", &owned, 0, start, end, &plan, &Default::default(), abi)
}

/// (a) `double f(double x) { return x + x; }` = `addsd xmm0, xmm0; ret`.
/// Lifted under the declared shape [F64] -> F64, the IR entry defines the
/// parameter from XMM0's cell, `ret` answers from it, and the emitted C
/// declares `double` on both sides.
#[test]
fn a_double_in_and_out_lifts_emits_and_interprets() {
    let bytes: &[u8] = &[
        0xF2, 0x0F, 0x58, 0xC0, // addsd xmm0, xmm0
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::F64],
        ret: RetKind::F64,
        member: false,
    };
    let (f, rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(f.params.len(), 1, "one declared parameter");
    assert_eq!(f.param_tys, vec![Ty::f64()], "the DECLARED type is double");
    assert!(matches!(f.ret_ty, Ty::Float { .. }), "ret_ty is float");
    assert_eq!(rep.float_params, 1, "the shape seeds one float parameter");
    assert_eq!(rep.ret_f64, 1, "the declared F64 return is counted");
    // The machine answer: 3.5 + 3.5 = 7.0, through the XMM0 cell.
    let exec = run_budget(&f, &[3.5f64.to_bits()], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => assert_eq!(v, 7.0),
        other => panic!("unexpected: {other:?}"),
    }
    // The C: declared double in, declared double out, bits moved, not
    // converted (`(uint64_t)d` would be the lie this whole model avoids).
    let out = emit_c(&f).expect("the float function emits");
    let sig = out
        .text
        .lines()
        .find(|l| l.starts_with("double f("))
        .unwrap_or_else(|| panic!("no double signature in:\n{}", out.text));
    assert!(
        sig.contains("double a0"),
        "the parameter is DECLARED double: {sig}"
    );
    assert!(
        out.text.contains("rdec_f64_bits(a0)"),
        "the entry moves the parameter's BITS:\n{}",
        out.text
    );
    assert!(
        out.text.contains("return (double)"),
        "the answer is returned as the declared double:\n{}",
        out.text
    );
}

/// (b) `int f(int a, double b)` = `mov eax, ecx; ret` with shape [Int, F64].
/// The float argument consumes XMM0, NOT an integer slot: the entry defines
/// RCX and XMM0, and leaves RDX/R8/R9/XMM1-3 undefined. The same bytes under
/// the integer model define all four integer registers — that model says
/// nothing about who is a parameter, which is exactly why it stays the shape
/// of every symbol without a declared signature.
#[test]
fn a_float_argument_consumes_no_integer_slot() {
    let names: std::collections::HashMap<u64, String> = Default::default();
    let abi = FnAbi {
        args: vec![ArgKind::Int, ArgKind::F64],
        ret: RetKind::Int,
        member: false,
    };
    let l = Lifter::new_abi("f", 0, &names, &abi);
    let regs = l.entry_regs();
    let defined =
        |g: Grp| regs.iter().find(|(k, _)| *k == g).map(|(_, v)| v.is_some()) == Some(true);
    assert!(defined(Grp::Rcx), "`int a` arrives in RCX");
    assert!(defined(Grp::Xmm(0)), "`double b` arrives in XMM0");
    assert!(
        !defined(Grp::Rdx),
        "the float parameter does NOT consume RDX — the int f(int,double) rule"
    );
    assert!(!defined(Grp::R8) && !defined(Grp::R9), "no other seeds");
    assert!(
        !defined(Grp::Xmm(1)),
        "only the declared float registers are parameters"
    );
    // The integer model, for contrast: every one of RCX/RDX/R8/R9 is a
    // parameter, because nothing declared otherwise.
    let i = Lifter::new("f", 0, &names);
    assert!(i
        .entry_regs()
        .iter()
        .filter(|(g, v)| v.is_some() && *g != Grp::Rsp)
        .all(|(g, _)| matches!(g, Grp::Rcx | Grp::Rdx | Grp::R8 | Grp::R9)));
}

/// (b, the counterfactual) `mov eax, edx; ret` — "return b" under the
/// integer model. Under the DECLARED shape [Int, F64] that second parameter
/// is a double, RDX is not a parameter, and the read of RDX is the
/// materialised `undef` — LOUD in the emitted C, where the integer model
/// reads a real parameter. A declared shape is a constraint, not a decoration.
#[test]
fn reading_an_undeclared_integer_register_under_a_declared_shape_is_undef() {
    let bytes: &[u8] = &[
        0x89, 0xD0, // mov  eax, edx
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::Int, ArgKind::F64],
        ret: RetKind::Int,
        member: false,
    };
    let (f, _rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA — undef is a value, not a graph defect");
    let out = emit_c(&f).expect("emits, with the undef LOUD in the text");
    assert!(
        out.text.contains("undef"),
        "the read of a non-parameter RDX is the materialised undef, said out loud:\n{}",
        out.text
    );
    // Same bytes, integer model: `return b` with b in RDX, emitted.
    let (g, _r) = {
        let owned = bytes.to_vec();
        let ranges = [Code {
            base: 0,
            bytes: owned.as_slice(),
        }];
        let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
            std::collections::HashMap::new();
        seeds.insert(0, (vec![0u64], vec![]));
        let plan = discover_all(&ranges, &seeds);
        let start = plan.functions[0];
        let end = owned.len() as u64;
        lift_one("g", &owned, 0, start, end, &plan, &Default::default())
    };
    emit_c(&g).expect("under the integer model this is `return b`");
}

/// (b2) The MEMBER rule, machine-verified by the corpus bytes: a
/// `__thiscall` callee spends slot 0 — RCX *and* XMM0 — on `this`, so the
/// first declared `double` arrives in XMM1. Under the paired seeding
/// [Int(this), F64] with `member: true`, RCX and XMM1 are entry definitions
/// and XMM0/RDX are not. (Free functions keep the per-class rule; the test
/// above pins it.)
#[test]
fn a_members_first_double_arrives_in_xmm1() {
    let names: std::collections::HashMap<u64, String> = Default::default();
    let abi = FnAbi {
        args: vec![ArgKind::Int, ArgKind::F64],
        ret: RetKind::F64,
        member: true,
    };
    let regs = Lifter::new_abi("m", 0, &names, &abi).entry_regs();
    let defined =
        |g: Grp| regs.iter().find(|(k, _)| *k == g).map(|(_, v)| v.is_some()) == Some(true);
    assert!(defined(Grp::Rcx), "this = RCX");
    assert!(
        defined(Grp::Xmm(1)),
        "the member's first declared double = XMM1 (the corpus grow saves it from there)"
    );
    assert!(
        !defined(Grp::Xmm(0)),
        "slot 0 belongs to `this`, pair and all"
    );
    assert!(!defined(Grp::Rdx), "no integer beyond this");
}

/// Every parameter class past its registers is loud: five declared integers
/// found no fifth register, the gap is counted and visible in the IR, and no
/// parameter is invented.
#[test]
fn a_signature_over_the_register_budget_refuses_instead_of_guessing() {
    let bytes: &[u8] = &[0xC3]; // ret
    let abi = FnAbi {
        args: vec![ArgKind::Int; 5],
        ret: RetKind::Int,
        member: false,
    };
    let (f, rep) = lift_abi(bytes, &abi);
    assert_eq!(f.params.len(), 4, "the budget seeds four, not five");
    assert_eq!(rep.gaps, 1, "the fifth parameter is a counted gap");
    assert!(
        rdec_core::fmt_ir::format_function(&f).contains("beyond the register-class budget"),
        "the gap is visible in the IR, not hidden in a counter"
    );
}

/// A declared F32 return compares 32 bits, and the lifter says so in the IR:
/// the returned value is the cell's low 32 bits viewed as a float — the
/// `movss` the ABI caller uses zeroes 127:32, so the type's width is the
/// answer's width. `f32 f(void)` = `mov eax, 0x40000000; movd xmm0, eax; ret`.
#[test]
fn a_float_return_answers_from_xmm0_at_its_declared_width() {
    let bytes: &[u8] = &[
        0xB8, 0x00, 0x00, 0x80, 0x3F, // mov eax, 0x3F800000   (1.0f)
        0x66, 0x0F, 0x6E, 0xC0, // movd  xmm0, eax
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: Vec::new(),
        ret: RetKind::F32,
        member: false,
    };
    let (f, rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.ret_f32, 1);
    let out = emit_c(&f).expect("emits");
    assert!(
        out.text.starts_with("float f(") || out.text.contains("\nfloat f("),
        "the C declares the declared return type float:\n{}",
        out.text
    );
    let exec = run_budget(&f, &[0, 0, 0, 0], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => assert_eq!(v, 1.0, "the 32-bit float pattern"),
        other => panic!("unexpected: {other:?}"),
    }
}
