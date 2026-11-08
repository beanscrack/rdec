// The Windows x64 FLOAT ABI, modelled from the DECLARED shape, pinned on
// hand-assembled bytes.
//
// The integer model seeds RCX/RDX/R8/R9 as the four parameters and answers
// `ret` from RAX. The float ABI homes each parameter by REGISTER PAIR:
// declared parameter `k` lives in the pair that owns RCX/RDX/R8/R9[k] AND
// XMM0-3[k], so `int f(int a, double b)` arrives with `a` in RCX (pair 0) and
// `b` in XMM1 (pair 1). A class does not consume the other class's registers —
// `b` spends no integer register and `a` no XMM — which is why the PAIR
// POSITION, not a per-class counter, is the index into both register arrays.
// A member's `this` spends pair 0 in both classes, so a member's first
// `double` is XMM1 for the same reason a free `f(int,double)`'s is. The kinds
// arrive per function as a `FnAbi` (the C++ pipeline builds it from the
// demangled signature; the lifter never reads the code to guess it), and a
// declared float return answers `ret` from XMM0's cell.
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
/// The pair rule homes `a` in pair 0 (RCX) and `b` in pair 1 — XMM1, not XMM0.
/// Neither parameter touches the other class: the entry defines RCX and XMM1
/// and leaves RDX (pair 1's integer half, spent on a double) and XMM0 (pair 0's
/// float half, spent on an int) undefined. The same bytes under the integer
/// model define all four integer registers — that model says nothing about who
/// is a parameter, which is exactly why it stays the shape of every symbol
/// without a declared signature.
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
    assert!(defined(Grp::Rcx), "`int a` arrives in RCX (pair 0)");
    assert!(defined(Grp::Xmm(1)), "`double b` arrives in XMM1 (pair 1)");
    assert!(
        !defined(Grp::Rdx),
        "the float parameter does NOT consume RDX — the int f(int,double) rule"
    );
    assert!(!defined(Grp::R8) && !defined(Grp::R9), "no other seeds");
    assert!(
        !defined(Grp::Xmm(0)),
        "pair 0's float half is spent on the `int a`: XMM0 is not a parameter"
    );
    assert!(
        !defined(Grp::Xmm(2)),
        "only the declared pairs are parameters"
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
/// first declared `double` arrives in XMM1. Under the pairing
/// [Int(this), F64] with `member: true`, RCX and XMM1 are entry definitions
/// and XMM0/RDX are not. It is the SAME rule the free test above pins: `this`
/// is declared parameter 0, so it spends pair 0, and the `double` at position 1
/// lands in pair 1. The `member` flag says `this` is there; the pair rule says
/// where everything lands.
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

// ---- THE PAIR RULE FOR FREE FUNCTIONS -----------------------------------
//
// The machine fact, established with `llvm-objdump -d` of a harness driver and
// a runtime probe (commit 0b4b2dd): Windows x64 homes declared parameter `k`
// in the register PAIR that owns RCX/RDX/R8/R9[k] AND XMM0-3[k]. A free
// `int f(int a, double b)` therefore receives `b` in XMM1, and opens with
// `movsd %xmm1,0x8(%rsp)` — the same first instruction, for the same reason,
// that a member's first `double` opens with (its `this` spends pair 0).
//
// Indexing the float registers by the parameter's index WITHIN the float class
// (XMM0, XMM1, ... for the 1st, 2nd float) seeds a register the caller did not
// deliver that parameter in: the parameter's value comes from a register that
// holds something else. That is a miscompile, not a spelling — and the corpus
// has no mixed free signature, so none of its 1364 functions observes it. These
// two shapes are pinned on bytes so the rule is a test, not a paragraph.

/// The witness, `int f(int a, double b)` consuming BOTH parameters:
///
/// ```text
/// cvtsi2sd xmm0, ecx     F2 0F 2A C1    xmm0 = (double)a
/// addsd    xmm0, xmm1    F2 0F 58 C1    xmm0 = (double)a + b     <- pair 1
/// cvttsd2si eax, xmm0    F2 0F 2C C0    eax = (int)((double)a + b)
/// ret                    C3
/// ```
///
/// The IR must seed parameter 1 (the declared `double`) from XMM1's cell and
/// parameter 0 (the `int`) from RCX, and XMM0 — pair 0's float half, spent on
/// the int — must NOT be a parameter. The value follows the home: 3 + 2.5 = 5.5
/// truncates to 5. Seeded from XMM0, the `addsd` reads the materialised `undef`
/// (0.0 in the interpreter, LOUD in the emitted C) and the answer is 3: the
/// miscompile, caught by the shape AND by the answer.
#[test]
fn a_free_double_at_pair_one_is_seeded_from_xmm1_and_its_value_follows() {
    let bytes: &[u8] = &[
        0xF2, 0x0F, 0x2A, 0xC1, // cvtsi2sd xmm0, ecx
        0xF2, 0x0F, 0x58, 0xC1, // addsd  xmm0, xmm1
        0xF2, 0x0F, 0x2C, 0xC0, // cvttsd2si eax, xmm0
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::Int, ArgKind::F64],
        ret: RetKind::Int,
        member: false,
    };
    let names: std::collections::HashMap<u64, String> = Default::default();
    let (f, rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(f.params.len(), 2, "two declared parameters");
    assert_eq!(
        f.param_tys,
        vec![Ty::u64(), Ty::f64()],
        "declared `int a`, declared `double b`"
    );
    assert_eq!(rep.float_params, 1, "one declared float parameter");
    assert_eq!(rep.gaps, 0, "both parameters found a register pair");
    // Which entry register carries which declared parameter.
    let home_of = |g: Grp| {
        Lifter::new_abi("f", 0, &names, &abi)
            .entry_regs()
            .into_iter()
            .find(|(k, _)| *k == g)
            .and_then(|(_, v)| v)
            .and_then(|v| f.params.iter().position(|p| *p == v))
    };
    assert_eq!(
        home_of(Grp::Rcx),
        Some(0),
        "`int a` is parameter 0, seeded from RCX (pair 0)"
    );
    assert_eq!(
        home_of(Grp::Xmm(1)),
        Some(1),
        "`double b` is parameter 1, seeded from XMM1 (pair 1) — the machine delivers it there"
    );
    assert_eq!(
        home_of(Grp::Xmm(0)),
        None,
        "XMM0 is NOT a parameter: pair 0's float half is spent on `int a`"
    );
    assert_eq!(
        home_of(Grp::Rdx),
        None,
        "RDX is NOT a parameter: pair 1 is a double"
    );
    // Nothing in the body reads a register the signature did not declare — the
    // `addsd`'s source IS the seeded XMM1. (Seeded by class index, XMM1 is
    // undeclared and the read is counted as an undef site.)
    assert!(
        rep.undef_sites.is_empty(),
        "every read is a declared pair: {:?}",
        rep.undef_sites
    );
    // The answer, through the IR interpreter: only XMM1's delivery makes it 5.
    let exec = run_budget(&f, &[3, 2.5f64.to_bits()], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(
            v, 5,
            "(int)(3 + 2.5) = 5 — `b` arrived in the register the body reads"
        ),
        other => panic!("unexpected: {other:?}"),
    }
    let out = emit_c(&f).expect("emits");
    // The C spells the seeding: the declared double arrives as `a1` and its
    // BITS are what land in the register cell the `addsd` reads. (The entry
    // block does materialise the shared `undef` for the pairs the signature
    // spends on the other class — XMM0 here — and nothing reads it; that is
    // what `undef_sites.is_empty()` above measures.)
    let sig = out
        .text
        .lines()
        .find(|l| l.contains("f(uint64_t a0"))
        .unwrap_or_else(|| panic!("no f(uint64_t a0, ...) signature in:\n{}", out.text));
    assert!(
        sig.contains("double a1"),
        "parameter 1 is DECLARED double: {sig}"
    );
    assert!(
        out.text.contains("rdec_f64_bits(a1)"),
        "the entry moves the declared double's BITS into its cell:\n{}",
        out.text
    );
}

/// The shape that makes class-index seeding unmistakably wrong,
/// `int g(double a, int b, double c)`: pairs 0, 1, 2 are
/// `XMM0 / RDX / XMM2` — the integer is at position 1 (RDX, not RCX) and the
/// second double at position 2 (XMM2, not XMM1). Per-class counting spells
/// that `XMM0 / RCX / XMM1`: two of the three parameters seeded from the wrong
/// register.
///
/// ```text
/// cvtsi2sd xmm3, edx     F2 0F 2A DA    xmm3 = (double)b
/// addsd    xmm0, xmm2    F2 0F 58 C2    xmm0 = a + c              <- pair 2
/// addsd    xmm0, xmm3    F2 0F 58 C3    xmm0 = a + c + (double)b  <- pair 1
/// cvttsd2si eax, xmm0    F2 0F 2C C0    eax = (int)that
/// ret                    C3
/// ```
#[test]
fn a_three_parameter_mixed_signature_seeds_three_pairs() {
    let bytes: &[u8] = &[
        0xF2, 0x0F, 0x2A, 0xDA, // cvtsi2sd xmm3, edx
        0xF2, 0x0F, 0x58, 0xC2, // addsd  xmm0, xmm2
        0xF2, 0x0F, 0x58, 0xC3, // addsd  xmm0, xmm3
        0xF2, 0x0F, 0x2C, 0xC0, // cvttsd2si eax, xmm0
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::F64, ArgKind::Int, ArgKind::F64],
        ret: RetKind::Int,
        member: false,
    };
    let names: std::collections::HashMap<u64, String> = Default::default();
    let (f, rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(f.params.len(), 3, "three declared parameters");
    assert_eq!(
        f.param_tys,
        vec![Ty::f64(), Ty::u64(), Ty::f64()],
        "declared `double a`, `int b`, `double c`"
    );
    assert_eq!(rep.float_params, 2, "two declared float parameters");
    assert_eq!(rep.gaps, 0, "all three found a register pair");
    let home_of = |g: Grp| {
        Lifter::new_abi("g", 0, &names, &abi)
            .entry_regs()
            .into_iter()
            .find(|(k, _)| *k == g)
            .and_then(|(_, v)| v)
            .and_then(|v| f.params.iter().position(|p| *p == v))
    };
    assert_eq!(home_of(Grp::Xmm(0)), Some(0), "`double a` = pair 0 = XMM0");
    assert_eq!(home_of(Grp::Rdx), Some(1), "`int b` = pair 1 = RDX");
    assert_eq!(home_of(Grp::Xmm(2)), Some(2), "`double c` = pair 2 = XMM2");
    assert_eq!(
        home_of(Grp::Xmm(1)),
        None,
        "XMM1 is not a parameter — it is pair 1's float half, spent on the int"
    );
    assert_eq!(
        home_of(Grp::Rcx),
        None,
        "RCX is not a parameter — pair 0's int half is spent on a double"
    );
    assert_eq!(home_of(Grp::R8), None, "pair 3 is not declared");
    assert!(
        rep.undef_sites.is_empty(),
        "every read is a declared pair: {:?}",
        rep.undef_sites
    );
    // 1.5 + (-0.25) + 7 = 8.25 -> 8. Every term is needed: drop any one home and
    // the answer is 1, 8.25-without-c, or 8.25-without-b.
    let exec = run_budget(&f, &[1.5f64.to_bits(), 7, (-0.25f64).to_bits()], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v, 8, "(int)(1.5 + 7 + -0.25) = 8"),
        other => panic!("unexpected: {other:?}"),
    }
}

/// `cvtsi2sd xmm0, ecx` is a SIGNED integer -> double conversion, and the C it
/// emitted did not COMPILE. The emitter built the source cast out of the string
/// `"(int64_t)"` inside the template `({s})({x})`, which spells `((int64_t))(x)`,
/// a cast with no operand, and clang answers `expected expression`. No corpus
/// function performs a signed int -> float conversion, so no gate ever saw it.
/// this test is the witness, and the `((int64_t))(` negative assertion is the
/// exact shape of the bug.
///
/// The prefixes, recorded because the FIRST version of this test got them wrong:
/// `F2` is scalar DOUBLE (`cvtsi2sd`), `F3` is scalar SINGLE (`cvtsi2ss`). With
/// `F3` the lifter is right to produce a `float`, and the emitted C says `(float)`.
/// Reading the test's own bytes back from `llvm-objdump` is the only way to be
/// sure which instruction was assembled.
#[test]
fn a_signed_int_to_double_conversion_emits_c_that_clang_accepts() {
    let bytes: &[u8] = &[
        0xF2, 0x0F, 0x2A, 0xC1, // cvtsi2sd xmm0, ecx   (F2 = scalar double)
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::Int],
        ret: RetKind::F64,
        member: false,
    };
    let (f, rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.ret_f64, 1, "a declared double answer comes from XMM0");
    let out = emit_c(&f).expect("emits");
    assert!(
        out.text.contains("(double)((int32_t)("),
        "the signed source is cast at ITS OWN width: `cvtsi2sd xmm0, ecx` reads a \
         32-bit cell, so the C must say `(double)((int32_t)(x))`\n{}",
        out.text
    );
    assert!(
        !out.text.contains("((int64_t))("),
        "`((int64_t))(x)` is a cast with no operand; clang: expected expression\n{}",
        out.text
    );
    // The cast WIDTH is part of the conversion, which is why the pin above is
    // `int32_t` and not the widened `(int64_t)` this emitter used to spell: a
    // cell holding 0xCB01F9D0 read as `int64_t` is +3405904336 (the zero-extend
    // of a 32-bit unsigned), and the same cell read at its own width is
    // -889062960. The machine reads the second.
    let exec = run_budget(&f, &[5, 0, 0, 0], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => {
            assert_eq!(
                v, 5.0,
                "the interpreter converts the same value the C casts"
            )
        }
        other => panic!("unexpected: {other:?}"),
    }
    let exec = run_budget(&f, &[0xCB01_F9D0, 0, 0, 0], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => assert_eq!(
            v, -889_062_960.0,
            "a 32-bit signed source with the sign bit set converts NEGATIVE"
        ),
        other => panic!("unexpected: {other:?}"),
    }
}

/// The single-precision sibling, pinned next to it so the F2/F3 distinction is
/// in the test suite and not in a comment: `cvtsi2ss` converts to `float`, and
/// the emitted C must say `(float)`, with the value the machine's low 32 bits
/// carry - not a widened double.
#[test]
fn a_signed_int_to_single_conversion_converts_to_float_not_double() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x2A, 0xC1, // cvtsi2ss xmm0, ecx   (F3 = scalar single)
        0xC3, // ret
    ];
    let abi = FnAbi {
        args: vec![ArgKind::Int],
        ret: RetKind::F32,
        member: false,
    };
    let (f, _rep) = lift_abi(bytes, &abi);
    verify_ssa(&f).expect("valid SSA");
    let out = emit_c(&f).expect("emits");
    assert!(
        out.text.contains("(float)(float)((int32_t)(") && out.text.contains("rdec_f32_bits"),
        "the single-precision form must convert to float at the source's own 32-bit \
         width and store f32 bits:\n{}",
        out.text
    );
    // 16777217 is not representable in a float: the f32 conversion is observable,
    // and the interpreter must show it, not silently carry the double.
    let exec = run_budget(&f, &[16777217, 0, 0, 0], 20000);
    match exec.outcome {
        Outcome::Ret(Val::Fl(v)) => assert_eq!(v, 16777216.0, "the f32 rounding is modelled"),
        other => panic!("unexpected: {other:?}"),
    }
}
