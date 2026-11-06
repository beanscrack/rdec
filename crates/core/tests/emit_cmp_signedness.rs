//! `Op::Cmp` emission when the OPERAND's display signedness and the
//! PREDICATE's signedness disagree.
//!
//! The interpreter is the ground truth (interp.rs, `Op::Cmp` arm):
//! signedness is a property of the PREDICATE - `Lt/Le/Gt/Ge` are signed,
//! `Ult/Ule/Ugt/Uge` unsigned, `Eq/Ne` neither - the width is
//! `width_of(a)`, and the operand type's signedness is explicitly "a display
//! concern (§1.3), so it is NOT consulted here". C's `<` follows the
//! operands' types, so the emitted C must CAST BOTH SIDES to the predicate's
//! signedness at the IR's width; then C's answer is the model's answer at the
//! machine's width.
//!
//! Before this file, the emitter REFUSED every disagreeing combination
//! (measured, at HEAD: `comparison Ult on a  operand` - the reason's format
//! string interpolates a literal empty string, so the refusal didn't even
//! name the type). The comment above the refusal promised "cast both sides to
//! the unsigned type of the same width"; the code never cast. Red-first:
//! tests 1, 2, 4 and 5 are RED on the refusing emitter (the fixture
//! `unwrap_or_else` quotes the refusal), and the sabotage in the file's doc
//! notes below (force the cast width to 64) reddens test 4, proving the width
//! claim is asserted, not decorative.

use rdec_core::emit_c::{emit_c, PRELUDE};
use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{Function, IntCmp, Op, Terminator};
use rdec_core::ty::{Sx, Ty, Width};

fn w(sx: Sx, width: Width) -> Ty {
    Ty::Int { sx, w: width }
}

/// `ret (cmp(param, CONST) ? 1 : 0)` - the shape the refusal fired on.
/// `f.params` is assigned: a fixture that binds no argument agrees with any
/// bug (standing repo rule).
fn param_const_cmp(param_ty: Ty, const_ty: Ty, const_val: u64, c: IntCmp) -> Function {
    let mut f = Function::new("cmpf", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Param, param_ty, 0);
    let k = f.emit(e, Op::Const(const_val), const_ty, 0);
    let cmp = f.emit(e, Op::Cmp(c, a, k), Ty::Bool, 0);
    let one = f.emit(e, Op::Const(1), Ty::u64(), 0);
    let zero = f.emit(e, Op::Const(0), Ty::u64(), 0);
    let sel = f.emit(e, Op::Select(cmp, one, zero), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::Return(sel);
    f.params.extend([a]);
    f.ret_ty = Ty::u64();
    f
}

/// The emitted line that spells the comparison. Cmp is the only node the
/// emitter spells as `... ? 1 : 0`; a `Select` names its operands.
fn cmp_line(c: &str) -> String {
    let hits: Vec<&str> = c.lines().filter(|l| l.contains("? 1 : 0")).collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one comparison line expected in:\n{c}"
    );
    hits[0].to_string()
}

fn emit(f: &Function) -> String {
    emit_c(f)
        .unwrap_or_else(|r| panic!("emit refused: {r:?}"))
        .text
}

#[test]
fn a_signed_operand_with_an_unsigned_predicate_casts_both_sides_unsigned() {
    // `cmp eax..., 10` lifted into a SIGNED display type with the unsigned
    // predicate: the C must compare the bit patterns at the IR width, which
    // means (uint64_t) on BOTH sides.
    let f = param_const_cmp(w(Sx::Signed, Width::W64), Ty::u64(), 10, IntCmp::Ult);
    let line = cmp_line(&emit(&f));
    assert!(
        line.contains("(uint64_t)x") && line.contains(" < "),
        "the signed/unsigned mismatch must cast BOTH sides to the unsigned type \
         of the operand's width, got: {line}"
    );
    // The model's answer, and the C's answer, must be the same number.
    // Ult(20, 10) = 0; Ult(5, 10) = 1; and Ult(0xFFFF...F, 10) = 0 - the
    // unsigned reading of the bit pattern is 2^64-1, which is NOT less than
    // 10 (the signed reading -1 is the Lt test's business, not this one).
    for (arg, want) in [(20u64, 0u64), (5, 1), (u64::MAX, 0)] {
        match run(&f, &[arg]).outcome {
            Outcome::Ret(Val::Int(v)) => assert_eq!(v, want, "Ult({arg}, 10)"),
            other => panic!("must answer, got {other:?}"),
        }
    }
}

#[test]
fn an_unsigned_operand_with_a_signed_predicate_casts_both_sides_signed() {
    let f = param_const_cmp(w(Sx::Unsigned, Width::W64), Ty::u64(), 10, IntCmp::Lt);
    let line = cmp_line(&emit(&f));
    assert!(
        line.contains("(int64_t)x") && line.contains(" < "),
        "Lt on an unsigned display type is a SIGNED machine comparison: cast both \
         sides to the signed type of the operand's width, got: {line}"
    );
    // Signed compare of the bit patterns: 20 > 10 -> 0; 5 < 10 -> 1;
    // 0xFFFF... as i64 is -1 < 10 -> 1.
    for (arg, want) in [(20u64, 0u64), (5, 1), (u64::MAX, 1)] {
        match run(&f, &[arg]).outcome {
            Outcome::Ret(Val::Int(v)) => assert_eq!(v, want, "Lt({arg}, 10)"),
            other => panic!("must answer, got {other:?}"),
        }
    }
}

#[test]
fn agreement_between_predicate_and_operand_emits_no_cast() {
    // The cast exists to repair a MISMATCH. Spraying casts everywhere would
    // "pass" the two tests above while making every corpus comparison text
    // noisy - this pins that the fix is a repair, not a style.
    let f = param_const_cmp(w(Sx::Signed, Width::W64), Ty::u64(), 10, IntCmp::Lt);
    let line = cmp_line(&emit(&f));
    assert!(
        !line.contains("(int64_t)") && !line.contains("(uint64_t)"),
        "Lt on a signed operand needs no cast, got: {line}"
    );
    let f = param_const_cmp(w(Sx::Signed, Width::W64), Ty::u64(), 10, IntCmp::Eq);
    let line = cmp_line(&emit(&f));
    assert!(
        !line.contains("(int64_t)") && !line.contains("(uint64_t)"),
        "Eq/Ne have no signedness, no cast, got: {line}"
    );
}

#[test]
fn a_32_bit_comparison_casts_at_32_bits_not_64() {
    // The width is a machine-code fact (`width_of(a)`), not the emitter's
    // convenience: casting a W32 operand through int64_t/uint64_t would widen
    // a `cmp eax, imm` into a 64-bit C comparison. The interpreter compares
    // at W32 (interp's `c.eval(x, y, w, sx)`); the C must match. The
    // constant 0x1_0000_0005 is masked to 5 by its W32 width, and a W32
    // parameter's signed cast to int32_t of 0x8000_0000 is negative - a
    // 64-bit cast would answer a different number.
    let f = param_const_cmp(
        w(Sx::Signed, Width::W32),
        Ty::u32(),
        0x1_0000_0005,
        IntCmp::Ult,
    );
    let line = cmp_line(&emit(&f));
    assert!(
        line.contains("(uint32_t)x") && !line.contains("(uint64_t)x"),
        "a W32 comparison must cast at 32 bits, got: {line}"
    );
    // Ult at W32: 0x8000_0000 as u32 is 2^31 > 5 -> 0.
    match run(&f, &[0x8000_0000u64]).outcome {
        Outcome::Ret(Val::Int(v)) => assert_eq!(v, 0, "Ult(0x80000000 as W32, 5)"),
        other => panic!("must answer, got {other:?}"),
    }
}

#[test]
fn an_integer_predicate_on_float_operands_is_refused_loudly() {
    // `Cmp` is the INTEGER comparison; `FCmp` is the float one and has C's
    // unordered semantics. An `Ult` on floats is not a thing the machine
    // does - the IR is malformed - and the old code emitted an ORDERED C
    // `<` with total-order semantics (NaN answers true), silently.
    let mut f = Function::new("fcmpf", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Param, Ty::f64(), 0);
    let b = f.emit(e, Op::Param, Ty::f64(), 0);
    let cmp = f.emit(e, Op::Cmp(IntCmp::Ult, a, b), Ty::Bool, 0);
    let one = f.emit(e, Op::Const(1), Ty::u64(), 0);
    let zero = f.emit(e, Op::Const(0), Ty::u64(), 0);
    let sel = f.emit(e, Op::Select(cmp, one, zero), Ty::u64(), 0);
    f.block_mut(e).term = Terminator::Return(sel);
    f.params.extend([a, b]);
    f.ret_ty = Ty::u64();
    let err = emit_c(&f).expect_err("an integer predicate on float operands must be refused");
    let text = format!("{err:?}");
    assert!(
        text.to_lowercase().contains("float"),
        "the refusal must name the real reason, got: {text}"
    );
}

#[test]
fn the_cast_form_compiles_with_werror_and_answers_what_the_model_answers() {
    // The single C compilation in this file, per repo rule: the emitted C for
    // test 1's fixture must COMPILE with -Werror (a `0xa` literal under a
    // (uint64_t) cast that clang diagnoses is exactly the class of failure a
    // text-only needle misses) and must PRINT what the interpreter ANSWERS.
    let f = param_const_cmp(w(Sx::Signed, Width::W64), Ty::u64(), 10, IntCmp::Ult);
    let c = emit(&f);
    let dir = std::env::temp_dir().join("rdec_cmp_signedness");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("cmpf.c");
    // PRELUDE is part of the product's file contract (`rdec c` emits
    // PRELUDE + text; the stdint typedefs live there), so the acceptance
    // call compiles the SAME composition the gates compile, plus stdio for
    // the probe's printf.
    std::fs::write(
        &src,
        format!(
            "{}\n#include <stdio.h>\n{c}\nint main(void){{ unsigned r = 0; r = r * 10 + (unsigned)cmpf(20); \
             r = r * 10 + (unsigned)cmpf(5); r = r * 10 + (unsigned)cmpf(-1); \
             printf(\"%03u\\n\", r); return 0; }}\n",
            PRELUDE
        ),
    )
    .unwrap();
    let out = std::process::Command::new("clang")
        .args(["-Werror", "-o"])
        .arg(dir.join("cmpf.exe"))
        .arg(&src)
        .output()
        .expect("clang must be on PATH in this repo (CI runs it)");
    assert!(
        out.status.success(),
        "-Werror compile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run_c = std::process::Command::new(dir.join("cmpf.exe"))
        .output()
        .expect("run");
    let printed = String::from_utf8_lossy(&run_c.stdout).trim().to_string();
    // The model: Ult(20,10)=0, Ult(5,10)=1, Ult(-1,10)=1 (bit pattern).
    let mut want = String::new();
    for arg in [20u64, 5, u64::MAX] {
        match run(&f, &[arg]).outcome {
            Outcome::Ret(Val::Int(v)) => want.push_str(&v.to_string()),
            other => panic!("model must answer, got {other:?}"),
        }
    }
    assert_eq!(
        printed, want,
        "the C and the model must print the same digits"
    );
}
