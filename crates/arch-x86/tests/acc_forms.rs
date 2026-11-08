//! The x86 ACCUMULATOR forms: `cdq`/`cqo`, `div`/`idiv`, `mul`/`imul` with one
//! operand, and `imul r64,r/m64,imm32`. They are the shape a compiler emits for
//! every C `/` and `%` on signed or unsigned integers, so refusing them refuses
//! the function: 14 of the corpus's 42 refusals were `cdq`/`div`, 6 more `imul`.
//!
//! EVERY expected number in this file is a silicon measurement, not a
//! derivation. The byte sequences were executed on this host with VirtualAlloc
//! PAGE_EXECUTE_READWRITE and the arguments in ecx/edx/r8 (Win64), so the
//! machine — not this file's arithmetic — set the expectations:
//!
//! ```text
//! idiv32 (mov rax,rcx ; cdq ; idiv r8d)      a=-7 d=2 -> eax 0xFFFFFFFD   edx 0xFFFFFFFF
//!                                          a=7  d=-2 -> eax 0xFFFFFFFD   edx 0x00000001
//!                                          a=-1 d=3  -> eax 0            edx 0xFFFFFFFF
//!                                          a=3  d=-1 -> eax 0xFFFFFFFD   edx 0
//!                                          a=100 d=7 -> eax 14           edx 2
//! div32  (mov rax,rcx ; xor edx,edx ; div r8d)
//!                                          a=0xFFFFFFF9 d=2 -> eax 0x7FFFFFFC edx 1
//!                                          a=0xFFFFFFFF d=3 -> eax 0x55555555 edx 0
//!                                          a=0x9ABCDEF0 d=0x76543210 -> eax 1 edx 0x2468ACE0
//! imul32 one-operand (imul r8d)            a=-7 d=2 -> eax 0xFFFFFFF2 edx 0xFFFFFFFF
//!                                          a=0x12345678 d=0x9ABCDEF0 -> eax 0x242D2080 edx 0xF8CC93D6
//! mul32  one-operand (mul r8d)             a=0xFFFFFFF9 d=2 -> eax 0xFFFFFFF2 edx 1
//!                                          a=0xFFFFFFFF d=3 -> eax 0xFFFFFFFD edx 2
//! imul64 one-operand                       a=-7 d=2 -> rax -14  rdx -1
//! mul64  one-operand                       a=-2 d=0x100000001 -> rax 0xFFFFFFFDFFFFFFFE rdx 0x100000000
//! imul rax,rcx,2AAAAAAB                    rcx=-7 -> 0xFFFFFF FED5555553 ; rcx=3 -> 0x80000001
//! imul rax,rcx,92492493(imm32 as negative) rcx=-7 -> 0x2FFFFFFFB
//! cmp ecx,edx ; cdq ; setc al              (7, 0xFFFFFFFE) -> al=1 ; (0xFFFFFFF9, 2) -> al=0
//! ```
//!
//! The last row is the EFLAGS claim: `cdq` writes EDX and touches NO flag, and
//! the machine proves it — the CF set by `cmp` is read by `setc` two
//! instructions later, across the `cdq`. A lifter that treats cdq as
//! flag-clearing answers 0 for every vector and the fixture goes red on the
//! `al = 1` legs.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::{
    lift_one, lift_one_sigs, no_tables, CalleeTable, FnAbi, RelocMap, RelocTarget, Report,
};
use rdec_core::emit_c::emit_c;
use rdec_core::interp::{run_budget, Fault, Outcome};
use rdec_core::ir::Function;
use rdec_core::verify::verify_ssa;

fn lift(bytes: &[u8]) -> (Function, Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64; 4], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 1, "one discovered function");
    let start = plan.functions[0];
    let end = owned.len() as u64;
    let (mut f, rep) = lift_one("f", &owned, 0, start, end, &plan, &Default::default());
    for _ in 0..16 {
        let mut ch = false;
        ch |= rdec_core::fuse::flag_elimination(&mut f);
        ch |= rdec_core::passes::prune_phis(&mut f);
        ch |= rdec_core::passes::const_fold(&mut f);
        ch |= rdec_core::passes::copy_prop(&mut f);
        ch |= rdec_core::passes::simplify_branches(&mut f);
        ch |= rdec_core::passes::dead_store_elimination(&mut f);
        ch |= rdec_core::passes::dead_code_elimination(&mut f);
        if !ch {
            break;
        }
    }
    (f, rep)
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
            rdec_core::ir::Op::Opaque(t) => Some(t.to_string()),
            _ => None,
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// `mov rax,rcx ; cdq ; idiv r8d ; ret` and the remainder sibling.
///
/// The pair IS the dividend: `cdq` fills EDX with the sign of EAX, and the
/// 32-bit `idiv` divides the 64-bit signed EDX:EAX by the signed r/m32,
/// quotient -> EAX, remainder -> EDX, truncating TOWARD ZERO (the C rule, not
/// the floor rule): -7/2 = -3 remainder -1. A floor-dividing model answers -4
/// with remainder +1, and the two legs of the negative vector cannot agree.
#[test]
fn cdq_idiv32_divides_the_pair_and_truncates_toward_zero() {
    let q: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x99, // cdq
        0x41, 0xF7, 0xF8, // idiv r8d
        0xC3, // ret
    ];
    let r: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x99, // cdq
        0x41, 0xF7, 0xF8, // idiv r8d
        0x89, 0xD0, // mov eax,edx  (the remainder, W32 write zeroes the upper half)
        0xC3, // ret
    ];
    for (bytes, want) in [
        (
            q,
            vec![
                (0xFFFFFFFFFFFFFFF9u64, 2u64, 0xFFFFFFFDu64),
                (7, 0xFFFFFFFE, 0xFFFFFFFD),
                (0xFFFFFFFF, 3, 0),
                (3, 0xFFFFFFFF, 0xFFFFFFFD),
                (100, 7, 14),
            ],
        ),
        (
            r,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 0xFFFFFFFF),
                (7, 0xFFFFFFFE, 1),
                (0xFFFFFFFF, 3, 0xFFFFFFFF),
                (3, 0xFFFFFFFF, 0),
                (100, 7, 2),
            ],
        ),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(rep.gaps, 0, "cdq + idiv r/m32 is modelled");
        for (a, d, want) in want {
            match answer(&f, &[a, 0, d, 0]) {
                Outcome::Ret(v) => assert_eq!(
                    v.bits() & 0xFFFF_FFFF,
                    want,
                    "a={a:#x} d={d:#x}: the machine's answer, quoted in the header"
                ),
                other => panic!("a={a:#x} d={d:#x}: unexpected {other:?}"),
            }
        }
    }
}

/// `mov rax,rcx ; xor edx,edx ; div r8d ; ret` (+ the remainder sibling).
///
/// Unsigned: the zeroed high half is the idiom, and the dividend is read as
/// UNSIGNED, which is the whole difference from the signed form — 0xFFFFFFF9/2
/// is 0x7FFFFFFC remainder 1, not -3 remainder -1. One vector pins the
/// signedness on each side.
#[test]
fn xor_zero_then_div32_divides_unsigned() {
    let q: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x31, 0xD2, // xor edx,edx
        0x41, 0xF7, 0xF0, // div r8d
        0xC3, // ret
    ];
    let r: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x31, 0xD2, // xor edx,edx
        0x41, 0xF7, 0xF0, // div r8d
        0x89, 0xD0, // mov eax,edx
        0xC3, // ret
    ];
    for (bytes, want) in [
        (
            q,
            vec![
                (0xFFFFFFFFFFFFFFF9u64, 2u64, 0x7FFFFFFCu64),
                (0xFFFFFFFF, 3, 0x55555555),
                (0x123456789ABCDEF0, 0x76543210, 1),
                (100, 7, 14),
            ],
        ),
        (
            r,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 1),
                (0xFFFFFFFF, 3, 0),
                (0x123456789ABCDEF0, 0x76543210, 0x2468ACE0),
                (100, 7, 2),
            ],
        ),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(rep.gaps, 0, "xor + div r/m32 is modelled");
        for (a, d, want) in want {
            match answer(&f, &[a, 0, d, 0]) {
                Outcome::Ret(v) => assert_eq!(v.bits() & 0xFFFF_FFFF, want, "a={a:#x} d={d:#x}"),
                other => panic!("a={a:#x} d={d:#x}: unexpected {other:?}"),
            }
        }
    }
}

/// `mov rax,rcx ; imul r8d ; ret` and `mul r8d` — one operand, F7 /5 and /4.
///
/// The destination pair is IMPLICIT: RDX:EAX (EDX:EAX at 32 bits). `imul` is
/// signed, `mul` unsigned, and the two differ ONLY in the high half, which is
/// exactly the pair the corpus's `mul_wide` returns. `mul` of 0xFFFFFFF9 x 2
/// carries into EDX = 1; `imul` of the same bytes sign-extends to EDX =
/// 0xFFFFFFFF. A model that spells both with the same high half answers one of
/// the two legs wrong.
#[test]
fn one_operand_mul_imul32_fill_the_implicit_pair() {
    // `mov eax,edx` after the multiply returns the HIGH half, W32-written.
    let imul_lo: &[u8] = &[0x48, 0x89, 0xC8, 0x41, 0xF7, 0xE8, 0xC3];
    let imul_hi: &[u8] = &[0x48, 0x89, 0xC8, 0x41, 0xF7, 0xE8, 0x89, 0xD0, 0xC3];
    let mul_lo: &[u8] = &[0x48, 0x89, 0xC8, 0x41, 0xF7, 0xE0, 0xC3];
    let mul_hi: &[u8] = &[0x48, 0x89, 0xC8, 0x41, 0xF7, 0xE0, 0x89, 0xD0, 0xC3];
    for (bytes, want) in [
        (
            imul_lo,
            vec![
                (0xFFFFFFFFFFFFFFF9u64, 2u64, 0xFFFFFFF2u64),
                (0x12345678, 0x9ABCDEF0, 0x242D2080),
                (100, 7, 700),
            ],
        ),
        (
            imul_hi,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 0xFFFFFFFF),
                (0x12345678, 0x9ABCDEF0, 0xF8CC93D6),
                (100, 7, 0),
            ],
        ),
        (
            mul_lo,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 0xFFFFFFF2),
                (0xFFFFFFFF, 3, 0xFFFFFFFD),
                (100, 7, 700),
            ],
        ),
        (
            mul_hi,
            vec![(0xFFFFFFFFFFFFFFF9, 2, 1), (0xFFFFFFFF, 3, 2), (100, 7, 0)],
        ),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(rep.gaps, 0, "one-operand mul/imul r/m32 is modelled");
        for (a, d, want) in want {
            match answer(&f, &[a, 0, d, 0]) {
                Outcome::Ret(v) => assert_eq!(v.bits() & 0xFFFF_FFFF, want, "a={a:#x} d={d:#x}"),
                other => panic!("a={a:#x} d={d:#x}: unexpected {other:?}"),
            }
        }
    }
}

/// `mov rax,rcx ; imul r8` / `mul r8` at 64 bits — the corpus's `mul_wide`.
///
/// The 64-bit high half is a 128-bit product's top, which the model carries as
/// `MulSmul`/`MulHi`. The interpreter has both; the C emitter refused both
/// ("needs __int128"), which is why this function was refused end to end. The
/// second half of this test is the emission claim: a function whose answer IS
/// the high half must EMIT C, and clang must accept it.
#[test]
fn one_operand_mul_imul64_high_half_lifts_and_emits() {
    let imul_hi: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x49, 0xF7, 0xE8, // imul r8
        0x48, 0x89, 0xD0, // mov rax,rdx
        0xC3, // ret
    ];
    let mul_hi: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x49, 0xF7, 0xE0, // mul r8
        0x48, 0x89, 0xD0, // mov rax,rdx
        0xC3, // ret
    ];
    let imul_lo: &[u8] = &[0x48, 0x89, 0xC8, 0x49, 0xF7, 0xE8, 0xC3];
    for (bytes, want) in [
        (
            imul_hi,
            vec![
                (0xFFFFFFFFFFFFFFF9u64, 2u64, 0xFFFFFFFFFFFFFFFFu64),
                (0x123456789ABCDEF0, 0x76543210, 0x86A1C97),
                (0xFFFFFFFFFFFFFFFE, 0x100000001, 0xFFFFFFFFFFFFFFFF),
            ],
        ),
        (
            mul_hi,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 1),
                (0x123456789ABCDEF0, 0x76543210, 0x86A1C97),
                (0xFFFFFFFFFFFFFFFE, 0x100000001, 0x1_0000_0000),
            ],
        ),
        (
            imul_lo,
            vec![
                (0xFFFFFFFFFFFFFFF9, 2, 0xFFFFFFFFFFFFFFF2),
                (0x123456789ABCDEF0, 0x76543210, 0x530ECA7E5618CF00),
            ],
        ),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(
            rep.gaps,
            0,
            "the 64-bit one-operand form lifts: {:?}",
            opaque_texts(&f)
        );
        for (a, d, want) in want {
            match answer(&f, &[a, 0, d, 0]) {
                Outcome::Ret(v) => assert_eq!(v.bits(), want, "a={a:#x} d={d:#x}"),
                other => panic!("a={a:#x} d={d:#x}: unexpected {other:?}"),
            }
        }
        let out = emit_c(&f).expect("the high half emits C");
        assert!(
            !out.text.contains("/* refusal"),
            "the emitted C must not carry a refusal comment:\n{}",
            out.text
        );
    }
}

/// `imul rax,rcx,imm32` — the 3-operand form with the immediate SIGN-EXTENDED
/// to the operand size. The gap here was mechanical and invisible: iced reports
/// that immediate as `OpKind::Immediate32to64`, and `src` listed every other
/// immediate kind except that one, so every `imul` with a 32-bit immediate on a
/// 64-bit operand — clang's magic-number division — was refused.
#[test]
fn imul_imm32_sign_extends_to_the_operand_width() {
    let pos: &[u8] = &[0x48, 0x69, 0xC1, 0xAB, 0xAA, 0xAA, 0x2A, 0xC3];
    let neg: &[u8] = &[0x48, 0x69, 0xC1, 0x93, 0x24, 0x49, 0x92, 0xC3];
    for (bytes, want) in [
        (
            pos,
            vec![
                (0xFFFFFFFFFFFFFFF9u64, 0xFFFFFFFE_D5555553u64),
                (7, 0x1_2AAAAAAD),
                (3, 0x8000_0001),
                (0xFFFFFFFF, 0x2AAAAAAAD5555555),
            ],
        ),
        (
            neg,
            vec![
                (0xFFFFFFFFFFFFFFF9, 0x2_FFFFFFFB),
                (7, 0xFFFFFFFD_00000005),
                (0xFFFFFFFF, 0x92492493_6DB6DB6D),
            ],
        ),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(
            rep.gaps,
            0,
            "the imm32-to-64 form is modelled: {:?}",
            opaque_texts(&f)
        );
        for (a, want) in want {
            match answer(&f, &[a, 0, 0, 0]) {
                Outcome::Ret(v) => assert_eq!(v.bits(), want, "rcx={a:#x}"),
                other => panic!("rcx={a:#x}: unexpected {other:?}"),
            }
        }
    }
}

/// `cdq` writes EDX and touches NO flag. The machine proves it with `setc`
/// reading the `cmp`'s CF two instructions later, across the `cdq` (probe:
/// (7, 0xFFFFFFFE) -> al=1, (0xFFFFFFF9, 2) -> al=0, both measured).
#[test]
fn cdq_leaves_flags_alone() {
    let bytes: &[u8] = &[
        0x39, 0xD1, // cmp ecx,edx     (CF = ecx <edx unsigned)
        0x99, // cdq               (writes edx; must not touch EFLAGS)
        0x0F, 0x92, 0xC0, // setc al
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "cdq + setc lift");
    for (a, b, want) in [
        (7u64, 0xFFFFFFFEu64, 1u64),
        (0x12345678, 0x9ABCDEF0, 1),
        (3, 0xFFFFFFFF, 1),
        (0xFFFFFFFFFFFFFFF9, 2, 0),
        (0xFFFFFFFF, 3, 0),
        (100, 7, 0),
    ] {
        match answer(&f, &[a, b, 0, 0]) {
            Outcome::Ret(v) => assert_eq!(v.bits() & 0xFF, want, "ecx={a:#x} edx={b:#x}"),
            other => panic!("ecx={a:#x} edx={b:#x}: unexpected {other:?}"),
        }
    }
}

/// The 64-bit `div`/`idiv` divide a 128-bit `rdx:rax`. The model has no 128-bit
/// integer type, so the honest answer is a refusal that says WHY — and a
/// refusal that names the mnemonic is the map for the slice that adds the type.
/// Silently modelling it as a 64-bit division would answer every real
/// `__int128`-style division wrong, and the corpus's own 32-bit division would
/// keep passing while the 64-bit one lied.
#[test]
fn pair_dividend_at_64_bits_refuses_with_a_reason() {
    for bytes in [
        &[
            0x48, 0x89, 0xC8, 0x48, 0x99, // mov rax,rcx ; cqo
            0x49, 0xF7, 0xF8, // idiv r8
            0xC3,
        ][..],
        &[
            0x48, 0x89, 0xC8, 0x48, 0x89, 0xD0, // mov rax,rcx ; mov rdx,rax
            0x49, 0xF7, 0xF0, // div r8
            0xC3,
        ][..],
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA: a refusal is still IR");
        assert_eq!(rep.gaps, 1, "the 128-bit dividend is a counted gap");
        let t = opaque_texts(&f).join(" | ");
        assert!(
            t.contains("idiv") || t.contains("div"),
            "the refusal names the mnemonic -> {t}"
        );
        assert!(
            t.contains("128"),
            "the refusal says the pair dividend is 128-bit -> {t}"
        );
    }
}

/// Division by zero is a #DE fault on the machine, not a number. The
/// interpreter's `Fault::DivZero` is the model's answer and the emitter's guard
/// is the C spelling of the same fact; a model that answered 0 here would agree
/// with the machine on nothing and disagree on the one input that matters.
#[test]
fn division_by_zero_faults_instead_of_answering_zero() {
    let idiv: &[u8] = &[0x48, 0x89, 0xC8, 0x99, 0x41, 0xF7, 0xF8, 0xC3];
    let (f, _rep) = lift(idiv);
    match run_budget(&f, &[7, 0, 0, 0], 20000).outcome {
        Outcome::Fault(Fault::DivZero) => {}
        other => panic!("expected the divide-by-zero fault, got {other:?}"),
    }
    let div: &[u8] = &[0x48, 0x89, 0xC8, 0x31, 0xD2, 0x41, 0xF7, 0xF0, 0xC3];
    let (f, _rep) = lift(div);
    match run_budget(&f, &[7, 0, 0, 0], 20000).outcome {
        Outcome::Fault(Fault::DivZero) => {}
        other => panic!("expected the divide-by-zero fault, got {other:?}"),
    }
}

/// `cqo` is the 64-bit sibling: RDX = all ones when RAX is negative, 0
/// otherwise. Pinned on its own because the 32-bit `cdq` passing does not
/// prove the 64-bit shift count (63, not 31).
#[test]
fn cqo_sign_extends_at_63_bits() {
    let bytes: &[u8] = &[
        0x48, 0x89, 0xC8, // mov rax,rcx
        0x48, 0x99, // cqo
        0x48, 0x89, 0xD0, // mov rax,rdx
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "cqo is modelled");
    for (a, want) in [
        (0xFFFFFFFFFFFFFFF9u64, 0xFFFFFFFFFFFFFFFFu64),
        (7, 0),
        (0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF),
        (0x7FFF_FFFF_FFFF_FFFF, 0),
    ] {
        match answer(&f, &[a, 0, 0, 0]) {
            Outcome::Ret(v) => assert_eq!(v.bits(), want, "rax={a:#x}"),
            other => panic!("rax={a:#x}: unexpected {other:?}"),
        }
    }
}

/// `mov eax,imm32 ; cwde ; ret` — the sign-replicate that writes INTO the
/// accumulator. Three rows, all silicon-measured on this host (the same bytes,
/// executed from executable memory, printed these values):
///
/// ```text
/// mov eax,0x12348123 ; cwde   -> RAX 0x00000000FFFF8123
/// mov eax,0x00007FFF ; cwde   -> RAX 0x0000000000007FFF
/// mov eax,0xFFFF0005 ; cwde   -> RAX 0x0000000000000005
/// ```
///
/// The THIRD row is the one that earns the fixture: it says the instruction reads
/// ONLY bits 0..16 of the accumulator. A lifter that sign-extends the 32-bit cell
/// answers 0xFFFFF005 there and passes both of the first two rows, so a fixture
/// with only "a negative short" would bless a real misread of the architecture.
/// The first row carries the other half: a 32-bit write zeroes bits 63:32, so
/// RAX is 0x00000000FFFF8123 and not a sign-extended 64-bit value.
#[test]
fn cwde_sign_extends_ax_alone_and_zeroes_the_upper_half() {
    for (bytes, want) in [
        (
            &[0xB8u8, 0x23, 0x81, 0x34, 0x12, 0x98, 0xC3][..],
            0xFFFF_8123u64,
        ),
        (&[0xB8u8, 0xFF, 0x7F, 0x00, 0x00, 0x98, 0xC3][..], 0x7FFF),
        (&[0xB8u8, 0x05, 0x00, 0xFF, 0xFF, 0x98, 0xC3][..], 0x5),
    ] {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(
            rep.gaps,
            0,
            "cwde is modelled; opaque texts: {:?}",
            opaque_texts(&f)
        );
        match answer(&f, &[0, 0, 0, 0]) {
            Outcome::Ret(v) => assert_eq!(
                v.bits(),
                want,
                "{:02x?}: the machine's answer, quoted in the header",
                &bytes[..4]
            ),
            other => panic!("{:02x?}: unexpected {other:?}", &bytes[..4]),
        }
    }
}

/// The two siblings stay REFUSED, on purpose: `cbw` (`66 98`) and `cdqe`
/// (`48 98`) have ZERO sites in any corpus this repo gates — counted over
/// `corpus/obj`, `corpus/msvc/obj` and `corpus/cpp/obj` with llvm-objdump, whose
/// AT&T names are `cbtw` and `cdqe` (the `cwde` sites print as `cwtl`, which is
/// how the three MSVC witnesses were nearly census'd to zero). A modelled
/// instruction with no corpus witness is a feature no gate measures: the arm
/// would be green by construction, and a T2 leg over MSVC objects does not exist
/// yet to test it. Landing either one means a snippet that emits it, in the same
/// commit as the arm.
#[test]
fn cbw_and_cdqe_stay_refused_until_a_corpus_site_exists() {
    for (bytes, name) in [
        (
            &[0xB8u8, 0x81, 0x12, 0x34, 0x12, 0x66, 0x98, 0xC3][..],
            "cbw",
        ),
        (
            &[0xB8u8, 0x00, 0x00, 0x00, 0x80, 0x48, 0x98, 0xC3][..],
            "cdqe",
        ),
    ] {
        let (f, rep) = lift(bytes);
        assert_eq!(rep.gaps, 1, "{name} must stay a loud refusal");
        let texts = opaque_texts(&f);
        assert_eq!(texts.len(), 1, "{name}: one refusal, {texts:?}");
        assert!(
            texts[0].contains(name),
            "{name}: the refusal names the mnemonic, got {:?}",
            texts[0]
        );
    }
}

/// `inc`/`dec` with a MEMORY destination, and the flag they do NOT touch.
///
/// Corpus witnesses: 3 sites, every one `incl (%rcx)`, every one in an MSVC
/// object (clang lowers the same C as `addl $1, (%rcx)`, which the lifter
/// already modelled, so the census on a clang-only corpus reports this arm as
/// dead code — and llvm-objdump prints it as `incl`, the same AT&T-name trap
/// that nearly census'd `cwde` to zero).
///
/// The arithmetic is the easy half; the fixture's teeth are in F1/F4 as a PAIR.
/// `inc`/`dec` PRESERVE CF, so modelling the bytes as `add $1` — which clobbers
/// CF — is a different machine, and any later `setc`/`adc` reads it. F1 enters
/// with CF=1 (`stc`) and F4 with CF=0 (`clc`); each returns `(CF << 1) | cell`:
///
/// ```text
/// F1  stc ; inc dword [rcx+8]  (cell 9 -> 10, CF)  -> 12          (CF stayed 1)
/// F2  stc ; dec dword [rcx+8]  (cell 9 ->  8, CF)  -> 10          (CF stayed 1)
/// F3  inc dword [rcx]          (0x7FFFFFFF -> 0x80000000, OF)     -> 0x80000002
/// F4  clc ; inc dword [rcx]    (cell 5 ->  6, CF)  ->  6          (CF stayed 0)
/// ```
///
/// measured on this host (`inc_mem_probe.c`, VirtualAlloc PAGE_EXECUTE_READWRITE,
/// cells in a frame the body takes with `sub rsp,0x20`). One row alone cannot
/// separate "CF preserved" from "CF forced": F1 rules out forced-0, F4 rules out
/// forced-1.
///
/// THOSE ARE SILICON ANSWERS, and the model reproduces all four: CF after
/// `inc`/`dec` is the PREVIOUS flags value's CF, carried in the flags node as
/// `cf_in` (SS1.2 item 3, closed), so F1 answers 12 and F2 answers 10. The
/// emitter's rule for a carried CF is `getflag(CF, cf_in)`, which the fusion
/// pass threads back to the producer — `cmp ecx,edx`'s `ecx < edx` above.
///
/// F3 is the signed-overflow half: 0x7FFFFFFF + 1 lands on INT_MIN, `seto`
/// answers 1, and the cell wraps to 0x80000000 — a saturating or
/// sign-preserving model gets 0x7FFFFFFF or 0x80000000+0 there.
///
/// F1/F2 also carry a displacement (`[rcx+8]`), so the address expression under
/// test is base+disp, not just base+0.
#[test]
fn inc_dec_on_memory_write_the_cell_back_and_preserve_cf() {
    const FRAME: &[u8] = &[0x48, 0x83, 0xEC, 0x20]; // sub rsp,0x20 - the cells must NOT
                                                    // sit at entry [rsp], which is the
                                                    // return address: a first version of
                                                    // the probe wrote [rsp] and `ret`
                                                    // jumped to 5.
    const UNFRAME: &[u8] = &[0x48, 0x83, 0xC4, 0x20]; // add rsp,0x20
    fn body(mid: &[&[u8]]) -> Vec<u8> {
        let mut v = FRAME.to_vec();
        for m in mid {
            v.extend_from_slice(m);
        }
        v.extend_from_slice(UNFRAME);
        v.push(0xC3);
        v
    }
    let cells = [
        &[0x48u8, 0xC7, 0x04, 0x24, 0x05, 0x00, 0x00, 0x00][..], // mov qword [rsp], 5
        &[0x48u8, 0xC7, 0x44, 0x24, 0x08, 0x09, 0x00, 0x00, 0x00][..], // mov qword [rsp+8], 9
    ];
    let read_cf = &[0x0Fu8, 0x92, 0xC0, 0x0F, 0xB6, 0xC0, 0x01, 0xC0][..]; // setc al; movzx eax,al; add eax,eax
    let read_of = &[0x0Fu8, 0x90, 0xC0, 0x0F, 0xB6, 0xC0, 0x01, 0xC0][..]; // seto al;  movzx eax,al; add eax,eax

    // CF is set by a COMPARISON, not by `stc`/`clc`: this lifter refuses
    // `stc` (measured - the fixture's first version named it in its own refusal),
    // and a comparison is what real code uses to set CF before an `inc`-then-`adc`
    // sequence anyway. `cmp ecx,edx` answers CF=1 when ecx < edx unsigned, so
    // (0,1) enters the inc with CF=1 and (1,0) enters it with CF=0. The
    // comparison is also what gives the flag node a predecessor: `inc`/`dec`
    // PRESERVE CF, which the model represents by threading the previous flags in
    // as an operand, and on a block with no flag producer there is no previous
    // CF to preserve - the arm refuses, correctly, and says so.
    let setcf = &[0x39u8, 0xD1][..]; // cmp ecx,edx
    let cases: Vec<(Vec<u8>, u64, [u64; 4], &str)> = vec![
        (
            body(&[
                cells[0],
                cells[1],
                setcf,
                &[0x48, 0x89, 0xE1][..], // mov rcx,rsp — AFTER the cmp on purpose: it
                // overwrites ECX, the comparison's first
                // operand, so a cmp placed after it compares a
                // stack address against the input and CF is a
                // constant 0 (measured: `cmpult 0x7ffe0, edx`).
                // A first version of this fixture had that
                // order, and its "CF = 1" leg proved nothing.
                &[0xFF, 0x41, 0x08], // inc dword [rcx+8]
                read_cf,
                &[0x03, 0x41, 0x08], // add eax, [rcx+8]
            ]),
            12,
            [0, 1, 0, 0],
            "F1: cell 9->10, CF preserved as 1 and read as 2*CF = 12 — the silicon number, \
             now the model's: eax = 2*CF + cell",
        ),
        (
            body(&[
                cells[0],
                cells[1],
                setcf,
                &[0x48, 0x89, 0xE1][..],
                &[0xFF, 0x49, 0x08], // dec dword [rcx+8]
                read_cf,
                &[0x03, 0x41, 0x08], // add eax, [rcx+8]
            ]),
            10,
            [0, 1, 0, 0],
            "F2: cell 9->8, CF preserved as 1 → 2*CF + cell = 10, the silicon number",
        ),
        (
            body(&[
                &[0xC7, 0x04, 0x24, 0xFF, 0xFF, 0xFF, 0x7F][..], // mov dword [rsp], 0x7FFFFFFF
                setcf,
                &[0x48, 0x89, 0xE1][..], // mov rcx,rsp
                &[0xFF, 0x01],           // inc dword [rcx]
                read_of,
                &[0x03, 0x01], // add eax, [rcx]
            ]),
            0x8000_0002,
            [0, 1, 0, 0],
            "F3: 0x7FFFFFFF wraps to 0x80000000 and OF is set",
        ),
        (
            body(&[
                cells[0],
                setcf,
                &[0x48, 0x89, 0xE1][..],
                &[0xFF, 0x01], // inc dword [rcx]
                read_cf,
                &[0x03, 0x01], // add eax, [rcx]
            ]),
            6,
            [1, 0, 0, 0],
            "F4: cell 5->6 with CF preserved as 0",
        ),
    ];

    for (bytes, want, ins, why) in cases {
        let (f, rep) = lift(&bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(
            rep.gaps,
            0,
            "{why}: memory inc/dec is modelled; opaque texts: {:?}",
            opaque_texts(&f)
        );
        match answer(&f, &ins) {
            Outcome::Ret(v) => assert_eq!(
                v.bits(),
                want,
                "{why}: silicon value from inc_mem_probe.c; the expected value also pins \
                 that the LAST write is a 32-bit op, so bits 63:32 are zeroed"
            ),
            Outcome::Fault(fl) => {
                panic!(
                    "{why}: faulted as {fl:?} — opaque texts: {:?}",
                    opaque_texts(&f)
                )
            }
            other => panic!("{why}: unexpected {other:?}"),
        }

        // The flag half of the contract: a function that READS CF after an
        // inc/dec must now EMIT, because the flags node carries the previous
        // flags value (`cf_in`) and the fusion pass threads the read back to the
        // producer. Before this closed, the emitter refused the whole function
        // with "CF after inc/dec needs the previous flags as an operand
        // (SS1.2 item 3)" — the refusal was the only thing keeping the
        // interpreter's silent 0 out of generated code. It is the value asserted
        // per case above that now carries that guarantee, so the emit has to
        // happen for the CF rows: a refusal here means the threading regressed
        // and the numbers above are being produced by the old undefined path.
        if let Err(rs) = emit_c(&f) {
            assert!(
                !why.contains("CF"),
                "{why}: a CF-preserving inc/dec refused to emit: {rs:?}"
            );
        }
    }
}

/// CF is preserved, not invented. With NO flag producer before the `inc` there
/// is no previous CF to preserve — silicon's CF at function entry is the
/// caller's, which this model does not know — so a function that reads CF there
/// must REFUSE, and must not emit a 0. This is the direction the threading must
/// not break: `cf_in` is `None` exactly when nothing produced flags.
#[test]
fn cf_with_no_previous_flag_producer_refuses_instead_of_inventing_zero() {
    let bytes: Vec<u8> = [
        &[0x48u8, 0xC7, 0x04, 0x24, 0x05, 0x00, 0x00, 0x00][..], // mov qword [rsp], 5
        &[0x48, 0x89, 0xE1][..],                                 // mov rcx,rsp
        &[0xFF, 0x01][..],                                       // inc dword [rcx]
        &[0x0F, 0x92, 0xC0, 0x0F, 0xB6, 0xC0][..], // setc al; movzx eax,al  -- eax = CF, no prior flags
        &[0xC3][..],                               // ret
    ]
    .concat();
    let (f, rep) = lift(&bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "the inc itself is modelled");
    match emit_c(&f) {
        Ok(_) => panic!(
            "invented a CF at function entry, where silicon's CF is the caller's and the \
             model has no producer for it: a 0 would be emitted as the preserved flag"
        ),
        Err(rs) => assert!(
            rs.iter().any(|r| r.reason.contains("inc/dec")),
            "the refusal must name the carried CF, not something else: {rs:?}"
        ),
    }
}

/// The arm must keep REFUSING what it cannot address. `FF 05 00 00 00 00` is
/// `inc dword [rip+0]`, and this harness carries no relocation table, so the
/// RIP-relative cell has no name — the gap path is the right answer, and the
/// test's point is that the new memory arm does not swallow it into a load from
/// a made-up address. A silently-wrong address is worse than a refusal: it
/// invents memory the program never had.
#[test]
fn inc_on_an_unnameable_rip_relative_cell_stays_refused() {
    let bytes: &[u8] = &[0xFF, 0x05, 0x00, 0x00, 0x00, 0x00, 0xC3]; // inc dword [rip+0] ; ret
    let (f, rep) = lift(bytes);
    assert!(
        rep.gaps > 0,
        "a RIP-relative inc with no relocation must stay refused; opaque texts: {:?}",
        opaque_texts(&f)
    );
    assert!(
        opaque_texts(&f)
            .iter()
            .any(|t| t.contains("inc/dec on an unmodelled destination")),
        "the refusal must NAME itself, not a generic gap: {:?}",
        opaque_texts(&f)
    );
}

/// A flag-setting op whose DESTINATION is memory: `cmpl %eax,(%rcx)`,
/// `testb $1,(%rcx)`.
///
/// Census first, AT&T names: 34 `cmp` sites with a memory destination over
/// `corpus/msvc/obj` (`cmpl %eax,0x20(%rsp)` x13, `cmpl %eax,(%rsp)` x8,
/// `cmpl $0x0,0x20(%rsp)` x3, plus imm/equal-width forms). The leg's census showed
/// only 3 - it records the FIRST refusal per function, and most of these functions
/// refuse for a different reason first. The modelled arm is therefore worth far
/// more than the census line suggested, and the census line is why: a per-function
/// census under-reports a class that appears many times per function.
///
/// THE ORDER IS THE CLAIM. Intel's first operand is AT&T's LAST, and `cmp m, r`
/// computes m - r, so the memory load must be the LEFT operand of the flag op.
/// Swapping operands computes r - m: CF, SF and OF all flip, and every `ja`/`jb`/
/// `setb` downstream walks the mirror image. F1/F2 are the silicon pair that pins
/// the direction (cell 5 vs eax 7 -> CF=1; cell 7 vs eax 5 -> CF=0). Everything is
/// measured on this host, `cmp_mem_probe.c`, returning `flag*8 + cell` so one run
/// answers the flag and proves the memory cell was the one read:
///
/// ```text
/// F1  [rcx]=5,  eax=7,   cmp [rcx],eax, setb  -> 13          CF=1  (5 below 7)
/// F2  [rcx]=7,  eax=5,   cmp [rcx],eax, setb  ->  7          CF=0  (7 not below 5)
/// F3  [rcx]=7,  eax=7,   cmp [rcx],eax, sete  -> 15          ZF=1
/// F4  [rcx]=0x7FFFFFFF, eax=-1, seto          -> 0x80000007  OF=1 (pos - neg = neg)
/// F5  [rcx]=3,  test [rcx],1, sete            ->  3          ZF=0 (3&1 = 1)
/// F6  [rcx]=2,  test [rcx],1, sete            -> 10          ZF=1 (2&1 = 0)
/// F7  [rcx+8]=5, eax=9,  cmp [rcx+8],eax,setb -> 13          base+disp address
/// ```
#[test]
fn cmp_and_test_with_a_memory_destination_compare_the_cell_in_order() {
    fn body(store: &[&[u8]], cc: &[u8], tail: &[&[u8]]) -> Vec<u8> {
        let mut v = vec![0x48u8, 0x83, 0xEC, 0x20]; // sub rsp,0x20 - entry [rsp] is
                                                    // the return address, never a cell
        for s in store {
            v.extend_from_slice(s);
        }
        v.extend_from_slice(&[0x48, 0x89, 0xE1]); // mov rcx,rsp
        for t in tail {
            v.extend_from_slice(t);
        }
        v.extend_from_slice(cc); // <setcc> al
        v.extend_from_slice(&[0x0F, 0xB6, 0xC0, 0xC1, 0xE0, 0x03]); // movzx eax,al; shl eax,3
        v.extend_from_slice(&[0x8B, 0x11, 0x01, 0xD0]); // mov edx,[rcx]; add eax,edx
        v.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0xC3]); // add rsp,0x20; ret
        v
    }
    fn st32(v: u32) -> Vec<u8> {
        let b = v.to_le_bytes();
        vec![0xC7, 0x04, 0x24, b[0], b[1], b[2], b[3]] // mov dword [rsp], v
    }
    fn eax(v: u32) -> Vec<u8> {
        let b = v.to_le_bytes();
        vec![0xB8, b[0], b[1], b[2], b[3]] // mov eax, v
    }
    const CMP_EAX: &[u8] = &[0x39, 0x01]; // cmp [rcx], eax  -> [rcx] - eax
    const SETB: &[u8] = &[0x0F, 0x92, 0xC0];
    const SETE: &[u8] = &[0x0F, 0x94, 0xC0];
    const SETO: &[u8] = &[0x0F, 0x90, 0xC0];

    let cases: Vec<(Vec<u8>, u64, &str)> = vec![
        (
            body(&[&st32(5)], SETB, &[&eax(7), CMP_EAX]),
            13,
            "F1: cell 5 vs eax 7 borrows, CF=1",
        ),
        (
            body(&[&st32(7)], SETB, &[&eax(5), CMP_EAX]),
            7,
            "F2: cell 7 vs eax 5 does NOT borrow - the direction F1 alone cannot pin",
        ),
        (
            body(&[&st32(7)], SETE, &[&eax(7), CMP_EAX]),
            15,
            "F3: equal, ZF=1",
        ),
        (
            body(&[&st32(0x7FFF_FFFF)], SETO, &[&eax(0xFFFF_FFFF), CMP_EAX]),
            0x8000_0007,
            "F4: 0x7FFFFFFF - (-1) = 0x80000000, OF=1",
        ),
        (
            body(
                // A DWORD store, not a byte store: the fixture reads the cell back
                // with `mov edx,[rcx]` (4 bytes), and silicon's high bytes were
                // STALE STACK (0x15fe03) that this file refuses to bake into an
                // expectation. The low byte - the one `test byte [rcx],1` reads -
                // is identical, so the flag answer silicon measured still applies.
                &[&st32(3)[..]],
                SETE,
                &[&[0xF6, 0x01, 0x01][..]], // test byte [rcx], 1
            ),
            3,
            "F5: 3 & 1 = 1, ZF=0",
        ),
        (
            body(&[&st32(2)[..]], SETE, &[&[0xF6, 0x01, 0x01][..]]), // test byte [rcx], 1
            10,
            "F6: 2 & 1 = 0, ZF=1 - the same test bit, opposite answer",
        ),
        (
            body(
                &[
                    // Cell written at BOTH slots: the shared read-back (`mov
                    // edx,[rcx]`) reads offset 0, while the comparison under test
                    // reads `[rcx+8]`. Both hold 5, so the cell in the expected
                    // value is the cell the disp8 form compared.
                    &[0xC7, 0x04, 0x24, 0x05, 0x00, 0x00, 0x00][..], // [rsp] = 5
                    &[
                        0x48, 0xC7, 0x44, 0x24, 0x08, 0x05, 0x00, 0x00, 0x00, // [rsp+8] = 5
                    ][..],
                ],
                SETB,
                &[
                    &[0xB8, 0x09, 0x00, 0x00, 0x00][..], // mov eax, 9
                    &[0x39, 0x41, 0x08][..],             // cmp [rcx+8], eax
                ],
            ),
            13,
            "F7: base+disp destination, cell 5 vs eax 9",
        ),
    ];

    for (bytes, want, why) in cases {
        let (f, rep) = lift(&bytes);
        verify_ssa(&f).expect("valid SSA");
        assert_eq!(
            rep.gaps,
            0,
            "{why}: memory cmp/test is modelled; opaque texts: {:?}",
            opaque_texts(&f)
        );
        match answer(&f, &[0, 0, 0, 0]) {
            Outcome::Ret(v) => {
                assert_eq!(v.bits(), want, "{why}: silicon value from cmp_mem_probe.c")
            }
            other => panic!("{why}: unexpected {other:?}"),
        }
        // cmp/test define every flag they are read for, so these functions EMIT:
        // unlike the `inc`/`dec` CF case, nothing here is left undefined, and a
        // refusal at this point would mean the arm lost the value.
        if let Err(rs) = emit_c(&f) {
            panic!("{why}: a memory cmp/test with defined flags REFUSED: {rs:?}");
        }
    }
}

/// A memory-destination compare whose address cannot be named stays REFUSED:
/// `83 3D 00 00 00 00 00` is `cmpl $0x0,[rip+0]`, and this harness has no
/// relocation table, so there is no cell to compare. Refusing is the honest
/// answer; comparing against an invented address would silently decide a branch
/// from memory the object never declared.
#[test]
fn cmp_against_an_unnameable_rip_relative_cell_stays_refused() {
    let bytes: &[u8] = &[0x83, 0x3D, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC3];
    let (f, rep) = lift(bytes);
    assert!(
        rep.gaps > 0,
        "a RIP-relative cmp with no relocation must stay refused; opaque texts: {:?}",
        opaque_texts(&f)
    );
    assert!(
        opaque_texts(&f)
            .iter()
            .any(|t| t.contains("flag setting with an unmodelled first operand")),
        "the refusal must NAME itself: {:?}",
        opaque_texts(&f)
    );
}

/// A RIP-relative destination whose displacement is NOT the instruction's last
/// field: `83 3D <disp32> <imm8>` is `cmpl $0x2, (%rip)` — an 8-bit immediate
/// FOLLOWS the displacement. Resolving by `next_ip - 4` (the arithmetic the
/// lifter used, correct for `39 0D <disp32>` where the disp32 really is last)
/// lands at offset 4, INSIDE a field that spans 2..6, while the table keys the
/// field's first byte (2): the relocation is missed and the site refuses.
///
/// Real witness: `corpus/msvc/obj/array_walk.O2.obj :: f @ 0x2f`. The same
/// assumption covers `81 /7 <disp32> <imm32>`, `C6 /0 <disp32> <imm8>`,
/// `F6 /7 <disp32> <imm8>` — a whole encoding family, in a shared helper.
///
/// This test supplies the relocation at offset 2 and requires the load to be
/// built AT THE TARGET ADDRESS (0x1000), not refused: the arithmetic version
/// fails it with a gap, and the leg's refusal census goes back up when it does.
#[test]
fn a_rip_destination_resolves_when_the_displacement_is_not_the_last_field() {
    let bytes: &[u8] = &[0x83, 0x3D, 0x00, 0x00, 0x00, 0x00, 0x02, 0xC3];
    let mut relocs: RelocMap = RelocMap::new();
    relocs.insert(
        2u64,
        RelocTarget {
            sym: "g_count".to_string(),
            addr: Some(0x1000),
        },
    );
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64; 4], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let start = plan.functions[0];
    let (f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        start,
        owned.len() as u64,
        &plan,
        &std::collections::HashMap::new(),
        &FnAbi::integer(),
        &CalleeTable::new(),
        &relocs,
        &no_tables(),
    );
    assert_eq!(
        rep.gaps,
        0,
        "the cell's address IS in the relocation table; opaque texts: {:?}",
        opaque_texts(&f)
    );
    // 0x1000 = 4096: the ONLY way that constant enters the IR is the relocation
    // target, so finding it in the dump proves the cmp loaded THE CELL at the
    // address the table names (the arithmetic version refuses this function and
    // the dump has no constant at all).
    let dump = format!("{:?}", f);
    assert!(
        dump.contains("Const(4096)"),
        "the load address must be the relocation's target 0x1000; IR dump: {dump}"
    );
}

/// The refusal must name the SHORTAGE, not guess at the operand kind.
///
/// `83 3D <disp32> <imm8>` = `cmpl $0x2, (%rip)` with a relocation that NAMES a
/// symbol but carries no address. That is the exact measured shape of the MSVC
/// leg's three remaining `flag setting` records: the leg's relocation map places
/// CODE sections, so a compare against a static in `.data` has a target with no
/// address. Calling that "an unmodelled first operand" sends the work queue to
/// the lifter, which already models the operand fine; the missing piece is an
/// address-space decision (where do data symbols live), which is a different
/// milestone's question.
#[test]
fn a_relocation_that_names_no_address_says_so_instead_of_blaming_the_operand() {
    let bytes: &[u8] = &[0x83, 0x3D, 0x00, 0x00, 0x00, 0x00, 0x02, 0xC3];
    let mut relocs: RelocMap = RelocMap::new();
    relocs.insert(
        2u64,
        RelocTarget {
            sym: "g_count".to_string(),
            addr: None, // names a symbol; the model has no address for it
        },
    );
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64; 4], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let start = plan.functions[0];
    let (f, rep) = lift_one_sigs(
        "f",
        &owned,
        0,
        start,
        owned.len() as u64,
        &plan,
        &std::collections::HashMap::new(),
        &FnAbi::integer(),
        &CalleeTable::new(),
        &relocs,
        &no_tables(),
    );
    assert!(rep.gaps > 0, "an address-less target stays refused");
    let texts = opaque_texts(&f);
    assert!(
        texts
            .iter()
            .any(|t| t.contains("flag setting against a cell whose relocation names no address")),
        "the refusal must name the missing address, not the operand kind: {texts:?}"
    );
    assert!(
        !texts
            .iter()
            .any(|t| t.contains("flag setting with an unmodelled first operand")),
        "the old key blamed the lifter for an address-space gap: {texts:?}"
    );
}

fn lift_raw(bytes: &[u8]) -> (Function, Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64; 4], vec![]));
    let plan = discover_all(&ranges, &seeds);
    let start = plan.functions[0];
    lift_one(
        "f",
        &owned,
        0,
        start,
        owned.len() as u64,
        &plan,
        &Default::default(),
    )
}

/// The interpreter's OWN carried-CF read, with no pass to thread it. The
/// post-pass fixture above proves the fusion threading and never reaches this
/// code: after the pass the CF read is a `cmpult` and the inc's flags node is
/// dead. Measured: with the interpreter's `cf_in` read sabotaged to
/// `Some(false)`, that fixture stayed GREEN (the pass threads the read itself),
/// so without this test the interpreter's arm has zero coverage. A raw run of
/// the same bytes is where the interpreter answers, and it must answer 12.
#[test]
fn raw_ir_reads_the_carried_cf_with_no_pass_to_thread_it() {
    let bytes: Vec<u8> = [
        &[0x48u8, 0x83, 0xEC, 0x20][..], // sub rsp,0x20
        &[
            0x48, 0xC7, 0x44, 0x24, 0x08, 0x09, 0x00, 0x00, 0x00, // mov qword [rsp+8], 9
        ][..],
        &[0x39, 0xD1][..],       // cmp ecx,edx  (ecx=0, edx=1 -> CF=1)
        &[0x48, 0x89, 0xE1][..], // mov rcx,rsp
        &[0xFF, 0x41, 0x08][..], // inc dword [rcx+8]  (9 -> 10)
        &[0x0F, 0x92, 0xC0, 0x0F, 0xB6, 0xC0, 0x01, 0xC0][..], // setc al; movzx; add eax,eax
        &[0x03, 0x41, 0x08][..], // add eax, [rcx+8]
        &[0x48, 0x83, 0xC4, 0x20][..], // add rsp,0x20
        &[0xC3][..],             // ret
    ]
    .concat();
    let (f, rep) = lift_raw(&bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "the shape lifts with no gap");
    match answer(&f, &[0, 1, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(
            v.bits(),
            12,
            "raw IR: the interpreter reads CF out of the flags value the inc carries \
             (the comparison's CF), the same 12 the post-pass fixture answers"
        ),
        Outcome::Fault(fl) => panic!(
            "the carried CF must be readable on raw IR; faulted {fl:?} — opaque texts: {:?}",
            opaque_texts(&f)
        ),
        other => panic!("unexpected {other:?}"),
    }
}

/// The register form of the same contract: `cmp ecx,edx ; inc eax ; setc al`.
/// `inc` writes a register, not a cell, and the CF it preserves is the
/// comparison's. Register `inc` reached this code through a different lifter
/// arm than the memory form, so the arm is pinned on its own.
#[test]
fn raw_ir_register_inc_carries_cf_to_a_later_read() {
    let bytes: Vec<u8> = [
        &[0x39u8, 0xD1][..],     // cmp ecx,edx  (ecx=0, edx=1 -> CF=1)
        &[0xFF, 0xC0][..],       // inc eax      (0 -> 1)
        &[0x0F, 0x92, 0xC0][..], // setc al  -> eax's low byte = CF
        &[0x0F, 0xB6, 0xC0][..], // movzx eax,al
        &[0xC3][..],             // ret
    ]
    .concat();
    let (f, rep) = lift_raw(&bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(
        rep.gaps, 0,
        "register inc is modelled; opcodes: 39D1 FFC0 0F92C0 0FB6C0"
    );
    match answer(&f, &[0, 1, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(
            v.bits(),
            1,
            "the comparison's CF must survive the register inc into the later setc"
        ),
        Outcome::Fault(fl) => panic!("the carried CF must be readable; faulted {fl:?}"),
        other => panic!("unexpected {other:?}"),
    }
}

/// The limit the leg's `lift` gate drew, pinned in one function. CF is preserved
/// WITHIN a basic block; across a join there is no value to carry, because SSA
/// has no flag phis (SS1.2 item 2). Shape:
///
/// ```text
/// bb0: sub rsp,0x20 ; mov rcx,rsp ; cmp ecx,edx ; je L2
/// bb1: inc dword [rcx]      ; jmp L3
/// bb2: inc dword [rcx+4]       (the `je` target)
/// L3:  inc dword [rcx+8] ; setc al ; movzx eax,al ; add rsp,0x20 ; ret
/// ```
///
/// L3 has two predecessors, so a `cf_in` naming a flags value defined in bb1 (or
/// bb2) is a use its block does not dominate. The MSVC leg's `lift` gate found
/// exactly that (`%63 defined in bb5 does not dominate use in bb8`, in
/// `early_returns.Od.obj :: $LN10`) when the lifter carried the flags value
/// across joins unconditionally. Here the lifter must emit NO carried value, the
/// IR must verify, and reading CF at the join must REFUSE — the comparison's CF
/// is true on one path and undefined on the other, and answering either way
/// invents a fact the machine only has per-path.
#[test]
fn cf_is_carried_within_a_block_and_not_across_a_join() {
    // Offsets: `FF 01` is 2 bytes and `FF 41 04` is 3, so L1=11, L2=15, L3=18.
    // (A first version counted L2 as 4 bytes, the jump targets landed
    // mid-instruction, and the decoder desynced into `ror` gaps.)
    let bytes: Vec<u8> = [
        &[0x48u8, 0x83, 0xEC, 0x20][..], // 0:  sub rsp,0x20
        &[0x48, 0x89, 0xE1][..],         // 4:  mov rcx,rsp
        &[0x39, 0xD1][..],               // 7:  cmp ecx,edx
        &[0x74, 0x04][..],               // 9:  je  L2 (11+4 = 15)
        &[0xFF, 0x01][..],               // 11: L1: inc dword [rcx]
        &[0xEB, 0x03][..],               // 13: jmp L3 (15+3 = 18)
        &[0xFF, 0x41, 0x04][..],         // 15: L2: inc dword [rcx+4]
        &[0xFF, 0x41, 0x08][..],         // 18: L3: inc dword [rcx+8]
        &[0x0F, 0x92, 0xC0][..],         // 21: setc al
        &[0x0F, 0xB6, 0xC0][..],         // 24: movzx eax,al
        &[0x48, 0x83, 0xC4, 0x20][..],   // 27: add rsp,0x20
        &[0xC3][..],                     // 31: ret
    ]
    .concat();
    let (f, _) = lift_raw(&bytes);
    // The IR must be structurally sound: this is the assertion that catches a
    // carried CF across the join, which is a hard verifier error, not a refusal.
    rdec_core::verify::verify_ssa(&f)
        .unwrap_or_else(|e| panic!("carried CF across a join is a hard SSA error: {e:?}"));
    assert!(
        f.blocks.len() >= 3,
        "the shape must have a join (bb0, two paths, L3); blocks lifted: {}",
        f.blocks.len()
    );
    match emit_c(&f) {
        Ok(_) => panic!(
            "emitted C that answers CF at a join: one path has the comparison's CF, the \
             other has none, and SSA has no flag phi to merge them (SS1.2 item 2)"
        ),
        Err(rs) => {
            // Measured refusal texts for this shape, both honest gaps:
            //     "flags value Sub { a: 4, b: 5, borrow: false } survived flag
            //      elimination"   (a pre-existing emitter conservatism: the frame
            //      `sub rsp,0x20`'s flags def, refused as a surviving flags value)
            // and the CF read at the join, which lowers to
            //     opaque "CF after inc/dec with no previous flag producer:
            //             silicon's CF is the caller's and this model has no
            //             value for it"
            // The second is what this fixture is about. The first refuses the
            // SAME function, and the guarantee under test is that no C is
            // emitted at all — a specific text is asserted when the pipeline
            // reaches it, and any named gap is accepted as a refusal.
            let text: String = rs
                .iter()
                .map(|r| r.reason.to_string())
                .collect::<Vec<_>>()
                .join(" | ");
            assert!(
                text.contains("inc/dec") || text.contains("survived flag elimination"),
                "the refusal must be a named gap, not silence: {text}"
            );
        }
    }
}
