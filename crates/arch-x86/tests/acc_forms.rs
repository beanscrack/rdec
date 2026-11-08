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
use rdec_arch_x86::lift::{lift_one, Report};
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
/// THOSE ARE SILICON ANSWERS, and the model does not reproduce F1/F2 — see the
/// flag block in the loop: CF after `inc`/`dec` is left UNDEFINED by this model
/// (SS1.2 item 3), the IR faults on reading it, and the emitter refuses the whole
/// function. The numbers asserted per case are the post-pass interpreter values,
/// with the silicon number in each case's own label.
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
                &[0x48, 0x89, 0xE1][..], // mov rcx,rsp
                setcf,
                &[0xFF, 0x41, 0x08], // inc dword [rcx+8]
                read_cf,
                &[0x03, 0x41, 0x08], // add eax, [rcx+8]
            ]),
            10,
            [0, 1, 0, 0],
            "F1: cell 9->10; silicon answers 12 because CF is PRESERVED, the model leaves \
             it undefined and reads of it refuse",
        ),
        (
            body(&[
                cells[0],
                cells[1],
                &[0x48, 0x89, 0xE1][..],
                setcf,
                &[0xFF, 0x49, 0x08], // dec dword [rcx+8]
                read_cf,
                &[0x03, 0x41, 0x08], // add eax, [rcx+8]
            ]),
            8,
            [0, 1, 0, 0],
            "F2: cell 9->8; silicon answers 10 (CF preserved as 1), same undefined-CF story",
        ),
        (
            body(&[
                &[0xC7, 0x04, 0x24, 0xFF, 0xFF, 0xFF, 0x7F][..], // mov dword [rsp], 0x7FFFFFFF
                &[0x48, 0x89, 0xE1][..],                         // mov rcx,rsp
                setcf,
                &[0xFF, 0x01], // inc dword [rcx]
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
                &[0x48, 0x89, 0xE1][..],
                setcf,
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

        // The flag half of the contract, and the reason the numbers above are
        // not the silicon numbers: a function that READS CF after an inc/dec is
        // REFUSED by the emitter, naming the gap. Silicon preserves CF, the
        // model leaves it undefined, and `fuse`'s own rule is that an undefined
        // bit keeps its getflag so the emitter refuses. Measured on this shape:
        //     Refusal { at: 19, reason: "opaque: CF after inc/dec needs the
        //                previous flags as an operand (SS1.2 item 3)" }
        // The interpreter's 0 for CF (the pipeline values above) is therefore
        // unreachable in any emitted function - a program that reads CF after
        // inc/dec never emits, so T2 can never compare a wrong CF against C.
        // Closing SS1.2 item 3 (thread the previous flags in as an operand) is
        // what would let F1 answer silicon's 12.
        let reads_cf = why.contains("CF");
        match emit_c(&f) {
            Err(rs) => {
                let named = rs.iter().any(|r| r.reason.contains("SS1.2 item 3"));
                assert!(
                    reads_cf && named,
                    "{why}: expected the SS1.2 item 3 refusal (silicon preserves CF); got {rs:?}"
                );
            }
            Ok(_) => assert!(
                !reads_cf,
                "{why}: a function that reads CF after inc/dec EMITTED C — the gap leaked \
                 into generated code, where a silent 0 would outlive any test"
            ),
        }
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
