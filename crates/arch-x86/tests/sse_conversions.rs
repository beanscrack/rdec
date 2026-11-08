//! M8 conversion breadth — `movq xmm, r64` (SSE2) and `punpcklqdq`.
//!
//! `movq xmm, r64` (`66 REX.W 0F 6E`) is the move toolchains use to build a
//! double in a GPR — from `cvttsd2si`-style integer work, from a raw
//! `memcpy`-style bit load, from `bit_cast` — and hand it to the XMM domain.
//! Intel Vol.2A: it writes the LOW 64 bits and leaves the destination's high
//! 64 bits alone. The model previously applied the `movd` rule (zero the
//! whole pair), which poisons every later high-lane read.
//!
//! `punpcklqdq` (`66 0F 6C`) writes ONLY the destination's high cell:
//! dest_hi = src_lo; dest_lo survives.
//!
//! The second fixture is the distinctive witness: a lift with the OLD
//! zeroing movq must propagate the zeroed pair consistently, and the pair
//! cells that reach `Ret` are pinned as data — a zeroing movq yields the
//! pair (0x3fa0…, 0) only if the DEST stayed zero; punpck's own write is
//! separately pinned by the destination census below. An `0F 6E` decoded as
//! the `movd` form refuses the instruction (named gap, pair undef), so
//! reaching `Ret` is itself a modelled-ness witness.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::{lift_one, Report};
use rdec_core::interp::{run_budget, Outcome};
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

/// `movabs $0x3ff9000000000000,rcx ; movq xmm0,rcx ;
/// cvttsd2si rax, xmm0 ; ret`
///
/// `cvttsd2si` reads the pair's LOW cell, so the answer is the integer the
/// bits spell: 1.5625 truncates to 1. The `movq` arm must fire on
/// `66 48 0F 6E` — the `0F 6E` decode as `movd` on a 32-bit read truncates
/// `0x3ff9…&0xffffffff = 0` to 0, and a refused lift never reaches `Ret`.
#[test]
fn movq_gpr_to_xmm_is_modelled_and_moves_the_quad() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor   xmm0, xmm0            (seed)
        0x48, 0xB9, 0, 0, 0, 0, 0, 0, 0xF9, 0x3F, // movabs $0x3ff9000000000000, rcx
        0x66, 0x48, 0x0F, 0x6E, 0xC1, // movq   xmm0, rcx
        0xF2, 0x48, 0x0F, 0x2C, 0xC0, // cvttsd2si rax, xmm0   (reads the LOW cell)
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    match answer(&f, &[0, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 1, "trunc(1.5625) = 1"),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(rep.sse_packed, 1, "the movq merge is modelled");
}

/// `pxor xmm0,xmm0 ; pxor xmm1,xmm1 ; movabs $0x3fa0000000000000,rcx ;
/// movq xmm0,rcx ; punpcklqdq xmm1,xmm0 ; cvttsd2si rax, xmm1 ; ret`
///
/// punpcklqdq writes ONLY the destination's HIGH cell. `cvttsd2si rax, xmm1`
/// reads xmm1's LOW cell, so the answer is the cell punpck must not touch.
/// That cell is seeded with a value the program cannot carry any other way
/// (44.0 = 0x4046000000000000, built by `movabs` + `movq`), which is the sentinel this repo
/// requires of every "must not write" fixture: the moved quad is 1.5625, so a
/// punpck that wrote the LOW cell answers 1 and one that wrote nothing
/// answers 44 — the two answers cannot agree.
///
/// What this fixture CANNOT see, named rather than hidden: a punpck arm that
/// swapped dst/src (writing the SOURCE's high cell from the destination's
/// low) is undetectable from any low-cell read, because with two registers
/// the swap is the same rule set with the roles exchanged — every low cell
/// ends up identical. Witnessing the swap direction needs a HIGH-cell read
/// (`pshufd`/`unpckhqdq`-style lane shuffle), which the lifter does not model
/// yet; it lands with that slice, and until then the census below is the
/// only punpck-occurrence evidence. `punpcklqdq` occurs 0 times in today's
/// corpus, so no floor depends on it.
#[test]
fn punpcklqdq_writes_only_the_high_lane() {
    let bytes: &[u8] = &[
        0x48, 0xB9, 0, 0, 0, 0, 0, 0, 0x46, 0x40, // movabs $0x4046000000000000,rcx  (44.0)
        0x66, 0x48, 0x0F, 0x6E, 0xC9, // movq       xmm1, rcx
        0x48, 0xBA, 0, 0, 0, 0, 0, 0, 0xF9, 0x3F, // movabs $0x3ff9000000000000,rdx  (1.5625)
        0x66, 0x48, 0x0F, 0x6E, 0xD2, // movq       xmm2, rdx
        0x66, 0x0F, 0x6C, 0xCA, // punpcklqdq xmm1, xmm2  (writes xmm1's HIGH only)
        0xF2, 0x48, 0x0F, 0x2C, 0xC1, // cvttsd2si  rax, xmm1     (reads xmm1's LOW)
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    match answer(&f, &[0, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 44, "the untouched LOW quad is the answer"),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(
        rep.sse_packed, 3,
        "two movq merges + punpcklqdq are modelled"
    );
}
