// M8 packed SIMD, first slice: the XMM register is a PAIR of W64 cells (lo,
// hi), and this file pins what that pair means, on hand-assembled bytes.
//
// Ground truth is `llvm-objdump -d` (clang 22.1.8 integrated assembler,
// `clang -c -target x86_64-pc-windows-msvc`, no `-O`, so no optimiser-version
// drift) over the same bytes, quoted byte-for-byte in each fixture comment.
// The assertions are on BEHAVIOR -- the bytes the interpreter's flat memory
// model holds after a run -- never on asm spellings, because the CI runner's
// clang lowers the same C differently and the T2 comparison is the judge there.
//
// The slice, and only the slice:
//  * 16-byte load/store: movdqa, movdqu, movaps, movups, movapd, movupd.
//    A load is two W64 loads (lo at addr, hi at addr+8); a store symmetric.
//  * lane-independent bitwise on the full 128 bits: pxor, pand, pandn, por.
//    Each is two INDEPENDENT 64-bit lane ops -- not lane-crossing, so the
//    W64 lane model expresses them exactly.
//  * the zeroing idiom `pxor xmm,xmm` zeroes BOTH cells (machine fact).
//  * scalar ops keep the pair honest, per the hardware, not per convenience:
//    - a scalar STORE (`movsd m64, xmm`) writes 8 bytes and leaves the
//      register's high lane exactly as it was -- preserved, observable here
//      by a following 16-byte store of the same register;
//    - a register/memory LOAD (`movsd xmm, ...`) zeroes bits 127:64 (the
//      Intel MOVSD merge form: load form zeroes the upper half, merge forms
//      like `addsd` preserve it). Pinned both ways: preservation where the
//      machine preserves, zeroing where the machine zeroes. Pinning the
//      wrong one here would make the decompiler agree with a comment
//      instead of with silicon, and T2 would be green on a lie.
//  * lane-independent packed ARITHMETIC, added by the second slice and pinned
//    by the fixtures at the end of this file: `paddq`/`psubq` (two u64 lanes),
//    `paddd`/`psubd` (four u32 lanes per 128 bits), `pcmpeqd`/`pcmpeqq` (per
//    lane equal -> all-ones, else zero). Each lane's answer is a function of
//    that lane's operands only, so no carry, borrow, or comparison result
//    crosses a lane -- the fixtures are built from the carry/borrow cases that
//    make a whole-register model answer a different BYTE.
//  * everything else packed (the shuffles `pshufd`/`shufps`/`punpck*`, the
//    multiplies `pmul*`, the shifts `psrad`/`pslld`, pmovmskb, movmskps...)
//    stays a loud `Op::Opaque` naming the exact mnemonic: the refusal census
//    is the next slice's map, so a folded/renamed gap is a lost map entry.
//    Lane crossing is the boundary -- see
//    `lane_crossing_packed_work_stays_loud_naming_the_mnemonic`.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::interp::{run_budget_seed, Outcome};
use rdec_core::ir::{Function, Op};
use rdec_core::verify::verify_ssa;

/// Lift + the pre-emission pass pipeline, exactly as the C gates run it, so
/// what these tests assert is what the emitter would see.
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

fn seed16(addr: u64, bytes16: &[u8; 16]) -> Vec<(u64, u8)> {
    bytes16
        .iter()
        .enumerate()
        .map(|(i, b)| (addr + i as u64, *b))
        .collect()
}

fn seed8v(addr: u64, v: u64) -> Vec<(u64, u8)> {
    v.to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(i, b)| (addr + i as u64, b))
        .collect()
}

/// The bytes the interpreter holds at `addr..addr+8` (all of them), `None` if
/// any byte was never written.
fn mem8(exec: &rdec_core::interp::Exec, addr: u64) -> Option<[u8; 8]> {
    let mut out = [0u8; 8];
    let mut seen = [false; 8];
    for (a, b) in &exec.mem {
        if *a >= addr && *a < addr + 8 {
            let i = (*a - addr) as usize;
            out[i] = *b;
            seen[i] = true;
        }
    }
    seen.iter().all(|s| *s).then_some(out)
}

/// The bytes the interpreter holds at `addr` (all 16 of them), `None` if any
/// byte was never written. The interpreter's memory is a byte map, so "never
/// written" is a fact the fixture can observe -- a half-store must show it.
fn mem16(exec: &rdec_core::interp::Exec, addr: u64) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    let mut seen = [false; 16];
    for (a, b) in &exec.mem {
        if *a >= addr && *a < addr + 16 {
            let i = (*a - addr) as usize;
            out[i] = *b;
            seen[i] = true;
        }
    }
    seen.iter().all(|s| *s).then_some(out)
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

fn run(f: &Function, inputs: &[u64], seed: &[(u64, u8)]) -> rdec_core::interp::Exec {
    run_budget_seed(f, inputs, 20000, seed)
}

// A 16-byte pattern whose halves DIFFER, so a lane order flipped in a store
// changes the bytes: lo = 00 11 .. 77, hi = 88 99 .. ff.
const PAT: [u8; 16] = *b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff";
const SRC: u64 = 0x1000;
const DST: u64 = 0x2000;

/// `movdqu (%rcx), %xmm0 ; movdqu %xmm0, (%rdx) ; ret` -- a 16-byte copy
/// through one register. Ground truth (`llvm-objdump -d` of these bytes):
///
/// ```text
/// 0: f3 0f 6f 01    movdqu (%rcx), %xmm0
/// 4: f3 0f 7f 02    movdqu %xmm0, (%rdx)
/// 8: c3             ret
/// ```
#[test]
fn movdqu_load_store_round_trips_16_bytes() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x7F, 0x02, // movdqu %xmm0,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let exec = run(&f, &[SRC, DST, 0, 0], &seed16(SRC, &PAT));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        got, PAT,
        "the 16 bytes must arrive in order (lo lane first)"
    );
    // No loud gaps survive: the load AND the store are modelled.
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// All five aligned/unaligned packed move families, each loading the same
/// 16 bytes and storing them 16 bytes apart. Ground truth (quoted from
/// `llvm-objdump -d` of the same bytes; every store uses xmm0):
///
/// ```text
/// movdqa  66 0f 6f 01  /  66 0f 7f 02
/// movaps  0f 28 01     /  0f 29 02
/// movups  0f 10 01     /  0f 11 42 10        (%rdx+0x10)
/// movapd  66 0f 28 01  /  66 0f 29 42 40     (%rdx+0x40)
/// movupd  66 0f 10 01  /  66 0f 11 42 20     (%rdx+0x20)
/// ret     c3
/// ```
#[test]
fn every_packed_move_family_round_trips_16_bytes() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0x6F, 0x01, // movdqa  (%rcx),%xmm0
        0x66, 0x0F, 0x7F, 0x02, // movdqa  %xmm0,(%rdx)
        0x0F, 0x28, 0x01, // movaps  (%rcx),%xmm0
        0x0F, 0x29, 0x02, // movaps  %xmm0,(%rdx)
        0x0F, 0x10, 0x01, // movups  (%rcx),%xmm0
        0x0F, 0x11, 0x42, 0x10, // movups  %xmm0,0x10(%rdx)
        0x66, 0x0F, 0x10, 0x01, // movupd  (%rcx),%xmm0
        0x66, 0x0F, 0x11, 0x42, 0x20, // movupd  %xmm0,0x20(%rdx)
        0x66, 0x0F, 0x28, 0x01, // movapd  (%rcx),%xmm0
        0x66, 0x0F, 0x29, 0x42, 0x40, // movapd  %xmm0,0x40(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let exec = run(&f, &[SRC, DST, 0, 0], &seed16(SRC, &PAT));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    for off in [0u64, 0x10, 0x20, 0x40] {
        let got = mem16(&exec, DST + off).unwrap_or_else(|| panic!("16 bytes at DST+{off:#x}"));
        assert_eq!(
            got, PAT,
            "DST+{off:#x} must hold the 16 bytes, lo lane first"
        );
    }
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `movdqu (%rcx),%xmm0 ; pxor %xmm0,%xmm0 ; movdqu %xmm0,(%rdx) ; ret`.
/// The zeroing idiom is a machine fact about ALL 128 bits, and the fixture
/// proves it reaches BOTH cells: the seed is 16 non-zero bytes, and all 16
/// bytes at the destination must be zero. Ground truth:
///
/// ```text
/// 0: f3 0f 6f 01      movdqu (%rcx), %xmm0
/// 4: 66 0f ef c0      pxor   %xmm0, %xmm0
/// 8: f3 0f 7f 02      movdqu %xmm0, (%rdx)
/// c: c3               ret
/// ```
#[test]
fn pxor_zeroes_both_cells() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0x66, 0x0F, 0xEF, 0xC0, // pxor   %xmm0,%xmm0
        0xF3, 0x0F, 0x7F, 0x02, // movdqu %xmm0,(%rdx)
        0xC3,
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let exec = run(&f, &[SRC, DST, 0, 0], &seed16(SRC, &PAT));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(got, [0u8; 16], "pxor xmm,xmm zeroes all 128 bits");
    assert_eq!(rep.sse_zeroing, 1, "the zeroing idiom is counted");
}

/// The scalar-store hi-preservation, and the ONLY shape in which "a scalar
/// movsd does not clobber the high bits" is true of the hardware: the STORE
/// form (`F2 0F 11 /r`) writes 8 bytes to memory and leaves the register
/// untouched. The fixture proves both halves: the destination's high 8 bytes
/// are NEVER written (a half store leaves them undefined -- visible as
/// `None` in the byte map), and a following 16-byte store of the SAME
/// register still emits the high lane the movdqu load put there. Ground truth:
///
/// ```text
/// 0: f3 0f 6f 01          movdqu (%rcx), %xmm0
/// 4: f2 0f 11 02          movsd  %xmm0, (%rdx)
/// 8: f3 0f 7f 42 10       movdqu %xmm0, 0x10(%rdx)
/// d: c3                   ret
/// ```
#[test]
fn scalar_movsd_store_preserves_the_high_lane() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF2, 0x0F, 0x11, 0x02, // movsd  %xmm0,(%rdx)
        0xF3, 0x0F, 0x7F, 0x42, 0x10, // movdqu %xmm0,0x10(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let mut seeds = seed16(SRC, &PAT);
    // The destination's HIGH half is seeded with a sentinel, and the tail
    // assertion compares against THAT. The original fixture compared
    // `tail == HI` (the register's own hi value), which a spilling
    // `movsd m64` satisfied by writing exactly HI at DST+8 — a sabotage
    // (store16-style spill in the scalar store form) PASSED it. "Not
    // written" must mean "the bytes the store does not own are exactly the
    // bytes that were there before", and the before-bytes must be distinct
    // from every value the program can carry. Test strengthened 2026-11-06
    // by the sabotage it had been hiding.
    seeds.extend(seed16(DST, &[0xA5u8; 16]));
    let exec = run(&f, &[SRC, DST, 0, 0], &seeds);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    // The scalar store wrote exactly its 8 bytes...
    let lo = mem8(&exec, DST);
    assert_eq!(
        lo,
        Some([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]),
        "movsd m64 stores the low 8 bytes"
    );
    let tail = mem8(&exec, DST + 8);
    const KEEP: [u8; 8] = [0xA5; 8]; // the sentinel seeded at DST+8
    assert_eq!(
        tail,
        Some(KEEP),
        "movsd m64 must NOT write bytes 8..16 of its destination: {tail:?}"
    );
    // ...and the register's high lane survived the scalar store: the 16-byte
    // store below it shows the pair intact.
    let got = mem16(&exec, DST + 0x10).expect("16 bytes stored at DST+0x10");
    assert_eq!(got, PAT, "the high lane survived the scalar store");
}

/// The LOAD form of the same instruction is the opposite machine fact:
/// `movsd xmm1, xmm2/m64` zeroes bits 127:64 of the destination. Pinned so
/// "preserve the high lane" cannot be over-generalised into the load path
/// (where the hardware zeroes): the seed at DST+0x10 is a 16-byte pattern,
/// the second load is 0x0123456789ABCDEF, and the answer must be
/// [that 8 bytes][eight zero bytes]. Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01        movdqu (%rcx), %xmm0
/// 4:  f2 0f 10 0a        movsd  (%rdx), %xmm1
/// 8:  f2 0f 10 c1        movsd  %xmm1, %xmm0
/// c:  f3 0f 7f 41 10     movdqu %xmm0, 0x10(%rcx)
/// 11: c3                 ret
/// ```
#[test]
fn movsd_load_form_zeroes_the_high_quadrant() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF2, 0x0F, 0x10, 0x0A, // movsd  (%rdx),%xmm1
        0xF2, 0x0F, 0x10, 0xC1, // movsd  %xmm1,%xmm0
        0xF3, 0x0F, 0x7F, 0x41, 0x10, // movdqu %xmm0,0x10(%rcx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    const L2: u64 = 0x0123456789ABCDEF;
    let seed: Vec<(u64, u8)> = seed16(SRC, &PAT)
        .into_iter()
        .chain(seed8v(DST, L2))
        .collect();
    let exec = run(&f, &[SRC, DST, 0, 0], &seed);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let mut want = [0u8; 16];
    want[..8].copy_from_slice(&L2.to_le_bytes());
    let got = mem16(&exec, SRC + 0x10).expect("16 bytes stored at SRC+0x10");
    assert_eq!(
        got, want,
        "movsd load form: low 8 = the load, high 8 = ZERO (Intel: zeroes 127:64)"
    );
}

/// `movq %rcx, %xmm0` / `movd`/`movq` gpr->xmm zero-extend to the FULL 128
/// bits: the high lane becomes zero. Ground truth:
///
/// ```text
/// 0: 66 48 0f 6e c1     movq %rcx, %xmm0
/// 5: f3 0f 7f 02        movdqu %xmm0, (%rdx)
/// 9: c3                 ret
/// ```
#[test]
fn movq_gpr_to_xmm_zeroes_the_high_lane() {
    let bytes: &[u8] = &[
        0x66, 0x48, 0x0F, 0x6E, 0xC1, // movq %rcx,%xmm0
        0xF3, 0x0F, 0x7F, 0x02, // movdqu %xmm0,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    const V: u64 = 0xDEADBEEFCAFEBABE;
    let exec = run(&f, &[V, DST, 0, 0], &[]);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let mut want = [0u8; 16];
    want[..8].copy_from_slice(&V.to_le_bytes());
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(got, want, "movq gpr->xmm zero-extends to 128 bits");
}

/// Packed `pxor` on two registers: lane-independent bitwise on the full 128
/// bits, two INDEPENDENT 64-bit lane xors. The AT&T listing puts the SOURCE
/// first: `pxor %xmm0, %xmm1` is Intel `PXOR xmm1, xmm0` - the DESTINATION is
/// xmm1, the register the listing prints SECOND, and the store reads the
/// DESTINATION. The seeds make every lane result non-zero and
/// lane-asymmetric at the byte level, so treating the op as 64-bit-only
/// (leaving hi = B_hi) is visible in the bytes. Operands: A (in xmm0) =
/// [0xF00FF00FF00FF00F, 0x0F0F0F0F0F0F0F0F], B (in xmm1, the DEST) =
/// [0x0FF00FF00FF00FF0, 0xF0F0F0F0F0F0F0F0], so the answer is all-ones in
/// BOTH lanes (hand-computed; the fixture spells the answer, not a formula).
/// Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01        movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 0a        movdqu (%rdx), %xmm1
/// 8:  66 0f ef c8        pxor   %xmm0, %xmm1
/// c:  f3 41 0f 7f 09     movdqu %xmm1, (%r9)
/// 11: c3                 ret
/// ```
#[test]
fn packed_pxor_is_two_independent_lane_xors() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0   (A, the source)
        0xF3, 0x0F, 0x6F, 0x0A, // movdqu (%rdx),%xmm1   (B, the destination)
        0x66, 0x0F, 0xEF, 0xC8, // pxor   %xmm0,%xmm1    Intel: PXOR xmm1,xmm0
        0xF3, 0x41, 0x0F, 0x7F, 0x09, // movdqu %xmm1,(%r9)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    // lo/hi lanes spelled per lane: A = [A_lo, A_hi], B = [B_lo, B_hi], and the
    // 128-bit seed constant is [hi << 64 | lo] for to_le_bytes.
    let a: [u8; 16] = 0x0F0F_0F0F_0F0F_0F0F_F00F_F00F_F00F_F00Fu128.to_le_bytes();
    let b: [u8; 16] = 0xF0F0_F0F0_F0F0_F0F0_0FF0_0FF0_0FF0_0FF0u128.to_le_bytes();
    let seed: Vec<(u64, u8)> = seed16(SRC, &a)
        .into_iter()
        .chain(seed16(SRC + 0x10, &b))
        .collect();
    let exec = run(&f, &[SRC, SRC + 0x10, 0, DST], &seed);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
        "pxor must answer all-ones in BOTH lanes (a 64-bit-only model leaves the high lane at B's value)"
    );
}

/// `pand`, `por` and `pandn` as lane pairs, registers seeded from three
/// 16-byte loads. DESTINATIONS are part of the encodings, stated here in Intel
/// reading (modrm reg field = first Intel operand = destination; the AT&T
/// listing prints source first):
///   `pand  %xmm0, %xmm1`  = PAND  xmm1, xmm0 : xmm1 = B & A
///   `por   %xmm2, %xmm1`  = POR   xmm1, xmm2 : xmm1 = (B & A) | C
///   `pandn %xmm2, %xmm0`  = PANDN xmm2, xmm0 : xmm2 = (~C) & A   (dest xmm2!)
/// With A = [0xFFF0FFF0FFF0FFF0, 0x0F0F0F0F0F0F0F0F] (lo/hi lanes),
/// B = [0x0FF00FF00FF00FF0, 0xF0F0F0F0F0F0F0F0],
/// C = [0x00FF00FF00FF00FF, 0xFF00FF00FF00FF00] the answers are
/// xmm1 = [0x0FFF0FFF0FFF0FFF, 0xFF00FF00FF00FF00] and
/// xmm2 = [0x000F000F000F000F, 0xF000F000F000F000] (hand-computed; spelled
/// here, not recomputed). The high lanes are NON-ZERO and lane-asymmetric: a
/// 64-bit-only model of any of the three ops changes the stored bytes;
/// swapping `pandn`'s NOT operand changes them too. The second store reads
/// xmm0, whose pair was NEVER written after its load - if it stores A exactly,
/// the `pandn` (destination xmm2) provably did not touch xmm0's cells.
/// Ground truth (llvm-objdump over these bytes):
///
/// ```text
/// 0:  f3 0f 6f 01        movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 0a        movdqu (%rdx), %xmm1
/// 8:  f3 41 0f 6f 10     movdqu (%r8), %xmm2
/// d:  66 0f db c8        pand   %xmm0, %xmm1
/// 11: 66 0f eb ca        por    %xmm2, %xmm1
/// 15: 66 0f df d0        pandn  %xmm2, %xmm0
/// 19: f3 41 0f 7f 09     movdqu %xmm1, (%r9)
/// 1e: f3 41 0f 7f 41 10  movdqu %xmm0, 0x10(%r9)
/// 24: c3                 ret
/// ```
#[test]
fn pand_pandn_por_are_lane_pairs() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0   A
        0xF3, 0x0F, 0x6F, 0x0A, // movdqu (%rdx),%xmm1   B
        0xF3, 0x41, 0x0F, 0x6F, 0x10, // movdqu (%r8),%xmm2    C
        0x66, 0x0F, 0xDB, 0xC8, // pand   %xmm0,%xmm1    Intel: PAND xmm1,xmm0
        0x66, 0x0F, 0xEB, 0xCA, // por    %xmm2,%xmm1    Intel: POR  xmm1,xmm2
        0x66, 0x0F, 0xDF, 0xD0, // pandn  %xmm2,%xmm0    Intel: PANDN xmm2,xmm0
        0xF3, 0x41, 0x0F, 0x7F, 0x09, // movdqu %xmm1,(%r9)
        0xF3, 0x41, 0x0F, 0x7F, 0x41, 0x10, // movdqu %xmm0,0x10(%r9)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    // [hi << 64 | lo] per register, lanes spelled low-first:
    let a: [u8; 16] = 0x0F0F0F0F0F0F0F0F_FFF0FFF0FFF0FFF0u128.to_le_bytes();
    let b: [u8; 16] = 0xF0F0F0F0F0F0F0F0_0FF00FF00FF00FF0u128.to_le_bytes();
    let c: [u8; 16] = 0xFF00FF00FF00FF00_00FF00FF00FF00FFu128.to_le_bytes();
    let seed: Vec<(u64, u8)> = seed16(SRC, &a)
        .into_iter()
        .chain(seed16(SRC + 0x10, &b))
        .chain(seed16(SRC + 0x20, &c))
        .collect();
    let exec = run(&f, &[SRC, SRC + 0x10, SRC + 0x20, DST], &seed);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST (xmm1)");
    assert_eq!(
        u128::from_le_bytes(got),
        0xFF00_FF00_FF00_FF00_0FFF_0FFF_0FFF_0FFF,
        "pand;por on the destination lane pair must answer the hand-computed value"
    );
    let got2 = mem16(&exec, DST + 0x10).expect("16 bytes stored at DST+0x10 (xmm0)");
    assert_eq!(
        u128::from_le_bytes(got2),
        0x0F0F_0F0F_0F0F_0F0F_FFF0_FFF0_FFF0_FFF0,
        "pandn's destination is xmm2 (modrm reg field = 2): xmm0 must keep A untouched"
    );
}

/// The scope boundary is part of the contract: packed work that is NOT
/// lane-independent -- shuffles, unpacks, multiplies, the shifts by an
/// immediate, and the mask reads -- must stay a loud gap naming the exact
/// mnemonic verbatim: the refusal census is the next lane's map, so each
/// must appear by name and nothing may be silently modelled. These are the
/// corpus's own loop-body forms (`array_walk.O2` emits all six), which is
/// why the boundary is drawn at lane crossing and not at "packed": modelling
/// the lane-independent arithmetic (`paddq`/`paddd`/`psubq`/`psubd`/
/// `pcmpeqd`/`pcmpeqq`, pinned by the fixtures below) does NOT quiet one of
/// these six names. Ground truth (`llvm-objdump -d` of these bytes):
///
/// ```text
/// 0:  66 44 0f 70 df f5     pshufd    $0xf5, %xmm7, %xmm11
/// 6:  66 41 0f f4 fa        pmuludq   %xmm10, %xmm7
/// b:  66 41 0f 62 fa        punpckldq %xmm10, %xmm7
/// 11: 44 0f c6 c9 88        shufps    $0x88, %xmm1, %xmm9
/// 16: 66 0f 72 e3 03        psrad     $0x3, %xmm3
/// 1b: 0f 50 c0              movmskps  %xmm0, %eax
/// 1e: c3                    ret
/// ```
#[test]
fn lane_crossing_packed_work_stays_loud_naming_the_mnemonic() {
    let bytes: &[u8] = &[
        0x66, 0x44, 0x0F, 0x70, 0xDF, 0xF5, // pshufd    $0xf5,%xmm7,%xmm11
        0x66, 0x41, 0x0F, 0xF4, 0xFA, // pmuludq   %xmm10,%xmm7
        0x66, 0x41, 0x0F, 0x62, 0xFA, // punpckldq %xmm10,%xmm7
        0x44, 0x0F, 0xC6, 0xC9, 0x88, // shufps    $0x88,%xmm1,%xmm9
        0x66, 0x0F, 0x72, 0xE3, 0x03, // psrad     $0x3,%xmm3
        0x0F, 0x50, 0xC0, // movmskps %xmm0,%eax
        0xC3,
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let texts = opaque_texts(&f);
    for nm in [
        "pshufd",
        "pmuludq",
        "punpckldq",
        "shufps",
        "psrad",
        "movmskps",
    ] {
        assert!(
            texts.iter().any(|t| t.contains(nm)),
            "{nm} must be a loud gap naming the mnemonic: {texts:?}"
        );
    }
    // Every one of the six is a gap: none is folded, renamed, or swallowed.
    assert_eq!(
        texts.len(),
        6,
        "exactly the six lane-crossing ops stay gaps: {texts:?}"
    );
    assert!(rep.gaps >= 6, "the gaps are counted: {}", rep.gaps);
}

// ---- M8 second slice: lane-independent packed arithmetic -----------------
//
// `paddq`/`paddd`/`psubq`/`psubd`/`pcmpeqd`/`pcmpeqq` are the packed ops whose
// lanes never talk to each other: every lane's answer is a function of that
// lane's two operands alone, so the W64 pair expresses them (the u64 ops
// directly, the u32 ops as the trunc / shift / recombine the fixtures pin by
// bytes). The fixtures are the carry and borrow cases, not the easy ones: in
// each pair the operands are chosen so that a whole-register model -- one
// 128-bit add, one 64-bit add per cell, a mask of 0/1 instead of all-ones --
// answers a DIFFERENT byte than the hardware. Those are the saboteur models
// this file names in its doc comment, and each fixture says which byte moves.
//
// Ground truth for every fixture is `llvm-objdump -d` of the same bytes
// (clang 22.1.8 integrated assembler, no -O), quoted verbatim in the comment.
// The expected answers are hand-computed and spelled as literals, never
// recomputed by the same expression the lifter emits.

/// Seed the two 16-byte operands of a register-register fixture: A at `SRC`
/// (loaded into xmm0 first), B at `SRC + 0x10` (loaded into xmm1 second).
fn seed_pair(a: u128, b: u128) -> Vec<(u64, u8)> {
    seed16(SRC, &a.to_le_bytes())
        .into_iter()
        .chain(seed16(SRC + 0x10, &b.to_le_bytes()))
        .collect()
}

/// `paddq %xmm0, %xmm1`: two INDEPENDENT 64-bit adds, and the carry out of
/// lane 0 goes nowhere. Operands (Intel reading: DEST = xmm1 = B, SRC = xmm0
/// = A), lanes spelled low-first:
///   A = [0xFFFFFFFFFFFFFFFF, 0x0000000000000001]
///   B = [0x0000000000000001, 0x0000000000000002]
/// The answer is [0x0, 0x3]: lane 0 wraps (all-ones + 1) and lane 1 is 1 + 2.
/// A model that adds the register as ONE 128-bit integer answers [0x0, 0x4]
/// -- the lane-0 carry leaks into byte 8, and the assertion names it. Ground
/// truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f d4 c8       paddq  %xmm0, %xmm1
/// d:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 11: c3                ret
/// ```
#[test]
fn paddq_is_two_independent_u64_adds() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0xD4, 0xC8, // paddq  %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x0000_0000_0000_0001_FFFF_FFFF_FFFF_FFFF;
    let b: u128 = 0x0000_0000_0000_0002_0000_0000_0000_0001;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x0000_0000_0000_0003_0000_0000_0000_0000,
        "lane 1 must be 1+2 = 3; a 128-bit add leaks lane 0's carry and answers 4 in byte 8"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `paddq 0x40(%rcx), %xmm0`: the memory-source form, which is the form the
/// corpus witness emits -- `copy_words.O2` adds four `.rdata` vector
/// constants, all `paddq (%rip), %xmm`. The 16 bytes at the operand address
/// are the source pair: two W64 loads, lo at addr, hi at addr+8.
///   A (dest, at SRC)   = [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
///   B (memory, at SRC+0x40) = [0x1, 0x1]
/// The answer is all zero: each lane wraps on its own. One 128-bit add
/// answers [0x0, 0x1] -- the carry out of lane 0 lands in byte 8. Ground
/// truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  66 0f d4 41 40    paddq  0x40(%rcx), %xmm0
/// 9:  f3 0f 7f 02       movdqu %xmm0, (%rdx)
/// d:  c3                ret
/// ```
#[test]
fn paddq_reads_a_memory_operand_as_a_lane_pair() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0x66, 0x0F, 0xD4, 0x41, 0x40, // paddq  0x40(%rcx),%xmm0
        0xF3, 0x0F, 0x7F, 0x02, // movdqu %xmm0,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = u128::MAX;
    let b: u128 = 0x0000_0000_0000_0001_0000_0000_0000_0001;
    let seed: Vec<(u64, u8)> = seed16(SRC, &a.to_le_bytes())
        .into_iter()
        .chain(seed16(SRC + 0x40, &b.to_le_bytes()))
        .collect();
    let exec = run(&f, &[SRC, DST, 0, 0], &seed);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x0000_0000_0000_0000_0000_0000_0000_0000,
        "both lanes wrap to zero; a 128-bit add answers 1 in byte 8"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `paddd %xmm0, %xmm1`: FOUR independent 32-bit adds. Lanes (a0..a3) are
/// spelled low-first inside each cell:
///   A = [0xFFFFFFFF, 0x00000001, 0x00000003, 0x00000002]
///   B = [0x00000001, 0xFFFFFFFF, 0x00000005, 0x00000004]
/// The answer is [0, 0, 8, 6]: lane 0 wraps, and lane 1 (1 + all-ones) wraps
/// too, with NO carry into lane 2. Model the cell as one 64-bit add and lane 1
/// answers 1 -- byte 4 is where the two models disagree (0x00 vs 0x01). The
/// high cell is the control: its two lanes add without a carry, so both
/// models agree there and the fixture proves the red is lane 1, not the cell.
/// Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f fe c8       paddd  %xmm0, %xmm1
/// d:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 11: c3                ret
/// ```
#[test]
fn paddd_is_four_independent_u32_adds() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0xFE, 0xC8, // paddd  %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x0000_0002_0000_0003_0000_0001_FFFF_FFFF;
    let b: u128 = 0x0000_0004_0000_0005_FFFF_FFFF_0000_0001;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x0000_0006_0000_0008_0000_0000_0000_0000,
        "lane 1 must wrap to 0; a 64-bit add of the cell answers 1 and moves byte 4"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `psubq %xmm0, %xmm1`: two INDEPENDENT 64-bit subs, borrow stays in its
/// lane. DEST = xmm1 = B, SRC = xmm0 = A:
///   A = [0x0000000000000001, 0x0000000000000001]
///   B = [0x0000000000000000, 0x1234567890ABCDEF]
/// The answer is [0xFFFFFFFFFFFFFFFF, 0x1234567890ABCDEE]: lane 0 borrows
/// (0 - 1), lane 1 is a plain -1. A 128-bit sub answers ...CDED in the high
/// lane, because lane 0's borrow crossed. Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f fb c8       psubq  %xmm0, %xmm1
/// d:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 11: c3                ret
/// ```
#[test]
fn psubq_borrow_stays_in_its_lane() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0xFB, 0xC8, // psubq  %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x0000_0000_0000_0001_0000_0000_0000_0001;
    let b: u128 = 0x1234_5678_90AB_CDEF_0000_0000_0000_0000;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x1234_5678_90AB_CDEE_FFFF_FFFF_FFFF_FFFF,
        "the low lane's borrow must not reach the high lane (a 128-bit sub answers ...CDED)"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `psubd %xmm0, %xmm1`: FOUR independent 32-bit subs, and this fixture is
/// the one a "sub = add of the negation" model must survive -- negating a
/// 64-bit cell and adding, without truncating each lane, propagates the
/// borrow exactly like a 64-bit subtract. DEST = xmm1 = B, SRC = xmm0 = A:
///   A = [0x00000001, 0x00000000, 0x80000001, 0x00000000]
///   B = [0x00000000, 0x00000001, 0x80000000, 0x12345678]
/// The answer is [0xFFFFFFFF, 0x00000001, 0xFFFFFFFF, 0x12345678]. A 64-bit
/// (or negate-and-add, untruncated) model answers lane 1 = 0x00000000 (byte 4)
/// and lane 3 = 0x12345677 (byte 12): two lanes show the crossing, one per
/// cell, so a fix that only truncates the low cell stays red. Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f fa c8       psubd  %xmm0, %xmm1
/// d:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 11: c3                ret
/// ```
#[test]
fn psubd_borrow_never_crosses_a_lane() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0xFA, 0xC8, // psubd  %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x0000_0000_8000_0001_0000_0000_0000_0001;
    let b: u128 = 0x1234_5678_8000_0000_0000_0001_0000_0000;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x1234_5678_FFFF_FFFF_0000_0001_FFFF_FFFF,
        "lane 1 must be 1 and lane 3 must be 0x12345678: a model that subtracts the cell answers 0 and 0x12345677 (bytes 4 and 12)"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `pcmpeqd %xmm0, %xmm1`: per 32-bit lane, equal -> all-ones, else zero. The
/// mask is the assertion -- a cast of the comparison to 0/1 is the same
/// boolean and the wrong bytes. DEST = xmm1 = B, SRC = xmm0 = A:
///   A = [0x00000007, 0x00010000, 0xDEADBEEF, 0x5A5A5A5A]
///   B = [0x00000007, 0x00000001, 0xDEADBEEF, 0x5A5A5A5A]
/// Lanes 0, 2, 3 are equal, lane 1 is not, so the answer is
/// [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF]. Two saboteurs die here:
/// a 0/1 mask answers 1 where the hardware answers all-ones (bytes 0-3), and
/// a compare of the whole 64-bit CELL answers zero for the entire low cell,
/// because its two lanes are unequal as a pair (bytes 0-3 again, and the
/// fixture's high cell is all-ones while its low cell is not, so a model that
/// treats the cells alike also moves). Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f 76 c8       pcmpeqd %xmm0, %xmm1
/// d:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 11: c3                ret
/// ```
#[test]
fn pcmpeqd_answers_an_all_ones_lane_mask() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0x76, 0xC8, // pcmpeqd %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x5A5A_5A5A_DEAD_BEEF_0001_0000_0000_0007;
    let b: u128 = 0x5A5A_5A5A_DEAD_BEEF_0000_0001_0000_0007;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0xFFFF_FFFF_FFFF_FFFF_0000_0000_FFFF_FFFF,
        "equal lanes are ALL ONES and lane 1 (unequal) is zero: 0x00000000FFFFFFFF in the low cell, all-ones in the high cell"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `pcmpeqq %xmm0, %xmm1`: the 64-lane sibling (SSE2, `66 0f 38 29 /r`).
///   A = [0x0000000000000005, 0x1122334455667788]
///   B = [0x0000000000000005, 0x1122334455667789]
/// Lane 0 is equal, lane 1 differs, so the answer is the lane-asymmetric
/// [0xFFFFFFFFFFFFFFFF, 0x0]: a model that compares the whole 128 bits answers
/// zero in BOTH lanes and dies on bytes 0-7. Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  f3 0f 6f 49 10    movdqu 0x10(%rcx), %xmm1
/// 9:  66 0f 38 29 c8    pcmpeqq %xmm0, %xmm1
/// e:  f3 0f 7f 0a       movdqu %xmm1, (%rdx)
/// 12: c3                ret
/// ```
#[test]
fn pcmpeqq_answers_an_all_ones_lane_mask() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0xF3, 0x0F, 0x6F, 0x49, 0x10, // movdqu 0x10(%rcx),%xmm1
        0x66, 0x0F, 0x38, 0x29, 0xC8, // pcmpeqq %xmm0,%xmm1
        0xF3, 0x0F, 0x7F, 0x0A, // movdqu %xmm1,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x1122_3344_5566_7788_0000_0000_0000_0005;
    let b: u128 = 0x1122_3344_5566_7789_0000_0000_0000_0005;
    let exec = run(&f, &[SRC, DST, 0, 0], &seed_pair(a, b));
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
        "lane 0 is all-ones and lane 1 is zero: a 128-bit compare answers zero in both"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}

/// `pcmpeqd 0x40(%rcx), %xmm0`: the memory-source form of the mask op, where
/// the DESTINATION is the register the listing prints second (Intel `DEST =
/// xmm0`). A = the register, B = the 16 bytes at SRC+0x40:
///   A = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
///   B = [0x11111111, 0x22222222, 0x33333334, 0x44444444]
/// The answer is [all-ones, all-ones, 0, all-ones]: the mask is
/// lane-asymmetric ACROSS the cells (low cell all-ones, high cell
/// 0xFFFFFFFF00000000), so the hi half of the memory operand must be the
/// bytes at addr+8 -- read lo twice, or ignore the high cell, and bytes
/// 8..16 move. Ground truth:
///
/// ```text
/// 0:  f3 0f 6f 01       movdqu (%rcx), %xmm0
/// 4:  66 0f 76 41 40    pcmpeqd 0x40(%rcx), %xmm0
/// 9:  f3 0f 7f 02       movdqu %xmm0, (%rdx)
/// d:  c3                ret
/// ```
#[test]
fn pcmpeqd_reads_a_memory_operand_as_a_lane_pair() {
    let bytes: &[u8] = &[
        0xF3, 0x0F, 0x6F, 0x01, // movdqu (%rcx),%xmm0
        0x66, 0x0F, 0x76, 0x41, 0x40, // pcmpeqd 0x40(%rcx),%xmm0
        0xF3, 0x0F, 0x7F, 0x02, // movdqu %xmm0,(%rdx)
        0xC3,
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let a: u128 = 0x4444_4444_3333_3333_2222_2222_1111_1111;
    let b: u128 = 0x4444_4444_3333_3334_2222_2222_1111_1111;
    let seed: Vec<(u64, u8)> = seed16(SRC, &a.to_le_bytes())
        .into_iter()
        .chain(seed16(SRC + 0x40, &b.to_le_bytes()))
        .collect();
    let exec = run(&f, &[SRC, DST, 0, 0], &seed);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    let got = mem16(&exec, DST).expect("16 bytes stored at DST");
    assert_eq!(
        u128::from_le_bytes(got),
        0xFFFF_FFFF_0000_0000_FFFF_FFFF_FFFF_FFFF,
        "only lane 2 differs, so the low cell is all-ones and the high cell is 0xFFFFFFFF00000000"
    );
    assert_eq!(opaque_texts(&f), Vec::<String>::new());
}
