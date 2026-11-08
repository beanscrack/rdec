// The scalar SSE subset, pinned on hand-assembled bytes -- no corpus file is
// read here, so every number below is checkable against the encoding in the
// comment above it.
//
// What these tests pin, and why each one is the shape it is:
//
//   movsd [rdx], xmm0 ; movsd xmm1, [rdx]  - a copy through memory. The cell is
//                                  BITS, so the store and the load must move the
//                                  pattern, not a converted number.
//   addsd + cvttsd2si at -1.75     - truncation, not rounding: -1 is the x86
//                                  answer, -2 is what a rounding conversion
//                                  would print. Inputs chosen so the two differ.
//   pxor xmm0, xmm0                - writes 0 and emits NO read of the previous
//                                  xmm0, the same contract `self_idiom.rs` pins
//                                  for `xor r,r`, for the same reason: this
//                                  interpreter is strict, so deleting a read of
//                                  undef turns a faulting run into a returning
//                                  one. The general fold is a lie; the encoding
//                                  is the fact.
//   cvttsd2si of 0.0/0.0           - NaN lands on 0x8000000000000000. In C,
//                                  `(int64_t)NaN` is undefined behaviour, so the
//                                  emitted C carries a guard, not a cast.
//   cvtsi2sd xmm0, eax of -1       - the source is read at ITS 32-bit width and
//                                  converted AS SIGNED. Reading the 64-bit
//                                  register unsigned gives 4294967295.0, a
//                                  different number, and the 64-bit destination
//                                  in the test is what makes the two visible.
//   paddd / shufps / andps / comisd - refused by name, at the address, counted.
//                                  The `comisd` refusal is not laziness about
//                                  flags: ZF/CF/PF is a three-bit answer and the
//                                  `jae`/`jp` spellings of it are not one boolean.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::{lift_one, Report};
use rdec_core::interp::{run_budget, Outcome};
use rdec_core::ir::{Function, Op};
use rdec_core::verify::verify_ssa;

fn lift(bytes: &[u8]) -> (Function, Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    // Hand-assembled bytes have no symbol table, so seed the entry address.
    // Seed the entry address AND the four integer argument registers: the
    // discovery plan's arg list is what makes R8 and R9 live-in parameters, and
    // a seed that names one argument leaves `cvtsi2sd xmm4, r8` reading undef.
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64; 4], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 1, "one discovered function");
    let start = plan.functions[0];
    let end = owned.len() as u64;
    let (mut f, rep) = lift_one("f", &owned, 0, start, end, &plan, &Default::default());
    // The pre-emission pipeline, as the C gates run it: what the tests assert is
    // what the emitter would see.
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

/// The bytes the machine wrote at `addr`, from the interpreter's memory
/// snapshot. `None` if nothing was stored there.
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

/// Reads of the shared `undef` value. The lifter always PLACES `undef` in the
/// entry block, so presence says nothing; the claim that carries weight is that
/// nothing consumes it.
fn undef_reads(f: &Function) -> usize {
    let Some(v) = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Undef))
        .map(|i| i.dest)
    else {
        return 0;
    };
    f.use_count(v)
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

/// `pxor xmm0,xmm0 ; cvtsi2sd xmm0,rcx ; movsd [rdx],xmm0 ; movsd xmm1,[rdx] ;
///  cvttsd2si eax,xmm1 ; ret`
///
/// A copy through memory: 7.0 goes in, 7.0 comes out, and the eight bytes in
/// between are 7.0's pattern (0x401C000000000000), not the integer 7.
#[test]
fn movsd_store_then_load_is_a_copy_through_memory() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor   xmm0, xmm0
        0xF2, 0x48, 0x0F, 0x2A, 0xC1, // cvtsi2sd xmm0, rcx
        0xF2, 0x48, 0x0F, 0x11, 0x02, // movsd  [rdx], xmm0
        0xF2, 0x0F, 0x10, 0x0A, // movsd  xmm1, [rdx]
        0xF2, 0x0F, 0x2C, 0xC1, // cvttsd2si eax, xmm1
        0xC3, // ret
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    let exec = run_budget(&f, &[7, 0, 0, 0], 20000);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    // The answer, as RAX: eax = cvttsd2si(7.0) = 7.
    match exec.outcome {
        Outcome::Ret(rdec_core::interp::Val::Int(v)) => assert_eq!(v, 7),
        other => panic!("unexpected: {other:?}"),
    }
    // And the memory in the middle holds the PATTERN, not the integer 7.
    let got = mem8(&exec, 0).expect("8 bytes stored at 0");
    assert_eq!(
        u64::from_le_bytes(got),
        7.0f64.to_bits(),
        "the store must write the double's bit pattern"
    );
}

/// `pxor xmm0,xmm0 ; cvtsi2sd xmm1,rcx(3) ; cvtsi2sd xmm2,rdx(2) ;
///  movapd xmm3,xmm1 ; divsd xmm3,xmm2 (1.5) ; subsd xmm0,xmm3 (-1.5) ;
///  cvtsi2sd xmm4,r8(1) ; cvtsi2sd xmm5,r9(4) ; divsd xmm4,xmm5 (0.25) ;
///  subsd xmm0,xmm4 (-1.75) ; cvttsd2si eax,xmm0 ; ret`
///
/// -1.75 is the input that separates the answers: truncate toward zero is -1,
/// round-to-nearest is -2. `addsd`/`subsd`/`divsd`/`movapd` are all exercised on
/// the way there, at F64.
#[test]
fn addsd_subsd_divsd_then_cvttsd2si_truncates_toward_zero() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor   xmm0, xmm0     -> 0.0
        0xF2, 0x48, 0x0F, 0x2A, 0xC9, // cvtsi2sd xmm1, rcx    -> 3.0
        0xF2, 0x48, 0x0F, 0x2A, 0xD2, // cvtsi2sd xmm2, rdx    -> 2.0
        0x66, 0x0F, 0x28, 0xD9, // movapd xmm3, xmm1   -> 3.0 (the copy)
        0xF2, 0x0F, 0x5E, 0xDA, // divsd  xmm3, xmm2   -> 1.5
        0xF2, 0x0F, 0x5C, 0xC3, // subsd  xmm0, xmm3   -> -1.5
        0xF2, 0x49, 0x0F, 0x2A, 0xE0, // cvtsi2sd xmm4, r8     -> 1.0
        0xF2, 0x49, 0x0F, 0x2A, 0xE9, // cvtsi2sd xmm5, r9     -> 4.0
        0xF2, 0x0F, 0x5E, 0xE5, // divsd  xmm4, xmm5     -> 0.25
        0xF2, 0x0F, 0x5C, 0xC4, // subsd  xmm0, xmm4   -> -1.75
        0xF2, 0x0F, 0x2C, 0xC0, // cvttsd2si eax, xmm0 -> -1
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    assert!(
        opaque_texts(&f).is_empty(),
        "every mnemonic here is modelled: {:?}",
        opaque_texts(&f)
    );
    assert_eq!(
        rep.sse_zeroing, 1,
        "the pxor is counted as the zeroing idiom"
    );
    assert_eq!(rep.sse.get("divsd"), Some(&2));
    assert_eq!(rep.sse.get("subsd"), Some(&2));
    assert_eq!(rep.sse.get("movapd"), Some(&1));
    match answer(&f, &[3, 2, 1, 4]) {
        // eax = -1, and a 32-bit write zeroes RAX's upper half.
        Outcome::Ret(v) => assert_eq!(v.bits(), 0xFFFF_FFFF, "-1.75 truncates to -1"),
        other => panic!("unexpected: {other:?}"),
    }
}

/// `pxor xmm0,xmm0 ; movsd [rdx],xmm0 ; mov eax,0 ; ret`
///
/// The zeroing idiom: the register becomes 0 and is NOT read. The `undef` the
/// lifter places in the entry block must have NO readers -- that is the
/// assertion that carries the claim, exactly as in the `xor r,r` test. The
/// general IR fold `xor(x,x) -> 0` would have had to delete that read, and here
/// deleting a read changes which executions fault.
///
/// `mov eax,0` is in the bytes for a boring reason: without it the function's
/// `ret` returns RAX, whose live-in is the shared `undef`, and the assertion
/// below would be measuring the return value rather than the idiom.
#[test]
fn pxor_zeroing_writes_zero_and_reads_nothing() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor  xmm0, xmm0
        0xF2, 0x48, 0x0F, 0x11, 0x02, // movsd [rdx], xmm0
        0xB8, 0x00, 0x00, 0x00, 0x00, // mov   eax, 0
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(undef_reads(&f), 0, "the idiom emits no read of xmm0");
    assert_eq!(rep.sse_zeroing, 1);
    assert_eq!(rep.sse.get("pxor"), Some(&1));
    let exec = run_budget(&f, &[0, 0, 0, 0], 20000);
    assert!(
        matches!(exec.outcome, Outcome::Ret(_)),
        "{:?}",
        exec.outcome
    );
    assert_eq!(mem8(&exec, 0), Some([0u8; 8]), "the stored value is +0.0");
    // And no `xor` of any kind reached the IR: the fold is not how this is modelled.
    let has_xor = f.blocks.iter().flat_map(|b| b.instrs.iter()).any(|i| {
        matches!(
            i.op,
            Op::Bin(rdec_core::ir::BinOp::Xor, _, _) | Op::Un(rdec_core::ir::UnOp::Not, _)
        )
    });
    assert!(!has_xor, "the zeroing is a constant, not an xor of undef");
}

/// `pxor xmm0,xmm0 ; pxor xmm1,xmm1 ; divsd xmm0,xmm1 (0.0/0.0 = NaN) ;
///  cvttsd2si rax,xmm0 ; ret`
///
/// x86 answers NaN with 0x8000000000000000. C's `(int64_t)NaN` is undefined
/// behaviour, so the IR carries a conversion the emitter must spell as a GUARD
/// (compare, select the magic) -- never a bare cast. A cast is the bug this
/// test exists to catch: on clang with the default fpu it happens to print
/// INT64_MIN, and on an optimiser that exploits the UB it prints anything.
#[test]
fn cvttsd2si_of_nan_lands_on_the_x86_magic_not_a_c_cast() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor   xmm0, xmm0
        0x66, 0x0F, 0xEF, 0xC9, // pxor   xmm1, xmm1
        0xF2, 0x0F, 0x5E, 0xC1, // divsd  xmm0, xmm1  -> NaN
        0xF2, 0x48, 0x0F, 0x2C, 0xC0, // cvttsd2si rax, xmm0
        0xC3, // ret
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    match answer(&f, &[]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 0x8000_0000_0000_0000),
        other => panic!("unexpected: {other:?}"),
    }
    // The emitted C carries the guard. The text assertion is the point: a bare
    // `(int64_t)x` here is C-UB, and the guard is what must be in the file.
    let out = rdec_core::emit_c::emit_c(&f).expect("the scalar subset emits");
    let c = out.text;
    println!("{c}");
    assert!(
        c.contains("0x8000000000000000ULL"),
        "the x86 magic constant is not in the emitted C:\n{c}"
    );
    // The range test, in the spelling that makes NaN fail it: a comparison
    // against the bounds, `!`-ed, so an unordered source picks the magic.
    assert!(
        c.contains(">= -9.223372036854776e18") && c.contains("< 9.223372036854776e18"),
        "the cvttsd2si range guard is not in the emitted C:\n{c}"
    );
    assert!(
        !c.contains("(int64_t)(x") || c.contains("? 0x8000000000000000ULL"),
        "a bare (int64_t) cast of a float is C-UB; the guard must wrap it:\n{c}"
    );
}

/// `pxor xmm0,xmm0 ; cvtsi2ss xmm0,ecx(=v) ; cvttss2si eax|xmm0 -> eax/rax ; ret`
/// — the 32-bit source SSE1 form, both destination widths, then the REX.W
/// source refusal.
///
/// EVERY expectation below is a silicon measurement, not a derivation: the
/// exact byte sequences were executed on this host (`VirtualAlloc`
/// PAGE_EXECUTE_READWRITE + a call with the argument in `ecx`), and the machine's
/// answers are quoted verbatim. That discipline is the point of this fixture's
/// history — the previous version asserted two things the hardware does not do:
///
///   1. "v = 0xCB01F9D0 is out of i32 range, so the magic fires". FALSE on both
///      counts. Read as int32 it is -889062960, which is IN range, and it is NOT
///      f32-exact: at that magnitude a float's ULP is 2^(29-23) = 64, and
///      889062960 = 64 x 13891608.75, so `cvtsi2ss` answers -889062976.0. The
///      machine's answers are eax = 0xCB01F9C0, rax = -889062976 — the f32
///      rounding is observable in the answer, which is exactly what makes this
///      leg worth pinning (the old fixture's -889062960 expectation, in range
///      both ways, hid it).
///   2. the magic leg was aimed at the wrong value. It fires for
///      v = 0x7FFFFFFF: 2147483647 rounds to the nearest float, 2^31
///      (ULP 128, distance 1 vs 127), and 2^31 is OUT of i32 range ->
///      eax = 0x80000000. The same value in the 64-bit destination is IN range
///      -> rax = +2147483648. One value, magic on one side, exact on the other.
///
/// The low-end pair is the discriminator: v = 0x80000000 read as int32 is
/// -2^31, which IS exactly representable and converts exactly, so eax answers
/// 0x80000000 for the OPPOSITE reason (a real conversion, not the invalid-op
/// magic). The rax leg separates them: -2147483648 for the real conversion,
/// 0x8000000000000000 for the magic. Measured: eax = 0x80000000, rax =
/// 0xFFFFFFFF80000000.
///
/// `F3 REX.W 0F 2A` is `cvtsi2ssq`, the SSSE3 64-bit-source extension: the
/// model speaks SSE1-plus, so it is refused by name as a named gap, counted,
/// and the refusal is asserted here — the form must not be silently modelled
/// at 64 bits.
#[test]
fn cvttss2si_answers_the_machine_f32_rounding_and_the_i32_magic() {
    let i32_form: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor     xmm0, xmm0
        0xF3, 0x0F, 0x2A, 0xC1, // cvtsi2ss xmm0, ecx
        0xF3, 0x0F, 0x2C, 0xC0, // cvttss2si eax, xmm0
        0xC3, // ret
    ];
    let (f, rep) = lift(i32_form);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "the SSE1 32-bit source form is modelled");

    // Machine: eax = 0xCB01F9C0. An f64 model (no f32 rounding) answers
    // 0xCB01F9D0; the machine answers the rounded -889062976.
    match answer(&f, &[0xCB01F9D0, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(
            v.bits(),
            0xCB01_F9C0,
            "f32(-889062960) = -889062976: the f32 ULP is 64 at this magnitude"
        ),
        other => panic!("unexpected: {other:?}"),
    }
    // Machine: eax = 0x80000000 — the invalid-op magic (2^31 does not fit i32).
    match answer(&f, &[0x7FFF_FFFF, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 0x8000_0000, "the i32 magic, not 0x7FFFFFFF"),
        other => panic!("unexpected: {other:?}"),
    }
    // Machine: eax = 0x80000000 — the SAME bits from the opposite cause: -2^31
    // is f32-exact and converts exactly. The i64 leg below is what tells the
    // two apart.
    match answer(&f, &[0x8000_0000, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 0x8000_0000, "-2^31 converts exactly"),
        other => panic!("unexpected: {other:?}"),
    }
    // Machine: eax = 0x01000000 from 16777217 — the 2^24 f32 rounding wall.
    match answer(&f, &[16_777_217, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 16_777_216, "the f32 rounding is modelled"),
        other => panic!("unexpected: {other:?}"),
    }

    let i64_form: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor      xmm0, xmm0
        0xF3, 0x0F, 0x2A, 0xC1, // cvtsi2ss  xmm0, ecx
        0xF3, 0x48, 0x0F, 0x2C, 0xC0, // cvttss2si rax, xmm0
        0xC3, // ret
    ];
    let (f, rep) = lift(i64_form);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.gaps, 0, "the SSE1 32-bit source form is modelled");
    // Machine: rax = -889062976 (NOT -889062960): the f32 leg of the rounding,
    // and not the magic, which the i32 leg above needed.
    match answer(&f, &[0xCB01F9D0, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits() as i64, -889_062_976, "rounded by f32, signed"),
        other => panic!("unexpected: {other:?}"),
    }
    // Machine: rax = +2147483648 — in i64 range, so the magic must NOT fire.
    match answer(&f, &[0x7FFF_FFFF, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits(), 0x8000_0000, "2^31 fits i64: exact, not magic"),
        other => panic!("unexpected: {other:?}"),
    }
    // Machine: rax = 0xFFFFFFFF80000000 = -2147483648. If the i32 leg's
    // 0x80000000 had been the magic, this leg would read 0x8000000000000000.
    match answer(&f, &[0x8000_0000, 0, 0, 0]) {
        Outcome::Ret(v) => assert_eq!(v.bits() as i64, -2_147_483_648, "the low-end conversion"),
        other => panic!("unexpected: {other:?}"),
    }

    // The SSSE3 extension stays a named refusal.
    let ssq: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor     xmm0, xmm0
        0xF3, 0x48, 0x0F, 0x2A, 0xC1, // cvtsi2ss xmm0, rcx  (REX.W: cvtsi2ssq)
        0xF3, 0x48, 0x0F, 0x2C, 0xC0, // cvttss2si rax, xmm0
        0xC3, // ret
    ];
    let (f, rep) = lift(ssq);
    verify_ssa(&f).expect("valid SSA: a refusal is still IR");
    assert_eq!(rep.gaps, 1, "the REX.W source form is a gap, counted");
    assert!(
        opaque_texts(&f).iter().any(|t| t.contains("cvtsi2ss")),
        "the refusal does not name the mnemonic -> {:?}",
        opaque_texts(&f)
    );
}

/// `mov eax,-1 ; cvtsi2sd xmm0,eax ; cvttsd2si rax,xmm0 ; ret`
///
/// The source is read at ITS 32-bit width and converted AS SIGNED. Reading the
/// 64-bit cell unsigned would give 4294967295.0, and the 64-bit destination on
/// the next line is what makes the two answers visible: -1 vs 4294967295.
#[test]
fn cvtsi2sd_reads_the_source_at_its_width_as_signed() {
    let bytes: &[u8] = &[
        0xB8, 0xFF, 0xFF, 0xFF, 0xFF, // mov      eax, -1
        0xF2, 0x0F, 0x2A, 0xC0, // cvtsi2sd xmm0, eax
        0xF2, 0x48, 0x0F, 0x2C, 0xC0, // cvttsd2si rax, xmm0
        0xC3, // ret
    ];
    let (f, _rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    match answer(&f, &[]) {
        Outcome::Ret(v) => assert_eq!(
            v.bits() as i64,
            -1,
            "cvtsi2sd of a 32-bit -1 is -1.0, not 4294967295.0"
        ),
        other => panic!("unexpected: {other:?}"),
    }
}

/// The packed subset and the ordered compare: refused by name, at the address,
/// counted as gaps. A refusal that does not name the mnemonic is the failure
/// mode this pins, because the refusal classes are the work list.
#[test]
fn packed_and_ordered_compare_are_refused_by_name() {
    let cases: &[(&str, &[u8])] = &[
        // `paddd` MOVED OUT of this table with the M8 packed-ARITHMETIC slice:
        // the lane-independent packed forms (paddq/paddd/psubq/psubd,
        // pcmpeqd/pcmpeqq) are per-lane IR over the XMM pair cells, so they are
        // IR now, not refusals. Positive coverage — per-lane carry/borrow
        // isolation and the all-ones mask, ground-truthed against llvm-objdump
        // — is `packed_simd.rs`. The lane-crossing forms below (shufps, andps)
        // and the ordered compare stay refused by name and stay tested here.
        (
            "shufps",
            &[
                0x0F, 0xC6, 0xC1, 0x00, // shufps xmm0, xmm1, 0
                0xC3,
            ][..],
        ),
        (
            "andps",
            &[
                0x0F, 0x54, 0xC1, // andps  xmm0, xmm1
                0xC3,
            ][..],
        ),
        (
            // `movapd` MOVED OUT of this table on 2026-11-06: the M8 packed
            // slice models the 16-byte load/store family (movap{s,d},
            // movdqa/movdqu/movups/movupd) over the XMM pair cells, so a
            // 128-bit store is IR now, not a refusal. Positive coverage
            // (16-byte round-trip, hi lane included, ground-truthed against
            // llvm-objdump) is `packed_simd.rs`; the mnemonics below stay
            // refused by name and stay tested here.
            "comisd",
            &[
                0x66, 0x0F, 0x2F, 0xC1, // comisd xmm0, xmm1
                0xC3,
            ][..],
        ),
        (
            "cvtsd2si",
            &[
                0x66, 0x0F, 0xEF, 0xC0, // pxor      xmm0, xmm0
                0xF2, 0x0F, 0x2D, 0xC0, // cvtsd2si eax, xmm0 (rounding form)
                0xC3,
            ][..],
        ),
    ];
    for (nm, bytes) in cases {
        let (f, rep) = lift(bytes);
        verify_ssa(&f).expect("valid SSA: a refusal is still IR");
        assert!(rep.gaps >= 1, "{nm}: the refusal was not counted as a gap");
        assert!(
            opaque_texts(&f).iter().any(|t| t.contains(nm)),
            "{nm}: the refusal does not name the mnemonic -> {:?}",
            opaque_texts(&f)
        );
        assert_eq!(
            rep.sse.get(*nm),
            Some(&1),
            "{nm}: the census counts what the decoder saw, modelled or refused"
        );
    }
}

/// The census is the auditable number: the per-mnemonic map the gate prints has
/// to add up to the instructions the lifter routed through the SSE path, and the
/// modelled ones must not be counted as gaps.
#[test]
fn census_counts_modelled_and_refused_together() {
    let bytes: &[u8] = &[
        0x66, 0x0F, 0xEF, 0xC0, // pxor   xmm0, xmm0
        0xF2, 0x48, 0x0F, 0x2A, 0xC1, // cvtsi2sd xmm0, rcx
        0xF2, 0x0F, 0x58, 0xC1, // addsd  xmm0, xmm1
        0x0F, 0xC6, 0xC1, 0x00, // shufps xmm0, xmm1, 0  (refused: lane-crossing)
        0xF2, 0x48, 0x0F, 0x2C, 0xC0, // cvttsd2si rax, xmm0
        0xC3, // ret
    ];
    let (f, rep) = lift(bytes);
    verify_ssa(&f).expect("valid SSA");
    assert_eq!(rep.sse.get("pxor"), Some(&1));
    assert_eq!(rep.sse.get("cvtsi2sd"), Some(&1));
    assert_eq!(rep.sse.get("addsd"), Some(&1));
    assert_eq!(rep.sse.get("shufps"), Some(&1));
    assert_eq!(rep.sse.get("cvttsd2si"), Some(&1));
    assert_eq!(
        rep.sse.values().sum::<u32>(),
        5,
        "five SSE instructions, counted once each"
    );
    // `paddd` used to sit in this stream as the refused one; the M8
    // packed-ARITHMETIC slice models it per lane, so the refused slot is now
    // `shufps`, a shuffle the lanes cannot express. The census invariant is
    // unchanged: modelled and refused are counted together, gaps are refusals.
    assert_eq!(rep.gaps, 1, "only the lane-crossing shuffle is a gap");
    assert_eq!(rep.sse_zeroing, 1, "pxor is the zeroing idiom");
}
