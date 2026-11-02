//! M1 decode gates.
//!
//! The roadmap requires two things of the decoder before anything is lifted:
//! boundaries agree with llvm-objdump on a real object, and the decoder never
//! panics on arbitrary bytes. The second one lives here; the first is
//! `rdec objdump-check`, which needs the binary and a corpus.

use rdec_arch_x86::decode::{decode, round_trip, Syntax};

/// Deterministic pseudo-random bytes: real coverage without a fuzzing harness,
/// and reproducible from the seed.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
}

#[test]
fn decoding_random_bytes_never_panics_and_always_advances() {
    let mut rng = Lcg(0x5eed_2026_1102);
    let mut windows = 0usize;
    let mut insns = 0usize;
    let mut refusals = 0usize;
    for _ in 0..40_000 {
        let n = 1 + (rng.next() % 24) as usize;
        let mut buf = vec![0u8; n];
        for b in buf.iter_mut() {
            *b = (rng.next() & 0xff) as u8;
        }
        // Bias toward the shapes real code has: prefixes and short forms.
        if rng.next().is_multiple_of(4) {
            let prefixes = [0x48u8, 0x4c, 0x66, 0x67, 0xf0, 0xf2, 0xf3, 0x0f, 0x3e, 0x2e];
            buf[0] = prefixes[(rng.next() % prefixes.len() as u64) as usize];
        }
        let base = rng.next() % 0x1_0000;
        let out = decode(&buf, base, Syntax::Intel);
        windows += 1;
        let mut last = base.wrapping_sub(1);
        for i in &out {
            if i.valid() {
                assert!(
                    i.len >= 1 && i.len <= 15,
                    "instruction length {} out of range",
                    i.len
                );
                assert!(
                    i.va >= last,
                    "addresses went backwards: {:#x} -> {:#x}",
                    last,
                    i.va
                );
                last = i.va;
                insns += 1;
            } else {
                refusals += 1;
                // A refusal must be the last entry: the sweep stops, it does
                // not skip bytes and invent instructions after them.
                assert!(
                    std::ptr::eq(i, out.last().unwrap()),
                    "refusal in mid-stream"
                );
            }
        }
    }
    assert!(
        insns > 10_000,
        "only {insns} instructions decoded; the generator is producing nothing but refusals"
    );
    println!("fuzz decode: {windows} windows, {insns} instructions, {refusals} refusals, 0 panics");
}

#[test]
fn decode_encode_decode_is_stable_on_real_code() {
    // `push rbx; sub rsp, 0x20; mov eax, ecx; add rax, 1; ret` — a shape the
    // corpus is full of, with a REX prefix, a 32-bit immediate form, and a ret.
    let bytes: &[u8] = &[
        0x53, 0x48, 0x83, 0xEC, 0x20, 0x8B, 0xC1, 0x48, 0x83, 0xC0, 0x01, 0xC3,
    ];
    let n = round_trip(bytes, 0).expect("round trip");
    assert_eq!(n, 5, "expected 5 instructions, got {n}");
    let insns = decode(bytes, 0, Syntax::Intel);
    let addrs: Vec<u64> = insns.iter().map(|i| i.va).collect();
    assert_eq!(addrs, vec![0, 1, 5, 7, 11], "boundaries: {addrs:?}");
    let ret = insns.last().unwrap();
    assert!(ret.is_ret, "the trailing 0xC3 must be a return");
}

#[test]
fn an_undecodable_byte_is_reported_as_a_gap_not_skipped() {
    // 0xC6 / 0xC7 with no ModRM byte is not a complete instruction: the sweep
    // must refuse at that address and stop, never advance past it silently.
    let out = decode(&[0x90, 0xC6], 0x1000, Syntax::Intel);
    assert_eq!(out.len(), 2, "{out:?}");
    assert!(out[0].valid(), "the nop must decode");
    assert!(!out[1].valid(), "the truncated opcode must be refused");
    assert_eq!(out[1].va, 0x1001);
}
