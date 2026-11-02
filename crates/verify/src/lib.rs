//! T1 oracle: check the x86 flag model against real EFLAGS.
//!
//! `cargo test` runs the boundary grid (fast, ~10^6 probes). `rdec flags --full`
//! adds N random operand pairs per op/width. The sabotage mode is the part that
//! makes this a test rather than a ritual: it injects a known-wrong flag into
//! one op at one width and asserts the harness reports *exactly* that op/width,
//! which proves the harness can fail and that its reporting is granular.

pub mod corpus;
pub mod hw;

/// Re-export under a stable name so the CLI does not hard-code the crate path.
pub use rdec_arch_x86::flags as model_shim;

use rdec_arch_x86::flags as model;
use rdec_core::ir::Flag;
use rdec_core::ty::Width;

#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    pub op: &'static str,
    pub w: Width,
    pub a: u64,
    pub b: u64,
    pub flag: Option<Flag>,
    /// `None` flag = the result value itself differed.
    pub model: u64,
    pub hw: u64,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.flag {
            Some(fl) => write!(
                f,
                "{}.{} a={:#x} b={:#x}: model {fl}={} hw {fl}={}",
                self.op,
                self.w.bits(),
                self.a,
                self.b,
                self.model != 0,
                self.hw != 0
            ),
            None => write!(
                f,
                "{}.{} a={:#x} b={:#x}: result model={:#x} hw={:#x}",
                self.op, self.w.bits(), self.a, self.b, self.model, self.hw
            ),
        }
    }
}

#[derive(Default)]
pub struct Report {
    pub probes: u64,
    pub flag_checks: u64,
    /// Model said "undefined", so hardware was not compared.
    pub undef_skips: u64,
    pub mismatches: Vec<Mismatch>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty()
    }
    pub fn summary(&self) -> String {
        format!(
            "{} probes, {} flag comparisons, {} undefined-skips, {} mismatches",
            self.probes,
            self.flag_checks,
            self.undef_skips,
            self.mismatches.len()
        )
    }
}

/// Sabotage coordinates: which op/width gets a deliberately wrong model bit.
#[derive(Copy, Clone, Debug)]
pub struct Sabotage {
    pub op: &'static str,
    pub w: Width,
    pub flag: Flag,
}

/// Operand grid: every power of two, every 2^k-1, the sign bits, all-ones, and
/// neighbours — the values that actually break flag formulas.
pub fn grid(w: Width) -> Vec<u64> {
    let bits = w.bits() as u64;
    let mask = w.mask();
    let mut v: Vec<u64> = Vec::new();
    for k in 0..bits {
        v.push((1u64 << k) & mask);
        v.push(((1u64 << k) - 1) & mask);
        if k + 1 < bits {
            v.push(((1u64 << k) + 1) & mask);
            v.push(((1u64 << (k + 1)) - 2) & mask);
        }
    }
    v.push(0);
    v.push(1);
    v.push(mask);
    v.push(mask - 1);
    v.push((mask >> 1) + 1); // INT_MIN
    v.push(mask >> 1); // INT_MAX
    v.sort();
    v.dedup();
    v
}

/// xorshift64* PRNG: deterministic, no deps, good enough for random probes.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

/// Run the full comparison.
pub fn check(sabotage: Option<Sabotage>, extra_random: u64) -> Report {
    let mut r = Report::default();
    for &w in [Width::W8, Width::W16, Width::W32, Width::W64].iter() {
        let mut vals = grid(w);
        if extra_random > 0 {
            let mut rng = Rng(0x9E3779B97F4A7C15 ^ w.bits() as u64);
            for _ in 0..extra_random {
                vals.push(rng.next_u64() & w.mask());
            }
        }
        // Binary ops over the full cross product.
        for &a in &vals {
            for &b in &vals {
                check_bin(&mut r, w, a, b, sabotage);
            }
        }
        // Unary + shift ops.
        for &a in &vals {
            check_unary(&mut r, w, a, sabotage);
            // Counts past 32 matter for 64-bit operands (6-bit count mask).
            for c in 0..=70u64 {
                check_shift(&mut r, w, a, c, sabotage);
            }
        }
    }
    r
}

fn sabotage_bit(s: Option<Sabotage>, op: &str, w: Width, flag: Flag, d: &mut model::FlagDef) {
    if let Some(s) = s {
        if s.op == op && s.w == w && s.flag == flag {
            d.set(flag, d.get(flag).map(|v| !v));
        }
    }
}

fn cmp_flags(
    r: &mut Report,
    op: &'static str,
    w: Width,
    a: u64,
    b: u64,
    seed: u64,
    m: &model::FlagDef,
    hw_flags: u64,
    sabotage: Option<Sabotage>,
) {
    for &fl in model::FlagDef::ALL.iter() {
        // "Unchanged" flags must equal the seeded value, and the model must not
        // claim to define them.
        if m.unchanged.contains(&fl) {
            r.flag_checks += 1;
            let mask = flag_mask(fl);
            let want = hw::bit(seed, mask);
            let got = hw::bit(hw_flags, mask);
            if want != got {
                r.mismatches.push(Mismatch {
                    op,
                    w,
                    a,
                    b,
                    flag: Some(fl),
                    model: want as u64,
                    hw: got as u64,
                });
            }
            continue;
        }
        match m.get(fl) {
            Some(model_val) => {
                r.flag_checks += 1;
                // Apply the (optional) sabotage to a copy of the definition.
                let mut def = *m;
                sabotage_bit(sabotage, op, w, fl, &mut def);
                let mv = def.get(fl).unwrap_or(model_val);
                let hw_val = hw::bit(hw_flags, flag_mask(fl));
                if mv != hw_val {
                    r.mismatches.push(Mismatch {
                        op,
                        w,
                        a,
                        b,
                        flag: Some(fl),
                        model: mv as u64,
                        hw: hw_val as u64,
                    });
                }
            }
            None => r.undef_skips += 1,
        }
    }
}

pub fn flag_mask(fl: Flag) -> u64 {
    match fl {
        Flag::CF => hw::CF,
        Flag::PF => hw::PF,
        Flag::AF => hw::AF,
        Flag::ZF => hw::ZF,
        Flag::SF => hw::SF,
        Flag::OF => hw::OF,
    }
}

fn check_bin(
    r: &mut Report,
    w: Width,
    a: u64,
    b: u64,
    s: Option<Sabotage>,
) {
    let seed = hw::ALL_SEED;
    r.probes += 6;

    let (mr, md) = model::add(w, a, b);
    let (hr, hf) = hw::add(w, a, b, seed);
    cmp_result(r, "add", w, a, b, mr, hr);
    cmp_flags(r, "add", w, a, b, seed, &md, hf, s);

    let (mr, md) = model::sub(w, a, b);
    let (hr, hf) = hw::sub(w, a, b, seed);
    cmp_result(r, "sub", w, a, b, mr, hr);
    cmp_flags(r, "sub", w, a, b, seed, &md, hf, s);

    for (name, op) in [("and", model::LogicOp::And), ("or", model::LogicOp::Or), ("xor", model::LogicOp::Xor)] {
        let (mr, md) = model::logic(w, a, b, op);
        let (hr, hf) = match op {
            model::LogicOp::And => hw::and(w, a, b, seed),
            model::LogicOp::Or => hw::or(w, a, b, seed),
            model::LogicOp::Xor => hw::xor(w, a, b, seed),
        };
        cmp_result(r, name, w, a, b, mr, hr);
        cmp_flags(r, name, w, a, b, seed, &md, hf, s);
    }

    let (mr, md) = model::adc(w, a, b, true);
    let (hr, hf) = hw::adc(w, a, b, true, seed);
    cmp_result(r, "adc", w, a, b, mr, hr);
    cmp_flags(r, "adc", w, a, b, seed, &md, hf, s);

    let (mr, md) = model::sbb(w, a, b, true);
    let (hr, hf) = hw::sbb(w, a, b, true, seed);
    cmp_result(r, "sbb", w, a, b, mr, hr);
    cmp_flags(r, "sbb", w, a, b, seed, &md, hf, s);

    let (mlo, mhi, md) = model::mul(w, a, b);
    let (hlo, hhi, hf) = hw::mul(w, a, b, seed);
    cmp_result(r, "mul", w, a, b, mlo, hlo);
    cmp_result(r, "mul", w, a, b, mhi, hhi);
    cmp_flags(r, "mul", w, a, b, seed, &md, hf, s);

    let (mlo, mhi, md) = model::imul(w, a, b);
    let (hlo, hhi, hf) = hw::imul(w, a, b, seed);
    cmp_result(r, "imul", w, a, b, mlo, hlo);
    cmp_result(r, "imul", w, a, b, mhi, hhi);
    cmp_flags(r, "imul", w, a, b, seed, &md, hf, s);
}

fn check_unary(r: &mut Report, w: Width, a: u64, s: Option<Sabotage>) {
    let seed = hw::ALL_SEED;
    r.probes += 3;
    for (name, mfn, hfn) in [
        ("inc", model::inc as fn(Width, u64) -> (u64, model::FlagDef), hw::inc as fn(Width, u64, u64) -> (u64, u64)),
        ("dec", model::dec, hw::dec),
        ("neg", model::neg, hw::neg),
    ] {
        let (mr, md) = mfn(w, a);
        let (hr, hf) = hfn(w, a, seed);
        cmp_result(r, name, w, a, 0, mr, hr);
        cmp_flags(r, name, w, a, 0, seed, &md, hf, s);
    }
}

fn check_shift(r: &mut Report, w: Width, a: u64, count: u64, s: Option<Sabotage>) {
    let seed = hw::ALL_SEED;
    r.probes += 3;
    let cases: [(
        &str,
        fn(Width, u64, u64) -> (u64, model::FlagDef),
        fn(Width, u64, u64, u64) -> (u64, u64),
    ); 3] = [
        ("shl", model::shl, hw::shl),
        ("shr", model::shr, hw::shr),
        ("sar", model::sar, hw::sar),
    ];
    for (name, mfn, hfn) in cases {
        let (mr, md) = mfn(w, a, count);
        let (hr, hf) = hfn(w, a, count, seed);
        cmp_result(r, name, w, a, count, mr, hr);
        cmp_flags(r, name, w, a, count, seed, &md, hf, s);
    }
}

fn cmp_result(r: &mut Report, op: &'static str, w: Width, a: u64, b: u64, m: u64, h: u64) {
    if w.trunc(m, false) != w.trunc(h, false) {
        r.mismatches.push(Mismatch {
            op,
            w,
            a,
            b,
            flag: None,
            model: m,
            hw: h,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model is believed only because this passes.
    #[test]
    fn flag_model_matches_hardware() {
        let r = check(None, 0);
        assert!(
            r.ok(),
            "flag model disagrees with hardware:\n{}",
            r.mismatches
                .iter()
                .take(40)
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(r.probes > 100_000, "grid too small: {}", r.summary());
    }

    /// The oracle can fail: a single injected bad bit must be caught, and only
    /// on the op/width it was injected into.
    #[test]
    fn harness_detects_sabotage() {
        let sab = Sabotage {
            op: "add",
            w: Width::W32,
            flag: Flag::OF,
        };
        let r = check(
            Some(sab),
            0,
        );
        assert!(!r.ok(), "sabotaged model passed the oracle");
        let bad: Vec<&Mismatch> = r
            .mismatches
            .iter()
            .filter(|m| m.flag == Some(Flag::OF))
            .collect();
        assert!(!bad.is_empty());
        assert!(
            bad.iter().all(|m| m.op == "add" && m.w == Width::W32),
            "sabotage leaked outside the injected op/width: {:?}",
            &bad[..bad.len().min(5)]
        );
    }

    #[test]
    fn grid_covers_boundaries() {
        let g = grid(Width::W8);
        for v in [0u64, 1, 0x7f, 0x80, 0xff] {
            assert!(g.contains(&v), "missing {v:#x} in u8 grid");
        }
        let g = grid(Width::W64);
        for v in [1u64 << 63, (1u64 << 63) - 1, u64::MAX] {
            assert!(g.contains(&v), "missing {v:#x} in u64 grid");
        }
    }
}
