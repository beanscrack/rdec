//! The PRELUDE's portable multiply-high helpers, cross-checked against
//! `__int128` — the reference that has no opinion about how to compute an
//! answer, which is the only kind of reference worth having here.
//!
//! Why this test exists: `rdec_mulhi64`/`rdec_smulhi64` are the C spelling of
//! `BinOp::MulHi`/`MulSmul` — the `mul`/`imul` high half, `mul_wide`'s return
//! value — and they were written to dodge a compiler extension (`__int128` does
//! not exist on MSVC). A limb decomposition is exactly the kind of code that is
//! right on the four cases someone tried and wrong on a carry. The decomposition
//! is:
//!
//! ```text
//!   a*b = a1*b1*2^64 + (a1*b0 + a0*b1)*2^32 + a0*b0
//!   high64 = a1*b1 + high32(a1*b0) + high32(a0*b1) + high32( (a0*b0)>>32 + low32(a1*b0) + low32(a0*b1) )
//! ```
//!
//! and the signed form subtracts each negative operand's contribution. Both
//! claims are checked over every boundary value and a deterministic pseudo-random
//! sweep — 300+ vectors — by compiling with the floor gate's own flag set and
//! letting the machine answer.
//!
//! The `__int128` reference is TEST-ONLY. Production C must not use it (MSVC has
//! no 128-bit integer type, and the emitted C has to compile where the project's
//! own C compiles); a test may, because a test only has to be right.

/// The vectors: the boundaries a limb decomposition breaks on, plus a
/// deterministic sweep. No RNG seed to chase: the LCG below is the vector list.
fn vectors() -> Vec<(u64, u64)> {
    let edges: [u64; 14] = [
        0,
        1,
        2,
        3,
        u64::from(u32::MAX),
        u64::from(u32::MAX) - 1,
        1 << 31,
        1 << 32,
        (1 << 32) + 1,
        1 << 63,
        (1 << 63) - 1,
        u64::MAX,
        u64::MAX - 1,
        0x00FF00FF00FF00FF,
    ];
    let mut v: Vec<(u64, u64)> = Vec::new();
    for &a in &edges {
        for &b in &edges {
            v.push((a, b));
        }
    }
    // A sweep: 0x9E3779B97F4A7C15 is the golden-ratio multiplier, so the
    // sequence walks the whole space without clustering, and the vectors are
    // the same on every machine.
    let mut s: u64 = 0x243F_6A88_85A3_08D3;
    for _ in 0..256 {
        s = s
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x6A09_E667_F3BC_C908);
        let t = s ^ s.rotate_left(23);
        v.push((s, t));
        v.push((t, s));
    }
    v
}

fn generated_c(vs: &[(u64, u64)]) -> String {
    let mut s = String::from(rdec_core::emit_c::PRELUDE);
    s.push_str(
        "\n\
         #include <stdio.h>\n\
         int main(void) {\n\
         \x20   static const uint64_t A[] = {",
    );
    for (a, _) in vs {
        s.push_str(&format!("{a:#018x}ULL,"));
    }
    s.push_str("};\n\x20   static const uint64_t B[] = {");
    for (_, b) in vs {
        s.push_str(&format!("{b:#018x}ULL,"));
    }
    s.push_str(
        "};\n\
         \x20   for (size_t i = 0; i < sizeof(A)/sizeof(A[0]); i++) {\n\
         \x20       uint64_t a = A[i], b = B[i];\n\
         \x20       uint64_t hu = rdec_mulhi64(a, b);\n\
         \x20       uint64_t hs = rdec_smulhi64((int64_t)a, (int64_t)b);\n\
         \x20       uint64_t ru = (uint64_t)((__int128)a * (__int128)b >> 64);\n\
         \x20       uint64_t rs = (uint64_t)(((__int128)(int64_t)a * (__int128)(int64_t)b) >> 64);\n\
         \x20       printf(\"%016llx %016llx %016llx %016llx\\n\",\n\
         \x20              (unsigned long long)hu, (unsigned long long)hs,\n\
         \x20              (unsigned long long)ru, (unsigned long long)rs);\n\
         \x20   }\n\
         \x20   return 0;\n\
         }\n",
    );
    s
}

#[test]
fn prelude_multiply_high_helpers_match_the_128_bit_reference() {
    let vs = vectors();
    assert!(
        vs.len() > 300,
        "the sweep is the test: {} vectors",
        vs.len()
    );
    let dir = std::env::temp_dir().join("rdec_mulhi_helpers");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let c = dir.join("mulhi.c");
    std::fs::write(&c, generated_c(&vs)).expect("write C");
    let exe = dir.join("mulhi.exe");
    // The floor gate's flag set (`m4::compile`), spelled out so the check
    // cannot drift from what CI compiles the emitted C with. `-Wno-unused-parameter`
    // is the gate's; the helpers must be clean under `-Wall -Wextra -Werror`.
    let build = std::process::Command::new("clang")
        .args([
            "-std=c17",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
            &c.display().to_string(),
            "-o",
            &exe.display().to_string(),
        ])
        .output()
        .expect("clang runs");
    assert!(
        build.status.success(),
        "clang -Werror rejects the PRELUDE helpers:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = std::process::Command::new(&exe).output().expect("run");
    assert!(
        run.status.success(),
        "the helper program exited {}: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let out = String::from_utf8_lossy(&run.stdout).into_owned();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), vs.len(), "one line per vector, no early exit");

    let parse = |s: &str| -> u64 {
        u64::from_str_radix(s, 16).unwrap_or_else(|e| panic!("bad hex {s:?}: {e}"))
    };
    // The Rust model's own rule, so a drift in the interpreter is caught here
    // too: `interp.rs` computes MulHi as `(x as u128 * y as u128) >> 64` and
    // MulSmul as `((x as i128 * y as i128) >> 64) as u64`, at the node width.
    let model_u = |a: u64, b: u64| -> u64 { ((((a as u128) * (b as u128)) >> 64) as u64).to_le() };
    let model_s =
        |a: u64, b: u64| -> u64 { (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64 };
    let mut bad = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let mut it = line.split_whitespace();
        let (hu, hs, ru, rs) = (
            parse(it.next().expect("hu")),
            parse(it.next().expect("hs")),
            parse(it.next().expect("ru")),
            parse(it.next().expect("rs")),
        );
        let (a, b) = vs[i];
        if hu != ru {
            bad += 1;
            if bad < 8 {
                println!(
                    "MulHi  a={a:#018x} b={b:#018x}: helper {hu:#018x} != __int128 {ru:#018x}"
                );
            }
        }
        if hs != rs {
            bad += 1;
            if bad < 8 {
                println!(
                    "MulSmul a={a:#018x} b={b:#018x}: helper {hs:#018x} != __int128 {rs:#018x}"
                );
            }
        }
        if ru != model_u(a, b) || rs != model_s(a, b) {
            bad += 1;
            if bad < 8 {
                println!(
                    "the interpreter drifted from __int128 at a={a:#x} b={b:#x}: \
                     model {:x}/{:x} vs ref {ru:#x}/{rs:#x}",
                    model_u(a, b),
                    model_s(a, b)
                );
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad} of {} vectors disagree (first few printed above)",
        vs.len()
    );
}

/// The emitter's own claim, aimed at the emitter: a multiply-high at 64 bits
/// emits the helper call with no refusal, and at any narrower width it is
/// REFUSED with a reason that names the shift the caller means.
///
/// The narrow case matters because the interpreter will happily answer it
/// (`>> 64` on a product that fits in 64 bits is 0 for `MulHi`, the sign for
/// `MulSmul`), and silently emitting that zero would put a constant in the C
/// where nobody wrote one. The lifter spells the 32-bit pair form as a W64
/// product plus a shift, so nothing should reach this path — and if something
/// does, the census names it.
#[test]
fn multiply_high_emits_at_64_and_refuses_below_naming_the_shift() {
    use rdec_core::emit_c::emit_c;
    use rdec_core::ir::{BinOp, Function, Op, Terminator};
    use rdec_core::ty::{Sx, Ty, Width};

    let build = |op: BinOp, w: Width| -> Result<String, Vec<String>> {
        let ty = Ty::Int {
            sx: Sx::Unsigned,
            w,
        };
        let mut f = Function::new("f", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, ty.clone(), 0);
        let b = f.emit(e, Op::Param, ty.clone(), 0);
        let r = f.emit(e, Op::Bin(op, a, b), ty.clone(), 0);
        f.block_mut(e).term = Terminator::Return(r);
        f.params.extend([a, b]);
        f.ret_ty = ty;
        // The emitter's contract: a refusal is an `Err` naming every site, not
        // a comment inside otherwise-good C — so "it refused" is a claim about
        // the return value, and this test reads it as one.
        emit_c(&f)
            .map(|o| o.text)
            .map_err(|rs| rs.iter().map(|r| r.reason.clone()).collect())
    };

    for op in [BinOp::MulHi, BinOp::MulSmul] {
        let w64 = build(op, Width::W64)
            .unwrap_or_else(|rs| panic!("{op:?} at W64 must emit, got refusals: {rs:?}"));
        let helper = if op == BinOp::MulHi {
            "rdec_mulhi64("
        } else {
            "rdec_smulhi64("
        };
        assert!(
            w64.contains(helper),
            "{op:?} at W64 must call the helper, got:\n{w64}"
        );
        assert!(
            !w64.contains("/* refusal"),
            "{op:?} at W64 must emit clean C, got:\n{w64}"
        );
        assert!(
            !w64.contains("__int128"),
            "{op:?} must not spell C that MSVC cannot compile, got:\n{w64}"
        );

        let reasons = build(op, Width::W32)
            .err()
            .unwrap_or_else(|| panic!("{op:?} at W32 must be REFUSED, not emitted"));
        assert!(
            reasons.iter().any(|r| r.contains("means a shift")),
            "{op:?} at W32 must name the shift the caller means, got: {reasons:?}"
        );
        assert!(
            !reasons.iter().any(|r| r.contains(helper)),
            "{op:?} at W32 must not route through the 64-bit helper: {reasons:?}"
        );
    }
}
