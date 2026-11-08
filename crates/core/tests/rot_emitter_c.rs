//! The EMITTED C leg for rotates: `emit_c`'s rotate expression, compiled with
//! the floor gate's own flag set and run, against a reference assembled bit by
//! bit. Nothing else in this repo can reach this code path.
//!
//! Why this file exists (measured, 2026-11-08): `emit_c` spells
//! `BinOp::RotL`/`RotR`, and the spelling it carried was arithmetically wrong:
//!
//! ```text
//!   ((x << cnt) | (x >> ((w - cnt) & w)))
//! ```
//!
//! The complement shift is masked with `& w` — a POWER OF TWO, which is not a
//! mask at all. At W32 with `cnt = 3` that is `29 & 32 == 0`, so the emitted C
//! computed `(x << 3) | (x >> 0)`: the wrapped bits were dropped and the value
//! itself was OR'd in. With `cnt = 0` it is `32 & 32 == 32`, a shift by the
//! width, which is undefined behaviour in C — a rotate-by-zero that clang is
//! free to answer any way it likes.
//!
//! No corpus function reaches this: `rol`/`ror` are NOT in `lift.rs`, so the
//! corpus has zero rotate sites, and the defect was latent. It is fixed anyway
//! because the emitter is a promise about what the pipeline can print, and
//! because a broken expression that no test compiles is exactly how a latent
//! defect survives to the day a `rol` arm lands. THE HOLE IS NAMED, not papered
//! over: until a `rol`/`ror` lift site exists with a corpus witness, no gate
//! measures rotates end to end, and this test is a floor under the spelling,
//! not a claim of corpus coverage.
//!
//! SABOTAGE PROVEN (2026-11-08): restoring the old `& w` spelling reddens
//! `emitted_rotate_expressions_match_the_bit_loop_reference` on the first
//! W32 rotate-by-3 vector; removing the value/result mask reddens the W32
//! high-bit vectors. Quotes in the commit message.

use rdec_core::emit_c::{emit_c, PRELUDE};
use rdec_core::ir::{BinOp, Function, Op, Terminator};
use rdec_core::ty::{Sx, Ty, Width};

/// THE REFERENCE: assemble the answer bit by bit. Deliberately shares no
/// mechanism with the model (which rotates with shifts) or with the emitted C
/// (which also rotates with shifts) — a reference written as
/// `(x << n) | (x >> (w - n))` would agree with a model that shifts by `w`.
fn rotate(x: u64, w: Width, cnt: u64, left: bool) -> u64 {
    let b = w.bits() as usize;
    let m = if b == 64 { u64::MAX } else { (1u64 << b) - 1 };
    let v = x & m;
    let n = (cnt % b as u64) as usize;
    let mut bit = [false; 64];
    for (i, slot) in bit.iter_mut().take(b).enumerate() {
        *slot = (v >> i) & 1 == 1;
    }
    let mut r = 0u64;
    for i in 0..b {
        let src = if left { (i + b - n) % b } else { (i + n) % b };
        if bit[src] {
            r |= 1u64 << i;
        }
    }
    r & m
}

fn rot_fn(name: &str, op: BinOp, w: Width) -> Function {
    let mut f = Function::new(name, 0);
    let e = f.entry;
    let ty = Ty::Int {
        sx: Sx::Unsigned,
        w,
    };
    let a = f.emit(e, Op::Param, ty.clone(), 0);
    let c = f.emit(e, Op::Param, ty.clone(), 0);
    let r = f.emit(e, Op::Bin(op, a, c), ty.clone(), 0);
    f.block_mut(e).term = Terminator::Return(r);
    f.params.extend([a, c]);
    f.ret_ty = ty;
    f
}

fn vectors() -> Vec<(u64, u64)> {
    let xs: [u64; 10] = [
        0,
        1,
        2,
        3,
        0x8000_0000,
        0xffff_ffff,
        0xdead_beef,
        0x8000_0000_0000_0000,
        u64::MAX,
        0x0123_4567_89ab_cdef,
    ];
    let cs: [u64; 16] = [
        0,
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        31,
        32,
        33,
        63,
        64,
        65,
        255,
        u64::MAX,
    ];
    let mut v = Vec::new();
    for &x in &xs {
        for &c in &cs {
            v.push((x, c));
        }
    }
    // A deterministic sweep so a width-dependent bug cannot hide in the edges.
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..400 {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        v.push((s.wrapping_mul(0x9E37_79B9_7F4A_7C15), s >> 3));
    }
    v
}

#[test]
fn emitted_rotate_expressions_match_the_bit_loop_reference() {
    let vs = vectors();
    assert!(
        vs.len() > 500,
        "the sweep is the test: {} vectors",
        vs.len()
    );

    // One function per (op, width) the emitter can spell, each through the
    // REAL emitter — this test compiles `emit_c`'s output, not a copy of its
    // rule. A test that re-derived the expression here would test the test.
    let fns = [
        ("rot32l", BinOp::RotL, Width::W32),
        ("rot32r", BinOp::RotR, Width::W32),
        ("rot64l", BinOp::RotL, Width::W64),
        ("rot64r", BinOp::RotR, Width::W64),
    ];
    // The gate's own composition puts the includes first (PRELUDE is the
    // helper block, not a translation unit); this harness prints, so it needs
    // stdio declared — an implicit `printf` is an error under -Werror.
    let mut src = String::from("#include <stdint.h>\n#include <stdio.h>\n");
    src.push_str(PRELUDE);
    src.push('\n');
    for (name, op, w) in fns {
        let out = emit_c(&rot_fn(name, op, w)).unwrap_or_else(|r| {
            panic!(
                "{name}: the emitter refused its own rotate: {:?}",
                r.iter().map(|x| x.reason.clone()).collect::<Vec<_>>()
            )
        });
        assert!(
            out.externs.is_empty(),
            "{name}: a rotate should call nothing, got {:?}",
            out.externs
        );
        src.push_str(&out.text);
        src.push('\n');
    }
    // The witness that the swept vectors are the ones compiled in: a count of
    // the table rows, printed from the same source the run reads.
    src.push_str("static const uint64_t V[][2] = {\n");
    for (x, c) in &vs {
        src.push_str(&format!("  {{ {x}ULL, {c}ULL }},\n"));
    }
    src.push_str("};\nint main(void){for(unsigned i=0;i<sizeof V/sizeof V[0];i++){printf(\"%llx %llx %llx %llx\\n\",(unsigned long long)rot32l((uint32_t)V[i][0],(uint32_t)V[i][1]),(unsigned long long)rot32r((uint32_t)V[i][0],(uint32_t)V[i][1]),(unsigned long long)rot64l(V[i][0],V[i][1]),(unsigned long long)rot64r(V[i][0],V[i][1]));}return 0;}\n");

    let dir = std::env::temp_dir().join("rdec_rot_emitter_c");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let c = dir.join("rot.c");
    std::fs::write(&c, &src).expect("write C");
    let exe = dir.join("rot.exe");
    // The floor gate's flag set (`m4::compile`), spelled out so this cannot
    // drift from what CI compiles the emitted C with.
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
        "clang -Werror rejects the emitted rotate text:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = std::process::Command::new(&exe).output().expect("run");
    assert!(
        run.status.success(),
        "the rotate program exited {}: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let out = String::from_utf8_lossy(&run.stdout).into_owned();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), vs.len(), "one line per vector, no early exit");

    let mut first_bad: Option<String> = None;
    let mut checked = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let got: Vec<u64> = line
            .split_whitespace()
            .map(|h| u64::from_str_radix(h, 16).unwrap_or_else(|e| panic!("bad hex {h:?}: {e}")))
            .collect();
        assert_eq!(
            got.len(),
            4,
            "line {i}: four answers expected, got {line:?}"
        );
        let (x, cnt) = vs[i];
        let want = [
            rotate(x, Width::W32, cnt & u64::from(u32::MAX), true),
            rotate(x, Width::W32, cnt & u64::from(u32::MAX), false),
            rotate(x, Width::W64, cnt, true),
            rotate(x, Width::W64, cnt, false),
        ];
        for k in 0..4 {
            checked += 1;
            if got[k] != want[k] && first_bad.is_none() {
                first_bad = Some(format!(
                    "vector {i}: x={x:#x} cnt={cnt:#x} slot {k}: C said {:#x}, bit loop says {:#x}",
                    got[k], want[k]
                ));
            }
        }
    }
    assert_eq!(checked, vs.len() * 4, "every slot of every vector checked");
    assert!(
        first_bad.is_none(),
        "the emitted rotate C disagrees with the bit-loop reference: {}",
        first_bad.unwrap()
    );
}

/// The emitter's rotate must be spelled from the SAME rule at every width, and
/// the text is the claim: the count is masked to `w-1`, the complement shift is
/// `(w - (cnt & (w-1))) & (w-1)`, and the value and result are masked to `w`.
/// Pinned on the text so a future 'simplification' back to `& w` is caught with
/// a sentence that says what broke, not a mismatched number three gates away.
#[test]
fn rotate_text_masks_the_count_and_the_value() {
    for (op, w) in [
        (BinOp::RotL, Width::W8),
        (BinOp::RotR, Width::W8),
        (BinOp::RotL, Width::W16),
        (BinOp::RotR, Width::W32),
        (BinOp::RotL, Width::W64),
    ] {
        let bits = w.bits();
        let out = emit_c(&rot_fn("rot_probe", op, w)).expect("emit");
        let text = out.text;
        assert!(
            text.contains(&format!("& {}ULL", bits - 1)),
            "{op:?} {bits}-bit rotate must mask the count with {} (w-1), not with {bits}: {text}",
            bits - 1
        );
        assert!(
            !text.contains(&format!("& {bits}ULL")),
            "{op:?} {bits}-bit rotate masks with `& {bits}`, a power of two: that is not a mask \
             (at cnt=3 it evaluates to 0 and drops the wrapped bits): {text}"
        );
        let m = if bits == 64 {
            "~0ULL".to_string()
        } else {
            format!("0x{:x}ULL", (1u64 << bits) - 1)
        };
        assert!(
            text.contains(&m),
            "{op:?} {bits}-bit rotate must mask the value and the result with {m}: {text}"
        );
    }
}
