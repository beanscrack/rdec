//! A phi's value arrives by the edge copy, so the phi op must not also emit a
//! statement.
//!
//! `emit_c` lowered a phi in two places: an assignment at the end of each
//! predecessor (the edge copy), and the phi's own op, whose C spelling was
//! `dest = (uint64_t)/* phi: assigned by edge copy */ 0`. That second one is a
//! real assignment, and it runs EVERY time control enters the block - including
//! on the back edge, after the latch has copied the new value in. The corpus hid
//! this for as long as no runnable function had a phi with non-zero incoming
//! values, so the reset looked like the value; T2, comparing the newly-runnable
//! loop functions against the machine, did not hide it: 41 of 1335 runs, with
//! `early_returns.O1::f(7,-1,40,3)` giving machine `0x2a` against C `0` - the
//! `mov eax,0x2a` edge copy into the ret block, wiped by that block's own phi
//! statement.
//!
//! So this file is the machine-code-free half of that proof: the IR's own
//! semantics against the C's, on the smallest functions where the two differ.

use rdec_core::ir::*;
use rdec_core::ty::Ty as T;
use rdec_core::verify::verify_ssa;
use std::process::Command;

/// The signature the emitter puts on a lifted function: four integer argument
/// registers, one unsigned answer.
fn sig(f: &mut Function) {
    f.ret_ty = T::u64();
    let e = f.entry;
    for _ in 0..4 {
        let v = f.new_value(T::u64());
        f.block_mut(e)
            .instrs
            .push(Instr::new(v, Op::Param, T::u64(), 0));
        f.params.push(v);
    }
}

/// e: if (1) goto h else goto d;  h: %p = phi [e]5, [d]0; return %p;  d: goto h
///
/// The taken edge is `e -> h`, so the answer is 5. A reset at `h` gives 0.
fn merge_phi() -> Function {
    let mut f = Function::new("mp", 0);
    sig(&mut f);
    let e = f.entry;
    let h = f.new_block(0x10);
    let d = f.new_block(0x20);
    let five = f.emit(e, Op::Const(5), T::u64(), 0);
    let zero = f.emit(e, Op::Const(0), T::u64(), 0);
    let t = f.emit(e, Op::Const(1), T::Bool, 0);
    f.block_mut(e).term = Terminator::Branch {
        cond: t,
        then_bb: h,
        else_bb: d,
    };
    let p = f.new_value(T::u64());
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            p,
            Op::Phi(vec![(five, e), (zero, d)].into_boxed_slice()),
            T::u64(),
            0x10,
        ),
    );
    f.block_mut(h).term = Terminator::Return(p);
    f.block_mut(d).term = Terminator::Jump(h);
    f
}

/// k = 0; s = 0
/// h: if (k >= 3) goto done else goto l;  l: s = s + 7; k = k + 1; goto h
/// done: return s                                     -> 21
///
/// The loop-carried form: a reset at `h` restarts every iteration from 0, so `k`
/// never reaches 3 and the C never returns at all.
fn loop_phi() -> Function {
    let mut f = Function::new("lp", 0);
    sig(&mut f);
    let e = f.entry;
    let h = f.new_block(0x10);
    let l = f.new_block(0x20);
    let done = f.new_block(0x30);
    let z = f.emit(e, Op::Const(0), T::u64(), 0);
    f.block_mut(e).term = Terminator::Jump(h);
    // The header's phi destinations exist before the latch is written: the
    // latch's values are the phis' loop-carried incoming edges.
    let kp = f.new_value(T::u64());
    let sp = f.new_value(T::u64());
    let seven = f.emit(l, Op::Const(7), T::u64(), 0x20);
    let s2 = f.emit(l, Op::Bin(BinOp::Add, sp, seven), T::u64(), 0x22);
    let one = f.emit(l, Op::Const(1), T::u64(), 0x24);
    let k2 = f.emit(l, Op::Bin(BinOp::Add, kp, one), T::u64(), 0x26);
    f.block_mut(l).term = Terminator::Jump(h);
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            kp,
            Op::Phi(vec![(z, e), (k2, l)].into_boxed_slice()),
            T::u64(),
            0x10,
        ),
    );
    f.block_mut(h).instrs.insert(
        1,
        Instr::new(
            sp,
            Op::Phi(vec![(z, e), (s2, l)].into_boxed_slice()),
            T::u64(),
            0x10,
        ),
    );
    let three = f.emit(h, Op::Const(3), T::u64(), 0x10);
    // `k < 3`: the test is in the header, so the value the header's phi carries
    // is what the exit reads - `s2`, defined in the latch, would not dominate
    // the exit block, and the verifier says so.
    let c = f.emit(h, Op::Cmp(IntCmp::Ult, kp, three), T::Bool, 0x12);
    f.block_mut(h).term = Terminator::Branch {
        cond: c,
        then_bb: l,
        else_bb: done,
    };
    f.block_mut(done).term = Terminator::Return(sp);
    f
}

#[test]
fn the_phi_op_emits_no_statement_of_its_own() {
    for f in [merge_phi(), loop_phi()] {
        verify_ssa(&f).expect("the IR is SSA");
        let out = rdec_core::emit_c::emit_c(&f).expect("it emits C");
        for b in &f.blocks {
            for ins in &b.instrs {
                if !matches!(ins.op, Op::Phi(_)) {
                    continue;
                }
                // Block-scoped: the phi's destination is written once per
                // INCOMING EDGE, at the end of the predecessor, and never inside
                // the block that owns the phi. The reset was exactly that
                // in-block write.
                let name = format!("x{} = ", ins.dest);
                let region = block_text(&out.text, b.id);
                let in_block = region
                    .lines()
                    .filter(|l| l.trim().starts_with(&name))
                    .count();
                assert_eq!(
                    in_block, 0,
                    "the phi's own block writes its destination (the reset):\n{region}"
                );
                let preds = f
                    .blocks
                    .iter()
                    .filter(|p| {
                        let mut s = Vec::new();
                        p.term.successors(&mut s);
                        s.contains(&b.id)
                    })
                    .count();
                let all = out
                    .text
                    .lines()
                    .filter(|l| l.trim().starts_with(&name))
                    .count();
                assert_eq!(
                    all, preds,
                    "one edge copy per predecessor ({preds}) for the phi in bb{}:\n{}",
                    b.id, out.text
                );
            }
        }
    }
}

/// The C text of one block: from its label to the next label.
fn block_text(text: &str, bb: u32) -> String {
    let label = format!("l{bb}: ");
    let Some(start) = text.find(&label) else {
        return String::new();
    };
    let rest = &text[start..];
    let end = rest[1..].find("\nl").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Run the emitted C and compare it with the interpreter's answer for the same
/// IR: the whole gate in miniature, with no machine code. If the C and the IR
/// disagree, the decompiler is wrong, and the test says so with both numbers.
#[test]
fn emitted_c_computes_what_the_ir_computes() {
    for (f, expect) in [(merge_phi(), 5u64), (loop_phi(), 21)] {
        let ir = rdec_core::interp::run(&f, &[]);
        let got = match ir.outcome {
            rdec_core::interp::Outcome::Ret(rdec_core::interp::Val::Int(v)) => v,
            other => panic!("the IR must return a value: {other:?}"),
        };
        assert_eq!(got, expect, "the IR's own answer");
        let out = rdec_core::emit_c::emit_c(&f).expect("it emits C");
        let text = format!(
            "{}\n{}\n#include <stdio.h>\nint main(void){{printf(\"%llu\\n\",(unsigned long long){}(1,2,3,4));return 0;}}\n",
            rdec_core::emit_c::PRELUDE,
            out.text,
            f.name
        );
        let dir = std::env::temp_dir().join("rdec_phi_edge_copy");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let c = dir.join(format!("{}.c", f.name));
        let exe = dir.join(format!("{}.exe", f.name));
        std::fs::write(&c, &text).expect("write C");
        // A stale exe from an earlier run is a silent wrong answer: the build
        // overwrites it, and if the build failed we must not run the old one.
        let _ = std::fs::remove_file(&exe);
        let build = Command::new("clang")
            .args([
                "-O0",
                &c.display().to_string(),
                "-o",
                &exe.display().to_string(),
            ])
            .output()
            .expect("clang runs");
        assert!(
            build.status.success(),
            "clang rejects the emitted C:\n{}\n{}",
            String::from_utf8_lossy(&build.stderr),
            text
        );
        let so = Command::new(&exe)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("run")
            .wait_with_output()
            .expect("output");
        let printed = String::from_utf8_lossy(&so.stdout).trim().to_string();
        assert_eq!(
            printed,
            expect.to_string(),
            "{}: the emitted C disagrees with the IR it came from (exit {:?})\n{}",
            f.name,
            so.status.code(),
            out.text
        );
    }
}
