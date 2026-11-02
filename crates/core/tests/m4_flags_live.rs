//! A `jge` in the shape the lifter actually produces: all five flag bits read, the
//! branch using `not(SF ^ OF)`. Flag elimination must fold the branch's condition
//! AND leave no `Flags` value behind, because that value is what the C emitter
//! refuses (37 functions' largest single class).
use rdec_core::fuse::flag_elimination;
use rdec_core::ir::{Flag, FlagsOp, Function, Op, Terminator};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

fn jge_fn() -> Function {
    let mut f = Function::new("jge", 0);
    let e = f.entry;
    let a = f.emit(e, Op::Param, Ty::i64(), 0);
    let b = f.emit(e, Op::Param, Ty::i64(), 0);
    let fv = f.emit(
        e,
        Op::Flags(FlagsOp::Sub {
            a,
            b,
            borrow: false,
        }),
        Ty::Flags,
        0,
    );
    // The lifter reads all five bits for a jcc; the branch uses two of them.
    let _zf = f.emit(e, Op::GetFlag(Flag::ZF, fv), Ty::Bool, 0);
    let sf = f.emit(e, Op::GetFlag(Flag::SF, fv), Ty::Bool, 0);
    let of = f.emit(e, Op::GetFlag(Flag::OF, fv), Ty::Bool, 0);
    let _cf = f.emit(e, Op::GetFlag(Flag::CF, fv), Ty::Bool, 0);
    let _pf = f.emit(e, Op::GetFlag(Flag::PF, fv), Ty::Bool, 0);
    let ne = f.emit(e, Op::Bin(rdec_core::ir::BinOp::Xor, sf, of), Ty::Bool, 0);
    let ge = f.emit(e, Op::Un(rdec_core::ir::UnOp::Not, ne), Ty::Bool, 0);
    let t = f.new_block(0x10);
    let el = f.new_block(0x20);
    f.block_mut(e).term = Terminator::Branch {
        cond: ge,
        then_bb: t,
        else_bb: el,
    };
    f.block_mut(t).term = Terminator::Return(a);
    f.block_mut(el).term = Terminator::Return(b);
    f.params.extend([a, b]);
    f
}

fn flag_nodes(f: &Function) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i.op, Op::Flags(_) | Op::GetFlag(..)))
        .count()
}

#[test]
fn a_jge_with_all_five_bits_read_leaves_no_flags_value() {
    let mut f = jge_fn();
    verify_ssa(&f).expect("fixture is valid IR");
    assert_eq!(flag_nodes(&f), 6, "fixture: one flags value, five reads");
    for _ in 0..16 {
        let mut ch = flag_elimination(&mut f);
        ch |= rdec_core::passes::copy_prop(&mut f);
        ch |= rdec_core::passes::const_fold(&mut f);
        ch |= rdec_core::passes::dead_code_elimination(&mut f);
        if !ch {
            break;
        }
    }
    let left = flag_nodes(&f);
    let dump = rdec_core::fmt_ir::format_function(&f);
    assert_eq!(
        left, 0,
        "flags value survived, the emitter will refuse:\n{dump}"
    );
    verify_ssa(&f).expect("the folded function is valid IR");
    rdec_core::emit_c::emit_c(&f)
        .unwrap_or_else(|r| panic!("still refused: {:?}\n{dump}", r.first()));
}

#[test]
fn folding_a_jge_preserves_meaning() {
    let mut f = jge_fn();
    let ss = rdec_core::tv::states(2, 40, 11);
    match rdec_core::tv::run_validated(&mut f, ("flag_elimination", flag_elimination), &ss, 1000) {
        Ok(_) => {}
        Err(fails) => panic!(
            "flag_elimination changed meaning: {:?}",
            fails.first().map(|x| (&x.pass, &x.inputs, &x.diffs))
        ),
    }
}
