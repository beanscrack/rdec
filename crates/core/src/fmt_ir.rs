//! Human-readable IR dumping. This is the decompiler's own debugger: every
//! stage of the pipeline prints through here, so pass bugs are visible as
//! diffs between two dumps.

use std::fmt::Write as _;

use crate::ir::*;

pub fn format_function(f: &Function) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "; {} @ {:#x}  ret={} params={}",
        f.name,
        f.addr,
        f.ret_ty,
        f.params.len()
    );
    let cfg = crate::cfg::Cfg::build(f);
    for &bb in &cfg.rpo {
        let b = f.block(bb);
        let _ = writeln!(s, "bb{}(at {:#x}):", bb, b.addr);
        for i in &b.instrs {
            let _ = writeln!(s, "    {}", format_instr(f, i));
        }
        let _ = writeln!(s, "    {}", format_term(f, &b.term));
    }
    for bb in cfg.unreachable() {
        let b = f.block(bb);
        let _ = writeln!(s, "bb{}(at {:#x}) [UNREACHABLE]:", bb, b.addr);
        for i in &b.instrs {
            let _ = writeln!(s, "    {}", format_instr(f, i));
        }
        let _ = writeln!(s, "    {}", format_term(f, &b.term));
    }
    s
}

pub fn vname(f: &Function, v: VId) -> String {
    match f.values.get(v as usize).and_then(|i| i.name.clone()) {
        Some(n) => n,
        None => format!("%{v}"),
    }
}

pub fn format_instr(f: &Function, i: &Instr) -> String {
    let dest = if i.dest == NO_VAL || i.ty.is_void() {
        String::new()
    } else {
        format!("{} = ", vname(f, i.dest))
    };
    let body = match &i.op {
        Op::Const(c) => format!("{c:#x}"),
        Op::ConstF(c) => format!("{c}"),
        Op::Undef => "undef".into(),
        Op::Param => "param".into(),
        Op::Copy(a) => format!("copy {}", vname(f, *a)),
        Op::Bin(op, a, b) => format!("{} {}, {}", bin_str(*op), vname(f, *a), vname(f, *b)),
        Op::Un(op, a) => format!("{} {}", un_str(*op), vname(f, *a)),
        Op::Cmp(c, a, b) => format!("cmp{} {}, {}", cmp_str(*c), vname(f, *a), vname(f, *b)),
        Op::Flags(fl) => format_flags(f, *fl),
        Op::GetFlag(g, fv) => format!("getflag {g}, {}", vname(f, *fv)),
        Op::Cast(k, t, a) => format!("{} {} {}[{}]", cast_str(*k), vname(f, *a), t, i.addr),
        Op::Load(t, a) => format!("load {}[{t}] {}", t, vname(f, *a)),
        Op::Store(t, a, v) => format!("store {}[{t}] = {}", vname(f, *a), vname(f, *v)),
        Op::Select(c, t, e) => format!(
            "select {}, {}, {}",
            vname(f, *c),
            vname(f, *t),
            vname(f, *e)
        ),
        Op::Phi(pairs) => {
            let mut s = "phi ".to_string();
            for (k, (v, b)) in pairs.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                }
                let _ = write!(s, "[bb{b}]{}", vname(f, *v));
            }
            s
        }
        Op::Call {
            callee,
            args,
            ret_ty,
            pure,
        } => {
            let t = match callee {
                Callee::Symbol(s) => s.clone(),
                Callee::Addr(a) => format!("sub_{a:x}"),
                Callee::Indirect(v) => format!("<{}>", vname(f, *v)),
            };
            let a: Vec<String> = args.iter().map(|v| vname(f, *v)).collect();
            format!("call {t}({}){} : {ret_ty}", a.join(", "), if *pure { " [pure]" } else { "" })
        }
        Op::Opaque(msg) => format!("opaque \"{msg}\""),
    };
    let ty = if i.ty.is_void() {
        String::new()
    } else {
        format!("   : {}", i.ty)
    };
    format!("{dest}{body}{ty}   ; {:#x}", i.addr)
}

pub fn format_term(f: &Function, t: &Terminator) -> String {
    match t {
        Terminator::Jump(b) => format!("br bb{b}"),
        Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => format!("br {}, bb{then_bb}, bb{else_bb}", vname(f, *cond)),
        Terminator::Return(v) if *v == NO_VAL => "ret".into(),
        Terminator::Return(v) => format!("ret {}", vname(f, *v)),
        Terminator::Unreachable => "unreachable".into(),
        Terminator::Exit => "exit".into(),
    }
}

fn format_flags(f: &Function, fl: FlagsOp) -> String {
    let (n, a, b) = match fl {
        FlagsOp::Add { a, b, carry } => (if carry { "flags.adc" } else { "flags.add" }, a, Some(b)),
        FlagsOp::Sub { a, b, borrow } => (if borrow { "flags.sbb" } else { "flags.sub" }, a, Some(b)),
        FlagsOp::And { a, b } => ("flags.and", a, Some(b)),
        FlagsOp::Or { a, b } => ("flags.or", a, Some(b)),
        FlagsOp::Xor { a, b } => ("flags.xor", a, Some(b)),
        FlagsOp::Shl { a, b } => ("flags.shl", a, Some(b)),
        FlagsOp::Shr { a, b } => ("flags.shr", a, Some(b)),
        FlagsOp::Sar { a, b } => ("flags.sar", a, Some(b)),
        FlagsOp::Neg { a } => ("flags.neg", a, None),
        FlagsOp::Inc { a } => ("flags.inc", a, None),
        FlagsOp::Dec { a } => ("flags.dec", a, None),
        FlagsOp::Mul { a, b, signed } => (if signed { "flags.imul" } else { "flags.mul" }, a, Some(b)),
    };
    match b {
        Some(b) => format!("{n} {}, {}", vname(f, a), vname(f, b)),
        None => format!("{n} {}", vname(f, a)),
    }
}

pub fn bin_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::MulHi => "mulhi",
        BinOp::MulSmul => "mulsmul",
        BinOp::UDiv => "udiv",
        BinOp::URem => "urem",
        BinOp::SDiv => "sdiv",
        BinOp::SRem => "srem",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Xor => "xor",
        BinOp::Shl => "shl",
        BinOp::Shr => "shr",
        BinOp::Sar => "sar",
        BinOp::RotL => "rotl",
        BinOp::RotR => "rotr",
    }
}

pub fn un_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "not",
        UnOp::Neg => "neg",
        UnOp::Popcount => "popcnt",
        UnOp::Lzcnt => "lzcnt",
        UnOp::Tzcnt => "tzcnt",
        UnOp::Sqrt => "sqrt",
    }
}

pub fn cmp_str(c: IntCmp) -> &'static str {
    match c {
        IntCmp::Eq => "e",
        IntCmp::Ne => "ne",
        IntCmp::Ult => "ult",
        IntCmp::Ule => "ule",
        IntCmp::Ugt => "ugt",
        IntCmp::Uge => "uge",
        IntCmp::Lt => "lt",
        IntCmp::Le => "le",
        IntCmp::Gt => "gt",
        IntCmp::Ge => "ge",
    }
}

pub fn cast_str(k: CastKind) -> &'static str {
    match k {
        CastKind::Trunc => "trunc",
        CastKind::ZExt => "zext",
        CastKind::SExt => "sext",
        CastKind::Bitcast => "bitcast",
        CastKind::SIToFp => "si2fp",
        CastKind::UIToFp => "ui2fp",
        CastKind::FpToSI => "fp2si",
        CastKind::FpToUI => "fp2ui",
        CastKind::FPExt => "fpext",
        CastKind::FPTrunc => "fptrunc",
    }
}
