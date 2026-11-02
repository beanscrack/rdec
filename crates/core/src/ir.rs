//! The RDec IR: typed three-address code, in SSA form after construction.
//!
//! Design notes that matter for the rest of the system:
//!
//! * Every def is a [`VId`] in a per-function arena, and every use is a
//!   `VId`. There is no aliasing of destinations, so passes are free to
//!   rename/replace a value with one `Vec` swap.
//! * Flag-defining x86 ops are modelled as a separate SSA value of type
//!   [`Ty::Flags`] ([`Op::Flags`]), read through [`Op::GetFlag`]. This makes
//!   "flags are dead" fall out of plain DCE, and makes jcc fusion a rewrite
//!   over `GetFlag` trees instead of a dataflow analysis.
//! * Memory is a flat 64-bit address space: [`Op::Load`] / [`Op::Store`] carry
//!   an address value and an access type. No `getelementptr`-style typed path
//!   until we do struct recovery.

use std::fmt;

use crate::ty::{Sx, Ty, Width};

pub type VId = u32;
pub type BbId = u32;

pub const NO_VAL: VId = u32::MAX;
pub const NO_BB: BbId = u32::MAX;

/// Integer binary ops. Wrapping semantics at the operand type's width; the
/// IR has no implicit truncation, every narrowing is an explicit [`Op::Cast`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Unsigned multiply high word.
    MulHi,
    /// Signed multiply high word.
    MulSmul,
    UDiv,
    URem,
    SDiv,
    SRem,
    And,
    Or,
    Xor,
    Shl,
    /// Logical shift right.
    Shr,
    /// Arithmetic shift right.
    Sar,
    /// Rotate left (x86 rol): width-aware.
    RotL,
    RotR,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    Not,
    Neg,
    /// Population count.
    Popcount,
    /// Leading zeros (lzcnt/bsr-idiom).
    Lzcnt,
    /// Trailing zeros (tzcnt/bsf-idiom).
    Tzcnt,
    Sqrt,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IntCmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Lt,
    Le,
    Gt,
    Ge,
}

impl IntCmp {
    pub const fn swap(self) -> IntCmp {
        match self {
            IntCmp::Eq => IntCmp::Eq,
            IntCmp::Ne => IntCmp::Ne,
            IntCmp::Ult => IntCmp::Ugt,
            IntCmp::Ule => IntCmp::Uge,
            IntCmp::Ugt => IntCmp::Ult,
            IntCmp::Uge => IntCmp::Ule,
            IntCmp::Lt => IntCmp::Gt,
            IntCmp::Le => IntCmp::Ge,
            IntCmp::Gt => IntCmp::Lt,
            IntCmp::Ge => IntCmp::Le,
        }
    }

    pub const fn negate(self) -> IntCmp {
        match self {
            IntCmp::Eq => IntCmp::Ne,
            IntCmp::Ne => IntCmp::Eq,
            IntCmp::Ult => IntCmp::Uge,
            IntCmp::Ule => IntCmp::Ugt,
            IntCmp::Ugt => IntCmp::Ule,
            IntCmp::Uge => IntCmp::Ult,
            IntCmp::Lt => IntCmp::Ge,
            IntCmp::Le => IntCmp::Gt,
            IntCmp::Gt => IntCmp::Le,
            IntCmp::Ge => IntCmp::Lt,
        }
    }

    pub const fn c_op(self) -> &'static str {
        match self {
            IntCmp::Eq => "==",
            IntCmp::Ne => "!=",
            IntCmp::Ult => "<",
            IntCmp::Ule => "<=",
            IntCmp::Ugt => ">",
            IntCmp::Uge => ">=",
            IntCmp::Lt => "<",
            IntCmp::Le => "<=",
            IntCmp::Gt => ">",
            IntCmp::Ge => ">=",
        }
    }

    /// Evaluate on raw u64 operands at `w`, honouring signedness.
    pub fn eval(self, a: u64, b: u64, w: Width, sx: Sx) -> bool {
        let (a, b) = if sx.signed() {
            (
                w.trunc(a, true) as i64 as u64,
                w.trunc(b, true) as i64 as u64,
            )
        } else {
            (w.trunc(a, false), w.trunc(b, false))
        };
        match (self, sx) {
            (IntCmp::Eq, _) => a == b,
            (IntCmp::Ne, _) => a != b,
            (IntCmp::Lt, Sx::Signed) => (a as i64) < b as i64,
            (IntCmp::Le, Sx::Signed) => (a as i64) <= b as i64,
            (IntCmp::Gt, Sx::Signed) => (a as i64) > b as i64,
            (IntCmp::Ge, Sx::Signed) => (a as i64) >= b as i64,
            (IntCmp::Lt, Sx::Unsigned) => a < b,
            (IntCmp::Le, Sx::Unsigned) => a <= b,
            (IntCmp::Gt, Sx::Unsigned) => a > b,
            (IntCmp::Ge, Sx::Unsigned) => a >= b,
            (IntCmp::Ult, _) => a < b,
            (IntCmp::Ule, _) => a <= b,
            (IntCmp::Ugt, _) => a > b,
            (IntCmp::Uge, _) => a >= b,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum CastKind {
    /// Truncate to N bits.
    Trunc,
    /// Zero-extend to N bits.
    ZExt,
    /// Sign-extend to N bits.
    SExt,
    /// Bitcast between same-width types (int <-> pointer, int <-> float).
    Bitcast,
    /// int -> float
    SIToFp,
    /// int (unsigned) -> float
    UIToFp,
    /// float -> int, truncating toward zero
    FpToSI,
    FpToUI,
    /// float widening / narrowing
    FPExt,
    FPTrunc,
}

/// One of the x86 EFLAGS bits we model.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Flag {
    CF,
    PF,
    AF,
    ZF,
    SF,
    OF,
}

/// An op that *defines* flags, i.e. produces a `Ty::Flags` value.
/// The operands are the same ones the x86 instruction had, so the flag
/// definitions below are per-op and local.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FlagsOp {
    /// a + b (+ carry_in). CF = carry out, OF = signed overflow,
    /// ZF/SF/PF/AF from the result.
    Add { a: VId, b: VId, carry: bool },
    /// a - b (- borrow_in). CF = borrow out.
    Sub { a: VId, b: VId, borrow: bool },
    /// a & b: CF=OF=0
    And { a: VId, b: VId },
    Or { a: VId, b: VId },
    Xor { a: VId, b: VId },
    /// a << b: CF = last bit shifted out, OF = MSB changed (count 1 only)
    Shl { a: VId, b: VId },
    /// a >> b logical
    Shr { a: VId, b: VId },
    /// a >> b arithmetic
    Sar { a: VId, b: VId },
    /// ~a + 1 (neg): CF = (a != 0)
    Neg { a: VId },
    /// a + 1: CF unchanged
    Inc { a: VId },
    /// a - 1: CF unchanged
    Dec { a: VId },
    /// a * b, full width: CF/OF = high word != sign-extension of low
    Mul { a: VId, b: VId, signed: bool },
}

/// A call target.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Callee {
    /// A symbol in the binary (or imported), by name.
    Symbol(String),
    /// An indirect call through this value.
    Indirect(VId),
    /// We know the address but not the name.
    Addr(u64),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    Const(u64),
    ConstF(f64),
    /// Uninitialized: reading it is `/*undef*/0`.
    Undef,
    /// A function parameter (def'ed in the entry block, never redefined).
    Param,
    /// SSA destructure artifact: `dest = src`. Always removed by copy
    /// propagation before C emission.
    Copy(VId),
    Bin(BinOp, VId, VId),
    Un(UnOp, VId),
    Cmp(IntCmp, VId, VId),
    /// Produce a `Ty::Flags` value.
    Flags(FlagsOp),
    /// Read one bit of a `Ty::Flags` value -> `Ty::Bool`.
    GetFlag(Flag, VId),
    Cast(CastKind, Ty, VId),
    Load(Ty, VId),
    Store(Ty, VId, VId),
    /// cond ? t : f  (cmov / setcc)
    Select(VId, VId, VId),
    /// (value, predecessor block) pairs.
    Phi(Box<[(VId, BbId)]>),
    Call {
        callee: Callee,
        ret_ty: Ty,
        args: Box<[VId]>,
        /// The call has side effects we do not model (printf, I/O, ...).
        /// Blocks DCE and blocks hoisting.
        pure: bool,
    },
    /// Something we could not model. Prints as a comment; blocks DCE.
    Opaque(Box<str>),
}

impl Op {
    /// Type of the result. `None` means the instruction has no destination,
    /// or (for `Select`/`Phi`) the type is the instruction's recorded type.
    pub fn ty(&self, f: &Function) -> Option<Ty> {
        Some(match self {
            Op::Const(_) => Ty::u64(),
            Op::ConstF(_) => Ty::f64(),
            Op::Undef | Op::Opaque(_) => return None,
            Op::Param | Op::Copy(_) => return None,
            Op::Bin(op, a, _) => {
                // Width comes from the operands; division/mul keep operand type.
                let ta = f.ty_of(*a);
                match op {
                    BinOp::MulHi | BinOp::MulSmul => ta.unwrap_or(Ty::u64()),
                    _ => ta.unwrap_or(Ty::u64()),
                }
            }
            Op::Un(op, a) => match op {
                UnOp::Sqrt => f.ty_of(*a).unwrap_or(Ty::f64()),
                _ => f.ty_of(*a).unwrap_or(Ty::u64()),
            },
            Op::Cmp(..) => Ty::Bool,
            Op::Flags(_) => Ty::Flags,
            Op::GetFlag(..) => Ty::Bool,
            Op::Cast(_, t, _) => t.clone(),
            Op::Load(t, _) => t.clone(),
            Op::Store(..) => Ty::Void,
            Op::Call { ret_ty, .. } => ret_ty.clone(),
            Op::Select(..) | Op::Phi(_) => return None,
        })
    }

    /// Values read by this op, in operand order.
    pub fn uses(&self, out: &mut Vec<VId>) {
        out.clear();
        match *self {
            Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Opaque(_) | Op::Param => {}
            Op::Copy(a) => out.push(a),
            Op::Bin(_, a, b) => out.extend([a, b]),
            Op::Un(_, a) | Op::Cast(_, _, a) => out.push(a),
            Op::Cmp(_, a, b) => out.extend([a, b]),
            Op::Flags(f) => match f {
                FlagsOp::Add { a, b, .. }
                | FlagsOp::Sub { a, b, .. }
                | FlagsOp::And { a, b }
                | FlagsOp::Or { a, b }
                | FlagsOp::Xor { a, b }
                | FlagsOp::Shl { a, b }
                | FlagsOp::Shr { a, b }
                | FlagsOp::Sar { a, b }
                | FlagsOp::Mul { a, b, .. } => out.extend([a, b]),
                FlagsOp::Neg { a } | FlagsOp::Inc { a } | FlagsOp::Dec { a } => out.push(a),
            },
            Op::GetFlag(_, f) => out.push(f),
            Op::Load(_, a) => out.push(a),
            Op::Store(_, a, v) => out.extend([a, v]),
            Op::Select(c, t, e) => out.extend([c, t, e]),
            Op::Phi(ref pairs) => out.extend(pairs.iter().map(|(v, _)| *v)),
            Op::Call { ref args, .. } => out.extend(args.iter().copied()),
        }
    }

    pub fn use_count(&self) -> usize {
        let mut v = Vec::new();
        self.uses(&mut v);
        v.len()
    }

    /// Pure, cheap, side-effect-free, and safe to duplicate into a use site?
    pub fn is_trivially_replicatable(&self) -> bool {
        match self {
            Op::Const(_) | Op::ConstF(_) | Op::Param | Op::Undef => true,
            Op::Copy(_) => true,
            Op::Bin(..) | Op::Un(..) | Op::Cmp(..) | Op::Cast(..) | Op::Select(..) => true,
            Op::Flags(_) => true,
            Op::GetFlag(..) => true,
            Op::Phi(_) | Op::Load(..) | Op::Store(..) | Op::Call { .. } | Op::Opaque(_) => false,
        }
    }

    /// Pure ops can be DCE'd when their result is unused.
    pub fn is_pure(&self) -> bool {
        match self {
            Op::Call { pure, .. } => *pure,
            Op::Store(..) | Op::Opaque(_) => false,
            _ => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub dest: VId,
    pub op: Op,
    /// Recorded type, authoritative for `Phi`/`Select`/`Undef`/`Opaque`/`Param`.
    pub ty: Ty,
    /// Address of the machine instruction this came from (for `; ` comments).
    pub addr: u64,
}

impl Instr {
    pub fn new(dest: VId, op: Op, ty: Ty, addr: u64) -> Instr {
        Instr { dest, op, ty, addr }
    }
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Jump(BbId),
    Branch {
        cond: VId,
        then_bb: BbId,
        else_bb: BbId,
    },
    Return(VId),
    /// Falls off the end / never executes.
    Unreachable,
    /// We ran out of modeled control flow (indirect jump, syscall, ...).
    /// Emits a `goto`/`exit()`-style escape so output stays compilable.
    Exit,
}

impl Terminator {
    pub fn successors(&self, out: &mut Vec<BbId>) {
        out.clear();
        match *self {
            Terminator::Jump(b) => out.push(b),
            Terminator::Branch { then_bb, else_bb, .. } => {
                out.push(then_bb);
                out.push(else_bb);
            }
            Terminator::Return(_) | Terminator::Unreachable | Terminator::Exit => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub id: BbId,
    /// Address of the first machine instruction, for symbolization + DOT.
    pub addr: u64,
    pub instrs: Vec<Instr>,
    pub term: Terminator,
}

impl Block {
    pub fn new(id: BbId, addr: u64) -> Block {
        Block {
            id,
            addr,
            instrs: Vec::new(),
            term: Terminator::Unreachable,
        }
    }

    pub fn push(&mut self, i: Instr) {
        self.instrs.push(i);
    }
}

/// Metadata for a value.
#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub ty: Ty,
    /// Name hint used by the C printer (`v12` if `None`).
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub addr: u64,
    pub blocks: Vec<Block>,
    pub entry: BbId,
    pub values: Vec<ValueInfo>,
    pub next_value: VId,
    pub next_block: BbId,
    /// Declared parameter order (entry-block `Op::Param` defs).
    pub params: Vec<VId>,
    pub ret_ty: Ty,
}

impl Function {
    pub fn new(name: impl Into<String>, addr: u64) -> Function {
        let mut f = Function {
            name: name.into(),
            addr,
            blocks: Vec::new(),
            entry: NO_BB,
            values: Vec::new(),
            next_value: 0,
            next_block: 0,
            params: Vec::new(),
            ret_ty: Ty::Void,
        };
        let e = f.new_block(0);
        f.entry = e;
        f
    }

    pub fn new_value(&mut self, ty: Ty) -> VId {
        let id = self.next_value;
        self.next_value += 1;
        if id as usize == self.values.len() {
            self.values.push(ValueInfo { ty, name: None });
        } else {
            self.values[id as usize].ty = ty;
        }
        id
    }

    pub fn new_block(&mut self, addr: u64) -> BbId {
        let id = self.next_block;
        self.next_block += 1;
        if id as usize == self.blocks.len() {
            self.blocks.push(Block::new(id, addr));
        }
        id
    }

    /// Append an op to `bb`, allocating its destination.
    pub fn emit(&mut self, bb: BbId, op: Op, ty: Ty, addr: u64) -> VId {
        let t = op.ty(self).unwrap_or(ty);
        let v = self.new_value(t.clone());
        self.blocks[bb as usize].push(Instr {
            dest: v,
            op,
            ty: t,
            addr,
        });
        v
    }

    /// Append an op with a known-void result (store, void call).
    pub fn emit_void(&mut self, bb: BbId, op: Op, addr: u64) {
        let v = self.new_value(Ty::Void);
        self.blocks[bb as usize].push(Instr {
            dest: v,
            op,
            ty: Ty::Void,
            addr,
        });
    }

    pub fn block(&self, b: BbId) -> &Block {
        &self.blocks[b as usize]
    }

    pub fn block_mut(&mut self, b: BbId) -> &mut Block {
        &mut self.blocks[b as usize]
    }

    pub fn ty_of(&self, v: VId) -> Option<Ty> {
        self.values.get(v as usize).map(|i| i.ty.clone())
    }

    pub fn set_ty(&mut self, v: VId, ty: Ty) {
        self.values[v as usize].ty = ty;
    }

    /// Locate the instruction defining `v`. Recomputed per query on purpose:
    /// use-lists are the classic source of decompiler bugs, and our functions
    /// are small enough that a scan is not the bottleneck.
    pub fn def(&self, v: VId) -> Option<(BbId, usize)> {
        for b in &self.blocks {
            for (i, ins) in b.instrs.iter().enumerate() {
                if ins.dest == v {
                    return Some((b.id, i));
                }
            }
        }
        None
    }

    /// All (block, instr index, operand slot) uses of `v`.
    pub fn uses_of(&self, v: VId) -> Vec<UseSite> {
        let mut out = Vec::new();
        for b in &self.blocks {
            for (ii, ins) in b.instrs.iter().enumerate() {
                let mut ops = Vec::new();
                ins.op.uses(&mut ops);
                for (slot, u) in ops.iter().enumerate() {
                    if *u == v {
                        out.push(UseSite {
                            bb: b.id,
                            instr: ii,
                            slot,
                        });
                    }
                }
            }
            let mut ops: Vec<VId> = Vec::new();
            if let Terminator::Branch { cond, .. } = b.term {
                if cond == v {
                    out.push(UseSite {
                        bb: b.id,
                        instr: usize::MAX,
                        slot: 0,
                    });
                }
            }
            let _ = &mut ops;
        }
        out
    }

    /// Rewrite every use of `old` to `new`, in operands and in terminators.
    pub fn replace_use_all(&mut self, old: VId, new: VId) {
        if old == new {
            return;
        }
        for b in self.blocks.iter_mut() {
            for i in b.instrs.iter_mut() {
                i.op.map_uses(&mut |u: &mut VId| {
                    if *u == old {
                        *u = new;
                    }
                });
            }
            if let Terminator::Branch { cond, .. } = &mut b.term {
                if *cond == old {
                    *cond = new;
                }
            }
        }
    }

    pub fn use_count(&self, v: VId) -> usize {
        self.uses_of(v).len()
    }

    pub fn blocks_in_addr_order(&self) -> Vec<BbId> {
        let mut v: Vec<BbId> = (0..self.blocks.len() as BbId).collect();
        v.sort_by_key(|b| self.block(*b).addr);
        v
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct UseSite {
    pub bb: BbId,
    /// `usize::MAX` marks a use in the block terminator.
    pub instr: usize,
    pub slot: usize,
}

/// `Op::ty` needs a `&Function` for operand types.
pub use crate::fmt_ir::{format_instr, format_term};

impl fmt::Display for Flag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Flag::CF => "CF",
            Flag::PF => "PF",
            Flag::AF => "AF",
            Flag::ZF => "ZF",
            Flag::SF => "SF",
            Flag::OF => "OF",
        })
    }
}
