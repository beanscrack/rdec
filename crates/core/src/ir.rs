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

/// Floating-point binary ops. Deliberately NOT `BinOp::Add` with float
/// operands: every consumer of `BinOp` computes in `u64` — the interpreter's
/// integer `binop`, the constant folder, the flag tables, `width::bits_needed`.
/// Reusing it for floats would make `addsd` an INTEGER ADD of two bit patterns,
/// which is not the machine's answer, and the operands' float types would not
/// stop it. `Op::FBin` is the op that means the float thing.
///
/// Each op happens at the precision of its OPERANDS: `addss` takes F32 and
/// answers F32, `addsd` takes F64 and answers F64. `Op::FBin`'s result type is
/// the type of its first operand for exactly that reason.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FBinOp {
    Add,
    Sub,
    Mul,
    /// IEEE division: by zero yields an infinity, it does not trap. This is why
    /// `FBin` is safe to replicate into use sites like `Bin` is.
    Div,
    /// `maxsd`/`maxss`. x86 returns the SECOND operand when either side is
    /// NaN, which is not `a > b ? a : b` — the emitter spells the NaN rule out,
    /// so the C agrees with the hardware on NaN inputs.
    Max,
    /// `minsd`/`minss`, with x86's NaN rule as for `Max`.
    Min,
    /// `sqrtsd`/`sqrtss`, at the operand's own precision.
    Sqrt,
}

/// Floating-point comparisons, named for the C operator they spell. Ordered:
/// `<`/`<=`/`>`/`>=`/`==` are `false` when either operand is NaN, and `!=` is
/// `true` — the IEEE/C rule, stated per predicate because the guard that uses
/// them (`cvttsd2si`, `maxsd`) depends on exactly that behaviour.
///
/// Only the C spellings exist here, on purpose. `ucomisd`/`comisd` set
/// ZF/CF/PF so that an unordered compare is a *three-bit* answer ("below",
/// "equal", "above-or-equal", "unordered"), and the `jcc` spelling that reads
/// it (`ja`/`jb`/`jp` combinations) is not one Bool. The lifter refuses those
/// mnemonics by name; this enum cannot express a claim about them.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FloatCmp {
    /// `x == y` — false when either side is NaN.
    Eq,
    /// `x != y` — TRUE when either side is NaN (x86's UN+NEQ, C's `!=`).
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl FloatCmp {
    /// C's operator for this predicate.
    pub const fn c_op(self) -> &'static str {
        match self {
            FloatCmp::Eq => "==",
            FloatCmp::Ne => "!=",
            FloatCmp::Lt => "<",
            FloatCmp::Le => "<=",
            FloatCmp::Gt => ">",
            FloatCmp::Ge => ">=",
        }
    }
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
    /// Square root, at the operand's own type: `double` for F64, `float` for
    /// F32 (`sqrtsd` / `sqrtss`).
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
    Add {
        a: VId,
        b: VId,
        carry: bool,
    },
    /// a - b (- borrow_in). CF = borrow out.
    Sub {
        a: VId,
        b: VId,
        borrow: bool,
    },
    /// a & b: CF=OF=0
    And {
        a: VId,
        b: VId,
    },
    Or {
        a: VId,
        b: VId,
    },
    Xor {
        a: VId,
        b: VId,
    },
    /// a << b: CF = last bit shifted out, OF = MSB changed (count 1 only)
    Shl {
        a: VId,
        b: VId,
    },
    /// a >> b logical
    Shr {
        a: VId,
        b: VId,
    },
    /// a >> b arithmetic
    Sar {
        a: VId,
        b: VId,
    },
    /// ~a + 1 (neg): CF = (a != 0)
    Neg {
        a: VId,
    },
    /// a + 1: CF unchanged
    Inc {
        a: VId,
    },
    /// a - 1: CF unchanged
    Dec {
        a: VId,
    },
    /// a * b, full width: CF/OF = high word != sign-extension of low
    Mul {
        a: VId,
        b: VId,
        signed: bool,
    },
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
    /// Floating-point arithmetic at the operands' float type (`addsd`, `mulss`,
    /// ...). The result type is the type of the FIRST operand: the operation
    /// happens at the operand precision, and recording F64 for an F32 `addss`
    /// would have the IR claim a wider answer than the machine produced.
    FBin(FBinOp, VId, VId),
    /// Floating-point comparison -> `Ty::Bool`, at the operands' float type.
    FCmp(FloatCmp, VId, VId),
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
    /// A call. `callee` carries the target's IDENTITY — the callee's function
    /// name as the generated translation unit spells it (`Symbol`), not an
    /// address: an address is only an identity inside one object's byte range,
    /// and the emitter, the interpreter and the gates all have to agree on who
    /// is being called. `Addr` is the loud case (a target with no symbol),
    /// `Indirect` the one no signature model reaches.
    ///
    /// `args` are the argument VALUES in the callee's DECLARED order: the
    /// caller's register cells for the pair each declared parameter owns
    /// (`ARG_REGS[pos]` for an integer kind, `ARG_XMM[pos]` for a float kind).
    /// `ret_ty` is the callee's declared return type, so the op's result is
    /// RAX's answer for an integer return and XMM0's answer for a float one.
    ///
    /// The op defines ONLY its result. The registers the call destroys are
    /// separate `Op::Clobber` definitions, one per caller-saved register the
    /// call does not itself define — a definition per register, because the
    /// block's outgoing register map is the transfer function SSA solves.
    Call {
        callee: Callee,
        ret_ty: Ty,
        args: Box<[VId]>,
        /// The call computes a value from its arguments and has no effects we
        /// do not model: DCE may delete it when the result is unused.
        /// `pure: false` is the loud default for printf, I/O, and anything
        /// whose behaviour is not modelled.
        pure: bool,
    },
    /// A caller-saved register destroyed by a call. It reads nothing and
    /// defines a value no C expression can name.
    ///
    /// This is the clobber, and it is an op of its own (not `Op::Opaque`) for
    /// one reason: an *unused* clobber is dead. DCE may delete a definition
    /// that no read observes, and that is what lets a function that calls a
    /// sibling and never re-reads RCX emit C at all. `Op::Opaque` deliberately
    /// blocks DCE (an unmodelled instruction must never vanish quietly), and
    /// applying that rule to the clobber is exactly why every function that
    /// calls anything used to be refused.
    ///
    /// Nothing is lost by the split. A *read* of this value stays loud
    /// everywhere it must be: the interpreter faults (`Fault::Clobbered`, never
    /// a zero), and the emitter refuses with the class it has always used
    /// ("opaque: clobbered by call"). A function that reads a caller-saved
    /// register after a call is a function whose answer the call destroyed, and
    /// that stays a refusal.
    Clobber,
    /// Something we could not model. Prints as a comment; blocks DCE.
    Opaque(Box<str>),
}

impl Op {
    /// Type of the result. `None` means the instruction has no destination,
    /// or (for `Select`/`Phi`) the type is the instruction's recorded type.
    pub fn ty(&self, f: &Function) -> Option<Ty> {
        Some(match self {
            // A constant has no intrinsic width: `1` is the same bit pattern at
            // every width, and which width it means is the *emitter's* claim.
            // Returning `Ty::u64()` here (the first version) made `Function::emit`
            // overwrite the width the lifter declared, so `add ecx, 1` became a
            // W32 + W64 mix - the lifter asked for uint32_t and silently got a
            // 64-bit constant. The value's stored type is the answer.
            Op::Const(_) => return None,
            Op::ConstF(_) => Ty::f64(),
            Op::Undef | Op::Opaque(_) => return None,
            // A register cell: the register file is one unsigned W64 word per
            // register, caller-saved clobber included (GPR or XMM cell alike).
            Op::Clobber => Ty::u64(),
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
            Op::FBin(_, a, _) => f.ty_of(*a).unwrap_or(Ty::f64()),
            Op::FCmp(..) => Ty::Bool,
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
            Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Opaque(_) | Op::Clobber | Op::Param => {}
            Op::Copy(a) => out.push(a),
            Op::Bin(_, a, b) => out.extend([a, b]),
            Op::FBin(_, a, b) | Op::FCmp(_, a, b) => out.extend([a, b]),
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
            // `FBin`/`FCmp` are in this group with `Bin`/`Cmp`: a pure
            // recomputation. Division by zero is NOT a trap in IEEE arithmetic
            // (it yields an infinity), so duplicating a `fdiv` cannot turn a
            // faulting execution into a returning one -- which is exactly the
            // reason an x86 `div` is NOT in this group.
            Op::FBin(..) | Op::FCmp(..) => true,
            Op::Bin(..) | Op::Un(..) | Op::Cmp(..) | Op::Cast(..) | Op::Select(..) => true,
            Op::Flags(_) => true,
            Op::GetFlag(..) => true,
            Op::Phi(_) | Op::Load(..) | Op::Store(..) | Op::Call { .. } | Op::Opaque(_) => false,
            // The clobber defines "a value the call destroyed": not a value any
            // use site may recompute, and not one it may name.
            Op::Clobber => false,
        }
    }

    /// Pure ops can be DCE'd when their result is unused.
    pub fn is_pure(&self) -> bool {
        match self {
            Op::Call { pure, .. } => *pure,
            Op::Store(..) | Op::Opaque(_) => false,
            // The clobber computes nothing and mutates nothing: it is a
            // DEFINITION whose only meaning is what a later read of it means.
            // Delete it when nothing reads it — the call that destroyed the
            // register is a separate, impure op and stays put.
            Op::Clobber => true,
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
    /// A computed jump whose destinations are known: one successor per table
    /// slot, `targets[i]` for index `i`. THE SUCCESSORS ARE THE TABLE - there is
    /// no target list beside the CFG, so the verifier, the interpreter and the
    /// emitter all see the same control flow, and a target that is not a block of
    /// this function is a verifier failure, not a note.
    ///
    /// The index is a 64-bit value. An index at or past `targets.len()` is not
    /// modelled: a machine code jump table covers exactly the slots it has, and
    /// the arm of a C `switch` that is not in the table is not a destination this
    /// terminator can name. The interpreter faults there and the emitter refuses;
    /// neither of them guesses.
    Switch {
        index: VId,
        targets: Vec<BbId>,
    },
    /// Falls off the end / never executes.
    Unreachable,
    /// We ran out of modeled control flow (indirect jump, syscall, ...).
    /// Emits a `goto`/`exit()`-style escape so output stays compilable.
    Exit,
    /// `return callee(args);` — the machine code left the function through a
    /// `jmp` whose relocation named the callee (the `e9 rel32` +
    /// `IMAGE_REL_AMD64_REL32 -> g` shape). The call's value IS this function's
    /// answer; control never returns to the caller's code.
    ///
    /// `value` is defined by an `Op::Call` in the SAME block, and `verify_ssa`
    /// requires it with a named callee: a `jmp` computes nothing, so a
    /// `TailReturn` of anything else means the IR invented a tail call, and an
    /// unnamed callee means nobody's answer is being returned.
    ///
    /// NO successor, which is the reason the terminator exists: a tail call has
    /// no fall-through edge, so the bytes after it are unreachable from it.
    /// Ending the block with `Return` on a fall-through edge would be the lie —
    /// it claims the block ran to its end.
    TailReturn(VId),
    /// The block ended in a call to a callee that never returns (`abort`, the
    /// CRT exit family, a C++ throw). The call defines nothing, the caller-saved
    /// registers are destroyed as at any call, and nothing executes after it —
    /// so, like a tail call, NO successor: the code after the call is
    /// unreachable and the function has no fall-through answer. Falling through
    /// to an `Exit` would invent an edge the machine code does not have.
    NoReturn,
}

impl Terminator {
    pub fn successors(&self, out: &mut Vec<BbId>) {
        out.clear();
        match *self {
            Terminator::Jump(b) => out.push(b),
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                out.push(then_bb);
                out.push(else_bb);
            }
            Terminator::Switch { targets: ref t, .. } => out.extend(t.iter().copied()),
            Terminator::Exit
            | Terminator::Unreachable
            | Terminator::Return(_)
            | Terminator::TailReturn(_)
            | Terminator::NoReturn => {}
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
    /// The DECLARED C type of each parameter, parallel to `params` when the
    /// ABI is known. A Windows x64 float parameter arrives in XMM0 as a bit
    /// cell, so its IR value is `Ty::u64` (the register file is one unsigned
    /// W64 word per register, float cells included) while the C signature
    /// spells it `float`/`double`. The emitter declares the parameter with
    /// this type and moves bits through the memcpy helpers; when the list is
    /// empty (no declared signature), the parameter's own value type is the
    /// declared type, which is the integer-ABI behaviour the C corpus has
    /// always had.
    pub param_tys: Vec<Ty>,
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
            param_tys: Vec::new(),
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
            // Terminator operands are uses. `Return` was missing here, which is
            // how DCE came to delete the expression a function returns (B1).
            let top: Option<VId> = match b.term {
                Terminator::Branch { cond, .. } => Some(cond),
                Terminator::Return(x) | Terminator::TailReturn(x) if x != NO_VAL => Some(x),
                // The switch's selector is a use: a pass that deletes its
                // definition leaves a jump on a value that does not exist.
                Terminator::Switch { index, .. } => Some(index),
                _ => None,
            };
            if top == Some(v) {
                out.push(UseSite {
                    bb: b.id,
                    instr: usize::MAX,
                    slot: 0,
                });
            }
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
            // Match with a guard so the rewrite is one arm per terminator.
            match &mut b.term {
                Terminator::Branch { cond, .. } if *cond == old => *cond = new,
                Terminator::Return(x) | Terminator::TailReturn(x) if *x == old => *x = new,
                Terminator::Switch { index, .. } if *index == old => *index = new,
                _ => {}
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
