//! M2 step 1: lift x86-64 machine code into `rdec-core` IR.
//!
//! The contract, and why the code is shaped this way:
//!
//! * Each register is ONE `W64` SSA value. A 32-bit read is `trunc`, a 32-bit
//!   write is `zext` - which is exactly the x86 rule that a 32-bit write clears
//!   the upper half. 8/16-bit writes are the mask/or pair the hardware performs.
//!   One representation per register is what keeps SSA, the verifier and the
//!   interpreter honest.
//! * Each XMM register is a PAIR of `W64` SSA cells — its low 64 bits
//!   (`Grp::Xmm(n)`) and its high 64 bits (`Grp::XmmHi(n)`) — bit patterns,
//!   not floats. That is a choice, and the reasons are here rather than in a
//!   comment at the first `movsd` because the emitter's C types follow from it.
//!   - The alternative is a cell typed `Ty::Float{F64}` (a scalar double view),
//!     and it breaks at the first `movss`: `movss` writes 32 bits of the
//!     register and zeroes 127:32, so the cell would have to be a `float` one
//!     instruction and a `double` the next, and a phi over a loop-carried
//!     register would have two types. The register file's job is to be ONE
//!     value per lane; a type that changes with the instruction defeats it.
//!   - At bits, every modelled op is expressible: `movsd`/`movss` move bits in
//!     and out, and each arithmetic, comparison or conversion op starts with an
//!     explicit `Cast(Bitcast, Ty::Float{w}, bits)` view (and writes the result
//!     back through the inverse). The C follows the same shape - `uint64_t`
//!     cells, `rdec_bits_f64(...)` / `rdec_f64_bits(...)` at the uses - so
//!     nobody has to guess where the number stops being a bit pattern.
//!   - The pair, not one cell, is what makes the 128-bit data path modelable
//!     WITHOUT a new type: a `movdqu`/`movaps`/`movdqa`/`movup*`/`movapd` load
//!     is two W64 loads (lo at addr, hi at addr+8), a store is two stores, and
//!     the lane-independent bitwise set (`pxor`/`pand`/`pandn`/`por`) is two
//!     independent W64 ops - none of it is lane-crossing, so the W64 lane
//!     model expresses all of it exactly. The same rule reaches the packed
//!     ARITHMETIC whose lanes never talk: `paddq`/`psubq` are two u64 lane
//!     ops, `paddd`/`psubd` are two W32 lane ops per cell (each op typed at its
//!     LANE's width, so the carry or borrow has no wider cell to travel in),
//!     and `pcmpeqd`/`pcmpeqq` answer all-ones / zero per lane. The pair's rule for scalar ops is
//!     the hardware's, per form, never a simplification: the scalar STORE
//!     (`movsd m64, xmm`) writes 8 bytes and leaves the register's high cell
//!     exactly as it was; the register/memory LOAD (`movsd xmm, ...`,
//!     `movss xmm, ...`, `movd`/`movq` gpr→xmm) zeroes the high cell (Intel's
//!     merge forms), and the write names that zero; the merge arithmetic
//!     (`addsd`-family, `cvtsi2sd`-family) writes the low cell and preserves
//!     the high one. The ops that cross lanes (`shufps`, `punpck*`,
//!     `pshufd`), the packed arithmetic that is not lane-independent
//!     (`pmul*`, the `psl*`/`psr*` shifts, saturation), the
//!     mask reads (`movmskps`) and the conversions between packed shapes
//!     (`cvtp*`) STAY refused by name, with the mnemonic and the address, and
//!     counted: the refusal census is the next slice's map. A model that
//!     silently ignores a 128-bit read is the miscompile; a refusal is the
//!     work item. Pinned by `crates/arch-x86/tests/packed_simd.rs`.
//! * Flag-defining instructions produce a `Ty::Flags` value; `jcc`/`setcc`/
//!   `cmovcc` read it with `GetFlag`. Dead flags then die by DCE, and `fuse.rs`
//!   turns `(cmp, jl)` into a signed compare. Flag semantics are width-dependent
//!   and width comes from the operand TYPES, so operands handed to a `FlagsOp`
//!   are truncated to the instruction's operand width first.
//! * A call clobbers the caller-saved registers. That is modelled, as an
//!   [`Op::Clobber`] definition per clobbered register, because leaving a value
//!   live across a call is a silent miscompile, and the silent kind is the one
//!   that ships. A DIRECT call to a target the pipeline names and types is
//!   modelled further: the call op carries the callee's identity and its
//!   declared arguments, and DEFINES its answer — RAX for a declared integer
//!   return, XMM0's cell for a declared float one — so a function that calls a
//!   decompiled sibling and uses the answer emits C. The clobbers do not shrink
//!   for it: the call defines its outputs and clobbers the rest. Everything the
//!   identity or the shape does not reach — indirect, no symbol, more arguments
//!   than the register pairs, an untypable callee — keeps its loud shape and its
//!   refusal, and each class is counted in `Report`.
//! * The Windows x64 ABI homes every parameter by REGISTER PAIR: declared
//!   parameter `k` arrives in the pair that owns RCX/RDX/R8/R9[k] AND
//!   XMM0-3[k], so a float parameter consumes no integer register and an
//!   integer parameter no XMM, and the index into each class's four registers
//!   is the DECLARED POSITION. A `__thiscall` callee spends pair 0 - RCX and
//!   XMM0 - on `this`, so its first `double` arrives in XMM1; a free
//!   `int f(int a, double b)` gets the same XMM1 for the same reason, because
//!   `b` is at position 1. Which kind each parameter has is NOT visible in the
//!   machine code - it is a property of the declared signature - so the lifter
//!   does not infer it: the caller hands in an `FnAbi` per function
//!   (`lift_one_sig`), and the C++ pipeline builds it from the DEMANGLED
//!   SIGNATURE, saying out loud that the evidence is the decorated name, not
//!   observed answers. An empty `FnAbi` is the integer model exactly as it was:
//!   the four integer registers seeded as parameters, an integer return. A
//!   signature the front end cannot fully type keeps the integer model and is
//!   counted and named - never guessed. `ret` follows the same declared kind: a
//!   float/double function answers in XMM0 (the cell's low 32/64 bits), and the
//!   emitted C declares the float return type.
//! * Everything outside the subset becomes `Op::Opaque(text)` - in the IR,
//!   counted, printed. Never a plausible guess.
//!
//! Block boundaries come from `discover`, so the lifter never invents one that
//! the decoder and the descent did not prove.

use std::collections::{HashMap, HashSet};

use iced_x86::{Decoder, Instruction, Mnemonic, OpKind, Register};
use rdec_core::ir::{
    BbId, BinOp, Callee, CastKind, FBinOp, Flag, FlagsOp, Function, Instr, IntCmp, Op, Terminator,
    UnOp, VId,
};
use rdec_core::ty::{FWidth, Ty, Width};

use crate::discover::{Plan, Tables};

/// A named register group: every width view of one register names one SSA value.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Grp {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    /// One XMM register's LOW 64 bits, by number. `Grp` is the register-file
    /// key, and the SSA machinery (live-ins, phis, the block transfer function,
    /// call clobbers) is keyed on it — so an XMM register is a register here,
    /// not a side table that the SSA pass forgets to consult. The register is
    /// the PAIR (`Xmm`, `XmmHi`): the scalar model reads and writes this cell,
    /// the packed data path reads and writes both, and the module doc says
    /// which form preserves which. The value is a bit pattern; see the module
    /// doc for why bits and not a float.
    Xmm(u8),
    /// The same register's HIGH 64 bits (bits 127:64). It exists so a
    /// 16-byte load/store and a lane-independent bitwise op can be modelled
    /// honestly; a scalar `movsd m64, xmm` must leave it untouched and a
    /// `movsd xmm, ...` must zero it, and both rules need a cell to act on.
    XmmHi(u8),
}

/// What one width-view of a register means. High-byte forms (AH/CH/DH/BH) are
/// refused: they alias a value we model as one 64-bit word, and modelling that
/// alias badly is worse than not lifting the function.
fn split(reg: Register) -> Option<(Grp, Width)> {
    use Grp::*;
    use Register as R;
    use Width::*;
    let (g, w) = match reg {
        R::AL => (Rax, W8),
        R::AX => (Rax, W16),
        R::EAX => (Rax, W32),
        R::RAX => (Rax, W64),
        R::CL => (Rcx, W8),
        R::CX => (Rcx, W16),
        R::ECX => (Rcx, W32),
        R::RCX => (Rcx, W64),
        R::DL => (Rdx, W8),
        R::DX => (Rdx, W16),
        R::EDX => (Rdx, W32),
        R::RDX => (Rdx, W64),
        R::BL => (Rbx, W8),
        R::BX => (Rbx, W16),
        R::EBX => (Rbx, W32),
        R::RBX => (Rbx, W64),
        R::SPL => (Rsp, W8),
        R::SP => (Rsp, W16),
        R::ESP => (Rsp, W32),
        R::RSP => (Rsp, W64),
        R::BPL => (Rbp, W8),
        R::BP => (Rbp, W16),
        R::EBP => (Rbp, W32),
        R::RBP => (Rbp, W64),
        R::SIL => (Rsi, W8),
        R::SI => (Rsi, W16),
        R::ESI => (Rsi, W32),
        R::RSI => (Rsi, W64),
        R::DIL => (Rdi, W8),
        R::DI => (Rdi, W16),
        R::EDI => (Rdi, W32),
        R::RDI => (Rdi, W64),
        R::R8L => (R8, W8),
        R::R8W => (R8, W16),
        R::R8D => (R8, W32),
        R::R8 => (R8, W64),
        R::R9L => (R9, W8),
        R::R9W => (R9, W16),
        R::R9D => (R9, W32),
        R::R9 => (R9, W64),
        R::R10L => (R10, W8),
        R::R10W => (R10, W16),
        R::R10D => (R10, W32),
        R::R10 => (R10, W64),
        R::R11L => (R11, W8),
        R::R11W => (R11, W16),
        R::R11D => (R11, W32),
        R::R11 => (R11, W64),
        R::R12L => (R12, W8),
        R::R12W => (R12, W16),
        R::R12D => (R12, W32),
        R::R12 => (R12, W64),
        R::R13L => (R13, W8),
        R::R13W => (R13, W16),
        R::R13D => (R13, W32),
        R::R13 => (R13, W64),
        R::R14L => (R14, W8),
        R::R14W => (R14, W16),
        R::R14D => (R14, W32),
        R::R14 => (R14, W64),
        R::R15L => (R15, W8),
        R::R15W => (R15, W16),
        R::R15D => (R15, W32),
        R::R15 => (R15, W64),
        // XMM registers: one group per register, always read as the whole
        // 64-bit low cell. The 128-bit view does not exist in this model, which
        // is why packed ops are refused rather than half-modelled.
        R::XMM0 => (Xmm(0), W64),
        R::XMM1 => (Xmm(1), W64),
        R::XMM2 => (Xmm(2), W64),
        R::XMM3 => (Xmm(3), W64),
        R::XMM4 => (Xmm(4), W64),
        R::XMM5 => (Xmm(5), W64),
        R::XMM6 => (Xmm(6), W64),
        R::XMM7 => (Xmm(7), W64),
        R::XMM8 => (Xmm(8), W64),
        R::XMM9 => (Xmm(9), W64),
        R::XMM10 => (Xmm(10), W64),
        R::XMM11 => (Xmm(11), W64),
        R::XMM12 => (Xmm(12), W64),
        R::XMM13 => (Xmm(13), W64),
        R::XMM14 => (Xmm(14), W64),
        R::XMM15 => (Xmm(15), W64),
        // AH/CH/DH/BH alias a value we model as one 64-bit word: refuse.
        _ => return None,
    };
    Some((g, w))
}

/// Every modelled register, in a fixed order: the SSA pass walks registers in
/// this order, which is what makes the emitted phi order deterministic. The XMM
/// cells follow the GPRs: the SSA pass is keyed on `Grp`, so adding them here
/// is what gives an XMM register a live-in, a phi at a join, and a clobber at a
/// call, with no second copy of that machinery. The pair (lo, hi) of one
/// register is adjacent, so a register's cells read together in every walk.
const GRPS: [Grp; 48] = [
    Grp::Rax,
    Grp::Rcx,
    Grp::Rdx,
    Grp::Rbx,
    Grp::Rsp,
    Grp::Rbp,
    Grp::Rsi,
    Grp::Rdi,
    Grp::R8,
    Grp::R9,
    Grp::R10,
    Grp::R11,
    Grp::R12,
    Grp::R13,
    Grp::R14,
    Grp::R15,
    Grp::Xmm(0),
    Grp::XmmHi(0),
    Grp::Xmm(1),
    Grp::XmmHi(1),
    Grp::Xmm(2),
    Grp::XmmHi(2),
    Grp::Xmm(3),
    Grp::XmmHi(3),
    Grp::Xmm(4),
    Grp::XmmHi(4),
    Grp::Xmm(5),
    Grp::XmmHi(5),
    Grp::Xmm(6),
    Grp::XmmHi(6),
    Grp::Xmm(7),
    Grp::XmmHi(7),
    Grp::Xmm(8),
    Grp::XmmHi(8),
    Grp::Xmm(9),
    Grp::XmmHi(9),
    Grp::Xmm(10),
    Grp::XmmHi(10),
    Grp::Xmm(11),
    Grp::XmmHi(11),
    Grp::Xmm(12),
    Grp::XmmHi(12),
    Grp::Xmm(13),
    Grp::XmmHi(13),
    Grp::Xmm(14),
    Grp::XmmHi(14),
    Grp::Xmm(15),
    Grp::XmmHi(15),
];

fn ty_u(w: Width) -> Ty {
    Ty::uint(w)
}

/// The packed arithmetic this lifter models as pure IR over the XMM pair, and
/// the LANE WIDTH each one runs at. The two u64 forms are two lane ops of the
/// pair; the two u32 forms are two lane ops PER CELL at W32, so the lane's own
/// add/sub truncation is what stops a carry or borrow crossing (`Lifter::cell32`);
/// the compares answer all-ones / zero per lane (`Lifter::mask_at`).
///
/// What is deliberately NOT here: the 8- and 16-bit lanes (`paddb`, `paddw`,
/// `psubw`...), the saturating forms, the multiplies, the shifts, the
/// comparisons other than EQ, and every shuffle/unpack — all stay loud
/// `Op::Opaque` gaps naming the mnemonic, because a half-modelled lane width is
/// the silent kind of miscompile and the census is the next slice's map.
enum PackedArith {
    Add64,
    Sub64,
    Eq64,
    Add32,
    Sub32,
    Eq32,
}

/// The Windows x86-64 integer argument registers, indexed by DECLARED
/// PARAMETER POSITION (pair index), not by "the n-th integer": pair `k` owns
/// `ARG_REGS[k]` and `ARG_XMM[k]` together. Four pairs, integer return; a
/// parameter past them is a loud gap, not a drop.
const ARG_REGS: [Grp; 4] = [Grp::Rcx, Grp::Rdx, Grp::R8, Grp::R9];

/// The float argument registers, indexed by the same DECLARED PARAMETER
/// POSITION (pair index) as `ARG_REGS`. The Windows x64 ABI homes parameter `k`
/// in the pair that owns `ARG_REGS[k]` AND `ARG_XMM[k]`: a float parameter
/// consumes no integer register and an integer parameter no XMM, so the index
/// into each class is the position, never a per-class counter. `int f(int a,
/// double b)` arrives with `a` in RCX (pair 0) and `b` in XMM1 (pair 1) - RDX is
/// NOT "used" by `b`, and XMM0 (pair 0's float half, spent on the `int`) holds
/// nothing this function may read. Pinned on hand-assembled bytes in
/// `tests/float_abi.rs`; the corpus' member `grow` says the same thing from the
/// other side (`movsd %xmm1, 0x8(%rsp)`).
const ARG_XMM: [Grp; 4] = [Grp::Xmm(0), Grp::Xmm(1), Grp::Xmm(2), Grp::Xmm(3)];

/// Which register class one declared parameter arrives in. The kind is a
/// property of the DECLARED signature (the demangled C++ name, the C prototype
/// the front end was given) - never a conclusion drawn from what the lifted
/// code does with the register, because a type inferred from observed answers
/// is the gate that reads the answer off the answer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArgKind {
    /// An integer-class parameter: delivered in `ARG_REGS[pos]`, the integer
    /// half of the pair its DECLARED POSITION owns.
    Int,
    /// `float`: delivered in `ARG_XMM[pos]`, the float half of that same pair.
    /// The cell's value is the 32-bit bit pattern zero-extended to 64 - the
    /// `movss` a scalar caller must use zeroes bits 127:32, so bits 63:32 are
    /// architecturally zero.
    F32,
    /// `double`: delivered in `ARG_XMM[pos]` like `F32`, and the cell's value is
    /// the bits. `int f(int a, double b)` puts `b` in XMM1, pair 1.
    F64,
}

/// Which register `ret` answers from, per the declared signature.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum RetKind {
    /// The integer answer in RAX (the model as it has always been). The
    /// default: no declared signature means the integer model.
    #[default]
    Int,
    /// A `float` answer in XMM0's low 32 bits.
    F32,
    /// A `double` answer in XMM0's low 64 bits.
    F64,
}

/// The declared shape of one function: parameter kinds in declaration order,
/// and the return kind. An empty `args` is "no signature was given": seed the
/// four integer argument registers, exactly the pre-float-ABI model, so every
/// caller that knows nothing keeps the behaviour it has always had.
#[derive(Clone, Debug, Default)]
pub struct FnAbi {
    pub args: Vec<ArgKind>,
    pub ret: RetKind,
    /// Member function (`this` is `args[0]`). Windows x64 lays parameters
    /// into four PAIRED slots — slot k owns `RCX/RDX/R8/R9[k]` AND
    /// `XMM0-3[k]` — and a `__thiscall` callee takes `this` in slot 0, so
    /// DECLARED parameter k lands in slot k of its register class: a
    /// member's first `double` arrives in XMM1, not XMM0. The corpus says
    /// so as machine fact: `?grow@Circle@@QEAANN@Z` opens
    /// `movsd %xmm1, 0x8(%rsp)` — the declared parameter is saved from XMM1.
    /// The pairing is the rule for free functions TOO (`int f(int a, double b)`
    /// = RCX + XMM1, and a float consumes no integer slot), so this flag does
    /// not pick a seeding rule — it says `this` is declared parameter 0 and
    /// therefore spends pair 0 in both classes.
    pub member: bool,
}

impl FnAbi {
    /// "No declared signature": the four integer registers as parameters, an
    /// integer return. Identical to the model every C-corpus gate measures.
    pub fn integer() -> FnAbi {
        FnAbi {
            args: Vec::new(),
            ret: RetKind::Int,
            member: false,
        }
    }
    /// True when seeding this ABI produced exactly the integer model.
    pub fn is_integer(&self) -> bool {
        self.args.is_empty() && self.ret == RetKind::Int
    }
    /// The register PAIRS this shape's parameters occupy, in DECLARED order,
    /// with the kinds that live in them — the single statement of the pair rule,
    /// read by the seed (`new_abi`, which creates the parameters) and by every
    /// CALL SITE (which has to read the same registers to pass the arguments).
    /// A pair position with no register (Windows x64 has four pairs) is dropped
    /// here and counted by the seed, never invented.
    ///
    /// The reason one function serves both jobs is the bug it prevents: a caller
    /// that computed "which registers" its own way would pass `b` in RDX to a
    /// callee that reads it from XMM1, and nothing downstream could see it.
    pub fn seeded_pairs(&self) -> Vec<(Grp, ArgKind)> {
        if self.args.is_empty() {
            // The integer model: the four integer registers, four pairs.
            return ARG_REGS.iter().map(|g| (*g, ArgKind::Int)).collect();
        }
        self.args
            .iter()
            .enumerate()
            .filter_map(|(pos, &k)| {
                let g = match k {
                    ArgKind::Int => ARG_REGS.get(pos).copied(),
                    ArgKind::F32 | ArgKind::F64 => ARG_XMM.get(pos).copied(),
                };
                g.map(|g| (g, k))
            })
            .collect()
    }
    /// The C-declared type of one parameter of this shape, as `new_abi` records
    /// it in `Function::param_tys` (a float parameter's IR value is the register
    /// cell; its DECLARED type is what the C spells).
    pub fn declared_ty(k: ArgKind) -> Ty {
        match k {
            ArgKind::Int => Ty::u64(),
            ArgKind::F32 => Ty::f32(),
            ArgKind::F64 => Ty::f64(),
        }
    }
}

/// What the front end DECLARES about one function: the shape, and whether it is
/// a shape at all. `typed: false` is the loud case — a signature the demangler
/// could not fully type, for which the subject keeps the integer model. A call
/// site reads this and refuses to pretend: an untypable callee gets no defined
/// return value, so the caller that uses its answer is refused, exactly as it
/// was before calls were modelled.
#[derive(Clone, Debug, Default)]
pub struct FnShape {
    pub abi: FnAbi,
    pub typed: bool,
}

/// What a call site knows about one callee: its identity in the generated
/// translation unit, and its declared shape.
#[derive(Clone, Debug, Default)]
pub struct CalleeSig {
    /// The C name the pipeline gives this function - the identity `Op::Call`
    /// records and the emitter declares. Not the COFF name: one C file holds
    /// every function of every object, and `o7_f` is who `f` is there.
    pub cname: String,
    pub abi: FnAbi,
    /// The declared shape is a shape (see `FnShape`).
    pub typed: bool,
    /// The callee NEVER RETURNS: declared by the front end, from a `noreturn`
    /// declaration in the source it read, from `lift::noreturn_symbol` for the
    /// CRT/C++ termination family, or not at all. Nothing in a `call`'s encoding
    /// says the callee stays away, so this is a claim the front end owns, and the
    /// lifter models the noreturn shape ONLY when it is set. Marking a returning
    /// callee noreturn deletes its caller's answer; leaving a noreturn callee
    /// undeclared leaves the caller's `Exit`, which is loud.
    pub noreturn: bool,
}

/// The termination family: symbols whose calls do not return. A NAME TABLE, kept
/// in one place and auditable, because `noreturn` is a declaration and the table
/// is the front end's declaration for the C runtime. MSVC decorates (`_abort`,
/// `_exit`) and decorates C++ (`?terminate@std@@YAXXZ`), so the check strips a
/// leading `_` and also accepts a decorated name containing one of the stems.
///
/// Deliberately NOT in here: `_alloca`/`__chkstk` (they return), `_purecall`
/// (it calls the terminator; its own contract is not `noreturn`), and
/// `_longjmp`/`setjmp` (control leaves, but not through the call's own return -
/// the model has no longjmp, and a block that ends in one is a loud `Exit`).
pub fn noreturn_symbol(name: &str) -> bool {
    const PLAIN: &[&str] = &[
        "abort",
        "exit",
        "_exit",
        "_Exit",
        "quick_exit",
        "_c_exit",
        "_cexit",
        "__fastfail",
        "_fastfail",
        "_CxxThrowException",
        "__CxxThrowException",
        "__crt_debugger_hook",
        "__report_gsfailure",
        "__report_rangecheckfailure",
        "_invalid_parameter_noinfo",
        "_invalid_parameter_noinfo_noreturn",
        "terminate",
    ];
    const STEMS: &[&str] = &[
        "?terminate@std@@",
        "?_Xbad_alloc@std@@",
        "?_Xbad_function_call@std@@",
        "?_Xlength_error@std@@",
        "?_Xout_of_range@std@@",
        "?_Xrange@std@@",
        "?_Xruntime_error@std@@",
    ];
    if STEMS.iter().any(|s| name.contains(s)) {
        return true;
    }
    let bare = name.strip_prefix('_').unwrap_or(name);
    PLAIN.contains(&bare) || PLAIN.contains(&name)
}

/// The callees of one object, keyed by the COFF symbol the discovery plan found
/// at the call target. Keyed by the COFF name because that is what the lifter
/// can look up from an address; the value carries the name the C uses.
pub type CalleeTable = HashMap<String, CalleeSig>;

/// An empty table: no call target is typable, which is the model as it was
/// before direct calls were modelled at all.
pub fn no_callees() -> CalleeTable {
    CalleeTable::new()
}

/// What the COFF relocation attached to a displacement field says: the symbol
/// that field names, and the address of that symbol in the object's flat
/// per-object space when the object DEFINES it (`Some`). An external symbol -
/// one no COFF symbol in this object resolves - carries its name and `None`:
/// the refusal can name it, and the pipeline cannot pretend it linked.
///
/// This is the evidence a `call`'s immediate does not carry: clang's COFF
/// objects leave the unrelocated next-IP in the displacement and put the
/// target in the relocation at its address (`callq 0x15` at 0x10 + the
/// REL32 relocation at 0x11 naming `h` at 0x50). Reading only the immediate
/// is why `calls: 0 modelled`.
#[derive(Clone, Debug)]
pub struct RelocTarget {
    pub sym: String,
    pub addr: Option<u64>,
}

/// Relocations keyed by the address of the displacement field they apply to,
/// in the same per-object virtual space as `names` - keyed at the field's FIRST
/// byte. A direct `call`/`jmp` displacement really is the instruction's last
/// field, so those resolve by arithmetic (`next_ip - disp_len`). A RIP-relative
/// displacement is NOT: `cmpl $0x2, (%rip)` puts an `imm8` after the disp32, so
/// those resolve by scanning the instruction's span - see `rip_reloc`.
pub type RelocMap = HashMap<u64, RelocTarget>;

/// An empty map: no displacement has a relocation, which is the model exactly
/// as it was before the relocation table was read as an evidence source.
pub fn no_relocs() -> RelocMap {
    RelocMap::new()
}

/// An empty relocation map: no call target and no RIP-relative address has a
/// named symbol, which is the model as it was.
static EMPTY_RELOCS: std::sync::LazyLock<RelocMap> = std::sync::LazyLock::new(RelocMap::new);

/// No jump table resolved for any site: every indirect jump keeps the loud
/// `Op::Opaque("indirect jmp")` refusal it has always had.
pub fn no_tables() -> Tables {
    Tables::new()
}

static EMPTY_TABLES: std::sync::LazyLock<Tables> = std::sync::LazyLock::new(Tables::new);

/// Caller-saved: a call may overwrite these. The Windows x64 ABI says the
/// WHOLE XMM register is caller-saved, and the pair model has two cells to
/// destroy per register — a high lane left alive across a call is the same
/// class of bug as an int in ECX across one. The CLOBBER COUNT, though, stays
/// at the ABI's register count (9 GPRs + 16 XMMs, minus the defined output):
/// `calls.rs` pins "RAX is defined BY THE CALL, so it is not among the
/// clobbers: 25 caller-saved registers minus the one the call defines", and
/// the clobber is a STATEMENT ABOUT A REGISTER — a cell-level pair is two
/// IR ops for the same statement, and counting the statement twice would
/// move a floor this slice must not move.
static CALLER_SAVED: &[&[Grp]] = &[
    &[
        Grp::Rax,
        Grp::Rcx,
        Grp::Rdx,
        Grp::R8,
        Grp::R9,
        Grp::R10,
        Grp::R11,
        Grp::Rsi,
        Grp::Rdi,
    ],
    &[
        Grp::Xmm(0),
        Grp::XmmHi(0),
        Grp::Xmm(1),
        Grp::XmmHi(1),
        Grp::Xmm(2),
        Grp::XmmHi(2),
        Grp::Xmm(3),
        Grp::XmmHi(3),
        Grp::Xmm(4),
        Grp::XmmHi(4),
        Grp::Xmm(5),
        Grp::XmmHi(5),
        Grp::Xmm(6),
        Grp::XmmHi(6),
        Grp::Xmm(7),
        Grp::XmmHi(7),
        Grp::Xmm(8),
        Grp::XmmHi(8),
        Grp::Xmm(9),
        Grp::XmmHi(9),
        Grp::Xmm(10),
        Grp::XmmHi(10),
        Grp::Xmm(11),
        Grp::XmmHi(11),
        Grp::Xmm(12),
        Grp::XmmHi(12),
        Grp::Xmm(13),
        Grp::XmmHi(13),
        Grp::Xmm(14),
        Grp::XmmHi(14),
        Grp::Xmm(15),
        Grp::XmmHi(15),
    ],
];

/// The condition a `jcc`/`setcc`/`cmovcc` tests, expressed over flag bits.
#[derive(Copy, Clone)]
enum Cond {
    E,
    Ne,
    S,
    Ns,
    L,
    Ge,
    G,
    Le,
    A,
    Ae,
    B,
    Be,
    O,
    No,
    P,
    Np,
}

/// Map a condition-code suffix to the flag expression. iced prints the canonical
/// name of each mnemonic, and its aliases (SETAE/SETNB, SETNA/SETBE, ...) all
/// spell the same test, so both spellings are listed. Names, not enum variants,
/// because the alias set is what the disassembly of real code contains.
fn cond_suffix(s: &str) -> Option<Cond> {
    Some(match s {
        "e" | "z" => Cond::E,
        "ne" | "nz" | "nz?" => Cond::Ne,
        "s" => Cond::S,
        "ns" => Cond::Ns,
        "l" | "nge" | "nge?" => Cond::L,
        "ge" | "nl" | "nl?" => Cond::Ge,
        "g" | "nle" | "nle?" => Cond::G,
        "le" | "ng" | "ng?" => Cond::Le,
        "a" | "nbe" | "nbe?" => Cond::A,
        "ae" | "nb" | "nb?" | "nbelow" => Cond::Ae,
        "b" | "nae" | "nae?" | "below" => Cond::B,
        "be" | "na" | "na?" => Cond::Be,
        // Overflow and parity are different bits: an earlier map folded "pe"
        // (parity even) into the overflow condition, which is a silent
        // miscompile of `jo`/`jp`, caught here by the dead-code lint.
        "o" => Cond::O,
        "no" => Cond::No,
        "p" | "pe" => Cond::P,
        "np" | "po" => Cond::Np,
        _ => return None,
    })
}

/// The condition a conditional-transfer mnemonic tests.
fn cond_of(n: &str) -> Option<Cond> {
    let s = n
        .strip_prefix("cmov")
        .or_else(|| n.strip_prefix("set"))
        .or_else(|| n.strip_prefix('j'))?;
    cond_suffix(s)
}

/// What the last instruction of a block said about control flow. A block's
/// fallthrough is a property of how the block ENDS - a lesson already paid for
/// once in `discover`.
enum Pending {
    Fall,
    Jump(u64),
    Jcc(u64, Cond),
    Ret,
    /// A TAIL CALL answered: the block's last instruction is the `Op::Call` the
    /// `jmp` became, and the block ends by leaving the function with THAT call's
    /// value (`Terminator::TailReturn`). It carries the call's own value, not the
    /// register read `Pending::Ret` performs: the verifier's rule is that a
    /// `TailReturn` is defined by a call in the same block, and the bitcast view
    /// that rule rejects is the one that exists to model a partial write of RAX -
    /// a tail call writes no register the caller can read, so there is nothing to
    /// view. The bits are the same cell.
    TailCall(VId),
    /// A call to a callee that never returns ended the block: no value, no
    /// successor (`Terminator::NoReturn`).
    NoReturn,
    /// An indirect jump/call, or a terminator we do not model.
    OpaqueTerm,
    /// A terminator built on the spot: the switch over a jump table's resolved
    /// targets. `close` installs it as it installs any other, so a block's
    /// terminator is still set in exactly one place.
    Switch(Terminator),
}

/// Which instruction made this call site: a `call`, which returns, or a `jmp`
/// whose relocation named the callee, which does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Site {
    Call,
    Tail,
}

/// A block: its start, and the (address, length) of each instruction in it.
pub struct BlockInsns {
    pub start: u64,
    pub insns: Vec<(u64, usize)>,
}

/// Split a proven instruction stream into blocks at the proven block starts.
/// The entry stack pointer, in the flat address space both oracles model:
/// `rdec_core::emit_c::MEM_SIZE` is 1 MiB, so this sits mid-range with 512 KiB of
/// room below for frames and pushes, and the address mask keeps it in range.
pub const STACK_BASE: u64 = 1 << 19;

pub fn split_blocks(plan: &Plan, start: u64, end: u64) -> Vec<BlockInsns> {
    // Block starts are the union of what discovery recorded and every branch
    // destination in range. The second set is what makes loops liftable: a
    // backward `jmp` to the middle of an already-decoded block claims no bytes,
    // so discovery cannot record a block there - but the header must be a block
    // of its own, or the back edge has nowhere to land, the header is not a join,
    // and the loop-carried value never gets a phi.
    let mut starts: Vec<u64> = plan
        .blocks
        .iter()
        .chain(plan.targets.iter())
        .copied()
        .filter(|a| *a >= start && *a < end)
        .collect();
    starts.push(start);
    starts.sort_unstable();
    starts.dedup();
    // A destination that is not an instruction boundary (it points into the middle
    // of one, which the decoder says cannot happen for a real jump) would produce
    // an empty block and a split that is not a split. Drop it, say nothing:
    // `objdump-check` is the gate on instruction boundaries.
    starts.retain(|a| *a == start || plan.insns.iter().any(|(ia, _)| ia == a));
    starts
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let stop = starts.get(i + 1).copied().unwrap_or(end);
            BlockInsns {
                start: *s,
                insns: plan
                    .insns
                    .iter()
                    .filter(|(a, _)| *a >= *s && *a < stop)
                    .map(|(a, l)| (*a, *l))
                    .collect(),
            }
        })
        .collect()
}

/// Evidence about one lifted function, so the gate counts real things.
#[derive(Default, Clone, Debug)]
pub struct Report {
    /// Loud gaps: `Op::Opaque` for an unmodelled instruction or operand.
    pub gaps: usize,
    /// Registers clobbered by calls - modelled, not gaps.
    pub clobbers: usize,
    /// Terminators we had to close with `Exit`.
    pub exits: usize,
    /// Calls whose target we could not name from the symbol table.
    pub unnamed_calls: usize,
    /// Direct calls to a callee the pipeline named AND typed: modelled as a call
    /// that defines its own answer, so the caller's use of that answer emits C.
    /// Counted at the call site, which is the only place the evidence is.
    pub calls_modelled: usize,
    /// `call *%rax`: no target, so no signature, so no answer.
    pub calls_indirect: usize,
    /// A direct call to a target with no COFF symbol: an address, not an identity.
    pub calls_no_symbol: usize,
    /// A direct call with more register arguments than the model has pairs
    /// (Windows x64 has four): the stack-argument case, named not dropped.
    pub calls_over_args: usize,
    /// A direct call to a callee whose declared shape the front end could not
    /// type: the identity is known, the signature is not, so the answer is not.
    pub calls_untypable: usize,
    /// Direct `call`s whose displacement field carries a REL32 relocation that
    /// resolves to a symbol DEFINED in this object: the target is the symbol's
    /// address, which is what lets the `names`/`sigs` lookup model the call.
    /// Counted at the call site, next to `calls_modelled`, so the gate can show
    /// the relocation evidence and the modelling evidence as two numbers.
    pub calls_reloc_resolved: usize,
    /// Direct `call`s whose displacement carries a relocation naming a symbol
    /// this object does not define (external, undefined): the name is known,
    /// the address is not, so the call keeps its loud shape.
    pub calls_reloc_external: usize,
    /// Direct `call`s whose displacement field has NO relocation. For an
    /// intra-section call clang links the target into the immediate, so the
    /// immediate IS the evidence; counted so the number of calls resolved from
    /// the table is auditable against the number of calls in the object.
    pub calls_reloc_absent: usize,
    /// RIP-relative operands (`lea`, `mov`, arithmetic, scalar SSE loads) whose
    /// displacement relocation names a symbol the object defines: the address
    /// became a named constant, so the `lea of an address we cannot name`
    /// class is cut by this number, counted at the operand.
    pub rip_named: usize,
    /// Blocks the entry cannot reach; their lifted contents dropped, counted.
    pub unreachable_blocks: usize,
    /// Live-in placeholders SSA construction resolved and deleted.
    pub scaffolding_removed: usize,
    /// Live-ins whose incoming value is `undef`: the register is read at a point
    /// where no modelled path defines it. At a function entry that is the ABI
    /// question (is this register an argument?), not a dataflow question.
    pub undef_live_ins: usize,
    /// Functions whose answer provably reaches `RAX` at 32-bit width, so bits
    /// 63:32 are zero on every path and the emitted C returns `uint32_t`. This
    /// is a machine-code fact (a 32-bit register write zeroes the upper half),
    /// counted so the width modelling can be watched, not assumed.
    pub ret_w32: usize,
    /// Register self-idioms translated as the machine-code facts they are:
    /// `xor r,r`/`sub r,r` write 0 at the operation width without reading `r`,
    /// `and r,r`/`or r,r` write `r`'s own value. Counted because the number is
    /// what makes `RAX` defined on paths that used to read an undefined
    /// register, and a reader must be able to ask how many paths moved.
    pub self_idioms: usize,
    /// SSE mnemonics translated, by name, counted at the instruction whether the
    /// op was modelled or refused. It is in the Report because the gate prints
    /// it next to the `self_idioms` count for the same reason: the number is
    /// checkable against `llvm-objdump -d` over the same address ranges, so a
    /// claim about SSE coverage is measured, not trusted.
    pub sse: std::collections::BTreeMap<String, u32>,
    /// `pxor/xorps/xorpd xmm,xmm` translated as the machine fact it is: all 128
    /// bits become 0 and the register is not read. Counted apart from
    /// `self_idioms` so the GPR number stays comparable to the history.
    pub sse_zeroing: usize,
    /// Packed data movement and lane-independent bitwise modelled by the pair
    /// (a 16-byte load, a 16-byte store, a full-register packed move, a
    /// `pxor`/`pand`/`pandn`/`por` on two registers): one per instruction,
    /// counted where the modelling happens, checkable against
    /// `llvm-objdump -d` over the same ranges.
    pub sse_packed: usize,
    /// Functions whose `ret` reaches it with XMM0 defined and RAX not: a
    /// float/double answer, in the register the ABI says and the model does not
    /// follow. T2 compares RAX only, so these are outside its claim; the C++
    /// gate prints the number so the shape is visible rather than inferred.
    pub ret_xmm0: usize,
    /// Parameters seeded from a DECLARED float kind (XMM0-3). Counted at the
    /// seed, so the number is "how many float parameters the declared
    /// signatures named", checkable against the demangled texts, never against
    /// what the code does with them.
    pub float_params: usize,
    /// Functions whose `ret` answered from XMM0 at F32/F64 width, per the
    /// declared return kind. The emitted C declares `float`/`double`; T2's
    /// XMM0 class compares the bit pattern at exactly this width.
    pub ret_f32: usize,
    pub ret_f64: usize,
    /// Where a register read resolved to `undef`: the address of the read and
    /// which register. `undef_live_ins` counts them; this names them, because
    /// "13 undef live-ins" is not a work item and "`0x29` reads RDX, which no
    /// modelled path writes" is. Same rule as the self-idiom count: a gap a
    /// reader cannot check is a gap that gets quietly believed.
    pub undef_sites: Vec<(u64, Grp)>,
    /// Computed jumps modelled as control flow: one per dispatch whose table
    /// resolved. Counted at the `jmp *`, where the evidence is.
    pub jump_tables: usize,
    /// Real successor edges those tables added: one per table slot that names a
    /// block of this function. `jump_table_edges` is the number of case bodies
    /// the CFG can now walk into, which is the claim M1's coverage makes.
    pub jump_table_edges: usize,
    /// Dispatches whose table did NOT fully resolve: every resolved slot is
    /// discovered and counted, and the dispatch keeps its `indirect jmp` refusal,
    /// because the alternative is a switch over half the case bodies.
    pub jump_table_partial: usize,
    /// Table slots whose target is not a block of this function: a counted gap,
    /// because an edge that goes nowhere is worse than a refusal that says so.
    pub jump_table_missing: usize,
    /// TAIL CALLS modelled: a `jmp` whose relocation names a defined function
    /// symbol, lifted as `return callee(args);` - a `Call` that defines the
    /// answer and a `Terminator::TailReturn` that leaves with it. Counted at the
    /// `jmp`, where the evidence is, beside the call census it belongs to.
    pub tail_calls: usize,
    /// A `jmp` that looks like a tail call (a relocation names a symbol) but has
    /// no callee the pipeline can call: the symbol is not defined here, or the
    /// front end declared no shape for it. Counted, and the block keeps its
    /// `Exit` - the verifier refuses the same shape in the IR, so a silent one
    /// must not be creatable here either.
    pub tail_calls_unnamed: usize,
    /// Calls to a callee the front end DECLARED noreturn (`CalleeSig::noreturn`).
    /// The call defines nothing, the caller-saved registers are destroyed, and
    /// the block ends in `Terminator::NoReturn`.
    pub noreturn_calls: usize,
    /// Those calls BY NAME: the site address and the callee's COFF symbol. A
    /// function whose IR ends in a noreturn exit has no fall-through answer, so
    /// the list is the work item; "2 noreturn calls" is not.
    pub noreturn_sites: Vec<(u64, String)>,
    /// Instructions the lifter did NOT lift, because the block that holds them
    /// already ended: after a tail call, after a call the front end declared
    /// noreturn, and after a call whose callee the object does not define. The
    /// bytes are inside the function's range — `nr.O0.obj` leaves `int3` + `nop`
    /// after `callq abort`, and clang leaves `nopw` after a tail `jmp` — so they
    /// are real bytes the object owns, and the count is the audit that the model
    /// said "these are not this function's code" out loud. Lifting them is the
    /// failure mode: `mov eax,0x190; ret` after `call abort` would give the
    /// function an answer the machine code never produces.
    pub bytes_past_ended_block: usize,
}

/// `op reg, reg` where both operands name the SAME register group. Each variant
/// is a fact about what the hardware computes, stated where the instruction is
/// translated - and that location is the whole point.
///
/// As an IR value identity (`x ^ x -> 0` in `const_fold`) the same rule is a
/// lie about meaning: the interpreter is strict, so an `xor(undef, undef)` read
/// FAULTS when it is reached, and deleting the read turns a faulting execution
/// into a returning one - which translation validation correctly reports as a
/// change in meaning. Deleting an unused `getflag` read already bit this project
/// exactly once for the same reason. The hardware does not read the register at
/// all here: the operand is an address whose value is never consumed, so the
/// lifter must not emit a read of it. That is a property of the encoding, and
/// the encoding is only visible in the lifter.
enum SelfIdiom {
    /// `xor r,r`: the result is 0, the register is not read.
    ZeroXor,
    /// `sub r,r`: the result is 0 (and no borrow), the register is not read.
    ZeroSub,
    /// `and r,r`: the result is the register's own value - a real read.
    KeepAnd,
    /// `or r,r`: the result is the register's own value - a real read.
    KeepOr,
}

impl SelfIdiom {
    /// Classify one instruction, or say it is not a self-idiom.
    ///
    /// Deliberately absent: `adc`/`sbb` (`sbb r,r` is `-CF`, so 0 only when the
    /// carry flag is 0 - a value identity that is not true, and the general
    /// path's `adc/sbb: carry chain not modelled` gap stays as the loud answer),
    /// `add r,r` (the result is `2*r`, not a constant), and `cmp`/`test`, which
    /// write no destination at all.
    fn of(m: Mnemonic, ins: &Instruction) -> Option<SelfIdiom> {
        if ins.op_count() != 2
            || ins.op_kind(0) != OpKind::Register
            || ins.op_kind(1) != OpKind::Register
        {
            return None;
        }
        if split(ins.op_register(0)) != split(ins.op_register(1)) {
            return None;
        }
        match m {
            Mnemonic::Xor => Some(SelfIdiom::ZeroXor),
            Mnemonic::Sub => Some(SelfIdiom::ZeroSub),
            Mnemonic::And => Some(SelfIdiom::KeepAnd),
            Mnemonic::Or => Some(SelfIdiom::KeepOr),
            _ => None,
        }
    }
}

pub struct Lifter<'a> {
    f: Function,
    regs: HashMap<Grp, VId>,
    flags: Option<VId>,
    /// The block `flags` was DEFINED in. SSA has no flag phis (SS1.2 item 2), so
    /// a flags value may only be used where its defining block dominates it, and
    /// block identity is the dominance fact the lifter can check at emit time.
    /// Measured: carrying a flags value across a join made an `inc`'s carried CF
    /// a use that its block did not dominate — the MSVC leg's `lift` gate caught
    /// it as `%63 defined in bb5 does not dominate use in bb8`.
    flags_block: BbId,
    bb_of: HashMap<u64, BbId>,
    order: Vec<u64>,
    cur: BbId,
    /// Address of the instruction being lifted. Every IR node records it, so a
    /// verifier complaint can be traced back to a byte offset - with the block
    /// address only, half a dozen values share `; 0x25` and a width complaint is
    /// unattributable.
    at: u64,
    pending: Pending,
    /// This block's own terminator has ALREADY ended it — a tail call, a call the
    /// front end declared noreturn, a call to a symbol the object does not define.
    /// The bytes the block table puts after that instruction are inside the
    /// function's range (clang leaves `int3` + padding, a tail `jmp` leaves `nopw`)
    /// and are not this function's code: the lifting loop counts them
    /// (`Report::bytes_past_ended_block`) and does not lift them. Reset per block.
    ended_block: bool,
    /// The declared shape this function was lifted under: which parameter
    /// arrives in which register class, and what `ret` answers from. Seeded
    /// from the caller's evidence (a demangled signature), never inferred from
    /// the code's behaviour.
    abi: FnAbi,
    /// Registers `Lifter::new` seeded as entry definitions (the four ARG_REGS
    /// of an empty ABI, the per-class registers of a declared one, plus RSP).
    /// The entry block marks these in `cur_defs`: a register the transfer
    /// function does not see defined at the entry is read as `undef` by every
    /// later block, which is the silent kind of wrong.
    seeded: Vec<Grp>,
    /// (block, register, placeholder) for every register a block READ before
    /// writing. These are the live-ins that become phis in `ssa_registers`.
    live_in: Vec<(BbId, Grp, VId)>,
    /// The register map at the end of each block, and which registers the block
    /// wrote. Together they are the transfer function the SSA pass solves.
    block_out: HashMap<BbId, HashMap<Grp, VId>>,
    block_def: HashMap<BbId, HashSet<Grp>>,
    cur_defs: HashSet<Grp>,
    /// One `undef` value, defined in the entry block, used for every
    /// "no path reaches this edge" slot. Defining it once in the entry block is
    /// the only placement that dominates every use: a phi's incoming value for
    /// an edge from an unreachable block cannot be defined in that block, and a
    /// non-phi instruction before a phi is a structural error.
    undef: Option<VId>,
    pub rep: Report,
    names: &'a HashMap<u64, String>,
    /// The callees of this object, keyed by COFF symbol: what a `call` target
    /// resolves to, and what its declared shape says to pass. An empty table is
    /// "no target is typable", which is the model exactly as it was.
    sigs: &'a CalleeTable,
    /// The object's relocations, keyed by the address of the displacement
    /// field they name. Empty is the model as it was: no call target and no
    /// RIP-relative address is resolved from the table.
    relocs: &'a RelocMap,
    /// The object's resolved jump tables, keyed by dispatch site. Empty is the
    /// model as it was: `jmp *%rax` refuses loudly.
    tables: &'a Tables,
    /// The instructions of a dispatch the scan matched, by address: what to skip,
    /// and what to capture on the way to the switch over the table's resolved
    /// targets.
    skip: HashMap<u64, Step>,
    /// The table index, captured at the slot read: the load overwrites the index
    /// register, so the selector has to be read before it goes.
    table_index: Option<VId>,
    fmt: Box<dyn iced_x86::Formatter>,
}

/// One instruction of a matched jump-table dispatch, by address.
#[derive(Clone, Debug)]
enum Step {
    /// `lea (%rip),%base`: names the table. The model branches on the table's
    /// SLOT, so no code address is built and none is needed.
    Lea,
    /// `movslq (%base,%idx,4),%tmp`: the slot read. `.rdata`'s bytes are not in
    /// the flat memory the model runs on, so this load is not a value - it is the
    /// evidence for the edge, and the index is the selector.
    Load(Grp),
    /// `add %tmp,%base`: the target's address arithmetic, subsumed by the edges.
    Add,
    /// `jmp *%base`: the dispatch, replaced by the switch over these targets.
    Site(Vec<u64>),
}

/// An empty callee table: the lifter sees no typable call target.
static EMPTY_SIGS: std::sync::LazyLock<CalleeTable> = std::sync::LazyLock::new(CalleeTable::new);

impl<'a> Lifter<'a> {
    /// Hand this lifter the object's resolved jump tables. Without them every
    /// indirect jump keeps its `Op::Opaque("indirect jmp")` refusal; with them a
    /// dispatch whose table resolved becomes a `Terminator::Switch` whose
    /// successors are the case bodies.
    pub fn with_tables(mut self, tables: &'a Tables) -> Self {
        self.tables = tables;
        self
    }

    /// Lift under the integer model: RCX/RDX/R8/R9 seeded as the four
    /// parameters, `ret` answers from RAX. Identical in behaviour to every
    /// gate that ran before the float ABI existed - `FnAbi::integer()` seeds
    /// exactly what this constructor always seeded.
    pub fn new(name: &str, addr: u64, names: &'a HashMap<u64, String>) -> Lifter<'a> {
        Lifter::new_abi(name, addr, names, &FnAbi::integer())
    }

    /// Lift under a DECLARED shape, with no callee table: every call this
    /// function makes stays a loud call. This is the constructor every gate and
    /// test used before direct calls were modelled, and it seeds exactly what it
    /// always seeded.
    pub fn new_abi(
        name: &str,
        addr: u64,
        names: &'a HashMap<u64, String>,
        abi: &FnAbi,
    ) -> Lifter<'a> {
        Lifter::new_sig(name, addr, names, abi, &EMPTY_SIGS, &EMPTY_RELOCS)
    }

    /// Lift with the declared shapes of this object's CALLEES available, so a
    /// direct call to a named, typed sibling becomes a call that DEFINES its
    /// answer instead of a call plus a pile of clobbers.
    ///
    /// `abi` is this function's own declared shape: `abi.args` is its parameter
    /// list in declaration order, and parameter `pos` is seeded from the register
    /// PAIR its position owns: `ARG_REGS[pos]` for an integer kind,
    /// `ARG_XMM[pos]` for a float kind. That is the Windows x64 rule as measured
    /// — a float parameter consumes no integer register and an integer parameter
    /// no XMM, so the index is the declared POSITION, not a per-class count. A
    /// member needs no special case: `this` is `args[0]` and spends pair 0 in
    /// BOTH classes, which is why its first `double` is XMM1 and a free
    /// `int f(int a, double b)`'s is XMM1 too. A parameter past pair 3 has no
    /// register: a loud, counted gap, never an invented value.
    /// An empty `abi.args` is the integer model: the four integer registers.
    pub fn new_sig(
        name: &str,
        addr: u64,
        names: &'a HashMap<u64, String>,
        abi: &FnAbi,
        sigs: &'a CalleeTable,
        relocs: &'a RelocMap,
    ) -> Lifter<'a> {
        let mut f = Function::new(name, addr);
        f.ret_ty = match abi.ret {
            RetKind::Int => Ty::i64(),
            RetKind::F32 => Ty::f32(),
            RetKind::F64 => Ty::f64(),
        };
        let entry = f.entry;
        let mut regs = HashMap::new();
        let mut seeded: Vec<Grp> = Vec::new();
        let mut float_params = 0usize;
        // One parameter per declared kind, seeded at its DECLARED POSITION by the
        // pair rule in `FnAbi::seeded_pairs` — the SAME rule every call site
        // reads, so a caller can never hand an argument in a register this seed
        // did not create (that drift is invisible everywhere downstream).
        // `int f(int a, double b)` = RCX + XMM1 and
        // `int g(double a, int b, double c)` = XMM0 + RDX + XMM2, and a member's
        // `this` at pair 0 spends both classes, which is why its first `double`
        // is XMM1 — the bytes picked this, and `tests/float_abi.rs` pins it.
        //
        // A float parameter's IR value is the XMM cell - one unsigned W64 word,
        // the register file's rule, signedness included (a register cell is
        // unsigned; signedness is a property of a *read*, and the emitter spells
        // both) - and its DECLARED type (`param_tys`) is what the C signature
        // spells; the emitter moves bits through the memcpy helpers.
        // An empty `abi.args` is the integer model: the same loop, four Ints.
        let pairs = abi.seeded_pairs();
        let over_budget = abi.args.len().saturating_sub(pairs.len());
        for (g, k) in pairs {
            if k != ArgKind::Int {
                float_params += 1;
            }
            let v = f.new_value(Ty::u64());
            f.params.push(v);
            f.param_tys.push(FnAbi::declared_ty(k));
            f.block_mut(f.entry)
                .push(Instr::new(v, Op::Param, Ty::u64(), addr));
            regs.insert(g, v);
            seeded.push(g);
        }
        if over_budget > 0 {
            // Loud, counted, and visible in the IR: a parameter that
            // founds no register is an `Opaque`, never a guess.
            f.emit_void(
                entry,
                Op::Opaque(
                    format!("{over_budget} declared parameters beyond the register-class budget")
                        .into(),
                ),
                addr,
            );
        }
        // The machine stack is memory. In a flat-memory model the entry stack
        // pointer is a fixed address, which makes every rsp-relative access a real
        // load/store at the SAME place in both oracles - the IR interpreter's
        // sparse memory and the emitted C's byte array - so the differential can
        // compare them at all. Modelling it as `Opaque` was defensible for the
        // interpreter and fatal for emission: it refused every -O0 function,
        // because every -O0 function has a stack frame.
        //
        // The value is not a guess about the program: no corpus function's
        // behaviour depends on the absolute address, only on offsets from it, and
        // `stack2reg` (M6) is the pass that turns these offsets into locals.
        let rsp = f.emit(f.entry, Op::Const(STACK_BASE), Ty::u64(), addr);
        regs.insert(Grp::Rsp, rsp);
        seeded.push(Grp::Rsp);
        let mut l = Lifter {
            f,
            regs,
            flags: None,
            flags_block: 0,
            bb_of: HashMap::new(),
            order: Vec::new(),
            cur: entry,
            at: addr,
            pending: Pending::Fall,
            ended_block: false,
            abi: abi.clone(),
            seeded,
            live_in: Vec::new(),
            block_out: HashMap::new(),
            block_def: HashMap::new(),
            cur_defs: HashSet::new(),
            undef: None,
            rep: Report::default(),
            names,
            sigs,
            relocs,
            tables: &EMPTY_TABLES,
            skip: HashMap::new(),
            table_index: None,
            fmt: crate::decode::formatter(crate::decode::Syntax::Intel),
        };
        l.rep.float_params = float_params;
        l.rep.gaps += over_budget;
        l
    }

    /// The entry register map, resolved to IR values where the seed defines
    /// one. `None` means the register is NOT an entry definition under this
    /// ABI: an integer parameter does not make RDX a parameter, and a float
    /// parameter does not make XMM0's neighbours parameters. This is the
    /// window through which a caller can check the per-class seeding - the
    /// claim "a float argument consumes no integer slot" is assertable, not
    /// narrated.
    pub fn entry_regs(&self) -> Vec<(Grp, Option<VId>)> {
        ARG_REGS
            .iter()
            .chain(ARG_XMM.iter())
            .map(|g| (*g, self.regs.get(g).copied()))
            .collect()
    }

    /// The mnemonic, lowercased, from iced's formatter: iced's `Mnemonic` has no
    /// `Display` without its `formatter` feature, and the canonical names are
    /// what the alias families (`setae`/`setnb`, `cmova`/`cmovnbe`) spell out.
    fn name_of(&mut self, ins: &Instruction) -> String {
        let mut s = String::new();
        self.fmt.format(ins, &mut s);
        s.split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
    }

    fn addr(&self) -> u64 {
        self.at
    }

    /// The id of the shared `undef` value. The value is allocated here and its
    /// instruction is placed only by `place_undef`, at the end of the pass: an
    /// `undef` emitted eagerly lands after the instructions that use it, and the
    /// verifier's program-order rule is right to reject that.
    fn undef_val(&mut self) -> VId {
        if let Some(v) = self.undef {
            return v;
        }
        let v = self.f.new_value(Ty::u64());
        self.undef = Some(v);
        v
    }

    /// Put the shared `undef` in the entry block, after its phi prefix: entry
    /// dominates every reachable use, the phi-prefix rule forbids putting it
    /// first, and no reachable use can precede it.
    fn place_undef(&mut self) {
        let Some(v) = self.undef else { return };
        if self.f.def(v).is_some() {
            return;
        }
        let entry = self.f.entry;
        let a = self.f.block(entry).addr;
        let pos = self
            .f
            .block(entry)
            .instrs
            .iter()
            .take_while(|i| matches!(i.op, Op::Phi(_)))
            .count();
        self.f
            .block_mut(entry)
            .instrs
            .insert(pos, Instr::new(v, Op::Undef, Ty::u64(), a));
    }

    fn gap(&mut self, why: &str, ty: Ty) -> VId {
        self.rep.gaps += 1;
        self.f.emit(
            self.cur,
            Op::Opaque(why.to_string().into()),
            ty,
            self.addr(),
        )
    }

    fn read64(&mut self, g: Grp) -> VId {
        if let Some(v) = self.regs.get(&g) {
            return *v;
        }
        // A register read before being written in this block is a LIVE-IN, not
        // undefined: the value comes in along a CFG edge. `ssa_registers`
        // replaces the placeholder with the incoming value, or a phi at a join.
        let v = self.f.emit(
            self.cur,
            Op::Opaque(format!("live-in {g:?}").into()),
            Ty::u64(),
            self.addr(),
        );
        self.live_in.push((self.cur, g, v));
        self.regs.insert(g, v);
        v
    }

    /// The register as the instruction sees it, at width `w`.
    fn read(&mut self, g: Grp, w: Width) -> VId {
        let full = self.read64(g);
        if w == Width::W64 {
            return full;
        }
        self.cast(CastKind::Trunc, full, w)
    }

    /// Install a register value at a point that is a DEFINITION of it but is not
    /// a `write`: a call's RAX, a caller-saved clobber, a push/pop's RSP.
    ///
    /// `block_def` is the transfer function `ssa_registers` solves: "this block
    /// produced a new value for this register, so its successors read *that*".
    /// A register installed here without the mark is read by the successors as
    /// the block's *incoming* value, which is a silent miscompile of exactly the
    /// kind this project does not ship: after a call, RCX keeps the value the
    /// caller passed in, and a function whose entry block ends with `push rbp`
    /// hands every later block an undefined RSP, so its frame slots have no
    /// address and T2 files the whole function under "memory outside the
    /// modelled frame". Marking is the difference between "the model says the
    /// register is gone" and "the model forgot to say".
    fn set_reg(&mut self, g: Grp, v: VId) {
        self.regs.insert(g, v);
        self.cur_defs.insert(g);
    }

    /// Write a width-`w` result, honouring x86's partial-write rules.
    ///
    /// The value is first read at the width the WRITE has: a `W32` write of a
    /// wider value truncates, because that is what the hardware does. Skipping
    /// that step is what made T2's last 39 mismatches: `lea (%rcx,%rdx), %eax`
    /// computes a 64-bit address and writes 32 bits of it, and the old code
    /// "zero-extended" the full 64-bit sum - a no-op on an already-64-bit value,
    /// so the emitted C added the caller's whole 64-bit registers and returned
    /// `0x100000000` where the machine returned `0` (saturation,
    /// args=[0x80000000,0x80000000]).
    fn write(&mut self, g: Grp, w: Width, v: VId) {
        let v64 = match w {
            Width::W64 => v,
            // A 32-bit write zeroes the upper half: `zext` of the low 32 bits.
            Width::W32 => {
                let low = self.narrow(v, Width::W32);
                self.cast(CastKind::ZExt, low, Width::W64)
            }
            // An 8/16-bit write leaves the rest of the register alone.
            _ => {
                let old = self.read64(g);
                let mask: u64 = if w == Width::W8 { 0xff } else { 0xffff };
                let m = self.const_of(!mask, Width::W64);
                let keep = self.bin(BinOp::And, old, m);
                let low = self.narrow(v, w);
                let up = self.cast(CastKind::ZExt, low, Width::W64);
                self.bin(BinOp::Or, keep, up)
            }
        };
        self.regs.insert(g, v64);
        self.cur_defs.insert(g);
    }

    fn bin(&mut self, op: BinOp, a: VId, b: VId) -> VId {
        let t = self.f.ty_of(a).unwrap_or(Ty::u64());
        self.f.emit(self.cur, Op::Bin(op, a, b), t, self.addr())
    }

    /// Name a value without computing anything: the register's own bits, written
    /// back where they came from.
    fn copy(&mut self, v: VId, w: Width) -> VId {
        self.f.emit(self.cur, Op::Copy(v), ty_u(w), self.addr())
    }

    /// Translate a register self-idiom. See `SelfIdiom` for why this is the
    /// lifter's job and not a fold's.
    ///
    /// The FLAGS are still produced, with the same shape the general path gives
    /// (`Flags(Xor{a,a})`, `Flags(Sub{a,a,borrow:false})`, `Flags(And{a,a})`),
    /// so flag definedness is unchanged - a `jz` after `xor r,r` sees a defined
    /// flags value on both sides - and `fuse.rs` can still lower the condition
    /// off it. For the constant cases the operands are the zero that was just
    /// written, which is exactly the flags hardware sets: ZF=1, SF=0, PF=1,
    /// CF=0, OF=0.
    fn self_idiom(&mut self, d: Grp, dw: Width, si: SelfIdiom) {
        self.rep.self_idioms += 1;
        match si {
            SelfIdiom::ZeroXor | SelfIdiom::ZeroSub => {
                // No `read` of `d`: the instruction consumes no operand value.
                let zero = self.const_of(0, dw);
                self.write(d, dw, zero);
                self.set_flags(match si {
                    SelfIdiom::ZeroXor => FlagsOp::Xor { a: zero, b: zero },
                    _ => FlagsOp::Sub {
                        a: zero,
                        b: zero,
                        borrow: false,
                    },
                });
            }
            SelfIdiom::KeepAnd | SelfIdiom::KeepOr => {
                // These DO read the register: the answer is its own value.
                let v = self.read(d, dw);
                let r = self.copy(v, dw);
                self.write(d, dw, r);
                self.set_flags(match si {
                    SelfIdiom::KeepAnd => FlagsOp::And { a: v, b: v },
                    _ => FlagsOp::Or { a: v, b: v },
                });
            }
        }
    }

    fn const_of(&mut self, v: u64, w: Width) -> VId {
        self.f.emit(self.cur, Op::Const(v), ty_u(w), self.addr())
    }

    fn set_flags(&mut self, op: FlagsOp) {
        let v = self.f.emit(self.cur, Op::Flags(op), Ty::Flags, self.addr());
        self.flags = Some(v);
        self.flags_block = self.cur;
    }

    /// Flags are undefined after ops the model does not define them for. Reading
    /// them is then a counted gap, not a stale value.
    fn clear_flags(&mut self) {
        self.flags = None;
    }

    /// Every caller-saved register the call destroys, as one `Op::Clobber`
    /// DEFINITION each, written into the block's register map. `defined` is the
    /// one register this call writes an answer into — a modelled call does not
    /// clobber the register it defines, because its value is the call's, not a
    /// destroyed one.
    ///
    /// Not one clobber is dropped for the sake of emission: a register the model
    /// says is destroyed has no value, and every reader of that value (the
    /// interpreter's `Fault::Clobbered`, the emitter's "opaque: clobbered by
    /// call", the gate census) stays exactly as loud as it has always been.
    /// What DCE may delete is an *unread* clobber, and an unread clobber is
    /// genuinely dead — the call op itself stays, impure, in the same place.
    fn clobber_all(&mut self, defined: Option<Grp>) {
        // Counted PER CELL, one Op::Clobber per cell: an IR node defines one
        // value, so a "statement per register, ops per cell" count would
        // break the invariant `calls.rs` checks — every clobber the report
        // counts is in the IR. The register-level truth (25 caller-saved
        // registers) is unchanged; what moved is the cell count: an XMM
        // register is now two cells, so a call destroys 41 - 1 defined = 40.
        // A high lane left alive across a call is the same class of bug as an
        // int in ECX across one — the count going UP is the model getting
        // more precise, and the floor moved with the root cause, not around it.
        let mut n = 0;
        for group in CALLER_SAVED {
            for &g in *group {
                if Some(g) == defined {
                    continue;
                }
                let v = self.f.emit(self.cur, Op::Clobber, Ty::u64(), self.addr());
                self.set_reg(g, v);
                n += 1;
            }
        }
        self.rep.clobbers += n;
    }

    fn get_flag(&mut self, fl: Flag) -> Option<VId> {
        let f = self.flags?;
        Some(
            self.f
                .emit(self.cur, Op::GetFlag(fl, f), Ty::Bool, self.addr()),
        )
    }

    /// The Bool a condition code tests, built from flag reads. `fuse.rs`,
    /// constant folding and DCE simplify this; the lifter's job is to be literal.
    fn cond(&mut self, c: Cond) -> Option<VId> {
        let (zf, sf, of, cf, pf) = (
            self.get_flag(Flag::ZF),
            self.get_flag(Flag::SF),
            self.get_flag(Flag::OF),
            self.get_flag(Flag::CF),
            self.get_flag(Flag::PF),
        );
        let (zf, sf, of, cf, pf) = (zf?, sf?, of?, cf?, pf?);
        let v = match c {
            Cond::E => zf,
            Cond::Ne => self.bnot(zf),
            Cond::S => sf,
            Cond::Ns => self.bnot(sf),
            // signed: L is SF != OF, GE is SF == OF
            Cond::L => self.bitbool(BinOp::Xor, sf, of),
            Cond::Ge => {
                let d = self.bitbool(BinOp::Xor, sf, of);
                self.bnot(d)
            }
            Cond::G => {
                let nz = self.bnot(zf);
                let nge = self.bitbool(BinOp::Xor, sf, of);
                let ge = self.bnot(nge);
                self.bitbool(BinOp::And, nz, ge)
            }
            Cond::Le => {
                let l = self.bitbool(BinOp::Xor, sf, of);
                self.bitbool(BinOp::Or, zf, l)
            }
            // unsigned: A is CF==0 && ZF==0, BE is CF==1 || ZF==1
            Cond::A => {
                let ncf = self.bnot(cf);
                let nzf = self.bnot(zf);
                self.bitbool(BinOp::And, ncf, nzf)
            }
            Cond::Ae => self.bnot(cf),
            Cond::B => cf,
            Cond::Be => self.bitbool(BinOp::Or, cf, zf),
            Cond::O => of,
            Cond::No => self.bnot(of),
            Cond::P => pf,
            Cond::Np => self.bnot(pf),
        };
        Some(v)
    }

    /// Logical not over a Bool (1-bit) value: `x ^ 1` for the interpreter, and
    /// the fusion pass reads it as negation.
    fn bnot(&mut self, v: VId) -> VId {
        self.f
            .emit(self.cur, Op::Un(UnOp::Not, v), Ty::Bool, self.addr())
    }

    /// A bitwise op over Bool values: for 1-bit words, AND/OR/XOR are the
    /// logical connectives.
    fn bitbool(&mut self, o: BinOp, x: VId, y: VId) -> VId {
        self.f
            .emit(self.cur, Op::Bin(o, x, y), Ty::Bool, self.addr())
    }

    fn reg_op(&self, ins: &Instruction, i: u32) -> Option<(Grp, Width)> {
        if ins.op_kind(i) != OpKind::Register {
            return None;
        }
        split(ins.op_register(i))
    }

    fn imm_of(&self, ins: &Instruction, i: u32, w: Width) -> Option<u64> {
        // 0x0F = "imm8/16/32/64 all optional", iced's mask for "any immediate".
        let v = ins.try_immediate(i).ok()? as i64;
        Some(match w {
            Width::W8 => (v as u8) as u64,
            Width::W16 => (v as u16) as u64,
            Width::W32 => (v as u32) as u64,
            _ => v as u64,
        })
    }

    /// An operand as a value: register, immediate, or memory load. `None` when
    /// it needs a model we do not have (RIP-relative, unaddressable index).
    fn src(&mut self, ins: &Instruction, i: u32, w: Width) -> Option<VId> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let (g, gw) = split(ins.op_register(i))?;
                // A wider source read at the operation's width is a trunc; the
                // 32-bit-write-clears-upper-half rule is already enforced by `write`.
                let v = self.read(g, gw);
                Some(self.narrow(v, w))
            }
            OpKind::Immediate8
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate16
            | OpKind::Immediate32
            // `imul rax,rcx,imm32` — the form every compiler uses for division
            // by a magic constant — reports its immediate as
            // `Immediate32to64`, and iced sign-extends it. It was the one
            // immediate kind missing from this list, so the whole 3-operand
            // 64-bit `imul` family was refused as "an unmodelled operand" while
            // the arithmetic next to it lifted fine.
            | OpKind::Immediate32to64
            | OpKind::Immediate64 => {
                let v = self.imm_of(ins, i, w)?;
                Some(self.const_of(v, w))
            }
            OpKind::Memory => {
                let a = self.addr_of(ins, i)?;
                let t = ty_u(w);
                Some(
                    self.f
                        .emit(self.cur, Op::Load(t.clone(), a), t, self.addr()),
                )
            }
            _ => None,
        }
    }

    /// Address value of a memory operand. `None` for anything we cannot name:
    /// a RIP-relative displacement with no relocation naming a defined symbol,
    /// an unaddressable index.
    ///
    /// A RIP-relative displacement is NOT self-describing in an object file:
    /// the bytes hold `target - next_ip` UNRELOCATED, and the real target lives
    /// in the REL32 relocation at the displacement's address. With that
    /// relocation, the address of a defined symbol — code, `.rdata`, a `.bss`
    /// static — is a constant we can name: an address in the object's flat
    /// per-object space, which is what the emitter's `RDEC_ADDR` masks into
    /// `rdec_mem`. Without it, the refusal names the symbol the relocation
    /// carried, so the gap is a work item with an identity.
    fn addr_of(&mut self, ins: &Instruction, op: u32) -> Option<VId> {
        if ins.op_kind(op) != OpKind::Memory {
            return None;
        }
        let base = ins.memory_base();
        if base == Register::RIP {
            // Resolved by SCANNING the instruction's span (`rip_reloc`), not by
            // `next_ip - 4`: that arithmetic assumes the disp32 is the last field
            // and is false for every RIP form with a trailing immediate
            // (`cmpl $0x2, (%rip)` = `83 3D <disp32> <imm8>`).
            match self.rip_reloc(ins).and_then(|rt| rt.addr) {
                Some(a) => {
                    self.rep.rip_named += 1;
                    Some(self.const_of(a, Width::W64))
                }
                None => None,
            }
        } else if base == Register::None {
            None
        } else {
            let (bg, _) = split(base)?;
            let mut v = self.read64(bg);
            let index = ins.memory_index();
            if index != Register::None {
                let (ig, _) = split(index)?;
                let iv = self.read64(ig);
                let scale = ins.memory_index_scale();
                if scale > 1 {
                    let s = self.const_of(scale as u64, Width::W64);
                    let scaled = self.bin(BinOp::Mul, iv, s);
                    v = self.bin(BinOp::Add, v, scaled);
                } else {
                    v = self.bin(BinOp::Add, v, iv);
                }
            }
            let disp = ins.memory_displacement64() as i64;
            if disp != 0 {
                let d = self.const_of(disp as u64, Width::W64);
                v = self.bin(BinOp::Add, v, d);
            }
            Some(v)
        }
    }

    /// The relocation THIS INSTRUCTION carries, by scanning its own bytes: a
    /// direct branch has exactly one, on its displacement, and the address of
    /// that field is what the container records. Scanning the instruction's span
    /// (instead of guessing the encoding width) is what keeps a `jmp rel8` and a
    /// `jmp rel32` the same question, and returns a copy so the caller can mutate
    /// `self` while holding it.
    fn reloc_on(&self, ins: &Instruction) -> Option<RelocTarget> {
        let a = self.addr();
        (0..ins.len() as u64)
            .find_map(|k| self.relocs.get(&(a + k)))
            .cloned()
    }

    /// The relocation carrying THIS instruction's RIP-relative displacement.
    ///
    /// NOT `next_ip - 4`. That arithmetic is right only when the disp32 is the
    /// instruction's LAST field: true for `39 0D <disp32>` (`cmpl (%rip), %ecx`),
    /// FALSE for every RIP form with a trailing immediate -
    /// `83 3D <disp32> <imm8>` (`cmpl $0x2, (%rip)`), `81 /7 <disp32> <imm32>`,
    /// `C6 /0 <disp32> <imm8>`, `F6 /7 <disp32> <imm8>`. Measured:
    /// `corpus/msvc/obj/array_walk.O2.obj :: f @ 0x2f` is `cmpl $0x2, (%rip)`;
    /// its disp32 occupies offsets 2..6 of an 8-byte instruction, so `next_ip - 4`
    /// = 4 lands INSIDE the field while the table keys the field's FIRST byte (2),
    /// and the site refused as "flag setting with an unmodelled first operand" even
    /// after the memory-destination arm had landed. One off-by-two, in a shared
    /// helper, for a whole encoding family.
    ///
    /// The rule is the one `reloc_on` already uses for branch targets: scan the
    /// instruction's OWN span, which keeps every encoding width the same question.
    /// A span carrying TWO distinct relocation targets is not guessed - the model
    /// cannot say which field the RIP base means - so it refuses and the caller's
    /// text names the arm (`lea of an address we cannot name`, ...).
    fn rip_reloc(&self, ins: &Instruction) -> Option<RelocTarget> {
        let a = self.addr();
        let mut found: Option<RelocTarget> = None;
        for k in 0..ins.len() as u64 {
            if let Some(rt) = self.relocs.get(&(a + k)) {
                match &found {
                    // The same target recorded at two offsets (a padded field) is
                    // one answer, not a conflict.
                    Some(prev) if prev.sym == rt.sym && prev.addr == rt.addr => {}
                    Some(_) => return None,
                    None => found = Some(rt.clone()),
                }
            }
        }
        found
    }

    /// The symbol named by the RIP-relative relocation on this instruction,
    /// for refusal text: `": h"`, or `""` when no relocation is there at all.
    fn rip_sym(&self, ins: &Instruction) -> Option<String> {
        self.rip_reloc(ins).map(|rt| format!(": {}", rt.sym))
    }

    /// Resolve a direct branch target from the relocation at its displacement
    /// field. `disp_len` is the ENCODING's width (rel32 = 4, rel16 = 2), so the
    /// lookup key is `next_ip - disp_len`. `count` routes the three outcomes
    /// into the call census (only call sites count: a `jcc`/`jmp` target is
    /// block-edge evidence for `discover`, and this slice resolves calls).
    /// Falls back to the immediate when the table has nothing to say.
    fn resolve_reloc_target(
        &mut self,
        ins: &Instruction,
        imm_target: u64,
        disp_len: u64,
        count: bool,
    ) -> u64 {
        match self.relocs.get(&ins.next_ip().wrapping_sub(disp_len)) {
            Some(rt) => match rt.addr {
                Some(a) => {
                    if count {
                        self.rep.calls_reloc_resolved += 1;
                    }
                    a
                }
                // Names a symbol the object does not define: an external call
                // has no COFF symbol to link, so it keeps the loud bucket it had.
                None => {
                    if count {
                        self.rep.calls_reloc_external += 1;
                    }
                    imm_target
                }
            },
            // No relocation: an intra-section target the compiler already
            // linked into the immediate - the immediate IS the evidence here.
            None => {
                if count {
                    self.rep.calls_reloc_absent += 1;
                }
                imm_target
            }
        }
    }

    /// Memory access size -> a modelled width. 128/256-bit accesses (SSE, AVX)
    /// have no integer width: they are unmodelled, and `None` makes the caller
    /// say so.
    fn w_of_ms(ms: iced_x86::MemorySize) -> Option<Width> {
        use iced_x86::MemorySize as M;
        use Width::*;
        match ms {
            M::UInt8 | M::Int8 => Some(W8),
            M::UInt16 | M::Int16 => Some(W16),
            M::UInt32 | M::Int32 => Some(W32),
            M::UInt64 | M::Int64 => Some(W64),
            // UInt52 (far ptr), 128/256/512, floats, BCD: no integer width here.
            _ => None,
        }
    }

    fn mem_w(&self, ins: &Instruction, op: u32, default: Width) -> Width {
        if ins.op_kind(op) != OpKind::Memory {
            return default;
        }
        Self::w_of_ms(ins.memory_size()).unwrap_or(default)
    }

    /// The width an operand has *as a source*: its own size, not the operation's.
    fn src_w(&self, ins: &Instruction, op: u32, default: Width) -> Width {
        match ins.op_kind(op) {
            OpKind::Register => split(ins.op_register(op))
                .map(|(_, w)| w)
                .unwrap_or(default),
            OpKind::Memory => self.mem_w(ins, op, default),
            _ => default,
        }
    }

    fn target(&self, ins: &Instruction) -> Option<u64> {
        for i in 0..ins.op_count() {
            match ins.op_kind(i) {
                OpKind::NearBranch16 => return Some(ins.near_branch16() as u64),
                OpKind::NearBranch32 => return Some(ins.near_branch32() as u64),
                OpKind::NearBranch64 => return Some(ins.near_branch64()),
                _ => {}
            }
        }
        None
    }

    /// The operand width an instruction operates at.
    fn op_w(&self, ins: &Instruction) -> Width {
        self.reg_op(ins, 0)
            .map(|(_, w)| w)
            .or_else(|| Self::w_of_ms(ins.memory_size()))
            .unwrap_or(Width::W64)
    }

    fn narrow(&mut self, v: VId, w: Width) -> VId {
        let cur = self
            .f
            .ty_of(v)
            .and_then(|t| t.width())
            .unwrap_or(Width::W64);
        if w == Width::W64 || cur == w {
            return v;
        }
        self.cast(CastKind::Trunc, v, w)
    }

    fn cast(&mut self, k: CastKind, v: VId, w: Width) -> VId {
        // Sign extension is a statement about the SIGNEDNESS OF THE SOURCE, and
        // registers are modelled as one unsigned W64 value, so `movsx` used to
        // emit SExt from a `uint32_t` - which the C emitter correctly refuses
        // ("SExt from unsigned ... needs a signed source type", 12 corpus
        // functions). Re-read the source as signed at its own width first: a
        // same-width Bitcast keeps every bit and changes only how the top bit is
        // read, which is exactly what movsx's mnemonic asserts.
        let v = if k == CastKind::SExt {
            let src = self.f.ty_of(v);
            let sw = src.clone().and_then(|t| t.width()).unwrap_or(w);
            let st = Ty::int(sw, rdec_core::ty::Sx::Signed);
            if src != Some(st.clone()) {
                self.f.emit(
                    self.cur,
                    Op::Cast(CastKind::Bitcast, st.clone(), v),
                    st,
                    self.addr(),
                )
            } else {
                v
            }
        } else {
            v
        };
        let t = ty_u(w);
        if self.f.ty_of(v) == Some(t.clone()) {
            return v;
        }
        self.f
            .emit(self.cur, Op::Cast(k, t.clone(), v), t, self.addr())
    }

    // ---- SSE: the scalar subset -------------------------------------------
    //
    // The XMM cells are bit patterns (`Grp::Xmm`, module doc), so every op here
    // opens with an explicit float view and closes with the inverse. Nothing in
    // this section invents a C rule: each arm says what the hardware does, and
    // what it cannot express it refuses by name.

    /// The XMM register file: one PAIR of `W64` bit cells per register (lo,
    /// hi), read through the same `read64` the GPRs use, so a read before a
    /// write is a live-in here too and a loop-carried float gets a phi for free.
    fn xmm(&mut self, i: u8) -> VId {
        self.read64(Grp::Xmm(i))
    }

    fn set_xmm(&mut self, i: u8, v: VId) {
        self.set_reg(Grp::Xmm(i), v);
    }

    /// The register's high cell (bits 127:64). A scalar op PRESERVES it (the
    /// store and merge forms) or ZEROES it (the load forms, per Intel) - the
    /// module doc lists which form does which; nothing here improvises.
    fn xmm_hi(&mut self, i: u8) -> VId {
        self.read64(Grp::XmmHi(i))
    }

    fn set_xmm_hi(&mut self, i: u8, v: VId) {
        self.set_reg(Grp::XmmHi(i), v);
    }

    /// Zero the high cell: the explicit spelling of Intel's merge-form rule
    /// (`movsd xmm, ...`, `movss xmm, ...`, `movd`/`movq` into xmm zero the
    /// upper bits). Writing the zero is the model being explicit: the cell
    /// exists, and this instruction's answer for it is 0.
    fn zero_xmm_hi(&mut self, i: u8) {
        let z = self.const_of(0, Width::W64);
        self.set_xmm_hi(i, z);
    }

    /// Splice `bits` (32/64 bits) into the low cell of lane `lane`: the
    /// destination register is NOT zeroed, its untouched lane survive —
    /// that merge rule is what a scalar write is on silicon. The arithmetic
    /// is lane-exact, not a lane-swap:
    ///
    /// - movq xmm, r64: 16 ones = `lo&~mask | bits<<0`
    /// - pinsrd xmm, r32, imm: 0xfffffff = 32 ones = `lo&~(mask<<0) | zext32<<0`
    ///
    /// movq's mask IS all 64 low bits, so the `>>shift`/`<<shift` is 0 there
    /// and the arm's rule drops out of the arithmetic. No model cell is
    /// faked; every lane is a W64 cell, and the shift moves mask and value
    /// together.
    fn splice_lane(&mut self, d: u8, bits: VId, lane: u32, wide: bool) {
        let z32;
        let (mask, val) = if wide {
            (
                self.const_of(u64::MAX, Width::W64),
                self.narrow(bits, Width::W64),
            )
        } else {
            let lo = self.narrow(bits, Width::W32);
            z32 = self.cast(CastKind::ZExt, lo, Width::W64);
            (self.const_of(0xffff_ffff, Width::W64), z32)
        };
        let ones = self.const_of(u64::MAX, Width::W64);
        let inv = self.bin(BinOp::Xor, mask, ones);
        if lane == 0 {
            let lo = self.xmm(d);
            let keep = self.bin(BinOp::And, lo, inv);
            let new_l = self.bin(BinOp::Or, keep, val);
            self.set_xmm(d, new_l);
        } else {
            let hi = self.xmm_hi(d);
            let keep = self.bin(BinOp::And, hi, inv);
            let new_h = self.bin(BinOp::Or, keep, val);
            self.set_xmm_hi(d, new_h);
        }
    }

    /// Load 16 bytes at `a` into the pair (lo cell, hi cell): two W64 loads,
    /// lo at addr and hi at addr+8 - little-endian lane order, which is what
    /// the hardware's byte order is. The model does not charge for alignment:
    /// an unaligned `movdqa` faults on silicon and lands in T2's crash bucket;
    /// the model invents no fault semantics, the same way an unaligned scalar
    /// load never faulted here.
    fn load16(&mut self, a: VId, d: u8) {
        let at = self.addr();
        let t = Ty::u64();
        let lo = self.f.emit(self.cur, Op::Load(t.clone(), a), t.clone(), at);
        let e8 = self.const_of(8, Width::W64);
        let a8 = self.bin(BinOp::Add, a, e8);
        let hi = self.f.emit(self.cur, Op::Load(t, a8), Ty::u64(), at);
        self.set_xmm(d, lo);
        self.set_xmm_hi(d, hi);
        self.rep.sse_packed += 1;
    }

    /// Store the pair as 16 bytes at `a`: lo at addr, hi at addr+8. The pair
    /// of stores IS the lane contract - swapping them is a miscompile that
    /// the T2 memory axis and `packed_simd.rs` both catch by bytes.
    fn store16(&mut self, a: VId, s: u8) {
        let at = self.addr();
        // Reads hoisted before the emits: `emit` appends to the block that
        // `read64`'s live-in machinery also touches (the existing rule).
        let lo = self.xmm(s);
        let hi = self.xmm_hi(s);
        let t = Ty::u64();
        self.f.emit_void(self.cur, Op::Store(t.clone(), a, lo), at);
        let e8 = self.const_of(8, Width::W64);
        let a8 = self.bin(BinOp::Add, a, e8);
        self.f.emit_void(self.cur, Op::Store(t, a8, hi), at);
        self.rep.sse_packed += 1;
    }

    /// The 16 bytes at `a` read as a lane pair (lo at addr, hi at addr+8):
    /// the LOAD half of `load16`, spelled once so a packed op that takes its
    /// source from memory reads exactly the pair a register destination holds.
    fn load16_at(&mut self, a: VId) -> (VId, VId) {
        let at = self.addr();
        let t = Ty::u64();
        let lo = self.f.emit(self.cur, Op::Load(t.clone(), a), t.clone(), at);
        let e8 = self.const_of(8, Width::W64);
        let a8 = self.bin(BinOp::Add, a, e8);
        let hi = self.f.emit(self.cur, Op::Load(t, a8), Ty::u64(), at);
        (lo, hi)
    }

    /// A packed SSE operand as a lane pair: a register's two cells, or the 16
    /// bytes at a memory operand. `None` means the operand is a kind this
    /// model cannot read (an unresolvable RIP displacement, a broadcast) and
    /// the CALLER makes that loud — the model does not guess at a constant it
    /// cannot name.
    fn packed_pair(&mut self, ins: &Instruction, i: u32) -> Option<(VId, VId)> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let s = self.xmm_op(ins, i)?;
                // Both reads before any emit: `emit` appends to the block that
                // `read64`'s live-in machinery also touches.
                let lo = self.xmm(s);
                let hi = self.xmm_hi(s);
                Some((lo, hi))
            }
            OpKind::Memory => {
                let a = self.addr_of(ins, i)?;
                Some(self.load16_at(a))
            }
            _ => None,
        }
    }

    /// The u32 lane `k` (0 = low) of a W64 cell, typed W32. Lane 1 shifts the
    /// cell up first — a shift, not a byte read, because the lane lives in a
    /// register — and the count is a W64 constant so the `Shr` pair agrees on
    /// a width (the verifier's mixed-width rule, and x86's own shift-count
    /// convention in `Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar`).
    fn lane32(&mut self, v: VId, k: u8) -> VId {
        let at = if k == 0 {
            v
        } else {
            let s = self.const_of(32, Width::W64);
            self.bin(BinOp::Shr, v, s)
        };
        self.cast(CastKind::Trunc, at, Width::W32)
    }

    /// Two W32 lane values back into one W64 cell: `(hi << 32) | lo`, each
    /// lane zero-extended first so the two lanes share no bit on the way.
    fn join32(&mut self, lo: VId, hi: VId) -> VId {
        let lo64 = self.cast(CastKind::ZExt, lo, Width::W64);
        let hi64 = self.cast(CastKind::ZExt, hi, Width::W64);
        let s = self.const_of(32, Width::W64);
        let up = self.bin(BinOp::Shl, hi64, s);
        self.bin(BinOp::Or, lo64, up)
    }

    /// Apply `f` to the two u32 lanes of a cell pair and join the answers.
    /// This is where "lane-independent" is a property of the IR and not of a
    /// comment: `f` sees W32-typed values only, so the add carry / sub borrow
    /// it produces is a 32-bit one — the interpreter truncates the lane's own
    /// answer and the C prints `uint32_t` — and it cannot reach the next lane.
    /// Modelling the op at W64 and masking after is the same bytes on the easy
    /// operands and a different BYTE on the carry cases; `packed_simd.rs`'s
    /// fixtures are those cases, and the saboteur model is in that file.
    fn cell32<F>(&mut self, a: VId, b: VId, f: F) -> VId
    where
        F: Fn(&mut Self, VId, VId) -> VId,
    {
        let (a0, a1) = (self.lane32(a, 0), self.lane32(a, 1));
        let (b0, b1) = (self.lane32(b, 0), self.lane32(b, 1));
        let r0 = f(self, a0, b0);
        let r1 = f(self, a1, b1);
        self.join32(r0, r1)
    }

    /// `a == b` -> `Ty::Bool`, at the operands' width. Signedness is a
    /// property of the predicate (`IntCmp::Eq` has none), so W32 lanes and W64
    /// lanes both read their own bits.
    fn cmp_eq(&mut self, a: VId, b: VId) -> VId {
        self.f
            .emit(self.cur, Op::Cmp(IntCmp::Eq, a, b), Ty::Bool, self.addr())
    }

    /// The lane mask `PCMPEQ*` writes: ALL ONES for equal, zero for unequal.
    /// The Bool -> integer spelling is the codebase's own — `setcc` builds
    /// `Select(cond, 1, 0)` at the destination width — with the machine's arm:
    /// all-ones, not 1. `w` is the lane width (`W32` for `pcmpeqd`, `W64` for
    /// `pcmpeqq`), so the mask, the comparison and the join all agree on it.
    fn mask_at(&mut self, c: VId, w: Width) -> VId {
        let bits = w.bits();
        let ones = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let t = ty_u(w);
        let ones = self
            .f
            .emit(self.cur, Op::Const(ones), t.clone(), self.addr());
        let zero = self.f.emit(self.cur, Op::Const(0), t.clone(), self.addr());
        self.f
            .emit(self.cur, Op::Select(c, ones, zero), t, self.addr())
    }

    /// The lane-independent packed arithmetic, by canonical name. The lane
    /// WIDTH is part of the answer: `paddd` is four u32 lanes and `paddq` two
    /// u64 lanes, and reading one as the other is the miscompile the fixtures
    /// name. `paddb`/`paddw` and the saturating forms are NOT here — 8- and
    /// 16-bit lanes need their own lane helpers, and refusing them by name
    /// keeps them in the census as the work item they are.
    fn packed_arith(nm: &str) -> Option<PackedArith> {
        Some(match nm {
            "paddq" => PackedArith::Add64,
            "psubq" => PackedArith::Sub64,
            "pcmpeqq" => PackedArith::Eq64,
            "paddd" => PackedArith::Add32,
            "psubd" => PackedArith::Sub32,
            "pcmpeqd" => PackedArith::Eq32,
            _ => return None,
        })
    }

    /// The operand is `XMMn`; that is the whole question, so this is the only
    /// place that asks it.
    fn xmm_op(&self, ins: &Instruction, i: u32) -> Option<u8> {
        if ins.op_kind(i) != OpKind::Register {
            return None;
        }
        use Register as R;
        Some(match ins.op_register(i) {
            R::XMM0 => 0,
            R::XMM1 => 1,
            R::XMM2 => 2,
            R::XMM3 => 3,
            R::XMM4 => 4,
            R::XMM5 => 5,
            R::XMM6 => 6,
            R::XMM7 => 7,
            R::XMM8 => 8,
            R::XMM9 => 9,
            R::XMM10 => 10,
            R::XMM11 => 11,
            R::XMM12 => 12,
            R::XMM13 => 13,
            R::XMM14 => 14,
            R::XMM15 => 15,
            _ => return None,
        })
    }

    /// The bit pattern read as a float at `fw`. In the C this is
    /// `rdec_bits_f64(x)` / `rdec_bits_f32(x)` -- a memcpy, never a conversion.
    fn fview(&mut self, bits: VId, fw: FWidth) -> VId {
        let t = Ty::Float { w: fw };
        self.f.emit(
            self.cur,
            Op::Cast(CastKind::Bitcast, t.clone(), bits),
            t,
            self.addr(),
        )
    }

    /// The inverse view: a float value's bits, for writing back into a cell.
    fn fbits(&mut self, fl: VId) -> VId {
        let t = Ty::uint(Width::W64);
        self.f.emit(
            self.cur,
            Op::Cast(CastKind::Bitcast, t.clone(), fl),
            t,
            self.addr(),
        )
    }

    /// `Cast` to an arbitrary type. `cast` builds an integer type from a width,
    /// which is right for the GPRs and wrong here: a float conversion's result
    /// type is a float.
    fn cast_ty(&mut self, k: CastKind, v: VId, t: Ty) -> VId {
        self.f
            .emit(self.cur, Op::Cast(k, t.clone(), v), t, self.addr())
    }

    fn fbinop(&mut self, o: FBinOp, a: VId, b: VId) -> VId {
        let t = self.f.ty_of(a).unwrap_or_else(Ty::f64);
        self.f.emit(self.cur, Op::FBin(o, a, b), t, self.addr())
    }

    /// A scalar SSE operand: an XMM cell viewed at `fw`, or memory loaded as a
    /// float at `fw` (which the emitter spells as a bit reinterpretation).
    fn fsrc(&mut self, ins: &Instruction, i: u32, fw: FWidth) -> Option<VId> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let s = self.xmm_op(ins, i)?;
                let b = self.xmm(s);
                Some(self.fview(b, fw))
            }
            OpKind::Memory => {
                let a = self.addr_of(ins, i)?;
                let t = Ty::Float { w: fw };
                Some(
                    self.f
                        .emit(self.cur, Op::Load(t.clone(), a), t, self.addr()),
                )
            }
            _ => None,
        }
    }

    /// A refusal that names the mnemonic, the reason, and (through `self.at`)
    /// the address -- the same shape as `unmodelled instruction: movsd`, so the
    /// gate's refusal classes sort by mnemonic on their own.
    fn sse_gap(&mut self, nm: &str, why: &str) {
        let text = format!("unmodelled instruction: {nm} ({why})");
        self.f
            .emit_void(self.cur, Op::Opaque(text.into()), self.addr());
        self.rep.gaps += 1;
    }

    /// Translate one SSE instruction: the scalar subset, and nothing else.
    ///
    /// Matching on the canonical mnemonic NAME is deliberate. The name is what
    /// the census counts, what a refusal prints, and what `llvm-objdump -d`
    /// shows at that address, so the three are checkable against each other; an
    /// enum match would let the gate's numbers and the object's bytes drift
    /// apart in silence.
    fn sse(&mut self, nm: &str, ins: &Instruction) {
        use FWidth::{F32, F64};
        // ---- the zeroing idiom, and only the idiom -------------------------
        if matches!(nm, "pxor" | "xorps" | "xorpd") {
            if let (Some(d), Some(s)) = (self.xmm_op(ins, 0), self.xmm_op(ins, 1)) {
                if d == s {
                    // `pxor xmm0, xmm0`: all 128 bits become 0 and the register
                    // is NOT read - both cells, because the machine zeroes all
                    // 16 bytes. The SSE sibling of `xor r,r`, modelled here for
                    // the reason `SelfIdiom` gives: as a general IR fold
                    // (`xor(x,x) -> 0`) the rule is a lie about meaning, because
                    // this interpreter is strict -- a read of an undefined xmm
                    // faults when it is reached, and deleting the read turns a
                    // faulting execution into a returning one. The operand is an
                    // address whose value is never consumed, and only the
                    // encoding knows that.
                    self.rep.sse_zeroing += 1;
                    let z = self.const_of(0, Width::W64);
                    self.set_xmm(d, z);
                    self.set_xmm_hi(d, z);
                    // EFLAGS untouched. Setting flags here would hand a later
                    // `jz` a flags value the machine left stale.
                    return;
                }
                // `pxor` on two registers is lane-independent bitwise on the
                // full 128 bits: two INDEPENDENT 64-bit xors, no lane crossing,
                // so the lane model expresses it exactly. `xorps`/`xorpd` are
                // the same bitwise answer on float masks, but the slice names
                // the integer set: they keep their loud gap below.
                if nm == "pxor" {
                    let (dl, dh) = (self.xmm(d), self.xmm_hi(d));
                    let (sl, sh) = (self.xmm(s), self.xmm_hi(s));
                    let lo = self.bin(BinOp::Xor, dl, sl);
                    let hi = self.bin(BinOp::Xor, dh, sh);
                    self.set_xmm(d, lo);
                    self.set_xmm_hi(d, hi);
                    self.rep.sse_packed += 1;
                    return;
                }
            }
            self.sse_gap(nm, "a 128-bit bitwise xor: a mask, not a scalar op");
            return;
        }
        // ---- lane-independent bitwise, both cells --------------------------
        // `pand`/`pandn`/`por` (Intel: DEST &= SRC; DEST = (NOT DEST) & SRC;
        // DEST |= SRC). Like `pxor`, each is two independent 64-bit lane ops.
        // EFLAGS untouched, as for every SSE op: giving a later `jz` flags the
        // machine left stale is the miscompile.
        if matches!(nm, "pand" | "pandn" | "por") {
            if let (Some(d), Some(s)) = (self.xmm_op(ins, 0), self.xmm_op(ins, 1)) {
                let (dl, dh) = (self.xmm(d), self.xmm_hi(d));
                let (sl, sh) = (self.xmm(s), self.xmm_hi(s));
                let (lo, hi) = match nm {
                    "pand" => (self.bin(BinOp::And, dl, sl), self.bin(BinOp::And, dh, sh)),
                    "por" => (self.bin(BinOp::Or, dl, sl), self.bin(BinOp::Or, dh, sh)),
                    // pandn: NOT of the DESTINATION, then AND with the source
                    // (Intel PANDN; the operand order is the AT&T pair the
                    // decoder reports, dest first).
                    _ => {
                        let nl =
                            self.f
                                .emit(self.cur, Op::Un(UnOp::Not, dl), Ty::u64(), self.addr());
                        let nh =
                            self.f
                                .emit(self.cur, Op::Un(UnOp::Not, dh), Ty::u64(), self.addr());
                        (self.bin(BinOp::And, nl, sl), self.bin(BinOp::And, nh, sh))
                    }
                };
                self.set_xmm(d, lo);
                self.set_xmm_hi(d, hi);
                self.rep.sse_packed += 1;
                return;
            }
            self.sse_gap(
                nm,
                "a 128-bit bitwise op with a source this model cannot read",
            );
            return;
        }
        // ---- lane-independent packed arithmetic, one lane at a time --------
        //
        // `paddq`/`psubq` are two u64 lanes; `paddd`/`psubd` are four u32 lanes
        // per 128 bits (two per cell); `pcmpeqd`/`pcmpeqq` answer all-ones /
        // zero per lane. Every one of them is LANE-INDEPENDENT — a lane's
        // answer is a function of that lane's two operands only — so the pair
        // of W64 cells expresses it exactly, and the rule that keeps the model
        // honest is that each lane op is typed at the LANE's width: a W32
        // `Add`/`Sub` truncates to 32 bits in the interpreter and prints as
        // `uint32_t` in the C, so the carry (add) and the borrow (sub) die
        // inside the lane. Spelling them at W64 and masking afterwards is the
        // same bytes on friendly operands and a different byte on the carry
        // cases, which is what `packed_simd.rs` pins with operands chosen for
        // exactly that disagreement.
        //
        // The DESTINATION is the register the AT&T listing prints second
        // (Intel's first), read and written as a pair like every other packed
        // op; the source is a register or 16 bytes of memory — `paddq (%rip),
        // %xmm` over a `.rdata` vector constant is the form the corpus's
        // vectorised witness emits. EFLAGS untouched, as for every SSE op:
        // giving a later `jz` flags the machine left stale is the miscompile.
        if let Some(op) = Self::packed_arith(nm) {
            let Some(d) = self.xmm_op(ins, 0) else {
                self.sse_gap(nm, "no XMM register destination");
                return;
            };
            let Some((sl, sh)) = self.packed_pair(ins, 1) else {
                self.sse_gap(
                    nm,
                    "a packed source this model cannot read (memory it cannot address)",
                );
                return;
            };
            // Reads before the emits below, per the rule in `sse`'s scalar arm.
            let (dl, dh) = (self.xmm(d), self.xmm_hi(d));
            let (lo, hi) = match op {
                PackedArith::Add64 => (self.bin(BinOp::Add, dl, sl), self.bin(BinOp::Add, dh, sh)),
                PackedArith::Sub64 => (self.bin(BinOp::Sub, dl, sl), self.bin(BinOp::Sub, dh, sh)),
                PackedArith::Eq64 => {
                    let e = self.cmp_eq(dl, sl);
                    let lo = self.mask_at(e, Width::W64);
                    let e = self.cmp_eq(dh, sh);
                    let hi = self.mask_at(e, Width::W64);
                    (lo, hi)
                }
                PackedArith::Add32 => {
                    let lo = self.cell32(dl, sl, |s, x, y| s.bin(BinOp::Add, x, y));
                    let hi = self.cell32(dh, sh, |s, x, y| s.bin(BinOp::Add, x, y));
                    (lo, hi)
                }
                PackedArith::Sub32 => {
                    let lo = self.cell32(dl, sl, |s, x, y| s.bin(BinOp::Sub, x, y));
                    let hi = self.cell32(dh, sh, |s, x, y| s.bin(BinOp::Sub, x, y));
                    (lo, hi)
                }
                PackedArith::Eq32 => {
                    let eq = |s: &mut Self, x, y| {
                        let c = s.cmp_eq(x, y);
                        s.mask_at(c, Width::W32)
                    };
                    let lo = self.cell32(dl, sl, eq);
                    let hi = self.cell32(dh, sh, eq);
                    (lo, hi)
                }
            };
            self.set_xmm(d, lo);
            self.set_xmm_hi(d, hi);
            self.rep.sse_packed += 1;
            return;
        }
        // ---- scalar arithmetic --------------------------------------------
        if let Some((op, fw)) = Self::sse_arith(nm) {
            if ins.op_kind(0) != OpKind::Register || self.xmm_op(ins, 0).is_none() {
                self.sse_gap(nm, "no XMM register destination");
                return;
            }
            let d = self.xmm_op(ins, 0).unwrap();
            // `self.xmm(d)` must be read BEFORE the view is emitted: `fview`
            // appends an instruction, which grows `self.f.blocks` that
            // `read64`'s live-in machinery also touches.
            let bits = self.xmm(d);
            let a = self.fview(bits, fw);
            let b = match self.fsrc(ins, 1, fw) {
                Some(v) => v,
                None => {
                    self.sse_gap(nm, "no readable float source");
                    return;
                }
            };
            let r = self.fbinop(op, a, b);
            let bits = self.fbits(r);
            // Merge-form arithmetic: `addsd` and friends write the low lane
            // and leave bits 127:64 of the destination alone (Intel). Writing
            // only the low cell IS the preservation - the high cell keeps its
            // binding, so it flows to the successors unchanged.
            self.set_xmm(d, bits);
            return;
        }
        // ---- moves, by the size the mnemonic names ------------------------
        match nm {
            "movsd" | "movss" => {
                let fw = if nm == "movsd" { F64 } else { F32 };
                let n = if nm == "movsd" {
                    Width::W64
                } else {
                    Width::W32
                };
                if let Some(d) = self.xmm_op(ins, 0) {
                    let v = match ins.op_kind(1) {
                        OpKind::Register => {
                            let s = match self.xmm_op(ins, 1) {
                                Some(s) => s,
                                None => {
                                    self.sse_gap(nm, "source is not an XMM register");
                                    return;
                                }
                            };
                            // The low bits move; `movsd` zeroes 127:64 and
                            // `movss` zeroes 127:32 (Intel's load-merge form).
                            // With the pair, the zeroing is not implicit any
                            // more - it gets an explicit write, below: the
                            // high cell exists now, and this instruction's
                            // answer for it is 0.
                            //
                            // The read is hoisted out of the `emit` argument
                            // list: `emit` takes `&mut self.f`, and `self.xmm`
                            // takes `&mut self`, which the borrow checker
                            // correctly refuses to stack.
                            let src = self.xmm(s);
                            self.f
                                .emit(self.cur, Op::Copy(src), Ty::uint(n), self.addr())
                        }
                        OpKind::Memory => {
                            let a = match self.addr_of(ins, 1) {
                                Some(a) => a,
                                None => {
                                    self.sse_gap(nm, "load from an address we cannot name");
                                    return;
                                }
                            };
                            let t = ty_u(n);
                            self.f.emit(self.cur, Op::Load(t, a), ty_u(n), self.addr())
                        }
                        _ => {
                            self.sse_gap(nm, "source is neither an XMM register nor memory");
                            return;
                        }
                    };
                    let _ = fw; // the precision lives in the cell's users, not the cell
                    let up = self.cast(CastKind::ZExt, v, Width::W64);
                    self.set_xmm(d, up);
                    // The load-merge zeroing, written explicitly: `movsd`
                    // clears 127:64 and `movss` clears 127:32 (the `movss`
                    // 63:32 zeroing is the ZExt above; the whole high cell is
                    // this line). Pinned by `packed_simd.rs` -- this is the
                    // form the machine ZEROES, the opposite of the store form
                    // below, which PRESERVES.
                    self.zero_xmm_hi(d);
                    return;
                }
                if ins.op_kind(0) == OpKind::Memory {
                    let s = match self.xmm_op(ins, 1) {
                        Some(s) => s,
                        None => {
                            self.sse_gap(nm, "store from a non-XMM source");
                            return;
                        }
                    };
                    let a = match self.addr_of(ins, 0) {
                        Some(a) => a,
                        None => {
                            self.sse_gap(nm, "store to an address we cannot name");
                            return;
                        }
                    };
                    let bits = self.xmm(s);
                    let low = self.narrow(bits, n);
                    // A float-typed store, so the emitter writes the VALUE's bits
                    // (`rdec_f64_bits`), and a bit-typed source is bitcast to the
                    // float the store's type names. Both halves of that are the
                    // same memcpy; the IR names the type, which is the point.
                    //
                    // The scalar store form touches NEITHER register cell:
                    // `movsd m64, xmm` writes 8 bytes to memory and leaves the
                    // register's pair exactly as it was (Intel). Nothing is
                    // written, so nothing can drift - and `packed_simd.rs`
                    // proves the preservation by storing the full pair after
                    // this store and reading back both lanes.
                    let v = self.cast_ty(CastKind::Bitcast, low, Ty::Float { w: fw });
                    self.f
                        .emit_void(self.cur, Op::Store(Ty::Float { w: fw }, a, v), self.addr());
                    return;
                }
                self.sse_gap(nm, "form the model does not have");
                return;
            }
            "movapd" | "movaps" | "movdqa" | "movdqu" | "movups" | "movupd" => {
                // The full 128-bit data path, per the pair: a move is a pair
                // of copies, a load is two W64 loads (lo at addr, hi at
                // addr+8), a store is two W64 stores at the same split. The
                // aligned and unaligned forms share the model: alignment is
                // the machine's business (an unaligned `movdqa` faults on
                // silicon, and T2 files that crash); inventing fault
                // semantics is not.
                if let Some(d) = self.xmm_op(ins, 0) {
                    match ins.op_kind(1) {
                        OpKind::Register => {
                            let s = match self.xmm_op(ins, 1) {
                                Some(s) => s,
                                None => {
                                    self.sse_gap(nm, "source is not an XMM register");
                                    return;
                                }
                            };
                            // The 128-bit copy: the pair copied cell by cell.
                            let src_lo = self.xmm(s);
                            let src_hi = self.xmm_hi(s);
                            let lo =
                                self.f
                                    .emit(self.cur, Op::Copy(src_lo), Ty::u64(), self.addr());
                            let hi =
                                self.f
                                    .emit(self.cur, Op::Copy(src_hi), Ty::u64(), self.addr());
                            self.set_xmm(d, lo);
                            self.set_xmm_hi(d, hi);
                            self.rep.sse_packed += 1;
                            return;
                        }
                        OpKind::Memory => {
                            let a = match self.addr_of(ins, 1) {
                                Some(a) => a,
                                None => {
                                    self.sse_gap(nm, "load from an address we cannot name");
                                    return;
                                }
                            };
                            self.load16(a, d);
                            return;
                        }
                        _ => {
                            self.sse_gap(nm, "source is neither an XMM register nor memory");
                            return;
                        }
                    }
                }
                if ins.op_kind(0) == OpKind::Memory {
                    let s = match self.xmm_op(ins, 1) {
                        Some(s) => s,
                        None => {
                            self.sse_gap(nm, "store from a non-XMM source");
                            return;
                        }
                    };
                    let a = match self.addr_of(ins, 0) {
                        Some(a) => a,
                        None => {
                            self.sse_gap(nm, "store to an address we cannot name");
                            return;
                        }
                    };
                    // A 128-bit STORE, modelled as the pair deserves: both
                    // cells, lo at addr and hi at addr+8. The half-store the
                    // one-cell model refused (`array_walk`'s `movaps
                    // %xmm0,(%rax)`) left 8 bytes undefined in the model and
                    // real on the machine; with the pair there is no half.
                    self.store16(a, s);
                    return;
                }
                self.sse_gap(nm, "form the model does not have");
                return;
            }
            "movd" | "movq" => {
                let n = if nm == "movd" { Width::W32 } else { Width::W64 };
                if let Some(d) = self.xmm_op(ins, 0) {
                    let v = match self.src(ins, 1, n) {
                        Some(v) => v,
                        None => {
                            self.sse_gap(nm, "no readable source");
                            return;
                        }
                    };
                    if nm == "movq" {
                        // `movq xmm, r/m64` MERGES: the low 64 bits are
                        // written, the destination's high lane survives
                        // (Intel Vol.2A). This is the toolchain's
                        // int-to-XMM-pair move.
                        self.splice_lane(d, v, 0, true);
                    } else {
                        // `movd xmm, r/m32` zero-extends to the full 128
                        // bits: ZExt to cell width is the 63:32 half, the
                        // explicit high-cell zero is the 127:64 half.
                        let up = self.cast(CastKind::ZExt, v, Width::W64);
                        self.set_xmm(d, up);
                        self.zero_xmm_hi(d);
                    }
                    self.rep.sse_packed += 1;
                    return;
                }
                if let Some(s) = self.xmm_op(ins, 1) {
                    let bits = self.xmm(s);
                    if let Some((d, dw)) = self.reg_op(ins, 0) {
                        // xmm -> gpr: the low bits, and the destination's own
                        // write rule (a 32-bit write clears the upper half)
                        // finishes the job.
                        let low = self.narrow(bits, dw);
                        self.write(d, dw, low);
                        return;
                    }
                    if ins.op_kind(0) == OpKind::Memory {
                        let a = match self.addr_of(ins, 0) {
                            Some(a) => a,
                            None => {
                                self.sse_gap(nm, "store to an address we cannot name");
                                return;
                            }
                        };
                        let low = self.narrow(bits, n);
                        self.f
                            .emit_void(self.cur, Op::Store(ty_u(n), a, low), self.addr());
                        return;
                    }
                }
                self.sse_gap(nm, "form the model does not have");
                return;
            }
            _ => {}
        }
        // ---- conversions --------------------------------------------------
        if nm == "cvtsi2sd" || nm == "cvtsi2ss" {
            let (fw, is64) = if nm == "cvtsi2sd" {
                (F64, true)
            } else {
                (F32, false)
            };
            let Some(d) = self.xmm_op(ins, 0) else {
                self.sse_gap(nm, "no XMM destination");
                return;
            };
            // The source is read at ITS width and CONVERTED AS SIGNED: `cvtsi2sd
            // xmm0, eax` reads 32 bits, not the 64-bit register. Reading at W32
            // is the width rule; the signed re-read at that width is what makes
            // -1 convert to -1.0 and not to 4294967295.0.
            let sw = match ins.op_kind(1) {
                OpKind::Register => match self.reg_op(ins, 1) {
                    Some((_, w)) => w,
                    None => {
                        self.sse_gap(nm, "source is not a general-purpose register");
                        return;
                    }
                },
                OpKind::Memory => match Self::w_of_ms(ins.memory_size()) {
                    Some(w) => w,
                    None => {
                        self.sse_gap(nm, "source has no readable size");
                        return;
                    }
                },
                _ => {
                    self.sse_gap(nm, "source is not an integer operand");
                    return;
                }
            };
            let v = match self.src(ins, 1, sw) {
                Some(v) => v,
                None => {
                    self.sse_gap(nm, "no readable integer source");
                    return;
                }
            };
            // `cvtsi2ss` with a 64-bit source is the SSSE3 `cvtsi2ssq`
            // extension: the model speaks SSE1-plus, so the 64-bit source
            // form is refused as a named gap, exactly like the `sd`-named
            // 64-bit MOV forms. The 32-bit source keeps the rule above.
            if !is64 && sw == Width::W64 {
                self.sse_gap(
                    nm,
                    "cvtsi2ss is SSE1 32-bit: REX.W source is cvtsi2ssq (SSSE3)",
                );
                return;
            }
            let sv = self.cast_ty(CastKind::Bitcast, v, Ty::int(sw, rdec_core::ty::Sx::Signed));
            let f = self.cast_ty(CastKind::SIToFp, sv, Ty::Float { w: fw });
            let bits = self.fbits(f);
            // Merge form: cvtsi2sd writes the low lane, the destination's high
            // lane survives (Intel). The single `set_xmm` is the preservation.
            self.set_xmm(d, bits);
            return;
        }
        if matches!(nm, "cvttsd2si" | "cvttss2si") {
            let fw = if nm == "cvttsd2si" { F64 } else { F32 };
            let Some((d, dw)) = self.reg_op(ins, 0) else {
                self.sse_gap(nm, "no integer destination");
                return;
            };
            let v = match self.fsrc(ins, 1, fw) {
                Some(v) => v,
                None => {
                    self.sse_gap(nm, "no readable float source");
                    return;
                }
            };
            // Truncate toward zero, and x86's answer for NaN or an out-of-range
            // source is the destination's sign bit. `FpToSI`'s C spelling is the
            // GUARDED form for that reason: a bare C cast is undefined
            // behaviour exactly where the machine has a defined answer.
            let t = Ty::int(dw, rdec_core::ty::Sx::Signed);
            let cvt = self.cast_ty(CastKind::FpToSI, v, t);
            self.write(d, dw, cvt);
            return;
        }
        if nm == "cvtss2sd" || nm == "cvtsd2ss" {
            let (from, to) = if nm == "cvtss2sd" {
                (F32, F64)
            } else {
                (F64, F32)
            };
            let Some(d) = self.xmm_op(ins, 0) else {
                self.sse_gap(nm, "no XMM destination");
                return;
            };
            let v = match self.fsrc(ins, 1, from) {
                Some(v) => v,
                None => {
                    self.sse_gap(nm, "no readable float source");
                    return;
                }
            };
            let k = if to == F64 {
                CastKind::FPExt
            } else {
                CastKind::FPTrunc
            };
            let r = self.cast_ty(k, v, Ty::Float { w: to });
            let bits = self.fbits(r);
            self.set_xmm(d, bits);
            return;
        }
        if nm == "punpcklqdq" {
            // The lane-pair packing,Dst = (lo lane of Dst, lo lane of src):
            // the low 64 bits of the dest stay, the low 64 bits of the source
            // move to the high lane. This is also the standard
            // float->int/reinterpret trick's second half: the pair's halves
            // are model cells in their own right, and "lane" talks about
            // memory-address lanes, not bytes.
            let Some(d) = self.xmm_op(ins, 0) else {
                self.sse_gap(nm, "no XMM destination");
                return;
            };
            let Some(s) = self.xmm_op(ins, 1) else {
                self.sse_gap(nm, "punpcklqdq needs an XMM source");
                return;
            };
            let slo = self.xmm(s);
            self.set_xmm_hi(d, slo);
            self.rep.sse_packed += 1;
            return;
        }
        if matches!(nm, "cvtsd2si" | "cvtss2si") {
            // The ROUNDING forms: they convert with MXCSR's rounding mode, which
            // the model does not carry. `cvttsd2si` (truncate) is modelled;
            // inventing round-to-nearest here would be a guess about a register
            // the function may have changed.
            self.sse_gap(
                nm,
                "a rounding-mode-dependent conversion; MXCSR is not modelled",
            );
            return;
        }
        if matches!(nm, "ucomisd" | "comisd" | "ucomiss" | "comiss") {
            // The refusal is not laziness about flags. `ucomisd` sets
            // ZF/CF/PF to a THREE-bit answer (below / equal / above-or-equal /
            // unordered), and the `jcc` spellings that read it are combinations
            // (`ja` = CF=0&&ZF=0, `jb` = CF=1, `jae` = ZF=1||CF=0, `jp` = PF=1)
            // whose unordered cases are not one boolean. `ja`/`jb`/`je`/`jne`
            // alone C would spell; `jae`/`jbe`/`jp`/`jnp` would not, and picking
            // the subset that works in the corpus is the guess this project
            // refuses to make. The compare stays a named gap.
            self.sse_gap(
                nm,
                "an ordered float compare: ZF/CF/PF is a three-bit answer and the jcc spellings of it are not one boolean",
            );
            return;
        }
        // Everything else in the family: packed arithmetic (`paddd`, `paddq`,
        // `pmul*`, `addpd`...), shuffles and unpacks (lane-crossing: no W64
        // lane model reaches them), mask reads (`movmskps`), the packed<->packed
        // conversions (`cvtdq2ps`, `cvtpd2ps`, ...), the rounding-mode compares,
        // the non-slice mask moves (`xorps`/`andps`/`andnps`/`orps` and their
        // `pd` siblings), and the MXCSR accesses. All stay refused by name:
        // the refusal census, mnemonic verbatim, is the next lane's map.
        self.sse_gap(
            nm,
            "packed/vector SSE: it reads or writes bits this model does not carry",
        );
    }

    /// The scalar SSE arithmetic, by canonical name. The SUFFIX is the precision,
    /// and the base must be one of these words exactly: that is what keeps
    /// `addpd` (base `addp`) and `cvtsd2si` out of a match that would happily read
    /// them as `add`/`cvt`.
    fn sse_arith(nm: &str) -> Option<(FBinOp, FWidth)> {
        use FWidth::{F32, F64};
        // The SUFFIX IS TWO LETTERS: `addsd` is `add` + `sd`, and stripping one
        // letter reads the base as `adds`, which is in no table. Getting this wrong
        // is not a crash -- it is `addsd` falling through to the packed refusal,
        // which is why the test names the mnemonics it expects to be modelled.
        let (base, fw) = match nm.strip_suffix("sd") {
            Some(b) => (b, F64),
            None => (nm.strip_suffix("ss")?, F32),
        };
        let op = match base {
            "add" => FBinOp::Add,
            "sub" => FBinOp::Sub,
            "mul" => FBinOp::Mul,
            "div" => FBinOp::Div,
            // x86's `maxsd`/`minsd` return the SECOND operand when either side is
            // NaN -- the IR op carries that rule, not C's `fmax`.
            "max" => FBinOp::Max,
            "min" => FBinOp::Min,
            "sqrt" => FBinOp::Sqrt,
            _ => return None,
        };
        Some((op, fw))
    }

    /// The SSE family, by canonical name: everything here reaches `Lifter::sse`,
    /// where the scalar subset is modelled and the rest is refused by name. The
    /// census counts at this gate, so the number the gates print is the number of
    /// SSE instructions the lifter SAW -- modelled and refused together, which is
    /// the only census that can be checked against `llvm-objdump -d`.
    ///
    /// What is NOT in this list falls through to the integer paths and lands in
    /// `unmodelled instruction: <mnemonic>` -- loud and counted there, so a missing
    /// name here is a renamed gap, never a silent one.
    fn sse_family(nm: &str) -> bool {
        // `cvtsi2sd`/`cvtsi2ss`/`cvtss2sd`/`cvtsd2ss` all END in `sd`/`ss`, so the
        // arithmetic matcher must be tried AFTER the conversion names, and the
        // packed forms must be in this list: what is not listed here falls through
        // to the integer paths, where the generic decoder's own
        // `unmodelled instruction: <mnemonic>` gap is loud and counted.
        matches!(
            nm,
            "movsd"
                | "movss"
                | "movapd"
                | "movaps"
                | "movdqa"
                | "movdqu"
                | "movd"
                | "movq"
                | "movups"
                | "movupd"
                | "punpcklqdq"
                | "addsd"
                | "subsd"
                | "mulsd"
                | "divsd"
                | "maxsd"
                | "minsd"
                | "sqrtsd"
                | "addss"
                | "subss"
                | "mulss"
                | "divss"
                | "maxss"
                | "minss"
                | "sqrtss"
                | "addpd"
                | "subpd"
                | "mulpd"
                | "divpd"
                | "maxpd"
                | "minpd"
                | "sqrtpd"
                | "addps"
                | "subps"
                | "mulps"
                | "divps"
                | "maxps"
                | "minps"
                | "sqrtps"
                | "pxor"
                | "xorps"
                | "xorpd"
                | "andps"
                | "andpd"
                | "andnps"
                | "andnpd"
                | "orps"
                | "orpd"
                | "comisd"
                | "ucomisd"
                | "comiss"
                | "ucomiss"
                | "shufps"
                | "shufpd"
                | "unpcklps"
                | "unpckhps"
                | "unpcklpd"
                | "unpckhpd"
                | "movmskps"
                | "movmskpd"
                | "cvtsi2sd"
                | "cvtsi2ss"
                | "cvttsd2si"
                | "cvttss2si"
                | "cvtsd2si"
                | "cvtss2si"
                | "cvtss2sd"
                | "cvtsd2ss"
                | "cvtdq2ps"
                | "cvtps2dq"
                | "cvtpd2ps"
                | "cvtps2pd"
                | "paddd"
                | "paddq"
                | "paddb"
                | "paddw"
                | "psubd"
                | "psubq"
                | "pmuludq"
                | "pmullw"
                | "pmulhw"
                | "pand"
                | "pandn"
                | "por"
                | "pcmpgtd"
                | "pcmpeqd"
                | "pcmpeqq"
                | "pslld"
                | "psrld"
                | "psrad"
                | "movntdq"
                | "movntpd"
                | "movntps"
                | "ldmxcsr"
                | "stmxcsr"
        )
    }

    /// One instruction. Control-transfer instructions set `self.pending`.
    fn one(&mut self, ins: &Instruction) {
        let m = ins.mnemonic();
        let nm = self.name_of(ins);
        let w = self.op_w(ins);

        // A jump-table dispatch, instruction by instruction. See
        // `scan_dispatches`: the slot read's bytes are `.rdata`, not modelled
        // memory, so the four instructions are replaced by the edges the
        // relocations name - and the register the slot read wrote keeps no
        // definition, which is what makes a later read of it a live-in (a counted
        // gap) instead of a value the model does not have.
        match self.skip.get(&self.at).cloned() {
            Some(Step::Lea) | Some(Step::Add) => return,
            Some(Step::Load(idx)) => {
                self.table_index = Some(self.read64(idx));
                return;
            }
            Some(Step::Site(targets)) => {
                let idx = match self.table_index {
                    Some(v) => v,
                    None => {
                        // The scan matched the shape; the sweep did not reach the read
                        // that supplies its index. Say so, keep the refusal.
                        self.gap("jump table dispatch with no reachable slot read", Ty::Void);
                        self.pending = Pending::OpaqueTerm;
                        return;
                    }
                };
                match self.switch_term(self.at, idx, &targets) {
                    Some(t) => {
                        self.table_index = None;
                        self.pending = Pending::Switch(t);
                    }
                    None => {
                        // Not every slot names a body: keep the refusal. The slots
                        // that did resolve stay discovered, and the count says a
                        // table was here and half of it is not at hand.
                        self.table_index = None;
                        self.gap("indirect jmp", Ty::Void);
                        self.pending = Pending::OpaqueTerm;
                    }
                }
                return;
            }
            None => {}
        }

        // SSE, scalar subset. Counted at this gate, before the modelled/refused
        // split, so the census the gates print is "what the decoder saw".
        if Self::sse_family(&nm) {
            *self.rep.sse.entry(nm.clone()).or_default() += 1;
            self.sse(&nm, ins);
            return;
        }

        // Conditional transfers: jcc / setcc / cmovcc, by canonical name.
        if matches!(
            m,
            Mnemonic::Je
                | Mnemonic::Jne
                | Mnemonic::Jl
                | Mnemonic::Jge
                | Mnemonic::Jg
                | Mnemonic::Jle
                | Mnemonic::Ja
                | Mnemonic::Jae
                | Mnemonic::Jb
                | Mnemonic::Jbe
                | Mnemonic::Js
                | Mnemonic::Jns
                | Mnemonic::Jo
                | Mnemonic::Jno
                | Mnemonic::Jp
                | Mnemonic::Jnp
        ) {
            let c = match cond_of(&nm) {
                Some(c) => c,
                None => {
                    self.gap("conditional jump with a condition we do not map", Ty::Bool);
                    self.pending = Pending::OpaqueTerm;
                    return;
                }
            };
            match self.target(ins) {
                Some(t) => self.pending = Pending::Jcc(t, c),
                None => {
                    self.gap("conditional jump with no readable target", Ty::Bool);
                    self.pending = Pending::OpaqueTerm;
                }
            }
            return;
        }

        match m {
            Mnemonic::Nop => {}
            Mnemonic::Lea => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("lea without a register destination", ty_u(w));
                        return;
                    }
                };
                let a = match self.addr_of(ins, 1) {
                    Some(a) => a,
                    None => {
                        // No relocation names a defined symbol at this
                        // displacement: an address the pipeline cannot speak
                        // of. If a relocation IS there, its symbol is the
                        // refusal's identity - the work item with a name.
                        let sfx = self.rip_sym(ins).unwrap_or_default();
                        self.gap(&format!("lea of an address we cannot name{sfx}"), Ty::u64())
                    }
                };
                self.write(d, dw, a);
            }
            Mnemonic::Mov | Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd => {
                if let Some((d, dw)) = self.reg_op(ins, 0) {
                    let sw = self.src_w(ins, 1, dw);
                    let v = match self.src(ins, 1, sw) {
                        Some(v) => v,
                        None => self.gap("mov with an unmodelled source", ty_u(sw)),
                    };
                    let v = match m {
                        Mnemonic::Movzx => self.cast(CastKind::ZExt, v, dw),
                        Mnemonic::Movsx | Mnemonic::Movsxd => self.cast(CastKind::SExt, v, dw),
                        _ => self.narrow(v, dw),
                    };
                    self.write(d, dw, v);
                    return;
                }
                // mov to memory: a store.
                let a = match self.addr_of(ins, 0) {
                    Some(a) => a,
                    None => {
                        let sfx = self.rip_sym(ins).unwrap_or_default();
                        self.gap(&format!("mov to an address we cannot name{sfx}"), Ty::u64());
                        return;
                    }
                };
                let sw = self.mem_w(ins, 0, w);
                let v = match self.src(ins, 1, sw) {
                    Some(v) => self.narrow(v, sw),
                    None => {
                        self.gap("mov store with an unmodelled source", ty_u(sw));
                        return;
                    }
                };
                self.f
                    .emit_void(self.cur, Op::Store(ty_u(sw), a, v), self.addr());
            }
            Mnemonic::Add
            | Mnemonic::Adc
            | Mnemonic::Sub
            | Mnemonic::Sbb
            | Mnemonic::And
            | Mnemonic::Or
            | Mnemonic::Xor => {
                let dst = self.reg_op(ins, 0);
                // The self-idiom is decided BEFORE any operand is read: this is
                // the one place that can know the two operands are the same
                // register, and `xor r,r`/`sub r,r` must not emit a read of it.
                if let Some((d, dw)) = dst {
                    if let Some(si) = SelfIdiom::of(m, ins) {
                        self.self_idiom(d, dw, si);
                        return;
                    }
                }
                let (lhs, rhs, dw, mem_addr) = match dst {
                    Some((d, dw)) => {
                        let lhs = self.read(d, dw);
                        let rhs = match self.src(ins, 1, dw) {
                            Some(v) => v,
                            None => {
                                let g = self.gap("arithmetic with an unmodelled operand", ty_u(dw));
                                self.write(d, dw, g);
                                self.clear_flags();
                                return;
                            }
                        };
                        (lhs, rhs, dw, None)
                    }
                    None => {
                        // Arithmetic on memory: load, operate, store.
                        let a = match self.addr_of(ins, 0) {
                            Some(a) => a,
                            None => {
                                let sfx = self.rip_sym(ins).unwrap_or_default();
                                self.gap(
                                    &format!("arithmetic on an address we cannot name{sfx}"),
                                    Ty::u64(),
                                );
                                return;
                            }
                        };
                        let dw = self.mem_w(ins, 0, w);
                        let t = ty_u(dw);
                        let lhs =
                            self.f
                                .emit(self.cur, Op::Load(t.clone(), a), t.clone(), self.addr());
                        let rhs = match self.src(ins, 1, dw) {
                            Some(v) => v,
                            None => {
                                self.gap("arithmetic with an unmodelled operand", t.clone());
                                self.clear_flags();
                                return;
                            }
                        };
                        (lhs, rhs, dw, Some(a))
                    }
                };
                let bo = match m {
                    Mnemonic::Add | Mnemonic::Adc => BinOp::Add,
                    Mnemonic::Sub | Mnemonic::Sbb => BinOp::Sub,
                    Mnemonic::And => BinOp::And,
                    Mnemonic::Or => BinOp::Or,
                    _ => BinOp::Xor,
                };
                let r = self.bin(bo, lhs, rhs);
                match (dst, mem_addr) {
                    (Some((d, _)), _) => self.write(d, dw, r),
                    (None, Some(a)) => {
                        self.f
                            .emit_void(self.cur, Op::Store(ty_u(dw), a, r), self.addr())
                    }
                    _ => {}
                }
                // ADC/SBB carry in a flag we do not thread: model the carry-less
                // sum, and say so loudly by leaving the flags undefined.
                if matches!(m, Mnemonic::Adc | Mnemonic::Sbb) {
                    self.gap("adc/sbb: carry chain not modelled", ty_u(dw));
                    self.clear_flags();
                } else {
                    self.set_flags(match m {
                        Mnemonic::Add => FlagsOp::Add {
                            a: lhs,
                            b: rhs,
                            carry: false,
                        },
                        Mnemonic::Sub => FlagsOp::Sub {
                            a: lhs,
                            b: rhs,
                            borrow: false,
                        },
                        Mnemonic::And => FlagsOp::And { a: lhs, b: rhs },
                        Mnemonic::Or => FlagsOp::Or { a: lhs, b: rhs },
                        _ => FlagsOp::Xor { a: lhs, b: rhs },
                    });
                }
            }
            Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("shift on an unmodelled destination", ty_u(w));
                        return;
                    }
                };
                let lhs = self.read(d, dw);
                let cnt = match self.src(ins, 1, Width::W8) {
                    Some(v) => self.cast(CastKind::ZExt, v, Width::W64),
                    None => {
                        let g = self.gap("shift with an unmodelled count", ty_u(dw));
                        self.write(d, dw, g);
                        self.clear_flags();
                        return;
                    }
                };
                // x86 masks the count - 5 bits, 6 for 64-bit operands. Modelled,
                // because the hardware table says so and T1 measured it.
                let mask = self.const_of(if dw == Width::W64 { 0x3f } else { 0x1f }, Width::W64);
                let masked = self.bin(BinOp::And, cnt, mask);
                let cnt = self.narrow(masked, dw);
                let bo = match m {
                    Mnemonic::Shl => BinOp::Shl,
                    Mnemonic::Shr => BinOp::Shr,
                    _ => BinOp::Sar,
                };
                let r = self.bin(bo, lhs, cnt);
                self.write(d, dw, r);
                self.set_flags(match m {
                    Mnemonic::Shl => FlagsOp::Shl { a: lhs, b: cnt },
                    Mnemonic::Shr => FlagsOp::Shr { a: lhs, b: cnt },
                    _ => FlagsOp::Sar { a: lhs, b: cnt },
                });
            }
            Mnemonic::Inc | Mnemonic::Dec => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        // `inc`/`dec` on a MEMORY destination: read the cell,
                        // add/subtract one, write the SAME address back, and set
                        // the same flags as the register form. Measured
                        // witnesses: 3 sites, every one `incl (%rcx)`, every one
                        // in an MSVC object — clang lowers the same C as
                        // `addl $1, (%rcx)`, which is modelled, so a
                        // clang-only census calls this arm dead code (and
                        // objdump prints it as `incl`, not `inc`).
                        //
                        // The flags are the load-bearing claim, not the
                        // arithmetic: `inc`/`dec` PRESERVE CF (the ices-eye view
                        // and the SDM agree, and the fixture pins it with
                        // `stc ; incl (%rcx) ; setc al`, which the machine
                        // answers 1), so FlagsOp::Inc/Dec — which write
                        // SF/ZF/OF/AF/PF and leave CF alone — is the same
                        // operator as the register form uses. A `add $1`
                        // modelling of the same bytes would clobber CF and
                        // change a later `setc`/`adc` answer.
                        //
                        // `addr_of` returns Some ONLY for a memory operand it
                        // can express (None for RIP without a relocation, None
                        // for no base), so a register destination can never
                        // reach this arm and be read through memory.
                        match self.addr_of(ins, 0) {
                            Some(a) => {
                                let t = ty_u(w);
                                let lhs =
                                    self.f
                                        .emit(self.cur, Op::Load(t.clone(), a), t, self.addr());
                                let one = self.const_of(1, w);
                                let r = self.bin(
                                    if m == Mnemonic::Inc {
                                        BinOp::Add
                                    } else {
                                        BinOp::Sub
                                    },
                                    lhs,
                                    one,
                                );
                                self.f
                                    .emit_void(self.cur, Op::Store(ty_u(w), a, r), self.addr());
                                let cf_in = if self.flags_block == self.cur {
                                    self.flags
                                } else {
                                    None
                                };
                                self.set_flags(if m == Mnemonic::Inc {
                                    FlagsOp::Inc { a: lhs, cf_in }
                                } else {
                                    FlagsOp::Dec { a: lhs, cf_in }
                                });
                            }
                            None => {
                                self.gap("inc/dec on an unmodelled destination", ty_u(w));
                            }
                        }
                        return;
                    }
                };
                let lhs = self.read(d, dw);
                let one = self.const_of(1, dw);
                let r = self.bin(
                    if m == Mnemonic::Inc {
                        BinOp::Add
                    } else {
                        BinOp::Sub
                    },
                    lhs,
                    one,
                );
                self.write(d, dw, r);
                let cf_in = if self.flags_block == self.cur {
                    self.flags
                } else {
                    None
                };
                self.set_flags(if m == Mnemonic::Inc {
                    FlagsOp::Inc { a: lhs, cf_in }
                } else {
                    FlagsOp::Dec { a: lhs, cf_in }
                });
            }
            Mnemonic::Neg => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("neg on an unmodelled destination", ty_u(w));
                        return;
                    }
                };
                let lhs = self.read(d, dw);
                let r = self
                    .f
                    .emit(self.cur, Op::Un(UnOp::Neg, lhs), ty_u(dw), self.addr());
                self.write(d, dw, r);
                self.set_flags(FlagsOp::Neg { a: lhs });
            }
            Mnemonic::Not => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("not on an unmodelled destination", ty_u(w));
                        return;
                    }
                };
                let lhs = self.read(d, dw);
                let r = self
                    .f
                    .emit(self.cur, Op::Un(UnOp::Not, lhs), ty_u(dw), self.addr());
                self.write(d, dw, r);
                // `not` leaves EFLAGS untouched: keep the previous flags value.
            }
            // ---- the ACCUMULATOR family: the implicit `RDX:RAX` pair. Every
            // C `/` and `%` compiles to one of these, so refusing them refuses
            // the function. All expectations in
            // crates/arch-x86/tests/acc_forms.rs are silicon measurements (the
            // probe recipe is in that file's header), including the EFLAGS
            // claims below.
            Mnemonic::Cwd | Mnemonic::Cdq | Mnemonic::Cqo => {
                // Sign-replicate the accumulator into the high half of the
                // dividend pair: `EDX = EAX >> 31` is the whole instruction
                // (and `RDX = RAX >> 63` for `cqo` — the shift count is the
                // width, which is why the 32-bit form passing does not prove
                // the 64-bit one).
                //
                // NO flag is touched, and that is a measured claim, not an
                // omission: `cmp ecx,edx ; cdq ; setc al` reads the `cmp`'s CF
                // across the `cdq` on the machine. A lifter that treats the
                // sign-replicate as flag-clearing answers 0 for every `setc`
                // leg, so the fixture keeps a flag read in the sequence.
                let (dw, count) = match m {
                    Mnemonic::Cwd => (Width::W16, 15u64),
                    Mnemonic::Cdq => (Width::W32, 31),
                    _ => (Width::W64, 63),
                };
                let a = self.read(Grp::Rax, dw);
                // The shift count carries the OPERATION's width: the verifier's
                // width-homogeneity rule rejects a node mixing W32 operands with
                // a W64 count, and the rule is right — a mixed-width node is a
                // statement about no real machine width.
                let sh = self.const_of(count, dw);
                let hi = self.bin(BinOp::Sar, a, sh);
                self.write(Grp::Rdx, dw, hi);
            }
            Mnemonic::Cwde => {
                // `EAX = (int32_t)(int16_t)AX` — the sign-replicate that writes
                // INTO the accumulator, unlike cwd/cdq/cqo above which write the
                // high half of a dividend pair. Two machine facts decide the
                // shape, both measured by executing the bytes on this host
                // (VirtualAlloc + PAGE_EXECUTE_READWRITE, scratch/
                // sign_extend_probe.c), and the fixture pins both:
                //   - it reads ONLY its own 16-bit cell. EAX=0xFFFF0005 gives
                //     EAX=0x5 — not 0xFFFFF005 (extending the 32-bit cell) and
                //     not 0xFFFF...F005. A lifter that sign-extends from W32
                //     passes the 0x12348123 case and fails this one, which is
                //     why the case is in the table.
                //   - the 32-bit write ZEROES bits 63:32: EAX=0x12348123 leaves
                //     RAX=0xFFFF8123, so the ZExt on write is architecture, not
                //     a choice.
                // NO flag is touched (same measured claim as cdq: a `cmp` before
                // it is read by a `setc` after it).
                //
                // The witnesses are MSVC-only: clang lowers the same C with
                // `movsx eax,ax`, which is modelled, so a clang-only corpus shows
                // this arm as dead code. And llvm-objdump prints the byte as
                // `cwtl`, not `cwde` — an Intel-mnemonic census over objdump
                // output counts ZERO sites and makes a live instruction look
                // unexercised.
                let a = self.read(Grp::Rax, Width::W16);
                let v = self.cast(CastKind::SExt, a, Width::W32);
                self.write(Grp::Rax, Width::W32, v);
            }
            Mnemonic::Mul | Mnemonic::Imul | Mnemonic::Div | Mnemonic::Idiv
                if ins.op_count() == 1 =>
            {
                // F7 /4 /5 /6 /7 with ONE operand: the operand is the divisor
                // (or multiplier) and the other operand is the accumulator
                // pair. The operand's OWN width sets the size of everything —
                // `imul rcx` is a 64-bit pair, `idiv r8d` a 32-bit one.
                let signed = matches!(m, Mnemonic::Imul | Mnemonic::Idiv);
                let divide = matches!(m, Mnemonic::Div | Mnemonic::Idiv);
                let aw = self.src_w(ins, 0, w);
                let b = match self.src(ins, 0, aw) {
                    Some(v) => v,
                    None => {
                        self.gap("accumulator form with an unmodelled operand", ty_u(aw));
                        self.clear_flags();
                        return;
                    }
                };
                let a = self.read(Grp::Rax, aw);
                match (divide, aw) {
                    (false, Width::W64) => {
                        // The 128-bit product's halves: the low half is the
                        // ordinary `Mul`, the high half is `MulHi`/`MulSmul`,
                        // which the interpreter computes exactly and the C
                        // emitter spells as a portable 32x32-limb decomposition
                        // (no `__int128`, so the C compiles where the model's
                        // u64 cells do).
                        let lo = self.bin(BinOp::Mul, a, b);
                        let hi = self.bin(if signed { BinOp::MulSmul } else { BinOp::MulHi }, a, b);
                        self.write(Grp::Rax, Width::W64, lo);
                        self.write(Grp::Rdx, Width::W64, hi);
                        self.set_flags(FlagsOp::Mul { a, b, signed });
                    }
                    (false, Width::W32) => {
                        // 32x32 fits in 64 bits under EITHER reading, so the
                        // full product is EXACT in the model's W64 cell: widen
                        // both halves with the mnemonic's own signedness,
                        // multiply once, and the pair's halves are the low 32
                        // bits and bits 32..63 of that product. Signed 32x32
                        // and unsigned 32x32 differ ONLY in the high half, and
                        // this spelling is where that difference lives.
                        let k = if signed {
                            CastKind::SExt
                        } else {
                            CastKind::ZExt
                        };
                        let pa = self.cast(k, a, Width::W64);
                        let pb = self.cast(k, b, Width::W64);
                        let p = self.bin(BinOp::Mul, pa, pb);
                        let lo = self.narrow(p, Width::W32);
                        let sh32 = self.const_of(32, Width::W64);
                        let hi = self.bin(BinOp::Shr, p, sh32);
                        let hi32 = self.narrow(hi, Width::W32);
                        self.write(Grp::Rax, Width::W32, lo);
                        self.write(Grp::Rdx, Width::W32, hi32);
                        // EFLAGS: `mul`/`imul` DO define CF/OF (the carry out of
                        // the pair) and leave SF/ZF/AF/PF undefined. The model's
                        // `FlagsOp::Mul` rule compares a high and low half of the
                        // SAME width, and this pair's halves live in a W64 cell,
                        // so reusing it here would compare the wrong bits and
                        // agree with the machine on nothing. Undefined is the
                        // machine's own answer for four of the five flags, and
                        // for CF/OF it is the honest one: a function that reads a
                        // flag after a 32-bit multiply hits the existing loud
                        // "flags undefined" refusal at the READ site instead of
                        // quietly getting a bit nobody computed.
                        self.clear_flags();
                    }
                    (true, Width::W32) => {
                        // The pair IS the dividend: (EDX << 32) | EAX as one
                        // 64-bit value. `cdq` put the sign of EAX in EDX, which
                        // is why the signed form's pair read as a signed 64-bit
                        // dividend equals the signed 32-bit value it came from.
                        // The mnemonic's signedness lives in the DIVISION, not
                        // the assembly of the pair — `div` and `idiv` on the
                        // same bytes differ only in how the pair and the divisor
                        // are read.
                        let hi = self.read(Grp::Rdx, Width::W32);
                        let hi64 = self.cast(CastKind::ZExt, hi, Width::W64);
                        let a64 = self.cast(CastKind::ZExt, a, Width::W64);
                        let sh32 = self.const_of(32, Width::W64);
                        let shifted = self.bin(BinOp::Shl, hi64, sh32);
                        let pair = self.bin(BinOp::Or, shifted, a64);
                        let d64 = self.cast(
                            if signed {
                                CastKind::SExt
                            } else {
                                CastKind::ZExt
                            },
                            b,
                            Width::W64,
                        );
                        let q = self.bin(if signed { BinOp::SDiv } else { BinOp::UDiv }, pair, d64);
                        let r = self.bin(if signed { BinOp::SRem } else { BinOp::URem }, pair, d64);
                        let q32 = self.narrow(q, Width::W32);
                        let r32 = self.narrow(r, Width::W32);
                        self.write(Grp::Rax, Width::W32, q32);
                        self.write(Grp::Rdx, Width::W32, r32);
                        // `div`/`idiv` leave ALL flags undefined on the machine.
                        self.clear_flags();
                        // NOT modelled, and deliberately so: a quotient that
                        // does not fit EAX is a #DE fault on the machine (the
                        // classic divide-by-zero, and `INT_MIN / -1`), and the
                        // model answers with the truncated quotient instead. The
                        // C side guards a zero divisor; overflow stays a named
                        // limitation in the roadmap, not a silent 0.
                    }
                    (true, Width::W64) => {
                        // A 64-bit `div`/`idiv` divides a 128-bit `rdx:rax`. The
                        // model's widest integer cell is 64 bits, so the honest
                        // answer is a refusal that names the missing type — the
                        // map for the slice that adds it. Modelling it as a
                        // 64-bit division would answer every real 128-bit
                        // dividend wrong while the 32-bit cases above kept
                        // passing, which is the failure mode a gate must not
                        // allow.
                        self.gap(
                            &format!(
                                "{m:?} r/m64: the 128-bit rdx:rax dividend needs a 128-bit integer type"
                            ),
                            ty_u(Width::W64),
                        );
                        self.clear_flags();
                    }
                    _ => {
                        self.gap(
                            &format!("{m:?} on an 8/16-bit accumulator pair is not modelled"),
                            ty_u(aw),
                        );
                        self.clear_flags();
                    }
                }
            }
            Mnemonic::Cmp | Mnemonic::Test => {
                let w2 = self.op_w(ins);
                let lhs = match self.src(ins, 0, w2) {
                    Some(v) => v,
                    None => match self.addr_of(ins, 0) {
                        // A flag-setting op whose DESTINATION is memory:
                        // `cmpl %eax,(%rcx)`, `testb $1,(%rcx)` - Intel's first
                        // operand, which llvm-objdump's AT&T output prints LAST,
                        // so an Intel-order grep finds none of these (the same
                        // name trap as `cwde` and `incl`). Measured census over
                        // `corpus/msvc/obj`: 34 `cmp` sites with a memory
                        // destination (`cmpl %eax,0x20(%rsp)` x13,
                        // `cmpl %eax,(%rsp)` x8, `cmpl $0x0,0x20(%rsp)` x3, ...),
                        // the reason the leg's census printed only 3 - the census
                        // records the FIRST refusal per function, and most of
                        // these functions refuse for a different reason first.
                        //
                        // The load is the cell the machine compares, and the
                        // ORDER is the whole claim: `cmp m, r` computes m - r, so
                        // the load must be the LEFT operand of the flag op.
                        // Swapping operands - the tempting simplification -
                        // computes r - m and flips CF/SF/OF, taking every `ja`
                        // and `setb` downstream to the mirror image. Silicon
                        // pins the pair: cell 5 vs eax 7 answers CF=1, cell 7 vs
                        // eax 5 answers CF=0 (`cmp_mem_probe.c`, F1=13, F2=7).
                        //
                        // `addr_of` again refuses what it cannot express (RIP
                        // with no relocation, no base), so an unaddressable
                        // destination stays refused rather than comparing an
                        // invented cell.
                        Some(a) => {
                            let t = ty_u(w2);
                            self.f
                                .emit(self.cur, Op::Load(t.clone(), a), t, self.addr())
                        }
                        None => {
                            // Two DIFFERENT shortages live behind `addr_of`
                            // returning None, and the census must not call them
                            // one thing:
                            //   * the addressing form is genuinely not expressible
                            //     (no base, or RIP with no relocation at all), and
                            //   * the relocation IS there and names a symbol, but
                            //     the model has no address for that target - the
                            //     leg's relocation map places CODE sections, so a
                            //     compare against a static in `.data`/`.bss` has a
                            //     target with no address.
                            // Measured on `array_walk.O2.obj :: f @ 0x2f`
                            // (`cmpl $0x2, (%rip)`): op_kind(0)=Memory,
                            // memory_base=RIP, the map holds a relocation at 0x31
                            // inside the 7-byte span - and its `addr` is None. The
                            // old key called that "an unmodelled first operand",
                            // which points the work queue at the lifter when the
                            // missing piece is an address-space decision.
                            let named_no_address = self.rip_reloc(ins).is_some();
                            self.gap(
                                if named_no_address {
                                    "flag setting against a cell whose relocation names no address"
                                } else {
                                    "flag setting with an unmodelled first operand"
                                },
                                ty_u(w2),
                            );
                            self.clear_flags();
                            return;
                        }
                    },
                };
                let rhs = match self.src(ins, 1, w2) {
                    Some(v) => v,
                    None => {
                        self.gap("flag setting with an unmodelled second operand", ty_u(w2));
                        self.clear_flags();
                        return;
                    }
                };
                self.set_flags(if m == Mnemonic::Cmp {
                    FlagsOp::Sub {
                        a: lhs,
                        b: rhs,
                        borrow: false,
                    }
                } else {
                    FlagsOp::And { a: lhs, b: rhs }
                });
            }
            Mnemonic::Imul => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("imul with an unmodelled destination", ty_u(w));
                        self.clear_flags();
                        return;
                    }
                };
                let (a, b) = if ins.op_count() == 3 {
                    let a = match self.src(ins, 1, dw) {
                        Some(v) => self.narrow(v, dw),
                        None => self.gap("imul with an unmodelled operand", ty_u(dw)),
                    };
                    let b = match self.src(ins, 2, dw) {
                        Some(v) => self.narrow(v, dw),
                        None => self.gap("imul with an unmodelled operand", ty_u(dw)),
                    };
                    (a, b)
                } else {
                    let a = self.read(d, dw);
                    let b = match self.src(ins, 1, dw) {
                        Some(v) => v,
                        None => self.gap("imul with an unmodelled operand", ty_u(dw)),
                    };
                    (a, b)
                };
                let r = self.bin(BinOp::Mul, a, b);
                self.write(d, dw, r);
                self.set_flags(FlagsOp::Mul { a, b, signed: true });
            }
            Mnemonic::Push => {
                let eight = self.const_of(8, Width::W64);
                let sp = self.read64(Grp::Rsp);
                let nsp = self.bin(BinOp::Sub, sp, eight);
                self.set_reg(Grp::Rsp, nsp);
                let v = match self.src(ins, 0, Width::W64) {
                    Some(v) => v,
                    None => {
                        self.gap("push of an unmodelled value", Ty::u64());
                        return;
                    }
                };
                self.f
                    .emit_void(self.cur, Op::Store(Ty::u64(), nsp, v), self.addr());
            }
            Mnemonic::Pop => {
                let sp = self.read64(Grp::Rsp);
                let v = self
                    .f
                    .emit(self.cur, Op::Load(Ty::u64(), sp), Ty::u64(), self.addr());
                let eight = self.const_of(8, Width::W64);
                let nsp = self.bin(BinOp::Add, sp, eight);
                self.set_reg(Grp::Rsp, nsp);
                match self.reg_op(ins, 0) {
                    Some((d, _)) => self.write(d, Width::W64, v),
                    None => {
                        self.gap("pop to an unmodelled destination", Ty::u64());
                    }
                }
            }
            Mnemonic::Jmp => {
                if ins.op_kind(0) == OpKind::Register || ins.op_kind(0) == OpKind::Memory {
                    self.f
                        .emit_void(self.cur, Op::Opaque("indirect jmp".into()), self.addr());
                    self.rep.gaps += 1;
                    self.pending = Pending::OpaqueTerm;
                    return;
                }
                // A DIRECT `jmp`. In an unlinked object its target is the
                // relocation at the displacement, not the immediate the bytes
                // hold: `tc.O2.obj` has `jmp 0x17` at 0x12 - 0x17 is the
                // unrelocated next-IP - with `IMAGE_REL_AMD64_REL32 g` at 0x13,
                // and `llvm-nm` says `T g` at 0x0. A relocation naming a DEFINED
                // function symbol is a TAIL CALL: the callee gets the argument
                // registers the caller already filled per the pair rule, its
                // answer is the caller's answer, and control never comes back -
                // `return callee(args);`. A `jmp` with NO relocation keeps its
                // immediate as the evidence, which is every `jmp` in the corpus:
                // 65 sites, 0 of them relocated, every one a loop back edge or an
                // if/else join inside the function that holds it.
                if let Some(rt) = self.reloc_on(ins) {
                    let callee = rt.addr.and_then(|a| match self.names.get(&a) {
                        Some(raw) => self.sigs.get(raw).map(|s| (a, raw.clone(), s.noreturn)),
                        None => None,
                    });
                    match callee {
                        Some((_, _, true)) => {
                            // `jmp abort`: control leaves the function through a
                            // callee that never returns. No value, no fall-through.
                            let sym = rt.sym;
                            self.noreturn_site(ins, &sym);
                        }
                        Some((a, _, false)) => {
                            if self.bb_of.contains_key(&a) {
                                // The named address is a block of THIS function:
                                // control flow inside it, not a call.
                                self.pending = Pending::Jump(a);
                            } else {
                                self.emit_call_site(ins, a, Site::Tail);
                            }
                        }
                        // A `jmp` whose relocation names a symbol with no address
                        // (an import) or no declared shape: the name is evidence,
                        // the callee is not callable, and the `Exit` stays.
                        None => {
                            self.rep.tail_calls_unnamed += 1;
                            self.gap(
                                &format!(
                                    "tail call to '{}': the object gives no callable callee for the symbol the relocation named",
                                    rt.sym
                                ),
                                Ty::Void,
                            );
                            self.pending = Pending::OpaqueTerm;
                        }
                    }
                    return;
                }
                match self.target(ins) {
                    Some(t) => self.pending = Pending::Jump(t),
                    None => {
                        self.gap("jmp with no readable target", Ty::Void);
                        self.pending = Pending::OpaqueTerm;
                    }
                }
            }
            Mnemonic::Call => {
                // A call destroys the caller's scratch registers: one
                // `Op::Clobber` definition per caller-saved register, because
                // leaving a value live across a call is a silent miscompile. The
                // flags are clobbered too, and their cell is `None` (see
                // `clear_flags`) — an `Op::Flags` there would let a compare before
                // the call answer a flag after it.
                //
                // What is modelled on TOP of that: a DIRECT call to a target the
                // pipeline names AND types is also a VALUE — the callee's answer,
                // defined by the call, landing in the register its declared
                // `RetKind` says carries it. The identity's evidence is the COFF
                // symbol table at the target address; the shape's is the declared
                // signature the front end resolved for that symbol.
                if ins.op_kind(0) == OpKind::Register || ins.op_kind(0) == OpKind::Memory {
                    // No target, so no signature, so no answer. Loud as it was.
                    self.f
                        .emit_void(self.cur, Op::Opaque("indirect call".into()), self.addr());
                    self.rep.gaps += 1;
                    self.rep.calls_indirect += 1;
                    self.clobber_all(None);
                    self.clear_flags();
                    return;
                }
                let t = self.target(ins).unwrap_or(0);
                // The relocation table speaks at the address of the displacement
                // field. clang's COFF objects leave the UNRELOCATED next-IP in a
                // `call`'s rel32 and name the real target in the REL32 relocation
                // AT THAT FIELD (`callq 0x15` at 0x10, relocation at 0x11 naming
                // `h` at 0x50), so the immediate is never the callee's address:
                // reading only `target()` is precisely why `calls: 0 modelled`.
                // iced_x86 reports a 64-bit-mode direct call as `NearBranch64`
                // (verified against the decoder on `call_helper.O0.obj`);
                // the ENCODING is rel32, and a rel32 displacement is the
                // instruction's last 4 bytes, so the relocation's address is
                // `next_ip - 4`.
                let t = match ins.op_kind(0) {
                    OpKind::NearBranch16 => self.resolve_reloc_target(ins, t, 2, true),
                    OpKind::NearBranch32 | OpKind::NearBranch64 => {
                        self.resolve_reloc_target(ins, t, 4, true)
                    }
                    _ => t,
                };
                // A call to a callee the front end DECLARED noreturn ends the
                // block. `nr.O0.obj` is the ground truth: `callq` at 0x49 with
                // `IMAGE_REL_AMD64_REL32 abort` at 0x4a, and 0x4e is `int3` - not
                // the caller's code. The name comes from the RELOCATION, which is
                // also the only name an import has.
                if let Some(rt) = self.reloc_on(ins) {
                    if self.sigs.get(rt.sym.as_str()).is_some_and(|s| s.noreturn) {
                        let sym = rt.sym;
                        self.noreturn_site(ins, &sym);
                        return;
                    }
                    // The relocation names a symbol this object does NOT define
                    // (`U abort` in `llvm-nm`): an import. There is no address for
                    // the callee, so the address-keyed lookup that gives a call its
                    // declared shape is silent, and the call defines no answer -
                    // the model does not answer a value whose callee it cannot see.
                    // The block then ENDS: the pipeline has no body for the callee,
                    // so it cannot certify that control comes back to the next
                    // instruction at all (the same bytes can be `printf` or `abort`,
                    // and the object does not say which). The `Exit` is the loud
                    // shape - the same shape the tail-call arm gives a `jmp` whose
                    // symbol has no callable callee, and for the same reason: the
                    // relocation is a name, not a callee. What is NOT done here is
                    // a noreturn model: `noreturn` is a declaration
                    // (`CalleeSig::noreturn`), and an undeclared callee keeps the
                    // gap and the `Exit` rather than an invented end-of-function.
                    if rt.addr.is_none() {
                        let sym = rt.sym;
                        self.import_site(ins, &sym);
                        return;
                    }
                }
                self.emit_call_site(ins, t, Site::Call);
            }
            Mnemonic::Ret => {
                // A float/double answer leaves in XMM0. Counted at the `ret`,
                // where the register map is this block's: XMM0 written, RAX not.
                // Not an IR gap -- the emitted C returns the integer RAX, which
                // is what T2 compares -- but the C++ gate prints the number,
                // because "N of these functions answer in XMM0" is the reason
                // their C looks like it returns nothing, and a reader should not
                // have to infer that from the disassembly.
                if self.regs.contains_key(&Grp::Xmm(0)) && !self.regs.contains_key(&Grp::Rax) {
                    self.rep.ret_xmm0 += 1;
                }
                self.pending = Pending::Ret;
            }
            _ if nm.starts_with("set") => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("setcc without a register destination", ty_u(w));
                        return;
                    }
                };
                let c = match cond_of(&nm) {
                    Some(c) => c,
                    None => {
                        self.gap("setcc with a condition we do not map", Ty::Bool);
                        return;
                    }
                };
                let cond = match self.cond(c) {
                    Some(v) => v,
                    None => self.gap("setcc reading undefined flags", Ty::Bool),
                };
                let one = self.const_of(1, Width::W8);
                let zero = self.const_of(0, Width::W8);
                let v = self.f.emit(
                    self.cur,
                    Op::Select(cond, one, zero),
                    ty_u(Width::W8),
                    self.addr(),
                );
                self.write(d, dw, v);
            }
            _ if nm.starts_with("cmov") => {
                let (d, dw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("cmovcc without a register destination", ty_u(w));
                        return;
                    }
                };
                let c = match cond_of(&nm) {
                    Some(c) => c,
                    None => {
                        self.gap("cmovcc with a condition we do not map", Ty::Bool);
                        return;
                    }
                };
                let cond = match self.cond(c) {
                    Some(v) => v,
                    None => self.gap("cmovcc reading undefined flags", Ty::Bool),
                };
                let old = self.read(d, dw);
                let new = match self.src(ins, 1, dw) {
                    Some(v) => v,
                    None => {
                        self.gap("cmovcc with an unmodelled source", ty_u(dw));
                        return;
                    }
                };
                let v = self
                    .f
                    .emit(self.cur, Op::Select(cond, new, old), ty_u(dw), self.addr());
                self.write(d, dw, v);
            }
            Mnemonic::Xchg => {
                let (a, aw) = match self.reg_op(ins, 0) {
                    Some(x) => x,
                    None => {
                        self.gap("xchg with an unmodelled operand", ty_u(w));
                        return;
                    }
                };
                let (b, bw) = match self.reg_op(ins, 1) {
                    Some(x) => x,
                    None => {
                        self.gap("xchg with an unmodelled operand", ty_u(w));
                        return;
                    }
                };
                let va = self.read(a, aw);
                let vb = self.read(b, bw);
                let vba = self.narrow(vb, aw);
                let vab = self.narrow(va, bw);
                self.write(a, aw, vba);
                self.write(b, bw, vab);
            }
            _ => {
                // Loud, counted, and it blocks DCE.
                let text = format!("unmodelled instruction: {nm}");
                match self.reg_op(ins, 0) {
                    Some((d, dw)) => {
                        let g = self.gap(&text, ty_u(dw));
                        self.write(d, dw, g);
                    }
                    None => {
                        self.f
                            .emit_void(self.cur, Op::Opaque(text.into()), self.addr());
                        self.rep.gaps += 1;
                    }
                }
                self.clear_flags();
            }
        }
    }

    /// Lift a whole function: build the block table, then each block, then its
    /// terminator from the block's own last instruction.
    pub fn lift_function(&mut self, bytes: &[u8], base: u64, blocks: &[BlockInsns]) -> Function {
        for (i, b) in blocks.iter().enumerate() {
            let id = if i == 0 {
                self.f.entry
            } else {
                self.f.new_block(b.start)
            };
            self.bb_of.insert(b.start, id);
            self.order.push(b.start);
        }
        // Which instructions this function's resolved jump tables consume. Set
        // before any block is lifted, because the sweep has to skip a dispatch's
        // `lea`/`movslq`/`add` at the moment it meets them, and the block that
        // holds the `jmp` is not necessarily the first.
        self.skip = self.scan_dispatches(bytes, base, blocks);
        for (i, b) in blocks.iter().enumerate() {
            let id = match self.bb_of.get(&b.start) {
                Some(x) => *x,
                None => continue,
            };
            self.cur = id;
            self.pending = Pending::Fall;
            self.ended_block = false;
            if i > 0 {
                // Each block starts with an EMPTY register map. Carrying the map
                // over in address order (the first version) is what produced
                // uses that no dominating def reached: the value was defined in
                // a block that is not the predecessor, and no phi existed.
                self.regs.clear();
            }
            self.cur_defs.clear();
            if i == 0 {
                // The entry block DEFINES the registers the ABI and the frame
                // model hand it: under the integer model RCX/RDX/R8/R9 carry
                // the four integer arguments; under a declared signature each
                // kind seeded its own class (`seeded` lists exactly those),
                // and RSP is the modelled stack base. They are seeded, never
                // written by an instruction, so nothing else marks them - and
                // a register `block_def` does not list is invisible to the
                // transfer function, which then hands every block after the
                // entry the entry's INCOMING value for it: `undef`. That is
                // the defect behind most of T2's "returns RAX unwritten"
                // bucket - in `early_returns.O1`, `mov eax, edx` at 0x4 is the
                // source's `return b`, and the IR read RDX as undefined.
                for g in self.seeded.clone() {
                    self.cur_defs.insert(g);
                }
            }
            for (va, len) in b.insns.iter() {
                if self.ended_block {
                    // This block's terminator already left the function (tail
                    // call, noreturn call, call to a symbol the object does not
                    // define). The bytes the block table put after it are padding
                    // or the next codelet, not this function's code: count them,
                    // say so, and do not lift them. Lifting `mov eax,0x190; ret`
                    // after `call abort` is the silent half-truth this guard
                    // exists to refuse.
                    self.rep.bytes_past_ended_block += 1;
                    eprintln!(
                        "bytes past the end of bb{id} in {}: instruction at {va:#x} not lifted (the block's terminator left the function)",
                        self.f.name
                    );
                    continue;
                }
                self.at = *va;
                let off = (*va).saturating_sub(base) as usize;
                if off + *len > bytes.len() {
                    self.gap("instruction runs past the section", Ty::Void);
                    continue;
                }
                let mut d = Decoder::with_ip(
                    64,
                    &bytes[off..off + *len],
                    *va,
                    iced_x86::DecoderOptions::NONE,
                );
                let mut ins = iced_x86::Instruction::default();
                d.decode_out(&mut ins);
                if ins.code() as u16 == 0 {
                    // Code::INVALID: the bytes in this block are not an
                    // instruction. Loud, and the block keeps going.
                    self.gap("decoder refused its own bytes", Ty::Void);
                } else {
                    self.one(&ins);
                }
            }
            self.close(i);
            self.block_out.insert(id, self.regs.clone());
            self.block_def.insert(id, self.cur_defs.clone());
        }
        self.ssa_registers();
        self.record_return_width();
        // Take, do not move: the caller reads `rep` after this returns, and a
        // report snapshotted before lifting would always claim zero gaps.
        std::mem::replace(&mut self.f, Function::new("", 0))
    }

    /// Record the width at which this function's answer reaches `RAX`.
    ///
    /// A 32-bit register write zeroes bits 63:32 architecturally, and `write`
    /// above models that write as `zext` - so when every returned value provably
    /// fits in 32 bits, `RAX` IS `zero_extend_32(value)` on every path. The
    /// emitted C has to return `uint32_t` to match it: an `int64_t` return that
    /// sign-extends the same 32 bits is an over-extension in the OUTPUT, and T2
    /// measured it - saturation's `f(1,-100)` gives machine `RAX = 0xffffff9d`
    /// against the C's `0xffffffffffffff9d`, 39 of 732 runs.
    ///
    /// The width is read off the IR the machine code produced, never off observed
    /// answers: a width inferred from answers is the gate that collapses to one
    /// bit and can never fail.
    fn record_return_width(&mut self) {
        // A declared float return already owns `ret_ty` (F32/F64): the
        // integer-width question is about RAX, and XMM0's answer is typed by
        // the declaration, not by a bits analysis of integer writes.
        if self.f.ret_ty.is_float() {
            return;
        }
        if rdec_core::width::return_width(&self.f) == 32 {
            self.f.ret_ty = ty_u(Width::W32);
            self.rep.ret_w32 += 1;
        }
    }

    /// Turn the per-block register transfer into SSA: phi at joins and at
    /// back-edge-only headers, incoming values solved to a fixed point over the
    /// CFG edges, live-in placeholders rewritten inside the block that read them.
    fn ssa_registers(&mut self) {
        // Blocks the entry cannot reach. Their instructions - every def in them -
        // are lifted artifacts the CFG says no path executes, and the verifier
        // rejects a definition in an unreachable block, correctly. Empty them,
        // exit loudly, count each: the bytes exist in the object, and reaching
        // them is a discovery/structuring question (M5), not a thing to hide by
        // leaving dead defs in the IR.
        let first = rdec_core::cfg::Cfg::build(&self.f);
        let unreach: HashSet<BbId> = first.unreachable().into_iter().collect();
        for i in 0..self.f.blocks.len() {
            let id = self.f.blocks[i].id;
            if !unreach.contains(&id) {
                continue;
            }
            let n = self.f.block(id).instrs.len();
            self.rep.gaps += 1;
            self.rep.unreachable_blocks += 1;
            self.f.block_mut(id).instrs.clear();
            self.f.block_mut(id).term = Terminator::Exit;
            if n > 0 {
                eprintln!(
                    "unreachable bb{id} in {}: {n} lifted instructions dropped",
                    self.f.name
                );
            }
        }
        // The CFG *after* the clearing: an emptied block exits, so its edges are
        // gone, and a phi built from the old predecessor list would not match the
        // verifier's. Every decision below uses this CFG.
        let cfg = rdec_core::cfg::Cfg::build(&self.f);
        let all: Vec<BbId> = self.f.blocks.iter().map(|b| b.id).collect();
        let reachable: Vec<BbId> = all
            .iter()
            .copied()
            .filter(|b| !unreach.contains(b))
            .collect();

        for &g in GRPS.iter() {
            // Where a phi is needed for this register.
            let mut readers: HashSet<BbId> = HashSet::new();
            for &(b, gg, _) in self.live_in.iter() {
                if gg == g {
                    readers.insert(b);
                }
            }
            let mut phi_of: HashMap<BbId, VId> = HashMap::new();
            for b in &reachable {
                let preds = cfg.preds[*b as usize].clone();
                // A join has >= 2 incoming edges. A single predecessor that this
                // block dominates is a back edge - a loop header whose only in
                // edge is the latch - and a value defined in the latch does not
                // dominate the header, so that edge needs a phi too.
                let back = preds.len() == 1 && cfg.dominates(*b, preds[0]);
                if preds.len() < 2 && !back {
                    continue;
                }
                // Only where the register is actually read at a start point:
                // here, or at the start of a successor.
                let needed = readers.contains(b)
                    || first
                        .preds
                        .iter()
                        .enumerate()
                        .any(|(p, ps)| ps.contains(b) && readers.contains(&(p as BbId)));
                if !needed {
                    continue;
                }
                let undef = self.undef_val();
                let pairs: Vec<(VId, BbId)> = preds.iter().map(|&p| (undef, p)).collect();
                let v = self.f.new_value(Ty::u64());
                let addr = self.f.block(*b).addr;
                self.f.block_mut(*b).instrs.insert(
                    0,
                    Instr::new(v, Op::Phi(pairs.into_boxed_slice()), Ty::u64(), addr),
                );
                phi_of.insert(*b, v);
            }
            // Solve in/out to a fixed point over the edges.
            let mut in_val: HashMap<BbId, VId> = HashMap::new();
            let mut out_val: HashMap<BbId, VId> = HashMap::new();
            for round in 0..64usize {
                let mut changed = false;
                for b in &reachable {
                    let preds = cfg.preds[*b as usize].clone();
                    let phi = phi_of.get(b).copied();
                    let undef = self.undef_val();
                    let new_in: VId = match phi {
                        Some(phi) => {
                            let raw: Vec<VId> = preds
                                .iter()
                                .map(|&p| out_val.get(&p).copied().unwrap_or(undef))
                                .collect();
                            // An incoming value that IS this phi is the honest
                            // answer, not a placeholder to be removed: it says
                            // the block does not write the register, so the
                            // value the register holds on re-entry is the value
                            // it had at the header - `phi(x, phi)`, the standard
                            // SSA for a loop-invariant register. `prune_phis`
                            // then drops the self edge and collapses the phi to
                            // `copy x`, which is the value the machine carries.
                            //
                            // Substituting `undef` here (what this did) made
                            // every loop that reads a register the loop body
                            // does not write - `early_returns`' `cmp edx, r9d`,
                            // the loop-carried RAX of `mov eax, 1` - an
                            // undefined read, and T2 filed 16 functions under
                            // "returns RAX unwritten on some path" for it.
                            //
                            // A self-reference with NO base case (every incoming
                            // edge is the phi) is the real gap: no path into the
                            // loop ever writes the register. That one is undef.
                            //
                            // And the self-reference has to be AVAILABLE, which is
                            // the verifier's rule for a phi operand: the value
                            // must be live at the END of its predecessor. A header
                            // dominates its latch in a natural loop, so the edge
                            // carries the header's own value; in an irreducible
                            // loop it does not, and the honest answer for that
                            // edge is `undef` - there is no value of the register
                            // that reaches it from the header.
                            let base = raw
                                .iter()
                                .zip(preds.iter())
                                .any(|(v, p)| *v != phi && !unreach.contains(p));
                            let pairs: Vec<(VId, BbId)> = raw
                                .iter()
                                .zip(preds.iter())
                                .map(|(v, p)| {
                                    let carried = *v == phi && base && cfg.dominates(*b, *p);
                                    let v = if unreach.contains(p) || (*v == phi && !carried) {
                                        undef
                                    } else {
                                        *v
                                    };
                                    (v, *p)
                                })
                                .collect();
                            if let Some(ix) = self
                                .f
                                .block_mut(*b)
                                .instrs
                                .iter()
                                .position(|i| i.dest == phi && matches!(i.op, Op::Phi(_)))
                            {
                                if let Op::Phi(old) = &mut self.f.block_mut(*b).instrs[ix].op {
                                    if old.to_vec() != pairs {
                                        *old = pairs.into_boxed_slice();
                                        changed = true;
                                    }
                                }
                            }
                            phi
                        }
                        None if preds.is_empty() => undef,
                        None => {
                            let p = preds[0];
                            out_val.get(&p).copied().unwrap_or(undef)
                        }
                    };
                    if in_val.get(b).copied() != Some(new_in) {
                        in_val.insert(*b, new_in);
                        changed = true;
                    }
                    let o = match (self.block_def.get(b), self.block_out.get(b)) {
                        (Some(d), Some(m)) if d.contains(&g) => m[&g],
                        _ => new_in,
                    };
                    if out_val.get(b).copied() != Some(o) {
                        out_val.insert(*b, o);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
                if round == 63 {
                    self.rep.gaps += 1;
                    eprintln!("SSA: fixed point not reached for {g:?} in {}", self.f.name);
                }
            }
            for (b, gg, old) in self.live_in.clone() {
                if gg != g {
                    continue;
                }
                let Some(new) = in_val.get(&b).copied() else {
                    continue;
                };
                if Some(new) == self.undef {
                    self.rep.undef_live_ins += 1;
                    // Name the read, not just count it. The address is the
                    // placeholder's own: `read64` stamped it with the address of
                    // the instruction that read the register, so this line points
                    // at the machine-code byte where the undefined value enters.
                    let at = self
                        .f
                        .block(b)
                        .instrs
                        .iter()
                        .find(|i| i.dest == old)
                        .map(|i| i.addr)
                        .unwrap_or(0);
                    self.rep.undef_sites.push((at, gg));
                }
                if old != new {
                    rewrite_in_block(self.f.block_mut(b), old, new);
                }
            }
        }
        // Delete the placeholders SSA construction invented, now that every
        // group has been solved and rewritten - not per group. Doing it
        // interleaved deleted a placeholder a *later* group's phi had already
        // taken as an incoming value, and the verifier caught the dangling
        // reference: "value %28 is used but never defined".
        //
        // Why delete at all: the placeholder is our plumbing, not a fact about
        // the machine code. `Op::Opaque` blocks DCE on purpose (an unmodelled
        // instruction must never vanish quietly), and that rule must not extend
        // to scaffolding, or every live-in in every loop header survives into the
        // emitted C as a refusal.
        let mut seen: HashSet<u32> = HashSet::new();
        for (b, _g, old) in self.live_in.clone() {
            if !seen.insert(old) {
                continue;
            }
            if self.f.use_count(old) != 0 {
                continue;
            }
            let is_scaffolding = self.f.block(b).instrs.iter().any(|i| {
                i.dest == old && matches!(&i.op, Op::Opaque(t) if t.starts_with("live-in "))
            });
            if !is_scaffolding {
                continue;
            }
            self.f.block_mut(b).instrs.retain(|i| i.dest != old);
            self.rep.scaffolding_removed += 1;
        }
        self.place_undef();
    }

    /// Find the jump-table dispatches of this function: the four-instruction idiom
    /// clang emits for a computed jump (`lea (%rip),%base` naming the table,
    /// `movslq (%base,%idx,4),%tmp` reading a slot, `add %tmp,%base` folding it in,
    /// `jmp *%base`), matched against a table discovery resolved FROM THE
    /// RELOCATIONS. The match is on the shape the machine code has, the table on
    /// the container's evidence; both are required, and the site key is discovery's
    /// link from the naming relocation to this `jmp`.
    ///
    /// The four addresses are recorded so the sweep skips them: `.rdata`'s bytes
    /// are not in the flat memory the model runs on, so lifting the slot read as a
    /// `Load` would make the interpreter fault on a table byte and the emitted C
    /// read uninitialised memory. The index register's value at the read is the
    /// selector, captured there because the read overwrites that register.
    fn scan_dispatches(
        &self,
        bytes: &[u8],
        base: u64,
        blocks: &[BlockInsns],
    ) -> HashMap<u64, Step> {
        let mut out: HashMap<u64, Step> = HashMap::new();
        if self.tables.is_empty() {
            return out;
        }
        for b in blocks {
            let mut v: Vec<iced_x86::Instruction> = Vec::new();
            for (va, len) in &b.insns {
                let off = (*va).saturating_sub(base) as usize;
                if off + *len > bytes.len() {
                    continue;
                }
                let mut d = Decoder::with_ip(
                    64,
                    &bytes[off..off + *len],
                    *va,
                    iced_x86::DecoderOptions::NONE,
                );
                let mut ins = iced_x86::Instruction::default();
                d.decode_out(&mut ins);
                if ins.code() as u16 == 0 {
                    continue;
                }
                v.push(ins);
            }
            for i in 3..v.len() {
                let site = &v[i];
                if site.mnemonic() != Mnemonic::Jmp || site.op_kind(0) != OpKind::Register {
                    continue;
                }
                let table = match self.tables.get(&site.ip()) {
                    Some(t) if !t.targets.is_empty() => t,
                    _ => continue,
                };
                let (add, load, lea) = (&v[i - 1], &v[i - 2], &v[i - 3]);
                // `add %tmp,%base`: the fold, into the register the jump reads.
                if add.mnemonic() != Mnemonic::Add
                    || add.op_kind(0) != OpKind::Register
                    || add.op_register(0) != site.op_register(0)
                    || add.op_kind(1) != OpKind::Register
                {
                    continue;
                }
                // `movslq (%base,%idx,4),%tmp`: a 4-byte slot, indexed, off the
                // register the `lea` wrote. `movslq` is iced's `Movsxd`:
                // sign-extend a memory 32-bit value into a 64-bit register.
                // `Movsx` is the 8/16-bit form.
                //
                // WHERE THE DELTA LANDS IS A LOWERING ARTIFACT, NOT A FACT ABOUT
                // THE JUMP. clang 22.1.8 loads the slot delta into the BASE
                // register and adds it into the index register; clang 20.1.8 - the
                // CI runner's compiler, whose bytes were dumped from CI in run
                // 42166188651 - loads it into the INDEX register and adds the base
                // in:
                //   clang 22:  movslq (%rax,%rcx,4),%rcx ; addq %rcx,%rax
                //   clang 20:  movslq (%rcx,%rax,4),%rax ; addq %rcx,%rax
                // The `.rdata` bytes, the 8 relocations and the 8 targets are
                // byte-identical on both; only the register roles swap. Requiring
                // the delta to be the fold's SOURCE operand made the corpus's two
                // switches model locally and be REFUSED on CI - which is why
                // `t2 --corpus` compared 1412 here and 1364 there, both green. The
                // machine facts are: the jump reads the fold's destination; the
                // fold's two operands are the relocation-named table base and the
                // delta loaded at 4-byte scale off that base; every slot target is
                // a block this function owns.
                let (ld, aa, ab, lr) = (
                    split(load.op_register(0)),
                    split(add.op_register(0)),
                    split(add.op_register(1)),
                    split(lea.op_register(0)),
                );
                // The fold is `base + delta` in one of exactly two role
                // assignments (the jump already reads `add`'s destination, checked
                // above):
                //   dest == base && src == delta   (clang 22)
                //   dest == delta && src == base   (clang 20)
                // A destination or source equal to neither is rejected, and so is
                // the degenerate `add r,r`, which doubles one register and is not a
                // fold of anything.
                //
                // WITNESSED, 2026-11-06: `jump_table_fixtures.rs::
                // a_degenerate_add_of_a_register_with_itself_is_not_a_fold`
                // patches JT_A's own bytes (same instruction sizes, so the
                // REL32 addend convention quoted in that file is unchanged)
                // into `movslq (%rax,%rcx,4),%rax ; add %rax,%rax ; jmp *%rax`.
                // The role check alone ACCEPTS that shape (both roles are
                // %rax); the doubling dispatches to 2*(base+delta). Without
                // `aa == ab` the fixture goes red with `jump_tables: 1` - the
                // guard is the load-bearing line.
                if load.mnemonic() != Mnemonic::Movsxd
                    || load.op_kind(1) != OpKind::Memory
                    || lea.op_kind(0) != OpKind::Register
                    || aa == ab
                    || !((aa == lr && ab == ld) || (aa == ld && ab == lr))
                    || load.memory_index_scale() != 4
                    || load.memory_base() != lea.op_register(0)
                {
                    continue;
                }
                let Some((idx, _)) = split(load.memory_index()) else {
                    continue;
                };
                // `lea (%rip),%base`: the RIP-relative form is what the naming
                // relocation applies to, and it is what makes the table reachable.
                if lea.mnemonic() != Mnemonic::Lea
                    || lea.op_kind(1) != OpKind::Memory
                    // iced spells a RIP-relative memory operand as base = RIP.
                    || lea.memory_base() != Register::RIP
                {
                    continue;
                }
                out.insert(lea.ip(), Step::Lea);
                out.insert(load.ip(), Step::Load(idx));
                out.insert(add.ip(), Step::Add);
                out.insert(site.ip(), Step::Site(table.targets.clone()));
            }
        }
        out
    }

    /// The computed jump over a resolved table: `Terminator::Switch`, whose
    /// successors ARE the table. One edge per slot, each naming a block of this
    /// function - there is no target list beside the CFG, so the verifier, the
    /// interpreter and the emitter all see the same N destinations, and the
    /// verifier is the thing that proves every one of them is a block.
    ///
    /// `None` means the table is not fully at hand: a slot whose target is not a
    /// block of this function. Then nothing is emitted and the caller keeps the
    /// refusal, because a switch over the targets that happened to resolve is a
    /// partial switch - a body with a hole in it - and this repo does not emit
    /// those. The missing slot is counted, and named.
    fn switch_term(&mut self, site: u64, idx: VId, targets: &[u64]) -> Option<Terminator> {
        let mut arms: Vec<BbId> = Vec::with_capacity(targets.len());
        for t in targets {
            match self.bb_of.get(t) {
                Some(&b) => arms.push(b),
                None => {
                    self.rep.gaps += 1;
                    self.rep.jump_table_missing += 1;
                    self.f.emit_void(
                        self.cur,
                        Op::Opaque(
                            format!("jump table target {t:#x} is not a block of this function")
                                .into(),
                        ),
                        site,
                    );
                }
            }
        }
        if !arms.is_empty() && arms.len() == targets.len() {
            self.rep.jump_tables += 1;
            self.rep.jump_table_edges += arms.len();
            Some(Terminator::Switch {
                index: idx,
                targets: arms,
            })
        } else {
            self.rep.jump_table_partial += 1;
            None
        }
    }

    /// Emit one call SITE. `Site::Call` is a `call`: the callee's answer is
    /// defined, the caller-saved registers are destroyed, and the NEXT
    /// instruction runs. `Site::Tail` is a `jmp` whose relocation named this
    /// callee: the same arguments, the same destroyed registers, the same answer,
    /// and no next instruction - the block ends by leaving the function with that
    /// answer. One body for both, because the machine gives a tail call the same
    /// registers a call gets; only the successor differs, and that difference is
    /// the `site` at the end.
    fn emit_call_site(&mut self, ins: &Instruction, t: u64, site: Site) {
        // More register arguments than the model has pairs (Windows x64
        // has four): the 5th+ argument lives on the stack, unmodelled.
        let over_args = ins.op_count() > 1;
        if over_args {
            self.gap("call with more than four register arguments", Ty::i64());
            self.rep.calls_over_args += 1;
        }
        let sig = self.names.get(&t).and_then(|r| self.sigs.get(r));
        // The callee's declared return type decides what the call defines.
        let typed_call = sig.is_some() && sig.unwrap().typed && !over_args;
        // A TAIL CALL must name a callee the pipeline can call: the `jmp` leaves the
        // function with the call's value, so an unnamed or untyped callee has no
        // answer to leave with. Refuse at the `jmp`, keep the `Exit`, and COUNT it
        // - the verifier refuses the same shape in the IR, so a silent one must
        // not be creatable here either.
        if site == Site::Tail && !(sig.is_some() && typed_call) {
            self.rep.tail_calls_unnamed += 1;
            let nm = self.reloc_on(ins).map(|rt| rt.sym).unwrap_or_default();
            self.gap(
                &format!("tail call to '{nm}': no callee the pipeline can call and return through"),
                Ty::Void,
            );
            self.pending = Pending::OpaqueTerm;
            self.ended_block = true;
            return;
        }
        let ret_ty = if typed_call {
            match sig.unwrap().abi.ret {
                RetKind::Int => Ty::i64(),
                RetKind::F32 => Ty::f32(),
                RetKind::F64 => Ty::f64(),
            }
        } else {
            Ty::i64()
        };
        // The arguments are the CALLER's cells for the pairs the CALLEE's
        // declared shape occupies — the same `seeded_pairs` rule that seeds
        // the callee's parameters, so a declared float argument goes in the
        // XMM cell the callee reads, not the integer register. A call we
        // cannot type keeps the integer model's four-integer-register
        // spelling, because that is the shape it has always had.
        let pairs: Vec<Grp> = match sig {
            Some(s) if s.typed && !over_args => {
                s.abi.seeded_pairs().into_iter().map(|(g, _)| g).collect()
            }
            _ => ARG_REGS.to_vec(),
        };
        let mut args: Vec<VId> = Vec::with_capacity(pairs.len());
        for g in pairs {
            let a = self.read64(g);
            args.push(a);
        }
        let callee = match sig {
            Some(s) => Callee::Symbol(s.cname.clone()),
            None => {
                self.rep.unnamed_calls += 1;
                self.rep.calls_no_symbol += 1;
                Callee::Addr(t)
            }
        };
        if sig.is_some() && !typed_call {
            // Identity known, signature not: the answer is not known, so
            // the call defines nothing (RAX stays clobbered below) and the
            // caller that uses the answer is refused, as it always was.
            self.rep.calls_untypable += 1;
        }
        let v = self.f.new_value(ret_ty.clone());
        let a = self.addr();
        self.f.block_mut(self.cur).push(Instr::new(
            v,
            Op::Call {
                callee,
                ret_ty: ret_ty.clone(),
                args: args.into_boxed_slice(),
                pure: false,
            },
            ret_ty.clone(),
            a,
        ));
        // The callee's answer lands where its declared return type says:
        // RAX for an integer, XMM0's cell for a float. The W64 view of a
        // float answer is a BITCAST (the cell holds the bits — converting
        // here would answer a different number), and the `float` case zexts
        // the low 32 bits, which is the Windows x64 rule that the upper
        // half of XMM0 is zeroed after a `float` return — the same rule the
        // `cvtsi2ss` / `movss` paths follow.
        if typed_call {
            match sig.unwrap().abi.ret {
                RetKind::Int => {
                    let view = self.cast(CastKind::Bitcast, v, Width::W64);
                    self.set_reg(Grp::Rax, view);
                }
                RetKind::F64 => {
                    let view = self.cast(CastKind::Bitcast, v, Width::W64);
                    self.set_reg(Grp::Xmm(0), view);
                }
                RetKind::F32 => {
                    let low = self.cast(CastKind::Bitcast, v, Width::W32);
                    let wide = self.cast(CastKind::ZExt, low, Width::W64);
                    self.set_reg(Grp::Xmm(0), wide);
                }
            }
            match site {
                Site::Call => self.rep.calls_modelled += 1,
                Site::Tail => self.rep.tail_calls += 1,
            }
        } else {
            // RAX is an unsigned W64 register value; `Op::Call`'s type is
            // its own declared return type, so the register reads the same
            // bits as unsigned. Without the view, any later partial write
            // of RAX reports the signed/unsigned mismatch the verifier
            // flags for a param.
            let view = self.cast(CastKind::Bitcast, v, Width::W64);
            self.set_reg(Grp::Rax, view);
        }
        // Caller-saved registers are gone. Modelled, loudly, per register
        // — except the one this call DEFINES, whose value the call wrote.
        self.clobber_all(if typed_call {
            match sig.unwrap().abi.ret {
                RetKind::Int => Some(Grp::Rax),
                RetKind::F32 | RetKind::F64 => Some(Grp::Xmm(0)),
            }
        } else {
            None
        });
        // The tail call's own ending: the block leaves the function with the
        // value this call defined, and there is NO fall-through successor.
        self.clear_flags();
        if site == Site::Tail {
            self.pending = Pending::TailCall(v);
            // `return callee(args);` — the function left through this call, so
            // the bytes after the `jmp` (clang's `nopw` padding) are not its code.
            self.ended_block = true;
        }
    }

    /// A call to a callee the front end DECLARED noreturn
    /// (`CalleeSig::noreturn`). The call DEFINES NOTHING - there is no answer,
    /// because control does not come back - the caller's scratch registers are
    /// destroyed as at any call, and the block ENDS: the code after the call is
    /// unreachable from it, and the function has no fall-through return.
    ///
    /// The callee's identity is the symbol the RELOCATION named. An import is
    /// `U` in `llvm-nm`: the object has no address for it, so the address table
    /// is silent and the relocation is the only name there is.
    fn noreturn_site(&mut self, ins: &Instruction, sym: &str) {
        let cname = self
            .sigs
            .get(sym)
            .map(|s| s.cname.clone())
            .unwrap_or_else(|| sym.to_string());
        if ins.op_count() > 1 {
            self.gap("call with more than four register arguments", Ty::i64());
            self.rep.calls_over_args += 1;
        }
        let pairs: Vec<Grp> = match self.sigs.get(sym) {
            Some(s) if s.typed && ins.op_count() <= 1 => {
                s.abi.seeded_pairs().into_iter().map(|(g, _)| g).collect()
            }
            _ => ARG_REGS.to_vec(),
        };
        let mut args: Vec<VId> = Vec::with_capacity(pairs.len());
        for g in pairs {
            let a = self.read64(g);
            args.push(a);
        }
        let at = self.addr();
        let v = self.f.new_value(Ty::Void);
        self.f.block_mut(self.cur).push(Instr::new(
            v,
            Op::Call {
                callee: Callee::Symbol(cname),
                ret_ty: Ty::Void,
                args: args.into_boxed_slice(),
                pure: false,
            },
            Ty::Void,
            at,
        ));
        // Every caller-saved register is destroyed and NOTHING is defined:
        // `clobber_all(None)` is the shape of a call that answers nothing.
        self.clobber_all(None);
        self.clear_flags();
        self.rep.noreturn_calls += 1;
        self.rep.noreturn_sites.push((at, sym.to_string()));
        self.pending = Pending::NoReturn;
        // The call IS the end of the block: what follows it in the function's
        // range (`int3` + padding in `nr.O0.obj`) is not this function's code.
        self.ended_block = true;
    }

    /// A call whose RELOCATION names a symbol this object does not define: an
    /// import, `U` in `llvm-nm`. The name is the object's evidence and it is all
    /// the object gives, so this is the loud call shape and nothing more:
    ///
    /// * the callee is the NAMED symbol (`Callee::Symbol`), because the
    ///   relocation states the identity — the address-keyed shape lookup that
    ///   gives a call its declared signature is silent for a callee with no
    ///   address here, so there is no declared return kind and the call
    ///   DEFINES NOTHING (`Ty::Void`). A model that answered `int64_t` here
    ///   would be answering for a function body it has never seen.
    /// * the caller-saved registers are destroyed as at any call
    ///   (`clobber_all(None)`) and the flags with them;
    /// * the counted gap says which symbol, in the report and in the IR
    ///   (`Op::Opaque`), so the refusal class is `rdec c`'s census, not a
    ///   comment;
    /// * the block ENDS in `Terminator::Exit`: the model cannot certify that
    ///   control comes back from a callee it cannot see. This is the shape the
    ///   corpus has always given an unmodellable terminator — counted in
    ///   `rep.exits`, refused by the emitter as `unmodelled control flow
    ///   (Exit)`, excluded from T2's runnable set.
    ///
    /// The census `calls_reloc_external` already counted this site at
    /// `resolve_reloc_target`, which is where the container's answer is read.
    fn import_site(&mut self, ins: &Instruction, sym: &str) {
        let cname = self
            .sigs
            .get(sym)
            .map(|s| s.cname.clone())
            .unwrap_or_else(|| sym.to_string());
        if ins.op_count() > 1 {
            self.gap("call with more than four register arguments", Ty::i64());
            self.rep.calls_over_args += 1;
        }
        let pairs: Vec<Grp> = match self.sigs.get(sym) {
            Some(s) if s.typed && ins.op_count() <= 1 => {
                s.abi.seeded_pairs().into_iter().map(|(g, _)| g).collect()
            }
            _ => ARG_REGS.to_vec(),
        };
        let mut args: Vec<VId> = Vec::with_capacity(pairs.len());
        for g in pairs {
            let a = self.read64(g);
            args.push(a);
        }
        let at = self.addr();
        let v = self.f.new_value(Ty::Void);
        self.f.block_mut(self.cur).push(Instr::new(
            v,
            Op::Call {
                callee: Callee::Symbol(cname),
                ret_ty: Ty::Void,
                args: args.into_boxed_slice(),
                pure: false,
            },
            Ty::Void,
            at,
        ));
        self.rep.calls_untypable += 1;
        self.gap(
            &format!(
                "call to '{sym}': the relocation names a symbol this object does not define, so there is no callee to call and no answer to define"
            ),
            Ty::Void,
        );
        self.clobber_all(None);
        self.clear_flags();
        self.pending = Pending::OpaqueTerm;
        // The block ends at the call: the model cannot name what comes after a
        // callee it cannot see, and the bytes are not this function's to run.
        self.ended_block = true;
    }

    /// A block's terminator, from what its last instruction said: the `Pending`
    /// its final instruction left behind says which exit the block takes, and a
    /// block whose pending is `Fall` falls through to the next block in the table
    /// or, when there is none, ends the function with an `Exit`.
    fn close(&mut self, i: usize) {
        let fall = self
            .order
            .get(i + 1)
            .and_then(|a| self.bb_of.get(a).copied());
        let term = match std::mem::replace(&mut self.pending, Pending::Fall) {
            Pending::Fall => match fall {
                Some(bb) => Terminator::Jump(bb),
                None => {
                    self.rep.exits += 1;
                    Terminator::Exit
                }
            },
            // The value the function answers with, in the register its declared
            // return kind says carries it. Shared with `Pending::TailCall`: the
            // callee put its answer in RAX (or XMM0) FOR THE CALLER, which is what
            // makes `return callee(args);` the same control flow the `jmp` is.
            // A TAIL CALL answers the function with the CALL's own value and
            // leaves no fall-through edge: `return callee(args);`. The value is
            // the `Op::Call` result, not the register read - the verifier's rule
            // is that a `TailReturn` is defined by a call in the same block, and
            // the bitcast view `Pending::Ret` reads is not a call. The bits are
            // the same: the view exists to model a partial write of the register,
            // and a tail call writes none.
            Pending::TailCall(v) => {
                match self.abi.ret {
                    RetKind::F32 => self.rep.ret_f32 += 1,
                    RetKind::F64 => self.rep.ret_f64 += 1,
                    RetKind::Int => {}
                }
                Terminator::TailReturn(v)
            }
            // The call never returned, so nothing after it runs and the function
            // has no fall-through answer. `rep.noreturn_calls` counted it and
            // `rep.noreturn_sites` names the callee.
            Pending::NoReturn => Terminator::NoReturn,
            Pending::Ret => match self.abi.ret {
                RetKind::Int => {
                    let r = self.read64(Grp::Rax);
                    Terminator::Return(r)
                }
                // A declared float/double function answers in XMM0: the cell's
                // bits ARE the answer, viewed as a float for the declared
                // return type. `f32` compares 32 bits - the `movss` the ABI
                // caller must use zeroes 127:32, so the low 32 are the whole
                // answer; the truncation to 32 is the width the type names,
                // not a mask to force an agreement. The kind is the declared
                // signature's, never a reading of what the code did.
                RetKind::F32 => {
                    self.rep.ret_f32 += 1;
                    let bits = self.read64(Grp::Xmm(0));
                    let low = self.narrow(bits, Width::W32);
                    let v = self.cast_ty(CastKind::Bitcast, low, Ty::f32());
                    Terminator::Return(v)
                }
                RetKind::F64 => {
                    self.rep.ret_f64 += 1;
                    let bits = self.read64(Grp::Xmm(0));
                    let v = self.cast_ty(CastKind::Bitcast, bits, Ty::f64());
                    Terminator::Return(v)
                }
            },
            Pending::Jump(t) => match self.bb_of.get(&t).copied() {
                Some(bb) => Terminator::Jump(bb),
                None => {
                    self.rep.exits += 1;
                    Terminator::Exit
                }
            },
            Pending::Jcc(t, c) => match (self.cond(c), self.bb_of.get(&t).copied(), fall) {
                (Some(cond), Some(th), Some(el)) => Terminator::Branch {
                    cond,
                    then_bb: th,
                    else_bb: el,
                },
                _ => {
                    self.rep.exits += 1;
                    self.gap(
                        "conditional branch with no block, or flags undefined",
                        Ty::Bool,
                    );
                    Terminator::Exit
                }
            },
            Pending::OpaqueTerm => {
                self.rep.exits += 1;
                Terminator::Exit
            }
            Pending::Switch(t) => t,
        };
        self.f.block_mut(self.cur).term = term;
    }
}

/// Lift one function out of an object's discovered code, under the integer
/// model (RCX/RDX/R8/R9 as parameters, RAX as the answer).
pub fn lift_one(
    name: &str,
    bytes: &[u8],
    base: u64,
    start: u64,
    end: u64,
    plan: &Plan,
    names: &HashMap<u64, String>,
) -> (Function, Report) {
    lift_one_sig(
        name,
        bytes,
        base,
        start,
        end,
        plan,
        names,
        &FnAbi::integer(),
    )
}

/// Lift one function under a DECLARED shape: the parameter kinds and return
/// kind come from the caller's evidence (the C++ pipeline builds them from the
/// demangled signature and says so in the report). `FnAbi::integer()` is
/// `lift_one` exactly.
// The parameter list is the lift entry's, and `abi` is the eighth: bundling the
// five code-range arguments (bytes/base/start/end/plan) into a struct is a
// refactor of every caller in the repo, not a lint fix.
#[allow(clippy::too_many_arguments)]
pub fn lift_one_sig(
    name: &str,
    bytes: &[u8],
    base: u64,
    start: u64,
    end: u64,
    plan: &Plan,
    names: &HashMap<u64, String>,
    abi: &FnAbi,
) -> (Function, Report) {
    lift_one_sigs(
        name,
        bytes,
        base,
        start,
        end,
        plan,
        names,
        abi,
        &no_callees(),
        &no_relocs(),
        &no_tables(),
    )
}

/// Lift one function with a declared shape AND the declared shapes of its
/// callees, so a direct call to a named, typed sibling becomes a call op that
/// defines its answer. An empty `sigs` is `lift_one_sig` exactly: every call
/// keeps the loud shape (call + clobber per caller-saved register) it has always
/// had, and the report counts each refused class.
// The parameter list is the lift entry's, and `abi`/`sigs` are the evidence:
// bundling the five code-range arguments into a struct is a refactor of every
// caller in the repo, not a lint fix.
#[allow(clippy::too_many_arguments)]
pub fn lift_one_sigs(
    name: &str,
    bytes: &[u8],
    base: u64,
    start: u64,
    end: u64,
    plan: &Plan,
    names: &HashMap<u64, String>,
    abi: &FnAbi,
    sigs: &CalleeTable,
    relocs: &RelocMap,
    tables: &Tables,
) -> (Function, Report) {
    let blocks = split_blocks(plan, start, end);
    let mut l = Lifter::new_sig(name, start, names, abi, sigs, relocs).with_tables(tables);
    let f = l.lift_function(bytes, base, &blocks);
    let rep = l.rep.clone();
    (f, rep)
}

/// Replace uses of `old` with `new` inside ONE block, instructions and
/// terminator. A global replace would be wrong: `old` is the live-in value of
/// this block, and other blocks legitimately name the same incoming value.
fn rewrite_in_block(b: &mut rdec_core::ir::Block, old: VId, new: VId) {
    for i in b.instrs.iter_mut() {
        if i.dest == old {
            continue;
        }
        rewrite_op(&mut i.op, old, new);
    }
    rewrite_term(&mut b.term, old, new);
}

fn map(v: VId, old: VId, new: VId) -> VId {
    if v == old {
        new
    } else {
        v
    }
}

fn rewrite_op(op: &mut Op, old: VId, new: VId) {
    match op {
        Op::Copy(a) => *a = map(*a, old, new),
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) => {
            *a = map(*a, old, new);
            *b = map(*b, old, new);
        }
        // The float pair has to be rewritten too: a phi resolution that skips a
        // `fadd`'s operands leaves that operand naming a block-local live-in,
        // which is exactly the dangling reference this function exists to
        // remove. It is listed with `Bin`/`Cmp`, not in the catch-all below, so
        // the day an op is added the compiler asks the question.
        Op::FBin(_, a, b) | Op::FCmp(_, a, b) => {
            *a = map(*a, old, new);
            *b = map(*b, old, new);
        }
        Op::Un(_, a) | Op::Cast(_, _, a) | Op::Load(_, a) | Op::GetFlag(_, a) => {
            *a = map(*a, old, new);
        }
        Op::Store(_, a, v) => {
            *a = map(*a, old, new);
            *v = map(*v, old, new);
        }
        Op::Select(c, t, f) => {
            *c = map(*c, old, new);
            *t = map(*t, old, new);
            *f = map(*f, old, new);
        }
        Op::Phi(pairs) => {
            for (v, _) in pairs.iter_mut() {
                *v = map(*v, old, new);
            }
        }
        Op::Call { args, .. } => {
            for a in args.iter_mut() {
                *a = map(*a, old, new);
            }
        }
        Op::Flags(f) => rewrite_flags(f, old, new),
        Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Param | Op::Clobber | Op::Opaque(_) => {}
    }
}

fn rewrite_flags(f: &mut FlagsOp, old: VId, new: VId) {
    let m = |v: &mut VId| *v = map(*v, old, new);
    match f {
        FlagsOp::Add { a, b, .. }
        | FlagsOp::Sub { a, b, .. }
        | FlagsOp::And { a, b }
        | FlagsOp::Or { a, b }
        | FlagsOp::Xor { a, b }
        | FlagsOp::Shl { a, b }
        | FlagsOp::Shr { a, b }
        | FlagsOp::Sar { a, b }
        | FlagsOp::Mul { a, b, .. } => {
            m(a);
            m(b);
        }
        FlagsOp::Neg { a } => m(a),
        // An SSA rename must follow the carried flags value too: a stale `cf_in`
        // is a dangling use, which verify_ssa rejects as a hard error.
        FlagsOp::Inc { a, cf_in } | FlagsOp::Dec { a, cf_in } => {
            m(a);
            if let Some(v) = cf_in {
                *v = map(*v, old, new);
            }
        }
    }
}

fn rewrite_term(t: &mut Terminator, old: VId, new: VId) {
    match t {
        Terminator::Return(v) | Terminator::TailReturn(v) => *v = map(*v, old, new),
        Terminator::Branch { cond, .. } => *cond = map(*cond, old, new),
        // A switch's index is a value like any other, and SSA rewriting rewrites
        // it: the destinations are blocks, not values, so they are untouched.
        Terminator::Switch { index, .. } => *index = map(*index, old, new),
        Terminator::Jump(_) | Terminator::Unreachable | Terminator::Exit | Terminator::NoReturn => {
        }
    }
}
