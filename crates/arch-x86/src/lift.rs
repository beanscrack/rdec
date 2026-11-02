//! M2 step 1: lift x86-64 machine code into `rdec-core` IR.
//!
//! The contract, and why the code is shaped this way:
//!
//! * Each register is ONE `W64` SSA value. A 32-bit read is `trunc`, a 32-bit
//!   write is `zext` - which is exactly the x86 rule that a 32-bit write clears
//!   the upper half. 8/16-bit writes are the mask/or pair the hardware performs.
//!   One representation per register is what keeps SSA, the verifier and the
//!   interpreter honest.
//! * Flag-defining instructions produce a `Ty::Flags` value; `jcc`/`setcc`/
//!   `cmovcc` read it with `GetFlag`. Dead flags then die by DCE, and `fuse.rs`
//!   turns `(cmp, jl)` into a signed compare. Flag semantics are width-dependent
//!   and width comes from the operand TYPES, so operands handed to a `FlagsOp`
//!   are truncated to the instruction's operand width first.
//! * A call clobbers the caller-saved registers. That is modelled, as an
//!   `Opaque` per clobbered register, because leaving a value live across a call
//!   is a silent miscompile, and the silent kind is the one that ships.
//! * Everything outside the subset becomes `Op::Opaque(text)` - in the IR,
//!   counted, printed. Never a plausible guess.
//!
//! Block boundaries come from `discover`, so the lifter never invents one that
//! the decoder and the descent did not prove.

use std::collections::{HashMap, HashSet};

use iced_x86::{Decoder, Instruction, Mnemonic, OpKind, Register};
use rdec_core::ir::{
    BbId, BinOp, Callee, CastKind, Flag, FlagsOp, Function, Instr, Op, Terminator, UnOp, VId,
};
use rdec_core::ty::{Ty, Width};

use crate::discover::Plan;

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
        // AH/CH/DH/BH alias a value we model as one 64-bit word: refuse.
        _ => return None,
    };
    Some((g, w))
}

/// Every modelled register, in a fixed order: the SSA pass walks registers in
/// this order, which is what makes the emitted phi order deterministic.
const GRPS: [Grp; 16] = [
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
];

fn ty_u(w: Width) -> Ty {
    Ty::uint(w)
}

/// The Windows x86-64 integer calling convention, as far as we model it: four
/// integer arguments, integer return. More arguments is a loud gap, not a drop.
const ARG_REGS: [Grp; 4] = [Grp::Rcx, Grp::Rdx, Grp::R8, Grp::R9];

/// Caller-saved: a call may overwrite these.
const CALLER_SAVED: [Grp; 9] = [
    Grp::Rax,
    Grp::Rcx,
    Grp::Rdx,
    Grp::R8,
    Grp::R9,
    Grp::R10,
    Grp::R11,
    Grp::Rsi,
    Grp::Rdi,
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
    /// An indirect jump/call, or a terminator we do not model.
    OpaqueTerm,
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
    /// Blocks the entry cannot reach; their lifted contents dropped, counted.
    pub unreachable_blocks: usize,
}

pub struct Lifter<'a> {
    f: Function,
    regs: HashMap<Grp, VId>,
    flags: Option<VId>,
    bb_of: HashMap<u64, BbId>,
    order: Vec<u64>,
    cur: BbId,
    /// Address of the instruction being lifted. Every IR node records it, so a
    /// verifier complaint can be traced back to a byte offset - with the block
    /// address only, half a dozen values share `; 0x25` and a width complaint is
    /// unattributable.
    at: u64,
    pending: Pending,
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
    fmt: Box<dyn iced_x86::Formatter>,
}

impl<'a> Lifter<'a> {
    pub fn new(name: &str, addr: u64, names: &'a HashMap<u64, String>) -> Lifter<'a> {
        let mut f = Function::new(name, addr);
        f.ret_ty = Ty::i64();
        let entry = f.entry;
        let mut regs = HashMap::new();
        for g in ARG_REGS {
            let v = f.new_value(Ty::i64());
            f.params.push(v);
            f.block_mut(f.entry)
                .push(Instr::new(v, Op::Param, Ty::i64(), addr));
            regs.insert(g, v);
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
        Lifter {
            f,
            regs,
            flags: None,
            bb_of: HashMap::new(),
            order: Vec::new(),
            cur: entry,
            at: addr,
            pending: Pending::Fall,
            live_in: Vec::new(),
            block_out: HashMap::new(),
            block_def: HashMap::new(),
            cur_defs: HashSet::new(),
            undef: None,
            rep: Report::default(),
            names,
            fmt: crate::decode::formatter(crate::decode::Syntax::Intel),
        }
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

    /// Write a width-`w` result, honouring x86's partial-write rules.
    fn write(&mut self, g: Grp, w: Width, v: VId) {
        let v64 = match w {
            Width::W64 => v,
            // A 32-bit write zeroes the upper half: `zext`.
            Width::W32 => self.cast(CastKind::ZExt, v, Width::W64),
            _ => {
                let old = self.read64(g);
                let mask: u64 = if w == Width::W8 { 0xff } else { 0xffff };
                let m = self.const_of(!mask, Width::W64);
                let keep = self.bin(BinOp::And, old, m);
                let up = self.cast(CastKind::ZExt, v, Width::W64);
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

    fn const_of(&mut self, v: u64, w: Width) -> VId {
        self.f.emit(self.cur, Op::Const(v), ty_u(w), self.addr())
    }

    fn set_flags(&mut self, op: FlagsOp) {
        let v = self.f.emit(self.cur, Op::Flags(op), Ty::Flags, self.addr());
        self.flags = Some(v);
    }

    /// Flags are undefined after ops the model does not define them for. Reading
    /// them is then a counted gap, not a stale value.
    fn clear_flags(&mut self) {
        self.flags = None;
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
    /// RIP-relative (needs the data/symbol model), unaddressable index.
    fn addr_of(&mut self, ins: &Instruction, op: u32) -> Option<VId> {
        if ins.op_kind(op) != OpKind::Memory {
            return None;
        }
        let base = ins.memory_base();
        if base == Register::RIP || base == Register::None {
            return None;
        }
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
        let t = ty_u(w);
        if self.f.ty_of(v) == Some(t.clone()) {
            return v;
        }
        self.f
            .emit(self.cur, Op::Cast(k, t.clone(), v), t, self.addr())
    }

    /// One instruction. Control-transfer instructions set `self.pending`.
    fn one(&mut self, ins: &Instruction) {
        let m = ins.mnemonic();
        let nm = self.name_of(ins);
        let w = self.op_w(ins);

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
                        // Usually an address in another section: needs the data
                        // and symbol model, which is M4/M6 work.
                        self.gap("lea of an address we cannot name", Ty::u64())
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
                        self.gap(
                            "mov to an address we cannot name (rip-relative data?)",
                            Ty::u64(),
                        );
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
                                self.gap("arithmetic on an address we cannot name", Ty::u64());
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
                        self.gap("inc/dec on an unmodelled destination", ty_u(w));
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
                self.set_flags(if m == Mnemonic::Inc {
                    FlagsOp::Inc { a: lhs }
                } else {
                    FlagsOp::Dec { a: lhs }
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
            Mnemonic::Cmp | Mnemonic::Test => {
                let w2 = self.op_w(ins);
                let lhs = match self.src(ins, 0, w2) {
                    Some(v) => v,
                    None => {
                        self.gap("flag setting with an unmodelled first operand", ty_u(w2));
                        self.clear_flags();
                        return;
                    }
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
                self.regs.insert(Grp::Rsp, nsp);
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
                self.regs.insert(Grp::Rsp, nsp);
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
                match self.target(ins) {
                    Some(t) => self.pending = Pending::Jump(t),
                    None => {
                        self.gap("jmp with no readable target", Ty::Void);
                        self.pending = Pending::OpaqueTerm;
                    }
                }
            }
            Mnemonic::Call => {
                if ins.op_kind(0) == OpKind::Register || ins.op_kind(0) == OpKind::Memory {
                    self.f
                        .emit_void(self.cur, Op::Opaque("indirect call".into()), self.addr());
                    self.rep.gaps += 1;
                } else {
                    let t = self.target(ins).unwrap_or(0);
                    let callee = match self.names.get(&t) {
                        Some(n) => Callee::Symbol(n.clone()),
                        None => {
                            self.rep.unnamed_calls += 1;
                            Callee::Addr(t)
                        }
                    };
                    let args: Vec<VId> = ARG_REGS.iter().map(|g| self.read64(*g)).collect();
                    if ins.op_count() > 1 {
                        self.gap("call with more than four register arguments", Ty::i64());
                    }
                    let v = self.f.new_value(Ty::i64());
                    let a = self.addr();
                    self.f.block_mut(self.cur).push(Instr::new(
                        v,
                        Op::Call {
                            callee,
                            ret_ty: Ty::i64(),
                            args: args.into_boxed_slice(),
                            pure: false,
                        },
                        Ty::i64(),
                        a,
                    ));
                    self.regs.insert(Grp::Rax, v);
                }
                // Caller-saved registers are gone. Modelled, loudly, per register.
                for g in CALLER_SAVED {
                    let v = self.f.emit(
                        self.cur,
                        Op::Opaque("clobbered by call".into()),
                        Ty::u64(),
                        self.addr(),
                    );
                    self.regs.insert(g, v);
                    self.rep.clobbers += 1;
                }
                self.clear_flags();
            }
            Mnemonic::Ret => {
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
        for (i, b) in blocks.iter().enumerate() {
            let id = match self.bb_of.get(&b.start) {
                Some(x) => *x,
                None => continue,
            };
            self.cur = id;
            self.pending = Pending::Fall;
            if i > 0 {
                // Each block starts with an EMPTY register map. Carrying the map
                // over in address order (the first version) is what produced
                // uses that no dominating def reached: the value was defined in
                // a block that is not the predecessor, and no phi existed.
                self.regs.clear();
            }
            self.cur_defs.clear();
            for (va, len) in b.insns.iter() {
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
        // Take, do not move: the caller reads `rep` after this returns, and a
        // report snapshotted before lifting would always claim zero gaps.
        std::mem::replace(&mut self.f, Function::new("", 0))
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
                            // An incoming value that is the phi itself means the
                            // value is undefined along that edge (the block does
                            // not define the register and the edge is the loop
                            // back edge); say so with undef, not self-reference.
                            let pairs: Vec<(VId, BbId)> = preds
                                .iter()
                                .map(|&p| {
                                    let v = out_val.get(&p).copied().unwrap_or(undef);
                                    let v = if v == phi || unreach.contains(&p) {
                                        undef
                                    } else {
                                        v
                                    };
                                    (v, p)
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
                if old != new {
                    rewrite_in_block(self.f.block_mut(b), old, new);
                }
            }
        }
        self.place_undef();
    }

    /// A block's terminator, from what its last instruction said.
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
            Pending::Ret => {
                let r = self.read64(Grp::Rax);
                Terminator::Return(r)
            }
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
        };
        self.f.block_mut(self.cur).term = term;
    }
}

/// Lift one function out of an object's discovered code.
pub fn lift_one(
    name: &str,
    bytes: &[u8],
    base: u64,
    start: u64,
    end: u64,
    plan: &Plan,
    names: &HashMap<u64, String>,
) -> (Function, Report) {
    let blocks = split_blocks(plan, start, end);
    let mut l = Lifter::new(name, start, names);
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
        Op::Const(_) | Op::ConstF(_) | Op::Undef | Op::Param | Op::Opaque(_) => {}
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
        FlagsOp::Neg { a } | FlagsOp::Inc { a } | FlagsOp::Dec { a } => m(a),
    }
}

fn rewrite_term(t: &mut Terminator, old: VId, new: VId) {
    match t {
        Terminator::Return(v) => *v = map(*v, old, new),
        Terminator::Branch { cond, .. } => *cond = map(*cond, old, new),
        Terminator::Jump(_) | Terminator::Unreachable | Terminator::Exit => {}
    }
}
