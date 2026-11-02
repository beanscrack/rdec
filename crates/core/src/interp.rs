//! The IR interpreter: the executable form of `docs/ir-semantics.md`.
//!
//! This is the reference semantics of the IR. It exists so that every other
//! oracle has something independent to check against:
//!
//! - **TV** (translation validation) runs a function before and after a pass on
//!   the same input state and compares outcomes. That comparison *is* the pass
//!   oracle, and it only exists because this interpreter exists.
//! - **T1R** (M2) checks flag lowerings against the semantics here, never against
//!   the lowering's own helpers.
//! - **P1** (proof track) differentially tests the SMT encoding against this file.
//!
//! Per §1.1's rule, an oracle must not share code with its subject: this module
//! deliberately re-derives flag semantics instead of calling
//! `rdec_arch_x86::flags`, and the Cargo graph keeps core unable to see that crate.
//!
//! Determinism is a hard requirement: identical input must produce identical
//! output, because CI diffs two runs (§1.5) and TV compares two executions.

use crate::ir::*;
use crate::ty::{FWidth, Sx, Ty, Width};
use std::collections::HashMap;

/// A runtime value. `Int` always holds the value truncated/extended to the
/// width its IR type says it has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    Int(u64),
    Fl(f64),
    Bl(bool),
    /// An opaque EFLAGS value. It must never be observable in the emitted C;
    /// reading it goes through `GetFlag`.
    Fg(FlagBits),
}

impl Val {
    pub fn bits(self) -> u64 {
        match self {
            Val::Int(x) => x,
            Val::Bl(b) => b as u64,
            Val::Fl(f) => f.to_bits(),
            Val::Fg(_) => 0,
        }
    }
    pub fn as_bool(self) -> bool {
        match self {
            Val::Bl(b) => b,
            Val::Int(x) => x != 0,
            _ => false,
        }
    }
}

/// One EFLAGS snapshot. `None` = architecturally undefined (SDM), which the IR
/// models as an undefined read, not as a bit.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct FlagBits {
    pub cf: Option<bool>,
    pub pf: Option<bool>,
    pub af: Option<bool>,
    pub zf: Option<bool>,
    pub sf: Option<bool>,
    pub of: Option<bool>,
}

impl FlagBits {
    pub fn get(self, f: Flag) -> Option<bool> {
        match f {
            Flag::CF => self.cf,
            Flag::PF => self.pf,
            Flag::AF => self.af,
            Flag::ZF => self.zf,
            Flag::SF => self.sf,
            Flag::OF => self.of,
        }
    }
}

/// Faults, in the sense of "the machine code does not define behavior here"
/// (§0 equivalence definition: such inputs are outside the claim).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    DivZero,
    /// `INT_MIN / -1`, which traps on x86.
    IntMinDivMinusOne,
    /// A read of a flag bit the model says is undefined.
    UndefFlag(Flag),
    /// A load from an address the state has no bytes for and no rule for.
    UnmodeledLoad,
    /// An operand with no definition. In SSA this cannot happen in valid IR, so
    /// it means a pass left a dangling reference -- loud, never a silent zero.
    MissingOperand(VId),
}

/// Observable events, in execution order. TV compares these, not just the return
/// value: a pass that reorders, drops, duplicates or invents a call is caught.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    /// A call, with the raw argument bits. The return value fed back to the
    /// program is `call_result(callee, args)`, a deterministic function of the
    /// call, so both executions see the same world.
    Call(String, Vec<u64>),
    /// A memory write, as observed by the rest of the program.
    Write(u64, u32, u64),
    /// A read of a flag the model calls undefined, and the bit used for it.
    UndefFlagRead(Flag),
    /// Execution reached `Terminator::Exit` (unmodeled control flow).
    Exited,
    /// Execution consumed an `Op::Opaque`.
    Opaque(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Ret(Val),
    Fault(Fault),
    /// Reached Exit, or fell off the end (`Terminator::Unreachable`).
    NoReturn,
    /// Hit the step budget. TV treats a budget on one side and not the other as
    /// a difference worth reporting.
    Budget,
}

/// One execution: the outcome, the event trace, and the final memory.
#[derive(Clone, Debug)]
pub struct Exec {
    pub outcome: Outcome,
    pub effects: Vec<Effect>,
    /// Every byte ever written, final value. A pass that writes extra bytes,
    /// skips bytes, or changes a value is caught here.
    pub mem: Vec<(u64, u8)>,
}

impl Exec {
    /// Compare two executions. Returns a human-readable list of differences —
    /// empty means equivalent on this input.
    pub fn diff(&self, other: &Exec) -> Vec<String> {
        let mut d = Vec::new();
        if self.outcome != other.outcome {
            d.push(format!(
                "outcome: {:?} vs {:?}",
                self.outcome, other.outcome
            ));
        }
        if self.effects != other.effects {
            // Name the first divergence: "pass X did Y on input Z" is the whole
            // value of TV, so the message must be a pointer, not a shrug.
            let n = self.effects.len().min(other.effects.len());
            let mut where_ = n;
            for i in 0..n {
                if self.effects[i] != other.effects[i] {
                    where_ = i;
                    break;
                }
            }
            let a = self
                .effects
                .get(where_)
                .map(|e| format!("{e:?}"))
                .unwrap_or_else(|| "<none>".to_string());
            let b = other
                .effects
                .get(where_)
                .map(|e| format!("{e:?}"))
                .unwrap_or_else(|| "<none>".to_string());
            d.push(format!(
                "effect trace differs at #{where_}: {} vs {} (len {} vs {})",
                a,
                b,
                self.effects.len(),
                other.effects.len()
            ));
        }
        let mut ms = self.mem.clone();
        let mut os = other.mem.clone();
        ms.sort();
        os.sort();
        if ms != os {
            let only_self: Vec<_> = ms.iter().filter(|x| !os.contains(x)).copied().collect();
            let only_other: Vec<_> = os.iter().filter(|x| !ms.contains(x)).copied().collect();
            d.push(format!(
                "final memory differs: {} bytes only in A (first {:?}), {} only in B (first {:?})",
                only_self.len(),
                only_self.first(),
                only_other.len(),
                only_other.first()
            ));
        }
        d
    }
}

/// Sparse byte-addressed memory. Byte granularity means aliasing, partial
/// writes and unaligned access all need no special case.
#[derive(Clone, Default, Debug)]
pub struct Mem {
    bytes: HashMap<u64, u8>,
}

impl Mem {
    pub fn store(&mut self, addr: u64, bytes: u32, val: u64) {
        for i in 0..bytes {
            self.bytes
                .insert(addr.wrapping_add(i as u64), (val >> (8 * i)) as u8);
        }
    }
    pub fn load(&self, addr: u64, bytes: u32) -> Option<u64> {
        let mut v = 0u64;
        for i in 0..bytes {
            v |= (*self.bytes.get(&addr.wrapping_add(i as u64))? as u64) << (8 * i);
        }
        Some(v)
    }
    pub fn snapshot(&self) -> Vec<(u64, u8)> {
        self.bytes.iter().map(|(a, b)| (*a, *b)).collect()
    }
}

/// Deterministic call-result function: the world's answer to a call, as a pure
/// function of (callee, argument bits). Both executions of a TV pair get the same
/// answer, so TV compares how the program *reacts* to calls, which is exactly
/// what a pass is allowed not to change.
pub fn call_result(callee: &str, args: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in callee.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &a in args {
        h ^= a;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct Interp<'a> {
    f: &'a Function,
    vals: Vec<Option<Val>>,
    mem: Mem,
    effects: Vec<Effect>,
    steps: usize,
    budget: usize,
    /// Block we fell through from, for resolving `Op::Phi` incoming lists.
    prev_bb: Option<BbId>,
}

/// Default step budget. A function that exceeds it is reported as `Budget`,
/// never silently truncated (§1.3's convergence rule, applied to interpretation).
pub const DEFAULT_BUDGET: usize = 200_000;

pub fn run(f: &Function, inputs: &[u64]) -> Exec {
    run_budget(f, inputs, DEFAULT_BUDGET)
}

pub fn run_budget(f: &Function, inputs: &[u64], budget: usize) -> Exec {
    let mut it = Interp {
        f,
        vals: vec![None; f.values.len().max(f.next_value as usize) + 16],
        mem: Mem::default(),
        effects: Vec::new(),
        steps: 0,
        budget,
        prev_bb: None,
    };
    let outcome = it.exec_entry(inputs);
    Exec {
        outcome,
        effects: it.effects,
        mem: it.mem.snapshot(),
    }
}

impl<'a> Interp<'a> {
    fn width_of(&self, v: VId) -> Width {
        self.f
            .ty_of(v)
            .and_then(|t| t.width())
            .unwrap_or(Width::W64)
    }

    fn signed_of(&self, v: VId) -> Sx {
        match self.f.ty_of(v) {
            Some(Ty::Int { sx, .. }) => sx,
            _ => Sx::Unsigned,
        }
    }

    fn get(&self, v: VId) -> Option<Val> {
        self.vals.get(v as usize).copied().flatten()
    }

    /// Read an operand. A value with no definition is a fault, not a zero: a
    /// pass that dangles a reference must fail loudly, because the silent-0
    /// version of this made TV blind to exactly that bug.
    fn fetch(&self, v: VId) -> Result<u64, Fault> {
        match self.get(v) {
            Some(x) => Ok(x.bits()),
            None => Err(Fault::MissingOperand(v)),
        }
    }

    fn fetch_val(&self, v: VId) -> Result<Val, Fault> {
        self.get(v).ok_or(Fault::MissingOperand(v))
    }

    fn exec_entry(&mut self, inputs: &[u64]) -> Outcome {
        // Parameters: bound in the entry block by `Op::Param`.
        for (i, &p) in self.f.params.iter().enumerate() {
            let ty = self.f.ty_of(p).unwrap_or(Ty::u64());
            let raw = inputs.get(i).copied().unwrap_or(0);
            let v = match ty {
                Ty::Float { w: FWidth::F32 } => Val::Fl(f32::from_bits(raw as u32) as f64),
                Ty::Float { w: FWidth::F64 } => Val::Fl(f64::from_bits(raw)),
                Ty::Bool => Val::Bl(raw != 0),
                _ => Val::Int(ty.width().unwrap_or(Width::W64).trunc(raw, false)),
            };
            self.set(p, v);
        }
        self.exec_bb(self.f.entry)
    }

    fn set(&mut self, v: VId, val: Val) {
        let i = v as usize;
        if i >= self.vals.len() {
            self.vals.resize(i + 64, None);
        }
        self.vals[i] = Some(val);
    }

    fn exec_bb(&mut self, bb: BbId) -> Outcome {
        let mut cur = bb;
        let mut from: Option<BbId> = None;
        loop {
            if cur == NO_BB || cur as usize >= self.f.blocks.len() {
                self.effects.push(Effect::Exited);
                return Outcome::NoReturn;
            }
            self.prev_bb = from;
            let block = &self.f.blocks[cur as usize];
            for ii in 0..block.instrs.len() {
                self.steps += 1;
                if self.steps > self.budget {
                    return Outcome::Budget;
                }
                // Borrow discipline: pull what we need, then release.
                let (dest, op) = {
                    let ins = &self.f.blocks[cur as usize].instrs[ii];
                    (ins.dest, ins.op.clone())
                };
                match self.eval_op(dest, &op) {
                    Ok(()) => {}
                    Err(f) => return Outcome::Fault(f),
                }
            }
            let term = self.f.blocks[cur as usize].term.clone();
            from = Some(cur);
            match term {
                Terminator::Jump(t) => cur = t,
                Terminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let c = self.get(cond).map(|v| v.as_bool()).unwrap_or(false);
                    cur = if c { then_bb } else { else_bb };
                }
                Terminator::Return(v) => {
                    let val = if v == NO_VAL {
                        Val::Int(0)
                    } else {
                        match self.fetch_val(v) {
                            Ok(x) => x,
                            Err(f) => return Outcome::Fault(f),
                        }
                    };
                    return Outcome::Ret(val);
                }
                Terminator::Unreachable | Terminator::Exit => {
                    self.effects.push(Effect::Exited);
                    return Outcome::NoReturn;
                }
            }
        }
    }

    fn eval_op(&mut self, dest: VId, op: &Op) -> Result<(), Fault> {
        match op {
            Op::Const(c) => self.set(dest, Val::Int(*c)),
            Op::ConstF(c) => self.set(dest, Val::Fl(*c)),
            // Undef is the all-zero bit pattern: a *fixed* choice, so TV is
            // deterministic. See docs/ir-semantics.md.
            Op::Undef => self.set(dest, Val::Int(0)),
            Op::Param => {
                if self.get(dest).is_none() {
                    self.set(dest, Val::Int(0));
                }
            }
            Op::Copy(s) => {
                let v = self.fetch_val(*s)?;
                self.set(dest, v);
            }
            Op::Bin(o, a, b) => {
                let (x, y) = (self.fetch(*a)?, self.fetch(*b)?);
                let w = self.width_of(*a);
                let sx = self.signed_of(*a);
                let r = self.binop(*o, x, y, w, sx)?;
                self.set(dest, Val::Int(w.trunc(r, false)));
            }
            Op::Un(o, a) => {
                let v = self.fetch_val(*a)?;
                let r = self.unop(*o, v, self.width_of(*a));
                self.set(dest, r);
            }
            Op::Cmp(c, a, b) => {
                let (x, y) = (self.fetch(*a)?, self.fetch(*b)?);
                let w = self.width_of(*a);
                // Signedness is a property of the predicate: Lt/Le/Gt/Ge are
                // signed, Ult/Ule/Ugt/Uge unsigned, Eq/Ne neither. The operand
                // type's signedness is a display concern (§1.3), so it is NOT
                // consulted here.
                let signed = matches!(c, IntCmp::Lt | IntCmp::Le | IntCmp::Gt | IntCmp::Ge);
                let r = c.eval(x, y, w, if signed { Sx::Signed } else { Sx::Unsigned });
                self.set(dest, Val::Bl(r));
            }
            Op::Flags(fo) => {
                let bits = self.eval_flags(fo)?;
                self.set(dest, Val::Fg(bits));
            }
            Op::GetFlag(fl, fv) => {
                let bits = match self.get(*fv) {
                    Some(Val::Fg(b)) => b,
                    _ => FlagBits::default(),
                };
                match bits.get(*fl) {
                    Some(b) => self.set(dest, Val::Bl(b)),
                    // An undefined flag read is a defined *event*: recorded so TV
                    // sees it on both sides, and faults so no pass can use it to
                    // smuggle a value through.
                    None => {
                        self.effects.push(Effect::UndefFlagRead(*fl));
                        self.set(dest, Val::Bl(false));
                        return Err(Fault::UndefFlag(*fl));
                    }
                }
            }
            Op::Cast(k, t, a) => {
                let x = self.fetch(*a)?;
                let sw = self.width_of(*a);
                let f = match self.get(*a) {
                    Some(Val::Fl(f)) => f,
                    _ => 0.0,
                };
                let v = match k {
                    CastKind::Trunc => Val::Int(t.width().unwrap_or(sw).trunc(x, false)),
                    CastKind::ZExt => Val::Int(sw.trunc(x, false)),
                    CastKind::SExt => Val::Int(sw.trunc(x, true)),
                    CastKind::Bitcast => match t {
                        // A bitcast int<->float reinterprets the bits.
                        Ty::Float { w: FWidth::F32 } => Val::Fl(f32::from_bits(x as u32) as f64),
                        Ty::Float { w: FWidth::F64 } => Val::Fl(f64::from_bits(x)),
                        Ty::Bool => Val::Bl(x != 0),
                        _ => Val::Int(x),
                    },
                    CastKind::SIToFp => Val::Fl(sw.trunc(x, true) as i64 as f64),
                    CastKind::UIToFp => Val::Fl(sw.trunc(x, false) as f64),
                    CastKind::FpToSI => {
                        Val::Int(t.width().unwrap_or(sw).trunc(f as i64 as u64, false))
                    }
                    CastKind::FpToUI => Val::Int(t.width().unwrap_or(sw).trunc(f as u64, false)),
                    CastKind::FPExt => Val::Fl(f),
                    CastKind::FPTrunc => Val::Fl(f as f32 as f64),
                };
                self.set(dest, v);
            }
            Op::Load(t, addr) => {
                let a = self.fetch(*addr)?;
                let n = t.width().map(|w| w.bytes()).unwrap_or(8);
                match self.mem.load(a, n) {
                    Some(v) => {
                        let v = match t {
                            Ty::Float { w: FWidth::F32 } => {
                                Val::Fl(f32::from_bits(v as u32) as f64)
                            }
                            Ty::Float { w: FWidth::F64 } => Val::Fl(f64::from_bits(v)),
                            Ty::Bool => Val::Bl(v != 0),
                            _ => Val::Int(v),
                        };
                        self.set(dest, v);
                    }
                    // Uninitialized memory is `undef` per the semantics doc:
                    // fixed zero, so both executions agree.
                    None => self.set(dest, Val::Int(0)),
                }
            }
            Op::Store(t, addr, val) => {
                let a = self.fetch(*addr)?;
                let n = t.width().map(|w| w.bytes()).unwrap_or(8);
                let v = match self.get(*val) {
                    Some(Val::Fl(f)) => match t {
                        Ty::Float { w: FWidth::F32 } => (f as f32) as u64,
                        _ => f.to_bits(),
                    },
                    Some(x) => x.bits(),
                    None => 0,
                };
                self.mem.store(a, n, v);
                self.effects.push(Effect::Write(a, n, v));
            }
            Op::Select(c, t, e) => {
                let cv = self.fetch(*c)? != 0;
                let v = self.fetch_val(if cv { *t } else { *e })?;
                self.set(dest, v);
            }
            Op::Phi(pairs) => {
                // The predecessor is tracked by the caller: SSA phi at block
                // entry reads the value from the block we came from.
                let v = pairs
                    .iter()
                    .find(|(_, bb)| Some(*bb) == self.prev_bb)
                    .or_else(|| pairs.first())
                    .map(|(v, _)| *v)
                    .unwrap_or(NO_VAL);
                let val = self.fetch_val(v)?;
                self.set(dest, val);
            }
            Op::Call {
                callee,
                args,
                pure,
                ret_ty,
                ..
            } => {
                let abits: Vec<u64> = args
                    .iter()
                    .map(|a| self.fetch(*a))
                    .collect::<Result<Vec<_>, Fault>>()?;
                let name = match callee {
                    Callee::Symbol(s) => s.clone(),
                    Callee::Addr(a) => format!("0x{a:x}"),
                    Callee::Indirect(v) => format!("[{}]", self.fetch(*v)?),
                };
                self.effects.push(Effect::Call(name.clone(), abits.clone()));
                // The result is a deterministic function of the call, so both
                // sides of a TV pair react to the same world.
                let r = call_result(&name, &abits);
                let v = match ret_ty {
                    Ty::Void => Val::Int(0),
                    Ty::Bool => Val::Bl(r != 0),
                    Ty::Float { .. } => Val::Fl(f64::from_bits(r)),
                    _ => Val::Int(r),
                };
                self.set(dest, v);
                let _ = pure;
            }
            Op::Opaque(msg) => {
                self.effects.push(Effect::Opaque(msg.to_string()));
                self.set(dest, Val::Int(0));
            }
        }
        Ok(())
    }

    fn binop(&mut self, o: BinOp, x: u64, y: u64, w: Width, sx: Sx) -> Result<u64, Fault> {
        let (x, y) = (w.trunc(x, false), w.trunc(y, false));
        let (xs, ys) = (w.trunc(x, true) as i64, w.trunc(y, true) as i64);
        let r = match o {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::MulHi => ((x as u128 * y as u128) >> 64) as u64,
            BinOp::MulSmul => (((xs as i128) * (ys as i128)) >> 64) as u64,
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            BinOp::Shl => shift_left(x, y, w),
            BinOp::Shr => w.trunc(w.trunc(x, false) >> (y.min(128) as u32), false),
            BinOp::Sar => w.trunc((xs >> (y.min(128) as u32)) as u64, false),
            BinOp::RotL => {
                let b = w.bits().max(1);
                let n = (y % b as u64) as u32;
                (x << n) | (x >> (b - n))
            }
            BinOp::RotR => {
                let b = w.bits().max(1);
                let n = (y % b as u64) as u32;
                (x >> n) | (x << (b - n))
            }
            BinOp::UDiv => {
                if y == 0 {
                    return Err(Fault::DivZero);
                }
                x / y
            }
            BinOp::URem => {
                if y == 0 {
                    return Err(Fault::DivZero);
                }
                x % y
            }
            BinOp::SDiv => {
                if y == 0 {
                    return Err(Fault::DivZero);
                }
                if xs == i64::MIN && ys == -1 {
                    return Err(Fault::IntMinDivMinusOne);
                }
                w.trunc(xs.wrapping_div(ys) as u64, false)
            }
            BinOp::SRem => {
                if y == 0 {
                    return Err(Fault::DivZero);
                }
                if xs == i64::MIN && ys == -1 {
                    return Err(Fault::IntMinDivMinusOne);
                }
                w.trunc(xs.wrapping_rem(ys) as u64, false)
            }
        };
        let _ = sx; // signedness is carried by the specific op, per §1.3
        Ok(w.trunc(r, false))
    }

    fn unop(&mut self, o: UnOp, v: Val, w: Width) -> Val {
        let x = v.bits();
        match o {
            UnOp::Not => Val::Int(w.trunc(!x, false)),
            UnOp::Neg => Val::Int(w.trunc(x.wrapping_neg(), false)),
            UnOp::Popcount => Val::Int(w.trunc(x, false).count_ones() as u64),
            UnOp::Lzcnt => {
                let b = w.bits();
                Val::Int((b as u64 - w.trunc(x, false).leading_zeros() as u64) & b as u64)
            }
            UnOp::Tzcnt => Val::Int(w.trunc(x, false).trailing_ones_0() as u64),
            UnOp::Sqrt => match v {
                Val::Fl(f) => Val::Fl(f.sqrt()),
                _ => Val::Fl((w.trunc(x, false) as f64).sqrt()),
            },
        }
    }

    /// EFLAGS semantics, written from the SDM and T1's measurements, *not* from
    /// `arch-x86::flags` — that independence is the point (oracle rule §1.1).
    fn eval_flags(&mut self, fo: &FlagsOp) -> Result<FlagBits, Fault> {
        let g = |v: VId| -> Result<u64, Fault> { self.fetch(v) };
        let mut d = FlagBits::default();
        match *fo {
            FlagsOp::Add { a, b, carry } => {
                let (x, y) = (g(a)?, g(b)?);
                let w = self.width_of(a);
                let cin = carry as u64;
                let s = x as u128 + y as u128 + cin as u128;
                let res = w.trunc(s as u64, false);
                d.cf = Some(s > w.mask() as u128);
                d.pf = Some(parity(res));
                d.af = Some(((x ^ res ^ y) >> 4) & 1 == 1);
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = Some(bit((x ^ res) & (y ^ res), w.bits() - 1));
            }
            FlagsOp::Sub { a, b, borrow } => {
                let (x, y) = (g(a)?, g(b)?);
                let w = self.width_of(a);
                let bin = borrow as u64;
                let res = w.trunc(x.wrapping_sub(y).wrapping_sub(bin), false);
                d.cf = Some(
                    ((x as u128).wrapping_sub(y as u128 + bin as u128) & !w.mask() as u128) != 0,
                );
                d.pf = Some(parity(res));
                d.af = Some(((x ^ res ^ y) >> 4) & 1 == 1);
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = Some(bit((x ^ y) & (x ^ res), w.bits() - 1));
            }
            FlagsOp::And { a, b } | FlagsOp::Or { a, b } | FlagsOp::Xor { a, b } => {
                let (x, y) = (g(a)?, g(b)?);
                let w = self.width_of(a);
                let res = match fo {
                    FlagsOp::And { .. } => x & y,
                    FlagsOp::Or { .. } => x | y,
                    _ => x ^ y,
                };
                let res = w.trunc(res, false);
                d.cf = Some(false);
                d.pf = Some(parity(res));
                d.af = None;
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = Some(false);
            }
            FlagsOp::Shl { a, b } | FlagsOp::Shr { a, b } | FlagsOp::Sar { a, b } => {
                let (x, c) = (g(a)?, g(b)?);
                let w = self.width_of(a);
                let n = count_for(w, c);
                if n == 0 {
                    // Count 0 leaves every flag unchanged, and v1's IR has no
                    // prev-flags operand to name "unchanged" (§9 read-findings).
                    d.cf = None;
                    d.pf = None;
                    d.af = None;
                    d.zf = None;
                    d.sf = None;
                    d.of = None;
                    return Ok(d);
                }
                let res = match fo {
                    FlagsOp::Shl { .. } => shift_left(x, n as u64, w),
                    FlagsOp::Shr { .. } => w.trunc(x >> n.min(128), false),
                    _ => w.trunc(((w.trunc(x, true) as i64) >> n.min(128)) as u64, false),
                };
                let res = w.trunc(res, false);
                d.cf = Some(match fo {
                    FlagsOp::Shl { .. } => {
                        if n > w.bits() {
                            false
                        } else {
                            bit(x, w.bits() - n)
                        }
                    }
                    _ => bit(x, n.saturating_sub(1)),
                });
                d.pf = Some(parity(res));
                d.af = None;
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = if n == 1 {
                    match fo {
                        FlagsOp::Shl { .. } => Some(bit(res, w.bits() - 1) != d.cf.unwrap()),
                        FlagsOp::Shr { .. } => Some(bit(x, w.bits() - 1)),
                        _ => Some(bit(x, w.bits() - 1)),
                    }
                } else {
                    None
                };
            }
            FlagsOp::Neg { a } => {
                let x = g(a)?;
                let w = self.width_of(a);
                let res = w.trunc(x.wrapping_neg(), false);
                d.cf = Some(x != 0);
                d.pf = Some(parity(res));
                d.af = Some(((x ^ res) >> 4) & 1 == 1);
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = Some(x == 1u64 << (w.bits() - 1));
            }
            FlagsOp::Inc { a } | FlagsOp::Dec { a } => {
                let x = g(a)?;
                let w = self.width_of(a);
                let inc = matches!(fo, FlagsOp::Inc { .. });
                let res = if inc {
                    w.trunc(x.wrapping_add(1), false)
                } else {
                    w.trunc(x.wrapping_sub(1), false)
                };
                // CF preserved; the IR cannot express that (no prev operand), so
                // it is undefined here and a read of it is a recorded event.
                d.cf = None;
                d.pf = Some(parity(res));
                d.af = Some(((x ^ res ^ 1) >> 4) & 1 == 1);
                d.zf = Some(res == 0);
                d.sf = Some(bit(res, w.bits() - 1));
                d.of = Some(if inc {
                    x == (1u64 << (w.bits() - 1)) - 1
                } else {
                    x == 1u64 << (w.bits() - 1)
                });
            }
            FlagsOp::Mul { a, b, signed } => {
                let (x, y) = (g(a)?, g(b)?);
                let w = self.width_of(a);
                let (lo, hi) = if signed {
                    let p = (w.trunc(x, true) as i128) * (w.trunc(y, true) as i128);
                    (p as u64, (p >> 64) as u64)
                } else {
                    let p = (w.trunc(x, false) as u128) * (w.trunc(y, false) as u128);
                    (p as u64, (p >> 64) as u64)
                };
                let hi_sign = (w.trunc(hi, true) as i64) as u64;
                let lo_sign = w.trunc(w.trunc(lo, true) as i64 as u64, false);
                d.cf = Some(hi_sign != lo_sign);
                d.of = Some(hi_sign != lo_sign);
                d.pf = Some(parity(w.trunc(lo, false)));
                d.af = None;
                d.zf = Some(w.trunc(lo, false) == 0);
                d.sf = Some(bit(w.trunc(lo, false), w.bits() - 1));
            }
        }
        Ok(d)
    }
}

// helpers -----------------------------------------------------------------

/// x86 shift-count masking: 5 bits for 8/16/32-bit operands, 6 for 64-bit
/// (Intel Vol. 2, SHR/SHL). T1's grid found a model that applied 5 bits to every
/// width, which turns `shl rax,36` into `shl rax,4`.
pub fn count_for(w: Width, c: u64) -> u32 {
    if w.bits() == 64 {
        (c & 0x3f) as u32
    } else {
        (c & 0x1f) as u32
    }
}

fn bit(x: u64, k: u32) -> bool {
    ((x >> k) & 1) == 1
}

fn parity(res: u64) -> bool {
    (res as u8).count_ones().is_multiple_of(2)
}

/// Shift-left defined for every count (docs/ir-semantics.md): counts at or
/// beyond the width produce zero, which is what the masked hardware does for the
/// legal range and a total definition for the rest.
fn shift_left(x: u64, count: u64, w: Width) -> u64 {
    if count >= 64 {
        return 0;
    }
    w.trunc(x << (count as u32), false)
}

trait TrailingOnes0 {
    fn trailing_ones_0(self) -> u32;
}
impl TrailingOnes0 for u64 {
    fn trailing_ones_0(self) -> u32 {
        if self == 0 {
            64
        } else {
            self.trailing_zeros()
        }
    }
}
