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
use std::collections::{HashMap, HashSet};

/// A runtime value. `Int` always holds the value truncated/extended to the
/// width its IR type says it has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    Int(u64),
    /// A float value. F32-typed values are carried as the `f64` that exactly
    /// represents them (every f32 is exactly an f64), and `fbin`/`fcmp` take a
    /// precision flag so an F32 op rounds through `f32` like the machine and
    /// the emitted C do. The bit pattern a `Val::Fl` answers with for `bits()`
    /// is the F64 pattern; the F32 pattern comes from `f32_bits`.
    Fl(f64),
    Bl(bool),
    /// An opaque EFLAGS value. It must never be observable in the emitted C;
    /// reading it goes through `GetFlag`.
    Fg(FlagBits),
}

/// The bit pattern a float value of the given width occupies in memory / a
/// register cell: 32 bits for `FWidth::F32`, 64 for `FWidth::F64`.
pub fn float_bits(w: FWidth, f: f64) -> u64 {
    match w {
        // The F32 narrowing is a real rounding, `as` on an out-of-range f64
        // saturates, and both the interpreter and the emitted C use the same
        // spelling. `f32::from_f64` is not a thing in stable Rust.
        FWidth::F32 => (f as f32).to_bits() as u64,
        FWidth::F64 => f.to_bits(),
    }
}

/// The float value a memory/register bit pattern stands for at the given
/// width. An F32 NaN's high bits are not part of the value, so they are
/// dropped: `f32::from_bits(bits as u32) as f64`.
pub fn float_of_bits(w: FWidth, bits: u64) -> f64 {
    match w {
        FWidth::F32 => f32::from_bits(bits as u32) as f64,
        FWidth::F64 => f64::from_bits(bits),
    }
}

/// Float arithmetic at a precision. The F32 arm rounds every answer through
/// `f32` -- that is `addss`, and it is what the emitted C does by naming the C
/// `float` type. Computing in f64 and truncating afterwards is double rounding
/// and would make this reference disagree with the machine.
pub fn fbin(o: FBinOp, a: f64, b: f64, w: FWidth) -> f64 {
    match w {
        FWidth::F64 => match o {
            FBinOp::Add => a + b,
            FBinOp::Sub => a - b,
            FBinOp::Mul => a * b,
            FBinOp::Div => a / b,
            // x86's rule, not C's ternary: NaN on either side answers `b`.
            FBinOp::Max => {
                if a.is_nan() || b.is_nan() {
                    b
                } else if a > b {
                    a
                } else {
                    b
                }
            }
            FBinOp::Min => {
                if a.is_nan() || b.is_nan() {
                    b
                } else if a < b {
                    a
                } else {
                    b
                }
            }
            FBinOp::Sqrt => a.sqrt(),
        },
        FWidth::F32 => {
            let (a, b) = (a as f32, b as f32);
            (match o {
                FBinOp::Add => a + b,
                FBinOp::Sub => a - b,
                FBinOp::Mul => a * b,
                FBinOp::Div => a / b,
                FBinOp::Max => {
                    if a.is_nan() || b.is_nan() {
                        b
                    } else if a > b {
                        a
                    } else {
                        b
                    }
                }
                FBinOp::Min => {
                    if a.is_nan() || b.is_nan() {
                        b
                    } else if a < b {
                        a
                    } else {
                        b
                    }
                }
                FBinOp::Sqrt => a.sqrt(),
            }) as f64
        }
    }
}

/// Float comparison, IEEE/C rules: every ordered predicate is false and `!=` is
/// true when either side is NaN.
pub fn fcmp(c: FloatCmp, a: f64, b: f64) -> bool {
    match c {
        FloatCmp::Eq => a == b,
        FloatCmp::Ne => a != b,
        FloatCmp::Lt => a < b,
        FloatCmp::Le => a <= b,
        FloatCmp::Gt => a > b,
        FloatCmp::Ge => a >= b,
    }
}

/// `cvttsd2si`/`cvttss2si`: truncate toward zero, and NaN or an out-of-range
/// source answers the destination width's sign bit -- the SDM's
/// "Integer Invalid Operation" result, 0x80000000 / 0x8000000000000000. Rust's
/// `as` saturates and C's cast is undefined behaviour for these inputs; this is
/// the machine's answer, spelled out. `x >= -2^(n-1) && x < 2^(n-1)` is the
/// in-range window, so -2^(n-1) itself converts exactly and 2^(n-1) does not.
pub fn fptosi_x86(x: f64, w: Width) -> u64 {
    let bits = w.bits();
    let magic = 1u64 << (bits - 1);
    let lo = -(2f64.powi(bits as i32 - 1));
    let hi = 2f64.powi(bits as i32 - 1);
    if x.is_nan() || !(x >= lo && x < hi) {
        magic
    } else if bits == 64 {
        // `as i64` is exact inside the window, and the window excludes the one
        // value (2^63) that would not fit.
        x.trunc() as i64 as u64
    } else {
        // The 32-bit destination: the answer is 32 bits. Sign-extending to 64
        // (the old spelling) handed a 30-bit value back as 64 bits, and the
        // `zext` into the register showed the sign bits — a defect invisible
        // below 2^31 where the value fits either way.
        ((x.trunc() as i64) as u32) as u64
    }
}

/// The unsigned form (`cvttsd2usi`): NaN, a negative source, or a source at or
/// above 2^n answers the sign bit, the same invalid-operation result.
pub fn fptoui_x86(x: f64, w: Width) -> u64 {
    let bits = w.bits();
    let magic = 1u64 << (bits - 1);
    let hi = 2f64.powi(bits as i32);
    if x.is_nan() || x < 0.0 || x >= hi {
        magic
    } else {
        x.trunc() as u64
    }
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
    /// A computed jump whose index is at or past the end of the table: the
    /// machine would have jumped to a slot the object does not have.
    SwitchOutOfRange {
        index: u64,
        len: usize,
    },
    DivZero,
    /// `INT_MIN / -1`, which traps on x86.
    IntMinDivMinusOne,
    /// A read of a flag bit the model says is undefined.
    UndefFlag(Flag),
    /// A flag the MODEL needs in order to compute a flag was not in the flag
    /// dict at that node. `UndefFlag` is the x86 "this flag is undefined after
    /// that instruction" fault; this one is structural - the IR asks the
    /// interpreter for a value the node never carried. The payload is the
    /// operation, so a T2 row says which. The field is `&'static str` because
    /// `Fault` is `Copy` (jump-table tests compare faults by value).
    MissingFlag(&'static str),
    /// A load from an address the state has no bytes for and no rule for.
    UnmodeledLoad,
    /// An operand with no definition. In SSA this cannot happen in valid IR, so
    /// it means a pass left a dangling reference -- loud, never a silent zero.
    MissingOperand(VId),
    /// A call to a target named in the IR that is NOT in the model. The answer
    /// is not guessed: `call_result`'s deterministic hash is the world's answer
    /// for a callee outside the object (an import), and inventing one for a
    /// callee that should have been in the program is how a missing function
    /// becomes a plausible number.
    UnknownCallee,
    /// The call graph was followed deeper than the interpreter's nesting bound.
    /// A real recursion is a fact about the program; a runaway is a bug in the
    /// model, and both answer the same way: stop, say which.
    CallDepth,
    /// The function ended in a call to a callee that never returns
    /// (`Terminator::NoReturn`). There is no answer to give: the machine code
    /// killed the process (or unwound) at that call. A fault naming the cause,
    /// never a value conjured out of a register the call destroyed.
    NoReturn,
    /// A read of a caller-saved register the call destroyed (`Op::Clobber`).
    /// The value is not "0", and a decompiler that answers 0 here turns a
    /// destroyed register into a plausible one.
    Clobbered(VId),
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
    /// Steps the run charged against its budget: one per block entry plus one
    /// per instruction. The budget is a COUNT, so the bound is checkable
    /// (`crates/core/tests/interp_block_transfer.rs`) instead of inferred from
    /// "it finished"; a run that reports `Budget` at 20 001 steps proves the
    /// bound fires, which a wall-clock measurement cannot.
    ///
    /// `Exec::diff` deliberately ignores it: TV compares *meaning* (outcome,
    /// effects, memory) across two IR states, and a pass that legitimately
    /// deletes an instruction changes the step count without changing the
    /// answer. Comparing steps there would report every good optimisation as a
    /// difference.
    pub steps: usize,
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
    /// The functions this run can call: the callee is looked up by NAME and
    /// interpreted from its own body, in this same run, with the arguments bound
    /// to its parameters. An empty table is "no sibling is in the model", and a
    /// call to a named target then faults (`Fault::UnknownCallee`) instead of
    /// answering a hash.
    program: &'a [Function],
    /// The `Op::Call` that closes a `Terminator::NoReturn` block (see
    /// `noreturn_dests`). One question is asked of it: when its callee is not in
    /// `program`, the run ends at the call itself.
    noret: HashSet<VId>,
    /// How deep the call graph has been followed in this run.
    depth: usize,
    vals: Vec<Option<Val>>,
    mem: Mem,
    effects: Vec<Effect>,
    steps: usize,
    budget: usize,
    /// Set by a nested call that ended the run: `Budget` (the callee exhausted
    /// the shared step budget) or `NoReturn` (the callee reached `Exit`). The
    /// caller cannot name an answer in either case, so the run reports the
    /// callee's outcome, not a value.
    abort: Option<Outcome>,
    /// Block we fell through from, for resolving `Op::Phi` incoming lists.
    prev_bb: Option<BbId>,
}

/// How deep `Op::Call` may nest inside one run. The corpus's functions call one
/// level deep; a bound is here so a model bug (a callee that calls its caller)
/// reports itself instead of consuming the step budget.
pub const MAX_CALL_DEPTH: usize = 32;

/// Default step budget. A function that exceeds it is reported as `Budget`,
/// never silently truncated (§1.3's convergence rule, applied to interpretation).
pub const DEFAULT_BUDGET: usize = 200_000;

pub fn run(f: &Function, inputs: &[u64]) -> Exec {
    run_budget(f, inputs, DEFAULT_BUDGET)
}

/// The values defined by the `Op::Call` that closes a `Terminator::NoReturn`
/// block: the call the front end DECLARED never returns.
///
/// One question is asked of this set. `Op::Call` with a callee that is not in
/// `program` is a hole (`Fault::UnknownCallee`) because the call's ANSWER would
/// have to be invented. A noreturn call defines no answer — its type is
/// `Ty::Void`, and control never comes back to read anything it left behind —
/// so the callee's body cannot change this run's outcome: the run ends at the
/// call either way. Answering `Fault::NoReturn` there names the cause the IR
/// already states, where `UnknownCallee` would report a missing function that
/// the model never needed.
fn noreturn_dests(f: &Function) -> HashSet<VId> {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::NoReturn))
        .filter_map(|b| b.instrs.last())
        .filter(|i| matches!(i.op, Op::Call { .. }))
        .map(|i| i.dest)
        .collect()
}

pub fn run_budget(f: &Function, inputs: &[u64], budget: usize) -> Exec {
    run_program_budget(f, inputs, &[], budget)
}

/// Run with the memory PRE-SEEDED at given bytes, as (address, byte).
///
/// The caller owns the seed: T2's caller-buffer vectors preload the same
/// window bytes the two harnesses memcpy before the call, so the IR leg
/// starts the run reading the same bytes the machine buffer and `rdec_mem`
/// window hold. A load of a byte nothing wrote stays `Fault::UnmodeledLoad`
/// (the seed never silences a read outside the window): seeding is the point
/// of this entry, and a memory model that reads unseeded bytes as 0 is the
/// silent-0 the interpreter exists to refuse.
pub fn run_budget_seed(f: &Function, inputs: &[u64], budget: usize, seed: &[(u64, u8)]) -> Exec {
    run_program_budget_seed(f, inputs, &[], budget, seed)
}

/// Run `f` with `program` as the set of functions it may call: a call to a
/// named target in the program executes that function's own body, in this run,
/// with the arguments bound to its parameters and the same flat memory.
pub fn run_program(f: &Function, inputs: &[u64], program: &[Function]) -> Exec {
    run_program_budget(f, inputs, program, DEFAULT_BUDGET)
}

pub fn run_program_budget(
    f: &Function,
    inputs: &[u64],
    program: &[Function],
    budget: usize,
) -> Exec {
    run_program_budget_seed(f, inputs, program, budget, &[])
}

/// `run_program_budget` with the memory pre-seeded (see `run_budget_seed`).
pub fn run_program_budget_seed(
    f: &Function,
    inputs: &[u64],
    program: &[Function],
    budget: usize,
    seed: &[(u64, u8)],
) -> Exec {
    let mut it = Interp {
        f,
        program,
        noret: noreturn_dests(f),
        depth: 0,
        vals: vec![None; f.values.len().max(f.next_value as usize) + 16],
        mem: Mem::default(),
        effects: Vec::new(),
        steps: 0,
        budget,
        abort: None,
        prev_bb: None,
    };
    for &(addr, byte) in seed {
        it.mem.store(addr, 1, byte as u64);
    }
    let outcome = it.exec_entry(inputs);
    Exec {
        outcome,
        steps: it.steps,
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

    /// Read a float operand. A non-float value here means the IR wired an
    /// integer into a float op, which `verify` rejects; answering 0.0 would let
    /// that bug run, so it is a fault.
    fn fetch_fl(&self, v: VId) -> Result<f64, Fault> {
        match self.get(v) {
            Some(Val::Fl(f)) => Ok(f),
            Some(_) => Err(Fault::MissingOperand(v)),
            None => Err(Fault::MissingOperand(v)),
        }
    }

    /// The precision a float op runs at: the operand's own `Ty::Float` width.
    /// A missing type answers F64, which is the width `ty_of` records for
    /// `Op::FBin` when the operand has no type, so the two agree.
    fn float_width_of(&self, v: VId) -> FWidth {
        match self.f.ty_of(v) {
            Some(Ty::Float { w }) => w,
            _ => FWidth::F64,
        }
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

    /// Run one callee in this run: its own values, the SAME memory, effects,
    /// step counter and program table (a callee may call a third function, and
    /// its writes are the caller's writes). The arguments are bound to the
    /// callee's parameters in declaration order, read at each parameter's own
    /// type — the same rule `exec_entry` uses for the top-level function, so a
    /// float parameter binds as the bit cell its caller passed.
    fn exec_callee(&mut self, callee: &'a Function, args: &[u64]) -> Result<Val, Fault> {
        if self.depth >= MAX_CALL_DEPTH {
            return Err(Fault::CallDepth);
        }
        self.depth += 1;
        let saved_f = std::mem::replace(&mut self.f, callee);
        let saved_vals = std::mem::replace(
            &mut self.vals,
            vec![None; callee.values.len().max(callee.next_value as usize) + 16],
        );
        let saved_prev = self.prev_bb.take();
        // The callee's own noreturn call sites, not the caller's: the set is a
        // fact about `self.f`, and `self.f` is about to be the callee.
        let saved_noret = std::mem::replace(&mut self.noret, noreturn_dests(callee));
        let out = self.exec_entry(args);
        self.f = saved_f;
        self.vals = saved_vals;
        self.noret = saved_noret;
        self.prev_bb = saved_prev;
        self.depth -= 1;
        match out {
            Outcome::Ret(v) => Ok(v),
            Outcome::Fault(f) => Err(f),
            // A callee that ran out of budget, or reached control the model does
            // not name, leaves the caller with no answer to have. Say which.
            Outcome::Budget | Outcome::NoReturn => {
                self.abort = Some(out.clone());
                Ok(Val::Int(0))
            }
        }
    }

    fn exec_bb(&mut self, bb: BbId) -> Outcome {
        let mut cur = bb;
        let mut from: Option<BbId> = None;
        loop {
            if cur == NO_BB || cur as usize >= self.f.blocks.len() {
                self.effects.push(Effect::Exited);
                return Outcome::NoReturn;
            }
            // A CONTROL TRANSFER IS WORK, and the budget must charge it.
            //
            // The budget used to be charged only per instruction, so a block
            // with no instructions at all cost nothing, and a cycle made of
            // such blocks ran forever: `bb1: br bb1` — the lifted shape of a
            // `jmp .` — enters `bb1`, executes zero instructions, jumps to
            // `bb1`, charges zero steps, and `self.steps > self.budget` can
            // never become true. Measured on a 270-byte hostile object (the
            // fuzzer's seed 1, iteration 1992, `.text` = 152 random bytes;
            // `crates/arch-x86/tests/fuzz_findings.rs::the_seed1_hang_input_
            // terminates_under_a_step_budget`), `run_budget(.., 20_000)` did
            // not take 20 000 steps: it took over 300 SECONDS and never
            // finished. That is a DoS on every user of the interpreter —
            // `rdec fuzz`, `rdec t2`, and TV — from one 4 KB object, and no
            // wall-clock deadline in a driver can interrupt it from outside.
            //
            // The fix is the bound the model already promises, applied to the
            // unit that actually advances: charge one step per block entry.
            // Every lap of every cycle now spends budget, so every run is
            // bounded by `budget` in units of work the machine can count, and
            // a function that cannot finish says `Outcome::Budget` — loud,
            // counted, and the same outcome the instruction path reports.
            self.steps += 1;
            if self.steps > self.budget {
                return Outcome::Budget;
            }
            self.prev_bb = from;
            let block = &self.f.blocks[cur as usize];
            for ii in 0..block.instrs.len() {
                self.steps += 1;
                if self.steps > self.budget {
                    return Outcome::Budget;
                }
                // Borrow discipline: pull what we need, then release.
                let (dest, op, ty) = {
                    let ins = &self.f.blocks[cur as usize].instrs[ii];
                    (ins.dest, ins.op.clone(), ins.ty.clone())
                };
                match self.eval_op(dest, &op, &ty) {
                    Ok(()) => {}
                    Err(f) => return Outcome::Fault(f),
                }
                // A nested call that ended the run (budget, or a callee that
                // reached `Exit`) is the run's outcome; the value it wrote is
                // not an answer and must not flow into the next instruction.
                if let Some(o) = self.abort.clone() {
                    return o;
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
                Terminator::Switch { index, targets } => {
                    // The table covers exactly `targets.len()` slots. An index at
                    // or past the end names no destination this function has, so
                    // the run ends on a fault: a fault, never a guess. An index
                    // with no value at all is treated the same way - the run stops
                    // rather than picking a block.
                    let i = match self.fetch_val(index) {
                        Ok(v) => v.bits(),
                        Err(f) => return Outcome::Fault(f),
                    };
                    match targets.get(i as usize) {
                        Some(&b) => cur = b,
                        None => {
                            return Outcome::Fault(Fault::SwitchOutOfRange {
                                index: i,
                                len: targets.len(),
                            })
                        }
                    }
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
                // The machine code left this function through the call that
                // produced `v`, so the callee's answer IS this function's answer.
                // The callee ran in this run, executed by the `Op::Call` that
                // defines `v`: present, its own body ran; absent, that op already
                // raised `Fault::UnknownCallee` and no value was invented.
                Terminator::TailReturn(v) => {
                    let val = match self.fetch_val(v) {
                        Ok(x) => x,
                        Err(f) => return Outcome::Fault(f),
                    };
                    return Outcome::Ret(val);
                }
                // A call that never returns: there is no answer, and the run says
                // so with a fault naming the cause. Not `Outcome::NoReturn`, which
                // is the shape "ran out of modelled control flow", and never a
                // value read out of the register the call destroyed.
                Terminator::NoReturn => {
                    self.effects.push(Effect::Exited);
                    return Outcome::Fault(Fault::NoReturn);
                }
                Terminator::Unreachable | Terminator::Exit => {
                    self.effects.push(Effect::Exited);
                    return Outcome::NoReturn;
                }
            }
        }
    }

    fn eval_op(&mut self, dest: VId, op: &Op, ty: &Ty) -> Result<(), Fault> {
        match op {
            Op::Const(c) => {
                // The type says how to read the bits. An integer-typed constant
                // is an integer; a FLOAT-typed one — which is what folding a
                // bit-cast of a constant produces (`movsd xmm0,` an immediate
                // pattern, `movapd` of a mask) — is the SAME bits read as a
                // float. Reading it as an integer here would leave a `Val::Int`
                // in a float slot, and the strict interpreter would fault on the
                // next float op that consumed it.
                match ty {
                    Ty::Float { w } => self.set(dest, Val::Fl(float_of_bits(*w, *c))),
                    _ => self.set(dest, Val::Int(*c)),
                }
            }
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
            // Float arithmetic. F32 operands round the answer through `f32`,
            // because that is what `addss` does and what the emitted C does by
            // naming the C `float` type; computing in f64 and truncating after
            // would make the reference answer DOUBLE-rounded (it is the classic
            // double-rounding bug, and TV would report a clean tree that the
            // hardware disagrees with).
            Op::FBin(o, a, b) => {
                let (x, y) = (self.fetch_fl(*a)?, self.fetch_fl(*b)?);
                let w = self.float_width_of(*a);
                self.set(dest, Val::Fl(fbin(*o, x, y, w)));
            }
            Op::FCmp(c, a, b) => {
                let (x, y) = (self.fetch_fl(*a)?, self.fetch_fl(*b)?);
                self.set(dest, Val::Bl(fcmp(*c, x, y)));
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
                    CastKind::SExt => {
                        // Sign from the SOURCE, then keep the DESTINATION's
                        // bits. Signing alone leaves a value typed T(dw)
                        // carrying bits above dw — `sext` of an int8 to a W16
                        // cell answered 0xFFFFFFFFFFFFFF80 for 0x80 — and that
                        // cell is a lie twice over: the type says W16, and the
                        // C this op emits, `(uint16_t)(int16_t)x`, answers
                        // 0xFF80. x86 cannot show the difference through a
                        // register (every write at the destination width
                        // truncates), which is why T2 never saw it; a consumer
                        // that reads the value at 64 bits does.
                        Val::Int(t.width().unwrap_or(sw).trunc(sw.trunc(x, true), false))
                    }
                    CastKind::Bitcast => match t {
                        Ty::Float { w: FWidth::F32 } => Val::Fl(f32::from_bits(x as u32) as f64),
                        Ty::Float { w: FWidth::F64 } => Val::Fl(f64::from_bits(x)),
                        Ty::Bool => Val::Bl(x != 0),
                        // int <- float: the cell encoding is the value's OWN
                        // width in bits, zero-extended. An F32 value occupies
                        // the low 32 bits of the register and bits 63:32 are
                        // zero -- that is `movss`'s architectural contract -- so
                        // writing the f64 pattern here would put 32 bits of
                        // exponent into a cell whose next use is a 4-byte store.
                        _ => match self.f.ty_of(*a) {
                            Some(Ty::Float { w: FWidth::F32 }) => {
                                Val::Int(float_bits(FWidth::F32, f))
                            }
                            _ => Val::Int(x),
                        },
                    },
                    CastKind::SIToFp => {
                        // `cvtsi2sd`/`cvtsi2ss`: the integer is read at ITS own
                        // width, the answer is at the TARGET's. An F32 target
                        // rounds the exact value through f32, which is what the
                        // hardware does -- keeping the unrounded f64 here would
                        // make the reference answer differ from the machine (and
                        // from the emitted C) for every integer above 2^24.
                        // signedness is the operand's TYPED property, like the
                        // width: a `uint32_t` cell converts as its raw bits,
                        // an `int32_t` cell converts signed.
                        let signed = matches!(
                            self.f.ty_of(*a),
                            Some(Ty::Int {
                                sx: crate::ty::Sx::Signed,
                                ..
                            })
                        );
                        let v = (sw.trunc(x, signed) as i64) as f64;
                        Val::Fl(match t {
                            Ty::Float { w: FWidth::F32 } => v as f32 as f64,
                            _ => v,
                        })
                    }
                    CastKind::UIToFp => {
                        let v = sw.trunc(x, false) as f64;
                        Val::Fl(match t {
                            Ty::Float { w: FWidth::F32 } => v as f32 as f64,
                            _ => v,
                        })
                    }
                    // `cvttsd2si`/`cvttss2si`: truncate toward zero, and x86's
                    // answer for NaN or an out-of-range source is the destination
                    // width's sign bit (0x80000000 / 0x8000000000000000). A Rust
                    // `as` SATURATES, so the rule is spelled out here; `emit_c`
                    // emits the same guard in C, because the same cast is
                    // undefined behaviour in C and must not be relied on.
                    CastKind::FpToSI => {
                        let dw = t.width().unwrap_or(sw);
                        Val::Int(dw.trunc(fptosi_x86(f, dw), false))
                    }
                    CastKind::FpToUI => {
                        let dw = t.width().unwrap_or(sw);
                        Val::Int(dw.trunc(fptoui_x86(f, dw), false))
                    }
                    CastKind::FPExt => Val::Fl(f),
                    CastKind::FPTrunc => Val::Fl(match t {
                        Ty::Float { w: FWidth::F32 } => f as f32 as f64,
                        _ => f,
                    }),
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
                        // The F32 store writes the f32 BITS. `(f as f32) as u64`
                        // -- what this used to do -- converts the value to an
                        // integer, so `movss` of 1.5 stored 1.
                        Ty::Float { w: FWidth::F32 } => float_bits(FWidth::F32, f),
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
                let v = match callee {
                    // A sibling in the model: run ITS body, here, with the
                    // arguments bound to its parameters. This is what lets the
                    // strict interpreter see through the call — a value the
                    // callee never defines faults inside the callee, and a
                    // callee that is not in the program at all is a fault, not a
                    // hash.
                    Callee::Symbol(s) => {
                        self.effects.push(Effect::Call(s.clone(), abits.clone()));
                        match self.program.iter().find(|g| g.name == *s) {
                            Some(callee) => {
                                let r = self.exec_callee(callee, &abits)?;
                                match ret_ty {
                                    Ty::Void => Val::Int(0),
                                    Ty::Bool => Val::Bl(r.bits() != 0),
                                    // The callee answered a float; the caller's
                                    // declared return type says at which
                                    // precision to read it.
                                    Ty::Float { w } => match r {
                                        Val::Fl(f) => Val::Fl(f),
                                        other => Val::Fl(float_of_bits(*w, other.bits())),
                                    },
                                    _ => Val::Int(r.bits()),
                                }
                            }
                            // Named, and not in the model: a hole, said out loud.
                            // The one exception is the call that closes a
                            // `NoReturn` block: it defines nothing and control
                            // never comes back, so the callee's absence cannot
                            // change the answer — there is no answer. The run
                            // ends at the call, naming that as the cause.
                            None => {
                                if self.noret.contains(&dest) {
                                    self.effects.push(Effect::Exited);
                                    return Err(Fault::NoReturn);
                                }
                                return Err(Fault::UnknownCallee);
                            }
                        }
                    }
                    // A target outside the object (an import) or through a
                    // register: the world's answer, as a deterministic function
                    // of the call, so both executions of a TV pair agree.
                    _ => {
                        let name = match callee {
                            Callee::Addr(a) => format!("0x{a:x}"),
                            Callee::Indirect(v) => format!("[{}]", self.fetch(*v)?),
                            Callee::Symbol(s) => s.clone(),
                        };
                        self.effects.push(Effect::Call(name.clone(), abits.clone()));
                        let r = call_result(&name, &abits);
                        match ret_ty {
                            Ty::Void => Val::Int(0),
                            Ty::Bool => Val::Bl(r != 0),
                            Ty::Float { .. } => Val::Fl(f64::from_bits(r)),
                            _ => Val::Int(r),
                        }
                    }
                };
                self.set(dest, v);
                let _ = pure;
            }
            // The register's value is destroyed. There is no bit pattern to
            // write, and 0 is a guess: fault.
            Op::Clobber => return Err(Fault::Clobbered(dest)),
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
            BinOp::Shr => {
                // The machine masks the shift count (5 bits for 8/16/32-bit
                // operands, 6 for 64). The interpreter is public: passes-
                // fused and fuzz-built IR present counts directly, and the
                // old `y.min(128)` was UB for a u64 in 64..=128 — debug
                // panics, RELEASE WRAPS (found by the P-track suite
                // `interp_arith_reference.rs`, 2026-11-06). Masking here is
                // identity on every lifted count (the lifter masks), so the
                // corpus floors cannot move; it makes the arms TOTAL and
                // machine-faithful for IR that bypasses the lifter.
                let n = (y & if w == Width::W64 { 63 } else { 31 }) as u32;
                w.trunc(w.trunc(x, false) >> n, false)
            }
            BinOp::Sar => {
                let n = (y & if w == Width::W64 { 63 } else { 31 }) as u32;
                w.trunc((xs >> n) as u64, false)
            }
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
                // CF is the borrow out of the width's top bit: for an N-bit
                // `a - b - bin`, CF = (a as unsigned N bits) < (b + bin). The
                // operands are truncated FIRST, because a value carried in a
                // 64-bit cell can hold bits above a 32-bit operation, and only
                // the N low bits take part in the borrow.
                //
                // The old spelling - `(x - y) & !mask != 0` after widening - is
                // the same test for every width but one: for N = 64 the
                // complement of the mask is 0, so it answered "no borrow" for
                // every 64-bit subtraction. `subq $0x7,%rax; ja` - the dense
                // switch's bounds guard, at 0x10 of
                // `corpus/obj/switch_dense.O0.obj` - then read "above" for every
                // index below the table, and only the table's last case was
                // reachable. Widening to 128 bits and comparing the truncated
                // operands is exact at all five widths.
                let (xt, yt) = (w.trunc(x, false), w.trunc(y, false));
                d.cf = Some((xt as u128) < (yt as u128) + bin as u128);
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
                        FlagsOp::Shl { .. } => {
                            // OF for a 1-bit shift is "the sign changed", which
                            // needs the CF the shift just set. The FlagsOp node
                            // does not carry CF, so a node whose flag dict has
                            // none is a lifted IR that this interpreter cannot
                            // execute: fault LOUD (the message names the
                            // operation) - an `unwrap()` here is a panic in the
                            // middle of a T2 comparison, and a silent `None`
                            // would be a missing flag nobody notices.
                            let Some(cf) = d.cf else {
                                return Err(Fault::MissingFlag("shl"));
                            };
                            Some(bit(res, w.bits() - 1) != cf)
                        }
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
            FlagsOp::Inc { a, cf_in } | FlagsOp::Dec { a, cf_in } => {
                let x = g(a)?;
                let w = self.width_of(a);
                let inc = matches!(fo, FlagsOp::Inc { .. });
                let res = if inc {
                    w.trunc(x.wrapping_add(1), false)
                } else {
                    w.trunc(x.wrapping_sub(1), false)
                };
                // CF is PRESERVED across inc/dec, and the flags node carries the
                // value it preserves: read the CF out of THAT value. With none
                // carried there is nothing to preserve — silicon's CF at function
                // entry is the caller's — so it stays undefined and a read of it
                // faults, rather than answering 0.
                d.cf = match cf_in {
                    Some(p) => match self.get(p) {
                        Some(Val::Fg(bits)) => bits.cf,
                        _ => None,
                    },
                    None => None,
                };
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

/// Shift-left defined for every count by the machine's own masking rule
/// (docs/ir-semantics.md: "Shift-count masking is hardware, not taste").
/// The old "counts at or beyond the width produce zero" was NOT what the
/// masked hardware does for raw counts: `1 << 32` at W32 masks to `<< 0`
/// and yields 1, not 0. The lifter pre-masks (`count_for`), so this arm is
/// identity for every lifted count; it is total and machine-faithful for
/// IR built directly (passes fusion, fuzz).
fn shift_left(x: u64, count: u64, w: Width) -> u64 {
    // Machine masking, same rule as Shr/Sar (see the note there): the old
    // `count >= 64 -> 0` disagreed with the machine for a raw count of
    // 32..63 at width 32 (x86 masks to 5 bits: `1 << 32` at W32 is `1 << 0`).
    let n = (count & if w == Width::W64 { 63 } else { 31 }) as u32;
    w.trunc(((x as u128) << n) as u64, false)
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

// Unit tests live at the BOTTOM of the file: the panic-site ratchet
// (`scripts/unwrap_ratchet.sh`) cuts each file at its first `#[cfg(test)]`, so a
// mid-file test module makes the ratchet blind to every production line below it.
#[cfg(test)]
mod x86_cast_tests {
    use super::*;
    #[test]
    fn fptosi_w32_window_is_the_machines_and_the_answer_is_32_bits() {
        // In range: -889062960 is INSIDE [-2^31, 2^31) — the earlier version of
        // this test asserted the magic here on a magnitude comparison it got
        // backwards. Silicon: `cvtsi2sd xmm0,eax(-889062960)` is exact as an
        // f64, `cvttsd2si eax,xmm0` answers 0xCB01F9D0 (probe run on this host).
        assert_eq!(fptosi_x86(-889062960.0, Width::W32), 0xCB01_F9D0);
        // Out of range on the high side: the window is exclusive at 2^31, so
        // 2^31 itself is the magic — the value the f32 round of 2147483647
        // lands on (see cvttss2si_answers_the_machine_f32_rounding_and_the_i32_magic).
        assert_eq!(fptosi_x86(2_147_483_648.0, Width::W32), 0x8000_0000);
        assert_eq!(fptosi_x86(3405904320.0, Width::W32), 0x8000_0000);
        // In-range W32 answers are 32 bits — no 64-bit sign extension. That is
        // the defect this arm fixes: a 30-bit answer returned as 64 bits shows
        // its sign bits through the `zext` into the register cell.
        assert_eq!(fptosi_x86(-1.0, Width::W32), 0xFFFF_FFFF);
        assert_eq!(fptosi_x86(12582912.0, Width::W32), 12582912);
        assert_eq!(fptosi_x86(-889062960.0, Width::W64), (-889062960i64) as u64);
        // -2^31 itself converts exactly (the window is inclusive at the low
        // end), so 0x80000000 has two causes and this is the honest one.
        assert_eq!(fptosi_x86(-2147483648.0, Width::W32), 0x8000_0000);
        // NaN is invalid operation on every width.
        assert_eq!(fptosi_x86(f64::NAN, Width::W32), 0x8000_0000);
        assert_eq!(fptosi_x86(f64::NAN, Width::W64), 0x8000_0000_0000_0000);
    }
}
