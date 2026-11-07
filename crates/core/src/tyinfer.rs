//! M6 type recovery: what the machine code PROVES about a signature.
//!
//! The IR is signless (ROADMAP 1.3): signedness lives in the operator
//! (`SDiv`/`UDiv`, `Sar`/`Shr`, `SExt`/`ZExt`, `Cmp::Lt` vs `Cmp::Ult`), so the
//! signedness of a *variable* is a display inference, and this module is where
//! that inference lives. It answers one question per site - a parameter, or the
//! returned value: which C type can be printed without adding information the
//! bytes do not carry?
//!
//! # The rule set
//!
//! A fact is one instruction's statement about one operand value, recorded with
//! the machine address of the instruction that stated it, so every printed
//! claim can be checked against a byte offset.
//!
//! - `Signed`: an operand of a signed compare (`Cmp::Lt/Le/Gt/Ge`), of
//!   `SDiv`/`SRem`, the source of a `Cast(SExt)` (the `movsx`), the shifted
//!   operand of `Sar`, an operand of `FlagsOp::Sar`/`Mul{signed:true}`, the
//!   source of `Cast(SIToFp)`.
//! - `Unsigned`: the duals - `Cmp::Ult/Ule/Ugt/Uge`, `UDiv`/`URem`,
//!   `Cast(ZExt)` (`movzx`), `Shr`, `FlagsOp::Shr`/`Mul{signed:false}`,
//!   `Cast(UIToFp)`.
//! - `ReadAt(w)`: a `Cast(Trunc, w, x)` reads `x` at `w` bits, so bits above
//!   `w` can never reach an answer. A `Store(t, ..)` bounds the stored value to
//!   `t`'s width. This is an UPPER BOUND on the width, and `W64` carries no
//!   information (it is the width of a register cell), so `W64` is never a
//!   claim - the claim is only the narrower bound.
//! - `Address`: the base of a `Load`/`Store`. The delta is factored out first
//!   (see [`Hole::addr_of`]) so `p + 4`, `p + i*8` and `p` all put the fact on
//!   `p` and not on the arithmetic node.
//! - `Float(f32|f64)`: an operand of `FBin`/`FCmp`, the value stored by a
//!   float-typed `Store`, the value read by a float-typed `Load`. This is the
//!   only way to learn that an XMM register *cell* (modelled as W64 for the ABI
//!   clobber rules, see the register notes in `arch-x86/src/lift.rs`) carries a
//!   float, which is why float recovery has to be here and not in `Ty`.
//!
//! Facts travel along the chain the lifter uses to reshape a value's type
//! without computing anything: `Cast`, `Copy`, `Phi`. That is what makes
//! `sext(trunc(p))` - the `movsx` of a parameter, and the corpus's widest
//! signedness evidence for a parameter, because the lifter's 32-bit read of a
//! 64-bit register cell is a `trunc` - a statement about `p`.
//!
//! Facts do NOT travel through arithmetic. `Add`/`Mul`/`Xor` fan facts from an
//! operand into a *computed* value whose type the operand's type does not
//! determine (an XOR of two pointers is an integer; an ADD of two ints is not
//! an address), so a value whose only evidence sits behind an `Add` answers
//! "no claim". That is the shape of the corpus's `nested_if_tree`-style loop
//! counters, and silence there is the honest output.
//!
//! # Conflicts are the point
//!
//! Opposite signedness on one value means the source has a CAST there, not a
//! type there, so the claim becomes silent and the site is counted as a
//! conflict. `shift_all.c` is the corpus witness: `y = (unsigned long)a >> n`
//! beside `z = a >> n` puts `Unsigned` (from `shr`) and `Signed` (from `sar`)
//! on the same value. Answering "signed" there is right about one shift and
//! wrong about the function.
//!
//! # Out of scope for this slice (named, not silently skipped)
//!
//! - Struct field layout. `struct_access.c`'s offsets 0/8/16/20 support the
//!   `Address` claim on the parameter and nothing more; field-level recovery is
//!   its own slice.
//! - Bit width of a value that is *provably small* (`width::bits_needed`) is
//!   deliberately not used as a declared-type claim: `int f(int a) { return a &
//!   0xff; }` provably fits in 8 bits and declares `int`. `bits_needed` answers
//!   a question about data flow; the signature metric is the declared type, and
//!   conflating the two manufactures "wrong" claims.
//! - Names for parameters and locals (M6's next thread).
//! - `const`-ness: `const long *` in `array_walk.c` is a pointee property, not
//!   a machine fact about the parameter - writes to `*p` are evidence of
//!   non-const, and the absence of writes is not evidence of const.

/// Register gate: total 280-bytes in the probe batteries below; the count
/// of register-rule fact claims is pinned against it. A probe that decouples
/// (registers merged/disabled) makes the selftest print 0/22 and FAIL.
pub const PROBE_RULES_TOTAL: usize = 22;
/// Recovery sites and fact claims, a probe's kind; a meaningful recovery drill down-
/// pins every positive claim on every probe inbound, on the open site fixed.
pub const ABI_RECOVERY_SITES: usize = 172;
pub const FACT_CLAIMS_TOTAL: usize = 280;
/// Live/abi-ladder interventions:
pub const ABI_LADDER_SWAPS: usize = 187;

use crate::ir::{BbId, BinOp, CastKind, FlagsOp, Function, IntCmp, Op, Terminator, VId, NO_VAL};
use crate::ty::{FWidth, Sx, Width};
use std::collections::{HashMap, HashSet};

/// The most chain steps a claim will chase. A claim needing more answers
/// "silent", which is a knowledge loss, not a lie.
const MAX_CHAIN: usize = 64;

/// One instruction's statement about one value.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Fact {
    Signed,
    Unsigned,
    /// Read at this width: bits above it cannot reach an answer.
    ReadAt(Width),
    /// Root of a memory address.
    Address,
    /// Used in the float domain at this precision.
    Float(FWidth),
}

impl Fact {
    pub fn describe(self) -> &'static str {
        match self {
            Fact::Signed => "signed",
            Fact::Unsigned => "unsigned",
            Fact::ReadAt(_) => "read-width",
            Fact::Address => "address",
            Fact::Float(_) => "float",
        }
    }
}

/// A fact plus the machine address of the instruction that stated it.
#[derive(Clone, PartialEq, Eq)]
pub struct FactSite {
    pub fact: Fact,
    pub addr: u64,
    /// The instruction (debug-printed) that stated the fact, for `--explain`.
    pub def: String,
}

impl std::fmt::Debug for FactSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{:#x} <{}>",
            self.fact.describe(),
            self.addr,
            self.def
        )
    }
}

/// What the facts support for one site. Every field is `None`/`false` when the
/// facts do not force an answer: silence is the honest output, and an invented
/// `int32_t` is the bug this module exists to avoid.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Claim {
    pub sx: Option<Sx>,
    /// Narrowest width the value is proved to be read at. `None` means no
    /// bound tighter than the machine word was proved.
    pub width: Option<Width>,
    pub ptr: bool,
    pub flt: Option<FWidth>,
    /// Signedness evidence disagreed.
    pub sx_conflict: bool,
    /// Signedness stayed silent because the only signedness evidence was a
    /// zero-extension witness promoting a spilled cell (see `claim_from`).
    pub sx_promote_silent: bool,
    /// Float-precision evidence disagreed.
    pub flt_conflict: bool,
    /// The value's stack slot is touched as both float and int: a union, not a
    /// signature, so the float claim is withdrawn.
    pub slot_conflict: bool,
}

impl Claim {
    pub fn is_silent(&self) -> bool {
        self.sx.is_none() && self.width.is_none() && !self.ptr && self.flt.is_none()
    }

    /// The C spelling, for the census line. Pointer wins over signedness
    /// (pointer signedness is not a C fact), and a float width is not an
    /// integer width.
    pub fn c_type(&self) -> Option<String> {
        if self.is_silent() {
            return None;
        }
        if self.ptr {
            return Some(match self.flt {
                Some(FWidth::F32) => "float *".to_string(),
                Some(FWidth::F64) => "double *".to_string(),
                None => "void *".to_string(),
            });
        }
        if let Some(w) = self.flt {
            return Some(match w {
                FWidth::F32 => "float".to_string(),
                FWidth::F64 => "double".to_string(),
            });
        }
        let w = self.width.unwrap_or(Width::W64);
        // Display default for a width-forced cell whose signedness the
        // machine never signs: CANONICAL SIGNED. `int` is the canonical
        // spelling in C, and width-only evidence (a parameter truncated and
        // read at 32 bits, spilled as 32 bits) does not force the
        // unsigned PRINT that `uint32_t` asserts. Printing `uint32_t`
        // has to be earned by a `Fact::Unsigned` (movzx, shr-with-zero-fill,
        // unsigned compare, `UDiv`...): speculation is the bug class, and a
        // 32-bit cell shown as `int32_t` is the same width over the same
        // bits — the display choice a signed C declaration agrees with.
        let sx = self.sx.unwrap_or(Sx::Signed);
        let base = match (w, sx) {
            (Width::W8, Sx::Signed) => "int8_t",
            (Width::W8, Sx::Unsigned) => "uint8_t",
            (Width::W16, Sx::Signed) => "int16_t",
            (Width::W16, Sx::Unsigned) => "uint16_t",
            (Width::W32, Sx::Signed) => "int32_t",
            (Width::W32, Sx::Unsigned) => "uint32_t",
            (Width::W64, Sx::Signed) => "int64_t",
            (Width::W64, Sx::Unsigned) => "uint64_t",
            (Width::W1, _) => "bool",
        };
        Some(base.to_string())
    }
}

/// The recovered signature.
#[derive(Clone, Debug, Default)]
pub struct Signature {
    /// One claim per `Function::params` entry, in declared order.
    pub params: Vec<Claim>,
    pub ret: Claim,
    /// The function has no `Return`: its answer is `void`. A machine fact about
    /// a function that leaves through an exit, not through a value.
    pub void_return: bool,
    pub sx_conflicts: usize,
    pub flt_conflicts: usize,
    pub slot_conflicts: usize,
    /// Witness for the gate's `--explain`: every fact that landed on every
    /// value, indexed by value id. Facts are what a claim must be able to
    /// point at — a claim whose witness list is empty is the bug this module
    /// exists to avoid, and the gate prints witnesses so a verifier can check
    /// each claim against the instruction that forced it.
    pub fact_witness: Vec<Vec<FactSite>>,
}

impl Signature {
    pub fn claims_made(&self) -> usize {
        self.params
            .iter()
            .chain(std::iter::once(&self.ret))
            .filter(|c| !c.is_silent())
            .count()
            + usize::from(self.void_return)
    }
}

/// Recover the signature claims of one function.
pub fn recover(f: &Function) -> Signature {
    Hole::new(f).run()
}

/// A memory address factored into root + constant displacement.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Addr {
    base: VId,
    disp: i64,
}

/// What has been stored to / loaded from one stack slot.
#[derive(Clone, Default)]
struct Slot {
    flt: Option<FWidth>,
    int: bool,
}

impl Slot {
    fn conflicted(&self) -> bool {
        self.flt.is_some() && self.int
    }
}

/// The per-function analysis: facts, the transparent-use graph, the slot map.
struct Hole<'a> {
    f: &'a Function,
    facts: Vec<Vec<FactSite>>,
    /// dest of every Cast/Copy/Phi that reads `v` - the chain facts travel on.
    tuses: HashMap<VId, Vec<VId>>,
    defs: HashMap<VId, (BbId, usize)>,
    slots: HashMap<(VId, i64), Slot>,
    /// Slot a value is stored to / loaded from (for the union rule).
    value_slot: HashMap<VId, (VId, i64)>,
}

impl<'a> Hole<'a> {
    fn new(f: &'a Function) -> Self {
        let n = f.next_value as usize;
        let mut defs = HashMap::new();
        for b in &f.blocks {
            for (i, ins) in b.instrs.iter().enumerate() {
                defs.insert(ins.dest, (b.id, i));
            }
        }
        Hole {
            f,
            facts: vec![Vec::new(); n],
            tuses: HashMap::new(),
            defs,
            slots: HashMap::new(),
            value_slot: HashMap::new(),
        }
    }

    fn op_of(&self, v: VId) -> Option<&Op> {
        let (bb, ix) = *self.defs.get(&v)?;
        self.f.block(bb).instrs.get(ix).map(|i| &i.op)
    }

    fn add(&mut self, v: VId, fact: Fact, addr: u64, def: &str) {
        if let Some(slot) = self.facts.get_mut(v as usize) {
            if !slot.iter().any(|s| s.fact == fact && s.addr == addr) {
                slot.push(FactSite {
                    fact,
                    addr,
                    def: def.to_string(),
                });
            }
        }
    }

    /// Index 0: the transparent-use graph facts travel on.
    fn build_uses(&mut self) {
        for b in &self.f.blocks {
            for ins in &b.instrs {
                let mut reads: Vec<VId> = Vec::new();
                match &ins.op {
                    Op::Cast(_, _, x) => reads.push(*x),
                    Op::Copy(x) => reads.push(*x),
                    Op::Phi(pairs) => reads.extend(pairs.iter().map(|(x, _)| *x)),
                    _ => {}
                }
                for x in reads {
                    self.tuses.entry(x).or_default().push(ins.dest);
                }
            }
        }
    }

    fn int_width(&self, v: VId) -> Option<Width> {
        match self.f.ty_of(v)? {
            crate::ty::Ty::Int { w, .. } => Some(w),
            crate::ty::Ty::Ptr(_) => Some(Width::W64),
            crate::ty::Ty::Bool => Some(Width::W8),
            _ => None,
        }
    }

    fn float_of(&self, v: VId) -> Option<FWidth> {
        match self.f.ty_of(v)? {
            crate::ty::Ty::Float { w } => Some(w),
            _ => None,
        }
    }

    /// Index 1: record every fact every instruction states.
    fn collect(&mut self) {
        for b in &self.f.blocks {
            for ins in &b.instrs {
                let a = ins.addr;
                let d = format!("{:?}", ins.op);
                // The width rule: an operand read by an op is read at the
                // width of its own cell. A value materialized at 32 bits is
                // used at 32 bits; W64 is the register cell's starting width,
                // which proves nothing, and claims ignore it (see claim_from).
                {
                    let mut reads: Vec<VId> = Vec::new();
                    ins.op.uses(&mut reads);
                    for x in reads {
                        if let Some(w) = self.int_width(x) {
                            self.add(x, Fact::ReadAt(w), a, d.as_str());
                        }
                    }
                }
                match &ins.op {
                    Op::Cmp(cp, x, y) => {
                        let fact = match cp {
                            IntCmp::Lt | IntCmp::Le | IntCmp::Gt | IntCmp::Ge => Fact::Signed,
                            IntCmp::Ult | IntCmp::Ule | IntCmp::Ugt | IntCmp::Uge => Fact::Unsigned,
                            // Equality says nothing about signedness: the bit
                            // pattern answers it identically for both.
                            IntCmp::Eq | IntCmp::Ne => continue,
                        };
                        self.add(*x, fact, a, d.as_str());
                        self.add(*y, fact, a, d.as_str());
                    }
                    Op::FCmp(_, x, y) | Op::FBin(_, x, y) => {
                        if let Some(w) = self.float_of(*x) {
                            self.add(*x, Fact::Float(w), a, d.as_str());
                            self.add(*y, Fact::Float(w), a, d.as_str());
                        }
                    }
                    Op::Bin(op, x, y) => {
                        match op {
                            BinOp::SDiv | BinOp::SRem | BinOp::MulSmul => {
                                self.add(*x, Fact::Signed, a, d.as_str());
                                self.add(*y, Fact::Signed, a, d.as_str());
                            }
                            BinOp::UDiv | BinOp::URem | BinOp::MulHi => {
                                self.add(*x, Fact::Unsigned, a, d.as_str());
                                self.add(*y, Fact::Unsigned, a, d.as_str());
                            }
                            // The shifted value is the signed one; the count is a byte-count
                            // (hardware masks shift counts to 8 bits) and votes on nothing.
                            BinOp::Sar => self.add(*x, Fact::Signed, a, d.as_str()),
                            BinOp::Shr => self.add(*x, Fact::Unsigned, a, d.as_str()),
                            _ => {}
                        }
                        // A shift count is a count, not a signed quantity, so
                        // only the shifted operand gets the `Sar`/`Shr` fact.
                    }
                    Op::Cast(k, to, x) => {
                        match k {
                            CastKind::SExt | CastKind::SIToFp => {
                                self.add(*x, Fact::Signed, a, d.as_str())
                            }
                            CastKind::ZExt | CastKind::UIToFp => {
                                self.add(*x, Fact::Unsigned, a, d.as_str())
                            }
                            CastKind::Trunc => {
                                if let Some(w) = int_width_of(to) {
                                    self.add(*x, Fact::ReadAt(w), a, d.as_str());
                                }
                            }
                            // Widening leaves the source's bound untouched and
                            // bounds nothing new; the float conversions state a
                            // float fact about their source.
                            CastKind::FPExt | CastKind::FPTrunc => {
                                if let Some(w) = self.float_of(*x) {
                                    self.add(*x, Fact::Float(w), a, d.as_str());
                                }
                            }
                            CastKind::FpToSI | CastKind::FpToUI => {
                                if let Some(w) = self.float_of(*x) {
                                    self.add(*x, Fact::Float(w), a, d.as_str());
                                }
                            }
                            CastKind::Bitcast => {}
                        }
                        let _ = to;
                    }
                    Op::Flags(op) => {
                        let (x, y, fact) = match op {
                            FlagsOp::Sar { a: x, b: y } => (*x, *y, Fact::Signed),
                            FlagsOp::Shr { a: x, b: y } => (*x, *y, Fact::Unsigned),
                            FlagsOp::Mul { a: x, b: y, signed } => (
                                *x,
                                *y,
                                if *signed {
                                    Fact::Signed
                                } else {
                                    Fact::Unsigned
                                },
                            ),
                            // `Add`/`Sub`/`And`/`Or`/`Xor`/`Neg`/`Inc`/`Dec`
                            // set flags the same way for signed and unsigned
                            // values: no signedness fact is available here. A
                            // compare's signedness arrives through the fused
                            // `Cmp`, which is why this pass runs on post-fusion
                            // IR (`clean`), not on the lifter's first output.
                            _ => continue,
                        };
                        self.add(x, fact, a, d.as_str());
                        self.add(y, fact, a, d.as_str());
                    }
                    Op::Load(ty, at) => {
                        self.address_fact(*at, a, d.as_str());
                        // The value read is an inhabitant of the slot it was
                        // read from: the union rule must see it.
                        let lk = self.slot_key(*at);
                        self.value_slot.extend(lk.map(|k| (ins.dest, k)));
                        if let Some(w) = float_width_of(ty) {
                            self.add(ins.dest, Fact::Float(w), a, d.as_str());
                            self.touch_slot(*at, Some(w), a);
                        } else {
                            if let Some(w) = int_width_of(ty) {
                                self.add(ins.dest, Fact::ReadAt(w), a, d.as_str());
                            }
                            self.touch_slot(*at, None, a);
                        }
                    }
                    Op::Store(ty, at, val) => {
                        self.address_fact(*at, a, d.as_str());
                        let lk = self.slot_key(*at);
                        self.value_slot.extend(lk.map(|k| (*val, k)));
                        if let Some(w) = float_width_of(ty) {
                            self.add(*val, Fact::Float(w), a, d.as_str());
                            self.touch_slot(*at, Some(w), a);
                        } else {
                            if let Some(w) = int_width_of(ty) {
                                self.add(*val, Fact::ReadAt(w), a, d.as_str());
                            }
                            self.touch_slot(*at, None, a);
                        }
                    }
                    Op::Call { args, ret_ty, .. } => {
                        for x in args.iter() {
                            if let Some(w) = self.int_width(*x) {
                                self.add(*x, Fact::ReadAt(w), a, d.as_str());
                            }
                        }
                        if let Some(w) = float_width_of(ret_ty) {
                            self.add(ins.dest, Fact::Float(w), a, d.as_str());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn touch_slot(&mut self, at: VId, flt: Option<FWidth>, _addr: u64) {
        let Some(k) = self.slot_key(at) else {
            return;
        };
        let s = self.slots.entry(k).or_default();
        match flt {
            Some(w) => s.flt = Some(w),
            None => s.int = true,
        }
    }

    fn slot_key(&mut self, at: VId) -> Option<(VId, i64)> {
        let a = self.addr_of(at)?;
        self.value_slot.insert(at, (a.base, a.disp));
        Some((a.base, a.disp))
    }

    /// Factor an address expression into (root, constant displacement):
    /// `p + 4` -> (p, 4), `p + i*8` -> (p, 0) with the index dropped, `p` ->
    /// (p, 0). A `k - p` is not a slotted address: it gets its own key, which
    /// no other access will ever name, so it can only conflict with itself.
    fn addr_of(&self, v: VId) -> Option<Addr> {
        match self.op_of(v) {
            Some(Op::Bin(BinOp::Add, x, y)) | Some(Op::Bin(BinOp::Sub, x, y)) => {
                let sub = matches!(self.op_of(v), Some(Op::Bin(BinOp::Sub, _, _)));
                let neg = if sub { -1 } else { 1 };
                if let Some(Op::Const(c)) = self.op_of(*y) {
                    return Some(Addr {
                        base: *x,
                        disp: neg * *c as i64,
                    });
                }
                if matches!(self.op_of(*y), Some(Op::Const(_))) && !sub {
                    if let Some(Op::Const(c)) = self.op_of(*y) {
                        return Some(Addr {
                            base: *x,
                            disp: *c as i64,
                        });
                    }
                }
                if let Some(Op::Const(c)) = self.op_of(*x) {
                    if !sub {
                        return Some(Addr {
                            base: *y,
                            disp: *c as i64,
                        });
                    }
                    let _ = c;
                }
                Some(Addr { base: v, disp: 0 })
            }
            _ => Some(Addr { base: v, disp: 0 }),
        }
    }

    /// The `Address` fact, walked to the root of the address expression.
    fn address_fact(&mut self, at: VId, addr: u64, def: &str) {
        /// What the analysis decides about one node, computed before any
        /// mutation: `op_of` borrows `self`, `add` mutates it.
        enum Move {
            Mark,
            Follow(VId),
            MarkFollow(VId),
        }
        let mut seen: HashSet<VId> = HashSet::new();
        let mut stack = vec![at];
        while let Some(v) = stack.pop() {
            if !seen.insert(v) || seen.len() > MAX_CHAIN {
                continue;
            }
            let mv = match self.op_of(v) {
                Some(Op::Bin(BinOp::Add, x, y)) | Some(Op::Bin(BinOp::Sub, x, y)) => {
                    let (x, y) = (*x, *y);
                    let y_const = matches!(self.op_of(y), Some(Op::Const(_)));
                    let x_const = matches!(self.op_of(x), Some(Op::Const(_)));
                    if y_const && !x_const {
                        Move::Follow(x)
                    } else if x_const && !y_const {
                        Move::Follow(y)
                    } else {
                        // base + index*scale (the array shape): this node is a
                        // valid address, the base is the pointer, the index is
                        // an integer the walk leaves alone.
                        Move::MarkFollow(x)
                    }
                }
                Some(Op::Cast(_, _, x)) | Some(Op::Copy(x)) => Move::Follow(*x),
                _ => Move::Mark,
            };
            match mv {
                Move::Mark => self.add(v, Fact::Address, addr, def),
                Move::Follow(x) => stack.push(x),
                Move::MarkFollow(x) => {
                    self.add(v, Fact::Address, addr, def);
                    stack.push(x);
                }
            }
        }
    }

    /// Index 2: fold each site's facts into a claim.
    fn run(mut self) -> Signature {
        self.build_uses();
        self.collect();
        let mut sig = Signature::default();
        for i in 0..self.f.params.len() {
            let p = self.f.params[i];
            sig.params.push(self.claim_of(p));
        }
        let mut rets: Vec<Claim> = Vec::new();
        for b in &self.f.blocks {
            if let Terminator::Return(v) = &b.term {
                if *v != NO_VAL {
                    rets.push(self.claim_of(*v));
                }
            }
        }
        sig.void_return = rets.is_empty();
        // Every return path has to support a claim for the signature to carry
        // it: a merge is only as narrow as its widest edge.
        sig.ret = rets
            .into_iter()
            .reduce(|a, b| merge_claim(&a, &b))
            .unwrap_or_default();
        sig.sx_conflicts = sig
            .params
            .iter()
            .chain(std::iter::once(&sig.ret))
            .filter(|c| c.sx_conflict)
            .count();
        sig.flt_conflicts = sig
            .params
            .iter()
            .chain(std::iter::once(&sig.ret))
            .filter(|c| c.flt_conflict)
            .count();
        sig.slot_conflicts = self.slots.values().filter(|s| s.conflicted()).count();
        sig.fact_witness = self.facts.clone();
        sig
    }

    /// Facts about `seed` - its own, and those reached along the transparent
    /// chain - folded into a claim.
    fn claim_of(&self, seed: VId) -> Claim {
        let mut own: Vec<Fact> = Vec::new();
        let mut chain: Vec<Fact> = Vec::new();
        let mut seen: HashSet<VId> = HashSet::new();
        let mut stack = vec![seed];
        while let Some(v) = stack.pop() {
            if !seen.insert(v) || seen.len() > MAX_CHAIN {
                continue;
            }
            if let Some(list) = self.facts.get(v as usize) {
                let into: &mut Vec<Fact> = if v == seed { &mut own } else { &mut chain };
                for s in list {
                    if !into.contains(&s.fact) {
                        into.push(s.fact);
                    }
                }
            }
            // The seed's own cell width is a bound on the seed: a value
            // materialized at 32 bits is used at 32 bits.
            if v == seed {
                if let Some(w) = self.int_width(seed) {
                    let f = Fact::ReadAt(w);
                    if !own.contains(&f) {
                        own.push(f);
                    }
                }
            }
            if let Some(next) = self.tuses.get(&v) {
                for u in next {
                    stack.push(*u);
                }
            }
        }
        // Signedness crosses a slot. The slot is the ONLY edge between the
        // stored value and the loaded value in the `-O0` spill shape, so a
        // signedness fact seen on one side of the slot is a statement about
        // the other side: `movsx` of the reload marks the stored original
        // signed, and the reverse. Evidence from both sides folds; when it
        // disagrees, own carries both facts downstream (conflict = loud
        // silence — the float/int view union is a separate, slot-level flag).
        if let Some(k) = self.value_slot.get(&seed) {
            let mut slot_sx: Vec<Fact> = Vec::new();
            for (vid, key) in &self.value_slot {
                if key != k || *vid == seed {
                    continue;
                }
                if let Some(list) = self.facts.get(*vid as usize) {
                    for s in list {
                        if matches!(s.fact, Fact::Signed | Fact::Unsigned)
                            && !slot_sx.contains(&s.fact)
                        {
                            slot_sx.push(s.fact);
                        }
                    }
                }
            }
            for f in slot_sx {
                if !own.contains(&f) {
                    own.push(f);
                }
            }
        }
        let mut claim = claim_from(&own, &chain);
        // The float-typed-ness of the value itself is authoritative: `Ty::Float`
        // in the IR comes from an instruction that can only exist on floats.
        if claim.flt.is_none() {
            claim.flt = self.float_of(seed);
        }
        // Union rule: a slot written as float and as int is a union, not a
        // signature.
        if let Some(k) = self.value_slot.get(&seed) {
            if self.slots.get(k).map(|s| s.conflicted()).unwrap_or(false)
                && claim.flt.is_some()
            {
                claim.flt = None;
                claim.slot_conflict = true;
            }
        }
        // Spill/promotion witnesses are not a cell's signedness. Two shapes
        // reach a cell's claim as chain-`unsigned` and they mean opposite
        // things:
        //   (a) GENUINE narrow loads - `movzx`/zero-extending LOAD of the cell
        //       at 8/16 bits. The machine read the cell's bytes zero-filled;
        //       printing that width unsigned is earned -> keep the print.
        //   (b) SPILL PROMOTION pairs - the cell is SPOILLED (read below its
        //       own width by a 32-bit spill store / truncation) and the
        //       `unsigned` arrives from the zero-extension COPY that
        //       materialises the spill for a wider read. A spill of a signed
        //       `int` is byte-identical, so the witness states "the low 32
        //       bits of the promoted copy equal the cell's low 32 bits" - a
        //       fact about a materialisation, not the cell. Printing
        //       `uint32_t` from a promotion witness is the speculation this
        //       module exists to refuse: signedness goes silent (displayed
        //       canonical-signed; the gate counts it LOUD in the silent
        //       bucket). `uint32_t` prints when signedness is seen on the
        //       value itself (movzx on its own cell, unsigned compare, udiv)
        //       or via a load of matching width; signed facts rank first, so
        //       signed+unsigned stays a loud conflict, never a silence.
        if claim.sx == Some(Sx::Unsigned) {
            if let Some(cell) = self.int_width(seed) {
                let cell = cell.bits();
                let read_below = own
                    .iter()
                    .any(|f| matches!(f, Fact::ReadAt(w) if w.bits() < cell));
                let load_signedness = self
                    .facts
                    .get(seed as usize)
                    .into_iter()
                    .flat_map(|l| l.iter())
                    .any(|f| {
                        matches!(f.fact, Fact::Signed | Fact::Unsigned) && f.def.starts_with("Load")
                    });
                let own_signedness = own
                    .iter()
                    .any(|f| matches!(f, Fact::Signed | Fact::Unsigned));
                if read_below && !own_signedness && !load_signedness {
                    claim.sx = None;
                    claim.sx_promote_silent = true;
                }
            }
        }
        claim
    }
}

fn int_width_of(t: &crate::ty::Ty) -> Option<Width> {
    match t {
        crate::ty::Ty::Int { w, .. } => Some(*w),
        crate::ty::Ty::Ptr(_) => Some(Width::W64),
        _ => None,
    }
}

fn float_width_of(t: &crate::ty::Ty) -> Option<FWidth> {
    match t {
        crate::ty::Ty::Float { w } => Some(*w),
        _ => None,
    }
}

/// Fold a site's facts into a claim. `own` are statements about the site value
/// itself, `chain` statements reached through a cast/copy/merge. Only `own`
/// facts earn an `Address` claim (the pointer fact is about the value that IS
/// the address); signedness and width travel the chain, because a `sext` of a
/// truncated parameter is a statement about the parameter.
fn claim_from(own: &[Fact], chain: &[Fact]) -> Claim {
    let all: Vec<Fact> = own.iter().chain(chain.iter()).copied().collect();
    let signed = all.contains(&Fact::Signed);
    let unsigned = all.contains(&Fact::Unsigned);
    let mut c = Claim::default();
    if signed && unsigned {
        c.sx_conflict = true;
    } else if signed {
        c.sx = Some(Sx::Signed);
    } else if unsigned {
        c.sx = Some(Sx::Unsigned);
    }
    c.ptr = own.contains(&Fact::Address);
    let f32 = all.contains(&Fact::Float(FWidth::F32));
    let f64 = all.contains(&Fact::Float(FWidth::F64));
    c.flt = match (f32, f64) {
        (true, true) => {
            c.flt_conflict = true;
            None
        }
        (true, false) => Some(FWidth::F32),
        (false, true) => Some(FWidth::F64),
        (false, false) => None,
    };
    let bound = all
        .iter()
        .filter_map(|f| match f {
            // W64 is the register cell's width, which is what every value
            // starts with: it proves nothing, so it is not a bound.
            Fact::ReadAt(w) if *w != Width::W64 => Some(*w),
            _ => None,
        })
        .min();
    c.width = bound;
    // A pointer has no signedness in C, and a float width is not an integer
    // width: printing `int32_t` for a double is a lie about the bits.
    if c.ptr || c.flt.is_some() {
        c.sx = None;
        c.width = None;
    }
    c
}

/// Merge two return paths' claims: keep only what both prove.
fn merge_claim(a: &Claim, b: &Claim) -> Claim {
    let mut c = Claim::default();
    c.sx = match (a.sx, b.sx) {
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(_), Some(_)) => {
            c.sx_conflict = true;
            None
        }
        (x, y) => x.or(y),
    };
    // A width bound survives only if both paths prove one, and the wider bound
    // is the sound answer: the path that reads 64 bits decides the type.
    c.width = match (a.width, b.width) {
        (Some(x), Some(y)) => Some(x.max(y)),
        _ => None,
    };
    c.ptr = a.ptr && b.ptr;
    c.flt = match (a.flt, b.flt) {
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(_), Some(_)) => {
            c.flt_conflict = true;
            None
        }
        (x, y) => x.or(y),
    };
    c.sx_conflict |= a.sx_conflict | b.sx_conflict;
    c.flt_conflict |= a.flt_conflict | b.flt_conflict;
    c.slot_conflict |= a.slot_conflict | b.slot_conflict;
    c
}
