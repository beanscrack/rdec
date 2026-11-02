//! x86 EFLAGS semantics, as a pure Rust model.
//!
//! This module is the *subject* of the T1 oracle (`rdec-verify flags`): every
//! definition here is checked against real EFLAGS read back with `pushfq` on
//! this machine, over a grid of operands including every boundary value. That is
//! the difference between a flag model that is believed and one that is measured.
//!
//! `None` means the flag is *undefined* for that instruction per the Intel SDM
//! (e.g. AF after a shift, OF after a shift by more than 1, the high flags after
//! a 16/32-bit shift in 64-bit mode). The model deliberately returns `None`
//! there instead of inventing a value: the lifter must not depend on undefined
//! flags, and the verifier is allowed to assert it.

use rdec_core::ir::Flag;
use rdec_core::ty::Width;

/// Which flags an op defines, with `None` = architecturally undefined.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FlagDef {
    pub cf: Option<bool>,
    pub pf: Option<bool>,
    pub af: Option<bool>,
    pub zf: Option<bool>,
    pub sf: Option<bool>,
    pub of: Option<bool>,
    /// Flags the instruction leaves at their previous value. The harness seeds
    /// EFLAGS with a known pattern so "unchanged" is testable.
    pub unchanged: &'static [rdec_core::ir::Flag],
}

impl FlagDef {
    pub fn get(&self, f: rdec_core::ir::Flag) -> Option<bool> {
        match f {
            Flag::CF => self.cf,
            Flag::PF => self.pf,
            Flag::AF => self.af,
            Flag::ZF => self.zf,
            Flag::SF => self.sf,
            Flag::OF => self.of,
        }
    }
    pub fn set(&mut self, f: rdec_core::ir::Flag, v: Option<bool>) {
        match f {
            Flag::CF => self.cf = v,
            Flag::PF => self.pf = v,
            Flag::AF => self.af = v,
            Flag::ZF => self.zf = v,
            Flag::SF => self.sf = v,
            Flag::OF => self.of = v,
        }
    }
    /// All six flags, for iteration in the harness.
    pub const ALL: [rdec_core::ir::Flag; 6] = [
        rdec_core::ir::Flag::CF,
        rdec_core::ir::Flag::PF,
        rdec_core::ir::Flag::AF,
        rdec_core::ir::Flag::ZF,
        rdec_core::ir::Flag::SF,
        rdec_core::ir::Flag::OF,
    ];
}

/// Shift counts are masked to 5 bits for 8/16/32-bit operands and to **6 bits**
/// for 64-bit operands (Intel Vol. 2, SHR/SHL: "the count is masked to 5 bits,
/// or to 6 bits if the operand size is 64 bits"). Getting this wrong turns
/// `shl rax, 36` into `shl rax, 4`, which is what the T1 grid found.
pub const fn mask_count(w: Width, count: u64) -> u32 {
    if w.bits() == 64 {
        (count & 0x3f) as u32
    } else {
        (count & 0x1f) as u32
    }
}

fn msb(w: Width) -> u32 {
    w.bits() - 1
}

fn as_u(w: Width, x: u64) -> u64 {
    w.trunc(x, false)
}

fn as_i(w: Width, x: u64) -> i64 {
    w.trunc(x, true) as i64
}

fn pf(res: u64) -> bool {
    (res as u8).count_ones() % 2 == 0
}

/// Bit `k` of `x`. Explicit helper because `x & y >> k` parses as `x & (y >> k)`
/// in Rust, which is exactly the bug this file needed a hardware oracle to find.
fn bit(x: u64, k: u32) -> bool {
    ((x >> k) & 1) == 1
}

fn af_add(a: u64, b: u64, res: u64) -> bool {
    bit(a ^ res ^ b, 4)
}

fn zf(w: Width, res: u64) -> bool {
    as_u(w, res) == 0
}
fn sf(w: Width, res: u64) -> bool {
    bit(as_u(w, res), msb(w))
}

// ---------------------------------------------------------------------------
// per-op definitions
// ---------------------------------------------------------------------------

/// `add a, b` (and the result value).
pub fn add(w: Width, a: u64, b: u64) -> (u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let res = as_u(w, a.wrapping_add(b));
    let mut d = FlagDef::default();
    d.cf = Some((a as u128 + b as u128) > w.mask() as u128);
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, b, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    // Signed overflow: both operands same sign, result opposite sign.
    d.of = Some(bit((a ^ res) & (b ^ res), msb(w)));
    (res, d)
}

/// `adc a, b` with an explicit carry-in (flags use the carry chain).
pub fn adc(w: Width, a: u64, b: u64, cin: bool) -> (u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let cin = cin as u64;
    let res = as_u(w, a.wrapping_add(b).wrapping_add(cin));
    let mut d = FlagDef::default();
    d.cf = Some((a as u128 + b as u128 + cin as u128) > w.mask() as u128);
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, b ^ cin, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    // Signed overflow: operands same sign, result opposite.
    let sa = (a >> msb(w)) & 1;
    let sb = (b >> msb(w)) & 1;
    let sr = (res >> msb(w)) & 1;
    d.of = Some(sa == sb && sr != sa);
    (res, d)
}

/// `sub a, b`.
pub fn sub(w: Width, a: u64, b: u64) -> (u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let res = as_u(w, a.wrapping_sub(b));
    let mut d = FlagDef::default();
    d.cf = Some(a < b);
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, b, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    // Signed overflow on a-b: operands of opposite sign and result sign != a.
    d.of = Some(bit((a ^ b) & (a ^ res), msb(w)));
    (res, d)
}

/// `sbb a, b`.
pub fn sbb(w: Width, a: u64, b: u64, bin: bool) -> (u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let bin = bin as u64;
    let res = as_u(w, a.wrapping_sub(b).wrapping_sub(bin));
    let mut d = FlagDef::default();
    d.cf = Some(((a as u128).wrapping_sub(b as u128 + bin as u128) & !w.mask() as u128) != 0
        || (a as u128) < (b as u128 + bin as u128));
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, b ^ bin, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    let sa = (a >> msb(w)) & 1;
    let sb = (b >> msb(w)) & 1;
    let sr = (res >> msb(w)) & 1;
    d.of = Some(sa != sb && sr != sa);
    (res, d)
}

/// `and a, b` / `test a, b` / `or` / `xor`: CF and OF are cleared.
pub fn logic(w: Width, a: u64, b: u64, op: LogicOp) -> (u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let res = match op {
        LogicOp::And => a & b,
        LogicOp::Or => a | b,
        LogicOp::Xor => a ^ b,
    };
    let mut d = FlagDef::default();
    d.cf = Some(false);
    d.pf = Some(pf(res));
    // AF is architecturally undefined for logic ops.
    d.af = None;
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = Some(false);
    (as_u(w, res), d)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
}

/// `shl a, n`
pub fn shl(w: Width, a: u64, count: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let n = mask_count(w, count);
    let mut d = FlagDef::default();
    if n == 0 {
        // A zero-count shift leaves every flag untouched.
        d.unchanged = &FlagDef::ALL;
        return (a, d);
    }
    let res = as_u(w, a << n);
    // The last bit shifted out is bit (width - n). Shifting by exactly the
    // width ejects bit 0 (hardware says CF=1 for `shl al,8` with al&1), and
    // only counts *beyond* the width leave CF clear.
    d.cf = Some(if n > w.bits() {
        false
    } else {
        bit(a, w.bits() - n)
    });
    d.pf = Some(pf(res));
    d.af = None;
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = if n == 1 {
        // OF after shl 1 = (new sign bit) XOR carry-out.
        Some(sf(w, res) != d.cf.unwrap())
    } else {
        None
    };
    (res, d)
}

/// `shr a, n`
pub fn shr(w: Width, a: u64, count: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let n = mask_count(w, count);
    let mut d = FlagDef::default();
    if n == 0 {
        d.unchanged = &FlagDef::ALL;
        return (a, d);
    }
    let res = as_u(w, a >> n);
    // The last bit shifted out is bit n-1 of the operand; for n beyond the
    // width there is nothing left, so CF is 0.
    d.cf = Some(if n - 1 >= w.bits() as u32 {
        false
    } else {
        (a >> (n - 1)) & 1 == 1
    });
    d.pf = Some(pf(res));
    d.af = None;
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = if n == 1 { Some(sf(w, a)) } else { None };
    (res, d)
}

/// `sar a, n`
pub fn sar(w: Width, a: u64, count: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let n = mask_count(w, count);
    let mut d = FlagDef::default();
    if n == 0 {
        d.unchanged = &FlagDef::ALL;
        return (a, d);
    }
    let res = as_u(w, (as_i(w, a) >> n as u32) as u64);
    // SAR shifts in the sign bit, so for n-1 at or beyond the width the last
    // bit out is the sign bit. Evaluating on the sign-extended value covers
    // every n uniformly.
    d.cf = Some((as_i(w, a) >> (n - 1) as u32) & 1 == 1);
    d.pf = Some(pf(res));
    d.af = None;
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    // SAR by 1 clears OF; SAR by >1 leaves it undefined.
    d.of = if n == 1 { Some(false) } else { None };
    (res, d)
}

/// `inc a`: CF is preserved, which the harness tests by seeding CF.
/// OF marks the signed overflow, i.e. the operand was INT_MAX.
pub fn inc(w: Width, a: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let res = as_u(w, a.wrapping_add(1));
    let mut d = FlagDef::default();
    d.cf = None;
    d.unchanged = &[rdec_core::ir::Flag::CF];
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, 1, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = Some(a == as_u(w, (1u64 << msb(w)) - 1));
    (res, d)
}

/// `dec a`: CF is preserved. OF marks INT_MIN - 1.
pub fn dec(w: Width, a: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let res = as_u(w, a.wrapping_sub(1));
    let mut d = FlagDef::default();
    d.cf = None;
    d.unchanged = &[rdec_core::ir::Flag::CF];
    d.pf = Some(pf(res));
    d.af = Some(af_add(a, 1, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = Some(a == as_u(w, 1u64 << msb(w)));
    (res, d)
}

/// `neg a`: CF is set unless the operand is zero.
pub fn neg(w: Width, a: u64) -> (u64, FlagDef) {
    let a = as_u(w, a);
    let res = as_u(w, 0u64.wrapping_sub(a));
    let mut d = FlagDef::default();
    d.cf = Some(a != 0);
    d.pf = Some(pf(res));
    d.af = Some(af_add(0, a, res));
    d.zf = Some(zf(w, res));
    d.sf = Some(sf(w, res));
    d.of = Some(a == as_u(w, 1u64 << msb(w)));
    (res, d)
}

/// `mul b` (unsigned, `a` implicit): CF=OF=1 iff the high half is non-zero.
pub fn mul(w: Width, a: u64, b: u64) -> (u64, u64, FlagDef) {
    let (a, b) = (as_u(w, a), as_u(w, b));
    let full = (a as u128) * (b as u128);
    let lo = as_u(w, full as u64);
    let hi = as_u(w, (full >> w.bits()) as u64);
    let mut d = FlagDef::default();
    d.cf = Some(hi != 0);
    d.of = Some(hi != 0);
    d.pf = None;
    d.af = None;
    d.zf = None;
    d.sf = None;
    (lo, hi, d)
}

/// `imul b`: CF=OF=1 iff the high half is not the sign-extension of the low.
pub fn imul(w: Width, a: u64, b: u64) -> (u64, u64, FlagDef) {
    let (a, b) = (as_i(w, a) as i128, as_i(w, b) as i128);
    let full = a * b;
    let lo = as_u(w, full as i64 as u64);
    let hi = as_u(w, (full >> w.bits()) as i64 as u64);
    let sext = as_u(w, ((full as i64) >> (w.bits() - 1)) as u64);
    let mut d = FlagDef::default();
    d.cf = Some(hi != sext);
    d.of = Some(hi != sext);
    d.pf = None;
    d.af = None;
    d.zf = None;
    d.sf = None;
    (lo, hi, d)
}
