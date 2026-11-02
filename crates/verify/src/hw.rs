//! Execute one x86-64 flag-defining instruction and read EFLAGS back.
//!
//! This is the ground truth for the T1 oracle. Every sequence is
//! `pushfq; push seed; popfq; <op>; pushfq; pop out; popfq` — seed the flags so
//! "instruction leaves CF alone" (`inc`/`dec`/shift-by-0) is observable, and
//! restore the caller's flags at the end.

use core::arch::asm;
use rdec_core::ty::Width;

/// EFLAGS bit positions.
pub const CF: u64 = 1 << 0;
pub const PF: u64 = 1 << 2;
pub const AF: u64 = 1 << 4;
pub const ZF: u64 = 1 << 6;
pub const SF: u64 = 1 << 7;
pub const OF: u64 = 1 << 11;

pub const ALL_SEED: u64 = CF | PF | AF | ZF | SF | OF;

pub fn bit(flags: u64, mask: u64) -> bool {
    flags & mask != 0
}

// Two-operand `op a, b` forms: add, sub, and, or, xor, cmp.
macro_rules! binop_fn {
    ($name:ident, $mnem:literal) => {
        /// Returns `(result_of_a, eflags_after)`.
        pub fn $name(w: Width, a: u64, b: u64, seed: u64) -> (u64, u64) {
            let mut res: u64;
            let mut flags: u64;
            unsafe {
                match w {
                    Width::W8 => {
                        let mut x: u8 = a as u8;
                        let y: u8 = b as u8;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " al, dl", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("al") x,
                            in("dl") y,
                            lateout("al") x,
                        );
                        res = x as u64;
                    }
                    Width::W16 => {
                        let mut x: u16 = a as u16;
                        let y: u16 = b as u16;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " ax, dx", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("ax") x,
                            in("dx") y,
                            lateout("ax") x,
                        );
                        res = x as u64;
                    }
                    Width::W32 => {
                        let mut x: u32 = a as u32;
                        let y: u32 = b as u32;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " eax, edx", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("eax") x,
                            in("edx") y,
                            lateout("eax") x,
                        );
                        res = x as u64;
                    }
                    Width::W64 => {
                        let mut x: u64 = a;
                        let y: u64 = b;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " rax, rdx", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("rax") x,
                            in("rdx") y,
                            lateout("rax") x,
                        );
                        res = x;
                    }
                    Width::W1 => unimplemented!("no 1-bit x86 ops"),
                }
            }
            (res, flags)
        }
    };
}

binop_fn!(add, "add");
binop_fn!(sub, "sub");
binop_fn!(and, "and");
binop_fn!(or, "or");
binop_fn!(xor, "xor");

// Shifts: count must be in cl.
macro_rules! shift_fn {
    ($name:ident, $mnem:literal) => {
        pub fn $name(w: Width, a: u64, count: u64, seed: u64) -> (u64, u64) {
            let mut res: u64;
            let mut flags: u64;
            unsafe {
                match w {
                    Width::W8 => {
                        let mut x: u8 = a as u8;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " al, cl", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("al") x,
                            in("cl") count as u8,
                            lateout("al") x,
                        );
                        res = x as u64;
                    }
                    Width::W16 => {
                        let mut x: u16 = a as u16;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " ax, cl", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("ax") x,
                            in("cl") count as u8,
                            lateout("ax") x,
                        );
                        res = x as u64;
                    }
                    Width::W32 => {
                        let mut x: u32 = a as u32;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " eax, cl", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("eax") x,
                            in("cl") count as u8,
                            lateout("eax") x,
                        );
                        res = x as u64;
                    }
                    Width::W64 => {
                        let mut x: u64 = a;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " rax, cl", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("rax") x,
                            in("cl") count as u8,
                            lateout("rax") x,
                        );
                        res = x;
                    }
                    Width::W1 => unimplemented!(),
                }
            }
            (res, flags)
        }
    };
}

shift_fn!(shl, "shl");
shift_fn!(shr, "shr");
shift_fn!(sar, "sar");

// Unary: inc, dec, neg.
macro_rules! unary_fn {
    ($name:ident, $mnem:literal) => {
        pub fn $name(w: Width, a: u64, seed: u64) -> (u64, u64) {
            let mut res: u64;
            let mut flags: u64;
            unsafe {
                match w {
                    Width::W8 => {
                        let mut x: u8 = a as u8;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " al", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("al") x,
                            lateout("al") x,
                        );
                        res = x as u64;
                    }
                    Width::W16 => {
                        let mut x: u16 = a as u16;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " ax", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("ax") x,
                            lateout("ax") x,
                        );
                        res = x as u64;
                    }
                    Width::W32 => {
                        let mut x: u32 = a as u32;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " eax", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("eax") x,
                            lateout("eax") x,
                        );
                        res = x as u64;
                    }
                    Width::W64 => {
                        let mut x: u64 = a;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " rax", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("rax") x,
                            lateout("rax") x,
                        );
                        res = x;
                    }
                    Width::W1 => unimplemented!(),
                }
            }
            (res, flags)
        }
    };
}

unary_fn!(inc, "inc");
unary_fn!(dec, "dec");
unary_fn!(neg, "neg");

// Accumulator multiply: `mul`/`imul rbx` with rdx:rax as the wide result.
macro_rules! mul_fn {
    ($name:ident, $mnem:literal) => {
        /// Returns `(lo, hi, eflags_after)`, with `a` in the accumulator.
        pub fn $name(w: Width, a: u64, b: u64, seed: u64) -> (u64, u64, u64) {
            let lo: u64;
            let hi: u64;
            let mut flags: u64;
            unsafe {
                match w {
                    Width::W8 => {
                        let mut al_v: u8 = a as u8;
                        // `ah` is not addressable as an asm operand on x86-64, so
                        // read the whole 16-bit accumulator: al = lo, ah = hi.
                        let mut ax_out: u16 = 0;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " sil", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("al") al_v,
                            in("sil") b as u8,
                            lateout("ax") ax_out,
                        );
                        lo = (ax_out & 0xff) as u64;
                        hi = (ax_out >> 8) as u64;
                    }
                    Width::W16 => {
                        let mut x: u16 = a as u16;
                        let mut y: u16 = 0;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " si", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("ax") x,
                            in("si") b as u16,
                            lateout("ax") x,
                            lateout("dx") y,
                        );
                        lo = x as u64;
                        hi = y as u64;
                    }
                    Width::W32 => {
                        let mut x: u32 = a as u32;
                        let mut y: u32 = 0;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " esi", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("eax") x,
                            in("esi") b as u32,
                            lateout("eax") x,
                            lateout("edx") y,
                        );
                        lo = x as u64;
                        hi = y as u64;
                    }
                    Width::W64 => {
                        let mut x: u64 = a;
                        let mut y: u64 = 0;
                        asm!(
                            concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                                    $mnem, " rsi", "\n\t", "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                            f = lateout(reg) flags,
                            s = in(reg) seed,
                            in("rax") x,
                            in("rsi") b,
                            lateout("rax") x,
                            lateout("rdx") y,
                        );
                        lo = x;
                        hi = y;
                    }
                    Width::W1 => unimplemented!(),
                }
            }
            (lo, hi, flags)
        }
    };
}

mul_fn!(mul, "mul");
mul_fn!(imul, "imul");

/// `adc`/`sbb` need an incoming carry, seeded via CF before the op.
pub fn adc(w: Width, a: u64, b: u64, cin: bool, seed: u64) -> (u64, u64) {
    let seed = if cin { seed | CF } else { seed & !CF };
    binop_seeded("adc", w, a, b, seed)
}

pub fn sbb(w: Width, a: u64, b: u64, bin: bool, seed: u64) -> (u64, u64) {
    let seed = if bin { seed | CF } else { seed & !CF };
    binop_seeded("sbb", w, a, b, seed)
}

/// Runtime-selected binary op, for the adc/sbb forms (the mnemonic is a
/// compile-time constant via two monomorphized closures).
fn binop_seeded(mnem: &str, w: Width, a: u64, b: u64, seed: u64) -> (u64, u64) {
    match (mnem, w) {
        ("adc", Width::W8) => adc8(a, b, seed),
        ("adc", Width::W16) => adc16(a, b, seed),
        ("adc", Width::W32) => adc32(a, b, seed),
        ("adc", Width::W64) => adc64(a, b, seed),
        ("sbb", Width::W8) => sbb8(a, b, seed),
        ("sbb", Width::W16) => sbb16(a, b, seed),
        ("sbb", Width::W32) => sbb32(a, b, seed),
        ("sbb", Width::W64) => sbb64(a, b, seed),
        _ => unreachable!("binop_seeded: {mnem} {w:?}"),
    }
}

macro_rules! adc_sbb2 {
    ($name:ident, $mnem:literal, $ty:ty, $acc:tt, $src:tt) => {
        fn $name(a: u64, b: u64, seed: u64) -> (u64, u64) {
            let x: $ty = a as $ty;
            let y: $ty = b as $ty;
            let mut out: $ty = 0;
            let mut flags: u64;
            unsafe {
                asm!(
                    concat!("pushfq", "\n\t", "push {s}", "\n\t", "popfq", "\n\t",
                            $mnem, " ", $acc, ", ", $src, "\n\t",
                            "pushfq", "\n\t", "pop {f}", "\n\t", "popfq"),
                    f = lateout(reg) flags,
                    s = in(reg) seed,
                    in($acc) x,
                    in($src) y,
                    lateout($acc) out,
                );
            }
            (out as u64, flags)
        }
    };
}

adc_sbb2!(adc8, "adc", u8, "al", "dl");
adc_sbb2!(adc16, "adc", u16, "ax", "dx");
adc_sbb2!(adc32, "adc", u32, "eax", "edx");
adc_sbb2!(adc64, "adc", u64, "rax", "rdx");
adc_sbb2!(sbb8, "sbb", u8, "al", "dl");
adc_sbb2!(sbb16, "sbb", u16, "ax", "dx");
adc_sbb2!(sbb32, "sbb", u32, "eax", "edx");
adc_sbb2!(sbb64, "sbb", u64, "rax", "rdx");
