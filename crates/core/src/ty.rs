//! Types in the RDec IR.
//!
//! The IR is typed, and the type of every [`Op`] result is recoverable from the
//! op itself (see [`Op::ty`]) except for a few cases (`Select`, `Phi`,
//! `GetFlag`) where the type is recorded per-instruction. Keeping types in the
//! op means a pass cannot accidentally construct a value whose printed C would
//! be a lie.

use std::fmt;

/// A machine width in bits. Only widths x86-64 can actually produce.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Width {
    W1,
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    pub const fn bits(self) -> u32 {
        match self {
            Width::W1 => 1,
            Width::W8 => 8,
            Width::W16 => 16,
            Width::W32 => 32,
            Width::W64 => 64,
        }
    }

    pub const fn bytes(self) -> u32 {
        self.bits() / 8
    }

    /// Mask with 2^bits - 1, as u64 (identity for W64).
    pub const fn mask(self) -> u64 {
        match self {
            Width::W1 => 1,
            Width::W8 => 0xff,
            Width::W16 => 0xffff,
            Width::W32 => 0xffff_ffff,
            Width::W64 => u64::MAX,
        }
    }

    /// Truncate + sign/zero extend `x` to this width.
    pub const fn trunc(self, x: u64, signed: bool) -> u64 {
        let m = self.mask();
        let v = x & m;
        if signed && self.bits() < 64 && (v >> (self.bits() - 1)) & 1 == 1 {
            // sign extend
            v | !m
        } else {
            v
        }
    }

    pub const fn from_bits(bits: u32) -> Option<Width> {
        match bits {
            1 => Some(Width::W1),
            8 => Some(Width::W8),
            16 => Some(Width::W16),
            32 => Some(Width::W32),
            64 => Some(Width::W64),
            _ => None,
        }
    }
}

/// Signedness is part of the *type*, not part of the operator, so that
/// `a < b` in C prints with the right cast pair and so signedness survives a
/// copy-propagation chain.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Sx {
    Signed,
    Unsigned,
}

impl Sx {
    pub const fn signed(self) -> bool {
        matches!(self, Sx::Signed)
    }
    pub const fn other(self) -> Sx {
        match self {
            Sx::Signed => Sx::Unsigned,
            Sx::Unsigned => Sx::Signed,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FWidth {
    F32,
    F64,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    /// No value. Only valid as the type of a store/call-without-dest.
    Void,
    /// `true`/`false`, produced by comparisons and `get_flag`, consumed by
    /// branches and `select`.
    Bool,
    Int {
        w: Width,
        sx: Sx,
    },
    Float {
        w: FWidth,
    },
    /// Pointer to pointee. `Ptr(Void)` is `void *`.
    Ptr(Box<Ty>),
    /// The opaque x86 EFLAGS value produced by flag-defining ops. Never
    /// printed; must not survive the flag-elimination pass.
    Flags,
    /// Function type: (ret, params).
    Func(Box<Ty>, Box<[Ty]>),
    /// A struct we recovered from a vtable/layout analysis. Opaque by width
    /// until we do real struct recovery.
    Struct {
        name: String,
        size: u32,
    },
}

impl Ty {
    pub const VOID: Ty = Ty::Void;
    pub const BOOL: Ty = Ty::Bool;
    pub const FLAGS: Ty = Ty::Flags;

    pub const fn int(w: Width, sx: Sx) -> Ty {
        Ty::Int { w, sx }
    }

    pub const fn uint(w: Width) -> Ty {
        Ty::Int {
            w,
            sx: Sx::Unsigned,
        }
    }

    pub const fn sint(w: Width) -> Ty {
        Ty::Int { w, sx: Sx::Signed }
    }

    pub fn u8() -> Ty {
        Ty::uint(Width::W8)
    }
    pub fn u16() -> Ty {
        Ty::uint(Width::W16)
    }
    pub fn u32() -> Ty {
        Ty::uint(Width::W32)
    }
    pub fn u64() -> Ty {
        Ty::uint(Width::W64)
    }
    pub fn i32() -> Ty {
        Ty::sint(Width::W32)
    }
    pub fn i64() -> Ty {
        Ty::sint(Width::W64)
    }
    pub fn f32() -> Ty {
        Ty::Float { w: FWidth::F32 }
    }
    pub fn f64() -> Ty {
        Ty::Float { w: FWidth::F64 }
    }
    pub fn void_ptr() -> Ty {
        Ty::Ptr(Box::new(Ty::Void))
    }
    pub fn ptr_to(t: Ty) -> Ty {
        Ty::Ptr(Box::new(t))
    }

    pub const fn is_int(&self) -> bool {
        matches!(self, Ty::Int { .. })
    }
    pub const fn is_float(&self) -> bool {
        matches!(self, Ty::Float { .. })
    }
    pub const fn is_ptr(&self) -> bool {
        matches!(self, Ty::Ptr(_))
    }
    pub const fn is_bool(&self) -> bool {
        matches!(self, Ty::Bool)
    }
    pub const fn is_void(&self) -> bool {
        matches!(self, Ty::Void)
    }

    pub const fn sx(&self) -> Option<Sx> {
        match self {
            Ty::Int { sx, .. } => Some(*sx),
            Ty::Ptr(_) => Some(Sx::Unsigned),
            _ => None,
        }
    }

    /// Width in bits, for integer/float/pointer types.
    pub const fn width(&self) -> Option<Width> {
        match self {
            Ty::Int { w, .. } => Some(*w),
            Ty::Bool => Some(Width::W1),
            Ty::Ptr(_) => Some(Width::W64),
            Ty::Float { w } => Some(match w {
                FWidth::F32 => Width::W32,
                FWidth::F64 => Width::W64,
            }),
            Ty::Struct { size, .. } => Width::from_bits(*size * 8),
            _ => None,
        }
    }

    /// Same integer width and same signedness? Used to decide whether a cast
    /// is needed when printing binary ops.
    pub fn int_equivalent(&self, other: &Ty) -> bool {
        self == other
    }

    /// Sign-extend/truncate a raw u64 to this type's own interpretation.
    pub const fn normalize(&self, x: u64) -> u64 {
        match self {
            Ty::Int { w, sx } => w.trunc(x, sx.signed()),
            Ty::Bool => x & 1,
            Ty::Ptr(_) => x,
            _ => x,
        }
    }

    /// C integer/float keyword for scalar types (not for ptr/func/struct).
    pub fn c_scalar(&self) -> Option<&'static str> {
        match self {
            Ty::Void => Some("void"),
            Ty::Bool => Some("bool"),
            Ty::Int {
                w: Width::W8,
                sx: Sx::Signed,
            } => Some("int8_t"),
            Ty::Int {
                w: Width::W8,
                sx: Sx::Unsigned,
            } => Some("uint8_t"),
            Ty::Int {
                w: Width::W16,
                sx: Sx::Signed,
            } => Some("int16_t"),
            Ty::Int {
                w: Width::W16,
                sx: Sx::Unsigned,
            } => Some("uint16_t"),
            Ty::Int {
                w: Width::W32,
                sx: Sx::Signed,
            } => Some("int32_t"),
            Ty::Int {
                w: Width::W32,
                sx: Sx::Unsigned,
            } => Some("uint32_t"),
            Ty::Int {
                w: Width::W64,
                sx: Sx::Signed,
            } => Some("int64_t"),
            Ty::Int {
                w: Width::W64,
                sx: Sx::Unsigned,
            } => Some("uint64_t"),
            Ty::Float { w: FWidth::F32 } => Some("float"),
            Ty::Float { w: FWidth::F64 } => Some("double"),
            _ => None,
        }
    }
}

impl fmt::Display for Ty {
    /// Prints `T` such that `format!("{name}{rest}")` is a C declarator for a
    /// variable named `name` — i.e. pointer stars trail the name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Func(ret, args) => {
                // `ret (*)(args)` — used only in prototypes.
                write!(f, "{} (*)(", ret)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Ty::Ptr(inner) => {
                let s = inner.to_string();
                // `int*` -> `int **p` reads as `int **` + name.
                write!(f, "{s} *")
            }
            Ty::Struct { name, .. } => write!(f, "struct {name}"),
            other => f.write_str(other.c_scalar().unwrap_or("/*?ty*/int")),
        }
    }
}
