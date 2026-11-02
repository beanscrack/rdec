//! x86-64 front end.
//!
//! Layout (ROADMAP.md): `flags` is the EFLAGS semantics model, checked against
//! real hardware by `rdec-verify`. `decode` (iced glue) and `lift` (per-op
//! translation to RDec IR) land in M1/M2.

pub mod decode;
pub mod flags;
pub mod loader;
