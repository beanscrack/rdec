//! Jump tables over EMBEDDED object bytes: the shape is the claim, not the
//! compiler's version.
//!
//! WHY THIS FILE EXISTS. The tests here used to read
//! `corpus/obj/switch_dense.O0.obj` and `switch_fallthrough.O0.obj`, objects
//! `ci.yml` GENERATES with the runner's clang. That runner has clang 20.1.8,
//! this machine has 22.1.8, and the two lower a C `switch` differently, so four
//! tests were green locally and red in CI (run 42164392990) while asserting
//! nothing about the decompiler: they were asserting which clang built the
//! bytes. The corpus objects stay in `jump_tables.rs`, where every assertion on
//! them is derived from the object's OWN relocations, so any clang that emits a
//! jump table at all satisfies it. The claims below are about a *shape*, so the
//! shape's bytes live here, as constants.
//!
//! WHERE THE BYTES COME FROM. Each fixture is hand-written assembly, assembled
//! by `clang -c -target x86_64-pc-windows-msvc` — clang's integrated assembler
//! IS llvm-mc, so the encodings and the relocation records are the toolchain's,
//! not hand-assembled, and not the optimiser's: no `-O` level, and therefore no
//! version, is in the loop. The objects are then read back with `llvm-objdump`
//! and the output is quoted verbatim below, so every expected number in this
//! file is auditable against the tool, not against the code under test.
//!
//! GROUND TRUTH — `jt_a.obj`, the REGISTER-INDIRECT dispatch (the form clang
//! 22.1.8 emits at `-O0`), `llvm-objdump -d -r`:
//!
//! ```text
//! Disassembly of section .text:
//!
//! 0000000000000000 <f>:
//!        0: 48 89 4c 24 08             movq    %rcx, 0x8(%rsp)
//!        5: 48 8b 4c 24 08             movq    0x8(%rsp), %rcx
//!        a: 48 8d 05 00 00 00 00       leaq    (%rip), %rax    # 0x11 <f+0x11>
//!              000000000000000d:  IMAGE_REL_AMD64_REL32  .rdata
//!       11: 48 63 0c 88                movslq  (%rax,%rcx,4), %rcx
//!       15: 48 01 c8                   addq    %rcx, %rax
//!       18: ff e0                      jmpq    *%rax
//!       1a: b8 07 00 00 00             movl    $0x7, %eax
//!       1f: c3                         retq
//!       20: b8 0b 00 00 00             movl    $0xb, %eax
//!       25: c3                         retq
//!       26: b8 0d 00 00 00             movl    $0xd, %eax
//!       2b: c3                         retq
//!       2c: b8 11 00 00 00             movl    $0x11, %eax
//!       31: c3                         retq
//!       32: b8 13 00 00 00             movl    $0x13, %eax
//!       37: c3                         retq
//!       38: b8 17 00 00 00             movl    $0x17, %eax
//!       3d: c3                         retq
//!       3e: b8 1d 00 00 00             movl    $0x1d, %eax
//!       43: c3                         retq
//!       44: b8 1f 00 00 00             movl    $0x1f, %eax
//!       49: c3                         retq
//!
//! RELOCATION RECORDS FOR [.text]:
//! OFFSET           TYPE                     VALUE
//! 000000000000000d IMAGE_REL_AMD64_REL32    .rdata
//!
//! RELOCATION RECORDS FOR [.rdata]:
//! OFFSET           TYPE                     VALUE
//! 0000000000000000 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000004 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000008 IMAGE_REL_AMD64_REL32    .text
//! 000000000000000c IMAGE_REL_AMD64_REL32    .text
//! 0000000000000010 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000014 IMAGE_REL_AMD64_REL32    .text
//! 0000000000000018 IMAGE_REL_AMD64_REL32    .text
//! 000000000000001c IMAGE_REL_AMD64_REL32    .text
//! ```
//!
//! `llvm-objdump -s -j .rdata`:
//!
//! ```text
//! Contents of section .rdata:
//!  0000 1e000000 28000000 32000000 3c000000  ....(...2...<...
//!  0010 46000000 50000000 5a000000 64000000  F...P...Z...d...
//!  0020 28000000 30000000 38000000 40000000  (...0...8...@...
//! ```
//!
//! The last twelve bytes are deliberate: four 4-byte slots that carry NO
//! relocation, whose values happen to decode as in-code targets (`0x28` in slot
//! 8 is `.text+0x4`). A recogniser that read table slots by VALUE would claim
//! them; the rule here is the relocation, so the table is 8 entries and the run
//! ends at `.rdata+0x20`. `a_table_run_ends_where_its_relocations_end` pins it.
//!
//! COFF `REL32` arithmetic, the same as `discover::table_target`: the slot's own
//! bytes are the addend and the linker writes `symbol + addend - (site + 4)`, so
//! a dispatch that computes `table_base + entry` lands the slot at
//! `sym_offset + addend - (slot - table_base) - 4`. Entry 0 by hand:
//! `0 + 0x1e - (0 - 0) - 4 = 0x1a`, the first `movl $0x7, %eax` above; entry 7:
//! `0x64 - 28 - 4 = 0x44`, the last.
//!
//! GROUND TRUTH — `jt_b.obj`, the MEMORY-INDIRECT dispatch (the form that does
//! NOT use a base register), `llvm-objdump -d -r`, quoted verbatim:
//!
//! ```text
//! Disassembly of section .text:
//!
//! 0000000000000000 <f>:
//!        0: 48 89 4c 24 08             movq    %rcx, 0x8(%rsp)
//!        5: 48 8b 4c 24 08             movq    0x8(%rsp), %rcx
//!        a: ff 24 8d 00 00 00 00       jmpq    *(,%rcx,4)
//!              000000000000000d:  IMAGE_REL_AMD64_ADDR32  .rdata
//!       11: b8 07 00 00 00             movl    $0x7, %eax
//!       16: c3                         retq
//!       17: b8 0b 00 00 00             movl    $0xb, %eax
//!       1c: c3                         retq
//!       1d: b8 0d 00 00 00             movl    $0xd, %eax
//!       22: c3                         retq
//!       23: b8 11 00 00 00             movl    $0x11, %eax
//!       28: c3                         retq
//!
//! Contents of section .rdata:
//!  0000 15000000 1f000000 29000000 33000000  ........)...3...
//! ```
//!
//! The one fact that matters, and it is the assembler's, not ours: the naming
//! relocation is `IMAGE_REL_AMD64_ADDR32`, not `REL32`. With a SIB base of
//! `disp32` the displacement is NOT RIP-relative in 64-bit mode, so the field
//! needs an absolute 32-bit relocation and the assembler gives it one. The
//! `.rdata` entries are the SAME `REL32 -> .text` class as fixture A (addends
//! `0x15 = 0x11 + 0 + 4`, `0x1f = 0x17 + 4 + 4`, ...). What the table is named
//! by, and what the jump reads, are two separate facts — and this repo's
//! recognition today needs the naming relocation to be in the REL32 family, so
//! this object yields NO table, the four bodies stay unclaimed, and the `jmp`
//! stays a counted gap. `the_memory_indirect_dispatch_the_assembler_emits_is_a_counted_gap`
//! pins exactly that, red direction included: a recogniser that "helpfully"
//! parsed this as a switch would be inventing eight destinations from bytes it
//! cannot name.
//!
//! GROUND TRUTH — `jt_c.obj`, the fallthrough switch (same dispatch shape as A,
//! bodies that fall into each other, and a bounds check that routes
//! out-of-range inputs past the table), `llvm-objdump -d -r`:
//!
//! ```text
//! Disassembly of section .text:
//!
//! 0000000000000000 <f>:
//!        0: 31 d2                      xorl    %edx, %edx
//!        2: 48 89 4c 24 08             movq    %rcx, 0x8(%rsp)
//!        7: 48 8b 44 24 08             movq    0x8(%rsp), %rax
//!        c: 48 83 e8 03                subq    $0x3, %rax
//!       10: 77 25                      ja      0x37 <f+0x37>
//!       12: 48 8b 4c 24 08             movq    0x8(%rsp), %rcx
//!       17: 48 8d 05 00 00 00 00       leaq    (%rip), %rax    # 0x1e <f+0x1e>
//!              000000000000001a:  IMAGE_REL_AMD64_REL32  .rdata
//!       1e: 48 63 0c 88                movslq  (%rax,%rcx,4), %rcx
//!       22: 48 01 c8                   addq    %rcx, %rax
//!       25: ff e0                      jmpq    *%rax
//!       27: 83 c2 01                   addl    $0x1, %edx
//!       2a: 83 c2 02                   addl    $0x2, %edx
//!       2d: 83 c2 04                   addl    $0x4, %edx
//!       30: eb 07                      jmp     0x39 <f+0x39>
//!       32: 83 c2 08                   addl    $0x8, %edx
//!       35: eb 02                      jmp     0x39 <f+0x39>
//!       37: 31 d2                      xorl    %edx, %edx
//!       39: 89 d0                      movl    %edx, %eax
//!       3b: c3                         retq
//!
//! Contents of section .rdata:
//!  0000 2b000000 32000000 39000000 42000000  +...2...9...B...
//! ```
//!
//! Slot 0 -> `0x2b - 0 - 4 = 0x27` (the `+1` body), slot 1 -> `0x32 - 4 - 4 =
//! 0x2a` (the `+2` body, so slot 1 falls through the `+4`), slot 2 -> `0x2d - 8
//! - 4 = 0x2d`, slot 3 -> `0x42 - 12 - 4 = 0x32`. Answers, from the assembly
//! itself and NOT from the model: `f(0)=1+2+4=7`, `f(1)=2+4=6`, `f(2)=4`,
//! `f(3)=8`, and `f(4)`/`f(9)` take the `ja` to 0x37 and answer 0.
//!
//! `f.params` is assigned by the real `FnAbi::integer()` seed in
//! `Lifter::new_sig` (RCX/RDX/R8/R9 -> parameters 0..=3), and every test below
//! asserts it, because a fixture that reads an unbound `Op::Param` agrees with
//! any bug (`docs`/skill: the CF test that passed on broken code).

mod common;

use common::{code_ranges, discover_fixture, uncovered};
use rdec_arch_x86::discover::{Gap, Plan};
use rdec_arch_x86::lift::{lift_one_sigs, no_callees, no_relocs, Report};
use rdec_core::cfg::Cfg;
use rdec_core::interp::{run, Fault, Outcome, Val};
use rdec_core::ir::{Function, Terminator, NO_VAL};
use rdec_core::verify::verify_ssa;

// The bytes of `jt_a.obj` — the register-indirect dispatch quoted above, 558
// bytes, assembled by clang's integrated assembler from hand-written assembly.
const JT_A_OBJ: [u8; 558] = [
    0x64, 0x86, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00, 0xfe, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x50, 0x60, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0xc0, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x30, 0xc0, 0x2e, 0x72, 0x64, 0x61,
    0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
    0x08, 0x01, 0x00, 0x00, 0x38, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0x40, 0x48, 0x89, 0x4c, 0x24, 0x08, 0x48, 0x8b, 0x4c, 0x24, 0x08, 0x48, 0x8d,
    0x05, 0x00, 0x00, 0x00, 0x00, 0x48, 0x63, 0x0c, 0x88, 0x48, 0x01, 0xc8, 0xff, 0xe0, 0xb8, 0x07,
    0x00, 0x00, 0x00, 0xc3, 0xb8, 0x0b, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x0d, 0x00, 0x00, 0x00, 0xc3,
    0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x13, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x17, 0x00, 0x00,
    0x00, 0xc3, 0xb8, 0x1d, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x1f, 0x00, 0x00, 0x00, 0xc3, 0x0d, 0x00,
    0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
    0x32, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
    0x5a, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
    0x38, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x1c, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x01, 0x4a, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x65, 0x85, 0xee, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x72, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x03, 0x01, 0x30, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0xd8, 0xee,
    0x7f, 0x2e, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
];

// The bytes of `jt_b.obj` — the memory-indirect dispatch (ADDR32 naming
// relocation), 453 bytes, same source style.
const JT_B_OBJ: [u8; 453] = [
    0x64, 0x86, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00, 0xdd, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x50, 0x60, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xe7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0xc0, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x30, 0xc0, 0x2e, 0x72, 0x64, 0x61,
    0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0xe7, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0x40, 0x48, 0x89, 0x4c, 0x24, 0x08, 0x48, 0x8b, 0x4c, 0x24, 0x08, 0xff, 0x24,
    0x8d, 0x00, 0x00, 0x00, 0x00, 0xb8, 0x07, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x0b, 0x00, 0x00, 0x00,
    0xc3, 0xb8, 0x0d, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3, 0x0d, 0x00, 0x00,
    0x00, 0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x15, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x29,
    0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x2e,
    0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03,
    0x01, 0x29, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xe5, 0xee, 0xe5, 0xd6, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x2e, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x72, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x01, 0x10, 0x00, 0x00,
    0x00, 0x04, 0x00, 0x00, 0x00, 0x37, 0xbf, 0x55, 0xe6, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x00, 0x00,
];

// The bytes of `jt_c.obj` — the fallthrough switch with a bounds check, 472
// bytes.
const JT_C_OBJ: [u8; 472] = [
    0x64, 0x86, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x32, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x50, 0x60, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xfa, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0xc0, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x30, 0xc0, 0x2e, 0x72, 0x64, 0x61,
    0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0xfa, 0x00, 0x00, 0x00, 0x0a, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0x40, 0x31, 0xd2, 0x48, 0x89, 0x4c, 0x24, 0x08, 0x48, 0x8b, 0x44, 0x24, 0x08,
    0x48, 0x83, 0xe8, 0x03, 0x77, 0x25, 0x48, 0x8b, 0x4c, 0x24, 0x08, 0x48, 0x8d, 0x05, 0x00, 0x00,
    0x00, 0x00, 0x48, 0x63, 0x0c, 0x88, 0x48, 0x01, 0xc8, 0xff, 0xe0, 0x83, 0xc2, 0x01, 0x83, 0xc2,
    0x02, 0x83, 0xc2, 0x04, 0xeb, 0x07, 0x83, 0xc2, 0x08, 0xeb, 0x02, 0x31, 0xd2, 0x89, 0xd0, 0xc3,
    0x1a, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x32, 0x00,
    0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x03, 0x01, 0x3c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xba, 0x15, 0x2d, 0x70,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x72,
    0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x01,
    0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xbb, 0x20, 0x69, 0x1e, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
];

// The same dispatch as `JT_A_OBJ` with the table base and the index/temporary
// roles SWAPPED, which is what clang 20.1.8 lowers at `-O0`. Verbatim from CI run
// 42166188651, which dumped the CI runner's `corpus/obj/switch_dense.O0.obj` with
// `clang version 20.1.8` and `llvm-objdump -d -r`:
//
// ```text
//        1a: 48 8d 0d 00 00 00 00     leaq   (%rip), %rcx      # 0x21
//         000000000000001d:  IMAGE_REL_AMD64_REL32    .rdata
//        21: 48 63 04 81              movslq (%rcx,%rax,4), %rax
//        25: 48 01 c8                 addq   %rcx, %rax
//        28: ff e0                    jmpq   *%rax
// ```
//
// and this fixture's own `llvm-objdump -d -r`, whose four dispatch instructions are
// byte-identical to those (`48 8d 0d 00 00 00 00`, `48 63 04 81`, `48 01 c8`,
// `ff e0`), with its naming relocation on the `lea`'s displacement and 4
// `.rdata REL32 -> .text` slots with addends `0x1e 0x28 0x32 0x3c`:
//
// ```text
// 0000000000000000 <f>:
//        0: 48 89 4c 24 08            movq   %rcx, 0x8(%rsp)
//        5: 48 8b 44 24 08            movq   0x8(%rsp), %rax
//        a: 48 8d 0d 00 00 00 00      leaq   (%rip), %rcx      # 0x11
//         000000000000000d:  IMAGE_REL_AMD64_REL32    .rdata
//       11: 48 63 04 81               movslq (%rcx,%rax,4), %rax
//       15: 48 01 c8                  addq   %rcx, %rax
//       18: ff e0                     jmpq   *%rax
//       1a: b8 07 00 00 00            movl   $0x7, %eax
//       1f: c3                        retq
//       20: b8 0b 00 00 00            movl   $0xb, %eax
//       25: c3                        retq
//       26: b8 0d 00 00 00            movl   $0xd, %eax
//       2b: c3                        retq
//       2c: b8 11 00 00 00            movl   $0x11, %eax
//       31: c3                        retq
//
// Contents of section .rdata:
//  0000 1e000000 28000000 32000000 3c000000
// ```
//
// The object's bytes, embedded (the 462 bytes of `jt_d.obj`, 12 per line):
//
// ```text
// const JT_D_OBJ: [u8; 462] = [
//     0x64,0x86,0x04,0x00,0x00,0x00,0x00,0x00,0x28,0x01,0x00,0x00,
//     0x09,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x2e,0x74,0x65,0x78,
//     0x74,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x32,0x00,0x00,0x00,0xb4,0x00,0x00,0x00,0xe6,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x20,0x00,0x50,0x60,
//     0x2e,0x64,0x61,0x74,0x61,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xf0,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x40,0x00,0x30,0xc0,0x2e,0x62,0x73,0x73,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x80,0x00,0x30,0xc0,0x2e,0x72,0x64,0x61,
//     0x74,0x61,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x10,0x00,0x00,0x00,0xf0,0x00,0x00,0x00,0x00,0x01,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x40,0x00,0x30,0x40,
//     0x48,0x89,0x4c,0x24,0x08,0x48,0x8b,0x44,0x24,0x08,0x48,0x8d,
//     0x0d,0x00,0x00,0x00,0x00,0x48,0x63,0x04,0x81,0x48,0x01,0xc8,
//     0xff,0xe0,0xb8,0x07,0x00,0x00,0x00,0xc3,0xb8,0x0b,0x00,0x00,
//     0x00,0xc3,0xb8,0x0d,0x00,0x00,0x00,0xc3,0xb8,0x11,0x00,0x00,
//     0x00,0xc3,0x0d,0x00,0x00,0x00,0x06,0x00,0x00,0x00,0x04,0x00,
//     0x1e,0x00,0x00,0x00,0x28,0x00,0x00,0x00,0x32,0x00,0x00,0x00,
//     0x3c,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x04,0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,
//     0x08,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x0c,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x2e,0x74,0x65,0x78,
//     0x74,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
//     0x03,0x01,0x32,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x33,0x28,
//     0x58,0x99,0x01,0x00,0x00,0x00,0x00,0x00,0x2e,0x64,0x61,0x74,
//     0x61,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00,
//     0x03,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x02,0x00,0x00,0x00,0x00,0x00,0x2e,0x62,0x73,0x73,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x03,0x00,0x00,0x00,
//     0x03,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
//     0x00,0x00,0x03,0x00,0x00,0x00,0x00,0x00,0x2e,0x72,0x64,0x61,
//     0x74,0x61,0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x00,0x00,
//     0x03,0x01,0x10,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x2a,0x76,
//     0x8d,0xa5,0x04,0x00,0x00,0x00,0x00,0x00,0x66,0x00,0x00,0x00,
//     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
//     0x02,0x00,0x04,0x00,0x00,0x00,
// ];
// ```

/// The case bodies of fixture A, straight from the `llvm-objdump -d` quote:
/// one `movl $n, %eax; retq` per slot, `n` = 7, 11, 13, 17, 19, 23, 29, 31.
const A_BODIES: [u64; 8] = [0x1a, 0x20, 0x26, 0x2c, 0x32, 0x38, 0x3e, 0x44];
const A_ANSWERS: [u64; 8] = [7, 11, 13, 17, 19, 23, 29, 31];

/// Fixture D as an object: the bytes quoted in the doc comment above.
const JT_D_OBJ: [u8; 462] = [
    0x64, 0x86, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00, 0xe6, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x50, 0x60, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0xc0, 0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x30, 0xc0, 0x2e, 0x72, 0x64, 0x61,
    0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0xf0, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x30, 0x40, 0x48, 0x89, 0x4c, 0x24, 0x08, 0x48, 0x8b, 0x44, 0x24, 0x08, 0x48, 0x8d,
    0x0d, 0x00, 0x00, 0x00, 0x00, 0x48, 0x63, 0x04, 0x81, 0x48, 0x01, 0xc8, 0xff, 0xe0, 0xb8, 0x07,
    0x00, 0x00, 0x00, 0xc3, 0xb8, 0x0b, 0x00, 0x00, 0x00, 0xc3, 0xb8, 0x0d, 0x00, 0x00, 0x00, 0xc3,
    0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3, 0x0d, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x1e, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x0c, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x01, 0x32, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x33, 0x28, 0x58, 0x99, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x64, 0x61, 0x74,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x2e, 0x62, 0x73, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2e, 0x72, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x03, 0x01, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x2a, 0x76,
    0x8d, 0xa5, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
];

#[test]
fn the_role_swapped_fold_ci_s_clang_lowers_models_the_same_switch_as_the_local_one() {
    let (o, fx) = discover_bytes(&JT_D_OBJ);

    // Discovery is relocation-driven, so it never saw the swap: the same table,
    // the same four targets, every byte claimed, no gap. This is CI's coverage
    // claim measured on CI's own lowering.
    let tbl = fx.tables.get(&D_JMP).unwrap_or_else(|| {
        panic!(
            "the swapped-fold dispatch at {D_JMP:#x} must resolve a table, got {:?}",
            fx.tables
        )
    });
    assert_eq!(tbl.base_section, ".rdata");
    assert_eq!(tbl.entries, 4);
    assert_eq!(
        tbl.targets, D_BODIES,
        "targets from the object's relocations"
    );
    assert!(tbl.unresolved.is_empty());
    for body in D_BODIES {
        assert!(
            fx.plan.blocks.contains(&body),
            "body {body:#x} undiscovered: {:?}",
            fx.plan
        );
    }
    assert_eq!(
        fx.plan.gap_count(Gap::IndirectBranch),
        0,
        "the table is resolved, so the dispatch is not a gap: {:?}",
        fx.plan.gaps
    );
    assert!(
        fx.uncovered.is_empty(),
        "coverage: unclaimed .text: {:?}",
        fx.uncovered
    );

    // Lifting is register-driven, and this object is clang 20.1.8's byte-identical
    // lowering of the SAME switch fixture A encodes in clang 22.1.8's shape: the
    // loaded delta lands in the ADD's destination here, in its source operand
    // there. The fold check used to require the source form, which is why the two
    // corpus switches modelled locally and were REFUSED on CI (1412 compared here,
    // 1364 there, both gates green). A register role is a lowering artifact; the
    // machine facts are that the jump reads the fold's destination, the fold's two
    // operands are the relocation-named table base and the delta loaded at 4-byte
    // scale off that base, and every slot target is a block this function owns.
    // So: the SAME switch, on both lowerings - and the interpreter answers per
    // slot, which is the claim that a mis-parse could not fake.
    let (f, rep) = lift_text(&o, &fx);
    verify_ssa(&f).expect("valid IR");
    assert_eq!(rep.jump_tables, 1, "the swapped fold models its table");
    assert_eq!(rep.jump_table_edges, 4);
    let sw = f
        .blocks
        .iter()
        .filter(|b| matches!(&b.term, Terminator::Switch { .. }))
        .count();
    assert_eq!(
        sw,
        1,
        "exactly one computed jump:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    let indirect = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(&i.op, rdec_core::ir::Op::Opaque(t) if t.as_ref() == "indirect jmp"))
        .count();
    assert_eq!(
        indirect,
        0,
        "the dispatch is modelled, so it must not also be an `indirect jmp` gap:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    for (k, want) in D_ANSWERS.iter().enumerate() {
        match run(&f, &[k as u64, 0, 0, 0]).outcome {
            Outcome::Ret(v) => assert_eq!(v, Val::Int(*want), "f({k}) via the swapped-fold table"),
            other => panic!("f({k}) did not return: {other:?}"),
        }
    }
    // No bounds check in this fixture: a slot past the end faults, never guesses.
    for k in [4u64, 99] {
        assert_eq!(
            run(&f, &[k, 0, 0, 0]).outcome,
            Outcome::Fault(Fault::SwitchOutOfRange { index: k, len: 4 }),
            "slot {k} of a 4-slot table faults, it does not guess"
        );
    }
    // And the emitted C is accepted, naming every case body - the same claim
    // fixture A makes about the local lowering.
    let out = rdec_core::emit_c::emit_c_sig(&f, &rdec_core::emit_c::Protos::new())
        .unwrap_or_else(|rs| panic!("the modelled switch must emit C, refused: {rs:?}"));
    for body in D_BODIES {
        assert!(
            out.text.contains(&format!(
                "case {}:",
                D_BODIES.iter().position(|&b| b == body).unwrap()
            )),
            "no case for body {body:#x}:\n{}",
            out.text
        );
    }
    assert!(
        out.text.contains("__builtin_trap()"),
        "past-the-table must trap, not fall through:\n{}",
        out.text
    );
}

#[test]
fn the_case_bodies_of_a_refused_dispatch_are_still_named_blocks_with_claimed_bytes() {
    // The half of the claim that does NOT depend on the lowering, on its own: for
    // fixture D - the CI runner's dispatch - every case body is a discovered block
    // whose bytes are claimed and whose first instruction is the `movl $n, %eax`
    // the source says is there. When the fold check is widened to accept the
    // swapped roles (a LOGIC change, reported separately), the `no Switch`
    // assertions in the test above are the red-first pin; the discovery claim here
    // is the part that never depended on the lowering.
    let (o, fx) = discover_bytes(&JT_D_OBJ);
    let tbl = fx.tables.get(&D_JMP).expect("fixture D table");
    assert_eq!(tbl.targets, D_BODIES);
    let ranges = code_ranges(&o).0;
    let text = ranges
        .first()
        .expect(".text is the fixture's first code range");
    for (body, want) in D_BODIES.iter().zip(D_ANSWERS) {
        assert!(
            fx.plan.blocks.contains(body),
            "case body {body:#x} is a discovered block: {:?}",
            fx.plan
        );
        let at = (body - text.base) as usize;
        assert_eq!(
            &text.bytes[at..at + 5],
            &[0xb8, (want & 0xff) as u8, 0x00, 0x00, 0x00],
            "the case body at {body:#x} is `movl ${want}, %eax`"
        );
    }
    assert!(fx.uncovered.is_empty(), "coverage: {:?}", fx.uncovered);
}

/// Fixture D: the dispatch is the `jmpq *%rax` at `0x18`, and its four case bodies
/// are `0x1a, 0x20, 0x26, 0x2c` per the quote above.
const D_JMP: u64 = 0x18;
const D_BODIES: [u64; 4] = [0x1a, 0x20, 0x26, 0x2c];
const D_ANSWERS: [u64; 4] = [7, 11, 13, 17];

/// The four case bodies of fixture B, per its quote: `0x11, 0x17, 0x1d, 0x23`.
const B_BODIES: [u64; 4] = [0x11, 0x17, 0x1d, 0x23];

/// The case bodies of fixture C: slot 3 is `0x32`, the `+8`; `0x37` is the
/// out-of-range arm the `ja` reaches, and `0x39` the join.
const C_BODIES: [u64; 4] = [0x27, 0x2a, 0x2d, 0x32];

/// A memory-indirect dispatch whose naming relocation IS `REL32`.
///
/// Fixture B is what an assembler really emits, and its ADDR32 naming relocation
/// is outside the recognition rule today, so B alone cannot test the ONE
/// variable this file is about: the dispatch instruction. This object is fixture
/// B's bytes with the naming relocation spelled `REL32` — the shape the
/// recogniser accepts — built with the container builder in `tests/common`
/// (`loader.rs`'s `.bss` test technique: header, section headers, section
/// symbols with aux records, relocation table, string table, data). Table slots,
/// addends and code bytes are byte-identical to fixture B; the `.rdata` run
/// carries four extra look-alike slots, as in fixture A.
///
/// With the table resolved and the dispatch memory-indirect, discovery claims
/// every byte of `.text` and the LIFTER still refuses the jump. That pair is the
/// anti-mis-parse claim, and it is what a sabotage that let the memory form
/// through as a `Terminator::Switch` turns red.
fn memory_indirect_named_by_rel32() -> Vec<u8> {
    let text: Vec<u8> = vec![
        0x48, 0x89, 0x4c, 0x24, 0x08, // 0x00 movq %rcx, 0x8(%rsp)
        0x48, 0x8b, 0x4c, 0x24, 0x08, // 0x05 movq 0x8(%rsp), %rcx
        0xff, 0x24, 0x8d, 0x00, 0x00, 0x00, 0x00, // 0x0a jmpq *(,%rcx,4)
        0xb8, 0x07, 0x00, 0x00, 0x00, 0xc3, // 0x11 movl $0x7,%eax; retq
        0xb8, 0x0b, 0x00, 0x00, 0x00, 0xc3, // 0x17 movl $0xb,%eax; retq
        0xb8, 0x0d, 0x00, 0x00, 0x00, 0xc3, // 0x1d movl $0xd,%eax; retq
        0xb8, 0x11, 0x00, 0x00, 0x00, 0xc3, // 0x23 movl $0x11,%eax; retq
    ];
    // addend = target + slot + 4, exactly fixture B's `.rdata` bytes, then the
    // four unrelocated look-alikes.
    let rdata: Vec<u8> = vec![
        0x15, 0x00, 0x00, 0x00, // slot 0 -> .text+0x11
        0x1f, 0x00, 0x00, 0x00, // slot 1 -> .text+0x17
        0x29, 0x00, 0x00, 0x00, // slot 2 -> .text+0x1d
        0x33, 0x00, 0x00, 0x00, // slot 3 -> .text+0x23
        0x28, 0x00, 0x00, 0x00, // no relocation: NOT a slot
        0x30, 0x00, 0x00, 0x00, // no relocation: NOT a slot
        0x38, 0x00, 0x00, 0x00, // no relocation: NOT a slot
        0x40, 0x00, 0x00, 0x00, // no relocation: NOT a slot
    ];
    common::build_object(
        &[
            (".text", 0x6000_0020, &text),
            (".rdata", 0x4050_0040, &rdata),
        ],
        &[
            // .text+0x0d: the jmp's displacement field -> `.rdata`, REL32.
            common::RelocRow {
                section: 0,
                address: 0x0d,
                typ: 4,
                symbol: 1,
            },
            // The four real slots, each `REL32 -> .text` (symbol row 0 is the
            // section's own definition symbol, what `llvm-objdump -r` prints).
            common::RelocRow {
                section: 1,
                address: 0x00,
                typ: 4,
                symbol: 0,
            },
            common::RelocRow {
                section: 1,
                address: 0x04,
                typ: 4,
                symbol: 0,
            },
            common::RelocRow {
                section: 1,
                address: 0x08,
                typ: 4,
                symbol: 0,
            },
            common::RelocRow {
                section: 1,
                address: 0x0c,
                typ: 4,
                symbol: 0,
            },
        ],
        &[
            common::SymRow {
                name: ".text",
                secnum: 1,
                value: 0,
                storage: 3,
            },
            common::SymRow {
                name: ".rdata",
                secnum: 2,
                value: 0,
                storage: 3,
            },
            common::SymRow {
                name: "f",
                secnum: 1,
                value: 0,
                storage: 2,
            },
        ],
    )
}

/// Discovery + lift of `f` over its own `.text`, in the space discovery builds,
/// with the object's tables handed to the lifter — the same entry point
/// `rdec c`/`rdec lift` use, and the same one `jump_tables.rs`'s corpus helper
/// drives.
struct Fixture {
    plan: Plan,
    tables: rdec_arch_x86::discover::Tables,
    uncovered: Vec<(u64, usize)>,
}

fn discover_bytes(bytes: &[u8]) -> (rdec_arch_x86::loader::Object, Fixture) {
    let (o, plan, tables) = discover_fixture(bytes);
    let unc = uncovered(&o, &plan, ".text");
    (
        o,
        Fixture {
            plan,
            tables,
            uncovered: unc,
        },
    )
}

/// Lift the object's `.text` as one function, from its first discovered start
/// to the end of the section.
fn lift_text(o: &rdec_arch_x86::loader::Object, fx: &Fixture) -> (Function, Report) {
    let (ranges, sp) = code_ranges(o);
    let tsec = o
        .sections
        .iter()
        .position(|s| s.name == ".text")
        .expect(".text");
    let base = sp.base_of[sp.idx_of[&tsec]];
    let rg = ranges.iter().find(|r| r.base == base).expect(".text range");
    let start = *fx.plan.functions.first().expect("f is discovered");
    let end = rg.base + rg.bytes.len() as u64;
    lift_one_sigs(
        "f",
        rg.bytes,
        rg.base,
        start,
        end,
        &fx.plan,
        &Default::default(),
        &rdec_arch_x86::lift::FnAbi::integer(),
        &no_callees(),
        &no_relocs(),
        &fx.tables,
    )
}

// ---------------------------------------------------------------------------
// THE REGISTER-INDIRECT DISPATCH: modelled, end to end.
// ---------------------------------------------------------------------------

#[test]
fn the_embedded_register_indirect_dispatch_becomes_cfg_edges_that_reach_every_case_body() {
    let (_o, fx) = discover_bytes(&JT_A_OBJ);
    // The dispatch is `jmpq *%rax` at .text+0x18 (see the quote).
    let t = fx.tables.get(&0x18).unwrap_or_else(|| {
        panic!(
            "no table at 0x18: {:?}",
            fx.tables.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(t.entries, 8, "eight REL32-covered slots in .rdata");
    assert!(t.unresolved.is_empty(), "{:?}", t.unresolved);
    assert_eq!(
        t.targets, A_BODIES,
        "the eight bodies, per the objdump quote"
    );
    assert_eq!(
        fx.plan.gap_count(Gap::IndirectBranch),
        0,
        "{:?}",
        fx.plan.gaps
    );
    assert!(
        fx.uncovered.is_empty(),
        "uncovered .text: {:?}",
        fx.uncovered
    );

    let (f, rep) = lift_text(&_o, &fx);
    // The dispatch is no longer an `indirect jmp` refusal.
    let indirect = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(&i.op, rdec_core::ir::Op::Opaque(t) if t.as_ref() == "indirect jmp"))
        .count();
    assert_eq!(indirect, 0, "the computed jump is modelled, not refused");
    // The parameters are the seeded ones: four, from `FnAbi::integer()`.
    assert_eq!(f.params.len(), 4, "RCX/RDX/R8/R9 are parameters 0..=3");

    let cfg = Cfg::build(&f);
    for a in A_BODIES {
        let bb = f.blocks.iter().find(|b| b.addr == a).unwrap_or_else(|| {
            panic!(
                "case body {a:#x} is not a block: {:?}",
                f.blocks.iter().map(|b| b.addr).collect::<Vec<_>>()
            )
        });
        assert!(
            cfg.is_reachable(bb.id),
            "case body {a:#x} (bb{}) is unreachable",
            bb.id
        );
    }
    verify_ssa(&f).expect("the switch function is valid IR");
    let switches: Vec<&Terminator> = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Switch { .. }))
        .map(|b| &b.term)
        .collect();
    assert_eq!(switches.len(), 1, "one dispatch, exactly");
    let Terminator::Switch { index, targets } = switches[0] else {
        unreachable!()
    };
    let got: Vec<u64> = targets.iter().map(|&b| f.block(b).addr).collect();
    assert_eq!(
        got, A_BODIES,
        "the switch's successors, in table slot order"
    );
    assert_ne!(*index, NO_VAL, "the dispatch has a selector");
    // Counted at the `jmp`, where the evidence is.
    assert_eq!(rep.jump_tables, 1);
    assert_eq!(rep.jump_table_edges, 8);
    assert_eq!(rep.jump_table_missing, 0);
    assert_eq!(rep.jump_table_partial, 0);
}

#[test]
fn the_embedded_interpreter_runs_the_switch_and_the_case_bodies_answer() {
    let (_o, fx) = discover_bytes(&JT_A_OBJ);
    let (f, _rep) = lift_text(&_o, &fx);
    // The answers are the immediates in the objdump quote: `movl $0x7`, `$0xb`,
    // `$0xd`, `$0x11`, `$0x13`, `$0x17`, `$0x1d`, `$0x1f`.
    for (k, want) in A_ANSWERS.iter().enumerate() {
        match run(&f, &[k as u64, 0, 0, 0]).outcome {
            Outcome::Ret(v) => assert_eq!(v, Val::Int(*want), "f({k}) via the jump table"),
            other => panic!("f({k}) did not return: {other:?}"),
        }
    }
    // This fixture has NO bounds check, so an index past the table names no
    // destination the function owns. The model stops the run on a fault — never
    // a fallthrough, never a picked block.
    for k in [8u64, 99] {
        assert_eq!(
            run(&f, &[k, 0, 0, 0]).outcome,
            Outcome::Fault(Fault::SwitchOutOfRange { index: k, len: 8 }),
            "slot {k} of an 8-slot table faults, it does not guess"
        );
    }
}

#[test]
fn the_embedded_fallthrough_switch_answers_through_the_same_edges() {
    let (_o, fx) = discover_bytes(&JT_C_OBJ);
    let t = fx.tables.get(&0x25).unwrap_or_else(|| {
        panic!(
            "no table at the dispatch 0x25: {:?}",
            fx.tables.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(t.targets, C_BODIES, "slot order per the quote");
    assert_eq!(
        fx.plan.gap_count(Gap::IndirectBranch),
        0,
        "{:?}",
        fx.plan.gaps
    );
    assert!(
        fx.uncovered.is_empty(),
        "uncovered .text: {:?}",
        fx.uncovered
    );

    let (f, rep) = lift_text(&_o, &fx);
    verify_ssa(&f).expect("valid IR");
    assert_eq!(rep.jump_tables, 1);
    assert_eq!(rep.jump_table_edges, 4);
    assert_eq!(f.params.len(), 4);
    // 7 = +1+2+4 (slot 0 falls through slots 1 and 2), 6, 4, 8; the `ja` at 0x10
    // routes 4 and 9 to the zeroing arm at 0x37, so they answer 0.
    for (input, answer) in [(0u64, 7u64), (1, 6), (2, 4), (3, 8), (4, 0), (9, 0)] {
        match run(&f, &[input, 0, 0, 0]).outcome {
            Outcome::Ret(v) => assert_eq!(v, Val::Int(answer), "f({input})"),
            other => panic!("f({input}) did not return: {other:?}"),
        }
    }
}

#[test]
fn the_embedded_c_names_every_case_body_and_traps_past_the_table() {
    // The emitter's `switch` is the CFG's edge list in C spelling: one `case`
    // per slot, each `goto` naming the same label the rest of the file uses for
    // that block, and a `default` that traps - the model's standing answer for a
    // destination it does not have.
    let (mut f, _rep) = {
        let (_o, fx) = discover_bytes(&JT_A_OBJ);
        lift_text(&_o, &fx)
    };
    clean(&mut f);
    let out = rdec_core::emit_c::emit_c_sig(&f, &rdec_core::emit_c::Protos::new())
        .expect("the switch function emits C");
    assert!(out.text.contains("switch ("), "a switch statement");
    assert_eq!(
        out.text.matches("case ").count(),
        8,
        "one case per table slot:\n{}",
        out.text
    );
    assert!(
        out.text.contains("default: __builtin_trap();"),
        "the out-of-table arm traps:\n{}",
        out.text
    );
    for k in 0..8 {
        // M6 epilogue materialization (2026-11-06) moved this pin on purpose:
        // arms that flow into a pure return tail now spell `return EXPR;`
        // directly and the shared label is gone. The machine fact this fixture
        // owns is the SWITCH SHAPE — one case per slot, every arm transfers
        // control; the printer's choice between `goto l…` and a direct
        // `return` is exactly the readability progress the corpus census
        // measures, and a pin on the printer's current spelling pins today's
        // readability bug as law.
        let arm = format!("case {k}: ");
        let at = out
            .text
            .find(&arm)
            .unwrap_or_else(|| panic!("slot {k} has no case label"));
        let head = out.text[at + arm.len()..].lines().next().unwrap_or("");
        assert!(
            head.starts_with("goto l") || head.starts_with("return"),
            "slot {k} must transfer control (block label or direct return); got: {head}"
        );
    }

    // The ONE clang invocation in this file: the emitted C is compiled with the
    // floor gate's exact flags (`m4::compile`, spelled out so this cannot drift
    // from what CI compiles the emitted C with).
    let dir = std::env::temp_dir().join("rdec_jump_table_fixtures");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let c = dir.join("jt_a.c");
    std::fs::write(&c, format!("{}\n{}", rdec_core::emit_c::PRELUDE, out.text)).expect("write C");
    let build = std::process::Command::new("clang")
        .args([
            "-std=c17",
            "-c",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
            &c.display().to_string(),
            "-o",
            &dir.join("jt_a.o").display().to_string(),
        ])
        .output()
        .expect("clang runs");
    assert!(
        build.status.success(),
        "clang -Werror rejects the switch TU:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
}

// ---------------------------------------------------------------------------
// THE MEMORY-INDIRECT DISPATCH: refused by name, never mis-parsed.
// ---------------------------------------------------------------------------

#[test]
fn the_memory_indirect_dispatch_the_assembler_emits_is_a_counted_gap_not_a_switch() {
    // Fixture B, the bytes an assembler really produces: the naming relocation
    // is ADDR32, the table is therefore not named by anything the recognition
    // rule reads, and the shape stays a LOUD counted gap. Nothing here is
    // modelled, and nothing here pretends to be.
    let (o, fx) = discover_bytes(&JT_B_OBJ);
    assert!(
        fx.tables.is_empty(),
        "an ADDR32 naming relocation is not a REL32 table base: {:?}",
        fx.tables
    );
    assert_eq!(
        fx.plan.gap_count(Gap::IndirectBranch),
        1,
        "the `jmpq *(,%rcx,4)` at 0xa is a counted gap: {:?}",
        fx.plan.gaps
    );
    assert_eq!(
        fx.plan.gaps[0].va, 0xa,
        "the gap is recorded at the jump, where the evidence is"
    );
    // The four bodies are bytes the sweep cannot reach. Coverage says so, per
    // byte: 0x11..0x29 is 24 bytes, the four `movl $n, %eax; retq` bodies.
    assert_eq!(
        fx.uncovered,
        vec![(0x11, 24)],
        "the case bodies stay unclaimed, and the gap is counted in bytes: {:?}",
        fx.uncovered
    );
    // And the object really does name them: this is a recognition limit, not a
    // missing fact. Ground truth is the `.rdata` addends in the header quote.
    let named = common::relocated_table_targets(&o, ".rdata");
    assert_eq!(
        named.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
        B_BODIES,
        "the object's own relocations name the four bodies"
    );

    let (f, rep) = lift_text(&o, &fx);
    assert_eq!(rep.jump_tables, 0, "no table, so no switch");
    assert_eq!(rep.jump_table_edges, 0);
    assert!(
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::Switch { .. })),
        "a memory-indirect jump with no resolved table became a Switch:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // The refusal is named, and located at the jump.
    let refusals = match rdec_core::emit_c::emit_c_sig(&f, &rdec_core::emit_c::Protos::new()) {
        Ok(_) => panic!("the memory-indirect dispatch emitted C: it has no target set"),
        Err(rs) => rs,
    };
    assert_eq!(
        refusals
            .iter()
            .map(|r| (r.at, r.reason.as_str()))
            .collect::<Vec<_>>(),
        vec![(0xa, "opaque: indirect jmp")],
        "a named refusal at the jump, not a silent success"
    );
}

#[test]
fn a_resolved_table_behind_a_memory_indirect_dispatch_is_discovered_though_the_jump_is_refused() {
    // The one-variable experiment against fixture A: same table bytes, same
    // COFF arithmetic, same naming relocation class (`REL32 -> .rdata`), and the
    // ONLY difference is the dispatch instruction — `jmpq *(,%rcx,4)` reads the
    // slot directly instead of `lea`/`movslq`/`add`/`jmp *%rax`.
    //
    // Discovery needs NO register form: `jump_tables` derives the table from the
    // object's relocations (`REL32 -> .rdata` names the base, consecutive
    // `REL32 -> .text` slots are the entries) and `dispatch_site` walks from the
    // relocation's own instruction to the first indirect `jmp` — which here is
    // the instruction that CARRIES the relocation. So the case bodies are
    // discovered and claimed, and the coverage claim does not read the
    // instruction's shape. The LIFTER's rule is the narrow one: its
    // `scan_dispatches` requires the 4-instruction register lowering to know
    // which value is the selector, and a jump it cannot give a selector to is a
    // named refusal. Discovery and lifting are separate claims.
    let bytes = memory_indirect_named_by_rel32();
    let (o, fx) = discover_bytes(&bytes);
    let t = fx.tables.get(&0xa).unwrap_or_else(|| {
        panic!(
            "no table at the dispatch 0xa: {:?}",
            fx.tables.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(t.entries, 4, "the four relocated slots");
    assert!(t.unresolved.is_empty(), "{:?}", t.unresolved);
    assert_eq!(
        t.targets, B_BODIES,
        "every body the relocations name is discovered"
    );
    for a in B_BODIES {
        assert!(
            fx.plan.blocks.contains(&a),
            "case body {a:#x} is not a block"
        );
    }
    assert_eq!(
        fx.plan.gap_count(Gap::IndirectBranch),
        0,
        "{:?}",
        fx.plan.gaps
    );
    assert!(
        fx.uncovered.is_empty(),
        "a memory-indirect dispatch left bytes unclaimed: {:?}",
        fx.uncovered
    );

    // ...and the lifter refuses the same site, by name.
    let (f, rep) = lift_text(&o, &fx);
    assert_eq!(rep.jump_tables, 0, "the memory form is not a Switch");
    assert_eq!(rep.jump_table_edges, 0);
    assert!(
        !f.blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::Switch { .. })),
        "the lifter modelled a dispatch it has no selector for:\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    let refusals = match rdec_core::emit_c::emit_c_sig(&f, &rdec_core::emit_c::Protos::new()) {
        Ok(_) => panic!("a dispatch with no selector emitted C"),
        Err(rs) => rs,
    };
    assert_eq!(
        refusals
            .iter()
            .map(|r| (r.at, r.reason.as_str()))
            .collect::<Vec<_>>(),
        vec![(0xa, "opaque: indirect jmp")],
        "the refusal names the shape, at the jump"
    );
}

#[test]
fn a_table_run_ends_where_its_relocations_end() {
    // `.rdata` here is 32 bytes = 8 four-byte slots, four of them relocated.
    // The values of the unrelocated four (`0x28`, `0x30`, `0x38`, `0x40`) decode
    // as `.text+0x4, 0x8, 0xc, 0x10` — all inside the code, all plausible. The
    // rule is the relocation, so the table is four entries and those addresses
    // appear nowhere.
    let bytes = memory_indirect_named_by_rel32();
    let (_o, fx) = discover_bytes(&bytes);
    assert_eq!(fx.tables.len(), 1, "one table, at one dispatch");
    let t = fx.tables.values().next().expect("the table");
    assert_eq!(t.entries, 4, "the run stops where the relocations stop");
    assert_eq!(t.targets, B_BODIES, "the relocated slots only");
    for bogus in [0x4u64, 0x8, 0xc, 0x10] {
        assert!(
            !t.targets.contains(&bogus),
            "an unrelocated 4-byte value was read as a jump-table target: {bogus:#x}"
        );
        assert!(
            !fx.plan.blocks.contains(&bogus),
            "a block was started at {bogus:#x} from bytes that name nothing"
        );
    }
}

/// The pre-emission pipeline, as `rdec c` runs it.
fn clean(f: &mut Function) {
    for _ in 0..16 {
        let mut ch = false;
        ch |= rdec_core::fuse::flag_elimination(f);
        ch |= rdec_core::passes::prune_phis(f);
        ch |= rdec_core::passes::const_fold(f);
        ch |= rdec_core::passes::copy_prop(f);
        ch |= rdec_core::passes::simplify_branches(f);
        ch |= rdec_core::passes::dead_store_elimination(f);
        ch |= rdec_core::passes::dead_code_elimination(f);
        if !ch {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// THE `aa == ab` GUARD, WITNESSED. (Named gap in lift.rs: the guard rejecting
// the degenerate `add r,r` had no fixture; a first attempt died on the REL32
// addend convention. This one reuses JT_A's own bytes, so the convention is
// JT_A's, quoted above, and only the FOLD is patched.)
// ---------------------------------------------------------------------------

#[test]
fn a_degenerate_add_of_a_register_with_itself_is_not_a_fold() {
    // The shape: `lea (%rip),%rax` names the table; `movslq (%rax,%rcx,4),%rax`
    // loads the slot INTO the base register; `add %rax,%rax` doubles it;
    // `jmp *%rax` dispatches. The role check alone ACCEPTS this (`aa == lr &&
    // ab == ld`, both %rax), and the doubling means the dispatched address is
    // 2*(base + delta) - not a table entry. Recognizing it would fabricate
    // switch successors the machine code does not have.
    //
    // Byte-patched from JT_A_OBJ, same instruction sizes:
    //   48 63 0c 88  movslq (%rax,%rcx,4),%rcx  ->  48 63 04 90  movslq (%rax,%rcx,4),%rax
    //   48 01 c8     add    %rcx,%rax           ->  48 01 c0     add    %rax,%rax
    // The .rdata table, the naming relocation, and the `jmp *%rax` are
    // UNCHANGED: the table is real; the fold that reads it is fake.
    let mut obj = JT_A_OBJ;
    let pat = [0x48, 0x63, 0x0c, 0x88, 0x48, 0x01, 0xc8, 0xff, 0xe0];
    let i = obj
        .windows(pat.len())
        .position(|w| w == pat)
        .expect("JT_A's fold+dispatch bytes exist in its own object");
    obj[i..i + pat.len()].copy_from_slice(&[0x48, 0x63, 0x04, 0x90, 0x48, 0x01, 0xc0, 0xff, 0xe0]);

    let (o, fx) = discover_bytes(&obj);
    // The table evidence is untouched - discovery keys it at the jmp, as in JT_A.
    assert_eq!(
        fx.tables.keys().copied().collect::<Vec<_>>(),
        vec![0x18],
        "the .rdata table and its relocations are unchanged; only the fold is degenerate"
    );

    let (f, rep) = lift_text(&o, &fx);
    // Lift must NOT model it: the guard rejects the doubling fold.
    assert_eq!(
        rep.jump_tables, 0,
        "a degenerate `add %rax,%rax` doubles the base register - it is not `base + delta`, \
         and modelling it invents switch targets the machine code does not have"
    );
    let switches = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Switch { .. }))
        .count();
    assert_eq!(switches, 0, "no Switch terminator from a fake fold");
    // The dispatch stays a LOUD, counted refusal, exactly as an unmodelled
    // computed jump does everywhere else in this repo.
    let indirect = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(&i.op, rdec_core::ir::Op::Opaque(t) if t.as_ref() == "indirect jmp"))
        .count();
    assert_eq!(
        indirect, 1,
        "the dispatch must remain a loud `indirect jmp` gap, not a fabricated switch"
    );
    // The LOUD home of this refusal is the lift-level `Opaque` above: the
    // DISCOVERY side keys a table at the jmp from the relocations, so it
    // treats the site as a dispatch (no plan gap) - which is exactly why the
    // fold guard, not the plan, is the thing this test witnesses. The refusal
    // is counted in the lift report.
    assert!(
        rep.gaps >= 1,
        "the refused dispatch must be counted in the lift report, got {rep:?}"
    );
}
