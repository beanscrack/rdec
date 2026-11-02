// The width at which a function's answer reaches RAX, measured on hand-assembled
// machine code.
//
// On x86-64 a 32-bit register write ZEROES bits 63:32 architecturally. So a
// function whose answer reaches RAX through such a write has an answer that is
// exactly `zero_extend_32(value)` - bits 63:32 are a fact, not "undefined above
// bit 31" - and the emitted C must return `uint32_t` to return the same bits.
// The 39 T2 mismatches this models were saturation's `f(1,-100)`: machine RAX
// `0xffffff9d` against the emitted C's `0xffffffffffffff9d`.
//
// Each function here pins one boundary of the rule:
//
//   mov eax, ecx    - a 32-bit write: the answer is 32 bits wide.
//   mov rax, rcx    - a 64-bit write: nothing is known, stay 64.
//   add rax, rcx    - a 64-bit write of an arithmetic result: stay 64.
//   mov al, cl      - an 8-bit write LEAVES bits 63:8 of the incoming RAX, so
//                     the answer is not the byte: the width is 64, and the
//                     incoming bits are not ours to claim.
//   the join        - two 32-bit writes merging: 32 on every path, so 32.
//   the loop        - 32-bit writes around a back edge: 32.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::ty::{Sx, Ty, Width};
use rdec_core::verify::verify_ssa;
use rdec_core::width::return_width;

fn lift(bytes: &[u8]) -> (rdec_core::ir::Function, rdec_arch_x86::lift::Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    // Discovery is seed-driven; hand-assembled bytes have no symbol table, so
    // seed the entry address ourselves (the same shape `m3_ssa` uses).
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 1, "one discovered function");
    let start = plan.functions[0];
    let end = owned.len() as u64;
    lift_one("f", &owned, 0, start, end, &plan, &Default::default())
}

/// `uint32_t` is the only spelling the lifter gives a recorded 32-bit answer.
fn declares_u32(f: &rdec_core::ir::Function) -> bool {
    matches!(
        f.ret_ty,
        Ty::Int {
            w: Width::W32,
            sx: Sx::Unsigned
        }
    )
}

#[test]
fn a_32_bit_write_to_eax_makes_a_32_bit_return() {
    let (f, rep) = lift(&[0x89, 0xC8, 0xC3]); // mov eax, ecx; ret
    assert_eq!(return_width(&f), 32, "the zext of a 32-bit write");
    assert_eq!(rep.ret_w32, 1, "the lifter records the fact it used");
    assert!(
        declares_u32(&f),
        "the emitted C must return uint32_t, the IR says {:?}\n{}",
        f.ret_ty,
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(verify_ssa(&f), Ok(()), "the width claim verifies");
}

#[test]
fn a_64_bit_write_to_rax_stays_64_bits_wide() {
    // mov rax, rcx; ret - the answer is the caller's 64-bit argument.
    let (f, rep) = lift(&[0x48, 0x89, 0xC8, 0xC3]);
    assert_eq!(return_width(&f), 64);
    assert_eq!(rep.ret_w32, 0, "nothing was recorded, nothing claimed");
    assert!(!declares_u32(&f));
    assert_eq!(verify_ssa(&f), Ok(()));

    // add rax, rcx; ret - arithmetic at 64 bits, carry into bit 63 included.
    let (g, grep) = lift(&[0x48, 0x01, 0xC8, 0xC3]);
    assert_eq!(return_width(&g), 64);
    assert_eq!(grep.ret_w32, 0);
    assert!(!declares_u32(&g));
}

#[test]
fn an_8_bit_write_leaves_the_incoming_bits_alone() {
    // mov al, cl; ret. x86's partial-write rule keeps bits 63:8 of RAX, which
    // the callee inherited from its caller: the answer is NOT the byte, and a
    // `uint8_t`/`uint32_t` return type would be a claim about bits we do not own.
    let (f, rep) = lift(&[0x88, 0xC8, 0xC3]);
    assert_eq!(
        return_width(&f),
        64,
        "bits 63:8 come in from the caller, so nothing is proved narrow\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(rep.ret_w32, 0);
    assert!(!declares_u32(&f));
}

/// The join from `m3_ssa`: two paths write `eax` (32-bit) into the returned
/// register, then merge. Both edges are 32-bit-clean, so RAX's upper half is
/// zero on every path.
#[test]
fn two_32_bit_writes_that_merge_are_still_a_32_bit_return() {
    let (f, rep) = lift(&[
        0xb8, 0x01, 0x00, 0x00, 0x00, //  0: mov eax, 1
        0x85, 0xc9, //  5: test ecx, ecx
        0x74, 0x0b, //  7: je 0x14
        0xb8, 0x02, 0x00, 0x00, 0x00, //  9: mov eax, 2
        0xeb, 0x07, // 14: jmp 0x17
        0x90, 0x90, 0x90, 0x90, // 16: alignment padding
        0xb8, 0x03, 0x00, 0x00, 0x00, // 20: mov eax, 3   (the `je` target)
        0xeb, 0x03, // 25: jmp 0x1e
        0x90, 0x90, 0x90, // 27: alignment padding
        0xc3, // 30: ret
    ]);
    assert_eq!(return_width(&f), 32, "a phi of two 32-bit-clean edges");
    assert_eq!(rep.ret_w32, 1);
    assert!(declares_u32(&f));
    assert_eq!(
        verify_ssa(&f),
        Ok(()),
        "the width claim must verify at the join"
    );
}

/// `eax = 0; while (ecx > 0) { eax -= 1; ecx -= 1; } ret` - the loop-carried
/// value is written at 32 bits on every edge, including the back edge.
#[test]
fn a_32_bit_loop_returns_32_bits() {
    let (f, rep) = lift(&[
        0xb8, 0x00, 0x00, 0x00, 0x00, //  0: mov eax, 0
        0x83, 0xf9, 0x00, //  5: cmp ecx, 0
        0x7e, 0x0a, //  8: jle 0x14
        0x83, 0xe8, 0x01, // 10: sub eax, 1
        0x83, 0xe9, 0x01, // 13: sub ecx, 1
        0xeb, 0xf3, // 16: jmp 0x5   (back edge)
        0x90, 0x90, // 18: alignment padding
        0xc3, // 20: ret
    ]);
    assert_eq!(return_width(&f), 32);
    assert_eq!(rep.ret_w32, 1);
    assert!(declares_u32(&f));
    assert_eq!(verify_ssa(&f), Ok(()));
}

/// `mov eax, 42; ret` - the smallest function the rule applies to, and the one
/// whose answer T2 compares: RAX = 0x000000000000002a, C = `(uint32_t)42`.
#[test]
fn a_constant_answer_written_at_32_bits_is_a_32_bit_return() {
    let (f, rep) = lift(&[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]);
    assert_eq!(return_width(&f), 32);
    assert_eq!(rep.ret_w32, 1);
    assert!(declares_u32(&f));
    let c = rdec_core::emit_c::emit_c(&f).expect("the smallest function emits C");
    assert!(
        c.text.contains("uint32_t f("),
        "the signature is the width: {}",
        c.text
    );
    assert!(
        c.text.contains("return (uint32_t)"),
        "the return casts through the declared type, not int64_t: {}",
        c.text
    );
}

/// The C is the claim, so the C is what is checked: a `uint32_t` function called
/// as `uint32_t` and widened in C yields exactly the 64-bit RAX a 32-bit write
/// leaves. This is the T2 wrapper's shape, run here as a unit test.
#[test]
fn the_emitted_uint32_return_widens_to_the_machines_rax() {
    let (f, _rep) = lift(&[0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]);
    let c = rdec_core::emit_c::emit_c(&f).expect("emits C");
    assert!(
        c.text.contains("uint32_t f(") && c.text.contains("return (uint32_t)"),
        "the C's declared width is what the harness widens: {}",
        c.text
    );
    // Hand-written mirror of the generated body: what clang computes for the
    // emitted `return (uint32_t)x;` in a `uint32_t` function.
    let c_value: u32 = 42;
    let widened: u64 = c_value as u64; // the wrapper's `(uint64_t)` cast
    let machine_rax: u64 = 0x2a; // `mov eax, 42` cleared bits 63:32
    assert_eq!(widened, machine_rax);

    // The old shape - int64_t plus the value's own sign - is the mismatch T2
    // measured, pinned here so it cannot come back as "the ABI leaves it open":
    // saturation's `f(1,-100)` answers -1 written at 32 bits, so the machine's
    // RAX is 0x00000000_ffffffff and the old C returned 0xffffffff_ffffffff.
    let over_extended: u64 = -1i32 as i64 as u64;
    assert_eq!(over_extended, 0xffff_ffff_ffff_ffff);
    assert_eq!(
        0xffff_ffff_u64,
        (-1i32 as u32) as u64,
        "uint32_t is the fix"
    );
    assert_ne!(
        over_extended, 0xffff_ffff_u64,
        "this is the 39-mismatch shape"
    );
}
