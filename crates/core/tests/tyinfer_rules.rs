//! M6 type recovery, one test per rule.
//!
//! Each fixture is hand-built IR in the shape the lifter produces, and each
//! test pins ONE rule of the module header. The rules that produce silence are
//! as important as the ones that produce a claim: a claim the machine code does
//! not force is the bug this whole slice exists to avoid (the corpus's source
//! truth, scored separately, is what makes an over-claim measurable).

mod common;
use common::*;

use rdec_core::ir::{BinOp, CastKind, IntCmp, Terminator};
use rdec_core::ty::{FWidth, Sx, Width};
use rdec_core::tyinfer;
use rdec_core::verify;

fn ok(f: &rdec_core::ir::Function) {
    verify::verify(f).expect("fixture must verify");
}

// ---- signedness: the operator carries it, the facts travel the cast chain ----

#[test]
fn signed_compare_makes_its_operands_signed_through_the_trunc() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let five = k(&mut f, 5, Width::W32);
    cmp(&mut f, IntCmp::Lt, t, five);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "Lt proves its operands signed"
    );
    assert_eq!(
        sig.params[0].c_type().as_deref(),
        Some("int32_t"),
        "signed + read-at-32"
    );
}

#[test]
fn unsigned_compare_makes_its_operands_unsigned() {
    // `cmp` + `jb` promises a zero-fill reading for the bytes it compares: it
    // marks the operand it reads. When the compare reads the cell, the cell
    // prints `uint32_t`.
    let (mut f, c) = cells(1);
    let p = c[0];
    f.set_ty(p, u32());
    let five = k(&mut f, 5, Width::W32);
    cmp(&mut f, IntCmp::Ule, p, five);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(sig.params[0].sx, Some(Sx::Unsigned));
    assert_eq!(sig.params[0].c_type().as_deref(), Some("uint32_t"));
}

#[test]
fn unsigned_compare_on_a_truncated_copy_leaves_the_cell_silent() {
    // The spill shape (`fib_rec.O1`, `gcd_loop.O1`): the compare reads the
    // 32-bit COPY of the cell, and a signed `int` spilled to that copy answers
    // `jb` byte-identically on the low half. The witness is about the
    // materialisation, not the cell, so the cell's signedness stays silent and
    // the display is canonical-signed (the gate counts it in the loud silent
    // bucket). A `uint32_t` print must be earned on the cell itself.
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let five = k(&mut f, 5, Width::W32);
    cmp(&mut f, IntCmp::Ule, t, five);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx, None,
        "the compare read the copy, not the cell"
    );
    assert_eq!(
        sig.params[0].width,
        Some(Width::W32),
        "the trunc bounds the width, not the sign"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("int32_t"));
}

#[test]
fn equality_says_nothing_about_signedness() {
    // `Eq`/`Ne` answer identically for every signed interpretation: the claim
    // must be silence, not a default.
    let (mut f, c) = cells(1);
    let p = c[0];
    let five = k(&mut f, 5, Width::W64);
    cmp(&mut f, IntCmp::Eq, p, five);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(sig.params[0].sx.is_none(), "Eq is sign-blind");
    assert!(sig.params[0].is_silent(), "and leaves no claim at all here");
}

#[test]
fn a_signed_division_proves_its_operands_signed() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let three = k(&mut f, 3, Width::W64);
    bin(&mut f, BinOp::SDiv, Width::W64, p, three);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "an idiv with cdq is a signed divide"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("int64_t"));
}

#[test]
fn an_unsigned_division_proves_its_operands_unsigned() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let three = k(&mut f, 3, Width::W64);
    bin(&mut f, BinOp::UDiv, Width::W64, p, three);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(sig.params[0].sx, Some(Sx::Unsigned));
}

#[test]
fn arithmetic_shift_right_proves_the_shifted_operand_signed() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Sar, Width::W64, p, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "sar propagates a sign, so its input has one"
    );
}

#[test]
fn logical_shift_right_proves_the_shifted_operand_unsigned() {
    // `sar` and `shr` touch DIFFERENT cells for the same source expression
    // (`mov rax, rcx; shr rax, 32; sar rcx, 1`, the `int64_t -> uint32_t` cast
    // shape), pinning the copy-splitting rule for placement.
    let (mut f, c) = cells(1);
    let p = c[0];
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Shr, Width::W64, p, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(sig.params[0].sx, Some(Sx::Unsigned));
}

#[test]
fn opposite_shifts_of_one_value_are_a_conflict_not_a_type() {
    // `shift_all.c`: `y = (unsigned long)a >> n` beside `z = a >> n`. The right
    // answer is "this value's signedness is a cast", and printing a type for it
    // is the lie.
    let (mut f, c) = cells(1);
    let p = c[0];
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Sar, Width::W64, p, one);
    bin(&mut f, BinOp::Shr, Width::W64, p, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.params[0].sx_conflict,
        "sar and shr on one value: evidence disagrees"
    );
    assert!(
        sig.params[0].sx.is_none(),
        "and no signedness may be printed"
    );
    assert_eq!(sig.sx_conflicts, 1);
}

#[test]
fn sext_of_a_truncated_parameter_is_a_signed_int32() {
    // The corpus's widest signedness evidence: `movsxd rax, dword [rbp-4]` on
    // the W64 parameter cell. A rule that refuses to cross the lifter's
    // 64-to-32 `trunc` finds signedness nowhere in the corpus.
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let e = sext(&mut f, t);
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Add, Width::W64, e, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "sext signs the value it extends"
    );
    assert_eq!(
        sig.params[0].width,
        Some(Width::W32),
        "the trunc bounds the read at 32 bits"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("int32_t"));
}

#[test]
fn movzx_of_a_thirty_two_bit_cell_is_an_unsigned_int32() {
    // `movzx rax, ecx` where ecx IS the cell: the zero fill is a machine fact
    // about the cell's high half, which `uint32_t` is the display of.
    let (mut f, c) = cells(1);
    let p = c[0];
    f.set_ty(p, u32());
    let e = zext(&mut f, p);
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Add, Width::W64, e, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Unsigned),
        "the movzx of a 32-bit cell"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("uint32_t"));
}

#[test]
fn zext_of_a_truncated_parameter_prints_canonical_signed() {
    // The promote pair `zext(trunc(x))` is the spill shape: the 32-bit read of
    // the W64 cell is the spill store, the `movzx` materialises the copy for a
    // wider sale, and a signed `int` spilled is byte-identical. The witness is
    // about the copy; the cell prints canonical-signed and the gate counts the
    // silence in its loud bucket (see the 32-bit-cell twin above).
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let e = zext(&mut f, t);
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Add, Width::W64, e, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx, None,
        "a promote witness is a fact about the copy, not the cell"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("int32_t"));
}

// ---- width: a bound, narrowed by facts, never invented ----

#[test]
fn a_read_at_32_bits_bounds_the_parameter_at_32_bits() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let one = k(&mut f, 1, Width::W32);
    bin(&mut f, BinOp::Add, Width::W32, t, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].width,
        Some(Width::W32),
        "bits above 32 can never be read"
    );
    assert!(
        sig.params[0].sx.is_none(),
        "width says nothing about signedness"
    );
    // Width says nothing about signedness, so the signedness is not forced;
    // the printed form is then canonical-signed (see `Claim::c_type`). A
    // `uint32_t` print must be earned by an unsigned fact, never defaulted to.
    assert_eq!(
        sig.params[0].c_type().as_deref(),
        Some("int32_t"),
        "unforced signedness prints canonical-signed"
    );
}

#[test]
fn a_64_bit_read_is_not_a_claim() {
    // The register cell's starting width proves nothing; a claim must be a
    // bound NARROWER than the machine word.
    let (mut f, c) = cells(1);
    let p = c[0];
    let one = k(&mut f, 1, Width::W64);
    bin(&mut f, BinOp::Add, Width::W64, p, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.params[0].is_silent(),
        "an unbounded width is silence, not `uint64_t`"
    );
}

#[test]
fn a_stored_width_bounds_the_stored_value() {
    // `mov dword [rbp-8], ecx` at -O0: the home is 4 bytes, and nothing wider
    // can reach an answer through it.
    let (mut f, c) = cells(2);
    let p = c[0];
    let r = rsp(&mut f);
    let off = k(&mut f, (-8i64) as u64, Width::W64);
    let slot = bin(&mut f, BinOp::Add, Width::W64, r, off);
    let t = trunc(&mut f, p, Width::W32);
    store(&mut f, rdec_core::ty::Ty::uint(Width::W32), slot, t);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(sig.params[0].width, Some(Width::W32));
    assert!(
        sig.params[1].is_silent(),
        "q is untouched: q gets no claim, invented or borrowed"
    );
}

// ---- pointer: the base of a memory access, under arithmetic ----

#[test]
fn a_load_base_is_a_pointer() {
    let (mut f, c) = cells(1);
    let p = c[0];
    let v = load(&mut f, u64(), p);
    ret(&mut f, v);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(sig.params[0].ptr, "the base of a load IS an address");
    assert_eq!(sig.params[0].c_type().as_deref(), Some("void *"));
    assert!(
        sig.params[0].sx.is_none(),
        "pointer signedness is not a C fact"
    );
}

#[test]
fn the_pointer_fact_survives_a_constant_offset() {
    // The struct read: `mov rax, [rbp+8]` is `Load(Add(p, 8))`; the fact
    // belongs to `p`, not to the sum.
    let (mut f, c) = cells(1);
    let p = c[0];
    let four = k(&mut f, 4, Width::W64);
    let a = bin(&mut f, BinOp::Add, Width::W64, p, four);
    let v = load(&mut f, u64(), a);
    ret(&mut f, v);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(sig.params[0].ptr, "p + 4 is rooted at p");
}

#[test]
fn base_plus_scaled_index_points_at_the_base_only() {
    // The array shape `a[i]`: the base is the pointer, the index is not.
    let (mut f, c) = cells(2);
    let (base, idx) = (c[0], c[1]);
    let eight = k(&mut f, 8, Width::W64);
    let scaled = bin(&mut f, BinOp::Mul, Width::W64, idx, eight);
    let addr = bin(&mut f, BinOp::Add, Width::W64, base, scaled);
    let v = load(&mut f, u64(), addr);
    ret(&mut f, v);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(sig.params[0].ptr, "the base gets the pointer claim");
    assert!(
        sig.params[1].is_silent(),
        "the index gets a silent claim at worst, never a pointer"
    );
}

#[test]
fn values_behind_arithmetic_stay_silent() {
    // The walking rebuild: an Add of a signed fact never votes on its operands.
    // `bits.c` is the corpus witness - `f(a,b) = a*b` with every signedness
    // fact inside the multiply - and a chain that flowed through `Mul` would
    // promise `int64_t` for a parameter the source declares `uint64_t`.
    let (mut f, c) = cells(1);
    let p = c[0];
    let t = trunc(&mut f, p, Width::W32);
    let five = k(&mut f, 5, Width::W32);
    cmp(&mut f, IntCmp::Lt, t, five);
    let one = k(&mut f, 1, Width::W32);
    let s = bin(&mut f, BinOp::Add, Width::W32, t, one);
    let _e = sext(&mut f, s);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "the compare fact reaches the param through casts"
    );
    assert_eq!(
        sig.params[0].width,
        Some(Width::W32),
        "bounded by every cast on the path"
    );
}

#[test]
fn a_call_borrows_the_cell_it_does_not_type_it() {
    // `f(g(a))`: the call site says nothing about `a`'s type, and the callee
    // body's facts are NOT this function's facts - borrows, never claims.
    let (mut f, c) = cells(1);
    let p = c[0];
    call(&mut f, vec![p]);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.params[0].is_silent(),
        "no callee-local fact may land on the parameter"
    );
}

// ---- float: the only way to see a float inside a W64 cell ----

#[test]
fn a_float_store_names_a_double() {
    // `movsd [rbp-8], xmm0`: the XMM cell is W64 in the IR (the ABI clobber
    // model), so `Ty` cannot say "float" - the store type is the only evidence.
    let (mut f, c) = cells(1);
    let p = c[0];
    let r = rsp(&mut f);
    let off = k(&mut f, (-8i64) as u64, Width::W64);
    let slot = bin(&mut f, BinOp::Add, Width::W64, r, off);
    store(&mut f, f64(), slot, p);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].flt,
        Some(FWidth::F64),
        "the store type is float evidence"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("double"));
    assert_eq!(
        sig.slot_conflicts, 0,
        "one view of one slot is not a conflict"
    );
}

#[test]
fn the_float_fact_crosses_a_bitcast() {
    // The movd-bridge shape: `movd eax, xmm0; addss xmm0, ...` - the float
    // claim lives on the cast dest, and the register cell is the chain.
    let (mut f, c) = cells(1);
    let p = c[0];
    let b = cast(&mut f, CastKind::Bitcast, f32(), p);
    let one = konst_f32(&mut f, 1.0);
    fbin(&mut f, FWidth::F32, b, one);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].flt,
        Some(FWidth::F32),
        "the float fact reaches the cell through the movd bridge"
    );
    assert_eq!(sig.params[0].c_type().as_deref(), Some("float"));
}

#[test]
fn a_slot_touched_as_float_and_int_is_a_union_not_a_type() {
    // The `union { int i; float f; }` shape, and the -O0 `movd` bridge. An f32
    // home read through `movd` has an int view of a float; answering "float"
    // with confidence over the bridge is the alias - the claim is withdrawn.
    let (mut f, c) = cells(2);
    let (fp, ip) = (c[0], c[1]);
    let r = rsp(&mut f);
    let off = k(&mut f, (-8i64) as u64, Width::W64);
    let slot = bin(&mut f, BinOp::Add, Width::W64, r, off);
    store(&mut f, f32(), slot, fp);
    store(&mut f, u64(), slot, ip);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.params[0].slot_conflict,
        "two views of one slot is the union signal"
    );
    assert!(
        sig.params[0].flt.is_none(),
        "no float type may be printed over the bridge"
    );
    assert_eq!(sig.slot_conflicts, 1);
}

#[test]
fn an_int_home_borrows_nothing_from_the_float_it_sits_next_to() {
    // Sparse -O0 slots: consecutive homes `rbp-4`, `rbp-8` are independent
    // variables. Disjoint-displacement keys, so no phantom union; the int write
    // proves the int's width and nothing about the float.
    let (mut f, c) = cells(2);
    let (fp, ip) = (c[0], c[1]);
    let r = rsp(&mut f);
    let o1 = k(&mut f, (-4i64) as u64, Width::W64);
    let o2 = k(&mut f, (-8i64) as u64, Width::W64);
    let a = bin(&mut f, BinOp::Add, Width::W64, r, o1);
    let b = bin(&mut f, BinOp::Add, Width::W64, r, o2);
    store(&mut f, f32(), a, fp);
    let t = trunc(&mut f, ip, Width::W32);
    store(&mut f, rdec_core::ty::Ty::uint(Width::W32), b, t);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].flt,
        Some(FWidth::F32),
        "the float keeps its own claim"
    );
    assert_eq!(
        sig.params[1].width,
        Some(Width::W32),
        "the int keeps its own"
    );
    assert!(
        sig.params[1].sx.is_none(),
        "and no signedness bleeds across stores"
    );
    assert_eq!(sig.slot_conflicts, 0, "disjoint homes are disjoint");
}

// ---- merging and void ----

#[test]
fn merged_returns_keep_only_what_every_path_proves() {
    // One path returns the value read at 32 bits, the other reads all 64: the
    // merged answer cannot promise 32, because the 64-bit path can put a bit
    // above 32 into the answer.
    let (mut f, c) = cells(1);
    let p = c[0];
    let e = f.entry;
    let cond = f.emit(
        f.entry,
        rdec_core::ir::Op::Const(1),
        rdec_core::ty::Ty::Bool,
        4,
    );
    let b1 = f.new_block(8);
    let b2 = f.new_block(16);
    let t = f.emit(
        b1,
        rdec_core::ir::Op::Cast(CastKind::Trunc, u32(), p),
        u32(),
        24,
    );
    f.block_mut(b1).term = Terminator::Return(t);
    let one = f.emit(b2, rdec_core::ir::Op::Const(1), u64(), 32);
    let d = f.emit(b2, rdec_core::ir::Op::Bin(BinOp::Sub, p, one), u64(), 36);
    f.block_mut(b2).term = Terminator::Return(d);
    f.block_mut(e).term = Terminator::Branch {
        cond,
        then_bb: b1,
        else_bb: b2,
    };
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.ret.width.is_none(),
        "the widest path decides, and W64 is not a bound"
    );
    assert!(!sig.void_return);
}

#[test]
fn a_function_with_no_value_return_is_void() {
    let (mut f, _c) = cells(1);
    ret_void(&mut f);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.void_return,
        "no Return value: the answer is void, and that is a machine fact"
    );
}

#[test]
fn conflicts_from_both_paths_land_on_the_same_value() {
    // The SAR/SHR shapes on two CFG paths of one value (shift_all's `a` is
    // reached in an `int64_t` cast tree): each path adds its own shift, and
    // the disagreement lands on the value both paths touch.
    let (mut f, c) = cells(1);
    let p = c[0];
    let e = f.entry;
    let cond = f.emit(
        f.entry,
        rdec_core::ir::Op::Const(1),
        rdec_core::ty::Ty::Bool,
        4,
    );
    let b1 = f.new_block(8);
    let b2 = f.new_block(16);
    let one1 = f.emit(b1, rdec_core::ir::Op::Const(1), u64(), 24);
    let s1 = f.emit(b1, rdec_core::ir::Op::Bin(BinOp::Sar, p, one1), u64(), 28);
    f.block_mut(b1).term = Terminator::Jump(b2);
    let two = f.emit(b2, rdec_core::ir::Op::Const(2), u64(), 32);
    let s2 = f.emit(b2, rdec_core::ir::Op::Bin(BinOp::Shr, p, two), u64(), 36);
    f.block_mut(b2).term = Terminator::Return(s2);
    f.block_mut(e).term = Terminator::Branch {
        cond,
        then_bb: b1,
        else_bb: b2,
    };
    let _ = s1;
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert!(
        sig.params[0].sx_conflict,
        "signed fact on one path, unsigned on the other"
    );
    assert!(sig.params[0].sx.is_none());
}

#[test]
fn shift_count_stays_silent() {
    // The count of a shift votes on nothing: x86 masks shift counts to 8
    // bits, so `n` is neither signed nor unsigned. Sar marks the shifted
    // value; pinning the count here is the difference between a rule and a
    // spray — an operand WideRule duty binds, unsigned-marking one's
    // neighbour does not.
    let (mut f, c) = cells(2);
    let (p, n) = (c[0], c[1]);
    bin(&mut f, BinOp::Sar, Width::W64, p, n);
    let z = k(&mut f, 0, Width::W32);
    ret(&mut f, z);
    ok(&f);
    let sig = tyinfer::recover(&f);
    assert_eq!(
        sig.params[0].sx,
        Some(Sx::Signed),
        "the shifted value is signed"
    );
    assert!(
        sig.params[1].is_silent(),
        "the count is untouched by the shift's sign"
    );
}
