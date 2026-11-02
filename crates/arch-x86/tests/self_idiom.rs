// Register self-idioms, pinned on hand-assembled machine code.
//
// `xor r,r` / `sub r,r` compute a constant 0 WITHOUT reading the register;
// `and r,r` / `or r,r` write the register's own value. These are facts about the
// encoding, so the lifter is where they belong. Modelled as an IR value identity
// they would be a change in meaning: the interpreter is strict, so an
// `xor(undef,undef)` read faults when reached, and deleting that read turns a
// faulting execution into a returning one.
//
// The tests below pin each half of that contract:
//
//   xor eax, eax; ret            - returns 0, and the IR contains NO `undef`
//                                  at all: no read of the register was emitted.
//   sub eax, eax; je/jne         - the FLAGS of the self-op are produced and
//                                  exact, so a branch on them resolves.
//   and ecx, ecx; jns            - the value is the register's own (identity),
//                                  and SF comes from it.
//   sbb eax, eax; ret            - NOT folded to 0: it is 0 only when CF=0, so
//                                  the loud gap is the honest answer.
//   xor al, al; ret              - an 8-bit self-op keeps the incoming upper
//                                  bits, so it claims no constant width.
use rdec_arch_x86::discover::{discover_all, Code};
use rdec_arch_x86::lift::lift_one;
use rdec_core::interp::{run_budget, Outcome, Val};
use rdec_core::ir::{Function, Op, Terminator};
use rdec_core::verify::verify_ssa;
use rdec_core::width::return_width;

fn lift(bytes: &[u8]) -> (Function, rdec_arch_x86::lift::Report) {
    let owned = bytes.to_vec();
    let ranges = [Code {
        base: 0,
        bytes: owned.as_slice(),
    }];
    // Seed the entry address: hand-assembled bytes have no symbol table (the
    // same shape `ret_width` uses).
    let mut seeds: std::collections::HashMap<usize, (Vec<u64>, Vec<u64>)> =
        std::collections::HashMap::new();
    seeds.insert(0, (vec![0u64], vec![]));
    let plan = discover_all(&ranges, &seeds);
    assert_eq!(plan.functions.len(), 1, "one discovered function");
    let start = plan.functions[0];
    let end = owned.len() as u64;
    lift_one("f", &owned, 0, start, end, &plan, &Default::default())
}

/// The pre-emission pipeline, as `rdec c` runs it: a surviving `Ty::Flags` value
/// is a refusal, so the flag of a self-op has to be lowered by the passes for
/// the C to exist at all.
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

/// How many times the shared `undef` value is READ here.
///
/// The lifter always places `undef` in the entry block - it is the only
/// placement that dominates every use a join can have - so "is `undef` present"
/// says nothing. The assertion that carries the claim is that nothing consumes
/// it: an `xor eax,eax` prologue must leave `undef` with no readers, because the
/// lifter never emitted the read the fold would have had to delete.
fn undef_reads(f: &Function) -> usize {
    let Some(v) = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Undef))
        .map(|i| i.dest)
    else {
        return 0;
    };
    f.use_count(v)
}

fn opaque_texts(f: &Function) -> Vec<String> {
    let mut v: Vec<String> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| match &i.op {
            Op::Opaque(t) => Some(t.to_string()),
            _ => None,
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

fn answer(f: &Function, inputs: &[u64]) -> Outcome {
    run_budget(f, inputs, 20000).outcome
}

/// `xor eax, eax; ret` - the corpus prologue. The answer is 0, and the model
/// never reads RAX to get there.
#[test]
fn xor_register_with_itself_returns_zero_without_reading_it() {
    let (f, rep) = lift(&[0x31, 0xc0, 0xc3]);
    assert_eq!(rep.self_idioms, 1, "one self-idiom, counted");
    assert_eq!(rep.gaps, 0, "nothing here is unmodelled");
    assert_eq!(verify_ssa(&f), Ok(()));
    assert_eq!(
        answer(&f, &[]),
        Outcome::Ret(Val::Int(0)),
        "the machine returns RAX = 0\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(
        undef_reads(&f),
        0,
        "the lifter emitted no read of the undefined register\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // A provably-zero answer needs ZERO bits, and `return_width` claims a width
    // only when the answer is a value whose width it can name: 0 bits is not
    // "32 bits wide", it is "the constant 0". So the return type stays int64_t,
    // and `return 0;` matches the machine's RAX = 0 bit for bit.
    assert_eq!(
        return_width(&f),
        64,
        "a constant 0 claims no width, which is sound: int64_t 0 == RAX 0"
    );

    // And it survives the pipeline into C: after the passes the returned value
    // IS the constant 0, and the function emits - no `undef` read, no refusal.
    let mut g = f.clone();
    clean(&mut g);
    assert_eq!(verify_ssa(&g), Ok(()), "the passes keep it valid IR");
    let ret = g
        .blocks
        .iter()
        .find_map(|b| match &b.term {
            Terminator::Return(v) => Some(*v),
            _ => None,
        })
        .expect("a return");
    let (bb, ix) = g.def(ret).expect("the returned value is defined");
    assert!(
        matches!(g.block(bb).instrs[ix].op, Op::Const(0)),
        "the passes leave the answer as the constant 0, not a read: {:?}\n{}",
        g.block(bb).instrs[ix].op,
        rdec_core::fmt_ir::format_function(&g)
    );
    let out = rdec_core::emit_c::emit_c(&g).expect("the prologue emits C, not a refusal");
    assert!(
        out.externs.is_empty() && out.text.contains("return"),
        "the emitted C is self-contained and returns the constant:\n{}",
        out.text
    );
}

/// `xor rax, rax; ret` at the full 64-bit width: same fact, no read.
#[test]
fn xor_full_width_register_with_itself_returns_zero() {
    let (f, rep) = lift(&[0x48, 0x31, 0xc0, 0xc3]);
    assert_eq!(rep.self_idioms, 1);
    assert_eq!(rep.gaps, 0);
    assert_eq!(undef_reads(&f), 0);
    assert_eq!(answer(&f, &[]), Outcome::Ret(Val::Int(0)));
}

/// `sub eax, eax` writes 0 AND sets ZF, so both spellings of the branch on it
/// resolve - the flags value is still there, and its bits are the hardware's.
#[test]
fn sub_register_with_itself_leaves_flags_a_jz_can_read() {
    //   0: 29 c0           sub eax, eax
    //   2: 74 05           je  0x9          <- ZF=1, so taken
    //   4: b8 07 00 00 00  mov eax, 7       <- skipped
    //   9: c3              ret
    let taken = lift(&[0x29, 0xc0, 0x74, 0x05, 0xb8, 0x07, 0, 0, 0, 0xc3]);
    assert_eq!(taken.1.self_idioms, 1);
    assert_eq!(
        taken.1.gaps,
        0,
        "the branch's flags are defined: no gap\n{}",
        rdec_core::fmt_ir::format_function(&taken.0)
    );
    assert_eq!(taken.1.exits, 0, "the branch landed on a block");
    assert_eq!(
        answer(&taken.0, &[]),
        Outcome::Ret(Val::Int(0)),
        "ZF=1, so `je` is taken and 7 is never written"
    );

    // Same bytes, `jne`: not taken, so the fallthrough's 7 is the answer. A
    // model that set ZF=0 flips these two answers, which is why both are here.
    //   0: 29 c0           sub eax, eax
    //   2: 75 05           jne 0x9          <- ZF=1, so NOT taken
    //   4: b8 07 00 00 00  mov eax, 7
    //   9: c3              ret
    let (f, rep) = lift(&[0x29, 0xc0, 0x75, 0x05, 0xb8, 0x07, 0, 0, 0, 0xc3]);
    assert_eq!(rep.self_idioms, 1);
    assert_eq!(rep.gaps, 0);
    assert_eq!(
        answer(&f, &[]),
        Outcome::Ret(Val::Int(7)),
        "ZF=1, so `jne` falls through to the 7"
    );
}

/// `and eax, eax` / `or eax, eax` read the register and write its own value
/// back, and their flags are the value's own signs and zeros. Both halves are
/// pinned: the value (a constant fold would return 0 here) and the flag.
#[test]
fn and_or_with_themselves_keep_the_value_and_the_flags_of_that_value() {
    // Identity: `mov eax, ecx; and eax, eax; ret` - the answer is the argument.
    for and_or in [0x21u8, 0x09] {
        // 21 = AND eax,eax, 09 = OR eax,eax
        let (f, rep) = lift(&[0x89, 0xc8, and_or, 0xc0, 0xc3]);
        assert_eq!(rep.self_idioms, 1, "{and_or:#x} is a self-idiom");
        assert_eq!(rep.gaps, 0);
        assert_eq!(
            answer(&f, &[0x0000_0005]),
            Outcome::Ret(Val::Int(5)),
            "{and_or:#x} r,r leaves the register's own value, not a constant"
        );
        assert_eq!(
            answer(&f, &[0xffff_ffff]),
            Outcome::Ret(Val::Int(0xffff_ffff)),
            "{and_or:#x} r,r leaves the register's own value, not a constant"
        );
    }

    // SF: `mov eax, ecx; and eax, eax; js 0xc; mov eax,7; ret; mov eax,2; ret`
    // The branch direction is the self-op's SF, which only the register's own
    // sign bit can set.
    //   0: 89 c8           mov eax, ecx
    //   2: 21 c0           and eax, eax
    //   4: 78 06           js  0xc          <- SF=1 takes the jump
    //   6: b8 07 00 00 00  mov eax, 7       <- the SF=0 path
    //  11: c3              ret
    //  12: b8 02 00 00 00  mov eax, 2       <- the SF=1 path
    //  17: c3              ret
    let (f, rep) = lift(&[
        0x89, 0xc8, 0x21, 0xc0, 0x78, 0x06, 0xb8, 0x07, 0, 0, 0, 0xc3, 0xb8, 0x02, 0, 0, 0, 0xc3,
    ]);
    assert_eq!(rep.self_idioms, 1);
    assert_eq!(
        rep.gaps,
        0,
        "the self-op's flags are defined\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    assert_eq!(rep.exits, 0, "the branch landed on a block, not an Exit");
    assert_eq!(
        answer(&f, &[5]),
        Outcome::Ret(Val::Int(7)),
        "SF=0 for a positive value: the `js` is not taken"
    );
    assert_eq!(
        answer(&f, &[0x8000_0000]),
        Outcome::Ret(Val::Int(2)),
        "SF=1 for a negative value: the `js` is taken"
    );

    // ZF: `mov eax, ecx; or eax, eax; je 0xc; mov eax,7; ret; mov eax,3; ret`
    //   0: 89 c8           mov eax, ecx
    //   2: 09 c0           or eax, eax
    //   4: 74 06           je  0xc          <- ZF=1 (the argument was 0)
    //   6: b8 07 00 00 00  mov eax, 7
    //  11: c3              ret
    //  12: b8 03 00 00 00  mov eax, 3
    //  17: c3              ret
    let (g, grep) = lift(&[
        0x89, 0xc8, 0x09, 0xc0, 0x74, 0x06, 0xb8, 0x07, 0, 0, 0, 0xc3, 0xb8, 0x03, 0, 0, 0, 0xc3,
    ]);
    assert_eq!(grep.self_idioms, 1);
    assert_eq!(grep.gaps, 0, "the self-op's flags are defined");
    assert_eq!(
        answer(&g, &[0]),
        Outcome::Ret(Val::Int(3)),
        "ZF=1 for a zero argument: the `je` is taken"
    );
    assert_eq!(
        answer(&g, &[0x42]),
        Outcome::Ret(Val::Int(7)),
        "ZF=0 for a non-zero argument: the `je` is not taken"
    );
}

/// `sbb eax, eax` is 0 ONLY when CF=0. Folding it to 0 would be a value identity
/// that is not true, so the loud named gap is the honest answer and stays.
#[test]
fn sbb_register_with_itself_stays_a_named_gap() {
    let (f, rep) = lift(&[0x19, 0xc0, 0xc3]); // sbb eax, eax; ret
    assert_eq!(
        rep.self_idioms, 0,
        "sbb is not in the self-idiom set: its result depends on CF"
    );
    assert!(rep.gaps > 0, "the carry-in is still a loud gap");
    assert!(
        opaque_texts(&f)
            .iter()
            .any(|t| t.contains("sbb") && t.contains("carry")),
        "the gap names the carry chain, not a generic refusal: {:?}",
        opaque_texts(&f)
    );
}

/// `xor al, al` is the self-idiom at 8-bit width: x86's partial-write rule keeps
/// bits 63:8 of RAX, which came in from the caller. So the answer is NOT the
/// constant 0, and the model claims no narrow return width.
#[test]
fn an_8_bit_self_idiom_keeps_the_incoming_upper_bits() {
    let (f, rep) = lift(&[0x30, 0xc0, 0xc3]); // xor al, al; ret
    assert_eq!(rep.self_idioms, 1);
    assert_eq!(
        return_width(&f),
        64,
        "bits 63:8 are the caller's, so nothing is proved narrow\n{}",
        rdec_core::fmt_ir::format_function(&f)
    );
    // The register is read for its upper half: that read is real, and it is a
    // live-in whose incoming value no path defines - the ABI question, counted.
    assert_eq!(rep.undef_live_ins, 1, "the kept bits are an ABI question");
}
