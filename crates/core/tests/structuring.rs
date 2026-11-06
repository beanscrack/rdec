//! M5: control-flow structuring, on hand-built IR.
//!
//! Every fixture here is a `Function` assembled by hand - no compiler, no
//! corpus, no object file - so each test names the CFG shape it is about and the
//! C it must (or must not) produce. The emitted C is quoted in every assertion
//! message: a failure has to be readable without a debugger.
//!
//! The claims:
//!
//! 1. a single-if region emits `if (...)` with NO `goto` to the join, and the
//!    block bodies survive;
//! 2. an if/else emits both arms with the polarity the machine code tested - the
//!    condition is the compared value, not its negation, and the arms are not
//!    swapped (the classic structuring miscompile);
//! 3. a pre-test loop emits `while (...)`, and the interpreter answers the
//!    ZERO-ITERATION case, which is what a do-while/while swap gets wrong;
//! 4. ANTI-INVENTION: a region the patterns cannot prove (irreducible,
//!    multi-entry) stays label + `goto` and SAYS SO in the census - no `if` is
//!    invented;
//! 5. two guards merge into `a && b` only when the second block computes nothing
//!    the machine code would not run on that arm;
//! 6. every fixture that uses `Op::Param` sets `f.params`, so the emitted C binds
//!    real arguments: a fixture that binds nothing agrees with any bug.
use rdec_core::emit_c::{emit_c, text_census};
use rdec_core::interp::{run, Outcome, Val};
use rdec_core::ir::{BinOp, Function, Instr, IntCmp, Op, Terminator};
use rdec_core::structure::{flat_walk, structure_with_params, sum_reports, Report, StmtKind};
use rdec_core::ty::Ty;
use rdec_core::verify::verify_ssa;

fn u64_() -> Ty {
    Ty::u64()
}

/// The C name the emitter gives a value: `x{n}` (see `Emitter::name`).
fn nm(v: rdec_core::ir::VId) -> String {
    format!("x{v}")
}

/// Emit C for a hand-built fixture, quoting the IR in the panic message.
fn c(f: &Function) -> String {
    verify_ssa(f).unwrap_or_else(|e| {
        panic!(
            "fixture is not valid SSA: {e:?}\n{}",
            rdec_core::fmt_ir::format_function(f)
        )
    });
    match emit_c(f) {
        Ok(o) => o.text,
        Err(r) => panic!(
            "fixture refused: {:?}\n{}",
            r.first().map(|x| &x.reason),
            rdec_core::fmt_ir::format_function(f)
        ),
    }
}

/// Assert the emitted C satisfies `pred`, quoting the C when it does not.
fn wants(text: &str, pred: impl Fn(&str) -> bool, what: &str) {
    assert!(
        pred(text),
        "the emitted C must {what}\n---- emitted C ----\n{text}\n------------------"
    );
}

fn line_of(text: &str, needle: &str) -> usize {
    text.lines()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing {needle:?}\n---- emitted C ----\n{text}"))
}

fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

/// The control-flow CONDITION lines the structuring pass prints: the `if (...) {`
/// and `while (...) {` heads. A merge/polarity claim is a claim about a
/// condition, so it is checked on the condition lines: the whole-file grep is
/// taken by the emitter's own `(void)&&lN;` label uses, which carry `&&` in
/// every function that has a label and a parameter, structured or not.
fn cond_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|l| (l.starts_with("if (") || l.starts_with("while (")) && l.ends_with('{'))
        .collect()
}

/// The text of the add this function computes, as the emitter spells an
/// addition: `x{a} + x{b}` inside its parentheses. The emitter wraps the result
/// in the destination's width cast (`x5 = (uint64_t)(x0 + x4);`), so a needle
/// that starts at the `=` has to name the cast; a needle that names the
/// operands and their order is the claim, and survives the emitter's spelling.
fn add_text(f: &Function) -> String {
    let ins = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Bin(BinOp::Add, _, _)))
        .expect("the fixture computes an add");
    match ins.op {
        Op::Bin(BinOp::Add, a, b) => format!("({} + {})", nm(a), nm(b)),
        _ => unreachable!("the find above only returns an add"),
    }
}

/// The text of the compare this function's guard tests, spelled exactly as the
/// emitter spells `Op::Cmp`: `(x{a} < x{b}) ? 1 : 0`.
fn cmp_text(f: &Function) -> String {
    let ins = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Cmp(..)))
        .expect("the fixture compares something");
    match ins.op {
        Op::Cmp(k, a, b) => format!("({} {} {}) ? 1 : 0", nm(a), k.c_op(), nm(b)),
        _ => unreachable!("the find above only returns a compare"),
    }
}

fn structured_stmt(text: &str, kw: &str) -> bool {
    text.lines().any(|l| {
        let l = l.trim();
        l.ends_with('{') && l.starts_with(kw) && l.contains('(')
    })
}

/// A two-parameter function: `f(p0, p1)`.
///
/// `f.params` is set, and that is the point (claim 6): the emitter's argument
/// loads read `a0`/`a1`, so a fixture that left `params` empty would emit a body
/// whose every read of `p0` is an unwritten local - a shape that agrees with any
/// bug the structuring pass can make.
fn fn2(name: &str) -> Function {
    let mut f = Function::new(name, 0x1000);
    let e = f.entry;
    let p0 = f.emit(e, Op::Param, u64_(), 0x1000);
    let p1 = f.emit(e, Op::Param, u64_(), 0x1004);
    f.params = vec![p0, p1];
    f
}

/// Emit and return the C plus the census the gate prints for it - the emitter's
/// own report, which is the one that must agree with the text (a tree the emitter
/// rejected is not what `rdec c` reports).
fn emit(f: &Function) -> (String, Report) {
    let text = c(f);
    let rep = emit_c(f).expect("the fixture emits").structuring;
    (text, rep)
}

fn assert_binds_arguments(text: &str) {
    wants(
        text,
        |t| t.contains("(uint64_t)a0") && t.contains("(uint64_t)a1"),
        "bind a0 and a1 into the entry's parameter values",
    );
}

fn answer(f: &Function, inputs: &[u64]) -> u64 {
    let e = run(f, inputs);
    match &e.outcome {
        Outcome::Ret(Val::Int(v)) => *v,
        other => panic!("the fixture must return an integer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1. single if
// ---------------------------------------------------------------------------

/// `if (p0 == 0) { p0 += 7 }` and fall through to `return`.
///
/// ```text
/// e: c = (p0 == 0); if c goto t else j
/// t: s = p0 + 7;     jmp j
/// j: phi(s@t, p0@e); return
/// ```
fn single_if() -> Function {
    let mut f = fn2("single_if");
    let (e, p0) = (f.entry, f.params[0]);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let ct = f.emit(e, Op::Cmp(IntCmp::Eq, p0, zero), Ty::Bool, 0x100c);
    let t = f.new_block(0x1010);
    let j = f.new_block(0x1020);
    f.block_mut(e).term = Terminator::Branch {
        cond: ct,
        then_bb: t,
        else_bb: j,
    };
    let seven = f.emit(t, Op::Const(7), u64_(), 0x1010);
    let s = f.emit(t, Op::Bin(BinOp::Add, p0, seven), u64_(), 0x1014);
    f.block_mut(t).term = Terminator::Jump(j);
    let r = f.new_value(u64_());
    f.block_mut(j).instrs.push(Instr::new(
        r,
        Op::Phi(vec![(s, t), (p0, e)].into_boxed_slice()),
        u64_(),
        0x1020,
    ));
    f.block_mut(j).term = Terminator::Return(r);
    f
}

#[test]
fn a_single_if_emits_an_if_and_the_join_needs_no_goto() {
    let f = single_if();
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);

    wants(&text, |t| count(t, "if (") == 1, "emit exactly one `if`");
    wants(
        &text,
        |t| !t.contains("goto l"),
        "be free of `goto`: the untaken arm is the join, which the walk emits next, and the taken arm falls into it",
    );
    let tested = nm(f.blocks[f.entry as usize]
        .instrs
        .iter()
        .find(|i| matches!(i.op, Op::Cmp(..)))
        .expect("the entry tests something")
        .dest);
    let guard = line_of(&text, "(x0 == x2) ? 1 : 0");
    let head = line_of(&text, &format!("if ({tested})"));
    let body = line_of(&text, &add_text(&f));
    assert!(
        guard < head && head < body,
        "the compare must precede the `if`, and the body must be inside it\n---- emitted C ----\n{text}\n------------------"
    );
    wants(
        &text,
        |t| t.contains("x6 = x0;") && t.contains("x6 = x5;"),
        "keep the join's phi as edge copies - the value each arm hands it: `x6 = x0` on the skipped arm, `x6 = x5` at the end of the tested arm",
    );
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (1, 0, 0),
        "the census of this function is one `if`, no `else`, no loop\n---- emitted C ----\n{text}\n------------------"
    );

    for a in [0u64, 1, 7, u64::MAX] {
        assert_eq!(
            answer(&f, &[a, 3]),
            if a == 0 { a + 7 } else { a },
            "f({a})"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. if / else, with the machine code's polarity
// ---------------------------------------------------------------------------

/// `return p0 < p1 ? 11 : 22;`
fn if_else() -> Function {
    let mut f = fn2("if_else");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let ct = f.emit(e, Op::Cmp(IntCmp::Ult, p0, p1), Ty::Bool, 0x1008);
    let t = f.new_block(0x1010);
    let el = f.new_block(0x1020);
    let j = f.new_block(0x1030);
    f.block_mut(e).term = Terminator::Branch {
        cond: ct,
        then_bb: t,
        else_bb: el,
    };
    let eleven = f.emit(t, Op::Const(11), u64_(), 0x1010);
    f.block_mut(t).term = Terminator::Jump(j);
    let twenty = f.emit(el, Op::Const(22), u64_(), 0x1020);
    f.block_mut(el).term = Terminator::Jump(j);
    let r = f.new_value(u64_());
    f.block_mut(j).instrs.push(Instr::new(
        r,
        Op::Phi(vec![(eleven, t), (twenty, el)].into_boxed_slice()),
        u64_(),
        0x1030,
    ));
    f.block_mut(j).term = Terminator::Return(r);
    f
}

#[test]
fn an_if_else_keeps_the_condition_the_machine_code_tested() {
    let f = if_else();
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);

    // The value the branch tested, and the predicate the machine code used.
    let tested = f
        .blocks
        .iter()
        .find_map(|b| match b.term {
            Terminator::Branch { cond, .. } => Some(cond),
            _ => None,
        })
        .expect("the fixture branches");
    let k = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| i.dest == tested)
        .and_then(|i| match i.op {
            Op::Cmp(k, a, b) => Some((k, a, b)),
            _ => None,
        })
        .expect("the tested value is a compare");
    assert_eq!(k.0, IntCmp::Ult, "the fixture tests an unsigned <");
    assert_eq!(
        (k.1, k.2),
        (f.params[0], f.params[1]),
        "of p0 and p1, in that order"
    );

    wants(&text, |t| count(t, "if (") == 1, "emit one `if`");
    wants(&text, |t| t.contains("} else {"), "emit the `else` arm");
    wants(
        &text,
        |t| t.contains("(x0 < x1) ? 1 : 0"),
        "spell the compare the machine code performed - `p0 < p1`. Its negation with the arms in place is the polarity miscompile this test exists to catch",
    );
    wants(
        &text,
        |t| !t.contains("if (!") && !t.contains(">="),
        "test the compared value un-negated: `if (x)` where x is `!cmp`, or an inverted compare, is a miscompile",
    );
    wants(
        &text,
        |t| !t.contains("goto l"),
        "need no `goto`: both arms land on the join, which the walk emits next",
    );

    // The arms, in the machine code's order: the tested-true arm is `then`.
    let head = line_of(&text, "if (");
    let then_arm = line_of(&text, "0xbULL");
    let els = line_of(&text, "} else {");
    let else_arm = line_of(&text, "0x16ULL");
    assert!(
        head < then_arm && then_arm < els && els < else_arm,
        "the tested-true arm must be the `then` arm: 11 before `else`, 22 after it\n---- emitted C ----\n{text}\n------------------"
    );

    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (0, 1, 0),
        "the census of this function is one if/else\n---- emitted C ----\n{text}\n------------------"
    );

    for (a, b) in [(1u64, 2u64), (2, 1), (5, 5), (0, u64::MAX)] {
        assert_eq!(
            answer(&f, &[a, b]),
            if a < b { 11 } else { 22 },
            "f({a},{b})"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. pre-test loop
// ---------------------------------------------------------------------------

/// ```text
/// e: zero = 0; jmp h
/// h: kp = phi(zero@e, k2@l); c = (kp < p1); if c goto l else d
/// l: k2 = kp + 1; jmp h
/// d: return kp
/// ```
///
/// The test lives in the header, so `p1 == 0` runs the body ZERO times. A
/// do-while answers 1 for that input; the interpreter reading the structured
/// statements, and the C printed from them, must answer 0.
fn count_up() -> Function {
    let mut f = fn2("count_up");
    let (e, p1) = (f.entry, f.params[1]);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let h = f.new_block(0x1010);
    let l = f.new_block(0x1020);
    let d = f.new_block(0x1030);
    f.block_mut(e).term = Terminator::Jump(h);
    let kp = f.new_value(u64_());
    let ct = f.emit(h, Op::Cmp(IntCmp::Ult, kp, p1), Ty::Bool, 0x1014);
    f.block_mut(h).term = Terminator::Branch {
        cond: ct,
        then_bb: l,
        else_bb: d,
    };
    let one = f.emit(l, Op::Const(1), u64_(), 0x1020);
    let k2 = f.emit(l, Op::Bin(BinOp::Add, kp, one), u64_(), 0x1024);
    f.block_mut(l).term = Terminator::Jump(h);
    f.block_mut(d).term = Terminator::Return(kp);
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            kp,
            Op::Phi(vec![(zero, e), (k2, l)].into_boxed_slice()),
            u64_(),
            0x1010,
        ),
    );
    f
}

#[test]
fn a_pre_test_loop_is_a_while_that_runs_zero_times_when_the_guard_fails() {
    let f = count_up();
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);

    wants(&text, |t| count(t, "while (") == 1, "emit one `while`");
    wants(
        &text,
        |t| !t.contains("do {") && !t.contains("} while"),
        "not be a do-while: the test is in the header, so it runs BEFORE the body",
    );
    wants(
        &text,
        |t| !t.contains("goto l"),
        "need no `goto`: the `while` text is the back edge",
    );
    // The polarity, pinned: this fixture's body IS the guard's `then` arm, so
    // the machine runs the body while the tested value is true and the C
    // condition is that value with no `!`. The other way round - the value
    // negated, or the arms swapped - runs the body on the trip the machine code
    // skips, and the printed C looks exactly as tidy doing it.
    wants(
        &text,
        |t| !cond_lines(t).iter().any(|l| l.contains('!')),
        "keep the machine code's polarity on the loop: `while (x)` where x is the value the guard tested. A `!` on the condition means the body was moved to the arm the machine code skips",
    );
    let head = line_of(&text, "while (");
    let body = line_of(&text, &add_text(&f));
    assert!(
        head < body,
        "the test must precede the body in the text\n---- emitted C ----\n{text}\n------------------"
    );
    // THE ROTATION. `while (c)` re-tests `c` before every trip, and `c` is a
    // value the header's instructions COMPUTE, so the header's text - the
    // compare - must appear TWICE: once above the loop (the trip in from the
    // region's entry) and once at the BOTTOM of the body (the back edge).
    // Printing it once computes the tested value once and the C never ends.
    // The first cut of this pass did exactly that, and `t2 --corpus` lost 9
    // comparisons to C-side timeouts while the IR interpreter - which reads the
    // graph, not the C - went on answering every vector correctly. A claim
    // about the printed loop needs a claim about the printed loop.
    let tested = cmp_text(&f);
    let copies: Vec<usize> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(&tested))
        .map(|(i, _)| i)
        .collect();
    assert!(
        copies.len() >= 2,
        "the guard's compare `{tested}` appears {} time(s): a `while` whose test is computed once is an infinite loop, so the header's text must be spelled on the entry edge AND on the back edge\n---- emitted C ----\n{text}\n------------------",
        copies.len()
    );
    assert!(
        *copies.last().expect("checked non-empty") > head,
        "the LAST copy of the guard's compare must sit INSIDE the loop body (it is the back edge), not above the `while`\n---- emitted C ----\n{text}\n------------------"
    );
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (0, 0, 1),
        "the census of this function is one loop\n---- emitted C ----\n{text}\n------------------"
    );

    // THE assertion a do-while/while swap fails: guard false on entry, so the
    // body never runs and the answer is the initialiser.
    assert_eq!(
        answer(&f, &[7, 0]),
        0,
        "guard false on entry: ZERO iterations"
    );
    assert_eq!(answer(&f, &[7, 1]), 1, "guard true once: one iteration");
    assert_eq!(
        answer(&f, &[7, 5]),
        5,
        "guard true five times: five iterations"
    );
}

// ---------------------------------------------------------------------------
// 4. anti-invention
// ---------------------------------------------------------------------------

/// The same pre-test loop as `count_up`, with the guard's sense turned round so
/// the BODY is the `else` arm: the machine code tests `5 < kp` and leaves on the
/// `then` arm. `while` must print the tested value negated, `while (!(x))`, and
/// must NOT swap the arms: an un-negated condition here runs the body on the
/// trip the machine code skips, which is the same bug with its hat on.
///
/// ```text
/// e: zero = 0; jmp h
/// h: kp = phi(zero@e, k2@l); c = (5 < kp); if c goto d else l
/// l: k2 = kp + 1; jmp h
/// d: return kp
/// ```
fn count_up_else_guard() -> Function {
    let mut f = fn2("count_up_else");
    let (e, p1) = (f.entry, f.params[1]);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let h = f.new_block(0x1010);
    let l = f.new_block(0x1020);
    let d = f.new_block(0x1030);
    f.block_mut(e).term = Terminator::Jump(h);
    let kp = f.new_value(u64_());
    let five = f.emit(h, Op::Const(5), u64_(), 0x1010);
    let ct = f.emit(h, Op::Cmp(IntCmp::Ult, five, kp), Ty::Bool, 0x1014);
    f.block_mut(h).term = Terminator::Branch {
        cond: ct,
        then_bb: d,
        else_bb: l,
    };
    let one = f.emit(l, Op::Const(1), u64_(), 0x1020);
    let k2 = f.emit(l, Op::Bin(BinOp::Add, kp, one), u64_(), 0x1024);
    f.block_mut(l).term = Terminator::Jump(h);
    f.block_mut(d).term = Terminator::Return(kp);
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            kp,
            Op::Phi(vec![(zero, e), (k2, l)].into_boxed_slice()),
            u64_(),
            0x1010,
        ),
    );
    let _ = p1;
    f
}

/// Two arms of a guard that converge on a block NEITHER ARM OWNS, below the
/// join: the arms share `q`, so `q` is a multi-entry region and no `if`/`else`
/// pair expresses it. Inside a region whose exit is `j`, the shape is:
///
/// ```text
/// e: if (p0 == 0) goto g else j
/// g: if (p1 == 0) goto t else f        <- the nested guard
/// t: jmp q
/// f: jmp q                              <- both arms own an edge into q
/// q: if (p0 < p1) goto j else k
/// k: jmp j
/// j: return p1
/// ```
///
/// `q` has two predecessors from two different arms: `while`, `if` and `if/else`
/// each have one door, so the nested guard must stay a label and a `goto`.
fn arms_that_converge_below_the_join() -> Function {
    let mut f = fn2("converge_below");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let g = f.new_block(0x1010);
    let t = f.new_block(0x1020);
    let fl = f.new_block(0x1030);
    let q = f.new_block(0x1040);
    let k = f.new_block(0x1050);
    let j = f.new_block(0x1060);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let c0 = f.emit(e, Op::Cmp(IntCmp::Eq, p0, zero), Ty::Bool, 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: c0,
        then_bb: g,
        else_bb: j,
    };
    let c1 = f.emit(g, Op::Cmp(IntCmp::Eq, p1, zero), Ty::Bool, 0x1010);
    f.block_mut(g).term = Terminator::Branch {
        cond: c1,
        then_bb: t,
        else_bb: fl,
    };
    f.block_mut(t).term = Terminator::Jump(q);
    f.block_mut(fl).term = Terminator::Jump(q);
    let c2 = f.emit(q, Op::Cmp(IntCmp::Ult, p0, p1), Ty::Bool, 0x1040);
    f.block_mut(q).term = Terminator::Branch {
        cond: c2,
        then_bb: j,
        else_bb: k,
    };
    f.block_mut(k).term = Terminator::Jump(j);
    f.block_mut(j).term = Terminator::Return(p1);
    f
}

#[test]
fn a_loop_whose_body_is_the_else_arm_prints_the_tested_value_negated() {
    let f = count_up_else_guard();
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);
    wants(&text, |t| count(t, "while (") == 1, "emit one `while`");
    // The polarity, pinned: the body is the `else` arm of a test of `5 < kp`,
    // so the condition is that value spelled with `!`. An un-negated condition
    // is the arm swap in disguise - it runs the body on the trip the machine
    // code takes straight to the exit.
    let tested = nm(f
        .blocks
        .iter()
        .find_map(|b| match b.term {
            Terminator::Branch { cond, .. } => Some(cond),
            _ => None,
        })
        .expect("the fixture branches"));
    wants(
        &text,
        |t| {
            cond_lines(t)
                .iter()
                .any(|l| l.starts_with("while (!(") && l.contains(&tested))
        },
        "spell the `while`'s condition as the negation of the tested value: `while (!(x))` with x the value the branch tested",
    );
    wants(
        &text,
        |t| !t.contains("do {") && !t.contains("} while"),
        "not be a do-while: the test is in the header",
    );
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (0, 0, 1),
        "the census of this function is one loop\n---- emitted C ----\n{text}\n------------------"
    );
    // Zero trips (kp starts at 6: `5 < 6` is true, the machine leaves at once)
    // and six trips (kp starts at 0): the answers the negation is what makes
    // right, and the answers a flipped polarity gets wrong.
    assert_eq!(
        answer(&f, &[7, 6]),
        6,
        "guard true on entry: ZERO iterations"
    );
    assert_eq!(answer(&f, &[7, 0]), 6, "guard false for six trips");
}

#[test]
fn arms_that_converge_on_a_block_neither_arm_owns_are_not_an_if_else() {
    let f = arms_that_converge_below_the_join();
    let (text, rep) = emit(&f);
    let tree = structure_with_params(&f, true);
    assert_binds_arguments(&text);
    // The OUTER guard is a single-if: its skipped arm is the join `j`, and its
    // running arm is the region {g, t, f, q, k}, all of which land on `j`.
    // The NESTED guard is not: `q` has one predecessor in each arm, and no C
    // statement has two doors.
    assert!(
        rep.if_else == 0,
        "no if/else may be invented for two arms that share a block below the join\nreasons: {:?}\n---- emitted C ----\n{text}\n------------------",
        rep.reasons
    );
    assert_eq!(
        tree.report.if_else, 0,
        "and the walk proposes no `if/else` either - a proposal the label net drops is still a proposal\nreasons: {:?}\nstmts {:#?}",
        tree.report.reasons,
        tree.stmts
    );
    wants(
        &text,
        |t| !structured_stmt(t, "} else {") && !t.contains("} else {"),
        "print no `else` arm: the arms do not close, so an `else` would be an invention",
    );
    assert!(
        rep.gotos > 0 && text.contains("goto l"),
        "the un-provable shape keeps its label + `goto`: {}\n---- emitted C ----\n{text}\n------------------",
        rep.gotos
    );
    // And the census is the TEXT's: the emitter fell back to the flat form
    // because this tree's flat region names labels its text never prints, and
    // that fallback is counted, never silent.
    let fell_back = rep
        .reasons
        .get("region whose text left a named label undefined (flat form)")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        fell_back, 1,
        "the tree this CFG proposes leaves a named label undefined, and the flat fallback is counted in the census (the walk proposed ifs {} if_else {} whiles {})\nreasons: {:?}\n---- emitted C ----\n{text}\n------------------",
        tree.report.ifs,
        tree.report.if_else,
        tree.report.whiles,
        rep.reasons
    );
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (0, 0, 0),
        "the printed census is the flat form's: nothing is structured\n---- emitted C ----\n{text}\n------------------"
    );
    let tc = text_census(&text);
    assert_eq!(
        (tc.gotos, tc.labels, tc.ifs, tc.els, tc.whiles),
        (
            rep.gotos,
            rep.labels,
            rep.ifs + rep.if_else,
            rep.if_else,
            rep.whiles
        ),
        "the text scan is the printed census\n---- emitted C ----\n{text}\n------------------"
    );
    for (a, b) in [(0u64, 0), (1, 2), (3, 3), (u64::MAX, 1)] {
        assert_eq!(answer(&f, &[a, b]), b, "f({a},{b}) is p1 on every path");
    }
}

/// Every `goto lN;` in the text names an `lN:` the text defines. The C-side
/// version of [`rdec_core::structure::Structured::labels_defined`]: clang
/// answers `use of undeclared label` to a text that fails it, so the emitted
/// file must never contain one.
fn goto_targets_defined(text: &str) -> Vec<String> {
    let defined: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter_map(|l| {
            let rest = l.strip_prefix('l')?;
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            (!digits.is_empty() && rest[digits.len()..].starts_with(':'))
                .then(|| digits.parse::<u32>().expect("digits").to_string())
        })
        .collect();
    let mut bad = Vec::new();
    for l in text.lines() {
        let s = l.trim();
        if let Some(rest) = s.strip_prefix("goto l") {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !defined.contains(&digits) {
                bad.push(format!("{s}  (no `l{digits}:` in the text)"));
            }
        }
    }
    bad
}

// ---------------------------------------------------------------------------
// 4b. the flat fallback: a tree that names an undefined label is never printed
// ---------------------------------------------------------------------------

#[test]
fn the_emitted_text_never_names_a_label_it_does_not_define() {
    for (name, f) in fixtures() {
        let text = c(&f);
        let bad = goto_targets_defined(&text);
        assert!(
            bad.is_empty(),
            "{name}: the emitted C jumps to labels it never defines: {bad:?}\n---- emitted C ----\n{text}\n------------------"
        );
        // The `(void)&&lN;` uses the emitter prints for `-Wunused-label` must
        // name the same set: a use with no label line is a clang error.
        let defined: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter_map(|l| {
                let rest = l.strip_prefix('l')?;
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                (!digits.is_empty() && rest[digits.len()..].starts_with(':'))
                    .then(|| digits.clone())
            })
            .collect();
        for l in text.lines() {
            let s = l.trim();
            if let Some(rest) = s.strip_prefix("(void)&&l") {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                assert!(
                    defined.contains(&digits),
                    "{name}: `(void)&&l{digits};` names a label the text does not define\n---- emitted C ----\n{text}\n------------------"
                );
            }
        }
    }
}

/// The irreducible pair: two guards that branch to each other, so neither block is
/// a region entry and no C statement covers them.
fn irreducible() -> Function {
    let mut f = fn2("irreducible");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let x = f.new_block(0x1010);
    let y = f.new_block(0x1020);
    let z = f.new_block(0x1030);
    let one = f.emit(e, Op::Const(1), u64_(), 0x1008);
    let a = f.emit(e, Op::Cmp(IntCmp::Eq, p0, one), Ty::Bool, 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: a,
        then_bb: x,
        else_bb: y,
    };
    let two = f.emit(x, Op::Const(2), u64_(), 0x1010);
    let b = f.emit(x, Op::Cmp(IntCmp::Eq, p1, two), Ty::Bool, 0x1014);
    f.block_mut(x).term = Terminator::Branch {
        cond: b,
        then_bb: y,
        else_bb: z,
    };
    let three = f.emit(y, Op::Const(3), u64_(), 0x1020);
    let cc = f.emit(y, Op::Cmp(IntCmp::Eq, p0, three), Ty::Bool, 0x1024);
    f.block_mut(y).term = Terminator::Branch {
        cond: cc,
        then_bb: x,
        else_bb: z,
    };
    let zero = f.emit(z, Op::Const(0), u64_(), 0x1030);
    f.block_mut(z).term = Terminator::Return(zero);
    f
}

#[test]
fn an_irreducible_region_stays_labels_and_gotos_and_says_so() {
    let f = irreducible();
    let (text, rep) = emit(&f);
    let tree = structure_with_params(&f, true);
    assert_binds_arguments(&text);

    // NO structure invented: not one `if (...) {`, not one `while (...) {`.
    wants(
        &text,
        |t| !structured_stmt(t, "if (") && !structured_stmt(t, "while ("),
        "contain no structured `if` or `while`: this CFG is irreducible, so an `if` here would be an invention",
    );
    wants(
        &text,
        |t| t.contains("goto l") && t.contains(": /* at 0x"),
        "keep the flat spelling: label + `goto` is the honest answer",
    );
    let branches = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Branch { .. }))
        .count();
    assert_eq!(
        branches, 3,
        "the fixture is three guards, one of them a back edge"
    );
    let flat_branches = rep
        .reasons
        .get("unstructured conditional branch")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        flat_branches, branches,
        "every branch of an irreducible region is counted as flat, in the census\nreasons: {:?}\n---- emitted C ----\n{text}\n------------------",
        rep.reasons
    );
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles, rep.shorts),
        (0, 0, 0, 0),
        "nothing was structured: a tree the text does not support is never printed (the walk proposed {} structured regions)\n---- emitted C ----\n{text}\n------------------",
        tree.report.ifs + tree.report.if_else + tree.report.whiles
    );

    // The census is a fact about the text: a scan of the text agrees with it.
    let tc = text_census(&text);
    assert_eq!(
        (tc.ifs, tc.els, tc.whiles),
        (0, 0, 0),
        "the text scan agrees that nothing is structured\n---- emitted C ----\n{text}\n------------------"
    );
    assert_eq!(
        (tc.gotos, tc.labels),
        (rep.gotos, rep.labels),
        "the census's goto and label counts are the text's\n---- emitted C ----\n{text}\n------------------"
    );
    assert!(
        rep.gotos >= branches * 2,
        "each flat guard spells its arms with a goto: {}\n{text}",
        rep.gotos
    );

    for (a, b) in [(0u64, 0), (1, 2), (3, 3), (u64::MAX, 1)] {
        assert_eq!(
            answer(&f, &[a, b]),
            0,
            "every path of the fixture returns 0"
        );
    }
}

/// A loop whose header has three doors: `e`, the latch, and a block reached
/// without going through the header. `while` needs one door; this must not
/// become one.
///
/// ```text
/// e: if (p0 == 0) goto h else z
/// h: kp = phi(0@e, k2@l, 9@q); if (kp < 5) goto l else z
/// l: k2 = kp + 1; jmp h
/// q: jmp h                      <- a door the header does not dominate
/// z: if (p1 == 0) goto q else r
/// r: return p1
/// ```
fn multi_entry_loop() -> Function {
    let mut f = fn2("multi_entry_loop");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let h = f.new_block(0x1010);
    let l = f.new_block(0x1020);
    let q = f.new_block(0x1030);
    let z = f.new_block(0x1040);
    let r = f.new_block(0x1050);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let a = f.emit(e, Op::Cmp(IntCmp::Eq, p0, zero), Ty::Bool, 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: a,
        then_bb: h,
        else_bb: z,
    };
    let kp = f.new_value(u64_());
    let five = f.emit(h, Op::Const(5), u64_(), 0x1010);
    let ct = f.emit(h, Op::Cmp(IntCmp::Ult, kp, five), Ty::Bool, 0x1014);
    f.block_mut(h).term = Terminator::Branch {
        cond: ct,
        then_bb: l,
        else_bb: z,
    };
    let one = f.emit(l, Op::Const(1), u64_(), 0x1020);
    let k2 = f.emit(l, Op::Bin(BinOp::Add, kp, one), u64_(), 0x1024);
    f.block_mut(l).term = Terminator::Jump(h);
    f.block_mut(q).term = Terminator::Jump(h);
    let nine = f.emit(z, Op::Const(9), u64_(), 0x1040);
    let b = f.emit(z, Op::Cmp(IntCmp::Eq, p1, nine), Ty::Bool, 0x1044);
    f.block_mut(z).term = Terminator::Branch {
        cond: b,
        then_bb: q,
        else_bb: r,
    };
    f.block_mut(r).term = Terminator::Return(p1);
    f.block_mut(h).instrs.insert(
        0,
        Instr::new(
            kp,
            Op::Phi(vec![(zero, e), (k2, l), (nine, q)].into_boxed_slice()),
            u64_(),
            0x1010,
        ),
    );
    f
}

#[test]
fn a_multi_entry_loop_is_not_structured() {
    let f = multi_entry_loop();
    let (text, rep) = emit(&f);
    let tree = structure_with_params(&f, true);
    assert_binds_arguments(&text);
    // The PASS's own proposal, not only what survived into the printed file, for
    // the thing a header with three doors can never be: a `while`. C's `while` has
    // one door, so a proposed loop here is an invention, and the emitter's second
    // safety net - a tree that names a label its text never defines is dropped for
    // the flat form - hides it from a test that only reads the printed census.
    // (The walk does propose 2 `if`s in this fixture, and they are not inventions:
    // each is a guard whose arms `region` proved. What makes the tree unprintable
    // is an arm that names a label its text never defines, and the net prints the
    // flat form for exactly that, counted - which the text assertions below check.)
    assert_eq!(
        tree.report.whiles, 0,
        "the walk proposes no loop for a header with three doors: a proposal the label net drops is still a proposal\nreasons: {:?}\nstmts {:#?}",
        tree.report.reasons,
        tree.stmts
    );
    wants(
        &text,
        |t| !structured_stmt(t, "if (") && !structured_stmt(t, "while ("),
        "invent no loop and no `if` for a header with three doors: no C statement has a second way in",
    );
    wants(&text, |t| t.contains("goto l"), "keep the flat spelling");
    assert_eq!(
        (rep.ifs, rep.if_else, rep.whiles),
        (0, 0, 0),
        "nothing structured\nreasons: {:?}\n---- emitted C ----\n{text}\n------------------",
        rep.reasons
    );
    assert!(
        rep.gotos > 0 && rep.labels > 0,
        "the census counts the gotos and labels it left behind: {} gotos, {} labels\n{text}",
        rep.gotos,
        rep.labels
    );
    let tc = text_census(&text);
    assert_eq!(
        (tc.gotos, tc.labels),
        (rep.gotos, rep.labels),
        "and the counts are the text's\n---- emitted C ----\n{text}\n------------------"
    );
}

// ---------------------------------------------------------------------------
// 5. short-circuit, and the guard on merging
// ---------------------------------------------------------------------------

/// Two guards, one join: `if (p0 < p1) { if (<second test>) store(8, 0x40, 99) }`.
///
/// With `merge`, the second guard block computes ONLY its own compare, which is
/// exactly when `a && b` is the same test the machine code ran. Without it, the
/// block computes a store first - and a merged condition would run that store on
/// the arm the machine code takes straight to the join.
fn two_guards(merge: bool) -> Function {
    let mut f = fn2("two_guards");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let g2 = f.new_block(0x1010);
    let t = f.new_block(0x1020);
    let j = f.new_block(0x1030);
    let c1 = f.emit(e, Op::Cmp(IntCmp::Ult, p0, p1), Ty::Bool, 0x1008);
    f.block_mut(e).term = Terminator::Branch {
        cond: c1,
        then_bb: g2,
        else_bb: j,
    };
    if merge {
        let c2 = f.emit(g2, Op::Cmp(IntCmp::Ugt, p1, p0), Ty::Bool, 0x1010);
        f.block_mut(g2).term = Terminator::Branch {
            cond: c2,
            then_bb: t,
            else_bb: j,
        };
    } else {
        let addr = f.emit(g2, Op::Const(0x20), u64_(), 0x1010);
        let val = f.emit(g2, Op::Const(7), u64_(), 0x1014);
        f.emit_void(g2, Op::Store(u64_(), addr, val), 0x1018);
        let c2 = f.emit(g2, Op::Cmp(IntCmp::Ugt, p1, p0), Ty::Bool, 0x101c);
        f.block_mut(g2).term = Terminator::Branch {
            cond: c2,
            then_bb: t,
            else_bb: j,
        };
    }
    let a = f.emit(t, Op::Const(0x40), u64_(), 0x1020);
    let v = f.emit(t, Op::Const(99), u64_(), 0x1024);
    f.emit_void(t, Op::Store(u64_(), a, v), 0x1028);
    f.block_mut(t).term = Terminator::Jump(j);
    let z = f.emit(j, Op::Const(0), u64_(), 0x1030);
    f.block_mut(j).term = Terminator::Return(z);
    f
}

#[test]
fn two_guards_merge_into_one_condition_only_when_the_link_computes_nothing() {
    // The link block computes something: no merge, two nested `if`s, the store kept.
    let f = two_guards(false);
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);
    wants(
        &text,
        |t| !cond_lines(t).iter().any(|l| l.contains("&&")),
        "not merge the guards with `&&` in a condition: the second block stores, and a merged condition would run that store on the arm the machine code skips (the label uses the emitter prints, `(void)&&lN;`, are not conditions and are not what this claims)",
    );
    wants(
        &text,
        |t| t.contains("store_8") && count(t, "if (") == 2,
        "keep the store, inside the nested `if`",
    );
    assert_eq!(
        rep.shorts, 0,
        "no short-circuit merge was earned\nreasons: {:?}\n---- emitted C ----\n{text}\n------------------",
        rep.reasons
    );

    // The link block computes only its compare: the merge is exact.
    let g = two_guards(true);
    let (gt, grep) = emit(&g);
    assert_binds_arguments(&gt);
    wants(
        &gt,
        |t| cond_lines(t).iter().any(|l| l.contains("&&")) && count(t, "if (") == 1,
        "merge two bare guards into one `if (a && b)` condition",
    );
    // Both guards skip to the join on their `else` arm, so the machine's test is
    // `a && b` with NEITHER value negated. A `!` in the merged condition is a
    // folded guard whose polarity was flipped: the same program only if the arm
    // it skips is the arm the machine code skips, and it is not.
    wants(
        &gt,
        |t| !cond_lines(t).iter().any(|l| l.contains('!')),
        "fold both guards with the machine code's polarity: `a && b`, neither negated",
    );
    wants(
        &gt,
        |t| !t.contains("goto l"),
        "need no `goto` for the merged chain",
    );
    assert_eq!(
        (grep.ifs, grep.shorts),
        (1, 1),
        "one `if` region, built from a merged condition\n---- emitted C ----\n{gt}\n------------------"
    );
    let tc = text_census(&gt);
    assert_eq!(
        (tc.ifs, tc.els, tc.whiles, tc.gotos, tc.labels),
        (
            grep.ifs + grep.if_else,
            grep.if_else,
            grep.whiles,
            grep.gotos,
            grep.labels
        ),
        "the census and the text agree\n---- emitted C ----\n{gt}\n------------------"
    );
}

// ---------------------------------------------------------------------------
// 6. the census describes the text, for every fixture
// ---------------------------------------------------------------------------

fn fixtures() -> Vec<(&'static str, Function)> {
    vec![
        ("single_if", single_if()),
        ("if_else", if_else()),
        ("count_up", count_up()),
        ("count_up_else", count_up_else_guard()),
        ("converge_below", arms_that_converge_below_the_join()),
        ("irreducible", irreducible()),
        ("multi_entry_loop", multi_entry_loop()),
        ("two_guards_merge", two_guards(true)),
        ("two_guards_nested", two_guards(false)),
    ]
}

#[test]
fn every_fixture_s_census_matches_its_emitted_text() {
    let mut total = Report::default();
    for (name, f) in fixtures() {
        let (text, rep) = emit(&f);
        let tree = structure_with_params(&f, true);
        assert!(
            tree.labels_defined() || rep.flat > tree.report.flat,
            "{name}: a tree that names an undefined label must not be the one printed\n---- emitted C ----\n{text}\n------------------"
        );
        let tc = text_census(&text);
        assert_eq!(
            (tc.ifs, tc.els, tc.whiles),
            (rep.ifs + rep.if_else, rep.if_else, rep.whiles),
            "{name}: the structured census must be what the text shows\n---- emitted C ----\n{text}\n------------------"
        );
        assert_eq!(
            (tc.gotos, tc.labels),
            (rep.gotos, rep.labels),
            "{name}: the goto/label census must be what the text shows\n---- emitted C ----\n{text}\n------------------"
        );
        assert!(
            !text.contains("do {"),
            "{name}: this pass never emits a do-while\n---- emitted C ----\n{text}\n------------------"
        );
        assert!(
            !text.contains("/* rdec:") || text.contains("jump table"),
            "{name}: the body carries no 'trust me' comment\n---- emitted C ----\n{text}\n------------------"
        );
        total = sum_reports(&[total, rep.clone()]);
    }
    // The fixtures demonstrate `if`, `if`/`else` and `while`, and the two shapes
    // that must NOT be structured, all with a census that adds up.
    assert!(
        total.ifs >= 1 && total.if_else >= 1 && total.whiles >= 1,
        "the fixtures must contain an if, an if/else and a while: {total:?}"
    );
    assert!(
        total.gotos >= 6 && total.labels >= 4,
        "and the unstructured shapes keep their gotos and labels: {total:?}"
    );
}

#[test]
fn every_fixture_s_tree_either_defines_its_labels_or_is_replaced_and_counted() {
    // The statement tree's own compile check, per fixture, and the LOUD half of
    // the deal: `Emitter::run` drops a tree that fails `labels_defined`, prints
    // the flat form instead, and counts the reason
    // `region whose text left a named label undefined (flat form)` into the
    // census. So exactly one of the two must be true for every fixture - a tree
    // that names a label its text never defines, and a census that says the
    // fallback ran - and the emitted file must name no undefined label at all.
    // `o40_f` in the corpus is the shape: an arm whose flat terminator names
    // `l5` and `l7`, neither of which the arm's text prints, and clang answers
    // `use of undeclared label`.
    for (name, f) in fixtures() {
        let tree = structure_with_params(&f, true);
        let (text, rep) = emit(&f);
        let fell_back = rep
            .reasons
            .get("region whose text left a named label undefined (flat form)")
            .copied()
            .unwrap_or(0);
        assert_eq!(
            tree.labels_defined(),
            fell_back == 0,
            "{name}: the tree defines its labels = {}/the fallback ran = {fell_back}; one of the two must be true and the census must say which\nreasons: {:?}\nstmts {:#?}\n---- emitted C ----\n{text}\n------------------",
            tree.labels_defined(),
            rep.reasons,
            tree.stmts
        );
        let bad = goto_targets_defined(&text);
        assert!(
            bad.is_empty(),
            "{name}: the printed C names a label it does not define: {bad:?}\n---- emitted C ----\n{text}\n------------------"
        );
    }
}

#[test]
fn the_flat_fallback_structures_nothing_and_defines_every_label() {
    // `flat_walk` is the spelling the emitter falls back to when a tree names a
    // label its text never defines: one plain block per reachable block, in RPO.
    for (name, f) in fixtures() {
        let s = flat_walk(&f);
        assert!(
            s.labels_defined(),
            "{name}: the flat form must name only labels it defines"
        );
        assert_eq!(
            (s.report.ifs, s.report.if_else, s.report.whiles),
            (0, 0, 0),
            "{name}: the flat form structures nothing"
        );
        let plain = s
            .stmts
            .iter()
            .filter(|st| matches!(st.kind, StmtKind::Block))
            .count();
        assert_eq!(
            plain,
            f.blocks.len(),
            "{name}: one plain statement per block, the entry's params printed separately"
        );
    }
    let s = flat_walk(&irreducible());
    assert!(
        s.report.gotos > 0,
        "and it keeps the gotos, counted: {}",
        s.report.gotos
    );
}
