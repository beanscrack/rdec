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
use rdec_core::structure::{
    flat_walk, structure_with_params, sum_reports, term_gotos, Cont, Report, StmtKind,
};
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

/// Emit for a graph that is NOT valid SSA on purpose (the readability-refusal
/// fixture). `c` verifies first, and that is the right rule for every fixture the
/// census suite walks: a hand-built graph the pipeline cannot produce must not
/// join that family. It is still the graph the readability rule is about, so the
/// pass runs on it here, and what this test can demand of it is what the pass owes
/// any graph: a `return` only where the value is readable, a counted refusal, and a
/// label block for every `goto` it prints.
fn emit_unverified(f: &Function) -> (String, Report) {
    let out = emit_c(f).expect("the fixture emits");
    (out.text.clone(), out.structuring)
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
        ("early_return", early_return_chain(false)),
        ("early_return_store", early_return_chain(true)),
    ]
}

#[test]
fn every_fixture_s_census_matches_its_emitted_text() {
    let mut total = Report::default();
    for (name, f) in fixtures() {
        let (text, rep) = emit(&f);
        let tree = structure_with_params(&f, true);
        // If the tree names a label its text never defines, the emitter must have
        // printed the flat walk instead - and the way to see that in the census is
        // the flat walk's OWN numbers, not a proxy that a readability feature can
        // move (absorption lowers `flat`, which is what the old proxy compared).
        let flat = flat_walk(&f);
        assert!(
            tree.labels_defined() || rep.gotos == flat.report.gotos,
            "{name}: a tree that names an undefined label must not be the one printed \
             (printed gotos {}, tree {}, flat {})\n---- emitted C ----\n{text}\n------------------",
            rep.gotos,
            tree.report.gotos,
            flat.report.gotos
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
        // One plain statement per block - except a pure-return block that EVERY
        // edge into it spells as a `return`: its text is dead, and the flat form
        // drops the statement and its label together (no `goto` can name the label,
        // which is what the absorber proved and the `labels_defined` check above
        // keeps). The ledger still closes.
        assert_eq!(
            plain + s.report.absorbed,
            f.blocks.len(),
            "{name}: one plain statement per printed block, the entry's params printed              separately (+{} absorbed return blocks)",
            s.report.absorbed
        );
    }
    let s = flat_walk(&irreducible());
    assert!(
        s.report.gotos > 0,
        "and it keeps the gotos, counted: {}",
        s.report.gotos
    );
}

// ---------------------------------------------------------------------------
// 7. M6: the early `return` the CFG proves, and the fall-through the LAYOUT proves
//
// Two readability features, red before they are implemented:
//
// * a `goto lX;` whose target block computes NOTHING and returns a value that is
//   readable on THIS edge prints `return <that value>;` at the site - the classic
//   early return. The target's label block keeps its own print for the edges that
//   are not early-returned; when EVERY edge into it spells the return, the block's
//   text is dead and its label goes with it.
// * a `goto lN;` whose target is the very next statement in the printed layout is
//   a no-op: the flat form lays its blocks out so the chains land adjacent, and
//   drops those gotos. Backward jumps stay backward.
//
// The guard fixtures beside them are the halves that must NOT be given away: a
// target that computes (a store) keeps its `goto`, and a value that is not
// readable at the site keeps its `goto`.
// ---------------------------------------------------------------------------

/// The early-return shape, built so the structuring patterns CANNOT prove a
/// region (the nested guard's two arms converge on `q`, a block neither arm
/// owns), so the walk prints the label + `goto` form - the form the census calls
/// `fall-through` / `unstructured conditional branch` / `return`, and the form 8
/// corpus functions fall back to.
///
/// ```text
/// e:   c0 = (p0 == 0);  if c0 goto g else L     edge e->L carries `p1`
/// g:   c1 = (p1 == 0);  if c1 goto t else f     <- stays flat (q is multi-entry)
/// t:   jmp q
/// f:   jmp q
/// q:   c2 = (p0 < p1);  if c2 goto L else k     edge q->L carries `p0`
/// k:   z = p0 + 1;   jmp L                      edge k->L carries `z`
/// L:   r = phi(p1@e, p0@q, z@k);  return r
/// ```
///
/// The answer, hand-derived from the graph (never from the emitter): `p0 != 0`
/// takes the guard's `else` arm straight to `L` and returns `p1`; `p0 == 0 &&
/// p0 < p1` runs the nested guard, lands on the `q` edge and returns `p0`;
/// `p0 == 0 && p0 >= p1` runs `k` and returns `p0 + 1`.
fn early_return_chain(store_in_target: bool) -> Function {
    let mut f = fn2("early_return_chain");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let g = f.new_block(0x1010);
    let t = f.new_block(0x1020);
    let ff = f.new_block(0x1030);
    let q = f.new_block(0x1040);
    let k = f.new_block(0x1050);
    let l = f.new_block(0x1060);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let c0 = f.emit(e, Op::Cmp(IntCmp::Eq, p0, zero), Ty::Bool, 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: c0,
        then_bb: g,
        else_bb: l,
    };
    let c1 = f.emit(g, Op::Cmp(IntCmp::Eq, p1, zero), Ty::Bool, 0x1010);
    f.block_mut(g).term = Terminator::Branch {
        cond: c1,
        then_bb: t,
        else_bb: ff,
    };
    f.block_mut(t).term = Terminator::Jump(q);
    f.block_mut(ff).term = Terminator::Jump(q);
    let c2 = f.emit(q, Op::Cmp(IntCmp::Ult, p0, p1), Ty::Bool, 0x1040);
    f.block_mut(q).term = Terminator::Branch {
        cond: c2,
        then_bb: l,
        else_bb: k,
    };
    let one = f.emit(k, Op::Const(1), u64_(), 0x1050);
    let z = f.emit(k, Op::Bin(BinOp::Add, p0, one), u64_(), 0x1054);
    let _ = z;
    f.block_mut(k).term = Terminator::Jump(l);
    let r = f.new_value(u64_());
    f.block_mut(l).instrs.push(Instr::new(
        r,
        Op::Phi(vec![(p1, e), (p0, q), (z, k)].into_boxed_slice()),
        u64_(),
        0x106c,
    ));
    if store_in_target {
        // The one thing that makes the target NOT a pure tail: a store, AFTER the
        // phi prefix (a phi after a non-phi is not valid SSA, and `verify_ssa`
        // says so). The machine runs this store on the edge into `L`; a `return`
        // spelled at the goto site with no statement for it runs nothing.
        let a = f.emit(l, Op::Const(0x40), u64_(), 0x1070);
        let v = f.emit(l, Op::Const(99), u64_(), 0x1074);
        f.emit_void(l, Op::Store(u64_(), a, v), 0x1078);
    }
    f.block_mut(l).term = Terminator::Return(r);
    // The DECLARED return type the lifter records: the emitted C's `return` casts
    // through it, and a driver can only read an answer the signature declares.
    f.ret_ty = u64_();
    f
}

/// The return block of [`early_return_chain`]: the last of the seven blocks.
fn er_target() -> u32 {
    6
}

/// The (input, answer) table, derived from the graph by hand: `p0 != 0` takes the
/// guard's `else` arm straight to `L` on the `e` edge, which carries `p1`;
/// `p0 == 0 && p0 < p1` arrives on the `q` edge, which carries `p0`; `p0 == 0 &&
/// p0 >= p1` arrives on the `k` edge, which carries `p0 + 1`.
fn early_return_table() -> Vec<(u64, u64, u64)> {
    vec![(0, 0, 1), (0, 5, 0), (3, 7, 7), (7, 2, 2), (1, 1, 1)]
}

/// Count the `return (T)<name>;` lines that hand back value `v`.
fn returns_value(text: &str, v: u32) -> usize {
    // Count every `return (T)xN;`, wherever it sits: an early return also spells
    // the arm of an `if (c) return (uint64_t)x0; else goto l5;`.
    let n = nm(v);
    let tail = format!("{n};");
    text.split("return (")
        .skip(1)
        .filter(|seg| {
            seg.find(')')
                .map(|i| seg[i + 1..].starts_with(&tail))
                .unwrap_or(false)
        })
        .count()
}

#[test]
fn an_early_return_is_printed_at_the_goto_site_and_the_dead_label_block_goes() {
    let f = early_return_chain(false);
    let (text, rep) = emit(&f);
    assert_binds_arguments(&text);
    let l = er_target();
    assert!(matches!(f.blocks[l as usize].term, Terminator::Return(_)));

    // (1) the goto to the pure-return block is gone at every one of its three
    // incoming edges.
    wants(
        &text,
        |t| !t.contains(&format!("goto l{l};")),
        "print `return`, never `goto l6;`: all three edges into the pure-return block spell the return they arrive with",
    );
    // (2) the label block is dead text once every edge returns: no label, no body.
    wants(
        &text,
        |t| !t.contains(&format!("l{l}:")),
        "drop the label line of the block every edge returns from",
    );
    // (3) each early return spells the value that arrives on ITS edge: the phi
    // operand of the edge, never the phi at the join.
    for v in f.params.clone() {
        wants(
            &text,
            move |t| returns_value(t, v) >= 1,
            "the e and q edges return the parameter value THEY carry, spelled at the site",
        );
    }
    let z = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find(|i| matches!(i.op, Op::Bin(BinOp::Add, _, _)))
        .expect("the k block adds 1")
        .dest;
    wants(
        &text,
        move |t| returns_value(t, z) >= 1,
        "the k edge returns the value it computed, spelled at the site",
    );
    // (4) the gotos to blocks that DO compute work survive.
    wants(
        &text,
        |t| t.contains("goto l2;") || t.contains("goto l3;") || t.contains("goto l4;"),
        "a goto to a block that is not a pure return target stays a goto",
    );

    // (5) the census is the text, and the text got shorter: this function's flat
    // form starts at 9 `goto` statements and 7 labels.
    let tc = text_census(&text);
    assert_eq!(
        (tc.gotos, tc.labels),
        (rep.gotos, rep.labels),
        "the goto/label census must be what the text shows\n---- emitted C ----\n{text}\n------------------"
    );
    assert!(
        rep.gotos < 9 && rep.labels < 7,
        "three `goto l6;` and the dead label must be gone from a flat form that \
         started at 9 gotos and 7 labels: gotos {} labels {}\n---- emitted C ----\n{text}\n------------------",
        rep.gotos,
        rep.labels
    );

    // (6) the graph's answers, hand-computed. The interpreter reads the CFG, so
    // these are the values the C must reproduce (the next test compiles the C).
    for (a0, a1, want) in early_return_table() {
        assert_eq!(answer(&f, &[a0, a1]), want, "f({a0}, {a1})");
    }
    let bad = goto_targets_defined(&text);
    assert!(
        bad.is_empty(),
        "the printed C names a label it does not define: {bad:?}\n---- emitted C ----\n{text}\n------------------"
    );
}

#[test]
fn the_early_return_text_compiles_with_werror_and_answers_the_graph_s_values() {
    // The text-level proof. The interpreter reads the graph, so it cannot see a
    // mis-spelled return: compile PRELUDE + the emitted C exactly as `rdec c`
    // composes it, run the same vectors, and require the graph's digits.
    let f = early_return_chain(false);
    let text = c(&f);
    let dir = std::env::temp_dir().join("rdec_early_return");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("early_return_chain.c");
    let mut call = String::new();
    for (a0, a1, _) in early_return_table() {
        call.push_str(&format!(
            "  r = r * 100 + (unsigned)early_return_chain({a0}ULL, {a1}ULL);\n"
        ));
    }
    std::fs::write(
        &src,
        format!(
            "{}\n#include <stdio.h>\n{text}\nint main(void){{ unsigned r = 0;\n{call}  \
             printf(\"%010u\\n\", r); return 0; }}\n",
            rdec_core::emit_c::PRELUDE
        ),
    )
    .unwrap();
    let out = std::process::Command::new("clang")
        .args([
            "-std=c17",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
            "-o",
        ])
        .arg(dir.join("early.exe"))
        .arg(&src)
        .output()
        .expect("clang must be on PATH in this repo (CI runs it)");
    assert!(
        out.status.success(),
        "-Werror compile failed:\n{}\n---- emitted C ----\n{text}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run_c = std::process::Command::new(dir.join("early.exe"))
        .output()
        .expect("run");
    let printed = String::from_utf8_lossy(&run_c.stdout).trim().to_string();
    let mut want = String::new();
    for (a0, a1, w) in early_return_table() {
        assert_eq!(answer(&f, &[a0, a1]), w);
        want.push_str(&format!("{w:02}"));
    }
    assert_eq!(
        printed,
        want,
        "the emitted C and the graph must answer the same digits (vectors {:?})",
        early_return_table()
    );
}

#[test]
fn a_return_target_that_computes_keeps_its_goto() {
    // The statement guard: `L` now stores before it returns. The machine runs
    // that store on the edge into `L`, and a `return` spelled at the goto site
    // with no statement for it runs nothing - so the guard is the difference
    // between a decompiler and a miscompiler. This is the sabotage needle: print
    // the return for a target that computes, and this goes red.
    let f = early_return_chain(true);
    let (text, rep) = emit(&f);
    let l = er_target();
    assert_eq!(
        count(&text, &format!("goto l{l};")),
        3,
        "all three edges into a target that stores keep their goto\n---- emitted C ----\n{text}\n------------------"
    );
    wants(
        &text,
        |t| t.contains(&format!("l{l}:")),
        "and the label block stays, because a goto lands on it",
    );
    let tc = text_census(&text);
    assert_eq!(
        (tc.gotos, tc.labels),
        (rep.gotos, rep.labels),
        "the census must be the text\n---- emitted C ----\n{text}\n------------------"
    );

    // The semantic half, which the census cannot see: the target's store is loaded
    // straight back, so it IS the answer. Compile the emitted C and run the graph's
    // vectors. An early return printed over a target that computes returns 0 from
    // zeroed memory where the machine returns 99 - and this line is where that
    // shows up, in digits, not in a goto count.
    let answers = compile_answers(&c(&f), "early_return_store", "early_return_chain");
    let want: String = early_return_table()
        .iter()
        .map(|(a0, a1, w)| {
            assert_eq!(answer(&f, &[*a0, *a1]), *w, "graph f({a0}, {a1})");
            format!("{w:02}")
        })
        .collect();
    assert_eq!(
        answers, want,
        "the emitted C of the STORE variant must run the store: the graph's digits are \
         {want}\n---- emitted C ----\n{text}\n------------------"
    );
}

/// Compile `text` with the same flags `rdec c` uses, call `fn_name` on the
/// [`early_return_table`] vectors, and return the digits the C printed.
fn compile_answers(text: &str, stem: &str, fn_name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("rdec_{stem}"));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("{stem}.c"));
    let mut call = String::new();
    for (a0, a1, _) in early_return_table() {
        call.push_str(&format!(
            "  r = r * 100 + (unsigned){fn_name}({a0}ULL, {a1}ULL);\n"
        ));
    }
    std::fs::write(
        &src,
        format!(
            "{}\n#include <stdio.h>\n{text}\nint main(void){{ unsigned r = 0;\n{call}               printf(\"%010u\\n\", r); return 0; }}\n",
            rdec_core::emit_c::PRELUDE
        ),
    )
    .unwrap();
    let out = std::process::Command::new("clang")
        .args([
            "-std=c17",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Werror",
            "-o",
        ])
        .arg(dir.join(format!("{stem}.exe")))
        .arg(&src)
        .output()
        .expect("clang must be on PATH in this repo (CI runs it)");
    assert!(
        out.status.success(),
        "-Werror compile failed:\n{}\n---- emitted C ----\n{text}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run_c = std::process::Command::new(dir.join(format!("{stem}.exe")))
        .output()
        .expect("run");
    String::from_utf8_lossy(&run_c.stdout).trim().to_string()
}

/// The readability check's own witness, on a graph `verify_ssa` REJECTS.
///
/// ```text
/// e:  c0 = (p0 == 0);  if c0 goto g else L   edge e->L: x is NOT written on this path
/// g:  c1 = (p1 == 0);  if c1 goto t else f   <- stays flat (q is multi-entry)
/// t:  jmp q
/// f:  jmp q
/// q:  x = p0 + p1;     if (p0 < p1) goto L else k    edge q->L: x IS written here
/// k:                  jmp L                 edge k->L: every path to k ran q
/// L:  return x                   x's def (q) does not dominate L: the e edge skips q
/// ```
///
/// The shape is [`early_return_chain`]'s on purpose: the nested guard's two arms
/// converge on `q`, a block neither arm owns, so no region is provable and the
/// label + `goto` form runs. That is the only form where a jump to a return block is
/// a `goto` the pass has to spell - a provable region reaches its exit as the text's
/// own fall-through, and there is nothing there to convert.
///
/// `q` does not dominate `L` (the `e -> L` edge reaches `L` without it), so
/// `verify_ssa`'s def-dominates-use rule rejects the function: no lifted corpus
/// function can look like this, which is exactly why the check is defensive on the
/// corpus. Built by hand it is the graph the rule is about - the `q` and `k` edges
/// may print `return (uint64_t)x;` because that path ran `q`; the `e` edge may not,
/// because on that path the local was never written. A `return` printed at the `e`
/// site reads an unwritten local, and the graph's answer is not a C answer.
fn er_unreadable() -> Function {
    let mut f = fn2("er_unreadable");
    let (e, p0, p1) = (f.entry, f.params[0], f.params[1]);
    let g = f.new_block(0x1010);
    let t = f.new_block(0x1020);
    let ff = f.new_block(0x1030);
    let q = f.new_block(0x1040);
    let k = f.new_block(0x1050);
    let l = f.new_block(0x1060);
    let zero = f.emit(e, Op::Const(0), u64_(), 0x1008);
    let c0 = f.emit(e, Op::Cmp(IntCmp::Eq, p0, zero), Ty::Bool, 0x100c);
    f.block_mut(e).term = Terminator::Branch {
        cond: c0,
        then_bb: g,
        else_bb: l,
    };
    let c1 = f.emit(g, Op::Cmp(IntCmp::Eq, p1, zero), Ty::Bool, 0x1010);
    f.block_mut(g).term = Terminator::Branch {
        cond: c1,
        then_bb: t,
        else_bb: ff,
    };
    f.block_mut(t).term = Terminator::Jump(q);
    f.block_mut(ff).term = Terminator::Jump(q);
    let x = f.emit(q, Op::Bin(BinOp::Add, p0, p1), u64_(), 0x1040);
    let c2 = f.emit(q, Op::Cmp(IntCmp::Ult, p0, p1), Ty::Bool, 0x1044);
    f.block_mut(q).term = Terminator::Branch {
        cond: c2,
        then_bb: l,
        else_bb: k,
    };
    f.block_mut(k).term = Terminator::Jump(l);
    f.block_mut(l).term = Terminator::Return(x);
    f.ret_ty = u64_();
    f
}

/// The return block of [`er_unreadable`]: the last of the seven blocks.
fn er_only_target() -> u32 {
    6
}

/// The dominance/readability refusal: counted, with the goto kept, on the edge the
/// value is not readable on. `verify_ssa` rejects this graph (the fixture says so),
/// so this is the rule proven on the only graph that can show it - the named gap is
/// that no emitted corpus C can.
#[test]
fn an_early_return_is_refused_when_the_value_is_not_readable_at_the_site() {
    // The graph is invalid SSA on purpose (the fixture says so), so this walks
    // past `c`'s verify: the rule being tested is the pass's own, and the pass is
    // what must not invent a `return` on the edge that cannot read the value.
    let f = er_unreadable();
    assert!(
        verify_ssa(&f).is_err(),
        "this fixture is not valid SSA on purpose: x's def (block 4, `q`) does not \
         dominate its use in `L`, which is the shape the check guards"
    );
    let (text, rep) = emit_unverified(&f);
    let t = er_only_target();
    assert_eq!(
        rep.early_returns, 2,
        "the `q` edge (its own block wrote x) and the `k` edge (q dominates k) print the \
         return; the `e` edge may not\n---- emitted C ----\n{text}\n------------------"
    );
    assert_eq!(
        rep.reasons
            .get("early return refused: value not readable at the site"),
        Some(&1),
        "the refused edge is counted, loud, not silent: {:?}",
        rep.reasons
    );
    assert_eq!(
        count(&text, &format!("goto l{t};")),
        1,
        "the edge that cannot read the value keeps its goto\n---- emitted C ----\n{text}\n------------------"
    );
    assert_eq!(
        text_census(&text).gotos,
        rep.gotos,
        "the census is the text\n---- emitted C ----\n{text}\n------------------"
    );
    let bad = goto_targets_defined(&text);
    assert!(
        bad.is_empty(),
        "the printed C names a label it does not define: {bad:?}\n---- emitted C ----\n{text}\n------------------"
    );
    assert!(
        text.contains(&format!("l{t}:")),
        "the return block a `goto` still names must keep its label block\n---- emitted C ----\n{text}\n------------------"
    );
    assert_eq!(
        text_census(&text).labels,
        rep.labels,
        "the label census is the text\n---- emitted C ----\n{text}\n------------------"
    );
}

/// The blocks the flat form prints, in the order its text runs.
fn printed_order(s: &rdec_core::structure::Structured) -> Vec<u32> {
    s.stmts
        .iter()
        .filter(|st| matches!(st.kind, StmtKind::Block))
        .map(|st| st.bb)
        .collect()
}

/// The flat form absorbs every `goto` that points at the NEXT printed statement,
/// prints every reachable block exactly once, and keeps every back edge a
/// backward jump.
#[test]
fn the_flat_layout_absorbs_fall_through_gotos_and_keeps_back_edges_backward() {
    let f = count_up();
    let s = flat_walk(&f);
    // The flat form of this function spells 4 gotos: e->h, h's two branch arms,
    // and the latch l->h. `e: jmp h` is the one that the layout can absorb by
    // printing h next.
    assert!(
        s.report.gotos <= 3,
        "`e: jmp h` must print no goto once the layout puts h next: {} gotos, stmts {:#?}",
        s.report.gotos,
        s.stmts
    );
    // No PRINTED goto may point at the statement printed next to it: that jump is a
    // no-op. A statement whose terminator the layout absorbed (`Cont::Fall`) prints
    // nothing, so adjacency is legal there.
    let printed = printed_order(&s);
    let mut j = 0usize;
    for st in &s.stmts {
        if matches!(st.kind, StmtKind::Block) {
            if let Cont::Term = st.cont {
                if let Terminator::Jump(x) = f.blocks[st.bb as usize].term {
                    assert!(
                        printed.get(j + 1) != Some(&x),
                        "block {} jumps to {}, which is the very next printed statement: that goto is a no-op and must be gone\nstmts {:#?}",
                        st.bb,
                        x,
                        s.stmts
                    );
                }
            }
            j += 1;
        }
    }
    // The loop's back edge keeps its `goto`: the layout absorbs the jumps that land
    // on the next line of text, and a backward jump is not one of them - dropping it
    // would run the loop body one extra time. This is the assertion that makes the
    // adjacency requirement load-bearing: absorb every jump without checking
    // adjacency, and the latch's jump disappears right here.
    let latch = s
        .stmts
        .iter()
        .find(|st| st.bb == 2)
        .expect("the latch block is printed");
    assert!(
        matches!(latch.cont, Cont::Term),
        "the latch's backward jump to the header must stay a printed terminator: {:?}",
        latch.cont
    );
    // And the ledger is exact, not "at least one". This layout - `e`, the header,
    // the latch - puts the header on the entry's next line, so exactly ONE jump
    // lands on the text's own fall-through; and the pure-return block `x`, whose
    // every edge the text now spells `return`, is exactly ONE absorbed block.
    assert_eq!(
        s.report.fallthroughs, 1,
        "exactly the one fall-through this layout earns, stmts {:#?}",
        s.stmts
    );
    assert_eq!(
        s.report.absorbed, 1,
        "the pure-return block is absorbed, stmts {:#?}",
        s.stmts
    );
    assert_eq!(
        s.report.early_returns, 1,
        "the header's `then` arm into the pure-return block prints the return it \
         arrives with, stmts {:#?}",
        s.stmts
    );
    // Layout, not loss: every reachable block is printed at most once, and the only
    // blocks that print NO statement are absorbed return blocks - a block whose
    // every edge the text spells `return`, so nothing is left for it to run. The
    // ledger closes: printed + absorbed = reachable.
    let cfg = rdec_core::cfg::Cfg::build(&f);
    for b in &cfg.rpo {
        let n = printed.iter().filter(|p| **p == *b).count();
        assert!(
            n <= 1,
            "block {b} must be printed at most once, printed {printed:?}"
        );
        if n == 0 {
            let blk = &f.blocks[*b as usize];
            assert!(
                matches!(blk.term, Terminator::Return(_))
                    && blk
                        .instrs
                        .iter()
                        .all(|i| matches!(i.op, rdec_core::ir::Op::Phi(_))),
                "block {b} printed no statement and is not a pure-return block: {:?}",
                blk.term
            );
        }
    }
    assert_eq!(
        printed.len() + s.report.absorbed,
        cfg.rpo.len(),
        "the printed blocks plus the absorbed ones must be the reachable blocks"
    );
    // A back edge stays a backward jump in the text: the layout may not turn a
    // loop inside out.
    for b in &cfg.rpo {
        for s2 in &cfg.succs[*b as usize] {
            if cfg.dominates(*s2, *b) {
                let from = printed.iter().position(|p| *p == *b).unwrap();
                let to = printed.iter().position(|p| *p == *s2).unwrap();
                assert!(
                    to < from,
                    "the back edge {b} -> {s2} must print backward: {from} -> {to}, printed {printed:?}"
                );
            }
        }
    }
}

/// The same layout invariant over every fixture, so the DFS layout cannot buy a
/// pruned goto by losing a block or flipping a loop.
///
/// **The named gap, on `layout_order`'s back-edge guard.** The guard refuses a
/// layout that puts a back edge forward and prints RPO instead, and no fixture can
/// watch it refuse: the sabotage "delete the guard" leaves all 18 tests green. It is
/// a theorem of the walk, not a property the fixtures happen to satisfy. The layout
/// is a DFS preorder, and in a DFS of a flow graph every edge whose head dominates
/// its tail is an edge to a DFS ANCESTOR - the path the walk took from the entry to
/// the tail runs through the head, because every path does - and an ancestor has the
/// smaller preorder index. So the order the walk produces already prints every back
/// edge backward, and the check can only fire on a layout that is not a DFS preorder.
/// That is the shape of this file's `aa == ab` precedent: a live guard on a property
/// the pass cannot violate by construction, witnessed by the two sentences above
/// where no fixture can witness it, and recorded as un-witnessable rather than as
/// tested. The RPO order it falls back to is the order this pass printed before M6,
/// so the fallback is the shape the corpus tests have always covered.
#[test]
fn every_fixture_s_flat_layout_prints_each_block_once_and_back_edges_backward() {
    for (name, f) in fixtures() {
        let s = flat_walk(&f);
        let cfg = rdec_core::cfg::Cfg::build(&f);
        let printed = printed_order(&s);
        for b in &cfg.rpo {
            let n = printed.iter().filter(|p| **p == *b).count();
            assert!(
                n <= 1,
                "{name}: block {b} printed {n} times, must be at most once: {printed:?}"
            );
            if n == 0 {
                let blk = &f.blocks[*b as usize];
                assert!(
                    matches!(blk.term, Terminator::Return(_))
                        && blk
                            .instrs
                            .iter()
                            .all(|i| matches!(i.op, rdec_core::ir::Op::Phi(_))),
                    "{name}: block {b} printed no statement and is not a pure-return block: {:?}",
                    blk.term
                );
            }
        }
        // The fall-through ledger, counted independently of the pass: a `jmp` is
        // absorbed when - and only when - its target is the statement printed next.
        // Absorb one without checking adjacency and this number stops matching.
        let adjacent = printed
            .iter()
            .enumerate()
            .filter(|(i, bb)| {
                matches!(
                    f.blocks[**bb as usize].term,
                    Terminator::Jump(x) if printed.get(i + 1) == Some(&x)
                )
            })
            .count();
        assert_eq!(
            s.report.fallthroughs, adjacent,
            "{name}: the layout absorbed {} jumps, and {} of the printed statements              jump to the statement printed next: {printed:?}",
            s.report.fallthroughs, adjacent,
        );
        assert_eq!(
            printed.len() + s.report.absorbed,
            cfg.rpo.len(),
            "{name}: {} printed statements + {} absorbed return blocks != {} reachable blocks",
            printed.len(),
            s.report.absorbed,
            cfg.rpo.len()
        );
        let mut j = 0usize;
        for st in &s.stmts {
            let bb = st.bb;
            if matches!(st.kind, StmtKind::Block) {
                if let Cont::Term = st.cont {
                    if let Terminator::Jump(x) = f.blocks[bb as usize].term {
                        assert!(
                            printed.get(j + 1) != Some(&x),
                            "{name}: block {bb} jumps to the statement printed next to it: that goto is a no-op\nstmts {:#?}",
                            s.stmts
                        );
                    }
                }
                j += 1;
            }
            for s2 in &cfg.succs[bb as usize] {
                if cfg.dominates(*s2, bb) {
                    let from = printed.iter().position(|p| *p == bb).unwrap();
                    let to = printed.iter().position(|p| *p == *s2).unwrap();
                    assert!(
                        to < from,
                        "{name}: the back edge {bb} -> {s2} prints forward ({from} -> {to}): {printed:?}"
                    );
                }
            }
        }
        // And the census is the text, absorption and early return included: every
        // `goto` the report counts belongs to a statement that is printed, and every
        // early return is a `return` in the text, not a `goto`.
        let text_gotos = s
            .stmts
            .iter()
            .map(|st| {
                let g = match st.cont {
                    Cont::Term => term_gotos(&f.blocks[st.bb as usize].term, &st.tails),
                    _ => 0,
                };
                g + usize::from(matches!(st.kind, StmtKind::Goto))
            })
            .sum::<usize>();
        assert_eq!(
            s.report.gotos, text_gotos,
            "{name}: the goto census must be the printed gotos ({} vs {})",
            s.report.gotos, text_gotos
        );
    }
}
