//! SSA IR -> C text.
//!
//! The rule for this whole file is the project's rule: **no silent semantic
//! loss**. Anything the emitter cannot express in C is a `Refusal`, not a
//! comment, not a guess, not a zero. The gate counts refusals by reason, so
//! "we emitted C for every function" can never quietly mean "we emitted C that
//! means something else".
//!
//! Three design decisions that shape the output:
//!
//! * **Phis become edge-copies.** A phi's value is chosen by the edge taken, so
//!   the copy belongs at the *end of each predecessor*, before its terminator.
//!   That is the standard SSA-out transformation, and it is why the emitter needs
//!   the CFG: the copies are derived from the same (value, predecessor) pairs the
//!   verifier checked.
//! * **Memory is flat bytes.** Machine code addresses bytes, not objects, so the
//!   C uses one `static uint8_t[]` and `memcpy` at a masked address. The mask is
//!   a modelling choice, said out loud: an out-of-range address becomes a
//!   deterministic in-range one rather than undefined behaviour, because the
//!   differential harness needs the two sides to *agree*, and C UB does not agree
//!   with anything.
//! * **Widths are explicit.** Every destination is assigned through a cast to its
//!   declared type, so truncation is the C the IR claims, and shift counts are
//!   masked the way x86 masks them. An unmasked `a >> 64` is UB in C and would
//!   make the differential meaningless.
//!
//! Flags must be gone before this runs: `Op::Flags` / `Op::GetFlag` are refusals,
//! which is the loud version of `Ty::Flags`'s own doc comment ("must not survive
//! the flag-elimination pass").
//!
//! **Control flow is structure, where the CFG proves it.** The body is printed
//! from [`crate::structure`]: `if` / `if`-`else` / `while` for the regions the CFG
//! supports, and label + `goto` for every region it does not - irreducible loops,
//! multi-entry regions, a block with a predecessor the pattern cannot account for.
//! A label is emitted only where a `goto` lands on it, and only those labels get a
//! `(void)&&lN;` use, so `-Werror -Wunused-label` stays clean for a reason: every
//! address-of-label names a label the text really defines.
use crate::ir::{BbId, BinOp, Callee, CastKind, FBinOp, Function, IntCmp, Op, Terminator, UnOp};
use crate::structure::{Cond, Cont, Report, Stmt, StmtKind};
use crate::ty::{FWidth, Sx, Ty, Width};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// Flat memory size, a power of two so the address mask is a plain AND.
pub const MEM_BITS: u32 = 20;
pub const MEM_SIZE: u64 = 1 << MEM_BITS;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal {
    /// Machine address of the instruction we cannot express, when there is one.
    pub at: u64,
    pub reason: String,
}

#[derive(Debug)]
pub struct Out {
    pub text: String,
    /// The file-scope declarations this function's text needs. That includes a
    /// PROTOTYPE for a callee whose definition lives in the same translation
    /// unit: the gates read this list as "does this function call out", and a
    /// function that calls a sibling does — the differential harness that links
    /// the callee's own definition is a separate claim, not this one.
    pub externs: Vec<String>,
    /// What the structuring pass did to this function, counted from the
    /// statements it built. The `c` and `cpp` gates print it and cross-check it
    /// against a scan of `text`, so the census cannot drift from the C.
    pub structuring: Report,
}

/// What the emitted C text says about control flow, counted from the text itself.
///
/// This is the census's independent second opinion: [`Report`] is what the
/// structuring walk says it built, this is what the printed file contains, and
/// `rdec c` fails when the two disagree. The rules are exact spellings, so a
/// flat `if (x) goto l1; else goto l2;` line - which ends in `;`, not `{` - is
/// NOT counted as a structured `if`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextCensus {
    /// Structured `if` statements, with and without an `else` arm.
    pub ifs: usize,
    /// `} else {` arms: each one pairs with one structured `if`.
    pub els: usize,
    /// Structured conditions that fold tests with `&&` - the short-circuit
    /// spellings. A condition is one line, so this counts lines.
    pub shorts: usize,
    /// `while (...) {` statements.
    pub whiles: usize,
    /// `goto l` statements, including the flat form's and a `switch` arm's.
    pub gotos: usize,
    /// `lN:` labels.
    pub labels: usize,
}

/// Scan emitted C for the control-flow spellings [`Report`] counts.
pub fn text_census(text: &str) -> TextCensus {
    let mut t = TextCensus::default();
    for line in text.lines() {
        let s = line.trim();
        if (s.starts_with("if (") || s.starts_with("while (")) && s.ends_with('{') {
            if s.starts_with("while (") {
                t.whiles += 1;
            } else {
                t.ifs += 1;
            }
            t.shorts += s.matches(" && ").count();
        }
        if s == "} else {" {
            t.els += 1;
        }
        t.gotos += s.matches("goto l").count();
        // A label line: `l7: /* at 0x12 */`, at any indentation. The emitter's
        // own identifiers are `x<N>` locals and `a<N>` parameters, so no
        // statement can start with `l<digits>:` - this pattern is the label.
        if let Some(rest) = s.strip_prefix('l') {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with(':') {
                t.labels += 1;
            }
        }
    }
    t
}

/// What the emitter needs to know about a callee to call it: the DECLARED return
/// type and the DECLARED parameter types, the same values the emitter uses to
/// spell the callee's definition. One source, so a prototype and a definition
/// cannot drift into a `conflicting types` error.
#[derive(Clone, Debug)]
pub struct CalleeDecl {
    pub ret: Ty,
    pub params: Vec<Ty>,
}

/// Callee name (as `Callee::Symbol` spells it) -> declared shape.
pub type Protos = HashMap<String, CalleeDecl>;

/// The DECLARED C parameter types of a function, exactly as `run` spells its
/// definition: the ABI's declared type when there is one, the value's own type
/// otherwise, and the emitter's `max(1)` rule — a C function with no parameters
/// is spelled with one, so the prototype has to say the same thing.
pub fn c_param_tys(f: &Function) -> Vec<Ty> {
    let n = f.params.len().max(1);
    (0..n)
        .map(|i| {
            f.param_tys
                .get(i)
                .cloned()
                .or_else(|| f.params.get(i).and_then(|v| f.ty_of(*v)))
                .unwrap_or(Ty::u64())
        })
        .collect()
}

/// The declared shape of `f` as a callee: built from the function itself, so a
/// prototype the caller emits and the definition the callee emits are the same
/// information from the same place.
pub fn callee_decl(f: &Function) -> CalleeDecl {
    CalleeDecl {
        ret: f.ret_ty.clone(),
        params: c_param_tys(f),
    }
}

/// The prototype for one callee, spelled from its declared shape.
fn proto_text(name: &str, d: &CalleeDecl) -> String {
    let ret = c_ty(&d.ret).unwrap_or_else(|| "int64_t".into());
    let ps: Vec<String> = d
        .params
        .iter()
        .map(|t| c_ty(t).unwrap_or_else(|| "int64_t".into()))
        .collect();
    // `(void)`, not `()`: an empty parameter list is a declaration with NO
    // prototype, and `-Wextra -Werror` in C17 calls that an error.
    let ps = if ps.is_empty() {
        "void".to_string()
    } else {
        ps.join(", ")
    };
    format!("extern {ret} {name}({ps});")
}

pub fn emit_c(f: &Function) -> Result<Out, Vec<Refusal>> {
    emit_c_sig(f, &Protos::new())
}

/// Emit one function, calling callees whose declared shapes are in `protos`.
/// Every function the pipeline emits into one translation unit gets a prototype
/// entry, so a call to a sibling is a real C call to a declared function, with
/// the arguments passed as the callee's declared types.
pub fn emit_c_sig(f: &Function, protos: &Protos) -> Result<Out, Vec<Refusal>> {
    let ret = c_ty(&f.ret_ty).unwrap_or_else(|| "int64_t".into());
    let mut e = Emitter {
        f,
        protos,
        tail: crate::ir::NO_VAL,
        noret: false,
        ret: ret.clone(),
        body: String::new(),
        externs: Vec::new(),
        refusals: Vec::new(),
        indent: 1,
        param_n: 0,
        folded: HashSet::new(),
        copies: HashMap::new(),
    };
    let structuring = e.run(&ret)?;
    let mut text = String::new();
    text.push_str("/* generated by rdec: machine code -> C. Any semantic gap is a\n");
    text.push_str("   compile-time refusal, never a comment that means 'trust me'. */\n");
    for x in &e.externs {
        let _ = writeln!(text, "{x}");
    }
    if !e.externs.is_empty() {
        text.push('\n');
    }
    text.push_str(&e.body);
    Ok(Out {
        text,
        externs: e.externs,
        structuring,
    })
}

struct Emitter<'a> {
    f: &'a Function,
    /// The declared shape of every callee this pipeline can name.
    protos: &'a Protos,
    /// The value this block's `TailReturn` returns, when the block ends in a
    /// tail call. Its defining `Op::Call` is then folded INTO the return
    /// (`return callee(args);`) instead of getting an assignment of its own, so
    /// the C says the one thing the machine code did. Set per block by the block
    /// loop; `NO_VAL` in every block that does not end in a tail call.
    tail: u32,
    /// This block ends in `Terminator::NoReturn`, so the call that closes it is
    /// the function's last statement: its site address goes on the line above
    /// (`Emitter::instr`), which leaves `abort(...);` standing as the statement
    /// the machine code executed. `false` in every other block, so no ordinary
    /// statement changes its spelling.
    noret: bool,
    /// The C spelling of `f.ret_ty`: the function's declared return type, which
    /// the `return` statement casts through (see `term`).
    ret: String,
    body: String,
    externs: Vec<String>,
    refusals: Vec<Refusal>,
    /// The indentation of the statements being printed now: 1 at the top level,
    /// +1 for every `if` / `while` the structuring walk nested around them.
    indent: usize,
    /// The entry's leading `Op::Param` instructions, which the structuring walk
    /// hoists into a statement of their own (see `block_params`), so `block_body`
    /// knows where the entry's own statements start.
    param_n: usize,
    /// Tail-call results folded into a `return`: no declaration, no statement.
    /// Computed once, before the declarations, so the decl list and the
    /// statement list agree on which values C ever names.
    folded: HashSet<u32>,
    /// A phi's edge copies, keyed by the PREDECESSOR block that must run them,
    /// in the order the phis appear. A predecessor keeps its copies at its own
    /// end no matter which statement it ends up inside.
    copies: HashMap<BbId, Vec<(u32, u32)>>,
}

/// C type name for an IR type, or a refusal for a type C cannot spell here.
fn c_ty(t: &Ty) -> Option<String> {
    Some(match t {
        Ty::Void => "void".into(),
        Ty::Bool => "uint8_t".into(),
        Ty::Int { w, sx } => match (w, sx) {
            (Width::W1, _) => "uint8_t".into(),
            (Width::W8, Sx::Signed) => "int8_t".into(),
            (Width::W8, Sx::Unsigned) => "uint8_t".into(),
            (Width::W16, Sx::Signed) => "int16_t".into(),
            (Width::W16, Sx::Unsigned) => "uint16_t".into(),
            (Width::W32, Sx::Signed) => "int32_t".into(),
            (Width::W32, Sx::Unsigned) => "uint32_t".into(),
            (Width::W64, Sx::Signed) => "int64_t".into(),
            (Width::W64, Sx::Unsigned) => "uint64_t".into(),
        },
        Ty::Float { w } => match w {
            FWidth::F32 => "float".into(),
            FWidth::F64 => "double".into(),
        },
        Ty::Ptr(_) => "uint64_t".into(),
        // A Flags value that survived elimination, a function type used as a
        // value, and a recovered struct all need the type-recovery pass, not a
        // guess here.
        Ty::Flags | Ty::Func(..) | Ty::Struct { .. } => return None,
    })
}

fn bits_of(t: &Ty) -> u32 {
    match t {
        Ty::Int { w, .. } => w.bits(),
        Ty::Bool => 1,
        Ty::Ptr(_) => 64,
        _ => 64,
    }
}

/// Emit `f` as a C function definition, declaring every callee in `protos` that
/// it calls. `protos` is built once for the translation unit from the same
/// declared shapes that spell the callees' own definitions, so a prototype and a
/// definition of `g` cannot disagree. `emit_c_sig` is this function; the name is
/// kept for the readability of call sites that say "with prototypes".
pub fn emit_c_protos(f: &Function, protos: &Protos) -> Result<Out, Vec<Refusal>> {
    emit_c_sig(f, protos)
}

impl<'a> Emitter<'a> {
    fn refuse(&mut self, at: u64, reason: impl Into<String>) {
        self.refusals.push(Refusal {
            at,
            reason: reason.into(),
        });
    }

    /// Stop at the first refusal: a body with a hole in it is not a body.
    fn bail(&self) -> Result<(), Vec<Refusal>> {
        if self.refusals.is_empty() {
            Ok(())
        } else {
            Err(self.refusals.clone())
        }
    }

    fn name(&self, v: u32) -> String {
        format!("x{v}")
    }

    fn ty_of(&self, v: u32) -> Ty {
        self.f.ty_of(v).unwrap_or(Ty::u64())
    }

    /// The value of `v` as a C expression, cast to a target type where the IR's
    /// width claim needs it.
    fn use_(&self, v: u32) -> String {
        self.name(v)
    }

    fn decl(&mut self, v: u32, ty: &Ty) {
        let t = c_ty(ty).unwrap_or_else(|| "uint64_t".into());
        // Initialised: a decompiler that emits `uint32_t x3;` and then reads it
        // on an uninitialised path gets a `-Wmaybe-uninitialized` warning, and a
        // warning under `-Werror` is the build refusing to lie.
        // `unused`: after the passes, a value can be computed and then read by
        // nothing (a phi whose merge is only consumed on a path DCE removed). The
        // decompiler emits the local because the machine code computes the value;
        // `-Werror -Wunused-but-set-variable` would otherwise refuse the file.
        let _ = writeln!(
            self.body,
            "    {t} {} __attribute__((unused)) = 0;",
            self.name(v)
        );
    }

    /// The region's C spellings of its tests: `x3`, `!(x3)`, and for a merged
    /// short-circuit chain `(x3) && !(x7)`. The value named is the one the
    /// terminator branched on; a negation is C's `!` on that value, never a
    /// rewrite of the comparison that produced it.
    fn cond_text(&self, conds: &[Cond]) -> String {
        let parts: Vec<String> = conds
            .iter()
            .map(|c| {
                let x = self.name(c.cond);
                if c.negated {
                    format!("!({x})")
                } else {
                    x
                }
            })
            .collect();
        if parts.len() == 1 {
            return parts[0].clone();
        }
        format!("({})", parts.join(") && ("))
    }

    /// The indentation of the statement list being printed: four spaces a level.
    fn pad(&self) -> String {
        " ".repeat(4 * self.indent)
    }

    /// What runs when a statement's text ends.
    fn after(&mut self, bb: BbId, c: Cont) -> Result<(), Vec<Refusal>> {
        match c {
            Cont::Exit | Cont::Next(_) => {}
            Cont::Goto(x) => {
                let p = self.pad();
                let _ = writeln!(self.body, "{p}goto l{x};");
            }
            Cont::Term => {
                let t = self.f.blocks[bb as usize].term.clone();
                self.term(bb, &t)?;
            }
        }
        Ok(())
    }

    /// Print a statement list at nesting level `ind`.
    fn stmts(&mut self, stmts: &[Stmt], ind: usize) -> Result<(), Vec<Refusal>> {
        for s in stmts {
            self.indent = ind;
            // A `while`'s label line goes on its IN-LOOP header copy (see the
            // rotated-loop comment in this function), so the head prints none:
            // printing it twice is a duplicate label, which clang refuses.
            if s.label && !matches!(s.kind, StmtKind::While { .. }) {
                let a = self.f.blocks[s.bb as usize].addr;
                let _ = writeln!(self.body, "l{}: /* at {a:#x} */", s.bb);
            }
            match &s.kind {
                StmtKind::Goto => {
                    let x = s.bb;
                    let p = self.pad();
                    let _ = writeln!(self.body, "{p}goto l{x};");
                    continue;
                }
                StmtKind::Params => {
                    self.block_params(s.bb)?;
                }
                StmtKind::Block => {
                    self.block_body(s.bb)?;
                    self.indent = ind;
                    self.after(s.bb, s.cont)?;
                }
                StmtKind::If { conds, then, els } => {
                    self.block_body(s.bb)?;
                    self.indent = ind;
                    let test = self.cond_text(conds);
                    let _ = writeln!(self.body, "{}if ({test}) {{", self.pad());
                    self.stmts(then, ind + 1)?;
                    self.indent = ind;
                    match els {
                        Some(e) => {
                            let _ = writeln!(self.body, "{}}} else {{", self.pad());
                            self.stmts(e, ind + 1)?;
                            self.indent = ind;
                            let _ = writeln!(self.body, "{}}}", self.pad());
                        }
                        None => {
                            let _ = writeln!(self.body, "{}}}", self.pad());
                        }
                    }
                    self.after(s.bb, s.cont)?;
                }
                StmtKind::While { conds, body } => {
                    // THE ROTATED LOOP. `while (c)` evaluates `c` before every
                    // trip, and `c` is a value the header's instructions
                    // compute, so the header's text has to appear in TWO
                    // places: once above the loop (the trip in from the
                    // region's entry) and once at the BOTTOM of the body (the
                    // back edge). Printing it only above the loop - the first
                    // cut of this pass - computed the tested value once and the
                    // C spun forever: T2's C side timed out on every corpus
                    // loop, 9 comparisons lost (`1403 | 13` against the
                    // `1412 | 4` floor) while the IR interpreter, which does
                    // read the graph, kept answering correctly.
                    //
                    // The duplication is in the TEXT, never in the work: the
                    // machine reaches a loop header once per trip plus the trip
                    // that leaves, and the two copies run exactly that many
                    // times - the peeled copy on the entry edge, the in-loop
                    // copy on every back edge. A store in a header is therefore
                    // still the machine's store, run the machine's number of
                    // times, and the block needs no purity gate.
                    //
                    // `l{bb}:` goes on the IN-LOOP copy, not the peeled one: a
                    // `goto` to the header means "test again", which is the
                    // in-loop copy. The peeled copy is entry-edge code, and
                    // jumping to it from inside the loop would leave the loop.
                    self.block_body(s.bb)?;
                    self.indent = ind;
                    let test = self.cond_text(conds);
                    let _ = writeln!(self.body, "{}while ({test}) {{", self.pad());
                    self.stmts(body, ind + 1)?;
                    self.indent = ind + 1;
                    if s.label {
                        let a = self.f.blocks[s.bb as usize].addr;
                        let _ = writeln!(self.body, "l{}: /* at {a:#x} */", s.bb);
                    }
                    self.block_body(s.bb)?;
                    self.indent = ind;
                    let _ = writeln!(self.body, "{}}}", self.pad());
                    self.after(s.bb, s.cont)?;
                }
            }
        }
        Ok(())
    }

    /// One block's statements: its instructions, then its phi edge copies. Both
    /// belong to the block, so both stay with it when the block becomes the head
    /// of an `if`, the guard of a `while`, or a label + `goto` block.
    fn block_body(&mut self, bb: BbId) -> Result<(), Vec<Refusal>> {
        let b = &self.f.blocks[bb as usize];
        // The entry's argument loads are a statement of their own, above the walk
        // (`StmtKind::Params`), so that the entry's phi edge copy - which belongs
        // to the entry, as in the flat spelling - prints AFTER them.
        let skip = if bb == self.f.entry { self.param_n } else { 0 };
        // A tail call's `Op::Call` is folded into the `return` that ends its
        // block (see `Emitter::tail`): no declaration, no assignment statement.
        // `x12 = g(...); return (int64_t)x12;` is two statements where the machine
        // code has one, and the local it declares is `set but not used` under
        // `-Werror`.
        self.tail = match b.term {
            Terminator::TailReturn(v) if self.tail_foldable(bb, v) => v,
            _ => crate::ir::NO_VAL,
        };
        // The block's last act is a call that never returns: its statement is
        // the function's final C statement, so it is spelled alone (see
        // `Emitter::noret`).
        self.noret = matches!(b.term, Terminator::NoReturn);
        for ins in b.instrs.iter().skip(skip) {
            if self.folded.contains(&ins.dest) {
                continue;
            }
            // A phi's value arrives by the edge-copy at the end of its
            // predecessor, so the phi op has no C statement: emitting one
            // (`dest = 0`, the initialiser's "no incoming edge" case) runs on
            // EVERY entry to the block - including the back edge - and wipes
            // the loop-carried value the latch just copied in. The corpus
            // masked this for as long as no runnable function had a phi (the
            // preheader's incoming value happened to be 0); T2, comparing the
            // newly-runnable loop functions, did not: 41 of 1335 runs, e.g.
            // `early_returns.O1::f(7,-1,40,3)` machine 0x2a, C 0, where the
            // `mov eax,0x2a` edge copy into the ret block was overwritten by
            // its own phi statement.
            if matches!(ins.op, Op::Phi(_)) {
                continue;
            }
            self.instr(ins)?;
            self.bail()?;
        }
        for (dest, src) in self.copies.get(&bb).cloned().unwrap_or_default() {
            let p = self.pad();
            let (d, s) = (self.name(dest), self.use_(src));
            let _ = writeln!(self.body, "{p}{d} = {s};");
        }
        Ok(())
    }

    /// The entry's `Op::Param` assignments, printed before the walk so the entry's
    /// edge copy has a statement to follow. Folding the argument loads into the
    /// entry's own statement would put that copy BEFORE them, and the function
    /// would return 0: this is exactly where the flat emitter's order mattered, and
    /// structuring has to keep it.
    fn block_params(&mut self, bb: BbId) -> Result<(), Vec<Refusal>> {
        self.tail = crate::ir::NO_VAL;
        self.noret = false;
        for ins in &self.f.blocks[bb as usize].instrs {
            if !matches!(ins.op, Op::Param) {
                break;
            }
            self.instr(ins)?;
            self.bail()?;
        }
        Ok(())
    }

    fn run(&mut self, ret: &str) -> Result<Report, Vec<Refusal>> {
        let entry = self.f.entry;
        let a0 = self.f.blocks[entry as usize].addr;
        let _ = writeln!(self.body, "/* function {} @ {a0:#x} */", self.f.name);
        let _ = write!(self.body, "{ret} {}(", self.f.name);
        let tys = c_param_tys(self.f);
        let n = tys.len();
        for (i, t) in tys.iter().enumerate() {
            // The DECLARED type wins: a Windows x64 float parameter is a bit
            // cell in the IR (Ty::u64, one W64 word per register) and a
            // `double`/`float` in C. The assignment at the `Op::Param` moves
            // bits through the memcpy helpers, never a value conversion.
            let _ = write!(self.body, "{t}", t = c_ty(t).unwrap_or("int64_t".into()));
            let _ = write!(self.body, " a{i} __attribute__((unused))");
            if i + 1 < n {
                let _ = write!(self.body, ", ");
            }
        }
        let _ = writeln!(self.body, ") {{");

        // Structure the CFG before a word of the body is printed: the label set is
        // a property of the whole function, and the `(void)&&lN;` uses below have
        // to name exactly the labels the text is going to define.
        self.param_n = self.f.blocks[entry as usize]
            .instrs
            .iter()
            .take_while(|i| matches!(i.op, Op::Param))
            .count();
        let mut st = crate::structure::structure_with_params(self.f, true);
        if !st.labels_defined() {
            // The tree names a label it never defines: not a spelling of this
            // function. Print the flat form and say so in the census - a region
            // the CFG did not prove stays a label and a `goto`, which is the whole
            // conservative contract of the pass, enforced on the output and not on
            // the patterns' intentions.
            st = crate::structure::flat_walk(self.f);
            st.report
                .reason("region whose text left a named label undefined (flat form)");
        }

        // Mark every label a `goto` lands on. A pass that folds a branch can leave
        // a block that nothing jumps to, and `-Werror -Wunused-label` then refuses
        // the file; the structuring walk drops the labels of the blocks it folded
        // into statements, so the labels printed below are exactly the ones the
        // text defines, and this is the list that keeps the file clean. `&&l` is
        // the GNU label-address extension, which clang accepts; evaluating it as a
        // `(void)` expression has no effect on the program.
        for (b, used) in st.label_blocks.iter().enumerate() {
            if *used {
                let _ = writeln!(self.body, "    (void)&&l{b};");
            }
        }
        let _ = writeln!(self.body);

        // Tail-call results folded into a `return`: no declaration, no statement
        // (see `block_body`).
        self.folded = self
            .f
            .blocks
            .iter()
            .filter_map(|b| match b.term {
                Terminator::TailReturn(v) if self.tail_foldable(b.id, v) => Some(v),
                _ => None,
            })
            .collect();

        // Locals for every value, in definition order, so the text reads like the
        // machine code it came from.
        for b in &self.f.blocks {
            for i in &b.instrs {
                if self.folded.contains(&i.dest) {
                    continue;
                }
                let ty = i.op.ty(self.f).unwrap_or(i.ty.clone());
                // A store has a destination in the IR (so the verifier can track
                // it) and no value in C: declaring `void x12;` is a clang error,
                // "variable has incomplete type 'void'".
                if ty == Ty::Void {
                    continue;
                }
                self.decl(i.dest, &ty);
            }
        }

        // phi destinations are assigned by edge-copies, so they need a value
        // before the block that owns them: the initialiser above is the "no
        // incoming edge" case, made visible as `= 0`.
        let mut copies: HashMap<BbId, Vec<(u32, u32)>> = HashMap::new();
        for b in &self.f.blocks {
            for i in &b.instrs {
                if let Op::Phi(pairs) = &i.op {
                    for (v, p) in pairs.iter() {
                        copies.entry(*p).or_default().push((i.dest, *v));
                    }
                }
            }
        }
        self.copies = copies;

        self.indent = 1;
        self.stmts(&st.stmts, 1)?;
        let _ = writeln!(self.body, "}}");
        Ok(st.report)
    }

    fn bin(&mut self, op: BinOp, a: u32, b: u32, ty: &Ty, at: u64) -> Option<String> {
        let (x, y) = (self.use_(a), self.use_(b));
        let w = bits_of(ty);
        let signed = matches!(ty, Ty::Int { sx: Sx::Signed, .. });
        let cnt = format!("({y} & {}ULL)", u64::MAX >> (64 - w.max(1)));
        let s = match op {
            BinOp::Add => format!("({x} + {y})"),
            BinOp::Sub => format!("({x} - {y})"),
            BinOp::Mul => format!("({x} * {y})"),
            // The high/low halves of a full-width product: the widened multiply
            // is the only C that spells it, and 128-bit ints are a compiler
            // extension - clang has them, gcc has them, MSVC does not. Say so.
            BinOp::MulHi => {
                self.refuse(
                    at,
                    "MulHi needs __int128 (a compiler extension)".to_string(),
                );
                return None;
            }
            BinOp::MulSmul => {
                self.refuse(at, "MulSmul (overflow flag) is not expressible".to_string());
                return None;
            }
            BinOp::UDiv => format!("({y} == 0 ? 0 : {x} / {y})"),
            BinOp::URem => format!("({y} == 0 ? 0 : {x} % {y})"),
            // x86 idiv traps on INT_MIN / -1; C calls it undefined behaviour.
            // Guarding both is what lets the differential compare answers.
            BinOp::SDiv if signed => {
                format!("({y} == 0 ? 0 : {y} == -1 ? (uint64_t)(0ULL - (uint64_t){x}) : {x} / {y})")
            }
            BinOp::SDiv => format!("({y} == 0 ? 0 : {x} / {y})"),
            BinOp::SRem if signed => format!("({y} == 0 ? 0 : {y} == -1 ? 0 : {x} % {y})"),
            BinOp::SRem => format!("({y} == 0 ? 0 : {x} % {y})"),
            BinOp::And => format!("({x} & {y})"),
            BinOp::Or => format!("({x} | {y})"),
            BinOp::Xor => format!("({x} ^ {y})"),
            BinOp::Shl => format!("({x} << {cnt})"),
            BinOp::Shr => format!("({x} >> {cnt})"),
            BinOp::Sar if signed => format!("({x} >> {cnt})"),
            BinOp::Sar => {
                // `sar` is an ARITHMETIC shift by definition. The operand is
                // unsigned here only because registers are modelled as one
                // unsigned W64 value - a modelling artifact, not a fact about the
                // operand (the same root cause as `movsx`). Reading the source as
                // signed at the operation's width is what the instruction does;
                // the old '((uint64_t)x >> cnt)' silently produced a LOGICAL shift
                // and T2 caught it: f(0x80000000,0x80000000,1,0) gave machine
                // 0xa0000000 vs C 0x60000000, 76 of 732 runs.
                let st = match w {
                    8 => "int8_t",
                    16 => "int16_t",
                    32 => "int32_t",
                    _ => "int64_t",
                };
                format!("(({st}){x} >> {cnt})")
            }
            BinOp::RotL => format!("(({x} << {cnt}) | ({x} >> (({w}ULL - {cnt}) & {w}_ULL)))"),
            BinOp::RotR => format!("(({x} >> {cnt}) | ({x} << (({w}ULL - {cnt}) & {w}_ULL)))"),
        };
        Some(s)
    }

    fn cast(&mut self, k: CastKind, to: &Ty, v: u32, at: u64) -> Option<String> {
        let src = self.ty_of(v);
        let x = self.use_(v);
        let sw = bits_of(&src);
        let s_ok = matches!(src, Ty::Int { sx: Sx::Signed, .. });
        let d_ok = matches!(to, Ty::Int { sx: Sx::Signed, .. });
        let t = match c_ty(to) {
            Some(t) => t,
            None => {
                self.refuse(at, format!("type {to:?} has no C spelling here"));
                return None;
            }
        };
        let s = match k {
            CastKind::Trunc => format!("({t})({x})"),
            CastKind::ZExt => format!("({t})((uint64_t){x} & {})", mask(sw)),
            // Sign extension: widen through the *signed* source width, which is
            // the only C spelling that propagates the sign bit portably.
            CastKind::SExt if s_ok => format!("({t})({x})"),
            CastKind::SExt => {
                // The source is unsigned in the IR, so its top bit is data and a
                // plain widening C cast would drop the sign extension. Refuse:
                // this is the sign-of-the-operand question, not a formatting one.
                self.refuse(
                    at,
                    format!("SExt from unsigned {src:?} of width {sw}: needs a signed source type"),
                );
                return None;
            }
            CastKind::Bitcast => {
                // A bitcast between an integer and a float REINTERPRETS; a C
                // cast CONVERTS, and the two answers are different numbers.
                // `(float)1065353216` is 1065353216.0f, and the bit pattern a
                // `movss` put in the cell is 1.0f. The memcpy helpers do it.
                let from_f = matches!(src, Ty::Float { .. });
                let to_f = matches!(to, Ty::Float { .. });
                match (from_f, to_f) {
                    (false, true) => match to {
                        Ty::Float { w: FWidth::F32 } => format!("rdec_bits_f32((uint64_t){x})"),
                        _ => format!("rdec_bits_f64((uint64_t){x})"),
                    },
                    (true, false) => match src {
                        Ty::Float { w: FWidth::F32 } => format!("rdec_f32_bits({x})"),
                        _ => format!("rdec_f64_bits({x})"),
                    },
                    // Int<->int and pointer views keep the plain C cast.
                    _ => format!("({t}){x}"),
                }
            }
            // Integer -> float. `(double)x` on an unsigned `x` converts the
            // unsigned value, and the machine's `cvtsi2sd` of a register the IR
            // calls signed answers the signed one: the cast goes through the
            // signed C type of the source width, which is the only spelling that
            // propagates the sign bit.
            CastKind::SIToFp => {
                if !s_ok {
                    self.refuse(
                        at,
                        format!("SIToFp from unsigned {src:?}: needs a signed source type"),
                    );
                    return None;
                }
                // The cast spelled into `({s})({x})` below: `s` must be a TYPE, not a
                // cast already. `"(int64_t)"` produced `((int64_t))(x)` - a cast of
                // nothing - and clang answers `expected expression`, so every signed
                // int -> float conversion (cvtsi2sd from a signed source) emitted C that
                // did not compile. No corpus function hits it, which is exactly why the
                // hermetic test in crates/arch-x86/tests/float_abi.rs exists.
                let s = "int64_t".to_string();
                match to {
                    Ty::Float { w: FWidth::F32 } => format!("(float)(({s})({x}))"),
                    _ => format!("(double)(({s})({x}))"),
                }
            }
            CastKind::UIToFp => match to {
                Ty::Float { w: FWidth::F32 } => format!("(float)((uint64_t)({x}))"),
                _ => format!("(double)((uint64_t)({x}))"),
            },
            // `cvttsd2si`/`cvttss2si`. The C cast `(int64_t)d` truncates toward
            // zero, which matches, BUT it is undefined behaviour once d is NaN
            // or outside the destination's range, where x86 answers the
            // destination's sign bit (0x80000000 / 0x8000000000000000). A bare
            // cast would hand the optimizer an out-of-range float and let it
            // invent an answer, so the guard is emitted: the comparison is
            // false for NaN, which is exactly when x86 answers the magic value.
            // (`!(d < hi)` is written so a NaN fails the range test rather than
            // passing it, and the code carries no NaN test that -ffast-math
            // could delete... which is why the comment names the rule.)
            CastKind::FpToSI => {
                let dw = bits_of(to);
                let magic = format!("0x{:x}ULL", 1u64 << (dw - 1));
                let (lo, hi) = if dw >= 64 {
                    // 2^63 is not representable as an int64_t literal; -9223372036854775808.0
                    // is, and the window is [-2^63, 2^63).
                    (-9_223_372_036_854_775_808f64, 9_223_372_036_854_775_808f64)
                } else {
                    let h = 2f64.powi(dw as i32 - 1);
                    (-h, h)
                };
                format!(
                    "((uint64_t)(!({x} >= {lo:e} && {x} < {hi:e}) ? {magic} : (uint64_t)(int64_t)({x})))"
                )
            }
            CastKind::FpToUI => {
                let dw = bits_of(to);
                let magic = format!("0x{:x}ULL", 1u64 << (dw - 1));
                let hi = if dw >= 64 {
                    18_446_744_073_709_551_616_f64
                } else {
                    2f64.powi(dw as i32)
                };
                format!("((uint64_t)(!({x} >= 0.0 && {x} < {hi:e}) ? {magic} : (uint64_t)({x})))")
            }
            // `cvtss2sd` / `cvtsd2ss`: a precision change of the VALUE, which is
            // what a C cast between the float types is. Every f32 is exactly an
            // f64, so the widening is lossless and the narrowing is the same
            // round-to-nearest the hardware does.
            CastKind::FPExt => format!("(double)({x})"),
            CastKind::FPTrunc => format!("(float)({x})"),
            // No `_` arm: every CastKind is spelled above, and the day a kind is
            // added the compiler says so here. A `_ => refuse(...)` arm that can
            // never fire is a refusal that reads like a guard.
        };
        let _ = d_ok;
        Some(s)
    }

    /// The C expression for a call: the callee spelled from its DECLARED shape,
    /// with the prototype declared in the same step. `instr` uses this for a call
    /// statement and `term` uses it for the one case where the call IS the
    /// statement a tail call's `return` needs (see `Emitter::tail`), so the two
    /// spellings of a call can never drift apart.
    ///
    /// `None` means a refusal was pushed. The callers stop at the first refusal
    /// (`bail`), so a refusal does not need to travel as an `Err` through here.
    fn call_expr(&mut self, ins: &crate::ir::Instr) -> Option<String> {
        let at = ins.addr;
        let (callee, args, ret_ty) = match &ins.op {
            Op::Call {
                callee,
                args,
                ret_ty,
                ..
            } => (callee, args.as_ref(), ret_ty),
            _ => {
                self.refuse(at, "call_expr on a non-call".to_string());
                return None;
            }
        };

        let name = match callee {
            Callee::Symbol(s) => c_name(s),
            Callee::Addr(a) => format!("rdec_fn_{a:x}"),
            Callee::Indirect(_) => {
                self.refuse(at, "indirect call: no signature model".to_string());
                return None;
            }
        };
        // A callee the pipeline named: call it as its DECLARED shape says,
        // and declare that shape before the call. The prototype is spelled
        // from the same `CalleeDecl` the callee's own definition is spelled
        // from, so the two cannot drift into `conflicting types`.
        if let Callee::Symbol(sym) = callee {
            let decl = self
                .protos
                .get(sym.as_str())
                .or_else(|| self.protos.get(&name))
                .cloned();
            let Some(d) = decl else {
                // A call that DEFINES NOTHING - the noreturn shape - can be
                // declared where a value call cannot: the declaration cannot
                // corrupt an answer, because there is no answer, and the
                // argument list we spell is the one this call uses. Windows
                // x64 is caller-clean, so the CRT's `abort(void)` reading
                // registers it ignores is the same call. The name is the front
                // end's declaration (`lift::noreturn_symbol`), the `noreturn`
                // attribute says out loud what the IR's `NoReturn` terminator
                // means, and clang then accepts a function that ends with
                // `__builtin_unreachable()` after this call.
                if *ret_ty == Ty::Void {
                    if args.len() != 4 {
                        // The four-register spelling below is the only thing
                        // this branch can declare; a call that passes a
                        // different count has no signature to spell.
                        self.refuse(
                                    at,
                                    format!(
                                        "noreturn call to {name} passes {} args; no declared shape to spell",
                                        args.len()
                                    ),
                                );
                        return None;
                    }
                    // The attribute goes AFTER the declarator, the way libc spells
                    // it (`void abort(void) __attribute__((noreturn));`): the line
                    // reads as the declaration of a VOID function first and the
                    // never-returns claim second, which is the order the model
                    // earns them in - `extern void abort(...)` is what makes the
                    // call a statement with no value, and the attribute is the
                    // claim that lets the body end without a `return`.
                    let proto = format!(
                                "extern void {name}(int64_t, int64_t, int64_t, int64_t) __attribute__((noreturn));"
                            );
                    if !self.externs.contains(&proto) {
                        self.externs.push(proto);
                    }
                    let a: Vec<String> = args
                        .iter()
                        .map(|v| format!("(int64_t){}", self.use_(*v)))
                        .collect();
                    return Some(format!("{name}({})", a.join(", ")));
                }
                // Named, and no declared shape: the identity is known and
                // the signature is not, which is the emitter's own kind of
                // gap. An `extern int64_t name(int64_t,int64_t,int64_t,
                // int64_t)` guess would COMPILE, and clash with the real
                // definition the moment both are in this file.
                self.refuse(
                    at,
                    format!("call to a sibling with no declared shape: {name}"),
                );
                return None;
            };
            let f_class = |t: &Ty| matches!(t, Ty::Float { .. });
            if f_class(&d.ret) != f_class(ret_ty) {
                self.refuse(
                    at,
                    format!(
                        "call to {name} answers {} but its declared shape returns {}",
                        c_ty(ret_ty).unwrap_or_else(|| "int64_t".into()),
                        c_ty(&d.ret).unwrap_or_else(|| "int64_t".into())
                    ),
                );
                return None;
            }
            if args.len() != d.params.len() {
                self.refuse(
                    at,
                    format!(
                        "call to {name} passes {} args; its declared shape has {}",
                        args.len(),
                        d.params.len()
                    ),
                );
                return None;
            }
            let proto = proto_text(&name, &d);
            if !self.externs.contains(&proto) {
                self.externs.push(proto);
            }
            let a: Vec<String> = args
                .iter()
                .zip(d.params.iter())
                .map(|(v, t)| match t {
                    // A W64 register cell feeding a float parameter moves
                    // BITS through the memcpy helper. A `(float)x` cast
                    // CONVERTS the integer and answers a different number,
                    // and `(int64_t)x` truncates: the same rule the
                    // parameter, load and store paths follow.
                    Ty::Float { w: FWidth::F32 } => {
                        format!("rdec_bits_f32((uint64_t){})", self.use_(*v))
                    }
                    Ty::Float { w: FWidth::F64 } => {
                        format!("rdec_bits_f64((uint64_t){})", self.use_(*v))
                    }
                    other => format!(
                        "({}){}",
                        c_ty(other).unwrap_or("int64_t".into()),
                        self.use_(*v)
                    ),
                })
                .collect();
            Some(format!("{name}({})", a.join(", ")))
        } else {
            // A target with no symbol: no identity, no declared shape, so
            // no prototype to write. The old spelling stays, loudly named by
            // the `rdec_fn_` prefix, and the gate's `externs` census counts it.
            if args.len() > 4 {
                self.refuse(
                            at,
                            format!("call to {name} with {} args: the 4-register signature model cannot express it", args.len()),
                        );
                return None;
            }
            let mut a: Vec<String> = args
                .iter()
                .map(|v| format!("(int64_t){}", self.use_(*v)))
                .collect();
            while a.len() < 4 {
                a.push("0".into());
            }
            let sig = format!(
                "extern {} {name}(int64_t, int64_t, int64_t, int64_t);",
                c_ty(ret_ty).unwrap_or("int64_t".into())
            );
            if !self.externs.contains(&sig) {
                self.externs.push(sig);
            }
            Some(format!("{name}({})", a.join(", ")))
        }
    }

    /// Can this block's `TailReturn(v)` fold its call into the `return`? Exactly
    /// when `v` is defined by a call IN THIS BLOCK and the terminator is that
    /// value's only use: then the C has one statement for the machine's one
    /// `jmp`, and the local the call would have needed is never declared.
    fn tail_foldable(&self, bb: crate::ir::BbId, v: crate::ir::VId) -> bool {
        match self.f.def(v) {
            Some((d, i)) => {
                d == bb
                    && matches!(self.f.block(d).instrs[i].op, Op::Call { .. })
                    && self.f.use_count(v) == 1
            }
            None => false,
        }
    }

    fn instr(&mut self, ins: &crate::ir::Instr) -> Result<(), Vec<Refusal>> {
        // Statements print at the current nesting level (see `pad`).
        let p = self.pad();
        let at = ins.addr;
        let v = ins.dest;
        let ty = ins.op.ty(self.f).unwrap_or(ins.ty.clone());
        let lhs = self.name(v);
        let expr: Option<String> = match &ins.op {
            Op::Const(c) => Some(format!(
                "({})(0x{:x}ULL)",
                c_ty(&ty).unwrap_or("uint64_t".into()),
                c & mask_of(bits_of(&ty))
            )),
            Op::ConstF(c) => Some(if c.is_nan() {
                // Rust prints a NaN as `NaN`, which is not C. `__builtin_nan("")`
                // is the quiet NaN clang folds to a constant, and it names the
                // intent instead of hiding it behind a magic bit pattern.
                "__builtin_nan(\"\")".into()
            } else if c.is_infinite() {
                if *c > 0.0 {
                    "__builtin_huge_val()".into()
                } else {
                    "(-__builtin_huge_val())".into()
                }
            } else {
                format!("{c:?}")
            }),
            // `undef` initialises to 0 and says so: the value is genuinely
            // undefined, the C has to be a number, and a comment on the same line
            // is where the difference lives.
            Op::Undef => Some("/* undef */ 0".into()),
            Op::Param => {
                // A parameter declared `float`/`double` is a BIT CELL in the
                // IR: the assignment reinterprets the C argument's bits
                // (memcpy), it does not convert its value. `(uint64_t)d`
                // truncates toward zero and answers a different number -- the
                // memcpy helpers are the guard, the same pair the float
                // Load/Store arms use for `movsd`/`movss`.
                let i = param_index(self.f, v).unwrap_or(0);
                match self.f.param_tys.get(i) {
                    Some(Ty::Float { w: FWidth::F64 }) => Some(format!("rdec_f64_bits(a{i})")),
                    Some(Ty::Float { w: FWidth::F32 }) => Some(format!("rdec_f32_bits(a{i})")),
                    _ => Some(format!("a{i}")),
                }
            }
            Op::Copy(s) => Some(self.use_(*s)),
            Op::Bin(o, a, b) => match self.bin(*o, *a, *b, &ty, at) {
                Some(s) => Some(s),
                None => return Err(self.refusals.clone()),
            },
            // Float arithmetic. The C type of the RESULT is the operands' float
            // type, and an F32 op must cast the result back to `float`: C's
            // usual arithmetic conversions promote float operands to double for
            // the arithmetic itself, so `(a + b)` at F32 would keep double
            // precision the machine threw away. The cast is the op's precision,
            // spelled where the semantics live.
            Op::FBin(o, a, b) => {
                let (x, y) = (self.use_(*a), self.use_(*b));
                let f32 = matches!(self.ty_of(*a), Ty::Float { w: FWidth::F32 });
                let at32 = |s: String| if f32 { format!("(float)({s})") } else { s };
                Some(at32(match o {
                    FBinOp::Add => format!("({x} + {y})"),
                    FBinOp::Sub => format!("({x} - {y})"),
                    FBinOp::Mul => format!("({x} * {y})"),
                    FBinOp::Div => format!("({x} / {y})"),
                    // x86's `maxsd` answers the SECOND operand when either side
                    // is NaN. C's ternary picks the false arm on a NaN
                    // comparison, which is the same answer -- the plain ternary
                    // IS x86's rule, and the claim is worth writing down because
                    // `fmax` (which ignores NaN) is NOT.
                    FBinOp::Max => format!("({x} > {y} ? {x} : {y})"),
                    FBinOp::Min => format!("({x} < {y} ? {x} : {y})"),
                    // The F32 form calls `sqrtf`, not `sqrt` on a promoted
                    // double: the latter is double rounding, and FMA contraction
                    // in a long expression is why this is not pedantry.
                    FBinOp::Sqrt => {
                        if f32 {
                            format!("__builtin_sqrtf({x})")
                        } else {
                            format!("__builtin_sqrt({x})")
                        }
                    }
                }))
            }
            // Float comparison. The predicate is C's own: an ordered comparison
            // is false and `!=` is true when either side is NaN, which is the
            // same answer the machine gives a `je`/`jne`/`ja`/`jb` reading of
            // `ucomisd`. The flag spellings that need the third bit (the
            // unordered case: `jp`/`jnp`, and `jae`/`jbe` pairs) are refused by
            // the lifter, so there is no unordered case left to spell here.
            Op::FCmp(c, a, b) => {
                let (x, y) = (self.use_(*a), self.use_(*b));
                Some(format!("({x} {} {y}) ? 1 : 0", c.c_op()))
            }
            Op::Un(o, a) => {
                let x = self.use_(*a);
                let w = bits_of(&ty);
                Some(match o {
                    UnOp::Not => format!("~{x}"),
                    UnOp::Neg => format!("(uint64_t)(0ULL - (uint64_t){x})"),
                    UnOp::Popcount => format!("__builtin_popcountll((uint64_t){x})"),
                    UnOp::Lzcnt => format!(
                        "((uint64_t){x} == 0 ? {w} : __builtin_clzll((uint64_t){x}) - (64 - {w}))"
                    ),
                    UnOp::Tzcnt => {
                        format!("((uint64_t){x} == 0 ? {w} : __builtin_ctzll((uint64_t){x}))")
                    }
                    UnOp::Sqrt => format!("__builtin_sqrt({x})"),
                })
            }
            Op::Cmp(c, a, b) => {
                let ta = self.ty_of(*a);
                let tb = self.ty_of(*b);
                // `Cmp` is the INTEGER comparison; `FCmp` is the float one and
                // carries C's unordered predicate. An integer predicate on
                // float operands is not a thing the machine does - the IR is
                // malformed - and the old code emitted an ORDERED C `<` there,
                // silently turning NaN comparisons into total-order answers.
                if matches!(ta, Ty::Float { .. }) || matches!(tb, Ty::Float { .. }) {
                    self.refuse(
                        at,
                        format!("integer comparison {c:?} on a float operand (FCmp is the float predicate)"),
                    );
                    return Err(self.refusals.clone());
                }
                let op = match c {
                    IntCmp::Eq => "==",
                    IntCmp::Ne => "!=",
                    IntCmp::Ult => "<",
                    IntCmp::Ule => "<=",
                    IntCmp::Ugt => ">",
                    IntCmp::Uge => ">=",
                    IntCmp::Lt => "<",
                    IntCmp::Le => "<=",
                    IntCmp::Gt => ">",
                    IntCmp::Ge => ">=",
                };
                // The ground truth is the interpreter (see its `Op::Cmp` arm):
                // signedness is a property of the PREDICATE, the width is
                // `width_of(a)`, and the operand type's signedness is a
                // display concern (§1.3). C's comparison follows the OPERANDS'
                // types, so where the two disagree, cast BOTH sides to the
                // predicate's signedness at the IR width - then C's answer is
                // the model's answer at the machine's width. The old code
                // refused every disagreement with a reason that interpolated a
                // literal empty string (`comparison Ult on a  operand`).
                let (x, y) = (self.use_(*a), self.use_(*b));
                let want_signed = matches!(c, IntCmp::Lt | IntCmp::Le | IntCmp::Gt | IntCmp::Ge);
                let (x, y) = if matches!(c, IntCmp::Eq | IntCmp::Ne)
                    || want_signed == matches!(ta, Ty::Int { sx: Sx::Signed, .. })
                {
                    (x, y)
                } else {
                    let w = match ta {
                        Ty::Int { w, .. } => w,
                        Ty::Bool => Width::W1,
                        other => {
                            self.refuse(
                                at,
                                format!("comparison {c:?} on a {other:?}-typed operand"),
                            );
                            return Err(self.refusals.clone());
                        }
                    };
                    let tt = Ty::Int {
                        w,
                        sx: if want_signed {
                            Sx::Signed
                        } else {
                            Sx::Unsigned
                        },
                    };
                    let Some(t) = c_ty(&tt) else {
                        self.refuse(
                            at,
                            format!("comparison {c:?} cast to {tt:?} has no C spelling"),
                        );
                        return Err(self.refusals.clone());
                    };
                    (format!("({t}){x}"), format!("({t}){y}"))
                };
                Some(format!("({x} {op} {y}) ? 1 : 0"))
            }
            Op::Flags(o) => {
                self.refuse(at, format!("flags value {o:?} survived flag elimination"));
                return Err(self.refusals.clone());
            }
            Op::GetFlag(f, _) => {
                self.refuse(at, format!("getflag {f:?} survived flag elimination"));
                return Err(self.refusals.clone());
            }
            Op::Cast(k, to, a) => match self.cast(*k, to, *a, at) {
                Some(s) => Some(s),
                None => return Err(self.refusals.clone()),
            },
            Op::Load(t, a) => {
                let x = self.use_(*a);
                let n = (bits_of(t) / 8).max(1);
                if c_ty(t).is_none() {
                    self.refuse(at, format!("load of {t:?} has no C spelling here"));
                    return Err(self.refusals.clone());
                }
                let raw = format!("load_{n}(RDEC_ADDR({x}))");
                Some(match t {
                    // A float load reads a BIT PATTERN out of memory, so the C
                    // must reinterpret (`rdec_bits_f64`), never convert:
                    // `(double)load_8(a)` converts an integer and answers a
                    // different number. This is the `movsd xmm, m64` case.
                    Ty::Float { w: FWidth::F32 } => format!("rdec_bits_f32({raw})"),
                    Ty::Float { w: FWidth::F64 } => format!("rdec_bits_f64({raw})"),
                    _ => raw,
                })
            }
            Op::Store(t, a, val) => {
                let (x, y) = (self.use_(*a), self.use_(*val));
                let n = (bits_of(t) / 8).max(1);
                // The mirror case: `movsd m64, xmm` stores the double's bits. A
                // `(uint64_t)d` cast TRUNCATES the value to an integer, which is
                // a conversion, and the guard against that is the helper.
                let bits = match t {
                    Ty::Float { w: FWidth::F32 } => format!("rdec_f32_bits({y})"),
                    Ty::Float { w: FWidth::F64 } => format!("rdec_f64_bits({y})"),
                    _ => format!("(uint64_t){y}"),
                };
                let _ = writeln!(self.body, "{p}store_{n}(RDEC_ADDR({x}), {bits});");
                None
            }
            Op::Select(c, t, f) => {
                let (cc, tt, ff) = (self.use_(*c), self.use_(*t), self.use_(*f));
                Some(format!("({cc} ? {tt} : {ff})"))
            }
            Op::Phi(_) => Some("/* phi: assigned by edge copy */ 0".into()),
            Op::Call { .. } => self.call_expr(ins),

            // A caller-saved register the call destroyed, READ after the call: the
            // answer depends on a value the model says is gone. This is the refusal
            // class the corpus has always reported, under the same name.
            Op::Clobber => {
                self.refuse(at, "opaque: clobbered by call".to_string());
                return Err(self.refusals.clone());
            }
            Op::Opaque(why) => {
                self.refuse(at, format!("opaque: {why}"));
                return Err(self.refusals.clone());
            }
        };
        match expr {
            Some(e) if ty == Ty::Void => {
                // A call that returns nothing is a *statement*: `(void)f(x);` is
                // fine C, `void x = ...;` is not.
                if self.noret && matches!(ins.op, Op::Call { .. }) {
                    // The call that ends a `NoReturn` block: its address on the
                    // line above, so the statement itself stands alone and reads
                    // as the function's last act.
                    let _ = writeln!(self.body, "{p}/* {at:#x} */");
                    let _ = writeln!(self.body, "{p}{e};");
                } else {
                    let _ = writeln!(self.body, "{p}{e}; /* {at:#x} */");
                }
            }
            Some(e) => {
                let t = c_ty(&ty).unwrap_or("uint64_t".into());
                let _ = writeln!(self.body, "{p}{lhs} = ({t}){e}; /* {at:#x} */");
            }
            None => {
                let _ = writeln!(self.body, "{p}/* {at:#x} */");
            }
        }
        Ok(())
    }

    fn term(&mut self, bb: crate::ir::BbId, t: &Terminator) -> Result<(), Vec<Refusal>> {
        // Statements print at the current nesting level (see `pad`).
        let p = self.pad();
        match t {
            Terminator::Jump(bb) => {
                let _ = writeln!(self.body, "{p}goto l{bb};");
            }
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let c = self.use_(*cond);
                let _ = writeln!(
                    self.body,
                    "{p}if ({c}) goto l{then_bb}; else goto l{else_bb};"
                );
            }

            Terminator::Switch { index, targets } => {
                // A computed jump whose destinations are the CFG's own successors:
                // every slot gets a `case` naming the same label the rest of the
                // emitted C uses for that block, so the C's control flow and the
                // CFG's edges are the same N edges - the emitter adds no
                // destination and drops none. An index past the end of the table
                // has no destination in this function, so the emitted C traps:
                // that is the model's standing answer, and T2 compares a trap.
                //
                // `default` is required: C's `switch` on an integer with no
                // matching case falls through the statement, and the machine
                // executing a table slot the object does not have is a fault, not
                // a ride-through.
                if targets.is_empty() {
                    // `verify_ssa` refuses this shape outright, so no verified
                    // function reaches here. If one ever does, the emitted C traps
                    // on every index: a body that lost all its case labels is not
                    // a body, and it must not quietly compile.
                    let _ = writeln!(
                        self.body,
                        "{p}__builtin_trap(); /* rdec: switch over no target at all (a partially resolved table) */"
                    );
                    return Ok(());
                }
                let idx = self.use_(*index);
                let _ = writeln!(self.body, "{p}switch ({idx}) {{");
                for (k, &bb) in targets.iter().enumerate() {
                    let _ = writeln!(self.body, "{p}case {k}: goto l{bb};");
                }
                let _ = writeln!(
                    self.body,
                    "{p}default: __builtin_trap(); /* rdec: index past the end of the jump table */"
                );
                let _ = writeln!(self.body, "{p}}}");
            }
            Terminator::Return(v) => {
                if *v == crate::ir::NO_VAL {
                    let _ = writeln!(self.body, "{p}return;");
                } else {
                    let x = self.use_(*v);
                    // The cast is to the DECLARED return type, not to int64_t.
                    // The declared type is the width the answer reaches `RAX` at
                    // (the lifter records it from the machine code: a 32-bit
                    // write zeroes bits 63:32), and `verify` checks the value
                    // fits it. Casting through it is what makes the emitted C
                    // return the same bits the machine leaves in RAX: `(uint32_t)`
                    // keeps the zero-extension the hardware performed, where the
                    // old unconditional `(int64_t)` sign-extended a 32-bit answer
                    // into bits that the machine code had already cleared.
                    //
                    // A `void` function that returns a value is the IR lying about
                    // itself, and `(void)x` is not C: keep the old int64_t spelling
                    // for that shape rather than inventing a cast.
                    let rc: &str = if self.ret == "void" {
                        "int64_t"
                    } else {
                        &self.ret
                    };
                    let _ = writeln!(self.body, "{p}return ({rc}){x};");
                }
            }
            // `return callee(args);` — the machine code's `jmp` and the C's
            // `return` are one control-flow edge, so the C is one statement: the
            // call expression is spelled by `call_expr`, the same function a call
            // STATEMENT uses, and the local it would have assigned is never
            // declared (`Emitter::tail`). The address in the comment is the
            // `jmp`'s, so the line points at the instruction that earned it.
            Terminator::TailReturn(v) => {
                if !self.tail_foldable(bb, *v) {
                    self.refuse(
                        0,
                        format!(
                            "tail call returns v{v}, which this block cannot fold into a return"
                        ),
                    );
                    return Err(self.refusals.clone());
                }
                let Some((dbb, i)) = self.f.def(*v) else {
                    self.refuse(0, format!("tail call returns an undefined value v{v}"));
                    return Err(self.refusals.clone());
                };
                let ins = self.f.block(dbb).instrs[i].clone();
                let at = ins.addr;
                let Some(e) = self.call_expr(&ins) else {
                    return Err(self.refusals.clone());
                };
                // Cast through the DECLARED return type, exactly as `Return` does:
                // the callee answers in RAX and the caller's answer is the same
                // bits, at the width the caller's own machine code declares.
                let rc: &str = if self.ret == "void" {
                    "int64_t"
                } else {
                    &self.ret
                };
                // The `jmp`'s address on the line ABOVE, not after the `;`: this
                // line is the function's return statement, and the one piece of C
                // the whole file must say plainly is `return callee(args);`. A
                // trailing comment makes the statement a line that ends in `*/`,
                // which is how a reader grepping the output for the tail call
                // loses it.
                let _ = writeln!(self.body, "{p}/* {at:#x} tail call */");
                let _ = writeln!(self.body, "{p}return ({rc}){e};");
            }
            // A call that never returns. The call itself is an ordinary statement
            // (it defines nothing), and this is the end of the function: C needs a
            // value from a path that has none, and `__builtin_unreachable` is the
            // honest spelling of "control never gets here" — a fall-through to a
            // `return` would invent an answer the machine code never produces.
            Terminator::NoReturn => {
                let _ = writeln!(
                    self.body,
                    "{p}__builtin_unreachable(); /* rdec: the call never returns */"
                );
            }
            Terminator::Unreachable => {
                let _ = writeln!(self.body, "{p}__builtin_unreachable();");
            }
            // Unmodelled control flow: the machine code goes somewhere we did not
            // resolve. `abort()` is the loud version; a fallthrough would be a lie.
            Terminator::Exit => {
                let _ = writeln!(
                    self.body,
                    "{p}__builtin_trap(); /* rdec: unmodelled control flow (Exit) */"
                );
            }
        }
        Ok(())
    }
}

fn param_index(f: &Function, v: u32) -> Option<usize> {
    f.params.iter().position(|p| *p == v)
}

fn mask_of(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn mask(bits: u32) -> String {
    if bits >= 64 {
        "0xffffffffffffffffULL".into()
    } else {
        format!("0x{:x}ULL", (1u64 << bits) - 1)
    }
}

/// A C identifier from a machine-code symbol: C++ names are full of `$`, `?`,
/// `@` and `:`, none of which C allows.
fn c_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The load/store helpers, at the widths the IR uses.
/// The shared definitions of a translation unit, emitted ONCE: flat memory, the
/// address mask, and the width-typed load/store helpers.
///
/// The "once" is the point. A decompiler that puts `static uint8_t rdec_mem[]`
/// inside every function gets a redefinition error the moment two functions share
/// a file, which is the normal case; and the mask keeps every modelled address in
/// range, so an out-of-range access in machine code becomes a deterministic
/// in-range one in C instead of undefined behaviour the differential cannot compare.
///
/// The size here and `MEM_SIZE` in Rust are the same number; the test
/// `prelude_and_mem_size_agree` is what keeps them so.
pub const PRELUDE: &str = "\
#include <stdint.h>
#include <string.h>

/* rdec declares every callee it names with the shape the model has for it - the
four Windows x64 register arguments, and the return kind the front end declared.
When such a name is ALSO a C library builtin (`abort`, `exit`), clang's
builtin-aware -Wincompatible-library-redeclaration calls that a conflict: it is
not one. The declaration states the CALL SITE's shape, which is what the file
being compiled is a record of; on Windows x64 the argument registers are
caller-clean, so a callee that reads none of them is called the same way, and a
callee that reads them is called with exactly the values the machine code put
there. Silencing it here, and only this, is what lets the emitted C name
`abort` and keep the arguments it was called with. */
#pragma clang diagnostic ignored \"-Wincompatible-library-redeclaration\"

static uint8_t rdec_mem[1 << 20];
#define RDEC_ADDR(a) ((uint64_t)(a) & ((1 << 20) - 1))

static inline __attribute__((unused)) uint64_t load_1(uint64_t a){uint8_t b=0;memcpy(&b,&rdec_mem[a],1);return b;}
static inline __attribute__((unused)) uint64_t load_2(uint64_t a){uint16_t b=0;memcpy(&b,&rdec_mem[a],2);return b;}
static inline __attribute__((unused)) uint64_t load_4(uint64_t a){uint32_t b=0;memcpy(&b,&rdec_mem[a],4);return b;}
static inline __attribute__((unused)) uint64_t load_8(uint64_t a){uint64_t b=0;memcpy(&b,&rdec_mem[a],8);return b;}
static inline __attribute__((unused)) void store_1(uint64_t a,uint64_t v){uint8_t b=(uint8_t)v;memcpy(&rdec_mem[a],&b,1);}
static inline __attribute__((unused)) void store_2(uint64_t a,uint64_t v){uint16_t b=(uint16_t)v;memcpy(&rdec_mem[a],&b,2);}
static inline __attribute__((unused)) void store_4(uint64_t a,uint64_t v){uint32_t b=(uint32_t)v;memcpy(&rdec_mem[a],&b,4);}
static inline __attribute__((unused)) void store_8(uint64_t a,uint64_t v){memcpy(&rdec_mem[a],&v,8);}

/* Float <-> bit pattern, by memcpy, never by a C cast: `(double)bits` converts
a number and `(uint64_t)d` truncates a value, and the decompiler's job at a
`movsd`/`cvtsd2si` is to move BITS, not to convert a value. A union type-pun
would be the other option; memcpy is the spelling C guarantees (6.7.2 p6) and
it is what the load_ and store_ helpers above already do. */
static inline __attribute__((unused)) double rdec_bits_f64(uint64_t b){double d=0.0;memcpy(&d,&b,8);return d;}
static inline __attribute__((unused)) float rdec_bits_f32(uint64_t b){uint32_t u=(uint32_t)b;float f=0.0f;memcpy(&f,&u,4);return f;}
static inline __attribute__((unused)) uint64_t rdec_f64_bits(double d){uint64_t b=0;memcpy(&b,&d,8);return b;}
static inline __attribute__((unused)) uint64_t rdec_f32_bits(float f){uint32_t u=0;memcpy(&u,&f,4);return u;}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_and_mem_size_agree() {
        // The C prelude writes the size as a literal; drift here means the C
        // buffer and the Rust address mask disagree, and an access that is in
        // range for one is out of range for the other.
        assert_eq!(MEM_SIZE, 1 << 20);
        assert!(PRELUDE.contains("rdec_mem[1 << 20]"));
        assert!(PRELUDE.contains("& ((1 << 20) - 1)"));
    }

    /// The emitter's contract, tested on the smallest function that has each
    /// shape: a phi at a join becomes an edge-copy, an unmodelled thing is a
    /// refusal, and `Exit` compiles to a trap rather than a lie.
    #[test]
    fn a_join_becomes_edge_copies_and_a_refusal_is_loud() {
        use crate::ir::{Function, Op, Terminator};
        use crate::ty::Ty;
        let mut f = Function::new("t", 0);
        // The entry block is the code's first block. A fresh `Block` starts life
        // with `Terminator::Unreachable`, so taking a NEW block for `a` left the
        // entry with no edge to any of this code: the whole body was
        // unreachable, the structuring walk (which walks the reachable CFG, as
        // the machine does) printed the entry's `__builtin_unreachable()` and
        // nothing else, and the needles below had no text to match.
        let a = f.entry;
        let b = f.new_block(4);
        let c = f.new_block(8);
        let p0 = f.emit(a, Op::Param, Ty::u64(), 0);
        let one = f.emit(a, Op::Const(1), Ty::u64(), 0);
        let t = f.emit(a, Op::Bin(crate::ir::BinOp::Add, p0, one), Ty::u64(), 0);
        let cnd = f.emit(a, Op::Cmp(crate::ir::IntCmp::Eq, p0, one), Ty::Bool, 0);
        f.block_mut(a).term = Terminator::Branch {
            cond: cnd,
            then_bb: b,
            else_bb: c,
        };
        let phi = f.new_value(Ty::u64());
        f.block_mut(b).instrs.insert(
            0,
            crate::ir::Instr::new(phi, Op::Phi(vec![(t, a)].into_boxed_slice()), Ty::u64(), 4),
        );
        f.block_mut(b).term = Terminator::Return(phi);
        f.block_mut(c).term = Terminator::Return(p0);
        f.params.push(p0);

        let out = emit_c(&f).expect("a pure integer function with a join must emit C");
        // The phi is not in the text; the copy is, at the end of its predecessor.
        assert!(!out.text.contains("phi %"), "phi leaked into C text");
        assert!(
            out.text.contains("x1 = x0;") || out.text.contains("= x"),
            "no edge copy"
        );
        // The branch has to be IN the C. At HEAD this fixture's two edges printed
        // as `if (x) goto l1; else goto l2;`; M5's structuring prints the same two
        // edges as a C `if`, which is the whole point of the pass, so the needle
        // is the condition. A build that lost the branch - one arm, one return -
        // has no `if (` and is red.
        assert!(out.text.contains("if ("), "no branch in C text");

        // An opaque value is a refusal, not a comment.
        let mut g = Function::new("u", 0);
        let e = g.entry;
        g.emit(e, Op::Opaque("nope".into()), Ty::u64(), 0);
        g.block_mut(e).term = Terminator::Return(rdec_ir_no_val());
        let err = emit_c(&g).expect_err("an opaque value must be refused, not emitted");
        assert!(err[0].reason.contains("nope"));
    }

    fn rdec_ir_no_val() -> u32 {
        crate::ir::NO_VAL
    }

    /// The declared return type is the width the answer is returned AT, so the
    /// `return` casts through it: a `uint32_t` function keeps the zero-extension
    /// a 32-bit machine write performed, where an unconditional `(int64_t)` cast
    /// sign-extended bits the machine code had already cleared.
    #[test]
    fn the_return_cast_is_the_declared_return_type() {
        let mut f = Function::new("r32", 0);
        let e = f.entry;
        let a = f.emit(e, Op::Param, Ty::u32(), 0);
        f.params.push(a);
        let one = f.emit(e, Op::Const(1), Ty::u32(), 0);
        let s = f.emit(e, Op::Bin(BinOp::Add, a, one), Ty::u32(), 0);
        f.block_mut(e).term = Terminator::Return(s);
        f.ret_ty = Ty::u32();
        let out = emit_c(&f).expect("a uint32_t function emits C");
        assert!(out.text.contains("uint32_t r32("), "{}", out.text);
        assert!(
            out.text.contains("return (uint32_t)x2;"),
            "the return casts through the declared type: {}",
            out.text
        );

        // The 64-bit case is unchanged: int64_t in, int64_t out.
        let mut g = Function::new("r64", 0);
        let e = g.entry;
        let a = g.emit(e, Op::Param, Ty::u64(), 0);
        g.params.push(a);
        g.block_mut(e).term = Terminator::Return(a);
        g.ret_ty = Ty::i64();
        let out = emit_c(&g).expect("an int64_t function emits C");
        assert!(out.text.contains("return (int64_t)x0;"), "{}", out.text);
    }
}
