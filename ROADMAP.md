# RDec — from-scratch x86-64 → C decompiler in Rust

**Roadmap v2 · 2026-11-01 · supersedes v1.**
v1 is preserved in git (`git show e329819:ROADMAP.md`). This revision comes from
an audit that ran the code, not only read it. §9 is the audited state of the
tree, with reproduced bugs. §10 lists each change from v1 and the reason for it.

---

## 0. Thesis, and the claim made precise

Goal (unchanged): a production-grade decompiler whose distinguishing property
is **checked equivalence**. Every decompiled function ships with evidence that
its C behaves like the machine code it came from.

v1 promised "provably equivalent output" and a "machine-checked equivalence
proof", but every oracle it planned is a *test*. Tests are evidence, not proof,
and a verification-first project cannot afford to overclaim. So the claim is
graded, and the grade is printed per function in the output and in JSON:

| Level | Claim | Requires |
|---|---|---|
| **E0** | none | The function contains `Opaque`, `Exit`, an unresolved indirect jump, or exceeded a budget. C is still produced, with the gaps listed. |
| **E1** | tested-equivalent | No E0 conditions. T2 passes ≥ 10³ vectors (boundary + random + coverage-guided) at `-O0` **and** `-O2`, with UBSan silent. |
| **E2** | bounded-proved | E1, plus SMT proves lifted IR ≡ final IR for all inputs, with loops unrolled to *k* (*k* printed). |
| **E3** | proved | E1, plus an unbounded SMT proof using loop synchronization (product program + invariants). |

**Trusted base for E2/E3**, stated so it can be attacked:
- the lifter's per-instruction semantics (tested against silicon by T1L, not proved),
- the SMT encoding of the IR (differentially tested against the IR interpreter),
- the C printer (kept syntax-directed and covered by T2),
- the solver.

**Equivalence, defined.** Take the recovered signature and ABI, and any initial
state on which the machine code does not fault. Both versions must produce:
- the same return value(s),
- the same final contents of all memory outside the function's own frame,
- the same sequence of external calls (callee, arguments, and the return values fed back),
- preserved callee-saved registers,
- the same termination (return vs. noreturn call).

Inputs on which the machine code faults (#DE, #GP, #PF) are outside the claim.
On every input where the machine code is defined, the emitted C must have no UB.

**Defensible near-term claim:** "every function carries an evidence level: E1 on
modern clang and MSVC codegen, E2 for loop-free functions." No current tool
occupies that niche, and every milestone below advances it.

---

## 1. Non-negotiables

### 1.1 The oracle stack

v1 had an oracle for the flag model (T1) and one for end-to-end output (T2),
and nothing in between. Every middle-end pass sits between two oracles that
cannot see it, and the audit measured the cost: the existing passes contain
silent miscompiles that the verifier accepts (§9). The fix is one component, an
**IR interpreter**, with four oracles built on top of it.

| Tier | Oracle | Validates | Lands |
|---|---|---|---|
| T0 | Unit tests + golden IR dumps | pass mechanics | M0 |
| T1 | Flag model vs real EFLAGS (`pushfq` harness). **Exists.** | x86 flag model | M0 ✓, extended M2 |
| **T1R** | Rewrite tables vs the T1-checked model: exhaustive at 8 bits (all *a*, *b*, carry-in, flag seeds), sampled at 16/32/64 | jcc fusion, `cmp_pair`, idiom tables | M2 |
| **T1L** | Lifter vs silicon: run each instruction form's bytes in a sandbox page from random register, flag and memory states. Run the lifted IR in the interpreter from the same state. Compare GPRs, defined flags, xmm, touched memory. Faults are caught by a vectored handler and compared as faults. | lifter, every operand form | M2 |
| **TV** | Per-pass translation validation: under `--validate` and in CI, evaluate the IR before and after every pass on *K* random states. A difference names the pass, the function and the input. | every middle-end pass | M0 |
| T2 | Differential execution. The original object and the recompiled C are linked into one harness, with external calls bound to recording stubs. Built at `-O0` and `-O2` with clang and gcc, under UBSan. | lifter → C end to end, incl. C semantics | M4 |
| T3 | `-std=c11 -Wall -Werror` on every emitted file | non-compilable output | M4 |
| T4 | Verifier after **every** pass. Type errors are hard; warnings are escalated at phase boundaries. | malformed IR | M0 |
| T5 | Fuzzing: loader/decoder on random bytes; random *valid IR* through the pipeline under TV; csmith/yarpgen C → compile matrix → T2 | panics, exotic codegen | M1, M3, M9 |
| **T6** | SMT translation validation: lifted IR ≡ final IR | E2/E3 evidence | P track |

Rules:
- Every oracle ships with a sabotage test proving it can fail. T1 already does
  this, and every tier must.
- A pass enters the default pipeline only once TV covers it and T0 has a fixture
  for each of its rewrite rules.
- No soft error survives a phase boundary.
- An oracle must not share code with its subject. T1R checks lowerings against
  the T1-measured model, not against the lowering's own helpers. The SMT
  encoding is checked against the interpreter, never against itself.

### 1.2 Flags

Keep v1's central decision. Flag-defining ops produce one SSA value of type
`Flags`, read only through `get_flag`. That makes dead flags plain DCE and
fusion a table. The audit found the representation underspecified, and v2 fixes
it:

1. **Flags ops reference their integer result** (`res: VId`), because ZF, SF and
   PF are functions of the result. v1's `FlagsOp` has no link to the result, so
   the lowering compares the `Flags` value itself to zero (§9 B5).
2. **Carry/borrow-in is an operand** (`cin: VId`, a Bool), not a `bool` field.
   `adc`/`sbb` chains are dynamic, and v1 cannot represent them.
3. **Partial updates take the previous flags** (`prev: VId`). This covers
   `inc`/`dec` (CF kept), `bt*` (CF only), rotates (CF/OF only) and
   *variable-count shifts*: a count of 0 leaves every flag unchanged, so each
   bit is `count == 0 ? prev.bit : new.bit`.
4. **Signedness lives in the predicate.** `SF ≠ OF` after `cmp` is `slt` by
   construction. v1 picks signed or unsigned from the operand's type, so `jl` on
   a register typed unsigned becomes an unsigned compare.
5. **Fusion matches the whole boolean tree at its root.** The current
   implementation lowers each `get_flag` alone, which leaves the jcc matrix
   unreachable (§9 B3).
6. **The exact fallback is total.** Every (FlagsOp × flag) pair has an exact
   lowering, including PF (`__rdec_parity8(res)`), AF, and OF for every op.
   T1R enumerates every pair to prove totality. A `Flags`-typed value that
   reaches C emission is an internal error, not a note.
7. **Flags through phis.** Cross-block flag reuse (tail-merged code, carry
   chains spanning blocks) splits the `Flags` phi into per-bit Bool phis for the
   bits actually read.
8. **FP compares produce flags.** `ucomis*`/`comis*` set ZF, PF and CF. `jp`
   after `ucomisd` (the NaN check) is how real code reads PF.
9. **Undefined flags** (per the SDM) lower to `undef`. T1 runs on both an Intel
   and an AMD host: undefined bits differ by vendor, and defined bits must not.

### 1.3 IR semantics (new)

- **Signless integers** `i1…i64`. Signedness goes on operators (`sdiv/udiv`,
  `slt/ult`, `sar/shr`, `sext/zext`). The signedness of a *variable* is an M6
  display inference. v1 encodes signedness in both the type and the predicate,
  which is how item 4 above goes wrong.
- **Written semantics.** `docs/ir-semantics.md` defines every op; the interpreter
  is its executable form. Covered cases:
  - shift by ≥ width: the lifter masks explicitly, and the IR op is defined for every count;
  - division by zero and `INT_MIN / -1`: trap, which gives the function a may-fault effect;
  - `undef`: an arbitrary but fixed value (`freeze` semantics, no poison).
- **Explicit memory ordering.** `Load`, `Store` and `Call` thread a memory token,
  or MemorySSA is built before any code motion. M4's expression inlining, DSE
  and load forwarding all consume it.
- **Calls carry effects:** ABI clobbers, memory effect (none / argmem / read /
  readwrite), noreturn, may-unwind.
- **Constants carry their width.** Today every `Const` is `u64`.
- **A per-pass index instead of use-lists.** Build a def/use index in O(n) when a
  pass starts and discard it when the pass ends. This keeps v1's "no stale
  use-lists" property without an O(n) scan per query, which makes DCE O(n²) per
  sweep. Without the index, M9's 10⁵-instruction budget is unreachable.
- **The fixpoint driver reports non-convergence as a gap.** It must not stop
  silently after 32 rounds, especially since several passes make one rewrite per
  call.

### 1.4 Memory, stack, frame (v1 §0.3, kept)

v1's local slot promotion stands: provably static frame offsets become
variables, and anything address-taken, escaped or dynamically indexed stays
memory. Additions:
- The stack-pointer delta is computed at every program point from M2 on. An
  unknown delta anywhere makes the function E0.
- Win64 shadow space and the SysV red zone are ABI facts, not heuristics.
- Stack probes (`__chkstk`, `___chkstk_ms`, `__alloca`) and canaries
  (`__security_cookie`/`__security_check_cookie`,
  `__stack_chk_guard`/`__stack_chk_fail`) are recognized by name plus shape, and
  each has a corpus case.

### 1.5 Loud by design (v1 §0.4, kept)

Additions:
- **Per-function isolation.** Each function runs under `catch_unwind` with time
  and memory budgets; an overrun produces E0 with the reason. Decompilers ingest
  hostile input, and malware analysis is a primary use.
- **Determinism.** Output is byte-identical across runs and thread counts; CI
  diffs two runs.
- **The confidence report becomes the evidence record:** level, gaps, budgets
  used, oracle versions.

### 1.6 Output contract (new)

Emitted C is C11 + `<stdint.h>` + one generated header, `rdec.h` (helpers and
intrinsic shims). Equivalence is claimed only under
`-fwrapv -fno-strict-aliasing -fno-delete-null-pointer-checks`, with clang or
gcc. The header documents the flags and the T2 build enforces them.

Why: x86 `add` wraps, but signed `+` in C is UB on overflow, and an optimizing
compiler exploits it. Emitting every op as unsigned arithmetic with casts would
be correct but unreadable. Two well-known flags are the honest trade.

UB the emitter must still avoid by itself:
- shift count ≥ width: mask or use a helper;
- unaligned typed access where alignment is unproven: use `memcpy`-based
  `rdec_load32`-style helpers;
- out-of-bounds pointer arithmetic: keep arithmetic on `uintptr_t` until
  provenance is proven;
- sequencing: v1's no-inline-on-alias rule.

T2 at `-O0` alone does not detect any of this, because `-O0` does not exploit
UB. That is why E1 requires `-O2` and UBSan.

---

## 2. Architecture

Strict one-way layering. `rdec-core` links against no x86 type, and the Cargo
dependency graph enforces it.

```
crates/bin       loader (goblin): PE/COFF, ELF; sections, symbols, relocations,
                 imports; Win64 .pdata/.xdata and ELF .eh_frame as function bounds   [M1]
crates/arch-x86  decode (iced-x86), lift, ABIs (Win64, SysV), stack delta, frame,
                 idioms, EFLAGS model (flags.rs — exists)
crates/core      IR, interpreter, verifier, CFG/dominators/loops, SSA, passes,
                 structuring, C emission                          (IR + passes exist)
crates/types     type libraries (libc, CRT, Win32 subset) generated from headers;
                 analyst type database                                               [M4]
crates/verify    T1 (exists), corpus generator (exists), T1R, T1L, TV, T2 runners,
                 fuzz targets, SMT bridge (T6)
crates/cli       rdec: flags / selftest / corpus exist; disasm / lift / ir / cfg /
                 decompile / verify / status / diff to come
```

Dependencies: `iced-x86` (MIT; note that `iced` on crates.io is an unrelated GUI
toolkit) and `goblin` (BSD-3). clang, gcc, LLVM tools and the SMT solver
(bitwuzla or z3, driven as an external process over SMT-LIB2) are test- and
evidence-time tools only. None is linked into the product.

Pipeline: `lift → SSA (versioned phi) → prune → flag-elim → fold / copy-prop /
DCE → frame-slot mem2reg → idioms / strength reduction → structure → cgen`.

Every pass is `fn(&mut Function) -> bool`, run to a fixpoint. The driver runs
the verifier after every pass unconditionally, and TV after every pass under
`--validate` and in CI.

---

## 3. Corpus and ground truth (new section)

The corpus defines what "works" means, so it is specified like code.

- **Compiler matrix:** {clang, gcc, MSVC} × {O0, O1, O2, O3/Os/Oz; MSVC /Od /O1 /O2}.
  - The current corpus is clang-only, and compiler diversity is the largest
    generalization risk.
  - MSVC codegen is most of what Windows users decompile: `/GS` cookies,
    `__chkstk`, CFG guard dispatch, SEH, and its own switch and jump-table shapes.
  - Host status: clang 22.1.8 and llvm-objdump are on PATH; gcc and `cl` are not.
    gcc → WSL or Linux CI; MSVC → VS Build Tools.
- **Ground truth from debug info.** Build every object with `-g` / `/Zi` and
  decompile a stripped copy. Score against DWARF/PDB (signatures, types, names,
  function bounds), and generate T2 inputs from the true prototypes. Debug info
  is never decompiler input in a scored run. (Today `rdec corpus` compiles with
  `-O<n> -std=c17 -c`, without `-g`.)
- **Tiers:**
  - feature snippets: 30 snippets × 4 opt levels = 120 objects, exists;
  - real programs: zlib, lua, sqlite3, xxd, tcc, busybox applets;
  - generated programs: csmith, yarpgen.
- **Lifter priority comes from frequency.** An instruction histogram over the
  corpus orders lifter work. Coverage is measured over instruction occurrences,
  per compiler and opt level.
- **Pinned and hashed.** The manifest already has an fnv1a64 per object. Add the
  compiler version and the full flags.

---

## 4. Milestones

Each milestone has a **gate**: a command whose output is the evidence. A gate
must cover all code that exists at that point, not only the new code. v1's M0
gate ("green on an empty pipeline") passed while the existing passes
miscompiled. Estimates are focused weeks.

---

### M0 — Harness and IR semantics · 2–3 wk (v1: 1)

**Done** (e329819): workspace, T1 + sabotage (9.7 × 10⁹ probes, 0 mismatches),
verifier selftest 5/5, corpus generator + hashed manifest, fmt + clippy clean.

**Reopened.** v2's M0 gate covers the passes that already exist.
- IR semantics document and interpreter.
- TV driver; T0 harness with golden IR.
- Verifier fixes:
  - forward all messages (`verify_ssa` currently drops them, B7);
  - check operand types (width agreement on `Cmp`/`Bin`, Bool operands for
    branches and `Select`);
  - escalate warnings at phase boundaries.
- Fix §9's bugs **test-first**: each becomes a TV or T0 fixture that fails
  before its fix. Until a pass has been through this, it stays out of the
  default pipeline.
- `rdec status` prints §9's measured table from commands, so the state section
  is never hand-written again.
- CI (self-hosted Windows): `cargo test`, `clippy -D warnings`, `fmt --check`,
  and a determinism diff.

**Gate:**
- `cargo test` green, including fixtures for B1–B7 and §9's code-reading findings.
- `rdec selftest` green.
- TV catches a sabotaged pass by name (e.g. a DCE that drops one random live
  instruction).

**Closed 2026-11-01** (d326306 + the follow-up in this commit). The gate is green,
measured by `rdec status`:

| Gate item | Evidence |
|---|---|
| Interpreter + semantics document | `crates/core/src/interp.rs`, `docs/ir-semantics.md` |
| TV driver, sabotage caught by name | `crates/core/src/tv.rs`; `tv_catches_the_sabotaged_dce_by_name` |
| T0 fixtures, red before their fix | `crates/core/tests/m0_gate.rs` — 18 tests, one per §9 bug plus the read-findings, checked against ground truth / the interpreter, never against pass output |
| Verifier forwards messages, checks operand types, escalates | B7 test + `verifier_rejects_a_mixed_width_binop`, `verifier_rejects_a_select_with_an_integer_condition`, branch-on-non-bool as a hard error |
| §9 bugs fixed test-first | §9's "Closed" table: B1–B7, each with the test that pins it |
| `rdec status` prints §9 from commands | `rdec status` — tree, toolchain, fmt, clippy, test counts, selftest, T1 grid, corpus (120 objects, every COFF `Machine` re-parsed from the bytes, not trusted from clang's exit code) |
| CI | `.github/workflows/ci.yml`, GitHub-hosted `windows-latest`, on `github.com/beanscrack/rdec` (private). First green run: **41398695697** at a5b675c — all 11 steps: toolchain, fmt, clippy `-D warnings`, test (24), selftest, T1 grid, corpus determinism, status. Runner toolchain rustc 1.99.1 / clang 20.1.8; local is 1.99.0 / clang 22.1.8, and the corpus row records that drift (66,593 bytes on clang 20 vs 93,928 on clang 22) instead of failing on it. Nightly/dispatch adds the 9.7e9-probe `flags --full` sweep. |

Pipeline determinism is now a test (`pipeline_is_deterministic`), because every
TV result depends on it.

---

### M1 — Load, decode, discover · 2 wk

- goblin PE/COFF + ELF, with relocations applied or symbolized from day one. The
  corpus is `.obj`, so every call and RIP-relative reference is a relocation over
  a zero displacement.
- Function bounds from symbols, then `.pdata` (Win64) / `.eh_frame` (ELF), then
  recursive descent, then linear sweep. Record which source found each function.
- Intel listing via the iced formatter.

**Gate:**
- Instruction boundaries and lengths agree with `llvm-objdump` on every
  instruction reached by recursive descent across the corpus. v1 compared
  normalized operand text, which tests formatter conventions, not RDec.
- iced decode → encode → decode round-trips.
- Function-boundary precision and recall ≥ 99.5% against debug info, on the
  stripped corpus.
- 0 panics on 10 MB of random bytes plus 10⁶ fuzz iterations.

---

### M2 — IR, lifter, flags: verified against silicon · 4–6 wk (v1: 3–4)

- Flag representation per §1.2; IR per §1.3.
- Lifter, in corpus-frequency order:
  - **Integer core:** v1's list plus `movsxd cdqe cdq cqo xchg bswap bt bts btr
    btc rol ror shld shrd setcc cmovcc rep movs/stos endbr64`.
  - **Scalar SSE, moved up from M8:** `movss/movsd movd/movq`,
    `add/sub/mul/div/sqrt/min/max ss/sd`, `cvt*`, `ucomis/comis`, `xorps/pxor`
    zeroing, and `movups/movaps` as 128-bit copies. Every `float` and `double` on
    x86-64 is SSE, and `-O2` uses xmm registers for struct zeroing and copies.
    Without scalar SSE, a large share of real functions is E0.
  - **x87, AVX, other packed SIMD:** `Opaque`, counted.
- Stack-delta analysis and the frame model.
- T1 extensions:
  - `adc`/`sbb` with carry-in 0 **and** 1 (today only 1);
  - "unchanged" flags seeded both 0 and 1 per flag. Today there is one all-ones
    seed, so a model that hard-codes CF = 1 after `inc` passes. Volume does not
    fix this; seeds do.
  - exhaustive 8-bit;
  - the `--full` run nightly;
  - both an Intel and an AMD host.
- T1L and T1R.

**Gate:**
- T1, T1L, T1R: 0 mismatches. T1L uses ≥ 10⁴ random states per instruction
  form, for every form in the corpus histogram.
- T1R totality: every FlagsOp × flag pair has an exact lowering.
- Lift coverage ≥ 99% of instruction occurrences at O0–O2 for clang and MSVC;
  the residue is `Opaque`.
- IR goldens stable.

---

### M3 — CFG, SSA, core passes · 4–5 wk (v1: 3–4)

- CFG recovery with:
  - **noreturn** knowledge (`abort exit _exit __stack_chk_fail __fastfail
    ExitProcess`, plus propagation). v1 left this to M7, but without it CFG
    recovery runs off the end of one function into the next.
  - **tail calls** (a `jmp` to a known function entry).
  - **jump-table target resolution.** Structuring into `switch` stays in M5, but
    an unresolved `jmp [tbl + idx*8]` leaves the CFG incomplete.
- Dominators (Cooper–Harvey–Kennedy), dominance frontiers, natural loops, SSA,
  and frame-slot mem2reg with escape analysis.
- Passes behind TV; flag elimination per §1.2, under T1R; per-pass timing
  instrumentation.
- Decide the **stable ID scheme** now: function, block and value identities that
  survive re-decompilation. The M9 analyst database depends on it, and it is
  expensive to retrofit.

**Gate:**
- Verifier and TV green through the fixpoint on the whole corpus.
- 100% fixpoint convergence.
- 0 `Flags` values reach cgen.
- Fusion rate ≥ 95% (fused trees / total), with the residue visible as helpers.

---

#### M3 step 1 - measured (2026-11-02)

SSA construction over the lifted CFG: register live-ins captured per block, phi
inserted at joins and at back-edge-only headers, incoming values solved to a fixed
point over the CFG edges, live-in placeholders rewritten inside the reading block.

```
lift: 120 objects, 141 functions lifted, 141 pass verify_ssa, 63 with no gaps and no exits
evidence: 153 loud gaps, 64 Exit terminators, 17 unnamed call targets, 153 modelled register clobbers, 18 unreachable blocks dropped
interp: 115 returned, 25 ended at Exit, 0 hit the step budget, 141 reached an unmodelled op, 0 unmodelled loads, 1 arithmetic fault, 0 used a value that is never defined
OK: every lifted function is structurally valid IR
```

101 -> 141 valid IR. The 40 M2 failures were one class: a register read in a block
whose value comes from several predecessors, with no phi to merge them.

What the two checks establish, and what they do not:

- `verify_ssa` is structural - phi/edge agreement, dominance, phi-prefix form,
  operand types.
- The interpreter run is semantic about *definedness and control flow*: it consumes
  every operand strictly, so a dangling rename, a phi pointing at a value not live
  on its edge, or an unrewritten register placeholder faults loudly. 0 holes over
  141 functions. It says nothing yet about whether the *values* match x86 - that is
  T2's job at M4.
- 141 "reached an unmodelled op" is the flags-clobber `Opaque` at calls: every
  function that calls something reads it. Expected at this milestone.
- 18 unreachable blocks dropped, each counted as a gap: code discovery found, whose
  only entry is an indirect jump (M5). The bytes are in the object; the CFG we built
  cannot reach them. Emptied and counted, not left as dead defs.

Three verifier/lifter rules this milestone forced, each because the gate said so:

- **A phi operand is availability on its edge, not dominance of the phi.** The old
  rule rejected the standard loop-carried `%v = phi [%init, %head], [%next, %latch]`
  where `%next` is defined in the latch, and a self edge made it "defined later in
  this very block". The check now asks: defined in the predecessor, or in a block
  dominating the predecessor. (The use site's operand slot identifies the pair, so
  the edge is recoverable - no new IR metadata was needed.)
- **`undef` needs a legal home.** A block with no predecessors has no incoming
  value; the entry block is the only placement that dominates every use, it must
  sit after the phi prefix, and it must be placed at the end of the pass - emitted
  eagerly it lands after the instructions that use it, which the program-order rule
  correctly rejects.
- **A definition in an unreachable block is a defect, not a detail.** Clearing such
  a block removes its edges, so the CFG used for phi placement must be rebuilt after
  clearing; phis built from the old predecessor list do not match the verifier's.

Process rule this milestone caught, the hard way: three patches applied with plain
string replacement, against a remembered file shape, silently matched nothing and
left the gate printing counters it never computed (all zeros, one wrapped negative).
Every patch in this repo now asserts its anchor matches exactly once.


### M3 step 2 - measured (2026-11-02): split basic blocks at branch destinations

The hermetic loop fixture named the defect: a backward `jmp` into the middle of an
already-decoded block lands on an address that is not a block start, so the back edge
has nowhere to land, the header is not a join, no phi is created, and the function
returns the register's value from *before* the loop. Discovery's byte-claim invariant
(M1) makes that target claim no new bytes, so it cannot appear in `blocks` at all.

Fix: `Plan::targets` records every direct branch destination inside the code - kept
even when the bytes were already claimed - and lifting builds blocks from
`blocks | targets | {start}`, restricted to instruction boundaries.

Before -> after, same 141 corpus functions:

| measure | step 1 | step 2 |
|---|---|---|
| pass `verify_ssa` | 141 | 141 |
| functions with no gaps and no exits | 63 | 85 |
| `Exit` terminators | 64 | 27 |
| loud gaps | 153 | 126 |
| unreachable blocks dropped | 18 | 8 |
| interpreter: returned | 115 | 119 |
| interpreter: used a value nothing defines | 0 | 0 |
| `cargo test` | 38 | 41 |

`objdump-check` 124/124 and `discover` coverage 97.88% unchanged, so M1's gates did
not move. `clippy -D warnings` clean.

Remaining `Exit` terminators (27) and the largest remaining gap class are named, not
counted vaguely: a function entry reading a register the 4 modelled parameters do not
cover (r8, r9, bp, ...) gets an `opaque "live-in RXX"`, which is honest - the ABI does
not promise those are arguments - and is what a type/ABI inference pass must resolve.


## M4 — Calls, signatures, UB-free C, T2 · 4–5 wk (v1: 3)

- Call lifting: arguments and returns per ABI, clobbers, stack arguments, SysV
  varargs (`al`), Win64 shadow space. v1 limited M4 to leaf functions, but most
  functions call something, and T2 needs a call model anyway.
- Signature synthesis, plus type libraries for imports (`crates/types`).
- C emission per §1.6: precedence, cast placement, expression inlining over the
  memory token, sequencing safety, pointer-vs-integer display.
- T2 runner:
  - link the original object and the recompiled C into one harness, using
    symbol prefixing;
  - bind external calls to recording stubs;
  - compare return value, memory and call trace;
  - generate inputs from true prototypes;
  - build with clang and gcc at `-O0` and `-O2`, under UBSan.
- Evidence record per function (E0/E1).

**Gate:**
- T3 clean on all output.
- T2 100% on the leaf + direct-call corpus: ≥ 10³ vectors each, at `-O2`, UBSan
  silent.
- Every function has an evidence record.

---

#### M4 step 1 - measured (2026-11-02): the C emitter, end to end

`crates/core/src/emit_c.rs` + `rdec c --corpus`: lift -> passes (flag elimination,
phi prune, const fold, copy prop, branch simplify, dead-store elim, DCE) -> verify
again -> emit C -> write one translation unit -> compile it with
`clang -std=c17 -Wall -Wextra -Werror`.

```
emit-c: 141 lifted functions, 15 emitted C, 97 refused, 29 invalid after passes
refusal classes: 33 opaque: live-in Rax | 17 flags value sub{a | 9 opaque: live-in Rsi
                 | 6 opaque: live-in Rsp | 6 opaque: unmodelled instruction: cdq
                 | 6 sext from unsigned int
clang -Werror: OK (corpus/c/all.c -> corpus/c/all.obj)
FAIL: a pass broke SSA in 29 functions: array_walk.Oz :: f: %69 mixes widths W32 and W64
```

What is now proven, and what is not:

- **Proven:** the pipeline runs on real machine code, and for 15 functions the
  emitted C is accepted by clang under `-Werror` - real widths, real signedness,
  phi-free (edge-copies), flat-memory loads/stores. The emitter cannot emit a
  silent gap: an unmodellable instruction is a `Refusal` with an address and a
  reason, and the gate counts them by class.
- **Not proven:** that the C computes what the machine code computes. That is the
  T2 differential (M4 step 2), and it is the milestone's real exit test.

The named work queue, largest class first. Each is a specific defect, not a
mystery:

1. **33+9+6 live-in placeholders** (`Rax`, `Rsi`, `Rsp`) - a register read at a
   block start whose value the SSA solve never resolved to a definition. The
   interpreter's "0 uses of an undefined value" does not catch it: the placeholder
   is an `opaque`, defined, semantically empty. Fix: find why in_val keeps the
   placeholder (entry-bound Rsp should propagate).
2. **29 functions where a pass breaks SSA** (width mixes, W1-with-W64). The
   post-pass `verify_ssa` is what catches this; it is a real pass bug, and the
   named instance is `array_walk.Oz %69/%72/%77`.
3. **17 flags values surviving elimination** (`sub`'s flags consumed in a shape
   `fuse::flag_elimination` does not fold).
4. **6 `cdq`** - unmodelled instruction in the lifter.
5. **6 SExt from an unsigned source** - the emitter refuses; the C needs a cast
   through the signed type of the source width.
6. **Indirect calls** - refused, no signature model yet.

Emitter decisions worth keeping in mind:

- **Flat memory, one declaration per translation unit**: `static uint8_t
  rdec_mem[1<<20]` inside every emitted function is a redefinition error the moment
  two functions share a file, which is the normal case for a decompiler. The
  address mask turns an out-of-range machine access into a deterministic in-range
  one, because C UB cannot be diffed against anything.
- **The entry stack pointer is a constant address** (`STACK_BASE = 1<<19`), not an
  `Opaque`. As an opaque it refused every -O0 function, because every -O0 function
  has a stack frame; as a flat-model address it gives the interpreter and the C the
  *same* bytes to compare.
- **`-Werror` hygiene is the emitter's job**: `(void)&&lN;` marks intentional labels
  a branch-folding pass orphaned, `__attribute__((unused))` on locals the passes
  left set-but-unread, and on the prelude helpers. Deleting the block, or dropping
  the flag, would hide real facts to make a build green.
- Guards where x86 and C disagree: division by zero (`? 0`), `INT_MIN / -1`, shift
  counts masked the x86 way, `lzcnt/tzcnt` of zero.


### M4 step 2 - measured (2026-11-02): live-in scaffolding resolved, class 1 closed

SSA construction invented an `Op::Opaque("live-in Rg")` for every register read
before a write in a block, rewrote its uses to the resolved incoming value, and
**left the definition behind** - `Op::Opaque` blocks DCE on purpose, so the
scaffolding became a permanent refusal in the emitted C. Deleting the definition
once nothing refers to it:

| measure | before | after |
|---|---|---|
| emitted C accepted by clang -Werror | 15 | **25** |
| refused functions | 97 | 87 |
| interpreter: reached an unmodelled op | 141 | **47** |
| valid IR (lift gate) | 141 | 141 |
| live-in placeholders resolved | - | 439 (205 of them `undef`) |

The refusal classes `opaque: live-in Rax` (33), `Rsi` (9), `Rsp` (6) are gone.

Two things this step made explicit, both from the gate rather than from reading:

- **Deleting per group is wrong.** Deleting a placeholder as soon as group `g`'s
  rewrite made it unused broke two functions: a later group's phi had already
  taken that placeholder as an incoming value, and the verifier reported
  "value %28 is used but never defined". The sweep runs once, after every group
  has been solved and rewritten. `cargo test` stayed green through the broken
  version - only the corpus gate caught it, which is why the gate exists.
- **`undef` is the honest answer, and it is a count.** 205 live-ins resolve to
  `undef`: a register read at a point no modelled path defines, which at a
  function entry is the ABI question ("is r8 an argument here?"), not a dataflow
  one. The number is in the gate's evidence line so it cannot be forgotten.

Largest refusal class now: **37 flags values surviving elimination**. My first
explanation - flags merged by a phi at a join - was **wrong**, and gates killed it
twice: the verifier rejected the hand-built fixture (`%5 has Flags type from a
non-Flags op`: nothing in this IR merges a flags value, by design), and the test I
wrote for the fix failed on the premise, not on the code. The pass and its tests
were deleted; the diagnosis that survives is in `docs/m4-flags.md`:

the lifter reads **all five flag bits** for a `jcc` (defensively), the branch uses
`SF`/`OF` only, and fusion rewrites the root's cone - so the `Flags` value stays
partially consumed and the pass correctly refuses to half-rewrite it. DCE cannot
clear the unused reads first, because deleting an unused `getflag` is not
semantics-preserving here: the interpreter evaluates every instruction it reaches
and reading a flag the model calls undefined is a fault, so removal can turn a
faulting execution into a returning one - TV would report DCE as changing meaning.

M4 step 3 is therefore: fold **per flags value, not per root cone** - rewrite every
read of a flags value in one step - while preserving the definedness story (T1 is
the oracle for which bits an operation leaves defined). The measure is the refusal
count: 37 of 87 today.

### M4 step 3 - measured (2026-11-02): the fusion pass was never firing on live code

The per-flags-value sweep went in, the refusal count did not move, and a hermetic
fixture (`crates/core/tests/m4_flags_live.rs`: a `jge` with all five flag bits read,
`not(SF ^ OF)` feeding the branch) reproduced it exactly. The cause was not the
shape of the flags value at all - it was the root finder:

`flag_roots` marks "is this node interior to the tree", and the arm for a
terminator use returned `true`. A flag expression feeding a `br` is consumed by a
*terminator*, so every live jcc tree was classified interior, so the pass folded
only the flag reads nobody used, and every real `flags.sub` feeding a branch
survived. The comment above that arm already described the correct behaviour; the
code did the opposite of it. One polarity, in the arm, not in the table.

Before -> after, same 141 functions:

| measure | before | after |
|---|---|---|
| refused functions | 87 | **48** |
| refusals naming a flags value | 37 | **0** |
| `flags value sub{..}` class | 28 | **gone** |
| `flags value and{..}` class | 9 | **gone** |
| emitted C accepted by clang -Werror | 25 | 25 |
| `cargo test` | 43 | **45** |

`lift --corpus` is unchanged at 141/141 valid IR, so the lifter is not what moved.

What the fix exposed, named: **68 functions (was 29) where a pass breaks SSA**,
first instance `array_walk.O0 :: f: bb1: %94 mixes widths W32 and W64`. The fusion
lowerings are now running on real code for the first time, and they emit comparisons
at the wrong width - `ZF` of a 32-bit `sub` compared against a 64-bit zero. The fix
is in `exact_flag`/the scratch builder: constants take the operation's type
(`ctx.ty`), not `u64`. That is M4 step 4, and it is also the gate for the emission
count, since a function whose passes broke SSA is not emitted at all.

Process notes worth keeping:
- The test suite stayed green through a fusion pass that never fired on live code.
  The M0 flag tests do not cover "flags value consumed by a branch". They do now
  (`m4_flags_live.rs`), with a TV test next to it: folding a `jge` preserves meaning.
- The refusal dump (`corpus/c/refused_N.ir`, written by `rdec c` for the first three
  refusals) is what turned this from a theory into a fact: the post-pass IR showed
  nothing had been folded, which ruled out every "the table is missing a shape"
  hypothesis in one look.


## M5 — Control-flow structuring · 4–5 wk

- v1's scope: if/else, while, do/while, `for` recovery, and switch recovery from
  M3's resolved tables. Irreducible regions keep `goto`, with a warning, counted.
- Choose the algorithm against the corpus instead of by default. Candidates:
  - Sharir-style reduction;
  - condition-based refinement ("No More Gotos", Yakdan et al., NDSS 2015);
  - compiler-aware structuring (SAILR, Basque et al., USENIX Security 2024).
  SAILR's finding shapes the gate: forcing zero gotos duplicates code and adds
  condition variables, and the source's own control flow is the better target.

**Gate:**
- Per function, gotos and duplicated statements are measured against the source.
  v1's "goto = 0 on reducible CFGs" is kept as a ceiling check, not as the only
  metric.
- Every loop and switch case is T2-equivalent. Structuring is invisible to TV,
  so T2 is its oracle.
- ≥ 90% of jump tables come out as `switch`.

---

### M6 — Readability · 5–7 wk (v1: 4–6)

v1's scope (value-set and signedness inference, width narrowing, pointer
provenance, the idiom table, naming, struct recovery, strings and globals),
plus type-library propagation from call sites.

**Gate:**
- Scored against debug info: signedness, width, pointer-ness, struct field
  offsets, and parameter names recovered through type-library calls. Ratcheted.
- **The same metrics computed on Ghidra headless output for the same stripped
  corpus.** The readability target is relative to a real incumbent, not an
  absolute number picked in advance.

---

### M7 — Whole binary · 6–8 wk

v1's scope (noreturn and tail calls moved to M3), plus:
- **Library function identification** (FLIRT-style signatures generated from
  CRT/libc archives). In a stripped, statically linked `-O2` binary, library code
  is often most of the functions.
- Function fragments from hot/cold splitting (`.cold`, MSVC PGO chunks).
- MSVC specifics:
  - SEH/unwind-aware CFG (`.pdata`/`.xdata`, `__C_specific_handler` scopes);
  - CFG guard (`__guard_dispatch_icall_fptr`) and retpoline thunks, treated as
    indirect call or jump.

**Gate:** v1's real-binary gate, plus the % of functions at E1+ and the
precision/recall of library identification against ground truth.

---

### M8 — C++ and packed SIMD · 8–12 wk

v1's scope: demangling (Itanium + MSVC), vtables, RTTI, `this`,
constructors/destructors, exceptions, packed SSE/AVX → vector C, TLS. Scalar SSE
has moved to M2.

---

### M9 — Production · ongoing

v1's scope: fuzzing at scale, nightly T2 over generated C, performance budgets,
JSON output + stable IDs, the plugin API, editor integration, and the
contributor contract for the fusion and idiom tables. Additions:
- **Analyst loop:** renames, retypes, prototypes and struct definitions stored
  by stable ID and fed back as constraints, with incremental re-decompilation of
  only the affected functions. This is how interactive decompilers are actually
  used.
- Debug info (PDB/DWARF) as optional *input* when present, separate from its use
  as test ground truth.

**Gate:** v1's (0 fuzz crashers open > 48 h; a public, versioned conformance
score), plus the evidence-level distribution published with every release.

---

### P — Proof track (E2/E3) · starts after M4 · 6–10 wk to E2

- **P1:** SMT encoding of the IR, differentially tested against the interpreter.
  These are two semantics of one IR, and they must agree on random states.
- **P2 (E2):** bounded equivalence of lifted IR ≡ final IR, for loop-free
  functions and for loops unrolled to *k*.
- **P3 (E3):** loop synchronization through a CFG product plus invariant
  inference. Each function has a solver timeout and falls back to E2/E1; the
  proof never blocks output.

**Gate (P2):**
- E2 on ≥ 80% of loop-free leaf functions, ≤ 10 s each (initial threshold,
  ratcheted).
- Sabotage: a deliberately wrong rewrite yields a counterexample that replays in
  the interpreter.

---

## 5. Timeline

| | v1 | v2 | Why it moved |
|---|---|---|---|
| M0 | 1 | 2–3 | IR interpreter, TV, existing bugs fixed test-first |
| M1 | 1–2 | 2 | `.pdata`/`.eh_frame`, relocations |
| M2 | 3–4 | 4–6 | scalar SSE, T1L, T1R, revised flags |
| M3 | 3–4 | 4–5 | noreturn, tail calls, jump-table targets moved in |
| M4 | 3 | 4–5 | calls, UB-free output contract, T2 runner |
| M5 | 4–5 | 4–5 | — |
| M6 | 4–6 | 5–7 | Ghidra baseline, type libraries |
| **M0–M6** | **19–25** | **25–33** | scope moved in, not padding |

M7 takes 6–8 weeks, M8 8–12. The P track runs in parallel with M5–M6 if there
is a second person, and serially otherwise.

**Re-baseline after M2** using measured throughput (actual vs. estimated weeks
per milestone). Until then, any one-person estimate beyond six months is a guess.

---

## 6. Known killers (planned for, not discovered later)

1. Flag semantics: T1, now with T1R behind it.
2. Silent semantic loss: `Opaque`, `Exit`, evidence levels; loud by design.
3. Irreducible CFG: goto fallback, never a crash.
4. Memory aliasing / address-taken locals: conservative promotion over an
   explicit memory token.
5. Frame analysis (`alloca`, `__chkstk`, canary, no frame pointer): named cases
   with stack-delta analysis.
6. Jump tables: targets resolved in M3, structured in M5.
7. Type recovery: M6, measured against debug info *and* against Ghidra.
8. **UB in emitted C** (sequencing, signed overflow, shifts, aliasing,
   alignment): §1.6, with T2 at `-O2` under UBSan.
9. Performance blowups on 10⁵-instruction functions: per-pass index (§1.3),
   per-pass timing from M3, budgets in M9.
10. Scope explosion without an oracle: every feature ships with a gate that
    covers all existing code.
11. **Unverified rewrite tables.** The audit's headline: the table v1 calls "the
    single most important line" has wrong entries and is unreachable. T1R.
12. **Single-compiler corpus:** the §3 matrix.
13. **Hostile input:** fuzzing from M1, budgets, per-function isolation.
14. **Library-code noise:** M7 signatures.
15. **Oracles sharing code with their subjects:** §1.1 rule.

## 7. Non-goals (until stated otherwise)

32-bit x86, ARM64, x87 `long double` semantics (lifted as `Opaque`),
self-modifying, packed or obfuscated code (detected and reported, not
decompiled), and kernel-mode code. Each is loud: `Opaque` plus a gap in the
evidence record, and a corpus case proves the report fires. A non-goal is a
tested decision, not a silence.

## 8. Process rules

- Every bug fix lands with the fixture that would have caught it, in the oracle
  tier that *should* have caught it.
- No pass enters the default pipeline without TV coverage.
- The state section is regenerated by `rdec status`, never written by hand. It
  has been wrong twice: v1 said "does not compile" when the tree compiled, and a
  later edit said "SSA/IR verifier runs after every pass" when `run_pipeline`
  never calls it.
- Gates are commands in CI, not paragraphs.

---

## 9. State of the tree (audited 2026-11-01 at e329819)

### Measured

| Check | Result |
|---|---|
| `cargo build` | ok (rustc 1.99.0) |
| `cargo test` | 5 passed, all in `rdec-verify`. **`rdec-core`: 0 tests.** |
| `rdec selftest` | 5/5 (fixed in d403f9e; it had failed because branch-on-non-bool was a soft error) |
| `rdec flags` (default grid) | 593,400 probes, 0 mismatches; `--sabotage` → 15,129 mismatches, all `add.32 OF` |
| `rdec flags --full 20000` (per v1 §5) | 9.7 × 10⁹ probes, 0 mismatches, 331 s |
| T1 coverage | add sub and or xor adc sbb mul imul inc dec neg shl shr sar, 8–64 bits. **Gaps:** `adc`/`sbb` only with carry-in = 1; one all-ones flag seed |
| Corpus | 30 snippets × clang {O0, O1, O2, Oz} = 120 COFF objects, hashed manifest; no `-g`, so no ground truth yet |
| `run_pipeline` | never calls the verifier; DSE is not included; stops silently after 32 rounds |

### Confirmed bugs

Each was reproduced by building IR through `rdec-core`'s public API, running one
pass, and printing the result (scratch crate outside the tree). Bold rows are
the ones the verifier accepts.

| # | Where | Input → output | Verifier |
|---|---|---|---|
| B1 | `Function::uses_of` / `replace_use_all` ignore `Return` operands | `return a+b` → DCE deletes the add; `ret %2` dangles | rejects (but see B7) |
| **B2** | `dedup_blocks` key renumbers *free* operands and omits the returned value | `c ? p0+p1 : p2+p3` → always `p0+p1` | **accepts** |
| **B3** | `flag_elimination` lowers each `get_flag` alone (the "tree root" is the leaf) | jcc matrix unreachable; `cmp a,b; jl` → `(a <s 0) ^ opaque` | **accepts** |
| **B4** | `atom(SF)` for `cmp` | `cmp a,b; js` → `a < 0` (correct: `a − b <s 0`) | **accepts** |
| B5 | `FlagCtx.res` and the exact path use the `Flags` value as the arithmetic result | `test a,b; ja` → compares the `Flags` value to 0 | rejects, with no message (B7) |
| **B6** | Exact fallback covers only CF/OF/ZF/SF of add/sub | `test; jp` and `inc; jc` leave `get_flag` and a live `Flags` value | **accepts** |
| B7 | `verify_ssa` maps `verify()`'s errors to `Err(vec![])` | structural failures reported with zero messages | — |

### Closed 2026-11-01 at d326306 (M0 reopened → closed)

The gate was: reproduce each bug as a fixture that fails, fix it, and check the
result against ground truth or the interpreter — never against pass output.
`crates/core/tests/m0_gate.rs`, 15 tests, red before, green now.

| # | Fix | Evidence (test names) |
|---|---|---|
| B1 | `uses_of` / `replace_use_all` count `Return` and `Branch` operands | `b1_terminator_uses_are_counted`, `b1_dce_must_not_delete_a_returned_value` |
| **B2** | `dedup_blocks` rewrites uses along the positional mapping it just computed, canonicalises **only block-local values** (free operands and successor targets key by identity), and skips phi blocks | `b2_dedup_must_not_merge_blocks_with_different_operands` (both arms), `b2_dedup_must_not_merge_blocks_with_different_conditions` |
| **B3** | fusion matches the `get_flag` tree **at its root** (`crates/core/src/fuse.rs`); v1's per-leaf pass is deleted, not patched | `b3_jl_after_cmp_is_signed_less` |
| **B4** | SF lowers against a recomputed `a − b`, never against `a` | `b4_sf_after_cmp_is_the_sign_of_a_minus_b` |
| B5 | the arithmetic result is re-emitted from the operands; `FlagCtx.res` no longer exists | `b5_zf_after_test_is_the_result_of_a_and_b` |
| **B6** | every (FlagsOp × flag) pair is answered. Where the IR cannot express it — CF after `inc` (§1.2 item 3), OF after a variable shift, AF after logic/shift/mul — the lowering is `Op::Opaque` **with the reason**, so the function reports itself E0 | `b6_pf_after_test_is_lower_byte_parity_and_leaves_no_flags`, `b6_cf_after_inc_is_undefined_and_must_be_reported_not_invented` |
| B7 | `verify_ssa` keeps `verify()`'s messages; a non-bool branch condition is a hard error, not a warning | `b7_structural_failures_keep_their_messages`, `rdec selftest` 5/5 |

The missing oracle layer (§1.1), built before any of the fixes were trusted:

- `crates/core/src/interp.rs` — executable IR semantics, the reference for TV and
  later for T1R/P1. Writing it found a soundness hole in the harness: a dangling
  operand read as `0`, which is why the sabotage test had passed. Dangling uses
  are now `MissingOperand`.
- `crates/core/src/tv.rs` — `run(f, s) == run(pass(f), s)` over sampled and
  boundary states, comparing outcome, ordered effects and final memory.
  `tv_accepts_the_real_pipeline_on_a_valid_function`,
  `tv_catches_the_sabotaged_dce_by_name`.

Read-findings closed by the same work: `cmp_pair` (derived from the trichotomy
subsets instead of remembered, validated exhaustively —
`cmp_pair_table_is_arithmetically_correct`), `joint_pair` (deleted; the exact
form needs no pair table, and `CF ∧ ¬ZF` is `¬ZF`, not `ZF`), OF after `cmp`
(now exact), `add_value_op` / `add_const` (deleted: no callers, both broke
dominance), `const_bin` `MulHi`/`MulSmul` at narrow widths
(`const_fold_mulhi_does_not_shift_by_64_at_narrow_widths`), `merge_blocks` phi
incoming (`merge_blocks_updates_phi_incoming`), `Call::pure`'s inverted doc.

Open, with the milestone that owns each: phi-aware merging in
`simplify_branches`/`dedup_blocks` (M3 GVN); carry-in and previous-flags as real
operands (§1.2 items 1 and 3, M2 — today those cases are loud gaps, not lies);
flag values through phis (§1.2 item 7, M3); `Op::Const` typed `u64` (M3);
`def()`/`uses_of()` linear scans (M3); and the T1 coverage gaps named above
(`adc`/`sbb` with carry-in 0, more flag seeds), which M2's T1R closes against
silicon *and* the interpreter.

Gates at d326306: `cargo fmt --check` clean; `clippy --all-targets -D warnings`
clean; 20 tests green (15 gate + 5 oracle, including the T1 hardware run);
`selftest` 5/5; `corpus` 120/120 AMD64 objects. `flags.rs` is untouched by this
commit, so the measured T1 result (9.7 × 10⁹ probes, 0 mismatches) still stands.

### Measured on the first CI run (2026-11-02, GitHub-hosted windows-latest)

| item | result |
|---|---|
| fmt, clippy `-D warnings`, `cargo test`, `selftest`, `flags` grid | **passed on the runner**, so T1 is a valid hardware oracle there too (it is x86-64) |
| corpus step | **failed, and the gate was the bug.** `git diff --exit-code corpus/manifest.tsv` asserts byte-identical codegen against the committed manifest, so it compares clang *versions*: runner 20.1.8 vs manifest built with 22.1.8. Replaced with a check of the claim actually wanted — build twice on the same toolchain, byte-compare the manifests. The manifest now records its compiler and target. |
| corpus reproducibility (found by that new check, locally, on its first run) | COFF objects embed a build `TimeDateStamp`: two runs of the *same* clang over the *same* sources moved 120/120 object hashes at **identical sizes** — the signature of a fixed-width volatile field, not a codegen change. `/Brepro` is clang-cl only and `-Xclang -brepro` is unknown to clang 22 (both measured), so the stamp is normalized at object offset 4; `-g0` drops the CodeView sections that embed the source path. Pinned by `coff_stamp_normalization_targets_the_real_field`, because the guard must read `Machine` at offset **0** — an object header has no image signature field, and checking offset 2 (NumberOfSections) silently disables the normalization. That is how this first shipped. |

After the fix: two builds into the same directory produce byte-identical
manifests, every object stamp is 0, and the corpus is a fingerprint of codegen
rather than of the clock. Run 41398695697 (a5b675c) is green on all 11 steps.

One consequence to keep in mind: `status` measures the objects on disk, which the
corpus step just rebuilt with the runner's clang, so byte counts and hashes are
toolchain-specific by design. The committed manifest is the clang-22 fingerprint;
comparing them is a toolchain comparison, not a RDec check, which is exactly the
confusion that produced the failing gate.

### Found by reading

Each needs a fixture before it is fixed.

- `cmp_pair`: **8 of 16 entries are wrong.** `(ult ∧ ne)` and `(slt ∧ ne)` map to
  `≤` instead of `<`. `(uge ∨ ne)` maps to `ugt`, but it is constant true.
  `(sgt ∨ ne)` maps to `≥`, but it is `≠`. Each appears in both operand orders.
  It is latent today only because B3 makes it unreachable.
- `joint_pair`: `¬SF ∧ ¬OF` → `ge`, but `ge` is SF = OF. Test class:
  `CF ∧ ¬ZF` → `ZF` and `OF ∧ ¬SF` → `SF`; the correct results are `0`, or
  `¬ZF`/`¬SF` when the left side is negated.
- The signedness of `SF ⊕ OF` comes from the operand's type (§1.3).
- OF after `cmp` lowers to `Opaque` instead of the exact helper.
- `add_value_op` puts new ops in the entry block, so operands defined later don't
  dominate their use. `add_const` can reuse a constant from a non-dominating block.
- `simplify_branches`, `merge_blocks` and `dedup_blocks` don't update phi
  incoming lists.
- `const_bin`: `MulHi`/`MulSmul` shift by 64 at every width.
- `FlagsOp::Add { carry: bool }`: carry-in is not an operand, and `Inc`/`Dec`
  have no previous-flags operand.
- Every `Op::Const` is typed `u64` whatever type is requested, so narrow
  constants are never reused and comparisons mix widths.
- `def()` and `uses_of()` are linear scans: DCE is O(n²) per sweep, and the
  fixpoint multiplies that.
- The doc comment on `Call::pure` describes the opposite of the field.

**What this shows.** v1's thesis is right, and the tree demonstrates it from the
other side. About 1.1 kLOC of passes were written before any oracle could see
them, and four of the confirmed bugs produce wrong logic or leave raw flags
while every existing check passes. The flag model *had* an oracle, and its five
bugs were caught on the first run.

---

### Measured: M1 step 1 (load + decode), 2026-11-02

| item | result |
|---|---|
| COFF object reader | **Written from scratch.** goblin 0.10.7 ships no COFF *object* reader — `src/read/coff` does not exist in the crate — and a decompiler cannot outsource the container it targets. Parses file/section headers, the symbol table (aux records skipped), per-section relocations, long names via the string table (`/N` section names, offset-form symbol names). `crates/arch-x86/src/loader.rs` |
| Function extents | COFF records **no symbol size**, so extents are inferred as "next symbol in section, else section end" and tagged `SizeSource::{NextSymbol, SectionEnd}`. Nothing downstream can read an inferred extent as a declared one. |
| Section-definition trap | `.text`'s definition symbol sits at offset 0, the same address as the first function. An offset-ordered extent walk that keeps it gives the function **size 0**, and the first version of this loader then reported "no code symbols" — which the gate accepted as success. Extents now exclude definition symbols (`a_function_sharing_offset_zero_with_its_section_symbol_gets_an_extent`), and **a check that inspects zero symbols is a FAIL**, not an OK. |
| Entry-count trap | `NumberOfSymbols` counts auxiliary records. Walking main symbols until `symbols.len() == nsyms` runs off the table end — the exact error the first real object produced. The loop bound is entries consumed. |
| Decoder | iced-x86 1.22 (`op_code_info`, `instr_info`, `intel`/`nasm`). A refusal is emitted as a `len == 0` gap **and ends the sweep** — advancing past undecodable bytes is how a disassembler invents instructions out of data (`an_undecodable_byte_is_reported_as_a_gap_not_skipped`). |
| **Gate: boundaries vs llvm-objdump** | `rdec objdump-check --corpus`, all 120 objects: **124 symbols, 124 exact boundary matches, 0 mismatches**, each symbol also decode→encode→decode stable. Its own CI step; a sampled row in `rdec status`. |
| Fuzz | 40,000 pseudo-random windows (prefix-biased): 0 panics, every accepted instruction 1..=15 bytes, addresses monotonic, refusals terminal. |

CI on this commit: run 41401159339 concluded `failure` with **zero jobs and no
logs** — `name: objdump-check (M1: instruction boundaries ...)` has an unquoted
colon-space, so the workflow file never parsed, and GitHub's failure mode for
that is silent. Pinned by `crates/cli/tests/ci_hygiene.rs`, which also asserts
that it *found* workflow files: the first version ran from `crates/cli`, found
none, and passed while checking nothing. Green again at e230c68 (run
41401304705), 12 steps including the corpus-wide objdump gate on the runner.

Not yet M1: recursive-descent discovery from entry points and relocations,
indirect-jump classification (jump-table recovery), the coverage measurement
itself, and the disassembly-literal test vectors. M1's gate stays open until
coverage ≥ 99.5% is measured on the corpus.

### Measured: M1 step 2 (recursive descent), 2026-11-02 — **gate RED, by design**

`rdec discover --corpus` over all 120 objects:

```
discover: 120 objects, 124 symbol-table functions, 0 missing, 17 extra starts
code bytes: 5992, padding (int3, nop sequences, long 0x00 fills): 36
coverage: 5830/5956 non-padding code bytes = 97.88%; uncovered non-padding = 126 bytes in 2 runs
gaps: indirect branch 2
FAIL: coverage 97.88% is below the 99.5% M1 gate
```

| claim | measurement |
|---|---|
| Function discovery | **124 of 124** symbol-table functions recovered by descent from entry points + `REL32` call relocations. 0 missing. 17 *extra* starts (call targets with no symbol — local functions and thunks), reported, not punished. |
| Coverage | 97.88% of non-padding code. The 126-byte shortfall is **2 runs of real code behind the 2 recorded indirect jumps** — switch arms reached only through a jump table. M1's gate requires "no unclassified indirect jump remains", so M1 stays open and the gate fails on purpose; jump-table recovery is M5. The threshold was not moved to fit the measurement. |
| Padding | Defined **semantically**, not by byte pattern: a run the decoder renders entirely as `nop` is padding (LLVM aligns with `0f 1f 84 00 …`, `66 66 2e 0f 1f …`), as are int3 runs and long 0x00 fills. 36 bytes. Counting NOP padding as coverage failure understates; counting it as covered overstates — it is now neither. |
| Overlap invariant | Bytes are claimed once, globally (`Decoded`); a sweep that reaches an already-claimed address stops there, which makes overlap structurally impossible. Found because coverage read **114.30%**: a sweep ran past a branch target that a later sweep decoded again. `overlap_free` is checked per object and a violation fails the gate. |
| Ground truth | The object's own symbol table (present even at `-g0`), so the claim is measured, not asserted. |

Bugs found by building this, each pinned by a test:
- iced 1.22 models a near branch as `OpKind::NearBranch32`, **not** an immediate;
  `immediate(0)` on it panics ("Invalid operand: 0"). The first version returned
  no targets, so discovery silently lost every call and jump edge and the corpus
  looked fine at 124/124 (every function was a symbol seed).
- A block's fallthrough is a property of how the block **ends**. Recording it at
  a `call` while the sweep continued enqueued the block's own tail as a new
  block, swept it twice, and failed `overlap_free` in a unit test.
- Basing every code section at 0 makes two sections' offset 0x10 look like the
  same byte.
- A probe that loads manifest-relative paths from a crate directory loads
  nothing: `let Ok(obj) = load(path) else { continue }` reported "no overlap
  found" over an empty loop. Same lesson as the CI hygiene test, paid for twice.

### Measured: M2 step 1 (lifting x86-64 -> IR), 2026-11-02 — **gate RED, by design**

`rdec lift --corpus`, all 120 objects:

```
lift: 120 objects, 141 functions lifted, 101 pass verify_ssa, 63 with no gaps and no exits
evidence: 135 loud gaps, 64 Exit terminators, 17 unnamed call targets, 153 modelled register clobbers
FAIL: 40 lifted functions are not valid IR
```

| claim | measurement |
|---|---|
| Subset lifted | mov/movzx/movsx/movsxd/lea, add/sub/and/or/xor (register and memory destinations), shl/shr/sar with the hardware count mask, inc/dec, neg/not, cmp/test, imul, push/pop, xchg, jcc/jmp/call/ret, setcc, cmovcc. Everything else becomes `Op::Opaque` in the IR and is counted: 135 gaps, 64 `Exit`s, 17 unnamed call targets. |
| Register model | One `W64` SSA value per register; 32-bit read = `trunc`, 32-bit write = `zext` (the x86 clear-upper-half rule), 8/16-bit write = the mask/or pair. 153 register clobbers modelled after calls, per caller-saved register, as `Opaque` — a value kept live across a call is a silent miscompile. |
| Flags | `cmp`/`test`/arith emit a `Ty::Flags` SSA value; `jcc`/`setcc`/`cmovcc` read it with `GetFlag`, expanded to the flag-bit expression (L = SF^OF, A = ^CF & ^ZF, ...). Fusion (`fuse.rs`) and DCE simplify them; the lifter stays literal. |
| Structural soundness | 101 of 141 pass `verify_ssa`. The 40 failures are all one named class: a register value defined in one block and used in a join successor with no phi — SSA construction, which is M3. The gate stays red; the threshold was not moved. |

Two bugs found, each pinned by a test:
- **The IR discarded the width of a constant.** `Function::emit` takes the type
  from `Op::ty`, and `Op::Const(_) => Ty::u64()` — so the lifter's declared
  `uint32_t` for `add ecx, 1` was silently overwritten with 64-bit, and the
  verifier's mixed-width check fired two layers away in lifted code. A constant
  has no intrinsic width; the emitter's claim is the value's type
  (`Op::ty(Const) => None`), and a constant that does not fit its declared width
  is now a hard verifier error. Pinned by `a_constants_width_is_the_emitters_claim`.
- **Overflow and parity are different bits.** The condition-code map folded
  `pe` (parity even) into the signed-overflow condition, so `jo` and `jp` would
  have compiled to the same test. The dead-code lint found it (`Cond::No` never
  constructed), which is the whole argument for `-D warnings`.

Attribution lesson: the IR recorded each instruction's **block** address, so
`%51` and the instruction that produced it could not be matched — a width
complaint was untraceable. IR nodes now carry the instruction address.

## 10. Changes from v1

| Change | Reason |
|---|---|
| "Provably equivalent" → graded evidence levels E0–E3, with a definition of equivalence and a stated trusted base | v1's oracles are tests. The claim must match the evidence. |
| IR interpreter + TV, T1R, T1L added to the oracle stack | Nothing validated the passes, the rewrite tables or the lifter; §9 shows the result |
| Flag representation: result link, carry-in operand, prev-flags operand, phi splitting, total fallback, FP compares | B3–B6 and the read-through findings; `adc`, `inc`, variable shifts and `jp` were unrepresentable |
| Signless IR integers; signedness on predicates | Signedness taken from the type turned `jl` into an unsigned compare |
| Explicit memory token, call effects, per-pass index, convergence reporting | Code motion and inlining need memory order; scans don't scale; silent non-convergence |
| Output contract (C11 + `rdec.h` + `-fwrapv -fno-strict-aliasing -fno-delete-null-pointer-checks`); T2 at `-O2` with UBSan | x86 wraps and C doesn't; `-O0` hides UB |
| Corpus section: clang/gcc/MSVC matrix, debug-info ground truth, frequency-ordered lifting | A single-compiler corpus doesn't generalize; scoring needs ground truth |
| Scalar SSE: M8 → M2 | All x86-64 floating point and much `-O2` struct code is SSE |
| noreturn, tail calls, jump-table targets: M5/M7 → M3 | CFG recovery is wrong without them |
| Calls and type libraries in M4; T2 runner specified (stubs, call trace, symbol prefixing) | Most functions aren't leaves; T2 needs a call model |
| M1 gate: llvm-objdump boundary agreement + round-trip, not operand-text match; `.pdata`/`.eh_frame` bounds | Text matching tests the formatter, not RDec; unwind tables give near-exact bounds |
| M5: algorithm chosen on the corpus; gotos measured against the source | Zero gotos can cost readability (SAILR) |
| M6: Ghidra baseline | Readability has to be relative to an incumbent |
| M7: library signatures, MSVC SEH/CFG-guard, cold fragments | Real stripped Windows binaries are dominated by these |
| M9: analyst loop; stable IDs decided in M3 | Interactive use is the product; IDs are expensive to retrofit |
| P track for E2/E3 | Makes the proof claim real rather than rhetorical |
| Gates must cover all existing code; `rdec status` generates §9 | v1's M0 gate passed over miscompiling passes; the state section was wrong twice |
| Non-goals listed | A production tool says what it doesn't do, loudly |
| M0–M6: 19–25 → 25–33 weeks, re-baseline after M2 | Scope moved in |

## T2 - differential execution: measured (2026-11-02), RED by design

`rdec t2 --corpus` (crates/cli/src/t2.rs, from the delegated build) runs BOTH sides of
every function the emitter accepts: the original machine-code bytes reached through the
COFF symbol via a linked driver, and the generated C, on the same 4-argument vectors,
comparing the full 64-bit RAX. A crash is its own bucket; a function that cannot run on
both sides is named with a reason, never silently dropped.

```
t2: 141 lifted, 57 refused by the emitter (not run), 0 invalid after passes
runnable: 31 functions on both sides, 53 not runnable
compare: the full 64-bit RAX, every vector, on both sides
compared: 732 | match: 617 | mismatch: 115 | not compared: 0 | differing only above bit 31: 39
FAIL: the decompiled C disagrees with the machine code on 115 of 732 runs
```

It found a real miscompile, minimized: **`sar` was emitted as a LOGICAL shift**. The
emitter's unsigned arm rendered `((uint64_t)x >> cnt)` because registers are modelled as
one unsigned W64 value - a modelling artifact, not a fact about the operand, and the
same root cause as `movsx`. Fixed in `emit_c.rs`: read the source as signed at the
operation's width. Measured: **115 -> 39 mismatches of 732** (f(0x80000000,0x80000000,1,0):
machine 0xa0000000, C 0x60000000, now agrees).

The remaining 39 are one class, already bucketed by the harness: the C side sign-extends
a 32-bit result into an `int64_t` return (0xffffffffffffff9d) where the machine writes
EAX and leaves bits 32-63 of RAX as the ABI allows (0xffffff9d). Two honest options, not
yet chosen: return the value at the width the IR records for the return (which needs the
lifter to record the width observed at the return site), or compare at that width and
treat the upper bits as ABI-undefined. Masking the compare by inferring width from
observed answers was tried and rejected - it collapsed 19 functions to a 1-bit gate that
can never fail.

Also named: 29 functions are skipped because RAX is unwritten on some path, and the IR
dump shows why - at entry, `xor eax,eax` becomes `xor(trunc(%undef), trunc(%undef))` and
no pass folds `x ^ x -> 0`. An idempotent-op fold admits ~29 more functions to the gate.

## T2 - return width modelled: measured (2026-11-02), GREEN

The option chosen above was the first one, and the question it rests on was settled, not
assumed: on x86-64 a 32-bit register write **zeroes bits 63:32 architecturally**. A
function whose answer reaches RAX through such a write has an answer that is exactly
`zero_extend_32(value)` - bits 63:32 are a fact, not "undefined above bit 31". So the
emitted C returning `int64_t` and sign-extending was an over-extension in the OUTPUT, and
the harness's old framing ("the Win64 ABI leaves the upper halves undefined, so reading
the whole RAX is honest") was that over-extension wearing the ABI's clothes.

Three pieces, each measured:

* `rdec_core::width::bits_needed` - how many bits provably hold a value, read off the IR:
  a sound under-approximation that answers 64 ("any u64") the moment it cannot prove less
  (a cycle - a loop-carried phi - claims nothing, and `Phi` merges at its **widest** edge).
  `return_width` takes the max over every `Return` and gives 32 or 64.
* The lifter records it (`lift.rs::record_return_width`) as `f.ret_ty = uint32_t`, counted
  in `Report::ret_w32` and printed by both gates; `emit_c` casts the `return` through the
  declared type; `rdec t2`'s wrapper casts the **call expression**, whose type is that
  declared return type, so the C value is widened in C - defined behaviour, not a
  `uint64_t` pointer reading bits the callee never promised.
* `verify` checks the claim (a returned value needing more bits than the declared return
  type holds is a hard error), so no pass can quietly widen a returned value: the check
  fires on the IR, not on the emitted C, and `rdec c` counts the function as
  "invalid after passes" instead of emitting a lie.

The first measurement after that was still 39 mismatches, which is the check earning its
keep: `bits_needed` said 64 for saturation, and the reason was upstream in the lifter.
`lea (%rcx,%rdx), %eax` computes a 64-bit address and writes 32 bits of it, and
`write(g, W32, v)` "zero-extended" the full 64-bit sum - a no-op on an already-64-bit
value - so the emitted C added the caller's whole 64-bit registers and returned
`0x100000000` where the machine returned `0` for args=[0x80000000,0x80000000]. Fixed:
a W32 write truncates the value to 32 bits before the zext (the W8/W16 partial-write path
got the same narrowing).

```
lift: 120 objects, 141 functions lifted, 141 pass verify_ssa, 85 with no gaps and no exits
return width: 116 of 141 functions answer at 32-bit width
emit-c: 141 lifted functions, 84 emitted C, 57 refused, 0 invalid after passes; clang -Werror: OK
t2: runnable: 31 functions on both sides, 53 not runnable
return width: 30/31 of the runnable functions declare uint32_t
compared: 732 | match: 732 | mismatch: 0 | not compared (one side faulted): 0 | of the
  mismatches, differing only above bit 31: 0
OK: the decompiled C computes the same 64-bit result as the machine code on all 732 runs
```

The gate went 693 -> 732 matched of 732 compared, with the comparison set unchanged (same
31 functions, same 732 jobs) - no function was dropped to make it green, and the
"differing only above bit 31" bucket is kept as its own count: a nonzero value there names
a function whose recorded width is wrong, and it is still a mismatch.

The T2 gate is green for the first time. What it does NOT yet cover is the green part of
the corpus: 57 functions are refused by the emitter, 29 return RAX unwritten on some path,
14 touch memory outside the modelled frame. Width is modelled; coverage is the work.

## M4 steps 4-6 and M8 step 0 - measured (2026-11-02), all gates re-run by the parent

Everything below was verified by running the command on the committed tree, not read
from a report. Commits: `4a1be44`, `9ba1519`, `8bffd9b`+`3169162`, `26cdf48`,
`0459b3d`, `b692e79`, `08b399b`, `bac8258`, `ab189c2`. CI green on every one.

### The C emission count 56 -> 74 -> 84, from two type bugs in the flag lowerings

| fix | measure |
|---|---|
| fusion constants carry the operation's width (`SB::set_ty(Ctx.ty)`) - a fused SF of a 32-bit `sub` was `and(u32, 0x80000000u64)`, "mixes widths W32 and W64" | pass breaks 68 -> 21, emitted C 56 -> **74** |
| constant flag values are Bool (`CF`/`OF` of And/Or/Xor, ZF of shift-by-0, OF of SAR-by-1 returned `sb.c(0)`, an INTEGER) - the tree built `xor(bool, uint32_t)`, "mixes widths W1 and W32" | pass breaks 21 -> **0**, emitted C 74 -> **84** |
| `movsx` re-reads its source as signed at its own width (a same-width Bitcast) before SExt - the emitter was right to refuse `SExt` of a `uint32_t` | the 12-function `sext` class -> **gone**, emitted C 84, refused 67 -> **57** |

Accounting closes at every step: 141 = emitted + refused + pass-broken (84 + 57 + 0).
Note the ordering lesson: the type bug survived the width fix. It was never about width.

### T2 (differential execution) - landed, found a miscompile, and is now green

`rdec t2 --corpus`: links each corpus `.obj`, calls the machine function through its
COFF symbol (Win64 ABI, `RAX` read with inline asm), calls the generated C on the same
4-argument vectors, compares the FULL 64-bit `RAX`. Crashes are their own bucket;
unrunnable functions are named with a reason.

| T2 measure | first run | after `sar` | after self-idioms |
|---|---:|---:|---:|
| functions runnable | 31 | 31 | **44** |
| compared | 732 | 732 | **1044** |
| match | 617 | 693 | **1044** |
| mismatch | 115 | 39 | **0** |

What T2 caught that nothing else could: `BinOp::Sar` on an unsigned-typed operand was
emitted as `((uint64_t)x >> cnt)` - a **logical** shift. `sar` is arithmetic by
definition; the operand is unsigned only because registers are modelled as one unsigned
W64 value, the same root cause as `movsx`. 76 of 732 runs disagreed
(`f(0x80000000,0x80000000,1,0)`: machine `0xa0000000`, C `0x60000000`).

The last 39 were a deeper one: a 32-bit write of a 64-bit-typed value kept the full
64-bit computation, because the write path returned the operand unchanged when it was
already W64-typed - so `lea (%rcx,%rdx),%eax` emitted C that added the caller's whole
64-bit registers (`0x100000000` vs the machine's `0`). A write at width *w* truncates to
*w* first. The return width is now derived from the machine code (`width.rs` proves the
fewest bits that hold each returned value; a loop-carried cycle claims nothing, a phi
merges at its widest edge), and `verify_ssa` hard-errors when a returned value needs more
bits than the declared return type. 30 of 31 runnable functions declare `uint32_t`,
matching the architectural fact that a 32-bit write zeroes `RAX[63:32]`.

`xor eax,eax` (29 functions excluded as "returns RAX unwritten") was fixed in the
**lifter**, not `const_fold`: the strict interpreter faults on reading `undef`, so a
general `x^x -> 0` fold deletes a faulting read and changes meaning - the same trap as
DCE of a `getflag` read. `xor r,r`/`sub r,r` compute 0 **without reading** the operand,
which is a machine fact, and the lifter now emits that (flags still produced; `sbb r,r`
= -CF stays a named gap). The lifter's self-idiom counter says 81; an independent
`llvm-objdump` census of the corpus says 81.

**Scope of the T2 claim, stated plainly:** proven = the returned 64-bit value agrees on
1044 (function, input) pairs across 44 of 141 functions. Not proven = the other 97
functions (57 emitter refusals, 16 RAX-unwritten, 14 memory outside the modelled frame,
4 unlinkable file-static, 5 no COFF symbol, 1 `Exit`), and **memory effects** - T2
compares the return value, not writes to the flat store. A green T2 is a green sample.

### M8 step 0: MSVC C++ name demangling, gated against `llvm-nm` - 68/68

`crates/demangle/` (zero production deps) decodes MSVC x64 decorated names to
`Name{text, kind}` / `Unsupported{raw, reason}`, kinds = FreeFunction /
MemberFunction{class, constness} / Constructor(`??0`) / Destructor(`??1`) / StaticMember /
Vtable - the classification M8 consumes, not the pretty text. Gate `rdec demangle --cpp`
over 15 header-free C++ snippets: 69 defined function symbols, 68 supported, **68 agreed
with `llvm-nm`, 0 disagreed**, 1 Unsupported by design (template fixture). Fails on any
disagreement, on zero compared, and below a stated floor.

Measured, not copied: the planning brief's example `?area@Circle@@QEAA_NXZ` is **non**-const
(this-adjust `A` non-const, `B` const, `C` volatile, `D` cv); x64 refs are `AEAH`/`AEBH`
shaped and `CED` is three builtins, not a volatile ref. A 45-pair measured table is a unit
test. A name that demangles to something plausible is the failure mode, so templates,
rvalue refs, function-pointer types, arrays, RTTI, operator/new/delete names, data
symbols and anon-namespace labels each refuse with a named reason.

The C++ corpus exposed a **loader bug**: COFF long symbol names were read 4 characters
short (string-table offsets are relative to the table start, which includes the 4-byte
size field), so `??0Circle@@QEAA@N@Z` came out `ircle@@QEAA@N@Z`. Invisible to the C
corpus, whose names are short. The demangle gate is the proof it is fixed: symbols read
through that loader match `llvm-nm` by name 1:1, zero unmatched.

CI: `corpus --cpp` runs before the test step (the `.obj` files are generated, not
committed) and a step runs `demangle --cpp` with the same-toolchain two-build manifest
check - not `git diff` against the committed manifest, which compares clang 22.1.8 with
the runner's 20.1.8.

### Where this leaves M8

Names, ctor/dtor identification and the `this`-in-RCX convention are available and
measured. Nothing has lifted C++ machine code yet: `rdec cpp --corpus` is the first real
measurement (Kind census, lifted/valid/emitted/refused by class), in progress. Then
T2 for C++ free functions, then member calls with a pointer argument in the differential
(the harness currently excludes pointer args, so member functions are not T2-runnable -
stated, not faked), then vtable dispatch last.

## The phi miscompile, T2 at 1364 comparisons, and M8 step 1 - measured (2026-11-02)

Commits `48375c8`, `a569f0d`, `99da80e`. CI green on the stack; every number below was
re-run on the merged tree by the parent, not taken from a report.

### Third real miscompile: the emitter emitted a statement for a phi node

A phi's value arrives through the edge-copy at the end of its predecessor, so the phi op
has no C statement. The emitter emitted one anyway (`dest = 0`, the no-incoming-edge
initialiser), and a statement at the top of a block runs on EVERY entry - including the
back edge - so each iteration wiped the loop-carried value.

The corpus hid it for as long as no T2-runnable function had a phi (the preheader's
incoming value happened to be 0). T2 caught it the moment loop functions became runnable:
41 of 1335 runs, minimized `early_returns.O1 :: f(7,-1,40,3)` - machine `0x2a`,
decompiled C `0`, where the `mov eax,0x2a` edge copy into the return block was
overwritten by that block's own phi statement. Pinned by
`crates/arch-x86/tests/phi_edge_copy.rs`.

### T2 coverage 44 -> 64 functions, comparisons 1044 -> 1364, all matching

```
runnable: 64 functions on both sides, 20 not runnable (10 memory outside the modelled
          frame | 5 no COFF symbol to link | 4 file-static | 1 unmodelled control flow)
inputs: 12 vectors per function, 24 for the 50 loop-free ones -> 1368 comparisons
compared: 1364 | match: 1364 | mismatch: 0 | not compared (one side faulted): 4
OK: the decompiled C computes the same 64-bit result as the machine code on all 1364 runs
```

The "returns RAX unwritten" bucket is gone. The comparison did not shrink: same vectors
(more for loop-free functions), full 64-bit RAX, no masking, no width inferred from
observed answers. 4 jobs time out (`loop_do` at vector 8, a genuine infinite loop under
those inputs) and are counted as faults, never as matches.

**What T2 proves, exactly:** the returned 64-bit value agrees on 1364 (function, input)
pairs across 64 of 141 C functions. It does not prove memory effects (the flat store is
not compared), the other 77 functions, or anything about C++ - no C++ function has been
compared by T2 yet, because the harness takes integer args and reads RAX, so pointer and
float args are out of its range. Green sample, not a green decompiler.

### M8 step 1 - the first C++ measurement: 55 of 78 lifted functions emit accepted C

`rdec cpp --corpus` runs the same pipeline over the C++ corpus, naming functions from the
demangled text with the decorated name kept in a comment:

```
compared: 69 | lifted: 78 | valid IR: 78/78 | emitted C: 55 | refused: 23
this-in-RCX: 30 member/ctor/dtor functions lifted with the class pointer modelled as
             parameter 0; 18 of them emitted C
ctors identified: 1, dtors identified: 3 (the ??0/??1 prefix and the demangler Kind agree)
clang -Werror: OK
```

Kind census: 36 FreeFunction | 2 StaticMember | 13 MemberFunction | 13
MemberFunction(virtual) | 1 Constructor | 3 Destructor | 1 Unsupported(template).

Named refusal classes: 9 call-clobber, 6 `movsd`, 4 indirect call, 1 arithmetic with an
unmodelled operand, 1 rip-relative mov, 1 mov with unmodelled source, 1 `movss`. Lifted-IR
gap census: 126 call-clobber, 14 `movsd`, 5 indirect call, 3 `movss`, 2 rip-relative mov,
2 `addsd`, 1 by-value struct, 1 `cvttsd2si`, 1 `cvttss2si`.

Two findings that shape M8: **scalar SSE is the single biggest C++ cost** (every refusal
it causes is a float/double member), and **this clang build names zero vftable symbols**,
so vtable identity is not reachable from the `.obj` files - `.rdata` bytes plus REL32
relocations is the only route, which is why vtable dispatch stays last.

### Next, in order

1. Scalar SSE in the lifter and emitter (XMM file, `movsd/movss/movapd/addsd/...`,
   `pxor xmm,xmm` as the zeroing idiom modelled as a machine fact like `xor r,r`, and
   `cvttsd2si` guarded to the x86 magic constant because out-of-range is UB in C).
2. T2 for C++: float/pointer args in the differential - emission is not value-equivalence.
3. The 57 C refusals (call-clobber, `cdq`, `imul`, `div`, `lea` naming) and the 23 C++ ones.

## Gate ledger — every gate, one line, opened 2026-11-02 (tree `77e363c`), rows re-read from the gates 2026-11-08

Written because the roadmap recorded milestones as they were opened and left several of
their gate verdicts unstated. Each row is the command, the verdict from running it, and
what closing means. Rows are re-read from the command, not from memory, and a row that
disagrees with a section below it is a bug in the row: the sections are dated records,
this table is the live contract. The 2026-11-08 re-read found three rows stale — M1
discovery (RED at 97.88% -> GREEN at 100.00%), M3 noreturn/tail calls (NOT DONE ->
landed, with a corpus that exercises none of it), and the T2 fault story — which is why
the date and the tree are in the heading: a row is a measurement with an expiry.

| gate | command | verdict | closing condition |
|---|---|---|---|
| M1 step 1 decode/load | `rdec objdump-check --corpus` | **GREEN** — 124/124 boundary agreement | closed |
| M1 step 2 discovery | `rdec discover --corpus` | **GREEN — closed** (re-read 2026-11-08; this row said RED at 97.88% while section `:1562` had already recorded the closure, so the row was the stale one): 136 objects, 140 symbol-table functions, **0 missing, 0 extra starts**, coverage **6764/6764 non-padding code bytes = 100.00%**, uncovered 0 bytes in 0 runs, jump tables **2 recovered from relocations (12 slots, 12 targets, 0 unresolved)**, gaps none, exit 0. The 126 bytes in 2 runs that held M1 open were the 2 switch bodies reached only through a jump table, closed by target resolution (`f2fe789`). The 99.5% threshold was never moved to fit a measurement. | closed |
| M2 step 1 lifter | `rdec lift --corpus` | **CLOSED — recorded late.** Criterion was "every lifted function is valid IR"; it printed `FAIL: 40 lifted functions are not valid IR` at step 1, and now prints `141 functions lifted, 141 pass verify_ssa`, exit 0. It became true at `aa0d8e7` (M3 step 1) and the roadmap never said so. | closed |
| M2 instruction breadth, MSVC leg | `rdec msvc-gates --limit 40` | **MOVING — measured 2026-11-08.** The leg now emits **121 of 134** lifted (the row written earlier today: 118 emitted / 16 refused), census keys 6 → **5**: the class `inc/dec on an unmodelled destination` (3 sites, **all `incl (%rcx)`, all in MSVC objects** — clang lowers the same C as `addl $1, (%rcx)`, which was already modelled, so a clang-only census calls this arm dead code, and llvm-objdump prints it as `incl`: the same AT&T-name trap that nearly census'd `cwde` to zero) is gone. The arm loads the cell, adds/subtracts 1, stores the SAME address expression back, and sets `FlagsOp::Inc/Dec`; `addr_of` returns `Some` only for an operand it can express, so `inc dword [rip+0]` with no relocation stays REFUSED (fixture-pinned) instead of loading from an invented address. Silicon-pinned fixtures (`inc_mem_probe.c`): cell 9→10 via `[rcx+8]`, dec 9→8, 0x7FFFFFFF→0x80000000 with OF set. | the leg's coverage stays measured, not adopted as the M1 floor; and a class can be fully modelled with ZERO movement in this number — emission is per function, the census records one refusal per function, so a class's head count is its site count in the disassembly, never its record count (this slice: 34 memory-`cmp` sites, 3 records, 121 → 121) |
| M2 Win64 caller-saved clobber list | `rdec c --corpus` refusal census, `lift` clobber count | **CORRECTED — measured 2026-11-08 on branch `arity-wip` (`def1647`). Recorded as its own row because it is the largest correctness fix on that base and must not be read as a footnote of `arity`: the lifter's caller-saved list was SysV's, so it clobbered `RSI`/`RDI` after every modelled call — on Win64 those registers are NONVOLATILE (callee-saved). Witness: `movl %edx, %esi` after a call, where the machine's `ESI` survives and the model destroyed it (`rsi` arrives with the callee's answer, not the caller's argument). The list is now Win64's (`RAX RCX RDX R8 R9 R10 R11` + XMM args); **clobber cells 40 → 38**, and removing the two false clobbers alone closed **5 of the 8** `opaque: clobbered by call` refusals (the other 3 closed by the arity two-pass on the same branch). A register-ABI table is a machine fact about the TARGET platform, not a list of volatile registers: a SysV table under a Win64 target both over-clobbers (false refusals) and would under-clobber in the mirror case (the silent-miscompile direction) | a Win64-vs-SysV clobber fixture per ABI, pinned like `float_abi.rs` pins the pair rule |
| M2 memory-destination `cmp`/`test`, and RIP displacements | `acc_forms.rs` (9 fixtures incl. a silicon direction pair), `msvc-gates` census key | **ARM LANDED + SHARED-HELPER DEFECT FIXED + MISDIAGNOSED KEY RENAMED — measured 2026-11-08.** (a) `cmp`/`test` with a MEMORY destination (`cmpl %eax,(%rcx)`, `testb $1,(%rcx)`) loads the cell and compares **in Intel's order**, which is AT&T's last operand; the direction is pinned by a silicon pair (cell 5 vs eax 7 → CF=1 answers 13; cell 7 vs eax 5 → CF=0 answers 7 — F1 alone does NOT pin it, an `r − m` model answers 5 there), plus ZF/OF/byte-`test`/base+disp cases. Swapping `a`/`b` reddens this fixture AND the pre-existing `cdq_leaves_flags_alone`. These functions also assert `emit_c` succeeds (`cmp`/`test` define their flags, so a refusal means the arm lost the value). (b) `addr_of`/`rip_sym` resolved RIP displacements as `next_ip − 4` — right only when the disp32 is the instruction's LAST field. `cmpl $0x2, (%rip)` = `83 3D <disp32> <imm8>` has a trailing immediate: the field spans 2..6, the arithmetic lands at 4, the table keys 2 → the relocation was MISSED. Instrumented proof: `ZZ gap at 0x2f len=7 kind=Memory base=RIP hits=["0x31"]`. Covers `81 /7 id`, `C6 /0 ib`, `F6 /7 ib`, … Now resolved by scanning the instruction's span (the rule `reloc_on` already used); `RelocMap`'s doc corrected. (c) With (b) fixed the site STILL refuses, and the honest text now says why: the relocation names `g_count` with **no address** — the leg's map places CODE sections only, so a compare against a static in `.data` has no cell. New key `flag setting against a cell whose relocation names no address` (3 records moved to it; the old key asserted absent so the misdiagnosis cannot creep back). Census keys stay 5, records 13, emitted 121 of 134 | decide where data symbols live in the per-object space (M1/M4 scoping) → these 3 leg records become emissions; the clip rule for census keys is now shared between the two gates |
| CF across `inc`/`dec` (SS1.2 item 3) | fixture `acc_forms.rs` (memory + register, raw-IR and post-pass), `msvc-gates` lift gate | **CLOSED WITHIN A BASIC BLOCK — 2026-11-08.** `FlagsOp::Inc`/`Dec` now carry `cf_in: Option<VId>`: the flags value whose CF x86 preserves. Fusion lowers a CF read on a carried node to `getflag(CF, cf_in)` and the fixpoint lowers THAT by its producer's rule, so `cmp ecx,edx ; inc ; setc` emits `ecx < edx`; the interpreter reads the carried `Val::Fg`; `uses()` lists `cf_in`, so DCE cannot delete the producer under it. The fixture's F1/F2 now answer silicon's **12** and **10** and the CF-reading functions **emit**. Two edges are design, not loose ends: **(a) function entry** — with no producer, silicon's CF is the caller's, so the read lowers to an opaque refusal naming that, never a 0 (`m0_gate` b6 + a new fixture); **(b) across a join** — SSA has no flag phis (SS1.2 item 2), so `cf_in` is set only when the flags value was defined in the current block. Carrying it across a join is a use its block does not dominate, and the MSVC leg's `lift` gate caught exactly that on real objects (`%63 defined in bb5 does not dominate use in bb8` in `early_returns.Od.obj :: $LN10`; `%59` in `switch_fallthrough.Od.obj :: $LN9`) before the guard existed; a new fixture reproduces the hard error locally. Closing (b) needs flag phis = SS1.2 item 2. Measured with the change: clang leg unchanged at **156 lifted / 129 emitted / 27 refused / 0 invalid** (no clang-corpus function reads CF after an `inc`), MSVC leg gates exit 0 over 134 functions. | flag phis (SS1.2 item 2) for cross-block CF; the leg's 3 `incl` sites never read CF, so its headline staying 121/134 is the expected result, not a stall |
| M3 step 1 SSA | `rdec lift --corpus` | **GREEN** — 141/141 valid IR; strict interpreter: 0 uses of a value never defined | closed |
| M3 step 2 block splitting | `rdec lift --corpus` | **GREEN** — loops lift; Exits 64 -> 27, clean functions 63 -> 85 | closed |
| M3 noreturn / tail calls / jump-table targets | census + three floors inside `rdec c --corpus` | **LANDED, corpus witnesses present** (2026-11-08; the row before this one said "NOT DONE — zero handling" on the evidence of a `grep -ric` that returned 0, where the live count in `lift.rs` is 53). Jump-table resolution closed M1. Census now: **`tail calls: 3 modelled | 3 with no callable callee | noreturn calls: 4`** (`0xf -> abort`, `0x12 -> abort` x3) from the `tailcall_local` / `tailcall_external` / `noreturn_call` snippets. `cmd_c` FAILS on `tail + tail_unnamed == 0`, on `noreturn == 0`, and on any census site whose name `noreturn_symbol` does not declare — floor 3 exists because a sabotage that declared EVERY import noreturn moved one corpus function into `__builtin_unreachable()` and passed every other gate. The fixture found the real hole: the noreturn arm needed a signature entry, imports have none (`U abort`: no section, no size, no address), and in a linked binary every name in the termination family IS an import — the declaration is now made from the relocation's name, for imports only, `typed: false`, defines nothing. The emitted C is `o100_abort(x0,x1,x2,x3); __builtin_unreachable();` under an `__attribute__((noreturn))` declaration. | C++ corpus sites (that corpus imports no family name — measured with `llvm-nm`), and the `jmp`-to-import refusal class growing past 3. The CALLS side's remaining gap is now named: **frame activation** — modelled calls carry arity and clobbers, but every function runs in ONE shared frame (entry RSP is the `STACK_BASE` constant), so the frame-activation refusal (section 18) is the stop-gap until the frame becomes an implicit parameter |
| M3 stable IDs | (no gate exists yet) | **NOT DONE** — IR value numbers are per-function and not stable across passes, so cross-pass diffing is not available | needs a stable-id invariant pinned by a test |
| M4 C emitter | `rdec c --corpus` | **RED, by design** — 156 lifted / **129 emitted** / **27 refused** / 0 invalid after passes. 129/27 is a FRESH measurement on this tree **and** on `40d14e1` (measured directly, clean worktree, objects regenerated) — so the row's previous **125 emitted** was STALE, carried over from a 152-function tree, and the tell was its own arithmetic: 125 + 27 = 152 against 156 lifted. Commit `cca1aa4` published "125 → 129 after the folded-SExt fix", which is FALSE — the fix moved no emitted count; corrected here. clang -Werror OK. Classes (measured with `RDEC_C_CENSUS_ALL=1`, each site named): 8 call-clobber, 6 `lea` of an unnameable address, 6 `lea` naming `.rdata` (all six are `switch_dense`/`switch_fallthrough` at O1/O2/Oz — the jump-table base), **3 `tail call to 'ext'`** + **1 `call to 'ext'`** (the new `tailcall_external` witnesses, one per -O level: the honest answer is a refusal, a prototype for an unnamed callee would be invented), 2 live-in `Rsp`, 1 `shufps` (packed SIMD, M8). Counts moved 140 -> 156 lifted and 21 -> 27 refused because the corpus grew 16 functions, 4 of them refusal witnesses. The arithmetic classes stay gone: 10 `cdq`, 6 `imul`, 4 `div` landed 2026-11-08 (`acc_forms.rs`). Remaining named gaps: the 128-bit `rdx:rax` dividend of `div`/`idiv r/m64`, quotient overflow (`#DE`), CF/OF after a 32-bit `mul`/`imul` (declared undefined), and **frame activation** — the model has one frame per function, the machine one per activation, and a caller whose frame slots a callee clobbers while it is live is now REFUSED, class `frame slots alias the callee's` (section 18). **Re-measured 2026-11-08 on branch `frame-activation`** (base `arity-wip` `def1647`, which took the row's 129/27 to **137/19** by closing the 8 `opaque: clobbered by call` sites): **156 lifted / 135 emitted / 21 refused / 0 invalid**, clang -Werror OK; classes (each site named): 6 `lea` unnameable | 6 `lea` naming `.rdata` | 3 `tail call to 'ext'` | **2 `frame slots alias the callee's`** (`call_helper.O0::f` and the self-recursive `fib_rec.O0::f`) | 2 live-in `Rsp` | 1 `call to 'ext'` | 1 `shufps`; `opaque: clobbered by call` is GONE | **frame activation: model the frame as an implicit parameter so each activation gets distinct memory** (the refusal is the stop-gap; section 18), then refusals -> 0 with every gap named in between. **Re-measured 2026-11-08 on branch `frame2`** (base `ab64a4d`, this box's clang 22.1.8, objects regenerated): **156 lifted / 135 emitted / 21 refused / 0 invalid**, census byte-identical to the row above (6 `lea` unnameable | 6 `lea` `.rdata` | 3 `tail call to 'ext'` | **2 `frame slots alias the callee's`** | 2 live-in `Rsp` | 1 `call to 'ext'` | 1 `shufps`; sum 21 = refused), clang -Werror OK. The gate gained a **`frame traffic:`** line — **49 functions on named slots | 3 with unbounded (indexed) frame traffic (`o65_f`, `o66_f`, `o67_f` = `locals_addressed`, all leaves) | 104 with no frame traffic**, a partition of the 156 that the corpus gate asserts. What changed is the row's STATUS as a contract: 135/21 was a clang-version fact (the CI runner's clang 20.1.8 printed **136 emitted / 20 refused / 1 invalid** on the parent commit, because one frame refusal keyed on a frame address the 20.1.8 codegen spelled with an index register), and it is now a rule fact — an unbounded (frame-rooted, unresolvable) access counts as a whole-window touch on every codegen (section 18.1) |
| T2 differential, C | `rdec t2 --corpus` | **GREEN as a sample** — 112 harnesses over 115 runnable functions (**2472** planned comparisons), compared **2452**, match **2452**, mismatch **0**, differing only above bit 31: **0**. The 20 not-compared runs are ALL machine-side `#DE` (`exit 0xc0000094`, STATUS_INTEGER_DIVIDE_BY_ZERO) on `div_unsigned`/`divmod_signed` at vectors 1 and 22 — the zero-divisor vectors, where the machine traps, the interpreter faults `Fault::DivZero`, and the emitted C answers 0. The harness names the faulting side and excludes them from the match count, so "2452/2452" is a claim about the division cases the machine survives. Memory axis **408/408**, IR leg **408/408** (unchanged: the new snippets are pure return-value shapes). The **+192 since the 2260 measurement earlier today** is exactly 8 functions x 24 vectors — `tailcall_local::f` at four -O levels (its emitted C answers with `return helper(a, b);`, compared against the machine's sibling-call answer) and `noreturn_call::f` at four (the `abort(); __builtin_unreachable();` leg present, never driven, the return leg compared). The earlier +464 is recorded in the M4 slice below. **Re-measured 2026-11-08 on branch `frame-activation`** (base `arity-wip` `def1647`, where the arity slice took the tuple RED to **2607 | 2580 | 27 mismatch | 21** — the 27 all in `call_helper` 21/96 + `fib_rec` 6/59): the frame-activation refusal takes the two aliasing functions out of the compared set and the tuple is **2560 | 2560 | mismatch 0 | 20 not-compared**, exit 0 — **the base 2452 is untouched** (2560 = 2452 + 72 `call_helper` + 36 `fib_rec`; the refused `call_helper.O0::f` costs its 24 vectors and the refused `fib_rec.O0::f` its 23), memory axis **408/408**, IR leg **408/408** | XMM0 class, more coverage, memory effects beyond the caller window. **Re-measured 2026-11-08 on branch `frame2`** (base `ab64a4d`): tuple **UNMOVED at 2560 | 2560 | mismatch 0 | 20 not-compared**, exit 0, memory **408 | 408 | 0**, IR leg **408 | 408 | 0**, and a line-by-line diff of the gate's own output against the pre-fix run is EMPTY except the build's timing line — no function stopped being compared, because the widened rule's three unbounded functions (`locals_addressed`, see the M4 row) are leaves with no callee to alias with. The 20 `not compared` stay the machine-side `#DE` pair on `div_unsigned`/`divmod_signed` (section 18.1) |
| T2 differential, C++ | `rdec t2 --cpp` | **GREEN as a sample** (measured 2026-11-08, after this slice's emitter change) — compared **856**, match **856**, mismatch **0**, not-compared **0**, over 18 RAX-class + 16 window-class + 5 XMM0-class functions; 22 native drivers, 2 decompiled runners, 0 restarts after faults. Return value and XMM0 bit pattern; the window class compares the return, not memory. This row re-reads the gate because `SDiv`/`SRem` now read their operands with the op's signedness, and the C++ corpus is where signed `/` and `%` reach the emitter from C++ types — the change is green on 856 C++ machine runs, not only on the C corpus. | memory effects; the window class compares the return, not the window |
| M8 step 0 demangler | `rdec demangle --cpp` | **GREEN** — 68/68 supported symbols agree with `llvm-nm`, 1 template unsupported (named) | closed for the corpus; templates open |
| M8 step 1-2 C++ through the pipeline | `rdec cpp --corpus` | **GREEN as a sample** (re-read 2026-11-08; the row said "RED by design — lifted 78, emitted 62, refused 16", which is a dated tree: the gate today prints **69 measured / 69 lifted / valid IR 69/69 / emitted C 64 / refused 5** and EXITS 0 with "the rest are named gaps"). Refusal classes: 4 `indirect call` (vtable dispatch, M8's next step), 1 call-clobber. This slice moved nothing here — measured: the C++ corpus imports no name in the noreturn family (`llvm-nm` over `corpus/cpp/*.obj`), and `corpus/cpp` is untouched by the slice, so the shared sig-table change cannot reach it. The row's old numbers are a corpus/gate history this table never recorded, which is what a dated row is for. | refusals -> 0 (4 vtable dispatches + 1) |
| M8 step 3 C++ value-verified | `rdec t2 --cpp` | **GREEN as a sample** (row above) | — |
| M8 remainder | — | **NOT DONE** — packed SIMD (refused by name: `paddd/paddq/psubd/pmuludq/pcmpeqd/shufps`, mask `andps/orps`, GPR `movd/movq`), vtable dispatch and virtual calls. Measured: this clang build names ZERO vftable symbols in `.obj`, so vtable identity needs `.rdata` + REL32 relocation recovery | relocation reader, then vtable identity, then `call [this+k]` |
| M5 control-flow structuring | `e0dafc6` | **LANDED** — if/while/switch recovery; jump-table switch sites close M1's last gap. RE-MEASURED 2026-11-08 at `57747c0` (the row's 75 goto / 202 labels was an older, smaller corpus — the emitted set has grown since): **39/129 functions structured, 78 regions = 49 if + 15 if/else + 14 while + 0 short-circuit**, 228 blocks left as goto, and the reason census is the work list, in order: **143 return, 38 unstructured conditional branch, 32 fall-through, 10 region whose text left a named label undefined (flat form), 6 join is not this region's exit, 4 noreturn call, 3 tail call, 3 switch**. Emitted text after the readability exchange: **83 gotos / 246 labels / 64 if / 14 while** (the exchange prints 43 goto sites as `return`, absorbs 18 return blocks, 12 jumps by layout). `clang -Werror` OK on the emitted `corpus/c/all.c` | the 143 `return` gotos are M6's readability thread (early-return printing already absorbs 43+18); the 38 `unstructured conditional branch` are the structuring algorithm's real limit; flag phis (SS1.2 item 2) gate the multi-parent-join class |
| M6 readability | `47b9076` `bfe8156` | **PARTIAL** — early-return printing, layout-then-prune, return-tail materialization (refusals 29→0) | Ghidra baseline is a user install decision |
| M6 type recovery (tyinfer + `types`) | `8447945` `bd3f68b` + lane m6e + 2026-11-08 | **LANDED, GATED, IN CI, CENSUS CLEAN** — re-measured 2026-11-08 on the MERGED tree (148 objects, not the lane's 136): 128 functions scored, 640 sites, 324 unscoreable; signedness LIE 0 (exact 150, canonical-over-unsigned 30, sound-silent 132), width 153 exact + 27 narrower + WIDER 0, ptr 21/0, float 0/0, **void-return matches 4 / LIE 0** (the `carry_chain` family CLOSED — section 16), conflicts 25 loud; `types --selftest` battery **25/25** + register 4/4; `types --corpus` exits **0** (CI asserts 0 now, was 1); `--baseline .gate/types-baseline` **GREEN** on a baseline that declares `known-lies: 0`; CI runs the ratchet in `gates` and `release-parity` | the LIFTER's shape is unchanged and only the TYPE CLAIM was withdrawn: `lift.rs` still records `Return(read64(Rax))` unconditionally (a COFF object carries no signature), so the emitted C of a `void`-source function still prints a `return` with a value — T2-invisible (all four legs are outside the compared set) and now named, not silent; Ghidra baseline |
| M7 whole binary | `rdec pe-discover <exe>` (+ `pe-info`) | **GATE EXISTS (2026-11-08)** — on the real linked image: 418 functions, 56266/61431 .text bytes covered = **91.59%**, `overlap_free: true`, VERDICT line printing **213 gaps (213 descent + 0 reloc boundary)**; a 0-function run exits 1 with its coverage declared void. Seeds from three named sources with per-source provenance: `.pdata` (328 of 336 entries seeded, 4 chained + 4 version-2 refused and COUNTED — the seed census is the witness, measured: seeding a refused entry does not move the function count, because the linear sweep decodes with the same engine a seed uses), the entry point, and `.reloc` DIR64 code targets (47 targets in `.text`, 38 distinct, 7 `.reloc`-only with printed provenance). `.reloc` reader (`crates/arch-x86/src/reloc.rs`) asserts a PARTITION — padding + absolute + relative + unreadable-site + refused-kind == entries — cross-checked against a python raw-byte walk sharing no code (6 blocks, 782 entries = 3 ABSOLUTE + 779 DIR64, `sum(SizeOfBlock) == dir size`). NOT DONE: imports/IAT unread (160 indirect-call gaps stay anonymous, `is this gap an import?` unmeasured), PE32 has no `.reloc` leg (refused by name), the half-address relocation family is named not paired, ELF not started, frame-model interpretation not started | imports/IAT, then CRT signatures; the 7 `.reloc`-only starts are addresses, not verified bodies |
| M9 production | `289645b` `c08a894` | **PARTIAL** — fuzzing lane (0 crashes/0 hangs over 20k-input censuses ×3 seeds), nightly hardening workflow, release parity, panic-site ratchet, pass-pipeline + verify_ssa linear-time; the analyst loop is not built | analyst loop; fuzz generator extension to shift ops with raw counts (the P-track defect's domain) |
| P track (proof) | seed | **EXTENDED 2026-11-08** — the oracle has its own oracle: `interp_arith_reference.rs`, an independent machine-semantics reference, now over the 9 original liftable BinOps **plus the accumulator ops that landed**: `UDiv/URem/SDiv/SRem` at all four widths and `MulHi/MulSmul` at W64 (boundary set + 3000 deterministic cases per op x width, both `Sx` displays, raw shift counts, zero-divisor fault pinned). The multiply-high reference is a LIMB DECOMPOSITION, so interpreter (`u128`), table (limbs) and the shipped C helper are three implementations; `mulhi_helpers.rs` is the C-leg (PRELUDE helpers vs `__int128` over 300+ vectors, compiled with the floor gate's own flags, and the emitter's W64-emits / W32-refuses claim). First run found two real defects, both fixed in the interpreter: `Shr`/`Sar` UB for counts 64..=128 (debug panic, RELEASE WRAP) and `shift_left` disagreeing with the machine's count masking at raw 32..63 (W32) | **rotate leg LANDED 2026-11-08** (this row's own next-step item): `RotL/RotR` are now covered by the reference AND by a new emitted-C leg (`rot_emitter_c.rs`), which found the emitter's rotate expression to be arithmetically wrong — it masked the complement shift with `& w`, a POWER OF TWO, so a W32 rotate by 3 emitted `(x<<3) | (x>>0)` and rotate-by-0 shifted by the width (C UB). Both rotate arms in the interpreter were also wrong (unmasked result; `x >> (b - 0)` = **debug panic at W64, silent `x|x` in release**). Remaining: a `rol`/`ror` LIFT SITE with a corpus witness — rotates are IR surface the corpus never reaches (measured: `rol`/`ror` are not in `lift.rs`), so the emitted-C leg is a floor under the spelling, not corpus coverage; and C-emitter equivalence proofs beyond multiply-high and rotate |

The honest headline (updated 2026-11-08, floors above): closed structural gates
(decode, lift, SSA — verify_ssa linear-time), a differential that proves returned
values on the full emitted corpus (2452 comparisons, 0 mismatches, caller-buffer
memory 408/408) and 856 C++ vectors, the corpus's first tail-call and noreturn
witnesses sitting on an exercise floor that fails the build when a model has no
corpus site, control-flow structuring landed (75 goto / 202 labels across 97
emitted functions), the integer division and multiply
families emitting and machine-verified (arithmetic refusals gone: 43 -> 21 on the
old corpus, 27 on the 16-functions-larger one), a PE32+ loader
reading a real linked EXE, and a proof track that now covers the ops the lifter
can emit, three implementations deep. What remains: type recovery, whole-binary
discovery, packed arithmetic + vtables, and the analyst loop — correctness is
measured everywhere it claims; the gaps are named, counted, and printed.

### M8 step 4 - the float ABI, measured (`b60538e`, `0b4b2dd`)

| gate | verdict |
|---|---|
| `rdec cpp --corpus` | emitted C **62**, refused 16, clang -Werror OK; the four float C++ functions (`Circle::Circle(double)`, `grow`, `radius`, `fl`) now EMIT |
| `rdec t2 --cpp` | compared **688** / match **688** / **mismatch 0**; classes RAX 15 (360 runs) + WINDOW 12 (288) + **XMM0 5 (40 runs: 4 F64 at 64 bits, 1 F32 at its 32)**; buckets sum to 78 |
| `rdec t2 --corpus` | unchanged: 1364 / 1364 / 0, 4 faults |
| `c` / `lift` / `objdump-check` / `demangle` | 84/57/0 · 141/141 · 124/124 · 68/68 · 127 tests · clippy + fmt clean |

The machine fact that took five mismatches to find: **Windows x64 homes parameter position
`k` in the register PAIR that owns `RCX/RDX/R8/R9[k]` AND `XMM0-3[k]`** — so a member's
first `float`/`double` arrives in **XMM1** (`this` takes pair 0 in both classes), and a
free `double fip(int,double)` callee opens `movsd %xmm1,0x8(%rsp)` for the same reason.
Proof in the commit: `llvm-objdump -d` of the harness's own driver, where the padded
prototype put the real argument in XMM2 and the callee read its pair-owning XMM1 = 0.0.

**New named gap, latent, not yet a miscompile because the corpus has no witness:** the
lifter seeds *free*-function float parameters per register CLASS (`int f(int,double)` ->
XMM0) while clang's caller and callee home that `double` in **XMM1** by the pair rule.
`xmm_home_mismatch` in the T2 harness converts any such disagreement into a NAMED
exclusion (0 corpus functions hit it today) so it cannot masquerade as a mismatch.
Closing it is a lifter change (seed free floats by pair position) plus a corpus witness —
the corpus generator must grow an `int f(int a, double b)` fixture, and the gate must be
red before the fix and green after.

**Attribution, kept straight:** the parent's second hypothesis (the two sides preloading
different `this` images) was DISPROVED, not fixed — the machine answers equal `W[j][0]`
bit-for-bit on all five rows and the C answers are `W[j][0] + p`, which rounds to exactly
`p` where `W[j][0]` is tiny. One bug, not two. The disproof is in the commit message.

### The pair rule closed, and three things it exposed (2026-11-02)

`Lifter::new_abi` now seeds BOTH classes by declared position: a free
`int f(int a, double b)` seeds RCX + **XMM1**, `int g(double,int,double)` seeds
XMM0 + RDX + XMM2, and a member's `this` spends pair 0 in both classes so its first
`double` is XMM1 with no special case. Red-first, quoted in the commit; the corpus gates
moved by exactly nothing, which is the honest expectation with no corpus witness.

Three findings, all named, none papered over:

1. **The emitter's signed int -> float cast emitted C that does not compile.** `s` was
   the string `"(int64_t)"` inside `({s})({x})`, so every `cvtsi2sd`-from-signed produced
   `((int64_t))(x)` — clang: `expected expression`. Zero corpus witnesses, so no gate saw
   it. Pinned by two hermetic tests, including that the interpreter shows the f32 rounding
   (`cvtsi2ss` of 16777217 → 16777216.0) and so does the emitted C.
2. **The T2 home guard's lifter-side homes are a MIRROR of `new_abi`, not lifted data** —
   the seeded register per parameter is not visible on `Function`. `home_mismatch` takes
   the rule as `&dyn Fn` so a test can hand it the old class-index rule and prove it
   bites, which keeps it falsifiable. The remaining work is to carry the seeding out of
   the lifter as evidence and consume it in `m8.rs`.
3. **The corpus has no witness for the free pair rule.** Adding `int f(int,double)` to the
   generator shifts every corpus count (120 objects / 141 functions / 124 symbols / 68
   names / 1364 runs), so it needs a planned re-baseline; the witnesses are hermetic until
   then. This is the first entry in the ledger where a MISSING FIXTURE, not a missing
   feature, is what let a miscompile-class bug sit unnoticed.

Recorded as a testing fact: `F2` is the scalar-DOUBLE prefix, `F3` the scalar-SINGLE one.
The first draft of the `cvtsi2sd` witness used `F3`; the lifter correctly produced a
`float` and the emitted text looked like a lifter bug. Decode your own test bytes with
`llvm-objdump` before believing a red test.

### Direct calls modelled, and the evidence source that was missing (2026-11-05)

`Op::Call` defines its result; caller-saved destruction is a separate `Op::Clobber` per
register. That split is what moved the gates: `Op::Opaque` still blocks DCE, so an
unmodelled instruction can never vanish, while an unread clobber is genuinely dead and a
read clobber is still a loud refusal.

| gate | before | after |
|---|---|---|
| `c --corpus` | 84 emitted / 57 refused | **99 emitted / 42 refused**, 0 invalid; the 15 `clobbered by call` class is gone (57-15 = 42 exactly) |
| `cpp --corpus` | 62 emitted / 16 refused | **71 emitted / 7 refused** (9 call-clobber gone; 4 indirect call remain) |
| `t2 --corpus` | 1364 / 1364 / 0, 4 faults | **identical**, and the exclusion list now names `15 calls a symbol` where it was silent before |
| `t2 --cpp` | 688 / 688 / 0, buckets sum 78 | **identical** (the 9 newly-emitting functions moved refused -> excluded) |
| lift / objdump / demangle / tests | 141/141 · 124/124 · 68/68 · 133 | 141/141 · 124/124 · 68/68 · **143** |

**The finding that matters more than the numbers: `calls: 0 modelled` in both corpora.**
The emission rise came from the clobber split, not from modelled calls, because a clang
COFF object does not put the call target in the instruction. Verified with llvm-objdump on
`corpus/obj/call_helper.O0.obj`:

    10:  callq 0x15 <f+0x15>          ; the immediate is the unrelocated next-IP
    OFFSET 0000000000000011  IMAGE_REL_AMD64_REL32  h     ; the target is HERE

`Lifter::target()` reads the immediate, so it never equals the callee's address. The
missing piece is not "support calls" — it is **the COFF relocation table as an evidence
source**, threaded into lifting. One reader unlocks four recorded gaps at once: call
targets (the 15 `calls a symbol` T2 exclusions), the **jump tables that keep M1 red at
97.88%** with 2 unclassified indirect branches, the largest remaining C refusal class
(`14 lea of an address we cannot name`), and **vtable identity for M8**, where this clang
build names zero vftable symbols so `.rdata` + REL32 is the only route.

New refusal census, C (42 records in 42 functions, 8 classes): `14 lea of an address we
cannot name | 10 cdq | 6 imul | 4 unstructured conditional branch | 4 div | 2 live-in
Rsp`. C++ (7): `4 indirect call | 1 arithmetic-unmodelled | 1 rip-relative mov | 1
mov-unmodelled source`. Next by the ledger's own ranking: relocation-aware targets, then
`cdq`/`imul`/`div` (arithmetic facts, each with a red-first fixture), then M5 structuring
on top of real control flow.

### The relocation table is the evidence source M1 was missing (2026-11-05, `82c285f`)

Reading the COFF per-section relocation lists moved four recorded gaps at once, and
parent-verified:

| gate | before | after |
|---|---|---|
| `c --corpus` | 99 emitted / 42 refused | **105 / 36**, 0 invalid; `calls: 17 modelled` (was 0), `lea of an address we cannot name` 14 -> **6** |
| `cpp --corpus` | 71 / 7 | **74 / 4**; `calls: 7 modelled`, 9 of 222 relocs resolved as targets |
| `t2 --corpus` | 1364/1364/0, 4 faults | **identical**; the 15 `calls a symbol` exclusions gone, `unmodelled control flow (Exit)` 1 -> 16, buckets sum 141 |
| `t2 --cpp` | 688/688/0 | **760 / 760 / 0**, buckets 35 + 39 + 4 = 78 (callee definitions enter the TU by least-fixpoint, never referenced without being defined) |
| lift · objdump · demangle · tests | 141/141 · 124/124 · 68/68 · 143 | unchanged · **149**, `status` exit 0 with zero FAIL rows |

Two loader bugs were real and are fixed red-first. Symbol indexes in relocation records
count RAW table entries, aux records included; using the filtered index read `.llvm_addrsig`
where objdump reads `.xdata`. And `.bss` `PointerToRawData == 0` means "no data in the file",
not "data at offset 0" - the loader was reading the **file header** as `.bss` content, which
was a live miscompile with a T2 witness: `?count@Widget@@SAHH@Z`, 24 runs, machine `0x0`
against C `0x78664`. Parsed relocations are asserted equal to `llvm-objdump -r` on all 120
objects (173: 41 REL32, 132 ADDR32NB, 4 long names).

**M1's two jump tables are now a resolution problem with named evidence, not missing
information** (measured, not implemented; `discover` stays red at 97.88%). The 2 runs are the
O0 switch pair. `switch_dense.O0`: 80-byte run at `.text+0x2a`; table in `.rdata`, 32 bytes =
8 entries x 4, `IMAGE_REL_AMD64_REL32 -> .text` at 0x0,0x4,...,0x1c with the addend in the
entry bytes (`2e 3c 4a 58 66 74 82 90`) -> `.text+0x2a, 0x34, ... 0x8c`, the 8 case bodies,
first entry exactly the uncovered run's first byte; base named by `.text` `REL32 -> .rdata`
at 0x1d; dispatch `movslq` + `jmpq *%rax`. `switch_fallthrough.O0`: 46 bytes at `.text+0x32`,
`.rdata` 16 bytes = 4 entries. 12 relocations carry every target for both tables.

Next, by the ledger's own ranking: (1) jump-table target resolution in discovery, which
closes M1 and feeds M5 structuring; (2) `cdq` / `imul` / `div` arithmetic facts (10 + 6 + 4 =
the bulk of the 36 C refusals, each needing a red-first fixture); (3) the 6 remaining
non-RIP `lea` refusals, which are artifacts of mis-decoded blocks and therefore belong with
discovery, not with the emitter.

### M1 GREEN: 100.00% coverage, and the switch bodies are value-verified (2026-11-05, `f2fe789`)

The gate that was red since the beginning closed on the relocation evidence source, not on a
threshold change. `rdec discover --corpus`:

    coverage: 5956/5956 non-padding code bytes = 100.00%; uncovered = 0 bytes in 0 runs
    jump tables: 2 recovered from relocations (12 slots, 12 targets, 0 slots unresolved)
    gaps: none

N is derived from the count of consecutive 4-byte `.rdata` entries each carrying its own
`REL32 -> .text`, so nothing hardcodes 8 or 4; the C++ corpus, which is vtable- and
thunk-heavy, yields **0** tables, asserted. `Terminator::Switch{index, targets}` makes the
table the CFG's successor list, so verifier, interpreter, emitter, passes and T2 read the
same control flow; the interpreter faults `SwitchOutOfRange` past the end.

| gate | before | after |
|---|---|---|
| `discover --corpus` | 97.88% RED, 126 bytes / 2 runs, 2 indirect gaps | **100.00% GREEN**, 0 runs, gaps none |
| `t2 --corpus` | 1364/1364/0, 64 runnable | **1412 / 1412 / 0**, 66 runnable, same 4 timeouts |
| `c --corpus` | 105 / 36 / 0 | **107 / 34 / 0**, clang -Werror OK |
| `t2 --cpp` · `cpp` · `lift` · `objdump` · `demangle` | 760/760/0 · 74/4 · 141/141 · 124/124 · 68/68 | **unchanged** · tests 149 -> **163** |

Two real bugs came out of the switch path. **`CF` on a 64-bit `sub` was always 0**:
`((x-y-bin) & !mask) != 0` is exact at every width except the widest, where the complement of
the mask is 0 - so `subq $0x7,%rax; ja` read "above" for every index and the dense table
answered only for its last case. And the core passes did not know `Switch` (DCE deleted the
selector, `redirect` ignored the table), which is what made the switches comparable at all.

The CF test is the durable lesson here. It built two `Op::Param` nodes and never assigned
`f.params`, so nothing bound, both operands read as nothing, and CF answered "no borrow" for
every case: it **passed on the broken code**. After binding the params, restoring the old
`!mask` expression in this session makes it fail (`CF of 0x0 - 0x7: left false, right true`)
and the fix makes it pass. A fixture whose expected values come from the code under test
agrees with any bug - prove the red direction by putting the bug back.

M3 bookkeeping, grep-verified: jump-table target resolution is **done** by `f2fe789`
(`JumpTable` in `discover.rs`); `noreturn` and `tail_call` remain at **0** occurrences in
`lift.rs`/`discover.rs`/`ir.rs`. Of the three that remain, tail calls are now the cheap one:
the hard part - an indirect branch whose target comes from relocation evidence, consumed as
real CFG edges by every downstream reader - is exactly what landed, and a tail call is the
same evidence naming a function symbol instead of a table.

Next: (1) tail calls + `noreturn`, cheap on the new machinery and it feeds the T2 `Exit`
exclusion bucket (16 functions); (2) `cdq`/`imul`/`div` arithmetic facts (10 + 6 + 4 = the
bulk of the 34 remaining C refusals, each with a red-first fixture and a sabotage check);
(3) M5 structuring, which now has real control flow to structure.

### The corpus has 124 C functions, not 141: phantom starts deleted (`msg8` commit, 2026-11-06)

`rdec discover --corpus` printed `17 extra starts` for the entire project. They were
functions seeded from the **unrelocated displacement** of a `call`/`jmp` — the next-IP
immediate (`callq 0x15 <f+0x15>`), which in a clang COFF object is not the target at all. Now
targets seed from the symbol a relocation NAMES:

    discover: 120 objects, 124 symbol-table functions, 0 missing, 0 extra starts
    coverage: 5956/5956 = 100.00%; uncovered 0 bytes in 0 runs; jump tables 2/12/0; gaps: none

New gate baselines from here on (the old 141/107/78/74/760 numbers are history, and the
re-baseline is arithmetic, not drift):

| gate | was | now |
|---|---|---|
| `c --corpus` | 141 lifted, 107 emitted / 34 refused | **124 lifted, 86 emitted / 38 refused, 0 invalid** |
| `cpp --corpus` | 78 lifted, 74 emitted / 4 refused | **69 lifted, 64 emitted / 5 refused** |
| `lift --corpus` | 141/141 | **124/124** valid IR, exit 0 |
| `t2 --corpus` | 1412/1412/0, 41 excluded | **1412/1412/0, 20 excluded** (`16 memory outside the modelled frame \| 4 symbol is file-static`) |
| `t2 --cpp` | 760/760/0, buckets sum 78 | **856/856/0**, buckets `39 + 25 + 5` sum 69 == lifted |
| tests | 163 | **176**, clippy/fmt clean, `status` exit 0 |

141 − 17 phantoms = 124 lifted; 107 − 17 = 90, and 4 of those real functions now live in their
true caller where a call clobbers a register they read, so they are refused loudly: 90 − 4 = 86
emitted, 34 + 4 = 38 refused, 86 + 38 = 124. **Emitted-per-real-function went 75.9% -> 69.4%:
that is the honest cost of deleting invented functions, and it is the number to improve from
now on.** T2's exclusion list is the tell — the 16 `unmodelled control flow (Exit)` and 5
`no COFF symbol to link` entries were CFG guard-dispatch thunks (`lea rcx,[rip+_tail_call_
indirect@N]; jmp [rip+__guard_x86]`) that existed only as phantoms.

M3 status after this: jump-table resolution done, tail calls modelled, noreturn modelled,
stable IDs open. **The corpus has 0 tail calls and 0 noreturn calls** — 65 `jmp` sites, 0 with
a relocation naming a defined function (the relocated ones name `__guard_x86` x23); 31 call
sites, none to abort/exit/__fastfail/a throw. The model is pinned by hermetic fixtures with
embedded bytes and quoted objdump ground truth, never by a fabricated corpus witness.

Red-direction proofs on this commit, each put back and shown red then restored: `jmp` reads the
immediate -> `a_direct_jmp_inside_one_function_is_a_not_a_tail_call` red; `TailReturn` reads
RAX -> red (the verifier hard-errors, a `jmp` computes no value); `noreturn_site` falls through
-> 2 red; **noreturn read from the callee's name -> the anti-invention test goes red** (left 1,
right 0); emitter drops the tail-return statement -> the emitted-text test goes red.

Named next steps, in the order the evidence ranks them: (1) the 38 C refusals — `cdq` (10),
`imul` (6), `div` (4) arithmetic facts, and the grown `opaque: clobbered by call` class (8)
which is now about real callers, so it is worth modelling register liveness across a call;
(2) the MSVC CFG guard dispatcher, the only route to the 16 guard-thunk regions and the
corpus-visible tail-call witness; (3) M5 structuring, now on a truthful CFG of 124 real
functions; (4) stable IDs, M3's last open item.

### Gate ledger: `fcee470` — recognition rules must not depend on the compiler version

CI was red on 4 tests that were green locally, and the fix is a fact about the project's
verification architecture, not about jump tables.

The CI probe (run 42166188651) dumped the runner's OWN objects. CI's clang is 20.1.8, this box
is 22.1.8, and for the same C source the two compilers emit the same `.rdata` bytes, the same 8
relocations and the same 8 targets, with the jump-table fold's register roles swapped:

    clang 22:  movslq (%rax,%rcx,4),%rcx ; addq %rcx,%rax      delta loaded into the BASE
    clang 20:  movslq (%rcx,%rax,4),%rax ; addq %rcx,%rax      delta loaded into the INDEX

The fold check pinned a lowering (`split(load.op_register(0)) == split(add.op_register(1))`),
so the corpus's two switches modelled locally and were REFUSED on CI. Measured damage, both
runs green: CI census `switches: 0 modelled` and `t2 --corpus compared: 1364` against **1412**
locally. The floors are CI-calibrated, which is exactly why a green gate was measuring a
smaller claim than local was.

The rule now cites machine facts that survive any lowering: the indirect jump reads the fold's
destination; the fold's two operands are the relocation-named table base and the delta loaded
at 4-byte scale off that base; every slot target is a block this function owns. Exactly two
role assignments satisfy that; `add r,r` is rejected as degenerate.

Post-merge verification, parent-run: `t2 --corpus` on CI now reports **compared: 1412 | match:
1412 | mismatch: 0** — the number that was the symptom is now the proof. Locally unchanged
(124 lifted / 86 emitted / 38 refused / 0 invalid; 1412/1412/0; 856/856/0; 100.00% coverage;
180 tests; fmt/clippy clean; `status` exit 0).

Two rules promoted to the skill: (1) when local and CI numbers differ, that difference is a
finding — report both numbers and identify which claim each side measures; (2) a recognition
rule must never name a register role one compiler chose, and every accepted shape gets an
embedded fixture (fixtures A and D in `jump_table_fixtures.rs` are the two roles of ONE switch).

Open, named: the `aa == ab` (degenerate `add`) guard has no fixture witness — a hand-built
object for it needs the builder's REL32 addend convention, and a first attempt produced targets
`[0x2, 0x4, 0x6, 0x8]`. It is defensive, not load-bearing for any observed shape.

Two lanes are in flight on this HEAD: production robustness (`rdec fuzz`, hostile-input hardening
across loader/demangler/discovery/decoder/emitter) and M5's first structuring slice (`if` /
`if-else` / `while` recovery, conservatively, with T2 mismatch 0 as the semantic proof).

### Gate ledger: production hardening — `c584a10` ratchet + release-mode parity

**Release-mode parity, measured on `c584a10` in an isolated worktree with its own
`CARGO_TARGET_DIR`** (the shared working tree had two children editing it, and a build
there measures their in-flight code, not a commit):

    discover  120 objects, 124 functions, 0 missing, 0 extra starts,
              5956/5956 = 100.00%, 2 tables / 12 slots / 0 unresolved, gaps: none
    c         124 lifted / 86 emitted / 38 refused / 0 invalid; clang -Werror: OK
    lift      OK: every lifted function is structurally valid IR
    objdump   OK;  demangle 68/68
    t2 C      compared 1412 | match 1412 | mismatch 0 | 4 faults (the same loop_do timeouts)
    t2 C++    compared 856 | match 856 | mismatch 0; buckets sum 69 == lifted 69
    status    exit 0, zero FAIL rows
    cargo test --release --workspace   180 passed, 0 failed

Every number equals the debug-mode floor, and the suite passes with `debug_assert!`
compiled out — so the decoder's asserts are exercised by the debug suite and are not
load-bearing for any answer the release build gives. This is now a standing CI job
(`release-parity`), not a one-off measurement: `--release` is what a user runs.

**Panic-site ratchet.** Every `unwrap()`/`expect()`/`panic!()`/`todo!()`/`unimplemented!()`
in the repo was read (99 sites; 30 in library non-test code). One is a certain panic on
realistic input — `loader.rs` sorting symbols by `symbols[i].section.unwrap()`, where
`section` is `None` for `SectionNumber == 0` absolute symbols that every linked object
carries. The demangler's `peek().unwrap()` sites and `interp.rs`'s `d.cf.unwrap()` are
assigned to the robustness lane with red-first fixture requirements. `cfg.rs:66`,
`flags.rs:241`, `lift.rs:1851` and the four `sig.unwrap()`s were checked and are guarded —
recorded so the next audit does not re-litigate them, and so nobody "fixes" a non-bug.
`scripts/unwrap_ratchet.sh` + `.gate/unwrap-baseline` pins the per-crate count, fails on a
rise, fails on a fall (with an instruction to lower the baseline), fails on a missing
baseline, and runs in both CI jobs.

### Named gap: the middle end runs out of pass budget on ordinary functions, and said "converged"

Found by `crates/core/tests/pipeline_scaling.rs`, which exists to ask the questions the
corpus cannot ask (its biggest function is ~40 blocks).

`run_pipeline` loops to a fixpoint capped at 32 rounds, and `merge_blocks` / `dedup_blocks`
collapse **one** thing per call. A phi-free carry chain of 250 blocks therefore needs far
more rounds than the pipeline has, hit the cap, and the report the gates print said:

    ladder(250): converged after 33 rounds

That is a false claim about the work: it stopped, it did not converge. Fixed: the report now
distinguishes the two, with `PipelineReport::not_converged` and the line
`stopped: pass budget (32 rounds) exhausted, passes still changing`. Red direction proven on
the final tree: restoring `if !changed || iter > 32` reproduces `converged after 33 rounds |
not_converged=false` and the test fails; the fix passes. A converging shape (`pairs(8)`,
`ladder(5)`) is asserted NOT to be flagged, so "always report non-convergence" cannot pass.

Two things this exposes, both open, both measured here, neither papered over:

1. **Super-linear middle end.** Printed scaling census (debug, this box):
   50 blocks 23.7 ms, 100 blocks 63.3 ms, 200 blocks 219.4 ms — 2x the blocks costs ~3.5x,
   and a 20k-block function did not finish in 60 seconds (kept as
   `a_20k_block_function_finishes_and_answers_the_right_number`, `#[ignore]`d with that
   measurement in the reason, run it with `cargo test --release -- --ignored`).
   The scaling test's own bound (200/50 blocks ratio < 25) is a regression net for the growth
   getting WORSE, not a claim that today's growth is acceptable.
2. **The budget is the reason, and the fix is not a bigger number.** Raising 32 makes the
   pipeline slower (rounds x passes is where the quadratic lives). The right fix is in
   `passes.rs`: apply all merges/dedups in one call so rounds become a small constant —
   which touches the same file M5's structuring pass lands in, so it is sequenced after.

Also named: `rdec_core::emit_c` refuses `Op::Cmp` whose operand is an `Op::Const`
(`comparison Ult on a  operand` — IR constants are width-agnostic, `Op::Const`'s `ty()` is
`None`, and the emitter's comparison path needs a width from the operand). No corpus function
shows it because the lifter's comparisons go through flags + `GetFlag`. Fixture cost: three
iterations to route around it.

---

## M5 — Control-flow structuring: LANDED 2026-11-06 at `e0dafc6` (CI green)

The algorithm is pattern-directed region recognition in `crates/core/src/structure.rs`, chosen
against the corpus, not imported: a region is structured only when the forward-closure and
single-join gates PROVE it, and the honest fallback is the label + `goto` form that was always
correct. `No More Gotos`-style edge splitting is not used, because it duplicates the guard's
WORK, and duplicated work is duplicated semantics in a decompiler whose contract is equivalence.

Measured before → after on the corpus (86 emitted functions):

    goto statements   318 → 123        labels   323 → 206
    structured        0/86 → 35/86 functions
    regions           72: 50 if | 12 if/else | 10 while
    blocks left as goto: 38 fall-through, 121 return, 34 unstructured conditional
                         branch, 8 region whose text left a named label undefined,
                         5 join is not this region's exit, 2 switch
    t2 --corpus  1412 | 1412 | mismatch 0 | not compared 4     (the floor, exactly)
    t2 --cpp     856 | 856 | 0
    clang -Werror: OK

**The bug T2 caught, and the class of bug.** Structuring's first loop form printed the header's
WORK (the compare that computes the tested value) ONCE, above the loop. The C never
re-evaluated the guard, so every corpus loop spun, and the interpreter timed out on it:
`t2 --corpus` dropped from 1412 compared to **1403 with 13 not-compared**, while `mismatch`
stayed at 0 — because the IR interpreter reads the GRAPH and the C side faults before
comparing, so the comparison count is the only thing that moves. A green gate measuring LESS
is the cardinal sin, not a detail. The fix rotates the TEXT (the guard's work is spelled on
the entry edge and again at the body bottom, the back edge; label on the in-loop copy, so a
`goto` to the header means "test again"): text duplication, not work duplication, so a store
in a header still runs exactly the machine's number of times. `t2.rs` now prints
`NOT COMPARED: <function> <vector> <side> <fault>` for every lost comparison, so a shrinking
claim cannot hide behind a clean mismatch count again. This is why the gate reports the
4-tuple `(compared, match, mismatch, not-compared)` and the FLOOR includes the compared count.

The same class, caught by the same mechanism earlier in the pass: `pre_loop` decided
body-in-then vs body-in-else BACKWARDS (36 mismatches, `mismatch` non-zero — this class IS
detectable when both sides run). Polarity is asserted per fixture by quoting the emitted
`if (`/`while (` condition line, with the negation asserted ABSENT, so "the body moved to the
arm the machine code skips" is a test failure, not a review finding.

Red directions proven by sabotage on the final tree (each applied alone, test quoted, restored):
guard polarity negated; if/else tests the negated value; `pre_loop`'s body arm inverted; `while`
→ do-while; **header printed once** (the shipped bug); `region()`'s gates dropped (irreducible
CFG must stay a `goto`, and `an_irreducible_region_stays_...` is the anti-invention test);
emitter printing a tree with an undefined label. Two gates were measured and are reported as
NOT load-bearing (`pre_loop`'s body-predecessors gate and arm-exclusivity gate: a later gate
already refuses the fixture) — a gate nobody can break is not evidence.

Honest narrowings, not weakened needles: `a_multi_entry_loop_is_not_structured` asserts the
pass proposes **no loop**, not zero `if`s — measurement showed the walk proposes 2 provable
`if`s there, and the reason that tree is unprintable is an undefined label, which is caught
and counted. Fixtures that use `Op::Param` assign `f.params` (a fixture that binds no argument
agrees with any bug). Test-time C compilation is the single `-Werror` acceptance call.

M5 gate status: if/else/while are recovered and T2-equivalent; jump tables are modelled as
`Terminator::Switch` with resolved targets (M1's relocation evidence) and the emitter keeps
them as `switch` (2 tables / 12 slots in the corpus, 0 unresolved). The 90%-as-`switch` gate
is NOT claimed: the corpus has 2 tables, and a percentage over 2 is a story, not a measurement.
`for`-loop recovery (three-clause canonicalisation) is not built; `for` appears in the corpus
as a `while` with an induction variable in the body, which is T2-equivalent and readable, so
this is a readability gap (M6), not a semantic one.

## Fuzz/robustness lane: LANDED 2026-11-06 at `289645b`

See the commit for the full account. The load-bearing facts:

1. **A 4 KB object was a 300-second DoS on the tool's own users**, and no wall-clock deadline
   outside the interpreter could interrupt it: the step budget charged instructions, the input
   lifted to an EMPTY self-loop block (`bb1: br bb1`), so every lap charged 0 and
   `steps > budget` could never fire. Fixed by charging one step per block entry — a control
   transfer is work. 300 s+ → 2 ms; fuzzer seed 1: never finished → 12.3 s for 3000 inputs.
   `Exec::diff` deliberately ignores `steps`: TV compares MEANING, and a pass that legitimately
   deletes an instruction must not show up as a difference.
2. **The fuzzer finds injected bugs, proven, not asserted.** Re-injecting the demangler
   truncation panic reddens the CI fuzz test naming seed, iteration, generator, and the repro.
   Deleting the step charge reddens the hand-counted step accounts (3 vs 5) and turns the
   self-loop test into the production hang (timeout kill).
3. **A stack overflow is not a panic.** `?f@@YAH` + ~1000 nested `P` produced
   `STATUS_STACK_OVERFLOW (0xc00000fd)`, which `catch_unwind` cannot catch: the demangler is
   depth-bounded (`MAX_TYPE_DEPTH`) and reports counted `Unsupported`.
4. Six census runs (seeds 1, 2, 20261106 × debug, release): 0 crashes, 0 hangs, 0 over-budget,
   **no release-only crash class**. Debug ~12 s, release ~2.8 s per 3000 inputs.

The ratchet moved DOWN where the lane deleted real panic sites, per the gate's own instruction:
`core 2 → 1`, `demangle 3 → 1`; `arch-x86 7`, `verify 4`, `cli 14` unchanged.

Named decreases, said out loud: the `m4` API adaptation dropped the census counters
`discover.bytes`, `discover.tables`, `discover.table_slots`, `discover.table_unresolved` and
the per-`Gap` `discover.<gap>` loop, because master's `Vec<Subject>` carries no plan census.
Re-based equivalents from the lifter's own report: `lift.jump_tables`, `lift.jump_table_edges`,
`lift.jump_table_partial`, `lift.jump_table_missing`, `lift.insns`. A census counter is removed
by name, never by silence.

## Middle end at production scale: LANDED 2026-11-06 at `fe425cf`

The scaling test's named gap is closed with the root cause MEASURED, not guessed
(`RDEC_PASS_TRACE=*** on `ladder(250)`, 256 lines):

* `merge_blocks` collapsed **exactly one edge per call** — `changed=true` in 32 of
  32 rounds, 251 live blocks → 220, every delta 1. A 250-block chain owes ~250
  collapses; the 32-round budget fired on IR that converges in a handful of sweeps.
* `dead_code_elimination` deleted NOTHING (0 changed rounds) and ate **89.3%** of
  the runtime (1962 ms of 2196 ms): it asked `Function::use_count` — a full node
  scan — once per instruction, every round, and the per-round cost never shrank
  (round 1: 67.2 ms; round 32: 70.0 ms).
* Shape of the defect: `time = rounds(∝ blocks, capped 32) × round-cost(∝ nodes²)`.

The fix is structural, not a bigger budget: every pass collapses to its OWN
**bounded** fixpoint (`nodes+blocks+16`, min 32); merges sweep every eligible edge
per call (chains halve per sweep); the per-node passes answer from per-sweep tables
so a sweep is O(nodes). `PipelineReport` gained `rounds` and `pass_invocations`
and stays truthful: `not_converged` is true only when the outer budget is
exhausted with passes still changing, or a fixpoint bound is exhausted — never a
function of size. A driver bug fell out of the wiring: a fixpoint loop's LAST call
always reports "no change", so the round must OR across calls — the pre-fix shape
hid a false-convergence class.

Measured (debug): 50 blocks 100 ms → 0.92 ms; 1k 8.5 s → 17.2 ms; 5k >1500 s
(killed) → 89.6 ms; **20k: never finished → 357 ms, converged after 2 rounds,
6.3 MB of C emitted, answer 10002 = N/2+2 hand-derived**. Growth 1k→5k is ×5.2
for ×5 blocks — linear. The 20k probe is un-`#[ignore]`d and asserts the answer.

Red-first and sabotage, all in the test file's history: the 250-block convergence
test quoted `stopped: pass budget (32 rounds) exhausted` BEFORE `src/` was touched;
single-collapse-per-call and cap-at-1 sabotages redden it; a synthetic oscillating
pass is caught by the bound (RED, terminates in 65 s) and with the bound REMOVED
never answers (exit=124) — the bound is load-bearing; no oscillating input is
reachable in the corpus, so the bound is stated as defensive. Honest redundancy
finding: capping only the INNER loop with the driver's re-call net intact stays
GREEN — what is load-bearing is "re-call until no change, multi-collapse per
call", not the counter.

Floors: census lines byte-identical (diffed against a reverse-patched baseline),
t2 1412|1412|0|4, t2 --cpp 856/856/0, cpp 64/5, discover 100.00% gaps none.

New named gap, measured, not papered: **`verify_ssa` on 20k blocks costs 274.7 s**
(quadratic; `crates/core/src/verify.rs` was outside this lane). The release-only
ignored test carries the number in its reason. Fix shape: the same per-sweep-table
pattern; the acceptance test is the 20k probe un-ignored on the SSA-verification
leg.

## M9 evidence: 20k-input fuzz census at `78f23c7` (clean worktree, own target dir)

    debug   seed 7     20000 inputs  7.4 s   crashes 0  hangs 0  over-budget 0  slowest 10 ms
    debug   seed 123   20000 inputs  7.6 s   crashes 0  hangs 0  over-budget 0  slowest 10 ms
    debug   seed 31337 20000 inputs  7.6 s   crashes 0  hangs 0  over-budget 0  slowest  9 ms
    release seed 7     20000 inputs  3.0 s   crashes 0  hangs 0  over-budget 0  slowest  2 ms
    (load refusals ~4.4k and gaps ~2.7k per run, all counted loudly; ~4.5 KB inputs)

The throughput is itself a datapoint: the fuzz census used to take 12.2 s for 3000
inputs with a 146 ms slowest input (`289645b`); at `78f23c7` the same surface does
20000 inputs in 7.4 s, slowest 10 ms - the pass-pipeline fixpoint work (fe425cf)
sped up every front-half consumer, which is what "the middle end scales" means
when measured at the product's own fuzzer.

## T2 memory-effect equivalence is real (2026-11-06) — the return-blind scribble is now caught

The gate that mattered most now exists: T2 compares CALLER BUFFERS, not just return
values. `t2 --corpus` = `1700/1700/mismatch 0/not-compared 4` (the 1412 return axis
preserved exactly, +288 window vectors; the 4 named `loop_do` faults kept), plus
`memory: 288 | match 288 | mismatch 0` and `ir leg: 288 | 288 | 0` on the same seed
bytes. Mapping printed by the gate: `native buffer byte k == rdec_mem[0x2000+k] ==
IR byte at 0x2000+k`; admission is a named rule (every memory touch outside the frame
must chain from param 0; rejections name the instruction).
Fixtures: `store_to_buffer` (the return-blind scribble), `copy_words`,
`read_only_words` x4 opts = 12 objects; 15 pre-existing window functions admitted.
SABOTAGE (the point of the lane, all four directions in `crates/cli/tests/t2_memory_effects.rs`):
store-shift-by-8 -> `return 0x10 AGREES and the caller buffer DIFFERS: word 1 (byte
offset 0x8)` with return `mismatch: 0`; spurious prologue store -> `read_only_words`
red on memory only; comparator sabotaged to lengths-equals-bytes flips the SAME run
green - the red lives in the bytes.
Floors moved UP by design: `c --corpus` 124/86/38 -> 136/96/40 (the +2 refusals are
`copy_words` O2/Oz lowering to `movdqu` - the M8 packed-SIMD gap with named corpus
witnesses), discover 5956 -> 6514 (100.00%), objdump-check 136 exact, workspace 226.
The CARGO_TARGET_DIR trap fired a third time (this session: leaked from a dispatch
build into the parent shell; and mid-run in the child): every result above was
re-measured after `unset`. The export rule for dispatch commands is a hazard, not
just for interactive shells - use `env CARGO_TARGET_DIR=... cargo ...` per-command.

## `verify_ssa` is linear: the 20k-block verify leg 274.7 s -> 9.5 ms (release), contract pinned first

The named gap from the perf ledger is closed. Census (same ladder, ms per single
verify_ssa, release): 500 blocks 119.3 -> 0.3; 2k 2732.8 -> 1.0; 8k 43211.7 -> 3.9;
20k **271775 -> 9.5** (debug 1136005 -> 40.4). x40-blocks growth: x2278 -> x34.8 —
work-per-node, not quadratic. The shape is the perf lane's own: a `Sweep` (one pass
-> def site, def census, use sites, block-id index) + `DomIndex` (DFS [pre,post]
intervals, O(1) dominance, exhaustively parity-checked against `Cfg::dominates`).
CONTRACT FIRST: `crates/core/tests/verify_ssa_rejections.rs` — 54 fixtures pinning
every rejection verify_ssa implements, written and sabotage-proven BEFORE the
rewrite, and proven to pass against HEAD's OLD verifier in a clean clone (the suite
pins the contract, not the new implementation). 13-entry sabotage table: each check
deleted -> its fixture goes red with `was ACCEPTED by verify_ssa`; `DomIndex::
dominates -> true` reddens a diamond parity test. Named unwitnessed check: the
`bad use-site for %{v} in bb{}` message is defensive (stays green when deleted, as
the suite header says).
All 49 diagnostic strings preserved; the 20k verify test un-ignored with measured
bounds (release 600 ms = 50x headroom, debug 5 s = 118x). Floors: t2 1700/1700/0/4 +
memory 288 + ir 288 byte-preserved THROUGH the verifier rewrite (the oracle is
load-bearing for every leg — lift 136/136, workspace 283/0, fmt/clippy/ratchet/status
clean). Parent re-measured everything on c5374de + patch; child's environmental
notes (clang 0xc0000142 under lane load, mid-build file swap) are its own, not tree state.

## M6 first landing: 24 gotos out of the emitted C (123->99 class delta), floors byte-preserved

The readability pass on the census's own two biggest reasons, integrated from the
isolated lane (base 78f23c7, applied to master untouched-since: structure.rs,
emit_c.rs, structuring.rs, m4.rs):
- **early return**: a `goto l_end;` whose target block is a pure `return EXPR;`
  prints `return EXPR;` at the site, spelling the phi copies on the edge above it
  (10 sites; 2 return blocks absorbed -> label gone). Guarded by SSA dominance and
  a pure-tail check; the refusal is a NEW COUNTED REASON, printed by the census:
  `early return refused: target computes` (29 in this corpus — the returns are
  `leave; ret` epilogs whose block ends in `add/mov`; peepholing that is a
  passes/ir question, the next M6 thread, not a print trick).
- **layout-then-prune**: flat DFS layout absorbs 14 jumps whose target prints next
  (fall-through 38 -> 24; remainder is back edges + multi-parent joins the layout
  provably cannot place).
Readability on the current corpus: 124 -> 100 goto, 217 -> 215 labels, `readability
exchange:` line reports the ledger (10 + 2 + 14); blocks-left-as-goto 211 -> 195.
Floors re-measured by the parent on master+patch: t2 1700/1700/0/4 + memory 288 +
ir leg 288 BYTE-PRESERVED (the compared count is the floor), t2 --cpp 856, c
136/96/40/0 + clang -Werror OK (no emission flipped), structuring 12 -> 18,
workspace 289/0, fmt/clippy/ratchet un-piped exit 0, status exit 0.
Red-first with sabotage: dominance->true, absorb-without-adjacency, print-return-
over-a-store (that fixture RUNS 5 vectors: the C returns 00 where the graph returns
99), absorb-a-named-return-block — all witnessed red. NAMED GAP (honest, like
`aa == ab` was before its witness): the DFS back-edge refusal has no red fixture —
18/18 stay green when the check is deleted; the back-edge-stays-backward property
is asserted as a layout THEOREM test over every fixture, not a refusal path.

## M7 probe (2026-11-06): a real linked EXE exists; the loader's first answer about it is LOUD

Artifact (reproducible): `clang -O1 -c` of `main.c` (`int main(){return (int)f(7);}`) +
`corpus/src/arith_basic.c`, linked by `clang main.obj arith.obj -o m7.exe` (VS BuildTools
14.44 link under the hood). 111,104 bytes; sections: .text 0xeff7 at VMA 0x140001000,
.rdata, .data, **.pdata 0xfc0** (SEH unwind tables, ~300+ functions of CRT), .fptable,
_RDATA, **.reloc** — the two structures M1 never saw in a .obj.
Today's first behavior on it (measured): `rdec discover m7.exe` prints
`section 1 (?) has SizeOfRawData 1750344141 at PointerToRawData 1881174889, past the
end of a 111104-byte file` — the COFF-object reader reads the PE IMAGE_HEADER as a
section table and REFUSES with exact bytes. No silent misparse: the audit standard
holds on images it was never built for. Exit code NOT yet asserted (the measuring
command piped through head — the exit-code lesson cuts both ways; a probe test must
run the binary un-piped).
M7 scope this pins: (1) PE image headers + section VA->file mapping; (2) .pdata/.xdata
SEH unwind (function starts WITHOUT symbols — the whole-binary discovery source);
(3) .reloc base relocations as evidence (the REL32 story re-used at image base
0x140000000); (4) CRT machine code => library signatures (discover must not promise
100% on 61 KB of CRT it will never emit); (5) entry point + IAT (imports are calls
to a boundary that must stay LOUD). The m7.exe corpus (plus a /RELEASE build with a
symbol table) is the fixture set.
GHIDRA BASELINE (M6 gate) is BLOCKED ON THE HOST, not on the code: no ghidra install;
java is 27 (Ghidra 11.x supports 17/21). Headless Ghidra = ~500 MB download + a second
JDK: a heavyweight install that needs the user's go-ahead, not a silent autonomous
step. When approved: run headless analyzeHeadless over corpus/c/all.c objects, emit
its C, and diff against `rdec c` per function as a readability/type-recovery baseline.

## M7 first slice landed: PE32+ image loader (`pe.rs`) + `rdec pe-info` — 26 new tests, floors unmoved

The object loader's refusal on m7.exe is no longer the last word: `crates/arch-x86/src/pe.rs`
reads MZ -> e_lfanew -> PE sig -> COFF header -> optional magic 0x20b (PE32 `0x10b`
refused BY NAME with the reason: 4-byte ImageBase), image base, entry RVA, subsystem,
data directories bounded by NumberOfRvaAndSizes, section table; API `va_to_off`,
`read_at`, `section_named`, `entry_va`. Every refusal names field/got/expected/offset/
file-length in loader.rs style; every address arithmetic is checked (rva+vsize RVA
overflow, rawdata past EOF, `PointerToRawData==0` with nonzero size — the .bss lesson
carried over — overlapping sections, zero alignments, table past EOF). `rdec pe-info`
prints the whole picture and ends with a LOUD `loaded, not yet read:` line naming the
next slices (.pdata/.xdata unwind, .reloc, imports). Exit 0 / exit 2.
m7.exe reads clean: image base 0x140000000, entry RVA 0x1280, 7 sections, EXCEPTION
dir 0x1c000/0xfc0 == .pdata, BASERELOC == .reloc, 16 dirs declared, 5 filled. The
parity leg cross-checks llvm-objdump and documents its quirk: objdump's `Size` column
is min(VirtualSize, SizeOfRawData) (m7 .data prints 0xc00 = raw, vsize 0x1b60 BSS tail).
Sabotage: overflow guard removed -> the fixture STILL refuses (with a DIFFERENT,
wrong diagnosis: section overlap) — the class is covered, the diagnosis degrades;
rawdata-EOF guard removed -> hostile image ACCEPTED, fixture red; magic 0x10b
accepted -> PE32 loads 4-bytes-off, red. Parent re-measured on master+patch: 22+4
tests, workspace 315/0, t2 1700/1700/0/4 + memory 288 + ir 288, ratchet/fmt/clippy/
status exit 0; refusal exit 2 measured UN-PIPED (the first measure through `| head`
printed 0 — the pipe lesson, fifth demonstration, now in the commit).

## Epilogue DCE: the 29 `target computes` refusals are at 0 — return tails materialize on the edges

Investigation first, and it falsified the handed premise (reported, not papered):
(A) RSP/RBP teardown was ALREADY DCE'd (lift lowers `mov rsp,rbp`/`pop` to pure
Copy/Load/Add nodes; the existing sweep deletes them; a fixture pins it). (B) the
frame-slot store is NOT deletable — `Exec::diff` compares the effect trace and
memory, and the T2 memory axis only sees the caller window; invisible is not dead —
kept, pinned. What the 29 actually compute: pure chains materializing the answer.
The fix is `passes::materialize_return_tails`: copy a pure tail's chain onto each
incoming edge (phi arms substituted), leave a phi in the tail, `Return(phi)` — the
originals die in the same DCE fixpoint, so it is a MOVE. Gates: pure, no
Store/Call/Clobber, FAULTING ops only move onto a pred whose only successor is the
tail (a div cannot appear on a switch arm it never runs — sabotage S3 runs p0=0:
machine returns 0, a copied div raises), operands `value_readable` at the edge.
Census prints the work: `epilogue: 72 return tails materialized on 108 edges | 267
kept (163 store/call/flag, 0 faults on an arm it did not run, 0 unreadable, 104
value computed outside the tail)`. The win: 29 -> 0 refusals, readability 100 -> 75
goto / 215 -> 201 labels, exchange `39 printed-as-return | 16 absorbed | 10 flat`.
SECOND WIN, forced by the first: the tail move exposed a latent quadratic — the
deletion cascade ran against block order, one level per sweep; `dce_sweep` now runs
it on a worklist inside one sweep. cascade fixture 4000 blocks 23.0 s -> 12.8 ms;
`clean` at 4k 31.5 s -> 23.8 ms.
COLLISION handled in the open: `jump_table_fixtures.rs` pinned `case {k}: goto l` —
the arms now spell `return` (the win on the fixture's own shape). The pin asserted
the printer's CURRENT SPELLING, the exact anti-pattern this repo's rules name;
relaxed to the machine fact (every arm transfers control; switch/8 cases/default
trap stay pinned). Parent re-measured all floors on master+patch: t2
1700/1700/0/4 + memory 288 + ir 288 byte-identical, structuring 18/18 untouched,
workspace 326/0, ratchet/fmt/clippy/status exit 0.

## M8 packed SIMD first slice: the XMM pair — movdqa/movdqu/movaps/movups family + lane bitwise

XMM registers are a PAIR of W64 cells (lo+hi; GRPS 32->48, CALLER_SAVED covers
both). Modelled: movap{s,d}, movdqa/movdqu/movups/movupd (loads = two W64 loads
lo@addr hi@addr+8, stores symmetric, reg-reg pair copies, the stack-spill shape),
pxor/pand/pandn/por as two independent lane ops, pxor/xorps/xorpd zeroing zeros
BOTH cells; scalar load-merge (`movsd xmm,...`) zeroes the hi cell explicitly,
scalar store (`movsd m64,xmm`) touches NEITHER cell; every other packed op stays a
loud Opaque naming the mnemonic. The refusal census is the next lane's map: 43
refusals, paddq 1 (copy_words.O2), pcmpeqd 1, cdq 10, clobbered-by-call 8, imul 6,
lea-unnamed 6, lea-.rdata 4, div 4, live-in Rsp 2, live-out flags 1. The
movdqa/movdqu-shape classes are GONE (modelled).
ACCEPTANCE DEVIATION, root-caused: `copy_words` O2/Oz stay refused on THIS clang
because they vectorize to movdqu + paddq (read-modify-write), and paddq is out of
this slice's scope — the refusal names the mnemonic verbatim in the census. The
pure-`movups` witness is the new corpus fixture `copy_block` (O1/O2/Oz): +3
emitted (97 emitted / 43 refused, -Werror OK), t2 `1796/1796/0/4`, memory axis
288 -> 384/384/0 and ir leg 384/384/0 - vector stores byte-compared in the caller
window, which is the oracle this slice was built to feed.
CLOBBER COUNT, the invariant the lane could not square: "one statement per ABI
register" broke `calls.rs`'s report == IR check (an IR node defines ONE value; the
pair needs two ops). Resolution is per-cell counting with the pinned 24 -> 40 move
carrying its root cause (9 GPR + 32 XMM cells - the defined output): clobbering the
high lane across a call is the POINT of the pair model, and the register-level ABI
fact (25 caller-saved) is unchanged. The scalar-era `movapd` refusal pin in
`sse_scalar.rs` retired with the reason (it is IR now; positive coverage in
`packed_simd.rs`).
SABOTAGE (all five, parent-run for the last three): store-lane-swap ->
`the 16 bytes must arrive in order (lo lane first)` left/right byte-quoted;
hi-load dropped -> 4 fixtures red; movsd load-merge zeroing removed ->
`low 8 = the load, high 8 = ZERO` red with bytes; scalar store spilling hi at
addr+8 -> FIRST RUN PASSED — the fixture compared `tail == HI` and the spill
wrote exactly HI: a test written to pass, caught by sabotage. Strengthened: the
destination high half is seeded with a 0xA5 sentinel the program cannot carry,
the same sabotage now reddens it with `Some([136,...,255]) != Some([165 x 8])`,
and the lesson is a comment in the fixture. Nine tests, all restored green.

## M7 slice 2 lands: `.pdata` reader, and one v2 entry refuses ITSELF, not the table

`crates/arch-x86/src/pdata.rs` parses the EXCEPTION directory as RUNTIME_FUNCTION +
UNWIND_INFO (v1 code arrays: frame/epilog descriptions, x12 magic), with a hard table
contract (sorted, EndAddress inside .text, 4-byte-aligned UnwindData, size%12) refused
as structural defects. Chaining and unknown UNWIND_INFO *versions* are per-entry
refusals: named, COUNTED (`chaining_refusals()` / `version_refusals()`), code array
never read. The API invariant `functions + chaining + version == entry_count` is
asserted on synthetic fixtures AND on the real linked image, where it is the
product: **m7.exe = 336 entries = 328 functions + 4 chained (0xbc52, 0xbf38, 0xc713,
0xc86c — export-thunk blocks) + 4 version-2 (0xe8e0, 0xe9f0, 0xefb0, 0xf3f0 —
CET-hardened CRT thunks)**. The first cut made v2 a whole-table refusal; the real
image showed what that costs — one CRT thunk would throw away the 332 true starts of
a symbol-free EXE (llvm-nm: `no symbols` is pinned; `.pdata` IS the start source).
`rdec pe-functions m7.exe` exits 0, prints the 8 refusals verbatim, and recites
328 + 4032/12 + 4 + 4. Sabotage (silent v2 drop) reddens the unit census and the
real-image cross-check. The probe artifact is REPRODUCIBLE (recipe in this file;
rebuilt byte-identical at 111,104) and the real-image leg self-skips LOUDLY when it
is absent. Floors: workspace 42 test groups green, fmt/clippy clean.

## M7 slice 2 landed: .pdata reader + `rdec pe-functions` (CI green 5m18s)

pe.rs PeImage + pdata.rs: EXCEPTION directory table, RUNTIME_FUNCTION -> v1 
UNWIND_INFO code arrays (frames/epilogues/x12 magic), full table contract 
(sorted, EndAddress in .text, 4-aligned UnwindData, size%12 -> structural REFUSAL).
KEY LESSON (measured): VERSION and CHAINING refusals are PER ENTRY - one v2 CRT 
thunk forbids the WHOLE TABLE. m7.exe = 336 entries = 328 functions + 4 chained 
(0xbc52/0xbf38/0xc713/0xc86c, export blocks sharing extended info) + 4 v2 
(0xe8e0/0xe9f0/0xefb0/0xf3f0, CET CRT thunks) - census asserted as a partition on 
synthetic AND real images; the real leg cross-checks an independent raw-byte walk 
(same indices are exact) and LOUD-skips when the probe is absent. llvm-nm: the 
linked EXE is symbol-free - .pdata IS the function-start source, pinned. 
Reproducible probe: clang -O1 main.c{int main(){return (int)f(7);}} + 
corpus/src/arith_basic.c, linked via clang/MSVC-link -> 111,104 bytes, rebuild 
verified byte-identical this slice. CI-only clippy escapes (1.99 doc lints) stay 
fix-forward: first red 33s doc_lazy_continuation on a doc line starting with +.

NEXT: discover speaking .pdata (start seeds from the table), .reloc, imports/IAT 
(named boundary + refusal), CRT library signatures; then M6 type recovery.

## M4 slice: the accumulator forms land — every C `/` and `%` the corpus compiles to

`cdq`/`cqo`/`cwd`, `div`/`idiv`, one-operand `mul`/`imul`, and
`imul r64,r/m64,imm32`. These are not exotic instructions; they are what a
compiler emits for integer division and for a wide multiply, so refusing them
refused the function. `crates/arch-x86/tests/acc_forms.rs` (9 fixtures, red
first), `crates/core/tests/interp_arith_reference.rs` (the P-track extension the
suite's own header asked for), `crates/core/tests/mulhi_helpers.rs` (the C leg).

THE MODEL, in the three lines that matter:
- `cdq`/`cqo` is `EDX = EAX >> 31` / `RDX = RAX >> 63`, and it touches NO flag.
  The flag claim is MEASURED, not assumed: the fixture runs `cmp ecx,edx ; cdq ;
  setc al` and reads the `cmp`'s CF across the `cdq` — a lifter that clears flags
  here answers 0 on the `al = 1` legs.
- The 32-bit pair IS a 64-bit value: dividend = `(EDX << 32) | EAX`, divisor =
  the operand read with the mnemonic's signedness, `SDiv`/`SRem`/`UDiv`/`URem` at
  W64, halves written back as W32. `div` and `idiv` are the same assembly; the
  whole difference is the reading, and that is where the model puts it.
- The 32-bit multiply needs no high-half op at all: 32x32 fits 64 bits under
  either reading, so widen both halves (`SExt` for `imul`, `ZExt` for `mul`),
  one `Mul`, low 32 bits -> EAX, bits 32..63 -> EDX. At 64 bits the high half is
  `MulHi`/`MulSmul`, which the interpreter had and the C emitter refused
  ("needs `__int128`") — so the emitter got the portable spelling instead.

THE EMITTER, two changes with teeth:
- `rdec_mulhi64`/`rdec_smulhi64` in the PRELUDE: four 32x32 limb products, every
  operation unsigned, the only wrap the final sum, which IS the answer mod 2^64.
  No `__int128`, because the emitted C has to compile where the project's own C
  compiles and MSVC has no 128-bit integer type. `mulhi_helpers.rs` compiles the
  helpers with the floor gate's flags and compares them against `__int128` over
  300+ vectors (the boundaries a limb decomposition breaks on + a golden-ratio
  sweep), and the P-track table's reference is the SAME decomposition written a
  third time in Rust. Interpreter (`u128`), table (limbs), shipped C (limbs),
  reference (`__int128`): four readings, one answer.
- `SDiv`/`SRem` are now read with the OP's signedness, not the operand TYPE's.
  Registers are modelled as unsigned cells, and the pair dividend lands in one,
  so the old rendering of a signed division on an unsigned cell emitted a
  `(uint64_t)/` — 0x7FFFFFFC where the machine answers 0xFFFFFFFD for -7/2. The
  `cdq`+`idiv` path walks straight into that. `INT_MIN / -1` keeps its guard (C
  UB, machine `#DE`, model wraps to `INT_MIN`, pinned by name).

GAPS, named not hidden:
- `div`/`idiv r/m64` divides a 128-bit `rdx:rax`. Refused, with the missing type
  in the reason: a 64-bit rendering would answer every real `__int128`-style
  division wrong while the 32-bit cases kept passing, which is the one failure
  mode a gate must not allow. Zero corpus sites.
- Quotient overflow. Divide by zero: the interpreter FAULTS (`Fault::DivZero`,
  pinned at every width), the emitted C answers 0 (it must return something to
  compare). `INT_MIN / -1`: the model's guard is `xs == i64::MIN && ys == -1`, a
  W64-shaped test, so the model FAULTS at W64 (`Fault::IntMinDivMinusOne`) and
  answers the WRAPPED quotient at W8/W16/W32 — where the machine traps at every
  width. Both directions are pinned by name
  (`int_min_over_minus_one_faults_at_64_and_wraps_below_it`), which is what makes
  "quotient overflow is not modelled at the narrow widths" a gap with a red test
  to turn green instead of an asymmetry nobody noticed. The divergences are
  written down at the lift site and at the emitter.
- CF/OF after a 32-bit `mul`/`imul` are defined by the machine and NOT modelled:
  the model's `FlagsOp::Mul` rule compares halves of the same width and this
  pair's halves live in a W64 cell. Flags are declared undefined, so a function
  that reads a flag after such a multiply hits the existing loud refusal at the
  READ site — the machine's own answer for SF/ZF/AF/PF, and for CF/OF the honest
  one: a counted refusal beats a bit nobody computed.

THE IMMEDIATE32TO64 GAP was the invisible one: `src` listed every immediate kind
except the one iced reports for `imul r64,r/m64,imm32`, so clang's entire
magic-constant-division idiom was refused as "an unmodelled operand" while the
arithmetic beside it lifted fine. One variant, and the imm32 sign-extension the
fixture pins (0x92492493 must multiply as -1840700269, measured) is the whole
content of the form.

MEASURED (parent-run, un-piped, exit codes as verdicts):
- `c --corpus`: 140 lifted, **119 emitted, 21 refused** (from 97/43), 0 invalid,
  clang -Werror OK. The arithmetic classes are GONE: 10 `cdq` + 6 `imul` + 4
  `div` = 20 of the 42 refusal records. What is left is 8 call-clobber, 6+4
  `lea`-of-an-unnameable-address, 2 live-in `Rsp`, 1 `shufps`.
- `t2 --corpus`: **compared 2260 / match 2260 / mismatch 0**, memory **408/408**,
  IR leg **408/408** (from 1796 / 384). The +464 are the four division/multiply
  families at four opt levels — `divmod_signed` (signed `/` and `%`),
  `div_unsigned`, `div_pow2`, `mul_wide` (the 64-bit `imul` high half, so the
  PRELUDE helper is compared against the machine, not just against `__int128`).
- THE 20 NOT-COMPARED RUNS ARE THE MACHINE TRAPPING, not the model drifting:
  every one is `exit 0xc0000094` = STATUS_INTEGER_DIVIDE_BY_ZERO, on the
  **machine-code side**, on `div_unsigned`/`divmod_signed` vectors 1 and 22 — the
  zero-divisor vectors. The harness names the faulting side and refuses to count
  them as matches, which is the accounting that makes "2260/2260" a real claim:
  the corpus's division cases where the machine survives are all compared, and
  the ones where it raises `#DE` are named. Same three-way story the fixtures
  pin: machine `#DE`, interpreter `Fault::DivZero`, emitted C answers 0. The
  infinite-loop fixtures that used to fill this bucket are not in it.
- `t2 --cpp`: **856 / 856 / 0 mismatch / 0 not-compared** over 18 RAX + 16
  window + 5 XMM0 functions. This gate matters more for this slice than the
  numbers suggest: the C++ corpus is where signed `/` and `%` reach the emitter
  from C++ signed types, so the `SDiv`/`SRem` signedness change is green on 856
  machine runs, not only on the C corpus.
- `c --corpus` exits 1 by design (refusals > 0), and the refusal census is the
  map: `lea`-of-an-unnameable-address is the largest class now, and it is a
  discovery artifact, not an instruction set.

SABOTAGE, all parent-run, all restored: `cdq` count 31 -> 32 reddens
`cdq_idiv32…` (the pair's high half is then the value, not the sign); pair `SDiv`
-> `UDiv` reddens it with the machine's number quoted (`left: 4294967292 right:
4294967293`); the 32-bit multiply widened unsigned reddens
`one_operand_mul_imul32…` on the high half (`1` vs `0xFFFFFFFF`); dropping the
carry limb from the PRELUDE helper reddens the `__int128` cross-check; dropping
the signed correction terms from the TABLE's reference reddens
`multiply_high_matches_a_limb_decomposition_at_64_bits`
(`model 0xffffffffffffffff, machine 0x0`) — which is the reference being wrong,
the way a reference has to be provable-wrong for the triangle to mean anything.

TWO TEST-WRITING DEFECTS the gates caught in the tests, not the code: the
divisibility sweep skipped a RAW zero divisor where the op sees the TRUNCATED one
(at W8 the edge value 0x1_0000_0000 IS a zero divisor, and the interpreter
faults, exactly as the machine #DEs), and the emitter's refusal claim was asserted
against text containing `/* refusal` when `emit_c` returns refusals as `Err` — so
that leg had never run. Both tests were rewritten to read the real contract, and
the rewritten legs are green.

---

### M3 slice 5 - the corpus grows its first tail-call and noreturn witnesses (2026-11-08)

**The gap was the evidence, not the feature.** The tail-call and noreturn model
shipped pinned by hand-built bytes in `crates/arch-x86/tests/tail_calls.rs`,
whose own header records that the corpus had no site for them. Every floor since
printed `tail calls: 0 modelled | noreturn calls: 0`, and the census doc comment
stated it in the present tense as if that were a stable property. A feature no
corpus reaches is a feature no gate measures.

**Gate first.** `cmd_c` now carries three floors, all reading `tail_counts` —
the same function that produced the census line printed above them, so a verdict
and a print cannot disagree:
1. *exercise*: `tail + tail_unnamed == 0` fails, naming the counts and the C
   shape that would fix it. Run against the pre-slice corpus it printed the FAIL
   verbatim — that red had to exist before any fixture was written.
2. *noreturn exercise*: `noreturn calls == 0` fails.
3. *anti-invention*: every symbol the noreturn census names must be a name
   `lift::noreturn_symbol` declares. Floor 3 exists because of a sabotage, see
   below.

**The witnesses** (`crates/verify/src/corpus.rs`; manifest +12 objects, and no
existing object hash moved, which is the deterministic-corpus design doing its
job — a corpus change arrives as a manifest diff, not a silent behaviour
change). Shapes measured per -O level with `llvm-objdump -d -r` as ground truth,
quoted in the source comment:

| snippet | measured | bucket |
|---|---|---|
| `tailcall_local` | O0 `callq helper`; O1/O2/Oz `jmp helper` + `IMAGE_REL_AMD64_REL32 helper`, a symbol this object DEFINES | the modelled tail call: `return helper(a, b);` |
| `tailcall_external` | O0 `callq ext`; O1/O2/Oz `jmp ext` -> `ext`, NOT defined here | the refusal bucket: the object gives no callee, and inventing a prototype is a silent miscompile |
| `noreturn_call` | all four levels `callq abort` -> `abort` (`U` in `llvm-nm`) | termination: the block ends at the call |

`noinline` on `helper` is load-bearing: clang inlines a visible helper at O1+
and the `jmp` never exists. The abort leg is `x > 100`, which the differential's
vectors never drive, so the function stays value-comparable *while* its CFG needs
the noreturn rule — a witness that pays for itself in two gates.

**The fixture found a real hole.** `noreturn calls` stayed **0** while the CFG
rule visibly fired (the census's `bytes past the end of bb2 … not lifted` lines
appeared at exactly the post-`call` addresses). Cause: the noreturn arm requires
a signature entry, and an import has none — `U abort` carries no section, no
size, no address, so it never reaches the symbol table the declaration loop
reads. M3's noreturn line names `abort exit _exit __stack_chk_fail __fastfail
ExitProcess`, and in a linked binary **every one of those is an import**, so the
feature was unreachable exactly where the milestone asked for it. The
declaration is now made where declarations are made — the CLI's front-end table —
from the NAME the relocation carries, for imports only (a defined symbol's entry
wins), and only for names the table declares. `typed: false` + default ABI:
declaring termination is a statement about CONTROL, not a return type the object
never gave, so the call reads and clobbers the six argument registers and defines
nothing.

**What the emitted C says** (`corpus/c/all.c`, the point of the whole slice):

```c
extern void o100_abort(int64_t, int64_t, int64_t, int64_t) __attribute__((noreturn));
...
l1: /* at 0xf */
    o100_abort((int64_t)x0, (int64_t)x1, (int64_t)x2, (int64_t)x3);
    __builtin_unreachable(); /* rdec: the call never returns */
```

The arguments the machine code put in the registers, named; the block ends; the
`return` survives on the leg that takes it. No fall-through into code that is not
this function's.

**MEASURED** (parent-run, un-piped, exit code = verdict):
- `c --corpus`: 156 lifted / **125 emitted / 27 refused** / 0 invalid after
  passes, clang -Werror OK. Refusals 21 -> 27 because the corpus grew 140 -> 156
  functions, 4 of which are new refusal witnesses; the `noreturn_call` functions
  EMIT. Classes: 8 call-clobber, 6 + 6 `lea` of an unnameable address (2 of them
  `.rdata`), **3 `tail call to 'ext'`**, 2 live-in `Rsp`, 1 `call to 'ext'`, 1
  `shufps`.
- census: **`tail calls: 3 modelled | 3 with no callable callee | noreturn
  calls: 4 | 0xf -> abort | 0x12 -> abort x3`**.
- `objdump-check`: 156 symbols, 156 exact boundary matches, 0 mismatches.
- `discover`: 148 objects, 156 functions, 0 missing, 0 extra starts, coverage
  **6996/6996 = 100.00%**, and a new NAMED gap class — `call to a symbol this
  object does not define (an import: the name is known, the code is not) 8` = 4
  x `ext` + 4 x `abort`, counted rather than absorbed.
- structuring census gains named goto reasons: `4 noreturn call`, `3 tail call`.

**SABOTAGE, and the hole it opened.**
- `noreturn: true` -> `false` in the new declaration: census reads 0, floor 2
  goes RED with its message verbatim. The corpus witness, not a comment, is what
  satisfies the gate.
- Declare **every** import noreturn (delete the name-table condition) — the
  invention direction, the dangerous one: census `noreturn calls: 5`
  (`0xf -> ext`), emitted 130 / refused 26, coverage unchanged 100.00%,
  discover exit 0. **Every gate above floor 3 passed.** One corpus function
  changed from `Exit` to `__builtin_unreachable()`, and nothing measured it.
  Floor 3 was written the same hour and reddens it: `FAIL: anti-invention floor
  - the noreturn census names ["ext"], and the front end does not declare those
  noreturn.`

**Ledger consequence.** The M3 row flips from `NOT DONE — zero handling` — the
row whose stated evidence was a `grep -ric` that returned 0, where the live
count is 53 — to LANDED with corpus witnesses and a named remaining gap: the C++
corpus is still all-zero on this line, because its functions are leaf arithmetic
and the demangler's noreturn knowledge has no corpus site to land on. The M4 row
is re-read at 156/125/27.
## M7 slice 3/4 lands: `.reloc` base relocations, and `rdec pe-discover` becomes a gate

`crates/arch-x86/src/reloc.rs` reads IMAGE_DIRECTORY_ENTRY_BASERELOC: PageRVA/SizeOfBlock blocks,
16-bit entries (type<<12 | offset), EVERY winnt.h AMD64 nibble 0..15 NAMED (8 is the hole winnt.h
itself leaves), ABSOLUTE padding COUNTED, never dropped. Totality is a PARTITION -
padding + absolute + relative + unreadable-site + refused-kind == entries - asserted on a
three-block hermetic image and on the real image, where it cross-checks a RAW-BYTE WALK written in
python that shares no code with the reader (6 blocks, 782 entries = 3 ABSOLUTE + 779 DIR64,
per-block counts, sum(SizeOfBlock) 0x64c == dir size). Refusals name field/got/expected/offset:
blocks past EOF, SizeOfBlock < 8, SizeOfBlock past the table end, (SizeOfBlock-8) not a multiple of
2, blocks not ascending by PageRVA (the pair is named), PageRVA not 64KiB-aligned, an entry whose
end passes its own block. Targets: DIR64 reads the 8 bytes at the site (value - ImageBase must be a
readable 32-bit RVA, else `unreadable_site`), REL32 is site+4+signed-addend; the half-address family
(HIGH/LOW/HIGHLOW/HIGHADJ/RELOC64/HIGHADJ_1/HIGHADJ_2) and the foreign-arch kinds are REFUSED BY
NAME - this reader pairs no halves, so a name is the honest floor and a value would be a guess.
MEASURED, and it corrects the brief three times: (1) `llvm-objdump -R` prints NOTHING for a PE
image (LLVM reads it as COFFObjectFile) - the oracle here is `dumpbin /RELOCATIONS` + the raw walk;
(2) the brief's `size % 8` directory rule REFUSES m7.exe (0x64c % 8 = 4): the stride is the 2-byte
ENTRY, the 8-byte header bound belongs to SizeOfBlock PER BLOCK (S3 below reddens if %8 returns);
(3) PageRVA is the page a block PATCHES, not the page it sits in (m7 blocks ascend 0x15000 ->
0x1c000 inside one 0x64c-byte table; fixture block 2 sits at table rva 0x301c patching page 0x3000).
And the offset-outside-page rule is a HOLE in the brief: a 16-bit offset cannot exceed 0xFFF, so the
reachable defect is an offset in the page whose END passes its own block - pinned that way.
m7.exe measured: ZERO relocation sites inside .text - MSVC keeps .text position-independent, every
site is in .rdata/.data. The 47 DIR64 relocations that NAME .text code address 38 distinct
addresses; 31 are .pdata starts and **7 are not: 0x15a8, 0x15ac, 0x1f20, 0x1fa8, 0x3e40, 0x8c14,
0xa9d8**. The other 732 DIR64 are data-to-data pointers (vtables): census, NOT gaps - counting them
as gaps would bury the boundaries that are real.
pe-discover now seeds from three sources and says which is which: `.pdata` parsed starts (census
prints `seeded 2 of 4 entries; a refused entry is NEVER a seed` - counting ADDRESSES PUSHED, so
seeding a refusal moves the number while the parsed count sits still), the entry point (inside
.text only; outside it is named and not seeded), and `.reloc` code targets, printed with their own
provenance line so the `.pdata` census stays the `.pdata` census. A relocation whose SITE is in
.text and whose target is not is the IAT/indirect-call shape: named by site + target + target
section, counted as a gap, `callee NOT invented` (the import directory is not read by this slice).
ABSOLUTE padding has no site (width 0) and is excluded from the site census - counting it invented
a boundary-shaped event at a byte the loader ignores. m7.exe: 418 functions, 56266/61431 = 91.59%
coverage, 213 gaps (53 indirect-branch + 160 indirect-call, ZERO of them reloc boundaries - MSVC
puts every site in .rdata/.data), overlap_free, exit 0 (413 functions / 56191 bytes before the
`.reloc` leg; +7 seeds). The VERDICT line now prints `213 gaps (213 descent + 0 reloc boundary)`: it
shipped counting ONLY the reloc boundaries, so a census of 213 printed a verdict of 0 gaps -
the quiet-zero class this repo has been burned by, caught by reading the run's own output.
THE GATE (crates/cli/tests/pe_discover.rs, 7 tests, hand-built PE32+ in pe_fixtures style): the
non-vacuity floor is asserted TWICE - the test demands functions > 0, and a 0-function run must
exit NONZERO with its coverage numbers declared void (the fixture strips .pdata, .reloc and parks
the entry point in .data). LEAK LESSON, measured, not assumed: seeding a chained refusal did NOT
move the function count - the linear sweep decodes with the SAME engine a seed uses, so a decodable
refused address is a function with or without the leak. The count is the wrong witness; the seed
census is the right one, and that is what is pinned (S4 turns it `seeded 3 of 4`, red). A refused
`.reloc` table costs only itself (functions 3, VERDICT OK, the refusal names SizeOfBlock 6, block 0).
Sabotages, all applied one site at a time, quoted, restored green: S1 census drops padding ->
`the_entry_census_is_a_partition_and_counts_padding` red on `balanced(): false`; S2 SizeOfBlock
bound loosened by 6 -> subtract-with-overflow at reloc.rs:422, the refusal test red; S3 directory
stride back to %8 -> the real-image leg red, m7.exe refused by a rule a healthy image cannot pass;
S4 chained refusal pushed as a seed -> census red; S5 vacuity gate disabled -> `a 0-function
discovery must exit nonzero: exit=0` red.
Floors re-measured in-lane (un-piped): fmt 0, clippy -D warnings 0, workspace **45 suites / 390
tests / 0 failed** (floor 43/364 - the delta is exactly the two new files), ratchet 0 with all five
baselines untouched (core 1, arch-x86 7, demangle 1, verify 4, cli 14 - this slice adds NO panic
site), selftest 0, objdump-check 0 (140 symbols / 140 exact boundary matches / 0 refusals / 0
mismatches), discover 0 (136 objects, 140 functions, coverage 6764/6764 = 100.00%), lift 0 (140 /
140 valid IR / 103 clean), c 0 (98 emitted, 42 refused, 0 invalid, clang -Werror OK), t2 compared
1820 / match 1820 / mismatch 0 / fault 4, memory 408/408/0, IR leg 408/408/0 - byte-identical to
the floor. COFF objects still go through the object loader and the loader's PE refusal is untouched
test-wise.
NOT PROVEN: imports/IAT (a boundary names an RVA, never a callee - next slice, and this census is
its map); PE32 (magic 0x10b) has no `.reloc` leg because pe.rs refuses it BY NAME; the 16-bit
half-address types are named, not paired; the 7 `.reloc`-only starts are addresses the image names,
not verified function bodies (0x15ac is 4 bytes behind 0x15a8: at least one is not a start - the
sweep decides, and its verdict is coverage, not a proof); the m7.exe legs skip LOUDLY when the probe
is absent from the reaped temp dir.



## Section 10: the MSVC leg of the corpus matrix (cl 19.44.35228, 128 objects, CI job `msvc`)

Section 3 promised the matrix `{clang, gcc, MSVC}`; the corpus path was clang-only.
`rdec corpus --msvc` (`crates/cli/src/corpus.rs`) compiles the SAME `corpus/src/*.c`
with cl into COFF at `corpus/msvc/obj/<snippet>.<opt>.obj`, writes the same manifest
shape, and every gate reaches it through ONE new flag, `--corpus-dir DIR`:
`m1::corpus_objects` is the manifest reader, and objdump-check / discover / lift /
`c` each gained the flag in the file that already owns their arg parsing. A gate
forked per toolchain is where a leg quietly stops checking anything, so a gate over
MSVC objects IS the gate over clang objects with a different denominator. The clang
path is untouched: nothing here is called by `rdec_verify::corpus::generate`, and
`git diff --exit-code corpus/manifest.tsv` is clean after a clang rebuild (measured).

FLAGS, recorded per leg in the manifest header (`# flags[leg]:`): `/nologo /c /Gw-
/GS- /Od`, `… /O1 …`, `… /O2 /GS …`, `… /O1 /Os …`. `/GS` (the security cookie) is ON
for exactly one leg — pinned by `every_leg_has_flags_and_the_cookie_leg_is_named`.
`/Oxs` and `/Ox` are NOT legs, and that is a measurement: their `.text` is
byte-identical to `/O2` on 32/32 snippets (only `.drectve`/debug records differ), so
a leg for them gates nothing. `/Os` IS a leg: its `.text` differs from `/O1` on 30/32
and from `/O2` on 29/32. `/Gw` is spelled out because it only does anything under
`/GL`, and `/GL` replaces per-function COFF code with an MSVC IL bitstream — there
would be no `.text` to decompile at all. `/Gs` stays default: `__chkstk` needs a
frame over 4096 bytes and no snippet has one.

THE `/GS` RECORD, HONEST: on this corpus the cookie is a no-op — 0 objects reference
`__security_cookie`, 0 reference `__chkstk`, and `/O2 /GS` `.text` is byte-identical
to `/O2 /GS-` `.text`. The lifter therefore refuses NOTHING by those names, because
no snippet has the local-buffer shape that makes MSVC install a cookie. The roadmap
WANTS those shapes: the corpus needs a snippet with a stack buffer, and
`corpus/src/*.c` is not this lane's file. Recorded as the next step.

EXCLUDED BY NAME, not silently (manifest `# excluded:` lines, and `rdec corpus
--msvc` refuses to report OK unless it built exactly 32×4): `call_helper.c`
(clang-only `__attribute__((noinline))`; cl: C2143) and `mul_wide.c` (clang-only
`__int128`; cl: C4235). Result: **128 objects, 101,212 bytes, 130 code symbols**.

WHAT cl EMBEDS, measured (two builds of the same 32 sources into the same directory,
2 s apart): the COFF `TimeDateStamp` at object **offset 4** moved in 32/32, and the
8-byte payload at **`.chks64` + 8** moved with it, because that checksum is taken
over the object *including the stamp*; `.chks64` bytes 0..8 (file identity) did not
move. Zeroing BOTH is what makes the leg reproducible: 32/32 objects differ before
normalization, 0/32 after. `.debug$S` (`S_OBJNAME`) and the FILE symbol's aux record
carry the source and `.obj` PATHS, so the claim is "cl twice into ONE directory" —
never two directories, and never cl-vs-clang bytes. Offsets pinned twice:
`msvc_stamp_offsets_pin_the_real_fields` (unit, both fields + the non-AMD64 guard)
and `the_normalization_rule_burns_off_a_real_clock` (integration: put a fake clock
and a fake checksum into a real cl object, run the rule, require the on-disk bytes).

MEASURED MSVC SHAPES the front end has to speak, asserted over generated objects:
C symbols carry a leading underscore (`_f` in 128/128), every function is its own
COMDAT (`.text$in$_f`; the corpus has exactly 128 code sections for 128 objects),
and the section-definition symbol shares offset 0 with `.text` — the `is_code`
dot-name skip is what keeps it out of the function list. `.bss` is 4 bytes with
`PointerToRawData == 0`, which means **NO DATA IN FILE**, not data at offset 0: the
assertion is on the loader's section bytes, never on the header field.

GATES: `.github/workflows/ci.yml` job `msvc` (parallel to `gates`, so the clang
job's ~21 min is untouched) runs `scripts/msvc-corpus.ps1` (cl, located by vswhere,
banner printed; corpus twice; manifest sha256 must match) then
`scripts/msvc-gates.ps1 -Limit 40`; nightly.yml runs `-Full`. `cl` is NOT on PATH on
windows-latest, and `call vcvars64.bat` is a cmd builtin a Rust child process cannot
call, so `scripts/msvc-env.ps1` writes a wrapper batch whose LAST LINE is `cl %*` —
with a bare `%*`, cmd.exe eats cl's own switches and all 128 spawns die with
`cl exit 1` (measured). Measured legs: objdump-check 40 symbols / 40 exact boundary
matches / 0 mismatches; discover 130 functions / 0 missing / 0 extra starts; lift
130/130 valid IR, 101 with no gaps and no exits; the emitted C TU compiles with
`clang -Werror`. Every stage is vacuity-guarded (0 objects, 0 code symbols, 0
functions, a refusal naming nothing, a pass that breaks SSA ⇒ exit 1), and
`an_empty_msvc_corpus_fails_the_gates_with_their_vacuity_text` runs the shared gates
over an empty manifest and requires them to FAIL with their vacuity text.

REFUSAL CENSUS BY MNEMONIC over MSVC objects — the roadmap's "lifter priority comes
from frequency", measured on a new toolchain for the first time (130 functions: 99
emitted, 31 refused, 0 invalid):

```
12  cdq                                        4  clobbered by call
 4  div                                         3  cwde
 3  flag setting with an unmodelled first operand
 3  inc/dec on an unmodelled destination
 2  lea of an address we cannot name
```

`cdq` is 39% of MSVC's refusals and 2.4% of clang's (clang spells the same `x/y`
source with `cqto`, refused once): MSVC's `idiv` lowering, not a new rdec gap.
KEY LESSON (measured): a census keyed at the first `:` CANNOT see a mnemonic —
`opaque: unmodelled instruction: cdq (memory operand)` is one bucket for every
unmodelled instruction, which is right for the clang gate's class list and useless
for "which instruction next". The MSVC census cuts one level deeper (`census_key`),
keeps the class shape for reasons that name no instruction, and counts a keyless
refusal as a FAILURE — an empty bucket cannot hide a dropped refusal.

MEASURED, NOT GATED (proposal, with the numbers): MSVC lowers `switch` into a table
of `mov eax,k; jmp` thunks INSIDE `.text$mn`, reached through an indirect branch
(56 bytes at 0x2c in `switch_dense`; the reachability gap reads `indirect branch`),
and pads with 2-byte `66 90` fills the padding classifier does not name. Coverage
over the leg is 6932/7048 = **98.35%** (116 uncovered bytes in 8 runs), under the M1
99.5% floor. `msvc-gates` runs the same `discover` code, parses `missing`/`extra`
(0/0 — both gated) and prints the coverage with this reason. The floor is comparable
on this leg after (a) `padding_run` names `66 90`/`90` fills (arch-x86) and (b) M2's
indirect-branch/jump-table work claims the thunk table.

WHAT THIS DOES NOT PROVE: no linked MSVC image (`/c` only, no CRT), so there is no
`t2` on this leg — the value-verified decompilation of cl code is UNPROVEN; no
`/GS` cookie or `__chkstk` site exists in the corpus, so the cookie path is
unmeasured; 32 snippets is 32 snippets, and a mnemonic absent from them is absent
from the census; no gcc leg; the MSVC manifest is generated in CI and not committed,
so cross-runner byte identity is NOT claimed (cl versions differ across images) —
same-toolchain determinism is.

NEXT: a stack-buffer snippet (cookie + `__chkstk` shapes, then the census speaks
them); the gcc leg of section 3; the two discover fixes above; then `cdq`/`cwde`
sign-extension modelling, which the census just ranked first on MSVC.


### 11. M1 slice 6: the padding claim becomes a function, and it learns the fill MSVC pads with (2026-11-08)

The MSVC leg's coverage gap was partly a bug in this repo's own classifier.
`discover --corpus-dir corpus/msvc` read 98.35% where the clang corpus reads 100.00%
on the same sources, and the lane recorded the reason as "MSVC lowers `switch` into
thunks plus 2-byte `66 90` fills the padding classifier does not name". The thunk half
is a real M2 hole. The `66 90` half was a gap in one predicate, in this file, named by
a mnemonic prefix — and the honest share it owns is MEASURED below: 2 bytes of 116, not
the gap. The claim "MSVC's coverage was a discovery bug" is what this slice set out to
test, and it came back mostly false.

WHAT `66 90` IS. The canonical 2-byte NOP: `xchg ax, ax` — exchanging AX with
itself, architecturally a no-op, and the fill MSVC emits for 2-byte alignment
inside `.text$mn`. Our decoder prints it as `xchg ax,ax` (measured: the sabotage
run below printed the decoder's own text as its witness). The padding classifier
asked the decoder — good design, and the reason `0f 1f 84 00 ...` and
`66 2e 0f 1f 84 00 ...` were already named — but then asked it the wrong question:
`text.starts_with("nop")`. LLVM aligns with `nopw`/`0f 1f` (prefix matches); MSVC
pads with `xchg` (prefix does not match). One toolchain's alignment is code and the
other's is padding, on a mnemonic prefix.

THE CHANGE. The inline predicate is now `m1::is_padding_run` + `is_nop_text`: the
architectural no-op family, spelled as the decoder spells it, space-insensitive
(the Intel printer's spacing is not a semantic property of the byte). Same three
doors as before — all-`0xCC` at any length, a decoder-verified all-no-op run, and a
run of 4+ `0x00` (three zero bytes are `add [rax],al`, a real instruction stream,
so they stay charged as uncovered) — plus the `xchg ax,ax` spelling, and an empty
run is padding nothing.

It is a function with tests, not an inline condition, because widening the padding
claim is the cheapest way to make coverage go up on paper: every byte reclassified
as padding leaves the denominator. So the narrow edges are pinned as assertions
(`crates/cli/src/m1.rs`, `mod padding_run_tests`, at the bottom of the file for the
panic-site ratchet):

- `the_two_byte_fill_msvc_pads_with_is_padding` — `66 90 66 90` is padding, AND the
  decoder's spelling is asserted (`xchgax,ax` after space-stripping), so the
  predicate is anchored to a measured fact about the decoder, not to a hope.
- `one_byte_and_multibyte_llvm_nops_are_padding` — `90 90 90`, the 8-byte
  `0f 1f 84 00 …`, the 10-byte `66 2e 0f 1f 84 00 …`.
- `int3_pads_at_any_length_and_zeroes_need_four_bytes` — one `0xCC` pads, sixteen
  pad, `00 00 00` does NOT (the assertion prints the decoder's text as the witness),
  four `00` do, an empty run does not.
- `a_run_that_moves_data_or_jumps_indirectly_is_code` — the MSVC switch thunk's
  load (`8b 04 25 00 20 40 00` = `mov eax, [0x402000]`), the indirect branch that
  reaches it (`ff 25 00 00 00 00`), and a `push rbp; mov rbp, rsp` prologue all
  stay UNCOVERED, where M1's 99.5% floor can see them.
- `a_trap_inside_a_nop_run_is_not_padding` — `90 90 CC 90` is not padding: `0xCC`
  is a trap, not a no-op, and one trap inside a nop run keeps the whole run
  uncovered. The claim is one-directional.

PROVEN CAPABLE OF FAILING. Sabotage: the nop family neutered
(`false && t.starts_with("nop") || …"xchgzz"`) → `cargo test -p rdec-cli
padding_run_tests` exit 101, 3 of 5 tests red, and the failure message printed
`66 90 66 90 is two canonical NOPs; decoder text was ["xchg ax,ax", "xchg ax,ax"]`
— the sabotage is what established the decoder's spelling as a measured fact.
Restored: 5 passed, exit 0.

MEASURED, clang side unchanged (this slice must not buy clang coverage, and did
not): `rdec discover --corpus` exit 0 — 148 objects, 156 symbol-table functions, 0
missing, 0 extra starts, code bytes 7066, padding 70, **coverage 6996/6996 =
100.00%**, 0 uncovered non-padding bytes in 0 runs. `cargo clippy -p rdec-cli
--all-targets -- -D warnings` 0. `bash scripts/unwrap_ratchet.sh` 0 — crates/cli 14
panic sites, exactly baseline: the new code adds no panic site, and the test module
is at the bottom of the file so the ratchet still sees everything above it.

MEASURED RE-READ on the merged tree (this slice's binary, 2026-11-08, cl
19.44.35228, 132 objects). Every number below is printed by the gate named above it.

- `discover --corpus-dir corpus/msvc` — 132 objects, 134 symbol-table functions, 0
  missing, 0 extra starts; code bytes 7098, padding 5, **coverage 6979/7093 = 98.39%**
  (the lane measured 6932/7048 = 98.35%), uncovered 114 bytes in 7 runs (the lane: 116
  bytes in 8 runs). So the padding rule owns **2 bytes of 116 and one gap run of eight**.
  The 114 that remain are the `switch` thunks reached through an indirect branch, and the
  M1 99.5% floor that refuses to adopt this leg's number is unchanged: the leg is
  measured, not gated.
- `objdump-check --corpus-dir corpus/msvc` — 132 symbols, **132 exact boundary
  matches**, 0 extent-boundary refusals, 0 mismatches.
- `lift --corpus-dir corpus/msvc` — 134 lifted, **134 pass `verify_ssa`**, 118 with no
  gaps and no exits.
- `c --corpus-dir corpus/msvc` — 134 lifted, 115 emitted, **19 refused**, 0 invalid,
  clang `-Werror` OK. Census, from `msvc-gates`, one key per line: 4 `clobbered by call`
  | 3 `cwde` | 3 `flag setting with an unmodelled first operand` | 3 `inc/dec on an
  unmodelled destination` | 3 `tail call to 'ext'` | 2 `lea of an address we cannot
  name` | 1 `call to 'ext'`. Against the lane's 31 refusals: **`cdq` 12 → 0 and `div` 4 →
  0**. Both of the leg's biggest classes were fixed by the M4 accumulator slice, which
  never targeted MSVC — `cdq` was 39% of the leg's refusals because MSVC signs the `idiv`
  dividend through the accumulator, and that is exactly the pair M4 lifted. `cwde` 3 was
  the surviving sibling, and it landed the same day (section 12): the leg re-measures at
  **118 emitted / 16 refused**, `cwde` 3 → 0, 6 census keys — and the fixture written for
  those three sites is what found the folded-sign-extension defect described there. The 3 `tail call to 'ext'` + 1 `call to 'ext'` are new
  here because the M3 witnesses arrived after the lane branched: the tail-call witness
  compiles under cl (it uses no attribute), so the second leg inherited M3's coverage
  for free.
- `msvc-gates --limit 40` — exit 0, "132 objects, 134 code symbols, census printed,
  nothing vacuous".
- **The leg fails an exercise floor, legitimately.** `c --corpus-dir` exits 1 on "the
  corpus contains 0 noreturn calls, so nothing in the corpus exercises the noreturn CFG
  rule" — because `noreturn_call.c` is one of the four cl-excluded snippets. This is a
  gap in the leg, not a floor to bypass: MSVC spells that declaration
  `__declspec(noreturn)`, clang on the windows-msvc target accepts the same spelling, so
  the leg gets its own witness in the next slice. Exempting the second leg from a floor
  is the laundering this repo's discipline exists to prevent.
- Full local suite on the merged tree: `cargo test` **48 suites / 426 tests / 0
  failed**; `fmt --all --check`, `clippy --all-targets -- -D warnings`, the panic-site
  ratchet (crates/cli 14 = baseline), `selftest`, `objdump-check --corpus`,
  `discover --corpus` (clang 6996/6996 = 100.00%), `lift --corpus`, `flags`,
  `t2 --corpus`, `t2 --cpp`, `pe-discover` on the linked image all exit 0.

One print-shape finding, recorded not fixed: `c`'s `refusal classes:` line clips at the
end of the line, and this leg's 7th key (`call to 'ext'`) was cut in that output. The
counts are printed (19 records in 19 functions, 7 keys) and `msvc-gates` prints one key
per line, so nothing is lost from the census — but a report that truncates its own list
is the shape a real drop would hide in, and the clip goes with the next `m4` census
touch.

WHAT THIS DOES NOT FIX, named so the next slice inherits it honest: the indirect
branch is still unclaimed — an MSVC `switch` reaches its `mov eax, k; jmp` thunks
through a memory-indirect jump, so those runs stay uncovered and stay in the gap
census (that is M2's indirect-branch jump-table work, the same owner the clang
corpus's 2 recovered tables came from). The padding rule change makes the *denominator*
honest for a second toolchain; it does not claim a single new instruction. And a
fill is named for the encodings our decoder prints: a toolchain that pads with a
form printing anything other than `nop*`/`xchg ax,ax` will show up as uncovered
bytes, which is the loud direction, not the quiet one.


## M6 type recovery: landed on master, `types` in CI (lane m6e, 2026-11-08)

MERGE. `git merge --no-commit --no-ff d295999` onto `m6-type-recovery` (bd3f68b) — auto-clean,
zero conflicts: the two sides touch disjoint files (master: lift.rs, emit_c.rs, interp.rs, m1.rs,
the SSE tests, main.rs's `pe-discover` hunk; M6: tyinfer.rs, m6.rs, truth.tsv, tyinfer_rules.rs,
main.rs's `types` hunks, scripts/truth_extract.py). Chosen over a rebase because the merge is
conflict-free (main.rs's hunks are 50+ lines apart) and a rebase would rewrite the two landed SHAs
this ledger cites (8447945, bd3f68b). The master-owned files (lift.rs, emit_c.rs, interp.rs, m1.rs)
AND width.rs, passes.rs, m4.rs, t2.rs are byte-identical to d295999 in the merged tree
(`git diff --quiet d295999 -- <file>` per file); no M6 semantic hunk had to be adapted, because
type recovery is a read-only analysis — `tyinfer::recover` is called only from `crates/cli/src/m6.rs`
and its own test file. That is the structural half of the emitted-C byte-stability claim; the
measured half is the c/t2 tuple below, which is identical to master's floor.

THREE CHECKS THE BRANCH SHIPPED RED — all in M6's OWN files, all found by running the CI steps
and nothing else:
  - `cargo fmt --check` red at tyinfer.rs:718 (one `if` the slice-3 commit never formatted).
  - `cargo clippy --all-targets -- -D warnings` red: `crates/core/tests/tyinfer_rules.rs:327`
    bound an unused `q` in `let (p, q) = (c[0], c[1]);`, and `-D warnings` makes the whole
    `rdec-core` test target fail to COMPILE — a lane that only ran `cargo test -p rdec-core`
    (where an unused binding is a warning, and clippy never ran) sees green.
  - `scripts/unwrap_ratchet.sh` red: crates/cli had 15 panic sites vs the baseline's 14 —
    `m6.rs:447 opt_wanted.as_deref().unwrap()`, a hand-guarded unwrap. Fixed by binding the
    Option (`if let Some(ow) = opt_wanted.as_deref()`), baseline UNCHANGED at 14. The comment
    that replaced it must not QUOTE the pattern: the ratchet is a byte-exact grep and counted
    the comment.
  The slice-3 ledger said "clippy clean, all core suites green". True of its own core suite;
  false of the three CI steps. Lesson: a lane runs `fmt --check`, `clippy --all-targets
  -D warnings` and `unwrap_ratchet.sh`, or it ships red CI — which is what this repo's CI
  history says it keeps doing.

CI WIRING. `.github/workflows/ci.yml` gains a `types` step in the `gates` job, placed
IMMEDIATELY AFTER `corpus (C + C++ objects…)` and in the same job (the gate reads
`corpus/manifest.tsv`, the objects it names, and `corpus/truth.tsv`; the objects are generated,
not committed — the trap that shipped red CI twice, runs 41415743892 and 42148930772), and the
release leg goes at the end of `release-parity`'s gate block, whose corpus step is above it. The
step runs the raw census, ASSERTS its exit code is 1, requires the battery green, and takes its
verdict from the ratchet:
  types --corpus                                  (exit 1 expected, census printed)
  types --selftest                                (must be 0)
  types --corpus --baseline .gate/types-baseline  (the step's verdict)
with `$LASTEXITCODE` checked after every one. The ordering and the exit-code checks are pinned by
`crates/cli/tests/types_ci_wiring.rs` (3 tests; each shown RED by sabotage: move the step above
the corpus step, delete the `if ($LASTEXITCODE …)` lines).

THE GATE IS NOT WEAKENED — the ratchet moves the exit code, never the scorer. `score_return` and
`score_int_grid` are byte-identical to bd3f68b's; `types --corpus` still prints all 4 LIE lines
and exits 1. New: `--baseline PATH` groups the printed lies into `subject|kind` FAMILIES
(derived from the scorer's own text, so there is no parallel counter to stop counting; an
unmappable text lands in `UNMAPPED(...)` and can never match a row) and compares against
`.gate/types-baseline` (`carry_chain|void-return 4`). Red on a RISE, red on a FALL ("lower the
baseline in this commit"), red on an unnamed family, error on a missing/empty file. Measured
end-to-end on the corpus, not only in a unit test: as-shipped 0; baseline 3 -> 1 "NEW over-truth
claims"; baseline 5 -> 1 "lower the baseline in this commit"; an extra row -> 1; a missing file
-> 2. `types --selftest` gains a 4th register gate (register 4/4) pinning that the baseline file
parses and names printable families; the 22-probe battery and `PROBE_RULES_TOTAL` are untouched
(22/22).

A BUG THE CI FORM CAUGHT BEFORE CI DID: the first arg-split skipped the `--baseline` FLAG's index
instead of its VALUE's, so `types --corpus --baseline .gate/types-baseline` handed the TSV to the
COFF loader — exit 2, "object with a 28515-byte optional header" — on the exact command line the
CI step runs. `baseline_arg_split_pins_the_baseline_path_value` pins it, and was run RED against
the pre-fix shape (`… .map(|i| i + 1)` removed) before it was run green.

CENSUS (this tree, local clang 22.1.8 — CI's clang is 20.1.8, so a CI census is a different
machine's claim): 128 functions scored, 640 sites, 308 unscoreable. signedness exact 144,
canonical-display-over-unsigned 27, sound-silent 141, LIE 0. width exact 150, narrower(sound) 21,
WIDER-LIE 0. pointer on-ptr 21, LIE 0. float on-float 0, LIE 0. void-return matches 0, LIE 4.
conflicts (loud by design) signedness 12, slot 0, precision 0. Byte-identical to the slice-3
numbers, so the merge moved no census counter — the silent-drop sin this lane was built to avoid.

THE GAP THAT REMAINS, MEASURED: 4 void-return lies, ONE family — `carry_chain` at O0/O1/O2/Oz.
`corpus/src/carry_chain.c` is `void f(unsigned long a, unsigned long b, unsigned long *lo,
unsigned long *hi) { …; *lo = l; *hi = h; }`. At O0 the tail is `movq 0x18(%rsp), %rax` /
`movl %ecx, (%rax)` / `addq $0x20, %rsp` / `retq`: RAX at `ret` is the spilled `hi` POINTER, left
over from the address role of the last store. `--explain carry_chain.O0` prints the pass's claim
verbatim: `return bb0 v84: claim Some("void *") facts read-width@0x43[Load W64] read-width@0x48
address@0x48` — the pointer claim is TRUE of that cell (it is `unsigned long *hi`), and the
scorer calls it over truth because the declared return is void. `void_return: false`, because
`sig.void_return = rets.is_empty()` is the machine fact "leaves through an exit, not a value",
and the IR has a value-returning terminator. The owner of the lie is the LIFTER: `Pending::Ret`
with `RetKind::Int` emits `Terminator::Return(read64(Rax))` unconditionally (lift.rs:4415-4419),
and `RetKind` for the C corpus is always `Int` — the lifter has no signature, because a COFF
object carries none.
WHAT A FIX WOULD HAVE TO CLAIM, measured on a twin, not theorised: the hand-built pair
(void `g` and `unsigned long *k` over the same body, clang -O0, same object format) has
BYTE-IDENTICAL bodies; `k` differs by exactly ONE appended instruction —
`movq 0x18(%rsp), %rax` at 0x9a, the answer re-materialised — before the same epilogue. So the
decidable fact is the ROLE of RAX's last write: in the void shape it is READ by a store as that
store's address and nothing writes RAX afterwards; in the returning shape it is never read at all
(it is the answer). A `RetKind::Void` rule has to claim exactly that — "RAX's reaching definition
at every return is consumed only as a memory address, so the function has no answer cell" — and
its fixture pair is this one (the void tail must go void, the pointer tail must stay a value, and
a rule that fires on both is the miscompile). The cost it must pay is named and is SMALLER than
feared: the four `carry_chain` objects are NOT in T2's compared set (all four are named by the
gate as `not run [memory outside the modelled frame] … no base arithmetic from param 0`; measured
alone: `t2 corpus/obj/carry_chain.Oz.obj` -> "1 lifted, 0 functions T2 can run", exit 1), so
closing the family removes NO comparison from the 1820/1820/0/4 tuple. It does have to keep
`verify`'s return-width bound and `emit_c`'s Void spelling honest, and truth.tsv keeps declaring
`void` — the fix that is NOT allowed is a scorer that stops scoring declared-void return sites.
Until that lifter change lands with machine evidence, the 4 stay in the census, in the printed
LIE list, and in `.gate/types-baseline`.

FLOORS (merged tree, un-piped, this box, per-command `CARGO_TARGET_DIR` outside the tree):
fmt 0; clippy -D warnings 0; ratchet 0 (core 1, arch-x86 7, demangle 1, verify 4, cli 14 — the
baseline is unchanged, nothing new and nothing removed); cargo test 45 suites / 403 passed /
0 failed (master's 43/364 + tyinfer_rules 28 + the 7 ratchet tests + the 3 CI-wiring tests);
selftest 0; objdump-check 0 (140 symbols, 140 exact, 0 refusals, 0 mismatches — the corpus grew
past the 124 the older floors quoted); discover 0 (100.00%, 6764/6764 non-padding bytes, gaps
none); lift 0 (140 lifted, 140 valid IR, 103 clean); c --corpus 98 emitted / 42 refused / 0
invalid with clang -Werror OK (the gate's own exit is 1 BY DESIGN while refusals > 0, as on
master); t2 --corpus compared 1820 / match 1820 / mismatch 0 / not-compared 4 with memory
408/408/0 and the IR leg 408/408/0 — the tuple is master's, byte for byte; t2 --cpp 856/856/0
over 39 compared functions; cpp --corpus 69 lifted / 64 emitted / 5 refused; demangle 68/68;
status 0. types --selftest 0 (battery 22/22, register 4/4); types --corpus 1 (the 4 named lies);
types --corpus --baseline 0.

WHAT THIS DOES NOT PROVE: not CI (nothing was pushed; the CI-form command line was run locally,
on clang 22.1.8, and CI's 20.1.8 is a different lowering — `objdump-check`/`c`/`t2` numbers can
move there for reasons that are the compiler's); not the C++ lane's coverage (the corpus has no
float/void C parameters: float 0/0 is silence, not skill); not that the baseline's family names
are the RIGHT decomposition (the C++ `--explain` lane can add families, and the ratchet forces
every one to be named in the same commit); not that a `RetKind::Void` rule is sound in general —
the twin pair above is the fixture that would have to prove it, and it is not written here.
---

## 12. M2 slice: `cwde`, and the constant folder that turned every folded sign-extension into a zero-extension (2026-11-08)

**What was asked.** The MSVC leg's refusal census ended with three classes that clang
does not have. One was `cwde` — 3 sites, byte `98`, at offset 8 in three `cl` objects.

**The census almost missed it.** `llvm-objdump` prints that byte as `cwtl` (AT&T), not
`cwde` (Intel). The first census grepped objdump output for `cwde` across all three
corpora and counted **zero**, which reads exactly like "nobody emits this". The
witnesses were found in the refusal text, not the disassembly. Any mnemonic census that
mixes an Intel-name key with an AT&T-name source has a silent-zero hole: `cwde`/`cwtl`,
`cdq`/`cqto`, `cwd`/`cltw`, `cbw`/`cbtw`. Measured site counts, all corpora:
`cwde` 3 (MSVC only), `cbw` 0, `cdqe` 0, and **0 in the clang corpus** for all three —
clang lowers the same C as `movsx eax,ax`, which was already modelled. So this arm's
entire witness set exists only on the second toolchain: a clang-only corpus would show a
correct, live, unexercised feature.

**Silicon first** (`scratch/sign_extend_probe.c`, executed from `VirtualAlloc`
`PAGE_EXECUTE_READWRITE`, not reasoned about):

| bytes | machine RAX | the claim it pins |
|---|---|---|
| `mov eax,0x12348123 ; cwde` | `0x00000000FFFF8123` | sign 16→32, **upper 32 zeroed** |
| `mov eax,0x00007FFF ; cwde` | `0x0000000000007FFF` | positive stays positive |
| `mov eax,0xFFFF0005 ; cwde` | `0x0000000000000005` | reads **only** bits 0..16 |
| `mov eax,0x12341281 ; cbw` | `0x000000001234FF81` | 8→16, upper 16 **preserved** |
| `mov eax,0x80000000 ; cdqe` | `0xFFFFFFFF80000000` | 32→64 |

Row 3 is the row that pays: a lifter that sign-extends the 32-bit cell answers
`0xFFFFF005` there and passes rows 1 and 2. Rows 1 and 3 together pin both halves of the
architecture, and `acc_forms.rs::cwde_sign_extends_ax_alone_and_zeroes_the_upper_half`
carries all three.

`cbw` and `cdqe` stay **refused**, pinned by
`cbw_and_cdqe_stay_refused_until_a_corpus_site_exists` (gaps == 1, refusal names the
mnemonic). Zero sites in any corpus means an arm that is green by construction; landing
one requires a snippet that emits it, in the same commit.

**What the fixture then found was not a `cwde` bug.** The fixture's first run was red on
the *immediate* case only:

```
RUNTIME ecx=0x12348123 -> RAX=0xffff8123   (correct, matches silicon)
RUNTIME ecx=0xffff0005 -> RAX=0x5          (correct)
CONST   mov eax,0x12348123 -> RAX=0x8123   (machine: 0xffff8123)
```

Same instruction, same lifter arm, same IR. The difference is that the immediate version
folds, and `const_cast`'s SExt arm read:

```rust
CastKind::SExt => w.trunc(x, true),   // w = the DESTINATION width
```

An SExt destination is **wider by construction**, so asking "is bit `dst-1` set?" of a
value that lives in the low `src` bits asks about a bit that is always 0. Every
sign-extension the folder ever applied to a constant was a zero-extension. Nothing else
in the pipeline can see this: the fold output is what a hand-written expectation is
written against, T2's vectors arrive in registers (Params, not `Const`s), and every test
that mentions `const_fold` calls it as a *step of a pipeline*, never as the thing under
test. That is the repo's worst defect class — a silent miscompile, no refusal, no gap, no
T2 signal — and it was found by a fixture written for a 3-site MSVC mnemonic.

**Two fixes, because the two evaluators must agree.** `passes.rs::const_cast` now signs
from the source width and keeps the destination's bits; `interp.rs`'s `Op::Cast` SExt arm
had the same shape and now matches it. The interpreter's version was independently
inconsistent with two neighbours: `Trunc` already returns a destination-width cell, `ZExt`
fits inside one, and the C this op emits — `(uint16_t)(int16_t)x` — answers `0xFF80` while
the interpreter answered `0xFFFFFFFFFFFFFF80`. x86 cannot show the difference through a
register (every write at the destination width truncates), which is exactly why T2's
2452 comparisons never saw it.

**New gate: `crates/core/tests/fold_agrees_with_interp.rs`.** Run the folded program and
the unfolded program on the same inputs, over 22 cast shapes (widening SExt/ZExt,
narrowing Trunc, same-width Bitcast, at every width pair) × 16 boundary vectors
(`0x80`, `0x8000`, `0x80000000`, `0x8000000000000000`, …) = 352 cells. Anti-vacuity is
asserted, not assumed: ≥300 cells swept **and** ≥90% of the programs must actually have
the cast removed by the folder, because "the folder folded nothing, so the answers agree"
is the vacuous pass this file exists to prevent. It bites on the first cell of the defect
it was written for: revert the folder and the first failure is
`sext W8->W16 of 0x80: the folder changed the answer` (`0xFFFFFFFFFFFFFF80` vs `0xFF80`).
Breadth lives here; the silicon truth lives in `acc_forms.rs`, because an agreement test
between two of our own programs is satisfied by both of them being wrong.

**M6 merged the same day.** `m6-landed-m6e` (tyinfer + `rdec types` + the CI baseline
ratchet) is in; the merge conflicted only in `ROADMAP.md` (kept the lane's two M6 rows and
this tree's M7 row, sections unioned). Re-measured on the merged 148-object tree, not
carried from the lane's 136: 128 functions scored, 640 sites, signedness LIE 0 (exact 150
/ canonical 30 / sound-silent 132), width 153 + 27 narrower, WIDER 0, ptr 21/0, conflicts
25, `--selftest` 22/22 + register 4/4, and the baseline ratchet **GREEN with 0 new
families** — the M3 slice's 12 objects added scoreable sites and no type claims, so
`.gate/types-baseline` needed no edit. The raw gate stays exit 1 by design on the 4 named
`carry_chain` void-return lies; the root cause is a lifter return-model gap (a COFF object
carries no signature; `lift.rs` emits `Return(read64(Rax))` unconditionally), not a type
inference gap, and it is the named next item on the type-recovery lane.

**Gate tuple, parent-run on this tree, un-piped, exit code = verdict** (`/tmp/floor_fold.log`,
the 16-gate floor): fmt 0, clippy `-D warnings` 0, `cargo test` 0 (**51 suites, 465
tests**), panic-site ratchet 0, `selftest` 0, `types --selftest` 0 (battery 22/22, register
4/4), `types --corpus` **1 — by design, the same 4 named `carry_chain` lies, no new
family**, `types --corpus --baseline .gate/types-baseline` 0, `objdump-check --corpus` 0
(156/156 exact boundaries), `discover --corpus` 0, `lift --corpus` 0, `flags` 0 (593,400
probes / 3,950,981 flag comparisons / 0 mismatches), `t2 --corpus` 0 (**2472 planned / 2452
compared / 2452 match / 0 mismatch / 20 not-compared `#DE`**, memory 408/408, IR leg
408/408), `t2 --cpp` 0 (856 / 856 / 0, not-compared 0), `pe-discover` on the linked image 0
(418 functions, 91.59%), `msvc-gates --limit 40` 0 (132 objects, 134 symbols, 40/40
boundaries, **118 emitted / 16 refused — `cwde` 3 → 0, leg refusals 19 → 16**, 6 census
keys, "nothing vacuous"). clang `c --corpus` on this tree: 156 lifted / **129 emitted** /
27 refused / 0 invalid. **`40d14e1`, measured in a clean worktree with the corpus
regenerated, prints the same 129 / 27** — so the ledger row's `125 emitted` was stale (its
own arithmetic said so: 125 + 27 = 152 against 156 lifted), and the claim this slice first
published from that pair — that the folder fix emitted four more functions, leaving a
four-function hole — is FALSE on both halves. The row now carries the correction. The only
corpus-wide movement this slice can claim is on the MSVC leg: `cwde` 3 → 0, refusals 19 →
16, which the fix touched directly.

**A number this slice first read wrong, recorded so it can't be read that way again.** The
T2 gate prints `-> 2472 comparisons` (planned inputs) and, further down, `compared: 2452`.
The first read of this run called that "+20 growth over the ledger's 2452". It is not
growth: the 20 are the machine-side `#DE` zero-divisor vectors, which are planned, run,
faulted, and excluded from the compared count by design. The ledger row now prints
planned-and-compared together for exactly that reason.

**Not proven.** No corpus function is *measured* as value-changed by the folder fix:
the fold path has no site census, so the blast radius is "every folded sign-extension of a
constant", of which the count is unmeasured. T2 staying at 2452/2452 proves only that no
*differentially-compared* function changed value — the class the fix targets is precisely
the one T2 does not reach. `cwde`'s 3 MSVC sites are refusals→lifted, not
value-verified: there is still no linked MSVC image, so no T2 leg over `cl` code exists.

## 13. `inc`/`dec` on a memory destination, the flag the model refuses to guess, and a floor script that was lying (2026-11-08)

**Census first, with AT&T names.** The MSVC leg's census carried
`inc/dec on an unmodelled destination` at 3 sites. Counting them with Intel names
(`grep -E '\b(inc|dec)\b'`) returns nothing useful, so the count was taken the way
`cwde` was: over `corpus/msvc/obj/*.obj` with `llvm-objdump -d`, filtering the AT&T
spellings. Result: **3 sites, all `incl (%rcx)`, all in MSVC objects**. clang lowers
the same C as `addl $1, (%rcx)`, which the lifter already modelled — so on a
clang-only corpus this arm is dead code, and a lane that census'd only `corpus/obj`
would have "proven" the refusal class empty. The `cwde` lesson, applied two days
later on the first try.

**The arm.** Load the cell, add/subtract 1, store back through the SAME address
expression, `set_flags(FlagsOp::Inc/Dec)`. The safety property is in `addr_of`: it
returns `Some` only for a memory operand it can express — `None` for `RIP` with no
relocation, `None` for no base — so a register destination can never be re-read
through memory, and `inc dword [rip+0]` in a harness with no relocation table stays
**refused** (fixture-pinned) instead of loading from an invented address. A silently
wrong address invents memory the program never had; a refusal does not.

**Silicon first (`inc_mem_probe.c`, VirtualAlloc RWX).** Each body returns
`(flag << 1) | cell`, so one run answers arithmetic and flags together:

```text
F1  cmp->CF=1 ; inc dword [rcx+8]  (cell 9 -> 10)   -> 12   (CF PRESERVED as 1)
F2  cmp->CF=1 ; dec dword [rcx+8]  (cell 9 ->  8)   -> 10   (CF PRESERVED as 1)
F3              inc dword [rcx]    (0x7FFFFFFF)     -> 0x80000002   (OF set, cell wraps)
F4  cmp->CF=0 ; inc dword [rcx]    (cell 5 ->  6)   ->  6   (CF PRESERVED as 0)
```

F1/F4 are a pair because one row cannot separate "CF preserved" from "CF forced":
F1 rules out forced-0, F4 rules out forced-1. F1/F2 carry `[rcx+8]`, so the address
expression under test is base+disp, not base+0. The probe had its own bug, worth the
sentence: the first version put the cells at the entry `[rsp]`, which is the **return
address**, and `ret` jumped to 5 — a segfault that reads like a bad encoding. Every
body now opens `sub rsp,0x20` and closes `add rsp,0x20`.

**The fixture found two things the plan did not contain.**

1. **`stc` and `clc` are refused by this lifter** (measured: the fixture's first
   version named `unmodelled instruction: stc` in its own refusal). CF has to be set
   by a comparison — `cmp ecx,edx` with `(0,1)` gives CF=1, with `(1,0)` gives 0 —
   which is the same trick the `cdq` fixture uses, and is what real code does before
   an `inc`-then-`adc` sequence anyway.
2. **The model does not preserve CF across `inc`/`dec`; it leaves CF undefined**, and
   that turns out to be the right shape of answer. Raw IR: reading CF faults
   `UndefFlag(CF)`. After the standard passes, the interpreter answers **0** — which
   is where a lesser fixture would have written `assert_eq!(v, 10)` and called the
   model done. The real question is what reaches generated code, and the emitter
   answers it:

   ```text
   Refusal { at: 19, reason: "opaque: CF after inc/dec needs the previous flags as an
          operand (SS1.2 item 3)" }
   ```

   The function is REFUSED, by name. So the interpreter's 0 never reaches C, and T2
   can never compare a wrong CF against compiled C, because a program that reads CF
   after `inc`/`dec` never emits at all. This is SS1.2 item 3 (thread the previous
   flags in as an operand), pre-existing — the register form has the same path — and
   this slice inherits it and pins it rather than closing it. The fixture asserts the
   leak direction: a CF-reading case MUST be refused, an OF-reading case must emit.

**Sabotage, both directions.** Spell `inc`'s flags as `FlagsOp::Add` — the `add $1`
temptation, which *defines* CF, as 0 — and the function emits, firing the leak
assertion ("a function that reads CF after inc/dec EMITTED C"). Store the
un-incremented `lhs` instead of `r` and the arithmetic assertion reddens. Both
restored; `acc_forms` 13/13.

**Leg movement, measured.** `msvc-gates --limit 40`: **121 of 134 emitted** (was 118),
**13 refused** (was 16), census keys **6 → 5**, and the `inc/dec` class is gone. The
3-function delta is exactly the 3 witnesses — the census denominator accounts for it,
it is not a vibe. Next class on the leg: `flag setting with an unmodelled first
operand` (3 sites, `array_walk.O2 @ 0x2f`) — memory-operand `test`/`cmp`, same shape
of work.

**The floor script was lying, and the sabotage run caught it.** The first
`floor_m6.sh` run printed `FLOOR_EXIT=0` while its clippy gate exited **101** (five
`needless_borrow` in the new fixture: `&cells[0]` is `&&[u8]`). The script's exit was
the LAST gate's exit, not an aggregate over the gates — so the headline number and
the pass condition could disagree, the exact defect class this repo has a rule
against, present in my own harness and invisible until a red gate happened to be
mid-list. `run` now takes `--expect N`, every gate declares the exit code it must
produce (`types --corpus` declares 1, by design, the 4 named `carry_chain` lies),
mismatches accumulate, and the script's exit is the aggregate. The floor in the
commit message below is that corrected script's verdict.

**NOT PROVEN, stated plainly.** (1) T2 has no leg over MSVC objects, so this arm's
arithmetic is silicon-fixture-pinned, not machine-differential; the observable on the
corpus is the census/refusal movement, not a value comparison. (2) The arm is written
generically over operand width, so `inc byte [rcx]` (`FE /0`) would lift, with **zero
corpus sites** — same no-witness rule that keeps `cbw`/`cdqe` refused, and no
fixture is claimed for it. (3) The 3 sites live in functions that emit, so their
values reach the emitted C; they are covered by leg-level emission, not by a
per-site differential. (4) `stc`/`clc` remain unmodelled: if a corpus ever grows a
site, the census will name it.
## 14. 2026-11-08 — M2: `cmp`/`test` with a memory destination, and a
## shared-helper off-by-two that had been hiding behind it

Triggered by the M2 row's "next class": `flag setting with an unmodelled first
operand`, 3 census records on the MSVC leg. It turned out to be three separate
things wearing one name, and the census text was pointing at the one that was
already fine.

### 14.1 The census under-reports a per-instruction class, badly

Counting memory-destination `cmp` sites in the leg's disassembly, with AT&T names
(the Intel-name census is the trap this file has now recorded three times —
`cwde`/`cwtl`, `inc`/`incl`):

```text
34  cmp sites with a memory destination
    cmpl %eax,0x20(%rsp)   x13
    cmpl %eax,(%rsp)       x8
    cmpl $0x0,0x20(%rsp)   x3
    ... imm and equal-width forms
```

The leg's census printed **3**. Not a bug: the census records the FIRST refusal
per function, and a function with thirteen memory compares that also contains a
call-clobber refusal is one record of "clobbered by call". Consequence for how
rows are read: **a class can be modelled end to end and move the leg's headline
by zero**, because emission is per function and the census is per function. The
head count for a class is its site count in the disassembly, never its census
record count.

### 14.2 The arm, and the direction pin

`crates/arch-x86/src/lift.rs`, `Mnemonic::Cmp | Mnemonic::Test`: when
`src(ins, 0, w)` finds no register, the arm now tries `addr_of(ins, 0)` and loads
the cell. The order is the entire claim — Intel's first operand is AT&T's LAST,
and `cmp m, r` computes **m − r**, so the load is the LEFT operand of the flag
op. Swapping computes r − m, flips CF/SF/OF, and sends every `ja`/`jb`/`setb`
downstream to the mirror image.

Silicon pair that pins the direction (`cmp_mem_probe.c`, returning `flag*8 + cell`
so one run answers the flag AND proves the cell read was the memory one):

```text
F1  [rcx]=5,  eax=7,   cmp [rcx],eax, setb  -> 13   CF=1  (5 below 7)
F2  [rcx]=7,  eax=5,   cmp [rcx],eax, setb  ->  7   CF=0  (7 not below 5)
F3  [rcx]=7,  eax=7,   cmp [rcx],eax, sete  -> 15   ZF=1
F4  [rcx]=0x7FFFFFFF, eax=-1,  seto         -> 0x80000007  OF=1
F5  cell 3, test [rcx],1, sete               ->  3   ZF=0
F6  cell 2, test [rcx],1, sete               -> 10   ZF=1
F7  [rcx+8]=5, eax=9,  cmp [rcx+8],eax,setb -> 13   base+disp
```

F1 alone does NOT pin the direction (an `r − m` model gives CF=0 → 5, and F1's
cell 5 would answer… 5). F1+F2 together do, and both are in the fixture.
Sabotage, run through the real binary: swapping `a`/`b` in `FlagsOp::Sub` reddens
the new fixture (F1 answers 5, silicon says 13) **and** independently reddens the
pre-existing `cdq_leaves_flags_alone` — one semantic break, two fixtures.

Fixture hygiene worth keeping: F5/F6 use a DWORD store, not the byte store the
probe used, because the shared read-back (`mov edx,[rcx]`) reads 4 bytes and
silicon's high bytes were **stale stack** (0x15fe03) — a fixture must not bake a
garbage byte into an expectation. `cmp`/`test` define their flags, so these
functions also assert `emit_c(...).is_ok()`: a refusal here would mean the arm
lost the value (contrast §13, where the CF gap makes the emitter refuse on
purpose).

### 14.3 A shared-helper off-by-two: `next_ip - 4`

Fixing the arm did not move the leg, so one real object was instrumented
(`array_walk.O2.obj :: f @ 0x2f`, `cmpl $0x2, (%rip)`):

```text
ZZ gap at 0x2f len=7 kind=Memory base=RIP hits=["0x31"]
```

The operand kind is Memory, the base is RIP, the relocation is in the map, inside
the instruction's span — and `addr_of` still returned None. `addr_of` and
`rip_sym` were computing the displacement field as `next_ip - 4`, which is right
only when the disp32 is the instruction's LAST field. `83 3D <disp32> <imm8>` has
an **immediate after** the displacement: the field spans offsets 2..6 of a 7-byte
instruction, `next_ip - 4` = 4 lands INSIDE it, and the table keys the field's
first byte (2). The same assumption covers `81 /7 <disp32> <imm32>`,
`C6 /0 <disp32> <imm8>`, `F6 /7 <disp32> <imm8>` — a whole encoding family, in a
helper used by every RIP-relative `lea`/`mov`/`cmp`/`test`.

Fix: `rip_reloc` resolves by SCANNING the instruction's own span, the rule
`reloc_on` already used for branch targets ("scan the span, which keeps every
encoding width the same question"). A span carrying two distinct targets is
refused, not guessed. `RelocMap`'s doc corrected: it keys the field's FIRST byte;
call/jump displacements really are last, RIP displacements are not.

Fixture: `a_rip_destination_resolves_when_the_displacement_is_not_the_last_field`
supplies the relocation at offset 2 and requires the load to be built at the
target (0x1000 = `Const(4096)` in the IR dump — the only way that constant enters
the IR is the relocation). The arithmetic version fails it with a gap.

### 14.4 What the 3 records actually are: an address-space gap, not a lifter gap

With the span fix in, the site still refuses — for the reason the probe was built
to find: the relocation names `g_count` and carries **no address**. The leg's
relocation map places CODE sections (`idx_of`/`base_of` are built over code
sections), so a compare against a static in `.data` has a target with no cell.

The census called that "flag setting with an unmodelled first operand", which
points the work queue at the lifter, which models this operand correctly. The
refusal now splits the two causes:

```text
flag setting against a cell whose relocation names no address   (relocation present, addr None)
flag setting with an unmodelled first operand                   (form genuinely inexpressible)
```

Fixture `a_relocation_that_names_no_address_says_so_instead_of_blaming_the_operand`
asserts the new text PRESENT and the old text ABSENT, so the misdiagnosis cannot
quietly come back. All 3 leg records moved to the new key: the leg prints

```text
refusal records: 13 in 13 refused functions, 5 census keys
      4  clobbered by call
      3  flag setting against a cell whose relocation…
      3  tail call to 'ext'
      2  lea of an address we cannot name
      1  call to 'ext'
```

and the emitted count stays **121 of 134** — honest, per §14.1: `array_walk.O2 ::
f` also refuses on `live-in Rdi` and a conditional branch, so this class was
never the binding constraint on that function's emission. Closing the class
requires deciding where data symbols live in the per-object address space
(an M1/M4 scoping call), not a lifter arm.

### 14.5 One clip rule

The new key is longer than 46 chars, which put the leg's raw `&low[..46]` under
load — the same defect that invented `__Imag` in §12, plus a UTF-8 panic risk on a
multi-byte character straddling byte 46. The leg now shares `m4::clip` (announced
cut, word-boundary break, char-boundary-safe), pinned by
`a_long_census_key_announces_its_cut_and_breaks_on_a_word`. The first attempt at
this patch dropped `census_key`'s colon-cut and the pre-existing key test caught
it in the same run.

### 14.6 State

`acc_forms` 17 tests; `rdec-cli` bin 50; full suite 51 suites / 0 failed;
fmt and `clippy -D warnings` clean. Legs: clang `c --corpus` 156 lifted /
129 emitted / 27 refused / 0 invalid (unchanged — the clang corpus has no
memory-destination `cmp` with these shapes); MSVC 121 of 134 emitted, 13 refused,
5 census keys with the class renamed. Floor tuple in the commit message.

---

## Slice 15 (2026-11-08) — M2: CF is PRESERVED across `inc`/`dec`, and the flags node carries the value (SS1.2 item 3 closed within a block)

The emitter used to refuse any function that read CF after an `inc`: `opaque: CF
after inc/dec needs the previous flags as an operand (SS1.2 item 3)`. That was the
honest answer for a model with no such operand — silicon PRESERVES CF (measured:
F1 answers 12, the model answered 10). The operand is the flags value being
preserved, so the flags node now carries it.

```text
FlagsOp::Inc { a, cf_in: Option<VId> }        (Dec the same)

fuse    CF of a carried node lowers to getflag(CF, cf_in); the pass splices it
        (SO::Flag → Op::GetFlag) and the fixpoint lowers THAT read by its own
        producer's rule — `cmp ecx,edx ; inc ; setc` becomes `ecx < edx` in C.
interp  reads the CF out of the carried Val::Fg (raw IR, no pass involved).
ir      uses() lists cf_in: it is a real SSA use, so DCE cannot delete the
        producer out from under the node that preserves it (measured: dropping
        it from uses() reddens the fixture, the fusion cone shrinks with it).
lift    cf_in = self.flags, ONLY when self.flags_block == self.cur (see below).
fmt_ir  prints the carried value as the second operand: a dump that dropped it
        would print two programs with different CF semantics identically.
```

**Function entry is a refusal, not a 0.** With no producer, silicon's CF at entry
is the caller's and the model has no value for it, so `cf_in = None` and the read
lowers to an opaque refusal naming that. `m0_gate`'s b6 test pins the shape, and a
new fixture pins the lifter's side of it.

What the flip found in the FIXTURE, not the model
-----------------------------------------------

F1's bytes were `mov rcx,rsp` THEN `cmp ecx,edx`. The comparison's first operand
is ECX, which the address setup had just overwritten, so the cmp compared a stack
address with the input and CF was a **constant 0** — visible in the IR as
`cmpult 0x7ffe0, edx`. The fixture's "CF = 1" leg proved nothing, and the old
assertion (the model's 10, plus "the emitter refuses, naming the gap") could not
see it: a refusal is a green test for a program whose flag is constant. The
comparison now runs before the address setup, and F1 answers **12**, F2 **10** —
the silicon numbers — with the CF-reading functions emitting C.

What the leg's gate found in the MODEL
--------------------------------------

Carrying the flags value across a join is a use its block does not dominate:

```text
corpus/msvc/obj/early_returns.Od.obj      :: $LN10: %63 defined in bb5 does not dominate use in bb8
corpus/msvc/obj/switch_fallthrough.Od.obj ::  $LN9: %59 defined in bb3 does not dominate use in bb5
```

The leg's `lift` gate went red on those two lines on the first run — the same gate
that had been green for nine pushes, on a change that touched no leg-specific code.
SSA has no flag phis (SS1.2 item 2 is a PREREQUISITE, not a follow-up), so the
lifter sets `cf_in` only when the flags value was defined in the current block:
CF is preserved **within** a basic block; across a join the read refuses and names
the missing producer. `cf_is_carried_within_a_block_and_not_across_a_join` builds
the join in raw bytes and asserts the hard SSA error plus the refusal; removing
the guard prints the leg's exact text locally.

Coverage found the same way
---------------------------

Sabotaging the interpreter's carried-CF read to `Some(false)` left the post-pass
fixture **GREEN**: by then the pass has replaced the read with `cmpult`, so the
interpreter's flags arm is unreachable there. The interpreter's path is pinned by
two raw-IR tests (memory and register form, `run_budget` on unpassed IR), which is
where it answers — 12 and 1.

Sabotages, each reddening a named test
--------------------------------------

| Sabotage | Red |
|---|---|
| interpreter CF → `Some(false)` | `raw_ir_reads_the_carried_cf_with_no_pass_to_thread_it` (and ONLY that — the discovery above) |
| lifter `cf_in: None` (memory arm) | the F-series fixture (10 vs 12) + the raw-IR test |
| `uses()` drops `cf_in` | the F-series fixture (the fusion cone loses the producer) |
| dominance guard removed | `cf_is_carried_within_a_block_and_not_across_a_join`, with the leg's `%N defined in bbX does not dominate use in bbY` |

Measured
--------

| | before | after |
|---|---|---|
| `acc_forms` | 17 tests | **21 tests**, 0 failed |
| F1 (`inc [rcx+8]`, CF from a `cmp`) | 10 (CF undefined → 0) | **12** = silicon |
| F2 (`dec [rcx+8]`) | 8 | **10** = silicon |
| CF-reading functions | emit refused | **emit** |
| clang `c --corpus` | 156 / 129 / 27 / 0 | **156 / 129 / 27 / 0** (no corpus function reads CF after an `inc`) |
| MSVC leg gates | exit 0 | exit 0, 134 functions, 132 verify |
| `cargo test` / fmt / clippy | 0 | **0 / 0 / 0** |

Rules this slice leaves behind
------------------------------

1. **A flags value is an SSA value, with SSA's obligations.** Any new operand on a
   flags op must appear in `uses()`, in the SSA rename map, in the pass's operand
   walk, and in the IR dump — and may only be used where its block dominates. The
   corpus leg is the fastest dominator of them all: it found the violation on
   production-shaped objects that no unit fixture resembled.
2. **A refusal assertion can hide a dead fixture leg.** F1 asserted "the model
   answers X and the emitter refuses" for a program whose flag was constant 0.
   When a fixture's expected value equals "the model's undefined answer", check the
   premise IN THE IR, not in the prose.
3. **Preserve-within-a-block is the honest half.** Closing the join needs flag
   phis (SS1.2 item 2); naming that as the prerequisite is the result, not a
   deferral.

## Slice 16 (2026-11-08) — M6: the last LIE in the ledger, closed by a CONSUMED-RETURN rule

`types --corpus` carried exactly one red family for two slices: `carry_chain|void-return 4`,
one per opt level of a function whose source declares

```c
void f(unsigned long a, unsigned long b, unsigned long *lo, unsigned long *hi)
```

while the compiled tail leaves a value in RAX and the lifter records `Return(value)`.
The type pass then faithfully reported a claim over that cell and the scorer called it a
LIE (`return claim over declared void`) — an over-truth claim, the only one left in the
census. It is now 0, and the gate exits 0.

### The two measured legs (facts, not adjectives)

The four legs are two shapes, and `--explain` naming each witness is what made that
visible:

* **-O0** returns the cell the loop stores **through**. `--explain carry_chain.O0`:
  `return bb0 v84` with `store-address@0x48` on the returned value — the value's own role
  is a store destination.
* **-O1/-O2** return the computed cell whose **trunc copy** the body stores through an
  out-parameter: `Store(Int W32, 3, v27)` with `v27 = Cast(Trunc, W32, v24)`, and
  `return bb0 v24`. The `stored-value` fact lands on `v27`, not on the returned `v24`.

That second shape is the whole lesson of the slice. The claim walk follows **uses**
(the value's materialisations), so consumption of a *copy* is consumption of the value —
and a rule written against the seed's OWN facts fixed one leg and left three red
(measured: `void-return matches 1 LIE 3`). `Fact::StoreAddr` (destination role) and
`Fact::StoredValue` (stored source) are checked on **own OR chain**, and the withdrawal
runs LAST in `claim_from`, after every field is computed: the first version reset the
claim in the middle of the function, so the width bound was computed back onto the
"withdrawn" claim (`silent=false, consumed=true`) and the probe failed. A "silence
everything" statement has to be the last statement.

The rule is stated where it lives: a returned value the body consumes has its only
positive role evidence inside the body, and a COFF object carries no prototype to say the
cell is an answer, so printing its type as a return type prints information the machine
never showed. **Cost, said plainly:** a C function that both stores a value and returns it
(`*out = s; return s;`) loses its RETURN claim (the cell keeps every fact it earned as a
parameter or site, and the site census is unchanged — measured: signedness/width/ptr
buckets byte-identical, 150/30/132, 153/27, 21/0). A returned value the body never
consumes keeps its claim; that leg is a probe, not prose.

### The probe-builder trap that made three probes pass for the wrong reason

`crates/cli/src/m6.rs`'s `Probe::finish` fills blocks whose terminator is `Unreachable`.
`Probe::new` gives the entry `Return(NO_VAL)`, so a probe that "returns" a value through
`finish` is a **void function**: `void_return = true`, `ret` a default silent claim, and
every assertion about `sig.ret` passes for reasons unrelated to the rule under test. Added
`finish_ret`, which sets the ENTRY terminator, and the three consumed-return probes use it.
Any future probe that asserts on `sig.ret` needs it.

### The baseline's new contract: a declared zero

`.gate/types-baseline` no longer carries a row. A rows-less file used to be the "gate that
checks nothing" and was refused; now the file must carry a `# known-lies: N` header whose
value equals the sum of its rows, so:

* zero rows + `known-lies: 0` = the closed-census state, legal, and non-vacuous — every
  printed lie is a RISE against zero, and the RAW gate (which CI now asserts exits **0**,
  was 1) is the checker;
* rows + a marker that disagrees with their sum = refused (a baseline cannot drift from
  its own declaration);
* no marker = refused, same as the old empty-file rule.

`crates/cli/tests/types_ci_wiring.rs` enforces the pairing: a baseline that declares zero
forces ci.yml's raw-gate step to assert exit 0 (it reads the step's own line and quotes it
in the failure). The old "scorer that stops looking at declared-void sites" prohibition is
inherited unchanged — the scorer's `void_return || ret.is_silent()` test is untouched, and
the withdrawal happens in the claim layer, where silence is the module's honest output.

### Sabotage pairs (each direction has a witness)

* Drop the chain terms → `a return claim consumed by the body through a stored copy is
  withdrawn` FAILS and the corpus goes RED with `void-return matches 1 LIE 3`.
* Fire on every return → `a return claim the body never consumed keeps its type` FAILS.
  The corpus cannot witness this direction (`void-return matches 4 LIE 0`,
  `pointer on-ptr 21` unchanged — no scored corpus site returns an unconsumed pointer),
  which is exactly why the pair probe is load-bearing and not decorative.

### Measurements

`types --corpus`: 128 functions, 640 sites, LIE 0 in every bucket,
`void-return matches 4 LIE 0`, exit 0. `types --selftest`: battery **25/25**
(`PROBE_RULES_TOTAL` 22 → 25, three new probes), register 4/4, exit 0.
`types --corpus --baseline .gate/types-baseline`: GREEN, 0 named, 0 new.
Emitted C is untouched by this slice by construction: `tyinfer::recover` is called only
from the gate, so T2's tuple and the C census are unchanged by the same argument the
m6e merge row used.

### What this did NOT close

The LIFTER still records `Return(read64(Rax))` for these functions
(`crates/arch-x86/src/lift.rs:4415`), so the emitted C of a `void` source function prints
a `return` with a value. T2 cannot see it — all four legs are outside the compared set
(`not run [memory outside the modelled frame]`) — and its owner is M4's void-return model:
a `RetKind::Void`/no-value terminator with a `verify` bound, not a type-recovery rule.

---

## 17. T1's full grid re-proved on today's flag model (measured locally, 2026-11-08)

CI runs the SAMPLED grid on every push (593,400 probes / 3,950,981 flag
comparisons, ~30 s); the 9.7-billion-probe `--full` sweep is a nightly job, which
means every flag-model change this repo landed today — `cwde`, memory `inc`/`dec`,
memory `cmp`/`test` in Intel order, `CF` preserved across `inc`/`dec`, the
constant-fold SExt fix — was pushed on sampled evidence. It has now been re-proved
at full depth on the owner's machine, at `57747c0`, release build:

```
9728273400 probes, 69666730981 flag comparisons, 17835591539 undefined-skips, 0 mismatches
elapsed 359.1741734s
OK: flag model matches hardware on every probe          T1FULL=0
```

**Zero mismatches over 9,728,273,400 probes and 69.7 billion flag comparisons.**
The probe count is above the nightly's 9,704,437,200 because the model now has more
flag-setting sites to probe than when that number was recorded — the expected
direction for a growing model, and the reason a probe count is printed rather than
assumed. `undefined-skips` (17.8e9) are the SDM's undefined-flag cases the oracle
deliberately does not assert; they are counted so a change that silently STARTS
asserting an undefined flag shows up in this number.

What this does NOT buy: the full grid covers the flag model's tables, not the
lifter's flag plumbing (which flags node a given instruction writes) — that is the
`lift` gate, T1R/T1L and T2's territory, and it runs sampled in CI. The nightly
keeps the full sweep as its regression net; this section is a dated, locally
measured re-proof of the model under all of M2's additions.

---

## 18. Calls: arity, the Win64 clobber fix, and frame activation (measured 2026-11-08, branches `arity-wip` `def1647` + `frame-activation`)

Three facts about calls landed in two steps on the same base. All numbers are
from `rdec c --corpus` / `rdec t2 --corpus` on this machine (clang 22.1.8).

**Arity recovery (`arity-wip`, `def1647`).** A callee's parameter count is
recovered from the callee's OWN lifted body, not from a name, an ABI, or an
observed answer: `Function::args_read` is **max READ parameter index + 1** —
terminator reads included, a self-forwarding write (`%x = add %x, 1` where the
definition reads its own cell) is NOT a read, and a phi destination counts as
a read reached by walking the phi's destinations to their producers. The
soundness direction is one-way: over-estimating keeps today's loud refusal,
under-estimating would be the miscompile, so the number is never a count of
*leading* reads. The anti-invention floor is **the 4-pair default**: a callee
the pipeline cannot see into (an import, a relocation naming no symbol, a
defined symbol with no lifted body) keeps four pairs, stated in the table,
printed by the `arity:` line — the default is LOUD, never a claim. The
mechanism is a **two-pass** per object: pass 1 lifts every subject with the
table as-is; each defined symbol's pass-1 body lowers its table entry; pass 2
re-lifts with the enriched table, and every printed counter comes from pass 2.
**Recursion is covered by the same rule**: `f`'s own pass-1 body is the
evidence for `f`'s calls to itself (`fib_rec`). Measured:
`arity: 156 of 156 in-object callees recovered, 4 left at the 4-pair
default`; `c --corpus` 129/27 → **137/19**, the `opaque: clobbered by call`
class GONE, clang -Werror OK.

**The Win64 caller-saved fix (same base; its own ledger row above, because it
is the largest correctness fix there).** The lifter's caller-saved list was
SysV's: it clobbered `RSI`/`RDI` after every modelled call, and on Win64 those
registers are nonvolatile. The witness is `movl %edx, %esi` after a call —
the machine's `ESI` survives, the model's did not. The list is Win64's now;
**clobber cells 40 → 38**, and removing the two false clobbers closed **5 of
the 8** clobber refusals on its own (arity closed the other 3). A register
table is a machine fact about the TARGET platform; the mirror error
(under-clobbering a volatile register) is the silent-miscompile direction,
which is why this is an ABI fact pinned per platform, not a tuning list.

**The frame-activation discovery (`frame-activation`).** `t2 --corpus` on
`def1647` went RED in exactly the two newly-emitted families: **2607 | 2580 |
27 mismatch | 21 not-compared** — 21 in `call_helper` (of 96), 6 in
`fib_rec` (of 59). Root cause, reproduced OUTSIDE the harness
(`arith_probe/probe.c` links `corpus/c/all.c` against
`call_helper.O0.obj` and drives both sides with the same `(a, b)`): the model
gives EVERY function the same frame base — entry RSP is the `STACK_BASE`
constant (`1 << 19`) — so a caller's frame slots and a callee's frame slots
ALIAS in the model's flat memory. The machine gives every ACTIVATION a frame;
the model has one per FUNCTION. Witness in the emitted text: `o8_f`
(`call_helper.O0::f`) uses slots `0x7fff8` (`a`), `0x7fffc` (`b`), `0x7fff0`,
`0x7fff4`, and its callee `o8_h` uses `0x7fffc` and `0x7fff8` — the same
addresses; the `-O0` leg answers `3*(a*a+1)` where the machine answers
`3*(a*a+b)`: `h`'s frame clobbered `f`'s slot for `b`. The `-O1` leg agrees on
every vector (no frame slots). `fib_rec` is the same defect through
recursion: activation 2 shares activation 1's frame.

**What the refusal now covers.** The gate (`crates/cli/src/m4.rs`,
`frame_alias_map`; consumed by `c` as a census-classed refusal and by `t2` as
a not-run) computes, per function, the frame-window addresses it TOUCHES
(every `Load`/`Store` whose address resolves through `Const`/`Copy`/`add`/`sub`
of constants into `[STACK_BASE - 0x1000, STACK_BASE)` — the lifter
materialises frame addresses as plain constants), and refuses the CALLER when
a symbol-named, same-object callee shares a slot that the caller LOADS
CONTROL-WISE AFTER the call (same block, later instruction, or any block
forward-reachable from the call's successors), or when the callee is the
caller itself (recursion; text says `calls itself`). Two measured bounds:
the **import leg** (a callee with no lifted body shares NOTHING — refusing on
imports would punish every `memcpy` caller) and the **liveness bound** — the
literal shared-set intersection over-fired on `tailcall_local.O0::f`, whose
IR reloads both slots BEFORE the call and whose t2 vectors MATCHED at
`def1647`: the refusal may only cost what the defect's reach covers, so a
caller that never reads a shared slot after the call keeps its emission (a
modelled `TailReturn` call site is skipped for the same reason — the caller's
activation dies at the machine's `jmp`). Measured: `c --corpus` **135
emitted / 21 refused** (refusals `o8_f`↔`o8_h` both using `0x7fff8`;
`o52_f calls itself`), `t2 --corpus` **2560 | 2560 | 0 | 20** exit 0 with the
base **2452 untouched** (2560 = 2452 + 72 + 36), memory 408/408, IR leg
408/408, `types --corpus` exit 0 with an unchanged census, `lift --corpus`
exit 0, MSVC leg 134/125/9 with no frame-class record. Fixtures:
`m4.rs::frame_alias_tests` (alias, anti-over-fire incl. import / disjoint /
tail-call / pre-load legs, recursion incl. the no-frame-recursion pair) and
`crates/cli/tests/frame_activation.rs` (corpus gate, loud skip, class-sum ==
refused-count parsed from the gate's own lines); each direction was shown RED
by its sabotage (always-emit, ignore-the-callee's-set, drop-the-recursion-
clause).

**The named next step.** The refusal is a stop-gap that names its own
obsolescence: **model the frame as an implicit parameter** (entry RSP becomes
a parameter; callee writes go to a fresh activation window, e.g. RSP' =
caller-RSP − frame-size, per `docs`/§1.4's stack-delta plan), so each
activation gets distinct memory and `call_helper.O0`/`fib_rec` EMIT and are
compared again. The liveness rule is then subsumed: with per-activation
frames the touched sets are disjoint by construction, and the refusal
shrinks to zero on its own — which is the test that the model replaced it.

### 18.1 The refusal moved with the compiler that built the inputs (measured 2026-11-08, branch `frame2`, base `ab64a4d`)

**The divergence, verbatim.** The `gates` job on `ab64a4d` went red in two
tests (`crates/cli/tests/arity_corpus.rs:83`,
`crates/cli/tests/frame_activation.rs:78`) on ONE fact, the same
`emit-c:` line read two ways:

```
CI   (windows-latest): emit-c: 156 lifted functions, 135 emitted C, 20 refused, 1 invalid after passes
                       assertion `left == right` failed: 0 invalid … left: 1  right: 0
this box (same commit): emit-c: 156 lifted functions, 135 emitted C, 21 refused, 0 invalid after passes
```

Same lifted, same emitted, a DIFFERENT BUCKET: on CI one of the two frame
refusals did not fire, so that function EMITTED, and the differential pass
then caught the emitted C as `invalid`. `.github/workflows/ci.yml`'s own
`toolchain` step prints the runner's clang — **20.1.8** — where this box
builds the corpus with **22.1.8** (the version the manifest header records).
The corpus objects are generated per machine, so the two runs are not the
same input: a 20.1.8 object spelled one frame store as
`mov %edx, (%rsp,%rcx,4)` → `add 0x7fffc, mul(%rcx, 4)`, whose address
`const_addr` cannot resolve, so the function recorded NO frame traffic,
`inter.is_empty()` said "shares nothing", and the caller emitted.

**`invalid` is the witness, not the bug.** The `1 invalid` is the pipeline
being honest: the emitted C computed a wrong answer because the callee's
activation overwrote a slot the caller reads back. Over-approximating on this
defect is loud (a refusal: counted, censused, named); under-approximating is
silent (a wrong answer, visible only if the vectors happen to cover it — the
direction rule this repo has now been paid on twice, after the clobber list).
**A refusal keyed on a compiler's spelling is not a refusal.**

**The rule.** `frame_touch(f) -> FrameTouch { slots, unbounded }` — the
resolvable in-window slots as before, PLUS `unbounded` when the function has a
`Load`/`Store` whose address is FRAME-ROOTED and not constant-resolvable:
frame-rooted means its operand tree contains an in-window `Const`, walking
`Copy`/`Cast` every `Bin` arm (`Mul`, `Shl`, …), `Select` and `Phi` at
`const_addr`'s own depth bound. A `Param`-rooted tree (the seeded
caller-buffer shape `t2`'s memory comparison covers) contains no in-window
constant and stays out, exactly as `const_addr`'s doc-comment promises.
Sharing in `frame_alias_map`: an unbounded callee may write ANY window address,
so it shares the caller's WHOLE known frame; an unbounded caller's live set IS
the window, so it shares it with any callee that touches the frame at all
(every frame-rooted load after the call is then a shared load — the wildcard
widens WHICH load, never WHETHER liveness is required); both bounded is the
literal intersection. Imports stay exempt, `Terminator::TailReturn` call sites
stay exempt, self-calls keep the recursion wording, and the refusal text says
WHICH side is unbounded.

**Measured in this tree (clang 22.1.8, objects regenerated here).** `c
--corpus`: **156 lifted / 135 emitted / 21 refused / 0 invalid** — UNMOVED,
and that is the point: the widening costs no emission on this codegen. Its own
line: `frame traffic: 49 functions on named slots | 3 with unbounded (indexed)
frame traffic (3: o65_f, o66_f, o67_f) | 104 with no frame traffic` = 156, a
partition of the lifted set asserted by the corpus gate (the buckets move with
the compiler, the total does not). The three unbounded functions are all
`locals_addressed` — `int *p = &a; … p = (p == &a) ? &b : &a;` — a moving
pointer over two locals, and every one of them is a LEAF, which is why the
refusal count did not move. `t2 --corpus` exit 0: **2560 | 2560 | mismatch 0 |
20 not-compared**, memory **408/408/0**, IR leg **408/408/0**, and the
gate-to-gate output diff against the pre-fix run names NO lost comparison.
`lift --corpus` exit 0 (156/156 verify), `types --corpus` exit 0 with an
unchanged census, `.gate/types-baseline` UNCHANGED (the ratchet is
lie-family-keyed and the family counts did not move). MSVC leg,
`RDEC_CL=… scripts/msvc-cl.cmd rdec msvc-gates --limit 40`: exit 0,
`msvc-gates OK: 132 objects, 134 code symbols, census printed, nothing
vacuous`, and its `c` leg prints **134 lifted / 125 emitted / 9 refused / 0
invalid** over 4 census keys, none of them the frame class (the leg's census
runs through `msvc_census`, which prints its own `emit-c:` line and no
frame-traffic line); the generated `corpus/msvc/manifest.tsv` came back
byte-identical to the committed one.

**Fixtures, and one gap they found.** `m4::frame_alias_tests` gained (d) an
indexed-callee refusal and (e) an unbounded-caller refusal, both HAND-BUILT IR
with no compiler in the loop — (d) is the proof the rule does not depend on any
one clang's spelling: it is the pre-fix bug reproduced with `Const`/`Param`/
`Mul`/`Add` alone, red on `ab64a4d` (`the caller EMITTED: … refusal map: []`)
and green on the rule; (f) pins the anti-over-fire side (a `Param`-rooted
computed address is NOT frame-rooted); (g) pins that the wildcard does not
delete liveness (an unbounded caller that reloads its frame cells before the
call, plain and tail-call legs). Sabotages, each shown red: bounded-only
(unbounded treated as touching nothing) reddens (d)+(e)+(g)'s liveness;
"unbounded skips liveness" reddens (g); deleting the liveness gate for ALL
sides reddens the pre-call-reload leg; and deleting the `TailReturn`
exemption kept every pre-existing fixture GREEN — the exemption had NO witness
on this base, so a leg was added for it (`tail_caller_post_load`) and the
deletion now reddens it.

---

### 18.2 What `invalid` actually means: the correction 18.1 owed (measured 2026-11-08)

`9876439` (the unbounded-frame rule) merged as `510c3e4`, and CI printed the IDENTICAL fact it was
supposed to fix: `135 emitted C, 20 refused, 1 invalid` — same bucket, same count, the runner's
clang unchanged. The frame-spelling hole and the CI failure are therefore NOT the same defect, and
the reason two fixes were written against the wrong mechanism is a reporting gap. Both are recorded
here rather than edited out of 18.1.

- **`invalid` is `verify_ssa` FAILING AFTER THE PASSES** (`m4.rs`, the `pass_broken` counter): a
  structural IR break. It is NOT "the emitted C miscomputes". A frame-aliasing caller that escapes
  a refusal lands in `emitted`, and the gate that catches a wrong ANSWER is `t2` — which was GREEN
  (`2560 | 2560 | 0`) on the very CI run whose `invalid` was read as a caught miscompile. 18.1's
  "the differential pass caught the emitted C as `invalid`" is wrong, and the corpus floors' own
  assertion text carried that sentence into CI's log; both now state what the counter counts.
- **The name existed and was invisible.** `m4.rs` builds the `first_pass_break` string, prints it
  only on the final `FAIL:` line, and the corpus floor tests capture stdout and discard it, so the
  CI log carried a bare count. The gate now prints
  `invalid record: <obj> :: <sym>: <verifier errors>` **unconditionally** — unconditionally, unlike
  the refusal records behind `RDEC_C_CENSUS_ALL`, because a bucket that must be empty makes every
  member a bug. Proven by injecting a synthetic verifier failure at the first subject:
  `invalid record: corpus\obj\arith_basic.O0.obj :: f: SABOTAGE(synthetic): value %7 used before
  def` alongside `emit-c: 156 lifted functions, 134 emitted C, 21 refused, 1 invalid`; the file was
  restored byte-identical (sha256 `a0be65da…`). The two floors now splice those record lines into
  the assertion message, so a red CI prints the function, not just the number.

WHAT STANDS from 18.1: the frame-spelling hole is real — the hand-built fixture is red on
bounded-only code (`the caller EMITTED ... refusal map: []`), the bounded/unbounded rule closes it,
T2 is green, and the census partition (`49 named | 3 unbounded | 104 none` = 156 lifted) is on the
gate's output. WHAT IS OPEN: the CI defect, a post-pass SSA break on clang 20.1.8 codegen, whose
member the next CI run will name. A count you cannot name is not a diagnosis, on either side of the
pipeline.

## 19. Data symbols: what the answer has to cover, measured on the MSVC leg (2026-11-08)

The roadmap has carried "where do data symbols live in the per-object address space" as an
open scoping call for the clang leg's `lea` records. Today it acquired an MSVC-leg witness, a
measurement of what a naive answer misses, and the tool the decision has to be made with.

`rdec msvc-gates --limit 40` (my tree, HEAD ab64a4d, exit 0): `134 lifted functions, 125 emitted
C, 9 refused, 0 invalid`, census `3 flag setting against a cell whose relocation names no
address | 3 tail call to 'ext' | 2 lea of an address we cannot name | 1 call to 'ext'` (sums to
9). The 3 + 2 + 1 = 6 records naming a data target are this decision; the 3 + 1 'ext' records are
the extern-prototype blessing, a separate decision.

The mechanism, at the instruction, because the census key says it and the bytes confirm it. The
first record is `corpus/msvc/obj/array_walk.O2.obj :: f @ 0x2f`, whose instruction is
`cmpl $0x2, (%rip)` — a comparison against a `.data` static. Its relocation, read from the COFF
tables (`python scripts/coff-relocs.py corpus/msvc/obj/array_walk.O2.obj --at 0x20 0x45`):

    .text rva=0x0031 DIR32 sym='J' -> section .debug$S value=0xa

and `lift.rs`'s own comment at the refusal site records the same fact from the Rust side: "the
leg's relocation map places CODE sections, so a compare against a static in `.data`/`.bss` has a
target with no address." The lifter's refusal text — "a cell whose relocation names no address" —
is accurate, and it is an address-space decision, not a lifter gap. My first reading of this site
("the cell must be a stack slot, widen the rule to frame slots") was WRONG, and the refusal text
is what corrected it: the census key is the ledger telling you which component owns the work.

Two things the decision must now cover, both measured today:

1. The clang leg (the original ask): 12 records, `6 lea of an address we cannot name` + `6 ...
   naming .rdata`. A relocation-directory walk (PE format) settles the clang objects.
2. The MSVC leg (new): an MSVC global reference is a **DIR32 relocation against a symbol whose
   section is `.debug$S`** — the corpus is built `/nologo /c /Gw- /O2 /GS` with no `/Z7`, yet the
   symbol table is CodeView-shaped, so naming an MSVC data symbol is a **COFF symbol-table +
   string-table walk, not a relocation-directory walk**. An answer scoped to "the PE `.reloc`
   directory" closes the clang records and leaves all 3 MSVC records standing. Say the scope out
   loud when making the call.

Inspection tool added with this row: `scripts/coff-relocs.py` (`--at LO HI`, `--summary`). It
exists because **`objdump -r` prints NOTHING for an MSVC x64 object** — binutils refuses the file
(`objdump -d` on it is empty too), so until today there was no way to read an MSVC object's
relocations from this repo, and the 3 records could not be inspected at all. It prints target
sections and marks CodeView relocations, because CodeView's own relocations live in the same
array as the data references — anything that names an address has to say which target sections it
accepts, and that filter is part of the decision.

Nothing in the clang corpus floors moves from this; the 6 MSVC records and the 12 clang records
stay in the census, loud, until the address space is chosen.