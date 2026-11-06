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

## Gate ledger — every gate, one line, measured 2026-11-02 (tree `77e363c`)

Written because the roadmap recorded milestones as they were opened and left several of
their gate verdicts unstated. Each row is the command, the verdict from running it, and
what closing means. `grep -ric "noreturn\|tail_call\|jump_table" crates/arch-x86/src/
lift.rs crates/arch-x86/src/discover.rs` returns 0 — that is the evidence for M3's
"not done" items below, not a memory.

| gate | command | verdict | closing condition |
|---|---|---|---|
| M1 step 1 decode/load | `rdec objdump-check --corpus` | **GREEN** — 124/124 boundary agreement | closed |
| M1 step 2 discovery | `rdec discover --corpus` | **RED, by design** — 124/124 functions, coverage 97.88% vs the 99.5% requirement, 126 bytes in 2 runs, 2 indirect branches unclassified | jump-table recovery resolves those 2 sites; the threshold is not moved |
| M2 step 1 lifter | `rdec lift --corpus` | **CLOSED — recorded late.** Criterion was "every lifted function is valid IR"; it printed `FAIL: 40 lifted functions are not valid IR` at step 1, and now prints `141 functions lifted, 141 pass verify_ssa`, exit 0. It became true at `aa0d8e7` (M3 step 1) and the roadmap never said so. | closed |
| M3 step 1 SSA | `rdec lift --corpus` | **GREEN** — 141/141 valid IR; strict interpreter: 0 uses of a value never defined | closed |
| M3 step 2 block splitting | `rdec lift --corpus` | **GREEN** — loops lift; Exits 64 -> 27, clean functions 63 -> 85 | closed |
| M3 noreturn / tail calls / jump-table targets | (no gate exists yet) | **NOT DONE** — zero handling in the lifter or discovery; the roadmap moved these into M3 (`ROADMAP.md:363-367`) and they were never built. A `jmp` to a known entry is still a call-shaped exit; `abort`/`__fastfail` are not noreturn, so CFG recovery is wrong at those. | gate first, then the feature: a fixture per item, red, then green |
| M3 stable IDs | (no gate exists yet) | **NOT DONE** — IR value numbers are per-function and not stable across passes, so cross-pass diffing is not available | needs a stable-id invariant pinned by a test |
| M4 C emitter | `rdec c --corpus` | **RED, by design** — 84 emitted / **57 refused** / 0 invalid after passes; clang -Werror OK. Top classes: 15 call-clobber, 14 `lea` of an unnameable address, 10 `cdq`, 6 `imul`, 4 unstructured conditional branch, 4 `div` | refusals -> 0 with every gap named in between |
| T2 differential, C | `rdec t2 --corpus` | **GREEN as a sample** — runnable 64 of 141, compared 1364, match 1364, mismatch 0, 4 faults (genuine `loop_do` infinite loops). **Proves the returned 64-bit value only.** Memory effects are NOT compared; 77 functions are not in the compared set. | add memory-effect comparison; widen coverage |
| T2 differential, C++ | `rdec t2 --cpp` | **GREEN as a sample** — 27 functions (15 RAX-class + 12 window-class), 648 runs, 648 matches, 0 mismatch, 0 faults, buckets sum to 78. Return value only; the window class compares the return, not memory. | XMM0 class (in progress), more coverage, memory effects |
| M8 step 0 demangler | `rdec demangle --cpp` | **GREEN** — 68/68 supported symbols agree with `llvm-nm`, 1 template unsupported (named) | closed for the corpus; templates open |
| M8 step 1-2 C++ through the pipeline | `rdec cpp --corpus` | **RED by design, improving** — lifted 78, valid 78/78, emitted C 62, refused 16; clang -Werror OK | refusals -> 0 |
| M8 step 3 C++ value-verified | `rdec t2 --cpp` | **GREEN as a sample** (row above) | — |
| M8 remainder | — | **NOT DONE** — packed SIMD (refused by name: `paddd/paddq/psubd/pmuludq/pcmpeqd/shufps`, mask `andps/orps`, GPR `movd/movq`), vtable dispatch and virtual calls. Measured: this clang build names ZERO vftable symbols in `.obj`, so vtable identity needs `.rdata` + REL32 relocation recovery | relocation reader, then vtable identity, then `call [this+k]` |
| M5 control-flow structuring | — | **NOT STARTED** — no if/while/switch recovery: output is goto-style with `(void)&&label` anchors. The roadmap's own estimate is 4-5 weeks. It is also what closes M1 (jump tables) | starts with the 2 jump-table sites |
| M6 readability + type recovery | — | **NOT STARTED** — includes the Ghidra baseline comparison | — |
| M7 whole binary | — | **NOT STARTED** — library signatures, SEH, cold fragments; ELF support | — |
| M9 production | — | **NOT STARTED** — fuzzing at scale, performance budgets, the analyst loop | — |
| P track (proof) | — | **NOT STARTED** — the formal/proof track has no measured section | — |

The honest headline: the decompiler has three closed structural gates (decode, lift,
SSA), a differential that proves returned values on 64/141 C functions and 27/78 C++
functions with zero mismatches, and no structuring pass — so what it emits is correct
where it is correct, and unreadable everywhere.

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
