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

### M4 — Calls, signatures, UB-free C, T2 · 4–5 wk (v1: 3)

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

### M5 — Control-flow structuring · 4–5 wk

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
