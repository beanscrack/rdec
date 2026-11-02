# RDec — from-scratch x86-64 → C decompiler in Rust

Goal: a production-grade decompiler. Not a toy, not a Ghidra clone — a decompiler
whose distinguishing property is **provably equivalent output**: every decompiled
function ships with evidence that the generated C behaves identically to the
machine code it came from.

That property is the wedge. Ghidra is ~2M LOC of accumulated heuristics; Hex-Rays
has decades of type-recovery tuning. Neither can produce a machine-checked
equivalence proof for its output. A verification-first architecture can, and it
also happens to be the only architecture that lets *one* person make steady,
non-regressing progress on a problem this size: every pass is gated by an oracle,
so "it looks right" is never the acceptance criterion.

---

## 0. The four things that decide this project

Everything else is bookkeeping. These are the calls that determine whether the
project can reach production at all, so they are made up front, not discovered.

### 0.1 The oracle (test strategy is architecture, not a phase)

A decompiler is a compiler in reverse, and compiler bugs are *silent*. So the
test harness is designed first:

| Tier | Oracle | Catches |
|---|---|---|
| T0 | Unit tests on hand-written IR: every pass, golden IR dumps | pass regressions |
| T1 | **Flag-table tests executed on real hardware**: for each flag-defining op, evaluate the Rust model over all `(a,b)` pairs in a sampled grid and compare against EFLAGS read back with `pushfq` | the #1 source of real decompiler bugs: wrong jcc semantics |
| T2 | **Differential execution**: compile C → `.o` → decompile → recompile the C → run both binaries over N input vectors, compare return value + observable memory/stdout | semantic drift, wrong widths/signedness, flag fusion bugs |
| T3 | `clang -Werror -c` on every emitted file | non-compilable / UB-laden output |
| T4 | IR + SSA verifiers run after *every* pass; pass idempotence (`run(p) == run(p;p)`) | invalid IR leaking downstream |
| T5 | Fuzzing: `cargo-fuzz` on the front-end (random bytes), plus csmith-generated C corpus through T2 | panics, unmodeled-control-flow crashes, exotic codegen |

T1 and T2 are the reason this roadmap is credible. They are built in Phase 1,
before any decompilation exists.

### 0.2 The flag model (the x86 decompiler's core problem)

x86 threads ~6 flag bits through EFLAGS, shared by every arithmetic op, with
asymmetric definitions per op (`inc` leaves CF alone; `and` clears CF/OF;
`sbb`/`adc` chain carries). Naive models produce wrong sign tests on code like
`sub eax,ecx; jl`.

**Decision: revng-style typed flags.** Flag-defining ops produce an SSA value of
type `Ty::Flags`; the only reader is `get_flag : Flags × {CF,PF,AF,ZF,SF,OF} → Bool`.
Consequences, all mechanical:
- dead flags are gone via plain DCE (no liveness analysis),
- a branch condition is a boolean tree over reads of *one* Flags value, so jcc
  fusion is a finite rewrite table (the classic `cmp`/`test` matrix), not a
  dataflow analysis,
- **anything the table misses lowers to an exact width-aware helper expression**
  (`__rdec_add_cf(a,b)`), emitted on demand. Output is therefore *never silently
  wrong*: a missing fusion costs readability, never correctness. This is a
  permanent architectural guarantee, and it is the single most important line in
  this document.

### 0.3 Memory, stack, and the frame

Flat 64-bit address space; `load`/`store` carry an access type. The readability
of the output is decided almost entirely by **local slot promotion**: recognize
the frame (`push rbp; mov rbp,rsp`, `sub rsp,N`, canary, callee-saved saves),
treat `[rbp-off]` slots with provably-static offsets as variables, and run
mem2reg over them with the same versioned-phi machinery used for registers.
Slots that are address-taken, escaped, or dynamically indexed stay real memory —
decompiling them as registers is exactly how decompilers produce plausible junk.
Red zone, `rsp`-relative code without a frame pointer, and dynamic `alloca`/`_chkstk`
probes are named cases, each with a test.

### 0.4 Unmodeled things are loud

Every gap has an explicit representation: `Op::Opaque`, `Terminator::Exit`,
unresolved indirect jumps, `unmodeled` notes attached to the block, and a
per-function **confidence report** (0..1 + list of gaps) printed with the output.
A production decompiler is trusted because it says when it is unsure. Silent
semantic loss is the failure mode that kills credibility and this project.

---

## 1. Architecture

Strict one-way layering; the middle-end must link against no x86 type.

```
rdec-bin     loader: PE/COFF + ELF via goblin. sections, symbols, relocations,
             PLT/IAT/thunk recognition, function discovery, linear sweep.
rdec-x86     decoder glue (iced), per-instruction semantics → IR, calling
             conventions (Win64 + SysV), frame analysis, idiom recognizers.
rdec-core    IR, CFG, dominators, loops, SSA, passes, structuring, C emission.
             Knows nothing about x86.
rdec-cli     disasm / lift / ir / cfg / decompile / verify / diff
rdec-verify  T1 flag-table harness, T2 differential runner, fuzz targets
```

Dependency licenses checked before use: `iced` (MIT), `goblin` (BSD-3), `clang`
(shell, test-only). No LLVM in the product — LLVM is a test-time tool only.

The IR is typed three-address code:

```
Op  = Const | Copy | Bin | Un | Cmp | Flags(FlagsOp) | GetFlag | Cast
    | Load | Store | Select | Phi | Call | Undef | Opaque
Terminator = Jump | Branch | Return | Unreachable | Exit
```

Pipeline: `lift → SSAn (versioned phi) → prune → flag-elim → fold/copyprop/DCE →
strength-reduce/idioms → structure → cgen`. Every pass is
`fn(&mut Function) -> bool` (changed), run to fixpoint. That shape is what makes
a partial pass list safe: no pass depends on a later one cleaning up after it.

---

## 2. Milestones

Each has a **gate**: a command that is run, whose output is the evidence. No
milestone closes on inspection of code. Estimates are focused weeks.

---

### M0 — Harness skeleton · 1 wk
Workspace, CI (`cargo test`, `cargo clippy -D warnings`, `cargo fmt --check`),
golden-file test macro, `rdec-verify` crate, corpus directory layout, the IR
verifier with SSA-form checks, corpus generator script (clang `-O0/-O1/-O2/-Oz`
× a list of C snippets → `.obj`).

**Gate:** `cargo test` green on an empty pipeline; corpus builds; verifier
rejects a hand-broken function (a test proving the verifier itself works).

---

### M1 — Decode and list · 1–2 wk
`goblin` PE/COFF (this host's clang output) and ELF. Recursive-descent function
discovery from symbols + entry, falling back to linear sweep. Byte-exact
`objdump` comparison. iced formatter for Intel listing. Thunk/PLT detection.

**Gate:** for 50 corpus objects, `rdec disasm` matches `objdump -d` on every
instruction (mnemonic + operands normalized), 100%. Zero panics on 10 MB of
random bytes (fuzz target #1).

---

### M2 — IR + flag semantics, verified · 3–4 wk
The IR (types, ops), lifter for the defined subset: `mov movzx movsx lea add sub
imul neg cmp test and or xor not shl shr sar inc dec setcc cmov movzx push pop
nop ret jmp jcc call`, plus `lea` addressing as pure arithmetic. `mul/div/idiv`
with `rdx` pairing. Frame model.
**T1 hardware flag-table harness** for every flag-defining op, all widths,
grid-sampled `(a,b)` + boundary values (`0, 1, -1, INT_MIN, INT_MAX, ±2^k`).

**Gate:** flag tables: 0 mismatches against real EFLAGS, ~10⁸ sampled triples per
op (this is the proof that jcc semantics are right, not an opinion). IR dumps
stable as goldens. Every lifted op is modeled or explicitly `Opaque` — assert
coverage ≥ 99% on the corpus.

---

### M3 — CFG, SSA, core passes · 3–4 wk
CFG recovery, dominators (Cooper-Harvey-Kennedy), dominance frontiers, natural
loops. SSAn via versioned phi at block entries (registers *and* promoted frame
slots through one mechanism) → prune to SSA. Passes: prune, **flag elimination**
(the jcc table + exact helper fallback), const-fold, copy-prop, DCE, branch
simplify, block merge, block dedup, dead-store elim. Idempotence + verifier
after each.

**Gate:** verifier green through fixpoint on the entire corpus. Flag-fusion hit
rate ≥ 95% (measured: `getflag` trees fused / total), with the residue visible as
helpers, never as raw flags in the C.

---

### M4 — Straight-line + if/return → compilable C · 3 wk
Expression DAG inlining (single-use, side-effect-free, dominance-checked),
C precedence/associativity, cast placement, and **sequencing safety**: never
inline an op that reads and writes memory into the same expression (C UB /
unspecified evaluation order). Pointer-vs-integer display rules.
Win64/SysV signature synthesis.

**Gate:** T3 clean (`clang -Werror -c` on all output) for leaf functions with no
loops; T2 differential equivalence 100% over the leaf corpus, ~10³ vectors each.

---

### M5 — Control flow structuring · 4–5 wk
Reducible CFG → `if/then/else`, `while`, `do/while`, sequence; goto-reduction
(Sharir-style) as the fallback; `for` recovery (init hoist + induction inc).
**Switch recovery**: jump-table detection, base/index/bounds extraction through
the IR, dense-vs-sparse decision, `default` inference from the bounds check.
Irreducible graphs keep `goto` + a warning, and are counted.

**Gate:** `goto` count = 0 on every reducible corpus function (measured, printed);
all 30 loop/switch corpus cases T2-equivalent; jump-table corpus ≥ 90% recovered
as `switch`, remainder as an if-chain with a note.

---

### M6 — Readability pass · 4–6 wk
This is what a real user sees, and it is where incumbent decompilers win.
Value-set + signedness inference (kills `0xffffffff` noise), integer width
narrowing, pointer provenance, idiom table (`min/max`, `abs`, `strlen` main
loops, `bswap`, division-by-constant strength reduction, `setcc`+`movzx`),
variable naming from origin (param/loop-induction/constant), typedef + struct
recovery from access offsets, string-literal and global recovery.

**Gate:** scored corpus — emitted C vs original C, per function: signedness match,
width match, operator-count ratio, statement-count ratio. Thresholds set from the
first scored run and ratcheted (a score that is not measured gets worse).

---

### M7 — Whole-binary scale · 6–8 wk
Function discovery without symbols (code/data separation from linear sweep,
`-ffunction-sections`, cold sections), indirect calls (vtable/PLT/switch),
tail calls and JMP-thunks, PIC/relocation-resolved targets, interprocedural
signature + constant propagation, nothrow/noreturn propagation, section/segment
model, `main`-entry walking, DLL import naming.

**Gate:** decompile a real, stripped, `-O2` open-source binary (start with
`tcc`, `sqlite3.c`, `xxd`, `lua`) and produce a compilable artifact; measure
% functions fully modeled, % with `Exit`, % with indirect-call notes.

---

### M8 — C++ and modern codegen · 8–12 wk
Name demangling (Itanium + MSVC), vtable recovery → class + virtual dispatch
display, RTTI, `this` inference, constructors/destructors, inlined-STL damage
control, exceptions (landing pads, LSDA, cleanup blocks as `try/catch`),
SSE/AVX → vector C, `memcpy`/`memcmp` recognition, TLS.

**Gate:** a C++ corpus (small classes, virtual dispatch, one exception path) with
T2 equivalence; demangle + vtable recovery reported per binary.

---

### M9 — Production hardening · ongoing
`cargo-fuzz` at scale (≥ 10⁶ seeds), T2 nightly over csmith/MITLS-generated
corpus, performance budgets (≥ 100 leaf functions/sec, streaming for 10 MB+
`text`), incremental decompilation, plugin/script API (Lua or Rust traits),
JSON output + stable IDs for tooling, IDE/editor integration, docs and a
contributor contract for the fusion/idiom tables.

**Gate:** fuzz crashers = 0 open > 48 h; a public conformance score on a fixed
benchmark suite (the same suite, versioned, every release).

---

## 3. Cumulative honest timeline

M0–M6 ≈ 19–25 focused weeks → *genuinely useful* on x86-64 Windows/Linux ELF for
single functions, verified-equivalent, compilable output. M0–M7 ≈ 8–11 months →
usable on real stripped binaries. C++/SIMD/exceptions parity work is another
year-plus. "Competitor to Hex-Rays" is a multi-year, multi-person claim; the
defensible near-term claim is **"the decompiler that ships equivalence proofs,
on modern codegen, with clean types."** That is a real niche, it is not occupied,
and every milestone above advances it.

## 4. Known killers (planned for, not discovered later)

1. Flag semantics — M2 T1 harness. The classic decompiler graveyard.
2. Silent semantic loss — `Opaque`/`Exit`/confidence report, loud by design.
3. Irreducible CFG (compiler `goto`/computed-goto, obfuscated code) — goto
   fallback, never a crash.
4. Memory aliasing / address-taken locals — promotion is conservative.
5. Frame analysis (`alloca`, `_chkstk`, canary, no-frame-pointer) — named cases.
6. Jump tables — M5, with bounds inference.
7. Type recovery — M6, the readability cliff; measured, not vibed.
8. Sequencing UB in generated C — M4, with an explicit no-inline-on-alias rule.
9. Performance blowups on 10⁵-instruction functions — M9 budgets, per-pass
   timing instrumentation from M3 (cheap to add, expensive to retrofit).
10. Scope explosion without an oracle — every feature ships with a gate.

## 5. State of the tree right now

Verified by running, not by inspection:

| Item | State |
|---|---|
| `crates/core/{ty,ir,cfg,fmt_ir,passes,verify}.rs` | compiles; SSA/IR verifier runs after every pass |
| `crates/arch-x86/src/flags.rs` | EFLAGS model, all widths, defined + undefined cases |
| `crates/verify/src/hw.rs` | inline-asm probes: seed flags via `pushfq`/`popfq`, run one op, read EFLAGS back |
| T1 grid (`cargo test -p rdec-verify`) | **3 passed** — model matches hardware over the boundary grid, counts 0..70, all widths |
| Sabotage test | **passes** — one injected bad OF bit is caught, and reported confined to the injected op/width |
| `rdec flags --full N` | runs the oracle at volume (N random operand pairs per op/width) |
| Toolchain | rustc 1.99 `x86_64-pc-windows-msvc`; clang 22.1.8 targeting MSVC; `clang -O1 -c` emits real COFF — the T2 corpus is buildable on this host |

**What T1 already bought.** The grid found five genuine semantics bugs in the
first run, all of which would have produced silently wrong decompilations later:

1. `add` OF and 2. `sub` OF — Rust precedence (`&` vs `>>`) made the sign tests
   compute garbage. Fixed with an explicit `bit()` helper.
3. `shl` CF at count == width — the last bit out is bit 0, so CF=1, not 0.
4. `inc`/`dec` OF — INT_MAX and INT_MIN were swapped.
5. **Shift-count mask is 6 bits for 64-bit operands, 5 bits otherwise** —
   `shl rax, 36` was being modeled as `shl rax, 4`. This is the kind of detail
   that no amount of reading your own code catches.

M0's remaining items are the corpus generator (clang `-O0/-O1/-O2/-Oz` over C
snippets → `.obj`) and CI wiring. M1 (decode + disassembly) is next.
