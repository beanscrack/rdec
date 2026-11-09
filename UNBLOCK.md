# What blocks the roadmap, and the decision each item needs

Dated 2026-11-08, measured at `f925c26`. Every count below is a gate's printed line from this
repo, not an estimate. Self-drivable work keeps landing (see ROADMAP §§17-21); what follows is the
set that cannot move without a call from you. Ordered by how much of the census each unlocks.

## 1. Where do data symbols live in the per-object address space?

Closes **12 clang records + 3 MSVC records**, and it is the oldest open scoping call.

- clang leg: `6 opaque: lea of an address we cannot name` + `6 ... naming .rdata`
  (of `156 lifted, 135 emitted, 21 refused`).
- MSVC leg: `3 flag setting against a cell whose relocation names no address`. The first is
  `corpus/msvc/obj/array_walk.O2.obj :: f @ 0x2f`, instruction `cmpl $0x2, (%rip)`, whose
  relocation reads (from the COFF tables, `python scripts/coff-relocs.py <obj> --at 0x20 0x45`):
  `.text rva=0x0031 DIR32 sym='J' -> section .debug$S value=0xa`.

**The decision has a scope you have to state out loud.** Naming the clang targets is a PE
relocation-directory walk. Naming the MSVC targets is a **COFF symbol-table + string-table walk**
(the corpus is built `/nologo /c /Gw- /O2 /GS`, no `/Z7`, and the reference lands on a CodeView-
shaped symbol). An answer scoped to "the PE `.reloc` directory" closes the 12 clang records and
leaves all 3 MSVC records standing. Say: PE walk, COFF walk, or both.
Details: ROADMAP §19. Note `objdump -r` prints NOTHING for an MSVC object — binutils refuses the
file — which is why `scripts/coff-relocs.py` exists.

## 2. Extern-call ABI blessing

Closes **4 clang + 4 MSVC refusals**: `3 tail call to 'ext'` + `1 call to 'ext'` on each leg.
The model refuses a call whose callee the object does not define, because it has no return width,
no clobber set, no argument count. What is needed: which externs are blessed with a prototype, and
under what return model (integer-only? floats through XMM0? noreturn list for `exit`/`abort`?).
This is a policy call about inventing a signature, so the pipeline will not guess it.

## 3. Flag phis as the home for cross-block control flow

Prerequisite, not a fix: the M5 structuring row is a dated contract, `83 goto / 246 labels`
today, and its census is `143 return | 38 unstructured conditional | 32 fall-through | 10
flat-form | 6 join-not-exit | 4 noreturn | 3 tail | 3 switch`. The **38 + 6 = 44** goto records
need flags to exist as SSA values across blocks (SS1.2 item 2), because those regions join on a
flag produced in one predecessor. Needs: a scope call on modelling flags as first-class values
(phis over the flag register, undefined-propagation rules, T1's contract) — it changes what T1
proves, which is why it is not mine to decide alone.

## 4. The stack-cookie probe (your shell, five minutes)

`corpus/src/stack_buffer.c.pending` is still empty: no `__security_cookie` / `__chkstk` site has
ever been measured, so the MSVC `/GS` cookie path is modelled by refusal only. My shell probes of
this are unstable (the cl driver needs a vcvars environment). Run the recipe in
`scripts/msvc-cl.cmd` + `rdec corpus --msvc` from a vcvars64 shell and the `.pending` file gains a
real site.

## 5. 128-bit and float returns under recovered arity

The arity/ABI merge (ROADMAP §§18, 18.1) models returns from the lifted body. 128-bit returns
(RDX:RAX) and the float-return width under a recovered prototype are modelled narrowly, T2-clean,
and named in the rows. Recommendation: **accept as designed** and fund only when a corpus site
needs it. Your call is accept vs fund.

## 6. Coverage holes I am not allowed to close alone (corpus-scope)

Each of these needs a new snippet under `corpus/src`, which moves EVERY baseline (lift counts,
T2 vectors, the types census, the MSVC leg, the C++ leg). Tell me which to fund and I will do the
mechanical re-pinning:

- **A 2-argument template.** The C++ gates cannot see reversed template arguments — the sabotage
  `pick2<unsigned int, int>` for `pick2<int, unsigned int>` keeps `demangle --cpp`, `cpp --corpus`
  and the corpus agreement gate ALL GREEN, because the corpus's only `??$` symbol is single-arg
  (ROADMAP §21.4). The hermetic table catches it; the corpus does not.
- **`rol`/`ror` lift sites.** The rotate emitter and interpreter legs are floored against a
  bit-loop reference and a compiled-C reference (ROADMAP §16), but `lift.rs` has no rotate arm, so
  no corpus function reaches them. Same for `cbw`/`cdqe`: refused by test, zero corpus sites.
- **MSVC frame-aliasing.** The frame rule is pinned by hand-built IR; the MSVC corpus produces
  zero frame-class records, so that leg does not exercise it.

## 7. Open defect, no decision needed (FYI, being worked)

CI is red on one fact: `pass-broke-ssa record: corpus\obj\locals_addressed.O2.obj :: f: after pass
1 'flag_elimination': %50 defined in bb2 is not available on the edge bb6 -> bb7`. This box (clang
21.1.8) does not reproduce it; the runner (20.1.8) does. The per-pass verifier added at `c2123ee`
named the pass on its first run, and a lane is fixing that pass now.

## 8. Hygiene

`git worktree list` carries 27 entries, mostly detached lane trees under `C:/tmp` and
`C:/Users/fortn/.codex/worktrees` from past sessions. Say the word and I prune the ones with no
unmerged commits; I have not touched them.
