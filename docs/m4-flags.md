# Why 37 flags values survive elimination (root cause, from the IR)

Largest C-emission refusal class: `flags value Sub { a, b, borrow } survived flag
elimination`. The premise I started from - flags merged by a phi at a join - is
**false**, and the verifier said so when I built that fixture:

    fixture is not valid IR: ["bb2: %5 has Flags type from a non-Flags op"]

Nothing in the IR merges a `Ty::Flags` value, and the invariant is deliberate. The
pass I wrote for that shape (`split_flag_phis`) is gone; nothing was committed for
it. The real shape is in the lifted IR, `corpus/obj/array_walk.O0.obj :: f`:

```
bb1(at 0x1c):
    %24 = trunc %23 uint32_t[31]   ; 0x1f
    %27 = load uint32_t[uint32_t] %26   ; 0x1f
    %28 = flags.sub %24, %27   : /*?ty*/int   ; 0x1f
    %29 = getflag ZF, %28   : bool   ; 0x23      <- unused by the branch
    %30 = getflag SF, %28   : bool   ; 0x23
    %31 = getflag OF, %28   : bool   ; 0x23
    %32 = getflag CF, %28   : bool   ; 0x23      <- unused by the branch
    %33 = getflag PF, %28   : bool   ; 0x23      <- unused by the branch
    %34 = xor %30, %31   : bool   ; 0x23
    %35 = not %34   : bool   ; 0x23
    br %35, bb3, bb2                             ; jge
```

The chain is `not(SF ^ OF)` = "signed greater-or-equal", and `fuse` handles that
shape: `lower_tree` has an arm for `Not`, and `exact_flag` answers every
(FlagsOp x flag) pair. The obstruction is the **three flag reads the branch never
uses** - `%29` ZF, `%32` CF, `%33` PF.

Why they are there: the lifter lowers a `jcc` by reading the flag bits the
condition needs *defensively*, and for this one it emits all five. Rule 1 of the
fusion pass is "match the tree at its root; leaves are replaced only as part of the
root's rewrite", so the rewrite covers the SF/OF cone, and `%28` is still read by
three leaves outside it - the flags value is partially consumed, and the pass
correctly refuses to half-rewrite it. `%28` survives, and the emitter refuses.

Why DCE does not clean it up first, which is the part worth writing down:
removing an unused `getflag` is **not** semantics-preserving in this IR. The
interpreter evaluates every instruction it reaches, and reading a flag the model
calls undefined is a fault (`Fault::UndefFlag`). So deleting `%29` can turn a
faulting execution into a returning one, and the TV harness would (correctly)
report the pass as changing meaning. DCE leaves flag reads alone, the flags value
stays partially consumed, and the refusal is real.

## The fix, in the order that keeps the oracle valid

1. Fold **per flags value, not per root cone**: when a flags value is consumed by
   flag reads at all, rewrite *every* read of it in one step, so nothing is left
   pointing at the `Flags` op. `lower_tree`/`exact_flag` already answer every
   (FlagsOp x flag) pair, so this is a change in *which* consumers get rewritten,
   not in the lowering table.
2. Preserve the definedness story while doing it. The interpreter faults on an
   undefined flag read, so each rewritten read must keep that: for a FlagsOp whose
   model says a bit is defined (all five for `sub`), the lowered expression is
   equivalent and no fault existed to preserve; for a bit the model calls
   undefined, the rewrite must keep a faulting read - not silently compute a bit
   the hardware leaves undefined. T1 is the oracle for which bits are defined per
   operation; `rdec flags` covers 9.7B probes.
3. Gate: the emission refusals for this class are the measure (37 today, of 87
   refusals). The T2 differential is the correctness check once C runs, and TV on
   this pass is the check before that.

Note on process: this is the second time in this milestone that a plausible
mechanism (flags phi) was killed by a gate rather than by an argument - the
verifier rejected the fixture, and the tests I wrote for the pass failed on the
premise, not the code. The pass and its tests were deleted, not left in the tree.
