# IR semantics

Roadmap §1.3's deliverable. This file defines what the IR *means*;
`crates/core/src/interp.rs` is this document written in Rust, and it is the
reference that translation validation (`tv.rs`), the T0 fixtures
(`tests/m0_gate.rs`) and later T1R/P1 compare against.

Consequences of that arrangement, stated up front:

- If the interpreter and this file disagree, **one of them is a bug**. Fix the
  one that is wrong, then re-run `cargo test -p rdec-core`.
- `interp.rs` shares no code with `passes.rs`. A pass and the reference that
  checks it must not fail the same way for the same reason.
- Every rule here is executable. Anything not expressible in the interpreter is
  not a semantic rule of this IR; it is a note, and it is marked as one.

## 1. Values and types

| Type | Value in the interpreter | Notes |
|---|---|---|
| `Int(w, sx)` | `Val::Int(u64)`, the low `w` bits significant | The *signedness lives in the type*, never in the value. An `i32` is 32 significant bits in a `u64` container; every op truncates to its operand width. |
| `Bool` | `Val::Bl(bool)` | Produced by `Cmp`, consumed by `Branch`/`Select`. |
| `Float` | `Val::Fl(f64)` | `F32`/`F64` bit patterns round-trip through `Cast(Bitcast)`. |
| `Flags` | `Val::Fg(FlagBits)` | Opaque, block-local, see §4. |
| `Ptr` | address bits as `Int` | Memory is byte-addressed, see §5. |

A `Val` carries no type of its own: the type comes from the defining
`Instr::ty` via `Function::ty_of`. That is why the verifier's "op type !=
recorded type" check is load-bearing: without it, the interpreter would be
reading a type the IR never declared.

## 2. Ops

`op(v1..vn)` is the notation; `w` is the operand width; every result is
truncated to the result type's width.

- `Const(c)` / `ConstF(c)` — the literal bits. (Open, M3: v1 types every
  `Const` as `u64`, so narrow constants are never reused and a `Cmp` can mix
  widths. The verifier now rejects the mixed-width `Cmp` instead of trusting the
  constant's type.)
- `Param` — the entry-block formal; `run(f, inputs)` supplies positional args.
- `Bin(Add|Sub|Mul|And|Or|Xor|Shl|Shr|Sar|Div|Rem|MulHi|MulSmul)` —
  arithmetic at the operand width. `MulHi`/`MulSmul` return the high half of the
  full-width product (never a shift by 64 at narrow widths — that was a real
  bug, `const_fold_mulhi_does_not_shift_by_64_at_narrow_widths`).
  `Div`/`Rem` by zero → `Fault::DivZero`; `INT_MIN / -1` →
  `Fault::IntMinDivMinusOne` (both trap on x86, so the interpreter traps too).
- **Shift-count masking is hardware, not taste**: `count_for(w, c)` masks to 6
  bits for 64-bit operands and 5 bits for 8/16/32-bit. T1 caught v1 masking
  64-bit counts to 5 bits, modelling `shl rax, 36` as `shl rax, 4`.
- `Cmp(pred, x, y)` → `Bool`. Signed predicates read the operands as signed,
  unsigned predicates as unsigned; the signedness comes from the operand *type*,
  which is why `SF ⊕ OF` needs the type and not just the bits.
- `Cast(ZExt|SExt|Trunc|Bitcast|FloatToInt|IntToFloat|...)` — the width rules
  are the C rules with the signedness of the *source* type.
- `Select(c, x, y)` — `c` must be `Bool`; arms share one type (verifier: hard).
- `Copy(v)` — identity. Emitted by fusion, removed by copy propagation. It is a
  real op so passes can rewrite a root in place without renumbering.
- `Undef` — a value that is never read in valid IR; the interpreter returns the
  all-zero bits *and the verifier is what keeps it out of reach*.
- `Opaque(reason)` — see §6.

## 3. Control flow

`Jump`, `Branch{cond, then, else}`, `Switch{idx, targets, default}`,
`Return(v)`, `Exit`, `Unreachable`.

Phi semantics are per *edge*: the incoming value is chosen by the predecessor
block, so the interpreter carries `prev_bb` into each block. A pass that
rewrites blocks without updating phi incoming lists produces results the
interpreter will disagree with, and TV reports it.

`Return(NO_VAL)` is `void`. Falling off the end (`Unreachable`) is
`Outcome::NoReturn`.

## 4. Flags

`Flags(FlagsOp)` produces a `Flags` value: one instruction's flag definition.
`GetFlag(flag, fv)` is its only reader.

- `FlagBits` stores `Option<bool>` per flag. `None` means *architecturally
  undefined* (SDM), not "false", not "unknown-but-there".
- Reading a defined flag yields `Bool`.
- Reading an undefined flag is a defined event: the interpreter records
  `Effect::UndefFlagRead(flag)` **and** faults (`Fault::UndefFlag`). The effect
  makes it visible to TV on both sides; the fault makes it impossible for a pass
  to smuggle a value through an undefined bit.
- `Flags` must never reach the emitted C. `fuse::flag_elimination` lowers every
  (FlagsOp × flag) pair; the verifier rejects a Flags-typed value consumed by
  anything but `GetFlag`.

## 5. Memory

A sparse byte map, little-endian, `Load(ty, addr)` / `Store(ty, addr, v)`.
Unaligned accesses are allowed (x86 allows them; alignment modelling is M4).
A load from a byte with no rule → `Fault::UnmodeledLoad`.

`Exec::mem` is *every byte ever written, with its final value*, in address
order. TV compares it, so a pass that invents a store, drops one, or changes a
stored value is caught even when the return value is unchanged.

## 6. Gaps are values you can see

- `Op::Opaque(reason)` evaluates to zero bits **and** appends
  `Effect::Opaque(reason)`. The function that contains it is evidence level E0:
  decompilation that says "I do not know this" is a correct output; a plausible
  invention is not. Today's users: CF after `inc`/`dec` (no previous-flags
  operand in the IR, §1.2 item 3), OF after a shift by a count other than 1,
  AF after logic/shift/mul.
- `Fault::MissingOperand(v)` is the soundness tripwire. In valid SSA every
  operand has a definition, so this fault means a pass left a dangling use.
  It exists because the first version of the interpreter read `0` for a missing
  operand, which made the sabotage test pass while proving nothing.

## 7. External calls

`Call{callee, args, pure}` records `Effect::Call(name, arg_bits)` and feeds back
`call_result(callee, args)`: an FNV-1a hash of the callee and arguments. That is
deliberately not a C model — it is a *deterministic world*, so a pass cannot
pass TV by making two executions see different return values, and `pure: false`
calls are ordered by the effect list.

## 8. Outcomes, and what "equivalent" means

`Outcome::Ret(Val) | Fault(Fault) | NoReturn | Budget`, with the effects list
and final memory as `Exec`.

> **E1 equivalence (the definition TV checks).** Two functions are equivalent
> over a set of states when, for every state, they produce the same outcome, the
> same ordered effect list, and the same final memory.

The step budget exists so a pass that introduces a loop cannot hang the harness;
a budget on one side and not the other counts as a difference.

## 9. Determinism contract

TV compares two runs, so the interpreter must be a function of its input:
no iteration over a `HashMap` in a way that reaches output, no time, no RNG
(the sampler in `tv.rs` is a fixed-seed xorshift), stable sorts everywhere, and
the pipeline itself is checked (`pipeline_is_deterministic`). A nondeterministic
reference turns every equivalence result into noise, which is worse than having
no reference at all.
