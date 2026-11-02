# Next step: M2 lifting (x86-64 -> IR), then the first end-to-end decompile

State at ff68a18: M0 closed; M1 step 1 green (objdump boundary gate 124/124);
M1 step 2 discovery 124/124 functions, coverage 97.88%, gate red by design
(2 unclassified indirect jumps = jump tables, M5). 37 tests, fmt/clippy clean.

Plan, in order, each closed by running something:

1. `lift.rs` in crates/arch-x86: decode -> rdec-core IR for a stated subset
   (mov/lea/add/sub/xor/and/or/shl/shr/sar/cmp/test/push/pop/movzx/movsx/call/ret/jcc/jmp).
   Anything outside the subset becomes `Op::Opaque`, never a guess.
   Gate: every lifted function passes `verify_ssa`, and `tv.rs` proves each pass
   preserves the interpreter's semantics.
2. Flags: lift `cmp`/`test` to a `Flags` SSA value and reuse `fuse.rs` - the T1
   hardware table (9.7e9 probes, 0 mismatches) is already the oracle.
3. `emit_c.rs`: IR -> C text. Gate: clang -Werror -c compiles it.
4. T2 differential: original object's function + decompiled C, same inputs,
   compare return values and memory. First target: `arith_basic`, `branchy`,
   `early_returns` in corpus/src, all 4 opt levels.
5. Only then C++ (M8 subset): MSVC demangle, `this` in RCX, member call as a
   plain call, ctor/dtor as functions, struct-by-value ABI. vtables last.

Rules that produced every bug fixed so far: measure, do not assert; a gate that
checks nothing must fail; a number that cannot be true (coverage 114%) is the
best signal available; gaps are loud (`Opaque`, `Exit`, named `Gap`).
