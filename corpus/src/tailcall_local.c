/* rdec corpus: sibling call to a function this object defines: `jmp helper` + a relocation naming it, the tail-call CFG and the return-with-callee's-answer model */
__attribute__((noinline)) long helper(long a, long b) { return a * 3 + b; } long f(long a, long b) { return helper(a, b); }
