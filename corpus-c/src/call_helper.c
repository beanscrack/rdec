/* rdec corpus: non-inlined call: call/ret, Win64 arg registers, callee-saved preservation */
__attribute__((noinline)) static int h(int x) { return x * x + 1; } int f(int a, int b) { return h(a) + h(b) + h(a + b); }
