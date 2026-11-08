/* rdec corpus: call to a noreturn callee mid-function: `call abort` on a leg the vectors never take, so CFG must end the block at the call */
__attribute__((noreturn)) void abort(void); long f(long x) { if (x > 100) abort(); return x * 2; }
