/* rdec corpus: reducible while-loop, induction variable, loop-carried phi */
int f(int n) { int s = 0; for (int i = 0; i < n; i++) s += i * 3; return s; }
