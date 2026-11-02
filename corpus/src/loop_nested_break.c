/* rdec corpus: nested loops with early exit: multi-level control flow */
int f(int n) { int s = 0; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { if (j > i) break; s += j; } if (s > 1000) break; } return s; }
