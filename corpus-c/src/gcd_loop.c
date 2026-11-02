/* rdec corpus: loop with modulo, no induction variable (not a for) */
int f(int a, int b) { while (b != 0) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }
