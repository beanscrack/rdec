/* rdec corpus: signed jcc: jl/jle/jg/jge flag fusion */
int f(int a, int b) { int r = 0; if (a < b) r += 1; if (a <= b) r += 2; if (a > b) r += 4; if (a >= b) r += 8; return r; }
