/* rdec corpus: idiv: rdx:rax pairing, sign correction, div-by-power-of-2 strength reduction */
int f(int a, int b) { return a / b + a % b; }
