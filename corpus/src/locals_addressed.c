/* rdec corpus: address-taken locals: must stay memory, never promoted to a register var */
int f(int n) { int a = n, b = n + 1; int *p = &a; for (int i = 0; i < n; i++) { *p += i; p = (p == &a) ? &b : &a; } return a + b; }
