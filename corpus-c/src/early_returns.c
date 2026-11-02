/* rdec corpus: multiple exits: goto-reduction pressure */
int f(int a, int b) { if (a < 0) return -1; if (b == 0) return 0; if (a > 100 && b > 100) return 1; for (int i = 0; i < b; i++) { if (a == i) return i; } return 42; }
