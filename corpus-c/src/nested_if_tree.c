/* rdec corpus: deep if/else nesting, else-if chains */
int f(int a, int b, int c) { if (a > 0) { if (b > 0) { if (c > 0) return 1; else return 2; } else if (c > 0) return 3; else return 4; } else { if (b + c > 10) return 5; return 6; } }
