/* rdec corpus: branchless select: cmov must become a ternary, not an if */
int f(int a, int b) { int m = a > b ? a : b; int n = a < b ? a : b; return m * 2 - n; }
