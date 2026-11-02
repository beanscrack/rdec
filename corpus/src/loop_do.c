/* rdec corpus: do/while (bottom-tested) vs while (top-tested) */
int f(int n) { int s = 0; do { s ^= n; n >>= 1; } while (n != 0); return s; }
