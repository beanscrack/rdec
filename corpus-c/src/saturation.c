/* rdec corpus: saturating arithmetic: cmp/cmov/setcc idiom cluster */
int f(int a, int b) { long t = (long)a + (long)b; if (t > 2147483647L) return 2147483647; if (t < -2147483648L) return (-2147483647 - 1); return (int)t; }
