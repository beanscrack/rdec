/* rdec corpus: and/or/xor/test: ZF/SF set, CF/OF cleared, PF */
int f(int a, int b) { int r = 0; if ((a & b) == 0) r |= 1; if ((a | b) < 0) r |= 2; if ((a ^ b) == 0) r |= 4; return r; }
