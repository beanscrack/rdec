/* rdec corpus: sub-register writes (al/ax): partial-flag merge semantics */
int f(int a) { char c = (char)a; short s = (short)(a >> 8); return (int)c * 3 + (int)s; }
