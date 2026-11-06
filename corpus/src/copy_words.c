/* rdec corpus: fixed-count store loop through a caller buffer: 16 words written, 16 returned */
long long f(long long *b) { int n = 16; for (int i = 0; i < n; i++) b[i] = b[i] + (long long)i; return n; }
