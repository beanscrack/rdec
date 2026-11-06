/* rdec corpus: writes through a caller buffer: b[3] is a scribble the return value cannot see */
long long f(long long *b) { b[0] = b[1] + b[2]; b[3] = b[0] * 2; return b[0]; }
