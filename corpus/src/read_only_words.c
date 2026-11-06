/* rdec corpus: read-mostly over a caller buffer: MUST NOT write; a spurious store is a miscompile */
long long f(const long long *b) { long long s = b[0] + b[1] * 2 + b[2] * 3 + b[3] * 4; return s; }
