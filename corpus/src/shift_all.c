/* rdec corpus: shl/shr/sar over signed+unsigned, incl. counts >= width */
long f(long a, unsigned n) {long x = a << (n & 63u);long y = (unsigned long)a >> (n & 63u);long z = a >> (n & 63u);return x + y + z; }
