/* rdec corpus: unsigned jcc: below/above-or-equal (the CF side) */
unsigned f(unsigned a, unsigned b) { unsigned r = 0; if (a < b) r += 1; if (a >= b) r += 2; if (a > b) r += 4; return r; }
