/* rdec corpus: add/adc carry threading through 128-bit add (CF as data) */
void f(unsigned long a, unsigned long b, unsigned long *lo, unsigned long *hi) { unsigned long l = a + b; unsigned long h = (unsigned long)(l < a); *lo = l; *hi = h; }
