/* rdec corpus: add/sub/imul widths; the bread and butter of expression recovery */
int f(int a, int b) { int t = a * 3 + b; return (t ^ a) - (b >> 2); }
