/* rdec corpus: div on unsigned: zero-extension of rdx */
unsigned f(unsigned a, unsigned b) { return a / b - a % b; }
