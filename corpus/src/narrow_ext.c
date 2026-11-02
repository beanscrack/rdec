/* rdec corpus: movzx/movsx: signedness of sub-word loads */
int f(char a, unsigned char b, short c, unsigned short d) { return a + b + c + d; }
