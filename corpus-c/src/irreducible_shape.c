/* rdec corpus: computed-goto-ish shape: two-entry loop, goto fallback must not crash */
int f(int n) { int s = 0, i = 0; goto mid; top: s += i; i++; mid: if (i < n) { if (s < 500) goto top; else goto top; } return s; }
