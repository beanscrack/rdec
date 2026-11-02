/* rdec cpp corpus: x64 pointer/reference prefixes (PEA, PEB, QEA, PEC, AEA, AEB), volatile, ** */
struct P2 { int a; int b; };
int byval(P2 p) { return p.a + p.b; }
int one_ptr(int *p, const int *q, int *const r, volatile int *v) { return *p + *q + *r + *v; }
int two_ptr(int **pp, char *s, const char *c, char *const *cc) { return **pp + *s + *c + **cc; }
int refs(int &a, const int &b, long long &c, P2 &p, const P2 &q) { return a + b + (int)c + p.a + q.b; }
int ptr64fn(void *a, const void *b) { return (int)(long long)a + (int)(long long)b; }
