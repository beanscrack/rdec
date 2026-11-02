/* rdec corpus: struct field offsets: the raw material for struct recovery */
struct S { int a; char b; long c; short d; }; int f(struct S *s) { s->a += 1; s->c = s->a * 2; return s->a + s->b + (int)s->c + s->d; }
