/* rdec cpp corpus: class nested in a class: ?get@Inner@Outer@@ scope chains */
struct Outer { int a; struct Inner { int b; int get() const; int bump(int d); }; Inner in; int use(int v); };
int Outer::Inner::get() const { return b; }
int Outer::Inner::bump(int d) { b += d; return b; }
int Outer::use(int v) { in.bump(v); return in.get() + a; }
