/* rdec cpp corpus: virtual methods (UEAA/UEBA), virtual dtor, override chain, multiple inheritance */
struct Base { virtual int value() const; virtual int twice() const; virtual ~Base(); int seed; };
struct Mid : Base { int value() const override; int twice() const override; ~Mid() override; int extra; };
struct Leaf : Mid { int value() const override; int probe() const; int offset; };
int Base::value() const { return seed; }
int Base::twice() const { return value() * 2; }
Base::~Base() {}
int Mid::value() const { return Base::value() + extra; }
int Mid::twice() const { return Base::twice() + 1; }
Mid::~Mid() {}
int Leaf::value() const { return Mid::value() + offset; }
int Leaf::probe() const { return twice(); }
struct Alpha { virtual int get() const; int a; };
struct Beta { virtual int get3() const; int b; };
struct Derived : Alpha, Beta { int get() const override; int get3() const override; int d; };
int Alpha::get() const { return a; }
int Beta::get3() const { return b; }
int Derived::get() const { return d; }
int Derived::get3() const { return d + 1; }
int use_derived(Derived *p) { return p->get() + p->get3(); }
