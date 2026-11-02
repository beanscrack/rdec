/* rdec cpp corpus: const overload pair (A vs B this-adjust) and reference-returning members */
class Buffer { int n_; public: int size() const; int &slot(int i); const int &slot(int i) const; };
int Buffer::size() const { return n_; }
int &Buffer::slot(int i) { (void)i; return n_; }
const int &Buffer::slot(int i) const { (void)i; return n_; }
