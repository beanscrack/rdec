/* rdec cpp corpus: static member functions (SA codes) + a static data symbol to filter out */
class Widget { static int live_; public: static int count(int seed); static int seed_and_count(int seed); int add_and_count(int seed); };
int Widget::live_ = 0;
int Widget::count(int seed) { return live_ + seed; }
int Widget::seed_and_count(int seed) { live_ = seed; return live_; }
int Widget::add_and_count(int seed) { live_ += seed; return live_; }
