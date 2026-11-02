/* rdec cpp corpus: instantiated template function: the loud-gap fixture, must land in Unsupported */
template <typename T> T tmax(T a, T b) { return a > b ? a : b; }
int use_template(int a, int b) { return tmax(a, b) + tmax(a * 2, b * 2); }
