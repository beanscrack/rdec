/* rdec cpp corpus: scoped enum (W4 + the ?A wrapper) and unscoped enum in return/argument position */
enum class Kind : int { kA = 1, kB = 2 };
enum Legacy { kLow, kHigh };
int use_kind(Kind k, Legacy l);
Kind pick(Kind a, Kind b);
int classify(int v);
int use_kind(Kind k, Legacy l) { return (int)k + (int)l; }
Kind pick(Kind a, Kind b) { return a > b ? a : b; }
int classify(int v) { return v > 0 ? 1 : 0; }
