/* rdec cpp corpus: struct with methods: member codes QEAA/QEA, thiscall, the `@@` scope form */
struct Point { int x, y; int sum(); void shift(int dx, int dy); };
int Point::sum() { return x + y; }
void Point::shift(int dx, int dy) { x += dx; y += dy; }
