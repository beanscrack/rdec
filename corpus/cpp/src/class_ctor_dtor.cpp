/* rdec cpp corpus: ctor ??0 / dtor ??1 with the @ return marker, private data, const method */
class Circle { double r; public: Circle(double radius); ~Circle(); double radius() const; double grow(double by); };
Circle::Circle(double radius) : r(radius) {}
Circle::~Circle() {}
double Circle::radius() const { return r; }
double Circle::grow(double by) { r += by; return r; }
