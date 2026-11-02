/* rdec cpp corpus: overloaded free functions: same name, argument-list-distinguishing encodings */
int add(int a, int b); double add(double a, double b); long add(long a, long b); long long add(long long a, long long b);
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
long add(long a, long b) { return a + b; }
long long add(long long a, long long b) { return a + b; }
