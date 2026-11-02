/* rdec cpp corpus: every x64 builtin letter, in return and argument position (C E D F G H I J K _J _K M N _N _W) */
signed char sc(signed char v); unsigned char uc(unsigned char v); char ch(char v); short sh(short v);
unsigned short us(unsigned short v); int in(int v); unsigned int un(unsigned int v); long lo(long v);
unsigned long ul(unsigned long v); long long ll(long long v); unsigned long long ull(unsigned long long v);
float fl(float v); double dbl(double v); bool bl(bool v); wchar_t wc(wchar_t v);
signed char sc(signed char v) { return v; }
unsigned char uc(unsigned char v) { return v; }
char ch(char v) { return v; }
short sh(short v) { return v; }
unsigned short us(unsigned short v) { return v; }
int in(int v) { return v; }
unsigned int un(unsigned int v) { return v; }
long lo(long v) { return v; }
unsigned long ul(unsigned long v) { return v; }
long long ll(long long v) { return v; }
unsigned long long ull(unsigned long long v) { return v; }
float fl(float v) { return v; }
double dbl(double v) { return v; }
bool bl(bool v) { return v; }
wchar_t wc(wchar_t v) { return v; }
int chain(signed char a, unsigned char b, char c, short d, unsigned short e, int f, unsigned int g, long h,
          unsigned long i, long long j, unsigned long long k, float l, double m, bool n, wchar_t o);
int chain(signed char a, unsigned char b, char c, short d, unsigned short e, int f, unsigned int g, long h,
          unsigned long i, long long j, unsigned long long k, float l, double m, bool n, wchar_t o)
{ return (int)(a + b + c + d + e + f + g + h + i + j + k + (long long)l + (long long)m + (int)n + (int)o); }
