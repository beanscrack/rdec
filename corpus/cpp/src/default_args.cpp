/* rdec cpp corpus: default arguments: mangled away; exercises the @Z multi-arg terminator */
int window(int x, int y = 8, int w = 640, int h = 480);
int window(int x, int y, int w, int h) { return x * 1 + y * 2 + w * 3 + h * 4; }
