/* rdec corpus: pointer indexing, [base+index*scale+disp] addressing */
long f(const long *a, int n) { long s = 0; for (int i = 0; i < n; i++) s += a[i] * (i + 1); return s; }
