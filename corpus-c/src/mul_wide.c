/* rdec corpus: mul/imul high half: 128-bit product, rdx as data not flag */
long f(long a, long b) { return (long)(((__int128)a * (__int128)b) >> 64); }
