/* rdec corpus: fallthrough edges: CFG shape that structuring gets wrong */
int f(int k) { int r = 0; switch (k) { case 0: r += 1; case 1: r += 2; case 2: r += 4; break; case 3: r += 8; } return r; }
