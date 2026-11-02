/* rdec corpus: setcc + movzx: boolean materialized into an integer */
int f(int a, int b) { return (a < b) + (a == b) * 2 + (a > b) * 4; }
