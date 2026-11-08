/* rdec corpus: sibling call to a symbol this object does NOT define: `jmp ext` - the refusal bucket, where a prototype would have to be invented */
extern long ext(long); long f(long a) { return ext(a + 1); }
