/* rdec corpus: packed 16-byte block moves through a caller buffer: movups data movement, no packed arithmetic */
long long f(long long *b) { b[0]=b[8]; b[1]=b[9]; b[2]=b[10]; b[3]=b[11]; b[4]=b[12]; b[5]=b[13]; b[6]=b[14]; b[7]=b[15]; return 8; }
