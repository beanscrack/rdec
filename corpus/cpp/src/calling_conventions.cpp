/* rdec cpp corpus: __cdecl/__stdcall/__fastcall free functions (x64 unifies them; the @YA letter set must map all) */
__stdcall int winapi_add(int a, int b);
__fastcall int fast_scale(int a, int b);
__cdecl int cdecl_base(int a, int b);
int winapi_add(int a, int b) { return a + b; }
int fast_scale(int a, int b) { return a * b; }
int cdecl_base(int a, int b) { return a - b; }
