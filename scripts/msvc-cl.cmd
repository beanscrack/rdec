@echo off
rem vcvars64 environment, then invoke cl: RDEC_CL points at THIS file, so
rem the driver is one executable to the generator (`cl` itself needs vcvars).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl %*
