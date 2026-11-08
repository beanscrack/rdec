# Locate the MSVC toolset and make `cl` runnable by our generator.
#
# On windows-latest (and in a plain git-bash shell on a dev box) `cl` is NOT on
# PATH: the toolset lives under the VS install and expects a vcvars64 environment.
# vswhere is the locator that exists at a fixed path on every image.
#
# Why a wrapper batch: `rdec corpus --msvc` spawns ONE executable per object, and
# `call vcvars64.bat` is a cmd builtin, not a program - a Rust child process
# cannot call it. The wrapper is `call vcvars64.bat` + `cl %*`: the generator
# invokes `driver <flags> <src> /Fo<obj>`, so the LAST LINE has to be `cl`, not a
# bare `%*` - with a bare `%*` cmd.exe eats cl's own switches and every spawn dies
# with "cl exit 1" (measured on all 128). RDEC_CL points at the wrapper, which is
# then one executable that behaves exactly like `clang`.
#
# Dot-source it:  . .\scripts\msvc-env.ps1 ; Set-RdecMsvcEnv
function Set-RdecMsvcEnv {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "no vswhere at $vswhere - this runner has no Visual Studio installer"
    }
    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) {
        Write-Error "vswhere found no installation with the VC x64 toolset (Component.VC.Tools.x86.x64)"
    }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { Write-Error "no vcvars64 at $vcvars" }

    $tmp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
    $wrap = Join-Path $tmp "rdec-cl.cmd"
    "@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`ncl %*`r`n" | Set-Content -Encoding ascii -NoNewline $wrap

    $env:RDEC_CL = $wrap
    Write-Host "MSVC install: $vsPath"
    Write-Host "RDEC_CL     : $wrap"
    # LOUD: the compiler banner, the same string the manifest records under `# cl:`.
    # A leg that builds with an unrecorded compiler is a leg nobody can reproduce.
    # The wrapper IS cl, so no `cl` argument here: `& $wrap` runs cl with no args,
    # which prints the banner on stderr, a usage line on stdout, and exits 1 (cl's
    # usage path). `Select-Object -First 1` alone grabbed the usage line, not the
    # version - select the line that names the version.
    # The banner is on stderr, and Windows PowerShell 5.1 turns a native
    # command's stderr, taken with `2>&1`, into an ERROR RECORD. Under
    # $ErrorActionPreference = 'Stop' — which is what a CI step runs with —
    # capturing a version banner then aborts the job: measured on windows-latest,
    # run 42472240749, where "Microsoft (R) C/C++ Optimizing Compiler Version
    # 19.51.36260 for x64" arrived as NativeCommandError and the msvc job died 43
    # seconds in, before a single object was compiled. Locally this line never
    # ran, because the generator spawns the driver directly from Rust; the lane's
    # own validation ran it on a shell where the same construct does not throw.
    # Route the probe through cmd so the streams merge BEFORE PowerShell sees
    # them: there is then no error record to throw, and `cl` with no arguments
    # still exits 1, which `cmd /c` reports in $LASTEXITCODE without ending us.
    $banner = (cmd /c "`"$wrap`" 2>&1") | Select-String -Pattern "Version" | Select-Object -First 1
    Write-Host "cl          : $banner"
    if (-not "$banner") { Write-Error "cl printed no banner through $wrap" }
}

Set-RdecMsvcEnv
