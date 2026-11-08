# The MSVC leg's gate battery, with the toolchain paths a runner does not set.
#
# `--limit N` is the push-CI form (bounded llvm-objdump spawns); `--full` is the
# nightly form (every symbol). The battery itself lives in
# `crates/cli/src/corpus.rs::cmd_msvc_gates`; this script only supplies cl (for
# the banner the gates print) and llvm-objdump, and forwards the exit code - a
# wrapper that swallows the code is a gate that always passes.
param(
    [string]$CorpusDir = "corpus/msvc",
    [int]$Limit = 0,
    [switch]$Full
)
$ErrorActionPreference = "Stop"
. $PSScriptRoot\msvc-env.ps1

# windows-latest ships LLVM under Program Files, and the runner's PATH does not
# always carry it. objdump-check refuses to run without objdump rather than
# passing on nothing, so make the tool findable and let it be loud if not.
$llvm = Join-Path $env:ProgramFiles "LLVM\bin"
if (Test-Path (Join-Path $llvm "llvm-objdump.exe")) {
    $env:PATH = "$llvm;$env:PATH"
    Write-Host "llvm-objdump: $((Get-Command llvm-objdump.exe).Source)"
} else {
    Write-Host "llvm-objdump: not under $llvm - relying on PATH (objdump-check fails loudly if absent)"
}

$argv = @("run", "-q", "-p", "rdec-cli", "--", "msvc-gates", "--corpus-dir", $CorpusDir)
if ($Full) { $argv += "--full" } elseif ($Limit -gt 0) { $argv += @("--limit", "$Limit") }
& cargo @argv
$rc = $LASTEXITCODE
Write-Host "msvc-gates exit code: $rc"
exit $rc
