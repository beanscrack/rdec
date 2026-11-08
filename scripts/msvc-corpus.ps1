# The MSVC leg of the corpus matrix: generate, then generate AGAIN and compare
# manifests. Same-toolchain determinism, never cross-compiler byte equality.
#
# Why twice into the SAME directory: cl embeds the `.obj` path in `.debug$S`
# (`S_OBJNAME`) and in the FILE symbol's aux record, so building into a *different*
# directory legitimately moves bytes inside `.debug$S`. The question this leg can
# answer is "does cl produce the same corpus from the same inputs twice", and that
# is exactly two runs into one directory. Comparing cl objects against clang
# objects would test the vendors, not rdec - the runner's clang 20.1.8 vs a local
# 22.1.8 already moved a gate from 1364 to 1412 comparisons.
#
# Normalized fields (see crates/cli/src/corpus.rs): the COFF TimeDateStamp at
# object offset 4, and the 8-byte checksum at `.chks64` + 8, which hashes the
# stamp. The rewrite lands on disk, so a byte diff of the two runs is EMPTY.
param(
    [string]$Src = "corpus/src",
    [string]$Out = "corpus/msvc"
)
$ErrorActionPreference = "Stop"
. $PSScriptRoot\msvc-env.ps1

$td = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
Write-Host "rdec exe: $td\debug\rdec.exe"

function Invoke-MsvcCorpus([string]$tag) {
    Write-Host "--- corpus --msvc ($tag) ---"
    & cargo run -q -p rdec-cli -- corpus --msvc --src $Src --out $Out
    if ($LASTEXITCODE -ne 0) { Write-Error "corpus --msvc ($tag) exited $LASTEXITCODE" }
}

Invoke-MsvcCorpus "A"
$manifest = Join-Path $Out "manifest.tsv"
$a = Get-Content $manifest -Raw
$hashA = (Get-FileHash $manifest -Algorithm SHA256).Hash
Invoke-MsvcCorpus "B"
$hashB = (Get-FileHash $manifest -Algorithm SHA256).Hash

$objects = Get-ChildItem -Path (Join-Path $Out "obj") -Filter *.obj -File
$bytes = ($objects | Measure-Object -Property Length -Sum).Sum
Write-Host "msvc corpus: $($objects.Count) objects, $bytes bytes, manifest $manifest"
Write-Host "manifest sha256 leg A: $hashA"
Write-Host "manifest sha256 leg B: $hashB"
if ($hashA -ne $hashB) {
    Write-Host "--- leg A manifest ---"
    Write-Host $a
    Write-Host "--- leg B manifest (now on disk) ---"
    Write-Host (Get-Content $manifest -Raw)
    Write-Error "MSVC corpus is NOT deterministic across two builds: manifest hashes differ"
}
Write-Host "MSVC corpus determinism OK: two cl builds, identical manifest ($hashA)"
