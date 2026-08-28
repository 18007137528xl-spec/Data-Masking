# Regression test for the installer's native-command handling.
#
# Windows PowerShell 5.1 turns a native program's stderr into an error record,
# and under $ErrorActionPreference = 'Stop' that is fatal. PowerShell 7 does
# not. That difference shipped a broken installer once already: CI ran pwsh 7
# and passed while 5.1 users died on the pytest probe. This test pins the
# behaviour the helper has to provide, under settings harsher than either
# shell's default.

# Exercise Invoke-Native under the harshest settings: stop on errors, and
# treat native non-zero exits as errors too. The helper must never throw, and
# must still report the exit code faithfully.

$root = Split-Path -Parent $PSScriptRoot
$src = Get-Content (Join-Path $root 'setup.ps1') -Raw
$fn = [regex]::Match($src, '(?s)function Invoke-Native \{.*?\n\}').Value
if (-not $fn) { Write-Host 'could not extract Invoke-Native'; exit 1 }
Invoke-Expression $fn

# python3 on Linux/macOS runners, python on Windows.
$py = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } else { 'python' }

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$cases = @(
    @{ n = 'stderr output then exit 1'; a = @('-c', 'import sys; sys.stderr.write("boom"); sys.exit(1)') },
    @{ n = 'traceback (the old pytest probe)'; a = @('-c', 'import nope_not_a_module') },
    @{ n = 'find_spec probe, module missing'; a = @('-c', 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("nope") else 1)') },
    @{ n = 'find_spec probe, module present'; a = @('-c', 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("json") else 1)') },
    @{ n = 'plain success with output'; a = @('-c', 'print("hello")') }
)

$allOk = $true
foreach ($c in $cases) {
    try {
        $r = Invoke-Native $py $c.a
        Write-Host ("  ok     {0,-38} exit={1}  Ok={2}" -f $c.n, $r.ExitCode, $r.Ok)
    } catch {
        Write-Host ("  THREW  {0,-38} {1}" -f $c.n, $_.FullyQualifiedErrorId)
        $allOk = $false
    }
}

Write-Host ''
Write-Host "helper never throws          : $allOk"

$r = Invoke-Native $py @('-c', 'import sys; sys.exit(3)')
Write-Host "exit code propagated (3)     : $($r.ExitCode -eq 3), Ok=$($r.Ok)"

$r = Invoke-Native $py @('-c', 'print("captured")')
Write-Host "stdout captured              : $($r.Text.Trim() -eq 'captured')"

$r = Invoke-Native $py @('-c', 'import sys; sys.stderr.write("to-stderr")')
Write-Host "stderr captured, not fatal   : $($r.Text -match 'to-stderr')"

Write-Host ''
Write-Host "ErrorActionPreference restored: $($ErrorActionPreference -eq 'Stop')"

if (-not $allOk) { exit 1 }
