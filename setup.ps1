<#
.SYNOPSIS
    Install deidkit on Windows and run a self-check against synthetic data.

.DESCRIPTION
    Double-click setup.bat to run this. It creates a virtual environment,
    installs dependencies, generates a fabricated study, runs the pipeline
    end to end, and verifies the guarantees the design promises.

    No real subject data is involved at any point. The synthetic study is
    entirely fabricated.

.PARAMETER Core
    Install only core dependencies, skipping Presidio, the SAS readers,
    Faker and Parquet. Faster; free-text detection falls back to the
    built-in pattern detector.

.PARAMETER SkipModel
    Skip the spaCy language model download (~560 MB). Without it Presidio
    cannot start and the pipeline silently falls back to the pattern
    detector -- the self-check reports which one is actually active.

.PARAMETER SkipSelfCheck
    Install only; do not generate synthetic data or run the pipeline.
#>

[CmdletBinding()]
param(
    [switch]$Core,
    [switch]$SkipModel,
    [switch]$SkipSelfCheck
)

$ErrorActionPreference = 'Stop'
$script:failed = $false

# ----------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------
function Write-Step   ($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Write-Ok     ($m) { Write-Host "  [ ok ] $m" -ForegroundColor Green }
function Write-Warn   ($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Write-Fail   ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; $script:failed = $true }
function Write-Info   ($m) { Write-Host "         $m" -ForegroundColor DarkGray }

# ----------------------------------------------------------------------
# running external programs
# ----------------------------------------------------------------------
function Invoke-Native {
    <#
    .SYNOPSIS
        Run an external program, returning its combined output and exit code.

    .DESCRIPTION
        Windows PowerShell 5.1 turns anything a native program writes to
        stderr into an error record, and under $ErrorActionPreference = 'Stop'
        that record is fatal. So a perfectly ordinary probe -- asking python
        whether a package imports, and expecting it to fail -- kills the whole
        script with a NativeCommandError instead of returning a non-zero exit
        code to be handled.

        PowerShell 7 does not behave this way, which is exactly how this
        shipped: CI runs pwsh 7, users run 5.1, and the difference is
        invisible until someone without pytest installed runs the installer.

        Every caller here checks the exit code and decides what it means, so
        stderr must stay informational. This helper drops the preference to
        'Continue' for the duration of the call and restores it after.
    #>
    param(
        [Parameter(Mandatory)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Exe @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    [pscustomobject]@{
        Output   = ($output | ForEach-Object { "$_" })
        Text     = (($output | ForEach-Object { "$_" }) -join "`n")
        ExitCode = $code
        Ok       = ($code -eq 0)
    }
}

Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "  deidkit setup" -ForegroundColor White
Write-Host "  de-identification pipeline for inbound EDC data" -ForegroundColor DarkGray
Write-Host "  ---------------------------------------------------------------"

# ----------------------------------------------------------------------
# 1. Python
# ----------------------------------------------------------------------
Write-Step "Checking Python"

$python = $null
foreach ($candidate in @('python', 'python3', 'py')) {
    try {
        $r = Invoke-Native $candidate @('-c', "import sys; print('%d.%d' % sys.version_info[:2])")
        $probe = $r.Text.Trim()
        if ($r.Ok -and $probe -match '^\d+\.\d+$') {
            $parts = $probe.Split('.')
            if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) {
                $python = $candidate
                Write-Ok "found $candidate (Python $probe)"
                break
            }
            Write-Warn "$candidate is Python $probe; 3.10 or newer is required"
        }
    } catch { }
}

if (-not $python) {
    Write-Fail "no suitable Python found"
    Write-Info "Install Python 3.10+ from https://www.python.org/downloads/windows/"
    Write-Info "During installation, tick 'Add Python to PATH' -- this is the"
    Write-Info "step most often missed, and without it this script cannot find it."
    exit 1
}

# ----------------------------------------------------------------------
# 2. virtual environment
# ----------------------------------------------------------------------
Write-Step "Creating the virtual environment"

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (Test-Path $venvPython) {
    Write-Ok "reusing the existing .venv"
} else {
    $r = Invoke-Native $python @('-m','venv','.venv')
    if (-not (Test-Path $venvPython)) {
        Write-Fail "could not create .venv"
        $r.Output | ForEach-Object { Write-Info $_ }
        exit 1
    }
    Write-Ok "created .venv"
}

# Call the venv's python directly rather than activating. Activation needs a
# permissive execution policy and only affects the current shell; invoking the
# interpreter by path works regardless and leaves no state behind.
Write-Info "using $venvPython"

# ----------------------------------------------------------------------
# 3. dependencies
# ----------------------------------------------------------------------
Write-Step "Installing dependencies"

$r = Invoke-Native $venvPython @('-m','pip','install','--upgrade','pip','--quiet')
if (-not $r.Ok) { Write-Warn "could not upgrade pip; continuing" }

$target = if ($Core) { '-e', '.' } else { '-e', '.[all]' }
Write-Info ("pip install " + ($target -join ' '))

$r = Invoke-Native $venvPython (@('-m','pip','install') + $target + @('--quiet'))
if (-not $r.Ok) {
    $r.Output | ForEach-Object { Write-Info $_ }
    Write-Fail "dependency installation failed"
    Write-Info "If this is a network or proxy problem, set HTTP_PROXY and"
    Write-Info "HTTPS_PROXY, or add your internal index with --index-url."
    exit 1
}
Write-Ok ("installed " + $(if ($Core) { "core dependencies" } else { "all optional extras" }))

# ----------------------------------------------------------------------
# 4. spaCy model for Presidio
# ----------------------------------------------------------------------
if (-not $Core -and -not $SkipModel) {
    Write-Step "Downloading the spaCy language model"
    Write-Info "about 560 MB; needed for Presidio's person-name detection"

    $r = Invoke-Native $venvPython @('-m','spacy','download','en_core_web_lg')
    if ($r.Ok) {
        Write-Ok "model installed"
    } else {
        Write-Warn "model download failed -- the pipeline will fall back to the"
        Write-Info "built-in pattern detector. It is solid on emails, phones and"
        Write-Info "facility names, weaker on bare person names. Re-run later with:"
        Write-Info "  .venv\Scripts\python.exe -m spacy download en_core_web_lg"
    }
} elseif ($SkipModel) {
    Write-Step "Skipping the spaCy model (requested)"
}

# ----------------------------------------------------------------------
# 5. which detector is actually live
# ----------------------------------------------------------------------
Write-Step "Verifying the free-text detector"

$r = Invoke-Native $venvPython @('-c', 'from deidkit.freetext import build_detector; print(build_detector().name)')
if (-not $r.Ok) {
    Write-Fail "deidkit did not import; the installation is broken"
    $r.Output | ForEach-Object { Write-Info $_ }
    exit 1
}
$detector = $r.Text.Trim()
if ($detector -eq 'presidio') {
    Write-Ok "Presidio NER is active"
} else {
    Write-Warn "the built-in pattern detector is active, not Presidio"
    Write-Info "Worth knowing rather than assuming: the fallback is deliberate so"
    Write-Info "the pipeline runs anywhere, which also means a missing model looks"
    Write-Info "like success unless you check. This is that check."
}

if ($SkipSelfCheck) {
    Write-Step "Skipping the self-check (requested)"
    Write-Host "`nInstallation complete.`n" -ForegroundColor Green
    exit 0
}

# ----------------------------------------------------------------------
# 6. unit tests
# ----------------------------------------------------------------------
Write-Step "Running the test suite"

# find_spec returns $null rather than raising, so a missing pytest is a
# clean 'no' instead of a traceback on stderr.
$r = Invoke-Native $venvPython @('-c', 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("pytest") else 1)')
if (-not $r.Ok) {
    Write-Warn "pytest is not installed, so the suite was not run"
    Write-Info "A missing test runner is not a test failure, so this is a warning --"
    Write-Info "but it does mean this install is unverified. Install it with:"
    Write-Info "  .venv\Scripts\python.exe -m pip install pytest"
} else {
    $r = Invoke-Native $venvPython @('-m','pytest','tests','-q')
    $testRc = $r.ExitCode
    $r.Output | ForEach-Object { Write-Info $_ }
    if ($testRc -eq 0) {
        Write-Ok "all tests passed"
    } else {
        Write-Fail "tests failed -- do not use this install against real data"
    }
}

# ----------------------------------------------------------------------
# 7. synthetic study
# ----------------------------------------------------------------------
Write-Step "Generating a synthetic study"
Write-Info "entirely fabricated: no real subject, site or investigator"

$r = Invoke-Native $venvPython @('scripts\make_synthetic_study.py','out\quarantine\study_demo')
$r.Output | ForEach-Object { Write-Info $_ }
if (-not $r.Ok) { Write-Fail "could not generate synthetic data"; exit 1 }
Write-Ok "synthetic study written to out\quarantine\study_demo"

# ----------------------------------------------------------------------
# 8. vault key
# ----------------------------------------------------------------------
Write-Step "Generating a development vault key"

$keyFile = Join-Path $PSScriptRoot 'out\dev-vault.key'
if (Test-Path $keyFile) {
    $key = (Get-Content $keyFile -Raw).Trim()
    Write-Ok "reusing the existing development key"
} else {
    $r = Invoke-Native $venvPython @('-m','deidkit.cli','keygen')
    # keygen prints the key on stdout and its warning on stderr; the merged
    # stream means the key is the first line that looks like a Fernet key.
    $key = ($r.Output | Where-Object { $_ -match '^[A-Za-z0-9_=-]{40,}$' } | Select-Object -First 1)
    if (-not $r.Ok -or -not $key) { Write-Fail "keygen produced no key"; exit 1 }
    $key = $key.Trim()
    New-Item -ItemType Directory -Force -Path (Split-Path $keyFile) | Out-Null
    Set-Content -Path $keyFile -Value $key -NoNewline
    Write-Ok "wrote out\dev-vault.key"
}

Write-Warn "This key is on disk beside the vault it opens. That is fine for"
Write-Info "synthetic data and wrong for anything else: in production the key"
Write-Info "comes from a KMS or HSM, and no principal that can read a data tier"
Write-Info "may read the key. out\ is git-ignored."

$env:DEIDKIT_VAULT_KEY = $key

# ----------------------------------------------------------------------
# 9. pipeline end to end
# ----------------------------------------------------------------------
Write-Step "Profiling the drop and drafting a contract"

$r = Invoke-Native $venvPython @('-m','deidkit.cli','profile','out\quarantine\study_demo',
    '-o','contracts\demo.yaml','--review','out\steward_review.csv')
$r.Output | ForEach-Object { Write-Info $_ }
if (-not $r.Ok) { Write-Fail "profile failed"; exit 1 }
Write-Ok "contract draft at contracts\demo.yaml"

Write-Step "Running the pipeline"

$operator = if ($env:USERNAME) { $env:USERNAME } else { 'unknown' }
$r = Invoke-Native $venvPython @('-m','deidkit.cli','run','out\quarantine\study_demo',
    '-c','contracts\demo.yaml','-o','out\tier_deidentified',
    '--vault','out\vault\demo.db','--operator',$operator,'--format','csv')
$r.Output | ForEach-Object { Write-Info $_ }
if (-not $r.Ok) { Write-Fail "pipeline run failed"; exit 1 }
Write-Ok "published tier at out\tier_deidentified"

# ----------------------------------------------------------------------
# 10. assert the guarantees on the real output
# ----------------------------------------------------------------------
Write-Step "Checking the published tier against the design guarantees"

$r = Invoke-Native $venvPython @('scripts\selfcheck.py',
    'out\quarantine\study_demo','out\tier_deidentified')
$checkRc = $r.ExitCode
$r.Output | ForEach-Object {
    if ($_ -like 'FAIL *') { Write-Fail ($_ -replace '^FAIL ', '') }
    elseif ($_ -like 'PASS *') { Write-Ok ($_ -replace '^PASS ', '') }
    else { Write-Info $_ }
}
if ($checkRc -ne 0) { $script:failed = $true }

# ----------------------------------------------------------------------
# done
# ----------------------------------------------------------------------
Write-Host "`n  ---------------------------------------------------------------"
if ($script:failed) {
    Write-Host "  Setup finished WITH FAILURES. Do not run this install against" -ForegroundColor Red
    Write-Host "  real data until they are resolved." -ForegroundColor Red
    exit 1
}

Write-Host "  Setup complete and verified." -ForegroundColor Green
Write-Host ""
Write-Host "  Free-text detector : $detector"
Write-Host "  Published tier     : out\tier_deidentified"
Write-Host "  Manifest           : out\tier_deidentified\manifest.json"
Write-Host "  Review queue       : out\tier_deidentified\review_queue.csv"
Write-Host "  Steward sheet      : out\steward_review.csv"
Write-Host ""
Write-Host "  Next: open out\steward_review.csv. Every rule needs a steward's" -ForegroundColor DarkGray
Write-Host "  confirmation before the contract is committed -- the suggestions" -ForegroundColor DarkGray
Write-Host "  come from naming convention, not from understanding your study." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  To use the tools directly in a new shell:" -ForegroundColor DarkGray
Write-Host "    .venv\Scripts\Activate.ps1        (then: deidkit --help)" -ForegroundColor DarkGray
Write-Host "  or without activating:" -ForegroundColor DarkGray
Write-Host "    .venv\Scripts\python.exe -m deidkit.cli --help" -ForegroundColor DarkGray
Write-Host ""
