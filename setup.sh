#!/usr/bin/env bash
# Install deidkit and test the install on invented data.
#
# The Linux/macOS counterpart of setup.ps1. The install test runs on a
# fabricated study: no real data is read, nothing is signed. Real drops are
# reviewed and signed off in the console (./serve.sh).
#
#   ./setup.sh                 full install, all extras, install test
#   ./setup.sh --core          core dependencies only
#   ./setup.sh --skip-model    skip the 560 MB spaCy model
#   ./setup.sh --skip-check    install only, no install test

set -euo pipefail

CORE=0
SKIP_MODEL=0
SKIP_CHECK=0
for arg in "$@"; do
    case "$arg" in
        --core) CORE=1 ;;
        --skip-model) SKIP_MODEL=1 ;;
        --skip-check) SKIP_CHECK=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

if [ -t 1 ]; then
    C='\033[36m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; D='\033[90m'; N='\033[0m'
else
    C=''; G=''; Y=''; R=''; D=''; N=''
fi
step() { printf "\n${C}=== %s${N}\n" "$1"; }
ok()   { printf "  ${G}[ ok ]${N} %s\n" "$1"; }
warn() { printf "  ${Y}[warn]${N} %s\n" "$1"; }
fail() { printf "  ${R}[FAIL]${N} %s\n" "$1"; FAILED=1; }
info() { printf "         ${D}%s${N}\n" "$1"; }
FAILED=0

printf "\n  deidkit setup\n"
printf "  ${D}de-identification pipeline for inbound EDC data${N}\n"
printf "  ---------------------------------------------------------------\n"

# ----------------------------------------------------------------------
step "Checking Python"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
        maj=${v%%.*}; min=${v##*.}
        if [ "${maj:-0}" = "3" ] && [ "${min:-0}" -ge 10 ] 2>/dev/null; then
            PY="$c"; ok "found $c (Python $v)"; break
        fi
    fi
done
[ -n "$PY" ] || { fail "no Python 3.10+ found"; exit 1; }

# ----------------------------------------------------------------------
step "Creating the virtual environment"
VENV_PY=".venv/bin/python"
if [ -x "$VENV_PY" ]; then
    ok "reusing the existing .venv"
else
    "$PY" -m venv .venv
    ok "created .venv"
fi
info "using $VENV_PY"

# Invoke the interpreter by path rather than activating: it works the same in
# a non-interactive shell and leaves no state behind.

# ----------------------------------------------------------------------
step "Installing dependencies"
"$VENV_PY" -m pip install --upgrade pip --quiet || warn "could not upgrade pip"
if [ "$CORE" = "1" ]; then TARGET="-e ."; LABEL="core dependencies"
else TARGET="-e .[all]"; LABEL="all optional extras"; fi
info "pip install $TARGET"
# shellcheck disable=SC2086
"$VENV_PY" -m pip install $TARGET --quiet
ok "installed $LABEL"

# ----------------------------------------------------------------------
if [ "$CORE" = "0" ] && [ "$SKIP_MODEL" = "0" ]; then
    step "Downloading the spaCy language model"
    info "about 560 MB; needed for Presidio's person-name detection"
    if "$VENV_PY" -m spacy download en_core_web_lg >/dev/null 2>&1; then
        ok "model installed"
    else
        warn "model download failed -- falling back to the pattern detector"
        info "re-run later: .venv/bin/python -m spacy download en_core_web_lg"
    fi
fi

# ----------------------------------------------------------------------
step "Verifying the free-text detector"
DETECTOR=$("$VENV_PY" -c 'from deidkit.freetext import build_detector; print(build_detector().name)' 2>/dev/null || echo "")
[ -n "$DETECTOR" ] || { fail "deidkit did not import; the install is broken"; exit 1; }
if [ "$DETECTOR" = "presidio" ]; then
    ok "Presidio NER is active"
else
    warn "the built-in pattern detector is active, not Presidio"
    info "The fallback is deliberate so the pipeline runs anywhere, which also"
    info "means a missing model looks like success unless you check. This is that check."
fi

if [ "$SKIP_CHECK" = "1" ]; then
    step "Skipping the install test (requested)"
    printf "\n${G}Installation complete, but untested.${N}\n"
    printf "No development key was made, so set DEIDKIT_KEY_URI (or\n"
    printf "DEIDKIT_VAULT_KEY) before you run ./serve.sh.\n\n"; exit 0
fi

# ----------------------------------------------------------------------
step "Running the test suite"
if ! "$VENV_PY" -c 'import pytest' 2>/dev/null; then
    warn "pytest is not installed, so the suite was not run"
    info "A missing test runner is not a test failure, so this is a warning --"
    info "but it does mean this install is unverified. Install it with:"
    info "  .venv/bin/python -m pip install pytest"
else
    TEST_OUT=$("$VENV_PY" -m pytest tests/ -q 2>&1) && TEST_RC=0 || TEST_RC=$?
    printf '%s\n' "$TEST_OUT" | sed 's/^/         /'
    if [ "$TEST_RC" = "0" ]; then
        ok "all tests passed"
    else
        fail "tests failed -- do not use this install against real data"
    fi
fi

# ----------------------------------------------------------------------
step "Generating a synthetic study"
info "entirely fabricated: no real subject, site or investigator"
"$VENV_PY" scripts/make_synthetic_study.py out/quarantine/study_demo 2>&1 | sed 's/^/         /'
ok "written to out/quarantine/study_demo"

# ----------------------------------------------------------------------
step "Generating a development vault key"
KEYFILE="out/dev-vault.key"
if [ -f "$KEYFILE" ]; then
    ok "reusing the existing development key"
else
    mkdir -p out
    "$VENV_PY" -m deidkit.cli keygen 2>/dev/null > "$KEYFILE"
    chmod 600 "$KEYFILE"
    ok "wrote $KEYFILE"
fi
warn "This key sits on disk beside the vault it opens."
info "Fine for synthetic data, wrong for anything else: in production the key"
info "comes from a KMS or HSM, and no principal that can read a data tier may"
info "read the key. out/ is git-ignored."
DEIDKIT_VAULT_KEY=$(tr -d '\n' < "$KEYFILE")
export DEIDKIT_VAULT_KEY

# ----------------------------------------------------------------------
# Clear the previous run's published output. The virtualenv and the dev key
# are deliberately reused -- they are slow to rebuild and the vault depends on
# the key -- but a stale tier is a trap: the layout has changed before, and a
# leftover file from an older version fails today's checks for a reason that
# has nothing to do with this install.
step "Clearing any previous published output"
rm -rf out/tier_deidentified out/tier_deidentified_review contracts/demo.yaml
ok "removed the previous tier, review directory and draft contract"
info "keeping .venv and out/dev-vault.key: both are reused on purpose"

step "Install test: profiling the invented study"
info "this exercises the pipeline on fabricated data only. Nothing here is"
info "a review: there is no decision sheet to fill in and nothing to sign."
# --keep-dates and --blind-treatment are the configuration this project
# asked for: dates retained as recorded, treatment names relabelled. Dates
# force tier: lds, which the run output states.
# The profiler's own output is written for a steward reviewing real data
# ("the steward MUST review...", "this is a DRAFT"). None of that applies to an
# invented study, so it is shown only if the step fails.
PROFILE_OUT=$("$VENV_PY" -m deidkit.cli profile out/quarantine/study_demo \
    -o contracts/demo.yaml --keep-dates --blind-treatment 2>&1) || {
    printf '%s\n' "$PROFILE_OUT" | sed 's/^/         /'
    fail "profile failed"; exit 1
}
ok "test contract drafted"

step "Install test: running the pipeline on the invented study"
# --unreviewed is the escape hatch for synthetic data and CI: an installer has
# no one to approve anything. On real data 'deidkit run' refuses an unapproved
# contract, and the manifest here records "reviewed": false, so this output can
# never pass for a published tier. Real sign-off happens in the console.
RUN_OUT=$("$VENV_PY" -m deidkit.cli run out/quarantine/study_demo \
    -c contracts/demo.yaml -o out/tier_deidentified --unreviewed \
    --vault out/vault/demo.db --operator "${USER:-unknown}" --format csv 2>&1) || {
    printf '%s\n' "$RUN_OUT" | sed 's/^/         /'
    fail "pipeline run failed"; exit 1
}
ok "test output at out/tier_deidentified"

# ----------------------------------------------------------------------
step "Install test: checking the output against the design guarantees"
CHECK_OUT=$("$VENV_PY" scripts/selfcheck.py \
    out/quarantine/study_demo out/tier_deidentified 2>&1) && CHECK_RC=0 || CHECK_RC=$?
while IFS= read -r line; do
    case "$line" in
        PASS\ *) ok "${line#PASS }" ;;
        FAIL\ *) printf "  ${R}[FAIL]${N} %s\n" "${line#FAIL }" ;;
        *) info "$line" ;;
    esac
done <<< "$CHECK_OUT"
[ "$CHECK_RC" = "0" ] || FAILED=1

# ----------------------------------------------------------------------
printf "\n  ---------------------------------------------------------------\n"
if [ "$FAILED" = "1" ]; then
    printf "  ${R}Setup finished WITH FAILURES.${N} Do not run this install against\n"
    printf "  ${R}real data until they are resolved.${N}\n\n"
    exit 1
fi
printf "  ${G}Setup complete and verified.${N}\n\n"
printf "  Free-text detector : %s\n" "$DETECTOR"
printf "  ${D}Install test       : passed, on invented data (out/tier_deidentified)${N}\n"
printf "  ${D}                     nothing to open or sign there -- it can be deleted${N}\n\n"
printf "  ${C}Ready. To de-identify a real drop, run ./serve.sh in this folder.${N}\n"
printf "  ${C}It starts the console and opens it in your browser; the review${N}\n"
printf "  ${C}and the sign-off both happen there.${N}\n\n"
printf "  ${D}In a new shell: source .venv/bin/activate  (then: deidkit --help)${N}\n\n"
