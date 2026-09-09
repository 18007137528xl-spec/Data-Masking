#!/usr/bin/env bash
# Install deidkit and run a self-check against synthetic data.
#
# The Linux/macOS counterpart of setup.ps1. No real subject data is involved:
# the synthetic study is entirely fabricated.
#
#   ./setup.sh                 full install, all extras, self-check
#   ./setup.sh --core          core dependencies only
#   ./setup.sh --skip-model    skip the 560 MB spaCy model
#   ./setup.sh --skip-check    install only

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
    step "Skipping the self-check (requested)"
    printf "\n${G}Installation complete.${N}\n\n"; exit 0
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

step "Profiling the drop and drafting a contract"
# --keep-dates and --blind-treatment are the configuration this project
# asked for: dates retained as recorded, treatment names relabelled. Dates
# force tier: lds, which the run output states.
# If a decision sheet already carries decisions, someone has reviewed it, and
# re-profiling would overwrite an afternoon's work that exists in exactly one
# place. So the sheet is left alone and only the draft is refreshed.
FILLED=0
if [ -f out/plan.csv ]; then
    FILLED=$("$VENV_PY" - <<'PY' 2>/dev/null || echo 0
import pandas as pd
try:
    f = pd.read_csv("out/plan.csv", dtype=str, keep_default_na=False,
                    encoding="utf-8-sig")
    print(int((f.get("decision", pd.Series(dtype=str)).astype(str).str.strip() != "").sum()))
except Exception:
    print(0)
PY
)
fi

DECISION_ARGS=(--decisions out/plan.csv)
if [ "${FILLED:-0}" -gt 0 ]; then
    DECISION_ARGS=()
    warn "out/plan.csv already has $FILLED decision(s): leaving it untouched"
    info "Re-profiling would overwrite a review that exists in one place only."
    info "To use it:  .venv/bin/python -m deidkit.cli approve out/plan.csv \\"
    info "              -c contracts/demo.yaml --data out/quarantine/study_demo \\"
    info "              -o contracts/demo.approved.yaml --approved-by you@example.com"
fi

"$VENV_PY" -m deidkit.cli profile out/quarantine/study_demo \
    -o contracts/demo.yaml --review out/steward_review.csv \
    "${DECISION_ARGS[@]}" \
    --keep-dates --blind-treatment 2>&1 | sed 's/^/         /'
ok "contract draft at contracts/demo.yaml"
[ "${FILLED:-0}" -gt 0 ] || ok "decision sheet at out/plan.csv -- one row per column, open it"

step "Running the pipeline -- WITHOUT a steward's approval"
warn "nobody has signed off contracts/demo.yaml, so this runs with --unreviewed"
info "An installer cannot stop and wait for a person, so it takes the escape"
info "hatch. On real data 'deidkit run' REFUSES an unapproved contract, and the"
info "manifest below records \"reviewed\": false. The last section shows the two"
info "commands that do the real round trip -- try them on this synthetic study."
# --unreviewed: nobody has signed off this contract, and on real data the run
# would refuse. The manifest records "reviewed": false so this output cannot be
# mistaken for a published tier. The real path is:
#   deidkit profile <dir> --decisions plan.csv   (fill in the decision column)
#   deidkit approve plan.csv -c <draft> --data <dir> -o approved.yaml
"$VENV_PY" -m deidkit.cli run out/quarantine/study_demo \
    -c contracts/demo.yaml -o out/tier_deidentified --unreviewed \
    --vault out/vault/demo.db --operator "${USER:-unknown}" --format csv 2>&1 |
    sed 's/^/         /'
ok "published tier at out/tier_deidentified"

# ----------------------------------------------------------------------
step "Checking the published tier against the design guarantees"
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
printf "  Published tier     : out/tier_deidentified\n"
printf "  Manifest           : out/tier_deidentified/manifest.json\n"
printf "  Review queue       : out/tier_deidentified/review_queue.csv\n"
printf "  Decision sheet     : out/plan.csv\n"
printf "  ${Y}Reviewed           : NO -- this ran with --unreviewed${N}\n\n"
printf "  ${D}The tier above is a demonstration, not a publishable output: no${N}\n"
printf "  ${D}steward approved the rules it used. To do it properly:${N}\n\n"
printf "  1. open ${C}out/plan.csv${N} -- one row per column, lowest confidence first\n"
printf "     put ${C}OK${N} in the decision column to accept a proposal, or\n"
printf "     ${C}CHANGE${N} plus a decision_treatment to overrule it.\n"
printf "     A blank row blocks the run. That is the whole point.\n\n"
printf "  2. sign it off, and run again against the approved contract:\n\n"
printf "     ${D}%s${N}\n" '.venv/bin/python -m deidkit.cli approve out/plan.csv \'
printf "     ${D}%s${N}\n" '    -c contracts/demo.yaml --data out/quarantine/study_demo \'
printf "     ${D}    -o contracts/demo.approved.yaml --approved-by you@example.com${N}\n\n"
printf "     ${D}%s${N}\n" '.venv/bin/python -m deidkit.cli run out/quarantine/study_demo \'
printf "     ${D}%s${N}\n" '    -c contracts/demo.approved.yaml -o out/tier_reviewed \'
printf "     ${D}    --vault out/vault/demo.db --format csv${N}\n\n"
printf "  ${D}The second command has no --unreviewed, and it will only work${N}\n"
printf "  ${D}because of the first.${N}\n\n"
printf "  ${D}In a new shell: source .venv/bin/activate  (then: deidkit --help)${N}\n\n"
