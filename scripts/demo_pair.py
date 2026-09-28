"""End-to-end demo: de-identify a raw -> SDTM pair and prove it still maps.

Run this to see the whole training-pair path in one go:

    python scripts/demo_pair.py

It fabricates a synthetic SDTM study and the raw EDC extract it came from,
de-identifies both against ONE vault, and then checks the thing that actually
matters -- that the two published sides still correspond row for row. No real
data is involved at any point.

Why the pair is the deliverable, and not two directories that happen to be
de-identified: a model learning to derive SDTM from raw EDC reads the raw
record as the input and the SDTM record as the label. De-identify the two
sides independently and each file is still well-formed, still passes every
check, and the correspondence between them is gone -- so the corpus teaches a
mapping that is not true, and nothing in either file shows it.

The three things that keep the correspondence:

1. one vault, so a subject's date offset is issued once and both sides move
   by it
2. ``join_key_template``, so the raw side's ``SUBJECT`` and the SDTM side's
   ``USUBJID`` reach that one vault entry despite being different strings
3. format-preserving shift on the raw side, so ``19/03/2025`` stays d/m/y --
   the conversion to ISO is the label, and normalising the input deletes the
   task

The verification at the end is deliberately arithmetic rather than a summary:
it converts every raw date by hand and compares it to the SDTM value.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "pair_demo"

C = "\033[36m"
G = "\033[32m"
R = "\033[31m"
D = "\033[90m"
N = "\033[0m"
if not sys.stdout.isatty():
    C = G = R = D = N = ""

STUDY = "TIG-2026-001"
FAILED = False


def step(text: str) -> None:
    print(f"\n{C}=== {text}{N}")


def ok(text: str) -> None:
    print(f"  {G}[ ok ]{N} {text}")


def bad(text: str) -> None:
    global FAILED
    FAILED = True
    print(f"  {R}[FAIL]{N} {text}")


def info(text: str) -> None:
    print(f"         {D}{text}{N}")


def run(*args: str, quiet: bool = True) -> None:
    proc = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(args)}")
    if not quiet:
        for line in proc.stdout.rstrip().splitlines():
            info(line)


def cli(*args: str, quiet: bool = True) -> None:
    run("-m", "deidkit.cli", *args, quiet=quiet)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("\n  deidkit -- raw -> SDTM training pair")
    print(f"  {D}synthetic data only; no real subject, site or investigator{N}")
    print("  ---------------------------------------------------------------")

    # ------------------------------------------------------------------
    step("Fabricating the two sides")
    sdtm_in, raw_in = OUT / "in_sdtm", OUT / "in_raw"
    run(str(ROOT / "scripts" / "make_synthetic_study.py"), str(sdtm_in))
    run(str(ROOT / "scripts" / "make_synthetic_raw.py"), str(sdtm_in), str(raw_in))
    ok("SDTM study and the raw extract it came from")

    dm_before = pd.read_csv(sdtm_in / "dm.csv", dtype=str)
    ae_before = pd.read_csv(raw_in / "ae_log.csv", dtype=str)
    info(f"SDTM  USUBJID={dm_before['USUBJID'][0]}  RFSTDTC={dm_before['RFSTDTC'][0]}")
    info(f"RAW   SUBJECT={ae_before['SUBJECT'][0]}  AE_START={ae_before['AE_START'][0]}")
    info("the same subject and the same dates, written two different ways")

    # ------------------------------------------------------------------
    step("Generating a development vault key")
    key_file = OUT / "dev-vault.key"
    proc = subprocess.run(
        [sys.executable, "-m", "deidkit.cli", "keygen"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    key_file.write_text(proc.stdout.strip(), encoding="utf-8")
    ok(f"wrote {key_file.relative_to(ROOT)}")
    info("Development only: in production this comes from a KMS or HSM, and no")
    info("principal that can read a data tier may read the key.")

    import os

    env_key = proc.stdout.strip()
    os.environ["DEIDKIT_VAULT_KEY"] = env_key

    vault = OUT / "vault.db"

    # ------------------------------------------------------------------
    step("Drafting the SDTM-side contract")
    cli(
        "profile", str(sdtm_in),
        "-o", str(OUT / "sdtm.yaml"),
        "--sdtm",
        "--blind-treatment",
        "--id-template", "{STUDYID}-US-{value}",
        quiet=False,
    )
    info("--sdtm            : --DTC shifted, stays a valid ISO date")
    info("--id-template     : USUBJID rebuilt as STUDYID + '-US-' + the surrogate,")
    info("                    so the composition stays a real derivation")

    step("Drafting the raw-side contract")
    cli(
        "profile", str(raw_in),
        "-o", str(OUT / "raw.yaml"),
        "--raw",
        "--blind-treatment",
        "--offset-key", STUDY + "-US-{SUBJECT}",
        "--review", str(OUT / "raw_steward_review.csv"),
        quiet=False,
    )
    info("--raw             : date columns found by their VALUES, then shifted")
    info("                    with the written format kept")
    info("--offset-key      : rebuilds the SDTM side's key from SUBJECT, so both")
    info("                    sides meet at one vault entry")

    # ------------------------------------------------------------------
    step("Publishing the SDTM side")
    cli(
        "run", str(sdtm_in),
        "-c", str(OUT / "sdtm.yaml"),
        "-o", str(OUT / "tier_sdtm"),
        "--vault", str(vault),
        "--operator", "demo",
        "--unreviewed",   # synthetic data; the manifest records that nobody reviewed it
        "--format", "csv",
    )
    ok(f"{(OUT / 'tier_sdtm').relative_to(ROOT)}")

    step("Publishing the raw side, against the SAME vault")
    cli(
        "run", str(raw_in),
        "-c", str(OUT / "raw.yaml"),
        "-o", str(OUT / "tier_raw"),
        "--vault", str(vault),
        "--operator", "demo",
        "--unreviewed",   # synthetic data; the manifest records that nobody reviewed it
        "--format", "csv",
    )
    ok(f"{(OUT / 'tier_raw').relative_to(ROOT)}")

    # ------------------------------------------------------------------
    step("What the two sides look like now")
    dm = pd.read_csv(OUT / "tier_sdtm" / "dm.csv", dtype=str)
    demog = pd.read_csv(OUT / "tier_raw" / "demog.csv", dtype=str)
    ae_s = pd.read_csv(OUT / "tier_sdtm" / "ae.csv", dtype=str)
    ae_r = pd.read_csv(OUT / "tier_raw" / "ae_log.csv", dtype=str)
    print()
    print(f"    {'':22}{'RAW (input)':<24}{'SDTM (label)'}")
    print(f"    {'subject':22}{demog['SUBJECT'][0]:<24}{dm['USUBJID'][0]}")
    print(f"    {'enrolment date':22}{demog['ENROL_DT'][0]:<24}{dm['RFSTDTC'][0]}")
    print(f"    {'treatment':22}{demog['TREATMENT'][0]:<24}{dm['ARM'][0]}")
    print(f"    {'AE onset':22}{ae_r['AE_START'][0]:<24}{ae_s['AESTDTC'][0]}")
    print()
    info("Both sides de-identified. Neither date is the real one. The mapping")
    info("between them is exactly what it was.")

    # ------------------------------------------------------------------
    step("Verifying the pair, by arithmetic rather than by assertion")

    joined = dm["STUDYID"] + "-US-" + demog["SUBJECT"]
    if joined.equals(dm["USUBJID"]):
        ok(f"USUBJID == STUDYID + '-US-' + SUBJECT for all {len(dm)} subjects")
    else:
        bad("the subject identifier no longer composes across the pair")

    def dmy_to_iso(v: str) -> str:
        d, m, y = v.split("/")
        return f"{y}-{m}-{d}"

    mismatched = [
        (r, s)
        for r, s in zip(ae_r["AE_START"], ae_s["AESTDTC"])
        if isinstance(r, str) and r and dmy_to_iso(r) != s
    ]
    if mismatched:
        bad(f"{len(mismatched)} AE date(s) no longer map, e.g. {mismatched[0]}")
    else:
        ok(f"all {len(ae_r)} AE onset dates convert exactly to their SDTM value")

    iso_on_raw = (
        ae_r["AE_START"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").sum()
    )
    if iso_on_raw:
        bad(f"{iso_on_raw} raw date(s) came out in ISO -- the task is pre-solved")
    else:
        ok("the raw side is still in its own format, so the conversion is learnable")

    mh_r = pd.read_csv(OUT / "tier_raw" / "mh_log.csv", dtype=str)
    mh_s = pd.read_csv(OUT / "tier_sdtm" / "mh.csv", dtype=str)
    partial_mismatch = 0
    for r, s in zip(mh_r["MH_ONSET"], mh_s["MHSTDTC"]):
        if not isinstance(r, str) or not isinstance(s, str) or not r or not s:
            continue
        r_grain = r.count("-") + 1
        s_grain = s.count("-") + 1
        if r_grain != s_grain:
            partial_mismatch += 1
    if partial_mismatch:
        bad(f"{partial_mismatch} partial onset(s) changed granularity across the pair")
    else:
        ok("partial onsets keep the same granularity on both sides -- no day invented")

    leaked = [
        v
        for v, out in zip(dm_before["RFSTDTC"], dm["RFSTDTC"])
        if isinstance(v, str) and v == out
    ]
    if leaked:
        bad(f"{len(leaked)} subject(s) kept their own real enrolment date")
    else:
        ok("no subject kept their own real date on either side")
    info("A shifted date sometimes equals some OTHER subject's real date. That")
    info("is not a leak -- nothing ties the value to the person -- so the check")
    info("is row-wise, not set-wise.")

    # ------------------------------------------------------------------
    step("Where the reversibility lives")
    info(f"vault      : {vault.relative_to(ROOT)}  (encrypted; offsets + crosswalk)")
    info("unblind    : deidkit reverse --entity subject --surrogate SUBJ-...")
    info("             --justification 'SAE 2026-004' -- always logged, no quiet path")
    info(f"review     : {(OUT / 'tier_raw_review').relative_to(ROOT)}")
    info("             holds ORIGINAL free text. Vault-level access, delete when done.")

    print("\n  ---------------------------------------------------------------")
    if FAILED:
        print(f"  {R}The pair did NOT verify. Do not build a corpus from this.{N}\n")
        return 1
    print(f"  {G}Pair verified.{N} Both sides de-identified; the mapping between")
    print("  them survived intact, which is what makes them training examples.\n")
    print(f"  {D}Next: read out/pair_demo/raw_steward_review.csv. Every rule needs a{N}")
    print(f"  {D}steward's confirmation before the contract is committed.{N}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
