"""Un-normalise the synthetic SDTM study into a synthetic raw EDC extract.

This is the *input* side of a raw -> SDTM training pair. It is generated from
the SDTM study rather than independently, because the point of a pair is that
the two sides describe the same events: same subjects, same dates, same
verbatim terms -- differing only in the shape a model has to learn to convert.

What gets un-normalised, and why each one matters to the pair:

* **Column names.** ``AESTDTC`` becomes ``AE_START``, ``MHSTDTC`` becomes
  ``MH_ONSET``. Raw EDC has no suffix convention; a tool that finds dates by
  name finds nothing here.
* **Date formats.** dd/mm/yyyy in one form, mm/dd/yyyy in another, ddMMMyyyy,
  yyyy/mm/dd, ``Mar-2015`` for a partial. Real CRFs differ per form, sometimes
  per site, and the conversion to ISO 8601 is the single most mechanical thing
  a derivation model has to get right.
* **The subject identifier.** Raw holds ``101-0004``; SDTM holds
  ``TIG-2026-001-US-101-0004``. Deriving USUBJID is part of the mapping -- and
  it is also why the offset key has to be reconstructed with a template, or
  the two sides never meet in the vault.
* **Nothing else.** Verbatim terms, severities, doses and lab values are
  carried across unchanged. Where the two sides differ, that difference is the
  label; inventing extra divergence would teach the model noise.

No real subject, site or investigator appears here: everything traces back to
``make_synthetic_study.py``, which fabricates all of it.

    python scripts/make_synthetic_raw.py IN_SDTM_DIR OUT_RAW_DIR
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

SEED = 20260826
_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _parts(value: object) -> tuple[int, int | None, int | None] | None:
    """Split an ISO / partial ISO date without inventing precision."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    bits = text.split("-")
    try:
        y = int(bits[0])
    except ValueError:
        return None
    m = int(bits[1]) if len(bits) > 1 else None
    d = int(bits[2]) if len(bits) > 2 else None
    return y, m, d


def dmy(value: object) -> str:
    """19/03/2025 -- European numeric, the most ambiguous form there is."""
    p = _parts(value)
    if not p or p[1] is None or p[2] is None:
        return ""
    y, m, d = p
    return f"{d:02d}/{m:02d}/{y:04d}"


def mdy(value: object) -> str:
    """03/19/2025 -- US numeric. Indistinguishable from the above by shape."""
    p = _parts(value)
    if not p or p[1] is None or p[2] is None:
        return ""
    y, m, d = p
    return f"{m:02d}/{d:02d}/{y:04d}"


def ddmonyyyy(value: object) -> str:
    """19MAR2025 -- the SAS date form, ubiquitous in clinical data transfer."""
    p = _parts(value)
    if not p or p[1] is None or p[2] is None:
        return ""
    y, m, d = p
    return f"{d:02d}{_MON[m - 1].upper()}{y:04d}"


def dash_mon(value: object) -> str:
    """19-Mar-2025, or Mar-2015 / 2015 when the day or month is not recorded.

    Partial dates have to survive as partial. Medical history onset is
    year-only nearly half the time, and a raw side that quietly imputed a day
    would train the model to invent one.
    """
    p = _parts(value)
    if not p:
        return ""
    y, m, d = p
    if m is None:
        return f"{y:04d}"
    if d is None:
        return f"{_MON[m - 1]}-{y:04d}"
    return f"{d:02d}-{_MON[m - 1]}-{y:04d}"


def slashed_iso(value: object) -> str:
    """2025/03/19 -- ISO order, non-ISO punctuation."""
    p = _parts(value)
    if not p or p[1] is None or p[2] is None:
        return ""
    y, m, d = p
    return f"{y:04d}/{m:02d}/{d:02d}"


def main(sdtm_dir: str, out_dir: str) -> None:
    src = Path(sdtm_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(src / f"{name}.csv", dtype=str, keep_default_na=False)

    dm, ae, mh, cm, ex, vs, lb = (
        read(n) for n in ("dm", "ae", "mh", "cm", "ex", "vs", "lb")
    )

    # SUBJECT is the raw identifier. USUBJID is derived from it on the SDTM
    # side, which is exactly why --offset-key exists.
    written: list[tuple[str, pd.DataFrame]] = []

    demog = pd.DataFrame({
        "SUBJECT": dm["SUBJID"],
        "SITE": dm["SITEID"],
        "INVESTIGATOR": dm["INVNAM"],
        "BIRTH_DT": dm["BRTHDTC"].map(dash_mon),
        "SEX_CD": dm["SEX"],
        "RACE_TXT": dm["RACE"],
        "ETHNIC_TXT": dm["ETHNIC"],
        "COUNTRY_CD": dm["COUNTRY"],
        "SITE_ZIP": dm["SITEZIP"],
        "TREATMENT": dm["ARM"],
        "ENROL_DT": dm["RFSTDTC"].map(dmy),
        "EXIT_DT": dm["RFENDTC"].map(dmy),
        "CONTACT_PHONE": dm["SUBJPHONE"],
    })
    written.append(("demog", demog))

    written.append(("ae_log", pd.DataFrame({
        "SUBJECT": ae["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "AE_NO": ae["AESEQ"],
        "AE_TERM": ae["AETERM"],
        "AE_START": ae["AESTDTC"].map(dmy),
        "AE_STOP": ae["AEENDTC"].map(dmy),
        "SEVERITY": ae["AESEV"],
        "SERIOUS_YN": ae["AESER"],
        "RELATED": ae["AEREL"],
        "OUTCOME": ae["AEOUT"],
    })))

    written.append(("mh_log", pd.DataFrame({
        "SUBJECT": mh["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "MH_NO": mh["MHSEQ"],
        "MH_TERM": mh["MHTERM"],
        # Partial onsets keep their granularity: Mar-2015, or bare 2015.
        "MH_ONSET": mh["MHSTDTC"].map(dash_mon),
        "ONGOING_YN": mh["MHONGO"],
    })))

    written.append(("conmeds", pd.DataFrame({
        "SUBJECT": cm["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "CM_NO": cm["CMSEQ"],
        "MEDICATION": cm["CMTRT"],
        "INDICATION": cm["CMINDC"],
        "DOSE_AMT": cm["CMDOSE"],
        "DOSE_UNIT": cm["CMDOSU"],
        "ROUTE_CD": cm["CMROUTE"],
        # ddMMMyyyy: unambiguous, and a format the ISO-only shift cannot read.
        "CM_START": cm["CMSTDTC"].map(ddmonyyyy),
    })))

    written.append(("dosing", pd.DataFrame({
        "SUBJECT": ex["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "CYCLE_NO": ex["EXSEQ"],
        "STUDY_DRUG": ex["EXTRT"],
        "DOSE_MG": ex["EXDOSE"],
        "FREQ_CD": ex["EXDOSFRQ"],
        "ROUTE_CD": ex["EXROUTE"],
        "DOSE_DT": ex["EXSTDTC"].map(slashed_iso),
    })))

    # Vitals uses US numeric order while the AE form uses European. Both are
    # all-numeric and look identical; only the values settle which is which,
    # which is the case for inferring the order per column rather than once
    # per study.
    written.append(("vitals", pd.DataFrame({
        "SUBJECT": vs["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "VISIT_NO": vs["VISITNUM"],
        "MEASURE_CD": vs["VSTESTCD"],
        "RESULT": vs["VSORRES"],
        "RESULT_UNIT": vs["VSORRESU"],
        "VISIT_DT": vs["VSDTC"].map(mdy),
    })))

    written.append(("labs", pd.DataFrame({
        "SUBJECT": lb["USUBJID"].str.rsplit("-", n=2).str[-2:].str.join("-"),
        "VISIT_NO": lb["VISITNUM"],
        "TEST_CD": lb["LBTESTCD"],
        "TEST_NAME": lb["LBTEST"],
        "RESULT": lb["LBORRES"],
        "RESULT_UNIT": lb["LBORRESU"],
        "COLLECT_DT": lb["LBDTC"].map(ddmonyyyy),
    })))

    for name, frame in written:
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:<9} {len(frame):>6} rows  {len(frame.columns):>3} cols")

    print(f"\nwrote synthetic raw EDC extract to {out}")
    print(f"date forms: dd/mm/yyyy, mm/dd/yyyy, ddMMMyyyy, yyyy/mm/dd, Mon-yyyy")
    print("subject id: SUBJECT (site-number), NOT USUBJID -- use --offset-key")
    _ = rng  # reserved: per-site format variation


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
