"""End-to-end guarantees.

These tests assert the properties the design actually promises, not just that
the code runs: no absolute date survives, MH/AE clinical content is byte-
identical to the input, surrogates are not derived from the originals, reruns
reproduce, and reverse lookups are always logged.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deidkit import Contract, DeidPipeline, Vault, load_study
from deidkit.contract import Treatment
from deidkit.freetext import PatternDetector, apply_adjudication
from deidkit.profile import draft_contract
from deidkit.risk import measure
from deidkit.transforms import cap_numeric, parse_dtc, study_day, zip3

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_synthetic_study.py"

ISO_DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def study(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("quarantine") / "study"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(d)], check=True, capture_output=True
    )
    return d


@pytest.fixture(scope="module")
def raw(study: Path):
    frames, sums = load_study(study)
    return frames, sums


@pytest.fixture(scope="module")
def contract(raw) -> Contract:
    frames, _ = raw
    c, _ = draft_contract(frames, source="TEST-001")
    return c


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(
        tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest"
    )
    yield v
    v.close()


@pytest.fixture()
def result(raw, contract, vault):
    frames, sums = raw
    return DeidPipeline(contract, vault, operator="pytest").run(
        frames, checksums=sums
    )


# ----------------------------------------------------------------------
# unit-level
# ----------------------------------------------------------------------
def test_partial_dates_keep_their_granularity():
    assert parse_dtc("2015").granularity == "year"
    assert parse_dtc("2015-03").granularity == "month"
    assert parse_dtc("2015-03-14").granularity == "day"
    assert parse_dtc("2015-03-14T10:30").granularity == "day"
    assert parse_dtc("").granularity == "none"
    assert parse_dtc(None).granularity == "none"
    # never invents a day
    assert parse_dtc("2015").to_date() is None
    assert parse_dtc("2015-03").to_date() is None


def test_study_day_has_no_day_zero():
    from datetime import date

    anchor = date(2026, 3, 1)
    assert study_day(anchor, anchor) == 1
    assert study_day(date(2026, 3, 2), anchor) == 2
    assert study_day(date(2026, 2, 28), anchor) == -1
    assert study_day(anchor, anchor, "day0") == 0


def test_age_cap_keeps_exact_values_below_the_cap():
    out = cap_numeric(pd.Series([45, 89, 90, 103, None]), cap=90)
    assert list(out[:2]) == ["45", "89"]
    assert out[2] == "90+" and out[3] == "90+"
    assert pd.isna(out[4])


def test_zip3_suppresses_low_population_prefixes():
    # 036 is one of the 17 prefixes with population under 20,000
    out = zip3(pd.Series(["03601", "02115", "9"]))
    assert out[0] == "000"
    assert out[1] == "021"  # leading zero must survive: this stays a string
    assert pd.isna(out[2])


def test_detector_ignores_clinical_vocabulary():
    det = PatternDetector()
    # eponymous conditions must not flood the review queue
    for clean in (
        "Guillain-Barre syndrome",
        "Crohn disease flare",
        "Glasgow Coma Scale 14",
        "Foley catheter placed",
    ):
        assert det.detect(clean) == [], clean
    # real identifiers must be caught
    assert det.detect("Fell at St Mary's Hospital, seen by Dr. Okafor")
    assert det.detect("called (617) 555-0142")
    assert det.detect("emailed to j.a@x.example")


# ----------------------------------------------------------------------
# the load-bearing guarantees
# ----------------------------------------------------------------------
def test_no_absolute_date_survives_anywhere(result):
    """The whole point of the study-day reparameterisation."""
    offenders: list[str] = []
    for name, frame in result.frames.items():
        for col in frame.columns:
            joined = " ".join(
                str(v) for v in frame[col].dropna().head(500).tolist()
            )
            if ISO_DATE.search(joined):
                offenders.append(f"{name}.{col}")
    assert not offenders, f"absolute dates leaked into {offenders}"


def test_mh_and_ae_clinical_content_is_byte_identical(raw, result):
    """'Retained in full' has to be literally true, not approximately."""
    frames, _ = raw
    for dom, cols in (
        ("AE", ["AETERM", "AEDECOD", "AEBODSYS", "AESEV", "AESER", "AEREL",
                "AEOUT", "AEACN"]),
        ("MH", ["MHTERM", "MHDECOD", "MHBODSYS", "MHONGO"]),
    ):
        for col in cols:
            before = frames[dom][col]
            after = result.frames[dom][col]
            pd.testing.assert_series_equal(
                before.astype("string").reset_index(drop=True),
                after.astype("string").reset_index(drop=True),
                check_names=False,
                obj=f"{dom}.{col}",
            )


def test_rare_ae_terms_are_not_pooled_away(raw, result):
    """Rare preferred terms are retained by decision, not silently rolled up."""
    frames, _ = raw
    rare = "Guillain-Barre syndrome"
    if rare in set(frames["AE"]["AEDECOD"]):
        assert rare in set(result.frames["AE"]["AEDECOD"])


def test_ae_intervals_are_preserved_exactly(raw, result):
    """AE duration must be recoverable from the study-day columns."""
    frames, _ = raw
    before, after = frames["AE"], result.frames["AE"]
    dur_before = []
    for s, e in zip(before["AESTDTC"], before["AEENDTC"]):
        ds, de = parse_dtc(s).to_date(), parse_dtc(e).to_date()
        dur_before.append(None if not (ds and de) else (de - ds).days)
    dur_after = (after["AEENDY"] - after["AESTDY"]).tolist()

    compared = 0
    for b, a in zip(dur_before, dur_after):
        if b is None:
            continue
        assert b == a, f"AE duration changed: {b} -> {a}"
        compared += 1
    assert compared > 50, "not enough non-null durations to be meaningful"


def test_surrogates_are_not_derived_from_originals(raw, result):
    """A surrogate must not embed, or be a hash of, the original."""
    frames, _ = raw
    originals = frames["DM"]["USUBJID"].astype(str).tolist()
    surrogates = result.frames["DM"]["USUBJID"].astype(str).tolist()

    assert len(set(surrogates)) == len(set(originals))
    for orig, surr in zip(originals, surrogates):
        # site and enrolment order must not survive: "TIG-2026-001-US-001-0042"
        tail = orig.split("-")[-1]
        assert tail not in surr, f"{tail!r} from {orig!r} survives in {surr!r}"
        assert surr != orig

    # And the mapping must not be reproducible without the vault: two vaults
    # issue different surrogates for the same subject.
    assert not set(originals) & set(surrogates)


def test_joins_survive_the_surrogate(raw, result):
    """Consistent surrogates across domains, or the data is useless."""
    frames, _ = raw
    for dom in ("AE", "MH", "LB", "VS"):
        before = set(frames[dom]["USUBJID"])
        after = set(result.frames[dom]["USUBJID"].dropna())
        assert len(after) == len(before)
        # every domain's subjects must resolve into DM's surrogate space
        assert after <= set(result.frames["DM"]["USUBJID"].dropna())


def test_direct_identifiers_are_gone(raw, result):
    dm_out = result.frames["DM"]
    for gone in ("INVNAM", "BRTHDTC", "SUBJPHONE", "SUBJID"):
        assert gone not in dm_out.columns, f"{gone} should have been dropped"


def test_screening_does_not_modify_data(raw, result):
    """screen_freetext produces a queue; it must never rewrite a value."""
    frames, _ = raw
    assert not result.review_queue.empty, "planted PHI was not detected at all"
    pd.testing.assert_series_equal(
        frames["AE"]["AETERM"].astype("string").reset_index(drop=True),
        result.frames["AE"]["AETERM"].astype("string").reset_index(drop=True),
        check_names=False,
    )


def test_screening_finds_the_planted_identifiers(result):
    queue = result.review_queue
    types = {
        t
        for row in queue["entity_types"]
        for t in str(row).split(",")
        if t
    }
    assert {"FACILITY", "PERSON"} & types, f"weak detection: {types}"
    assert queue["verdict"].eq("").all(), "verdicts must start empty"


def test_adjudication_only_touches_redact_rows(raw, result):
    frames, _ = raw
    queue = result.review_queue.copy()
    queue["verdict"] = "PASS"
    queue.loc[queue.index[0], "verdict"] = "REDACT"
    queue.loc[queue.index[0], "replacement"] = "Fall at hospital"

    out, stats = apply_adjudication(result.frames, queue)
    assert stats["redactions_applied"] == 1
    assert stats["rows_unreviewed"] == 0

    row = queue.iloc[0]
    assert out[row["domain"]].at[row["row_id"], row["column"]] == "Fall at hospital"
    # everything else untouched
    changed = (
        out["AE"]["AETERM"].astype("string")
        != result.frames["AE"]["AETERM"].astype("string")
    ).sum()
    assert changed == 1


def test_unreviewed_rows_are_reported_not_assumed(result):
    queue = result.review_queue.copy()
    queue["verdict"] = ""
    _, stats = apply_adjudication(result.frames, queue)
    assert stats["rows_unreviewed"] == len(queue)
    assert stats["redactions_applied"] == 0


# ----------------------------------------------------------------------
# reproducibility and reversibility
# ----------------------------------------------------------------------
def test_rerun_against_the_same_vault_is_identical(raw, contract, vault):
    frames, sums = raw
    p = DeidPipeline(contract, vault, operator="pytest")
    a = p.run(frames, checksums=sums)
    b = p.run(frames, checksums=sums)
    for dom in a.frames:
        pd.testing.assert_frame_equal(a.frames[dom], b.frames[dom], obj=dom)


def test_reverse_lookup_requires_and_records_a_justification(raw, result, vault):
    frames, _ = raw
    original = str(frames["DM"]["USUBJID"].iloc[0])
    surrogate = str(result.frames["DM"]["USUBJID"].iloc[0])

    with pytest.raises(Exception):
        vault.reverse("subject", surrogate, justification="  ")

    got = vault.reverse(
        "subject", surrogate, justification="SAE-2026-0031 safety report"
    )
    assert got == original

    log = vault.access_log()
    assert log[-1]["action"] == "reverse"
    assert "SAE-2026-0031" in log[-1]["justification"]
    assert log[-1]["operator"] == "pytest"


def test_wrong_key_refuses_to_open_the_vault(tmp_path):
    from deidkit.vault import VaultError

    path = tmp_path / "v.db"
    Vault(path, key=Vault.generate_key()).close()
    with pytest.raises(VaultError, match="does not open"):
        Vault(path, key=Vault.generate_key())


# ----------------------------------------------------------------------
# contract enforcement
# ----------------------------------------------------------------------
def test_undeclared_column_halts_the_run(raw, contract, vault):
    from deidkit.pipeline import ContractMismatch

    frames, _ = raw
    tampered = {k: v.copy() for k, v in frames.items()}
    tampered["DM"]["PATIENT_EMAIL"] = "a@b.example"

    with pytest.raises(ContractMismatch, match="not in contract"):
        DeidPipeline(contract, vault).run(tampered)


def test_retained_in_full_rejects_value_destroying_rules(raw):
    from pydantic import ValidationError

    from deidkit.contract import DomainContract, FieldRule

    with pytest.raises(ValidationError, match="retained_in_full"):
        DomainContract(
            name="AE",
            retained_in_full=True,
            fields=[
                FieldRule(column="AETERM", treatment=Treatment.SCREEN_FREETEXT),
                FieldRule(
                    column="AEDECOD",
                    treatment=Treatment.POOL_RARE,
                    min_count=5,
                ),
            ],
        )


def test_manifest_records_what_a_determination_needs(result):
    m = result.manifest
    assert m["tier"] == "deidentified"
    assert m["contract"]["contract_version"]
    assert m["inputs"]["checksums_sha256"]
    assert m["risk"]["k_min"] >= 1
    assert m["anchor"]["subjects_resolved"] == 120
    assert m["transformation"]["AE"]["retained_in_full"] is True
    assert m["freetext_screening"]["flagged_rows"] > 0


# ----------------------------------------------------------------------
# risk measurement
# ----------------------------------------------------------------------
def test_risk_nulls_do_not_split_equivalence_classes():
    frame = pd.DataFrame(
        {"a": ["x", "x", "x", "x"], "b": [None, None, None, None]}
    )
    r = measure(frame, ["a", "b"], k_target=2)
    assert r.k_min == 4 and r.n_classes == 1


def test_risk_reports_a_reduction_path_when_the_target_is_missed():
    frame = pd.DataFrame(
        {
            "site": list("AABBCC") * 4,
            "sex": list("MF") * 12,
            "age": [str(20 + i) for i in range(24)],
        }
    )
    r = measure(frame, ["site", "sex", "age"], k_target=4)
    assert not r.k_met
    assert r.reduction_path, "a missed target must come with a way forward"
    assert r.reduction_path[-1]["k_min"] >= r.k_min
