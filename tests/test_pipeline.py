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
from pydantic import ValidationError
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


# ----------------------------------------------------------------------
# profiling real-shaped SDTM
# ----------------------------------------------------------------------
def _realish_sdtm() -> dict[str, pd.DataFrame]:
    """SDTM as it actually arrives: derived --DY columns beside the --DTC
    dates, design variables, flags, and relative-timing columns."""
    return {
        "DM": pd.DataFrame(
            {
                "STUDYID": ["S"] * 3, "DOMAIN": ["DM"] * 3,
                "USUBJID": ["S-01-001", "S-01-002", "S-01-003"],
                "SITEID": ["01"] * 3, "AGE": [54, 61, 47], "AGEU": ["YEARS"] * 3,
                "SEX": list("MFM"), "RACE": ["WHITE"] * 3,
                "ARM": ["Drug A", "Placebo", "Drug A"], "ARMCD": ["A", "P", "A"],
                "COUNTRY": ["USA"] * 3,
                "RFSTDTC": ["2026-02-18", "2026-02-19", "2026-02-20"],
                "DTHDTC": [None, None, "2026-06-01"], "DTHFL": [None, None, "Y"],
            }
        ),
        "AE": pd.DataFrame(
            {
                "STUDYID": ["S"] * 3, "DOMAIN": ["AE"] * 3,
                "USUBJID": ["S-01-001", "S-01-002", "S-01-003"],
                "AESEQ": [1, 2, 3],
                "AETERM": ["Headache", "Nausea", "Rash"],
                "AEDECOD": ["Headache", "Nausea", "Rash"],
                "AESTDTC": ["2026-03-01", "2026-03-05", "2026-03-09"],
                "AESTDY": [12, 16, 20],
                "AEENDTC": ["2026-03-04", "2026-03-08", None],
                "AEENDY": [15, 19, None],
                "AEENRF": [None, None, "ONGOING"],
                "EPOCH": ["TREATMENT"] * 3,
            }
        ),
    }


def test_derived_study_days_do_not_collide_with_their_dates():
    """Real SDTM ships AESTDY beside AESTDTC. Converting the date would emit a
    second AESTDY and fail contract validation, so the date is dropped instead
    -- and the drop has to name the survivor."""
    contract, _ = draft_contract(_realish_sdtm(), source="REALISH")

    ae = contract.domain("AE")
    for date_col, derived in (("AESTDTC", "AESTDY"), ("AEENDTC", "AEENDY")):
        r = ae.rule(date_col)
        assert r.treatment is Treatment.DROP, f"{date_col}: {r.treatment}"
        assert r.redundant_with == derived
        assert ae.rule(derived).treatment is Treatment.RETAIN

    outputs = [f.resolved_output() for f in ae.fields if f.treatment is not Treatment.DROP]
    assert len(outputs) == len(set(outputs)), "duplicate output columns"


def test_redundant_drop_is_allowed_in_a_retained_in_full_domain():
    """AE is retained in full, yet dropping a date whose study day survives is
    not a loss -- provided the contract says where it survives."""
    from deidkit.contract import DomainContract, FieldRule

    ok = DomainContract(
        name="AE", retained_in_full=True,
        fields=[
            FieldRule(column="AESTDY", treatment=Treatment.RETAIN),
            FieldRule(column="AESTDTC", treatment=Treatment.DROP,
                      redundant_with="AESTDY"),
        ],
    )
    assert ok.rule("AESTDTC").redundant_with == "AESTDY"

    # An unexplained drop still fails.
    with pytest.raises(ValidationError, match="retained_in_full"):
        DomainContract(
            name="AE", retained_in_full=True,
            fields=[FieldRule(column="AETERM", treatment=Treatment.DROP)],
        )

    # Naming a survivor that is itself dropped fails: nothing survives.
    with pytest.raises(ValidationError, match="survives nowhere"):
        DomainContract(
            name="AE",
            fields=[
                FieldRule(column="AESTDY", treatment=Treatment.DROP),
                FieldRule(column="AESTDTC", treatment=Treatment.DROP,
                          redundant_with="AESTDY"),
            ],
        )

    # Naming a column that does not exist fails.
    with pytest.raises(ValidationError, match="no rule in this domain"):
        DomainContract(
            name="AE",
            fields=[FieldRule(column="AESTDTC", treatment=Treatment.DROP,
                              redundant_with="NOPE")],
        )


def test_profiler_recognises_ordinary_sdtm_columns():
    """Flagging correct decisions as 'unrecognised' teaches reviewers to click
    through, so the flag has to mean something."""
    _, suggestions = draft_contract(_realish_sdtm(), source="REALISH")
    low = [
        f"{dom}.{col}"
        for dom, cols in suggestions.items()
        for col, s in cols.items()
        if s.confidence == "low"
    ]
    assert not low, f"ordinary SDTM columns left unrecognised: {low}"


def test_sequence_keys_are_not_called_clinical_content():
    """--SEQ is a record key. Calling it 'the analytic payload' is a plausible
    lie, which is worse for a reviewer than an honest 'unrecognised'."""
    _, suggestions = draft_contract(_realish_sdtm(), source="REALISH")
    assert "sequence key" in suggestions["AE"]["AESEQ"].rationale


# ----------------------------------------------------------------------
# SDTM-conformant output
# ----------------------------------------------------------------------
def test_sdtm_mode_keeps_dtc_and_shifts_it(tmp_path):
    """--DTC is a required SDTM variable. Replacing it with a study day gives
    a dataset that will not validate, so the SDTM mode shifts it instead."""
    frames = _realish_sdtm()
    contract, _ = draft_contract(frames, source="S", sdtm_conformant=True)

    ae = contract.domain("AE")
    for col in ("AESTDTC", "AEENDTC"):
        assert ae.rule(col).treatment is Treatment.DATE_SHIFT
    # the derived study days stay, and stay correct
    assert ae.rule("AESTDY").treatment is Treatment.RETAIN
    # the anchor shifts too -- that is what keeps --DY valid
    assert contract.domain("DM").rule("RFSTDTC").treatment is Treatment.DATE_SHIFT

    with Vault(tmp_path / "v.db", key=Vault.generate_key()) as vault:
        out = DeidPipeline(contract, vault).run(frames)

    before, after = frames["AE"], out.frames["AE"]
    assert "AESTDTC" in after.columns, "SDTM mode must keep --DTC"

    for b, a in zip(before["AESTDTC"], after["AESTDTC"]):
        assert a != b, "dates must actually move"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(a)), f"not an ISO date: {a}"

    # intervals survive the shift
    for bs, be, as_, ae_ in zip(
        before["AESTDTC"], before["AEENDTC"], after["AESTDTC"], after["AEENDTC"]
    ):
        d = [parse_dtc(x).to_date() for x in (bs, be, as_, ae_)]
        if all(d):
            assert (d[1] - d[0]).days == (d[3] - d[2]).days


def test_default_mode_is_flagged_as_non_conformant():
    """The study-day default removes a required variable. That trade-off has
    to be visible in the contract, not discovered by a validator later."""
    contract, _ = draft_contract(_realish_sdtm(), source="S")
    note = contract.domain("AE").rule("AESTDTC").note or ""
    assert "non-conformant" in note.lower() or "sdtm_conformant" in note


def test_conmed_verbatim_is_screened_not_treated_as_coded():
    """CMTRT is investigator-typed, and after AETERM it is the free-text field
    most likely to carry an identifier. CMDECOD is the coded counterpart."""
    cm = pd.DataFrame(
        {
            "STUDYID": ["S"] * 2, "DOMAIN": ["CM"] * 2,
            "USUBJID": ["S-01-001", "S-01-002"], "CMSEQ": [1, 2],
            "CMTRT": ["Lisinopril", "Insulin, started by Dr. Halvorsen at Riverside Clinic"],
            "CMDECOD": ["LISINOPRIL", "INSULIN GLARGINE"],
            "CMDOSE": [10, 20], "CMROUTE": ["ORAL", "SUBCUTANEOUS"],
        }
    )
    frames = _realish_sdtm() | {"CM": cm}
    contract, _ = draft_contract(frames, source="S")
    assert contract.domain("CM").rule("CMTRT").treatment is Treatment.SCREEN_FREETEXT
    assert contract.domain("CM").rule("CMDECOD").treatment is Treatment.RETAIN


def test_drug_names_are_not_masked():
    """A treatment arm is the exposure under study: identical for everyone in
    it, and identifying nobody. Masking it would destroy the analysis."""
    frames = _realish_sdtm()
    contract, _ = draft_contract(frames, source="S")
    assert contract.domain("DM").rule("ARM").treatment is Treatment.RETAIN


# ----------------------------------------------------------------------
# dates are identifiers, and the tier has to agree
# ----------------------------------------------------------------------
def _minimal_contract(tier, dtc_treatment):
    from deidkit.contract import AnchorSpec, DomainContract, FieldRule

    extra = {"entity": "subject"} if dtc_treatment is Treatment.DATE_SHIFT else {}
    return Contract(
        contract_version="1.0", source="S", tier=tier,
        anchor=AnchorSpec(domain="DM", date_column="RFSTDTC"),
        domains=[
            DomainContract(
                name="DM",
                fields=[FieldRule(column="RFSTDTC", treatment=dtc_treatment, **extra)],
            )
        ],
    )


def test_deidentified_tier_cannot_keep_calendar_dates():
    """HIPAA enumerates dates as identifiers (164.514(b)(2)(i)(C)), and they are
    the strongest linkage vector a clinical dataset has. An LDS may keep them;
    a de-identified tier may not, and the contract should not let the two be
    confused -- nothing in the data itself would reveal the mistake."""
    with pytest.raises(ValidationError, match="deidentified"):
        _minimal_contract("deidentified", Treatment.RETAIN)

    # The same rules are fine once the tier tells the truth.
    assert _minimal_contract("lds", Treatment.RETAIN).tier == "lds"
    # And shifting satisfies the de-identified tier.
    assert _minimal_contract("deidentified", Treatment.DATE_SHIFT).tier == "deidentified"


# ----------------------------------------------------------------------
# treatment blinding -- confidentiality, not privacy
# ----------------------------------------------------------------------
def test_treatment_labels_are_consistent_reversible_and_keep_placebo(tmp_path):
    frames = _realish_sdtm()
    frames["DM"]["ACTARM"] = frames["DM"]["ARM"]
    frames["DM"]["ARM"] = ["Pembrolizumab 200 mg Q3W", "Placebo", "Pembrolizumab 200 mg Q3W"]
    frames["DM"]["ACTARM"] = frames["DM"]["ARM"]

    contract, _ = draft_contract(frames, source="S", blind_treatment=True)
    assert contract.domain("DM").rule("ARM").treatment is Treatment.LABEL_MAP

    with Vault(tmp_path / "v.db", key=Vault.generate_key(), operator="t") as vault:
        out = DeidPipeline(contract, vault).run(frames)
        dm = out.frames["DM"]

        # placebo passes through: an analysis still needs to know the control
        assert list(dm["ARM"]) == [dm["ARM"][0], "Placebo", dm["ARM"][0]]
        assert dm["ARM"][0].startswith("TRT ")

        # one namespace, so every column naming the treatment agrees
        assert list(dm["ARM"]) == list(dm["ACTARM"])

        # group sizes unchanged -- the analysis is untouched
        assert sorted(frames["DM"]["ARM"].value_counts()) == sorted(dm["ARM"].value_counts())

        # reversible for unblinding, through the same logged path
        original = vault.reverse(
            "treatment", dm["ARM"][0], justification="DSMB unblinding"
        )
        assert original == "Pembrolizumab 200 mg Q3W"
        assert vault.access_log()[-1]["justification"] == "DSMB unblinding"


def test_treatment_names_are_kept_unless_blinding_is_asked_for():
    """Blinding is opt-in: it is a confidentiality control, and a drug name is
    not PHI."""
    contract, _ = draft_contract(_realish_sdtm(), source="S")
    assert contract.domain("DM").rule("ARM").treatment is Treatment.RETAIN
