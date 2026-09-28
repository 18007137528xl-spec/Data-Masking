"""End-to-end guarantees.

These tests assert the properties the design actually promises, not just that
the code runs: no absolute date survives, MH/AE clinical content is byte-
identical to the input, surrogates are not derived from the originals, reruns
reproduce, and reverse lookups are always logged.
"""

from __future__ import annotations

import json
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
    # Rows needing judgment start blank; rows the policy settled carry a
    # verdict AND a verdict_source, so nothing reads as a human's ruling.
    blank = queue["verdict"].astype(str).str.strip() == ""
    assert blank.any(), "every row was auto-decided; nothing left for a human"
    assert (
        queue.loc[~blank, "verdict_source"].astype(str).str.strip() != ""
    ).all(), "a pre-filled verdict must say where it came from"
    assert (queue["reviewer"].astype(str).str.strip() == "").all()


def test_adjudication_only_touches_redact_rows(raw, result):
    frames, _ = raw
    queue = result.review_queue.copy()
    queue["verdict"] = "PASS"
    queue["replacement"] = ""
    # Pick an AE row by name rather than by position: the queue is now ordered
    # judgment-first, so index 0 is whichever domain needed a human.
    target = queue.index[(queue["domain"] == "AE").to_numpy().argmax()]
    queue.loc[target, "verdict"] = "REDACT"
    queue.loc[target, "replacement"] = "Fall at hospital"

    out, stats = apply_adjudication(result.frames, queue)
    assert stats["redactions_applied"] == 1
    assert stats["rows_unreviewed"] == 0

    row = queue.loc[target]
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


# ----------------------------------------------------------------------
# blinding beyond the ARM column
# ----------------------------------------------------------------------
def test_blind_terms_are_derived_not_guessed():
    from deidkit.blinding import derive_terms

    terms = derive_terms(["Pembrolizumab 200 mg Q3W", "Placebo", "Placebo tablet"])
    assert "pembrolizumab" in terms
    assert "q3w" in terms
    # words that would match everywhere and mean nothing
    for noise in ("mg", "placebo", "tablet", "200"):
        assert noise not in terms, f"{noise!r} would flood the queue"


def test_study_drug_in_free_text_is_flagged():
    """Relabelling ARM is undone by one investigator writing the compound into
    AETERM -- and free text is where a model memorises."""
    from deidkit.blinding import CompositeDetector, StudyDrugRecognizer

    det = CompositeDetector(PatternDetector(), StudyDrugRecognizer(["pembrolizumab"]))
    found = det.detect("Rash 3 days after pembrolizumab infusion")
    assert [f.entity_type for f in found] == ["STUDY_DRUG"]
    assert det.detect("Rash on forearm") == []


def test_blinding_audit_reports_what_relabelling_cannot_reach():
    """Everything upstream is intent; the audit is outcome."""
    from deidkit.blinding import audit

    frames = {
        "AE": pd.DataFrame({"AETERM": ["Rash after pembrolizumab", "Headache"]}),
        "EX": pd.DataFrame({"EXDOSFRQ": ["Q3W", "Q6W"]}),
        "DM": pd.DataFrame({"ARM": ["TRT A", "TRT B"]}),
    }
    rep = audit(frames, ["pembrolizumab", "q3w"], skip=[("DM", "ARM")])
    assert not rep.held
    found = {(x.domain, x.column, x.term) for x in rep.leaks}
    assert ("AE", "AETERM", "pembrolizumab") in found
    assert ("EX", "EXDOSFRQ", "q3w") in found

    clean = audit(
        {"AE": pd.DataFrame({"AETERM": ["Rash after study drug"]})},
        ["pembrolizumab"],
    )
    assert clean.held


def test_review_queue_is_not_a_domain_and_not_published(tmp_path):
    """The queue holds the ORIGINAL text of every flagged row -- that is what
    makes it reviewable, and what makes it unredacted PHI. It must not be
    discovered as data, nor sit in the directory analysts read."""
    from deidkit.io import discover

    tier = tmp_path / "tier"
    tier.mkdir()
    pd.DataFrame({"USUBJID": ["A"]}).to_csv(tier / "dm.csv", index=False)
    pd.DataFrame({"text": ["Fell at St Mary's Hospital"]}).to_csv(
        tier / "review_queue.csv", index=False
    )
    (tier / "manifest.json").write_text("{}")

    assert set(discover(tier)) == {"DM"}, "the queue must not be read back as data"


# ----------------------------------------------------------------------
# verbatim or controlled vocabulary?
# ----------------------------------------------------------------------
def test_textcheck_distinguishes_a_picklist_from_typed_text():
    """--TERM is the investigator's wording and --DECOD is the MedDRA term,
    but some EDC builds make the site pick from a coded list. Which one you
    have decides whether the column needs adjudication, so measure it."""
    from deidkit.textcheck import characterise

    # A pick-list: the term IS the coded term, every time.
    picklist = pd.DataFrame(
        {
            "AETERM": ["Headache", "Nausea", "Headache", "Rash"],
            "AEDECOD": ["Headache", "Nausea", "Headache", "Rash"],
        }
    )
    assert characterise(picklist, "AETERM").verdict == "controlled"

    # Typed text: abbreviations, added detail, and an identifier.
    typed = pd.DataFrame(
        {
            "AETERM": [
                "Elevated ALT",
                "mild nausea after dosing",
                "Fell at St Mary's Hospital, seen by Dr. Okafor",
                "Headache",
            ],
            "AEDECOD": [
                "Alanine aminotransferase increased",
                "Nausea",
                "Fall",
                "Headache",
            ],
        }
    )
    assert characterise(typed, "AETERM").verdict == "verbatim"


def test_textcheck_fails_toward_review():
    """Calling verbatim text 'controlled' skips review and publishes whatever
    was typed; the reverse wastes a reviewer's minutes. One weak signal must
    not buy 'controlled'."""
    from deidkit.textcheck import characterise

    # Short, one term per code, no detector hits -- but the wording still
    # diverges from the coding, which is a human writing shorthand.
    shorthand = pd.DataFrame(
        {
            "MHTERM": ["GERD", "Crohn disease", "Asthma", "Hypertension"],
            "MHDECOD": [
                "Gastrooesophageal reflux disease",
                "Crohn's disease",
                "Asthma",
                "Hypertension",
            ],
        }
    )
    p = characterise(shorthand, "MHTERM")
    assert p.verdict == "mixed", p.verdict
    assert p.match_coded_rate is not None and p.match_coded_rate < 0.95


# ----------------------------------------------------------------------
# the configuration this project asked for
# ----------------------------------------------------------------------
def test_keep_dates_retains_dtc_verbatim_and_forces_the_lds_tier(tmp_path):
    """Dates as recorded, no shift and no study-day conversion. HIPAA
    enumerates dates as identifiers, so this is only lawful on a Limited Data
    Set -- the contract sets the tier to match rather than letting a tier
    claim to be de-identified while shipping a calendar."""
    frames = _realish_sdtm()
    contract, _ = draft_contract(frames, source="S", keep_dates=True)

    assert contract.tier == "lds"
    for dom, col in (("AE", "AESTDTC"), ("DM", "RFSTDTC"), ("DM", "DTHDTC")):
        assert contract.domain(dom).rule(col).treatment is Treatment.RETAIN

    with Vault(tmp_path / "v.db", key=Vault.generate_key()) as vault:
        out = DeidPipeline(contract, vault).run(frames)

    for col in ("AESTDTC", "AEENDTC"):
        pd.testing.assert_series_equal(
            frames["AE"][col].astype("string").reset_index(drop=True),
            out.frames["AE"][col].astype("string").reset_index(drop=True),
            check_names=False, obj=col,
        )


def test_arm_and_product_columns_get_separate_label_namespaces(tmp_path):
    """ARM is a regimen, EXTRT is the compound. One shared namespace gave EXTRT
    a label that read like a third arm."""
    frames = _realish_sdtm()
    frames["DM"]["ARM"] = ["Pembro 200 mg Q3W", "Placebo", "Pembro 400 mg Q6W"]
    frames["EX"] = pd.DataFrame(
        {
            "STUDYID": ["S"] * 3, "DOMAIN": ["EX"] * 3,
            "USUBJID": frames["DM"]["USUBJID"],
            "EXSEQ": [1, 2, 3],
            "EXTRT": ["Pembro", "Placebo", "Pembro"],
            "EXDOSE": [200, 0, 400],
        }
    )
    contract, _ = draft_contract(frames, source="S", blind_treatment=True)

    arm = contract.domain("DM").rule("ARM")
    ext = contract.domain("EX").rule("EXTRT")
    assert arm.treatment is ext.treatment is Treatment.LABEL_MAP
    assert arm.entity != ext.entity
    assert arm.prefix == "TRT" and ext.prefix == "DRUG"

    with Vault(tmp_path / "v.db", key=Vault.generate_key()) as vault:
        out = DeidPipeline(contract, vault).run(frames)

    arms = set(out.frames["DM"]["ARM"].dropna())
    prods = set(out.frames["EX"]["EXTRT"].dropna())
    assert "Placebo" in arms and "Placebo" in prods
    assert all(a.startswith("TRT ") for a in arms - {"Placebo"})
    assert all(p.startswith("DRUG ") for p in prods - {"Placebo"})
    # two arms, one compound: the labels no longer imply a third arm
    assert len(arms - {"Placebo"}) == 2
    assert len(prods - {"Placebo"}) == 1


# ----------------------------------------------------------------------
# what the manifest may say about the most exposed subjects
# ----------------------------------------------------------------------
def test_the_manifest_reports_exposure_without_naming_the_exposed(result):
    """The manifest is the artifact that travels.

    It gets attached to a determination, pasted into a ticket, mailed to a
    reviewer. The quasi-identifier values of the smallest equivalence classes
    are already in the published table, so this is not about secrecy -- it is
    that away from the data, a ranked list of who is unique on the QI set is a
    targeting aid living in the one file whose purpose is to attest the release
    is safe. Counts and the distribution say everything a determination needs.
    """
    risk = result.manifest["risk"]
    assert "smallest_classes" not in risk
    assert "class_size_histogram" in risk
    assert risk["n_classes_below_k"] >= 0
    # nothing anywhere in the risk block should be a QI value
    blob = json.dumps(risk)
    for value in ("WHITE", "NOT HISPANIC OR LATINO", "SITE-"):
        assert value not in blob, f"{value!r} leaked into the manifest risk block"


def test_the_steward_copy_keeps_the_values(result):
    """The same report, asked for explicitly, still carries what a human needs
    to decide what to generalise."""
    detail = result.risk_report.to_dict(include_class_values=True)
    assert detail["smallest_classes"]
    assert "values" in detail["smallest_classes"][0]


# ----------------------------------------------------------------------
# the second review loop, and the tier that is actually released
# ----------------------------------------------------------------------
def test_a_screened_tier_declares_itself_unfinished(result):
    """screen_freetext does not modify data, so a flagged row is published
    exactly as received. The manifest has to say so: "33 rows flagged" reads
    like 33 rows handled, and the difference is whether unredacted text is
    sitting in the tier someone is about to share."""
    adj = result.manifest["adjudication"]
    assert adj["required"] is True
    assert adj["complete"] is False
    blank = (result.review_queue["verdict"].astype(str).str.strip() == "").sum()
    assert adj["pending_rows"] == blank
    assert adj["pending_rows"] <= len(result.review_queue)


def test_an_untouched_queue_counts_as_unreviewed_not_unrecognised():
    """pandas reads a blank verdict column as NaN, and float('nan') is truthy.

    That made a queue nobody had opened report as N rows with an
    *unrecognised* verdict beside "rows unreviewed: 0" -- which reads like it
    was reviewed and merely mistyped.
    """
    frames = {"AE": pd.DataFrame({"AETERM": ["headache", "rash"]})}
    queue = pd.DataFrame(
        {
            "domain": ["AE", "AE"],
            "row_id": [0, 1],
            "column": ["AETERM", "AETERM"],
            "verdict": [float("nan"), None],
            "replacement": [float("nan"), float("nan")],
        }
    )
    _, stats = apply_adjudication(frames, queue)
    assert stats["rows_unreviewed"] == 2
    assert stats["rows_unknown_verdict"] == 0


def test_adjudication_records_who_and_what_in_the_final_manifest(
    tmp_path, result, study
):
    """The released tier was the only one without a manifest: data written,
    counts printed to a console, nothing durable saying who ruled on what."""
    from deidkit import cli, io as dio

    tier = tmp_path / "tier"
    dio.write_study(result.frames, str(tier), fmt="csv")
    (tier / "manifest.json").write_text(
        json.dumps(result.manifest, default=str), encoding="utf-8"
    )
    queue_path = tmp_path / "queue.csv"
    queue = result.review_queue.copy()
    queue["verdict"] = "PASS"
    queue.to_csv(queue_path, index=False)

    final = tmp_path / "final"
    args = cli.build_parser().parse_args(
        [
            "adjudicate",
            str(tier),
            "--queue",
            str(queue_path),
            "-o",
            str(final),
            "--format",
            "csv",
            "--operator",
            "steward@example.com",
        ]
    )
    assert cli.cmd_adjudicate(args) == 0

    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    adj = manifest["adjudication"]
    assert adj["complete"] is True
    assert adj["pending_rows"] == 0
    assert adj["adjudicated_by"] == "steward@example.com"
    assert adj["rows_passed"] == len(queue)
    # and the upstream record survives rather than being replaced
    assert manifest["contract"]["contract_version"]
    assert manifest["inputs"]["checksums_sha256"]


def test_publishing_with_unruled_rows_needs_saying_so(tmp_path, result):
    from deidkit import cli, io as dio

    tier = tmp_path / "tier"
    dio.write_study(result.frames, str(tier), fmt="csv")
    queue_path = tmp_path / "queue.csv"
    result.review_queue.to_csv(queue_path, index=False)  # no verdicts at all

    def run(extra: list[str]) -> int:
        return cli.cmd_adjudicate(
            cli.build_parser().parse_args(
                ["adjudicate", str(tier), "--queue", str(queue_path),
                 "-o", str(tmp_path / "final"), "--format", "csv", *extra]
            )
        )

    assert run([]) == 1, "an unruled queue must not silently publish"
    assert run(["--allow-pending"]) == 0
    manifest = json.loads(
        (tmp_path / "final" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["adjudication"]["published_with_pending_rows"] is True
    assert manifest["adjudication"]["complete"] is False


# ----------------------------------------------------------------------
# what the tool decides, and what it escalates
# ----------------------------------------------------------------------
def test_the_policy_escalates_only_the_judgment_calls():
    """A queue that also carries the mechanical rows stops being read.

    Measured on the synthetic study: 33 flagged rows were 13 study-drug
    mentions, 4 bare contact details and 16 name / facility / in-text-date
    hits. Two thirds had one defensible answer, and the ones that mattered
    were buried among them.
    """
    from deidkit.freetext import DEFAULT_SCREEN_POLICY as P

    # not PHI at all -- a compound name is a blinding matter on its own axis
    assert P["STUDY_DRUG"] == "pass"
    # no clinical reading exists for these
    assert P["EMAIL_ADDRESS"] == "redact"
    assert P["PHONE_NUMBER"] == "redact"
    # only a person knows whether the name is the investigator or a relative
    assert P["PERSON"] == "queue"
    assert P["FACILITY"] == "queue"
    assert P["DATE_IN_TEXT"] == "queue"


def test_an_unknown_entity_type_is_escalated_not_auto_redacted():
    """A detector upgrade must not silently acquire an auto-decision."""
    from deidkit.freetext import Finding, _row_decision, resolve_policy

    verdict, _ = _row_decision(
        [
            Finding(
                entity_type="SOMETHING_NEW",
                start=0,
                end=3,
                text="abc",
                score=0.9,
                detector="test",
            )
        ],
        resolve_policy(),
    )
    assert verdict == "", "an unrecognised entity type must go to a human"


def test_a_mixed_row_goes_to_a_human_rather_than_being_half_handled():
    """'... by Dr Almeida, see fax 617-555-0198' holds one mechanical finding
    and one judgment call. Auto-redacting the phone number before a person
    sees the name would hide the part that needed them."""
    from deidkit.freetext import Finding, _row_decision, resolve_policy

    verdict, _ = _row_decision(
        [
            Finding(
                entity_type="PHONE_NUMBER", start=0, end=3, text="617",
                score=0.9, detector="test",
            ),
            Finding(
                entity_type="PERSON", start=5, end=9, text="Alma",
                score=0.9, detector="test",
            ),
        ],
        resolve_policy(),
    )
    assert verdict == ""


def test_redaction_is_span_level_so_the_sentence_survives():
    """A reviewer typing REDACT must not lose the clinical content.

    Replacing the whole cell would take the drug, the indication and the
    route along with the fax number that was the actual problem.
    """
    from deidkit.freetext import Finding, redact_all_spans

    text = "Amoxicillin prescribed by GP, see fax 617-555-0198"
    out = redact_all_spans(
        text,
        [
            Finding(
                entity_type="PHONE_NUMBER",
                start=38,
                end=50,
                text="617-555-0198",
                score=0.9,
                detector="test",
            )
        ],
    )
    assert out.startswith("Amoxicillin prescribed by GP")
    assert "617-555-0198" not in out
    assert "<PHONE_NUMBER>" in out


def test_the_queue_puts_the_human_rows_first(result):
    """Ordering by confidence alone put a study-drug mention above a person's
    name, which is backwards: the top of the sheet should be the decisions
    only a person can make."""
    verdicts = result.review_queue["verdict"].astype(str).str.strip()
    if not (verdicts == "").any() or (verdicts != "").sum() == 0:
        pytest.skip("this study produced no mixed queue")
    first_decided = (verdicts != "").idxmax()
    last_blank = (verdicts == "")[::-1].idxmax()
    assert first_decided > last_blank, "pre-decided rows appear above human rows"


def test_a_prefilled_verdict_always_says_where_it_came_from(result):
    """A pre-filled decision with no provenance reads as a human's."""
    q = result.review_queue
    filled = q[q["verdict"].astype(str).str.strip() != ""]
    assert not filled.empty
    assert (filled["verdict_source"].astype(str).str.strip() != "").all()
    assert (q["reviewer"].astype(str).str.strip() == "").all()


# ----------------------------------------------------------------------
# the approval digest
# ----------------------------------------------------------------------
def test_adding_an_optional_parameter_does_not_invalidate_approvals(contract):
    """Discovered the hard way: adding one optional field to FieldRule changed
    the digest of every contract in existence and invalidated every steward's
    sign-off across every study -- on an upgrade that altered the behaviour of
    none of them. A digest covers what the rules DO."""
    explicit = contract.model_dump(mode="json")
    for dom in explicit["domains"]:
        for f in dom["fields"]:
            f["screen_policy"] = None  # the default, spelled out
    assert Contract.model_validate(explicit).rules_digest() == contract.rules_digest()


def test_an_old_scheme_approval_is_told_apart_from_a_tampered_one(contract):
    from deidkit.contract import Approval

    signed = contract.model_copy(
        update={
            "approval": Approval(
                approved_by="steward@example.com",
                approved_at="2026-01-01T00:00:00+00:00",
                rules_fingerprint=contract.rules_digest(),
                digest_scheme=contract.DIGEST_SCHEME,
            )
        }
    )
    payload = signed.model_dump(mode="json")
    payload["approval"]["digest_scheme"] = 1
    with pytest.raises(ValidationError, match="older version"):
        Contract.model_validate(payload)

    payload["approval"]["digest_scheme"] = contract.DIGEST_SCHEME
    payload["approval"]["rules_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="edited after it was approved"):
        Contract.model_validate(payload)
