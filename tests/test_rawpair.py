"""The raw -> SDTM training pair.

The deliverable here is not one de-identified directory, it is two that still
correspond. A model learning to derive SDTM from raw EDC reads the raw record
as the input and the SDTM record as the label, so a de-identification that
breaks the correspondence produces a corpus that teaches something untrue --
and every individual file still looks perfectly well-formed, which is why
these properties need tests rather than inspection.

What is asserted, and what breaks if it is not:

* the written format survives the shift -- otherwise the raw side arrives
  pre-normalised and the format conversion, the single most mechanical thing
  the model has to learn, is absent from the corpus
* both sides move by the same offset -- otherwise raw day X no longer
  corresponds to SDTM day X
* partial dates keep their granularity and agree across the pair -- otherwise
  ``Mar-2015`` maps to a different month than ``2015-03`` did
* an ambiguous day/month order halts instead of guessing
* an unparsed value is never published silently
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deidkit import Contract, DeidPipeline, Vault, load_study
from deidkit import rawdates
from deidkit.contract import DomainContract, FieldRule, Treatment
from deidkit.pipeline import ContractMismatch
from deidkit.profile import draft_contract, profile_column
from deidkit.transforms import shift_dates, shift_partial

ROOT = Path(__file__).resolve().parents[1]
MK_SDTM = ROOT / "scripts" / "make_synthetic_study.py"
MK_RAW = ROOT / "scripts" / "make_synthetic_raw.py"


# ----------------------------------------------------------------------
# fixtures: one synthetic study, and the raw extract it came from
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def pair_dirs(tmp_path_factory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("pair")
    sdtm, raw = base / "sdtm", base / "raw"
    subprocess.run(
        [sys.executable, str(MK_SDTM), str(sdtm)], check=True, capture_output=True
    )
    subprocess.run(
        [sys.executable, str(MK_RAW), str(sdtm), str(raw)],
        check=True,
        capture_output=True,
    )
    return sdtm, raw


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


@pytest.fixture()
def published(pair_dirs, vault):
    """Both sides through the pipeline, against ONE vault."""
    sdtm_dir, raw_dir = pair_dirs
    sdtm_frames, sdtm_sums = load_study(sdtm_dir)
    raw_frames, raw_sums = load_study(raw_dir)

    sdtm_contract, _ = draft_contract(
        sdtm_frames,
        source="TEST-SDTM",
        sdtm_conformant=True,
        blind_treatment=True,
        subject_id_template="{STUDYID}-US-{value}",
    )
    raw_contract, _ = draft_contract(
        raw_frames,
        source="TEST-RAW",
        raw_edc=True,
        blind_treatment=True,
        join_key_template="TIG-2026-001-US-{SUBJECT}",
    )
    sdtm_out = DeidPipeline(sdtm_contract, vault, operator="pytest").run(
        sdtm_frames, checksums=sdtm_sums
    )
    raw_out = DeidPipeline(raw_contract, vault, operator="pytest").run(
        raw_frames, checksums=raw_sums
    )
    return sdtm_out, raw_out


# ----------------------------------------------------------------------
# unit: parsing and rendering
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected_template",
    [
        ("2025-03-19", "{y}-{m}-{d}"),
        ("2025/03/19", "{y}/{m}/{d}"),
        ("19-Mar-2025", "{d}-{mon}-{y}"),
        ("19MAR2025", "{d}{MON}{y}"),
        ("19 Mar 2025", "{d} {mon} {y}"),
        ("Mar-2025", "{mon}-{y}"),
        ("2025-03", "{y}-{m}"),
        ("2025", "{y}"),
    ],
)
def test_written_form_is_remembered(value, expected_template):
    p = rawdates.parse(value, "dmy")
    assert p is not None
    assert p.template == expected_template


def test_ambiguous_numeric_is_refused_rather_than_guessed():
    # 03/04/2025 is 3 April or 4 March. Guessing gives a date in the wrong
    # month that still looks like a date, which is worse than refusing.
    assert rawdates.parse("03/04/2025") is None
    assert rawdates.parse("03/04/2025", "dmy").month == 4
    assert rawdates.parse("03/04/2025", "mdy").month == 3


def test_order_is_inferred_only_when_the_data_proves_it():
    assert rawdates.infer_order(pd.Series(["19/03/2025", "05/06/2025"])) == "dmy"
    assert rawdates.infer_order(pd.Series(["03/19/2025", "06/05/2025"])) == "mdy"
    # Nothing above 12 anywhere: genuinely undecidable from the column.
    assert rawdates.infer_order(pd.Series(["03/04/2025", "05/06/2025"])) == "unknown"
    # No ambiguous forms at all -- reported as such, not as an inferred order.
    assert rawdates.infer_order(pd.Series(["19MAR2025"])) == "unambiguous"


def test_a_column_of_four_digit_numbers_is_not_a_date_column():
    # A bare year is a valid date, so counting year-only hits equally would
    # class lab values as dates and "shift" them.
    numbers = pd.Series(["1024", "2011", "3500", "1999"] * 20, name="LBRESULT")
    assert not profile_column(numbers).looks_raw_date


def test_a_mostly_partial_column_is_still_a_date_column():
    # Medical-history onset is year-only about half the time. Requiring
    # precision everywhere would report this as free text and publish it
    # untouched.
    onsets = pd.Series(
        ["Jul-2001", "2001", "15-Mar-2010", "1998", "Feb-2015"] * 20, name="MH_ONSET"
    )
    assert profile_column(onsets).looks_raw_date


# ----------------------------------------------------------------------
# unit: shifting
# ----------------------------------------------------------------------
def test_shift_preserves_each_value_s_own_format(vault):
    values = pd.Series(
        ["19/03/2025", "01/04/2025", "19-Mar-2025", "19MAR2025", "2025-03-19"]
    )
    subjects = pd.Series(["S1"] * 5)
    shifted = rawdates.shift_preserving_format(values, subjects, vault).values
    assert shifted.iloc[0].count("/") == 2 and len(shifted.iloc[0]) == 10
    assert shifted.iloc[2][2] == "-" and shifted.iloc[2][3:6].isalpha()
    assert shifted.iloc[3][2:5].isupper()
    assert shifted.iloc[4].count("-") == 2 and shifted.iloc[4][:2] == "20"
    # and none of them is still the original date
    assert not any(a == b for a, b in zip(values, shifted))


def test_partial_dates_keep_their_granularity_through_a_shift(vault):
    values = pd.Series(["Mar-2025", "2025"])
    shifted = rawdates.shift_preserving_format(
        values, pd.Series(["S1", "S1"]), vault
    ).values
    assert rawdates.parse(shifted.iloc[0], "dmy").granularity == "month"
    assert rawdates.parse(shifted.iloc[1], "dmy").granularity == "year"


def test_both_sides_of_a_partial_date_land_on_the_same_value(vault):
    """``Mar-2015`` on the raw side and ``2015-03`` on the SDTM side are the
    same date written twice. If the two paths shifted them differently, every
    partial date in the corpus would teach a mapping that is not true."""
    subj = pd.Series(["S1"])
    for raw_form, iso_form in [
        ("Mar-2015", "2015-03"),
        ("2015", "2015"),
        ("15-Mar-2015", "2015-03-15"),
    ]:
        r = rawdates.shift_preserving_format(
            pd.Series([raw_form]), subj, vault
        ).values.iloc[0]
        s = shift_dates(pd.Series([iso_form]), subj, vault).iloc[0]
        rp, sp = rawdates.parse(r, "dmy"), rawdates.parse(s, "dmy")
        assert (rp.year, rp.month, rp.day) == (sp.year, sp.month, sp.day), raw_form


def test_partial_shift_never_invents_precision():
    y, m = shift_partial(2015, 3, 400)
    assert (y, m) == (2016, 4)
    y, m = shift_partial(2015, None, 400)
    assert m is None


def test_unparsed_values_are_counted_and_sampled(vault):
    values = pd.Series(["19/03/2025", "03-15-25", "sometime in spring"])
    shift = rawdates.shift_preserving_format(values, pd.Series(["S1"] * 3), vault)
    assert shift.report["unparsed"] == 2
    assert shift.report["passed_through"] == 2
    assert shift.passed_through.tolist() == [False, True, True]
    # passed through unchanged, not nulled
    assert shift.values.iloc[1] == "03-15-25"


def test_reused_offsets_are_distinguished_from_new_ones(vault):
    subj = pd.Series(["S1", "S2"])
    v = pd.Series(["19/03/2025", "20/03/2025"])
    first = rawdates.shift_preserving_format(v, subj, vault)
    assert first.report["offset_keys_reused"] == 0
    second = rawdates.shift_preserving_format(v, subj, vault)
    # The signal that the other side of the pair used these same keys.
    assert second.report["offset_keys_reused"] == 2


# ----------------------------------------------------------------------
# contract-level guarantees
# ----------------------------------------------------------------------
def _one_column_contract(**rule_kw) -> Contract:
    return Contract(
        contract_version="t",
        source="t",
        tier="deidentified",
        anchor={"domain": "RAW", "subject_column": "SUBJECT", "date_column": "DT"},
        risk={"domain": "RAW", "k_target": 1},
        domains=[
            DomainContract(
                name="RAW",
                subject_key="SUBJECT",
                fields=[
                    FieldRule(column="SUBJECT", treatment=Treatment.RETAIN),
                    FieldRule(
                        column="DT",
                        treatment=Treatment.DATE_SHIFT_RAW,
                        entity="subject",
                        **rule_kw,
                    ),
                ],
            )
        ],
    )


def test_an_unparsed_value_halts_the_run_by_default(vault):
    frame = pd.DataFrame(
        {"SUBJECT": ["S1", "S2"], "DT": ["19/03/2025", "spring 2025"]}
    )
    pipe = DeidPipeline(_one_column_contract(), vault, operator="pytest")
    with pytest.raises(ContractMismatch, match="not shifted"):
        pipe.run({"RAW": frame})


def test_an_ambiguous_order_halts_the_run(vault):
    # Every component <= 12, so the column cannot settle its own order.
    frame = pd.DataFrame({"SUBJECT": ["S1", "S2"], "DT": ["03/04/2025", "05/06/2025"]})
    pipe = DeidPipeline(_one_column_contract(), vault, operator="pytest")
    with pytest.raises(ContractMismatch, match="date_order"):
        pipe.run({"RAW": frame})


def test_a_declared_order_lets_the_ambiguous_column_through(vault):
    frame = pd.DataFrame({"SUBJECT": ["S1", "S2"], "DT": ["03/04/2025", "05/06/2025"]})
    pipe = DeidPipeline(
        _one_column_contract(date_order="dmy"), vault, operator="pytest"
    )
    out = pipe.run({"RAW": frame}).frames["RAW"]
    assert out["DT"].notna().all()
    assert not out["DT"].isin(frame["DT"]).any()


def test_on_unparsed_redact_nulls_only_the_unparsed(vault):
    frame = pd.DataFrame(
        {"SUBJECT": ["S1", "S2"], "DT": ["19/03/2025", "spring 2025"]}
    )
    pipe = DeidPipeline(
        _one_column_contract(on_unparsed="redact"), vault, operator="pytest"
    )
    out = pipe.run({"RAW": frame}).frames["RAW"]
    assert out["DT"].isna().sum() == 1
    assert out["DT"].notna().sum() == 1


def test_a_date_column_retained_blocks_a_deidentified_tier():
    # The tier check reads is_date, so it catches a raw column with no --DTC
    # suffix to give it away.
    with pytest.raises(Exception, match="real calendar values"):
        Contract(
            contract_version="t",
            source="t",
            tier="deidentified",
            anchor={
                "domain": "RAW",
                "subject_column": "SUBJECT",
                "date_column": "DT",
            },
            risk={"domain": "RAW", "k_target": 1},
            domains=[
                DomainContract(
                    name="RAW",
                    fields=[
                        FieldRule(
                            column="VISIT_DT",
                            treatment=Treatment.RETAIN,
                            is_date=True,
                        )
                    ],
                )
            ],
        )


def test_the_manifest_does_not_carry_unshifted_date_samples(vault):
    frame = pd.DataFrame(
        {"SUBJECT": ["S1", "S2"], "DT": ["19/03/2025", "spring 2025"]}
    )
    pipe = DeidPipeline(
        _one_column_contract(on_unparsed="pass"), vault, operator="pytest"
    )
    result = pipe.run({"RAW": frame})
    raw_report = result.manifest["transformation"]["RAW"]["raw_dates"]["DT"]
    assert raw_report["passed_through"] == 1
    # The samples are real dates. The manifest ships inside the published
    # tier, so they must not be in it.
    assert "passed_through_samples" not in raw_report


# ----------------------------------------------------------------------
# the pair itself
# ----------------------------------------------------------------------
def test_the_two_sides_join_on_one_subject_identifier(published):
    sdtm, raw = published
    dm = sdtm.frames["DM"]
    demog = raw.frames["DEMOG"]
    assert len(demog) == len(dm)
    # USUBJID is composed over the surrogate; SUBJECT is the bare surrogate.
    joined = dm["STUDYID"] + "-US-" + demog["SUBJECT"]
    assert joined.equals(dm["USUBJID"])


def test_every_paired_date_still_maps_after_conversion(published):
    sdtm, raw = published
    ae_s = sdtm.frames["AE"]
    ae_r = raw.frames["AE_LOG"]
    assert len(ae_s) == len(ae_r)
    for dmy, iso in zip(ae_r["AE_START"], ae_s["AESTDTC"]):
        if not isinstance(dmy, str) or not dmy:
            continue
        d, m, y = dmy.split("/")
        assert f"{y}-{m}-{d}" == iso


def test_the_raw_side_publishes_no_iso_dates(published):
    """If the raw side came out ISO, the conversion the model is meant to
    learn is already done for it."""
    _, raw = published
    ae_r = raw.frames["AE_LOG"]
    assert not ae_r["AE_START"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").any()


def test_no_subject_keeps_their_own_enrolment_date(published, pair_dirs):
    """Row-wise, not set-wise, and the distinction is the point.

    With 120 subjects and offsets of six to eighteen months, one subject's
    shifted date will sometimes equal a *different* subject's real date. That
    is not a leak -- nothing links the value back to the person -- and a
    set-membership assertion would fail on it while proving nothing. What has
    to hold is that no row still carries its own subject's date.
    """
    sdtm_dir, _ = pair_dirs
    original = pd.read_csv(sdtm_dir / "dm.csv", dtype=str)
    sdtm, raw = published
    for value, iso_out, dmy_out in zip(
        original["RFSTDTC"], sdtm.frames["DM"]["RFSTDTC"], raw.frames["DEMOG"]["ENROL_DT"]
    ):
        if not isinstance(value, str) or not value:
            continue
        y, m, d = value.split("-")
        assert iso_out != value
        assert dmy_out != f"{d}/{m}/{y}"


def test_day_month_order_is_inferred_per_column_not_per_study(published):
    """The AE form is d/m/y and the vitals form is m/d/y -- the same shape
    with different meaning. One study-wide setting would corrupt one of
    them."""
    _, raw = published
    ae = raw.frames["AE_LOG"]["AE_START"].dropna()
    vs = raw.frames["VITALS"]["VISIT_DT"].dropna()
    assert any(int(v.split("/")[0]) > 12 for v in ae)
    assert any(int(v.split("/")[1]) > 12 for v in vs)


# ----------------------------------------------------------------------
# dates that arrive through Excel
# ----------------------------------------------------------------------
def test_a_datetime_is_parsed_and_its_time_survives():
    """An Excel cell typed as a datetime reads back as
    "2025-04-20 00:00:00", and the raw patterns accepted only a bare date --
    so 2853 real dates counted as UNPARSED and halted the run."""
    for value, tail in [
        ("2025-04-20 00:00:00", "00:00:00"),
        ("2025-04-20T14:30:00", "14:30:00"),
        ("2025-04-20 14:30", "14:30"),
    ]:
        p = rawdates.parse(value, "dmy")
        assert p is not None, value
        assert (p.year, p.month, p.day) == (2025, 4, 20)
        assert p.tail == tail


def test_a_numeric_date_with_a_clock_is_still_ambiguous():
    """A time tells you nothing about day/month order. Naming those groups
    d and m would hardcode d/m/y and put 03/04/2025 08:15 in the wrong month
    -- the exact mistake this module exists to refuse."""
    assert rawdates.parse("03/04/2025 08:15:00") is None
    assert rawdates.parse("03/04/2025 08:15:00", "dmy").month == 4
    assert rawdates.parse("03/04/2025 08:15:00", "mdy").month == 3


def test_shifting_a_datetime_keeps_the_time_of_day(vault):
    values = pd.Series(["2025-04-20 14:30:00", "2025-04-20 00:00:00"])
    out = rawdates.shift_preserving_format(
        values, pd.Series(["S1", "S1"]), vault
    ).values
    assert out.iloc[0].endswith(" 14:30:00")
    assert out.iloc[1].endswith(" 00:00:00")
    assert not any(a == b for a, b in zip(values, out))


def test_excel_midnight_is_dropped_per_column_on_evidence(tmp_path):
    """The clock Excel adds to a date-only field is Excel's, not the
    coordinator's -- nobody recorded midnight. But one real time anywhere in
    the column and the whole column keeps its times, because then the field
    genuinely holds times."""
    pytest.importorskip("openpyxl")
    from deidkit.io import read_table

    path = tmp_path / "ex.xlsx"
    pd.DataFrame(
        {
            "EXSTDAT": pd.to_datetime(["2025-04-20", "2025-05-19"]),
            "EXSTDTM": pd.to_datetime(
                ["2025-04-20 14:30:00", "2025-05-19 00:00:00"]
            ),
        }
    ).to_excel(path, index=False)
    frame = read_table(path)
    assert frame["EXSTDAT"].iloc[0] == "2025-04-20"
    assert frame["EXSTDTM"].iloc[0] == "2025-04-20 14:30:00"
    # the midnight in a column that has real times is data, not an artifact
    assert frame["EXSTDTM"].iloc[1] == "2025-05-19 00:00:00"


def test_pandas_own_missing_markers_count_as_missing():
    """One dtype change upstream was enough to turn empty cells into a halt.

    "isinstance(v, float) and isna(v)" covered numpy's NaN and nothing else,
    so pd.NA and pd.NaT fell through to str() -- which renders them "<NA>"
    and "NaT" -- and 63 empty cells were reported as dates in an
    unrecognised format.
    """
    import numpy as np

    from deidkit.rawdates import _blank
    from deidkit.transforms import parse_dtc

    for missing in (None, np.nan, pd.NA, pd.NaT, "", "  ", "NA", "<NA>", "NaT", "N/A"):
        assert _blank(missing), repr(missing)
        assert parse_dtc(missing).year is None, repr(missing)
    assert not _blank("2025-04-20")
    assert parse_dtc("2025-04-20").year == 2025


def test_a_string_column_with_pd_na_shifts_without_halting(vault):
    """The exact shape that failed: a StringDtype column holding pd.NA."""
    values = pd.Series(
        ["2025-04-20", pd.NA, "2025-05-19", pd.NA], dtype="string"
    )
    shift = rawdates.shift_preserving_format(
        values, pd.Series(["S1"] * 4), vault
    )
    assert shift.report["passed_through"] == 0, shift.report
    assert shift.report["shifted"] == 2
    assert shift.values.isna().sum() == 2
