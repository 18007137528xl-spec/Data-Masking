"""Columns the profiler has actually checked are settled, not sent to review.

A review queue of 80 columns out of 110 is not a review, it is the profiler
handing its job back. A row belongs in front of a person only when the tool
cannot settle it: a guess in the risky direction that the values did not
confirm. Everything below is either the safe direction (drop, shift, dummy)
or a retain the values were checked for and passed.
"""

from __future__ import annotations

import pandas as pd

from deidkit.profile import _name_like, draft_contract

N = 40
FILE = "EDC_VS_RAWDATA_BDM-AI-2025-001"


def _study(**cols: list[str]) -> dict[str, pd.DataFrame]:
    subjects = [f"1001-{i % 8 + 1:03d}" for i in range(N)]
    dm = pd.DataFrame({
        "SUBJECT": [f"1001-{i:03d}" for i in range(1, 9)],
        "SITE": ["1001"] * 8,
        "SEX": list("MFMFMFMF"),
    })
    frame = pd.DataFrame({"SUBJECT": subjects, **cols})
    return {FILE: frame, "EDC_DM_RAWDATA_BDM-AI-2025-001": dm}


def _cycle(values: list[str]) -> list[str]:
    return [values[i % len(values)] for i in range(N)]


def _conf(frames, col, **kw):
    _, sugg = draft_contract(frames, source="s", raw_edc=True, **kw)
    return sugg[FILE][col]


# --- numbers ---------------------------------------------------------------
def test_measurements_checked_across_every_value_are_settled():
    s = _conf(_study(ZQ=_cycle(["72.5", "68.1", "80", "91.3", "55.0"])), "ZQ")
    assert s.confidence == "high"
    assert s.rule.treatment.value == "retain"


def test_numbers_that_could_be_phones_or_ids_still_go_to_review():
    phones = _cycle(["13812345678", "13987654321", "13511112222"])
    assert _conf(_study(ZQ=phones), "ZQ").confidence == "medium"


def test_numbers_that_could_be_years_are_never_settled_as_retained():
    # The date reader usually claims a column of years first and shifts it.
    # Whatever claims it, it must not be published as-is without a person.
    years = _cycle(["1961", "1975", "1980", "1955"])
    for raw in (True, False):
        _, sugg = draft_contract(_study(ZQ=years), source="s", raw_edc=raw)
        s = sugg[FILE]["ZQ"]
        assert not (s.confidence == "high" and s.rule.treatment.value == "retain")


def test_numbers_written_like_record_numbers_still_go_to_review():
    codes = [f"{1000 + i}" for i in range(20)] * 2
    assert _conf(_study(ZQ=codes), "ZQ").confidence == "medium"
    padded = _cycle(["001", "002", "003"])
    assert _conf(_study(ZQ=padded), "ZQ").confidence == "medium"


# --- pick-lists ------------------------------------------------------------
def test_a_pick_list_with_every_value_checked_is_settled():
    for values in (["Y", "N"], ["Not Done", "Done"], ["轻度", "中度", "重度"],
                   ["Mild", "Moderate", "Severe"], ["Left Arm", "Right Arm"]):
        s = _conf(_study(ZQ=_cycle(values)), "ZQ")
        assert s.confidence == "high", values


def test_a_pick_list_of_people_is_not_settled():
    for values in (["张伟", "李娜", "王芳"], ["欧阳明", "陈静"],
                   ["Smith, John", "Chen, Wei"], ["John Smith", "Mary Jones"],
                   ["北京协和医院", "上海瑞金医院"]):
        s = _conf(_study(ZQ=_cycle(values)), "ZQ")
        assert s.confidence != "high", values


def test_name_check_counts_names_and_passes_answers():
    assert _name_like(["张伟", "Y", "Not Done", "Smith, John"]) == 2
    assert _name_like(["Y", "N", "UNKNOWN", "NOT DONE", "Severe"]) == 0


# --- the safe direction, or nothing to decide --------------------------------
def test_bookkeeping_columns_are_settled():
    frames = _study(
        FOLDERNAME=_cycle(["Screening", "Week 1"]),
        RECORDDATE=_cycle(["01JAN2025", "02JAN2025"]),
        VSTIM=_cycle(["08:30", "14:15"]),
        AGEU=_cycle(["YEARS"]),
    )
    _, sugg = draft_contract(frames, source="s", raw_edc=True)
    for col in ("FOLDERNAME", "RECORDDATE", "VSTIM", "AGEU"):
        assert sugg[FILE][col].confidence == "high", col


def test_free_text_replaced_with_dummy_is_settled():
    notes = [f"patient reported feeling unwell on visit {i} after lunch"
             for i in range(N)]
    s = _conf(_study(ZQ=notes), "ZQ", dummy_text=True)
    assert s.rule.treatment.value == "dummy_text"
    assert s.confidence == "high"
