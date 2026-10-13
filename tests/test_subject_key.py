"""Every file needs to know which column is the subject.

The date offset and the reassignment both look subjects up by it. When the
profiler did not recognise the column, a reviewer marking it surrogate_id /
entity=subject has said which one it is -- and that has to reach the
domain's key, or the run stops on "no subject key" with the fix already on
the page.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deidkit import DeidPipeline, Vault
from deidkit import decisions as dec
from deidkit.contract import Treatment
from deidkit.profile import _raw_subject_key_of, draft_contract
from deidkit.reassign import ReassignError

FILE = "EDC_AE_RAWDATA"


def _ae(subject_col: str) -> pd.DataFrame:
    rows = []
    for s in range(1, 7):
        for seq in (1, 2):
            rows.append({subject_col: f"101-00{s}", "AE_NO": str(seq),
                         "SEVERITY": "MILD" if seq == 1 else "SEVERE",
                         "OUTCOME": "RECOVERED" if s % 2 else "ONGOING"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", [
    "Subject ID", "SUBJECT_NO", "Subject Number", "subj_num", "PATNO",
    "受试者编号", "筛选号",
])
def test_subject_columns_are_found_under_the_names_they_go_by(name):
    contract, sugg = draft_contract({FILE: _ae(name)}, source="T", raw_edc=True,
                                    reassign=True)
    assert contract.domain(FILE).subject_key == name
    assert sugg[FILE][name].rule.entity == "subject"


def test_rave_subject_wins_over_its_database_id():
    cols = ["subjectId", "SiteNumber", "Subject", "SUBJECT_RAW"]
    assert _raw_subject_key_of(cols) == "Subject"
    frame = _ae("Subject")
    frame.insert(0, "subjectId", [str(5000 + i // 2) for i in range(len(frame))])
    contract, sugg = draft_contract({FILE: frame}, source="T", raw_edc=True)
    assert contract.domain(FILE).subject_key == "Subject"
    dup = sugg[FILE]["subjectId"].rule
    assert dup.treatment is Treatment.DROP
    assert dup.redundant_with == "Subject"  # the real column name, not SUBJECT


@pytest.fixture
def vault(tmp_path):
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


def test_a_reviewer_naming_the_subject_column_fixes_the_key(vault):
    frame = _ae("ZQ")  # a name nothing recognises
    frames = {FILE: frame}
    contract, sugg = draft_contract(frames, source="T", raw_edc=True,
                                    reassign=True)
    assert contract.domain(FILE).subject_key is None

    # Unfixed, the run says what to do about it.
    with pytest.raises(ReassignError, match="entity=subject"):
        DeidPipeline(contract, vault, operator="pytest").run(frames, checksums={})

    sheet = dec.build_sheet(contract, sugg)
    sheet["decision"] = "OK"
    row = sheet["column"] == "ZQ"
    sheet.loc[row, ["decision", "decision_treatment", "decision_params"]] = [
        "CHANGE", "surrogate_id", "entity=subject",
    ]
    approved, _ = dec.apply_sheet(sheet, contract, approved_by="t", frames=frames)
    assert approved.domain(FILE).subject_key == "ZQ"
    out = DeidPipeline(approved, vault, operator="pytest").run(frames, checksums={})
    assert out.manifest["reassignment"]["groups"][FILE]["mode"] == "values"


def test_rave_writes_one_site_three_ways_and_one_survives():
    frame = _ae("Subject")
    frame["siteid"] = "11"
    frame["Site"] = "Site A"
    frame["SiteNumber"] = "101"
    _, sugg = draft_contract({FILE: frame}, source="T", raw_edc=True)
    rules = {c: s.rule for c, s in sugg[FILE].items()}
    assert rules["SiteNumber"].treatment is Treatment.SURROGATE_ID
    for col in ("siteid", "Site"):
        assert rules[col].treatment is Treatment.DROP, col
        assert rules[col].redundant_with == "SiteNumber"
