"""The study's own number: one fixed surrogate, the same in data and protocol."""

from __future__ import annotations

import pandas as pd
import pytest

from deidkit import Vault
from deidkit import profile as pf
from deidkit.contract import Treatment


@pytest.fixture()
def vault(tmp_path):
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


def _frame():
    return pd.DataFrame({
        "STUDY": ["MDV3800-01"] * 6, "PROJECT": ["MDV3800-01"] * 6,
        "Subject": [f"1001-00{i}" for i in range(6)], "SiteNumber": ["1001"] * 6,
        "AESTDAT": ["01/JAN/2021"] * 6,
    })


def test_raw_study_columns_get_the_study_surrogate():
    _, sug = pf.draft_contract({"AE": _frame()}, source="x", raw_edc=True)
    for col in ("STUDY", "PROJECT"):
        s = sug["AE"][col]
        assert s.rule.treatment is Treatment.SURROGATE_ID, col
        assert s.rule.entity == "studyid" and s.confidence == "high", col


def test_sdtm_studyid_gets_the_study_surrogate():
    ae = pd.DataFrame({"STUDYID": ["MDV3800-01"] * 6, "DOMAIN": ["AE"] * 6,
                       "USUBJID": [f"MDV3800-01-1001-00{i}" for i in range(6)],
                       "AESTDTC": ["2021-01-01"] * 6})
    _, sug = pf.draft_contract({"AE": ae}, source="x")
    s = sug["AE"]["STUDYID"]
    assert s.rule.treatment is Treatment.SURROGATE_ID and s.rule.entity == "studyid"
    assert sug["AE"]["DOMAIN"].rule.treatment is Treatment.RETAIN


def test_the_same_number_gets_the_same_surrogate_in_data_and_protocol(vault):
    from deidkit import protocol as pr

    data_side = vault.surrogate_for("studyid", "MDV3800-01", preserve_format=True)
    term = pr.Term("MDV3800-01", "study_id")
    assert pr.propose(term, vault) == data_side
    assert data_side != "MDV3800-01" and len(data_side) == len("MDV3800-01")


def test_raw_cm_columns_frequency_indication_full_text_and_drug_names():
    n = 8
    cm = pd.DataFrame({
        "Subject": [f"1001-00{i}" for i in range(n)], "SiteNumber": ["1001"] * n,
        "CMTRT": ["ibuprofen", "paracetamol", "omeprazole", "aspirin", "metformin",
                  "insulin", "ibuprofen", "amlodipine"],
        "CMTRT_FUL": ["ibuprofen 200mg", "paracetamol", "omeprazole", "aspirin",
                      "metformin", "insulin glargine", "ibuprofen", "amlodipine"],
        "DRUG_SYN_TMS": ["IBUPROFEN", "PARACETAMOL", "OMEPRAZOLE", "ACETYLSALICYLIC ACID",
                         "METFORMIN", "INSULIN GLARGINE", "IBUPROFEN", "AMLODIPINE"],
        "OTH_FRQ_FUL": ["every other day", "", "", "", "once weekly", "", "", ""],
        "CMDOSFRQ": ["QD", "BID", "QD", "QD", "BID", "QD", "PRN", "QD"],
        "XX_INDC": ["pain", "fever", "GERD", "prophylaxis", "diabetes", "diabetes",
                    "pain", "hypertension"],
        "DRUGSTDAT": ["01/JAN/2021"] * n,
    })
    _, sug = pf.draft_contract({"CM": cm}, source="x", raw_edc=True, blind_treatment=True)
    s = sug["CM"]
    assert s["CMTRT_FUL"].rule.treatment is s["CMTRT"].rule.treatment
    assert s["DRUG_SYN_TMS"].rule.treatment is Treatment.LABEL_MAP
    assert s["DRUG_SYN_TMS"].rule.entity == "treatment_product"
    for col in ("OTH_FRQ_FUL", "CMDOSFRQ", "XX_INDC"):
        assert s[col].rule.treatment is Treatment.RETAIN, col
        assert s[col].confidence == "high", col
    assert s["DRUGSTDAT"].rule.treatment is Treatment.DATE_SHIFT_RAW
