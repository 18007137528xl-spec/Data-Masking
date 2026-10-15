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


def test_a_pick_list_of_anything_but_answers_is_not_settled():
    # No name in it, and still not settled: high needs every value to be an
    # ordinary answer, not merely the absence of a name.
    for values in (["Atrial fibrillation", "Gout"], ["甘露", "欧明"],
                   ["SHR-1210", "JS001"]):
        s = _conf(_study(ZQ=_cycle(values)), "ZQ")
        assert s.confidence != "high" or s.rule.treatment.value != "retain", values


# --- people filed under their role ----------------------------------------
def test_staff_names_are_dropped_whatever_the_column_is_called():
    for col in ("INV_NAME", "INVNAM_RAW", "PI_SIGN", "CRCNAME", "ENTERED_BY",
                "研究者签名", "医生姓名"):
        for raw in (True, False):
            _, sugg = draft_contract(_study(**{col: _cycle(["甘露", "欧明"])}),
                                     source="s", raw_edc=raw)
            s = sugg[FILE][col]
            assert s.rule.treatment.value == "drop", (col, raw)
            assert s.confidence == "high", (col, raw)


# --- the study drug --------------------------------------------------------
def _ex(**cols):
    frames = _study(**cols)
    frames["EDC_EX_RAWDATA"] = frames.pop(FILE)
    return frames


def test_product_columns_are_blinded_by_their_values():
    for values in (["SHR-1210", "Placebo"], ["Empagliflozin 10 mg", "Placebo"],
                   ["恩格列净片", "安慰剂"], ["BI 10773", "Empagliflozin"]):
        _, sugg = draft_contract(_ex(IPX=_cycle(values)), source="s",
                                 raw_edc=True, blind_treatment=True)
        s = sugg["EDC_EX_RAWDATA"]["IPX"]
        assert s.rule.treatment.value == "label_map", values
        assert s.rule.entity == "treatment_product"


def test_concomitant_medications_are_not_blinded():
    frames = _study(CMDRUG=_cycle(["Atorvastatin", "Metformin"]))
    frames["EDC_CM_RAWDATA"] = frames.pop(FILE)
    _, sugg = draft_contract(frames, source="s", raw_edc=True,
                             blind_treatment=True)
    assert sugg["EDC_CM_RAWDATA"]["CMDRUG"].rule.treatment.value != "label_map"


def test_one_drug_gets_one_label_whatever_its_case(tmp_path):
    from deidkit import Vault
    from deidkit.transforms import label_map

    v = Vault(tmp_path / "v.db", key=Vault.generate_key(), operator="t")
    a = label_map(pd.Series(["Empagliflozin", "Placebo"]), v,
                  entity="treatment_product", prefix="DRUG", keep_values=["Placebo"])
    b = label_map(pd.Series(["EMPAGLIFLOZIN ", "PLACEBO"]), v,
                  entity="treatment_product", prefix="DRUG", keep_values=["Placebo"])
    assert a[0] == b[0] and a[0].startswith("DRUG ")
    assert list(b)[1] == "PLACEBO"
    assert v.reverse("treatment_product", a[0], justification="t") == "Empagliflozin"
    v.close()


# --- day/month order ---------------------------------------------------------
def test_an_ambiguous_date_column_takes_the_order_its_siblings_prove():
    frames = _study(
        AESTDAT=_cycle(["03/04/2025", "05/06/2025"]),   # never above 12
        AEENDAT=_cycle(["23/04/2025", "15/06/2025"]),   # proves dmy
    )
    _, sugg = draft_contract(frames, source="s", raw_edc=True)
    s = sugg[FILE]["AESTDAT"]
    assert s.confidence == "high"
    assert s.rule.date_order == "dmy"
    assert "other columns" in s.rule.note


def test_an_ambiguous_date_column_alone_still_asks():
    frames = _study(AESTDAT=_cycle(["03/04/2025", "05/06/2025"]))
    _, sugg = draft_contract(frames, source="s", raw_edc=True)
    assert sugg[FILE]["AESTDAT"].confidence == "low"


# --- dictionary coding is the answer, not free text --------------------------
def test_meddra_columns_are_kept_however_they_are_spelled():
    terms = _cycle(["Blood pressure increased", "Upper respiratory tract infection",
                    "Alanine aminotransferase increased", "Headache", "Back pain"])
    names = ["AELLT", "AETERM_LLT", "LLT_NAME", "AELLTNM", "AETERM_LLT_TERM",
             "AEPT", "AETERM_PT", "PT_NAME", "SOC_NAME", "AEHLGT", "LLTNAME_ZH",
             "低位语", "LLT名称", "首选术语"]
    for raw in (True, False):
        _, sugg = draft_contract(_study(**{n: terms for n in names}), source="s",
                                 raw_edc=raw, dummy_text=True)
        for n in names:
            s = sugg[FILE][n]
            assert s.rule.treatment.value == "retain", (n, raw, s.rationale)
            assert s.confidence == "high", n
        # the verbatim term beside them is still free text
        _, sugg = draft_contract(_study(AETERM=terms), source="s", raw_edc=raw,
                                 dummy_text=True)
        assert sugg[FILE]["AETERM"].rule.treatment.value == "dummy_text"


def test_codes_shaped_like_a_compound_are_not_a_drug_on_their_own():
    # CN001 has exactly the shape of SHR-1210. Without a drug name or a
    # placebo beside it, it is a site or a screening number, not a product.
    for col in ("INVSITE", "ZQ"):
        _, sugg = draft_contract(_ex(**{col: _cycle(["CN001", "CN002", "US003"])}),
                                 source="s", raw_edc=True, blind_treatment=True)
        assert sugg["EDC_EX_RAWDATA"][col].rule.treatment.value != "label_map", col


def test_invsite_is_the_site():
    _, sugg = draft_contract(_study(INVSITE=_cycle(["CN001", "CN002"])),
                             source="s", raw_edc=True)
    rule = sugg[FILE]["INVSITE"].rule
    assert rule.treatment.value == "surrogate_id" and rule.entity == "site"
    # beside the protocol's site number it is the redundant one
    _, sugg = draft_contract(
        _study(INVSITE=_cycle(["CN001"]), SiteNumber=_cycle(["101"])),
        source="s", raw_edc=True)
    assert sugg[FILE]["INVSITE"].rule.redundant_with == "SiteNumber"


def test_placebo_is_kept_in_any_case(tmp_path):
    from deidkit import Vault
    from deidkit.transforms import label_map

    v = Vault(tmp_path / "v.db", key=Vault.generate_key(), operator="t")
    out = label_map(pd.Series(["SHR-1210", "PLACEBO", "placebo"]), v,
                    entity="treatment_product", prefix="DRUG", keep_values=["Placebo"])
    assert list(out)[1:] == ["PLACEBO", "placebo"]
    v.close()


# --- meanings padded inside longer names --------------------------------------
def test_names_padded_on_both_sides_are_still_read():
    frames = _study(
        ENDDT_RAW=_cycle(["12/03/2025", "23/04/2025"]),
        STRDT=_cycle(["12/03/2025", "23/04/2025"]),
        AEStrDt=_cycle(["12/03/2025", "23/04/2025"]),
        AE_HLGT_CD=_cycle(["10019231"]),
        MEDDRA_HLGT_TERM=_cycle(["Headaches"]),
        AE_OUT_C=_cycle(["1", "2"]),
        AEOutcome=_cycle(["Recovered", "Fatal"]),
        AE_ACTION_TAKEN=_cycle(["Dose not changed", "Drug withdrawn"]),
    )
    for raw in (True, False):
        _, sugg = draft_contract(frames, source="s", raw_edc=raw)
        got = {c: s for c, s in sugg[FILE].items()}
        for col in ("AE_HLGT_CD", "MEDDRA_HLGT_TERM", "AE_OUT_C", "AEOutcome",
                    "AE_ACTION_TAKEN"):
            assert got[col].rule.treatment.value == "retain", (col, raw)
            assert got[col].confidence == "high", (col, raw)
        for col in ("ENDDT_RAW", "STRDT", "AEStrDt"):
            assert not got[col].rationale.startswith("unrecognised"), (col, raw)
            if raw:
                assert got[col].rule.treatment.value == "date_shift_raw", col
                assert got[col].confidence == "high", col


def test_other_specify_beside_a_coded_field_is_still_free_text():
    _, sugg = draft_contract(
        _study(AE_OUT_OTH=[f"moved to another city {i}" for i in range(N)]),
        source="s", raw_edc=True)
    assert sugg[FILE]["AE_OUT_OTH"].rule.treatment.value != "retain"


def test_pt_name_alone_is_a_patient_not_a_preferred_term():
    _, sugg = draft_contract(_study(PT_NAME=_cycle(["John Smith", "Mary Lee"])),
                             source="s", raw_edc=True)
    assert sugg[FILE]["PT_NAME"].rule.treatment.value == "drop"


def test_start_and_end_dates_in_pieces_are_dropped():
    frames = _study(
        ENDDY_RAW=_cycle(["3", "17", "28"]),
        ENDMO_RAW=_cycle(["MAR", "APR"]),
        STRMO_RAW=_cycle(["1", "12"]),
        STRYY_RAW=_cycle(["2024", "2025"]),
        STRYY_XX=_cycle(["2024", "2025"]),
        ENDDY_XX=_cycle(["3", "17"]),
        AESTRDY_C=_cycle(["5", "9"]),
    )
    for raw in (True, False):
        _, sugg = draft_contract(frames, source="s", raw_edc=raw)
        for col in ("ENDDY_RAW", "ENDMO_RAW", "STRMO_RAW", "STRYY_RAW",
                    "STRYY_XX", "ENDDY_XX", "AESTRDY_C"):
            s = sugg[FILE][col]
            assert not s.rationale.startswith("unrecognised"), (col, raw)
            assert s.rule.treatment.value != "retain", (col, raw, s.rationale)
            if raw:
                assert s.rule.treatment.value == "drop", (col, s.rationale)
                assert s.confidence == "high", col


def test_a_study_day_is_not_a_day_of_the_month():
    _, sugg = draft_contract(_study(ENDDY_X=_cycle(["-3", "45", "112"])),
                             source="s", raw_edc=True)
    assert sugg[FILE]["ENDDY_X"].rule.treatment.value == "retain"


def test_months_are_never_read_as_names(monkeypatch):
    # Presidio's NER calls MAR and APR people or places. Stand in for it.
    from deidkit import profile as prof

    class Eager:
        def detect(self, text):
            from deidkit.freetext import Finding
            return [Finding("PERSON", 0, len(text), text, 0.9, "fake")]

    monkeypatch.setattr(prof, "_DETECTOR", Eager())
    assert prof._identifier_hits(["MAR", "APR", "Jun", "UNK", "12"]) == set()
    assert prof._identifier_hits(["John Smith"]) == {"person"}
    _, sugg = draft_contract(_study(ZQMONTH=_cycle(["MAR", "APR", "March"])),
                             source="s", raw_edc=True)
    assert sugg[FILE]["ZQMONTH"].confidence != "low"


# --- PT: usually MedDRA's preferred term, sometimes the patient ---------------
def _pt_frame(values):
    return pd.DataFrame({"PT": values, "AESEV": _cycle(["MILD", "SEVERE"])})


def test_pt_holding_terms_is_the_preferred_term_on_its_own():
    for values in (_cycle(["Headache", "Back pain"]), _cycle(["HEADACHE", "NAUSEA"]),
                   _cycle(["10019211", "10028813"])):
        for raw in (True, False):
            contract, sugg = draft_contract({FILE: _pt_frame(values)}, source="s",
                                            raw_edc=raw)
            s = sugg[FILE]["PT"]
            assert s.rule.treatment.value == "retain", (values[0], raw, s.rationale)
            assert s.confidence == "high"
            assert contract.domain(FILE).subject_key != "PT"


def test_pt_holding_patient_numbers_is_the_subject():
    contract, sugg = draft_contract(
        {FILE: _pt_frame([f"101-00{i % 8 + 1}" for i in range(N)])},
        source="s", raw_edc=True)
    s = sugg[FILE]["PT"]
    assert s.rule.treatment.value == "surrogate_id" and s.rule.entity == "subject"
    assert s.confidence == "high"
    assert contract.domain(FILE).subject_key == "PT"


def test_pt_that_could_be_either_asks():
    values = _cycle(["Headache", "101-001", "Nausea", "101-002", "x"])
    _, sugg = draft_contract({FILE: _pt_frame(values)}, source="s", raw_edc=True)
    assert sugg[FILE]["PT"].confidence == "medium"


def test_pt_id_is_never_a_dictionary_term():
    _, sugg = draft_contract(_study(PT_ID=_cycle(["1001", "1002"])), source="s",
                             raw_edc=False)
    assert sugg[FILE]["PT_ID"].rationale != "dictionary-coded term"


def test_the_console_saves_before_it_signs():
    from pathlib import Path
    import deidkit

    html = (Path(deidkit.__file__).parent / "console.html").read_text(encoding="utf-8")
    approve = html.index("api('/api/approve'")
    run = html.index("api('/api/run'")
    assert "await flush();" in html[approve - 200:approve]
    assert "await flush();" in html[run - 200:run]
    assert "pending.add" in html  # queued rows are kept, not overwritten
