"""Dummy text: every free-text value replaced, shape kept, same text -> same dummy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deidkit import DeidPipeline, Vault, load_study
from deidkit.contract import Treatment
from deidkit.dummytext import dummy_value
from deidkit.profile import draft_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


@pytest.fixture(scope="module")
def pair_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("dummy")
    sdtm, raw = base / "sdtm", base / "raw"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_study.py"),
                    str(sdtm)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_raw.py"),
                    str(sdtm), str(raw)], check=True, capture_output=True)
    return sdtm, raw


def _shape(s: str) -> str:
    return "".join(
        "A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else c
        for c in s
    )


def test_the_shape_survives_and_the_text_does_not(vault):
    text = "Severe headache, 2 days (Dr. Li)"
    d = dummy_value(text, vault)
    assert _shape(d) == _shape(text)
    assert d != text
    # no word survives. Short ones are left out of the check: a random draw
    # of two letters lands on "Li" again once in 676 runs, which is chance,
    # not a leak -- and a test that fails by chance teaches people to ignore it.
    words = lambda x: {w.strip(",.()") for w in x.lower().split() if len(w.strip(",.()")) >= 4}
    assert not words(d) & words(text)


def test_the_same_text_gets_the_same_dummy_whatever_its_case(vault):
    a = dummy_value("Headache", vault)
    b = dummy_value("HEADACHE", vault)
    c = dummy_value("  Headache ", vault)
    assert a.upper() == b and b.isupper() and a[0].isupper() and a[1:].islower()
    assert c == "  " + a + " "
    assert dummy_value("Nausea", vault) != dummy_value("Vomiting", vault)


def test_chinese_text_is_replaced_with_chinese_text(vault):
    d = dummy_value("患者头痛3天", vault)
    assert len(d) == 6 and d != "患者头痛3天"
    assert d[4].isdigit()
    assert all(0x4E00 <= ord(ch) <= 0x9FA5 for ch in d[:4] + d[5])


def test_blanks_stay_blank(vault):
    assert dummy_value("", vault) == ""
    assert dummy_value(None, vault) is None
    assert pd.isna(dummy_value(float("nan"), vault))


def test_every_free_text_column_is_dummied_and_nothing_is_queued(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    contract, _ = draft_contract(frames, source="T", sdtm_conformant=True,
                                 dummy_text=True)
    rules = {(d.name, f.column): f for d in contract.domains for f in d.fields}
    for col in [("AE", "AETERM"), ("MH", "MHTERM"), ("CM", "CMTRT")]:
        assert rules[col].treatment is Treatment.DUMMY_TEXT, col
    assert not any(f.treatment is Treatment.SCREEN_FREETEXT for f in rules.values())
    assert not next(d for d in contract.domains if d.name == "AE").retained_in_full

    out = DeidPipeline(contract, vault, operator="pytest").run(frames, checksums=sums)
    assert out.review_queue.empty
    ae_in, ae_out = frames["AE"], out.frames["AE"]
    assert (ae_out["AETERM"] != ae_in["AETERM"]).all()
    assert ae_out["AEDECOD"].equals(ae_in["AEDECOD"])  # coded content stays


def test_both_sides_of_a_pair_get_the_same_dummy(pair_dirs, vault):
    sdtm_frames, s_sums = load_study(pair_dirs[0])
    raw_frames, r_sums = load_study(pair_dirs[1])
    sc, _ = draft_contract(sdtm_frames, source="T", sdtm_conformant=True,
                           dummy_text=True)
    rc, _ = draft_contract(raw_frames, source="T", raw_edc=True, dummy_text=True,
                           join_key_template="TIG-2026-001-US-{SUBJECT}")
    s = DeidPipeline(sc, vault, operator="pytest").run(sdtm_frames, checksums=s_sums)
    r = DeidPipeline(rc, vault, operator="pytest").run(raw_frames, checksums=r_sums)
    a, b = r.frames["AE_LOG"]["AE_TERM"], s.frames["AE"]["AETERM"]
    assert (a.str.upper() == b.str.upper()).all()
