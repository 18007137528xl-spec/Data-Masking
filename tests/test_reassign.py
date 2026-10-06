"""Record reassignment between subjects.

The point is to cut the link between a person and their clinical content; the
constraint is that SDTM must still derive from raw afterwards. So the tests
pin both: records really move, and every interval a derivation reads -- the
study day of an event, the pairing of a raw date with its SDTM date -- is
exactly what it was.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from deidkit import DeidPipeline, Vault, load_study
from deidkit.profile import draft_contract, reassign_group
from deidkit.reassign import move_date

ROOT = Path(__file__).resolve().parents[1]
MK_SDTM = ROOT / "scripts" / "make_synthetic_study.py"
MK_RAW = ROOT / "scripts" / "make_synthetic_raw.py"


@pytest.fixture(scope="module")
def pair_dirs(tmp_path_factory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("reassign")
    sdtm, raw = base / "sdtm", base / "raw"
    subprocess.run([sys.executable, str(MK_SDTM), str(sdtm)], check=True,
                   capture_output=True)
    subprocess.run([sys.executable, str(MK_RAW), str(sdtm), str(raw)],
                   check=True, capture_output=True)
    return sdtm, raw


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


def _iso(v) -> date | None:
    if not isinstance(v, str) or len(v) < 10:
        return None
    return date.fromisoformat(v[:10])


def _sdtm_contract(frames, groups=None):
    contract, _ = draft_contract(
        frames, source="T", sdtm_conformant=True, reassign="subjects",
        subject_id_template="{STUDYID}-US-{value}",
    )
    if groups is not None:
        contract = contract.model_copy(update={"domains": [
            d.model_copy(update={"reassign": d.reassign if d.reassign in groups else None})
            for d in contract.domains
        ]})
    return contract


# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,group",
    [("AE", "AE"), ("EDC_AE_RAWDATA_US_X", "AE"), ("SUPPAE", "AE"),
     ("DM", None), ("EX", None), ("DS", None), ("EDC_LB_X", "LB"), ("demog", None)],
)
def test_which_domains_move(name, group):
    assert reassign_group(name) == group


def test_a_date_moves_in_the_form_it_was_written():
    assert move_date("2025-03-19", 10) == "2025-03-29"
    assert move_date("2025-03-19T08:15", 10) == "2025-03-29T08:15"
    assert move_date("19MAR2025", -19) == "28FEB2025"
    assert move_date("UN-MAR-2025", 31) == "UN-APR-2025"
    assert move_date("not a date", 10) == "not a date"


def test_the_anchor_domain_and_dosing_stay_put(pair_dirs):
    frames, _ = load_study(pair_dirs[0])
    contract = _sdtm_contract(frames)
    by = {d.name: d.reassign for d in contract.domains}
    assert by["DM"] is None and by["EX"] is None
    assert by["AE"] == "AE" and by["LB"] == "LB"


def test_records_move_and_study_days_do_not(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    contract = _sdtm_contract(frames)
    out = DeidPipeline(contract, vault, operator="pytest").run(frames, checksums=sums)

    dm_in, ae_in = frames["DM"], frames["AE"]
    dm_out, ae_out = out.frames["DM"], out.frames["AE"]
    ref_in = dict(zip(dm_in["USUBJID"], dm_in["RFSTDTC"].map(_iso)))
    ref_out = dict(zip(dm_out["USUBJID"], dm_out["RFSTDTC"].map(_iso)))
    # The surrogate each original subject was given, through DM's row order.
    surrogate = dict(zip(dm_in["USUBJID"], dm_out["USUBJID"]))

    moved = checked = 0
    for i in ae_in.index:
        before, after = _iso(ae_in.at[i, "AESTDTC"]), _iso(ae_out.at[i, "AESTDTC"])
        if before is None:
            continue
        owner_in = ae_in.at[i, "USUBJID"]
        filed_under = ae_out.at[i, "USUBJID"]
        moved += filed_under != surrogate[owner_in]
        # the study day this event had for its real owner ...
        day_before = (before - ref_in[owner_in]).days
        # ... is the study day it has for the subject it is now filed under
        day_after = (after - ref_out[filed_under]).days
        assert day_before == day_after, i
        checked += 1
    assert checked > 50
    assert moved == checked  # a derangement: nobody keeps their own AEs
    # the content itself is untouched
    assert ae_out["AETERM"].equals(ae_in["AETERM"])
    assert out.manifest["reassignment"]["groups"]["AE"]["reassigned"] > 0
    # DM is not reassigned: every subject keeps their own demographics
    assert dm_out["SEX"].equals(dm_in["SEX"])


def test_each_domain_moves_by_its_own_mapping(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    contract = _sdtm_contract(frames)
    DeidPipeline(contract, vault, operator="pytest").run(frames, checksums=sums)
    keys = sorted(frames["DM"]["USUBJID"])
    ae = vault.reassignment_map("AE", keys)
    mh = vault.reassignment_map("MH", keys)
    assert sorted(r for r, _ in ae.values()) == keys  # a bijection
    assert any(ae[k][0] != mh[k][0] for k in keys)


def test_the_raw_side_moves_exactly_as_the_sdtm_side_did(pair_dirs, vault):
    sdtm_frames, s_sums = load_study(pair_dirs[0])
    raw_frames, r_sums = load_study(pair_dirs[1])
    sdtm = DeidPipeline(_sdtm_contract(sdtm_frames, {"AE", "MH"}), vault,
                        operator="pytest").run(sdtm_frames, checksums=s_sums)
    raw_contract, _ = draft_contract(
        raw_frames, source="T", raw_edc=True, reassign="subjects",
        join_key_template="TIG-2026-001-US-{SUBJECT}",
    )
    assert {d.reassign for d in raw_contract.domains} >= {"AE", "MH"}
    raw = DeidPipeline(raw_contract, vault, operator="pytest").run(
        raw_frames, checksums=r_sums
    )

    ae_s, ae_r = sdtm.frames["AE"], raw.frames["AE_LOG"]
    dm, demog = sdtm.frames["DM"], raw.frames["DEMOG"]
    raw_to_usubjid = dict(zip(demog["SUBJECT"], dm["USUBJID"]))
    for i in ae_s.index:
        # filed under the same subject on both sides ...
        assert raw_to_usubjid[ae_r.at[i, "SUBJECT"]] == ae_s.at[i, "USUBJID"]
        # ... and the raw date still converts to the SDTM date
        dmy, iso = ae_r.at[i, "AE_START"], ae_s.at[i, "AESTDTC"]
        if isinstance(dmy, str) and dmy:
            d, m, y = dmy.split("/")
            assert f"{y}-{m}-{d}" == iso


def test_the_raw_side_alone_moves_records_and_keeps_dates(pair_dirs, vault):
    """A raw extract has no RFSTDTC. Reassigning it on its own moves each
    record to its new owner with its dates exactly as written -- and an SDTM
    side run afterwards against the same vault does the same, so the pair
    still corresponds."""
    raw_frames, r_sums = load_study(pair_dirs[1])
    sdtm_frames, s_sums = load_study(pair_dirs[0])
    raw_contract, _ = draft_contract(
        raw_frames, source="T", raw_edc=True, reassign="subjects",
        join_key_template="TIG-2026-001-US-{SUBJECT}",
    )
    raw = DeidPipeline(raw_contract, vault, operator="pytest").run(
        raw_frames, checksums=r_sums
    )
    assert raw.manifest["reassignment"]["groups"]["AE"]["dates"] == "kept as written"
    # records really moved: SUBJECT on AE no longer matches the original owner
    ae_in, ae_out = raw_frames["AE_LOG"], raw.frames["AE_LOG"]
    demog_in, demog_out = raw_frames["DEMOG"], raw.frames["DEMOG"]
    surrogate = dict(zip(demog_in["SUBJECT"], demog_out["SUBJECT"]))
    assert all(
        ae_out.at[i, "SUBJECT"] != surrogate[ae_in.at[i, "SUBJECT"]]
        for i in ae_in.index
    )

    sdtm = DeidPipeline(_sdtm_contract(sdtm_frames, {"AE", "MH"}), vault,
                        operator="pytest").run(sdtm_frames, checksums=s_sums)
    assert "warning" in sdtm.manifest["reassignment"]["groups"]["AE"]
    ae_s = sdtm.frames["AE"]
    raw_to_usubjid = dict(zip(demog_out["SUBJECT"], sdtm.frames["DM"]["USUBJID"]))
    for i in ae_s.index:
        assert raw_to_usubjid[ae_out.at[i, "SUBJECT"]] == ae_s.at[i, "USUBJID"]
        dmy, iso = ae_out.at[i, "AE_START"], ae_s.at[i, "AESTDTC"]
        if isinstance(dmy, str) and dmy:
            d, m, y = dmy.split("/")
            assert f"{y}-{m}-{d}" == iso


def test_the_real_owner_can_be_found_and_it_is_logged(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    DeidPipeline(_sdtm_contract(frames), vault, operator="pytest").run(
        frames, checksums=sums
    )
    keys = sorted(frames["DM"]["USUBJID"])
    mapping = vault.reassignment_map("AE", keys)
    donor = keys[0]
    recipient = mapping[donor][0]
    assert vault.reassigned_from("AE", recipient, justification="SAE-1") == donor
    assert vault.access_log()[-1]["action"] == "reassigned_from"


def test_turning_it_on_changes_the_signature(pair_dirs):
    frames, _ = load_study(pair_dirs[0])
    off, _ = draft_contract(frames, source="T", sdtm_conformant=True)
    on, _ = draft_contract(frames, source="T", sdtm_conformant=True, reassign=True)
    by_subject, _ = draft_contract(frames, source="T", sdtm_conformant=True,
                                  reassign="subjects")
    assert on.rules_digest() != by_subject.rules_digest()
    assert off.rules_digest() != on.rules_digest()


# ----------------------------------------------------------------------
# rows: every record dealt at random, independently
# ----------------------------------------------------------------------
def test_every_record_is_dealt_on_its_own(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    contract, _ = draft_contract(frames, source="T", sdtm_conformant=True,
                                 reassign="rows")
    assert next(d for d in contract.domains if d.name == "AE").reassign_mode == "rows"
    out = DeidPipeline(contract, vault, operator="pytest").run(frames, checksums=sums)
    ae_in, ae_out = frames["AE"], out.frames["AE"]
    dm_in, dm_out = frames["DM"], out.frames["DM"]
    rep = out.manifest["reassignment"]["groups"]["AE"]
    assert rep["mode"] == "rows" and "AESEQ" in rep["renumbered"]

    # nothing lost, nothing invented
    assert sorted(ae_out["AETERM"]) == sorted(ae_in["AETERM"])
    # each event keeps its study day, now counted from its new owner
    ref_in = dict(zip(dm_in["USUBJID"], dm_in["RFSTDTC"].map(_iso)))
    ref_out = dict(zip(dm_out["USUBJID"], dm_out["RFSTDTC"].map(_iso)))
    def days(frame, refs):
        return sorted(
            (t, (_iso(d) - refs[u]).days)
            for t, d, u in zip(frame["AETERM"], frame["AESTDTC"], frame["USUBJID"])
            if _iso(d)
        )
    assert days(ae_in, ref_in) == days(ae_out, ref_out)
    # AESEQ is a key again under each new owner
    for _, g in ae_out.groupby("USUBJID"):
        assert sorted(int(x) for x in g["AESEQ"]) == list(range(1, len(g) + 1))
    # and the rows are no longer in the original order
    assert list(ae_out["AETERM"]) != list(ae_in["AETERM"])


def test_rows_mode_on_raw_alone(pair_dirs, vault):
    raw_frames, r_sums = load_study(pair_dirs[1])
    contract, _ = draft_contract(raw_frames, source="T", raw_edc=True,
                                 reassign="rows")
    out = DeidPipeline(contract, vault, operator="pytest").run(
        raw_frames, checksums=r_sums
    )
    rep = out.manifest["reassignment"]["groups"]["AE_LOG"]
    assert rep["mode"] == "rows" and rep["dates"] == "kept as written"
    ae = out.frames["AE_LOG"]
    assert sorted(ae["AE_TERM"]) == sorted(raw_frames["AE_LOG"]["AE_TERM"])
    subjects = set(out.frames["DEMOG"]["SUBJECT"])
    assert set(ae["SUBJECT"]) <= subjects
    for _, g in ae.groupby("SUBJECT"):
        assert sorted(int(x) for x in g["AE_NO"]) == list(range(1, len(g) + 1))



# ----------------------------------------------------------------------
# values (the default): every column shuffled on its own as well
# ----------------------------------------------------------------------
def test_values_mode_breaks_every_link_on_a_row(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    contract, _ = draft_contract(frames, source="T", sdtm_conformant=True,
                                 reassign=True)
    ae_rule = next(d for d in contract.domains if d.name == "AE")
    assert ae_rule.reassign_mode == "values"
    out = DeidPipeline(contract, vault, operator="pytest").run(frames, checksums=sums)
    rep = out.manifest["reassignment"]["groups"]["AE"]
    assert rep["mode"] == "values"
    assert {"AEDECOD", "AEBODSYS", "AESEV", "AESTDTC"} <= set(rep["columns_shuffled"])
    assert not {"USUBJID", "STUDYID", "AESEQ"} & set(rep["columns_shuffled"])

    ae_in, ae_out = frames["AE"], out.frames["AE"]
    # every column keeps its values, just not its row
    for col in ("AEDECOD", "AEBODSYS", "AESEV"):
        assert sorted(ae_out[col].astype(str)) == sorted(ae_in[col].astype(str))
    # the dictionary pairing is broken: AEDECOD no longer predicts AEBODSYS
    pairs_in = set(zip(ae_in["AEDECOD"], ae_in["AEBODSYS"]))
    pairs_out = set(zip(ae_out["AEDECOD"], ae_out["AEBODSYS"]))
    assert len(pairs_out - pairs_in) > 10
    # whose record it is still resolves, and AESEQ is still a key
    assert set(ae_out["USUBJID"]) <= set(out.frames["DM"]["USUBJID"])
    for _, g in ae_out.groupby("USUBJID"):
        assert sorted(int(x) for x in g["AESEQ"]) == list(range(1, len(g) + 1))


def test_values_mode_on_raw_keeps_the_site_with_the_subject(pair_dirs, vault):
    raw_frames, r_sums = load_study(pair_dirs[1])
    contract, _ = draft_contract(raw_frames, source="T", raw_edc=True,
                                 reassign=True)
    out = DeidPipeline(contract, vault, operator="pytest").run(
        raw_frames, checksums=r_sums
    )
    ae = out.frames["AE_LOG"]
    assert set(ae["SUBJECT"]) <= set(out.frames["DEMOG"]["SUBJECT"])
    assert "AE_TERM" in out.manifest["reassignment"]["groups"]["AE_LOG"]["columns_shuffled"]


# ----------------------------------------------------------------------
# a drop of one file: no DM to take subjects and sites from
# ----------------------------------------------------------------------
def test_an_ae_file_on_its_own_is_shuffled(pair_dirs, vault):
    frames, sums = load_study(pair_dirs[0])
    only = {"AE": frames["AE"]}
    contract, _ = draft_contract(only, source="T", sdtm_conformant=True,
                                 reassign=True)
    assert contract.anchor.domain == "AE"
    out = DeidPipeline(contract, vault, operator="pytest").run(only, checksums={})
    rep = out.manifest["reassignment"]["groups"]["AE"]
    assert rep["mode"] == "values" and rep["subjects_from"] == "AE"
    ae_in, ae_out = frames["AE"], out.frames["AE"]
    pairs_in = set(zip(ae_in["AEDECOD"], ae_in["AEBODSYS"]))
    assert len(set(zip(ae_out["AEDECOD"], ae_out["AEBODSYS"])) - pairs_in) > 10
    assert ae_out["USUBJID"].nunique() <= ae_in["USUBJID"].nunique()
    for _, g in ae_out.groupby("USUBJID"):
        assert sorted(int(x) for x in g["AESEQ"]) == list(range(1, len(g) + 1))


def test_without_dm_the_site_still_follows_the_subject(vault):
    subj = [f"10{s}-00{n}" for s in (1, 2) for n in (1, 2, 3)]
    rows = []
    for k in subj:
        for seq in (1, 2):
            rows.append({"SUBJECT": k, "SITE": k[:3], "AE_NO": str(seq),
                         "AE_TERM": f"term {k} {seq}", "SEVERITY": "MILD" if seq == 1 else "SEVERE"})
    frame = pd.DataFrame(rows)
    contract, _ = draft_contract({"EDC_AE_RAWDATA": frame}, source="T",
                                 raw_edc=True, reassign=True)
    out = DeidPipeline(contract, vault, operator="pytest").run(
        {"EDC_AE_RAWDATA": frame}, checksums={}
    )
    ae = out.frames["EDC_AE_RAWDATA"]
    rep = out.manifest["reassignment"]["groups"]["EDC_AE_RAWDATA"]
    assert "SITE" not in rep["columns_shuffled"]
    # each subject sits at exactly one site, the one its number says
    assert (ae.groupby("SUBJECT")["SITE"].nunique() == 1).all()
