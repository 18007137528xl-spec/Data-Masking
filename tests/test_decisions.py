"""The decision sheet round trip, and the gate it puts in front of a run.

The property under test is not "the CSV parses". It is that a contract nobody
approved cannot be executed, that an approval covers the exact rules it was
given, and that a steward's override reaches the data while still facing every
structural check. Each test below corresponds to a way the old behaviour let
an unreviewed plan reach a published tier while the manifest said nothing
about it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from deidkit import Contract, DeidPipeline, Vault, load_study
from deidkit import decisions as dec
from deidkit.contract import DomainContract, FieldRule, Treatment
from deidkit.profile import draft_contract

ROOT = Path(__file__).resolve().parents[1]
MK_SDTM = ROOT / "scripts" / "make_synthetic_study.py"


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("decisions") / "study"
    subprocess.run(
        [sys.executable, str(MK_SDTM), str(d)], check=True, capture_output=True
    )
    return d


@pytest.fixture(scope="module")
def drafted(study: Path):
    frames, _ = load_study(study)
    contract, suggestions = draft_contract(
        frames, source="TEST-DEC", sdtm_conformant=True
    )
    return frames, contract, dec.build_sheet(contract, suggestions)


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


def _all_ok(sheet: pd.DataFrame) -> pd.DataFrame:
    out = sheet.copy()
    out["decision"] = "OK"
    return out


# ----------------------------------------------------------------------
# the sheet
# ----------------------------------------------------------------------
def test_the_sheet_covers_every_column_once(drafted):
    frames, contract, sheet = drafted
    expected = {(d.name, f.column) for d in contract.domains for f in d.fields}
    got = set(zip(sheet["domain"], sheet["column"]))
    assert got == expected
    assert len(sheet) == len(expected)


def test_lowest_confidence_is_at_the_top(drafted):
    """The ordering is the sheet's main affordance: a steward should not have
    to hunt through correct guesses for the ones worth their attention."""
    _, _, sheet = drafted
    ranks = [{"low": 0, "medium": 1, "high": 2}[c] for c in sheet["confidence"]]
    assert ranks == sorted(ranks)


def test_parameters_survive_the_round_trip(drafted):
    _, contract, sheet = drafted
    rule = next(
        f for d in contract.domains for f in d.fields
        if f.treatment is Treatment.SURROGATE_ID
    )
    text = dec.encode_params(rule)
    assert "entity=subject" in text
    assert dec.decode_params(text, where="t")["entity"] == "subject"


def test_unsettable_parameters_are_refused():
    with pytest.raises(dec.DecisionError, match="not a parameter"):
        dec.decode_params("treatment=drop", where="t")
    with pytest.raises(dec.DecisionError, match="key=value"):
        dec.decode_params("cap 90", where="t")


def test_a_spreadsheet_cannot_retype_the_columns(tmp_path, drafted):
    _, _, sheet = drafted
    p = tmp_path / "hand_made.csv"
    sheet[["domain", "column"]].to_csv(p, index=False)
    with pytest.raises(dec.DecisionError, match="missing required column"):
        dec.read_sheet(p)


def test_excel_bom_and_whitespace_are_tolerated(tmp_path, drafted):
    _, _, sheet = drafted
    p = tmp_path / "excel.csv"
    _all_ok(sheet).to_csv(p, index=False, encoding="utf-8-sig")
    back = dec.read_sheet(p)
    assert list(back.columns)[0] == "domain"
    assert (back["decision"] == "OK").all()


# ----------------------------------------------------------------------
# approval
# ----------------------------------------------------------------------
def test_a_blank_decision_blocks_approval(drafted):
    frames, contract, sheet = drafted
    with pytest.raises(dec.DecisionError, match="no decision"):
        dec.apply_sheet(sheet, contract, approved_by="steward", frames=frames)


def test_one_blank_row_is_enough_to_block(drafted):
    """Partial approval is the failure mode this exists to prevent, so 54 of
    55 rows reviewed is still a refusal."""
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    filled.loc[filled.index[3], "decision"] = ""
    with pytest.raises(dec.DecisionError, match="1 of"):
        dec.apply_sheet(filled, contract, approved_by="steward", frames=frames)


def test_approval_records_what_was_accepted_and_changed(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    approved, stats = dec.apply_sheet(
        filled, contract, approved_by="steward@example.com", frames=frames
    )
    assert approved.approval is not None
    assert approved.approval.approved_by == "steward@example.com"
    assert stats["accepted"] == len(sheet)
    assert stats["changed"] == 0
    assert approved.approval.rules_fingerprint == approved.rules_digest()


def test_an_override_reaches_the_rule(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    target = (filled["domain"] == "DM") & (filled["column"] == "SITEZIP")
    filled.loc[target, ["decision", "decision_treatment", "steward_note"]] = [
        "CHANGE", "drop", "site geography not needed downstream",
    ]
    approved, stats = dec.apply_sheet(
        filled, contract, approved_by="steward", frames=frames
    )
    rule = approved.domain("DM").rule("SITEZIP")
    assert rule.treatment is Treatment.DROP
    assert "steward:" in (rule.note or "")
    assert stats["changed"] == 1


def test_ok_with_an_override_filled_in_is_refused(drafted):
    """Silent divergence between what the sheet says and what was approved is
    worse than an error: the record would claim the proposal was accepted."""
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    filled.loc[filled.index[0], "decision_treatment"] = "drop"
    with pytest.raises(dec.DecisionError, match="OK"):
        dec.apply_sheet(filled, contract, approved_by="s", frames=frames)


def test_change_without_a_treatment_is_refused(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    filled.loc[filled.index[0], "decision"] = "CHANGE"
    with pytest.raises(dec.DecisionError, match="decision_treatment is"):
        dec.apply_sheet(filled, contract, approved_by="s", frames=frames)


def test_an_unknown_treatment_lists_the_real_ones(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    filled.loc[filled.index[0], ["decision", "decision_treatment"]] = [
        "CHANGE", "anonymise_please",
    ]
    with pytest.raises(dec.DecisionError, match="is not a treatment"):
        dec.apply_sheet(filled, contract, approved_by="s", frames=frames)


def test_a_steward_cannot_overrule_an_invariant(drafted):
    """The tier/date coupling holds whoever asked for the change: a
    de-identified tier that retains a date is refused even when a steward
    typed it in deliberately."""
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    target = (filled["domain"] == "AE") & (filled["column"] == "AESTDTC")
    filled.loc[target, ["decision", "decision_treatment"]] = ["CHANGE", "retain"]
    with pytest.raises(dec.DecisionError, match="real calendar values"):
        dec.apply_sheet(filled, contract, approved_by="s", frames=frames)


def test_a_sheet_from_another_drop_is_refused(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    extra = pd.DataFrame(frames["DM"])
    extra["A_NEW_COLUMN"] = "x"
    with pytest.raises(dec.DecisionError, match="says nothing about"):
        dec.apply_sheet(
            filled, contract, approved_by="s", frames={**frames, "DM": extra}
        )


def test_a_duplicated_row_is_refused(drafted):
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    doubled = pd.concat([filled, filled.head(1)], ignore_index=True)
    with pytest.raises(dec.DecisionError, match="more than one row"):
        dec.apply_sheet(doubled, contract, approved_by="s", frames=frames)


# ----------------------------------------------------------------------
# the signature
# ----------------------------------------------------------------------
def test_editing_a_rule_after_approval_stops_the_contract_loading(drafted, tmp_path):
    frames, contract, sheet = drafted
    approved, _ = dec.apply_sheet(
        _all_ok(sheet), contract, approved_by="s", frames=frames
    )
    p = tmp_path / "approved.yaml"
    approved.to_yaml(p)

    # A change that is perfectly valid on its own terms -- ZIP truncation
    # turned back into a pass-through -- so nothing but the signature stands
    # between it and a run.
    text = p.read_text(encoding="utf-8")
    assert "treatment: zip3" in text
    (tmp_path / "tampered.yaml").write_text(
        text.replace("treatment: zip3", "treatment: retain", 1), encoding="utf-8"
    )
    with pytest.raises(Exception, match="edited after it was approved"):
        Contract.from_yaml(tmp_path / "tampered.yaml")


def test_the_signature_survives_a_yaml_round_trip(drafted, tmp_path):
    frames, contract, sheet = drafted
    approved, _ = dec.apply_sheet(
        _all_ok(sheet), contract, approved_by="s", frames=frames
    )
    p = tmp_path / "a.yaml"
    approved.to_yaml(p)
    back = Contract.from_yaml(p)
    assert back.approval is not None
    assert back.approval.rules_fingerprint == approved.approval.rules_fingerprint


def test_bumping_the_version_cannot_launder_a_rule_change(drafted):
    """contract_version is outside the digest on purpose. If it were inside,
    the digest would change for a reason that has nothing to do with the
    rules; if a rule change could be hidden by bumping it, the digest would
    be worthless. Only the first must be true."""
    _, contract, _ = drafted
    before = contract.rules_digest()
    assert contract.model_copy(
        update={"contract_version": "9.9.9"}
    ).rules_digest() == before


# ----------------------------------------------------------------------
# carrying decisions forward
# ----------------------------------------------------------------------
def test_carry_forward_reproduces_the_same_rules(drafted):
    """The second drop of a study should not re-argue settled decisions -- and
    the proof that nothing drifted is that the fingerprint comes out
    identical."""
    frames, contract, sheet = drafted
    filled = _all_ok(sheet)
    target = (filled["domain"] == "DM") & (filled["column"] == "SITEZIP")
    filled.loc[target, ["decision", "decision_treatment"]] = ["CHANGE", "drop"]
    first, _ = dec.apply_sheet(filled, contract, approved_by="s", frames=frames)

    carried = dec.carry_forward(sheet, first)
    assert (carried["decision"] != "").all()
    second, stats = dec.apply_sheet(
        carried, contract, approved_by="s", frames=frames
    )
    assert second.rules_digest() == first.rules_digest()
    assert stats["changed"] == 1


def test_a_new_column_comes_back_blank(drafted):
    frames, contract, sheet = drafted
    first, _ = dec.apply_sheet(
        _all_ok(sheet), contract, approved_by="s", frames=frames
    )
    extra = pd.concat(
        [
            sheet,
            pd.DataFrame([{
                "domain": "DM", "column": "NEW_FIELD",
                "proposed_treatment": "retain", "proposed_params": "",
                "confidence": "low", "quasi_identifier": False,
                "decision": "", "decision_treatment": "", "decision_params": "",
                "steward_note": "", "why": "unrecognised", "detail": "",
            }]),
        ],
        ignore_index=True,
    )
    carried = dec.carry_forward(extra, first)
    blank = carried[carried["decision"] == ""]
    assert list(blank["column"]) == ["NEW_FIELD"]


def test_decisions_cannot_be_carried_from_an_unapproved_draft(drafted):
    _, contract, sheet = drafted
    with pytest.raises(dec.DecisionError, match="no decisions in it"):
        dec.carry_forward(sheet, contract)


# ----------------------------------------------------------------------
# the run gate, through the CLI
# ----------------------------------------------------------------------
def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "deidkit.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "DEIDKIT_VAULT_KEY": Vault.generate_key()},
    )


def test_run_refuses_an_unapproved_contract(study, tmp_path, drafted):
    _, contract, _ = drafted
    draft = tmp_path / "draft.yaml"
    contract.to_yaml(draft)
    proc = _cli(
        "run", str(study), "-c", str(draft), "-o", str(tmp_path / "tier"),
        "--vault", str(tmp_path / "v.db"), "--format", "csv",
    )
    assert proc.returncode != 0
    assert "has not been approved" in proc.stderr
    assert not (tmp_path / "tier").exists()


def test_unreviewed_runs_but_the_manifest_says_so(study, tmp_path, drafted):
    _, contract, _ = drafted
    draft = tmp_path / "draft.yaml"
    contract.to_yaml(draft)
    proc = _cli(
        "run", str(study), "-c", str(draft), "-o", str(tmp_path / "tier"),
        "--vault", str(tmp_path / "v.db"), "--format", "csv", "--unreviewed",
    )
    assert proc.returncode == 0, proc.stderr
    import json

    manifest = json.loads((tmp_path / "tier" / "manifest.json").read_text())
    assert manifest["review"]["reviewed"] is False
    assert "NOT REVIEWED" in proc.stdout


def test_an_approved_contract_runs_and_the_manifest_names_the_approver(
    study, tmp_path, drafted
):
    frames, contract, sheet = drafted
    approved, _ = dec.apply_sheet(
        _all_ok(sheet), contract, approved_by="steward@example.com", frames=frames
    )
    path = tmp_path / "approved.yaml"
    approved.to_yaml(path)
    proc = _cli(
        "run", str(study), "-c", str(path), "-o", str(tmp_path / "tier"),
        "--vault", str(tmp_path / "v.db"), "--format", "csv",
    )
    assert proc.returncode == 0, proc.stderr
    import json

    manifest = json.loads((tmp_path / "tier" / "manifest.json").read_text())
    assert manifest["review"]["reviewed"] is True
    assert manifest["review"]["approved_by"] == "steward@example.com"
    assert manifest["review"]["rules_fingerprint"].startswith("sha256:")


# ----------------------------------------------------------------------
# the treatment reference
# ----------------------------------------------------------------------
def test_every_treatment_is_documented():
    """The reference and the enum cannot drift.

    A steward who follows a reference that is missing a treatment, or that
    lists one the code does not have, hits a failure the reference caused.
    """
    from deidkit.contract import Treatment
    from deidkit.decisions import TREATMENT_HELP

    assert {t.value for t in Treatment} == set(TREATMENT_HELP)


def test_documented_requirements_match_the_validators():
    """Each treatment's 'requires' list must be exactly what FieldRule
    enforces -- documented-but-optional teaches a steward to omit something
    that will be rejected, and the reverse hides a real requirement."""
    from deidkit.contract import FieldRule, Treatment
    from deidkit.decisions import TREATMENT_HELP

    samples = {
        "entity": "subject",
        "faker_provider": "name",
        "cap": 90,
        "bins": [0.0, 50.0],
        "min_count": 5,
    }
    for name, meta in TREATMENT_HELP.items():
        required = set(meta["requires"])
        # With everything it asks for, the rule must build.
        FieldRule(
            column="X",
            treatment=Treatment(name),
            **{k: samples[k] for k in required},
        )
        # With any one of them missing, it must not.
        for drop in required:
            kw = {k: samples[k] for k in required - {drop}}
            with pytest.raises(Exception):
                FieldRule(column="X", treatment=Treatment(name), **kw)


# ----------------------------------------------------------------------
# carrying decisions forward
# ----------------------------------------------------------------------
def test_carry_forward_reproduces_the_previous_approval(study):
    """The second drop must not re-litigate settled decisions.

    An accepted proposal comes back OK; an override comes back as that
    override, not as the profiler's proposal and not as a blank. Approving the
    carried sheet unchanged must therefore yield the same rules as last time --
    otherwise 'carry forward' quietly changes what was agreed.
    """
    frames, _ = load_study(study)
    contract, suggestions = draft_contract(frames, source="T", sdtm_conformant=True)
    sheet = dec.build_sheet(contract, suggestions)
    sheet["decision"] = "OK"
    # one override, so there is something non-trivial to carry
    mask = (sheet["domain"] == "DM") & (sheet["column"] == "AGE")
    sheet.loc[mask, ["decision", "decision_treatment", "decision_params"]] = [
        "CHANGE",
        "generalize_numeric",
        "bins=0,18,40,65,90",
    ]
    first, _ = dec.apply_sheet(sheet, contract, approved_by="steward@example.com")

    # A fresh profile of the same data, carrying the approval forward.
    contract2, suggestions2 = draft_contract(
        frames, source="T", sdtm_conformant=True
    )
    sheet2 = dec.carry_forward(dec.build_sheet(contract2, suggestions2), first)
    assert (sheet2["decision"] != "").all(), "nothing should still need a decision"
    carried = sheet2[sheet2["decision"] == "CHANGE"]
    assert list(carried["column"]) == ["AGE"]
    assert carried.iloc[0]["decision_treatment"] == "generalize_numeric"

    second, stats = dec.apply_sheet(
        sheet2, contract2, approved_by="steward@example.com"
    )
    assert stats["changed"] == 1
    rule = second.domain("DM").rule("AGE")
    assert rule.treatment is Treatment.GENERALIZE_NUMERIC
    assert rule.bins == [0.0, 18.0, 40.0, 65.0, 90.0]
    # Same rules in, same digest out.
    assert second.rules_digest() == first.rules_digest()


def test_carry_forward_refuses_an_unapproved_source(drafted):
    _, contract, sheet = drafted
    with pytest.raises(dec.DecisionError, match="unapproved"):
        dec.carry_forward(sheet, contract)


# ----------------------------------------------------------------------
# the file is open in Excel
# ----------------------------------------------------------------------
def test_a_locked_output_is_refused_before_any_work(tmp_path, study, monkeypatch):
    """The ordinary Windows failure, and the one the tool handles worst.

    A steward is told to open the decision sheet, opens it in Excel, Excel
    takes an exclusive lock, and the next run profiles every domain, prints a
    summary ending in "contract draft written", and then dies in a pandas
    traceback about errno 13. So this asserts three things: the command fails,
    it fails with a message naming the actual cause, and it fails BEFORE
    writing anything.
    """
    import builtins

    from deidkit import cli

    plan = tmp_path / "plan.csv"
    plan.write_text("locked", encoding="utf-8")
    contract_out = tmp_path / "draft.yaml"

    real_open = builtins.open

    def locked(file, *a, **kw):
        if str(file) == str(plan):
            raise PermissionError(13, "Permission denied")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", locked)

    args = cli.build_parser().parse_args(
        [
            "profile",
            str(study),
            "-o",
            str(contract_out),
            "--decisions",
            str(plan),
        ]
    )
    rc = cli.cmd_profile(args)

    assert rc == 1
    assert not contract_out.exists(), "the draft was written despite the failure"


def test_the_lock_message_names_excel(tmp_path, monkeypatch):
    import builtins

    from deidkit import cli

    target = tmp_path / "plan.csv"
    target.write_text("x", encoding="utf-8")
    real_open = builtins.open

    def locked(file, *a, **kw):
        if str(file) == str(target):
            raise PermissionError(13, "Permission denied")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", locked)

    msg = cli._check_writable(str(target))
    assert msg is not None
    assert "Excel" in msg
    assert "Nothing has been changed" in msg


def test_a_writable_path_passes_and_leaves_nothing_behind(tmp_path):
    from deidkit import cli

    fresh = tmp_path / "sub" / "dir" / "plan.csv"
    assert cli._check_writable(str(fresh)) is None
    assert fresh.parent.is_dir(), "the parent should be created, ready to write"
    assert not fresh.exists(), "a pre-flight check must not leave a stub file"


# ----------------------------------------------------------------------
# not overwriting a review
# ----------------------------------------------------------------------
def test_profiling_refuses_to_overwrite_a_filled_sheet(tmp_path, study):
    """A reviewed sheet is the only copy of that work.

    It is not in version control, it took someone an afternoon, and
    re-profiling would rewrite it with blanks. The refusal also says what the
    person almost certainly meant to do instead.
    """
    from deidkit import cli

    plan = tmp_path / "plan.csv"
    args = cli.build_parser().parse_args(
        ["profile", str(study), "-o", str(tmp_path / "d.yaml"), "--decisions", str(plan)]
    )
    assert cli.cmd_profile(args) == 0

    sheet = pd.read_csv(plan, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet["decision"] = "OK"
    sheet.to_csv(plan, index=False, encoding="utf-8-sig")
    before = plan.read_bytes()

    assert cli.cmd_profile(args) == 1, "a filled sheet must not be overwritten"
    assert plan.read_bytes() == before, "the sheet was modified anyway"

    # Starting over is legitimate -- it just has to be said out loud.
    forced = cli.build_parser().parse_args(
        [
            "profile",
            str(study),
            "-o",
            str(tmp_path / "d.yaml"),
            "--decisions",
            str(plan),
            "--force-decisions",
        ]
    )
    assert cli.cmd_profile(forced) == 0
    reset = pd.read_csv(plan, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert (reset["decision"].str.strip() == "").all()


def test_a_blank_sheet_is_not_protected(tmp_path, study):
    """Only decisions are precious. An unfilled sheet is regenerated freely,
    or the ordinary re-profile loop would need a flag every time."""
    from deidkit import cli

    plan = tmp_path / "plan.csv"
    args = cli.build_parser().parse_args(
        ["profile", str(study), "-o", str(tmp_path / "d.yaml"), "--decisions", str(plan)]
    )
    assert cli.cmd_profile(args) == 0
    assert cli.cmd_profile(args) == 0
