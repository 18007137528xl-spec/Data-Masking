"""Command-line interface.

    deidkit keygen                    generate a vault key
    deidkit profile   <dir>           profile a drop and draft a contract
    deidkit approve   <plan.csv>      sign off a decision sheet -> contract
    deidkit treatments                what a decision sheet may ask for
    deidkit run       <dir>           transform, screen, measure, publish
    deidkit adjudicate <dir>          apply a reviewed free-text queue
    deidkit risk      <dir>           re-measure risk on a published tier
    deidkit vault     stats|log       inspect the crosswalk
    deidkit reverse   <surrogate>     break-glass re-identification (logged)

Uses argparse rather than a CLI framework: one fewer dependency in a package
that will be reviewed by people who are not its authors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import (
    decisions as dec,
    freetext,
    io as dio,
    profile as prof,
    risk as risk_mod,
    textcheck,
)
from .contract import Contract
from .pipeline import ContractMismatch, DeidPipeline
from .vault import Vault, VaultError


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


class ContractLoadError(RuntimeError):
    """A contract file that will not load. Reported, never raised at a user."""


def _load_contract(path: str) -> Contract:
    """Load a contract, turning validation failures into readable errors.

    The tamper check in particular has to read well: someone hitting it is
    being told that a file they were about to run against patient data has
    been edited since it was approved, and a pydantic traceback is not how to
    say that.
    """
    try:
        return Contract.from_yaml(path)
    except FileNotFoundError:
        raise ContractLoadError(f"no such contract file: {path}") from None
    except Exception as exc:
        # Pydantic wraps the validator's message in its own framing and then
        # appends the whole input dict. Both are noise here, and the input
        # dump in particular is a wall of YAML in the middle of a sentence
        # someone needs to read carefully.
        detail = str(exc)
        detail = re.split(r"\s*\[type=[a-z_]+,\s*input_value=", detail)[0]
        keep = []
        for line in detail.splitlines():
            line = line.strip()
            if not line or re.match(r"^\d+ validation error", line):
                continue
            if line.startswith("For further information"):
                continue
            keep.append(re.sub(r"^Value error,\s*", "", line))
        raise ContractLoadError(
            f"{path} will not load:\n  " + "\n  ".join(keep)
        ) from None


def _filled_decisions(path: str | None) -> int:
    """How many decisions a sheet already carries. 0 if it is blank or absent.

    Used to refuse to overwrite work. A steward may have spent an afternoon on
    83 rows, and 'profile' rewriting the sheet would destroy that silently and
    irreversibly -- the file is not in git, and the only copy is the one being
    overwritten.
    """
    if not path or not Path(path).exists():
        return 0
    try:
        frame = pd.read_csv(
            path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    except Exception:
        return 0  # unreadable is not "filled in"; let the write proceed
    if "decision" not in frame.columns:
        return 0
    return int((frame["decision"].fillna("").astype(str).str.strip() != "").sum())


def _check_writable(*paths: str | None) -> str | None:
    """Confirm every output path can be written, before doing any work.

    Returns an error message, or None.

    This exists because of a specific and very ordinary Windows failure. A
    steward is told to open the decision sheet; they open it in Excel; Excel
    takes an exclusive lock; the next run gets through profiling all seven
    domains, prints a full summary ending in "contract draft written", and only
    then dies in a pandas traceback about errno 13. Everything about that is
    misleading: the work is wasted, the summary implies success, and the
    message names a library rather than the spreadsheet holding the file.

    A file this tool is about to overwrite is a file it can check first.
    """
    for path in paths:
        if not path:
            continue
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"cannot create the folder for {path}: {exc}"
        try:
            if target.exists():
                with open(target, "r+b"):
                    pass
            else:
                # Touch and remove, so a pre-flight check leaves nothing behind.
                with open(target, "wb"):
                    pass
                target.unlink()
        except PermissionError:
            return (
                f"cannot write {path} -- it is locked by another program.\n"
                "On Windows that is almost always Excel holding the file open. "
                "Close it and\nrun this again, or write somewhere else with a "
                "different output path.\nNothing has been changed."
            )
        except OSError as exc:
            return f"cannot write {path}: {exc}\nNothing has been changed."
    return None


def _operator(args: argparse.Namespace) -> str:
    return args.operator or os.environ.get("USER") or "unknown"


def _open_vault(args: argparse.Namespace) -> Vault:
    return Vault(
        args.vault,
        operator=_operator(args),
        key_uri=getattr(args, "key_uri", None),
    )


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def cmd_keygen(args: argparse.Namespace) -> int:
    # With a key URI, provision the key inside the key service and never let
    # the plaintext touch stdout or a shell history.
    if args.key_uri:
        from .keyprovider import KeyProviderError, from_uri

        try:
            provider = from_uri(args.key_uri)
            print(provider.provision())
        except KeyProviderError as exc:
            return _err(str(exc))
        if not provider.managed:
            print(
                "\nThis key source is not backed by a managed key service. That is\n"
                "fine for development and wrong for PHI: set DEIDKIT_KEY_URI to a\n"
                "KMS or Key Vault URI for anything real.",
                file=sys.stderr,
            )
        return 0

    key = Vault.generate_key()
    print(key)
    print(
        "\nStore this in your KMS, not beside the vault file and not in version\n"
        "control. Without it the crosswalk cannot be opened -- which also means\n"
        "no subject in any published tier can ever be re-identified, including\n"
        "for safety reporting.\n"
        "\nFor a cloud deployment, prefer provisioning the key inside the key\n"
        "service so the plaintext never reaches a terminal:\n"
        "  deidkit keygen --key-uri 'awskms:<key-arn>?blob=/vault/key.blob'",
        file=sys.stderr,
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    # Before profiling seven domains, check the three files this will write.
    if problem := _check_writable(args.out, args.review, args.decisions):
        return _err(problem)

    # And before overwriting a decision sheet, check nobody has filled it in.
    filled = _filled_decisions(args.decisions)
    if filled and not args.force_decisions:
        return _err(
            f"{args.decisions} already has {filled} decision(s) in it, and "
            "profiling would overwrite them.\n\n"
            "If you have reviewed the sheet, the next step is not to profile "
            "again -- it is:\n"
            f"  deidkit approve {args.decisions} -c {args.out} "
            f"--data {args.directory} -o <approved.yaml>\n\n"
            "If you meant to start the review over, delete the sheet or pass "
            "--force-decisions.\nThe sheet is not in version control, so this "
            "refusal is the only copy of that work."
        )

    frames, _ = dio.load_study(args.directory)
    if not frames:
        return _err(dio.explain_empty(args.directory))

    contract, suggestions = prof.draft_contract(
        frames,
        source=args.source or Path(args.directory).name,
        tier=args.tier,
        anchor_domain=args.anchor_domain,
        anchor_date_column=args.anchor_date,
        k_target=args.k_target,
        sdtm_conformant=args.sdtm,
        blind_treatment=args.blind_treatment,
        keep_dates=args.keep_dates,
        raw_edc=args.raw,
        join_key_template=args.offset_key,
        subject_id_template=args.id_template,
    )

    table = prof.suggestion_table(suggestions)
    low = table[table["confidence"] == "low"]

    print(f"profiled {len(frames)} domain(s): {', '.join(sorted(frames))}")
    print(f"drafted {len(table)} column rules")
    print(f"anchor: {contract.anchor.domain}.{contract.anchor.date_column}")
    anchor_cols = set(map(str, frames.get(contract.anchor.domain, pd.DataFrame()).columns))
    if contract.anchor.date_column not in anchor_cols:
        print(
            f"        WARNING: {contract.anchor.date_column} is not in "
            f"{contract.anchor.domain}. No reference start date was found, and "
            "audit\n        timestamps (ENTRYDTC, LASTUPDDTC) are deliberately "
            "not used as one -- they date\n        the paperwork, not the "
            "patient. Any study-day conversion will refuse to run.\n"
            "        Point --anchor-domain/--anchor-date at the real "
            "randomisation or first-dose date."
        )
    if args.raw:
        print(
            "dates : shifted per subject, written format preserved "
            "(raw side of a raw -> SDTM pair)"
        )
        if args.offset_key:
            example = next(
                (
                    args.offset_key.format(
                        **{
                            c: str(frames[d][c].iloc[0])
                            for c in re.findall(r"{(\w+)}", args.offset_key)
                            if c in frames[d].columns
                        }
                    )
                    for d in frames
                    if all(
                        c in frames[d].columns
                        for c in re.findall(r"{(\w+)}", args.offset_key)
                    )
                    and len(frames[d])
                ),
                None,
            )
            print(f"offset key: {args.offset_key}")
            if example:
                print(
                    f"        first row resolves to {example!r} -- this must be "
                    "byte-identical to the key the SDTM side used"
                )
        else:
            print(
                "        NO --offset-key given: offsets are keyed on each "
                "table's subject column as-is. Correct only if the raw and "
                "SDTM sides spell the subject identifier identically."
            )
        unresolved = [
            f"{d.name}.{f.column}"
            for d in contract.domains
            for f in d.fields
            if f.treatment.value == "date_shift_raw"
            and f.date_order is None
            and "AMBIGUOUS" in (f.note or "")
        ]
        if unresolved:
            print(
                f"        {len(unresolved)} column(s) have an UNRESOLVED "
                "day/month order and will halt the run until date_order is set:"
            )
            for c in unresolved:
                print(f"          {c}")
        keyless = [d.name for d in contract.domains if not d.subject_key]
        if keyless:
            print(
                "        no subject key found in: "
                f"{', '.join(keyless)} -- set subject_key in the contract, or "
                "their dates cannot be shifted"
            )
    elif args.keep_dates:
        print("dates : retained as recorded -- tier forced to 'lds'")
    elif args.sdtm:
        print("dates : shifted per subject (SDTM-conformant; --DTC retained)")
    else:
        print("dates : converted to study days (--DTC dropped; NOT SDTM-conformant)")
    print(f"tier  : {contract.tier}")
    retained = [d.name for d in contract.domains if d.retained_in_full]
    if retained:
        print(f"retained in full: {', '.join(retained)}")
    print()
    if len(low):
        print(f"{len(low)} rule(s) the steward MUST review (low confidence):")
        for _, r in low.iterrows():
            print(f"  {r['domain']}.{r['column']:<20} -> {r['treatment']:<28} {r['rationale']}")
        print()

    contract.to_yaml(args.out)
    print(f"contract draft written to {args.out}")

    if args.decisions:
        sheet = dec.build_sheet(contract, suggestions)
        if args.carry_forward:
            try:
                previous = _load_contract(args.carry_forward)
                sheet = dec.carry_forward(sheet, previous)
            except (dec.DecisionError, ContractLoadError) as exc:
                return _err(str(exc))
            prefilled = int((sheet["decision"] != "").sum())
            print(
                f"carried {prefilled} decision(s) forward from "
                f"{args.carry_forward}; {len(sheet) - prefilled} still need one"
            )
        Path(args.decisions).parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(args.decisions, index=False, encoding="utf-8-sig")
        print(f"decision sheet written to {args.decisions}")
        print(
            "\nNEXT: open the decision sheet. Lowest confidence is at the top.\n"
            "  decision = OK      accept the proposal\n"
            "  decision = CHANGE  overrule it, and say what in "
            "decision_treatment\n"
            "Then: deidkit approve "
            f"{args.decisions} -c {args.out} --data {args.directory} "
            "-o <approved.yaml>\n"
            "A blank row blocks the run. That is deliberate: it is the "
            "difference between\n"
            "a contract that runs and a contract someone agreed with."
        )
    if args.review:
        Path(args.review).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.review, index=False)
        print(f"steward review sheet written to {args.review}")
    print(
        "\nThis is a DRAFT. Every rule needs a steward's confirmation before it\n"
        "is committed -- the suggestions come from naming convention and content\n"
        "heuristics, not from understanding your study."
    )
    return 0


def cmd_treatments(args: argparse.Namespace) -> int:
    print()
    print(dec.treatment_reference())
    print()
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    if problem := _check_writable(args.out):
        return _err(problem)
    try:
        contract = _load_contract(args.contract)
    except ContractLoadError as exc:
        return _err(str(exc))

    if contract.approval is not None:
        print(
            f"note: {args.contract} is already approved by "
            f"{contract.approval.approved_by} on "
            f"{contract.approval.approved_at[:10]}. Re-approving replaces that "
            "signature."
        )

    try:
        sheet = dec.read_sheet(args.plan)
    except dec.DecisionError as exc:
        return _err(str(exc))

    frames = None
    if args.data:
        frames, _ = dio.load_study(args.data)
        if not frames:
            return _err(dio.explain_empty(args.data))
    else:
        print(
            "note: --data was not given, so the sheet was checked against the\n"
            "      contract but not against a dataset. Passing it catches a "
            "sheet\n      written for a different drop."
        )

    try:
        approved, stats = dec.apply_sheet(
            sheet,
            contract,
            approved_by=args.approved_by or _operator(args),
            frames=frames,
            plan_file=args.plan,
            note=args.note,
        )
    except dec.DecisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    approved.to_yaml(args.out)
    print()
    print(f"approved by  : {approved.approval.approved_by}")
    print(f"at           : {approved.approval.approved_at}")
    print(f"decisions    : {stats['accepted']} accepted as proposed, "
          f"{stats['changed']} changed  (of {stats['total']})")
    print(f"fingerprint  : {approved.approval.rules_fingerprint}")
    print(f"tier         : {approved.tier}")
    print(f"written to   : {args.out}")
    if stats["changed"] == 0:
        print(
            "\nEvery row was accepted as proposed. That is a legitimate "
            "outcome, and it is\nalso what a sheet filled in by dragging one "
            "value down the column looks like.\nThe tool cannot tell those "
            "apart; only your process can."
        )
    print(
        "\nCommit this file. The signature covers these exact rules, so any "
        "later edit\nto any rule stops the contract from loading at all."
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        contract = _load_contract(args.contract)
    except ContractLoadError as exc:
        return _err(str(exc))

    # A contract with no approval is a machine's guess about someone's study.
    # Refusing it here is the whole point of the decision sheet -- and the
    # escape hatch is explicit and recorded, because development convenience
    # should never depend on a check being easy to forget.
    if contract.approval is None and not args.unreviewed:
        return _err(
            f"{args.contract} has not been approved.\n\n"
            "  deidkit profile <dir> --decisions plan.csv\n"
            "  # fill in the decision column\n"
            f"  deidkit approve plan.csv -c {args.contract} --data "
            "<dir> -o approved.yaml\n\n"
            "Suggestions come from naming convention and content heuristics, "
            "not from\nunderstanding your study, so an unreviewed contract is "
            "an untested claim.\nFor synthetic data or CI, pass --unreviewed: "
            "it runs, and the manifest\nrecords that nobody reviewed it."
        )
    frames, sums = dio.load_study(args.directory)
    if not frames:
        return _err(dio.explain_empty(args.directory))

    # The manifest and the review queue are written last, after every domain
    # has been transformed and the risk measured. A lock on either -- the queue
    # is a CSV, so someone is adjudicating it in Excel -- would throw that away
    # at the final step, so both are checked up front.
    out_dir = str(args.out).rstrip("/").rstrip("\\")
    manifest_path = args.manifest or f"{out_dir}/manifest.json"
    queue_path = args.review or f"{out_dir}_review/review_queue.csv"
    if problem := _check_writable(manifest_path, queue_path):
        return _err(problem)

    try:
        vault = _open_vault(args)
    except VaultError as exc:
        return _err(str(exc))

    with vault:
        pipeline = DeidPipeline(contract, vault, operator=_operator(args))
        try:
            result = pipeline.run(
                frames,
                checksums=sums,
                allow_missing=args.allow_missing,
                screen=not args.no_screen,
            )
        except ContractMismatch as exc:
            return _err(f"contract mismatch\n{exc}")

    written = dio.write_study(result.frames, args.out, fmt=args.format)

    result.manifest["vault"] = {"key_source": vault.key_source}
    # Recorded either way. A manifest that is silent about review lets an
    # unreviewed run be mistaken for a reviewed one later, which is exactly
    # the confusion this block exists to remove.
    result.manifest["review"] = (
        {"reviewed": True, **contract.approval.model_dump(mode="json")}
        if contract.approval is not None
        else {
            "reviewed": False,
            "reason": "--unreviewed: no steward approved this contract",
        }
    )
    dio.write_text(
        json.dumps(result.manifest, indent=2, default=str), manifest_path
    )

    # The queue holds the ORIGINAL text of every flagged row -- unredacted PHI,
    # and in a blinded study the compound name. It must not sit in the
    # directory analysts read, so it goes to a sibling by default.
    if not result.review_queue.empty:
        dio.write_text(result.review_queue.to_csv(index=False), queue_path)

    # The steward's copy of the risk report, with the quasi-identifier values
    # of the smallest classes. It goes to the review sibling, not the published
    # tier, for the same reason the free-text queue does: it is the material a
    # human needs to decide what to generalise, and it is also the most
    # targeting-useful thing the pipeline produces.
    risk_detail_path = None
    if result.risk_report is not None:
        risk_detail_path = str(Path(queue_path).parent / "risk_detail.json")
        dio.write_text(
            json.dumps(
                result.risk_report.to_dict(include_class_values=True),
                indent=2,
                default=str,
            ),
            risk_detail_path,
        )

    print(result.summary())
    print()
    for name, p in written.items():
        print(f"  wrote {name:<6} -> {p}")
    print(f"  wrote manifest -> {manifest_path}")
    if contract.approval is None:
        print(
            "\n  NOT REVIEWED: this ran on an unapproved contract and the "
            "manifest says so.\n  Do not treat this output as a published tier."
        )
    if not result.review_queue.empty:
        print(f"  wrote review queue -> {queue_path}")
    if risk_detail_path:
        print(f"  wrote risk detail  -> {risk_detail_path}")
        pol = (result.manifest.get("freetext_screening") or {}).get("policy") or {}
        needs = int(pol.get("needs_human", len(result.review_queue)))
        decided = len(result.review_queue) - needs
        print(
            f"\nfree-text queue: {len(result.review_queue)} flagged row(s), "
            f"{needs} need a person."
        )
        if decided:
            print(
                f"  {decided} were settled by policy "
                f"(auto-pass {pol.get('auto_pass', 0)}, "
                f"auto-redact {pol.get('auto_redact', 0)}) and are pre-filled "
                "with a verdict\n  you can overrule. The rows needing judgment "
                "are at the TOP of the sheet."
            )
        print(
            "Set 'verdict' to PASS or REDACT on the blank rows, then:\n"
            f"  deidkit adjudicate {args.out} --queue {queue_path} "
            "--out <final-dir>"
        )
        print(
            "\nThe queue contains the ORIGINAL text of each flagged row, which is "
            "what makes\nit reviewable and also means it is unredacted PHI. Keep "
            "it under the vault's\naccess controls, not the published tier's, and "
            "delete it once adjudicated."
        )
    if result.blinding_report and not result.blinding_report.held:
        print()
        print(result.blinding_report.summary())
        print(
            "\nThe relabelling cannot reach these. Adjudicate the free-text "
            "findings, and decide\nexplicitly about dose and regimen columns -- "
            "pooling them protects the blind but\ncosts exposure-response "
            "analysis."
        )

    if result.risk_report and not result.risk_report.k_met:
        print(
            "\nNOTE: the k target was not met. For a small trial this is normal. "
            "Record it\nin the determination as an accepted risk with the "
            "compensating controls named --\nan accepted risk and an unhandled "
            "omission look identical in the data."
        )
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    out_dir = str(args.out).rstrip("/").rstrip("\\")
    manifest_path = args.manifest or f"{out_dir}/manifest.json"
    if problem := _check_writable(manifest_path):
        return _err(problem)

    paths = dio.discover(args.directory)
    frames = {n: dio.read_table(p) for n, p in paths.items()}
    queue = pd.read_csv(args.queue)

    out_frames, stats = freetext.apply_adjudication(frames, queue)

    pending = int(stats["rows_unreviewed"]) + int(stats["rows_unknown_verdict"])
    # This is the tier meant for release. A row nobody ruled on is published
    # exactly as recorded, which for a flagged row means possible PHI in the
    # output -- so it halts here for the same reason a blank decision sheet
    # halts the run, and the escape hatch is explicit and recorded.
    if pending and not args.allow_pending:
        return _err(
            f"{stats['rows_unreviewed']} row(s) have no verdict"
            + (
                f" and {stats['rows_unknown_verdict']} have one that is not "
                "PASS or REDACT"
                if stats["rows_unknown_verdict"]
                else ""
            )
            + ".\n\nAn unruled row is published as recorded, and these rows "
            "were flagged as possible\nPHI, so this is the last place to catch "
            "it. Set 'verdict' to PASS or REDACT\nfor every row, or pass "
            "--allow-pending to publish the remainder unchanged --\nwhich is "
            "recorded in the manifest as a known gap, not hidden."
        )

    written = dio.write_study(out_frames, args.out, fmt=args.format)

    # The tier that gets released is the one that most needs a manifest, and
    # until now it was the only one without one: the data was written, the
    # counts went to the console, and nothing durable recorded who adjudicated
    # what. The chain of evidence broke at the last step.
    source_manifest = Path(args.directory) / "manifest.json"
    manifest: dict[str, Any] = {}
    if source_manifest.exists():
        try:
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    passed = (
        len(queue)
        - int(stats["redactions_applied"])
        - int(stats["rows_unreviewed"])
        - int(stats["rows_unknown_verdict"])
    )
    manifest["adjudication"] = {
        "required": True,
        "complete": pending == 0,
        "pending_rows": pending,
        "queue_rows": len(queue),
        "redactions_applied": int(stats["redactions_applied"]),
        "rows_passed": passed,
        "rows_unreviewed": int(stats["rows_unreviewed"]),
        "rows_unknown_verdict": int(stats["rows_unknown_verdict"]),
        "adjudicated_by": _operator(args),
        "adjudicated_at": _now_iso(),
        "source_tier": str(args.directory),
        "published_with_pending_rows": bool(pending and args.allow_pending),
    }
    manifest["inputs"] = {
        "source_tier": str(args.directory),
        "checksums_sha256": {n: dio.checksum(p) for n, p in written.items()},
    }
    dio.write_text(json.dumps(manifest, indent=2, default=str), manifest_path)

    print(f"redactions applied : {stats['redactions_applied']}")
    print(f"rows passed        : {passed}")
    print(f"rows unreviewed    : {stats['rows_unreviewed']}")
    if stats["rows_unknown_verdict"]:
        print(f"rows with an unrecognised verdict: {stats['rows_unknown_verdict']}")
    for name, p in written.items():
        print(f"  wrote {name:<6} -> {p}")
    print(f"  wrote manifest -> {manifest_path}")
    if pending:
        print(
            "\n  PUBLISHED WITH A KNOWN GAP: "
            f"{pending} flagged row(s) went out unruled.\n"
            "  The manifest records it. Close the queue and re-run to remove it."
        )
    else:
        print("\n  Queue closed: every flagged row has a verdict.")
    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    contract = Contract.from_yaml(args.contract)
    paths = dio.discover(args.directory)
    domain = args.domain or contract.risk.domain
    if domain not in paths:
        return _err(f"domain {domain!r} not found in {args.directory}")

    frame = dio.read_table(paths[domain])
    qis = args.qi or contract.quasi_identifiers(domain)
    if not qis:
        return _err(
            f"no quasi-identifiers declared for {domain}; flag them in the "
            "contract with is_quasi_identifier, or pass --qi"
        )
    report = risk_mod.measure(
        frame,
        qis,
        domain=domain,
        k_target=args.k_target or contract.risk.k_target,
        l_target=contract.risk.l_target,
        sensitive_columns=contract.risk.sensitive_columns,
    )
    print(report.summary())
    if args.out:
        Path(args.out).write_text(
            json.dumps(report.to_dict(include_class_values=True), indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_textcheck(args: argparse.Namespace) -> int:
    frames, _ = dio.load_study(args.directory)
    profiles = textcheck.characterise_study(frames)
    if not profiles:
        return _err("no --TERM / --TRT columns found")

    print("Is each text column verbatim, or already controlled vocabulary?")
    print()
    print(f"  {'column':<22} {'verdict':<11} evidence")
    print("  " + "-" * 96)
    for p in profiles:
        print(p.line())
    print()
    for p in profiles:
        print(f"{p.domain}.{p.column}  ->  {p.verdict}")
        for r in p.reasons:
            print(f"    - {r}")
        print()

    verbatim = [p for p in profiles if p.verdict in ("verbatim", "mixed")]
    if verbatim:
        print(
            "Columns reading as verbatim need screening and adjudication before\n"
            "they enter a training corpus: a human typed them, so a human may\n"
            "have typed an identifier into them."
        )
    else:
        print(
            "Every column looks like controlled vocabulary. The value space is\n"
            "closed and nobody typed prose into it, so the free-text concern\n"
            "largely does not apply -- treat them as coded fields.\n"
            "Rare terms remain quasi-identifiers either way."
        )
    if args.out:
        import json
        Path(args.out).write_text(
            json.dumps([p.to_dict() for p in profiles], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_vault(args: argparse.Namespace) -> int:
    try:
        vault = _open_vault(args)
    except VaultError as exc:
        return _err(str(exc))
    with vault:
        if args.action == "stats":
            print(json.dumps(vault.stats(), indent=2))
        else:
            log = vault.access_log()
            if not log:
                print("no reverse lookups recorded")
            else:
                for row in log:
                    print(
                        f"{row['at']}  {row['action']:<12} {row['entity']}/"
                        f"{row['surrogate']}  by {row['operator']}: "
                        f"{row['justification']}"
                    )
    return 0


def cmd_reverse(args: argparse.Namespace) -> int:
    try:
        vault = _open_vault(args)
    except VaultError as exc:
        return _err(str(exc))
    with vault:
        original = vault.reverse(
            args.entity, args.surrogate, justification=args.justification
        )
    if original is None:
        print(f"no mapping for {args.entity}/{args.surrogate} (the miss was logged)")
        return 1
    print(original)
    print(
        f"\nThis lookup was written to the vault access log under operator "
        f"{_operator(args)!r}.",
        file=sys.stderr,
    )
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deidkit",
        description="De-identify inbound EDC data for analysis and model training.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_vault_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--vault", required=True, help="crosswalk vault file")
        sp.add_argument("--operator", help="recorded in the manifest and access log")
        sp.add_argument(
            "--key-uri",
            help="where the vault key comes from: env:VAR, file:/path, "
            "awskms:<arn>?blob=<path>, azurekv:<vault-url>/<secret>, "
            "gcpkms:<resource>?blob=<path>. Defaults to $DEIDKIT_KEY_URI, "
            "then $DEIDKIT_VAULT_KEY.",
        )

    # keygen
    sp = sub.add_parser("keygen", help="generate or provision a vault key")
    sp.add_argument(
        "--key-uri",
        help="provision the key at this URI instead of printing one to stdout. "
        "Preferred for cloud deployments: the plaintext key never reaches a "
        "terminal or a shell history.",
    )
    sp.set_defaults(func=cmd_keygen)

    # profile
    sp = sub.add_parser("profile", help="profile a drop and draft a contract")
    sp.add_argument("directory")
    sp.add_argument("-o", "--out", default="contract.draft.yaml")
    sp.add_argument("--review", help="also write a steward review sheet (CSV)")
    sp.add_argument(
        "--decisions",
        help="write the DECISION SHEET here (CSV): one row per column, the "
        "proposed treatment and its parameters, lowest confidence first. Fill "
        "in the decision column, then 'deidkit approve' turns it back into a "
        "contract signed against those exact rules. Without an approval, "
        "'deidkit run' refuses to execute.",
    )
    sp.add_argument(
        "--carry-forward",
        metavar="APPROVED.YAML",
        help="pre-fill decisions from a previously APPROVED contract for this "
        "source. A column whose proposed rule is identical to the approved one "
        "comes back marked OK; anything new or changed comes back blank, and "
        "therefore blocking. This is what makes the second drop of a study "
        "cheap to review without making it a formality.",
    )
    sp.add_argument("--source", help="study / source identifier")
    sp.add_argument(
        "--tier", choices=["lds", "deidentified"], default="deidentified"
    )
    sp.add_argument("--anchor-domain")
    sp.add_argument("--anchor-date")
    sp.add_argument("--k-target", type=int, default=5)
    sp.add_argument(
        "--sdtm",
        action="store_true",
        help="the output must be conformant SDTM. Dates are shifted by a "
        "per-subject offset instead of being replaced by study days, so --DTC "
        "survives as a valid ISO date. Use this whenever the deliverable is "
        "SDTM rather than an analysis dataset.",
    )
    sp.add_argument(
        "--keep-dates",
        action="store_true",
        help="retain --DTC exactly as recorded, no shift and no study-day "
        "conversion. Forces tier: lds, because HIPAA enumerates dates as "
        "identifiers and only a Limited Data Set may carry them -- an LDS "
        "remains PHI, needs a DUA, and cannot feed a training corpus.",
    )
    sp.add_argument(
        "--raw",
        action="store_true",
        help="this drop is raw EDC, the input side of a raw -> SDTM training "
        "pair. Date columns are found by their values rather than a --DTC "
        "suffix, and shifted with their written form intact (19/03/2025 stays "
        "d/m/y) -- the conversion to ISO is what the model has to learn, so "
        "normalising it here would delete the task. Point this and the SDTM "
        "drop at the SAME vault: the offset is per subject, so both sides move "
        "together and the mapping between them holds exactly.",
    )
    sp.add_argument(
        "--force-decisions",
        action="store_true",
        help="overwrite a decision sheet that already has decisions in it. "
        "Starting a review over is a legitimate thing to do; doing it by "
        "accident is not, which is why it needs saying.",
    )
    sp.add_argument(
        "--offset-key",
        help="how to build the date-offset key from each table's own columns, "
        "e.g. 'TIG-2026-001-US-{SUBJECT}'. Needed when the two sides of a pair "
        "identify subjects differently -- SDTM keys on USUBJID, a raw extract "
        "usually holds only the site-subject number. Both sides must resolve "
        "to the SAME string or they get unrelated offsets and the pair breaks "
        "while both files still look correct.",
    )
    sp.add_argument(
        "--id-template",
        help="rebuild the subject identifier around its surrogate, e.g. "
        "'{STUDYID}-US-{value}'. For the SDTM side of a pair: USUBJID really "
        "is the study, the country and the raw subject number joined together, "
        "and a derivation model should learn that. Without it both sides "
        "publish the same bare surrogate and the corpus teaches "
        "'USUBJID = SUBJECT', which is true of no real study.",
    )
    sp.add_argument(
        "--blind-treatment",
        action="store_true",
        help="relabel treatment names to TRT A / TRT B across ARM, ARMCD, "
        "ACTARM and EXTRT. For blinding and commercial confidentiality, not "
        "privacy -- an arm identifies nobody. Reversible via the vault; "
        "Placebo passes through.",
    )
    sp.set_defaults(func=cmd_profile)

    # treatments
    sp = sub.add_parser(
        "treatments",
        help="what decision_treatment may be set to, and what each one needs",
    )
    sp.set_defaults(func=cmd_treatments)

    # approve
    sp = sub.add_parser(
        "approve",
        help="sign off a filled-in decision sheet -> an approved contract",
    )
    sp.add_argument("plan", help="the decision sheet CSV, with decisions filled in")
    sp.add_argument(
        "-c", "--contract", required=True,
        help="the DRAFT contract the sheet was written from. The sheet carries "
        "per-column decisions; structure (anchor, subject keys, k target) lives "
        "here, because a flat CSV cannot express the structural checks.",
    )
    sp.add_argument(
        "-o", "--out", required=True, help="where to write the approved contract"
    )
    sp.add_argument(
        "--data",
        help="the drop the sheet was written for. Strongly recommended: it is "
        "the only check that catches a sheet approved against a different "
        "extract.",
    )
    sp.add_argument(
        "--approved-by",
        help="who is approving. Defaults to --operator / $USER. Should be a "
        "person, not a service account.",
    )
    sp.add_argument("--note", help="free text recorded with the approval")
    sp.add_argument("--operator")
    sp.set_defaults(func=cmd_approve)

    # run
    sp = sub.add_parser("run", help="transform, screen, measure, publish")
    sp.add_argument("directory")
    sp.add_argument("-c", "--contract", required=True)
    sp.add_argument("-o", "--out", required=True)
    add_vault_args(sp)
    sp.add_argument("--manifest")
    sp.add_argument("--review")
    sp.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    sp.add_argument("--allow-missing", action="store_true")
    sp.add_argument("--no-screen", action="store_true")
    sp.add_argument(
        "--unreviewed",
        action="store_true",
        help="run a contract that no steward has approved. For synthetic data, "
        "development and CI. The manifest records \"reviewed\": false, so the "
        "output cannot later be mistaken for a reviewed tier.",
    )
    sp.set_defaults(func=cmd_run)

    # adjudicate
    sp = sub.add_parser("adjudicate", help="apply a reviewed free-text queue")
    sp.add_argument("directory")
    sp.add_argument("--manifest", help="where to write the final tier's manifest")
    sp.add_argument(
        "--allow-pending",
        action="store_true",
        help="publish even though some flagged rows have no verdict. They go "
        "out exactly as recorded, and the manifest records the gap rather than "
        "hiding it.",
    )
    sp.add_argument(
        "--operator", help="who adjudicated; recorded in the final manifest"
    )
    sp.add_argument("--queue", required=True)
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    sp.set_defaults(func=cmd_adjudicate)

    # risk
    sp = sub.add_parser("risk", help="measure risk on a published tier")
    sp.add_argument("directory")
    sp.add_argument("-c", "--contract", required=True)
    sp.add_argument("--domain")
    sp.add_argument("--qi", nargs="*")
    sp.add_argument("--k-target", type=int)
    sp.add_argument("-o", "--out")
    sp.set_defaults(func=cmd_risk)

    # textcheck
    sp = sub.add_parser(
        "textcheck",
        help="is a --TERM column verbatim text or already controlled vocabulary?",
    )
    sp.add_argument("directory")
    sp.add_argument("-o", "--out", help="write the measurements as JSON")
    sp.set_defaults(func=cmd_textcheck)

    # vault
    sp = sub.add_parser("vault", help="inspect the crosswalk")
    sp.add_argument("action", choices=["stats", "log"])
    add_vault_args(sp)
    sp.set_defaults(func=cmd_vault)

    # reverse
    sp = sub.add_parser("reverse", help="break-glass re-identification (logged)")
    sp.add_argument("surrogate")
    sp.add_argument("--entity", default="subject")
    sp.add_argument(
        "-j",
        "--justification",
        required=True,
        help="safety report or data query reference; written to the access log",
    )
    add_vault_args(sp)
    sp.set_defaults(func=cmd_reverse)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
