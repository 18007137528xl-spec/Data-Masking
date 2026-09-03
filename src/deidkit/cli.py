"""Command-line interface.

    deidkit keygen                    generate a vault key
    deidkit profile   <dir>           profile a drop and draft a contract
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
from pathlib import Path

import pandas as pd

from . import freetext, io as dio, profile as prof, risk as risk_mod, textcheck
from .contract import Contract
from .pipeline import ContractMismatch, DeidPipeline
from .vault import Vault, VaultError


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


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
    frames, _ = dio.load_study(args.directory)
    if not frames:
        return _err(f"no readable tables in {args.directory}")

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


def cmd_run(args: argparse.Namespace) -> int:
    contract = Contract.from_yaml(args.contract)
    frames, sums = dio.load_study(args.directory)
    if not frames:
        return _err(f"no readable tables in {args.directory}")

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

    manifest_path = args.manifest or f"{str(args.out).rstrip('/')}/manifest.json"
    result.manifest["vault"] = {"key_source": vault.key_source}
    dio.write_text(
        json.dumps(result.manifest, indent=2, default=str), manifest_path
    )

    # The queue holds the ORIGINAL text of every flagged row -- unredacted PHI,
    # and in a blinded study the compound name. It must not sit in the
    # directory analysts read, so it goes to a sibling by default.
    out_dir = str(args.out).rstrip("/")
    queue_path = args.review or f"{out_dir}_review/review_queue.csv"
    if not result.review_queue.empty:
        dio.write_text(result.review_queue.to_csv(index=False), queue_path)

    print(result.summary())
    print()
    for name, p in written.items():
        print(f"  wrote {name:<6} -> {p}")
    print(f"  wrote manifest -> {manifest_path}")
    if not result.review_queue.empty:
        print(f"  wrote review queue -> {queue_path}")
        print(
            f"\n{len(result.review_queue)} free-text row(s) need adjudication. "
            "Set 'verdict' to PASS or REDACT,\nthen: deidkit adjudicate "
            f"{args.out} --queue {queue_path} --out <final-dir>"
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
    paths = dio.discover(args.directory)
    frames = {n: dio.read_table(p) for n, p in paths.items()}
    queue = pd.read_csv(args.queue)

    out_frames, stats = freetext.apply_adjudication(frames, queue)
    written = dio.write_study(out_frames, args.out, fmt=args.format)

    print(f"redactions applied : {stats['redactions_applied']}")
    print(f"rows passed        : {len(queue) - stats['redactions_applied'] - stats['rows_unreviewed'] - stats['rows_unknown_verdict']}")
    print(f"rows unreviewed    : {stats['rows_unreviewed']}")
    if stats["rows_unknown_verdict"]:
        print(f"rows with an unrecognised verdict: {stats['rows_unknown_verdict']}")
    if stats["rows_unreviewed"]:
        print(
            "\nUnreviewed rows were left unchanged -- they are neither passed nor\n"
            "redacted. Publish only once the queue is empty, or record the\n"
            "remainder as a known gap."
        )
    for name, p in written.items():
        print(f"  wrote {name:<6} -> {p}")
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
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
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
    sp.set_defaults(func=cmd_run)

    # adjudicate
    sp = sub.add_parser("adjudicate", help="apply a reviewed free-text queue")
    sp.add_argument("directory")
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
