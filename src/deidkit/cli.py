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
import sys
from pathlib import Path

import pandas as pd

from . import freetext, io as dio, profile as prof, risk as risk_mod
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
    )

    table = prof.suggestion_table(suggestions)
    low = table[table["confidence"] == "low"]

    print(f"profiled {len(frames)} domain(s): {', '.join(sorted(frames))}")
    print(f"drafted {len(table)} column rules")
    print(f"anchor: {contract.anchor.domain}.{contract.anchor.date_column}")
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

    queue_path = args.review or f"{str(args.out).rstrip('/')}/review_queue.csv"
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
