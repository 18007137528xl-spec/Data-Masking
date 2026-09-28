"""Verify a published tier against the guarantees the design promises.

Run after a pipeline execution on the synthetic study. This is not a unit test
-- it inspects the actual published files, which is what an auditor would do.
Called by setup.ps1 and setup.sh, and usable on its own:

    python scripts/selfcheck.py out/quarantine/study_demo out/tier_deidentified

Exit code 0 means every guarantee held. Non-zero means do not point this
install at real data.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ISO_DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
DOMAINS = ("dm", "ae", "mh", "lb", "vs")

RETAINED_IN_FULL = {
    "ae": ["AETERM", "AEDECOD", "AEBODSYS", "AESEV", "AESER", "AEREL", "AEOUT", "AEACN"],
    "mh": ["MHTERM", "MHDECOD", "MHBODSYS", "MHONGO"],
}

MUST_BE_GONE = ("INVNAM", "BRTHDTC", "SUBJPHONE", "SUBJID")


def check(raw_dir: str, pub_dir: str) -> tuple[list[str], list[str]]:
    raw, pub = Path(raw_dir), Path(pub_dir)
    problems: list[str] = []
    passed: list[str] = []

    def read(base: Path, dom: str, **kw) -> pd.DataFrame | None:
        for ext in ("csv", "parquet"):
            p = base / f"{dom}.{ext}"
            if p.exists():
                if ext == "csv":
                    return pd.read_csv(p, **kw)
                return pd.read_parquet(p)
        return None

    # --- 0. what did this run claim to be? ------------------------------
    # The date check depends on the tier, so read it first. An LDS may carry
    # full dates (164.514(e)); a de-identified tier may not.
    manifest_path = pub / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tier = manifest.get("tier", "deidentified")

    # --- 1. dates match what the tier promises --------------------------
    dated: list[str] = []
    checked = 0
    for dom in DOMAINS:
        frame = read(pub, dom, dtype=str)
        if frame is None:
            continue
        checked += 1
        for col in frame.columns:
            joined = " ".join(frame[col].dropna().astype(str).head(400))
            if ISO_DATE.search(joined):
                dated.append(f"{dom.upper()}.{col}")

    if tier == "lds":
        if dated:
            passed.append(
                f"tier=lds: calendar dates retained in {len(dated)} column(s), "
                "which is what an LDS is for -- and why it stays PHI"
            )
        elif checked:
            problems.append(
                "tier=lds but no calendar dates found; if the dates were "
                "converted anyway, the tier should be 'deidentified'"
            )
    else:
        if dated:
            problems.append(f"tier={tier} but calendar dates survived in {dated}")
        elif checked:
            passed.append(f"no absolute dates survived any of {checked} domains")

    # --- 2. MH / AE clinical content byte-identical ---------------------
    modified: list[str] = []
    compared = 0
    for dom, cols in RETAINED_IN_FULL.items():
        before, after = read(raw, dom, dtype=str), read(pub, dom, dtype=str)
        if before is None or after is None:
            continue
        for col in cols:
            if col not in before.columns or col not in after.columns:
                continue
            compared += 1
            if not before[col].fillna("").equals(after[col].fillna("")):
                modified.append(f"{dom.upper()}.{col}")
    if modified:
        problems.append(
            f"columns declared retained-in-full were modified: {modified}"
        )
    elif compared:
        passed.append(
            f"MH and AE: {compared} clinical columns byte-identical to input"
        )

    # --- 3. direct identifiers removed ----------------------------------
    dm_pub = read(pub, "dm", dtype=str)
    if dm_pub is not None:
        still_there = [c for c in MUST_BE_GONE if c in dm_pub.columns]
        if still_there:
            problems.append(f"direct identifiers not dropped: {still_there}")
        else:
            passed.append("direct identifiers removed from DM")

    # --- 4. surrogates carry nothing of the original --------------------
    dm_raw = read(raw, "dm", dtype=str)
    if dm_raw is not None and dm_pub is not None and "USUBJID" in dm_pub.columns:
        embedded = None
        for orig, surr in zip(dm_raw["USUBJID"], dm_pub["USUBJID"]):
            tail = str(orig).split("-")[-1]
            if tail and tail in str(surr):
                embedded = (orig, surr)
                break
        if embedded:
            problems.append(
                f"a surrogate embeds the original: {embedded[0]} -> {embedded[1]}"
            )
        elif set(dm_raw["USUBJID"]) & set(dm_pub["USUBJID"].dropna()):
            problems.append("some subject identifiers passed through unchanged")
        else:
            n = dm_pub["USUBJID"].nunique()
            passed.append(f"{n} surrogates issued, none derived from the original")

    # --- 5. AE intervals recoverable ------------------------------------
    ae_pub = read(pub, "ae")
    if ae_pub is not None and {"AESTDY", "AEENDY"} <= set(ae_pub.columns):
        dur = (ae_pub["AEENDY"] - ae_pub["AESTDY"]).dropna()
        if len(dur) == 0:
            problems.append("no AE durations are recoverable from study days")
        elif (dur < 0).any():
            problems.append("negative AE durations: the anchor or parsing is wrong")
        else:
            passed.append(f"AE intervals recoverable for {len(dur)} events")

    # --- 6. joins survive ------------------------------------------------
    if dm_pub is not None and "USUBJID" in dm_pub.columns:
        dm_subjects = set(dm_pub["USUBJID"].dropna())
        orphaned: list[str] = []
        for dom in ("ae", "mh", "lb", "vs"):
            frame = read(pub, dom, dtype=str)
            if frame is None or "USUBJID" not in frame.columns:
                continue
            if not set(frame["USUBJID"].dropna()) <= dm_subjects:
                orphaned.append(dom.upper())
        if orphaned:
            problems.append(f"subjects in {orphaned} do not resolve into DM")
        else:
            passed.append("joins intact: every domain resolves into DM's surrogates")

    # --- 7. screening produced a queue and changed nothing ---------------
    # The queue lives BESIDE the published tier, not inside it: it holds the
    # original text of every flagged row, so it is unredacted PHI and belongs
    # under the vault's access controls rather than the analysts'.
    candidates = [
        pub.parent / f"{pub.name}_review" / "review_queue.csv",
        pub / "review_queue.csv",  # older layout
    ]
    queue_path = next((p for p in candidates if p.exists()), candidates[0])
    if (pub / "review_queue.csv").exists():
        problems.append(
            "the review queue is inside the published tier; it holds original "
            "unredacted text and must not sit where analysts read"
        )
    if (pub / "risk_detail.json").exists():
        problems.append(
            "the risk detail is inside the published tier; it lists the "
            "quasi-identifier values of the most exposed subjects, which is a "
            "targeting aid and belongs beside the review queue"
        )

    # --- 7b. the manifest must not carry the exposed subjects' QI values ----
    # The manifest is the artifact that travels: attached to a determination,
    # pasted into a ticket, mailed to a reviewer. Away from the data, a ranked
    # list of who is unique on the quasi-identifier set is exactly what an
    # attacker would want and the opposite of what the manifest is for.
    mf = pub / "manifest.json"
    if mf.exists():
        risk = json.loads(mf.read_text(encoding="utf-8")).get("risk") or {}
        if "smallest_classes" in risk:
            problems.append(
                "the manifest carries smallest_classes with quasi-identifier "
                "values; it should carry the distribution and counts only"
            )
        elif risk:
            passed.append(
                f"manifest risk is counts only: {risk.get('n_classes_below_k')} "
                f"class(es) below k, values held in the review sibling"
            )
    ae_raw = read(raw, "ae", dtype=str)
    if queue_path.exists() and ae_raw is not None:
        q = pd.read_csv(queue_path)
        rate = len(q) / len(ae_raw) if len(ae_raw) else 0
        if not q.empty and not (q["verdict"].fillna("") == "").all():
            problems.append("review queue arrived with verdicts pre-filled")
        else:
            passed.append(
                f"free-text queue: {len(q)} of {len(ae_raw)} AE rows flagged "
                f"({rate:.1%}), all awaiting adjudication"
            )
    elif ae_raw is not None:
        problems.append("no review queue was produced; screening did not run")

    # --- 7b. treatment names masked, where blinding was asked for -------
    terms = (manifest.get("blinding") or {}).get("terms_checked") or []
    if terms:
        import re as _re

        arm_leaks: list[str] = []
        for dom in ("dm", "ex"):
            frame = read(pub, dom, dtype=str)
            if frame is None:
                continue
            for col in ("ARM", "ACTARM", "ARMCD", "ACTARMCD", "EXTRT"):
                if col not in frame.columns:
                    continue
                joined = " ".join(frame[col].dropna().astype(str).head(400))
                for t in terms:
                    if _re.search(r"\b" + _re.escape(t) + r"\b", joined, _re.I):
                        arm_leaks.append(f"{dom.upper()}.{col}")
                        break
        if arm_leaks:
            problems.append(
                f"treatment columns still name the compound: {sorted(set(arm_leaks))}"
            )
        else:
            held = (manifest.get("blinding") or {}).get("held")
            note = "" if held else " (free text or dose columns still leak -- see the run output)"
            passed.append(
                f"treatment relabelled in every arm column, {len(terms)} term(s) "
                f"checked{note}"
            )

    # --- 8. manifest carries determination evidence ---------------------
    if manifest:
        m = manifest
        required = [
            ("tier", m.get("tier")),
            ("contract version", m.get("contract", {}).get("contract_version")),
            ("input checksums", m.get("inputs", {}).get("checksums_sha256")),
            ("risk metrics", m.get("risk")),
        ]
        missing = [name for name, value in required if not value]
        if missing:
            problems.append(f"manifest is missing {missing}")
        else:
            k = m["risk"].get("k_min")
            target = m["risk"].get("k_target")
            note = "" if m["risk"].get("k_met") else f" (below target {target})"
            passed.append(
                f"manifest complete: tier={m['tier']}, k={k}{note}, "
                f"{len(m['inputs']['checksums_sha256'])} inputs checksummed"
            )
    else:
        problems.append("no manifest was written")

    return problems, passed


def main(argv: list[str]) -> int:
    raw = argv[1] if len(argv) > 1 else "out/quarantine/study_demo"
    pub = argv[2] if len(argv) > 2 else "out/tier_deidentified"

    if not Path(raw).is_dir() or not Path(pub).is_dir():
        print(f"FAIL missing directory: {raw} or {pub}")
        return 2

    problems, passed = check(raw, pub)
    for p in passed:
        print(f"PASS {p}")
    for p in problems:
        print(f"FAIL {p}")

    if problems:
        print(
            "FAIL one or more guarantees did not hold -- do not point this "
            "install at real data"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
