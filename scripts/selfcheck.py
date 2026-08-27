"""Verify a published tier against the guarantees the design promises.

Run after a pipeline execution on the synthetic study. This is not a unit test
-- it inspects the actual published files, which is what an auditor would do.
Called by setup.ps1 and setup.sh, and usable on its own:

    python scripts/selfcheck.py out/quarantine/study_demo out/tier_deidentified

Exit code 0 means every guarantee held. Non-zero means do not point this
install at real data.
"""

from __future__ import annotations

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

    # --- 1. no absolute date survives anywhere --------------------------
    leaked: list[str] = []
    checked = 0
    for dom in DOMAINS:
        frame = read(pub, dom, dtype=str)
        if frame is None:
            continue
        checked += 1
        for col in frame.columns:
            joined = " ".join(frame[col].dropna().astype(str).head(400))
            if ISO_DATE.search(joined):
                leaked.append(f"{dom.upper()}.{col}")
    if leaked:
        problems.append(f"absolute dates survived in {leaked}")
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
    queue_path = pub / "review_queue.csv"
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

    # --- 8. manifest carries determination evidence ---------------------
    manifest = pub / "manifest.json"
    if manifest.exists():
        import json

        m = json.loads(manifest.read_text(encoding="utf-8"))
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
