"""Free-text screening for verbatim clinical fields.

The governing decision: **screen and adjudicate, never blanket-redact.**

Adverse-event and medical-history verbatim terms are required for regulatory
review and carry clinical nuance that MedDRA coding loses, so this module does
not modify data. It produces a *review queue* of candidate rows for a data
manager to adjudicate. Unflagged rows pass through untouched, which is what
makes "MH and AE retained in full" true in practice rather than aspirational.

Why this is affordable: a trial's ``AETERM`` column is a few thousand rows,
detector hit rates run 1-3%, and most hits are false positives from medical
vocabulary that looks like a name or a place. That is tens of rows for a human
to read -- versus betting that no investigator has ever typed a hospital name
into a verbatim field. In real trial data, they have.

Two detector backends:

* **Presidio** (``pip install presidio-analyzer`` plus a spaCy model) when
  available -- proper NER, extensible recognisers.
* A **built-in pattern detector** otherwise, so the pipeline runs anywhere.
  It is deliberately conservative about person names: blanket
  capitalised-bigram matching floods the queue with drug names and MedDRA
  terms, which trains reviewers to click through. Cued detection
  (``Dr. Chen``, ``at Mercy General``) yields a queue people actually read.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd

# ----------------------------------------------------------------------
# findings
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int
    text: str
    score: float
    detector: str


# ----------------------------------------------------------------------
# clinical vocabulary that reads as PHI but is not
# ----------------------------------------------------------------------

#: Tokens that trip person/location detectors but are medical vocabulary.
#: Extend this per therapeutic area -- it is the single highest-leverage way to
#: keep the review queue small enough that reviewers stay attentive.
CLINICAL_ALLOWLIST: frozenset[str] = frozenset(
    x.lower()
    for x in (
        # eponymous conditions and signs
        "Crohn", "Cushing", "Parkinson", "Alzheimer", "Hodgkin", "Bell",
        "Graves", "Paget", "Raynaud", "Wilson", "Addison", "Barrett",
        "Guillain", "Barre", "Sjogren", "Behcet", "Kawasaki", "Reye",
        "Charcot", "Marie", "Tooth", "Duchenne", "Huntington", "Menetrier",
        "Peyronie", "Dupuytren", "Osgood", "Schlatter", "Ehlers", "Danlos",
        "Marfan", "Turner", "Klinefelter", "Down", "Gilbert", "Meniere",
        "Tourette", "Asperger", "Bright", "Pott", "Colles", "Murphy",
        "Babinski", "Romberg", "Homan", "Chvostek", "Trousseau",
        # scales, scores, criteria
        "Glasgow", "Karnofsky", "Ashworth", "Braden", "Norton", "Apgar",
        "Ramsay", "Hamilton", "Beck", "Montgomery", "Asberg", "Bristol",
        "Likert", "Borg", "Wexner", "Mallampati", "Child", "Pugh",
        "Framingham", "Cockcroft", "Gault", "Modification", "RECIST",
        # anatomical / procedural eponyms
        "Foley", "Swan", "Ganz", "Hickman", "Broviac", "Whipple", "Billroth",
        "Roux", "Hartmann", "Kocher", "Pfannenstiel", "Seldinger", "Valsalva",
        "Purkinje", "Langerhans", "Kupffer", "Schwann", "Bowman", "Henle",
        # geography embedded in disease and organism names
        "Lyme", "Ebola", "Marburg", "Zika", "Denver", "Philadelphia",
        "Mediterranean", "Rocky", "Mountain", "West", "Nile", "German",
        "Spanish", "Japanese", "Indian", "African", "Asian", "European",
        # frequent unit / route / form tokens
        "Ringer", "Hartmann", "Luer", "French", "Charriere",
    )
)

# ----------------------------------------------------------------------
# built-in pattern detector
# ----------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
        0.95,
    ),
    (
        "PHONE_NUMBER",
        re.compile(
            r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?"
            r"(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
        ),
        0.75,
    ),
    (
        "US_SSN",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        0.9,
    ),
    (
        "MEDICAL_RECORD_NUMBER",
        re.compile(
            r"\b(?:MRN|M\.R\.N\.|medical\s+record(?:\s+(?:no|number|#))?)"
            r"[\s:#]*([A-Z0-9-]{4,})\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "URL",
        re.compile(r"\bhttps?://\S+|\bwww\.[\w.-]+\.\w{2,}\b", re.IGNORECASE),
        0.9,
    ),
    (
        "DATE_IN_TEXT",
        re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r"|\d{4}-\d{2}-\d{2}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4})\b",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        "PERSON",
        # Cued only: an honorific or an explicit role, then a capitalised name.
        re.compile(
            r"\b(?:Dr|Doctor|Prof|Professor|Mr|Mrs|Ms|Miss|Nurse|PI|"
            r"investigator|patient|subject|son|daughter|wife|husband|mother|"
            r"father|brother|sister)\.?\s+"
            r"([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,2})\b"
        ),
        0.6,
    ),
    (
        "FACILITY",
        re.compile(
            r"\b((?:[A-Z][\w'&.-]*\s+){0,3}"
            r"(?:Hospital|Clinic|Medical\s+Cent(?:er|re)|Health\s+Cent(?:er|re)|"
            r"Infirmary|Sanatorium|Nursing\s+Home|Hospice|Surgery\s+Cent(?:er|re)|"
            r"Emergency\s+Room|A&E|ER|ICU))\b"
        ),
        0.7,
    ),
    (
        "LOCATION",
        re.compile(
            r"\b(?:in|at|from|near|to)\s+"
            r"((?:St\.?\s+)?[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b"
        ),
        0.4,
    ),
    (
        "ZIP_CODE",
        re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),
        0.35,
    ),
    (
        "AGE_OVER_89",
        re.compile(r"\b(?:9\d|1\d{2})\s*(?:-|\s)?\s*year[s]?[\s-]old\b", re.IGNORECASE),
        0.8,
    ),
)


def _allowlisted(text: str) -> bool:
    """True when every alphabetic token is medical vocabulary."""
    toks = [t for t in re.findall(r"[A-Za-z]+", text) if len(t) > 1]
    if not toks:
        return False
    return all(t.lower() in CLINICAL_ALLOWLIST for t in toks)


class PatternDetector:
    """Dependency-free fallback detector."""

    name = "pattern"

    def detect(self, text: str) -> list[Finding]:
        found: list[Finding] = []
        for entity, pat, score in _PATTERNS:
            for m in pat.finditer(text):
                # Prefer the capture group when the pattern used a cue prefix,
                # so the reviewer sees the candidate identifier, not the cue.
                if m.groups():
                    start, end = m.span(1)
                    frag = m.group(1)
                else:
                    start, end = m.span()
                    frag = m.group()
                if _allowlisted(frag):
                    continue
                found.append(
                    Finding(entity, start, end, frag, score, self.name)
                )
        return _dedupe(found)


class PresidioDetector:  # pragma: no cover - optional dependency
    """Presidio-backed detector, used when the package is importable."""

    name = "presidio"

    def __init__(self, language: str = "en", score_threshold: float = 0.35) -> None:
        from presidio_analyzer import AnalyzerEngine

        self._engine = AnalyzerEngine()
        self._language = language
        self._threshold = score_threshold
        self._fallback = PatternDetector()

    def detect(self, text: str) -> list[Finding]:
        res = self._engine.analyze(
            text=text, language=self._language, score_threshold=self._threshold
        )
        found = [
            Finding(
                r.entity_type,
                r.start,
                r.end,
                text[r.start : r.end],
                float(r.score),
                self.name,
            )
            for r in res
            if not _allowlisted(text[r.start : r.end])
        ]
        # Presidio's NER is strong on names and weak on study-specific ID
        # formats; run both and merge rather than choosing.
        found.extend(self._fallback.detect(text))
        return _dedupe(found)


def _dedupe(findings: Sequence[Finding]) -> list[Finding]:
    """Keep the highest-scoring finding for each overlapping span."""
    ordered = sorted(findings, key=lambda f: (-f.score, f.start, f.end))
    kept: list[Finding] = []
    for f in ordered:
        if any(f.start < k.end and k.start < f.end for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def build_detector(prefer_presidio: bool = True) -> PatternDetector | PresidioDetector:
    if prefer_presidio:
        try:  # pragma: no cover
            return PresidioDetector()
        except Exception:
            pass
    return PatternDetector()


# ----------------------------------------------------------------------
# review queue
# ----------------------------------------------------------------------

REVIEW_COLUMNS = [
    "domain",
    "row_id",
    "subject",
    "column",
    "entity_types",
    "n_findings",
    "max_score",
    "text",
    "marked_text",
    "verdict",
    "replacement",
    "reviewer",
]


def mark(text: str, findings: Sequence[Finding]) -> str:
    """Bracket the candidate spans so a reviewer can see them at a glance."""
    out, cursor = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        out.append(text[cursor : f.start])
        out.append(f"[[{f.entity_type}:{text[f.start:f.end]}]]")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


def screen_column(
    frame: pd.DataFrame,
    column: str,
    *,
    domain: str,
    subject_column: str | None,
    detector: PatternDetector | PresidioDetector,
) -> pd.DataFrame:
    """Screen one free-text column. Returns candidate rows only; data untouched."""
    rows: list[dict[str, object]] = []
    if column not in frame.columns:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    for idx, value in frame[column].items():
        if pd.isna(value):
            continue
        text = str(value)
        if not text.strip():
            continue
        findings = detector.detect(text)
        if not findings:
            continue
        rows.append(
            {
                "domain": domain,
                "row_id": idx,
                "subject": (
                    frame.at[idx, subject_column]
                    if subject_column and subject_column in frame.columns
                    else None
                ),
                "column": column,
                "entity_types": ",".join(sorted({f.entity_type for f in findings})),
                "n_findings": len(findings),
                "max_score": round(max(f.score for f in findings), 3),
                "text": text,
                "marked_text": mark(text, findings),
                "verdict": "",  # PENDING -> reviewer writes PASS or REDACT
                "replacement": "",
                "reviewer": "",
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def screen(
    frames: dict[str, pd.DataFrame],
    targets: Iterable[tuple[str, str, str | None]],
    *,
    detector: PatternDetector | PresidioDetector | None = None,
) -> pd.DataFrame:
    """Screen every ``(domain, column, subject_column)`` target.

    Returns one review queue across all domains, highest-confidence first so a
    reviewer working top-down hits the real findings early.
    """
    det = detector or build_detector()
    parts = [
        screen_column(
            frames[dom], col, domain=dom, subject_column=subj, detector=det
        )
        for dom, col, subj in targets
        if dom in frames
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    queue = pd.concat(parts, ignore_index=True)
    return queue.sort_values(
        ["max_score", "n_findings"], ascending=False
    ).reset_index(drop=True)


def screening_summary(queue: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Hit rates per screened column, for the manifest."""
    if queue.empty:
        return {"flagged_rows": 0, "by_column": {}}
    by: dict[str, dict[str, object]] = {}
    for (dom, col), grp in queue.groupby(["domain", "column"], sort=True):
        total = len(frames[dom]) if dom in frames else 0
        by[f"{dom}.{col}"] = {
            "flagged": int(len(grp)),
            "screened": int(total),
            "hit_rate": round(len(grp) / total, 4) if total else None,
            "entity_types": sorted(
                {
                    e
                    for types in grp["entity_types"]
                    for e in str(types).split(",")
                    if e
                }
            ),
        }
    return {"flagged_rows": int(len(queue)), "by_column": by}


def apply_adjudication(
    frames: dict[str, pd.DataFrame], adjudicated: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Apply a reviewed queue.

    Only rows a human marked ``REDACT`` are changed; ``PASS`` rows are left
    exactly as they were. Rows still blank are unreviewed, and are reported --
    never silently treated as either verdict.
    """
    out = {k: v.copy() for k, v in frames.items()}
    applied, unreviewed, unknown = 0, 0, 0

    for _, row in adjudicated.iterrows():
        verdict = str(row.get("verdict", "") or "").strip().upper()
        if not verdict or verdict == "PENDING":
            unreviewed += 1
            continue
        if verdict == "PASS":
            continue
        if verdict != "REDACT":
            unknown += 1
            continue
        dom, col, rid = row["domain"], row["column"], row["row_id"]
        if dom not in out or col not in out[dom].columns:
            continue
        replacement = row.get("replacement")
        if replacement is None or (
            isinstance(replacement, float) and pd.isna(replacement)
        ):
            replacement = ""
        replacement = str(replacement).strip() or "[REDACTED]"
        out[dom].at[rid, col] = replacement
        applied += 1

    return out, {
        "redactions_applied": applied,
        "rows_unreviewed": unreviewed,
        "rows_unknown_verdict": unknown,
    }


def stats(queue: pd.DataFrame) -> dict[str, object]:
    if queue.empty:
        return {"flagged": 0}
    return {
        "flagged": int(len(queue)),
        "by_entity_type": (
            queue["entity_types"]
            .str.split(",")
            .explode()
            .value_counts()
            .to_dict()
        ),
    }


def finding_to_dict(f: Finding) -> dict[str, object]:
    return asdict(f)
