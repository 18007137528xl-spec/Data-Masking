"""Blinding: keeping the treatment out of the published data.

Relabelling ``ARM`` to ``TRT A`` is the easy half. The treatment leaks through
several other doors, and for a training corpus the worst of them is free text:
an investigator writes "rash 3 days after pembrolizumab infusion" into
``AETERM``, and the label on ``ARM`` is now decoration. Free text is also
exactly where a language model memorises, so a leak there survives into the
weights.

Three doors, and this module handles all three:

1. **Free text naming the drug.** The terms are not guessed -- they are derived
   from the treatment values the contract is already relabelling, so the
   detector always knows precisely what it is looking for.
2. **Dose and regimen.** ``EXDOSE = 200`` against ``EXDOSE = 400`` separates
   two arms perfectly, whatever ``ARM`` says. The profiler flags these; a
   steward decides, because pooling dose destroys exposure-response analysis
   and that may matter more than the blind.
3. **Anything missed.** :func:`audit` scans the published output for the
   original treatment strings and their distinctive tokens. Blinding you have
   not verified is blinding you are hoping for.

None of this is a privacy control. A treatment arm identifies nobody. This is
commercial confidentiality and blinded review, a separate axis from PHI, and
worth keeping separate in the head as well as the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

from .freetext import Finding

#: Words that appear in treatment strings but say nothing about which arm.
#: Matching on these would flag every row and teach reviewers to click through.
_GENERIC = frozenset(
    x.lower()
    for x in (
        "placebo", "control", "vehicle", "sham", "comparator", "standard",
        "care", "arm", "group", "cohort", "dose", "doses", "dosing",
        "tablet", "tablets", "capsule", "capsules", "injection", "infusion",
        "solution", "suspension", "syrup", "patch", "vial",
        "oral", "orally", "intravenous", "iv", "subcutaneous", "sc", "im",
        "topical", "inhaled", "nasal",
        "daily", "weekly", "monthly", "once", "twice", "thrice",
        "morning", "evening", "night", "bid", "tid", "qid", "qd", "prn",
        "high", "low", "mid", "medium", "and", "the", "with", "plus",
        "open", "label", "blinded", "double", "single", "active", "titration",
        "mg", "mcg", "ug", "kg", "ml", "unit", "units", "iu",
    )
)

#: A dose or regimen alone can separate the arms. Names, not suffixes: --DOSE
#: catches EXDOSE and CMDOSE alike, and only the study drug's matters.
DOSE_COLUMNS = (
    "EXDOSE", "EXDOSTOT", "EXDOSU", "EXDOSFRM", "EXDOSFRQ", "EXDOSRGM",
    "EXADJ", "EXTRTV",
)


def derive_terms(values: Iterable[str]) -> list[str]:
    """Distinctive terms for a set of treatment strings.

    ``"Pembrolizumab 200 mg Q3W"`` yields the full string, ``pembrolizumab``,
    and the regimen code ``q3w`` -- but not ``mg``, and not ``placebo``, which
    would match everywhere and say nothing.
    """
    terms: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in _GENERIC:
            continue
        terms.add(text.lower())
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text):
            low = tok.lower()
            if low in _GENERIC or low.isdigit():
                continue
            # Drug names are long; regimen codes (Q3W, BIW) are short but
            # distinctive because they mix letters and digits.
            if len(low) >= 5 or (any(c.isdigit() for c in low) and len(low) >= 3):
                terms.add(low)
    return sorted(terms, key=len, reverse=True)


class StudyDrugRecognizer:
    """Flags free text that names the study treatment.

    Slots into the free-text screen alongside the PHI detectors, so a mention
    of the drug lands in the same review queue by the same route -- with its
    own entity type, so a reviewer can see it is a blinding finding rather
    than a privacy one.
    """

    name = "study-drug"
    entity_type = "STUDY_DRUG"

    def __init__(self, terms: Sequence[str]) -> None:
        self.terms = [t for t in terms if t]
        self._pattern = (
            re.compile(
                r"\b(" + "|".join(re.escape(t) for t in self.terms) + r")\b",
                re.IGNORECASE,
            )
            if self.terms
            else None
        )

    def detect(self, text: str) -> list[Finding]:
        if not self._pattern:
            return []
        return [
            Finding(
                self.entity_type, m.start(), m.end(), m.group(), 0.9, self.name
            )
            for m in self._pattern.finditer(text)
        ]


class CompositeDetector:
    """Runs a PHI detector and a study-drug recognizer over the same text."""

    def __init__(self, base, drug: StudyDrugRecognizer | None) -> None:
        self._base = base
        self._drug = drug
        self.name = base.name + ("+study-drug" if drug and drug.terms else "")

    def detect(self, text: str) -> list[Finding]:
        found = list(self._base.detect(text))
        if self._drug:
            taken = [(f.start, f.end) for f in found]
            for f in self._drug.detect(text):
                if not any(f.start < e and s < f.end for s, e in taken):
                    found.append(f)
        return sorted(found, key=lambda f: f.start)


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------


@dataclass
class BlindingLeak:
    domain: str
    column: str
    term: str
    n_rows: int
    example: str

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "column": self.column,
            "term": self.term,
            "rows": self.n_rows,
            "example": self.example[:160],
        }


@dataclass
class BlindingReport:
    terms_checked: list[str] = field(default_factory=list)
    columns_scanned: int = 0
    leaks: list[BlindingLeak] = field(default_factory=list)

    @property
    def held(self) -> bool:
        return not self.leaks

    def to_dict(self) -> dict[str, object]:
        return {
            "terms_checked": self.terms_checked,
            "columns_scanned": self.columns_scanned,
            "held": self.held,
            "leaks": [x.to_dict() for x in self.leaks],
        }

    def summary(self) -> str:
        if not self.terms_checked:
            return "blinding: not requested"
        if self.held:
            return (
                f"blinding: held -- {len(self.terms_checked)} term(s) absent "
                f"from {self.columns_scanned} published columns"
            )
        lines = [f"blinding: LEAKED in {len(self.leaks)} place(s)"]
        for leak in self.leaks[:8]:
            lines.append(
                f"  {leak.domain}.{leak.column}: {leak.term!r} in "
                f"{leak.n_rows} row(s) -- {leak.example[:70]!r}"
            )
        return "\n".join(lines)


def audit(
    frames: dict[str, pd.DataFrame],
    terms: Sequence[str],
    *,
    skip: Sequence[tuple[str, str]] = (),
) -> BlindingReport:
    """Scan published output for the treatment terms.

    The check that matters, because everything upstream is intent and this is
    outcome. Run on the transformed frames, so anything the relabelling and
    the free-text screen between them failed to catch shows up here.
    """
    report = BlindingReport(terms_checked=list(terms))
    if not terms:
        return report

    skipset = {(d.upper(), c.upper()) for d, c in skip}
    patterns = {
        t: re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in terms
    }

    for dom, frame in frames.items():
        for col in frame.columns:
            if (dom.upper(), str(col).upper()) in skipset:
                continue
            series = frame[col]
            if series.dtype.kind in "ifb":  # numeric columns cannot spell a name
                continue
            report.columns_scanned += 1
            text = series.dropna().astype(str)
            if text.empty:
                continue
            for term, pat in patterns.items():
                hit = text[text.str.contains(pat, na=False)]
                if not hit.empty:
                    report.leaks.append(
                        BlindingLeak(dom, str(col), term, len(hit), hit.iloc[0])
                    )
    return report
