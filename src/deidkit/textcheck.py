"""Is this column verbatim text, or a controlled vocabulary?

The question decides how much care ``AETERM`` needs. In SDTM the answer is
supposed to be clear -- ``--TERM`` is the term as reported by the investigator,
``--DECOD`` is the dictionary-derived MedDRA Preferred Term -- but real studies
blur it. Some EDC builds make the site pick from a coded list, so the
"verbatim" field arrives already controlled. Some sponsors overwrite the
verbatim with the coded term before handing the data over.

It matters because the two cases call for different handling. Controlled
vocabulary has a closed value space that nobody typed prose into, so it can
enter a training corpus with the care owed to any coded field. Genuine verbatim
needs screening and adjudication first.

So this measures rather than assumes, using signals that do not require
knowing the dictionary:

* how often the term equals its coded counterpart
* how many distinct terms share one code -- a pick-list sits at 1.00
* whether values read as typed prose: length, punctuation, embedded numbers
* whether the PHI detector finds anything at all

The verdict is deliberately asymmetric. Calling verbatim text "controlled"
skips review and publishes whatever an investigator typed; calling controlled
vocabulary "verbatim" costs a reviewer some wasted minutes. "Controlled" has to
be earned on every axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .freetext import build_detector

Verdict = Literal["controlled", "verbatim", "mixed", "unknown"]

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(v: object) -> str:
    return _NORM.sub(" ", str(v).lower()).strip()


@dataclass
class TextProfile:
    domain: str
    column: str
    coded_column: str | None
    n: int
    n_distinct: int
    match_coded_rate: float | None
    terms_per_code: float | None
    mean_words: float
    sentence_rate: float
    detector_hit_rate: float
    verdict: Verdict
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "column": self.column,
            "coded_column": self.coded_column,
            "rows": self.n,
            "distinct_values": self.n_distinct,
            "matches_coded_term": self.match_coded_rate,
            "distinct_terms_per_code": self.terms_per_code,
            "mean_words": round(self.mean_words, 2),
            "sentence_like_rate": round(self.sentence_rate, 3),
            "detector_hit_rate": round(self.detector_hit_rate, 4),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }

    def line(self) -> str:
        m = "   n/a" if self.match_coded_rate is None else f"{self.match_coded_rate:6.1%}"
        t = " n/a" if self.terms_per_code is None else f"{self.terms_per_code:4.2f}"
        col = f"{self.domain}.{self.column}"
        return (
            f"  {col:<14} {self.verdict:<11} matches {m}   "
            f"terms/code {t}   words {self.mean_words:4.1f}   "
            f"detector {self.detector_hit_rate:5.1%}"
        )


def _guess_coded_column(frame: pd.DataFrame, column: str) -> str | None:
    for cand in (
        column.replace("TERM", "DECOD"),
        column.replace("TRT", "DECOD"),
        column[:2] + "DECOD",
    ):
        if cand != column and cand in frame.columns:
            return cand
    return None


def characterise(
    frame: pd.DataFrame,
    column: str,
    *,
    domain: str = "",
    coded_column: str | None = None,
    detector=None,
    sample: int = 2000,
) -> TextProfile:
    """Measure one text column against its coded counterpart."""
    if coded_column is None:
        coded_column = _guess_coded_column(frame, column)

    values = frame[column].dropna().astype(str)
    if values.empty:
        return TextProfile(
            domain, column, coded_column, 0, 0, None, None, 0.0, 0.0, 0.0,
            "unknown", ["column is empty"],
        )

    words = values.str.split().str.len()
    mean_words = float(words.mean())
    # A comma or full stop, or a multi-digit number, reads as typed prose
    # rather than a dictionary entry.
    sentence = float(values.str.contains(r"[.;,]|\d{2,}", regex=True).mean())

    match_rate: float | None = None
    terms_per_code: float | None = None
    if coded_column and coded_column in frame.columns:
        pair = frame[[column, coded_column]].dropna()
        if len(pair):
            left = pair[column].map(_norm)
            right = pair[coded_column].map(_norm)
            match_rate = float((left == right).mean())
            terms_per_code = float(pair.groupby(right)[column].nunique().mean())

    det = detector or build_detector()
    probe = values.drop_duplicates().head(sample)
    hit_rate = (
        sum(1 for v in probe if det.detect(v)) / len(probe) if len(probe) else 0.0
    )

    # --- verdict --------------------------------------------------------
    reasons: list[str] = []
    signals: list[bool] = []

    diverges = match_rate is not None and match_rate < 0.95
    signals.append(bool(diverges))
    if match_rate is None:
        reasons.append(
            "no coded counterpart to compare against, so the strongest signal "
            "is unavailable"
        )
    elif diverges:
        reasons.append(
            f"{1 - match_rate:.0%} of values differ from {coded_column} -- the "
            "text carries wording the coding does not (abbreviations, "
            "shorthand, added detail), which is what a human typing looks like"
        )
    else:
        reasons.append(
            f"{match_rate:.0%} of values equal {coded_column}: the column has "
            "effectively been replaced by the coded term"
        )

    many_terms = terms_per_code is not None and terms_per_code >= 1.15
    signals.append(bool(many_terms))
    if many_terms:
        reasons.append(
            f"{terms_per_code:.2f} distinct terms per code -- a pick-list would "
            "sit at 1.00"
        )

    prose = sentence >= 0.15 or mean_words >= 3.5
    signals.append(prose)
    if prose:
        reasons.append(
            f"values read as typed prose: {mean_words:.1f} words on average, "
            f"{sentence:.0%} carrying punctuation or embedded numbers"
        )

    detected = hit_rate > 0
    signals.append(detected)
    if detected:
        reasons.append(
            f"the detector already flags {hit_rate:.1%} of distinct values -- "
            "settles it on its own"
        )

    n_signals = sum(signals)
    if detected or n_signals >= 2:
        verdict: Verdict = "verbatim"
    elif n_signals == 1:
        verdict = "mixed"
        reasons.append(
            "one signal only, so this is treated as needing review: "
            "unnecessary review costs minutes, skipped review costs "
            "unreviewed identifiers"
        )
    elif match_rate is None:
        verdict = "unknown"
        reasons.append(
            "nothing suggests free text, but with no coded counterpart that is "
            "weak evidence -- read the samples before trusting it"
        )
    else:
        verdict = "controlled"
        reasons.append(
            "matches the coded term, one term per code, short values, no "
            "detector hits: controlled vocabulary on every axis"
        )

    return TextProfile(
        domain, column, coded_column, len(values), int(values.nunique()),
        match_rate, terms_per_code, mean_words, sentence, hit_rate,
        verdict, reasons,
    )


def characterise_study(
    frames: dict[str, pd.DataFrame], *, detector=None
) -> list[TextProfile]:
    """Characterise every plausible verbatim column across a study."""
    det = detector or build_detector()
    out: list[TextProfile] = []
    for dom, frame in sorted(frames.items()):
        for col in frame.columns:
            up = str(col).upper()
            if up.endswith(("TERM", "TRT", "COMMENT", "NARR", "MODIFY")):
                out.append(characterise(frame, str(col), domain=dom, detector=det))
    return out
