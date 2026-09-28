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
        # Severe cutaneous and immune reactions. Stevens-Johnson was missing,
        # which is the worst possible omission in this list: it is one of the
        # AEs a safety reviewer most needs to read, and a detector calling it
        # a person's name put it in front of a reviewer 13 times with REDACT
        # as an available answer.
        "Stevens", "Johnson", "Lyell", "Sweet", "Still", "Quincke",
        "Henoch", "Schonlein", "Wegener", "Goodpasture", "Churg", "Strauss",
        "Takayasu", "Buerger", "Behcet", "Hashimoto", "Graves", "Riedel",
        "Lambert", "Eaton", "Guillain", "Miller", "Fisher", "Devic",
        "Wernicke", "Korsakoff", "Creutzfeldt", "Jakob", "Gehrig",
        "Mallory", "Weiss", "Boerhaave", "Budd", "Chiari", "Zollinger",
        "Ellison", "Peutz", "Jeghers", "Lynch", "Gardner", "Cowden",
        "Fanconi", "Wiskott", "Aldrich", "DiGeorge", "Prader", "Willi",
        "Angelman", "Rett", "Brugada", "Wolff", "White", "Torsades",
        "Raynaud", "Sheehan", "Conn", "Nelson", "Meigs", "Ortner",
        # more scales and criteria
        "Barthel", "Rankin", "Fugl", "Meyer", "Berg", "Tinetti", "Epworth",
        "Zubrod", "Lansky", "Cornell", "Yesavage", "Katz", "Lawton",
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


#: WHO International Nonproprietary Name stems. The INN system exists so that
#: a drug name announces its class in its ending, which makes it the one
#: category of clinical vocabulary that can be recognised without a
#: dictionary -- and an NER model has no idea: Presidio read "Adalimumab" as a
#: person's name 21 times in one 188-row concomitant-medication table.
#:
#: Enumerating drugs is hopeless; enumerating their endings is not.
_INN_STEMS: tuple[str, ...] = (
    # biologics
    "mab", "cept", "kin", "ase", "tide", "parin",
    # small molecules by class
    "nib", "ciclib", "rafenib", "tinib", "zomib", "prazole", "statin",
    "cillin", "mycin", "micin", "cycline", "oxacin", "penem", "cephalo",
    "ceph", "cef", "olol", "pril", "sartan", "dipine", "azepam", "zolam",
    "barbital", "caine", "vir", "navir", "ovir", "fungin", "conazole",
    "azole", "setron", "triptan", "glitazone", "gliptin", "flozin",
    "formin", "sulin", "limus", "sporin", "profen", "coxib", "dronate",
    "trexate", "platin", "rubicin", "taxel", "tecan", "citabine", "fenac",
)

#: Short words that end in a stem by accident. "Case" is not an enzyme.
_INN_FALSE_FRIENDS = frozenset(
    {"case", "base", "phase", "release", "disease", "increase", "decrease",
     "please", "cease", "nurse", "course", "worse", "dose", "close"}
)


def _looks_like_drug_name(token: str) -> bool:
    """Recognise an INN by its ending rather than by a dictionary."""
    low = token.lower()
    if len(low) < 6 or low in _INN_FALSE_FRIENDS:
        return False
    return any(low.endswith(stem) for stem in _INN_STEMS)


#: Entity types the clinical allowlist may suppress. It exists to stop a
#: NER model calling "Adalimumab" a person, so it applies to the types that
#: mistake can produce -- and to nothing else.
#:
#: STUDY_DRUG is deliberately absent, and that is not a detail: a study-drug
#: finding IS a drug name, so an allowlist keyed on drug morphology would
#: suppress every one of them. That finding is the blinding audit's input,
#: on a different axis from PHI entirely, and silently deleting it would
#: leave the compound name in the corpus with nothing reporting it.
_ALLOWLISTABLE: frozenset[str] = frozenset(
    {"PERSON", "LOCATION", "ORGANIZATION", "FACILITY", "NRP", "GPE"}
)


def _suppressed(entity_type: str, text: str) -> bool:
    """Is this finding a clinical term a detector mistook for an identifier?"""
    if entity_type not in _ALLOWLISTABLE:
        return False
    return _allowlisted(text)


def _allowlisted(text: str) -> bool:
    """True when every alphabetic token is medical vocabulary."""
    toks = [t for t in re.findall(r"[A-Za-z]+", text) if len(t) > 1]
    if not toks:
        return False
    return all(
        t.lower() in CLINICAL_ALLOWLIST or _looks_like_drug_name(t) for t in toks
    )


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
            if not _suppressed(r.entity_type, text[r.start : r.end])
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
# screening policy: what the tool decides, and what it escalates
# ----------------------------------------------------------------------

#: Per-entity-type policy for a detection inside clinical free text.
#:
#: The queue exists for judgment calls, and it stops being useful the moment
#: it also carries the mechanical ones. On a real 120-subject study, 33 flagged
#: rows broke down as 13 study-drug mentions, 4 bare contact details, and 10
#: name/facility hits -- so two thirds of what a steward was asked to rule on
#: had only one defensible answer, and burying the 10 that mattered among them
#: is how a reviewer learns to fill the column by dragging.
#:
#: ``pass``   not PHI. Publishing it is the correct outcome.
#: ``redact`` PHI with no clinical reading. Redacted at the SPAN, so the rest
#:            of the sentence survives -- an AE verbatim keeps its clinical
#:            content and loses the phone number that was never part of it.
#: ``queue``  genuinely ambiguous. A person has to look.
DEFAULT_SCREEN_POLICY: dict[str, str] = {
    # Not PHI at all. A compound name is a BLINDING matter, on a different
    # axis entirely, and the blinding audit reports it separately -- so
    # putting it in a PHI queue asks the wrong question about the right
    # finding.
    "STUDY_DRUG": "pass",
    # Direct identifiers. There is no reading of an AE verbatim in which an
    # email address or a card number is clinical content, so asking a human
    # to confirm that 33 times teaches them to stop reading.
    "EMAIL_ADDRESS": "redact",
    "PHONE_NUMBER": "redact",
    "US_SSN": "redact",
    "CREDIT_CARD": "redact",
    "IBAN_CODE": "redact",
    "IP_ADDRESS": "redact",
    "URL": "redact",
    "MEDICAL_LICENSE": "redact",
    "US_DRIVER_LICENSE": "redact",
    "US_PASSPORT": "redact",
    # Judgment. Is that name the investigator or the subject's daughter? Is
    # the facility the study site or the local hospital that narrows the
    # subject to a town? Is the date in "since March 2019" clinically load-
    # bearing? Only a person knows, and these are what the queue is for.
    "PERSON": "queue",
    "FACILITY": "queue",
    "LOCATION": "queue",
    "ORGANIZATION": "queue",
    "NRP": "queue",
    "AGE": "queue",
    # The same finding under two names. The built-in detector calls it
    # DATE_IN_TEXT and Presidio calls it DATE_TIME, and a policy that knows
    # only one of them is a policy that works on whichever detector it was
    # written against -- which was the weaker one. Every entity name below
    # has to cover both vocabularies or the table is decorative.
    "DATE_IN_TEXT": "queue",
    "DATE_TIME": "queue",
    # Presidio's spellings for identifiers the fallback names differently.
    "US_BANK_NUMBER": "redact",
    "US_ITIN": "redact",
    "UK_NHS": "redact",
    "CRYPTO": "redact",
    "AU_TFN": "redact",
    "AU_MEDICARE": "redact",
    "IN_AADHAAR": "redact",
    "SG_NRIC_FIN": "redact",
}

#: Anything the policy does not name is escalated. A detector version that
#: adds an entity type must not silently acquire an auto-redact.
UNKNOWN_ENTITY_POLICY = "queue"


def resolve_policy(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The default policy with a steward's per-column overrides applied."""
    policy = dict(DEFAULT_SCREEN_POLICY)
    for key, value in (overrides or {}).items():
        action = str(value).strip().lower()
        if action not in {"pass", "redact", "queue"}:
            raise ValueError(
                f"screen_policy {key}={value!r}: expected pass, redact or queue"
            )
        policy[str(key).strip().upper()] = action
    return policy


def _row_decision(
    findings: Sequence[Finding], policy: dict[str, str]
) -> tuple[str, str]:
    """The policy verdict for one row, and why.

    A row is escalated if ANY finding in it needs judgment -- a sentence
    holding both a phone number and a person's name is a judgment call, and
    auto-redacting half of it before a human sees the rest would hide the part
    that mattered.
    """
    actions = {
        policy.get(f.entity_type, UNKNOWN_ENTITY_POLICY) for f in findings
    }
    types = ",".join(sorted({f.entity_type for f in findings}))
    if "queue" in actions:
        return "", ""
    if "redact" in actions:
        return "REDACT", f"policy: redact {types}"
    return "PASS", f"policy: not PHI ({types})"


def redact_all_spans(text: str, findings: Sequence[Finding]) -> str:
    """Replace every detected span, leaving the rest of the sentence intact."""
    out, cursor = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        out.append(text[cursor : f.start])
        out.append(f"<{f.entity_type}>")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


def redact_spans(text: str, findings: Sequence[Finding], policy: dict[str, str]) -> str:
    """Replace only the spans the policy redacts. The sentence survives."""
    out, cursor = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        if policy.get(f.entity_type, UNKNOWN_ENTITY_POLICY) != "redact":
            continue
        out.append(text[cursor : f.start])
        out.append(f"<{f.entity_type}>")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


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
    "verdict_source",
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
    policy: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Screen one free-text column. Returns candidate rows only; data untouched."""
    rows: list[dict[str, object]] = []
    policy = policy if policy is not None else resolve_policy()
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
        verdict, source = _row_decision(findings, policy)
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
                # Pre-filled where the policy has a single defensible answer,
                # blank where a person has to decide. A steward can overrule
                # any of it -- and now only reads the rows that need them.
                "verdict": verdict,
                "verdict_source": source,
                # Always pre-filled with SPAN-level redaction of every
                # finding, whatever the verdict. Without it, a reviewer who
                # types REDACT on "Amoxicillin prescribed 04/12/2026 by GP,
                # see fax 617-555-0142" gets the whole cell replaced by
                # [REDACTED] and loses the drug, the indication and the date
                # -- clinical content that was never the problem. With it,
                # REDACT means "remove the spans", and the reviewer can edit
                # the suggestion when they disagree about one of them.
                "replacement": redact_all_spans(text, findings),
                "reviewer": "",
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def screen(
    frames: dict[str, pd.DataFrame],
    targets: Iterable[tuple[str, str, str | None]],
    *,
    detector: PatternDetector | PresidioDetector | None = None,
    policies: dict[tuple[str, str], dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Screen every ``(domain, column, subject_column)`` target.

    Returns one review queue across all domains, with the rows that need a
    human FIRST. Ordering by confidence alone put a study-drug mention above a
    person's name, which is backwards: what a reviewer should meet at the top
    of the sheet is the decision only they can make.
    """
    det = detector or build_detector()
    parts = [
        screen_column(
            frames[dom],
            col,
            domain=dom,
            subject_column=subj,
            detector=det,
            policy=resolve_policy((policies or {}).get((dom, col))),
        )
        for dom, col, subj in targets
        if dom in frames
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    queue = pd.concat(parts, ignore_index=True)
    queue["_needs_human"] = (queue["verdict"].astype(str).str.strip() == "").astype(int)
    return (
        queue.sort_values(
            ["_needs_human", "max_score", "n_findings"], ascending=False
        )
        .drop(columns="_needs_human")
        .reset_index(drop=True)
    )


def policy_summary(queue: pd.DataFrame) -> dict[str, object]:
    """What the policy decided, and what it left for a person.

    In the manifest because an auto-decision that nobody can see is worse
    than a queue that is too long: the reader has to be able to tell how much
    of the screening a human actually ruled on.
    """
    if queue.empty:
        return {"needs_human": 0, "auto_redact": 0, "auto_pass": 0}
    verdicts = queue["verdict"].astype(str).str.strip().str.upper()
    return {
        "needs_human": int((verdicts == "").sum()),
        "auto_redact": int((verdicts == "REDACT").sum()),
        "auto_pass": int((verdicts == "PASS").sum()),
        "by_source": queue.get(
            "verdict_source", pd.Series(dtype=str)
        ).astype(str).replace("", "needs a human").value_counts().to_dict(),
    }


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
        raw_verdict = row.get("verdict", "")
        # An untouched queue comes back from pandas as NaN, and float("nan") is
        # truthy -- so `raw or ""` kept it, str() made it "NAN", and a queue
        # nobody had opened was reported as 33 rows with an *unrecognised*
        # verdict beside "rows unreviewed: 0". Which reads like it was
        # reviewed. Blank is blank however the reader spells it.
        verdict = (
            "" if raw_verdict is None or pd.isna(raw_verdict)
            else str(raw_verdict).strip().upper()
        )
        if not verdict or verdict in {"PENDING", "NAN", "NONE"}:
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
