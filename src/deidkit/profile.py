"""Profiling and contract drafting.

You cannot mask what you have not found, and you cannot review what nobody
wrote down. This module profiles an incoming drop and drafts a contract with a
suggested treatment per column.

The draft is a **draft**. Every suggestion is a guess from SDTM naming
convention plus content heuristics, and a data steward has to confirm it before
the contract is committed. What the draft buys is that the steward reviews a
filled-in table rather than authoring one from scratch -- which is the
difference between a contract that gets maintained and one that does not.

SDTM naming is conventional enough that the guesses land most of the time:
``--DTC`` is a date, ``--DECOD`` is a coded term, ``--TERM`` is verbatim text.
Where convention runs out, the profile falls back to the data: a column whose
values are long, mostly unique, and full of spaces is free text whatever it is
called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .contract import (
    AnchorSpec,
    Contract,
    DomainContract,
    FieldRule,
    RiskSpec,
    Treatment,
)
from .transforms import parse_dtc

# ----------------------------------------------------------------------
# column profile
# ----------------------------------------------------------------------


@dataclass
class ColumnProfile:
    column: str
    dtype: str
    n: int
    n_null: int
    n_unique: int
    mean_length: float
    max_length: int
    has_spaces: bool
    date_like_rate: float
    partial_date_rate: float
    samples: list[str] = field(default_factory=list)

    @property
    def null_rate(self) -> float:
        return self.n_null / self.n if self.n else 0.0

    @property
    def unique_rate(self) -> float:
        non_null = self.n - self.n_null
        return self.n_unique / non_null if non_null else 0.0

    @property
    def looks_free_text(self) -> bool:
        return (
            self.has_spaces
            and self.mean_length >= 12
            and self.unique_rate >= 0.30
        )

    @property
    def looks_date(self) -> bool:
        return self.date_like_rate >= 0.80

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "n": self.n,
            "null_rate": round(self.null_rate, 4),
            "n_unique": self.n_unique,
            "unique_rate": round(self.unique_rate, 4),
            "mean_length": round(self.mean_length, 1),
            "date_like_rate": round(self.date_like_rate, 3),
            "partial_date_rate": round(self.partial_date_rate, 3),
            "samples": self.samples,
        }


def profile_column(series: pd.Series, *, n_samples: int = 3) -> ColumnProfile:
    n = len(series)
    non_null = series.dropna()
    text = non_null.astype(str)
    lengths = text.str.len()

    date_hits = 0
    partial_hits = 0
    probe = text.head(500)
    for v in probe:
        p = parse_dtc(v)
        if p.year is not None:
            date_hits += 1
            if p.granularity != "day":
                partial_hits += 1

    return ColumnProfile(
        column=str(series.name),
        dtype=str(series.dtype),
        n=n,
        n_null=int(series.isna().sum()),
        n_unique=int(non_null.nunique()),
        mean_length=float(lengths.mean()) if len(lengths) else 0.0,
        max_length=int(lengths.max()) if len(lengths) else 0,
        has_spaces=bool(text.str.contains(r"\s", regex=True).mean() > 0.30)
        if len(text)
        else False,
        date_like_rate=date_hits / len(probe) if len(probe) else 0.0,
        partial_date_rate=partial_hits / date_hits if date_hits else 0.0,
        samples=[str(v)[:80] for v in non_null.head(n_samples)],
    )


def profile_frame(frame: pd.DataFrame) -> dict[str, ColumnProfile]:
    return {str(c): profile_column(frame[c]) for c in frame.columns}


# ----------------------------------------------------------------------
# treatment suggestion
# ----------------------------------------------------------------------

#: Domains retained in full by decision. Rare coded terms are NOT pooled here.
RETAINED_IN_FULL: frozenset[str] = frozenset({"MH", "AE", "FA", "CE"})

_SUBJECT_KEYS = {"USUBJID", "SUBJID"}
_SITE_KEYS = {"SITEID", "SITEGR1", "SITENUM"}

#: Direct identifiers to remove outright -- no analytic value at any tier.
_DROP_EXACT = {
    "INVNAM", "INVID", "SUBJNAM", "PATNAM", "NAME", "FIRSTNAME", "LASTNAME",
    "SURNAME", "ADDRESS", "ADDR1", "ADDR2", "STREET", "CITY", "PHONE", "TEL",
    "MOBILE", "EMAIL", "FAX", "SSN", "NHSNUM", "MRN", "MEDRECNO", "INSURANCE",
    "POLICYNO", "ACCOUNTNO", "IDNUMBER", "PASSPORT", "NEXTOFKIN", "EMERGCONT",
}

#: Coded / graded clinical content. Retained verbatim.
_RETAIN_SUFFIXES = (
    "DECOD", "BODSYS", "SEV", "SER", "REL", "OUT", "ACN", "CAT",
    "SCAT", "ORRES", "ORRESU", "STRESC", "STRESN", "STRESU", "TESTCD",
    "TEST", "SPEC", "POS", "LOC", "LAT", "DIR", "METHOD", "BLFL", "DRVFL",
    "STAT", "REASND", "TPT", "TPTNUM", "ELTM", "TOXGR", "GRPID", "REFID",
    "DOSE", "DOSU", "DOSFRM", "DOSFRQ", "ROUTE", "ONGO",
    # relative-timing flags: ONGOING / BEFORE / DURING / AFTER
    "ENRF", "STRF", "ENRTPT", "STRTPT", "ENTPT",
    # reference-range and baseline indicators
    "NRIND", "LOBXFL",
    # SDTM's Y/N flag convention: --FL (BLFL, DTHFL, DRVFL, ...)
    "FL",
)

#: Record sequence keys. Retained, but they are not clinical content -- saying
#: so would invite a steward to wave through a column they have not thought
#: about. A wrong-but-plausible rationale is worse than "unrecognised".
_SEQUENCE_SUFFIXES = ("SEQ", "SPID", "REFID", "GRPID")

#: Dose and regimen separate the arms even when the arm is relabelled.
from .blinding import DOSE_COLUMNS as _DOSE_COLUMNS  # noqa: E402

#: Columns naming the treatment. Relabelling ARM while EXTRT still spells out
#: the compound achieves nothing, so these move together.
_TREATMENT_EXACT = {
    "ARM", "ARMCD", "ACTARM", "ACTARMCD", "ACTARMUD", "EXTRT", "TRTA", "TRTP",
}

#: Study design and visit structure. Retained: these are what the analysis is
#: organised around, and they identify a protocol, not a person.
_DESIGN_EXACT = {
    "ARM", "ARMCD", "ACTARM", "ACTARMCD", "ARMNRS", "ACTARMUD",
    "EPOCH", "VISIT", "VISITNUM", "VISITDY", "TAETORD", "AGEU",
}

#: Substring cues for direct identifiers whose column name is decorated
#: (``SUBJPHONE``, ``PATEMAIL``, ``HOMEADDR1``). Exact-name matching alone
#: misses these, and a phone-number column that reaches the free-text screen
#: would publish every value untouched -- the screen does not rewrite data.
_DROP_PATTERN = re.compile(
    r"(PHONE|MOBILE|^TEL$|TELNO|EMAIL|FAX|SSN|SOCSEC|ADDR|STREET|POBOX|"
    r"NEXTOFKIN|EMERGCONT|KINNAM|GUARDIAN|INITIALS|INSURANC|POLICYNO|"
    r"PASSPORT|LICENSE|LICENCE|ACCOUNTNO|IPADDR|DEVICEID|BIOMETRIC)"
)

#: Investigator-entered verbatim fields. --TRT belongs here, not with the
#: coded content: CMTRT is the reported name of a concomitant medication,
#: typed by a human, and after AETERM it is the free-text field most likely to
#: carry an identifier. CMDECOD is the coded counterpart and stays untouched.
_VERBATIM_SUFFIXES = ("TERM", "TRT", "MODIFY", "LLT", "PTCD", "SPID")
_FREETEXT_NAMES = re.compile(
    r"(TERM|COMMENT|CMNT|NARR|NARRATIVE|DESC|DESCRIP|REASON|SPECIFY|"
    r"OTH|OTHER|NOTE|TEXT)$"
)


@dataclass
class Suggestion:
    rule: FieldRule
    confidence: str  # "high" | "medium" | "low"
    rationale: str


def suggest(
    profile: ColumnProfile,
    *,
    domain: str,
    anchor_column: str | None = None,
    sibling_columns: frozenset[str] = frozenset(),
    sdtm_conformant: bool = False,
    blind_treatment: bool = False,
) -> Suggestion:
    """Suggest a treatment for one profiled column.

    ``sibling_columns`` (upper-cased) is the rest of the domain, which some
    decisions genuinely depend on: SDTM ``DM`` carries both ``BRTHDTC`` and
    ``AGE``, and deriving age from the date of birth when age is already
    present would both collide and pointlessly retain a direct identifier.

    ``sdtm_conformant`` changes how dates are handled, and it is the more
    consequential switch:

    * **False** (analysis output): ``--DTC`` becomes a ``--DY`` study day and
      the date column goes away. Smallest attack surface, but the result is
      not SDTM -- ``--DTC`` is a required variable and Pinnacle 21 will say so.
    * **True** (SDTM output): ``--DTC`` keeps its ISO form and is shifted by a
      per-subject offset held in the vault. The dataset stays conformant, and
      any existing ``--DY`` stays correct for free, because it is measured
      from ``RFSTDTC`` -- which shifts by the same amount.
    """
    col = profile.column
    up = col.upper()

    def rule(t: Treatment, **kw: Any) -> FieldRule:
        return FieldRule(column=col, treatment=t, **kw)

    # --- structural / bookkeeping ------------------------------------
    if up in {"DOMAIN", "STUDYID"}:
        return Suggestion(rule(Treatment.RETAIN), "high", "structural column")

    # --- identifiers --------------------------------------------------
    if up in _SUBJECT_KEYS:
        # USUBJID is the join key across domains. SUBJID alongside it is
        # redundant AND encodes site plus enrolment order -- and surrogating
        # both would issue two unrelated surrogates for the same person,
        # because their original values differ.
        if up == "SUBJID" and "USUBJID" in sibling_columns:
            return Suggestion(
                rule(
                    Treatment.DROP,
                    redundant_with="USUBJID",
                    note="redundant with USUBJID, and encodes site and "
                    "enrolment order",
                ),
                "high",
                "redundant subject identifier",
            )
        return Suggestion(
            rule(
                Treatment.SURROGATE_ID,
                entity="subject",
                prefix="SUBJ",
                note="random surrogate, not derived from the original: EDC "
                "subject IDs encode country, site and enrolment order",
            ),
            "high",
            "subject identifier",
        )
    if up in _SITE_KEYS:
        return Suggestion(
            rule(
                Treatment.SURROGATE_ID,
                entity="site",
                prefix="SITE",
                is_quasi_identifier=True,
                note="a site with very few subjects is itself identifying; "
                "consider pooling low-enrolment sites into a region",
            ),
            "high",
            "site identifier",
        )
    if up in _DROP_EXACT or _DROP_PATTERN.search(up):
        return Suggestion(
            rule(Treatment.DROP), "high", "direct identifier, no analytic value"
        )

    # --- dose and regimen give the arm away ------------------------------
    if blind_treatment and up in _DOSE_COLUMNS:
        return Suggestion(
            rule(
                Treatment.RETAIN,
                note="REVIEW: unblinding risk. EXDOSE 200 against EXDOSE 400 "
                "separates the arms perfectly, whatever ARM says. Retained by "
                "default because pooling dose destroys exposure-response "
                "analysis -- a steward has to decide which matters more here.",
            ),
            "low",
            "dose/regimen: retained, but it unblinds",
        )

    # --- treatment naming ----------------------------------------------
    if up in _TREATMENT_EXACT and blind_treatment:
        return Suggestion(
            rule(
                Treatment.LABEL_MAP,
                entity="treatment",
                prefix="TRT",
                keep_values=["Placebo", "PLACEBO", "placebo"],
                note="blinding and commercial confidentiality, NOT privacy -- a "
                "treatment arm identifies nobody. One namespace across every "
                "column that names the treatment, or the labels disagree. "
                "Placebo passes through: most analyses need to know the control.",
            ),
            "high",
            "treatment name (blinded)",
        )

    # --- study design and visit structure -----------------------------
    if up in _DESIGN_EXACT:
        return Suggestion(
            rule(Treatment.RETAIN),
            "high",
            "study design / visit structure",
        )

    # --- already-derived study days -----------------------------------
    # Real SDTM exports usually carry --DY alongside --DTC. A study day is an
    # interval, not a date element, so it is already in the form this pipeline
    # would have produced.
    if up.endswith("DY") and not up.endswith("BODY") and len(up) > 2:
        note = "already a study day: an interval, not a date element"
        if sdtm_conformant:
            note += (
                "; stays correct under date shifting, because the reference "
                "date it is measured from shifts by the same offset"
            )
        return Suggestion(rule(Treatment.RETAIN, note=note), "high",
                          "derived study day")

    # --- dates --------------------------------------------------------
    if up in {"BRTHDTC", "BIRTHDTC", "DOB"}:
        if "AGE" in sibling_columns:
            return Suggestion(
                rule(
                    Treatment.DROP,
                    redundant_with="AGE",
                    note="AGE is already present in this domain, so the date of "
                    "birth adds nothing analytically and is a direct identifier",
                ),
                "high",
                "date of birth, redundant with AGE",
            )
        return Suggestion(
            rule(
                Treatment.DOB_TO_AGE,
                cap=90,
                is_quasi_identifier=True,
                note="ages at or above 90 collapse to one band: extreme ages "
                "are near-unique",
            ),
            "high",
            "date of birth",
        )
    if up.endswith("DTC") or profile.looks_date:
        is_anchor = bool(anchor_column and up == anchor_column.upper())

        if sdtm_conformant:
            # Every date shifts, the anchor included. Shifting the anchor is
            # what keeps --DY valid: study day is measured from the reference
            # start, so if both move by the same offset the difference holds.
            # Converting the anchor to "1" instead would leave a required SDTM
            # variable holding something that is not a date.
            note = (
                "shifted by a per-subject offset: --DTC stays a valid ISO date "
                "so the dataset remains SDTM-conformant, and every interval "
                "survives because one subject's dates all move together"
            )
            if is_anchor:
                note = (
                    "the reference start, shifted with the rest of the subject's "
                    "dates -- which is precisely why any --DY stays correct"
                )
            return Suggestion(
                rule(Treatment.DATE_SHIFT, entity="subject", note=note),
                "high",
                "anchor date (SDTM-conformant shift)" if is_anchor
                else "event date (SDTM-conformant shift)",
            )

        if is_anchor:
            return Suggestion(
                rule(
                    Treatment.DATE_TO_STUDY_DAY,
                    note="the anchor itself; becomes Day 1 by construction",
                ),
                "high",
                "anchor date",
            )
        # Data-driven: a column that is frequently partial (medical history
        # start dates especially) must not be forced to a study day, because
        # that requires imputing a day that was never recorded.
        if profile.partial_date_rate >= 0.20:
            return Suggestion(
                rule(
                    Treatment.PARTIAL_DATE_TO_YEAR_OFFSET,
                    note=f"{profile.partial_date_rate:.0%} of values are partial "
                    "dates; converted at the granularity actually held, no day "
                    "imputed",
                ),
                "high",
                "frequently-partial date",
            )
        derived = up[:-3] + "DY" if up.endswith("DTC") else None
        if derived and derived in sibling_columns:
            return Suggestion(
                rule(
                    Treatment.DROP,
                    redundant_with=derived,
                    note=f"{derived} is already present and carries this date's "
                    "analytic content as an interval. NOTE: this makes the "
                    "dataset non-conformant, since --DTC is required in SDTM. "
                    "Use sdtm_conformant=True if the output must validate.",
                ),
                "high",
                "event date, already derived to a study day",
            )
        return Suggestion(
            rule(
                Treatment.DATE_TO_STUDY_DAY,
                note="reparameterisation, not redaction: intervals preserved. "
                "NOTE: removes a required SDTM variable; use sdtm_conformant=True "
                "if the output must validate.",
            ),
            "high",
            "event date",
        )

    # --- quasi-identifiers -------------------------------------------
    if up in {"AGE"}:
        return Suggestion(
            rule(
                Treatment.CAP_NUMERIC,
                cap=90,
                is_quasi_identifier=True,
                note="exact years below 90, a single 90+ band above it: extreme "
                "ages are near-unique, ordinary ones are not",
            ),
            "high",
            "age",
        )
    if up in {"AGEU", "AGEGR1"}:
        return Suggestion(
            rule(Treatment.RETAIN, is_quasi_identifier=up != "AGEU"),
            "medium",
            "age unit / band",
        )
    if up in {"SEX", "GENDER", "RACE", "ETHNIC", "ETHNICITY", "COUNTRY"}:
        return Suggestion(
            rule(Treatment.RETAIN, is_quasi_identifier=True),
            "high",
            "demographic quasi-identifier",
        )
    if re.search(r"(ZIP|POSTCODE|POSTAL)", up):
        return Suggestion(
            rule(Treatment.ZIP3, is_quasi_identifier=True),
            "high",
            "postal geography",
        )

    # --- record keys ---------------------------------------------------
    if any(up.endswith(s) for s in _SEQUENCE_SUFFIXES):
        return Suggestion(
            rule(Treatment.RETAIN),
            "high",
            "record sequence key, not clinical content",
        )

    # --- coded clinical content --------------------------------------
    if any(up.endswith(s) for s in _RETAIN_SUFFIXES):
        return Suggestion(
            rule(Treatment.RETAIN),
            "high",
            "coded or graded clinical content -- the analytic payload",
        )

    # --- verbatim / free text ----------------------------------------
    if (
        any(up.endswith(s) for s in _VERBATIM_SUFFIXES)
        or _FREETEXT_NAMES.search(up)
        or profile.looks_free_text
    ):
        note = (
            "screened, not rewritten: candidates go to a review queue and "
            "unflagged rows publish as received"
        )
        if domain.upper() in RETAINED_IN_FULL:
            note += "; required for regulatory review in this domain"
        return Suggestion(
            rule(Treatment.SCREEN_FREETEXT, note=note),
            "high" if any(up.endswith(s) for s in _VERBATIM_SUFFIXES) else "medium",
            "verbatim free text",
        )

    # --- fallback -----------------------------------------------------
    if profile.unique_rate > 0.95 and profile.n_unique > 20:
        return Suggestion(
            rule(
                Treatment.DROP,
                note="REVIEW: near-unique values and no recognised meaning; "
                "drop is the fail-safe guess, not a considered decision",
            ),
            "low",
            "unrecognised, near-unique",
        )
    return Suggestion(
        rule(Treatment.RETAIN, note="REVIEW: unrecognised column"),
        "low",
        "unrecognised",
    )


# ----------------------------------------------------------------------
# contract drafting
# ----------------------------------------------------------------------


def draft_contract(
    frames: dict[str, pd.DataFrame],
    *,
    source: str,
    tier: str = "deidentified",
    contract_version: str = "0.1.0-draft",
    anchor_domain: str | None = None,
    anchor_date_column: str | None = None,
    subject_column: str = "USUBJID",
    k_target: int = 5,
    sdtm_conformant: bool = False,
    blind_treatment: bool = False,
) -> tuple[Contract, dict[str, dict[str, Suggestion]]]:
    """Draft a contract from profiled data.

    Returns the contract and the per-column suggestions, so the CLI can print
    confidence and rationale alongside for the steward to review.
    """
    anchor_domain = anchor_domain or ("DM" if "DM" in frames else next(iter(frames)))
    if anchor_date_column is None:
        candidates = ["RFXSTDTC", "RFSTDTC", "TRTSDTC", "RANDDTC"]
        cols = set(frames[anchor_domain].columns)
        anchor_date_column = next(
            (c for c in candidates if c in cols),
            next(
                (c for c in frames[anchor_domain].columns if str(c).endswith("DTC")),
                "RFSTDTC",
            ),
        )

    suggestions: dict[str, dict[str, Suggestion]] = {}
    domains: list[DomainContract] = []

    for name, frame in frames.items():
        profs = profile_frame(frame)
        siblings = frozenset(str(c).upper() for c in frame.columns)
        per_col = {
            col: suggest(
                p,
                domain=name,
                anchor_column=anchor_date_column if name == anchor_domain else None,
                sibling_columns=siblings - {col.upper()},
                sdtm_conformant=sdtm_conformant,
                blind_treatment=blind_treatment,
            )
            for col, p in profs.items()
        }
        suggestions[name] = per_col
        domains.append(
            DomainContract(
                name=name,
                subject_key=subject_column if subject_column in frame.columns else None,
                retained_in_full=name.upper() in RETAINED_IN_FULL,
                fields=[s.rule for s in per_col.values()],
            )
        )

    risk_domain = anchor_domain if anchor_domain in frames else next(iter(frames))
    contract = Contract(
        contract_version=contract_version,
        source=source,
        tier=tier,  # type: ignore[arg-type]
        anchor=AnchorSpec(
            domain=anchor_domain,
            subject_column=subject_column,
            date_column=anchor_date_column,
        ),
        risk=RiskSpec(domain=risk_domain, k_target=k_target),
        domains=domains,
    )
    return contract, suggestions


def suggestion_table(suggestions: dict[str, dict[str, Suggestion]]) -> pd.DataFrame:
    """Flatten suggestions for review as a spreadsheet."""
    rows = [
        {
            "domain": dom,
            "column": col,
            "treatment": s.rule.treatment.value,
            "confidence": s.confidence,
            "quasi_identifier": s.rule.is_quasi_identifier,
            "rationale": s.rationale,
            "note": s.rule.note or "",
            "steward_verdict": "",
        }
        for dom, cols in suggestions.items()
        for col, s in cols.items()
    ]
    order = {"low": 0, "medium": 1, "high": 2}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["confidence", "domain", "column"],
        key=lambda s: s.map(order) if s.name == "confidence" else s,
    ).reset_index(drop=True)
