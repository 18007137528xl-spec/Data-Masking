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
from . import rawdates
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
    #: Fraction of values that read as a date in *some* written form, not just
    #: ISO. Raw EDC columns arrive as 19/03/2025 or 19MAR2025, so ISO-only
    #: detection reports a date column as free text and screens it instead of
    #: treating it.
    raw_date_rate: float = 0.0
    #: Fraction carrying a month or a day. A column of bare years is far more
    #: likely to be a number than a date.
    raw_date_precise_rate: float = 0.0
    #: Day/month order the column itself proves, or "unknown".
    raw_date_order: str = "unambiguous"
    #: Written forms present, as templates -- never values.
    raw_date_formats: dict[str, int] = field(default_factory=dict)
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

    @property
    def looks_raw_date(self) -> bool:
        """A date column by its values, whatever its name says.

        Raw EDC has no naming convention to lean on -- VISITDT, AE_START,
        dt_onset are all real -- so this is the only reliable signal there.
        """
        return self.raw_date_rate >= 0.80 and self.raw_date_precise_rate >= 0.10

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
            "raw_date_rate": round(self.raw_date_rate, 3),
            "raw_date_precise_rate": round(self.raw_date_precise_rate, 3),
            "raw_date_formats": self.raw_date_formats,
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
    raw_rate, raw_precise = rawdates.date_like_rate(probe)
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
        raw_date_rate=raw_rate,
        raw_date_precise_rate=raw_precise,
        raw_date_order=rawdates.infer_order(probe) if raw_rate else "unambiguous",
        raw_date_formats=rawdates.describe_formats(probe) if raw_rate else {},
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

#: Columns naming the administered product rather than the arm. EXTRT holds
#: the compound; ARM holds the regimen. Different granularity, so they get
#: different label namespaces.
_PRODUCT_COLUMNS = {"EXTRT", "EXTRTV"}

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
    r"INVESTIGATOR|PHYSICIAN|CLINICIAN|COORDINATOR|PRINCIPALINV|"
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

#: Investigator-entered free text under raw EDC's own names. In SDTM these
#: arrive as CMTRT / AETERM / MHTERM and are caught by suffix; on a raw form
#: they are called MEDICATION, DIAGNOSIS, INDICATION. Left unrecognised they
#: fall through to "retain", which publishes every value untouched -- and a
#: medication field is exactly where a physician's name or a hospital ends up.
_RAW_FREETEXT_NAMES = re.compile(
    r"(MEDICATION|MEDICATIONNAME|CONMED|DRUGNAME|INDICATION|DIAGNOSIS|"
    r"CONDITION|PROCEDURE|SYMPTOM|EVENT|ILLNESS|HISTORY|COMPLAINT|FINDING|"
    r"REMARK|FREETEXT|VERBATIM)"
)

#: Treatment and product names as raw EDC spells them.
_RAW_TREATMENT_NAMES = {
    "TREATMENT", "TREATMENTARM", "TRTNAME", "TRTGROUP", "TRTGRP", "RANDARM",
    "RANDOMISEDARM", "RANDOMIZEDARM", "ARMTXT", "ARMDESC", "COHORT",
}
_RAW_PRODUCT_NAMES = {"STUDYDRUG", "STUDYMED", "IMPNAME", "DRUG", "COMPOUND"}

#: Site identifiers as raw EDC spells them.
_RAW_SITE_KEYS = {"SITE", "SITECODE", "SITENO", "CENTRE", "CENTER", "CENTREID"}

#: Type tags raw EDC hangs off the end of a column name: SEX_CD, RACE_TXT,
#: VISIT_NO, DOSE_AMT. They carry no meaning of their own, and leaving them on
#: makes SEX_CD unrecognised -- which for a quasi-identifier is not a cosmetic
#: miss: an unflagged QI is one the k-anonymity measurement never counts, so
#: the risk report comes back reassuring for the wrong reason.
_RAW_TYPE_TAGS = (
    "CD", "TXT", "TEXT", "NO", "NUM", "NAME", "DESC", "AMT", "UNIT", "VAL",
    "YN", "FLAG", "DT", "DTM", "TM", "ID",
)


def _stem(up: str) -> str:
    """Strip one trailing type tag, so SEX_CD is recognised as SEX."""
    for tag in sorted(_RAW_TYPE_TAGS, key=len, reverse=True):
        if up.endswith(tag) and len(up) > len(tag) + 1:
            return up[: -len(tag)]
    return up


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
    keep_dates: bool = False,
    raw_edc: bool = False,
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

    ``raw_edc`` says this drop is the *raw* side of a raw -> SDTM pair. Dates
    are then found by their values rather than a ``--DTC`` suffix, and shifted
    with their written format intact, because the format is what the model is
    being trained to convert.
    """
    col = profile.column
    up = col.upper()
    if raw_edc:
        # Raw EDC spells the same things with separators: AE_START, SITE_ZIP,
        # PATIENT_ID. Every recognizer below is written against SDTM's
        # separator-free names, so match on the squeezed form. SDTM names have
        # no separators, so this is a no-op there.
        up = re.sub(r"[ _\-.]", "", up)
    names = (up, _stem(up)) if raw_edc else (up,)

    def named(*sets: object) -> bool:
        """Is this column one of these names, allowing for a raw type tag?"""
        return any(n in s for n in names for s in sets)  # type: ignore[operator]

    def rule(t: Treatment, **kw: Any) -> FieldRule:
        return FieldRule(column=col, treatment=t, **kw)

    # --- structural / bookkeeping ------------------------------------
    if named({"DOMAIN", "STUDYID"}):
        return Suggestion(rule(Treatment.RETAIN), "high", "structural column")

    # --- identifiers --------------------------------------------------
    if raw_edc and named(_RAW_SUBJECT_KEYS) and "USUBJID" not in sibling_columns:
        # The raw side's subject column. It gets the SAME surrogate as the
        # SDTM side's USUBJID, because the contract's join_key_template makes
        # both look the vault up under one string -- without that, the two
        # published sides carry unrelated identifiers and cannot be assembled
        # into pairs at all.
        return Suggestion(
            rule(
                Treatment.SURROGATE_ID,
                entity="subject",
                prefix="SUBJ",
                note="random surrogate, shared with the SDTM side via the "
                "domain's join_key_template. Raw subject numbers encode site "
                "and enrolment order, so they are identifiers in their own "
                "right, not just keys",
            ),
            "high",
            "raw subject identifier",
        )

    if named(_SUBJECT_KEYS):
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
    if named(_SITE_KEYS) or (raw_edc and named(_RAW_SITE_KEYS)):
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
    if named(_DROP_EXACT) or _DROP_PATTERN.search(up):
        return Suggestion(
            rule(Treatment.DROP), "high", "direct identifier, no analytic value"
        )

    # --- dates, found by their values (raw EDC) -----------------------
    # Ahead of every name-based branch below, because in a raw extract the
    # values are the stronger evidence: VISIT_DT is a date column whose name
    # reduces to VISIT, and a name-first reading of it retains real dates.
    if raw_edc and (profile.looks_raw_date or profile.looks_date):
        # The raw side of a training pair. Same per-subject offset as the SDTM
        # side (same vault, same subject key), but re-emitted in the form it
        # arrived in: the conversion from that form to ISO is the label the
        # model is learning, and normalising it here would delete the task.
        order = profile.raw_date_order
        settled = order in {"dmy", "mdy", "ymd", "unambiguous"}
        # A raw date of birth is the one case where shifting is not enough on
        # its own. The offset is 6-18 months, so the age stays recoverable to
        # about a year -- fine for the great majority, and wrong above 89,
        # where HIPAA treats the age itself as an identifier and the SDTM side
        # of this same pair caps it at 90+. Left alone, the raw tier publishes
        # what the SDTM tier deliberately withheld.
        is_dob = named({"BRTHDTC", "BIRTHDTC", "DOB", "BIRTH", "BIRTHDATE"})
        if is_dob:
            return Suggestion(
                rule(
                    Treatment.DATE_SHIFT_RAW,
                    entity="subject",
                    is_date=True,
                    date_order=(
                        None if order in {"unambiguous", "unknown"} else order
                    ),
                    note="date of birth, shifted with the subject's other "
                    "dates -- which is what keeps age derivable from it, since "
                    "the enrolment date moved by the same amount. REVIEW: the "
                    "shift does NOT cap age, so subjects over 89 are published "
                    "here at an age the SDTM side collapses to '90+'. Either "
                    "null this column for those subjects, or record the "
                    "residual in the determination and keep it.",
                ),
                "low",
                "raw date of birth -- age remains recoverable",
            )
        return Suggestion(
            rule(
                Treatment.DATE_SHIFT_RAW,
                entity="subject",
                is_date=True,
                date_order=None if order in {"unambiguous", "unknown"} else order,
                note=(
                    "shifted by the subject's vault offset, keeping the written "
                    f"format ({', '.join(profile.raw_date_formats) or 'mixed'}). "
                    + (
                        f"day/month order established from the data: {order}"
                        if order in {"dmy", "mdy"}
                        else "no ambiguous all-numeric values, so no order to "
                        "establish"
                        if settled
                        else "AMBIGUOUS: no value in this column has a component "
                        "above 12, so 03/04/2025 could be either date. Set "
                        "date_order: dmy or mdy from the source specification -- "
                        "the run halts until you do, because a wrong order moves "
                        "every date into the wrong month and the output still "
                        "looks like dates"
                    )
                ),
            ),
            "high" if settled else "low",
            "raw date, format-preserving shift"
            if settled
            else "raw date, day/month order UNRESOLVED",
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
    if (
        named(_TREATMENT_EXACT)
        or (raw_edc and named(_RAW_TREATMENT_NAMES, _RAW_PRODUCT_NAMES))
    ) and blind_treatment:
        # Arm columns and product columns describe the treatment at different
        # granularity: ARM is a regimen ("Pembrolizumab 200 mg Q3W"), EXTRT is
        # the compound ("Pembrolizumab"). Sharing one namespace gave EXTRT a
        # label that looked like a third arm. Separate namespaces, separate
        # prefixes, so the labels no longer masquerade as the same scale.
        is_product = named(_PRODUCT_COLUMNS) or (
            raw_edc and named(_RAW_PRODUCT_NAMES)
        )
        return Suggestion(
            rule(
                Treatment.LABEL_MAP,
                entity="treatment_product" if is_product else "treatment",
                prefix="DRUG" if is_product else "TRT",
                keep_values=["Placebo", "PLACEBO", "placebo"],
                note="blinding and commercial confidentiality, NOT privacy -- a "
                "treatment arm identifies nobody. Arm columns share one "
                "namespace so their labels agree; product columns get their "
                "own, because a compound is not an arm. Placebo passes "
                "through: most analyses need to know the control.",
            ),
            "high",
            "product name (blinded)" if is_product else "treatment arm (blinded)",
        )

    # --- study design and visit structure -----------------------------
    if named(_DESIGN_EXACT):
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
    if named({"BRTHDTC", "BIRTHDTC", "DOB", "BIRTH", "BIRTHDATE"}):
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

        if keep_dates:
            # Retained as recorded. HIPAA enumerates dates as identifiers, so
            # this is only available on a Limited Data Set, which 164.514(e)
            # permits to carry full dates -- the contract sets tier: lds to
            # match, because the alternative is a tier that calls itself
            # de-identified while shipping a calendar.
            return Suggestion(
                rule(
                    Treatment.RETAIN,
                    note="retained as recorded. Requires tier: lds -- an LDS may "
                    "keep full dates (164.514(e)) but remains PHI: it needs a "
                    "data use agreement, and a model trained on it inherits "
                    "that scope",
                ),
                "high",
                "date, retained (LDS)",
            )

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
    if named({"AGE"}):
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
    if named({"AGEU", "AGEGR1"}):
        return Suggestion(
            rule(Treatment.RETAIN, is_quasi_identifier=up != "AGEU"),
            "medium",
            "age unit / band",
        )
    if named({"SEX", "GENDER", "RACE", "ETHNIC", "ETHNICITY", "COUNTRY"}):
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
        or (raw_edc and _RAW_FREETEXT_NAMES.search(up))
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


#: Subject-key names seen in raw EDC exports, in the order they are trusted.
#: A raw extract has no USUBJID; without a subject key the domain has no
#: offset to shift by, and every date in it would pass through unshifted.
_RAW_SUBJECT_KEYS = (
    "USUBJID", "SUBJID", "SUBJECT", "SUBJECTID", "SUBJECT_ID", "SUBJNO",
    "PATIENTID", "PATIENT_ID", "PATID", "PT_ID", "SCRNO", "SCREENINGNO",
)


def _raw_subject_key(frame: pd.DataFrame) -> str | None:
    upper = {str(c).upper(): str(c) for c in frame.columns}
    for cand in _RAW_SUBJECT_KEYS:
        if cand in upper:
            return upper[cand]
    return None


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
    keep_dates: bool = False,
    raw_edc: bool = False,
    join_key_template: str | None = None,
    subject_id_template: str | None = None,
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

    if raw_edc and keep_dates:
        raise ValueError(
            "--raw and --keep-dates ask for opposite things. Raw mode shifts "
            "every date so the drop can be de-identified and used for "
            "training; --keep-dates retains real calendar dates, which forces "
            "tier: lds and takes training off the table. Choose one."
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
                keep_dates=keep_dates,
                raw_edc=raw_edc,
            )
            for col, p in profs.items()
        }
        if subject_id_template:
            # Both sides of a pair share one surrogate, so without this the
            # SDTM side's identifier equals the raw side's and the corpus
            # teaches USUBJID = SUBJECT. Rebuilding the real composition over
            # the surrogate keeps the derivation true and still random.
            key = subject_column if subject_column in frame.columns else (
                _raw_subject_key(frame) if raw_edc else None
            )
            for col, sug in per_col.items():
                if (
                    col == key
                    and sug.rule.treatment is Treatment.SURROGATE_ID
                    and sug.rule.entity == "subject"
                ):
                    per_col[col] = Suggestion(
                        sug.rule.model_copy(
                            update={"output_template": subject_id_template}
                        ),
                        sug.confidence,
                        sug.rationale + ", composed via output_template",
                    )

        suggestions[name] = per_col
        domains.append(
            DomainContract(
                name=name,
                subject_key=(
                    subject_column
                    if subject_column in frame.columns
                    else _raw_subject_key(frame)
                    if raw_edc
                    else None
                ),
                join_key_template=join_key_template,
                retained_in_full=name.upper() in RETAINED_IN_FULL,
                fields=[s.rule for s in per_col.values()],
            )
        )

    if keep_dates and tier != "lds":
        # Not a silent override: retaining calendar dates is only lawful on an
        # LDS, and the contract validator enforces that. Setting it here means
        # the tier label matches the data instead of failing later.
        tier = "lds"

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
