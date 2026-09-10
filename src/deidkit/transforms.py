"""Transform primitives.

Every function here is pure with respect to its inputs plus the vault: given
the same input column, the same contract rule, and the same vault state, the
output is identical. That is what makes an analysis from six months ago
reproducible, and it is the reason surrogate and offset assignment lives in the
vault rather than in a random seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .contract import FieldRule
from .vault import Vault

# ----------------------------------------------------------------------
# ISO 8601 dates, SDTM flavour
# ----------------------------------------------------------------------

Granularity = Literal["day", "month", "year", "none"]

_ISO = re.compile(
    r"^\s*(?P<y>\d{4})"
    r"(?:-(?P<m>\d{2})"
    r"(?:-(?P<d>\d{2})"
    r"(?:[T ].*)?"
    r")?)?\s*$"
)


@dataclass(frozen=True)
class ParsedDate:
    """An SDTM ``--DTC`` value parsed without inventing precision.

    SDTM permits partial dates: ``2015``, ``2015-03``, ``2015-03-14``. Medical
    history in particular is routinely year-only. Imputing a day to make a
    study-day computable manufactures false precision and biases
    duration-of-condition analysis, so granularity is carried explicitly and
    partial values are converted at the granularity they actually have.
    """

    year: int | None
    month: int | None
    day: int | None
    granularity: Granularity

    @property
    def is_complete(self) -> bool:
        return self.granularity == "day"

    def to_date(self) -> date | None:
        if not self.is_complete:
            return None
        assert self.year and self.month and self.day
        try:
            return date(self.year, self.month, self.day)
        except ValueError:
            return None


_EMPTY = ParsedDate(None, None, None, "none")


def parse_dtc(value: object) -> ParsedDate:
    """Parse one ISO 8601 / SDTM ``--DTC`` value."""
    # pd.NA and pd.NaT are neither None nor float, and str() renders them
    # "<NA>" and "NaT" -- which no date parser matches and every date parser
    # would report as a bad value rather than a missing one.
    if value is None:
        return _EMPTY
    try:
        if pd.isna(value):
            return _EMPTY
    except (TypeError, ValueError):  # pragma: no cover - array-like input
        pass
    if isinstance(value, date):
        return ParsedDate(value.year, value.month, value.day, "day")
    text = str(value).strip()
    if not text or text.upper() in {
        "NA", "NAN", "NONE", "UNK", "UNKNOWN", ".", "<NA>", "NAT", "N/A"
    }:
        return _EMPTY
    m = _ISO.match(text)
    if not m:
        return _EMPTY
    y = int(m.group("y"))
    mo = int(m.group("m")) if m.group("m") else None
    d = int(m.group("d")) if m.group("d") else None
    if d is not None:
        return ParsedDate(y, mo, d, "day")
    if mo is not None:
        return ParsedDate(y, mo, None, "month")
    return ParsedDate(y, None, None, "year")


def shift_partial(
    year: int, month: int | None, offset: int
) -> tuple[int, int | None]:
    """Move a partial date by whole days, then truncate back to its own grain.

    A partial date names an interval, not a day, so it cannot be moved by a
    day count directly. It can be moved by shifting a representative point
    inside that interval -- mid-month, mid-year -- and then discarding the
    precision again, which is what this does. Nothing is invented: a
    month-granularity value comes back month-granularity.

    This is the single rule both date paths call, and it has to be, because on
    a raw -> SDTM training pair the same underlying partial date arrives in two
    different formats and must land on the same shifted value. If ``Mar-2025``
    on the raw side and ``2025-03`` on the SDTM side moved by different rules,
    every partial date in the corpus would teach the model a mapping that is
    not true.
    """
    if month is None:
        moved = date(year, 7, 1) + timedelta(days=offset)
        return moved.year, None
    moved = date(year, month, 15) + timedelta(days=offset)
    return moved.year, moved.month


def study_day(event: date, anchor: date, convention: str = "day1") -> int:
    """CDISC study day.

    Under the ``day1`` convention the anchor date is Day 1 and there is no
    Day 0: dates on or after the anchor are ``diff + 1``, dates before it are
    ``diff``. ``day0`` gives the plain difference, which some analysis
    pipelines prefer.
    """
    diff = (event - anchor).days
    if convention == "day0":
        return diff
    return diff + 1 if diff >= 0 else diff


# ----------------------------------------------------------------------
# date treatments
# ----------------------------------------------------------------------


def to_study_day(
    values: pd.Series,
    anchors: pd.Series,
    *,
    convention: str = "day1",
) -> tuple[pd.Series, pd.Series]:
    """Absolute dates -> signed study-day integers.

    Returns ``(study_day, granularity)``. Rows whose date is partial or whose
    subject has no anchor yield ``NA`` rather than a guess; the granularity
    column records which, so downstream analysis can stratify instead of
    quietly dropping them.
    """
    days: list[int | None] = []
    grains: list[str] = []
    for raw, anc in zip(values, anchors):
        p = parse_dtc(raw)
        grains.append(p.granularity)
        ev = p.to_date()
        anc_p = parse_dtc(anc).to_date()
        if ev is None or anc_p is None:
            days.append(None)
        else:
            days.append(study_day(ev, anc_p, convention))
    return (
        pd.Series(days, index=values.index, dtype="Int64"),
        pd.Series(grains, index=values.index, dtype="string"),
    )


def to_year_offset(
    values: pd.Series, anchors: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Partial (or complete) dates -> relative *year* offset + granularity.

    The treatment for medical-history start dates, which are commonly year-only.
    A condition first recorded in 2015 against a 2026 enrolment becomes ``-11``
    with granularity ``year`` -- duration-of-condition survives, no day is
    invented, and the absolute year is gone.
    """
    offs: list[int | None] = []
    grains: list[str] = []
    for raw, anc in zip(values, anchors):
        p = parse_dtc(raw)
        a = parse_dtc(anc)
        grains.append(p.granularity)
        if p.year is None or a.year is None:
            offs.append(None)
        else:
            offs.append(p.year - a.year)
    return (
        pd.Series(offs, index=values.index, dtype="Int64"),
        pd.Series(grains, index=values.index, dtype="string"),
    )


def shift_dates(
    values: pd.Series, subjects: pd.Series, vault: Vault, *, entity: str = "subject"
) -> pd.Series:
    """Retain a calendar date, moved by the subject's vault-held offset.

    Only for genuine seasonality analysis. ``to_study_day`` is preferred: it
    keeps every interval and removes the calendar entirely, whereas a shifted
    date still leaks day-of-week and season.
    """
    offsets = vault.offset_map(
        sorted({str(s) for s in subjects.dropna().unique()}), entity=entity
    )
    out: list[str | None] = []
    for raw, subj in zip(values, subjects):
        p = parse_dtc(raw)
        d = p.to_date()
        off = offsets.get(str(subj))
        if d is None or off is None:
            # A partial date keeps its own granularity: shifted through a
            # representative point and truncated back, so 2015-03 stays
            # month-granularity instead of collapsing to a year.
            if p.year is not None and off is not None:
                y, mo = shift_partial(p.year, p.month, off)
                out.append(f"{y:04d}-{mo:02d}" if mo else f"{y:04d}")
            else:
                out.append(None)
        else:
            out.append((d + timedelta(days=off)).isoformat())
    return pd.Series(out, index=values.index, dtype="string")


def dob_to_age(
    values: pd.Series, anchors: pd.Series, *, cap: int = 90
) -> pd.Series:
    """Date of birth -> age in years at the anchor, capped.

    Ages at or above ``cap`` collapse to a single band. Extreme ages are
    near-unique in any realistic population, which is why HIPAA Safe Harbor
    requires exactly this and why it is worth doing under Expert Determination
    too.
    """
    out: list[str | None] = []
    for raw, anc in zip(values, anchors):
        b = parse_dtc(raw)
        a = parse_dtc(anc)
        if b.year is None or a.year is None:
            out.append(None)
            continue
        age = a.year - b.year
        # Only adjust for not-yet-had-birthday when both dates are precise
        # enough to know.
        bd, ad = b.to_date(), a.to_date()
        if bd and ad and (ad.month, ad.day) < (bd.month, bd.day):
            age -= 1
        elif b.month and a.month and not (bd and ad) and a.month < b.month:
            age -= 1
        out.append(f"{cap}+" if age >= cap else str(age))
    return pd.Series(out, index=values.index, dtype="string")


# ----------------------------------------------------------------------
# identifier treatments
# ----------------------------------------------------------------------


def surrogate(
    values: pd.Series,
    vault: Vault,
    *,
    entity: str,
    prefix: str = "",
    length: int = 8,
) -> pd.Series:
    """Replace identifiers with random, non-derived surrogates from the vault.

    Consistent across domains and across drops, so joins survive. Note what is
    *not* happening: no hash of the original, so the EDC identifier's internal
    structure (``US-001-0042`` -> country, site, enrolment order) is destroyed
    rather than merely obscured.
    """
    mapping = vault.surrogate_map(
        entity,
        (str(v) for v in values.dropna().unique()),
        prefix=prefix,
        length=length,
    )
    return pd.Series(
        [None if pd.isna(v) else mapping[str(v)] for v in values],
        index=values.index,
        dtype="string",
    )


def label_map(
    values: pd.Series,
    vault: Vault,
    *,
    entity: str,
    prefix: str = "TRT",
    keep_values: Sequence[str] = (),
) -> pd.Series:
    """Map distinct values to stable neutral labels: ``TRT A``, ``TRT B``.

    Not a privacy control -- a treatment arm identifies nobody. This is for
    blinding and commercial confidentiality, and it is only worth anything
    applied across every column that names the treatment, which is why the
    mapping lives in one vault namespace rather than per column.

    Assignment order is random, so the labels do not leak the arms' order in
    the protocol, and it is held in the vault, so unblinding is possible and
    reversal is logged like any other re-identification.
    """
    keep = {str(k) for k in keep_values}
    subject_values = sorted(
        {str(v) for v in values.dropna().unique() if str(v) not in keep}
    )
    mapping = vault.label_map(entity, subject_values, prefix=prefix)
    return pd.Series(
        [
            None
            if pd.isna(v)
            else (str(v) if str(v) in keep else mapping[str(v)])
            for v in values
        ],
        index=values.index,
        dtype="string",
    )


def faker_column(values: pd.Series, *, provider: str, seed: int = 0) -> pd.Series:
    """Synthetic replacements for direct identifiers whose column must persist.

    Surrogates rather than nulls: a null breaks downstream pipelines, and the
    absence itself carries information. Distinct originals map to distinct
    fakes, so cardinality is preserved.
    """
    uniques = [v for v in values.dropna().unique()]
    try:  # pragma: no cover - optional dependency
        from faker import Faker

        fk = Faker()
        Faker.seed(seed)
        gen = getattr(fk, provider)
        mapping = {v: gen() for v in uniques}
    except Exception:
        mapping = {
            v: f"{provider.upper()}-{i:06d}" for i, v in enumerate(uniques, start=1)
        }
    return pd.Series(
        [None if pd.isna(v) else mapping[v] for v in values],
        index=values.index,
        dtype="string",
    )


# ----------------------------------------------------------------------
# quasi-identifier generalisation
# ----------------------------------------------------------------------

#: The 3-digit ZIP prefixes whose population fell below 20,000 in the 2000
#: census. HIPAA Safe Harbor requires these be reported as ``000``; retaining
#: them would leave a geography with too few people in it to hide anyone.
RESTRICTED_ZIP3: frozenset[str] = frozenset(
    {
        "036", "059", "063", "102", "203", "556", "692", "790",
        "821", "823", "830", "831", "878", "879", "884", "890", "893",
    }
)


def zip3(values: pd.Series) -> pd.Series:
    """Truncate to a 3-digit ZIP, suppressing low-population prefixes."""
    out: list[str | None] = []
    for v in values:
        if pd.isna(v):
            out.append(None)
            continue
        digits = re.sub(r"\D", "", str(v))
        if len(digits) < 3:
            out.append(None)
            continue
        p = digits[:3]
        out.append("000" if p in RESTRICTED_ZIP3 else p)
    return pd.Series(out, index=values.index, dtype="string")


def cap_numeric(values: pd.Series, *, cap: float) -> pd.Series:
    """Exact values below ``cap``; a single open band at or above it.

    The treatment for age: exact years are analytically useful and, below 90,
    not meaningfully identifying in any realistic cohort. At 90 and above the
    population thins out to the point where the age alone narrows a subject
    down, which is why HIPAA singles it out.
    """
    nums = pd.to_numeric(values, errors="coerce")
    out: list[str | None] = []
    for v in nums:
        if pd.isna(v):
            out.append(None)
        elif v >= cap:
            out.append(f"{cap:g}+")
        else:
            out.append(f"{v:g}")
    return pd.Series(out, index=values.index, dtype="string")


def generalize_numeric(
    values: pd.Series, *, bins: list[float], cap: float | None = None
) -> pd.Series:
    """Band a numeric column, with an optional open-ended top band."""
    nums = pd.to_numeric(values, errors="coerce")
    edges = sorted(set(bins))
    out: list[str | None] = []
    for v in nums:
        if pd.isna(v):
            out.append(None)
            continue
        if cap is not None and v >= cap:
            out.append(f"{cap:g}+")
            continue
        label = None
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi:
                label = f"{lo:g}-{hi - 1:g}" if float(hi).is_integer() else f"{lo:g}-{hi:g}"
                break
        if label is None:
            label = f"<{edges[0]:g}" if v < edges[0] else f"{edges[-1]:g}+"
        out.append(label)
    return pd.Series(out, index=values.index, dtype="string")


def pool_rare(
    values: pd.Series, *, min_count: int, pooled_value: str = "OTHER"
) -> tuple[pd.Series, list[str]]:
    """Categories occurring fewer than ``min_count`` times -> ``pooled_value``.

    Returns the transformed column and the list of pooled categories, so the
    manifest can record exactly what was suppressed. A silent cap reads as
    "we kept everything" when it did not.

    Not applied inside MH or AE: rare preferred terms and orphan indications
    are retained there by an explicit, documented decision.
    """
    counts = values.value_counts(dropna=True)
    rare = sorted(str(k) for k, c in counts.items() if c < min_count)
    rare_set = set(rare)
    out = pd.Series(
        [
            None
            if pd.isna(v)
            else (pooled_value if str(v) in rare_set else str(v))
            for v in values
        ],
        index=values.index,
        dtype="string",
    )
    return out, rare


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------


def describe_rule(rule: FieldRule) -> str:
    """One-line human description, for the manifest and CLI output."""
    t = rule.treatment.value
    bits = [t]
    if rule.entity:
        bits.append(f"entity={rule.entity}")
    if rule.cap is not None:
        bits.append(f"cap={rule.cap}")
    if rule.min_count is not None:
        bits.append(f"min_count={rule.min_count}")
    if rule.bins:
        bits.append(f"bins={len(rule.bins)}")
    if rule.faker_provider:
        bits.append(f"provider={rule.faker_provider}")
    return " ".join(bits)
