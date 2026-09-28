"""Format-preserving date shifting, for the raw side of a training pair.

Training a model to derive SDTM from raw EDC data means the training example
is a *pair*: a raw record and the SDTM record it should produce. The date is
part of what has to be learned -- the format conversion (``19/03/2025`` ->
``2025-03-19``), the ISO 8601 conventions, how a partial date is carried
through. Strip the dates and there is nothing left to learn.

Which is why shifting, not removing, is the right treatment here. Shift both
sides of the pair by the same per-subject offset and:

* the mapping relationship survives exactly -- raw day X still corresponds to
  SDTM day X
* the format-conversion signal survives, because each side keeps its own
  format
* no real calendar date survives on either side

Two things this module exists to get right, both of which the ISO-only shift
in ``transforms`` gets wrong on raw data:

**Format preservation.** A raw value shifted and re-emitted as ISO would teach
the model that the source is already ISO, which is the one thing it is
supposed to learn to fix. So the format is detected, the days are shifted, and
the value is re-emitted in the format it arrived in.

**Day/month ambiguity.** ``03/04/2025`` is either 3 April or 4 March, and
guessing wrong shifts a date into the wrong month while still looking
plausible -- silently corrupting the labels. So the order is inferred only
where the column itself proves it (some value with a component above 12), and
otherwise has to be declared.

The offsets come from the same vault as everything else, keyed by subject, so
processing the raw directory and the SDTM directory against one vault gives
both sides the same offset with no extra coordination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from .transforms import shift_partial
from .vault import Vault

#: ``unambiguous`` means the column contains no value whose day/month order
#: could be read two ways, so there was nothing to infer. It is reported
#: distinctly from ``ymd`` because claiming an order the data never showed is
#: how a wrong order gets written into a manifest and believed later.
Order = Literal["dmy", "mdy", "ymd", "unambiguous", "unknown"]

_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_MONTH_ABBR = {v: k for k, v in _MONTHS.items()}


@dataclass(frozen=True)
class RawShift:
    """The result of a format-preserving shift.

    ``passed_through`` is the row mask of values that were emitted unchanged.
    It is returned as a mask rather than inferred by comparing input to output,
    because a shifted value can legitimately equal its input -- a year-only
    date moved by four months usually lands in the same year -- and a caller
    that nulls "unchanged" rows would delete correctly-shifted data.
    """

    values: pd.Series
    report: dict[str, object]
    passed_through: pd.Series

    def __iter__(self):
        # Kept unpackable as (values, report): the mask is the addition.
        return iter((self.values, self.report))


@dataclass(frozen=True)
class RawDate:
    """A parsed raw date, remembering how it was written."""

    year: int | None
    month: int | None
    day: int | None
    #: A strftime-like template with {y} {m} {d} {mon} {MON} placeholders.
    template: str
    granularity: Literal["day", "month", "year", "none"]

    def to_date(self) -> date | None:
        if self.granularity != "day" or not (self.year and self.month and self.day):
            return None
        try:
            return date(self.year, self.month, self.day)
        except ValueError:
            return None

    def render(self, y: int, m: int | None, d: int | None) -> str:
        return (
            self.template
            .replace("{y}", f"{y:04d}")
            .replace("{yy}", f"{y % 100:02d}")
            .replace("{m}", f"{m:02d}" if m else "")
            .replace("{d}", f"{d:02d}" if d else "")
            .replace("{mon}", _MONTH_ABBR.get(m or 0, "").capitalize())
            .replace("{MON}", _MONTH_ABBR.get(m or 0, "").upper())
        )


# Ordered: the first pattern that matches wins, so put the unambiguous
# month-name and ISO forms ahead of the all-numeric ones.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 2025-03-19  2025-03  2025
    (re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})$"), "{y}-{m}-{d}"),
    (re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})$"), "{y}-{m}"),
    # A bare four-digit number is only read as a year inside a plausible range.
    # Without that guard a column of four-digit lab values or subject numbers
    # sniffs as a date column, and gets "shifted".
    (re.compile(r"^(?P<y>(?:19|20)\d{2})$"), "{y}"),
    # 2025/03/19
    (re.compile(r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})$"), "{y}/{m}/{d}"),
    # 19-Mar-2025   19MAR2025   19 Mar 2025
    (re.compile(r"^(?P<d>\d{1,2})-(?P<mon>[A-Za-z]{3})-(?P<y>\d{4})$"), "{d}-{mon}-{y}"),
    (re.compile(r"^(?P<d>\d{2})(?P<mon>[A-Za-z]{3})(?P<y>\d{4})$"), "{d}{MON}{y}"),
    (re.compile(r"^(?P<d>\d{1,2}) (?P<mon>[A-Za-z]{3}) (?P<y>\d{4})$"), "{d} {mon} {y}"),
    # Mar-2025
    (re.compile(r"^(?P<mon>[A-Za-z]{3})-(?P<y>\d{4})$"), "{mon}-{y}"),
    # ambiguous all-numeric: 19/03/2025  03/19/2025  19.03.2025
    (re.compile(r"^(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})$"), "{a}/{b}/{y}"),
    (re.compile(r"^(?P<a>\d{1,2})-(?P<b>\d{1,2})-(?P<y>\d{4})$"), "{a}-{b}-{y}"),
    (re.compile(r"^(?P<a>\d{1,2})\.(?P<b>\d{1,2})\.(?P<y>\d{4})$"), "{a}.{b}.{y}"),
)


#: An all-numeric d/m/y form. Matching this while ``parse`` returned nothing
#: means the value is a date whose order was never established, which is a
#: different problem from a format nobody recognised.
_AMBIGUOUS_NUMERIC = re.compile(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}$")


def _blank(v: object) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    return not str(v).strip() or str(v).strip().upper() in {
        "NA", "NAN", "NONE", "UNK", "UNKNOWN", ".", "", "NULL"
    }


def infer_order(values: pd.Series) -> Order:
    """Work out day/month order from the column, or admit it cannot.

    A component above 12 can only be a day, which settles it. If nothing in
    the column ever exceeds 12 in either position, the column is genuinely
    ambiguous and this returns ``unknown`` rather than assuming a locale --
    a wrong guess shifts dates into the wrong month while still looking like
    a date.
    """
    a_over, b_over = False, False
    seen = False
    for v in values.dropna().astype(str):
        for pat, tmpl in _PATTERNS:
            m = pat.match(v.strip())
            if not m:
                continue
            gd = m.groupdict()
            if "a" not in gd:
                break  # unambiguous form; tells us nothing about ordering
            seen = True
            a, b = int(gd["a"]), int(gd["b"])
            if a > 12:
                a_over = True
            if b > 12:
                b_over = True
            break
    if not seen:
        return "unambiguous"  # nothing in the column can be read two ways
    if a_over and not b_over:
        return "dmy"
    if b_over and not a_over:
        return "mdy"
    return "unknown"


def parse(value: object, order: Order = "unknown") -> RawDate | None:
    """Parse one raw date, keeping its written form."""
    if _blank(value):
        return None
    text = str(value).strip()
    for pat, tmpl in _PATTERNS:
        m = pat.match(text)
        if not m:
            continue
        gd = m.groupdict()
        y = int(gd["y"]) if gd.get("y") else None

        if "a" in gd:
            a, b = int(gd["a"]), int(gd["b"])
            if order == "dmy":
                d, mo = a, b
            elif order == "mdy":
                mo, d = a, b
            else:
                return None  # ambiguous and undeclared: refuse to guess
            # Keep the written order in the template.
            tmpl = tmpl.replace("{a}", "{d}" if order == "dmy" else "{m}")
            tmpl = tmpl.replace("{b}", "{m}" if order == "dmy" else "{d}")
        else:
            mo = (
                _MONTHS.get(gd["mon"].lower()) if gd.get("mon")
                else (int(gd["m"]) if gd.get("m") else None)
            )
            d = int(gd["d"]) if gd.get("d") else None

        gran = "day" if d else ("month" if mo else ("year" if y else "none"))
        return RawDate(y, mo, d, tmpl, gran)
    return None


def shift_preserving_format(
    values: pd.Series,
    subjects: pd.Series,
    vault: Vault,
    *,
    entity: str = "subject",
    order: Order | None = None,
) -> RawShift:
    """Shift raw dates by the subject's vault offset, keeping each format.

    Returns the shifted column and a report. Every value that was *not*
    shifted is counted and sampled, because this treatment passes such values
    through unchanged: a raw EDC extract holds formats no parser anticipates,
    and on the raw side of a training pair an intact value the tool does not
    understand is more useful than a deleted one.

    That passthrough is also the one way this treatment can leak a real date,
    so the counts are not diagnostics -- they are the finding. The caller
    decides what to do about them; ``pipeline`` refuses the run by default.
    """
    resolved: Order = order or infer_order(values)
    keys = sorted({str(s) for s in subjects.dropna().unique()})
    reused = vault.known_offset_keys(keys, entity=entity)
    offsets = vault.offset_map(keys, entity=entity)

    out: list[str | None] = []
    through: list[bool] = []
    unparsed = 0
    ambiguous = 0
    shifted = 0
    partial = 0
    no_offset = 0
    kept: list[str] = []

    def keep(text: str) -> None:
        if len(kept) < 5 and text not in kept:
            kept.append(text)

    for raw, subj in zip(values, subjects):
        if _blank(raw):
            out.append(None)
            through.append(False)
            continue
        text = str(raw).strip()
        off = offsets.get(str(subj))
        p = parse(text, resolved)

        if p is None:
            if _AMBIGUOUS_NUMERIC.match(text):
                ambiguous += 1
            else:
                unparsed += 1
            keep(text)
            out.append(text)
            through.append(True)
            continue
        if off is None:
            # No subject, so no offset. Nulling would be the safe move for a
            # published tier, but this column is a training label: a silent
            # null here is a corrupted pair. Report it and let the caller stop.
            no_offset += 1
            keep(text)
            out.append(text)
            through.append(True)
            continue

        exact = p.to_date()
        if exact is not None:
            moved = exact + timedelta(days=off)
            out.append(p.render(moved.year, moved.month, moved.day))
            through.append(False)
            shifted += 1
        elif p.year is not None:
            # Shifted through a representative point and truncated back to the
            # granularity actually recorded -- the same rule the SDTM side
            # applies, so the two sides of the pair still agree.
            y, mo = shift_partial(p.year, p.month, off)
            out.append(p.render(y, mo, None))
            through.append(False)
            partial += 1
        else:
            unparsed += 1
            keep(text)
            out.append(text)
            through.append(True)

    report: dict[str, object] = {
        "order": resolved,
        "shifted": shifted,
        "partial_shifted": partial,
        "unparsed": unparsed,
        "ambiguous_refused": ambiguous,
        "no_offset": no_offset,
        "passed_through": unparsed + ambiguous + no_offset,
        "passed_through_samples": kept,
        "formats": describe_formats(values),
        # Offsets already in the vault mean these subjects were shifted by an
        # earlier run -- the other side of the pair. All-new means they were
        # not, and the two sides moved by unrelated amounts.
        "offset_keys": len(keys),
        "offset_keys_reused": len(reused),
    }
    return RawShift(
        pd.Series(out, index=values.index, dtype="string"),
        report,
        pd.Series(through, index=values.index, dtype=bool),
    )


def describe_formats(values: pd.Series, limit: int = 8) -> dict[str, int]:
    """Which written forms appear, and how often. For a steward's review.

    Blanks are excluded: an empty cell is a missing date, not an unrecognised
    format, and counting the two together would report a column of clean dates
    with a high null rate as a parsing problem.
    """
    counts: dict[str, int] = {}
    for v in values.dropna().astype(str):
        if _blank(v):
            continue
        p = parse(v.strip(), "dmy")  # order irrelevant to the template
        key = p.template if p else "UNPARSED"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:limit])


def date_like_rate(values: pd.Series, *, probe: int = 500) -> tuple[float, float]:
    """How much of this column reads as a date, and how much of it is precise.

    Used to find date columns in raw EDC, where the column *name* carries no
    convention to go on -- ``VISITDT``, ``AE_START``, ``dt_onset`` are all in
    the wild.

    Two numbers rather than one, because either alone gets a real column
    wrong. Medical-history onset is year-only in roughly half its rows, so
    counting only precise values reports a genuine date column as free text
    and publishes it untouched. But a bare four-digit number *is* a valid
    year, so counting them equally reports a column of lab values or visit
    numbers as dates. A date column is one where nearly everything parses AND
    some of it carries a month or a day.
    """
    seen = 0
    hits = 0
    precise = 0
    for v in values.dropna().astype(str).head(probe):
        if _blank(v):
            continue
        seen += 1
        p = parse(v.strip(), "dmy")
        if p is None:
            continue
        hits += 1
        if p.granularity in {"day", "month"}:
            precise += 1
    if not seen:
        return 0.0, 0.0
    return hits / seen, precise / seen
