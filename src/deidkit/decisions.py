"""The steward decision sheet: the round trip between the draft and the run.

``deidkit profile`` proposes a treatment for every column and writes it here.
A steward opens the sheet, changes what is wrong, records a decision on every
row, and ``deidkit approve`` turns it back into a contract with a signature
bound to those exact rules.

The design rests on one distinction. A contract that runs is not the same
thing as a contract someone agreed with, and before this module existed the
tool could not tell them apart: the profiler drafted a rule for every column,
``run`` executed the draft, the manifest recorded what was done -- and nothing
anywhere recorded whether a person had ever looked. "Every rule needs a
steward's confirmation" was a printed sentence, not a gate.

Three things follow from that, and each is a refusal:

**A blank decision blocks the run.** Not a warning. A blank row means nobody
formed a view about that column, and the whole value of the manifest is that
it says something true -- an approval covering rows nobody read would make it
say something false.

**The signature covers the rules, not the file.** ``Contract.rules_digest()``
digests every rule and every parameter; approval stores it; the contract model
re-checks it on every load. Editing a treatment after approval breaks the
signature and the contract stops loading. Otherwise approval is a date stamp
that a one-line YAML edit walks straight past.

**A steward can overrule the suggestion, not the invariants.** An override
goes through the same ``FieldRule`` and ``Contract`` validators as anything
else, so a change that produces two rules writing one column, or a
de-identified tier that retains dates, is rejected with the reason -- whoever
asked for it.

The sheet carries per-column decisions only. Structure -- which domain is the
anchor, subject keys, the k target -- stays in the draft contract, which is
why ``approve`` reads both. A flat CSV cannot express the structural checks,
and pretending otherwise would mean writing them somewhere with no validation
at all.

Not to be confused with ``review_queue.csv``, the other sheet: that one is
per-*row* free-text adjudication, and it happens *after* a run. This one is
per-*column*, and it happens before.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import Approval, Contract, FieldRule, Treatment
from .profile import Suggestion

#: Columns the tool writes and the steward reads.
PROPOSED = (
    "domain",
    "column",
    "proposed_treatment",
    "proposed_params",
    "confidence",
    "quasi_identifier",
)
#: Columns the steward fills in.
DECISION = ("decision", "decision_treatment", "decision_params", "steward_note")
#: Explanatory columns, ignored on the way back.
CONTEXT = ("why", "detail")

COLUMNS = PROPOSED + DECISION + CONTEXT

#: Rule parameters a steward may set from the sheet. Deliberately not every
#: field on FieldRule: ``column`` and ``treatment`` have their own columns, and
#: ``note`` is the profiler's rationale, which a steward should not overwrite --
#: their own reasoning goes in ``steward_note`` where it is attributable.
SETTABLE = (
    "entity",
    "prefix",
    "faker_provider",
    "output_column",
    "output_template",
    "cap",
    "bins",
    "min_count",
    "pooled_value",
    "keep_values",
    "is_quasi_identifier",
    "is_date",
    "date_order",
    "on_unparsed",
    "redundant_with",
)

_INT_PARAMS = {"cap", "min_count"}
_BOOL_PARAMS = {"is_quasi_identifier", "is_date"}
_LIST_PARAMS = {"bins", "keep_values"}


#: What a steward can write in ``decision_treatment``, and what each one needs.
#:
#: Kept here rather than in a docstring or the README because three places read
#: it: the ``deidkit treatments`` reference, the error raised when an override
#: names something that is not a treatment, and the tests that assert the two
#: agree. A reference that drifts from the validators is worse than none --
#: someone follows it and the run fails anyway.
TREATMENT_HELP: dict[str, dict[str, object]] = {
    "retain": {
        "does": "publish the column unchanged",
        "requires": (),
        "optional": ("is_quasi_identifier", "output_column"),
        "cost": "none",
    },
    "drop": {
        "does": "remove the column entirely",
        "requires": (),
        "optional": ("redundant_with",),
        "cost": "total, unless redundant_with names a surviving column",
    },
    "surrogate_id": {
        "does": "random non-derived surrogate + vault crosswalk",
        "requires": ("entity",),
        "optional": ("prefix", "output_template"),
        "cost": "none; joins and reversibility both survive",
    },
    "faker": {
        "does": "replace with a plausible synthetic value",
        "requires": ("faker_provider",),
        "optional": (),
        "cost": "the real value is gone and NOT recoverable -- unlike "
        "surrogate_id, nothing is written to the vault",
    },
    "date_to_study_day": {
        "does": "date -> signed day relative to the subject's anchor",
        "requires": (),
        "optional": ("output_column",),
        "cost": "none for intervals; the calendar is gone, and so is --DTC, "
        "so the output is no longer conformant SDTM",
    },
    "partial_date_to_year_offset": {
        "does": "partial date -> relative year offset + granularity flag",
        "requires": (),
        "optional": (),
        "cost": "low; never imputes a day",
    },
    "dob_to_age": {
        "does": "date of birth -> age at anchor, capped",
        "requires": (),
        "optional": ("cap",),
        "cost": "low; cap defaults to 90 per Safe Harbor",
    },
    "date_shift": {
        "does": "keep an ISO date, moved by the subject's vault offset",
        "requires": ("entity",),
        "optional": (),
        "cost": "approximate; --DTC stays valid so SDTM still conforms",
    },
    "date_shift_raw": {
        "does": "same shift, re-emitted in the format it arrived in",
        "requires": ("entity",),
        "optional": ("date_order", "on_unparsed", "is_date"),
        "cost": "approximate; for the raw side of a raw -> SDTM pair",
    },
    "cap_numeric": {
        "does": "exact below the cap, one band at or above it",
        "requires": ("cap",),
        "optional": ("is_quasi_identifier",),
        "cost": "low; only the tail loses precision",
    },
    "generalize_numeric": {
        "does": "band a number into intervals",
        "requires": ("bins",),
        "optional": ("cap", "is_quasi_identifier"),
        "cost": "moderate; every value loses precision",
    },
    "zip3": {
        "does": "truncate to 3-digit ZIP, low-population prefixes -> 000",
        "requires": (),
        "optional": ("is_quasi_identifier",),
        "cost": "low",
    },
    "pool_rare": {
        "does": "categories below a threshold -> a pooled value",
        "requires": ("min_count",),
        "optional": ("pooled_value", "is_quasi_identifier"),
        "cost": "minor overall, total for the rare categories -- do not use "
        "in MH/AE, where the rare term is often the finding",
    },
    "label_map": {
        "does": "distinct values -> TRT A / TRT B, stable and reversible",
        "requires": ("entity",),
        "optional": ("prefix", "keep_values"),
        "cost": "none analytically; this is blinding, not privacy",
    },
    "screen_freetext": {
        "does": "detect PHI candidates into a review queue; DATA UNCHANGED",
        "requires": (),
        "optional": (),
        "cost": "none -- and note that it publishes every value as received; "
        "the protection is the adjudication step, not this rule",
    },
    "redact_freetext": {
        "does": "detect and replace in place",
        "requires": (),
        "optional": (),
        "cost": "high on clinical verbatim -- destroys content regulatory "
        "review needs; prefer screen_freetext and adjudicate",
    },
}


def treatment_reference() -> str:
    """The printable answer to "what can I put in decision_treatment?"."""
    lines = [
        "decision_treatment -- what you can write, and what it needs",
        "",
        "  decision = OK      accept the proposal as it stands",
        "  decision = CHANGE  overrule it; name the treatment below",
        "",
    ]
    width = max(len(k) for k in TREATMENT_HELP)
    for name, meta in TREATMENT_HELP.items():
        req = ", ".join(meta["requires"]) or "-"
        lines.append(f"  {name:<{width}}  {meta['does']}")
        lines.append(f"  {'':<{width}}  requires : {req}")
        if meta["optional"]:
            lines.append(
                f"  {'':<{width}}  optional : {', '.join(meta['optional'])}"
            )
        lines.append(f"  {'':<{width}}  cost     : {meta['cost']}")
        lines.append("")
    lines += [
        "decision_params -- key=value, separated by ';'",
        "",
        "  cap=90                     one band at 90 and above",
        "  bins=0,18,40,65,90         band edges for generalize_numeric",
        "  min_count=5                pool_rare threshold",
        "  entity=subject             which surrogate namespace",
        "  keep_values=Placebo        label_map: pass these through",
        "  date_order=dmy             date_shift_raw: 03/04/2025 is 3 April",
        "  on_unparsed=redact         date_shift_raw: null what will not parse",
        "  redundant_with=AESTDY      drop: the column the content survives in",
        "  is_quasi_identifier=true   count this column in the k measurement",
        "",
        "A parameter the treatment does not take is rejected by name, and a",
        "required one that is missing names itself. Neither is guesswork.",
        "",
        "What a steward CANNOT change here: the structural checks. Two rules",
        "writing one output column, a de-identified tier retaining dates, a",
        "retained_in_full domain given a value-destroying treatment -- these",
        "are refused whoever asks, because they are what the tier means.",
    ]
    return "\n".join(lines)


_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


class DecisionError(RuntimeError):
    """The decision sheet cannot be turned into a contract.

    Always raised with the specific rows named. A compliance step that fails
    with "invalid input" teaches people to guess.
    """


def _readable(exc: Exception) -> str:
    """Pull the message out of a pydantic ValidationError.

    The raw form wraps one useful sentence in a location path, a type tag, a
    truncated dump of the input and a documentation URL. A steward reading a
    refusal in a terminal needs the sentence; the rest trains them to skim
    past the part that matters.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    try:
        raw = errors()
    except Exception:  # pragma: no cover
        return str(exc)
    out = []
    for err in raw:
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        loc = ".".join(str(p) for p in err.get("loc", ()) if not isinstance(p, int))
        out.append(f"{loc}: {msg}" if loc and loc not in msg else msg)
    return "\n  ".join(out) or str(exc)


# ----------------------------------------------------------------------
# parameter encoding
# ----------------------------------------------------------------------
def encode_params(rule: FieldRule) -> str:
    """Render a rule's parameters as ``key=value; key=value``.

    Readable in a spreadsheet cell, and re-parseable. Empty and default
    values are omitted so the cell shows what is actually in force rather
    than a wall of blanks.
    """
    bits: list[str] = []
    for key in SETTABLE:
        value = getattr(rule, key, None)
        if value in (None, "", [], False) or (
            key == "pooled_value" and value == "OTHER"
        ):
            continue
        if key == "on_unparsed" and value == "fail":
            continue
        if isinstance(value, list):
            bits.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            bits.append(f"{key}={value}")
    return "; ".join(bits)


def decode_params(text: str, *, where: str) -> dict[str, Any]:
    """Parse ``key=value; key=value`` from a spreadsheet cell."""
    out: dict[str, Any] = {}
    if not text or not str(text).strip():
        return out
    for chunk in re.split(r"[;\n]+", str(text)):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise DecisionError(
                f"{where}: cannot read parameter {chunk!r}. Use "
                f"key=value, separated by ';' -- for example 'cap=90' or "
                f"'date_order=dmy'."
            )
        key, _, raw = chunk.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in SETTABLE:
            raise DecisionError(
                f"{where}: {key!r} is not a parameter that can be set here.\n"
                f"  settable: {', '.join(SETTABLE)}"
            )
        try:
            if key in _INT_PARAMS:
                out[key] = int(raw)
            elif key in _BOOL_PARAMS:
                out[key] = raw.strip().lower() in {"true", "yes", "y", "1"}
            elif key in _LIST_PARAMS:
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                out[key] = [float(p) for p in parts] if key == "bins" else parts
            else:
                out[key] = raw
        except ValueError as exc:
            raise DecisionError(f"{where}: {key}={raw!r} -- {exc}") from exc
    return out


# ----------------------------------------------------------------------
# writing the sheet
# ----------------------------------------------------------------------
def build_sheet(
    contract: Contract, suggestions: dict[str, dict[str, Suggestion]]
) -> pd.DataFrame:
    """One row per column of the drop, lowest confidence first.

    The ordering is the point of the sheet. 55 rules do not deserve equal
    attention: the ones the profiler is unsure about are the ones a steward
    can actually improve, so they are at the top rather than scattered
    alphabetically through three screens of correct guesses.
    """
    rules = {
        (d.name, f.column): f for d in contract.domains for f in d.fields
    }
    rows = []
    for dom, cols in suggestions.items():
        for col, sug in cols.items():
            rule = rules.get((dom, col), sug.rule)
            rows.append(
                {
                    "domain": dom,
                    "column": col,
                    "proposed_treatment": rule.treatment.value,
                    "proposed_params": encode_params(rule),
                    "confidence": sug.confidence,
                    "quasi_identifier": rule.is_quasi_identifier,
                    "decision": "",
                    "decision_treatment": "",
                    "decision_params": "",
                    "steward_note": "",
                    "why": sug.rationale,
                    "detail": rule.note or "",
                }
            )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["_order"] = frame["confidence"].map(_CONFIDENCE_ORDER).fillna(9)
    frame = (
        frame.sort_values(["_order", "domain", "column"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return frame


def carry_forward(sheet: pd.DataFrame, previous: Contract) -> pd.DataFrame:
    """Pre-fill decisions from an already-approved contract.

    On the second drop of a study, most columns are the ones a steward
    already ruled on. Asking again for all of them turns review into a
    formality, so a column whose proposed rule is byte-identical to the
    approved one comes back pre-filled -- and every column that is new, or
    whose rule changed, comes back blank and therefore blocking.

    Only ever called with a contract that carries an approval, because
    carrying a decision forward from a draft would be inventing one.
    """
    if previous.approval is None:
        raise DecisionError(
            "cannot carry decisions forward from an unapproved contract -- "
            "there are no decisions in it, only suggestions"
        )
    approved = {
        (d.name, f.column): (f.treatment.value, encode_params(f))
        for d in previous.domains
        for f in d.fields
    }
    out = sheet.copy()
    provenance = (
        f"carried forward from {previous.approval.approved_by} "
        f"on {previous.approval.approved_at[:10]}"
    )
    for i, row in out.iterrows():
        key = (row["domain"], row["column"])
        if key not in approved:
            continue  # a column nobody has ever ruled on: stays blank
        treatment, params = approved[key]
        if (row["proposed_treatment"], row["proposed_params"]) == (treatment, params):
            out.at[i, "decision"] = "OK"
            out.at[i, "steward_note"] = provenance
            continue
        # The profiler proposes something the steward already overruled once.
        # Carrying the override forward as an override -- rather than blanking
        # the row -- is what stops a standing decision from having to be
        # re-argued on every drop, while the sheet still shows both what was
        # proposed and what was decided.
        out.at[i, "decision"] = "CHANGE"
        out.at[i, "decision_treatment"] = treatment
        out.at[i, "decision_params"] = params
        out.at[i, "steward_note"] = provenance
    return out


# ----------------------------------------------------------------------
# reading the sheet back
# ----------------------------------------------------------------------
def read_sheet(path: str | Path) -> pd.DataFrame:
    """Read a decision sheet as it comes back from a spreadsheet.

    Everything is read as text and nothing is type-inferred. Excel is happy
    to turn a site-subject number like ``002-0002`` into a date and to strip
    the leading zeros off ``002``, and a de-identification plan silently
    rewritten by a spreadsheet is a category of bug worth spending one
    parameter to prevent.
    """
    frame = pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in ("domain", "column", "proposed_treatment", "decision")
               if c not in frame.columns]
    if missing:
        raise DecisionError(
            f"{path}: the sheet is missing required column(s): {missing}.\n"
            "It must be the sheet 'deidkit profile --decisions' wrote, with "
            "the decision columns filled in -- not a re-typed subset."
        )
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    for col in COLUMNS:
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    return frame


def _check_coverage(
    sheet: pd.DataFrame,
    contract: Contract,
    frames: dict[str, pd.DataFrame] | None,
) -> None:
    """The sheet must describe this drop exactly: no extra, no missing, no
    duplicates. Anything else means it is a sheet for something else."""
    pairs = list(zip(sheet["domain"], sheet["column"]))
    dupes = sorted({p for p in pairs if pairs.count(p) > 1})
    if dupes:
        raise DecisionError(
            "the sheet has more than one row for: "
            + ", ".join(f"{d}.{c}" for d, c in dupes)
            + "\nA column cannot have two decisions. Delete the duplicates."
        )

    expected = {(d.name, f.column) for d in contract.domains for f in d.fields}
    got = set(pairs)
    if missing := sorted(expected - got):
        raise DecisionError(
            "the sheet does not cover every column in the contract. Missing:\n"
            + "\n".join(f"  {d}.{c}" for d, c in missing[:20])
            + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
            + "\nA column with no row is a column with no decision."
        )
    if extra := sorted(got - expected):
        raise DecisionError(
            "the sheet has rows for columns that are not in the contract:\n"
            + "\n".join(f"  {d}.{c}" for d, c in extra[:20])
            + "\nEither the sheet is from a different drop, or rows were added "
            "by hand. Re-run 'deidkit profile --decisions' on this drop."
        )

    if frames is None:
        return
    # The strongest check available: does the sheet describe the data in front
    # of us, or a different extract with the same shape of contract?
    for dom, frame in frames.items():
        in_data = {str(c) for c in frame.columns}
        in_sheet = {c for d, c in got if d == dom}
        if drifted := sorted(in_data - in_sheet):
            raise DecisionError(
                f"{dom}: the data has column(s) the sheet says nothing about: "
                f"{drifted}\nThis drop is not the drop the sheet was written "
                "for. Re-profile it."
            )


def apply_sheet(
    sheet: pd.DataFrame,
    contract: Contract,
    *,
    approved_by: str,
    frames: dict[str, pd.DataFrame] | None = None,
    plan_file: str | None = None,
    note: str | None = None,
) -> tuple[Contract, dict[str, int]]:
    """Turn a filled-in sheet plus the draft contract into an approved one.

    Raises ``DecisionError`` listing every unreviewed row rather than
    approving part of a sheet: a partial approval is the thing this exists to
    prevent.
    """
    _check_coverage(sheet, contract, frames)

    blank = sheet[sheet["decision"] == ""]
    if len(blank):
        by_conf = blank["confidence"].value_counts().to_dict()
        listed = "\n".join(
            f"  {r.domain}.{r.column:<18} {r.proposed_treatment:<24} "
            f"({r.confidence}) {r.why}"
            for r in blank.head(25).itertuples()
        )
        raise DecisionError(
            f"{len(blank)} of {len(sheet)} row(s) have no decision "
            f"({by_conf}).\n{listed}"
            + (f"\n  ... and {len(blank) - 25} more" if len(blank) > 25 else "")
            + "\n\nPut 'OK' in the decision column to accept the proposal, or "
            "'CHANGE' with a decision_treatment to overrule it. A blank row is "
            "a column nobody formed a view about, and an approval that covered "
            "it would make the manifest claim something untrue."
        )

    verdicts = {
        (r.domain, r.column): (
            r.decision.upper(),
            r.decision_treatment,
            r.decision_params,
            r.steward_note,
        )
        for r in sheet.itertuples()
    }

    accepted = changed = 0
    new_domains = []
    for dom in contract.domains:
        fields = []
        for rule in dom.fields:
            verdict, treatment, params_text, note_text = verdicts[
                (dom.name, rule.column)
            ]
            where = f"{dom.name}.{rule.column}"

            if verdict in {"OK", "ACCEPT", "Y", "YES"}:
                if treatment or params_text:
                    raise DecisionError(
                        f"{where}: decision is 'OK' but decision_treatment / "
                        f"decision_params is filled in. 'OK' means accept the "
                        f"proposal as it stands -- if you are changing it, say "
                        f"CHANGE so the record shows a change was made."
                    )
                accepted += 1
                fields.append(
                    rule.model_copy(update={"note": _annotate(rule.note, note_text)})
                )
                continue

            if verdict not in {"CHANGE", "C", "OVERRIDE"}:
                raise DecisionError(
                    f"{where}: decision {verdict!r} is not understood.\n"
                    "  OK     -- accept the proposal\n"
                    "  CHANGE -- overrule it; put the treatment in "
                    "decision_treatment"
                )

            if not treatment:
                raise DecisionError(
                    f"{where}: decision is 'CHANGE' but decision_treatment is "
                    f"empty. What should it be instead?\n"
                    f"  treatments: {', '.join(t.value for t in Treatment)}"
                )
            try:
                chosen = Treatment(treatment.strip().lower())
            except ValueError:
                raise DecisionError(
                    f"{where}: {treatment!r} is not a treatment.\n"
                    f"  treatments: {', '.join(t.value for t in Treatment)}"
                ) from None

            # A steward's override starts from a clean rule, not from the
            # proposal's parameters: keeping 'cap=90' after switching away
            # from cap_numeric would silently carry a setting nobody chose.
            params = decode_params(params_text, where=where)
            try:
                fields.append(
                    FieldRule(
                        column=rule.column,
                        treatment=chosen,
                        note=_annotate(
                            f"steward override; proposed was "
                            f"{rule.treatment.value}",
                            note_text,
                        ),
                        **params,
                    )
                )
            except Exception as exc:  # pydantic validation, deliberately wide
                raise DecisionError(
                    f"{where}: {chosen.value} with {params or 'no parameters'} "
                    f"is not a valid rule.\n  {_readable(exc)}"
                ) from exc
            changed += 1

        new_domains.append(dom.model_copy(update={"fields": fields}))

    # Validators run here: the steward's changes face the same structural
    # checks as the draft did -- duplicate output columns, the tier/date
    # coupling, retained_in_full consistency.
    try:
        approved = contract.model_copy(
            update={"domains": new_domains, "approval": None}
        )
        approved = Contract.model_validate(approved.model_dump(mode="json"))
    except Exception as exc:
        raise DecisionError(
            f"the decisions do not make a valid contract.\n  {_readable(exc)}\n"
            "A steward can overrule a suggestion, but not the checks -- these "
            "hold whoever asked for the change."
        ) from exc

    stamped = approved.model_copy(
        update={
            "approval": Approval(
                approved_by=approved_by,
                approved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                rules_fingerprint=approved.rules_digest(),
                decisions_accepted=accepted,
                decisions_changed=changed,
                plan_file=Path(plan_file).name if plan_file else None,
                note=note,
            )
        }
    )
    # Round-trip through the model so the fingerprint check runs on the way in
    # too, rather than trusting the object we just built.
    stamped = Contract.model_validate(stamped.model_dump(mode="json"))
    return stamped, {"accepted": accepted, "changed": changed, "total": len(sheet)}


def _annotate(base: str | None, steward_note: str) -> str | None:
    if not steward_note:
        return base
    return f"{base} [steward: {steward_note}]" if base else f"[steward: {steward_note}]"
