"""Record reassignment: each subject's records in a domain go to another subject.

What the de-identified tier still links together after surrogates and date
shifts is a person's *content*: a 67-year-old woman at site 1001, and the
rare adverse event she had. Reassignment cuts that link. Subject A's adverse
events are filed under subject C, A's medical history under F, A's labs under
B -- each domain by its own random mapping, held in the vault.

It is the right trade for one purpose and the wrong one for most: the records
stop being true of anyone, so the tier is no use for analysis. For a corpus
whose job is to test whether a model derives SDTM from raw EDC correctly, that
does not matter -- as long as the derivation itself still holds. So the one
thing this module refuses to break is the relationship between a record and
the reference dates its derived values are measured from:

* where the drop has reference dates (SDTM's DM), dates move by
  ``ref(new owner) - ref(old owner)``, so a record on study day 15 for its old
  owner is on study day 15 for its new one and a carried --DY stays true.
  A raw extract has none, so there the dates stay as written: the model
  derives study days from the new owner's own raw dates
* subject-level columns repeated on the record (SUBJECT, SITE, STUDYID) are
  rewritten to the new owner's, so the record joins to the right DM row and
  the site still matches the subject number
* the mapping and the date delta live in the vault, so whichever side of a
  pair runs second moves exactly as the first did

Domains that *carry* the reference dates -- DM, EX, DS -- are never
reassigned: moving them would move the anchor everything else is measured
from.

This runs before every other treatment, on the original values, so shifting,
surrogates and screening then see an ordinary drop.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from . import rawdates
from .contract import Contract, DomainContract, FieldRule, Treatment
from .transforms import shift_partial
from .vault import Vault


class ReassignError(RuntimeError):
    """Reassignment cannot be done without breaking a derivation."""


_DATE_TREATMENTS = {
    Treatment.DATE_SHIFT,
    Treatment.DATE_SHIFT_RAW,
    Treatment.DATE_TO_STUDY_DAY,
    Treatment.PARTIAL_DATE_TO_YEAR_OFFSET,
}


def _is_date_rule(rule: FieldRule) -> bool:
    if rule.treatment in _DATE_TREATMENTS:
        return True
    return rule.treatment is Treatment.RETAIN and (
        rule.column.upper().endswith("DTC") or rule.is_date
    )


def _keys(frame: pd.DataFrame, dom: DomainContract) -> pd.Series | None:
    """The string each row's subject is known by in the vault.

    The same key the date offset uses: the subject column, or the domain's
    join_key_template -- which is what lets the raw side reach the SDTM
    side's entry.
    """
    tmpl = dom.join_key_template
    if tmpl:
        import re

        needed = set(re.findall(r"{(\w+)}", tmpl))
        if needed - set(map(str, frame.columns)):
            return None
        return frame.apply(
            lambda row: tmpl.format(**{c: str(row[c]) for c in needed}), axis=1
        ).astype("string")
    if dom.subject_key and dom.subject_key in frame.columns:
        return frame[dom.subject_key].astype("string")
    return None


def move_date(value: Any, days: int, order: str | None = None) -> Any:
    """Move one date by ``days`` in the form it was written. Partial dates
    move by the same rule every partial date uses; anything unreadable is
    returned untouched for the date treatment downstream to report."""
    if days == 0 or rawdates._blank(value):
        return value
    p = rawdates.parse(str(value).strip(), order or "unknown")
    if p is None:
        return value
    exact = p.to_date()
    if exact is not None:
        moved = exact + timedelta(days=days)
        return p.render(moved.year, moved.month, moved.day)
    if p.year is not None:
        y, mo = shift_partial(p.year, p.month, days)
        return p.render(y, mo, None)
    return value


def _same(a: Any, b: Any) -> bool:
    na, nb = pd.isna(a), pd.isna(b)
    if na or nb:
        return bool(na and nb)
    return str(a).strip() == str(b).strip()


def apply(
    frames: dict[str, pd.DataFrame], contract: Contract, vault: Vault
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Reassign the records of every domain the contract marks. Returns the
    new frames and a report of counts -- never keys or dates."""
    marked = [d for d in contract.domains if d.reassign and d.name in frames]
    if not marked:
        return frames, {}

    spec = contract.anchor
    if spec.domain not in frames:
        raise ReassignError(
            f"reassignment needs the anchor domain {spec.domain!r} in the drop: "
            "it is where each subject's reference date and site come from"
        )
    anchor_dom = contract.domain(spec.domain)
    if anchor_dom.reassign:
        raise ReassignError(
            f"{spec.domain} is the anchor domain and cannot be reassigned: "
            "every other domain's dates are measured from it"
        )
    anchor = frames[spec.domain].reset_index(drop=True)
    akeys = _keys(anchor, anchor_dom)
    if akeys is None:
        raise ReassignError(
            f"{spec.domain}: no subject key to reassign by -- set its "
            "subject_key or join_key_template"
        )
    row_of = {str(k): i for i, k in enumerate(akeys) if not pd.isna(k)}

    # Real reference dates, where the anchor domain has them.
    refs: dict[str, date | None] = {}
    if spec.date_column in anchor.columns:
        rule = anchor_dom.rule(spec.date_column)
        order = (rule.date_order if rule else None) or rawdates.infer_order(
            anchor[spec.date_column]
        )
        for k, i in row_of.items():
            p = rawdates.parse(anchor.at[i, spec.date_column], order)
            refs[k] = p.to_date() if p else None

    out = dict(frames)
    report: dict[str, Any] = {}
    groups: dict[str, list[DomainContract]] = {}
    for d in marked:
        groups.setdefault(d.reassign, []).append(d)

    for group, doms in groups.items():
        dom_keys = {d.name: _keys(frames[d.name], d) for d in doms}
        for name, ks in dom_keys.items():
            if ks is None:
                raise ReassignError(
                    f"{name}: no subject key to reassign by -- its subject_key "
                    "or join_key_template must name columns it has"
                )
        universe = set(row_of)
        for ks in dom_keys.values():
            universe |= {str(k) for k in ks.dropna()}
        mapping = vault.reassignment_map(group, universe, refs)

        kept = sorted(k for k in universe if mapping[k][0] == k)
        report[group] = {
            "domains": [d.name for d in doms],
            "subjects": len(universe),
            "reassigned": len([k for k in universe if mapping[k][0] != k]),
            "kept_own_records": len(kept),
            "rows": {},
        }

        for d in doms:
            frame = frames[d.name]
            ks = dom_keys[d.name]
            new = frame.copy()
            date_rules = [
                r for r in d.fields if r.column in frame.columns and _is_date_rule(r)
            ]
            date_cols = {r.column for r in date_rules}

            donors = [None if pd.isna(k) else str(k) for k in ks]
            moving = [
                i for i, k in enumerate(donors) if k is not None and mapping[k][0] != k
            ]
            report[group]["rows"][d.name] = len(moving)
            if not moving:
                out[d.name] = new
                continue

            missing = {
                mapping[donors[i]][0]
                for i in moving
                if mapping[donors[i]][0] not in row_of
            } | {donors[i] for i in moving if donors[i] not in row_of}
            if missing:
                raise ReassignError(
                    f"{d.name}: {len(missing)} subject(s) have records here but "
                    f"no row in {spec.domain}, so there is no reference date or "
                    "site to move their records to. Reassignment needs every "
                    "subject in the anchor domain."
                )
            if date_cols:
                undated = sorted({
                    donors[i] for i in moving if mapping[donors[i]][1] is None
                })
                if undated:
                    # No reference dates on this side -- a raw extract has no
                    # RFSTDTC. The records move and their dates stay exactly
                    # as written, which is all a raw corpus needs: the model
                    # derives study days from the new owner's own raw dates.
                    # The zero is stored, so an SDTM side run later against
                    # this vault moves its dates the same way and the pair
                    # still corresponds.
                    vault.settle_unanchored(group, undated)
                    for k in undated:
                        mapping[k] = (mapping[k][0], 0)
                    report[group]["dates"] = "kept as written"

            # Columns that repeat a subject-level value from the anchor
            # domain -- SUBJECT, SITE, STUDYID -- shown by the data rather
            # than assumed from names: every row must equal its own subject's
            # anchor value. DOMAIN ('AE' vs 'DM') fails this and is left alone.
            mirrors: dict[str, str] = {}
            for col in frame.columns:
                if col in date_cols:
                    continue
                candidates = [col] if col in anchor.columns else []
                if col == d.subject_key and spec.subject_column in anchor.columns:
                    candidates.append(spec.subject_column)
                for a in candidates:
                    if all(
                        _same(frame.iat[i, frame.columns.get_loc(col)],
                              anchor.at[row_of[donors[i]], a])
                        for i in moving
                    ):
                        mirrors[col] = a
                        break
            if d.subject_key and d.subject_key not in mirrors:
                raise ReassignError(
                    f"{d.name}.{d.subject_key}: does not match "
                    f"{spec.domain}.{spec.subject_column} row for row, so a "
                    "record cannot be relabelled to its new subject"
                )

            for col, a in mirrors.items():
                j = new.columns.get_loc(col)
                for i in moving:
                    new.iat[i, j] = anchor.at[row_of[mapping[donors[i]][0]], a]

            for r in date_rules:
                order = r.date_order or rawdates.infer_order(frame[r.column])
                j = new.columns.get_loc(r.column)
                for i in moving:
                    new.iat[i, j] = move_date(
                        frame.iat[i, j], mapping[donors[i]][1] or 0, order
                    )

            # The relabelled rows must now be found under their new owner.
            after = _keys(new, d)
            wrong = [
                i for i in moving if str(after.iat[i]) != mapping[donors[i]][0]
            ]
            if wrong:
                raise ReassignError(
                    f"{d.name}: {len(wrong)} relabelled row(s) still resolve to "
                    "their old subject -- the join_key_template uses a column "
                    f"that does not repeat {spec.domain}'s value"
                )
            out[d.name] = new

        if vault.is_unanchored(group):
            report[group]["dates"] = "kept as written"
            if refs:
                # This side HAS reference dates, but the group was first
                # reassigned without them. Consistency with the side that ran
                # first wins: dates stay as written here too. Study days that
                # this drop carries as numbers (--DY) were computed for the
                # old owner, so they must be derived again, not carried over.
                report[group]["warning"] = (
                    "records were reassigned first on a side with no reference "
                    "dates, so dates were kept as written. Any --DY, EPOCH or "
                    "baseline flag carried on these records was computed for "
                    "their old owner: derive it again from the new owner's "
                    "reference date before using this side as an answer key"
                )
        if kept:
            report[group]["note"] = (
                f"{len(kept)} subject(s) new to this group with nobody new to "
                "swap with kept their own records; they move on a later run "
                "only if issued together with others"
            )
    return out, report
