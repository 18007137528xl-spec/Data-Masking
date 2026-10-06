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

import re
import secrets
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


def _relabel(
    frame: pd.DataFrame,
    d: DomainContract,
    donors: list[str | None],
    moving: list[int],
    recipient: dict[int, str],
    delta: dict[int, int],
    anchor: pd.DataFrame,
    row_of: dict[str, int],
    spec: Any,
    date_rules: list[FieldRule],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Give rows ``moving`` to ``recipient[i]``: rewrite the subject-level
    columns to the new owner's values and move dates by ``delta[i]``.
    Returns the frame and the subject-level columns it rewrote."""
    new = frame.copy()
    date_cols = {r.column for r in date_rules}
    # Columns that repeat a subject-level value from the anchor domain --
    # SUBJECT, SITE, STUDYID -- shown by the data rather than assumed from
    # names: every row must equal its own subject's anchor value. DOMAIN
    # ('AE' vs 'DM') fails this and is left alone.
    mirrors: dict[str, str] = {}
    for col in frame.columns:
        if col in date_cols:
            continue
        candidates = [col] if col in anchor.columns else []
        if col == d.subject_key and spec.subject_column in anchor.columns:
            candidates.append(spec.subject_column)
        for a in candidates:
            j = frame.columns.get_loc(col)
            if all(
                _same(frame.iat[i, j], anchor.at[row_of[donors[i]], a])
                for i in moving
            ):
                mirrors[col] = a
                break
    if d.subject_key and d.subject_key not in mirrors:
        raise ReassignError(
            f"{d.name}.{d.subject_key}: does not match "
            f"{spec.domain}.{spec.subject_column} row for row, so a record "
            "cannot be relabelled to its new subject"
        )
    for col, a in mirrors.items():
        j = new.columns.get_loc(col)
        for i in moving:
            new.iat[i, j] = anchor.at[row_of[recipient[i]], a]
    for r in date_rules:
        order = r.date_order or rawdates.infer_order(frame[r.column])
        j = new.columns.get_loc(r.column)
        for i in moving:
            if delta[i]:
                new.iat[i, j] = move_date(frame.iat[i, j], delta[i], order)

    after = _keys(new, d)
    wrong = [i for i in moving if str(after.iat[i]) != recipient[i]]
    if wrong:
        raise ReassignError(
            f"{d.name}: {len(wrong)} relabelled row(s) still resolve to their "
            "old subject -- the join_key_template uses a column that does not "
            f"repeat {spec.domain}'s value"
        )
    return new, mirrors


def _sequence_columns(
    frame: pd.DataFrame, donors: list[str | None], skip: set[str]
) -> list[str]:
    """Columns that number each subject's records 1..n (AESEQ, AE_NO).

    Found by the data: every subject's values are exactly 1..n, and somebody
    has more than one. VISITNUM repeats within a subject and is not one; a
    dose that happens to be 1 for a subject with one record is not proof.
    """
    found = []
    for col in frame.columns:
        if col in skip:
            continue
        per: dict[str, list[int]] = {}
        ok = True
        for v, k in zip(frame[col], donors):
            if k is None:
                continue
            text = str(v).strip()
            if text.endswith(".0"):  # a number column read back as float
                text = text[:-2]
            if not text.isdigit():
                ok = False
                break
            per.setdefault(k, []).append(int(text))
        if not ok or not per or max(len(x) for x in per.values()) < 2:
            continue
        if all(sorted(x) == list(range(1, len(x) + 1)) for x in per.values()):
            found.append(col)
    return found


_SUBJECT_LEVEL_NAMES = re.compile(
    r"^(STUDYID|STUDY|PROJECT|USUBJID|SUBJID|SUBJECT|SUBJECTID|SUBJNUM|PATID|"
    r"SITEID|SITE|SITENUM|SITENUMBER|SITENO|STUDYSITEID|INVID|COUNTRY)$"
)


def _self_anchor(
    frame: pd.DataFrame, d: DomainContract, ks: pd.Series
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per subject, built from the domain itself, for a drop with no
    DM to read subjects and sites from.

    The columns kept are the subject-level ones: the subject key, and any
    column that never varies within a subject -- shown by subjects who have
    several records. With nobody holding more than one record there is no
    such evidence, and only the usual names (STUDYID, SUBJECT, SITE...) count.
    """
    keys = [None if pd.isna(k) else str(k) for k in ks]
    rows: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        if k is not None:
            rows.setdefault(k, []).append(i)
    multi = [ix for ix in rows.values() if len(ix) > 1]
    cols = []
    for j, col in enumerate(frame.columns):
        if col == d.subject_key:
            cols.append(col)
            continue
        if len(multi) >= 2:
            if all(
                all(_same(frame.iat[i, j], frame.iat[ix[0], j]) for i in ix)
                for ix in rows.values()
            ):
                cols.append(col)
        elif _SUBJECT_LEVEL_NAMES.match(re.sub(r"[ _\-.]", "", str(col).upper())):
            cols.append(col)
    firsts = [ix[0] for ix in rows.values()]
    anchor = frame.iloc[firsts][cols].reset_index(drop=True)
    return anchor, {k: n for n, k in enumerate(rows)}


def _rows_mode(
    frame: pd.DataFrame,
    d: DomainContract,
    ks: pd.Series,
    anchor: pd.DataFrame,
    row_of: dict[str, int],
    refs: dict[str, date | None],
    spec: Any,
    shuffle_values: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every record to a subject drawn at random, independently.

    With ``shuffle_values`` every other column is then shuffled on its own
    across the domain as well, so nothing on a row belongs together any more
    -- AEDECOD no longer goes with its AEBODSYS, nor the start date with the
    end date. Only the columns that say whose record it is (SUBJECT, SITE,
    STUDYID) and the per-subject sequence number stay put.

    No sets: a subject's three adverse events land on three different
    subjects, or the same one, by chance. The draw is fresh each run and not
    kept -- the original drop is the record of who had what.
    """
    rng = secrets.SystemRandom()
    pool = sorted(row_of)
    donors = [None if pd.isna(k) else str(k) for k in ks]
    stray = {k for k in donors if k is not None and k not in row_of}
    if stray:
        raise ReassignError(
            f"{d.name}: {len(stray)} subject(s) have records here but no row in "
            f"{spec.domain}, so their records have no subject list to be dealt "
            "from. Reassignment needs every subject in the anchor domain."
        )
    moving = [i for i, k in enumerate(donors) if k is not None]
    recipient = {i: rng.choice(pool) for i in moving}
    delta = {}
    for i in moving:
        a, b = refs.get(donors[i]), refs.get(recipient[i])
        # Moving dates to keep a study day true means nothing once the date
        # itself is dealt to a different row.
        delta[i] = (b - a).days if a and b and not shuffle_values else 0
    date_rules = [r for r in d.fields if r.column in frame.columns and _is_date_rule(r)]
    seqs = _sequence_columns(
        frame, donors, {r.column for r in date_rules} | {d.subject_key or ""}
    )
    new, mirrors = _relabel(frame, d, donors, moving, recipient, delta, anchor,
                            row_of, spec, date_rules)

    # Renumber 1..n under each new owner, so AESEQ is still a key.
    for col in seqs:
        counter: dict[str, int] = {}
        j = new.columns.get_loc(col)
        for i in moving:
            counter[recipient[i]] = counter.get(recipient[i], 0) + 1
            was = frame.iat[i, j]
            n = counter[recipient[i]]
            new.iat[i, j] = n if isinstance(was, (int, float)) else str(n)

    shuffled: list[str] = []
    if shuffle_values:
        import re as _re

        fixed = set(mirrors) | set(seqs) | {d.subject_key or ""}
        if d.join_key_template:
            fixed |= set(_re.findall(r"{(\w+)}", d.join_key_template))
        for col in new.columns:
            if col in fixed or new[col].nunique(dropna=False) < 2:
                continue
            perm = list(range(len(new)))
            rng.shuffle(perm)
            new[col] = new[col].iloc[perm].to_numpy()
            shuffled.append(col)

    # And deal the rows out in a random order, so position says nothing
    # about who a record came from or when it was entered.
    order = list(range(len(new)))
    rng.shuffle(order)
    new = new.iloc[order].reset_index(drop=True)
    return new, {
        "mode": "values" if shuffle_values else "rows",
        "columns_shuffled": shuffled,
        "domains": [d.name],
        "rows": len(moving),
        "subjects": len(pool),
        "renumbered": seqs,
        "dates": "moved by the reference-date difference"
        if any(delta.values()) else "kept as written",
    }


def apply(
    frames: dict[str, pd.DataFrame], contract: Contract, vault: Vault
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Reassign the records of every domain the contract marks. Returns the
    new frames and a report of counts -- never keys or dates."""
    marked = [d for d in contract.domains if d.reassign and d.name in frames]
    if not marked:
        return frames, {}

    spec = contract.anchor
    # The subject list, sites and reference dates normally come from DM. A
    # drop of one AE file has no DM -- the anchor is the AE file itself -- and
    # then each domain is dealt among its own subjects, with the subject-level
    # columns read from the domain (see _self_anchor).
    shared = None
    if spec.domain in frames and not contract.domain(spec.domain).reassign:
        anchor_dom = contract.domain(spec.domain)
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
        shared = (anchor, row_of, refs)

    out = dict(frames)
    report: dict[str, Any] = {}
    groups: dict[str, list[DomainContract]] = {}
    for d in marked:
        if d.reassign_mode in ("rows", "values"):
            ks = _keys(frames[d.name], d)
            if ks is None:
                raise ReassignError(
                    f"{d.name}: no subject key to reassign by -- its "
                    "subject_key or join_key_template must name columns it has"
                )
            if shared is not None:
                anchor_ctx = shared
                local_spec = spec
            else:
                a, r = _self_anchor(frames[d.name], d, ks)
                anchor_ctx = (a, r, {})
                local_spec = spec.model_copy(
                    update={"domain": d.name, "subject_column": d.subject_key or ""}
                )
            out[d.name], report[d.name] = _rows_mode(
                frames[d.name], d, ks, *anchor_ctx, local_spec,
                shuffle_values=d.reassign_mode == "values",
            )
            if shared is None:
                report[d.name]["subjects_from"] = d.name
            continue
        groups.setdefault(d.reassign, []).append(d)

    if groups and shared is None:
        raise ReassignError(
            f"moving each subject's records together needs the anchor domain "
            f"{spec.domain!r} in the drop, separate from the domains being "
            "moved: it is where each subject's reference date and site come "
            "from. Without it, use the default shuffle instead."
        )
    if shared is not None:
        anchor, row_of, refs = shared
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

            new, _ = _relabel(
                frame, d, donors, moving,
                {i: mapping[donors[i]][0] for i in moving},
                {i: mapping[donors[i]][1] or 0 for i in moving},
                anchor, row_of, spec, date_rules,
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
