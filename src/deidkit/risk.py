"""Re-identification risk measurement.

The step home-built masking tools skip, and the one that separates "we masked
it" from something defensible. Transformation without measurement is a claim;
measurement is evidence, and the numbers here are what an Expert Determination
under 45 CFR 164.514(b)(1) actually rests on.

Read the results with the small-trial caveat in mind: a 40-subject Phase I
study may have *no* quasi-identifier combination reaching k=5 without
generalising demographics into uselessness. That is expected. Expert
Determination explicitly permits weighing context -- internal-only recipients,
no egress, audited access -- so a missed threshold is a finding to document as
an accepted risk with its compensating controls named, not a bug to code
around. Which is why ``fail_on_target_miss`` defaults to false and the report
is always emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

#: Placeholder for missing quasi-identifier values when forming equivalence
#: classes. Nulls must not silently split classes -- two records both missing
#: RACE belong in the same class, and pandas' groupby would drop them.
_NULL = "\x00<NA>"


@dataclass
class RiskReport:
    """Measured re-identification risk over one declared quasi-identifier set."""

    domain: str
    quasi_identifiers: list[str]
    n_records: int
    n_classes: int

    k_min: int
    k_target: int
    k_met: bool

    records_below_k: int
    fraction_below_k: float

    prosecutor_risk_max: float
    """Worst-case: 1 / smallest class. The risk to the most exposed subject."""

    marketer_risk_mean: float
    """Average per-record risk = classes / records. The risk of a bulk
    re-identification attempt across the whole release."""

    l_min: int | None = None
    l_target: int | None = None
    l_met: bool | None = None
    sensitive_columns: list[str] = field(default_factory=list)

    smallest_classes: list[dict[str, object]] = field(default_factory=list)
    blocking: list[dict[str, object]] = field(default_factory=list)
    reduction_path: list[dict[str, object]] = field(default_factory=list)
    """Greedy sequence of quasi-identifier removals toward the target.

    A flat "k without column X" table is accurate but useless when every single
    removal leaves k unchanged, which is the normal case: it is the *product* of
    the QI cardinalities against the cohort size that drives k, so relief only
    arrives after several removals or generalisations. This records the actual
    sequence and where it lands -- including landing short, which is the honest
    answer for a small trial.
    """

    def class_size_histogram(self) -> dict[str, int]:
        """How many equivalence classes there are at each size.

        The distribution answers what the manifest needs to attest -- how
        concentrated the exposure is -- without naming who is in the exposed
        classes.
        """
        hist: dict[str, int] = {}
        for c in self.smallest_classes:
            key = str(c["size"])
            hist[key] = hist.get(key, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))

    def to_dict(self, *, include_class_values: bool = False) -> dict[str, object]:
        """Serialise for the manifest.

        ``smallest_classes`` carries the quasi-identifier VALUES of the most
        exposed subjects -- the exact combinations that make one person unique
        in this release. That belongs in a steward's working copy, and not in
        the manifest, for a reason that has nothing to do with the values being
        secret: they are already in the published table beside it. It is that
        the manifest is the artifact that travels. It gets attached to a
        determination, pasted into a ticket, mailed to a reviewer -- and once
        it is away from the data it is a ranked list of who is easiest to
        re-identify, in a file whose whole purpose is to attest that the
        release is safe.

        So the default is the distribution and the counts. Pass
        ``include_class_values=True`` for the steward copy, which is written
        beside the review queue and inherits its access controls.
        """
        d = {
            "domain": self.domain,
            "quasi_identifiers": self.quasi_identifiers,
            "n_records": self.n_records,
            "n_equivalence_classes": self.n_classes,
            "k_min": self.k_min,
            "k_target": self.k_target,
            "k_met": self.k_met,
            "records_below_k": self.records_below_k,
            "fraction_below_k": self.fraction_below_k,
            "prosecutor_risk_max": self.prosecutor_risk_max,
            "marketer_risk_mean": self.marketer_risk_mean,
            "class_size_histogram": self.class_size_histogram(),
            "n_classes_below_k": sum(
                1 for c in self.smallest_classes if int(c["size"]) < self.k_target
            ),
            "blocking_quasi_identifiers": self.blocking,
            "reduction_path": self.reduction_path,
        }
        if include_class_values:
            d["smallest_classes"] = self.smallest_classes
        if self.l_target is not None:
            d |= {
                "l_min": self.l_min,
                "l_target": self.l_target,
                "l_met": self.l_met,
                "sensitive_columns": self.sensitive_columns,
            }
        return d

    def summary(self) -> str:
        lines = [
            f"Risk report -- {self.domain}",
            f"  quasi-identifiers : {', '.join(self.quasi_identifiers) or '(none)'}",
            f"  records           : {self.n_records} in {self.n_classes} classes",
            f"  k                 : {self.k_min} (target {self.k_target}) "
            f"{'PASS' if self.k_met else 'MISS'}",
            f"  below target      : {self.records_below_k} records "
            f"({self.fraction_below_k:.1%})",
            f"  prosecutor (max)  : {self.prosecutor_risk_max:.3f}",
            f"  marketer (mean)   : {self.marketer_risk_mean:.3f}",
        ]
        if self.l_target is not None:
            lines.append(
                f"  l-diversity       : {self.l_min} (target {self.l_target}) "
                f"{'PASS' if self.l_met else 'MISS'}"
            )
        if self.blocking:
            lines.append("  k with one QI removed:")
            for b in self.blocking:
                lines.append(f"    - without {b['column']:<20} k = {b['k_min']}")
        if self.reduction_path:
            lines.append("  greedy path toward the target:")
            for step in self.reduction_path:
                lines.append(
                    f"    - also drop {str(step['column']):<20} k = {step['k_min']}"
                    + ("  <- target met" if step["target_met"] else "")
                )
            if not self.reduction_path[-1]["target_met"]:
                lines.append(
                    f"    target not reachable by removal alone: {self.n_records} "
                    "records cannot fill the class space these QIs create."
                )
                lines.append(
                    "    Generalise instead (age -> bands, site -> region), or "
                    "carry the residual"
                )
                lines.append(
                    "    risk into the determination with its contextual controls."
                )
        if self.smallest_classes:
            lines.append("  smallest classes:")
            for c in self.smallest_classes[:5]:
                vals = ", ".join(f"{k}={v}" for k, v in c["values"].items())
                lines.append(f"    - n={c['size']:<4} {vals}")
        return "\n".join(lines)


def _key_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Stringified QI columns with nulls made explicit."""
    out = pd.DataFrame(index=frame.index)
    for c in columns:
        out[c] = frame[c].astype("string").fillna(_NULL)
    return out


def measure(
    frame: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    *,
    domain: str = "",
    k_target: int = 5,
    l_target: int | None = None,
    sensitive_columns: Sequence[str] = (),
    n_smallest: int = 10,
    diagnose_blocking: bool = True,
) -> RiskReport:
    """Measure k-anonymity, l-diversity, and population risk metrics."""
    qis = [c for c in quasi_identifiers if c in frame.columns]
    missing = [c for c in quasi_identifiers if c not in frame.columns]
    if missing:
        raise KeyError(
            f"quasi-identifier columns absent from {domain or 'frame'}: {missing}. "
            "The contract declares them; the data does not have them."
        )

    n = len(frame)
    if n == 0 or not qis:
        return RiskReport(
            domain=domain,
            quasi_identifiers=list(qis),
            n_records=n,
            n_classes=0,
            k_min=0,
            k_target=k_target,
            k_met=False,
            records_below_k=0,
            fraction_below_k=0.0,
            prosecutor_risk_max=0.0,
            marketer_risk_mean=0.0,
        )

    keys = _key_frame(frame, qis)
    sizes = keys.groupby(qis, dropna=False, sort=False).size()

    k_min = int(sizes.min())
    n_classes = int(len(sizes))
    small = sizes[sizes < k_target]
    records_below = int(small.sum())

    # --- smallest classes, for diagnosis -----------------------------
    smallest: list[dict[str, object]] = []
    for key, size in sizes.nsmallest(n_smallest, keep="all").items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        smallest.append(
            {
                "size": int(size),
                "values": {
                    c: (None if v == _NULL else v)
                    for c, v in zip(qis, key_tuple)
                },
            }
        )

    # --- which QI is destroying k? ------------------------------------
    blocking: list[dict[str, object]] = []
    if diagnose_blocking and len(qis) > 1:
        for c in qis:
            rest = [x for x in qis if x != c]
            sub = int(keys.groupby(rest, dropna=False, sort=False).size().min())
            blocking.append({"column": c, "k_min": sub})
        blocking.sort(key=lambda b: -int(b["k_min"]))

    # --- greedy reduction path ----------------------------------------
    reduction: list[dict[str, object]] = []
    if diagnose_blocking and k_min < k_target and len(qis) > 1:
        remaining = list(qis)
        while len(remaining) > 1:
            best_col, best_k = None, -1
            for c in remaining:
                rest = [x for x in remaining if x != c]
                cand = int(
                    keys.groupby(rest, dropna=False, sort=False).size().min()
                )
                if cand > best_k:
                    best_col, best_k = c, cand
            assert best_col is not None
            remaining.remove(best_col)
            met = best_k >= k_target
            reduction.append(
                {
                    "column": best_col,
                    "remaining": list(remaining),
                    "k_min": best_k,
                    "target_met": met,
                }
            )
            if met:
                break

    # --- l-diversity ---------------------------------------------------
    l_min: int | None = None
    l_met: bool | None = None
    sens = [c for c in sensitive_columns if c in frame.columns]
    if l_target is not None and sens:
        combined = keys.join(frame[sens].astype("string"))
        per_class = combined.groupby(qis, dropna=False, sort=False)[sens].nunique()
        # A class is only as diverse as its least diverse sensitive column.
        l_min = int(per_class.min(axis=1).min())
        l_met = l_min >= l_target

    return RiskReport(
        domain=domain,
        quasi_identifiers=list(qis),
        n_records=n,
        n_classes=n_classes,
        k_min=k_min,
        k_target=k_target,
        k_met=k_min >= k_target,
        records_below_k=records_below,
        fraction_below_k=round(records_below / n, 4),
        prosecutor_risk_max=round(1.0 / k_min, 4),
        marketer_risk_mean=round(n_classes / n, 4),
        l_min=l_min,
        l_target=l_target,
        l_met=l_met,
        sensitive_columns=list(sens),
        smallest_classes=smallest,
        blocking=blocking,
        reduction_path=reduction,
    )
