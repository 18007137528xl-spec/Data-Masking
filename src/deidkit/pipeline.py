"""Pipeline orchestration.

Order of operations, and why:

1. **Validate** the data against the contract, fail-closed. A column present in
   the data but absent from the contract halts the run -- the alternative is a
   silent pass-through of an unreviewed field, which is exactly how identifiers
   reach a published tier.
2. **Resolve anchors** from the raw data, before any transformation. Study-day
   conversion needs each subject's real reference date.
3. **Screen free text** on the raw data. Screening reads; it does not write.
4. **Transform**, domain by domain, according to the contract.
5. **Measure risk** on the transformed output.
6. **Emit a manifest** recording all of the above.

The manifest is the point. It is what turns a pipeline run into evidence: rule
set version, per-column treatment applied, categories pooled, screening hit
rates, risk metrics achieved, input checksums, operator, timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import freetext, risk as risk_mod, transforms as tf
from .contract import (
    NEEDS_VAULT,
    Contract,
    DomainContract,
    FieldRule,
    Treatment,
)
from .vault import Vault

TOOL_VERSION = "0.1.0"


class ContractMismatch(RuntimeError):
    """The incoming data does not match the contract.

    Raised rather than warned: schema drift on a source that handles PHI is a
    review event, not something to work around at runtime.
    """


class MissingAnchor(RuntimeError):
    pass


@dataclass
class DomainResult:
    name: str
    frame: pd.DataFrame
    rows_in: int
    rows_out: int
    treatments: dict[str, str] = field(default_factory=dict)
    pooled_categories: dict[str, list[str]] = field(default_factory=dict)
    unconverted_dates: dict[str, int] = field(default_factory=dict)
    retained_in_full: bool = False


@dataclass
class RunResult:
    frames: dict[str, pd.DataFrame]
    review_queue: pd.DataFrame
    risk_report: risk_mod.RiskReport | None
    manifest: dict[str, Any]
    domains: dict[str, DomainResult]

    def summary(self) -> str:
        lines = [
            f"tier            : {self.manifest['tier']}",
            f"contract        : {self.manifest['contract']['contract_version']}",
            f"domains         : {len(self.frames)}",
        ]
        for name, d in self.domains.items():
            tag = "  [retained in full]" if d.retained_in_full else ""
            lines.append(
                f"  {name:<6} {d.rows_in:>7} rows -> {len(d.frame.columns):>3} cols{tag}"
            )
        lines.append(
            f"review queue    : {len(self.review_queue)} rows flagged for adjudication"
        )
        if self.risk_report:
            lines.append("")
            lines.append(self.risk_report.summary())
        return "\n".join(lines)


class DeidPipeline:
    def __init__(
        self,
        contract: Contract,
        vault: Vault,
        *,
        operator: str = "unknown",
        detector: Any | None = None,
    ) -> None:
        self.contract = contract
        self.vault = vault
        self.operator = operator
        self._detector = detector

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(
        self, frames: dict[str, pd.DataFrame], *, allow_missing: bool = False
    ) -> None:
        problems: list[str] = []

        for dom in self.contract.domains:
            if dom.name not in frames:
                problems.append(f"{dom.name}: domain in contract but not in data")
                continue
            have = set(frames[dom.name].columns)
            declared = {f.column for f in dom.fields}

            undeclared = sorted(have - declared)
            if undeclared:
                problems.append(
                    f"{dom.name}: columns present in data but not in contract "
                    f"{undeclared} -- add a rule for each (use 'drop' if not needed). "
                    "Refusing to pass unreviewed columns through."
                )
            absent = sorted(declared - have)
            if absent and not allow_missing:
                problems.append(
                    f"{dom.name}: columns in contract but not in data {absent} "
                    "-- schema drift; re-profile the source or pass allow_missing"
                )

        extra = sorted(set(frames) - {d.name for d in self.contract.domains})
        if extra:
            problems.append(
                f"domains present in data but not in contract: {extra} -- "
                "these will not be published"
            )

        if problems:
            raise ContractMismatch("\n".join(problems))

    # ------------------------------------------------------------------
    # anchors
    # ------------------------------------------------------------------
    def anchor_map(self, frames: dict[str, pd.DataFrame]) -> dict[str, str]:
        spec = self.contract.anchor
        if spec.domain not in frames:
            raise MissingAnchor(f"anchor domain {spec.domain!r} not in data")
        src = frames[spec.domain]
        for col in (spec.subject_column, spec.date_column):
            if col not in src.columns:
                raise MissingAnchor(
                    f"anchor column {col!r} not in {spec.domain}"
                )
        pairs = src[[spec.subject_column, spec.date_column]].dropna(
            subset=[spec.subject_column]
        )
        return {
            str(s): (None if pd.isna(d) else str(d))
            for s, d in zip(pairs[spec.subject_column], pairs[spec.date_column])
        }

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def _subject_series(
        self, frame: pd.DataFrame, dom: DomainContract
    ) -> pd.Series | None:
        """Original (pre-surrogate) subject identifiers for this domain."""
        key = dom.subject_key
        if key and key in frame.columns:
            return frame[key].astype("string")
        return None

    def transform_domain(
        self,
        frame: pd.DataFrame,
        dom: DomainContract,
        anchors: dict[str, str],
    ) -> DomainResult:
        result = DomainResult(
            name=dom.name,
            frame=pd.DataFrame(index=frame.index),
            rows_in=len(frame),
            rows_out=len(frame),
            retained_in_full=dom.retained_in_full,
        )

        subjects = self._subject_series(frame, dom)
        anchor_series = (
            subjects.map(lambda s: anchors.get(str(s)))
            if subjects is not None
            else pd.Series([None] * len(frame), index=frame.index, dtype="string")
        )

        out = result.frame
        for rule in dom.fields:
            if rule.column not in frame.columns:
                continue
            if rule.treatment is Treatment.DROP:
                result.treatments[rule.column] = "drop"
                continue

            self._apply_rule(rule, frame, out, subjects, anchor_series, result, dom)

        result.frame = out
        return result

    def _apply_rule(
        self,
        rule: FieldRule,
        frame: pd.DataFrame,
        out: pd.DataFrame,
        subjects: pd.Series | None,
        anchors: pd.Series,
        result: DomainResult,
        dom: DomainContract,
    ) -> None:
        col = frame[rule.column]
        name = rule.resolved_output()
        t = rule.treatment
        result.treatments[rule.column] = tf.describe_rule(rule)

        if t in (Treatment.RETAIN, Treatment.SCREEN_FREETEXT):
            # SCREEN_FREETEXT is deliberately non-mutating: the screening pass
            # produces a review queue, the data is published as received.
            out[name] = col

        elif t is Treatment.SURROGATE_ID:
            out[name] = tf.surrogate(
                col,
                self.vault,
                entity=rule.entity or "subject",
                prefix=rule.prefix or "",
            )

        elif t is Treatment.FAKER:
            out[name] = tf.faker_column(
                col, provider=rule.faker_provider or "word"
            )

        elif t is Treatment.DATE_TO_STUDY_DAY:
            days, grain = tf.to_study_day(
                col, anchors, convention=self.contract.anchor.convention
            )
            out[name] = days
            out[f"{name}_GRAN"] = grain
            unconverted = int(days.isna().sum() - col.isna().sum())
            if unconverted > 0:
                result.unconverted_dates[rule.column] = unconverted

        elif t is Treatment.PARTIAL_DATE_TO_YEAR_OFFSET:
            offs, grain = tf.to_year_offset(col, anchors)
            out[name] = offs
            out[f"{name}_GRAN"] = grain

        elif t is Treatment.DOB_TO_AGE:
            out[name] = tf.dob_to_age(col, anchors, cap=rule.cap or 90)

        elif t is Treatment.DATE_SHIFT:
            if subjects is None:
                raise ContractMismatch(
                    f"{dom.name}.{rule.column}: date_shift needs the domain's "
                    "subject_key to be set"
                )
            out[name] = tf.shift_dates(
                col, subjects, self.vault, entity=rule.entity or "subject"
            )

        elif t is Treatment.ZIP3:
            out[name] = tf.zip3(col)

        elif t is Treatment.LABEL_MAP:
            out[name] = tf.label_map(
                col,
                self.vault,
                entity=rule.entity or "treatment",
                prefix=rule.prefix or "TRT",
                keep_values=rule.keep_values,
            )

        elif t is Treatment.CAP_NUMERIC:
            out[name] = tf.cap_numeric(col, cap=float(rule.cap or 90))

        elif t is Treatment.GENERALIZE_NUMERIC:
            out[name] = tf.generalize_numeric(
                col, bins=rule.bins or [], cap=rule.cap
            )

        elif t is Treatment.POOL_RARE:
            pooled, rare = tf.pool_rare(
                col,
                min_count=rule.min_count or 1,
                pooled_value=rule.pooled_value,
            )
            out[name] = pooled
            if rare:
                result.pooled_categories[rule.column] = rare

        elif t is Treatment.REDACT_FREETEXT:
            # Detect-and-replace. Not the default for verbatim clinical text.
            det = self._detector or freetext.build_detector()
            out[name] = pd.Series(
                [
                    None
                    if pd.isna(v)
                    else _redact(str(v), det)
                    for v in col
                ],
                index=col.index,
                dtype="string",
            )

        else:  # pragma: no cover - Treatment is exhaustive
            raise ContractMismatch(f"unhandled treatment {t!r}")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        checksums: dict[str, str] | None = None,
        allow_missing: bool = False,
        screen: bool = True,
    ) -> RunResult:
        self.validate(frames, allow_missing=allow_missing)
        anchors = self.anchor_map(frames)

        # --- 3. screen free text on the raw data ----------------------
        queue = pd.DataFrame(columns=freetext.REVIEW_COLUMNS)
        screen_summary: dict[str, Any] = {"flagged_rows": 0, "by_column": {}}
        if screen:
            targets = [
                (d.name, f.column, d.subject_key)
                for d in self.contract.domains
                for f in d.fields
                if f.treatment is Treatment.SCREEN_FREETEXT
            ]
            if targets:
                queue = freetext.screen(
                    frames, targets, detector=self._detector
                )
                screen_summary = freetext.screening_summary(queue, frames)

        # --- 4. transform ---------------------------------------------
        results: dict[str, DomainResult] = {}
        for dom in self.contract.domains:
            if dom.name not in frames:
                continue
            results[dom.name] = self.transform_domain(
                frames[dom.name], dom, anchors
            )
        self.vault.commit()

        out_frames = {k: v.frame for k, v in results.items()}

        # --- 5. measure risk ------------------------------------------
        report: risk_mod.RiskReport | None = None
        spec = self.contract.risk
        if spec.domain in out_frames:
            qis = self.contract.quasi_identifiers(spec.domain)
            if qis:
                report = risk_mod.measure(
                    out_frames[spec.domain],
                    qis,
                    domain=spec.domain,
                    k_target=spec.k_target,
                    l_target=spec.l_target,
                    sensitive_columns=spec.sensitive_columns,
                )
                if spec.fail_on_target_miss and not report.k_met:
                    raise RuntimeError(
                        f"risk gate: k={report.k_min} below target "
                        f"{spec.k_target}\n{report.summary()}"
                    )

        # --- 6. manifest ----------------------------------------------
        manifest = self._manifest(
            results, checksums or {}, screen_summary, report, anchors
        )
        return RunResult(
            frames=out_frames,
            review_queue=queue,
            risk_report=report,
            manifest=manifest,
            domains=results,
        )

    def _manifest(
        self,
        results: dict[str, DomainResult],
        checksums: dict[str, str],
        screen_summary: dict[str, Any],
        report: risk_mod.RiskReport | None,
        anchors: dict[str, str],
    ) -> dict[str, Any]:
        missing_anchor = sorted(s for s, d in anchors.items() if not d)
        return {
            "tool": {"name": "deidkit", "version": TOOL_VERSION},
            "run": {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "operator": self.operator,
            },
            "tier": self.contract.tier,
            "contract": self.contract.fingerprint(),
            "inputs": {
                "checksums_sha256": checksums,
                "rows_in": {k: v.rows_in for k, v in results.items()},
            },
            "anchor": {
                "domain": self.contract.anchor.domain,
                "date_column": self.contract.anchor.date_column,
                "convention": self.contract.anchor.convention,
                "subjects_resolved": len(anchors) - len(missing_anchor),
                "subjects_without_anchor": len(missing_anchor),
            },
            "transformation": {
                name: {
                    "retained_in_full": d.retained_in_full,
                    "treatments": d.treatments,
                    "pooled_categories": d.pooled_categories,
                    "dates_unconverted": d.unconverted_dates,
                    "output_columns": list(d.frame.columns),
                }
                for name, d in results.items()
            },
            "freetext_screening": screen_summary,
            "risk": report.to_dict() if report else None,
            "notes": [
                "Surrogate identifiers are randomly generated, not derived from "
                "the original values (45 CFR 164.514(c)).",
                "Absolute dates are reparameterised to study day; intervals are "
                "preserved.",
                "Free-text columns under 'screen_freetext' are published as "
                "received; the review queue records candidates for adjudication.",
                "Domains marked retained_in_full carry rare coded terms by "
                "explicit decision -- see the determination for the accepted "
                "risk and its compensating controls.",
            ],
        }


def _redact(text: str, detector: Any) -> str:
    findings = detector.detect(text)
    if not findings:
        return text
    out, cursor = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        out.append(text[cursor : f.start])
        out.append(f"<{f.entity_type}>")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)
