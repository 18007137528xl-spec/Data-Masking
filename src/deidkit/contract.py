"""Versioned field contract.

The contract is the single source of truth for what happens to every column.
It is authored once per source (assisted by ``deidkit profile``), reviewed by a
data steward, committed to version control, and thereafter validated against
each incoming drop. Schema drift is an error, not a silent pass-through.

Fail-closed rule: a column present in the data but absent from the contract
halts the run. Unknown columns are never emitted.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Treatment(str, Enum):
    """What happens to a column.

    Chosen by the column's *analytic role*, not its data type.
    """

    # --- pass through -------------------------------------------------
    RETAIN = "retain"
    """No transformation. Measurements, coded clinical content, MH/AE."""

    DROP = "drop"
    """Remove the column entirely."""

    # --- identifiers --------------------------------------------------
    SURROGATE_ID = "surrogate_id"
    """Random, non-derived surrogate + crosswalk entry in the vault.

    Deliberately NOT a keyed hash of the original: HIPAA 164.514(c) requires a
    re-identification code not be derived from information about the individual.
    """

    FAKER = "faker"
    """Synthetic replacement, for direct identifiers whose column must persist."""

    # --- dates --------------------------------------------------------
    DATE_TO_STUDY_DAY = "date_to_study_day"
    """Absolute date -> signed integer day relative to the subject's anchor.

    A reparameterisation, not a redaction: every interval quantity survives.
    """

    PARTIAL_DATE_TO_YEAR_OFFSET = "partial_date_to_year_offset"
    """Partial ISO 8601 date (``2015``, ``2015-03``) -> relative year offset
    plus a granularity flag. Never imputes a day."""

    DOB_TO_AGE = "dob_to_age"
    """Date of birth -> age in years at anchor, with an upper cap band."""

    DATE_SHIFT = "date_shift"
    """Retain a calendar date but apply the subject's vault-held offset.

    Only for genuine seasonality analysis. Prefer DATE_TO_STUDY_DAY.
    """

    # --- quasi-identifiers --------------------------------------------
    CAP_NUMERIC = "cap_numeric"
    """Keep the exact value, but collapse everything at or above a cap.

    The precise shape HIPAA asks for on age: exact years below 90, a single
    ``90+`` band above it, because extreme ages are near-unique. Preferred over
    GENERALIZE_NUMERIC for age under Expert Determination -- banding every age
    loses precision the risk numbers do not require.
    """

    GENERALIZE_NUMERIC = "generalize_numeric"
    """Band a numeric column into intervals, with an optional top cap."""

    ZIP3 = "zip3"
    """Truncate to 3-digit ZIP; low-population prefixes -> ``000``."""

    POOL_RARE = "pool_rare"
    """Categories below a frequency threshold -> a pooled value."""

    # --- free text ----------------------------------------------------
    SCREEN_FREETEXT = "screen_freetext"
    """Detect candidate PHI and emit a review queue. Data is NOT modified.

    The default for MH/AE verbatim: screen and adjudicate, never blanket-redact.
    """

    REDACT_FREETEXT = "redact_freetext"
    """Detect and replace in place. Available, but not the default for verbatim
    clinical text -- it destroys content required for regulatory review."""


#: Treatments that leave the column's values byte-identical.
NON_MUTATING: frozenset[Treatment] = frozenset(
    {Treatment.RETAIN, Treatment.SCREEN_FREETEXT}
)

#: Treatments that require a per-subject anchor date to be resolvable.
NEEDS_ANCHOR: frozenset[Treatment] = frozenset(
    {
        Treatment.DATE_TO_STUDY_DAY,
        Treatment.PARTIAL_DATE_TO_YEAR_OFFSET,
        Treatment.DOB_TO_AGE,
    }
)

#: Treatments that write to (or read from) the crosswalk vault.
NEEDS_VAULT: frozenset[Treatment] = frozenset(
    {Treatment.SURROGATE_ID, Treatment.DATE_SHIFT}
)


class FieldRule(BaseModel):
    """The treatment assigned to one column of one domain."""

    model_config = ConfigDict(extra="forbid")

    column: str
    treatment: Treatment
    note: str | None = None

    # --- treatment parameters ----------------------------------------
    entity: str | None = Field(
        default=None,
        description="Surrogate namespace, e.g. 'subject' or 'site'. "
        "Distinct entities never share a surrogate space.",
    )
    prefix: str | None = Field(
        default=None, description="Human-readable surrogate prefix, e.g. 'SUBJ'."
    )
    faker_provider: str | None = Field(
        default=None, description="Faker provider name, e.g. 'name', 'street_address'."
    )
    output_column: str | None = Field(
        default=None,
        description="Rename the output. Defaults to a treatment-specific name "
        "for date conversions (e.g. AESTDTC -> AESTDY).",
    )
    cap: int | None = Field(
        default=None, description="Top cap for DOB_TO_AGE / GENERALIZE_NUMERIC (e.g. 90)."
    )
    bins: list[float] | None = Field(
        default=None, description="Explicit band edges for GENERALIZE_NUMERIC."
    )
    min_count: int | None = Field(
        default=None, description="Frequency threshold for POOL_RARE."
    )
    pooled_value: str = Field(
        default="OTHER", description="Replacement for POOL_RARE categories below threshold."
    )
    is_quasi_identifier: bool = Field(
        default=False,
        description="Include the OUTPUT column in the k-anonymity QI set.",
    )

    @model_validator(mode="after")
    def _check_params(self) -> FieldRule:
        t = self.treatment
        if t is Treatment.SURROGATE_ID and not self.entity:
            raise ValueError(f"{self.column}: surrogate_id requires 'entity'")
        if t is Treatment.FAKER and not self.faker_provider:
            raise ValueError(f"{self.column}: faker requires 'faker_provider'")
        if t is Treatment.POOL_RARE and self.min_count is None:
            raise ValueError(f"{self.column}: pool_rare requires 'min_count'")
        if t is Treatment.GENERALIZE_NUMERIC and not self.bins:
            raise ValueError(f"{self.column}: generalize_numeric requires 'bins'")
        if t is Treatment.CAP_NUMERIC and self.cap is None:
            raise ValueError(f"{self.column}: cap_numeric requires 'cap'")
        if t is Treatment.DATE_SHIFT and not self.entity:
            raise ValueError(
                f"{self.column}: date_shift requires 'entity' (the offset key, "
                "normally 'subject')"
            )
        return self

    def resolved_output(self) -> str:
        """Name of the column this rule emits."""
        if self.output_column:
            return self.output_column
        if self.treatment is Treatment.DATE_TO_STUDY_DAY:
            # SDTM convention: --DTC (character date) -> --DY (study day)
            if self.column.upper().endswith("DTC"):
                return self.column[:-3] + "DY"
            return self.column + "_STUDY_DAY"
        if self.treatment is Treatment.PARTIAL_DATE_TO_YEAR_OFFSET:
            return self.column + "_YR_REL"
        if self.treatment is Treatment.DOB_TO_AGE:
            return "AGE"
        return self.column


class DomainContract(BaseModel):
    """Contract for one SDTM-style domain (one input table)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    subject_key: str | None = Field(
        default=None,
        description="Column holding the subject identifier, used to look up the "
        "anchor date and the per-subject date offset. Usually USUBJID.",
    )
    retained_in_full: bool = Field(
        default=False,
        description="Documents a domain-level retention decision (MH, AE). "
        "Purely declarative -- rules still govern behaviour -- but it is "
        "asserted against the rules at load time.",
    )
    fields: list[FieldRule]

    @model_validator(mode="after")
    def _check(self) -> DomainContract:
        seen: set[str] = set()
        for f in self.fields:
            if f.column in seen:
                raise ValueError(f"{self.name}: duplicate rule for column {f.column!r}")
            seen.add(f.column)

        outs: dict[str, str] = {}
        for f in self.fields:
            if f.treatment is Treatment.DROP:
                continue
            out = f.resolved_output()
            if out in outs:
                raise ValueError(
                    f"{self.name}: rules for {outs[out]!r} and {f.column!r} both "
                    f"emit column {out!r}"
                )
            outs[out] = f.column

        if self.retained_in_full:
            # A domain declared "retained in full" may only reparameterise dates,
            # screen text, or pass through. Any value-destroying treatment is a
            # contradiction between the declaration and the rules.
            allowed = NON_MUTATING | {
                Treatment.DATE_TO_STUDY_DAY,
                Treatment.PARTIAL_DATE_TO_YEAR_OFFSET,
                Treatment.SURROGATE_ID,
            }
            bad = [
                f.column for f in self.fields if f.treatment not in allowed
            ]
            if bad:
                raise ValueError(
                    f"{self.name}: declared retained_in_full but these columns use "
                    f"value-destroying treatments: {sorted(bad)}. Either change the "
                    f"treatments or drop the retained_in_full declaration -- an "
                    f"accepted risk and an unhandled omission must not look alike."
                )
        return self

    def rule(self, column: str) -> FieldRule | None:
        for f in self.fields:
            if f.column == column:
                return f
        return None


class AnchorSpec(BaseModel):
    """Where the per-subject reference date (study Day 1) comes from.

    Every ``date_to_study_day`` conversion is relative to this. Absent an
    anchor for a subject, that subject's dates cannot be converted and the
    rows are held rather than silently emitted with a null day.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(description="Domain holding the anchor, e.g. 'DM' or 'EX'.")
    subject_column: str = "USUBJID"
    date_column: str = Field(description="e.g. 'RFXSTDTC' (first dose) or 'RFSTDTC'.")
    convention: Literal["day1", "day0"] = Field(
        default="day1",
        description="SDTM uses Day 1 for the anchor date with no Day 0; 'day0' "
        "gives a plain difference instead.",
    )


class RiskSpec(BaseModel):
    """Targets for the risk measurement gate."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(
        default="DM",
        description="Domain the QI set is measured on -- normally one row per subject.",
    )
    k_target: int = 5
    l_target: int | None = Field(
        default=None, description="l-diversity target on 'sensitive_columns'."
    )
    sensitive_columns: list[str] = Field(default_factory=list)
    fail_on_target_miss: bool = Field(
        default=False,
        description="If true, a run that misses the target fails. Default false: "
        "small trials routinely cannot reach k=5, and Expert Determination "
        "explicitly weighs contextual controls. Missing the target must be a "
        "documented accepted risk, not a silent one.",
    )


class Contract(BaseModel):
    """The full versioned field contract for one data source."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(
        description="Bump on any rule change. Recorded in every manifest so an "
        "analysis from six months ago remains reproducible."
    )
    source: str = Field(description="Data source / study identifier.")
    tier: Literal["lds", "deidentified"] = Field(
        description="Which output tier this contract produces. 'lds' remains PHI "
        "and requires a DUA; 'deidentified' is the Expert Determination output "
        "and is the only tier a training corpus may be built from."
    )
    anchor: AnchorSpec
    risk: RiskSpec = Field(default_factory=RiskSpec)
    domains: list[DomainContract]

    @model_validator(mode="after")
    def _check(self) -> Contract:
        names = [d.name for d in self.domains]
        if len(names) != len(set(names)):
            raise ValueError("duplicate domain names in contract")
        if self.anchor.domain not in names:
            raise ValueError(
                f"anchor domain {self.anchor.domain!r} is not in the contract"
            )
        if self.risk.domain not in names:
            raise ValueError(f"risk domain {self.risk.domain!r} is not in the contract")
        return self

    def domain(self, name: str) -> DomainContract:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(name)

    # --- serialisation ------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> Contract:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def to_yaml(self, path: str) -> None:
        """Write the contract for a steward to read and edit.

        Per-field defaults are omitted to keep the rules scannable, but the
        header, anchor, and risk blocks are always written in full: a steward
        cannot tune a k target that the file does not mention, and an anchor
        left implicit is an anchor nobody checks.
        """
        data = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        data["contract_version"] = self.contract_version
        data["source"] = self.source
        data["tier"] = self.tier
        data["anchor"] = self.anchor.model_dump(mode="json", exclude_none=True)
        data["risk"] = self.risk.model_dump(mode="json", exclude_none=True)
        # Reorder so the decisions a human makes come first.
        order = ["contract_version", "source", "tier", "anchor", "risk", "domains"]
        data = {k: data[k] for k in order if k in data}
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)

    def quasi_identifiers(self, domain: str) -> list[str]:
        """Output column names flagged as quasi-identifiers for a domain."""
        d = self.domain(domain)
        return [
            f.resolved_output()
            for f in d.fields
            if f.is_quasi_identifier and f.treatment is not Treatment.DROP
        ]

    def fingerprint(self) -> dict[str, Any]:
        """Compact record of the rule set, for the manifest."""
        return {
            "contract_version": self.contract_version,
            "source": self.source,
            "tier": self.tier,
            "domains": {
                d.name: {
                    "retained_in_full": d.retained_in_full,
                    "rules": {
                        f.column: f.treatment.value for f in sorted(
                            d.fields, key=lambda r: r.column
                        )
                    },
                }
                for d in sorted(self.domains, key=lambda x: x.name)
            },
        }
