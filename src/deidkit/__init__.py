"""deidkit -- de-identification for inbound EDC data.

Built for one specific set of constraints:

* structured EDC / SDTM-style tabular data
* US HIPAA, targeting Expert Determination (45 CFR 164.514(b)(1)) rather than
  Safe Harbor, because Safe Harbor's removal of all date elements finer than a
  year destroys interval-based clinical analysis
* a controlled re-identification path is required for safety reporting, so the
  output is *pseudonymised* and a crosswalk vault exists
* medical history and adverse events are retained in full, with dates
  reparameterised to study day and verbatim text screened for review rather
  than rewritten
* the de-identified tier -- and only that tier -- is the source for an
  internal model training corpus

Typical use::

    from deidkit import Contract, Vault, DeidPipeline, load_study

    contract = Contract.from_yaml("contracts/study_abc.yaml")
    frames, sums = load_study("data/quarantine/study_abc")

    with Vault("vault/study_abc.db", operator="xli") as vault:
        result = DeidPipeline(contract, vault, operator="xli").run(
            frames, checksums=sums
        )

    print(result.summary())
"""

from .contract import (
    AnchorSpec,
    Contract,
    DomainContract,
    FieldRule,
    RiskSpec,
    Treatment,
)
from .freetext import apply_adjudication, build_detector, screen
from .io import checksum, load_study, read_table, write_study, write_table
from .pipeline import ContractMismatch, DeidPipeline, RunResult
from .risk import RiskReport, measure
from .vault import Vault, VaultError

__version__ = "0.1.0"

__all__ = [
    "AnchorSpec",
    "Contract",
    "ContractMismatch",
    "DeidPipeline",
    "DomainContract",
    "FieldRule",
    "RiskReport",
    "RiskSpec",
    "RunResult",
    "Treatment",
    "Vault",
    "VaultError",
    "__version__",
    "apply_adjudication",
    "build_detector",
    "checksum",
    "load_study",
    "measure",
    "read_table",
    "screen",
    "write_study",
    "write_table",
]
