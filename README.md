# deidkit

De-identification pipeline for inbound EDC / SDTM clinical data.

Built for one specific set of constraints, and opinionated because of them:

| Constraint | Consequence in the design |
|---|---|
| Structured EDC / SDTM tabular data | Column treatments keyed to SDTM naming; SAS `.xpt` / `.sas7bdat` read natively |
| US HIPAA | Targets **Expert Determination** (§164.514(b)(1)), not Safe Harbor |
| Re-identification required for safety reporting | A crosswalk vault exists; output is *pseudonymised*, not anonymous |
| MH and AE retained in full | Verbatim text is **screened for review, never rewritten**; rare coded terms are not pooled |
| Internal model training is a downstream use | Only the de-identified tier may feed a corpus; the LDS tier may not |

## Why not Safe Harbor

Safe Harbor requires removing every date element more specific than a year.
That removes TEAE window determination, time to onset, duration, resolution,
and dosing-interval overlap — most of what EDC data is for. It also constrains
how a re-identification code may be built (§164.514(c)): the code must not be
derived from information about the individual, which rules out
`HMAC(subject_id)`.

So: surrogates here are **randomly generated and stored in a vault**, and dates
are **reparameterised to study day** rather than redacted. Study day is an
interval, not a date element, and it preserves every interval quantity exactly.

## Install

One-shot install with a verified self-check:

```bash
./setup.sh              # Linux / macOS
```
```
setup.bat               # Windows -- double-click; it launches setup.ps1
```

Both create a virtualenv, install dependencies, fabricate a synthetic study,
run the pipeline, and assert the design guarantees against the published
output.

Flags: `--core` skips the optional extras, `--skip-model` skips the 560 MB
spaCy download, `--skip-check` installs without running the pipeline. On
Windows, `setup.bat` accepts both these and the native PowerShell spellings
(`-Core`, `-SkipModel`, `-SkipSelfCheck`); `setup.ps1`, called directly, takes
only the PowerShell ones.

Or by hand:

```bash
pip install -e .                  # core
pip install -e '.[all]'           # + Presidio NER, SAS readers, Faker, Parquet
pip install -e '.[aws]'           # + KMS envelope keys and s3:// storage
```

Optional extras and what you lose without them:

| Extra | Gives you | Without it |
|---|---|---|
| `screening` | Presidio NER for free text | Built-in pattern detector: solid on emails, phones, facilities; weaker on bare person names |
| `sas` | `.xpt` / `.sas7bdat` input | CSV / TSV / Parquet only |
| `surrogates` | Realistic Faker values | `PROVIDER-000001`-style placeholders |
| `parquet` | Parquet tiers | CSV output |
| `aws` / `azure` / `gcp` | KMS-backed vault keys and `s3://` / `az://` / `gs://` storage | Local filesystem only |

## Workflow

```bash
# 0. one vault key, held in a KMS -- never beside the vault file
export DEIDKIT_VAULT_KEY=$(deidkit keygen)

# 1. profile a drop and draft a contract
deidkit profile data/quarantine/study_abc \
    -o contracts/study_abc.yaml \
    --review out/steward_review.csv

# 2. a data steward reviews EVERY rule, then the contract is committed

# 3. transform, screen, measure, publish
deidkit run data/quarantine/study_abc \
    -c contracts/study_abc.yaml \
    -o tiers/deidentified \
    --vault vault/study_abc.db \
    --operator xli

# 4. adjudicate the free-text queue, then publish the final tier
deidkit adjudicate tiers/deidentified \
    --queue tiers/deidentified/review_queue.csv \
    -o tiers/deidentified_final

# break-glass re-identification, always logged
deidkit reverse SUBJ-8F3K2P --vault vault/study_abc.db \
    -j "SAE-2026-0031 expedited safety report"
```

The profiler's output is a **draft**. Suggestions come from SDTM naming
convention plus content heuristics, not from understanding your study. Rules it
is unsure about are printed as low-confidence and must be reviewed.

## Training a model to derive SDTM from raw EDC

This is a different job from publishing an analysis dataset, and it changes
what the dates are for. A training example here is a **pair**: the raw record
is the input, the SDTM record is the label. The date is part of the label —
`19/03/2025` has to become `2025-03-19`, a year-only onset has to stay
year-only, `--DY` has to be counted from the reference start. Remove the dates
and the task is gone; keep the real ones and the tier is a Limited Data Set,
which is PHI and cannot feed a corpus.

Shifting is what fits: what the model needs is the format and the mapping
logic, not the calendar. Move both sides of the pair by the **same per-subject
offset** and the correspondence holds exactly — raw day X still maps to SDTM
day X, every interval survives, and no real date survives on either side.

```bash
export DEIDKIT_VAULT_KEY=$(deidkit keygen)

# the SDTM side: --DTC shifted, still valid ISO, still conformant
deidkit profile data/quarantine/sdtm_abc -o contracts/abc_sdtm.yaml \
    --sdtm --blind-treatment \
    --id-template '{STUDYID}-US-{value}'

# the raw side: dates found by their VALUES, shifted in their own format
deidkit profile data/quarantine/raw_abc -o contracts/abc_raw.yaml \
    --raw --blind-treatment \
    --offset-key 'STUDY-001-US-{SUBJECT}'

# both against ONE vault -- this is what makes the offsets agree
deidkit run data/quarantine/sdtm_abc -c contracts/abc_sdtm.yaml \
    -o tiers/sdtm --vault vault/abc.db --operator xli
deidkit run data/quarantine/raw_abc  -c contracts/abc_raw.yaml \
    -o tiers/raw  --vault vault/abc.db --operator xli
```

Three things keep the pair intact, and each fails silently if it is missing:

| | What it does | What breaks without it |
|---|---|---|
| one vault | issues each subject's offset once | the two sides move by unrelated amounts |
| `--offset-key` | rebuilds the SDTM key from the raw columns | raw `SUBJECT` and SDTM `USUBJID` never meet in the vault, so every offset is newly minted |
| `--raw` | shifts in the value's own written form | the raw side arrives pre-normalised to ISO and the conversion the model is meant to learn is already done |

`--id-template` covers the fourth: both sides share one surrogate, so without
it the SDTM identifier equals the raw one and the corpus teaches
`USUBJID = SUBJECT`, which is true of no real study.

Two things this mode refuses to guess. An all-numeric date column where
nothing exceeds 12 (`03/04/2025`) has no readable day/month order, so the run
halts until `date_order` is declared — a wrong order moves every date into the
wrong month and the output still looks like dates. And a value the parser does
not recognise would be published **unshifted**, so by default that halts too;
`on_unparsed: redact` or `pass` makes it a decision on the record instead.

See the whole path end to end, on fabricated data, with the correspondence
checked arithmetically at the end:

```bash
python scripts/demo_pair.py
```

## Field treatments

| Treatment | Use | Analytic cost |
|---|---|---|
| `retain` | Coded clinical content, measurements, MH/AE | None |
| `drop` | Direct identifiers with no analytic value | None |
| `surrogate_id` | Join keys — random, non-derived, vaulted | None; joins preserved |
| `faker` | Direct identifiers whose column must persist | None |
| `date_to_study_day` | All event dates | **None** — intervals preserved |
| `partial_date_to_year_offset` | Frequently-partial dates (MH start) | None; no day imputed |
| `dob_to_age` | Date of birth | Negligible |
| `cap_numeric` | Age — exact below 90, one `90+` band above | Negligible |
| `date_shift` | SDTM output, or where seasonality is analysed | Approximate |
| `date_shift_raw` | The raw side of a raw → SDTM pair; keeps the written format | Approximate |
| `zip3` | Postal geography, low-population prefixes suppressed | Low |
| `generalize_numeric` | Band a quasi-identifier to lift *k* | Moderate |
| `pool_rare` | Low-frequency categories **outside** MH/AE | Minor |
| `screen_freetext` | Verbatim clinical text — **queue, no rewrite** | None |
| `redact_freetext` | Available, not the default for verbatim | High |

## What the contract enforces

Fail-closed, deliberately:

- A column in the data with no rule **halts the run**. Unreviewed columns never
  reach a published tier.
- Two rules emitting the same output column is a validation error (SDTM `DM`
  carries both `BRTHDTC` and `AGE` — the profiler drops the former).
- A domain marked `retained_in_full` **rejects** value-destroying treatments.
  An accepted risk and an unhandled omission must not look alike in the data.

## The vault

- Surrogates are random, never derived from the original.
- The lookup index is an HMAC under a pepper derived from the vault key, so the
  index cannot be used for decryption and does not hold identifiers in clear.
- `reverse()` demands a non-empty justification and writes an access-log row
  before returning. There is no unlogged path to a real identifier.
- A wrong key **refuses to open** the vault rather than silently issuing a
  second, disjoint surrogate space for the same subjects.
- `destroy()` requires an explicit confirmation phrase and leaves a receipt.
  Destroying the crosswalk is what converts pseudonymised data into anonymous
  data — put it on a retention schedule with a date against it.

Deploy the vault file, its key, and the accounts that reach them separately
from every data tier. No principal that can read a tier may read the vault.
Never co-locate the vault in a backup set with a de-identified tier.

## Risk measurement

`deidkit risk` reports *k*-anonymity, *l*-diversity, prosecutor (worst-case)
and marketer (mean) risk over the quasi-identifier set the contract declares,
plus the smallest equivalence classes and a **greedy reduction path** toward the
target.

Read the numbers with the small-trial caveat in mind. On the bundled
120-subject synthetic study, retaining `SITEID`, banded `AGE`, `SEX`, pooled
`RACE`, and `ETHNIC` gives *k*=1 — not a bug, arithmetic: 120 records cannot
fill the class space five quasi-identifiers create. The reduction path says so
plainly and shows that reaching *k*=5 costs three of the five columns.

That is why `fail_on_target_miss` defaults to **false**. Expert Determination
explicitly permits weighing context — internal-only recipients, no egress,
audited access — so a missed threshold is a finding to document as an accepted
risk with its compensating controls named, not something to code around. The
manifest records the miss so it cannot quietly disappear.

## The manifest

Every run writes `manifest.json`: tool version, contract version and rule
fingerprint, input SHA-256 checksums, per-column treatment applied, categories
pooled, dates that could not be converted, free-text screening hit rates, risk
metrics achieved, operator, timestamp.

This is the point of the tool. It is what turns a pipeline run into the
evidence package a determination rests on.

## Tiers

```
quarantine  →  LDS (still PHI, DUA required)  →  de-identified (out of scope)
                                                        ↓
                                            training corpus (internal only)
```

Set `tier: lds` or `tier: deidentified` in the contract; it is recorded in
every manifest. **Never build a training corpus from the LDS tier** — an LDS
remains PHI, and a model trained on PHI is defensibly a PHI-derived asset whose
weights inherit that scope.

Before the corpus: deduplicate and near-deduplicate (repeated exposure is the
largest driver of verbatim memorisation), isolate the training service
principal from the vault, and probe the trained model for regurgitation against
the source. Whether verbatim MH/AE text enters the corpus at all is a separate
decision from retaining it in the analysis tier — it should not inherit by
default.

## Running on a cloud platform

The pipeline is a batch job: it reads a drop, writes tiers, and exits. There is
no service to attack and no long-lived process holding PHI.

```bash
docker build -t deidkit:0.1.0 --build-arg EXTRAS='[all]' \
    --build-arg SPACY_MODEL=en_core_web_lg .

docker run --rm -e DEIDKIT_KEY_URI='awskms:arn:...?blob=/vault/key.blob' \
    -v vault-mount:/vault deidkit:0.1.0 run \
        s3://quarantine/study_abc/ -c /app/contracts/study_abc.yaml \
        -o s3://tier-deid/study_abc/ --vault /vault/study_abc.db
```

The vault key must come from a key service, not an environment variable. The
runtime image sets `DEIDKIT_REQUIRE_MANAGED_KEY=1` so a misconfiguration fails
loudly instead of falling back. Provision the key inside the service so the
plaintext never reaches a shell history:

```bash
deidkit keygen --key-uri 'awskms:<key-arn>?blob=/vault/key.blob'
deidkit keygen --key-uri 'azurekv:https://myvault.vault.azure.net/deid-key'
deidkit keygen --key-uri 'gcpkms:<resource-name>?blob=/vault/key.blob'
```

`compose.yaml` runs the whole flow locally with volumes that model the
three-zone separation: only `pipeline` and `keygen` mount the vault, and the
`analyst` service can reach neither the vault nor quarantine.

**[DEPLOY.md](DEPLOY.md)** covers the per-cloud service mapping, the four IAM
separations that are load-bearing, why the vault needs a real filesystem rather
than object storage, and the pre-flight checklist.

## Development

```bash
python scripts/make_synthetic_study.py out/quarantine/study_demo
python -m pytest tests/ -q
```

The synthetic study is entirely fabricated and plants identifiers of the kind
that turn up in real verbatim fields, so the screen has something to find. It
also includes a single-subject site and a handful of over-89 subjects, so the
risk report and the age cap are both exercised.

The tests assert the guarantees, not just that the code runs: no absolute date
survives any domain, MH/AE clinical columns are byte-identical to the input, AE
durations are recoverable exactly from the study-day columns, surrogates do not
embed the originals, joins survive, screening does not modify data, unreviewed
queue rows are reported rather than assumed, and reverse lookups are always
logged.

## Not legal advice

The Expert Determination route, the Limited Data Set conditions, and the
construction of re-identification codes all turn on judgements that privacy
counsel and a qualified statistician need to make on the specifics of your data
and agreements. This tool implements a design; it does not certify it.

Section references are to 45 CFR Part 164, Subpart E.
