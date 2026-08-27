# Deploying deidkit on a cloud platform

The pipeline is a batch job, not a service. It reads a drop, writes tiers, and
exits. That makes it cheap to run and easy to isolate: there is no long-lived
process holding PHI in memory and no endpoint to attack.

What actually matters in a cloud deployment is not the compute — it is the
**boundaries**. Collapse them and the transformations stop meaning anything.

## The boundaries that are load-bearing

Four separations. Each one, if collapsed, breaks the design in a way the data
will not reveal:

| Separation | Collapsed how | What breaks |
|---|---|---|
| Quarantine ≠ published tiers | one bucket with prefixes, one role | An analyst who can read a prefix can read the raw drop. The transformations become advisory. |
| Vault ≠ every data tier | pipeline role also readable by analysis | Anyone who can read a tier can re-identify it. The output is no longer pseudonymised in any meaningful sense. |
| LDS tier ≠ de-identified tier | one "processed" bucket | A training corpus gets built from PHI, and the model weights inherit HIPAA scope. |
| Vault key ≠ vault storage | key file on the same volume | A single compromised volume yields both halves. Envelope encryption exists precisely to prevent this. |

The compose file in this repo encodes the same shape locally, which is the
point of it: only `pipeline` and `keygen` mount the vault volume, and the
`analyst` service can reach neither the vault nor quarantine. If a change makes
`analyst` work when it shouldn't, the isolation regressed.

## Service mapping

The job is a container that runs to completion, so any of these work:

| Role | AWS | Azure | GCP |
|---|---|---|---|
| Compute | ECS Fargate task, or Batch | Container Apps Job, or Container Instances | Cloud Run Job |
| Quarantine | S3 bucket, own KMS key | Blob container, own key | GCS bucket, CMEK |
| Tiers | separate S3 buckets per tier | separate containers | separate buckets |
| Vault | EFS or a small EBS volume | Azure Files | Filestore or a PD |
| Vault key | KMS CMK (envelope) | Key Vault secret | Cloud KMS (envelope) |
| Identity | task role per zone | managed identity per job | service account per job |
| Audit | CloudTrail (KMS Decrypt) | Key Vault diagnostics | Cloud Audit Logs |

All three providers cover the relevant storage, KMS and container services
under their HIPAA/BAA-eligible service lists, but **confirm current eligibility
against the provider's own list before you commit** — the lists change, and a
service being available in your region is not the same as being covered by your
agreement. You need a signed BAA in place regardless of which services you pick.

### Why a filesystem for the vault, not object storage

The vault is SQLite, which needs POSIX file locking. Object storage does not
provide it, and two concurrent runs against an S3-backed vault would corrupt
it. Mount a real filesystem, and run the pipeline **single-writer** — one job
at a time per study. If you outgrow that, replace the vault backend with a
managed database (Postgres with the same encrypted-at-rest columns); the
`Vault` class is the only thing that would change.

## The vault key

Never an environment variable in production. The runtime image sets
`DEIDKIT_REQUIRE_MANAGED_KEY=1` so a misconfiguration fails loudly rather than
falling back to one.

Provision the key **inside the key service**, so the plaintext never reaches a
terminal or a shell history:

```bash
# AWS -- envelope: a Fernet data key wrapped under your CMK. The wrapped blob
# is safe beside the vault, because opening it needs kms:Decrypt on the CMK.
deidkit keygen --key-uri 'awskms:arn:aws:kms:us-east-1:123456789012:key/abcd?blob=/vault/key.blob'

# Azure -- the secret lives in Key Vault; nothing on disk.
deidkit keygen --key-uri 'azurekv:https://myvault.vault.azure.net/deid-vault-key'

# GCP
deidkit keygen --key-uri 'gcpkms:projects/p/locations/us/keyRings/r/cryptoKeys/k?blob=/vault/key.blob'
```

Then point runs at the same URI via `DEIDKIT_KEY_URI`. The manifest records
which key source was used, so a run made with a development key is identifiable
after the fact.

Two consequences worth stating plainly:

- **Losing IAM access to the KMS key locks the vault as thoroughly as losing
  the key.** Enable deletion protection on the CMK, and write down the recovery
  path before you need it.
- **`deidkit keygen` refuses to overwrite an existing key.** Overwriting would
  orphan every surrogate already issued, and no subject in any published tier
  could ever be re-identified again — including for a safety report.

## IAM: the minimum that actually holds

Three principals. The pipeline is the only one that touches more than one zone.

```
pipeline-role
  s3:GetObject           on  quarantine/*
  s3:PutObject           on  tier-lds/*, tier-deid/*
  kms:Decrypt            on  vault-cmk, quarantine-cmk
  (filesystem)           on  the vault mount
  NO s3:PutObject on quarantine  -- nothing writes back to a drop

analyst-role
  s3:GetObject           on  tier-lds/*   (under a DUA)
  NO access to quarantine, the vault, the vault CMK, or tier-deid

training-role
  s3:GetObject           on  tier-deid/*
  NO access to quarantine, tier-lds, the vault, or the vault CMK
```

`training-role` being barred from `tier-lds` is not belt-and-braces. An LDS
remains PHI; a model trained on it is defensibly a PHI-derived asset. Enforcing
that in IAM means the mistake cannot be made by editing a config file.

Add an explicit `Deny` on the vault CMK for every principal except
`pipeline-role`. Allow-lists drift as roles accumulate policies; a deny does not.

## Deploying

```bash
# build (include Presidio and the cloud backends)
docker build -t deidkit:0.1.0 \
    --build-arg EXTRAS='[all]' \
    --build-arg SPACY_MODEL=en_core_web_lg .

# a run, reading and writing object storage directly
docker run --rm \
    -e DEIDKIT_KEY_URI='awskms:arn:...?blob=/vault/key.blob' \
    -e AWS_REGION=us-east-1 \
    -v vault-mount:/vault \
    deidkit:0.1.0 run \
        s3://quarantine/study_abc/ \
        -c /app/contracts/study_abc.yaml \
        -o s3://tier-deid/study_abc/ \
        --vault /vault/study_abc.db \
        --operator "$JOB_ID"
```

The default build is core-only and small; the free-text screen falls back to
the built-in pattern detector. That fallback is deliberate so the pipeline runs
anywhere, which also means **a missing model looks like success unless you
check**. Confirm which detector is live:

```bash
docker run --rm deidkit:0.1.0 \
    python -c "from deidkit.freetext import build_detector; print(build_detector().name)"
```

`presidio` is what you want in production. `pattern` means the model is absent.

## Contracts belong in version control, not in the bucket

The contract is the reviewed record of what happens to every column. Bake it
into the image or pull it from your Git provider at job start — do not read it
from a bucket an operator can edit, or the review step becomes decorative. The
manifest records `contract_version`, so a published tier can always be traced
to the exact reviewed rules that produced it.

## Logging: the one thing that will leak

Application logs are the most likely PHI leak in a cloud deployment, because
they are shipped somewhere central by default and retained for a long time.

- Set the log driver to capture the pipeline's own stdout only, and never
  enable row-level or SQL debug logging in an environment that sees real data.
- Turn off exception reporters that capture local variables (Sentry's default
  does). A stack trace through a transform holds column values.
- Check that failed-load reject files land in quarantine, not in a shared
  temp bucket.
- Give quarantine a short lifecycle policy, and confirm the bucket's own
  access logs are not written into a bucket with wider read access.

The pipeline writes checksums, row counts, treatment names and risk metrics.
It does not log values. Keep it that way.

## Cost

The compute is negligible: a 120-subject study runs in under a second, and a
large multi-study drop is minutes of one CPU. Expect the bill to be dominated
by storage and the KMS key, not the job — order of tens of dollars a month for
a handful of studies, before the spaCy model's image-registry footprint.

This is worth saying to a budget holder because it reframes the decision: the
cost of this pipeline is the **review effort** — the steward confirming rules,
the data manager adjudicating the free-text queue, the statistician writing the
determination — not the infrastructure.

## Before the first real run

- [ ] BAA signed, and every service used confirmed against the provider's
      current HIPAA-eligible list
- [ ] The incoming data agreement checked for its secondary-use and
      model-training clause — this is the actual gate, and no deployment
      decision resolves a contractual restriction
- [ ] Three separate buckets, three separate roles, explicit deny on the
      vault CMK
- [ ] Key provisioned in the KMS, deletion protection on, recovery path
      documented
- [ ] `DEIDKIT_REQUIRE_MANAGED_KEY=1` confirmed in the deployed task definition
- [ ] Detector confirmed as `presidio`, not `pattern`
- [ ] Log driver reviewed; variable-capturing error reporters disabled
- [ ] Crosswalk destruction date on the retention schedule
- [ ] A dry run on synthetic data in the real environment, with
      `scripts/selfcheck.py` passing

## Not legal advice

Service eligibility, the Expert Determination route, and the construction of
re-identification codes all turn on judgements your privacy counsel and a
qualified statistician need to make on your specific data and agreements. This
document describes how to deploy a tool; it does not certify a deployment.
