# What changed and why

<!-- One or two sentences. -->

---

## If this PR touches `contracts/`

A contract is the reviewed record of what happens to every column of a real
study's data. Changing it changes what reaches a published tier, so it needs a
human who understands the study, not just the code.

- [ ] A data steward has confirmed every changed rule
- [ ] `contract_version` was bumped — the manifest records it, and a published
      tier must be traceable to the exact rules that produced it
- [ ] No column moved to a treatment that leaves it published as received
      unless that was the intent
- [ ] Nothing was added to a `retained_in_full` domain that destroys values
- [ ] If the quasi-identifier set changed, `deidkit risk` was re-run and the
      new numbers are in the PR description

**Risk numbers after this change:**

```
k = ?    target = ?    records below target = ?
```

## If this PR touches transforms, the vault, or screening

- [ ] `pytest tests/ -q` passes locally
- [ ] `scripts/selfcheck.py` passes against a fresh synthetic run
- [ ] No change makes a surrogate derivable from the original value
- [ ] No change adds an unlogged path to a real identifier
- [ ] No change causes a `screen_freetext` column to be rewritten — that
      treatment publishes data as received by design

## If this PR touches the detector or the allowlist

- [ ] Checked against clinical vocabulary that reads as PHI but is not
      (eponymous conditions, scales, geography in disease names) — a noisy
      queue trains reviewers to click through it
- [ ] Recall did not regress on the planted identifiers in the synthetic study

## Always

- [ ] No real subject data, identifier, site name, or investigator name
      appears anywhere in this diff, including in test fixtures and commit
      messages
- [ ] No vault key, wrapped key blob, or `.db` file is committed
