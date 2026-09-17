# Cross-work authorship verification experiment

This is the next **development experiment**, not a new independent validation. All four
previously evaluated reference manifests are now exposed development sources. Their frozen
results and numerical implementations remain unchanged. No disputed scripture labels are
used as ground truth, and no production author-attribution feature is enabled.

**Measured result:** [the completed experiment](RESULTS.md) fails its preliminary criteria.
None of the ten pair-verifier calibration rounds meets both error limits and useful coverage.

## The change being tested

The earlier models learn a candidate ranking and apply a threshold to the winning score.
This experiment additionally learns what distinguishes a pair of works attributed to the
same author from a pair attributed to different authors. Same-author training pairs always
cross canonical work boundaries. Different-author pairs include writers sharing a genre/topic;
the report records how many such examples exist. These metadata are not predictive features.

The fixed recipe compares function/common-word rates, word endings and descriptive features,
plus character-pattern cosine similarity. It learns a symmetric regularized linear comparison
rule. Vocabulary, IDF, static scaling, pair scaling and the classifier fit verifier-training
authors only. Neither a learned comparison score nor its transformation is a historical
authorship probability. Word endings remain crude proxies, not validated morphology.

Training weights balance positive/negative classes. Negative weight is shared equally between
same-genre and other work-pairs when both exist, so easier cross-genre comparisons cannot
dominate training; a missing stratum is recorded. Within each negative stratum, author-pairs,
then canonical work-pairs, then passage-pairs are balanced. Positive authors, work-pairs and
passage-pairs are likewise balanced. At most eight deterministic
passage-pairs represent each work-pair. Numerous combinations of the same texts do not become
independent evidence. The matching labels guide training sampling/weights, not predictive features.
Total classifier fitting weight equals the number of canonical training works, so generating
more correlated pairs does not inflate the scale used by the fixed regularization parameter.

## Author and work separation

The combined sources contain 44 Greek works from 14 catalogue authors and 48 modern Hebrew
works from 12 authors. Exact source/text duplicates and known canonical-author aliases were
audited; the two Lucian witness editions remain excluded. Unmarked quotation and partial reuse
are still possible. Genre/topic labels differing only by spaces/underscores are normalized.

Each deterministic fold assigns entirely distinct author identities to:

1. Verifier training: all remaining authors with at least two works.
2. Calibration: two candidate authors and two unfamiliar authors.
3. Evaluation: two different candidate authors and two different unfamiliar authors.

Each candidate contributes exactly two canonical reference works. Its other works supply
queries. A reference work never supplies queries; adjacent chunks and alternate editions
cannot substitute for independent works. Low-work-count authors are preferentially unfamiliar
cases, preserving training and reference coverage. Singleton unfamiliar authors remain visible
in the counts; a singleton in the training remainder is explicitly excluded.

Eligible candidate authors rotate through pairs: six Greek folds and four Hebrew folds.
Greek's odd eligible-author count repeats one candidate. Authors/works recur across folds,
so pooled rates average passages within fold/work, then folds within work, works within author,
and authors equally. There is no claim that the folds are independent samples.

These sources support a **two-candidate prototype**. The strict separation leaves too little
training coverage for the intended four-candidate comparison. Any later larger gallery needs
its own calibration and more independent development authors.

## Whole-gallery decisions and the baseline

For each query, the verifier averages its passage-pair scores separately within each reference
work. A candidate's score is the weaker of its two work scores. This requires support across
works without pretending the two comparisons are independent statistical evidence.

Calibration applies to the highest candidate score from the entire two-candidate search.
Threshold choices consider unfamiliar acceptance **and wrong known attribution**. A feasible
threshold must simultaneously keep both error rates at or below 5% and correctly accept at
least 80% of known material. Ties abstain. If no threshold meets all targets, every active
decision abstains; the best error-constrained candidate threshold is saved only as a diagnostic.

The learned feature/model fit and threshold then transfer to a completely different candidate
gallery and unfamiliar authors, with the same gallery size. Test labels never choose a threshold.

For comparison, the original selected recipes (Greek function-word centroid; Hebrew character
linear classifier) use the exact same calibration/evaluation galleries and queries. Each
baseline fits its own reference gallery, then transfers the calibration threshold to the test
gallery. Its independently fitted feature spaces differ from the verifier's common fitted
space. This compares two strategies under the same author roles; it is not an identical-fit
ablation or a direct comparison with old benchmark percentages.

## Controls, evidence and stopping criteria

A fully same-genre experiment requires enough authors and works for **every** role. Greek
forensic oratory does not supply that coverage and must report the control unavailable.
All normalized Hebrew genres already match, so repeating that filter is not an independent
replication. Additional unknown-query diagnostics select texts sharing a genre/topic label
with at least one reference work; those subsets do not establish full topic control.

The preliminary development signal requires all calibration folds to be feasible, every
per-fold and pooled error/coverage target to hold, and the pooled genre-matched unfamiliar subset to
contain at least two authors and remain within the error target. Available per-fold matched
subsets must also meet that error limit. Rejecting everything cannot
pass. A positive development signal would still require fresh confirmation, adequate coverage
and uncertainty estimation before any broader reliability claim. We provide no population
confidence interval for this small, overlapping, exposed development design.

The full protocol, code hashes, runtime, source identities and all work partitions are written
to `development_lock.json` **before fitting any model**. A changed configuration cannot overwrite
that lock. Results contain pair-training work identities/weights, raw gallery scores, separate
reference-work evidence, calibration candidates and actual acceptance decisions.

## Reproduce

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen stylometry verification-study --out output/verification-development-v1 --check
```

Sources are already cached from previous studies. Add `--download` to fetch missing
checksum-pinned development texts; no paid APIs are used. Optional `--language grc` or
`--language hbo` selects one pipeline; use a separate output directory for a different selection.
`--manifest` can be repeated for a separately curated development corpus. The default includes
all four exposed manifests, not any new final-test source.

`--check` exits 1 if any main or eligible matched-control run fails the preliminary criteria,
or a requested language lacks a main run, after writing diagnostics.
Without it, a diagnostic run can finish normally even when empirical targets fail. Status
remains `development_only` even if the preliminary criteria pass. Results from this experiment
must not be labeled scripture attribution, Biblical Hebrew evidence, Arabic validation, or
independent validation on untouched authors.
