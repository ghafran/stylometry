# Real-text reliability evaluation

This pilot asks whether the current lexical features carry author information across separate
works, and whether a fixed decision rule can reject authors it has never seen. It also exercises
the actual application on single-author, mixed-author and same-genre inputs. The
[measured results](RESULTS.md) fail the acceptance criteria. These are useful failure measurements,
not proof of historical or scriptural authorship.

## Reproduce

From the repository root:

```bash
uv sync --frozen
uv run --frozen stylometry benchmark --download
uv run --frozen pytest -q
uv run --frozen pytest -q --run-empirical tests/test_empirical_acceptance.py
```

The benchmark command downloads checksum-pinned public reference texts into the ignored
`data/raw/benchmarks/` cache. Subsequent benchmark runs work offline. No AI profiles or paid APIs
are used. The regular software suite skips the empirical acceptance test; the final command
recomputes the real-author benchmark and **currently fails** because the reliability claim is
not met. `stylometry benchmark --check` also returns status 1 for a failed or inconclusive result,
after saving all diagnostic artifacts. Omitting `--check` permits a diagnostic run to complete
without requiring empirical success.

The main output is `output/benchmark/report.md`, with detailed attribution results and individual
predictions in `output/benchmark/attribution/`. `suite.json` records source-manifest hashes,
implementation hashes, package versions, protocol, work partitions, thresholds and controls.
The checked-in `results.json` is a compact snapshot; raw texts and the large prediction files
are recreated locally. `--language grc` or `--language hbo` selects a pipeline, `--no-controls`
skips app/perturbation checks, and repeatable `--manifest` supplies custom reference manifests.
Unavailable languages remain unvalidated.

## Sources and coverage

| Pipeline | Actual source language | Authors | Separate attribution works | Normalized tokens |
|---|---|---:|---:|---:|
| Greek (`grc`) | Ancient Greek | 6 | 18 | 199,918 |
| Hebrew (`hbo`) | Modern Hebrew (`heb`) | 4 | 16 | 107,754 |
| Arabic (`arb`) | Not acquired | 0 | 0 | 0 |

[greek_manifest.json](greek_manifest.json) pins the
[Perseus Greek corpus](https://github.com/PerseusDL/canonical-greekLit/tree/9839fe84c883e47ca5dab8872d8c684d9e14fe14)
at an immutable revision, with source and normalized-text SHA-256 values, edition metadata,
licenses and curation notes. Credit: the Perseus Digital Library and named source editors;
the repository is licensed CC BY-SA 4.0. Selected authors are Lysias, Isocrates, Demosthenes,
Plato, Xenophon and Lucian. Three orators provide a same-genre comparison; Plato and Xenophon
provide overlapping Socratic material. Attribution labels are conventional catalogue labels,
not independently established historical facts. Two editions of Lucian's *How to Write History*
are separate controls, excluded from attribution training and testing.

[additional_manifest.json](additional_manifest.json) pins the Project Ben-Yehuda public-domain
dump, with 16 original prose works by four modern Hebrew writers. See
[ADDITIONAL_SOURCES.md](ADDITIONAL_SOURCES.md) for selection, attribution, licensing, extraction
and the Arabic coverage gap. The `hbo` setting selects the app's Hebrew normalizer; the source
language is explicitly recorded as modern Hebrew. These texts cannot validate Biblical Hebrew.

The loaders verify raw bytes before parsing, remove identified editorial metadata, then verify
normalized text and token counts. Names, topics, dialogue and unmarked quotations within the
works can remain. An editor, period, genre or subject can still correlate with author labels.
No disputed scriptural attributions are used as ground truth.

## Fixed evaluation protocol

Source selection, modeling choices and thresholds were fixed before inspecting scores. A final
audit corrected grouping to use canonical work IDs instead of display titles, preventing alternate
editions supplied through different manifests from crossing partitions. That correction is
explicitly versioned as `real-author-pilot-v2-canonical-work-id`; final results were recomputed.
Thresholds and model choices were not adjusted to improve the scores. This was an internal
predeclared protocol, not an externally registered study.

- Split each work into nonoverlapping 100-, 500- and 1,000-token units; retain at most 12 evenly
  spaced units per work. The cap limits length imbalance; author centroids and reported rates
  weight works equally. Feature fitting and calibration still use the retained units.
- Rank canonical work IDs by a seeded hash within each author and assign them to three buckets.
  Rotate entire buckets through training, calibration and testing. Every test work is absent
  from both fitting and calibration. At least three independent works per author are needed.
- Fit character vocabulary, IDF, SVD, scaling and author centroids on training works only.
  Each work has equal weight in its author's centroid. Predict by cosine similarity.
- Evaluate predefined function/common words and the full lexical feature set separately.
  The former includes some common content verbs and must not be described as topic-free.
- Calibrate a verification threshold permitting at most 5% wrong-author matches on calibration
  data. Independently calibrate rejection to retain at least 95% of genuine calibration matches.
  Test every author in turn as wholly unseen, excluded from training and calibration.
- Report balanced accuracy, verification AUROC, false matches, missed same-author matches,
  unseen-author false acceptance, known-author retention, per-author errors and metadata subsets.
  Candidate scores and raw predictions remain auditable; they are not authorship probabilities.

Accuracy and error rates average within work, then within author. Conditional descriptive 95%
intervals resample whole works with predictions fixed. They do not refit the model or model
dependencies through shared training works and candidate authors. Boundary guards prevent zero
observed errors from implying zero uncertainty. AUROC intervals are unavailable when a
fold/author stratum has fewer than two works. These intervals do not quantify historical uncertainty.

Acceptance requires accuracy ≥80%, verification false matches ≤5%, missed same-author matches
≤20%, unseen-author acceptance ≤5%, and known-author retention ≥80%. The entire interval must
clear each target. Overall passing also requires at least eight authors and 60 independent works
in each evaluated language and length. A small corpus can demonstrate failures but cannot pass
the broader coverage requirement. Changing defaults marks a run exploratory and prevents a pass.

## Production and robustness controls

The reference classifier is a supervised diagnostic built from the app's lexical features.
Separate controls invoke the actual unsupervised clustering pipeline on 25- and 500-token units:
mixed authors with smoothing off/on, one author at a time, and shared genres. Scores compare
returned partitions with withheld catalogue labels. A style split within a writer can be real;
it does not prove multiple writers. Conversely, one group does not establish one writer.

Separate held-out works undergo spelling normalization, random deletion of 2% of words and
replacement of 20% with another author's text. The feature map remains fixed. These synthetic
perturbations measure prediction sensitivity, not manuscript authenticity. The two Lucian editions
provide one real same-work edition check; neither is included in reference training. Translation,
broad manuscript variation, AI-generated profiles and scripture transfer remain untested.

## What the failures mean

The app can help explore stylistic similarities and differences. This pilot does not justify
unique-person attribution, author counts, or claims that scriptural passages have a named author.
The current rejection rule is especially unreliable when the true author is absent. More data
alone is not evidence of improvement: future changes need new authors/works reserved as a fresh
final test set, matched controls and a rejection rule calibrated without seeing those final authors.
The current benchmark is now exposed evaluation data and must not be treated as untouched after
tuning. Preserve the measured failures instead of loosening the gates to obtain a green result.

See the [improvement plan](IMPROVEMENT_PLAN.md) and [first rejection experiment](REJECTION_DEVELOPMENT.md).
Run `uv run --frozen stylometry benchmark-rejection` to reproduce the development comparison.
The original protocol, acceptance test and results snapshot remain unchanged.

The follow-up [passage-model study](study_v1/README.md) implements larger passages, fixed model
comparisons, separate unfamiliar calibration authors and a locked fresh-author evaluation.
Its [results](study_v1/RESULTS.md) also fail the empirical acceptance gate. It is a separate
study; its different author panel must not be used as a direct before/after improvement claim.

The subsequent [pair-verification development experiment](verification_v1/README.md) learns
cross-work comparisons on the now-exposed combined corpus, with separate training, calibration
and evaluation author identities. Previous benchmark snapshots stay frozen.
