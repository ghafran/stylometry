# Passage models and fresh-author evaluation

This study implements the four proposed changes. **Its empirical acceptance gate still fails.**
The [results](RESULTS.md) distinguish identifying a writer from a known candidate set from
deciding whether the true writer is present at all. Neither is historical proof of authorship.

## What was implemented

1. **Raw-token passages.** `stylometry cluster-passages` extracts features after pooling
   500, 1,000 or 2,000 normalized words. Passages do not overlap or cross observed work,
   chapter, witness, source, author-label or gap boundaries. Gapped/supplied verses are
   excluded because internal token boundaries are unknown. Every retained token maps to
   an original source verse and token offset; short tails and insufficient passages are reported.
   No AI profiles or additional smoothing are used. Reports label the output as whole passages.
2. **Three fixed model recipes.** Function/common-word cosine centroids; centroids with word
   endings, descriptive features and training-observed adjacent function-word pairs; and
   character 3–5-gram TF-IDF with a regularized linear classifier. The latter uses fixed C=1,
   not a hyperparameter search. Word endings and order are linguistic proxies, not validated
   POS or dependency parsing. Vocabulary, IDF, scaling, bigrams and models fit training works
   only. Training mass balances authors and canonical works. Scores are not probabilities.
3. **Matched controls and multiple unfamiliar authors.** Models are fitted separately on
   qualifying shared genres/topics. Development rejection reserves two unfamiliar calibration
   authors and a separate outer unknown author. The old four-author Hebrew development panel
   is too small for this split and explicitly reports it unavailable. The fresh Hebrew panel
   supplies enough authors to exercise that protocol.
4. **A locked, fresh evaluation.** Development compares all three models at all three passage
   lengths. Selection first prefers useful rejection at the declared targets, then chooses
   matched-genre accuracy (overall accuracy only when no genre control is available).
   Ties use fixed model order, then shorter passages. The design, code hashes, runtime and
   source manifests are locked before scoring reserved authors. Mutations invalidate the lock.

The baseline files and original failing empirical test remain unchanged. This is a separate
study, not a replacement score for the previous benchmark.

## Reproduce

```bash
uv sync --frozen
uv run --frozen pytest -q

uv run --frozen stylometry author-study develop --download --out output/author-study-v1
uv run --frozen stylometry author-study seal \
  --model-lock output/author-study-v1/model_lock.json \
  --manifest benchmarks/greek_fresh_manifest.json \
  --manifest benchmarks/hebrew_fresh_manifest.json \
  --out output/author-study-v1/fresh
uv run --frozen stylometry author-study evaluate --download \
  --evaluation-lock output/author-study-v1/fresh/evaluation_lock.json \
  --out output/author-study-v1/fresh --check
```

The final command currently exits 1 after writing diagnostics. Source downloads are cached and
checksum-verified; subsequent runs are offline. No paid APIs are called. Fresh author sources
are not loaded by the development stage. `seal` inspects metadata, author/work roles and manifest
hashes; `evaluate` verifies them before loading and scoring. The curated author identities were
also independently audited using CTS textgroups and Hebrew publisher/authority identifiers;
the generic code uses manifest labels, so custom corpora still require alias/identity curation.

An existing study directory refuses a different model lock. Code or dependency changes require
a new study directory and new validation protocol. Re-running this exact locked configuration
is a reproduction of exposed data, not another independent test.

For the actual application, after building a source corpus:

```bash
uv run --frozen stylometry cluster-passages --language grc --tokens 1000
uv run --frozen stylometry html --out output/passages/grc
```

Optional `--corpus PATH --scope all` uses a separate source JSONL. The default output is
`output/passages/<language>/`, preserving verse-analysis outputs. `passages.json` contains
the source mapping and exclusions. Fewer than three complete passages stops clustering but
still writes the mapping. A demonstration on the exposed Greek reference works produced
191 complete 1,000-token passages and was exercised through clustering, Markdown and HTML.
Those reference works are flattened editions without retained chapter divisions; the demonstration
protects work boundaries but cannot reconstruct missing chapter metadata. Scripture source
records with chapter/witness metadata retain those boundaries.

## Evaluation and uncertainty

The [source audit](../FRESH_SOURCES.md) records 58 new whole works and eight new catalogue authors
per language. Each panel has four reference authors, two unfamiliar calibration authors and
two final unfamiliar authors. Reference works are assigned to disjoint training, calibration
and testing buckets using canonical IDs. Final unknown authors enter neither fitting nor
calibration. Different editions, internal books and chunks do not count as new authors/works.

The rejection threshold controls the maximum candidate score on unfamiliar calibration authors.
It requires ≤5% calibration false acceptance and ≥80% known calibration acceptance; otherwise
it abstains from every text. The final gate additionally measures nearest-candidate accuracy,
correct known acceptance and wrong known acceptance. Rejecting everything cannot pass.
Observed target misses fail this test panel; adequate confidence and corpus coverage are also
needed before a passing result is possible.

Forty bootstrap repetitions refit features, scaling, models and calibration, resampling whole
works within roles and unfamiliar authors as blocks. Known candidate identities remain fixed.
This is a rough pilot, not calibrated population uncertainty. A singleton author/role work
stratum makes confidence intervals unavailable; diagnostic resampling ranges are retained.
Zero-event unfamiliar-author guards count authors rather than multiplying evidence by their
numbers of works or chunks. Two unfamiliar authors cannot establish a universal 5% risk bound.
Broad passing requires much greater reference and unfamiliar-author coverage than these panels.

## Limits and remaining research

The selected Greek model uses function/common words; the Hebrew model uses character patterns.
Both selected 500 tokens from development, so larger passages cannot be assumed to perform
better. The fresh figures use different authors and candidate counts than earlier evaluations;
they are not a direct before/after improvement estimate.

Matched genres/topics are broad catalogue/manual labels and do not eliminate editorial, period,
translation or subject confounding. Greek includes conventional ancient attributions; Hebrew is
modern Hebrew. Ancient-language syntax tagging, a broader independently reviewed corpus,
translations/manuscript variants, Biblical Hebrew, Arabic and scripture transfer remain unvalidated.
Further model or threshold changes need newly reserved final authors and works. The completed
study supplies a reproducible way to measure that work without silently redefining success.
