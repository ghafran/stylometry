# Cross-work verification: measured results

**The experiment is implemented, but this verifier does not solve reliable attribution.**
It fails the development criteria in both languages and should not advance to a new final
confirmation study in its present form. Scripture attribution remains unvalidated.

The full software suite reports **522 passed, 1 skipped**, including 99 new verifier, gallery,
study-isolation and command tests. The skipped test is the original opt-in empirical benchmark.
The new real-text experiment ran to completion with `--check` and returned **exit status 1**
because its empirical development criteria were not met.

## What was implemented and evaluated

- A learned, symmetric same-author/different-author comparison using pairs of independent works.
- Training-only vocabulary and scaling, with authors separated before any feature fitting or pairing.
- Balanced positive/negative evidence, explicit same-genre negative weighting, and regularization
  scaled to the number of canonical training works rather than correlated passage-pair counts.
- Two independent reference works per candidate, with acceptance requiring support from both.
- Calibration of the complete candidate search, controlling wrong known assignments as well as
  unfamiliar-author acceptance, with abstention when useful correct acceptance is insufficient.
- A comparison against the earlier fixed model recipes on exactly the same panels and queries.
- Source/work/author alias checks, pre-fit code/data/runtime locks and full decision evidence.

The exposed combined corpus contains **92 works from 26 catalogue authors**, yielding 649
sampled, nonoverlapping 500-token passages. Each fold uses two candidate authors for calibration
and two different candidate authors for evaluation, plus two unfamiliar authors at each stage.
The verifier trains on six other Greek authors or four other Hebrew authors. Six Greek and
four Hebrew folds rotate those roles. Every candidate has two reference works and separate query
works; training, calibration and evaluation author groups never overlap within a fold.

## Results on the same development panels

| Language | Method | Accuracy among two known candidates | Correct known acceptance | Unfamiliar acceptance | Feasible calibration rounds |
|---|---|---:|---:|---:|---:|
| Greek | New pair verifier | 92.2% | **0.0%** | 0.0% | **0 / 6** |
| Greek | Fixed centroid baseline | 95.6% | 9.1% | 4.7% | 1 / 6 |
| Modern Hebrew | New pair verifier | 77.2% | **0.0%** | 0.0% | **0 / 4** |
| Modern Hebrew | Fixed character baseline | 93.9% | 0.0% | 0.0% | 0 / 4 |

Accuracy is a diagnostic before rejection, conditional on the writer being in a two-candidate
gallery. It is not general attribution accuracy. Rates average within fold/work, then across
folds within work, works within author, and authors equally. The earlier four-candidate fresh
study has different conditions; its percentages are not a direct comparison for this table.

At thresholds meeting both 5% calibration error limits, the pair verifier correctly retains
only **6.3–62.5%** of known Greek calibration material across rounds and **4.9–30.5%** of known
Hebrew material. Every round misses the required 80%. These candidate thresholds are therefore
inactive: the verifier abstains on **all 572 evaluation decisions** (401 distinct passages
across repeated panels). Zero false acceptance is accompanied by zero useful acceptance;
accuracy among accepted passages is undefined.

The Greek baseline's pooled 4.7% unfamiliar acceptance also must not be read as a passing result.
Its one feasible calibration round accepts **34.0% of unfamiliar material in that round's
evaluation**, while correctly accepting only 75.0% of known material. Pooling it with abstaining
rounds hides that failure. The per-round gate correctly rejects the overall result.

The new verifier has lower known-candidate ranking accuracy than the comparator in both
languages on these panels. There is no measured improvement here that justifies promoting it.
The comparison tests complete strategies: the baseline independently fits each gallery before
transferring a threshold; the verifier retains one training-only feature/model fit for both
galleries. It does not isolate one feature family's causal effect.

## Controls and interpretation

Greek training folds include 27–54 same-genre negative canonical work-pairs. They receive half
the negative training weight despite many more cross-genre pairs. Hebrew training negatives
all share its normalized broad literary-prose label, so the absent other-genre stratum is
explicitly recorded. Matching metadata never enter the predictive feature vector.

A full Greek same-genre experiment lacks enough separate authors for all required roles and
is reported unavailable. Its main candidate pairs span different genres; ranking accuracy
can still reflect genre. The narrower Greek unfamiliar-query genre subset includes three
authors and six works, while its topic subset includes only one author and two works.
Hebrew's broad genre/topic unfamiliar subsets contain four authors and eight works. The pair
verifier rejects all of these too. They do not establish successful topic-independent attribution.

This result shows that the implemented comparison model and decision statistic do not separate
known from unfamiliar material well enough under the specified targets. It does not establish
which of limited training-author coverage, coarse features, cross-author score transfer or the
conservative reference aggregation contributes most. More data alone is not a demonstrated fix.

The next development priority is a broader, independently curated set of comparable writers
with multiple secure works per author, followed by controlled representation/aggregation
comparisons within that development pool. The present result is a reason to keep researching
the representation and rejection tradeoff, not to loosen thresholds or spend another untouched
final corpus on this unchanged model.

## Evidence and scope

The protocol and source/model hashes were sealed at **2026-09-17 01:52:24 UTC**; the run completed
at **01:54:24 UTC**. No numerical code or design was changed after those results. Both earlier
studies' numerical source checksums still match their frozen snapshots.

- [Design and reproduction instructions](README.md)
- [Generated development report](development.md)
- [Compact machine-readable results](development.json)
- [Exact full results archive, including predictions and reference-work evidence](development.full.json.gz)
- [Pre-fit protocol, source identities, runtime and partitions](development_lock.json)

The archive decompresses to the exact original full JSON. Its compressed and uncompressed
SHA-256 values are recorded in the compact snapshot. Raw texts remain in the checksum-verified,
ignored source cache and are reconstructed by the documented loaders.

These are exposed-data development results from small, overlapping folds. No population
confidence interval is asserted, and two-candidate searches cannot validate larger searches.
Conventional source attributions, unmarked quotation, editorial effects, period and register
remain limitations. Modern Hebrew is not Biblical Hebrew; scripture transfer, Arabic and named
historical author identities remain unvalidated. The application continues to offer exploratory
style analysis; this experimental verifier is not a validated attribution feature.
