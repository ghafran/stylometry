# Fresh-author study results

**The implementation checks pass; the empirical reliability gate fails. Scripture attribution remains unvalidated.**

All four proposed steps are implemented: pooled passages, three model families with matched
genre/topic controls, multiple unfamiliar calibration authors, and a locked fresh-author test.
The software suite reports **423 passed, 1 skipped**. The skipped test is the original opt-in
empirical benchmark. The new fresh evaluation was actually run with `--check` and returned
status 1 because its reliability targets were not met.

## What the fresh test found

The new sources contain **58 whole works from 16 catalogue authors**, none of those authors
in the earlier development corpus. Each language has four candidate authors, two unfamiliar
calibration authors, and two separate final unfamiliar authors. Whole works, not passages,
are assigned to disjoint training, calibration and testing roles.

| Measure | Greek | Modern Hebrew |
|---|---:|---:|
| Selected model | Function/common-word centroid | Character-pattern linear classifier |
| Selected passage length | 500 tokens | 500 tokens |
| Accuracy when the author is among four known candidates | **88.5%** | **90.6%** |
| Known test works / passages | 6 / 46 | 8 / 59 |
| Final unfamiliar authors / works / passages | 2 / 3 / 27 | 2 / 4 / 45 |
| Calibration false acceptance at the candidate threshold | 4.2% | 4.2% |
| Known calibration material retained at that threshold | **30.3%** | **42.7%** |
| Known final material accepted by the active rule | **0.0%** | **0.0%** |
| Unfamiliar final material accepted by the active rule | **0.0%** | **0.0%** |
| Empirical result | **Failed** | **Failed** |

Rates weight authors equally, then works equally within authors, then passages within works.
They are not raw passage-count percentages or authorship probabilities.

The calibrated rule needs both ≤5% unfamiliar-author false acceptance and ≥80% known-author
acceptance. Neither language has a threshold meeting both targets. Its active behavior therefore
abstains on every final passage. Zero false acceptance here reflects zero useful acceptance;
it does not demonstrate a reliable attribution service. Accuracy among accepted passages is
undefined because none were accepted. These models remain reference-corpus experiments, not a
production scripture attribution feature.

Greek candidate calibration acceptance includes 28.2% correct and 2.1% wrong assignments.
Hebrew candidate calibration acceptance is 42.7% correct and 0.0% wrong. These are diagnostics
for a rejected threshold, not final-test claims or a threshold deployed by the app.

## Controls and uncertainty

Refitting the same frozen Greek design only on matched forensic oratory gives **95.8%**
accuracy, but this is a two-candidate comparison on three test works. Modern Hebrew's broad
literary-prose and social/psychological-fiction controls both give **90.6%**, using the same
four authors and eight test works as its main panel. Those two controls are not independent
replications and broad labels do not remove topic, period, editor or register effects.

Forty whole-work bootstrap repetitions refit the complete feature, model and calibration
pipeline, with unfamiliar authors sampled as blocks. Greek confidence intervals are unavailable
because several author/role strata have only one work. The Hebrew accuracy interval is roughly
**51.2–98.3%**, conditional on this candidate panel and this small resampling design. These are
rough pilot estimates, not calibrated population guarantees. With only two final unfamiliar
authors per language, neither panel can establish a general 5% unfamiliar-author risk limit.
Both also fall below the predeclared coverage requirement for a broad passing claim.

The earlier benchmark used different authors, candidate counts and designs. Comparing its
82.6%/69.3% accuracy directly with these 88.5%/90.6% figures would not establish improvement.
Modern Hebrew results do not validate Biblical Hebrew; edited literary Greek does not validate
scripture transfer. Named historical attribution and historical author counts remain unproven.

## Frozen evidence and reproduction

Model selection used only the old development corpus. The fresh manifest and model lock were
sealed at **2026-09-17 01:17:01 UTC**; evaluation completed at **01:18:57 UTC**. The source
manifests, numerical model code and numerical-library versions match the recorded locks.
No model or threshold was tuned after these fresh results. This corpus is now **exposed
evaluation data**; a future revised design needs another reserved final panel.

- [Reproduction instructions and passage command](README.md)
- [Development comparison](development.md) and [compact development data](development.json)
- [Frozen model and protocol](model_lock.json) and [original evaluation lock](evaluation_lock.json)
- [Full final predictions, partitions, calibration and controls](fresh_evaluation.json)
- [Source, rights and author-identity audit](../FRESH_SOURCES.md)

The evaluation lock retains the original machine's manifest paths for exact provenance. Use
the documented `seal` command to create a lock with local paths when reproducing elsewhere;
the new timestamp/path-specific lock digest will differ. Full development predictions are
recreated by the development command; their original file checksum is recorded in the compact
snapshot. Raw source files remain in the ignored checksum-verified cache.

The passage command was also exercised on the exposed Greek reference corpus: 191 complete
1,000-token passages were clustered and rendered in Markdown and HTML, with source-token
mappings and exclusions retained. That demonstrates the application path works; its clusters
do not prove author identities or counts.

The remaining scientific problem is useful unfamiliar-author rejection. Further research must
improve retained correct attributions while preserving a low false-match rate, then confirm
that result on substantially broader untouched authors and scripture-relevant controls.
