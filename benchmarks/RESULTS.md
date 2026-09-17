# Measured reliability: failed

Run: 2026-09-17T00:27:15.495134+00:00. Protocol: `real-author-pilot-v2-canonical-work-id`.

**The current evidence does not establish reliable unique-author or scripture attribution.**

The software suite passed **301 tests** and skipped the opt-in empirical test. The separately run empirical acceptance test **failed**; the CLI `--check` gate also returned status 1. Both independent executions produced identical metrics and work partitions. The reliability requirements were not weakened to make these checks pass.

The reference corpus contains 34 attribution works by ten catalogued authors: 18 ancient Greek works by six authors and 16 modern Hebrew works by four authors. Two editions of another Greek work provide a separate control. Source files and normalized texts are checksum-verified.

## All held-out results

Accuracy is balanced across authors and works. Verification compares a text with a candidate author; unseen-author acceptance measures the separate rejection rule when the true author is absent. These are different decisions with different calibration thresholds. Values are point estimates; conditional descriptive intervals, work assignments, per-author errors and metadata subsets are retained in [results.json](results.json).

| Language | Tokens | Features | Author accuracy | Wrong-author matches | Missed same-author matches | Unseen authors accepted |
|---|---:|---|---:|---:|---:|---:|
| Greek | 100 | Function/common words | 44.0% | 4.2% | 73.1% | 100.0% |
| Greek | 100 | Full lexical | 48.1% | 3.8% | 69.4% | 100.0% |
| Modern Hebrew | 100 | Function/common words | 41.1% | 6.1% | 82.3% | 100.0% |
| Modern Hebrew | 100 | Full lexical | 46.4% | 4.7% | 78.6% | 100.0% |
| Greek | 500 | Function/common words | 75.6% | 6.0% | 30.2% | 99.5% |
| Greek | 500 | Full lexical | 70.8% | 5.8% | 31.4% | 98.4% |
| Modern Hebrew | 500 | Function/common words | 59.0% | 3.7% | 67.7% | 100.0% |
| Modern Hebrew | 500 | Full lexical | 62.7% | 2.1% | 70.3% | 98.4% |
| Greek | 1000 | Function/common words | 85.0% | 8.0% | 20.8% | 90.5% |
| Greek | 1000 | Full lexical | 82.6% | 7.3% | 24.2% | 92.4% |
| Modern Hebrew | 1000 | Function/common words | 65.8% | 3.0% | 69.2% | 97.9% |
| Modern Hebrew | 1000 | Full lexical | 69.3% | 3.2% | 73.0% | 96.9% |

Every configuration failed the empirical gates. The unknown-author rule falsely accepted 90.5–100% of unseen-author material across configurations, far above the 5% target. The apparent strength of longer Greek passages therefore does not justify using this rule to assign an unknown text to a known writer.

At 1,000 tokens, the full lexical Greek accuracy is 82.6% with a conditional descriptive interval of 76.2–89.8%; modern Hebrew accuracy is 69.3% with an interval of 54.7–82.3%. These intervals resample fixed predictions, do not refit models, and do not account for shared-training dependencies. They are not uncertainty bounds for scripture authorship. The corpus is also below the predeclared minimum coverage for a pass.

## Actual app behavior

The production clustering controls withheld author labels from features. Across 20 single-author controls, the app returned one group each time. On mixed-author inputs, it generally merged known authors rather than recovering them:

| Language | Tokens per unit | Smoothing | Reference authors | Returned groups | Agreement with author labels (ARI) |
|---|---:|---:|---:|---:|---:|
| grc | 25 | 0.0 | 6 | 1 | 0.000 |
| grc | 25 | 0.7 | 6 | 1 | 0.000 |
| grc | 500 | 0.0 | 6 | 3 | 0.540 |
| grc | 500 | 0.7 | 6 | 1 | 0.000 |
| hbo | 25 | 0.0 | 4 | 1 | 0.000 |
| hbo | 25 | 0.7 | 4 | 1 | 0.000 |
| hbo | 500 | 0.0 | 4 | 1 | 0.000 |
| hbo | 500 | 0.7 | 4 | 2 | 0.002 |

In particular, default smoothing returned one group for six Greek authors at both tested lengths. A one-group result cannot establish a single writer. At 500 tokens, turning smoothing off returned three Greek groups with moderate author agreement, showing sensitivity to this setting.

## Robustness and limits

Spelling variants that normalize to the same text produced identical predictions. Deleting 2% of words retained approximately 97% of predictions; replacing 20% with another author’s text retained 88.3% for Greek and 86.2% for Hebrew. These perturbation runs use a separate held-out-work split and are descriptive controls, not the cross-work acceptance score. Both Lucian editions selected the same nearest reference author, with cosine similarity 0.997. One edition pair does not establish broad manuscript or translation invariance.

Arabic, Biblical Hebrew/Aramaic, scripture transfer and AI profiles remain unvalidated. Ancient author labels are catalogue assumptions; genre, topic, period and editorial practice can still correlate with authors. See the [protocol and source documentation](README.md).

The final audit changed grouping from display titles to canonical work IDs and aligned the robustness predictor with the main work-balanced centroid recipe. These corrections are versioned and all results above were recomputed afterward. No decision thresholds were tuned using the scores. Preliminary scores from the earlier partition basis are superseded by this snapshot.

The code now protects against held-out feature leakage and repeated editions, and preserves an explicit failing empirical acceptance gate. Improving author identification requires new reserved authors/works, stronger rejection calibration and matched editorial/topic controls. This exposed corpus must not be described as untouched evaluation data after future tuning.

Recreate the detailed report and individual predictions with `uv run --frozen stylometry benchmark --download`. Run `uv run --frozen stylometry benchmark --check` or `uv run --frozen pytest -q --run-empirical tests/test_empirical_acceptance.py` to enforce the current failing acceptance gate. No paid APIs are involved.
