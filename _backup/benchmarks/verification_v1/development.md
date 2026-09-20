# Cross-work pair-verification development

**Development only. No independent authorship or scripture validation.**

Verifier training authors, calibration authors and testing authors are disjoint within each fold. Each candidate has two independent reference works; the whole two-candidate search is calibrated. All previous corpora are now exposed development data.

| Language / control | Method | Nearest-candidate accuracy | Correct known acceptance | Wrong known acceptance | Unfamiliar acceptance | Feasible calibration folds |
|---|---|---:|---:|---:|---:|---:|
| grc / all genres | pair_verifier | 92.2% | 0.0% | 0.0% | 0.0% | 0/6 |
| grc / all genres | baseline | 95.6% | 9.1% | 0.0% | 4.7% | 1/6 |
| hbo / all genres | pair_verifier | 77.2% | 0.0% | 0.0% | 0.0% | 0/4 |
| hbo / all genres | baseline | 93.9% | 0.0% | 0.0% | 0.0% | 0/4 |

Zero false acceptance with zero correct acceptance is a failure to provide useful attribution. Candidate calibration rates for infeasible thresholds are saved separately from the active abstention decisions.

The baseline repeats the old fixed model recipe on these same galleries and queries. Its features/models are independently fitted to each calibration/test gallery; its threshold transfers between them. The pair verifier instead keeps its feature/model fit fixed while transferring to both galleries. These are deliberately different strategies, not identical training regimes.

## Comparable-genre controls

- grc / deliberative_oratory: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / epideictic_prose_hymn: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / forensic_oratory: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / medical_treatise: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / philosophical_dialogue: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / philosophical_essay: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / rhetorical_criticism: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / satirical_prose: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- grc / socratic_prose: insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors.
- hbo / literary_prose: identical to main panel; no independent replication.

Unknown-query subsets sharing a genre/topic label with a gallery reference are in the JSON, with author/work counts. They are narrower diagnostics, not fully controlled replications.

## Limits

- All source corpora were previously exposed; this is development, not a fresh confirmation.
- Two-candidate searches cannot validate four-candidate or unrestricted attribution.
- Rotating folds reuse authors and works. Rates balance author/work/fold, and no population confidence interval is supplied.
- Only two unfamiliar authors calibrate each fold; candidate thresholds do not certify population error rates.
- Shared genre/topic labels and exact duplicate checks do not remove quotations, editorial effects or all content shortcuts.
- Classical Greek and modern Hebrew do not establish scripture transfer, Biblical Hebrew, Arabic or named historical identities.

The pre-fit protocol, source hashes, runtime and partitions are in `development_lock.json`. All raw decision scores, work-level reference evidence, candidate calibration thresholds and final abstentions are in `development.json`. Scores are not historical authorship probabilities.
