# Unknown-author rejection development

**Development only. Author attribution remains unvalidated.**

The revised rule calibrates against a separate unfamiliar author, using the maximum candidate score—the quantity used to accept a text. It abstains on ties or when no threshold meets both calibration targets: ≤5% impostor acceptance and ≥80% known-author acceptance. The outer unknown test author enters neither training nor calibration.

Old and revised decisions below use the same models and test texts. Refusing all texts is not a successful authorship system.

Rates average within work and then author. Accepted accuracy includes mistakes on both known and unknown authors; it is unavailable if no text was accepted.

Actual settings: 1000 tokens per passage, at most 12 samples per work, seed 42. Protocol values describe defaults; overrides remain development-only.

| Language | Rule | Unknown accepted | Known accepted | Known correctly accepted | Known wrongly accepted | Balanced accuracy among accepted |
|---|---|---:|---:|---:|---:|---:|
| grc | Known-author calibration only | 92.0% | 96.6% | 83.8% | 12.8% | 69.7% |
| grc | Separate unfamiliar-author calibration | 0.0% | 0.0% | 0.0% | 0.0% | Unavailable |
| hbo | Known-author calibration only | 85.4% | 92.5% | 70.9% | 21.6% | 53.3% |
| hbo | Separate unfamiliar-author calibration | 0.0% | 0.0% | 0.0% | 0.0% | Unavailable |

grc: 0/18 scenarios had a feasible calibration threshold. Infeasible scenarios abstain from all test texts. Feasibility is a calibration point estimate, not validation. Actual source languages: grc.

hbo: 0/12 scenarios had a feasible calibration threshold. Infeasible scenarios abstain from all test texts. Feasibility is a calibration point estimate, not validation. Actual source languages: heb.

## Limits

- This corpus was already inspected. The experiment is development evidence, not fresh validation or a replacement for the frozen failed benchmark.
- One calibration impostor author per scenario cannot characterize the diversity of unseen writers. The 5% calibration target is not a certified bound on future false acceptance.
- Abstaining from every text eliminates false acceptance but fails useful known-author coverage. Both outcomes must be reported.
- Intervals resample fixed predictions by work; model/calibration refitting and dependencies from shared authors and training works are not modeled.
- Old and revised policies share each fitted model and candidate pool. Their scores are not directly comparable to the larger candidate pools in the original benchmark.
- Accuracy among accepted texts depends on the designed mix of known and unknown authors; it is not expected precision for a deployment population.
- Modern Hebrew is not Biblical Hebrew; scripture transfer and Arabic remain unvalidated.

Author/work partitions, calibration metrics, thresholds, source provenance and implementation hashes are in [rejection_development.json](rejection_development.json). Full decisions are regenerated in `output/rejection-development/experiment.json` by running `uv run --frozen stylometry benchmark-rejection`.
