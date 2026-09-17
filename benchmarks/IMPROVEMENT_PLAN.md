# Improving author discrimination

The evidence points to two distinct problems: the unknown-author decision rule lacks suitable
calibration, and the representation often distinguishes genre more strongly than writers. We
should improve each component on development data, then measure the frozen result on new works
and authors. A green software suite cannot substitute for that evaluation.

**Implementation update:** the passage builder, three-model comparison, matched-genre/topic
controls, two-impostor calibration, locked fresh-author evaluation and full-refit uncertainty
checks have now been implemented and run. See [the study and reproduction guide](study_v1/README.md)
and [fresh results](study_v1/RESULTS.md). The fresh empirical gate still fails; the changes must
not be described as established scripture attribution.

## 1. Calibrate an explicit unknown outcome — implemented as a development experiment

The original rejection threshold retains 95% of genuine known-author calibration matches.
That criterion does not constrain how often unfamiliar writers are accepted. The corrective
experiment calibrates the actual decision statistic, maximum candidate similarity, against
an entirely separate unfamiliar author. Both that author and the outer unknown test author
are excluded from feature/model fitting. The outer unknown author is excluded from calibration;
the calibration impostor is excluded from testing. Canonical works remain disjoint.

The revised rule chooses a threshold meeting ≤5% false acceptance on calibration impostors,
with equal author/work weighting. It must also retain ≥80% of known calibration material.
Exact candidate ties, missing calibration and incompatible targets yield abstention. Known
correct/wrong acceptance and unknown acceptance are reported separately. Old and revised rules
share exactly the same fitted scores, candidate pools and test texts.

Run `uv run --frozen stylometry benchmark-rejection`. The [development results](REJECTION_DEVELOPMENT.md)
show **0 of 18 Greek and 0 of 12 Hebrew scenarios** meeting both calibration targets. Consequently
the revised rule abstains on every test text: zero false accepts **and zero useful coverage**.
This corrects an unsafe decision policy but does not solve authorship discrimination. One
calibration impostor per scenario is also far too narrow to characterize future unknown writers.
The experiment is not enabled as a validated production attribution model.

This task definition matters: [PAN's open-set attribution benchmark](https://pan.webis.de/clef19/pan19-web/authorship-attribution.html)
explicitly allows the true author to be absent from the candidate list. A system needs an unknown
outcome and must evaluate its mistakes as well as its ability to identify familiar writers.

## 2. Measure comparable passages before computing features — implemented

`stylometry cluster-passages` now implements this representation with source mappings and
exclusion accounting. The reference study compared all three specified passage lengths.

Pool raw text into fixed 500-, 1,000- and 2,000-token passages, then extract features. Preserve
work, witness, chapter and gap boundaries. Use nonoverlapping passages for evaluation; if
overlapping windows aid the display, map them back to their shared underlying text instead of
counting them as independent observations. Mark units with insufficient text as unsupported.

The current ±5-unit smoothing has different effective support for different input lengths:
up to approximately 275 words with 25-word units versus 5,500 with 500-word units. Averaging
TF-IDF projections and lexical diversity statistics from short units is not equivalent to
extracting those statistics from pooled text. Compare pooled passages and smoothing choices
on development data before changing defaults. Passage length is an experimental factor, not
a claim that 1,000 words guarantees reliable identification.

## 3. Separate genre and topic from authorship — comparison implemented

The study now refits models within matched genres and topics. Its fixed candidates are a
function-word centroid, a centroid with word endings and adjacent function/common-word pairs,
and a regularized character-pattern classifier. The word-order and ending features are proxies;
no validated ancient-language syntactic or morphological tagger is being claimed.

In the saved Greek 500-token control without smoothing, the three returned groups align almost
perfectly with genre (ARI 0.956), compared with author ARI 0.540: forensic orators together,
Plato/Xenophon together, and Lucian separately. Under default smoothing the same initial
three-group proposal is rejected because its minimum stability score is 0.776, below 0.8.
Lowering that cutoff would expose a genre-related partition, not recover six distinct writers.

Keep the existing stability safeguards. Compare writers within shared genres and comparable
subjects, and require same-writer generalization across works. Investigate function-word,
morphological and syntactic patterns alongside character features using language-appropriate
analysis; audit those tools on ancient text before trusting their output. Compare regularized
supervised reference models with the current centroid baseline. Continue reporting clustering
as exploratory style groups. A reference classifier and an unsupervised partition solve
different tasks.

Research offers hypotheses, not a guarantee for this corpus. A [topic-confusion study](https://aclanthology.org/2021.findings-emnlp.359/)
found that part-of-speech and stylometric features helped distinguish topic changes from writer
differences. That motivates a matched-topic/genre experiment; it does not establish transfer to
ancient Greek, Biblical Hebrew or scripture.

## 4. Freeze the model, then use genuinely new evidence — first fresh evaluation completed

The locked study used 26 fresh Greek and 32 fresh modern Hebrew works from 16 new catalogue
authors. Each language reserved four reference authors, two calibration unfamiliar authors
and two final unfamiliar authors. Source selection and roles were fixed before scoring.
These corpora are now exposed evaluation data; another round of tuning needs another final set.

Use the current exposed corpus for development. Before selecting a final model, assemble
additional development authors for rejection calibration and reserve separate final authors
and works that do not influence feature selection, thresholds or parameter choices. Include
same-genre writers, multiple works and editors per writer, more than one edition, and explicit
quotation/translation controls. Assess uncertainty by resampling appropriate independent works
and authors with model fitting repeated, rather than only resampling fixed predictions.

Final validation must measure false attribution and useful coverage simultaneously. Keep
the original accuracy and error targets; failed or inconclusive results remain such. Evaluate
new authors and topics explicitly, as in [PAN's open-set verification scenario](https://pan.webis.de/clef21/pan21-web/author-identification.html).
The current minimum corpus coverage is a lower bound, not a guarantee of sufficient evidence.

Scripture use requires a separate transfer study with comparable language, period, genre,
translation status and textual transmission. Modern Hebrew cannot establish Biblical Hebrew
performance, and conventional attributions are not independently proven identities. Reliable
style comparisons may be achievable while historical named-author claims remain unresolved.

## Reporting corrections already made

Inferred clusters now consistently appear as style groups in report and HTML tables. A one-group
fallback means no supported partition; it does not prove one writer. Translation, editing,
scribal effects and genre are possible explanations of similarity and difference. Existing
file formats and machine-readable field names remain compatible.
