# stylometry

Exploratory style analysis across the manuscripts of the Bible and the Quran. Each verse receives
lexical features and optionally a blinded AI style profile. Within each language, verses are grouped
into style clusters labelled A1, A2, ... . These groups help locate differences worth examining; they
do not identify authors or establish how many people composed a text. A one-group result means the
analysis found no supported split, not that a single author has been proved.

## Reliability and verification

Run the offline regression and strategy controls with `uv run --frozen pytest -q`. No model API calls
or corpus downloads are required. Tests cover:

- Single Gaussian populations, including correlated dimensions, must not be forced into multiple groups.
- Clear synthetic style populations must survive smoothing and passage subsampling.
- Shared request effects must not score as successful recovery of held-out works.
- Missing profiles, manuscript gaps and chapter boundaries must break smoothing and reported runs.
- Anonymous prompt IDs must map correctly back to corpus IDs. Partial, duplicate, invalid or non-finite
  responses must fail without creating invented measurements.
- Resume must preserve the original request context and reject changed text or generation settings.
- Constant and tiny feature matrices and a forced single group must produce usable outputs.
- Held-out feature extraction must preserve its training vocabulary, IDF, projection and scaling.
- Reference texts must match pinned source and normalized-text checksums; alternate editions must
  not become independent works in training and test partitions.

The [real-text benchmark](benchmarks/README.md) downloads 18 Greek works by six catalogued authors,
16 modern Hebrew works by four authors, and two editions of one additional Greek work. It tests
separate training, calibration and test works, entirely unseen authors, 100/500/1,000-word passages,
feature choices, genre/topic subsets, the actual clustering pipeline and text perturbations.

```bash
uv run --frozen stylometry benchmark --download          # writes output/benchmark/report.md
uv run --frozen stylometry benchmark --check             # nonzero unless empirical gates pass
uv run --frozen pytest -q --run-empirical tests/test_empirical_acceptance.py
```

**The current empirical benchmark fails.** See [measured results](benchmarks/RESULTS.md).
Passing software tests does not establish reliable author identification. The opt-in empirical
acceptance test deliberately fails when the measured reliability requirements are unmet; it is
separate from the fast regression suite. No paid model calls are made by this benchmark.
Modern Hebrew does not validate Biblical Hebrew or Aramaic, and Arabic has no reference corpus
in this first pilot. Scripture attribution and AI profiles remain unvalidated.

The [improvement plan](benchmarks/IMPROVEMENT_PLAN.md) separates rejection calibration, passage
representation and genre confounding. `uv run --frozen stylometry benchmark-rejection` runs the
first corrective experiment with a separate unfamiliar calibration author and an outer unknown
test author. It is development only and does not replace the failed benchmark. In the current
corpus, none of 30 scenarios meets both calibration targets; the revised rule abstains from all
texts. See the [development results](benchmarks/REJECTION_DEVELOPMENT.md).

The next [passage/model study](benchmarks/study_v1/README.md) is implemented and has been run on
58 additional works by 16 new catalogue authors, with model choices locked before evaluation.
It compares 500/1,000/2,000-token passages, three feature/model families, matched genres/topics,
and rejection using two separate unfamiliar calibration authors. It refits models during
uncertainty checks. [Fresh results](benchmarks/study_v1/RESULTS.md) show useful known-candidate
discrimination but still fail safe, useful unknown-author attribution.

`uv run --frozen stylometry cluster-passages --language grc --tokens 1000` pools normalized
source words before feature extraction, without additional smoothing or AI calls. It preserves
chapter, witness and gap boundaries and records source-token mappings and excluded text in
`output/passages/grc/passages.json`. Use `--corpus PATH` for a separate source JSONL, and `--out`
to choose the output directory. Passage groups remain exploratory.

Automatic clustering always includes a one-group baseline. Candidate partitions are fitted on smoothed
features and scored on unsmoothed features; PCA is fitted on unsmoothed features. A split must improve
[Gaussian-mixture BIC](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html) over one Gaussian by at least 10 and have positive unsmoothed silhouette. BIC uses
a fixed sample of at most 4,000 observations, rotated into its full PCA basis before fitting diagonal
covariances. The selected split must then have adjusted Rand agreement of at least 0.8 in all five
80% passage subsamples. Fewer than five passages, unstable partitions or insufficient BIC improvement
produce one group. `--k` explicitly overrides selection and is labelled as forced. The BIC baseline is
always considered, even when `--kmin` is 2 or larger.

These thresholds are exploratory diagnostics, not calibrated authorship tests. Passage subsampling
keeps the feature map fixed and measures partition sensitivity; it does not validate the entire
pipeline on unseen data. Margins are centroid-distance measures, not probabilities. Full validation
still needs substantially broader reference corpora, matched genre/topic/editor controls, manuscript
and translation comparisons, and repeated blind model runs. The bundled lexical pilot measures
some of these failure modes and does not reproduce or validate the historical model results below.

**Existing profiles:** generate a new `--profiles` file for the blinded workflow. Complete legacy
profiles remain readable with a warning for exploration, but cannot be resumed into a new run and do
not receive held-out work-recovery scores without request provenance. Invalid profiles are rejected.
New profiles record prompt/schema versions, input text hashes, generation settings and request groups.
Resuming rejects incompatible text, model, settings or context. `--limit` is a soft cap rounded up to a
complete request so changing the limit cannot change the context of an already profiled verse.

[Grouped work-recovery evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) holds out whole chapters and AI requests together, fits tag vocabulary inside
training folds, and compares models on a common cohort. If independent groups or provenance are
insufficient, the score is unavailable rather than estimated from random verse splits. Consensus and
retest agreement measure agreement and repeatability, not accuracy against an authorial ground truth.

## Corpus

Three languages, analysed separately (stylometric features never cross a language boundary):

| Language | Witness / edition | Contents | Source | Licence |
|---|---|---|---|---|
| Greek | Codex Sinaiticus (S) | LXX (partial), complete NT, Barnabas, Hermas; first hand, scribes A/B/D per page | [itsee-birmingham/codex-sinaiticus](https://github.com/itsee-birmingham/codex-sinaiticus) | CC BY-NC-SA 3.0 |
| Greek | every NT witness to c. AD 400 (130 sigla) | P52, P104, P4, P64, P66, P46, P75, P45, P47 ... and the majuscules 02 Alexandrinus, 03 Vaticanus, 04 Ephraemi, 05 Bezae; original hand, correctors dropped, nomina sacra expanded | [CNTR transcriptions](https://github.com/Center-for-New-Testament-Restoration/transcriptions) | CC BY-SA 4.0 |
| Greek | Swete = Vaticanus OT | Swete's *Old Testament in Greek*, the text of B (56 books incl. OG and Theodotion Daniel) | [nathans/lxx-swete](https://github.com/nathans/lxx-swete) (from First1KGreek) | CC BY-SA 4.0 |
| Greek | Bonnet | Acts of John, Thomas, Philip, Barnabas | [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek) | CC BY-SA 4.0 |
| Greek | Lake | Apostolic Fathers (1-2 Clement, Ignatius, Polycarp, Didache, Barnabas, Hermas, Mart. Pol., Diognetus) | [jtauber/apostolic-fathers](https://github.com/jtauber/apostolic-fathers) | CC BY-SA 4.0 |
| Hebrew | Leningrad Codex (L) | complete Tanakh, Westminster/OSHB | [openscriptures/morphhb](https://github.com/openscriptures/morphhb) | text PD, morph CC BY 4.0 |
| Hebrew | Aleppo Codex (A) | Miqra according to the Masorah (Aleppo where extant) | [bdenckla/MAM-basics](https://github.com/bdenckla/MAM-basics) | CC BY-SA 4.0 |
| Hebrew | Samaritan Pentateuch (SP) | Schorch's transcription of MS Dublin CBL 751 | [DT-UCPH/sp](https://github.com/DT-UCPH/sp) | CC BY-NC 4.0 |
| Hebrew | Dead Sea Scrolls (Q) | all scrolls in Abegg's transcription: biblical scrolls incl. 1QIsaa as witnesses, non-biblical scrolls as works | [ETCBC/dss](https://github.com/ETCBC/dss) | CC BY-NC 4.0 |
| Hebrew | Nash Papyrus (N), Ketef Hinnom (KH) | Decalogue + Shema; the two silver amulets | readings after Cook/Albright and Barkay/Ahituv via Wikipedia | CC BY-SA 4.0 |
| Arabic | Quran (T) | Tanzil Uthmani text, 114 suras with Meccan/Medinan metadata | [tanzil.net](https://tanzil.net) | verbatim copies with attribution |

**Witnesses versus works.** Isaiah exists in L, A, 1QIsaa (Hebrew) and in S and Swete (Greek); Mark
in Sinaiticus, Vaticanus, Alexandrinus, P45 ... For authorship one witness per work per language is
enough, so one witness becomes the *primary* (`ISA`) and the others are kept as `ISA@1Qisaa`,
`MARK@P45` ... flagged `duplicate_of`. The primary is chosen by a priority list (Sinaiticus, Vaticanus,
Leningrad, Aleppo, Samaritan, then the editions) among witnesses that have at least 90% of the fullest
witness's text, so Sinaiticus is primary for the NT, Swete (Vaticanus) for the Greek OT where Sinaiticus
is fragmentary, Leningrad for the Tanakh, and Lake's edition for Hermas. `data/processed/verses.jsonl`
holds the primaries (or everything with `build-corpus --include-duplicates`); `witnesses.jsonl` always
holds every witness. Ketef Hinnom and Nash are far too short to fingerprint; they are in the corpus as
witnesses with their reconstruction share recorded.

**Witness comparison.** `stylometry witnesses --language grc` aligns every secondary witness against the
primary verse by verse (token-level similarity and a word diff), writes `witness_summary.csv` /
`witness_verses.csv`, and renders `site/witnesses.html`: all manuscripts of the language oldest first
(Ketef Hinnom c. 600 BC ... Leningrad AD 1008; P52 c. 150 ... Alexandrinus c. 420) with coverage, mean
similarity to the primary and a page per witness showing each verse with the differences highlighted.

`scripts/download_sources.sh` fetches everything (about 500 MB). The Samaritan and Dead Sea Scrolls data
and the Sinaiticus transcription are non-commercial licences, so derived outputs are too.

## Quick start

```bash
scripts/download_sources.sh
uv sync
uv run stylometry build-corpus                          # -> data/processed/verses.jsonl
uv run stylometry estimate --language hbo --scope all --model deepseek-flash
uv run stylometry profile --backend deepseek --model deepseek-flash --language grc --limit 100 --profiles data/processed/profiles_deepseek_flash.jsonl
uv run stylometry profile --backend deepseek --model deepseek-flash --scope all --yes --workers 8 --profiles data/processed/profiles_deepseek_flash.jsonl
uv run stylometry cluster --language grc --profiles data/processed/profiles_deepseek_flash.jsonl   # -> output/grc/
uv run stylometry cluster --language hbo --scope all --profiles ...                                # -> output/hbo/
uv run stylometry cluster --language arb --scope all --profiles ...                                # -> output/arb/
uv run stylometry report --language grc && uv run stylometry html --language grc                   # output/grc/report.md, output/grc/site/, output/index.html
uv run stylometry witnesses --language grc                                                          # output/grc/site/witnesses.html
uv run stylometry compare-models --set opus-5=data/processed/profiles.jsonl --set flash=...         # output/models/ (see Models and cost)
```

`uv run stylometry all --backend deepseek --model deepseek-flash --scope all --yes --profiles ...` chains
everything for every language present. Profiles are appended as complete validated requests arrive. Compatible complete requests are
skipped on resume; incomplete requests are regenerated with their original context.

### Scope and language

`--language grc|hbo|arb` picks the corpus; clustering is always single-language. `--scope` narrows it:
`christian` (Greek default: everything except the Septuagint), `nt`, `lxx`, `sinaiticus`, `noncanonical`,
`tanakh`, `quran`, `all`. `--works MARK ISA Q002` picks individual works (Quran suras are `Q001`-`Q114`).

### Models and cost

Any of these backends produce the per-verse profiles; keep each model's output in its own `--profiles`
file, because the style scales of different models are not on one scale. Every run appends its token
usage and cost to `<profiles>_runs.jsonl`; `stylometry compare-models` reads those to price each model.

| backend | model | needs | per-verse cost (measured, list price) |
|---|---|---|---|
| `sdk` / `batch` | claude-opus-5 (default) | `ANTHROPIC_API_KEY` | ≈ $0.006 sync, $0.003 batch (estimated) |
| `cli` | claude-opus-5 | local `claude` login | $0.012 |
| `deepseek` | deepseek-flash | `DEEPSEEK_API_KEY` | $0.0003 (half off-peak) |
| `deepseek` | deepseek-v4-pro | `DEEPSEEK_API_KEY` | $0.001; `--thinking` $0.004 |
| `openai` | gpt-5.5 / gpt-5.4-mini | `OPENAI_API_KEY` | $0.007 / $0.005 at low reasoning |

Keys are read from the environment or from a `.env` file in the repo root (gitignored).
`stylometry estimate --language ... --model ...` prints the projection for any scope. For the full
corpus (74,130 primary units: Greek 41,126, Hebrew 26,768, Arabic 6,236; about 3,800 requests):

| model | whole corpus | Greek only | Hebrew only | Arabic only |
|---|---|---|---|---|
| deepseek-flash (peak / off-peak) | $23 / $12 | $13 / $6.5 | $8.5 / $4.2 | $2 / $1 |
| deepseek-flash, two runs averaged | $47 / $23 | $26 / $13 | $17 / $8.4 | $4 / $2 |
| claude-opus-5 (sync / batch, estimated) | ≈ $460 / $230 | $255 / $128 | $166 / $83 | $39 / $19 |
| gpt-5.5, low reasoning (sync / batch) | $524 / $262 | $291 / $145 | $189 / $95 | $44 / $22 |

**Comparing models.** `stylometry compare-models` reports agreement, repeatability, descriptive work
recovery and cost from the profile files on disk and writes
`output/models/` (`index.html`, `model_comparison.md`, `models.csv`, and the clustering re-run with each
model's profiles under `cluster/`):

```
uv run stylometry compare-models --set opus-5=data/processed/profiles.jsonl \
    --set deepseek-flash=data/processed/profiles_deepseek_flash.jsonl \
    --set deepseek-flash_retest=data/processed/profiles_deepseek_flash_retest.jsonl \
    --set gpt-5.5=data/processed/profiles_gpt55.jsonl --set gpt-5.5_retest=data/processed/profiles_gpt55_retest.jsonl \
    --set deepseek-v4-pro=data/processed/profiles_deepseek.jsonl ...
```

The historical pilot below used unblinded prompts and random verse-level folds, so its work-recovery
numbers are optimistic and cannot establish model accuracy or author discrimination. They are retained
as historical measurements only. The revised evaluator uses common cohorts and grouped holdouts;
these numbers must be regenerated with new blinded profiles before comparing the revised strategy.
For profile files covering several languages, use `compare-models --language grc` (or `hbo`/`arb`)
to keep the evaluation and cluster comparisons within one language.
The pilot covered the first 75 verses of Mark, John, Romans, 1 Clement and Acts of John (375 verses).

| model | consensus score (75 Mark verses) | retest r | work recovery (375) | clustering ARI vs work, k free / fixed | $/verse |
|---|---|---|---|---|---|
| gpt-5.5 (low reasoning) | **0.86** | **0.92** | 83% | 0.64 / 0.59 | 0.0071 |
| claude-opus-5 (CLI, reference) | 0.82 | – | **87%** | 0.58 / 0.65 | 0.0117 |
| deepseek-flash, two runs averaged | 0.83 | – | – | – | 0.0006 |
| deepseek-flash | 0.80 | 0.61 | 84% | 0.59 / 0.64 | **0.0003** |
| deepseek-v4-pro + `--thinking` | 0.80 | – | – | – | 0.0039 |
| deepseek-v4-pro | 0.73 | – | – | – | 0.0010 |
| gpt-5.4-mini | 0.64 | – | – | – | 0.0048 |
| lexical features only, no AI | – | – | – | 0.46 / 0.69 | 0 |

The pilot suggests differences in agreement, repeatability and cost, but does not justify an
accuracy ranking or a whole-corpus model recommendation. A new blinded pilot with enough independent
passages is required. Numeric and categorical definitions changed between older prompts, which is
another reason not to pool legacy and current profiles in one evaluation.

## How it works

1. **Corpus** (`stylometry/corpus/`). One loader per source produces the common verse record (see
   `corpus/__init__.py`). Sinaiticus and Vaticanus keep the first hand only; the apocryphal Acts and the
   non-biblical scrolls have no verse numbers and are packed into ~25-word units at sentence or line
   boundaries; Dead Sea Scrolls units that are mostly reconstruction are dropped. Every text also gets a
   *bare* form per language (Greek: no accents, one sigma; Hebrew: no vowels or cantillation, final
   letters folded; Arabic: no tashkeel, alef variants merged) so manuscripts and editions compare fairly.
2. **Lexical features** (`stylometry/features.py`, no AI): closed-class word rates, word-ending rates,
   length and verse-initial connective habits (καί / waw / wa-), and a character 2-4-gram projection,
   with word lists per language in `stylometry/lang.py`.
3. **AI style profiles** (`stylometry/ai_profile.py`): the model reads text with anonymous unit IDs, without work/witness metadata, and
   returns per unit six 0-1 scales (register, foreign interference, hypotaxis, lexical richness,
   rhetorical polish, emotional intensity), five language-neutral categories (discourse mode, narrative
   tense, connective style, voice, quotation), device tags, diagnostic phrases and a one-line signature.
   The prompt carries language-specific anchors (Atticism, Late Biblical Hebrew, saj' ...).
4. **Clustering** (`stylometry/cluster.py`): blocks standardised and variance-equalised, each verse
   blended with its ±5 neighbours inside an uninterrupted chapter passage, PCA to 30 dims fitted on
   unsmoothed vectors, and k selected with the single-group and stability checks above (`--k-criterion`,
   `--k`). K-means groups are named by size. Per hand: effect-size markers, tag lift, phrases, representative verses; per verse: assignment
   margin and outlier score.
5. **Descriptive checks**: adjusted Rand index against work and against traditional groupings (Paul-undisputed /
   Deutero-Pauline / Pastorals, Torah / Former Prophets / Chronicler, Meccan / Medinan ...), purity per
   work, and for Sinaiticus a scribe × hand table.

## Outputs (`output/<language>/`)

`verse_assignments.csv`, `authors.json`, `work_by_author.csv`, `segments.csv`, `outliers.csv`,
`k_selection.csv`, `summary.json`, `report.md`, and `site/` with the dashboard (`index.html`), one page
per hand under `authors/` and one page per work under `works/`. `output/index.html` links the languages
and `output/models/` holds the model comparison.

## Reading the results honestly

- Clusters are exploratory style groups, not identified persons. Genre is the loudest stylistic signal, so the top
  split separates narrative from letter from vision; the works × hands table and the ARI figures show how
  much of the structure is book-level.
- The number of hands depends on the criterion; the k-selection table shows how sharp the optimum is.
- Consecutive minority-group verses are hypotheses for closer reading. Smoothing itself produces runs;
  quotations, topic or genre shifts may also explain them.
- LXX "authors" are translators; the Old Greek and Theodotion versions of Daniel are kept apart for that
  reason. The Quran's traditional Meccan/Medinan split is the external check for any stylistic split.
