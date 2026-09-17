# stylometry

Verse-level author discovery across the manuscripts of the Bible and the Quran. Every verse gets a
style profile - partly from classical stylometry, partly from an LLM - and the verses of each language
are clustered into stylistic *hands* named A1, A2, ... The outputs say how many hands a corpus needs,
which verses each hand wrote, what makes each hand recognisable, and how that lines up with traditional
attributions.

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
everything for every language present. Every step is resumable: profiles are appended as they arrive and
units already profiled are skipped.

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

**Which model?** `stylometry compare-models` answers this from the profile files on disk and writes
`output/models/` (`index.html`, `model_comparison.md`, `models.csv`, and the clustering re-run with each
model's profiles under `cluster/`):

```
uv run stylometry compare-models --set opus-5=data/processed/profiles.jsonl \
    --set deepseek-flash=data/processed/profiles_deepseek_flash.jsonl \
    --set deepseek-flash_retest=data/processed/profiles_deepseek_flash_retest.jsonl \
    --set gpt-5.5=data/processed/profiles_gpt55.jsonl --set gpt-5.5_retest=data/processed/profiles_gpt55_retest.jsonl \
    --set deepseek-v4-pro=data/processed/profiles_deepseek.jsonl ...
```

There is no ground truth for the style of a verse, so four proxies are used: agreement with a reference
model (the first `--set`); agreement with the consensus of all the *other* models; stability, i.e.
agreement of a model with itself on a second run (`NAME_retest` sets); and usefulness, i.e. how well the
profile alone recovers which of the pilot works a verse comes from, and what the project's own clustering
does with it. The pilot set is the first 75 verses of Mark, John, Romans, 1 Clement and the Acts of John
(375 verses, five different hands). Six configurations were run on the 75 Mark verses; Opus 5,
deepseek-flash and gpt-5.5 on all 375; flash and gpt-5.5 twice on Mark.

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

Reading: gpt-5.5 is the most accurate and by far the most stable model. deepseek-flash is noisy verse
by verse (a second run agrees with the first only at r = 0.61) but its errors average out: fed into the
clustering it recovers the five works as well as Opus or gpt-5.5, and averaging two flash runs lifts its
consensus agreement to 0.83 for $0.0006 per verse. deepseek-v4-pro is not better than flash at three
times the price; gpt-5.4-mini is the outlier and not worth its price. Categorical agreement with the
Opus reference is lower outside Mark (57-71%) for every model alike because those Opus profiles were
made with the first, Greek-only prompt, which labelled the letters `exhortation` / `author_first_person`
where the current prompt yields `speech` / `prophet_or_apostle`; the numeric scales are unaffected.
Recommendation: profile everything with deepseek-flash (twice, if you want the stability, still under $50
for the whole corpus), and spot-check a sample with gpt-5.5. DeepSeek V4 reasons by default at ~1,000
reasoning tokens per verse; the backend switches it off unless `--thinking` is given.

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
3. **AI style profiles** (`stylometry/ai_profile.py`): the model reads a chapter-sized run of units and
   returns per unit six 0-1 scales (register, foreign interference, hypotaxis, lexical richness,
   rhetorical polish, emotional intensity), five language-neutral categories (discourse mode, narrative
   tense, connective style, voice, quotation), device tags, diagnostic phrases and a one-line signature.
   The prompt carries language-specific anchors (Atticism, Late Biblical Hebrew, saj' ...).
4. **Clustering** (`stylometry/cluster.py`): blocks standardised and variance-equalised, each verse
   blended with its ±5 neighbours in the same work, PCA to 30 dims, k chosen by scoring each candidate
   partition on the verses' own unsmoothed vectors (`--k-criterion`, `--k`), k-means, hands renamed by
   size. Per hand: effect-size markers, tag lift, phrases, representative verses; per verse: assignment
   margin and outlier score.
5. **Validation**: adjusted Rand index against work and against traditional groupings (Paul-undisputed /
   Deutero-Pauline / Pastorals, Torah / Former Prophets / Chronicler, Meccan / Medinan ...), purity per
   work, and for Sinaiticus a scribe × hand table.

## Outputs (`output/<language>/`)

`verse_assignments.csv`, `authors.json`, `work_by_author.csv`, `segments.csv`, `outliers.csv`,
`k_selection.csv`, `summary.json`, `report.md`, and `site/` with the dashboard (`index.html`), one page
per hand under `authors/` and one page per work under `works/`. `output/index.html` links the languages
and `output/models/` holds the model comparison.

A run can cover one book: `--works GEN --out output/hbo-genesis` profiles and clusters Genesis on its
own. With a single work the cross-work statistics (ARI, purity per work) are degenerate, so the report
and the dashboard switch to a chapter breakdown, and for Hebrew the report adds a YHWH / Elohim table
per hand: the divine names are never clustering features, so any skew there is independent evidence.

## Reading the results honestly

- Clusters are stylistic hands, not identified persons. Genre is the loudest stylistic signal, so the top
  split separates narrative from letter from vision; the works × hands table and the ARI figures show how
  much of the structure is book-level.
- The number of hands depends on the criterion; the k-selection table shows how sharp the optimum is.
- Runs of several consecutive minority-hand verses are the interesting candidates; isolated flips are
  mostly noise.
- LXX "authors" are translators; the Old Greek and Theodotion versions of Daniel are kept apart for that
  reason. The Quran's traditional Meccan/Medinan split is the external check for any stylistic split.
