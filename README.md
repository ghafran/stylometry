# Stylometry

Discover possible shared authors within each language, tag every verse, and explore
the results from language down to collection, book, chapter and verse. The English
reference corpus provides 13 known authors against which to measure the results.

The previous application, tests, scripts, documentation, dependency files and
benchmarks are preserved in `_backup/`. The original source texts remain in
`data/raw/`. The new analysis and explorer do not load the previous application.
The source format readers retain compatible parsing logic where needed to read
the existing manuscripts.

## Run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```sh
uv sync --frozen
uv run stylometry run
uv run stylometry serve
```

Open http://127.0.0.1:8000, or open `output/index.html` directly. Everything runs
locally without API credentials, paid model calls, or new corpus downloads.

You can also build once and analyze selected languages:

```sh
uv run stylometry build
uv run stylometry analyze --language eng --out output/english
uv run stylometry analyze --language grc --language hbo --out output/ancient
uv run stylometry benchmark
uv run pytest
```

Language codes follow the source data: `eng` English, `grc` Greek, `hbo` Biblical
Hebrew, `arb` Arabic. Missing requested languages are an error. The source build
report lists included, missing and excluded inputs; a broken present source is an
error rather than an apparently successful partial build.

The built corpus selects **one primary manuscript witness per language and book**,
so alternate copies do not inflate the analysis. The current sources contain
242,519 parsed units; 154,158 primary units are analyzed and 88,361 alternate
manuscript units remain in the original inputs. These include secondary CNTR,
MAM and Samaritan readings. Every verse in the canonical input receives an output
row. See `data/processed/build_report.json` for each witness choice; the explorer
also shows this scope. A custom JSONL corpus analyzes every supplied row directly.

## What the output means

- **Inferred author** is an anonymous language-wide group, such as `eng-A001`.
  The same ID means the same estimated group in every collection and book of
  that run. Different languages are never matched. IDs may change after reruns
  with different texts or settings; they are not permanent person identifiers.
- **Estimated authors** counts distinct inferred groups. Parent totals use a
  union of author IDs, so an author appearing in three books is counted once.
- **Reference author** is an independently supplied English catalogue label. It
  is displayed separately and is never a discovery feature or supplied count.
- **Low evidence** marks an assigned verse whose own text is short, whose margin
  is small, whose manuscript has a gap, or whose group split is not stable.
- **Insufficient text** is an explicit unassigned tag. Every input verse remains
  in the output, including empty or short units. Zero groups with unassigned
  verses means there is insufficient evidence, not that there were zero writers.
- **Distance margin** compares a passage's distance to its assigned group center
  with the nearest other center. It is not a probability of correct authorship.

In the explorer, choose a language and drill into collections, books and chapters.
Search verse text or filter by an inferred author to find related passages across
books in that language. Text explorer, Hierarchy, and Author groups share the
corpus filters and search. Their counts are recomputed from matching text, even
when viewing a parent rollup with a chapter or evidence filter selected. Opening
an author or hierarchy row keeps existing filters; hierarchy breadcrumbs change
the displayed level without broadening the selection. Use the sidebar or Reset
filters to broaden the corpus. English prose uses paragraphs as verse-like text units;
the input adapter preserves chapter headings where available.

The **Contributions** tab charts each inferred author's share of the selected
language, collection, book or chapter. Switch between word counts and
verse/paragraph counts. Percentages include unassigned text; words count each
verse's own tokens once, without repeating its supporting passage. Selecting an
author highlights its bar while keeping the full selection as the denominator.
Text search and evidence filters do not change contribution totals.

## Analysis

Each language is fitted independently. Whole verses are pooled into disjoint
passages targeting 1,200 words, with at least 200 words needed for analysis.
Passages can span chapters within a book; they never cross a book, collection,
language or explicitly marked manuscript gap. Tiny tails may join the preceding
passage in the same uninterrupted book segment. A long verse remains intact.
All member verses receive that passage's assignment, and the exact passage
membership is exported for inspection. These are contextual verse tags, not
independent authorship tests of each short verse.

The text representation combines frequent-word rates and character patterns,
reduces them to at most 12 dimensions, and fits diagonal Gaussian mixtures.
The model compares 1 through at most 20 groups using BIC; 13 is never provided
to discovery. A multi-group model must improve BIC over one group by at least
10 and have at least three sampled passages per group. At most 2,000 passages
per language train the representation and mixture; all eligible passages are
then assigned. The seed makes this sampling reproducible. The report includes
every candidate score, the search ceiling, silhouette and three 80% subsample
agreement checks. These checks hold the feature map and group count fixed.

You can change the passage size, minimum evidence, search ceiling and fit sample:

```sh
uv run stylometry analyze --passage-tokens 1200 --min-tokens 200 \
  --max-authors 30 --fit-passages 3000 --seed 42
```

Authorship is an inference: genre, topic, translation, transmission and editorial
practice can also produce style differences. The displayed count is an
exploratory estimate, especially when subsample agreement is low or the selected
count reaches the search ceiling. One group is not proof of one author.
English performance does not establish accuracy in the other languages.

## English validation

The report keeps two different tests separate:

1. **Blind discovery:** after clustering without author labels, compare the
   discovered groups with the known authors using count error, adjusted Rand
   index and normalized mutual information. The group/author table shows splits
   and merges. Agreement is weighted by verses and is not a held-out score.
2. **Held-out book attribution:** a separate classifier learns labelled training
   books and predicts entire unseen test books from the same known candidates.
   Vocabulary, feature scaling and classifier fitting use training books only.
   Passage and book accuracy, balanced accuracy, per-author results and the
   exact split are exported. Up to 20 passages per work are sampled across the
   work. Disputed/joint Federalist papers are excluded from labelled scoring.

The expected 13 names are checked for source completeness. Neither matching the
number 13 nor a high closed-set attribution score by itself validates discovery
of unknown historical authors. Actual scores are generated by each run, never
hard-coded as passing results.

## Files

| File | Contents |
| --- | --- |
| `data/processed/verses.jsonl` | Canonical input records |
| `data/processed/build_report.json` | Input coverage and source accounting |
| `output/index.html` | Self-contained interactive explorer |
| `output/report.json` | Full result with tags, author groups, rollups and diagnostics |
| `output/verses.csv` | One row per verse, including reference and inferred authors |
| `output/rollups.csv` | Language, collection, book, chapter and verse summaries |
| `output/passages.json` | Exact supporting passage membership |
| `output/benchmark.json` | English benchmark and post-hoc discovery comparison |

Custom JSONL input uses `id`, `language`, `collection`, `book`, `book_title`,
`chapter`, `verse`, `text`, and optional `reference_author`, `source`, `witness`,
`has_gap`. IDs must be unique; chapter and verse references are strings. Supply
records in reading order. Generated artifacts are ignored by Git and can be
rebuilt. Original source licences still apply to the texts and derived exports;
licence/readme files remain alongside their sources in `data/raw/` and the old
source catalogue is retained in `_backup/README.md`.
