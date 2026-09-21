# Sources checked into this repository

The texts in this directory are redistributed from the projects below so the corpus can be rebuilt from
this repository alone. Each keeps the licence it was published under, and those licences are why this
file exists: most of them require attribution when the text is passed on, and one requires that it be
passed on unchanged.

**Nothing here is the work of this project.** The transcriptions, editions and encodings are the work of
the people and institutions named. The only files this project generates are under `data/processed/` and
`output/`, neither of which is checked in.

| Directory | Text | Licence | Attribution |
|---|---|---|---|
| `codex-sinaiticus/` | Codex Sinaiticus, full transcription | CC BY-NC-SA 3.0 | Codex Sinaiticus Project; ITSEE, University of Birmingham |
| `cntr/class1/` | 130 New Testament witnesses to c. AD 400 | CC BY-SA 4.0 | Center for New Testament Restoration, Alan Bunning |
| `vaticanus/swete/` | Greek Old Testament, Swete's edition | CC BY-SA 4.0 | Swete's text, digitised by the lxx-swete project |
| `first1kgreek/` | Early Greek works | CC BY-SA 4.0 | Open Greek and Latin / First1KGreek |
| `apostolic-fathers/texts/` | Apostolic Fathers, Lake's Greek text | CC BY-SA 4.0 | Open Apostolic Fathers project (jtauber) |
| `oshb/` | Westminster Leningrad Codex | Public domain text | Open Scriptures Hebrew Bible |
| `mam/` | Miqra according to the Masorah, Aleppo-based | CC BY-SA 4.0 | MAM, Benjamin Denckla |
| `samaritan/tf/` | Samaritan Pentateuch, Text-Fabric | CC BY-NC 4.0 | DT-UCPH Samaritan Pentateuch dataset |
| `dss/tf/2.0.1/` | Dead Sea Scrolls, Text-Fabric | CC BY-NC 4.0 | Abegg, Bowley and Cook; converted by Jacobs, Naaijer and Roorda (ETCBC) |
| `inscriptions/` | Ketef Hinnom amulets, Nash Papyrus | Published readings | Editors' published readings, transcribed here |
| `quran/` | Quran, Uthmani text and sura metadata | Tanzil terms | Tanzil.net. The terms require a verbatim copy with attribution; this copy is unmodified. |
| `bukhari/` | Sahih al-Bukhari, Arabic, with book divisions | Unlicense (public domain dedication) | hadith-api, Fawaz Ahmed. The text itself is 9th-century and long out of copyright; the Unlicense covers the digital edition. |
| `qudsi/` | Forty Hadith Qudsi, Arabic | Unlicense (public domain dedication) | hadith-api, Fawaz Ahmed. Text long out of copyright; the Unlicense covers the digital edition. |
| `english/` | Federalist Papers, 15 novels, 20 cross-genre works | Public domain in the United States | Project Gutenberg. The texts are out of copyright; PG's own licence covers its trademark and its front and back matter, which the loader strips and this project does not redistribute. |
| `benchmarks/` | Reference works by catalogued authors | per-manifest, recorded with each entry | See `benchmarks/*_manifest.json` in the repository root |

## What the hadith loaders keep

`data/raw/bukhari/` and `data/raw/qudsi/` hold the complete reports as published.
The fresh importers preserve the entire transmitted report, including the isnad,
framing, and citation notes. They do not infer a speaker boundary or assume that
quoted speech identifies the historical author. All numbered Bukhari reports are
retained, including records in unassigned book 0 and an empty source report, which
receives an insufficient-text result. Qudsi HTML line-break tags are removed.

## Witness and fragment policy

The import parses every supported source, then chooses one primary witness per
language and book. A preferred manuscript must have at least 90% of the fullest
witness's token coverage. Every alternate witness and its unit count is recorded
in `data/processed/build_report.json`; alternate editions are not counted as
independent authored works. Nonempty Dead Sea Scrolls fragments and reconstructed
readings are retained with reconstruction metadata. Repeated references within a
selected manuscript receive distinct IDs while preserving their source references.

## Why there is English here

The other three languages have no ground truth. Nobody can say who wrote Isaiah, so nothing measured
on it can be scored, and a strategy that looks convincing there might be measuring genre, length or
the editor's punctuation. Every English text has a settled author. Three collections test three
things: the Federalist papers hold genre and period constant across three authors, the novels give
several long works per author for whole-work holdout, and the cross-genre set has the same hand
writing fiction and essays, which is where a strategy that tracks genre rather than authorship shows
itself.

The English importer keeps full body paragraphs, including short dialogue, with no
word cap and no percentage-based skip. Verified opening-text anchors remove title
blocks, contents, and editorial prefaces; Gutenberg wrappers, illustrations,
transcriber notes, and verified publisher advertisements are excluded. Printed
Federalist bylines, publication metadata, salutations, and signatures are excluded
from analysis. Undisputed works provide the 13 reference authors; the 12 disputed
and three joint Federalist papers remain in the corpus with no single-author
reference label. Prose paragraphs stand in for verses. Sequential section numbers
are used for English chapters, with available printed headings retained separately.

## Non-commercial terms

Codex Sinaiticus, the Samaritan Pentateuch and the Dead Sea Scrolls datasets are non-commercial
licences. Any output derived from them, including everything this pipeline produces, inherits that
restriction. The Sinaiticus licence is also share-alike.

## What is not checked in, and why

Only material no loader reads:

- `codex-sinaiticus/archive/` and `beta-versions_not-for-release/`: superseded releases, and files the
  project marks as not for release. The loader reads `sinaiticus_full_v195.xml`.
- `apostolic-fathers/` everything but `texts/`: the upstream project's build scripts, intermediate
  conversions and a second analytical edition.
- 49 of the 80 Dead Sea Scrolls Text-Fabric features: morphological and lexical layers this project
  never opens. The 31 kept are the ones the loader requests plus the ones `otext.tf` names, which
  Text-Fabric loads to satisfy its declared text formats. The retained features include the text and reconstruction annotations needed by
  the current importer.

The original download and manifest scripts are preserved under `_backup/`. The fresh
`stylometry build` command reports missing sources and fails if a present supported
source cannot be parsed. Auxiliary `benchmarks/` downloads are reported separately
and excluded from the primary corpus.
