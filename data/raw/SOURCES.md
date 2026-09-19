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
| `benchmarks/` | Reference works by catalogued authors | per-manifest, recorded with each entry | See `benchmarks/*_manifest.json` in the repository root |

## What the Bukhari loader keeps

`data/raw/bukhari/` holds the whole collection, chain and all, exactly as published. The loader keeps
only the *matn*, the body of each report, and drops the *isnad*, the chain of transmitters prefixed to
it — 28% of the text. That is a judgement made in code, in `stylometry/corpus/bukhari.py`, not an edit
to the file on disk: the raw text stays complete so the decision can be checked or reversed.

## What the Hadith Qudsi loader keeps

The same treatment as Bukhari — matn only, chain dropped — plus the closing "narrated by X" citation
this edition appends to 39 of its 40 reports. The collection is small on purpose and is labelled so:
40 reports, about 2,400 tokens, one work. It is there as a third category beside the Qur'an and
Bukhari, not as something to cluster.

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
  Text-Fabric loads to satisfy its declared text formats. Parsing the trimmed set yields the same
  7,824 units from 178 witnesses as the full download.

Together these take `data/raw/` from about 770 MB to the 234 MB checked in without changing a single parsed unit.
`scripts/download_sources.sh` refetches anything missing, and `stylometry manifest --check` verifies that
what is present still matches the checked-in manifest.
