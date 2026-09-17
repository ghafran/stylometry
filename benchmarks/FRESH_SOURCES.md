# Fresh Greek reference sources

**Exposure status:** the locked evaluation completed on 2026-09-17 at 01:18:57 UTC.
These sources are now exposed evaluation data. The acquisition account below records their
status before that run. See the [frozen results](study_v1/RESULTS.md); future tuning needs a
new reserved final panel. The manifests remain unchanged after sealing.

`greek_fresh_manifest.json` was curated for a new evaluation after the original
six-author Greek pilot became exposed evaluation data. Its eight authors are
absent from that pilot. Source selection used catalogue metadata, source
authenticity review, text extraction and minimum lengths. The curator did **not**
run a classifier, fit a model, inspect feature vectors, obtain predictions or
calculate classification performance on these texts.

“Fresh” describes exposure to performance evaluation at acquisition. It does not
mean that the sources were unread: the curator inspected metadata and prose for
data quality. Subsequent evaluation must record that the corpus has been used;
it must not remain labelled an untouched test set after tuning against results.

## Predeclared author roles

| Evaluation role | Author | Independent works | Normalized tokens |
|---|---|---:|---:|
| Reference | Aeschines | 3 | 43,792 |
| Reference | Isaeus | 6 | 19,386 |
| Reference | Plutarch | 6 | 44,256 |
| Reference | Dionysius of Halicarnassus | 3 | 40,030 |
| Calibration unfamiliar | Antiphon | 2 | 9,748 |
| Calibration unfamiliar | Aelius Aristides | 3 | 7,647 |
| Final unfamiliar | Andocides | 2 | 11,158 |
| Final unfamiliar | Galen | 1 | 31,807 |
| **Total** | **8 catalogue authors** | **26** | **207,824** |

The manifest records these assignments as `evaluation_role` values `reference`,
`calibration_unknown` and `test_unknown`. The existing loader's separate `role`
field remains `attribution`; a caller must actually enforce the evaluation-role
partition. This manifest alone cannot stop accidental use of final unfamiliar
authors in calibration. The frozen evaluation protocol must do so.

The initial source proposal placed Antiphon among reference authors. Source
research identified specific doubts about *Against the Stepmother*. Before any
scoring, that text was excluded and the roles were revised: Antiphon contributes
his two remaining court speeches to unfamiliar-author calibration; Dionysius,
with three independent suitable works, becomes a reference author. The reason
was attribution and source availability, not observed performance.

## Works and selection

All files come from the [Perseus canonical Greek repository](https://github.com/PerseusDL/canonical-greekLit/tree/9839fe84c883e47ca5dab8872d8c684d9e14fe14)
at commit `9839fe84c883e47ca5dab8872d8c684d9e14fe14`. The manifest includes
immutable download URLs, the source and normalized-text SHA-256 checksums,
expected token counts, catalogue checksums, source editions and licenses.

* **Aeschines:** *Against Timarchus*, *On the Embassy*, *Against Ctesiphon*
  (`tlg0026.tlg001`–`tlg003`), three distinct court speeches.
* **Isaeus:** the estate speeches for Cleonymus, Menecles, Pyrrhus, Dicaeogenes,
  Philoctemon and Apollodorus (`tlg0017.tlg001`, `002`, `003`, `005`, `006`, `007`).
  The intervening Nicostratus speech was excluded because the extracted body is
  shorter than 2,000 tokens. All six retained works are separate legal cases.
* **Plutarch:** *How to Tell a Flatterer from a Friend*, *How a Man May Become
  Aware of His Progress in Virtue*, *On Superstition*, *On the E at Delphi*,
  *On the Pythian Oracles*, and *On the Obsolescence of Oracles*
  (`tlg0007.tlg070`, `071`, `080`, `090`, `091`, `092`). These are separately
  catalogued compositions within the *Moralia*, not artificial chapter splits.
* **Dionysius of Halicarnassus:** *On Thucydides*, *On Literary Composition*,
  *Letter to Pompeius Geminus* (`tlg0081.tlg010`, `012`, `015`). These critical
  compositions quote earlier authors. The unchanged parser removes marked
  quotations, but unmarked reuse can remain, including material from writers in
  the old pilot. They do not provide content-independent authorship evidence.
* **Antiphon:** *On the Murder of Herodes* and *On the Choreutes*
  (`tlg0028.tlg005`, `006`). Exclude the *Tetralogies* and *Against the Stepmother*.
* **Aelius Aristides:** prose hymns *To Zeus*, *Athena*, *Isthmian Oration to
  Poseidon* (`tlg0284.tlg001`, `002`, `003`). These are prose, not verse hymns or
  forensic speeches. Catalogue numbering must not be silently substituted for
  numbering in other editions; the CTS work identifiers and Greek edition
  labels identify exactly which works were selected.
* **Andocides:** *On the Mysteries* and *On His Return*
  (`tlg0027.tlg001`, `002`). The latter is a public deliberative appeal, not a
  forensic case. Exclude *Against Alcibiades* and *On the Peace with Sparta*.
* **Galen:** the complete *On the Natural Faculties* (`tlg0057.tlg010`). Its
  three internal books count as **one** independent work.

All 18 reference works contain more than 2,000 normalized Greek tokens. The
smallest final unfamiliar work, Andocides' *On His Return*, has 1,945. The source
cache holds complete source files; no model-dependent excerpting or rewriting
was performed during acquisition. Existing `tei-greek-v1` extraction was used
without modification, followed by the application's Greek tokenizer.

Each selected TEI file explicitly licenses the digital edition under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Credit the Perseus
Digital Library at Tufts University and the individual editors/publishers in
the manifest. Cached source files remain outside version control.

## Attribution review and limits

These labels are conventional catalogue attributions, not proof that the
historical composition of each work was simple or exclusively personal. For
speechwriters, the label means the attributed composer, not the litigant whose
voice the speech adopts. Authored dialogue, borrowed ideas and unmarked
quotations may survive extraction. A shared modern editor, edition tradition,
genre, subject, chronology or register can still correlate with an author.

The exclusions were informed by primary source editions and scholarly research:

* [Dobson's discussion of Antiphon's works, hosted by Perseus](https://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A1999.04.0075%3Achapter%3D2%3Asection%3D7)
  notes historical questions about *Against the Stepmother*. It was omitted
  rather than treated as secure ground truth.
* [Research on *Against Alcibiades*](https://www.cambridge.org/core/journals/classical-quarterly/article/abs/rhetoric-and-history-in-andocides-4-against-alcibiades/17E8048D4574313742573A3AAF9AE4D9)
  and [research disputing *On the Peace*](https://ejournals.epublishing.ekt.gr/index.php/tekmiria/article/view/28450/0)
  support excluding those attributed Andocides speeches.
* [The Perseus edition introduction to *On the E at Delphi*](https://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A2008.01.0243)
  identifies the Delphic work and its context. The selection excludes commonly
  designated Pseudo-Plutarch texts rather than assigning them to Plutarch.
* [Research on Aristides' prose hymns](https://academic.oup.com/book/5959/chapter-abstract/149319319)
  supports treating those texts as epideictic prose with a distinct generic
  context. They are not a matched substitute for Attic litigation speeches.

Genre/topic labels are broad manual annotations, not a blinded scholarly
matching study. Court oratory provides overlap among reference Aeschines and
Isaeus, calibration Antiphon, and final Andocides. Aristides' hymns and Plutarch's
religious essays/dialogues overlap in subject but differ in genre. Galen's
medical subject is particularly distinctive, so rejection of him alone would
not demonstrate discrimination between writers on closely matched topics.

There are only four candidate/reference authors, two calibration unfamiliar
authors and two final unfamiliar authors. The eight labels must not be reported
as eight independent candidate identities or eight unknown-author trials.
The 26 works do not satisfy an eight-reference-author/60-work coverage standard.
One Galen treatise cannot establish general unknown-author rejection reliability.
Edited literary Greek also does not validate scripture attribution, Koine
transfer, historical author counts, manuscripts, Hebrew or Arabic.

# Fresh Hebrew reference sources

`hebrew_fresh_manifest.json` adds eight catalogue authors absent from the old
Brenner, Gnessin, Berdyczewski and Bershadsky pilot. Acquisition used only source
metadata, rights, prose inspection, extraction, token counts and checksums. No
classification, feature-vector analysis, model fitting, predictions or empirical
benchmark was run on these sources by the curator. The same performance-exposure
meaning of “fresh” and the same requirement to enforce `evaluation_role` apply.

## Predeclared Hebrew author roles

| Evaluation role | Author | Independent works | Normalized tokens |
|---|---|---:|---:|
| Reference | Yehuda Steinberg | 6 | 23,312 |
| Reference | Asher Barash | 6 | 24,361 |
| Reference | Moshe Smilansky | 6 | 33,604 |
| Reference | Nechama Pukhachevsky | 6 | 37,418 |
| Calibration unfamiliar | David Frischman | 2 | 11,255 |
| Calibration unfamiliar | S. Ben-Zion | 2 | 7,860 |
| Final unfamiliar | Aharon Abraham Kabak | 2 | 24,695 |
| Final unfamiliar | Alexander Ziskind Rabinovich | 2 | 27,650 |
| **Total** | **8 catalogue authors** | **32** | **190,155** |

The reference partition contains 24 works and 118,695 tokens; unfamiliar-author
calibration contains four works and 19,115 tokens; final unfamiliar testing
contains four works and 52,345 tokens. These remain only four reference author
labels and two independent final unfamiliar author labels.

The parent approved the proposed author roles before scoring. During the
source-only audit, the provisional final unfamiliar author Jacob Steinberg was
replaced by Alexander Ziskind Rabinovich because Hebrew/Yiddish self-translation
complicated the original-language criterion. This source-language decision was
made before any performance evaluation. Yehuda Steinberg is a different writer
and retains his reference role.

## Hebrew works and provenance

Sources come from the [Project Ben-Yehuda public-domain repository](https://github.com/projectbenyehuda/public_domain_dump/tree/5e4277fead7b565f32cc4b352abd3565023a77d8)
at commit `5e4277fead7b565f32cc4b352abd3565023a77d8`, using its
[pinned catalogue](https://github.com/projectbenyehuda/public_domain_dump/blob/5e4277fead7b565f32cc4b352abd3565023a77d8/pseudocatalogue.csv).
The source IDs below are repository work IDs, each prefixed `benyehuda.m` in the
manifest. The manifest records Hebrew titles, author authority links, source
editions where available, immutable URLs, source and normalized-text SHA-256
checksums, expected token counts and the catalogue checksum.

| Author | Selected catalogue work IDs |
|---|---|
| Yehuda Steinberg | 458, 953, 1550, 2169, 2232, 2418 |
| Asher Barash | 34665, 34668, 34678, 34679, 34681, 34683 |
| Moshe Smilansky | 21329, 21330, 21331, 21332, 23056, 23058 |
| Nechama Pukhachevsky | 305, 2550, 2872, 5686, 6335, 6658 |
| David Frischman | 229, 647 |
| S. Ben-Zion | 2062, 2376 |
| Aharon Abraham Kabak | 34, 13354 |
| Alexander Ziskind Rabinovich | 7953, 8036 |

Eligible entries had one catalogue author, prose genre, no listed translator or
foreign original language, and 2,000–30,000 tokens after unchanged extraction.
The smallest retained work has exactly 2,000 tokens. Selection favored complete
modern social and psychological narratives. It excluded obvious historical or
biblical retellings, folklore anthologies, multi-volume parts, chapter fragments,
ambiguous part/version titles, duplicate texts and unclear editorial-preface
layouts. Headings, openings and endings were inspected for work identity and
metadata leakage. Complete source files were retained; chapters were not turned
into separate works, and no performance-dependent excerpting was used.

For Yehuda Steinberg, the first 60 prose catalogue records were inspected to
find six suitable independent narratives. For Asher Barash, early historical
tales and long novels were skipped before selecting six eligible modern
narratives. For the other authors, the first 20 eligible catalogue entries were
inspected and modern prose was selected within this broad genre. These are
documented convenience selections, not random or representative samples.

The unchanged `benyehuda-html-v1` parser removes source metadata, headings,
explicitly marked footnotes/reference anchors and the marked publisher footer,
including in layouts with prose outside paragraph tags. The existing Hebrew
tokenizer then supplies the normalized text and count. All 32 normalized texts
have distinct hashes; no author label or canonical work ID overlaps the old
Hebrew pilot. These checks do not rule out quotations or partial textual reuse.

The repository's [public-domain license statement](https://github.com/projectbenyehuda/public_domain_dump/blob/5e4277fead7b565f32cc4b352abd3565023a77d8/LICENSE)
provides the recorded rights basis. Credit Project Ben-Yehuda and its volunteers.
Downloaded source files remain in the ignored local cache and are reproducibly
addressed by their immutable URLs and checksums.

## Hebrew attribution and transfer limits

Original-Hebrew status is a catalogue-based assumption supported by limited
source inspection. A blank translator or original-language field does **not**
prove that every publication has no self-translation, rewriting or mixed
language history. Catalogued authorship is a useful conventional reference
label, not a guarantee that every retained word was composed by that author.
Unmarked quotations, editorial interventions and shared edition conventions can
remain. Genre and topic are broad manual annotations rather than an independent
matching study; social setting, subject, period and register can correlate with
author identity.

The sources are **modern literary Hebrew**. `source_language: "heb"` records
that fact; the loader's `language: "hbo"` only selects the existing Hebrew-script
normalizer. It must not be reported as evidence that the texts are Biblical
Hebrew. This corpus does not establish Biblical Hebrew or Aramaic transfer,
named scripture attribution, historical author counts or Arabic reliability.

David Frischman candidates were read during initial source-feasibility research,
but were neither included in the old benchmark nor scored. All eight new labels
were unexposed to classification-performance evaluation when acquired. Once
evaluation results influence development, this corpus must be treated as
exposed data and fresh confirmation sources will be needed.

## Completed canonical-author identity audit

A separate metadata audit checked identity separation beyond display names. Greek CTS
textgroups in the old panel (`tlg0540`, `tlg0010`, `tlg0014`, `tlg0059`, `tlg0032`, `tlg0062`)
are disjoint from the fresh reference (`tlg0026`, `tlg0017`, `tlg0007`, `tlg0081`), unfamiliar
calibration (`tlg0028`, `tlg0284`) and final unfamiliar (`tlg0027`, `tlg0057`) groups.
The three fresh roles are also pairwise disjoint.

Hebrew publisher identities and Wikidata authority identifiers similarly separate the old
authors (`Q939732`, `Q514289`, `Q1926717`, `Q7064133`) from fresh reference authors
(`Q4526788`, `Q2866491`, `Q1948937`, `Q12409829`), calibration authors (`Q705482`, `Q2905102`)
and final unfamiliar authors (`Q399957`, `Q12403993`). The fresh roles are pairwise disjoint.
The corresponding Ben-Yehuda author IDs are reference 117/1274/727/70, calibration 142/76,
and final unfamiliar 13/111.

The pinned catalogue leaves Nechama Pukhachevsky's authority field blank. The independent
read-only audit resolved her as [Wikidata Q12409829](https://www.wikidata.org/wiki/Q12409829),
which links [Ben-Yehuda author 70](https://benyehuda.org/author/70). This supplemental resolution
does not alter the locked manifest. Catalogue identity separation does not itself prove
historical authorship or absence of borrowed text.
