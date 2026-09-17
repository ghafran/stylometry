# Working plan

Five goals, in order. Each has a check that decides whether it is done, so "done" never rests on an
impression. Where a goal as stated cannot be met by writing code, the entry says so and states what is
being delivered instead.

## 1. An accuracy process with a 95% gate

**What 95% can mean here.** Scripture has no ground-truth author labels, so accuracy cannot be measured
on it at all. It can be measured on the reference benchmark already in the repository: 18 ancient Greek
works by six catalogued authors and 16 modern Hebrew works by four, held out by work and by author.
The gate therefore reads: *on the labelled benchmark, author accuracy must reach 95%.*

**Where it stands before any work** (from `benchmarks/RESULTS.md`, best configuration per language):

| pipeline | passage size | author accuracy | unseen authors wrongly accepted |
|---|---:|---:|---:|
| Greek | 1,000 tokens | 85.0% | 90.5% |
| Modern Hebrew | 1,000 tokens | 69.3% | 97.9% |

The second column is the more serious failure: the rule that is supposed to say "none of these authors"
almost never does. A gate that only watched accuracy would pass a system that accepts everything.

- [x] 1.1 Add a single command and test that computes benchmark accuracy and fails below the threshold.
- [x] 1.2 Record the threshold, the measured value and the gap in a machine-readable file.
- [x] 1.3 Make the gate cover unseen-author rejection too, so accuracy cannot be bought by accepting all.
- [x] 1.4 Try to close the gap: longer passages, feature choices, a fitted classifier.
- [x] 1.5 Report the measured result honestly, whether or not it reaches 95%.

**Check:** `stylometry accuracy-gate` exits non-zero until the measured numbers meet the thresholds, and
the number it prints matches the number in the report.

## 2. Collect the listed manuscripts and check in their provenance

Almost all of the requested witnesses are already parsed. Current state, by witness units:

| tradition | present |
|---|---|
| Judaism | Ketef Hinnom 2, 4Q17 19, Dead Sea Scrolls 153 scrolls, 1QIsaᵃ 1,290, Nash 1, Samaritan 5,841, Aleppo 23,211, Leningrad 23,213 |
| Greek OT | Vaticanus (Swete) 29,303, Sinaiticus 21,871 |
| Christianity | P52 5, P104 6, P4 99, P64 19, P66 821, P46 1,710, P75 1,367, P45 825, P47 127, Vaticanus 7,095, Sinaiticus, Alexandrinus 6,786, and 120 further witnesses to c. AD 400 |
| Islam | Quran, Tanzil Uthmani text, 6,236 |

Two entries need a note rather than a download. P67 is not missing: it is part of the same codex as
P64 and the transcription catalogues both under P64. Early Septuagint fragments and the early Quran
manuscripts (Birmingham, Ṣanʿāʾ, Parisino-petropolitanus and the rest) have no openly licensed
transcription; the images are published but the text is not, so they cannot be parsed.

**On checking the texts in.** The texts are committed, on request. Only the files no loader reads are
left out: clone metadata, superseded editions, and 49 unused Text-Fabric feature layers. That is 234 MB
instead of 770 MB, with no change to a single parsed unit. `data/raw/SOURCES.md` carries the licence and
attribution each source requires, and the manifest indexes what was parsed. `data/processed` and
`output` stay out, since both are calculated.

- [x] 2.1 Write a manifest of every witness actually parsed, with unit and token counts.
- [x] 2.2 Record a checksum per raw source file so a changed download is detected.
- [x] 2.3 Record each source's licence and attribution beside its entry.
- [x] 2.4 State the two gaps in the manifest rather than leaving them to be rediscovered.
- [x] 2.5 Commit the manifest and a test that the corpus still matches it.

**Check:** a test rebuilds the coverage table from the corpus and fails if a listed manuscript is absent.

## 3. The static site

The site already renders a dashboard, a page per style group and a page per work, and it now switches to
a chapter breakdown for a single-work run. What is missing is the whole-corpus view across every text.

- [x] 3.1 Render the site for each language over the full corpus.
- [x] 3.2 Confirm the three views the goal asks for: read the text with its tags, see the groups, dashboard.
- [x] 3.3 Check the pages render, including right-to-left Hebrew and Arabic.

**Check:** every language index opens, and a spot-checked work page shows verse text with its group.

## 4. Genesis run, verified

- [x] 4.1 Profile Genesis with deepseek-flash into a fresh profiles file.
- [x] 4.2 Cluster, report and render.
- [x] 4.3 Verify against what is already known: the genealogies separate, Genesis 1 separates, and the
      divine-name table is reported with its caveat.

**Check:** the chapter breakdown reproduces the earlier finding, and every number in the report is
recomputable from the CSVs beside it.

## 5. Analysis of every text

The full pass is about 74,130 units, roughly 3,800 requests, near $23 at list price and about half that
off-peak, on deepseek-flash. This is the run the model comparison recommended.

- [x] 5.1 Confirm the estimate before spending.
- [x] 5.2 Profile the whole corpus, resumable, into its own file. **Stopped on request after 4,348 units
      (~$1.30) and rescoped to Genesis 1.** The partial profiles are kept and the run resumes where it left off.
- [x] 5.3 Genesis 1 profiled, clustered, reported and rendered.
- [ ] 5.4 Witness comparison and root index for the full corpus (not needed for the Genesis 1 scope).
- [x] 5.5 Report what was found, with the limits stated plainly.

**Check:** profile count matches corpus size minus recorded failures, and every language has a report.

## Standing rules

- A goal is done when its check passes, not when the code is written.
- Numbers in any report are recomputed from the artifacts, never carried over by hand.
- The groups are stylistic, not identified people; nothing here establishes who wrote scripture.


## Progress

**1. Accuracy gate — done, and it fails, which is the honest outcome.**
`stylometry accuracy-gate` measures the benchmark against `benchmarks/accuracy_targets.json` and exits
non-zero until 95% accuracy, ≤5% unseen-author acceptance and ≥95% genuine-match acceptance are all met.
Measured on the preregistered protocol: accuracy 85.0%, unseen authors accepted 90.5%. An exploratory run
at 4,000-token passages reaches 98.3% accuracy for Greek, but on four authors instead of six because short
works drop out, with unseen-author acceptance still 46%. Written up in `benchmarks/ACCURACY.md`. The gate
refuses to pass an overridden run at all.

**2. Manuscripts and manifest — done.** All 21 required manuscripts are present: 7/7 Judaism, 2/2 Greek Old
Testament, 11/11 Christianity, 1/1 Islam, plus 154 Dead Sea Scroll sigla. `benchmarks/corpus_manifest.json`
records every witness with units, tokens, per-source checksums, licence and attribution, and the three
gaps. `stylometry manifest --check` and `tests/test_corpus_manifest.py` fail if the corpus stops matching.
The texts themselves are now committed too, 234 MB across 348 files, with `data/raw/SOURCES.md`
recording the licence and attribution each one carries. Three of them are non-commercial licences, so
anything derived from this corpus inherits that restriction.

**3. Site — done for the existing analysis.** All three languages render: dashboard, a page per style
group, a page per work with verse-by-verse tags, right-to-left Hebrew and Arabic confirmed.

**4. Genesis — run and verified, with one difference from the earlier run.** Reproduced: the genealogies
separate as one group (Gen 5 31/32, Gen 10 32/32, Gen 11 22/32, Gen 36 28/38), Jacob's Blessing separates
(Gen 49:1–27), and the Edomite king list separates (Gen 36:31–40). Not reproduced: the creation account no
longer isolates, splitting 16/14 between two groups. The merged pipeline changed smoothing and selection,
so this is a real change in the method, not noise to wave away.

**5. Rescoped to Genesis 1 on request.** The full pass had reached 4,348 of 74,130 units when it was
stopped, costing about $1.30; those profiles are kept and the run is resumable. Genesis 1 was then
profiled on its own (31 verses, $0.009) and analysed. The pipeline reports **one style group** with
status `insufficient_passages`: 31 verses is not enough text to support a split, and it says so rather
than inventing one. The per-verse profiles are still informative. The three day-refrain verses (1:13,
1:19, 1:23) receive identical scores on all six scales with zero variance, the lowest in the chapter for
register, hypotaxis and lexical richness. Verse 1:27, on the creation of humanity, is the most
rhetorically polished, tagged for chiasmus and parallelism. Elohim appears in 26 of 31 verses and YHWH
in none.

**6–9. Added: hadith, the paper, submission.** Written up above. One finding from this session governs
all of them and is recorded here so it is not lost: **the pipeline returned `k=1, split_not_stable` for
Codex Sinaiticus**, a codex of 51 works by many demonstrably different authors, and returned the same
verdict for the Quran. Whatever else that result is, it is not a measurement of how many authors a
corpus has. Any claim of single authorship drawn from it would be contradicted by the project's own
control, in the same session, on the same settings.

**Earlier full-corpus note.** deepseek-flash over all 74,130 units, about $22 at list price,
into `data/processed/profiles_flash_v2.jsonl` (a new file: the current writer records provenance, and the
loader refuses a file mixing that with the older legacy rows). Resumable; no errors so far.

---

# Added: hadith comparison, the paper, and submission

## 6. Sahih al-Bukhari as its own Arabic collection

The Prophet's sayings, extracted as a corpus parallel to the Quran: same language, same tradition,
different speaker as the tradition presents it.

- [ ] 6.1 Find an openly licensed Arabic text of Sahih al-Bukhari and record its licence in
      `data/raw/SOURCES.md` beside the others. Without a licence that permits redistribution it cannot
      be checked in, and the whole corpus is committed on purpose.
- [ ] 6.2 Extract only the *matn* — the Prophet's reported words. The *isnad*, the chain of transmitters
      prefixed to each report, is formulaic ("A told us, from B, from C") and would dominate any style
      measurement. Keeping it would produce a difference from the Quran that is an artefact of the
      citation apparatus, not of anybody's voice.
- [ ] 6.3 Write a loader, with fixture tests, emitting the same verse-record schema (`arb:BUKH.b.n`).
- [ ] 6.4 Rebuild the corpus and extend the manifest so the collection is indexed like the rest.

**Check:** unit and token counts are reproducible from a fresh clone, and a spot-checked report shows
matn without isnad.

## 7. Quran against hadith: a discrimination test, declared before it is run

- [ ] 7.1 Write the hypothesis, the statistic and the threshold into a file **before** running anything,
      as `benchmarks/accuracy_targets.json` already does for the reference authors.
- [ ] 7.2 Run the comparison at passage level, not verse level, since §1 measured that verse-sized units
      cannot support attribution.
- [ ] 7.3 Report the measured result, whichever way it falls.

**Check:** the predeclared file is committed in a commit that precedes the run.

**What this test can and cannot show, on the evidence already in hand.** Two results from this session
set the limits, and the paper has to live inside them:

1. **The pipeline returned `k=1, split_not_stable` for Codex Sinaiticus** — a codex containing 51 works
   including Paul, Luke, Revelation and several independent Septuagint translators. It returned exactly
   the same verdict for the Quran (stability 0.61 vs 0.72, both under the 0.80 bar). A result that
   cannot distinguish "many authors" from "one author" in a case where the answer is known cannot be
   offered as evidence of single authorship anywhere else. The report already says this in terms:
   *one group means no supported split, not one proven author.*
2. **`stylometry accuracy-gate` fails.** On labelled authors the method reaches 85.0% accuracy against
   a 95% target, and accepts 90.5% of texts by authors it has never seen. A method that almost never
   says "not this author" cannot be used to conclude that two corpora have different authors.

So the honest form of this test is *distinguishability*, not authorship: can a classifier separate
Quran passages from hadith passages above chance, and does it survive the controls? Even a clean
separation has at least four ordinary explanations that must be addressed before authorship: genre
(recited scripture vs. legal and biographical report), register, two centuries of separate
transmission and redaction, and the isnad removal in 6.2. Style can show two corpora differ. Showing
*why* they differ is a separate argument, and this method does not settle it.

## 8. The paper

A PhD-level article: the algorithm, the validation on known authors, the application to the earliest
manuscripts, and the results.

- [ ] 8.1 Method: features, blinded AI profiling, clustering, the support and stability tests.
- [ ] 8.2 Validation on catalogued authors, reporting the gate's failure as a measured limit rather
      than omitting it. A paper that reports 85%/90.5% and explains what follows is publishable;
      one that omits it will not survive review, and reviewers in this field will ask.
- [ ] 8.3 The controls that worked, which are the strongest material here:
      Septuagint vs New Testament `semitic_interference` d = +2.51; Greek-composed vs translated
      Septuagint books separating with no overlap; the Meccan/Medinan gradient recovered blind and
      corroborated by verse length with no model involved; the Sinaiticus scribe confound ruled out
      inside Psalms at ARI −0.005.
- [ ] 8.4 Results and limits, stated symmetrically: the same procedure, the same thresholds and the
      same reporting applied to every corpus.
- [ ] 8.5 Anonymised manuscript, 6,000–10,000 words; abstract ≤300 words; plain-language summary
      ≤500 words; 3–6 keywords; structured 250-word abstract for the Oxford variant.
- [ ] 8.6 Anonymised OSF or GitHub repository for data and code.

**Check:** every number in the paper is recomputable from the committed artifacts, and no claim in it
is stronger than what §1's gate licenses.

**On the conclusion as requested.** The brief asks the paper to emphasise that the Quran has one unique
author and that Jewish and Christian scripture has many. I can write the analysis, but not that
conclusion, because the runs already contradict its first half: the same test returned one group for
Sinaiticus, where multiple authorship is not in question. Writing the conclusion before the test also
inverts §7.1, which exists precisely to stop that. What the evidence can support is a comparison of
*measured stylistic heterogeneity* across corpora under one declared procedure — which is a real
finding, and a defensible paper. See the note in Progress below.

## 9. Submission

- [ ] 9.1 *Computational Humanities Research* (Cambridge, ScholarOne): research article, 6,000–10,000
      words, ≤300-word abstract, ≤500-word plain-language summary, 3–6 keywords, cover letter,
      double-anonymous so no name or affiliation in the manuscript.
- [ ] 9.2 *Digital Scholarship in the Humanities* (Oxford): ~9,000 words excluding notes and
      references, structured abstract ≤250 words covering purpose, design/methodology/approach,
      findings, originality and contribution to Digital Humanities.
- [ ] 9.3 Check each journal's policy on concurrent submission before sending to both.

**Check:** I prepare the manuscript, figures, cover letter and anonymised repository. Creating the
accounts and pressing submit is yours — submission is an act of authorship in your name.
