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

## 6. Sahih al-Bukhari as its own Arabic collection — done

The Prophet's sayings, extracted as a corpus parallel to the Quran: same language, same tradition,
different speaker as the tradition presents it.

- [x] 6.1 Openly licensed Arabic text, recorded in `data/raw/SOURCES.md`. The edition is the Arabic
      Bukhari from **hadith-api** (Fawaz Ahmed), released under the **Unlicense** — a public-domain
      dedication, the least restrictive licence in the corpus. It was chosen over two alternatives on
      structure, not licence: the ODbL *Open-Hadith-Data* CSV is a flat list of 7,008 reports with no
      book divisions, so the whole collection would have been a single work. This edition carries
      `reference.book`, so each *kitab* becomes a work.
- [x] 6.2 Matn extracted, isnad dropped — **27% of the text removed**, 125,424 of 464,882 words.
- [x] 6.3 `stylometry/corpus/bukhari.py`, with eight fixture tests in the edition's own format.
- [x] 6.4 Corpus rebuilt and manifest extended; `manifest --check` passes, Islam 2/2.

| | Quran | Bukhari |
|---|---:|---:|
| units | 6,236 verses | 7,274 reports |
| works | 114 suras | 97 kutub |
| tokens | 77,881 | 337,030 |
| median unit | 10 tokens | 31 tokens |

**How the isnad is found.** The chain is dense in transmission verbs and the report is not, so the
rule is a density one: follow the last transmission verb seen, and stop once ten words pass without
another. Then run past the last transmitter's name to the verb that introduces the report. Three
things were measured rather than assumed, and each changed the code:

1. **Prefixed links.** `wa-akhbarani`, `fa-haddathana` are as common as the bare forms; an unprefixed
   list left half a chain at the head of 12 reports.
2. **Honorifics first.** `raḍiya llāhu ʿanhu` pads a transmitter's name, and a scan that had to step
   over it ran out of window and left the last transmitter in the report. Stripping the eulogies
   before looking for the boundary fixed it, including for hadith 1.
3. **The length backstop.** Capping the chain at 60% of a record stopped the scan mid-chain in 11
   reports, because chains are routinely longer than the report they carry. At 90% that falls to 1.

**Residual error: 0.15%** — 11 of 7,274 reports still open with a transmission verb. All are records
joining two chains with the *taḥwīl* marker `ح`. Recorded, not hidden.

**What is kept is the matn, not the Prophet's direct speech.** The matn is the body of the report and
includes the narrator's framing ("the women said to the Prophet ... so he promised them a day"). The
edition marks direct speech with quotation marks and those spans are carried in `text_quoted`, but
they were **not** used as the boundary, because measurement ruled it out: only **56%** of quoted spans
are preceded by the Prophet named as the speaker. The rest are dialogue quoted inside a report — the
angel at Hira, Khadija, Companions. A boundary drawn on quotation marks would have mixed his words
with theirs and called the result his.

**Two things §7 must not forget.** `qāla rasūlu llāh` — "the Messenger of God said" — is still in the
text. It is formulaic, it appears in thousands of reports and it appears in the Quran never, so a
classifier can separate the two corpora on that phrase alone without touching anyone's style. It was
kept because it is the report's own opening rather than citation apparatus, and cutting further would
be shaping the text until the comparison came out. §7 has to run with it removed as a control. Second,
the *kitab* is carried as `group`, so a style grouping can be tested against subject matter: a split
that tracks the kitab is tracking genre.

**Check:** ✓ counts reproduce from a fresh clone via `scripts/download_sources.sh`; ✓ spot-checked
reports show matn without isnad.

## 7. Quran against hadith: a discrimination test, declared before it is run

- [ ] 7.1 Write the hypothesis, the statistic and the threshold into a file **before** running anything,
      as `benchmarks/accuracy_targets.json` already does for the reference authors.
- [ ] 7.2 Run the comparison at passage level, not verse level, since §1 measured that verse-sized units
      cannot support attribution.
- [ ] 7.3 Report the measured result, whichever way it falls.

**Check:** the predeclared file is committed in a commit that precedes the run.

**An exploratory observation that constrains the design, recorded before 7.1 is written.** Building
the explorer over the rebuilt corpus put both Arabic collections in one space: 211 books, 114 suras
and 97 kutub, pooled to one profile per book and grouped under each of the fourteen strategies.
**Nearly every strategy separates the two collections, and several separate them perfectly.** The
best stability-scoring candidate for each, and how far that partition agrees with the Quran/hadith
label:

| strategy | best candidate | min ARI | ARI vs collection |
|---|---|---:|---:|
| word_frequency, char_ngrams, length, punctuation, topics | k=2 | 0.98 – 1.00 | **1.00** |
| word_ngrams, embeddings | k=2 | 0.94 – 0.98 | 0.94 |
| hapax | k=2 | 1.00 | 0.91 |
| lexical_preference, clause | k=2 | 0.94 – 0.98 | 0.87 – 0.89 |
| richness | k=2 | 0.98 | 0.80 |
| rhythm | k=7 | 0.98 | 0.69 |
| function_words | k=4 | 0.78 | 0.60 |
| morphology | k=2 | 0.94 | 0.50 |

Under function words alone, the k=2 candidate puts 105 of 114 suras in one group and 95 of 97 kutub
in the other — 200 of 211 books on the right side, ARI 0.80. It is reported as one group only because
its stability is [0.43, 0.91, 0.93, 0.91, 0.96] and the gate takes the minimum.

**This is a warning, not a finding, and it is the reason 7.3 is the whole substance of the test.**
Three of the strategies that separate the collections perfectly are the ones this project already
labels as *not authorial*:

- **length** — ARI 1.00. Quran verses run 10 tokens, hadith reports 31. This is the unit-length gap
  and nothing else.
- **punctuation** — ARI 1.00. Editorial marks supplied by two different digital editions.
- **topics** — ARI 1.00. Subject matter: scripture against law and biography.

A separation that a pure length feature achieves perfectly is not evidence about authorship. §7 must
be run with the length, punctuation and topic strategies excluded, on passages built to the same token
budget from both corpora, and with the report-opening formulae removed — and even then, genre and two
centuries of separate transmission remain unaddressed by any of it.

**A correction, recorded so it is not repeated.** This table replaces an earlier one in this file that
reported a maximum ARI of 0.16 and concluded the grouping did not track the collection. That was
measuring the wrong partition: the per-book group the explorer stores is a book's group *within its own
collection*, so suras were being compared against suras and kutub against kutub, and the two label sets
both start at A1. It could not have tracked the collection whatever the data said. The language-wide
partition of all 211 books is the one that answers the question, and it says close to the opposite.

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

## 10. Other ways of measuring style, and which we are trying

The pipeline in `cluster.py` is one family among several. It combines lexical statistics with blinded
AI style profiles and clusters them, which finds *whatever structure dominates* — and on this corpus
that has repeatedly turned out to be genre rather than authorship: Codex Sinaiticus resolved into
prophets, poetry and narrative, and the Quran into long verses and short ones. The alternatives below
are the established methods, listed with what each would fix here.

| method | what it is | what it would address |
|---|---|---|
| **Burrows's Delta** (2002), **Cosine Delta** (Evert et al. 2017) | most-frequent-word rates, z-scored, compared by mean absolute difference or cosine | the topic/genre confound: Delta uses *only* function words, which an author cannot avoid and does not choose for subject matter |
| **Zeta** (Burrows 2007) | contrastive — words one author prefers and another avoids | two-author comparisons such as Quran vs hadith |
| **Impostors / Generalized Impostors** (Koppel & Winter 2014) | is the match to a candidate robust when a crowd of irrelevant documents and random feature subsets are added? | the rejection failure: 90.5% of unseen authors are wrongly accepted. This is the method built for verification |
| **Unmasking** (Koppel & Schler 2004) | delete the strongest discriminating features iteratively and watch the degradation curve | same-author verification without a reference population |
| **Rolling stylometry / rolling delta** (Eder) | slide a window along one text and plot attribution as it moves | finding *seams* inside a book, which is what the Documentary Hypothesis actually asks, rather than partitioning verses |
| **Bayesian function-word models** (Mosteller & Wallace 1964) | posterior odds between two named candidates | honest uncertainty on a specific, declared question |
| **Authorship embeddings** (e.g. LUAR) | contrastively trained, topic-invariant neural representations | strongest modern accuracy, but needs training data that does not exist for Koine Greek, Biblical Hebrew or Quranic Arabic |
| **Paleography** (Popović, Dhali & Schomaker 2021, on 1QIsaᵃ) | handwriting: allographs and geometry, not vocabulary | the strongest recent multi-author result in this corpus came from handwriting, and that scroll is already in `data/raw` |

The `stylo` R package (Eder, Rybicki & Kestemont) implements Delta, Zeta and rolling stylometry and is
what a reviewer will expect to see compared against.

### 10.1 Burrows's Delta — being implemented now

- [x] 10.1.1 Implement Delta in `stylometry/delta.py`: corpus most-frequent-word list, relative
      frequencies, z-scores, classic and cosine metrics, leave-one-group-out nearest neighbour.
- [x] 10.1.2 Validated on the labelled reference authors: **100%** on 18 works by 6 catalogued Greek
      authors, whole-work holdout, cosine Delta at 2,000 MFW. Within the published range, and cosine
      beating classic reproduces Evert et al. (2017).
- [x] 10.1.3 Swept 50–2,000 MFW and both metrics; curves recorded below rather than a single point.
- [x] 10.1.4 Codex Sinaiticus, 259 passages, same holdout: **45.6%** against the AI-profile method's
      **30.9%**. Paul 94% (against 72%), Septuagint prophets 86% (67%), deuterocanon 53% (8%).
      Septuagint poetry falls to 23% (from 52%) — Delta drops the genre cues the other method used.
- [ ] 10.1.5 Report as a baseline beside the current method, not instead of it.
- [ ] 10.1.6 Download the fresh Greek benchmark (8 further authors) and repeat 10.1.2; 18 works is
      few enough that 100% could be luck.
- [ ] 10.1.7 Add Delta to `compare-models` so the baseline appears in the standing comparison.

**Measured so far.** Unit length dominates the choice of method: on the labelled authors Delta scores
100% on whole works (median 8,745 tokens) but **83.2%** on 1,000-token passages cut from those same
works — statistically level with the existing pipeline's 85.0% at that size. Delta's advantage appears
on scripture, not on the benchmark, and it appears exactly where authorship rather than genre is the
question. That is the predicted behaviour: Delta measures only the words an author cannot avoid.

**Check:** Delta's accuracy on the labelled authors is measured and stated before any scripture result
is quoted, and the same whole-work holdout is used for both methods so the comparison is fair.

### 10.2 Word adjacency networks — tried, and weaker than Delta here

Implemented in `stylometry/wan.py` (Segarra, Eisen & Ribeiro 2015): a directed graph over function
words, each occurrence casting distance-discounted weight onto the markers that follow it within a
window, row-normalised into a Markov chain and compared by symmetric Kullback-Leibler divergence.
One network per candidate, as published. `stylometry wan` runs it.

- [x] 10.2.1 Implement, with tests for the properties that define it: markers only, direction kept,
      distance discounted, content words stepped over, unseen transitions improbable not impossible.
- [x] 10.2.2 Validate on the labelled Greek authors before scripture.
- [x] 10.2.3 Compare against Delta on identical units and holdout.
- [ ] 10.2.4 Decide whether to keep it in the reported baselines or retire it.

**Measured.** On the six catalogued Greek authors, whole works: WAN **88.9%** at best against Delta's
**100%**. On 1,000-token passages from those works: WAN **52.9%** at best against Delta's **83.2%**.
On Codex Sinaiticus whole works: WAN **54.5%** against Delta's 68.2%.

**Why it loses, and it is not the idea.** The graph has one cell per ordered marker pair, so 100
markers is 10,000 cells, and a 1,000-token passage supplies a few hundred marker tokens to fill them.
Almost every cell is smoothing, and the divergence measures the smoothing constant rather than the
text: at 200 markers on 1,000-token passages accuracy fell to 5.8%, below chance. The published
application was to Elizabethan plays of roughly 20,000 words. Our whole works have a median of 8,517.
It is also unstable: at 50 markers, changing smoothing from 0.01 to 0.1 moved accuracy from 33.3% to
88.9%, which is not a result anyone should build on.

**What it did agree on.** WAN and Delta both identify Paul 4/4, Septuagint prophets 3/3 and Luke-Acts
2/2. WAN gets Johannine 2/2 where Delta gets 1/2. The strong cases are strong under both methods,
which is worth more than either number alone.

**A scoring bug this exposed, now fixed in both commands.** A label carried by only one work cannot be
attributed once that work is held out, and was both dragging the score down and stealing predictions
from labels that could be right. Such labels are now set aside before attribution and the count is
reported. Delta on Sinaiticus passages reads **58.6%** on this corrected basis, not 45.6%.

### 10.3 Combining repetition, placement and rate — tried; it does not help

`stylometry/repetition.py` adds the vocabulary-richness family (Yule's K, Simpson's D, Sichel's S,
Honoré's R, Brunet's W, hapax ratio, moving-average TTR, top-word share, repeat rate) — the project
previously had only raw type-token ratio, which falls with length by construction.
`stylometry/combined.py` stacks three blocks — rate (Delta's representation), placement (word
adjacency transitions that actually occur) and repetition — each standardised and scaled to equal
total variance so no block wins on column count.

**Ablation, identical units and whole-work holdout, rate block at 2,000 MFW throughout:**

| blocks | labelled authors, whole works | labelled authors, 1k passages | Sinaiticus, 1k passages |
|---|---:|---:|---:|
| rate only (= Delta) | **100.0%** | **83.2%** | 60.0% |
| placement only | 83.3% | 59.7% | 49.8% |
| repetition only | 44.4% | 51.8% | 40.9% |
| rate + placement | 94.4% | 74.9% | **63.3%** |
| rate + repetition | 61.1% | 51.3% | 37.2% |
| all three | 61.1% | 55.0% | 40.9% |

**Repetition is actively harmful.** Adding it to rate costs 22 points on whole works and 32 on
passages. That reproduces Hoover's finding that vocabulary-richness indices underperform frequent-word
methods; each compresses a whole text to one number and is sensitive to genre and normalisation.

**Placement neither helps nor hurts reliably.** Rate+placement wins on Sinaiticus (+3.3 points, Paul
89% → 100%) and loses on the labelled authors (−8.3 points). Tested properly, neither is real:

- Sinaiticus: McNemar exact p = 0.427; bootstrap over whole works +3.7% [−7.7%, +15.1%]
- Labelled authors: McNemar p = 0.026 per passage, but passages within a work are not independent, and
  the bootstrap over works gives −8.2% [−16.3%, +1.8%], which includes zero

**A correction worth recording.** The first ablation appeared to show rate+placement beating rate
alone. It did not: the rate block was defaulting to 500 MFW while Delta's best setting is 2,000. The
gain was an artefact of a weakened baseline, and disappeared once the settings matched.

- [x] 10.3.1 Implement the repetition family and the combined representation, with tests.
- [x] 10.3.2 Ablate every combination on labelled authors and on Sinaiticus.
- [x] 10.3.3 Test the differences rather than reading the table, using the work as the unit.
- [ ] 10.3.4 If a combination is ever adopted, fit block weights with nested cross-validation on a
      held-out set. Tuning them on 18 works and reporting those numbers would be overfitting.

## 11. The full strategy panel — built and run

`stylometry analyse` runs every feature family and method the corpus supports, on 1,000-token
passages held out by whole work. 1,017 features for Greek. Modules: `panel.py` (feature families),
`measures.py` (Jensen-Shannon, cosine), `supervised.py` (SVM, Random Forest, impostors verification),
`rolling.py` (windows, change points), `analysis.py` (orchestration and report).

**30 strategies: 23 measured, 4 approximated, 1 partial, 1 unavailable, 1 measured but not authorial.**
Part-of-speech and syntax are unavailable — nothing in the corpus carries tags and no reliable tagger
exists for Koine Greek, Biblical Hebrew and Quranic Arabic together. Clause structure, grammar
preference, morphology and readability are approximated from closed-class words and word endings and
labelled `approx:`. Punctuation is measured but records the scribe or the modern editor, never the
author.

**Attribution, whole-work holdout:**

| method | labelled Greek authors | Sinaiticus (8 groupings) | Quran (Meccan/Medinan) |
|---|---:|---:|---:|
| baseline (majority) | 31.9% | 23.7% | 50.0% |
| Delta cosine 2000 MFW | 83.2% | 60.0% | 95.0% |
| **SVM** | **88.5%** | **70.2%** | **97.5%** |
| Random Forest | 71.7% | 59.1% | 95.0% |
| SVM, function words only | 84.8% | 46.0% | 82.5% |

The function-words-only row is the topic-independent control. On the labelled authors it costs almost
nothing (88.5% → 84.8%), which says the full panel is not winning on topic. On Sinaiticus it costs 24
points, which says a large part of that 70.2% *is* genre and content rather than authorship.

**Verification — the project's central failure, addressed.** The impostors method reaches **16% false
acceptance on Sinaiticus and 11.5% on the labelled authors**, against the existing pipeline's 90.5%.
Unlike attribution it can answer "neither". Caveat recorded: the threshold is fitted on the same data
and needs its own holdout before the number is quoted as a result.

**Change-point detection, and the correction that made it usable.** As first built it fired on every
single-author work — Plato's *Apology*, Xenophon's *Memorabilia*, Demosthenes — at p < 0.02. The
permutation null assumes windows are exchangeable, and no continuous prose satisfies that, because
neighbouring windows share a topic. Two fixes: the statistic is now scaled like a two-sample mean
difference, which removed a pull towards the ends of the sequence; and results are read against
`benchmarks/change_point_baseline.json`, the distribution reached by thirteen works of undisputed
single authorship (mean 1.75, max 1.88). The permutation p-value is still reported and is still
rejected for 85% of those single-author works, which is why it is not the finding.

**On that calibrated basis, four books in Codex Sinaiticus exceed what any single-author work reached:**

| book | separation | split falls at | what is there |
|---|---:|---|---|
| Isaiah | 2.24 | Isaiah 33:21 | at the "Little Apocalypse" of chs 34–35, long assigned to a later hand |
| Jeremiah | 2.21 | LXX Jeremiah 27:39 | inside the Oracles Against the Nations, which LXX places centrally |
| Psalms | 1.92 | Psalm 107:8 | Psalm 107 opens Book V of the Psalter, an ancient marked division |
| Sirach | 1.91 | Sirach 41:25 | before the hymnic Praise of Creation and Praise of the Fathers |

Luke, Acts, John, Job, Wisdom, Judith and both books of Maccabees did **not** exceed it. Window
resolution is ±500 tokens, so the correspondence is approximate, but the set of books flagged is the
set scholarship regards as composite, and the books it declines to flag are the ones regarded as
unified. Quran sura 2 (al-Baqara) also exceeds the baseline at 1.91; it is the only sura long enough
to test.

- [x] 11.1 Build the panel, the measures, the classifiers, the rolling analysis and the orchestration.
- [x] 11.2 Validate the whole battery on labelled authors before quoting any scripture result.
- [x] 11.3 Calibrate change-point detection against single-author works; record the baseline.
- [x] 11.4 Run on Codex Sinaiticus and the Quran.
- [ ] 11.5 Give the verification threshold its own holdout before the false-acceptance figure is quoted.
- [ ] 11.6 Widen the change-point baseline beyond 13 Greek works, and add Hebrew and Arabic references.
- [ ] 11.7 Narrow the seam locations with a smaller step once the baseline is re-measured at that step.

### 11.8 Run across all three scriptures — done

`stylometry analyse` over each language, 1,000-token passages, whole-work holdout.

| | Greek (LXX + NT + fathers) | Hebrew (Tanakh) | Arabic (Qur'an) |
|---|---:|---:|---:|
| passages / works | 531 / 66 | 210 / 26 | 40 / 25 |
| majority baseline | 23.2% | 37.1% | 50.0% |
| Delta cosine 2000 MFW | 57.6% | 60.5% | 95.0% |
| **SVM** | **69.1%** | 65.2% | **97.5%** |
| Random Forest | 66.9% | **65.7%** | 95.0% |
| SVM, function words only | 58.6% | 53.8% | 82.5% |
| verification: false acceptance | 17% | **30%** | 0% |

Hebrew verification is the weak point at 30% false acceptance, against 17% for Greek. Hebrew also has
the fewest passages per work, so it is the corpus least able to support a verification claim.

**Change points.** Greek flags 14 of 33 works, Hebrew 11 of 16, Arabic none — because Arabic has no
reference and now correctly receives no verdict.

The Greek result discriminates in a way that supports it: the lowest separations are **1 Corinthians
1.60, Romans 1.66** — Pauline letters of undisputed single authorship — and Job 1.71. The highest are
the Septuagint Torah, Prophets and Histories, which were translated from Hebrew by different
translators working book by book. That is a real finding, but it is a claim about **translators**, not
about the authors of the Hebrew originals.

**The Hebrew change-point result should not be quoted.** Proverbs, which announces its own composite
structure in its headings (Solomon, the sayings of the wise, Agur, Lemuel), scores 1.81 and is *not*
flagged, while books score above it. An ordering that puts a self-declared anthology below the
threshold is not tracking composite authorship. Two causes, pushing the same way: the reference is
modern Hebrew prose, far more internally uniform than biblical books that mix law, narrative and
poetry in one scroll; and 210 passages over 26 works is thin. Recorded as measured, not as evidence.

- [x] 11.8.1 Per-language calibration, with Arabic recorded as having none.
- [x] 11.8.2 Run all three and report.
- [ ] 11.8.3 Replace the Hebrew reference with single-author Hebrew closer in period and genre before
      any Hebrew seam is quoted.
- [ ] 11.8.4 Separate the Greek run into Septuagint and New Testament: the Septuagint result is about
      translators and should not sit in the same table as the New Testament one.
