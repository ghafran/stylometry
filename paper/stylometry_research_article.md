# Style Discovery and Authorship Attribution in a Multilingual Corpus of Scriptural and Literary Texts

## Abstract

Computational stylometry can identify recurrent textual patterns without establishing how many historical agents produced them. This distinction becomes consequential when an analysis combines literary works with translated, transmitted, and fragmentary religious texts. We examine the saved output of a language-specific stylometric pipeline applied to 154,158 textual units containing 6,348,728 normalized tokens in Arabic, English, Greek, and Biblical Hebrew. The pipeline constructs disjoint contextual passages, combines frequent-word rates with character n-grams, reduces the representation to at most twelve dimensions, and selects diagonal Gaussian mixtures using the Bayesian information criterion. It identifies five Arabic, nineteen English, eighteen Greek, and ten Hebrew groups. All four partitions improve substantially on their respective one-component baselines, but none satisfies the implementation’s resampling stability requirement. An English reference corpus provides a direct test of the interpretation of these groups: blind discovery yields nineteen groups for thirteen catalogue authors, with adjusted Rand index 0.260 and normalized mutual information 0.437. A separate supervised classifier correctly attributes 199 of 238 held-out passages and 24 of 25 held-out works. This divergence shows that successful closed-set attribution does not validate unsupervised author enumeration. Detailed inspection reveals author splitting and merging in English, strong collection separation in Arabic, overlapping collection memberships in Greek, and severe evidence attrition among Hebrew fragments. We conclude that the output supports exploratory mapping of textual heterogeneity and the design of targeted follow-up comparisons. Historical authorship claims require additional controls for genre, translation, transmission, witness selection, and the dependence introduced by passage-level assignment.

Keywords: stylometry; authorship attribution; unsupervised clustering; textual transmission; multilingual corpora; model validation

## 1 Introduction

The present analysis finds substantial stylistic heterogeneity in four language corpora while also exposing the limits of interpreting that heterogeneity as authorship. Its clearest result comes from the English control. A supervised model identifies the catalogue author of nearly every held-out work, whereas an unsupervised model divides the same broader corpus into groups that frequently split individual authors and combine different authors. These outcomes are compatible because the models answer different questions. Recognizing a work among known candidates requires a useful decision boundary; discovering historical writers requires a justified correspondence between latent textual groups and people. Our results establish the former under a particular evaluation design and do not establish the latter.

Authorship attribution has long used recurrent linguistic measurements to compare texts. Frequent lexical items, character sequences, and surface statistics can supply evidence that complements bibliographical and philological analysis. Stamatatos’s survey organizes the field around both representations and evaluation conditions, emphasizing that performance belongs to a specified problem rather than to a feature inventory in isolation (Stamatatos, 2009). That perspective is especially relevant when moving between modern literary benchmarks and ancient textual traditions. A procedure that discriminates among catalogued English writers encounters a different inferential problem when applied to a translated biblical book or a report transmitted through successive narrators.

Our object of study is the output of version 0.2.0 of the Stylometry project, examined on 22 September 2026. The application provides language-specific style groups and a navigable hierarchy of collections, books, chapters, and verses. English paragraphs occupy the same structural position as verses. This common interface makes heterogeneous materials searchable, but it also creates an interpretive risk: a label displayed beside a verse may appear to represent evidence derived from that verse alone. In the implemented analysis, labels usually derive from a much larger passage and are inherited by its member units. The apparent resolution of the display therefore exceeds the resolution of the statistical decision.

The article investigates three questions. First, what statistical structure does the pipeline recover, and how reproducible are its partitions under the perturbations actually tested? Second, how closely does blind English discovery correspond to reference authorship, and how does that correspondence relate to independent held-out attribution? Third, which features of the ancient corpora constrain historical interpretation? These questions place model diagnostics, corpus accounting, and substantive interpretation within one analysis. They also allow a useful result to be reported without equating every stylistic boundary with a change of writer.

The contribution is an empirical audit and interpretation of a completed computational analysis. We connect numerical outputs to the implementation that produced them, identify discrepancies between convenient descriptions and actual procedures, and examine concrete cases in which a plausible reading would exceed the evidence. For example, the English discovery returns three groups within the Federalist collection, yet almost all of that collection’s words occupy a single group shared by the three reference authors. The number three thus has little evidential value by itself. Likewise, three groups appear in both the Greek and Hebrew versions of Isaiah, but the systems do not align their feature spaces or test corresponding boundaries. Equal counts cannot establish a shared compositional history.

We use “style group” as an operational term for a fitted mixture component with assigned passages. It identifies an estimated region of a particular feature space in a particular run. It does not presuppose that the region represents an individual, a community, a translator, or an editorial layer. Such interpretations remain hypotheses to be tested against textual and historical alternatives. Throughout the article, we separate measurements stored in the saved output, descriptive quantities calculated from those measurements, and proposed analyses that have not yet been performed.

## 2 Conceptual and methodological background

### 2.1 What a stylometric representation measures

Stylometry depends on repeated linguistic choices, but repetition can have several causes. An author may favor a construction; a genre may require it; a topic may make it unavoidable; or an editor may regularize it. Frequent words are attractive partly because they occur often enough to support comparison across passages. Their frequency alone, however, does not make them independent of communicative context. Kestemont (2014) provides a theoretical discussion of function words that helps clarify why apparently unobtrusive lexical choices deserve analysis while still requiring linguistic explanation.

The present pipeline selects frequent words algorithmically. It does not identify function words through grammatical annotation or a curated list. Consequently, “frequent-word representation” is the correct description of its lexical component. A high-frequency content word can enter the vocabulary when the corpus contains a concentrated topic or a repeated formula. This distinction matters particularly for scriptural collections, where names, reporting expressions, recurrent invocations, and legal terminology may be frequent. Those features can be useful for describing textual organization while remaining ambiguous evidence about individual authorship.

Character n-grams add another source of information. They can represent word endings, fragments of lexical items, and patterns near word boundaries. Sapkota et al. (2015) show that different classes of character n-grams do not contribute equally to attribution. Their result motivates careful attention to how characters are extracted rather than treating all character features as interchangeable. Here, discovery uses n-grams within word boundaries after normalization, whereas the supervised benchmark uses ordinary character n-grams on a different text representation. The two systems therefore have overlapping motivations but materially different inputs.

Text length also changes the reliability of stylistic measurement. Eder’s investigations demonstrate that sample size and sampling procedure affect attribution, with consequences that vary across corpora and languages (Eder, 2015). The current target of 1,200 tokens is a practical design choice, not a demonstrated universal sufficiency threshold. Its 200-token minimum prevents the shortest contexts from receiving model assignments, but it does not establish that every context above that floor contains enough information for reliable historical inference. Minimum eligibility and validated adequacy are separate properties.

### 2.2 Attribution and discovery have different targets

In closed-set attribution, the candidate authors are known and represented in training. The classifier must choose among those candidates for a new text. In unsupervised discovery, neither author labels nor the number of authors need be supplied. The procedure instead estimates a partition according to a statistical objective. That partition might align with authors, but it can also align with other recurrent distinctions. The mere absence of author labels from fitting establishes blindness with respect to those labels; it does not establish blindness to their correlates, such as genre or publication conventions.

Prior work on unsupervised decomposition has explicitly addressed the possibility of separating textual contributions without identified samples of the contributing authors (Koppel et al., 2011). Such work makes clear why segmentation and discovery warrant their own evaluation. Our application differs from an experiment with known artificially combined sources: it fits a language-wide mixture across many works and then propagates passage assignments to constituent units. We therefore evaluate the resulting partition against known English authors only after fitting and avoid treating that comparison as a supervised prediction experiment.

The distinction can be expressed statistically. A classifier estimates a relationship between textual features and reference labels. A mixture model estimates a distribution of textual features through several components. There is no requirement that the number of components needed to approximate that distribution equal the number of reference labels. One writer can occupy several regions because of dialogue, narration, topic, or genre; several writers can occupy one region because they share conventions. Our interpretation follows this difference in the quantities being estimated.

### 2.3 Fit and stability answer different questions

The Bayesian information criterion, or BIC, compares fitted models through a likelihood term and a complexity penalty (Schwarz, 1978). Within this project, it decides how many diagonal Gaussian components to retain from a finite candidate set. A lower BIC supports that candidate under the chosen representation and model family. It does not supply a posterior probability that a specified number of historical authors existed. The distinction is particularly important when components are used to approximate distributions that need not be Gaussian or composed of discrete social entities.

Stability asks whether a partition persists when some aspect of the analysis changes. The broader literature distinguishes multiple perturbations and cautions against assuming that stability alone identifies a uniquely meaningful clustering (von Luxburg, 2010). Our diagnostics examine only repeated fits after subsampling the already constructed feature space at a fixed component count. This is a limited but informative test. Failure suggests sensitivity even under relatively constrained perturbations; success would still leave representation choice, corpus composition, and historical interpretation unresolved.

## 3 Materials and methods

### 3.1 Analytical snapshot and corpus accounting

We analyzed the saved report, passage membership file, English benchmark, and corpus build report. The article does not introduce a newly fitted discovery model or report unperformed sensitivity experiments. Counts and percentages not explicitly stored in the report were calculated from its exported rows. The inspected repository revision was 276465f9263a083d8806b97b4595582449f32e1f. The corpus fingerprint agreed across the build report, main report, and benchmark, supporting the conclusion that these artifacts refer to the same ordered input texts.

The source build parsed 242,519 units and selected 154,158 primary units. The remaining 88,361 units belonged to alternate manuscript witnesses or editions and were excluded from the primary analysis. Selection operates within language and book, retaining one primary witness. A preferred witness is eligible for that preference only when its token coverage reaches at least 90% of the fullest available witness. This procedure limits the inflation that would arise if parallel copies were counted as independent works, while making the analyzed corpus dependent on a specified editorial policy.

Four language identifiers organize all fitting: Arabic, English, Greek, and Biblical Hebrew. These are corpus categories, not assertions that every included text has an identical linguistic history. Together they contain 6,348,728 tokens under the discovery tokenizer. Token totals are implementation-dependent measures; they are not directly comparable estimates of linguistic information across writing systems. All 154,158 primary units remain in the exported report, including units that cannot be assigned a style.

Table 1. Corpus size and passage eligibility

| Language | Textual units | Tokens | All passages | Eligible passages | Assigned units |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arabic | 13,865 | 648,247 | 608 | 595 | 13,803 |
| English | 70,292 | 4,389,373 | 3,474 | 3,474 | 70,292 |
| Greek | 41,126 | 851,869 | 746 | 738 | 41,115 |
| Biblical Hebrew | 28,875 | 459,239 | 5,481 | 286 | 23,411 |
| Total | 154,158 | 6,348,728 | 10,309 | 5,093 | 148,621 |

Arabic includes the Qur’an, Sahih al-Bukhari, and Forty Hadith Qudsi. Greek includes the Septuagint collection, the New Testament, and a noncanonical collection assembled from the supported sources. Hebrew includes the Tanakh, selected Dead Sea Scrolls material, and three inscription or papyrus records. English includes eighty-five Federalist papers, fifteen novels, and twenty works in a collection designated Cross-genre. The latter designation describes corpus construction and must not be confused with a controlled train-in-one-genre, test-in-another evaluation.

The primary source accounting is more restrictive than the list of available downloads. Codex Sinaiticus supplies 20,139 selected units and the Vaticanus/Swete source supplies 17,414. First1KGreek supplies 1,833, and the Apostolic Fathers source supplies 1,740. The Open Scriptures Hebrew Bible supplies 23,213 units and the Dead Sea Scrolls source 5,659. Although CNTR, MAM, and the Samaritan source were parsed, none contributes selected primary units in this snapshot. Their presence on disk therefore does not demonstrate that their alternate readings influenced the fitted results.

### 3.2 Text preparation and documentary scope

English importers retain body paragraphs while excluding identified Gutenberg wrappers and editorial material. The Federalist adapter removes printed bylines and related publication framing so that explicit author names are not presented as attribution evidence. Twelve disputed and three joint Federalist papers remain available for blind discovery but lack a single-author reference label. The supervised benchmark excludes these fifteen works from labelled evaluation. This treatment preserves their textual presence without resolving disputed authorship by assumption.

The hadith input retains complete transmitted reports, including chains of transmission, narrative framing, and citation notes. The analysis therefore concerns the surface style of those digital reports. It does not isolate a particular quoted speaker or infer where one historical voice ends and another begins. The distinction is central when comparing hadith and Qur’anic text: a separation can reflect the report format, recurring transmission language, or editorial presentation, among other possibilities. A model trained on complete reports cannot directly answer a question about the style of an extracted speech segment that it was never given.

Fragmentary Hebrew material retains gap and reconstruction information. Explicit gaps interrupt passage construction, and a damaged unit is handled on its own rather than permitting surrounding material to bridge the gap. These choices prioritize traceable evidence over uniform assignment coverage. Reconstruction metadata remains available, but the discovery representation does not independently model uncertainty in each restored letter. A sufficiently long reconstructed or damaged unit can still be eligible, subject to a low-evidence flag.

### 3.3 Passage construction and inherited labels

Within each language, the system pools complete verses or paragraphs into nonoverlapping passages targeting 1,200 tokens. It normally preserves book boundaries while allowing a passage to span chapters. Whole units are kept intact, so a long paragraph can produce a passage substantially above the target. A final block below 200 tokens joins the preceding block only within the same uninterrupted segment; otherwise, it remains ineligible. Empty units and gap-marked units interrupt the ordinary sequence. The resulting analytical unit is the passage, even when the display presents results at verse level.

There is one consequential exception to the ordinary book-boundary rule. In the Qur’an collection, consecutive suras that are individually below the 200-token floor can be combined. Grouping stops when the floor is reached, with a remaining short tail permitted to join its neighboring group. The saved run contains eleven passages spanning multiple suras. One includes ten suras, Q105 through Q114, with 241 tokens and forty-eight verses. By contrast, the opening sura remains an ineligible twenty-nine-token passage because it is not joined across the intervening long sura. Describing this implementation simply as never crossing books would be inaccurate.

All units within an eligible passage inherit one mixture assignment. They also receive the supporting passage identifier, evidence-token count, and a distance margin. The exported membership lists permit reconstruction of these dependencies. A visual succession of identically tagged verses can therefore represent one decision propagated across multiple rows. It cannot be counted as repeated independent confirmation of that decision, and a transition between tags is localized initially to a passage boundary rather than to an independently detected compositional seam.

### 3.4 Discovery representation and mixture fitting

Discovery text undergoes case folding and Unicode NFKD normalization, followed by removal of combining marks. The tokenizer retains alphabetic sequences with limited internal apostrophe handling and excludes digits and underscores. The model then rejoins these tokens into normalized text. This treatment reduces some orthographic variation, but it also removes distinctions encoded through diacritics. There is no morphological parsing, syntactic annotation, lemmatization, or explicit attribution of quoted material to separate voices.

The lexical representation contains up to 200 frequent words, expressed as rates per passage token count. A training-fitted scaler standardizes their variation without centering the sparse matrix, and the rates are divided by the square root of the vocabulary size. The character representation contains up to 3,000 TF-IDF features built from two- through four-character n-grams within word boundaries. These two matrices are concatenated. The scaling is intended to temper imbalance between feature families; it does not demonstrate that their empirical contributions are exactly equal.

Truncated singular value decomposition reduces the combined representation to at most twelve dimensions. The retained dimension count is also constrained by sample and feature counts. Numerically negligible axes are removed before subsequent standardization, and standardized coordinates are clipped to the interval from minus eight to eight. These operations make fitting tractable and limit numerical artifacts. They also mean that the statistical groups summarize variation surviving a substantial projection, rather than differences in every original word or character feature.

At most 2,000 eligible passages per language fit the representation and mixture. The seed is 42, and sampling is without replacement. Arabic, Greek, and Hebrew use all 595, 738, and 286 eligible passages, respectively; English uses 2,000 of 3,474. Every eligible passage is then assigned using the fitted model. English discovery consequently includes assignments outside the fit sample, but this should not be called a held-out authorship evaluation: the target is an unlabeled partition, and the fit sample is drawn at passage rather than book level.

The candidate component count runs from one to the smaller of twenty and the integer part of the fit-sample size divided by eight. All four languages reach a candidate ceiling of twenty in this run. Each model uses diagonal covariance, covariance regularization of 0.05, two initializations, and at most 200 iterations. A multicomponent candidate must converge and place at least three fit passages in its smallest predicted group. Among supported candidates, selection minimizes BIC; a multicomponent solution must also improve BIC over the one-component baseline by at least ten.

For a model with K components and d dimensions, the diagonal Gaussian parameter count is K times the sum of twice d and one, minus one. BIC penalizes that count by the logarithm of the number of fit observations and subtracts twice the fitted log likelihood. This makes sample size, dimensionality, and model flexibility explicit parts of the selection rule. Because passages from the same work can remain dependent, the criterion should be understood here as the implementation’s model-selection score, without extending its result into a formal test of historical author number.

### 3.5 Diagnostics and evidence status

The report records a Euclidean silhouette score in the reduced space, using at most 1,000 fit observations while ensuring representation of every predicted group. Silhouette compares within-group and alternative-group distances and supplies a geometric diagnostic (Rousseeuw, 1987). The fitted mixture itself assigns by component probabilities involving covariance and mixture weights. Its assignments need not coincide with the nearest Euclidean center, so silhouette and mixture likelihood assess different aspects of the same partition.

The exported distance margin likewise uses Euclidean center distances. It subtracts the assigned-center distance from the nearest alternative-center distance and divides by the larger of those distances, with numerical protection near zero. A negative margin is possible because the assigned mixture component need not be the nearest center under this metric. The margin is neither a calibrated correctness probability nor a historical confidence estimate.

Three stability repetitions refit a mixture with the selected component count after sampling 80% of the fitting observations. The feature map remains fixed, and each trial uses one initialization. Trial predictions on the full fit sample are compared with the original labels using adjusted Rand index, or ARI (Hubert and Arabie, 1985). The implementation calls a partition stable only if all three ARIs are at least 0.8. These trials combine subsample perturbation with changes in initialization and do not estimate variability from rebuilding the complete pipeline.

A tagged unit receives low-evidence status when its own text has fewer than forty tokens, it has a gap, its margin is below 0.1, or the language partition is not marked stable. Since every fitted language fails the final condition in this run, all 148,621 assigned units are marked low evidence. The remaining 5,537 units are insufficient text. This global consequence must be made explicit: the flag does not distinguish a few problematic verses from otherwise validated historical assignments.

### 3.6 English evaluation design

Blind discovery is compared post hoc with thirteen independently supplied English author labels. The evaluated set contains 70,068 labelled paragraphs, all of which have discovery assignments. The remaining 224 English paragraphs belong to works without single-author reference labels. Evaluation records group-count error, ARI, and normalized mutual information, or NMI. The latter measures association between partitions with a normalization that should not be mistaken for correction for chance (Vinh et al., 2010). These metrics are paragraph-weighted, and adjacent paragraphs often share a passage prediction.

The independent supervised evaluation groups text by whole work before splitting. Each author needs at least two usable works. A seeded per-author shuffle selects 20% of works, rounded upward, for testing while retaining training material. The realized split contains eighty training works and twenty-five test works. It includes all thirteen reference authors. The word “book” in the implementation denotes this work-level identifier: an individual Federalist paper counts as one work even though its scale differs from a novel.

Benchmark passages are built within chapters, can split long textual units, and are sampled at evenly spaced positions, up to twenty per work. Thus, their construction differs from discovery passages. Of 4,094 available benchmark passages in eligible works, 861 are sampled: 623 for training and 238 for testing. The classifier combines training-fitted word TF-IDF with up to 800 terms, character three- and four-gram TF-IDF with up to 8,000 features, and standardized surface measurements. A class-balanced linear support vector machine predicts the author. Feature-family weights are 1.0, 0.75, and 0.10, respectively.

Vocabulary, inverse document frequencies, scaling, and classifier fitting use training works only. Work predictions use the most frequent passage prediction, with ties resolved lexicographically. This is a plurality rule; it does not require an absolute majority. We report passage accuracy, macro-averaged author recall, work accuracy, and work-level macro recall. No repeated outer splits, confidence intervals, or formal significance tests are supplied in the analyzed output, and none is inferred from the single saved experiment.

## 4 Results

### 4.1 Coverage and the effective amount of evidence

Assignment coverage is 99.55% of Arabic units, 100% of English units, 99.97% of Greek units, and 81.08% of Hebrew units. These percentages describe the availability of a contextual assignment, irrespective of its low-evidence status. The difference between rows and analytical observations is substantial. The entire corpus has 154,158 rows, but only 5,093 eligible passages. Counting rows as independent measurements would therefore exaggerate the empirical resolution even before accounting for dependence among passages from the same work.

The Hebrew case most clearly shows why total passage count can be misleading. It has 5,481 passages, exceeding English, but only 286 satisfy the evidence floor. Most Hebrew blocks are short or fragmented. The Tanakh contributes 23,213 assigned units out of 23,213, whereas the Dead Sea Scrolls collection contributes only 197 assigned units out of 5,659. Its assignment coverage is 3.48%. Of its 150,391 tokens, 144,441 occur in unassigned units, leaving 5,950 tokens associated with a style label. Statements about the collection’s stylistic diversity must therefore be restricted to a small selected portion of its text.

Eligible passage lengths also vary despite the common target. Their medians are 1,222 tokens in Arabic, 1,246 in English, 1,208 in Greek, and 1,205.5 in Hebrew. English includes an eligible passage of 3,659 tokens, reflecting preservation of complete units during discovery. Arabic includes eligible contexts near the evidence floor because short suras can be combined. These observations confirm that the target length standardizes scale only approximately. They do not warrant comparing a 200-token fragment and a long literary passage as equally informative observations.

### 4.2 Selected partitions and competing component counts

The selected group counts are five for Arabic, nineteen for English, eighteen for Greek, and ten for Hebrew. Every selection lies below the formal search ceiling. The result is therefore not mechanically truncated at twenty, although the English and Greek choices remain close enough to that ceiling to motivate a future expanded search. More fundamentally, all counts are conditional on the vocabulary, projection, sampling, and covariance assumptions specified above.

Table 2. Discovery model diagnostics

| Language | Groups | BIC improvement over one | Silhouette | Resampling ARI range | Mean ARI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arabic | 5 | 3,891.566 | 0.150 | 0.553–0.637 | 0.595 |
| English | 19 | 6,754.271 | 0.128 | 0.536–0.677 | 0.621 |
| Greek | 18 | 2,637.459 | 0.178 | 0.627–0.666 | 0.641 |
| Biblical Hebrew | 10 | 1,144.748 | 0.211 | 0.620–0.750 | 0.694 |

The large BIC improvements indicate that a single diagonal Gaussian is a comparatively poor description of each fitted language representation. They should not be compared across languages as a scale of historical complexity, because the corpora have different sample sizes and feature distributions. The silhouette values are positive but modest. Taken together with the stability results, they describe useful structure accompanied by appreciable overlap or ambiguity in the chosen geometric representation.

Inspection of neighboring candidate scores adds information that the selected count alone conceals. English BIC is 61,565.307 at nineteen components and 61,571.894 at sixteen, a difference of 6.587. Greek BIC is 22,663.646 at eighteen and 22,671.283 at seventeen, a difference of 7.637. Hebrew’s ten-component solution improves on eleven by 21.801. Arabic’s five-component solution improves on eight by 70.060. These are descriptive comparisons within each search, not probabilities over candidate counts. The ten-point rule in the software applies against the one-component baseline; it does not require a ten-point advantage over the runner-up.

All twelve resampling ARIs are below 0.8. The original partitions therefore fail the project’s stability rule even while strongly improving BIC. This is not a contradiction. A distribution can require more than one Gaussian component while admitting several substantially different assignments. The saved output supports language-specific heterogeneity more strongly than it supports any one exact partition. The report’s low-evidence flags accurately reflect that distinction, although the global nature of the flag should accompany any visualization derived from it.

### 4.3 English discovery splits and merges known authors

The English discovery identifies nineteen groups on labelled text against thirteen reference authors, yielding a count error of plus six. ARI is 0.260184 and NMI is 0.436775. These values indicate incomplete correspondence with reference authorship; neither can be restated as a percentage of authors correctly discovered. Their paragraph weighting also means that the numerical contribution of a work depends partly on how its edition divides prose into paragraphs.

Several groups nevertheless show concentrated author membership. Charlotte Bronte accounts for 6,527 of 7,021 labelled paragraphs in eng-S002, or 92.96%. Jane Austen accounts for 2,419 of 2,547 in eng-S010 and 3,028 of 3,179 in eng-S014, or 94.97% and 95.25%, respectively. Mark Twain accounts for 2,672 of 2,722 paragraphs in eng-S017, or 98.16%. These concentrations show that author-associated signal is present in portions of the partition. They also demonstrate why cluster purity is insufficient: two highly Austen-concentrated groups already split a single reference author.

Other groups combine many authors. The largest group by token support, eng-S001, includes labelled text by ten reference authors, with substantial contributions from Thomas Hardy, Charles Dickens, G. K. Chesterton, and Mark Twain. The mixture is therefore neither a simple oversegmentation of perfectly isolated authors nor a wholly uninformative partition. It combines selective concentration, within-author differentiation, and cross-author merging. Explaining these patterns requires examining which textual properties distinguish the groups, rather than assigning each group the name of its most frequent author.

The Federalist collection provides a particularly clear counterexample to author counting. It contains three discovered style identifiers, but eng-S011 accounts for 185,899 of its 189,773 tokens, approximately 97.96%. Within the labelled contingency table, that group contains 724 Hamilton paragraphs, seventy-six Jay paragraphs, and 262 Madison paragraphs. The discovery has mainly grouped the collection’s shared textual characteristics while allowing a few passages to join groups found elsewhere. A count of three within this collection does not amount to recovery of its three reference authors.

### 4.4 Held-out attribution succeeds under a different design

The supervised model correctly predicts 199 of 238 held-out passages, producing 83.61% accuracy. Macro-averaged author recall is 77.61%, revealing uneven performance that the overall rate partially obscures. Aggregating predictions by work yields twenty-four correct decisions out of twenty-five, or 96.00%, with work-level macro recall of 92.31%. The improvement with aggregation is empirical evidence that multiple sampled passages often provide a useful collective signal even when individual passage predictions vary.

Table 3. Held-out passage performance by reference author

| Reference author | Training works | Test works | Correct passages | Test passages | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alexander Hamilton | 40 | 11 | 27 | 27 | 100.00% |
| Arthur Conan Doyle | 3 | 1 | 20 | 20 | 100.00% |
| Charles Dickens | 2 | 1 | 16 | 20 | 80.00% |
| Charlotte Bronte | 2 | 1 | 20 | 20 | 100.00% |
| G. K. Chesterton | 3 | 1 | 20 | 20 | 100.00% |
| George Eliot | 2 | 1 | 12 | 20 | 60.00% |
| H. G. Wells | 3 | 1 | 15 | 20 | 75.00% |
| James Madison | 11 | 3 | 8 | 9 | 88.89% |
| Jane Austen | 2 | 1 | 20 | 20 | 100.00% |
| John Jay | 4 | 1 | 0 | 2 | 0.00% |
| Mark Twain | 3 | 1 | 18 | 20 | 90.00% |
| Robert Louis Stevenson | 3 | 1 | 9 | 20 | 45.00% |
| Thomas Hardy | 2 | 1 | 14 | 20 | 70.00% |

John Jay’s Federalist No. 64 is the only incorrectly attributed work. Both sampled passages are assigned to Hamilton. Because Jay contributes only two test passages and one test work, the observed zero cannot support a general conclusion that his writing is intrinsically unidentifiable. It demonstrates failure on the small held-out sample actually available. Conversely, perfect scores for other authors are limited to the tested works and should not be generalized to all their genres or periods.

Stevenson’s Virginibus Puerisque illustrates the aggregation rule. Nine passages are correctly assigned to Stevenson, five to Wells, three to Eliot, and one each to Doyle, Chesterton, and Twain. Stevenson wins the plurality despite receiving fewer than half of the votes. George Eliot’s Middlemarch is also correctly attributed at work level despite eight incorrect passage predictions. These examples explain how work-level success can coexist with substantial local error and why “majority vote” would be an imprecise description of the implementation.

The Federalist test subset reaches 35 correct passages out of 38, or 92.11%, but its macro recall is 62.96%. Hamilton’s larger contribution dominates the unbalanced average. The Novels and Cross-genre subsets each obtain 82 correct predictions out of 100. The latter result is evidence about held-out works drawn from a heterogeneous collection; the split does not enforce genre separation between training and test. It therefore cannot independently demonstrate that attribution is invariant to genre.

### 4.5 Arabic groups separate collections and divide the Qur’an

The Arabic partition shows complete separation between the two groups assigned to Qur’anic material and the three groups assigned to the hadith collections. Of 6,236 Qur’anic verses, 6,189 are assigned: 2,541 to arb-S003 and 3,648 to arb-S005. These groups contain 43,999 and 33,685 tokens, respectively. The larger verse count belongs to the smaller token total, illustrating how short-versus-long verse structure changes the apparent prominence of a group depending on the denominator.

Sahih al-Bukhari contributes 7,574 assigned reports out of 7,589. Of these, 5,335 occupy arb-S001, 1,832 occupy arb-S002, and 407 occupy arb-S004. All forty Hadith Qudsi reports occupy arb-S004, which therefore crosses the boundary between the two hadith collections. Collection labels were not supplied as model features, so these associations emerge from textual measurements. Nevertheless, content, framing, transmission formulas, and editorial conventions can all encode collection membership indirectly.

The two Qur’anic groups establish variation within the analyzed normalized text relative to the complete Arabic corpus. They do not identify two authors, two revelation periods, or two compositional strata. None of those hypotheses was supplied as an external annotation and tested against the partition. Al-Baqara contains both groups, with 210 verses assigned to arb-S003 and seventy-six to arb-S005. Al-Ikhlaas receives a single group through a passage shared with nine other short suras. Such differences show why sura-level counts require attention to supporting context before historical comparison.

### 4.6 Greek overlap and Hebrew evidence attrition

Greek has eighteen groups overall, of which sixteen appear in the Septuagint collection, ten in the New Testament, and seven in the noncanonical collection. These collection counts must not be added: the same language-specific identifier can appear in multiple collections. The overlap indicates that the partition does not simply reproduce the three collection labels, while also leaving open whether shared features derive from genre, vocabulary, translation practice, or other textual relations.

Some Greek groups are concentrated within a collection. For example, grc-S001 contains 71,313 tokens from noncanonical texts, 5,860 from the New Testament, and 1,201 from the Septuagint. Other groups cross boundaries more substantially. This distribution can guide close reading by identifying comparable passages, but a common identifier records similarity under the fitted representation rather than demonstrating literary dependence or common authorship. Those claims would require independent evidence and controls for ubiquitous linguistic patterns.

Isaiah illustrates the danger of comparing counts across languages. Greek Isaiah has three groups, with 1,013 of 1,289 units assigned to grc-S005. Hebrew Isaiah also has three groups, with 1,197 of 1,291 units assigned to hbo-S001. The predominance of one group in each version is descriptively clear, but the identities, boundaries, and textual supports differ. Because the pipelines are fitted independently and do not align verses for a cross-language test, the shared count of three cannot confirm a tripartite compositional hypothesis.

The Hebrew partition contains ten groups, nine represented in the Tanakh. The small assigned portion of the Dead Sea Scrolls occupies hbo-S009 and hbo-S010, while the sole assigned inscription record, the Nash Papyrus, occupies hbo-S008 on 206 tokens. The two Ketef Hinnom records remain unassigned. A single assigned papyrus record cannot establish the authorship or stylistic unity of an inscriptional tradition. More generally, the Hebrew results describe the distribution of usable surviving and encoded evidence, with particularly limited reach into the fragmentary material.

## 5 Discussion

### 5.1 The strongest finding is the separation of inferential tasks

The English comparison establishes the article’s main methodological conclusion within the project’s own data. There is sufficient author-associated information for strong closed-set work attribution under the saved split, but the unsupervised mixture does not recover the reference partition well. The discrepancy is not resolved by pointing to the high work accuracy, because the supervised and unsupervised systems have different objectives, feature capacities, training conditions, and passage construction rules. The classifier’s performance cannot retrospectively validate the meaning of mixture components.

This finding also limits the use of English as a calibration device for ancient materials. The benchmark establishes that one particular supervised pipeline can discriminate among its represented candidates. It does not estimate the probability that a Greek or Hebrew style identifier corresponds to one writer. The ancient corpora differ in script, morphology, documentary survival, editorial mediation, and the availability of ground truth. No mapping from English classification accuracy to historical authorship confidence has been fitted or justified.

The control nevertheless has substantial scientific value. It directly demonstrates that one author can appear in several groups and several authors can appear in one group. These are observed failure modes of author enumeration in the current system, rather than merely hypothetical cautions borrowed from the literature. They should shape every downstream interpretation of the non-English outputs. A candidate ancient attribution would need evidence capable of distinguishing the same alternatives that remain unresolved in English.

### 5.2 Heterogeneity is better supported than exact enumeration

All four languages favor multicomponent descriptions over a single diagonal Gaussian by substantial BIC margins. This supports heterogeneity in the fitted representation. It does not establish that the heterogeneous distribution is generated by a finite set of sharply separated historical styles. Gradients of register, topic, or orthographic practice can also require multiple components when approximated by this model family. The meaning of a component depends on the substantive account connecting its defining measurements to textual production.

The instability diagnostics further constrain enumeration. Each partition changes appreciably when the mixture is refitted on a subsample even though the feature map and component count remain fixed. There may be stable broad distinctions inside an unstable detailed partition, but the saved global ARIs do not identify which distinctions those are. A claim that a particular pair of groups is stable would require group-specific or pairwise evidence. The current output supplies a reason to investigate such structure, not an already completed demonstration.

A useful next analysis would examine co-assignment across many perturbations: how often do two passages remain together when the corpus, representation, and fitting seed change? Such a measure could reveal persistent cores and unstable boundaries without requiring labels to remain numerically identical across runs. In this project, style identifiers are ranked by token support and are inherently run-specific. Apparent persistence of a label string would therefore be weaker evidence than persistence of its member passages.

### 5.3 Corpus construction participates in the result

Selecting one witness per language and book improves the independence of the corpus relative to counting every manuscript copy as a separate work. It also creates a particular composite collection whose properties depend on the selection rule. A different primary witness could change orthography, the extent of text, and the placement of gaps. Because the model pools within languages, changing one substantial work can alter the feature vocabulary and shift assignments elsewhere. The corpus is part of the measurement procedure, rather than a neutral container for an invariant style signal.

The 90% coverage rule expresses a defensible preference for sufficiently complete evidence, but token coverage is not a complete measure of textual comparability. A witness may have similar total length while differing in which passages survive. Reconstruction conventions and divisions into units can also affect eligibility. A manuscript-based sensitivity study should therefore examine both alternative witnesses and aligned common textual spans. Comparing only aggregate word counts would leave the most consequential differences uncontrolled.

The Greek and Arabic cases make source mediation especially visible. Translated texts carry the effects of translation choices, and complete hadith reports contain layers of framing that are absent from a comparison based solely on quoted speech. The present pipeline does not separate these possible producers of linguistic regularity. Treating a discovered group as the fingerprint of an original historical speaker would collapse distinct stages of textual production into one inferred agent. The observed data are the normalized digital texts of transmitted works.

### 5.4 Missing assignments are substantively patterned

The absence of an assignment is not distributed uniformly across the corpus. It is concentrated in materials affected by fragmentation, gaps, or short independent units. In the Dead Sea Scrolls collection, the unassigned majority means that the visible style groups cannot be interpreted as an exhaustive inventory of stylistic variation. The eligible subset is selected by preservation and encoding conditions, which may correlate with textual type. Coverage is consequently part of the substantive result rather than a minor reporting inconvenience.

This pattern also cautions against interpreting greater assignability as stronger historical authenticity or coherence. A long, well-preserved text is more likely to satisfy the evidence floor, but length does not independently establish who composed it. Conversely, a short inscription can be historically informative while remaining statistically unsuitable for this representation. Documentary importance and stylometric eligibility answer different questions. The analysis benefits from retaining unassigned rows because they make that distinction inspectable.

The Qur’anic pooling exception presents the complementary tradeoff. Combining short suras increases coverage, but it transfers the evidence of the combined context to each constituent sura. This can be useful for exploratory browsing if the shared support is visible. It also suppresses the possibility of observing within-passage differences at the displayed resolution. Any argument about an individual short sura must return to its own evidence and evaluate whether an alternative method can support a separate decision.

### 5.5 Weighting determines the question being answered

The project presents unit counts and token counts, while fitting operates on passages. These are three different weighting schemes. Unit counts emphasize how the corpus is segmented into verses or paragraphs. Token counts emphasize textual volume. Passage-based fitting approximately equalizes local samples but still gives longer works more observations, and English subsampling occurs across passages without author or work balancing. The resulting model describes the sampled textual distribution, not an equal-weight comparison of authors or books.

The Qur’anic result provides a concrete illustration: arb-S005 contains more verses while arb-S003 contains more tokens. Neither statement is erroneous. They answer different questions about prevalence. Similarly, paragraph-weighted English ARI can change if an edition divides identical prose differently, even if passage labels remain unchanged. A future analysis should report agreement under passage, token, and equal-work weighting, accompanied by an explanation of the scholarly question each denominator serves.

Hierarchical summaries require equal care. The correct count of styles in a parent category is the union of identifiers across its children. Summing the Greek collection counts would produce thirty-three even though only eighteen Greek groups exist. Shared identifiers encode overlap, so an inflated sum would manufacture apparent diversity from repeated appearances of the same component. The project’s rollups use unions, which is an appropriate safeguard; interpretations and secondary tables should preserve it.

### 5.6 A research design for stronger historical claims

Further work should begin with controlled English experiments before assigning historical meaning to ancient groups. Repeated whole-work splits would measure sensitivity to the particular held-out titles, while author-balanced summaries would reduce the influence of Hamilton’s many short works. Dedicated genre-transfer experiments should train and test on explicitly separated genres. Topic masking or text distortion offers one possible control, motivated by Stamatatos’s cross-topic attribution results (Stamatatos, 2017), but its benefit would need to be measured in this corpus rather than assumed.

Discovery requires its own sensitivity program. Passage targets and minimum lengths should vary over a prespecified range; lexical-only, character-only, and combined representations should be compared; and the dimensionality and covariance assumptions should be challenged. Every repetition should rebuild vocabulary, scaling, and projection. Sampling whole works would probe corpus dependence more directly than repeatedly removing passages from a fixed feature map. These experiments could establish whether broad collection separations persist while finer partitions change.

Historical case studies should then replace unrestricted pattern matching with explicit competing hypotheses. For a proposed compositional boundary, the relevant test would compare predicted segmentation against that boundary while controlling for genre and textual length. For translation effects, aligned passages across versions and multiple translators or translation traditions would be preferable. For hadith, a comparison of full reports, transmission chains, and independently annotated speech segments could reveal which layer drives the observed separation. Each design should specify its target before inspecting the outcome.

Interpretability would also improve through feature-level analysis. The saved report does not provide the words or n-grams chiefly responsible for each group. Without those explanations, a group’s distribution can suggest a research question but offers limited evidence about its linguistic mechanism. Reporting discriminative patterns with representative and counterexample passages would let philologists assess whether a split tracks narration, formulaic language, lexical subject matter, or editorial practice. Such explanation should be tested on unseen or reserved text where feasible to avoid selecting only congenial examples.

### 5.7 Limitations of the present article

This study is a retrospective examination of one saved computational run and its accompanying implementation. The descriptive checks establish consistency among corpus identifiers, counts, assignments, and benchmark predictions. They do not constitute an independent replication from every raw source. The report records the application version and configuration but lacks a complete embedded execution environment and source-code hash. The inspected revision and input fingerprint improve traceability without proving the exact software state that generated every output file.

The article also does not supply ancient author labels, adjudicate contested attributions, or test detailed historical chronologies. No annotation of genre, translator, or transmission layer has been analyzed as an explanatory variable. Consequently, statements connecting groups to these factors remain possible interpretations or proposed controls. The concrete results concern assignments, overlaps, coverage, and diagnostic behavior. Keeping that boundary explicit permits meaningful use of the output without converting an exploratory analysis into a stronger experiment after the fact.

## 6 Conclusion

The analyzed output provides a traceable map of stylistic variation across 154,158 units in four languages. It identifies five Arabic, nineteen English, eighteen Greek, and ten Hebrew groups under a documented representation and mixture-selection procedure. These partitions improve on single-component baselines, yet all fail the implementation’s resampling stability requirement. Their exact membership and number should therefore remain provisional.

The English control establishes why authorship and stylistic grouping must be evaluated separately. Closed-set attribution succeeds for twenty-four of twenty-five held-out works, while blind discovery only partially corresponds to thirteen known authors and produces both splitting and merging. The ancient-text results inherit this unresolved interpretive problem and add further dependence on translation, transmission, preservation, and witness selection. Their strongest present use is to identify comparisons for close reading and controlled testing.

A defensible historical inference will require convergence between stable textual measurements, appropriate reference experiments, and independently specified philological hypotheses. The current system makes the underlying passages and coverage visible enough to support that next stage. Its scientific value lies in documenting where textual regularities appear, how strongly the model distinguishes them, and where the available evidence prevents a more specific conclusion.

## Data and reproducibility

The primary analytical artifacts are output/report.json, output/passages.json, output/benchmark.json, and data/processed/build_report.json in the Stylometry repository. The inspected code is in stylometry/engine.py, stylometry/benchmark.py, stylometry/corpus.py, and the source importers. The canonical input fingerprint is f73b651b0816a2bbf9d1ac9796a400174460c31807570495bc7b5ff4648d0717. It hashes ordered unit identifiers and texts; it does not independently attest every metadata field, reference label, or software dependency. The project configuration uses a 1,200-token target, 200-token evidence floor, twenty-component search ceiling, 2,000-passage fitting cap, and seed 42, with the Qur’an collection eligible for short-sura pooling.

The accompanying audit script reproduces this article’s descriptive summaries from saved outputs and records file checksums. The source texts remain attributable to their respective digital editions and transcription projects as documented in data/raw/SOURCES.md and the source directories. Availability in this repository should not be read as a blanket licence for unrestricted redistribution of every underlying text. No new historical author labels or synthetic experimental outcomes were introduced for this article.

## References

Eder, M. (2015). Does size matter? Authorship attribution, small samples, big problem. Digital Scholarship in the Humanities, 30(2), 167–182. [Publisher record](https://doi.org/10.1093/llc/fqt066).

Hubert, L., and Arabie, P. (1985). Comparing partitions. Journal of Classification, 2, 193–218. [Publisher record](https://doi.org/10.1007/BF01908075).

Kestemont, M. (2014). Function words in authorship attribution. From black magic to theory? Proceedings of the 3rd Workshop on Computational Linguistics for Literature, 59–66. [ACL Anthology](https://aclanthology.org/W14-0908/).

Koppel, M., Akiva, N., Dershowitz, I., and Dershowitz, N. (2011). Unsupervised decomposition of a document into authorial components. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 1356–1364. [ACL Anthology](https://aclanthology.org/P11-1136/).

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65. [Publisher record](https://doi.org/10.1016/0377-0427(87)90125-7).

Sapkota, U., Bethard, S., Montes-y-Gómez, M., and Solorio, T. (2015). Not all character n-grams are created equal: A study in authorship attribution. Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 93–102. [ACL Anthology](https://aclanthology.org/N15-1010/).

Schwarz, G. (1978). Estimating the dimension of a model. The Annals of Statistics, 6(2), 461–464. [Original article](https://doi.org/10.1214/aos/1176344136).

Stamatatos, E. (2009). A survey of modern authorship attribution methods. Journal of the American Society for Information Science and Technology, 60(3), 538–556. [Author manuscript](https://icsdweb.aegean.gr/stamatatos/papers/survey.pdf).

Stamatatos, E. (2017). Authorship attribution using text distortion. Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, Volume 1, 1138–1149. [ACL Anthology](https://aclanthology.org/E17-1107/).

Vinh, N. X., Epps, J., and Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. Journal of Machine Learning Research, 11, 2837–2854. [Journal article](https://www.jmlr.org/papers/v11/vinh10a.html).

von Luxburg, U. (2010). Clustering stability: An overview. Foundations and Trends in Machine Learning, 2(3), 235–274. [Author manuscript](https://arxiv.org/abs/1007.1075).
