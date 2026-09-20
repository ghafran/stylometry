"""The strategies, each one selectable on its own, with what it measures and what it cannot.

A stylometric result is only as good as the reader's ability to see which evidence produced it. So
every family is registered here as a separate, independently runnable block: its key, the plain
statement of what it measures, its standing (measured, approximated, unavailable, or measured but not
about authorship), and the function that turns passages into numbers.

The analysis then runs each strategy alone, all of them together, and all-but-one for each, so a
reader can ask three different questions: what can this strategy see by itself, what do they see
together, and what is lost when this one is removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import panel

MEASURED = "measured"
APPROXIMATED = "approximated"
PARTIAL = "partial"
UNAVAILABLE = "unavailable"
NOT_AUTHORIAL = "not authorial"


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    measures: str          # what it looks at, in one line
    status: str
    note: str = ""         # the caveat a reader needs before trusting it
    languages: tuple = ()  # empty means every language
    build: Callable | None = field(default=None, compare=False)

    def available_for(self, language: str) -> bool:
        return not self.languages or language in self.languages


def _per_doc(fn) -> Callable:
    """Wrap a per-document feature function into one that builds the whole matrix."""
    def build(docs, texts, language, pos=None, seed=0):
        rows, names = [], None
        for i, (tokens, text) in enumerate(zip(docs, texts)):
            values, block_names = fn(tokens, text, language, (pos or [None] * len(docs))[i])
            rows.append(values)
            names = names or block_names
        return (np.vstack(rows) if rows else np.zeros((0, 0))), (names or [])
    return build


def _ngram_block(kind: str, n: int, cap: int) -> Callable:
    def build(docs, texts, language, pos=None, seed=0):
        from collections import Counter

        counters = [panel.char_ngrams(t, n) if kind == "char" else panel.word_ngrams(d, n)
                    for d, t in zip(docs, texts)]
        total: Counter = Counter()
        for c in counters:
            total.update(c)
        keys = [k for k, _ in total.most_common(cap)]
        if not keys:
            return np.zeros((len(docs), 0)), []
        X = np.array([[c.get(k, 0) / max(sum(c.values()), 1) for k in keys] for c in counters])
        prefix = f"{kind}{n}"
        return X, [f"{prefix}:{k if isinstance(k, str) else ' '.join(k)}" for k in keys]
    return build


REGISTRY: list[Strategy] = [
    Strategy("function_words", "Function-word frequency",
             "Rates of words an author cannot avoid: and, but, of, the, therefore.",
             MEASURED,
             "The workhorse of the field. These words carry almost no subject matter, which is why "
             "they survive a change of topic.",
             build=_per_doc(lambda t, x, l, p: panel.function_word_rates(t, l))),

    Strategy("word_frequency", "Word-frequency profile",
             "The commonest words of the corpus and how often each passage uses them.",
             MEASURED,
             "What Burrows's Delta is computed from.",
             build=_ngram_block("word", 1, 500)),

    Strategy("char_ngrams", "Character n-grams",
             "Recurring letter sequences of two to five characters.",
             MEASURED,
             "Robust to spelling variation and damage, but they run over content words too, so they "
             "carry topic as well as style.",
             build=_ngram_block("char", 4, 300)),

    Strategy("word_ngrams", "Word n-grams",
             "Recurring sequences of two or three words.",
             MEASURED, build=_ngram_block("word", 2, 300)),

    Strategy("richness", "Vocabulary richness",
             "Yule's K, Simpson's D, Sichel's S, Honoré's R, Brunet's W and moving-average TTR.",
             MEASURED,
             "Measured here to be weak: added to word rates it cost 22 points on whole works. Each "
             "index compresses a whole text to one number.",
             build=_per_doc(lambda t, x, l, p: panel.richness_features(t))),

    Strategy("hapax", "Hapax legomena",
             "The share of the vocabulary a passage uses exactly once.",
             MEASURED,
             "Part of the richness family and subject to the same caution.",
             build=_per_doc(lambda t, x, l, p: (
                 np.array([panel.repetition.hapax_ratio(t), panel.repetition.sichel_s(t)]),
                 ["hapax:ratio", "hapax:dis_legomena"]))),

    Strategy("lexical_preference", "Lexical preferences",
             "Which of two words meaning the same thing an author reaches for.",
             MEASURED,
             "A small curated list per language, not a full synonym lexicon: the Hebrew first person "
             "anoki against ani is the classic case, long used in source criticism.",
             build=_per_doc(lambda t, x, l, p: panel.synonym_features(t, l))),

    Strategy("length", "Sentence and word length",
             "Word length, sentence length and their spread.",
             PARTIAL,
             "Sentences are only available where punctuation survives - Sinaiticus's scribal high "
             "points, the printed editors' commas. Unpointed Qumran material has none.",
             build=_per_doc(lambda t, x, l, p: panel.length_features(t, x))),

    Strategy("clause", "Clause structure",
             "How clauses are joined: coordination against subordination.",
             APPROXIMATED,
             "Read off closed-class words because no parser exists for these languages.",
             build=_per_doc(lambda t, x, l, p: panel.clause_features(t, x, l))),

    Strategy("pos", "Syntax and grammar (part of speech)",
             "Rates of each part of speech and of each ordered pair.",
             MEASURED,
             "Real morphological tags, from the Open Scriptures Hebrew Bible. Hebrew only: no tagged "
             "text and no reliable tagger exists for Koine Greek or Quranic Arabic.",
             languages=("hbo",),
             build=_per_doc(lambda t, x, l, p: panel.pos_features(p))),

    Strategy("morphology", "Morphology",
             "Word endings, standing in for inflection.",
             APPROXIMATED,
             "A suffix list, not a morphological analyser. For Hebrew the part-of-speech block is the "
             "better measurement.",
             build=_per_doc(lambda t, x, l, p: panel.morphology_rates(t, l))),

    Strategy("punctuation", "Punctuation patterns",
             "Rates of stops, commas and the scribal high point.",
             NOT_AUTHORIAL,
             "Ancient manuscripts were not punctuated consistently. What this measures is the scribe "
             "who copied the text or the modern editor who printed it - never the author.",
             build=_per_doc(lambda t, x, l, p: panel.punctuation_features(x))),

    Strategy("rhythm", "Rhythm and cadence",
             "Whether neighbouring sentences echo each other's length, and how bursty the series is.",
             MEASURED,
             build=_per_doc(lambda t, x, l, p: (
                 panel.length_features(t, x)[0][-2:],
                 ["rhythm:length_autocorrelation", "rhythm:burstiness"]))),

    Strategy("topics", "Semantic patterns",
             "How a passage's vocabulary distributes over latent topics.",
             NOT_AUTHORIAL,
             "This one measures subject matter on purpose. It is the control: a style result that "
             "tracks this block closely is probably reading content, not authorship.",
             build=lambda docs, texts, language, pos=None, seed=0: panel.topic_features(docs, seed=seed)),

    Strategy("embeddings", "Learned representations",
             "Vectors fitted to this corpus by factorising word co-occurrence.",
             NOT_AUTHORIAL,
             "Not a pretrained authorship embedding: those are trained contrastively on labelled "
             "author pairs to be topic-invariant, and none exists for these languages. Fitted here, "
             "they carry topic as readily as style.",
             build=lambda docs, texts, language, pos=None, seed=0: panel.embedding_features(docs, seed=seed)),
]

BY_KEY = {s.key: s for s in REGISTRY}


def build_matrix(keys, docs, texts, language="grc", pos=None, seed=0):
    """Features for the chosen strategies only, with their names and per-strategy column spans."""
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")
    blocks, names, spans = [], [], {}
    for key in keys:
        strategy = BY_KEY[key]
        if not strategy.available_for(language):
            spans[key] = (0, 0)
            continue
        X, block_names = strategy.build(docs, texts, language, pos, seed)
        X = np.asarray(X, dtype=float)
        if X.size == 0 or X.shape[1] == 0:
            spans[key] = (0, 0)
            continue
        start = sum(b.shape[1] for b in blocks)
        blocks.append(X)
        names += list(block_names)
        spans[key] = (start, start + X.shape[1])
    if not blocks:
        raise ValueError("no chosen strategy produced any feature for this language")
    return np.hstack(blocks), names, spans


def describe(language: str | None = None) -> list[dict]:
    """The registry as plain data, for a report or a picker."""
    return [{"key": s.key, "name": s.name, "measures": s.measures, "status": s.status,
             "note": s.note,
             "available": True if language is None else s.available_for(language),
             "languages": list(s.languages) or ["all"]}
            for s in REGISTRY]
