"""A panel of stylometric feature families, each answering a different question about a text.

The families here are the standard ones. What matters for this corpus is which of them the *data*
supports, so each block records what it measures and, where it is a stand-in for something the corpus
cannot give, says so rather than quietly substituting.

Three limits apply throughout and are not repairable by writing more code.

* **No part-of-speech tags.** Nothing in the corpus carries them and no reliable tagger for Koine
  Greek, Biblical Hebrew and Quranic Arabic is available here. Syntax and grammar-preference features
  are therefore approximated from closed-class words and word endings, which is weaker and is labelled
  as ``approx:``.
* **Punctuation is not the author's.** Ancient manuscripts were written without consistent
  punctuation. Codex Sinaiticus carries scribal high points and the printed editions carry a modern
  editor's commas; both measure the person who prepared the text, not the person who composed it.
  The block is built because it is informative about scribes and editions, and labelled so it is never
  read as authorship evidence.
* **Sentences are inferred, not given.** Where punctuation exists it is used; elsewhere the verse is
  the only available unit, and verse division is itself medieval.
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter

import numpy as np

from . import repetition
from .lang import function_words, suffixes

# Marks that end a sense-unit in the Greek sources: the scribal high point and the editorial stop.
SENTENCE_MARKS = ".;·˙:?！。"
CLAUSE_MARKS = ",·˙;"


def _is_mark(ch: str) -> bool:
    """Combining marks are vowel points and cantillation, not punctuation, and are excluded."""
    return unicodedata.category(ch).startswith("M")


def punctuation_counts(text: str) -> Counter:
    return Counter(ch for ch in text
                   if not ch.isalnum() and not ch.isspace() and not _is_mark(ch))


def sentences(text: str) -> list[list[str]]:
    """Split on sentence-ending punctuation; a text without any is one sentence."""
    out, current = [], []
    for token in text.split():
        stripped = "".join(c for c in token if c.isalnum() or _is_mark(c))
        if stripped:
            current.append(stripped)
        if any(c in SENTENCE_MARKS for c in token):
            if current:
                out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def word_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def char_ngrams(text: str, n: int) -> Counter:
    joined = " " + " ".join(text.split()) + " "
    return Counter(joined[i:i + n] for i in range(len(joined) - n + 1))


def collocations(tokens: list[str], window: int = 3, top: int = 40) -> list[tuple[tuple, float]]:
    """Word pairs that co-occur far more than chance, scored by pointwise mutual information.

    A collocation is not merely a frequent pair: ``of the`` is frequent because both words are. PMI
    divides the joint rate by what independence would predict, so what surfaces is the pairing the
    author actually favours. Pairs seen fewer than three times are dropped as noise.
    """
    if window < 1:
        raise ValueError("window must be positive")
    unigrams = Counter(tokens)
    total = max(len(tokens), 1)
    pairs: Counter = Counter()
    for i, a in enumerate(tokens):
        for b in tokens[i + 1:i + 1 + window]:
            pairs[(a, b)] += 1
    joint_total = max(sum(pairs.values()), 1)
    scored = []
    for (a, b), count in pairs.items():
        if count < 3:
            continue
        expected = (unigrams[a] / total) * (unigrams[b] / total)
        if expected > 0:
            scored.append(((a, b), math.log((count / joint_total) / expected)))
    return sorted(scored, key=lambda kv: -kv[1])[:top]


def repeated_phrases(tokens: list[str], n: int = 3, min_count: int = 3, top: int = 40):
    """Fixed expressions: n-grams the text returns to. Phraseology as opposed to single-word habit."""
    counts = word_ngrams(tokens, n)
    return [(phrase, c) for phrase, c in counts.most_common(top) if c >= min_count]


# --- the named blocks -------------------------------------------------------------------------------

def function_word_rates(tokens: list[str], language: str) -> tuple[np.ndarray, list[str]]:
    """Strategy: function-word frequency. Words an author cannot avoid and does not choose for topic."""
    words = function_words(language)
    counts = Counter(tokens)
    n = max(len(tokens), 1)
    return (np.array([counts[w] / n * 100 for w in words]), [f"fw:{w}" for w in words])


def morphology_rates(tokens: list[str], language: str) -> tuple[np.ndarray, list[str]]:
    """Strategy: morphology. Word endings stand in for inflection, with no tagger available."""
    endings = suffixes(language)
    n = max(len(tokens), 1)
    out = [sum(1 for w in tokens if len(w) > len(s) + 1 and w.endswith(s)) / n * 100 for s in endings]
    return np.array(out), [f"sfx:-{s}" for s in endings]


def length_features(tokens: list[str], text: str) -> tuple[np.ndarray, list[str]]:
    """Strategies: word length, sentence length, readability, rhythm and cadence."""
    lengths = [len(w) for w in tokens] or [0]
    sents = sentences(text)
    sent_lengths = [len(s) for s in sents] or [len(tokens)]
    mean_sent = float(np.mean(sent_lengths))
    mean_word = float(np.mean(lengths))
    names = [
        "len:mean_word", "len:sd_word", "len:long_word_ratio", "len:short_word_ratio",
        "len:mean_sentence", "len:sd_sentence", "len:max_sentence", "len:sentences_per_100w",
        "read:words_per_sentence_x_wordlen", "rhythm:length_autocorrelation", "rhythm:burstiness",
    ]
    # Rhythm: do neighbouring sentences echo each other's length, and how bursty is the series?
    series = np.array(sent_lengths, dtype=float)
    if len(series) > 2 and series.std() > 1e-9:
        centred = series - series.mean()
        autocorr = float((centred[:-1] * centred[1:]).mean() / series.var())
    else:
        autocorr = 0.0
    burstiness = float((series.std() - series.mean()) / (series.std() + series.mean())) if series.sum() else 0.0
    return np.array([
        mean_word, float(np.std(lengths)),
        float(np.mean([l >= 8 for l in lengths])), float(np.mean([l <= 3 for l in lengths])),
        mean_sent, float(np.std(sent_lengths)), float(max(sent_lengths)),
        len(sents) / max(len(tokens), 1) * 100,
        mean_sent * mean_word, autocorr, burstiness,
    ]), names


def clause_features(tokens: list[str], text: str, language: str) -> tuple[np.ndarray, list[str]]:
    """Strategies: clause structure and grammar preference, approximated from closed-class words.

    Without a parser, how clauses combine is read off the words that join them: coordinators against
    subordinators is the paratactic/hypotactic axis, which is the distinction that matters most for
    these languages.
    """
    from .lang import INITIAL_CONNECTIVES

    conn = INITIAL_CONNECTIVES[language]
    coordinators = set(conn.get("first", ())) | set(conn.get("second", ()))
    fw = set(function_words(language))
    subordinators = {w for w in fw if w not in coordinators}
    n = max(len(tokens), 1)
    clause_marks = sum(1 for ch in text if ch in CLAUSE_MARKS)
    coord = sum(1 for w in tokens if w in coordinators)
    subord = sum(1 for w in tokens if w in subordinators)
    return np.array([
        coord / n * 100, subord / n * 100,
        subord / max(coord, 1),
        clause_marks / n * 100,
        len(sentences(text)) / max(clause_marks + 1, 1),
    ]), ["approx:coordinator_rate", "approx:subordinator_rate", "approx:hypotaxis_ratio",
         "punct:clause_marks_per_100w", "approx:clauses_per_sentence"]


def punctuation_features(text: str) -> tuple[np.ndarray, list[str]]:
    """Strategy: punctuation. Records the scribe or the editor, never the author - see the module note."""
    counts = punctuation_counts(text)
    n = max(len(text.split()), 1)
    marks = [".", ",", "·", ";", ":", "?", "'", "’"]
    values = [counts[m] / n * 100 for m in marks]
    values.append(sum(counts.values()) / n * 100)
    return np.array(values), [f"punct:{m}" for m in marks] + ["punct:total_per_100w"]


def richness_features(tokens: list[str]) -> tuple[np.ndarray, list[str]]:
    """Strategies: vocabulary richness and hapax legomena."""
    return repetition.features(tokens), list(repetition.NAMES)


# --- the families that were missing ----------------------------------------------------------------

def pos_features(pos: list[str]) -> tuple[np.ndarray, list[str]]:
    """Strategies: syntax and grammar preference, from real tags where the corpus carries them.

    The Open Scriptures Hebrew Bible tags every morpheme, so Hebrew has genuine parts of speech rather
    than the closed-class approximation used elsewhere. Rates of each tag, and of each ordered pair,
    which is where grammatical habit shows: how often a conjunction is followed by a verb rather than
    a noun separates narrative chaining from nominal style without touching a single content word.

    Greek and Arabic carry no tags here, so this returns zeros for them, and the coverage table says
    so rather than letting an empty block look like a measurement.
    """
    tags = [t for t in (pos or []) if isinstance(t, str)]
    names = [f"pos:{name}" for name in POS_TAGS] + [f"posbi:{a}>{b}" for a, b in POS_BIGRAMS]
    if not tags:
        return np.zeros(len(names)), names
    n = len(tags)
    counts = Counter(tags)
    bigrams = Counter(zip(tags, tags[1:]))
    values = [counts[t] / n * 100 for t in POS_TAGS]
    total_bigrams = max(sum(bigrams.values()), 1)
    values += [bigrams[pair] / total_bigrams * 100 for pair in POS_BIGRAMS]
    return np.array(values), names


POS_TAGS = ["noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction",
            "particle", "suffix"]
POS_BIGRAMS = [("conjunction", "verb"), ("conjunction", "noun"), ("preposition", "noun"),
               ("verb", "noun"), ("noun", "verb"), ("noun", "noun"), ("particle", "noun"),
               ("verb", "preposition"), ("noun", "adjective"), ("verb", "verb")]


# Pairs that mean the same thing, where an author's choice between them is habit rather than content.
# This is the feature Dershowitz and Koppel used for biblical source criticism. The Hebrew first
# person is the classic case: anoki and ani are the same word, and the Priestly source prefers ani.
SYNONYM_PAIRS = {
    "hbo": [
        ("first_person", ["אנכי"], ["אני"]),
        ("relative", ["אשר"], ["ש"]),
        ("assembly", ["עדה"], ["קהל"]),
        ("maidservant", ["אמה"], ["שפחה"]),
        ("beget", ["הוליד"], ["ילד"]),
    ],
    "grc": [
        ("say_aorist", ["ειπεν"], ["εφη"]),
        ("but", ["δε"], ["αλλα"]),
        ("therefore", ["ουν"], ["διο"]),
        ("boy", ["παιδιον"], ["παισ"]),
        ("see", ["ιδου"], ["ιδε"]),
    ],
    "arb": [
        ("say", ["قال"], ["قل"]),
        ("indeed", ["ان"], ["انما"]),
        ("people", ["الناس"], ["القوم"]),
    ],
    # English pairs are the well-worn ones of the discipline: near-equivalent choices where the
    # author is picking without deliberating, which is what makes the choice habitual rather than
    # meant. Without these, lexical preference was the one strategy the validation corpus could not
    # score, which is the wrong strategy to have no evidence about.
    "eng": [
        ("while", ["while"], ["whilst"]),
        ("among", ["among"], ["amongst"]),
        ("amid", ["amid"], ["amidst"]),
        ("toward", ["toward", "towards"], ["to"]),
        ("upon", ["upon"], ["on"]),
        ("perhaps", ["perhaps"], ["maybe", "possibly"]),
        ("although", ["although"], ["though"]),
        ("cannot", ["cannot"], ["can't"]),
        ("shall", ["shall"], ["will"]),
        ("must", ["must"], ["ought"]),
        ("begin", ["begin", "began", "begun"], ["start", "started"]),
        ("seem", ["seem", "seemed", "seems"], ["appear", "appeared", "appears"]),
        ("very", ["very"], ["quite", "rather"]),
        ("big", ["big", "large"], ["great"]),
        ("said", ["said"], ["replied", "answered"]),
    ],
}


def synonym_features(tokens: list[str], language: str) -> tuple[np.ndarray, list[str]]:
    """Strategy: lexical preference - which of two words meaning the same thing an author reaches for.

    Reported as a share, ``a / (a + b)``, so it is a preference and not a frequency: a passage that
    uses neither returns the neutral 0.5, and one that uses both returns where it sits between them.
    A rate would confound preference with how often the idea comes up at all.
    """
    pairs = SYNONYM_PAIRS.get(language, [])
    names, values = [], []
    for label, first, second in pairs:
        a = sum(1 for t in tokens if t in first)
        b = sum(1 for t in tokens if t in second)
        names.append(f"syn:{label}")
        values.append(a / (a + b) if (a + b) else 0.5)
        names.append(f"syn:{label}_attested")
        values.append(float(min(a + b, 10)))
    return np.array(values) if values else np.zeros(0), names


def topic_features(docs: list[list[str]], n_topics: int = 12, seed: int = 0
                   ) -> tuple[np.ndarray, list[str]]:
    """Strategy: semantic patterns - how a passage's vocabulary distributes over latent topics.

    Latent Dirichlet Allocation over content words, fitted on the corpus being analysed. This is the
    one family here that deliberately measures *subject matter* rather than style, and it earns its
    place by being the control: if a style result and the topic block agree closely, the style result
    is probably reading content. It should never be quoted as evidence of authorship.

    Function words are excluded, since a topic model built on them would model nothing.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    names = [f"topic:{i + 1}" for i in range(n_topics)]
    if len(docs) < max(3, n_topics):
        return np.zeros((len(docs), n_topics)), names
    texts = [" ".join(d) for d in docs]
    counts = CountVectorizer(analyzer=str.split, min_df=3, max_df=0.5).fit_transform(texts)
    if counts.shape[1] < n_topics:
        return np.zeros((len(docs), n_topics)), names
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed,
                                    learning_method="batch", max_iter=20)
    return lda.fit_transform(counts), names


def embedding_features(docs: list[list[str]], dims: int = 50, window: int = 4, seed: int = 0
                       ) -> tuple[np.ndarray, list[str]]:
    """Strategy: learned representations - vectors fitted to this corpus, not a pretrained model.

    Words are embedded by factorising their co-occurrence counts, and a passage is the mean of its
    words' vectors. This is the distributional-semantics idea behind word2vec, reduced to the linear
    case, and it is what "neural embeddings" can honestly mean here.

    It is **not** a pretrained authorship embedding. Those - LUAR and its relatives - are trained
    contrastively on hundreds of thousands of labelled author pairs to be topic-invariant, and no such
    model exists for Koine Greek, Biblical Hebrew or Quranic Arabic. Fitted on the corpus under study,
    these vectors carry topic as readily as style, so they are reported beside the topic block and
    read the same way.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer

    names = [f"emb:{i + 1}" for i in range(dims)]
    if len(docs) < 3:
        return np.zeros((len(docs), dims)), names
    vocabulary = [w for w, c in Counter(t for d in docs for t in d).items() if c >= 5]
    if len(vocabulary) < dims + 1:
        return np.zeros((len(docs), dims)), names
    index = {w: i for i, w in enumerate(vocabulary)}
    co = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float32)
    for doc in docs:
        ids = [(i, index[t]) for i, t in enumerate(doc) if t in index]
        for a, (pos_a, col_a) in enumerate(ids):
            for pos_b, col_b in ids[a + 1:]:
                if pos_b - pos_a > window:
                    break
                co[col_a, col_b] += 1.0
                co[col_b, col_a] += 1.0
    k = min(dims, min(co.shape) - 1)
    vectors = TruncatedSVD(n_components=k, random_state=seed).fit_transform(np.log1p(co))
    out = np.zeros((len(docs), dims), dtype=np.float64)
    for row, doc in enumerate(docs):
        present = [index[t] for t in doc if t in index]
        if present:
            out[row, :k] = vectors[present].mean(axis=0)
    return out, names
