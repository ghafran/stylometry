"""Deterministic (non-AI) stylometric features for each verse unit, per language.

Blocks
------
fw:*   relative frequency (per 100 tokens) of predefined function/common words
sfx:*  relative frequency of word endings (crude morphology)
misc:* length, word-length, type-token ratio, verse-initial connective habits
cng:*  truncated-SVD projection of character 2-4-gram TF-IDF (spelling / morphology / phonology)

All verses passed in must share one language.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .lang import INITIAL_CONNECTIVES, function_words, initial_waw, suffixes

MISC_NAMES = [
    "misc:log_tokens",
    "misc:mean_word_len",
    "misc:long_word_ratio",
    "misc:type_token_ratio",
    "misc:initial_conjunction",
    "misc:second_particle",
    "misc:asyndeton_start",
]


def _static_features(verses: list[dict]) -> tuple[np.ndarray, list[str]]:
    if not verses:
        raise ValueError("lexical features need at least one verse")
    langs = {v.get("language", "grc") for v in verses}
    if len(langs) != 1:
        raise ValueError(f"lexical features need a single language, got {sorted(langs)}")
    lang = langs.pop()
    fw_list, sfx_list, conn = function_words(lang), suffixes(lang), INITIAL_CONNECTIVES[lang]

    texts = [v["text_bare"] for v in verses]
    if any(not isinstance(t, str) for t in texts):
        raise ValueError("text_bare must be a string for every verse")
    if not any(t.strip() for t in texts):
        raise ValueError("lexical features need nonempty text")
    toks = [t.split() for t in texts]
    n = np.array([max(len(t), 1) for t in toks], dtype=np.float32)

    cv = CountVectorizer(vocabulary=fw_list, analyzer=str.split, lowercase=False)
    fw = cv.transform(texts).toarray().astype(np.float32) / n[:, None] * 100
    names = [f"fw:{w}" for w in fw_list]

    sfx = np.zeros((len(verses), len(sfx_list)), dtype=np.float32)
    for i, words in enumerate(toks):
        for j, suf in enumerate(sfx_list):
            sfx[i, j] = sum(1 for w in words if len(w) > len(suf) + 1 and w.endswith(suf))
    sfx = sfx / n[:, None] * 100
    names += [f"sfx:-{s}" for s in sfx_list]

    misc = np.zeros((len(verses), len(MISC_NAMES)), dtype=np.float32)
    for i, words in enumerate(toks):
        lens = [len(w) for w in words] or [0]
        first_two = set(words[:2])
        initial = (1.0 if words and words[0] in conn["first"] else 0.0) or initial_waw(words, lang)
        misc[i] = [
            np.log1p(len(words)),
            float(np.mean(lens)),
            float(np.mean([l >= 8 for l in lens])),
            len(set(words)) / max(len(words), 1),
            initial,
            1.0 if len(words) > 1 and words[1] in conn["second"] else 0.0,
            1.0 if words and not (first_two & conn["any"]) and not initial else 0.0,
        ]
    names += MISC_NAMES

    return np.hstack([fw, sfx, misc]).astype(np.float32), names


class LexicalFeatureExtractor:
    """Train-fitted lexical feature map for honest held-out evaluation.

    Word lists and local rates are fixed; character vocabulary, IDF and SVD
    are learned exclusively in fit(). transform() never adapts to test data.
    Corpus clustering uses fit_transform(), preserving its transductive behavior.
    """

    def __init__(self, svd_dims: int = 40, seed: int = 0):
        if not isinstance(svd_dims, (int, np.integer)) or svd_dims < 0:
            raise ValueError("svd_dims must be a nonnegative integer")
        self.svd_dims = int(svd_dims)
        self.seed = seed

    def fit(self, verses: list[dict]) -> "LexicalFeatureExtractor":
        _, names = _static_features(verses)
        self.language_ = verses[0].get("language", "grc")
        self.vectorizer_ = None
        self.svd_ = None
        self.projection_dims_ = 0
        texts = [v["text_bare"] for v in verses]
        if self.svd_dims:
            min_df = 5 if len(texts) >= 5 else 1
            cng = None
            for cutoff in dict.fromkeys((min_df, 1)):
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=cutoff,
                                             max_features=30000, sublinear_tf=True)
                try:
                    cng = vectorizer.fit_transform(texts)
                    self.vectorizer_ = vectorizer
                    break
                except ValueError as exc:
                    if "empty vocabulary" not in str(exc) and "After pruning, no terms remain" not in str(exc):
                        raise
            if cng is not None and cng.shape[1]:
                k = min(self.svd_dims, cng.shape[1], max(1, len(verses) - 1))
                if cng.shape[1] == 1:
                    self.projection_dims_ = 1
                else:
                    self.svd_ = TruncatedSVD(n_components=k, random_state=self.seed)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        self.svd_.fit(cng)
                    self.projection_dims_ = self.svd_.components_.shape[0]
        self.feature_names_ = names + [f"cng:svd{i:02d}" for i in range(self.projection_dims_)]
        return self

    def transform(self, verses: list[dict]) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise ValueError("fit the lexical feature extractor before transforming observations")
        if not verses:
            return np.empty((0, len(self.feature_names_)), dtype=np.float32)
        if any(v.get("language", "grc") != self.language_ for v in verses):
            raise ValueError("held-out texts must have the same language as the fitted feature map")
        static, _ = _static_features(verses)
        if self.vectorizer_ is None:
            return static
        cng = self.vectorizer_.transform([v["text_bare"] for v in verses])
        projection = self.svd_.transform(cng) if self.svd_ is not None else cng.toarray()
        return np.hstack([static, projection]).astype(np.float32)

    def fit_transform(self, verses: list[dict]) -> np.ndarray:
        return self.fit(verses).transform(verses)


def lexical_features(verses: list[dict], svd_dims: int = 40, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    extractor = LexicalFeatureExtractor(svd_dims=svd_dims, seed=seed)
    X = extractor.fit_transform(verses)
    return X, extractor.feature_names_
