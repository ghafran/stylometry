"""Deterministic (non-AI) stylometric features for each verse unit, per language.

Blocks
------
fw:*   relative frequency (per 100 tokens) of the language's closed-class words
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


def lexical_features(verses: list[dict], svd_dims: int = 40, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    langs = {v.get("language", "grc") for v in verses}
    if len(langs) != 1:
        raise ValueError(f"lexical features need a single language, got {sorted(langs)}")
    lang = langs.pop()
    fw_list, sfx_list, conn = function_words(lang), suffixes(lang), INITIAL_CONNECTIVES[lang]

    texts = [v["text_bare"] for v in verses]
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

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=30000, sublinear_tf=True)
    cng = tfidf.fit_transform(texts)
    k = max(2, min(svd_dims, cng.shape[1] - 1, len(verses) - 1))
    proj = TruncatedSVD(n_components=k, random_state=seed).fit_transform(cng).astype(np.float32)
    names += [f"cng:svd{i:02d}" for i in range(k)]

    return np.hstack([fw, sfx, misc, proj]).astype(np.float32), names
