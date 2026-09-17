"""Burrows's Delta, the standard baseline in computational stylometry.

Delta (Burrows 2002) represents a document by the relative frequencies of the most frequent words in
the corpus, z-scored against the corpus, and compares documents by the mean absolute difference of
those z-scores. Cosine Delta (Smith & Aldridge 2011; Evert et al. 2017) keeps the representation and
replaces the metric with cosine distance, which measures better on most corpora.

Two properties matter for this project. Delta uses *only* the most frequent words - articles,
particles, prepositions, pronouns, auxiliaries - which are the words an author cannot avoid and does
not choose for subject matter. That is deliberate: it is how the method controls for topic and genre,
the confound that dominates the clustering in ``cluster.py``, whose features include word length,
type-token ratio, discourse mode and character n-grams over content. And Delta is a *distance*, not a
decision: it ranks candidates and never says "none of these", so it attributes but cannot verify.

No model is involved at any point.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

DEFAULT_MFW = 500


def most_frequent_words(docs: list[list[str]], n_words: int = DEFAULT_MFW) -> list[str]:
    """The n commonest word types across the whole corpus, ranked by total frequency.

    Ranked on the corpus rather than per document, so every document is described on the same axes.
    Ties break alphabetically, so the list does not depend on dictionary ordering.
    """
    if n_words < 1:
        raise ValueError("n_words must be positive")
    if not docs or not any(docs):
        raise ValueError("Delta needs at least one nonempty document")
    totals: Counter = Counter()
    for tokens in docs:
        totals.update(tokens)
    return [w for w, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:n_words]]


def frequencies(docs: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    """Relative frequency of each vocabulary word in each document."""
    out = np.zeros((len(docs), len(vocabulary)), dtype=np.float64)
    index = {w: i for i, w in enumerate(vocabulary)}
    for row, tokens in enumerate(docs):
        if not tokens:
            continue
        counts = Counter(tokens)
        for word, count in counts.items():
            column = index.get(word)
            if column is not None:
                out[row, column] = count / len(tokens)
    return out


def zscores(freq: np.ndarray) -> np.ndarray:
    """Standardise each word across documents; a word with no variation carries no information."""
    mean = freq.mean(axis=0)
    sd = freq.std(axis=0)
    quiet = sd < 1e-12
    scaled = (freq - mean) / np.where(quiet, 1.0, sd)
    scaled[:, quiet] = 0.0
    return scaled


def delta_distances(Z: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Pairwise Delta distances between the z-scored document profiles.

    ``classic`` is Burrows's original mean absolute difference; ``cosine`` is the later variant that
    measures better on most corpora and is the one reported by default here.
    """
    if metric == "classic":
        return np.abs(Z[:, None, :] - Z[None, :, :]).mean(axis=2)
    if metric == "cosine":
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        unit = Z / np.where(norms < 1e-12, 1.0, norms)
        return np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
    raise ValueError(f"unknown Delta metric {metric!r}; use 'classic' or 'cosine'")


def delta_matrix(docs: list[list[str]], n_words: int = DEFAULT_MFW,
                 metric: str = "cosine") -> tuple[np.ndarray, list[str]]:
    """Distances between documents, and the vocabulary they were measured on."""
    vocabulary = most_frequent_words(docs, n_words)
    return delta_distances(zscores(frequencies(docs, vocabulary)), metric), vocabulary


def attribute(docs: list[list[str]], labels: list[str], groups: list[str] | None = None,
              n_words: int = DEFAULT_MFW, metric: str = "cosine") -> list[str]:
    """Leave-one-group-out nearest neighbour: the classic Delta attribution rule.

    ``groups`` holds whatever must be held out together - the work a passage came from, so a passage
    is never attributed by its own neighbours, which is the mistake that makes stylometry look far
    better than it is. With no groups, each document is held out alone.

    The z-scores are computed once over the whole corpus, as Burrows does; only the *candidates* are
    held out. Returns one predicted label per document.
    """
    if len(docs) != len(labels):
        raise ValueError("every document needs a label")
    if groups is not None and len(groups) != len(docs):
        raise ValueError("every document needs a holdout group")
    if len(docs) < 2:
        raise ValueError("attribution needs at least two documents")
    held = list(groups) if groups is not None else [str(i) for i in range(len(docs))]
    distances, _ = delta_matrix(docs, n_words, metric)
    predictions = []
    for i in range(len(docs)):
        candidates = [j for j in range(len(docs)) if held[j] != held[i]]
        if not candidates:
            predictions.append(None)
            continue
        predictions.append(labels[min(candidates, key=lambda j: distances[i, j])])
    return predictions
