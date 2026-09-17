"""Word adjacency networks: style as the *arrangement* of function words, not their rates.

Segarra, Eisen & Ribeiro (IEEE Trans. Signal Processing, 2015) represent an author by a directed graph
over function words alone. Each occurrence of a marker word casts weight onto the markers that follow
it within a short window, discounted by distance. Row-normalised, the graph is a Markov chain: given
that this author has just written δέ, what do they write next?

This measures something no other feature in the project does. Delta counts how often an author reaches
for a word; this asks where they put it. For Koine Greek that is a pointed question, because the
postpositive particles - δέ, γάρ, μέν, οὖν - are grammatically barred from opening a clause and carry
almost no subject matter, so their placement is habit rather than content. The same property should
make the method resistant to the genre confound that dominates ``cluster.py``: what follows what among
function words has little to do with whether a passage is prophecy or narrative.

Markers default to the language's function words, so no content word can enter the graph.
"""
from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = 10


def markers_for(language: str, limit: int | None = None) -> list[str]:
    """The language's function words, which is what the method is defined over."""
    from .lang import function_words

    words = list(function_words(language))
    return words[:limit] if limit else words


def adjacency(tokens: list[str], markers: list[str], window: int = DEFAULT_WINDOW,
              decay: str = "inverse") -> np.ndarray:
    """Directed, distance-discounted counts of which marker follows which.

    ``inverse`` weights a marker d words later by 1/d, the paper's default; ``uniform`` counts every
    position in the window equally. Only marker-to-marker transitions are recorded, and intervening
    content words are passed over rather than breaking the link, so the graph describes the skeleton
    of the sentence.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if decay not in ("inverse", "uniform"):
        raise ValueError(f"unknown decay {decay!r}; use 'inverse' or 'uniform'")
    if not markers:
        raise ValueError("a word adjacency network needs at least one marker word")
    index = {w: i for i, w in enumerate(markers)}
    W = np.zeros((len(markers), len(markers)), dtype=np.float64)
    positions = [(i, index[t]) for i, t in enumerate(tokens) if t in index]
    for a, (pos_a, row) in enumerate(positions):
        for pos_b, col in positions[a + 1:]:
            d = pos_b - pos_a
            if d > window:
                break
            W[row, col] += 1.0 / d if decay == "inverse" else 1.0
    return W


def transition_matrix(W: np.ndarray, smoothing: float = 0.01) -> np.ndarray:
    """Row-normalise into a Markov chain, smoothed so an unseen transition is rare, not impossible.

    Without smoothing a marker the text never follows gives probability zero, and the likelihood
    score below would be negative infinity for any text that does use it - one unseen bigram would
    outweigh all the evidence.
    """
    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative")
    P = W + smoothing
    return P / P.sum(axis=1, keepdims=True)


def profile(tokens: list[str], markers: list[str], window: int = DEFAULT_WINDOW,
            decay: str = "inverse", smoothing: float = 0.01) -> np.ndarray:
    return transition_matrix(adjacency(tokens, markers, window, decay), smoothing)


def divergence(unknown: np.ndarray, candidate: np.ndarray) -> float:
    """How poorly the candidate's habits explain the unknown text's, lower being closer.

    The Kullback-Leibler divergence between the two chains, averaged over rows: for each marker, how
    surprised the candidate's chain is by what the unknown text actually does next. It is asymmetric
    by nature - the question "would this author have written this?" is not symmetric - so the two
    directions are averaged, giving a usable distance.
    """
    if unknown.shape != candidate.shape:
        raise ValueError("word adjacency networks must share a marker vocabulary")

    def one_way(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * (np.log(p) - np.log(q))) / len(p))

    return 0.5 * (one_way(unknown, candidate) + one_way(candidate, unknown))


def distances(profiles: list[np.ndarray]) -> np.ndarray:
    n = len(profiles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = divergence(profiles[i], profiles[j])
    return D


def attribute(docs: list[list[str]], labels: list[str], language: str = "grc",
              groups: list[str] | None = None, window: int = DEFAULT_WINDOW,
              n_markers: int | None = None, decay: str = "inverse",
              smoothing: float = 0.01) -> list[str]:
    """Leave-one-group-out nearest neighbour over adjacency networks.

    ``groups`` holds together whatever must be held out together - the work a passage came from - so
    a passage is never attributed by its own neighbours.
    """
    if len(docs) != len(labels):
        raise ValueError("every document needs a label")
    if groups is not None and len(groups) != len(docs):
        raise ValueError("every document needs a holdout group")
    if len(docs) < 2:
        raise ValueError("attribution needs at least two documents")
    markers = markers_for(language, n_markers)
    held = list(groups) if groups is not None else [str(i) for i in range(len(docs))]
    profiles = [profile(d, markers, window, decay, smoothing) for d in docs]
    D = distances(profiles)
    out = []
    for i in range(len(docs)):
        candidates = [j for j in range(len(docs)) if held[j] != held[i]]
        out.append(labels[min(candidates, key=lambda j: D[i, j])] if candidates else None)
    return out


def attribute_pooled(docs: list[list[str]], labels: list[str], language: str = "grc",
                     groups: list[str] | None = None, window: int = DEFAULT_WINDOW,
                     n_markers: int | None = None, decay: str = "inverse",
                     smoothing: float = 0.01) -> list[str]:
    """One network per candidate author, built from their held-in text - the method as published.

    A single document is thin evidence for a graph: 100 markers means 10,000 possible transitions,
    and a 1,000-token passage supplies a few hundred marker tokens to fill them. Segarra, Eisen and
    Ribeiro therefore pool each author's texts into one chain and score the unknown against that.
    Everything sharing the unknown's holdout group is excluded from every pool, so a work never
    contributes to the profile it is judged against.
    """
    if len(docs) != len(labels):
        raise ValueError("every document needs a label")
    if groups is not None and len(groups) != len(docs):
        raise ValueError("every document needs a holdout group")
    markers = markers_for(language, n_markers)
    held = list(groups) if groups is not None else [str(i) for i in range(len(docs))]
    counts = [adjacency(d, markers, window, decay) for d in docs]
    out = []
    for i in range(len(docs)):
        unknown = transition_matrix(counts[i], smoothing)
        best, best_d = None, float("inf")
        for label in sorted(set(labels)):
            pool = [counts[j] for j in range(len(docs)) if labels[j] == label and held[j] != held[i]]
            if not pool:
                continue
            candidate = transition_matrix(sum(pool), smoothing)
            if (d := divergence(unknown, candidate)) < best_d:
                best, best_d = label, d
        out.append(best)
    return out
