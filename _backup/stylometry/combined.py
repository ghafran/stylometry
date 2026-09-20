"""Repetition, placement and rate as three blocks of one representation.

Each answers a different question about a text. Rate asks which words an author reaches for, and is
what Delta measures. Placement asks where they put them, and is what a word adjacency network
measures. Repetition asks how much they lean on words already used. They are not obviously redundant,
so the question is whether combining them beats the strongest of them alone - which on this corpus is
rate, by a wide margin.

Blocks are standardised, then each is divided by its own total standard deviation before its weight is
applied, so a block cannot dominate merely by having more columns: placement contributes hundreds of
transition columns against repetition's nine, and without equalising, repetition would be inaudible.

The placement block is deliberately *not* the full transition matrix. With 100 markers that is 10,000
columns, nearly all of them zero on a passage-sized text, and measuring that is measuring the
smoothing constant. Only the transitions that actually occur across the corpus are kept.
"""
from __future__ import annotations

import math

import numpy as np

from . import repetition
from .delta import frequencies, most_frequent_words, zscores
from .wan import adjacency, markers_for

DEFAULT_WEIGHTS = {"rate": 1.0, "placement": 1.0, "repetition": 1.0}


def rate_block(docs: list[list[str]], n_words: int = 500) -> tuple[np.ndarray, list[str]]:
    """Delta's representation: z-scored relative frequencies of the most frequent words."""
    vocabulary = most_frequent_words(docs, n_words)
    return zscores(frequencies(docs, vocabulary)), [f"rate:{w}" for w in vocabulary]


def placement_block(docs: list[list[str]], language: str = "grc", n_markers: int = 100,
                    window: int = 10, keep: int = 500) -> tuple[np.ndarray, list[str]]:
    """Rates of the marker-to-marker transitions that actually occur, most frequent first.

    Each row is one document's adjacency counts, normalised by that document's total transition
    weight so length does not decide, then restricted to the ``keep`` commonest transitions corpus
    wide. A transition almost nobody uses is noise at this text length.
    """
    markers = markers_for(language, n_markers)
    counts = [adjacency(d, markers, window) for d in docs]
    stacked = np.stack(counts)
    totals = stacked.sum(axis=(1, 2), keepdims=True)
    rates = stacked / np.where(totals < 1e-12, 1.0, totals)
    corpus = stacked.sum(axis=0)
    flat = corpus.ravel()
    chosen = np.argsort(flat)[::-1][:keep]
    chosen = chosen[flat[chosen] > 0]
    names = [f"place:{markers[i // len(markers)]}>{markers[i % len(markers)]}" for i in chosen]
    return rates.reshape(len(docs), -1)[:, chosen], names


def repetition_block(docs: list[list[str]], mattr_window: int = 100) -> tuple[np.ndarray, list[str]]:
    """The vocabulary-richness indices, log-compressed where their range is extreme."""
    M = repetition.matrix(docs, mattr_window)
    heavy = [repetition.NAMES.index(n) for n in ("rep:yule_k", "rep:honore_r", "rep:brunet_w")]
    M[:, heavy] = np.log1p(np.clip(M[:, heavy], 0, None))
    return M, list(repetition.NAMES)


def _equalise(Z: np.ndarray, weight: float) -> np.ndarray:
    """Standardise, then scale to unit total variance, so column count cannot buy influence."""
    if Z.shape[1] == 0:
        return Z
    mean, sd = Z.mean(axis=0), Z.std(axis=0)
    S = (Z - mean) / np.where(sd < 1e-12, 1.0, sd)
    S[:, sd < 1e-12] = 0.0
    variance = float(np.var(S, axis=0).sum())
    return S / math.sqrt(variance) * weight if variance > 1e-12 else np.zeros_like(S)


def build(docs: list[list[str]], language: str = "grc", weights: dict | None = None,
          n_words: int = 500, n_markers: int = 100, window: int = 10,
          keep: int = 500) -> tuple[np.ndarray, list[str]]:
    """One matrix from whichever blocks carry positive weight."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if not any(w > 0 for w in weights.values()):
        raise ValueError("at least one block must carry positive weight")
    if any(w < 0 or not math.isfinite(w) for w in weights.values()):
        raise ValueError("block weights must be finite and nonnegative")
    parts, names = [], []
    builders = {
        "rate": lambda: rate_block(docs, n_words),
        "placement": lambda: placement_block(docs, language, n_markers, window, keep),
        "repetition": lambda: repetition_block(docs),
    }
    for block, make in builders.items():
        if weights.get(block, 0) > 0:
            Z, block_names = make()
            parts.append(_equalise(Z, weights[block]))
            names += block_names
    return np.hstack(parts), names


def attribute(docs: list[list[str]], labels: list[str], language: str = "grc",
              groups: list[str] | None = None, weights: dict | None = None,
              **kwargs) -> list[str]:
    """Leave-one-group-out nearest neighbour under cosine distance over the combined blocks."""
    if len(docs) != len(labels):
        raise ValueError("every document needs a label")
    if groups is not None and len(groups) != len(docs):
        raise ValueError("every document needs a holdout group")
    if len(docs) < 2:
        raise ValueError("attribution needs at least two documents")
    X, _ = build(docs, language, weights, **kwargs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.where(norms < 1e-12, 1.0, norms)
    D = 1.0 - unit @ unit.T
    held = list(groups) if groups is not None else [str(i) for i in range(len(docs))]
    out = []
    for i in range(len(docs)):
        candidates = [j for j in range(len(docs)) if held[j] != held[i]]
        out.append(labels[min(candidates, key=lambda j: D[i, j])] if candidates else None)
    return out
