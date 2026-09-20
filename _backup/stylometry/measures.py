"""Distances between texts represented as probability distributions.

Cosine similarity compares feature vectors by angle, which is what Delta uses. Jensen-Shannon compares
*distributions* - how much the word probabilities of two texts differ - and unlike Kullback-Leibler it
is symmetric, finite even when one text uses a word the other never does, and its square root is a
true metric. That makes it the safer choice when one side is short.
"""
from __future__ import annotations

import numpy as np


def _distribution(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    return counts / np.where(total < 1e-12, 1.0, total)


def jensen_shannon(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Divergence between two distributions, 0 when identical and 1 when disjoint (base 2)."""
    p, q = _distribution(np.asarray(p, dtype=float)), _distribution(np.asarray(q, dtype=float))
    if p.shape != q.shape:
        raise ValueError("distributions must share a vocabulary")
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / m[mask])))

    return float((0.5 * kl(p) + 0.5 * kl(q)) / np.log(base))


def jensen_shannon_matrix(counts: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence between the rows of a count matrix."""
    n = len(counts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = jensen_shannon(counts[i], counts[j])
    return D


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between feature vectors."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.where(norms < 1e-12, 1.0, norms)
    return np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
