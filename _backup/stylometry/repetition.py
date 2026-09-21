"""How much a text repeats itself, measured so that length does not decide the answer.

The project already carries ``misc:type_token_ratio``, which is the crudest member of this family and
falls with length by construction: a longer text runs out of new words. The measures here are the
standard vocabulary-richness indices, most built to be stable across length.

Yule (1944) devised K for an authorship problem - the *De Imitatione Christi* dispute - and framed it
as a repeat rate: how concentrated a text is on words it uses again. It is the one member of the
family generally held to be length-robust.

An honest caveat belongs with the whole family. Hoover's work in the early 2000s found these indices
perform poorly for attribution against frequent-word methods: each compresses a whole text to one
number, and they are sensitive to genre, editorial normalisation and orthography. They are included
here as a block to be tested, not as a method expected to win on its own.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

NAMES = [
    "rep:yule_k",
    "rep:simpson_d",
    "rep:sichel_s",
    "rep:honore_r",
    "rep:brunet_w",
    "rep:hapax_ratio",
    "rep:mattr",
    "rep:top_word_share",
    "rep:repeat_rate",
]


def _spectrum(tokens: list[str]) -> tuple[Counter, int, int]:
    """Counts of how many types occur exactly i times, with token and type totals."""
    counts = Counter(tokens)
    return Counter(counts.values()), len(tokens), len(counts)


def yule_k(tokens: list[str]) -> float:
    """Yule's characteristic constant: 10^4 (Σ i²V_i − N) / N². Higher means more repetitive."""
    spectrum, n, _ = _spectrum(tokens)
    if n < 2:
        return 0.0
    total = sum(i * i * v for i, v in spectrum.items())
    return 1e4 * (total - n) / (n * n)


def simpson_d(tokens: list[str]) -> float:
    """Probability that two tokens drawn without replacement are the same word."""
    spectrum, n, _ = _spectrum(tokens)
    if n < 2:
        return 0.0
    return sum(v * i * (i - 1) for i, v in spectrum.items()) / (n * (n - 1))


def sichel_s(tokens: list[str]) -> float:
    """Share of the vocabulary used exactly twice; held to be length-stable."""
    spectrum, _, types = _spectrum(tokens)
    return spectrum.get(2, 0) / types if types else 0.0


def honore_r(tokens: list[str]) -> float:
    """Leans on words used exactly once. Undefined when every word is a hapax, so that is capped."""
    spectrum, n, types = _spectrum(tokens)
    if n < 2 or not types:
        return 0.0
    hapax_share = spectrum.get(1, 0) / types
    if hapax_share >= 1.0 - 1e-9:
        return 100.0 * math.log(n) / 1e-9 ** 0.5  # a finite stand-in for an undefined value
    return 100.0 * math.log(n) / (1.0 - hapax_share)


def brunet_w(tokens: list[str]) -> float:
    """N^(V^-0.165); nearly independent of length by construction."""
    _, n, types = _spectrum(tokens)
    if n < 2 or types < 1:
        return 0.0
    return float(n ** (types ** -0.165))


def hapax_ratio(tokens: list[str]) -> float:
    spectrum, _, types = _spectrum(tokens)
    return spectrum.get(1, 0) / types if types else 0.0


def mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-average type-token ratio: TTR over fixed windows, so length cannot drive it.

    Short texts are measured whole rather than reported as missing.
    """
    if window < 2:
        raise ValueError("window must be at least two tokens")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def top_word_share(tokens: list[str], top: int = 10) -> float:
    """Share of the text taken by its ten commonest words: how far it leans on a few."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return sum(c for _, c in counts.most_common(top)) / len(tokens)


def repeat_rate(tokens: list[str]) -> float:
    """Share of tokens whose word occurs more than once - repetition at its plainest."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return sum(c for c in counts.values() if c > 1) / len(tokens)


def features(tokens: list[str], mattr_window: int = 100) -> np.ndarray:
    """One row of repetition measures, in the order of ``NAMES``."""
    return np.array([
        yule_k(tokens), simpson_d(tokens), sichel_s(tokens), honore_r(tokens), brunet_w(tokens),
        hapax_ratio(tokens), mattr(tokens, mattr_window), top_word_share(tokens), repeat_rate(tokens),
    ], dtype=np.float64)


def matrix(docs: list[list[str]], mattr_window: int = 100) -> np.ndarray:
    return np.vstack([features(d, mattr_window) for d in docs])
