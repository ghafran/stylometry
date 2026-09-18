"""Drill from a language down to a verse, under whichever strategy you choose.

The question this answers is not "what is the answer" but "what does this evidence look like". Pick a
language, see its collections; pick a strategy, see where those collections sit in the style space
that strategy defines; then open a collection to see its books, a book to see its chapters, a chapter
to see its verses. The strategy can be changed at any level and everything re-plots, which is the
point: a grouping that survives a change of strategy means something, and one that rearranges itself
每 time does not.

Each strategy defines its own space, so coordinates are only comparable within a strategy. They are
the first two principal components of that strategy's standardised features, fitted once over every
verse of the language, so a verse, its chapter and its book are all placed on the same axes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .strategies import BY_KEY, REGISTRY, describe

MAX_MARKERS = 6


def _standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    out = (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)
    out[:, sd < 1e-12] = 0.0
    return out


def strategy_space(key: str, verses: list[dict], language: str, seed: int = 0):
    """Standardised features for one strategy, and every verse's place in its first two components."""
    strategy = BY_KEY[key]
    if not strategy.available_for(language):
        return None
    docs = [v["text_bare"].split() for v in verses]
    texts = [v.get("text") or v["text_bare"] for v in verses]
    pos = [v.get("pos") or [] for v in verses]
    X, names = strategy.build(docs, texts, language, pos, seed)
    X = np.asarray(X, dtype=float)
    if X.size == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
        return None
    Z = _standardise(X)
    if Z.shape[1] < 2 or np.allclose(Z, 0):
        coords = np.zeros((len(verses), 2))
        if Z.shape[1] >= 1:
            coords[:, 0] = Z[:, 0]
    else:
        coords = PCA(n_components=2, random_state=seed).fit_transform(Z)
    return {"names": names, "Z": Z, "coords": coords}


def _markers(Z: np.ndarray, names: list[str], rows: list[int]) -> list[list]:
    """The features that most separate this group from the corpus, in standard deviations."""
    if not rows:
        return []
    mean = Z[rows].mean(axis=0)
    order = np.argsort(-np.abs(mean))[:MAX_MARKERS]
    return [[names[i], round(float(mean[i]), 2)] for i in order if abs(mean[i]) > 0.15]


def build(verses: list[dict], language: str, seed: int = 0, progress=print) -> dict:
    """The whole tree, with every node placed under every available strategy."""
    verses = [v for v in verses if v.get("text_bare", "").strip()]
    if len(verses) < 10:
        raise ValueError("the explorer needs at least ten verses")

    spaces: dict[str, dict] = {}
    for strategy in REGISTRY:
        if not strategy.available_for(language):
            continue
        progress(f"  {strategy.key} ...")
        space = strategy_space(strategy.key, verses, language, seed)
        if space is not None:
            spaces[strategy.key] = space
    if not spaces:
        raise ValueError(f"no strategy produced features for {language}")
    keys = list(spaces)

    # Index the hierarchy: collection > work > chapter > verse.
    by_collection: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, v in enumerate(verses):
        by_collection[v.get("collection") or "?"][v["work"]][str(v.get("chapter") or "?")].append(i)

    def node(rows: list[int], label: str, level: str, extra=None) -> dict:
        out = {
            "label": label, "level": level, "n_verses": len(rows),
            "n_tokens": int(sum(int(verses[i].get("n_tokens") or 0) for i in rows)),
            # Packed flat in `keys` order rather than keyed by strategy: repeating fourteen strategy
            # names on every one of forty thousand verses costs more than the numbers themselves.
            "xy": [round(float(x), 2) for k in keys for x in spaces[k]["coords"][rows].mean(axis=0)],
            "markers": {k: _markers(spaces[k]["Z"], spaces[k]["names"], rows) for k in keys},
        }
        if extra:
            out.update(extra)
        return out

    tree = []
    for collection in sorted(by_collection):
        works_out = []
        collection_rows: list[int] = []
        for work in sorted(by_collection[collection]):
            chapters_out = []
            work_rows: list[int] = []
            chapters = by_collection[collection][work]
            for chapter in sorted(chapters, key=lambda c: (len(c), c)):
                rows = chapters[chapter]
                work_rows += rows
                chapters_out.append(node(rows, chapter, "chapter", {
                    "v": [[
                        verses[i].get("verse") or "",
                        (verses[i].get("text") or verses[i]["text_bare"])[:240],
                        int(verses[i].get("n_tokens") or 0),
                        [round(float(x), 1) for k in keys for x in spaces[k]["coords"][i]],
                    ] for i in rows],
                }))
            collection_rows += work_rows
            title = verses[work_rows[0]].get("work_title") or work
            works_out.append({**node(work_rows, title, "work", {"code": work}), "children": chapters_out})
        tree.append({**node(collection_rows, collection, "collection"), "children": works_out})

    return {
        "language": language,
        "n_verses": len(verses),
        "keys": keys,          # the order every packed xy array follows
        "verse_fields": ["ref", "text", "n_tokens", "xy"],
        "strategies": [s for s in describe(language) if s["key"] in keys],
        "unavailable": [s["key"] for s in describe(language) if s["key"] not in keys],
        "tree": tree,
        "note": "Coordinates are the first two principal components of each strategy's standardised "
                "features, fitted over every verse of this language. They are comparable within a "
                "strategy and not between strategies.",
    }
