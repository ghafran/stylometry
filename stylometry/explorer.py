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
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .cluster import cluster, select_k, subsample_stability, supported_k_order
from .strategies import BY_KEY, REGISTRY, describe

MAX_MARKERS = 6

# A partition needs enough units to be worth testing. Eight is already generous: two groups fitted
# to eight documents is a weak claim, and the stability gate below is what actually has to carry it.
GROUP_MIN_UNITS = 8
GROUP_MAX_K = 8
GROUP_MIN_ARI = 0.8

# Section 1 of this project measured that verse-sized units cannot support attribution, which is why
# the analysis works on 1,000-token passages. A partition of units below that floor can still be
# stable - a hapax ratio over eighteen tokens takes few distinct values, so it clusters cleanly - and
# be a property of the arithmetic rather than of anybody's hand. The number is shown with that said.
GROUP_RELIABLE_TOKENS = 1000

# Why a level reports one group, in the words the page shows.
GROUP_REASONS = {
    "too_few_units": "too few units here to look for groups",
    "no_variation": "this strategy finds no variation between them",
    "no_supported_split": "no split is supported: they read as one style",
    "unstable": "a split scored well but did not survive resampling",
}


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


def group_units(P: np.ndarray, seed: int = 0) -> dict:
    """Partition sibling units - books of a collection, chapters of a book, verses of a chapter.

    The same rule the main pipeline uses, applied to whatever is being listed: a candidate must beat
    one Gaussian on BIC and have positive silhouette to be a candidate at all, and must then survive
    refitting on 80% subsamples with an adjusted Rand index of at least 0.8 against the full fit. If
    nothing survives, the answer is one group, and the reason is recorded rather than hidden.

    ``P`` is one pooled profile per unit. Pooling first is the point: a chapter of five hundred words
    is a document, where its individual verses of eighteen are not.
    """
    n = len(P)
    if n < GROUP_MIN_UNITS:
        return {"k": 1, "labels": [1] * n, "reason": "too_few_units", "n_units": n, "sizes": [n]}
    Z = _standardise(np.asarray(P, dtype=float))
    if Z.shape[1] == 0 or np.allclose(Z, 0) or not np.isfinite(Z).all():
        return {"k": 1, "labels": [1] * n, "reason": "no_variation", "n_units": n, "sizes": [n]}
    comps = min(30, Z.shape[1], n - 1)
    X = PCA(n_components=comps, random_state=seed).fit_transform(Z) if comps >= 2 else Z
    # Never ask for more groups than roughly four units each would allow; a k that leaves singleton
    # groups is arithmetic, not evidence.
    kmax = min(GROUP_MAX_K, n - 1, max(2, n // 4))
    _, ktable = select_k(X, 2, kmax, seed=seed)
    order = supported_k_order(ktable)
    if not order:
        return {"k": 1, "labels": [1] * n, "reason": "no_supported_split", "n_units": n, "sizes": [n]}
    scored = {row["k"]: row for row in ktable}
    for k in order:
        labels, _, _ = cluster(X, k, seed=seed)
        stability = subsample_stability(X, labels, k, seed=seed)
        if stability.get("available") and stability["min_ari"] >= GROUP_MIN_ARI:
            members = [int(a[1:]) for a in labels]
            return {"k": k, "labels": members, "reason": None, "n_units": n,
                    # `cluster` ranks by size, so A1 is always the largest group.
                    "sizes": [members.count(g) for g in range(1, k + 1)],
                    "min_ari": round(stability["min_ari"], 3),
                    "silhouette": round(float(scored[k]["silhouette"]), 3)}
    return {"k": 1, "labels": [1] * n, "reason": "unstable", "n_units": n, "sizes": [n],
            "rejected": [int(k) for k in order]}


def _markers(Z: np.ndarray, names: list[str], rows: list[int]) -> list[list]:
    """The features that most separate this group from the corpus, in standard deviations."""
    if not rows:
        return []
    mean = Z[rows].mean(axis=0)
    order = np.argsort(-np.abs(mean))[:MAX_MARKERS]
    return [[names[i], round(float(mean[i]), 2)] for i in order if abs(mean[i]) > 0.15]


def _profiles(Z: np.ndarray, row_sets: list[list[int]]) -> np.ndarray:
    """One pooled profile per unit: the mean of its verses in the strategy's standardised space."""
    return np.vstack([Z[rows].mean(axis=0) for rows in row_sets])


def _median_tokens(tokens: np.ndarray, row_sets: list[list[int]]) -> int:
    """How long the units being partitioned actually are, which decides whether to believe the split."""
    if not row_sets:
        return 0
    return int(np.median([int(tokens[rows].sum()) for rows in row_sets]))


def _partitions(Z: np.ndarray, shape: list, tokens: np.ndarray, seed: int = 0) -> dict:
    """Every sibling set the explorer lists, partitioned under one strategy.

    Each level groups the units it actually shows: a collection groups its books, a book groups its
    chapters, a chapter groups its verses. The language gets two - its collections, which is what the
    page lists, and all of its books, which is the number worth quoting.
    """
    limits = _one_thread()
    out: dict = {"books": {}, "chapters": {}, "verses": {}}
    collection_rows, all_book_rows = [], []
    for ci, (_, works) in enumerate(shape):
        book_rows = []
        for wi, (_, chapters) in enumerate(works):
            chapter_rows = [rows for _, rows in chapters]
            for chi, rows in enumerate(chapter_rows):
                singles = [[i] for i in rows]
                out["verses"][(ci, wi, chi)] = {**group_units(Z[rows], seed),
                                                "tokens": _median_tokens(tokens, singles)}
            out["chapters"][(ci, wi)] = {**group_units(_profiles(Z, chapter_rows), seed),
                                         "tokens": _median_tokens(tokens, chapter_rows)}
            book_rows.append([i for rows in chapter_rows for i in rows])
        out["books"][ci] = {**group_units(_profiles(Z, book_rows), seed),
                            "tokens": _median_tokens(tokens, book_rows)}
        all_book_rows += book_rows
        collection_rows.append([i for rows in book_rows for i in rows])
    out["collections"] = {**group_units(_profiles(Z, collection_rows), seed),
                          "tokens": _median_tokens(tokens, collection_rows)}
    out["language_books"] = {**group_units(_profiles(Z, all_book_rows), seed),
                             "tokens": _median_tokens(tokens, all_book_rows)}
    if limits is not None:
        limits.unregister()
    return out


def _one_thread():
    """Keep each worker to one BLAS thread.

    The matrices here are tiny - a chapter is thirty rows - so the maths is call-overhead bound, and
    twenty processes each opening twenty OpenMP threads is four hundred threads fighting over
    sixteen cores. Returns the limiter to release, or None when threadpoolctl is unavailable.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=1)


def _partition_task(args):
    Z, shape, tokens, seed = args
    return _partitions(Z, shape, tokens, seed)


def _pack(parts: list[dict]) -> dict:
    """A partition of this node's children, flat in `keys` order like every other packed array."""
    return {"gk": [p["k"] for p in parts],
            "gr": [p.get("reason") for p in parts],
            "gari": [p.get("min_ari") for p in parts],
            "gsz": [p.get("sizes") or [p["n_units"]] for p in parts],
            "gn": parts[0]["n_units"] if parts else 0,
            "gtok": parts[0].get("tokens", 0) if parts else 0}


def build(verses: list[dict], language: str, seed: int = 0, progress=print,
          workers: int | None = None) -> dict:
    """The whole tree, with every node placed and every sibling set grouped under every strategy."""
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

    # Index the hierarchy, then freeze it in the order the pages list it, so an index means the same
    # thing in the tree and in every partition.
    by_collection: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, v in enumerate(verses):
        by_collection[v.get("collection") or "?"][v["work"]][str(v.get("chapter") or "?")].append(i)
    shape = [(collection,
              [(work, [(c, by_collection[collection][work][c])
                       for c in sorted(by_collection[collection][work], key=lambda c: (len(c), c))])
               for work in sorted(by_collection[collection])])
             for collection in sorted(by_collection)]

    # Each strategy partitions independently, so they fan out. Serially this is hours: the work is
    # tens of thousands of very small fits, which is call-overhead bound, and threads lose to the GIL.
    if workers is None:
        workers = min(len(keys), os.cpu_count() or 1)
    tokens = np.array([int(v.get("n_tokens") or 0) for v in verses])
    groups = {}
    if workers > 1 and len(keys) > 1:
        progress(f"  grouping {len(keys)} strategies across {workers} processes ...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = pool.map(_partition_task, [(spaces[k]["Z"], shape, tokens, seed) for k in keys])
            for key, result in zip(keys, done):
                groups[key] = result
                progress(f"    grouped {key}")
    else:
        for key in keys:
            progress(f"  grouping under {key} ...")
            groups[key] = _partitions(spaces[key]["Z"], shape, tokens, seed)

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
    for ci, (collection, works) in enumerate(shape):
        works_out = []
        collection_rows: list[int] = []
        for wi, (work, chapters) in enumerate(works):
            chapters_out = []
            work_rows: list[int] = []
            for chi, (chapter, rows) in enumerate(chapters):
                work_rows += rows
                verse_part = {k: groups[k]["verses"][(ci, wi, chi)] for k in keys}
                chapters_out.append(node(rows, chapter, "chapter", {
                    "v": [[
                        verses[i].get("verse") or "",
                        (verses[i].get("text") or verses[i]["text_bare"])[:240],
                        int(verses[i].get("n_tokens") or 0),
                        [round(float(x), 1) for k in keys for x in spaces[k]["coords"][i]],
                        [verse_part[k]["labels"][j] for k in keys],
                    ] for j, i in enumerate(rows)],
                    "g": [groups[k]["chapters"][(ci, wi)]["labels"][chi] for k in keys],
                    **_pack([verse_part[k] for k in keys]),
                }))
            collection_rows += work_rows
            title = verses[work_rows[0]].get("work_title") or work
            works_out.append({
                **node(work_rows, title, "work", {
                    "code": work,
                    "g": [groups[k]["books"][ci]["labels"][wi] for k in keys],
                    **_pack([groups[k]["chapters"][(ci, wi)] for k in keys]),
                }),
                "children": chapters_out})
        tree.append({
            **node(collection_rows, collection, "collection", {
                "g": [groups[k]["collections"]["labels"][ci] for k in keys],
                **_pack([groups[k]["books"][ci] for k in keys]),
            }),
            "children": works_out})

    return {
        "language": language,
        "n_verses": len(verses),
        "keys": keys,          # the order every packed xy, g and gk array follows
        "verse_fields": ["ref", "text", "n_tokens", "xy", "group"],
        "strategies": [s for s in describe(language) if s["key"] in keys],
        "unavailable": [s["key"] for s in describe(language) if s["key"] not in keys],
        "tree": tree,
        # The headline: every book of this language, grouped. `collections` is the partition of the
        # handful of collections the language page lists, which is a much weaker thing.
        "book_groups": _pack([groups[k]["language_books"] for k in keys]),
        "collection_groups": _pack([groups[k]["collections"] for k in keys]),
        "group_reasons": GROUP_REASONS,
        "reliable_tokens": GROUP_RELIABLE_TOKENS,
        "note": "Coordinates are the first two principal components of each strategy's standardised "
                "features, fitted over every verse of this language. They are comparable within a "
                "strategy and not between strategies.",
        "group_note": "A style group is a partition of the units a level lists - a collection groups "
                      "its books, a book its chapters, a chapter its verses - fitted on pooled "
                      "profiles and kept only if it survives refitting on 80% subsamples at an "
                      "adjusted Rand index of 0.8 or better. One group means no split was supported, "
                      "not that one hand wrote it.",
    }
