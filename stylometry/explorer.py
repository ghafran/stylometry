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

# How many assumed authors the author view will offer to lay out. This is a limit on how much gets
# written into the page, not a claim: see `author_partitions`, which measures why the count cannot be
# inferred from the text and so has to be the reader's to set.
AUTHOR_OFFER_MAX = 60

# Why a level infers one author, in the words the page shows. Every one of these is "the text did
# not support more", never "one hand wrote it": the same test returned one for Codex Sinaiticus.
GROUP_REASONS = {
    "too_few_units": "too few units here to look for more than one",
    "no_variation": "this strategy finds no variation between them",
    "no_supported_split": "no division into more than one is supported",
    "unstable": "a division scored well but did not survive resampling",
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


def author_partitions(P: np.ndarray, seed: int = 0) -> dict:
    """Assume each style is a hand, and lay out that assumption at every author count worth offering.

    This is a different question from ``group_units`` and answers to a different standard. That one
    refuses to split unless the split survives resampling, which is right when the question is
    "is there evidence of more than one hand". Here the question is "if there are N hands, which
    text belongs to which", and N is supplied rather than discovered - because on the English corpus,
    where the answer is known, it could not be discovered: selecting N by silhouette returned 2 for
    every collection tested, whose true counts were 3, 5, 5 and 13. So N is offered as a range for
    the reader to set, and the honest reading of these labels is "under an assumption of N authors",
    never "there are N authors".

    The geometry is the one that recovered known English authors best on its *worst* collection
    rather than its best - length-weighted profiles, thirty components, k-means - at a mean adjusted
    Rand index of 0.43 against the truth. Being handed the right N, it still puts roughly a third of
    the works in the wrong hand.
    """
    n = len(P)
    if n < 3:
        return {"n_units": n, "range": [], "labels": {}}
    Z = _standardise(np.asarray(P, dtype=float))
    if Z.shape[1] == 0 or np.allclose(Z, 0) or not np.isfinite(Z).all():
        return {"n_units": n, "range": [], "labels": {}}
    comps = min(30, Z.shape[1], n - 1)
    X = PCA(n_components=comps, random_state=seed).fit_transform(Z) if comps >= 2 else Z
    # Two limits, neither of them a claim about how many hands there could be. Every count is stored
    # for every strategy, so the offer stops at AUTHOR_OFFER_MAX to keep the page from bloating; and
    # two books identical in this strategy's features cannot be told apart into two hands, so it also
    # stops at however many distinct profiles there are. Nothing here decides the answer is below it.
    distinct = len(np.unique(X, axis=0))
    top = min(n - 1, AUTHOR_OFFER_MAX, distinct)
    if top < 2:
        return {"n_units": n, "range": [1, 1], "labels": {1: [1] * n}}
    out = {1: [1] * n}
    for k in range(2, top + 1):
        labels, _, _ = cluster(X, k, seed=seed)
        out[k] = [int(a[1:]) for a in labels]
    return {"n_units": n, "range": [1, top], "labels": out}


def author_fit(labels_by_k: dict, truth: list[str]) -> dict | None:
    """How well an assumed-author partition matches authors that are actually known.

    Only English has this. It is the number that says what the same procedure is worth everywhere
    it cannot be checked.
    """
    known = [i for i, a in enumerate(truth) if a]
    if len(known) < 4:
        return None
    names = [truth[i] for i in known]
    true_k = len(set(names))
    if true_k < 2 or true_k not in labels_by_k:
        return None
    from sklearn.metrics import adjusted_rand_score
    at_true = adjusted_rand_score(names, [labels_by_k[true_k][i] for i in known])
    best_k, best = max(((k, adjusted_rand_score(names, [labels_by_k[k][i] for i in known]))
                        for k in labels_by_k), key=lambda t: t[1])
    return {"true_k": true_k, "n_known": len(known), "ari": round(float(at_true), 3),
            "best_k": int(best_k), "best_ari": round(float(best), 3),
            "recovered": _recovered(names, [labels_by_k[true_k][i] for i in known])}


def _recovered(truth: list[str], labels: list[int], share: float = 0.8) -> int:
    """How many authors a partition actually finds, counted from both sides.

    An adjusted Rand index flatters a corpus with one big author in it. The English language-level
    partition scores 0.908 at seven hands while putting Austen, Bronte, Dickens, Eliot and Hardy in
    one of them, because Hamilton's fifty-one papers are half the known books and he comes out
    clean. So count an author found only when most of their work lands in one group *and* most of
    that group is theirs. Recall alone scores four of those five novelists as perfectly recovered at
    the true count of thirteen: all three of Bronte, all three of Eliot, all three of Austen and all
    three of Hardy sit in one group - the same group, which holds fourteen books.
    """
    found = 0
    for name in set(truth):
        mine = [i for i, t in enumerate(truth) if t == name]
        for g in set(labels):
            inside = [i for i in mine if labels[i] == g]
            size = sum(1 for l in labels if l == g)
            if len(inside) >= share * len(mine) and len(inside) >= share * size:
                found += 1
                break
    return found


def _markers(Z: np.ndarray, names: list[str], rows: list[int]) -> list[list]:
    """The features that most separate this group from the corpus, in standard deviations."""
    if not rows:
        return []
    mean = Z[rows].mean(axis=0)
    order = np.argsort(-np.abs(mean))[:MAX_MARKERS]
    return [[names[i], round(float(mean[i]), 2)] for i in order if abs(mean[i]) > 0.15]


def _profiles(Z: np.ndarray, row_sets: list[list[int]], tokens: np.ndarray | None = None) -> np.ndarray:
    """One pooled profile per unit: its verses averaged in the strategy's standardised space.

    Weighted by length, because the unit's rate of a feature is what its whole text does, not what
    its verses do on average: a 300-token verse and an 8-token verse are not two equal opinions
    about how often this author writes *and*. Measured on the English corpus, where the authors are
    known, weighting lifts author recovery from an adjusted Rand index of 0.20 to 0.43.
    """
    if tokens is None:
        return np.vstack([Z[rows].mean(axis=0) for rows in row_sets])
    out = []
    for rows in row_sets:
        w = tokens[rows].astype(float)
        total = w.sum()
        out.append(Z[rows].T @ (w / total) if total > 0 else Z[rows].mean(axis=0))
    return np.vstack(out)


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
    out: dict = {"books": {}, "chapters": {}, "verses": {}, "author_books": {}}
    collection_rows, all_book_rows = [], []
    for ci, (_, works) in enumerate(shape):
        book_rows = []
        for wi, (_, chapters) in enumerate(works):
            chapter_rows = [rows for _, rows in chapters]
            for chi, rows in enumerate(chapter_rows):
                singles = [[i] for i in rows]
                out["verses"][(ci, wi, chi)] = {**group_units(Z[rows], seed),
                                                "tokens": _median_tokens(tokens, singles)}
            out["chapters"][(ci, wi)] = {**group_units(_profiles(Z, chapter_rows, tokens), seed),
                                         "tokens": _median_tokens(tokens, chapter_rows)}
            book_rows.append([i for rows in chapter_rows for i in rows])
        book_profiles = _profiles(Z, book_rows, tokens)
        out["books"][ci] = {**group_units(book_profiles, seed),
                            "tokens": _median_tokens(tokens, book_rows)}
        out["author_books"][ci] = author_partitions(book_profiles, seed)
        all_book_rows += book_rows
        collection_rows.append([i for rows in book_rows for i in rows])
    out["collections"] = {**group_units(_profiles(Z, collection_rows, tokens), seed),
                          "tokens": _median_tokens(tokens, collection_rows)}
    out["language_books"] = {**group_units(_profiles(Z, all_book_rows, tokens), seed),
                             "tokens": _median_tokens(tokens, all_book_rows)}
    # The same units again under the other question: not "is a split supported" but "if there are N
    # hands, whose is this". Books only - section 1 measured that verses cannot carry attribution.
    out["author_language"] = author_partitions(_profiles(Z, all_book_rows, tokens), seed)
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


def _pack_authors(parts: list[dict], known: list[str]) -> dict:
    """An assumed-author partition of these books at every offered count, flat in `keys` order.

    ``af`` is the only honest calibration the project has: where the authors are known it is how
    well this procedure recovers them, and where they are not it is null, which is most of the time.
    """
    return {"an": parts[0]["n_units"] if parts else 0,
            # Per strategy, not from the first one: a strategy that finds no variation between these
            # books offers no range at all, and must not inherit a slider from one that does.
            "arange": [p["range"] for p in parts],
            "al": [{str(k): v for k, v in p["labels"].items()} for p in parts],
            "af": [author_fit(p["labels"], known) for p in parts],
            "aknown": known if any(known) else []}


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

    # The language-wide partition of every book runs in the same nested order `_partitions` built it,
    # so a single running index maps it back onto the tree. It is carried per book because the
    # alternative - reading a book's group *within its own collection* and comparing that across
    # collections - is meaningless: both label sets start at A1 whatever the data says.
    book_index = 0
    tree = []
    for ci, (collection, works) in enumerate(shape):
        works_out = []
        collection_rows: list[int] = []
        collection_book_span = (book_index, book_index + len(works))
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
                        # Who the source says is speaking; empty outside the hadith collections.
                        verses[i].get("attribution") or "",
                    ] for j, i in enumerate(rows)],
                    "g": [groups[k]["chapters"][(ci, wi)]["labels"][chi] for k in keys],
                    **_pack([verse_part[k] for k in keys]),
                }))
            collection_rows += work_rows
            title = verses[work_rows[0]].get("work_title") or work
            first = verses[work_rows[0]]
            works_out.append({
                **node(work_rows, title, "work", {
                    "code": work,
                    # Its place in the language-wide book order, so a page can read this book out of
                    # the language's assumed-author partition the way `gl` does for style groups.
                    "bi": book_index,
                    # Who is actually known to have written it. Empty for every scripture corpus -
                    # which is the point of carrying an English corpus where it is not.
                    "author": (first.get("author") or "") if first.get("known_author") else "",
                    "genre": first.get("genre") or "",
                    "g": [groups[k]["books"][ci]["labels"][wi] for k in keys],
                    # ... and its group among every book of the language, which is a different thing.
                    "gl": [groups[k]["language_books"]["labels"][book_index] for k in keys],
                    **_pack([groups[k]["chapters"][(ci, wi)] for k in keys]),
                }),
                "children": chapters_out})
            book_index += 1
        lo, hi = collection_book_span
        tree.append({
            **node(collection_rows, collection, "collection", {
                "g": [groups[k]["collections"]["labels"][ci] for k in keys],
                # How this collection's books fall across the language-wide grouping: the line that
                # says whether a grouping of the language tracks its collections or cuts across them.
                "glsz": [[groups[k]["language_books"]["labels"][lo:hi].count(g)
                          for g in range(1, groups[k]["language_books"]["k"] + 1)] for k in keys],
                **_pack([groups[k]["books"][ci] for k in keys]),
                **_pack_authors([groups[k]["author_books"][ci] for k in keys],
                                [w["author"] for w in works_out]),
            }),
            "children": works_out})

    return {
        "language": language,
        "n_verses": len(verses),
        "keys": keys,          # the order every packed xy, g and gk array follows
        "verse_fields": ["ref", "text", "n_tokens", "xy", "group", "attribution"],
        "strategies": [s for s in describe(language) if s["key"] in keys],
        "unavailable": [s["key"] for s in describe(language) if s["key"] not in keys],
        "tree": tree,
        # The headline: every book of this language, grouped. `collections` is the partition of the
        # handful of collections the language page lists, which is a much weaker thing.
        "book_groups": _pack([groups[k]["language_books"] for k in keys]),
        "collection_groups": _pack([groups[k]["collections"] for k in keys]),
        # Carries the inferred partition as well as the settable ones, the way a collection node does,
        # so the language page opens at the same count the front page reports.
        "author_groups": {**_pack([groups[k]["language_books"] for k in keys]),
                          **_pack_authors([groups[k]["author_language"] for k in keys],
                                          [w["author"] for c in tree for w in c["children"]])},
        "group_reasons": GROUP_REASONS,
        "author_note": "At the book level the count can be overridden: `al` holds a partition at "
                       "every count from one to the number of books. The inferred count is not "
                       "reliable - on the English corpus, where the authors are known, inference "
                       "returned 2 whether the truth was 3, 5 or 13 - so it is offered as a start, "
                       "not an answer. Where authors are known, `af` reports how many of them the "
                       "partition at the true count recovers into a group that is both mostly theirs "
                       "and mostly no one else's; over all 120 English books that is 0 of 13, and "
                       "fitting each collection on its own, 6 of 13.",
        "reliable_tokens": GROUP_RELIABLE_TOKENS,
        "note": "Coordinates are the first two principal components of each strategy's standardised "
                "features, fitted over every verse of this language. They are comparable within a "
                "strategy and not between strategies.",
        "group_note": "Authors are inferred from style. Each level partitions the units it lists - "
                      "a collection its books, a book its chapters, a chapter its verses - on pooled "
                      "profiles, and the inferred count is the largest that survives refitting on 80% "
                      "subsamples at an adjusted Rand index of 0.8 or better. One author means no "
                      "division was supported, not that one hand wrote it. Where the authors are "
                      "actually known - English - the known count is used and the inference is "
                      "scored against it.",
    }
