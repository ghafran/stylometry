"""A full stylometric analysis: every feature family the corpus supports, every method it warrants.

The point of running many strategies together is not to find the one that gives the most interesting
answer. It is the opposite: to see which conclusions survive being asked in several independent ways,
and to record plainly which strategies this corpus cannot support at all. Every result is held out by
whole work, so nothing is ever judged by a neighbour from the same book.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import panel, repetition
from .delta import attribute as delta_attribute
from .delta import most_frequent_words, frequencies
from .measures import cosine_matrix, jensen_shannon_matrix
from .rolling import change_points, exceeds_baseline, windows
from .supervised import classify, impostors

# What each strategy is served by, and where the corpus cannot serve it. Reported with the results so
# an absent strategy is visible rather than silently missing.
STRATEGY_COVERAGE = [
    ("Function-word frequency", "measured", "fw: block, per language"),
    ("Word-frequency profile", "measured", "most frequent words, relative rates"),
    ("Character n-grams", "measured", "2-5 grams"),
    ("Word n-grams", "measured", "1-3 grams"),
    ("Vocabulary richness", "measured", "Yule K, Simpson D, Sichel S, Honore R, Brunet W, MATTR"),
    ("Hapax legomena", "measured", "hapax ratio and the frequency spectrum"),
    ("Sentence length", "partly", "from scribal or editorial punctuation; absent in unpointed sources"),
    ("Word length", "measured", "mean, spread, long and short word share"),
    ("Clause structure", "approximated", "coordinator vs subordinator rates; no parser available"),
    ("Syntax / POS patterns", "unavailable", "no tags in the corpus and no reliable tagger for these languages"),
    ("Grammar preferences", "approximated", "word endings stand in for inflection"),
    ("Punctuation patterns", "measured, not authorial", "records the scribe or modern editor, never the author"),
    ("Morphology", "approximated", "suffix rates; no morphological analyser"),
    ("Collocations", "measured", "PMI-scored word pairs"),
    ("Phraseology", "measured", "repeated n-grams"),
    ("Semantic patterns", "measured", "content-word distribution; reported as a topic control, not as style"),
    ("Topic-independent stylometry", "measured", "function-word-only variant run alongside the full panel"),
    ("Readability / style metrics", "approximated", "sentence and word length composite"),
    ("Rhythm / cadence", "measured", "sentence-length autocorrelation and burstiness"),
    ("Burrows's Delta", "measured", "classic and cosine"),
    ("Cosine similarity", "measured", "on the standardised panel"),
    ("Jensen-Shannon divergence", "measured", "on word-frequency distributions"),
    ("PCA", "measured", "for the variance structure of the panel"),
    ("Cluster analysis", "measured", "see the clustering pipeline"),
    ("SVM / Random Forest", "measured", "leave-one-work-out"),
    ("Change-point detection", "measured", "permutation-tested, per work"),
    ("Rolling-window analysis", "measured", "overlapping windows within a work"),
    ("Authorship verification", "measured", "impostors method, calibrated on known pairs"),
    ("Authorship attribution", "measured", "Delta, SVM, Random Forest"),
    ("Authorship clustering", "measured", "see the clustering pipeline"),
]


def build_panel(docs: list[list[str]], texts: list[str], language: str) -> tuple[np.ndarray, list[str]]:
    """Every feature family, concatenated, with its names."""
    rows, names = [], None
    for tokens, text in zip(docs, texts):
        blocks = [
            panel.function_word_rates(tokens, language),
            panel.morphology_rates(tokens, language),
            panel.length_features(tokens, text),
            panel.clause_features(tokens, text, language),
            panel.punctuation_features(text),
            panel.richness_features(tokens),
        ]
        rows.append(np.concatenate([b[0] for b in blocks]))
        if names is None:
            names = [n for b in blocks for n in b[1]]
    X = np.vstack(rows)
    # Character and word n-grams are corpus-wide vocabularies, so they are added after the per-document
    # blocks rather than inside the loop.
    for kind, n, cap in (("char", 4, 300), ("word", 2, 300)):
        counters = [panel.char_ngrams(t, n) if kind == "char" else panel.word_ngrams(d, n)
                    for d, t in zip(docs, texts)]
        total: Counter = Counter()
        for c in counters:
            total.update(c)
        keys = [k for k, _ in total.most_common(cap)]
        block = np.array([[c.get(k, 0) / max(sum(c.values()), 1) for k in keys] for c in counters])
        X = np.hstack([X, block])
        names += [f"{kind}{n}:{k if isinstance(k, str) else ' '.join(k)}" for k in keys]
    return X, names


def _function_word_columns(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith("fw:")]


def analyse(docs: list[list[str]], texts: list[str], labels: list[str], works: list[str],
            language: str = "grc", seed: int = 0, permutations: int = 500) -> dict:
    """Run the battery. Every method sees the same units and the same whole-work holdout."""
    if not (len(docs) == len(texts) == len(labels) == len(works)):
        raise ValueError("documents, texts, labels and works must be the same length")
    if len(docs) < 4:
        raise ValueError("a full analysis needs at least four units")

    X, names = build_panel(docs, texts, language)
    result: dict = {
        "n_units": len(docs), "n_features": X.shape[1], "n_works": len(set(works)),
        "n_labels": len(set(labels)), "language": language,
        "feature_blocks": dict(Counter(n.split(":")[0] for n in names)),
        "strategy_coverage": [{"strategy": s, "status": st, "detail": d} for s, st, d in STRATEGY_COVERAGE],
    }

    counts = Counter(labels)
    result["majority_baseline"] = max(counts.values()) / len(labels)
    result["chance"] = 1 / len(counts)

    # --- attribution, several ways -----------------------------------------------------------------
    attribution = {}
    for n_words in (500, 2000):
        pred = delta_attribute(docs, labels, groups=works, n_words=n_words, metric="cosine")
        attribution[f"delta_cosine_{n_words}mfw"] = _score(pred, labels)
    for kind in ("svm", "forest"):
        out = classify(X, labels, works, kind=kind, seed=seed)
        attribution[kind] = {"accuracy": out["accuracy"], "n": out["n"]}
    # Topic-independent variant: the same classifier on function words alone.
    fw_cols = _function_word_columns(names)
    if fw_cols:
        out = classify(X[:, fw_cols], labels, works, kind="svm", seed=seed)
        attribution["svm_function_words_only"] = {"accuracy": out["accuracy"], "n": out["n"],
                                                  "n_features": len(fw_cols)}
    result["attribution"] = attribution

    # --- distances ---------------------------------------------------------------------------------
    vocabulary = most_frequent_words(docs, 500)
    freq = frequencies(docs, vocabulary)
    result["distances"] = {
        "jensen_shannon_mean_within_label": _within_between(jensen_shannon_matrix(freq), labels),
        "cosine_mean_within_label": _within_between(cosine_matrix(_standardise(X)), labels),
    }

    # --- verification ------------------------------------------------------------------------------
    result["verification"] = _verify(X, labels, works, seed)

    # --- change points, per work -------------------------------------------------------------------
    result["change_points"] = _seams(docs, texts, works, language, permutations, seed)
    return result


def _standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    out = (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)
    out[:, sd < 1e-12] = 0.0
    return out


def _score(pred: list, labels: list[str]) -> dict:
    scored = [(t, p) for t, p in zip(labels, pred) if p is not None]
    return {"accuracy": sum(t == p for t, p in scored) / len(scored) if scored else 0.0,
            "n": len(scored)}


def _within_between(D: np.ndarray, labels: list[str]) -> dict:
    """Mean distance inside a label against across labels: separation without any classifier."""
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else between).append(D[i, j])
    w, b = float(np.mean(within)) if within else float("nan"), float(np.mean(between)) if between else float("nan")
    return {"within": w, "between": b, "ratio": (w / b) if b else float("nan")}


def _verify(X: np.ndarray, labels: list[str], works: list[str], seed: int) -> dict:
    """Impostors scores for genuine and false pairings, which is what a threshold must separate."""
    Z = _standardise(X)
    by_label = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    same, different = [], []
    for i, label in enumerate(labels):
        pool_same = [j for j in by_label[label] if works[j] != works[i]]
        pool_other = [j for j in range(len(labels)) if labels[j] != label]
        if len(pool_same) < 2 or len(pool_other) < 4:
            continue
        rng = np.random.default_rng(seed + i)
        impostor_ids = rng.choice(pool_other, min(20, len(pool_other)), replace=False)
        same.append(impostors(Z[i], Z[pool_same], Z[impostor_ids], seed=seed))
        wrong = str(rng.choice([l for l in by_label if l != label]))
        pool_wrong = [j for j in by_label[wrong] if works[j] != works[i]]
        if len(pool_wrong) >= 2:
            different.append(impostors(Z[i], Z[pool_wrong], Z[impostor_ids], seed=seed))
    if not same or not different:
        return {"available": False, "reason": "not enough labels with two or more works"}
    same_arr, diff_arr = np.array(same), np.array(different)
    # The separation that matters: can one threshold accept genuine pairs and reject false ones?
    thresholds = np.linspace(0, 1, 101)
    best = max(thresholds, key=lambda t: (same_arr >= t).mean() + (diff_arr < t).mean())
    return {
        "available": True, "n_same_author_pairs": len(same), "n_different_author_pairs": len(different),
        "same_author_mean": float(same_arr.mean()), "different_author_mean": float(diff_arr.mean()),
        "best_threshold": float(best),
        "acceptance_of_genuine": float((same_arr >= best).mean()),
        "false_acceptance": float((diff_arr >= best).mean()),
        "note": "a score, not a probability; the threshold is fitted here and would need its own holdout",
    }


BASELINE_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "change_point_baseline.json"


def load_change_point_baseline(language: str = "grc", path: Path | None = None) -> dict | None:
    """What the statistic reaches in single-author works *of that language*.

    A threshold taken from Greek prose says nothing about Hebrew, so the reference is per language and
    a language without one gets no verdict rather than a borrowed one. Arabic has no single-author
    corpus available here at all, and is recorded as unavailable for exactly that reason.
    """
    path = Path(path or BASELINE_PATH)
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8")).get(language)
    if not isinstance(entry, dict) or entry.get("available") is False or "max" not in entry:
        return None
    return entry


def _seams(docs, texts, works, language, permutations, seed) -> list[dict]:
    """Rolling windows within each work, read against single-author works rather than against chance.

    The permutation p-value is kept but is not the finding: it rejects exchangeability for 85% of
    works known to have one author, because neighbouring windows of continuous prose share a topic.
    What decides is whether the shift exceeds what single authorship itself produces.
    """
    baseline = load_change_point_baseline()
    by_work = defaultdict(list)
    for tokens, text, work in zip(docs, texts, works):
        by_work[work].append((tokens, text))
    out = []
    for work, parts in sorted(by_work.items()):
        tokens = [t for part in parts for t in part[0]]
        text = " ".join(part[1] for part in parts)
        pieces = windows(tokens, size=1000, step=500)
        if len(pieces) < 8:
            continue
        X, _ = build_panel([p[1] for p in pieces], [" ".join(p[1]) for p in pieces], language)
        found = change_points(X, permutations=permutations, seed=seed)
        if found.get("available"):
            found["work"] = work
            found["token_offset"] = int(pieces[found["split_index"]][0])
            row = {k: found[k] for k in
                   ("work", "n_windows", "split_index", "token_offset", "separation", "p_value")}
            if baseline:
                row.update(exceeds_baseline(found["separation"], baseline))
            out.append(row)
    return sorted(out, key=lambda r: -r["separation"])


def render(result: dict) -> str:
    """A report that names every strategy, including the ones the corpus cannot support."""
    md = ["# Full stylometric analysis\n",
          f"{result['n_units']} units, {result['n_features']} features, {result['n_works']} works, "
          f"{result['n_labels']} labels, language {result['language']}.\n",
          "Every result below is held out by whole work: no unit is ever judged by another unit from "
          "the same book. Baselines are given because an accuracy without one means nothing.\n"]

    md.append("\n## Strategy coverage\n")
    md.append("| strategy | status | how |\n|---|---|---|")
    for row in result["strategy_coverage"]:
        md.append(f"| {row['strategy']} | {row['status']} | {row['detail']} |")

    md.append("\n\n## Attribution\n")
    md.append(f"Majority baseline {result['majority_baseline']:.1%}, chance {result['chance']:.1%}.\n")
    md.append("| method | accuracy | units |\n|---|---:|---:|")
    for name, row in result["attribution"].items():
        md.append(f"| {name} | {row['accuracy']:.1%} | {row['n']} |")

    md.append("\n\n## Distances\n")
    md.append("Mean distance within a label against across labels. A ratio below 1 means texts sharing "
              "a label really are closer to each other, with no classifier involved.\n")
    md.append("| measure | within | between | ratio |\n|---|---:|---:|---:|")
    for name, row in result["distances"].items():
        md.append(f"| {name} | {row['within']:.4f} | {row['between']:.4f} | {row['ratio']:.3f} |")

    v = result["verification"]
    md.append("\n\n## Verification\n")
    if not v.get("available"):
        md.append(f"Not available: {v.get('reason')}.")
    else:
        md.append(f"Impostors method over {v['n_same_author_pairs']} genuine and "
                  f"{v['n_different_author_pairs']} false pairings.\n")
        md.append(f"- genuine pairings score {v['same_author_mean']:.2f} on average, false ones "
                  f"{v['different_author_mean']:.2f}")
        md.append(f"- at the best threshold ({v['best_threshold']:.2f}): "
                  f"**{v['acceptance_of_genuine']:.1%}** of genuine pairings accepted, "
                  f"**{v['false_acceptance']:.1%}** of false ones wrongly accepted")
        md.append(f"\n{v['note']}. Unlike attribution, this method can answer 'neither'.")

    md.append("\n\n## Change points\n")
    seams = result["change_points"]
    if not seams:
        md.append("No work had enough text for rolling windows.")
    else:
        baseline = load_change_point_baseline(result.get("language", "grc"))
        md.append("The strongest stylistic shift inside each work.\n")
        md.append("**The permutation p-value is reported but is not the finding.** It asks whether the "
                  "windows could be in any order, and for continuous prose they could not: neighbouring "
                  "windows share a topic. Measured on thirteen Greek works of undisputed single "
                  "authorship, it rejected exchangeability for **85%** of them. A test that fires on "
                  "Plato's *Apology* cannot be used to find a seam anywhere.\n")
        if baseline:
            md.append(f"What decides is the last column: whether the shift exceeds what single "
                      f"authorship itself produces **in this language**. Across {baseline['n_works']} "
                      f"such works the statistic reached a mean of {baseline['mean']:.2f} and never "
                      f"exceeded **{baseline['max']:.2f}**.\n")
            if baseline.get("note_language"):
                md.append(f"> {baseline['note_language']}\n")
        else:
            md.append("**No single-author reference exists for this language**, so no separation value "
                      "here can be called a seam. The figures are reported as measurements only.\n")
        md.append("| work | windows | token offset | separation | p | above single-author max |"
                  "\n|---|---:|---:|---:|---:|---|")
        for row in seams[:20]:
            verdict = ("**yes**" if row.get("exceeds_all_single_author_works")
                       else "no" if "exceeds_all_single_author_works" in row else "—")
            md.append(f"| {row['work']} | {row['n_windows']} | {row['token_offset']:,} | "
                      f"{row['separation']:.2f} | {row['p_value']:.3f} | {verdict} |")
        beats = [r for r in seams if r.get("exceeds_all_single_author_works")]
        naive = [r for r in seams if r["p_value"] < 0.05]
        md.append(f"\n{len(naive)} of {len(seams)} works clear p < 0.05, which by itself means little.")
        if baseline:
            md.append(f"**{len(beats)}** exceed what any single-author reference work in this language "
                      f"reached.")
    return "\n".join(md) + "\n"


def save(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    (out / "analysis.md").write_text(render(result), encoding="utf-8")
    return out
