"""Compare the AI profiling models on the verses they have all profiled.

    stylometry compare-models --set opus=data/processed/profiles.jsonl --set flash=data/processed/profiles_deepseek_flash.jsonl ...
        -> output/models/index.html, model_comparison.md, models.csv, agreement.csv, cluster/<set>/

There is no ground truth for the style of a verse, so a model is judged four ways:

  agreement   correlation of the six numeric scales and agreement of the categorical labels with a
              reference model (the first --set unless --reference is given) and with the consensus of
              all the OTHER models, which does not privilege any one of them;
  stability   agreement between two independent runs of the same model on the same verses.  A set named
              NAME_retest (or NAME-retest) is paired with NAME and excluded from the other tables;
  usefulness  how well the profile ALONE recovers which work a verse belongs to (eta² of each scale
              between works, cross-validated classifier accuracy), and what the full clustering does
              when fed the profiles (k found, agreement with the work boundaries);
  cost        measured $ per verse from the *_runs.jsonl sidecars that `stylometry profile` writes,
              projected onto the corpus.

Every measure is a proxy: a model can agree with the others and still be wrong, and recovering works
also rewards genre sensitivity.  Read them together.
"""
from __future__ import annotations

import csv
import hashlib
import html as h
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ai_profile import (
    CATEGORICAL_DIMS, NUMERIC_DIMS, PROMPT_VERSION, SCHEMA_VERSION, load_profiles, validate_profiles_for_corpus,
)

RETEST_SUFFIXES = ("_retest", "-retest")
DISCOUNT = 0.5  # Anthropic and OpenAI batch APIs, DeepSeek off-peak hours: all halve the list price
NEAR_BEST = 0.9  # a model counts as "as good as the best" when its consensus score is within 10%


@dataclass
class ModelSet:
    name: str
    path: Path
    profiles: dict[str, dict]
    model: str = ""
    backend: str = ""
    cost_per_verse: float | None = None
    runs: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.profiles)

    @property
    def ids(self) -> set[str]:
        return set(self.profiles)


# --- loading -----------------------------------------------------------------------------------------

def parse_set(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--set expects NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def runs_path(profiles_path: Path) -> Path:
    return profiles_path.with_name(profiles_path.stem + "_runs.jsonl")


def load_runs(profiles_path: Path) -> list[dict]:
    p = runs_path(profiles_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_set(name: str, path: Path, cost_override: float | None = None) -> ModelSet:
    profiles = load_profiles(path)
    if not profiles:
        raise SystemExit(f"{path}: no profiles")
    model = Counter(p.get("model", "") for p in profiles.values()).most_common(1)[0][0]
    backend = Counter(p.get("backend", "") for p in profiles.values()).most_common(1)[0][0]
    runs = load_runs(path)
    cost = cost_override
    if cost is None:
        spent = sum(float(r.get("cost_usd", 0.0)) for r in runs)
        done = sum(int(r.get("profiled", 0)) for r in runs)
        cost = spent / done if done else None
    return ModelSet(name, path, profiles, model, backend, cost, runs)


def base_name(name: str) -> str | None:
    for suf in RETEST_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


ENSEMBLE_MARK = "×"


def base_of_ensemble(name: str) -> str | None:
    return name.split(ENSEMBLE_MARK)[0] if ENSEMBLE_MARK in name else None


# --- agreement ---------------------------------------------------------------------------------------

def numeric_matrix(P: dict[str, dict], ids: list[str]) -> np.ndarray:
    return np.array([[float(P[i].get(d, 0.5)) for d in NUMERIC_DIMS] for i in ids], dtype=float)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _finite_mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def corr_by_dim(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    return {d: _corr(A[:, j], B[:, j]) for j, d in enumerate(NUMERIC_DIMS)}


def cat_agreement(P: dict[str, dict], Q: dict[str, dict], ids: list[str]) -> dict[str, float]:
    return {d: float(np.mean([P[i].get(d) == Q[i].get(d) for i in ids])) for d in CATEGORICAL_DIMS}


def tag_jaccard(P: dict[str, dict], Q: dict[str, dict], ids: list[str]) -> float:
    vals = []
    for i in ids:
        a, b = set(P[i].get("style_tags", [])), set(Q[i].get("style_tags", []))
        if a or b:
            vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals)) if vals else float("nan")


def pair_agreement(P: dict[str, dict], Q: dict[str, dict], ids: list[str]) -> dict:
    A, B = numeric_matrix(P, ids), numeric_matrix(Q, ids)
    r = corr_by_dim(A, B)
    cats = cat_agreement(P, Q, ids)
    return {
        "n": len(ids),
        "r_by_dim": r,
        "r_mean": _finite_mean(r.values()),
        "mad_by_dim": {d: float(np.mean(np.abs(A[:, j] - B[:, j]))) for j, d in enumerate(NUMERIC_DIMS)},
        "cat_by_dim": cats,
        "cat_mean": float(np.mean(list(cats.values()))),
        "tag_jaccard": tag_jaccard(P, Q, ids),
    }


def ensemble(name: str, members: list[ModelSet]) -> ModelSet:
    """Mean scales, majority labels, and up to eight most frequent tags across runs."""
    ids = set.intersection(*[m.ids for m in members])
    profiles = {}
    for i in ids:
        rec = {"id": i}
        for d in NUMERIC_DIMS:
            rec[d] = float(np.mean([float(m.profiles[i].get(d, 0.5)) for m in members]))
        for d in CATEGORICAL_DIMS:
            votes = Counter(m.profiles[i].get(d) for m in members)
            top = max(votes.values())
            rec[d] = next(m.profiles[i].get(d) for m in members if votes[m.profiles[i].get(d)] == top)
        tag_counts = Counter(t for m in members for t in set(m.profiles[i].get("style_tags", [])))
        rec["style_tags"] = sorted(tag_counts, key=lambda t: (-tag_counts[t], t))[:8]
        rec["distinctive_phrases"] = []
        rec["signature"] = f"Mean profile across {len(members)} runs; scales averaged and categories voted."
        rec["model"], rec["backend"] = members[0].model, members[0].backend
        memberships = [_request_groups(m.profiles[i]) for m in members]
        if all(memberships):
            requests = sorted(set().union(*memberships))
            source = members[0].profiles[i]["provenance"]
            rec["provenance"] = {
                **source, "request_groups": requests,
                "request_group": hashlib.sha256(json.dumps(requests).encode("utf-8")).hexdigest(),
                "generation_settings": {"aggregation": "mean-majority-top8", "members": [
                    {"model": m.model, "backend": m.backend,
                     "generation_settings": m.profiles[i]["provenance"].get("generation_settings", {})} for m in members
                ]},
            }
        profiles[i] = rec
    cost = sum(m.cost_per_verse for m in members) if all(m.cost_per_verse is not None for m in members) else None
    return ModelSet(name, members[0].path, profiles, f"{members[0].model} (mean of {len(members)} runs)",
                    members[0].backend, cost, [])


def consensus_agreement(sets: dict[str, ModelSet], ids: list[str], voters: list[str] | None = None) -> dict[str, dict]:
    """Each model against the mean scale / majority label of all the other models (ensembles do not vote)."""
    out: dict[str, dict] = {}
    names = list(sets)
    voters = voters or names
    mats = {n: numeric_matrix(sets[n].profiles, ids) for n in names}
    for n in names:
        others = [o for o in voters if o != n and o != base_of_ensemble(n)]
        if not others:
            continue
        mean_others = np.mean([mats[o] for o in others], axis=0)
        r = corr_by_dim(mats[n], mean_others)
        cats = {}
        for d in CATEGORICAL_DIMS:
            hits = []
            for i in ids:
                votes = Counter(sets[o].profiles[i].get(d) for o in others)
                top = votes.most_common(1)[0][0]
                hits.append(sets[n].profiles[i].get(d) == top)
            cats[d] = float(np.mean(hits))
        r_mean = _finite_mean(r.values())
        cat_mean = float(np.mean(list(cats.values())))
        out[n] = {
            "n": len(ids), "n_others": len(others), "r_by_dim": r, "r_mean": r_mean,
            "cat_by_dim": cats, "cat_mean": cat_mean,
            "score": (r_mean + cat_mean) / 2 if math.isfinite(r_mean) else None,
        }
    return out


# --- usefulness --------------------------------------------------------------------------------------

def _tag_docs(P: dict[str, dict], ids: list[str]) -> list[str]:
    return [" ".join(str(t).replace(" ", "_") for t in P[i].get("style_tags", [])) for i in ids]


def _fit_tag_vectorizer(P: dict[str, dict], train_ids: list[str]) -> CountVectorizer | None:
    """Learn the vocabulary on training passages only, including the min_df threshold."""
    vectorizer = CountVectorizer(analyzer=str.split, binary=True, min_df=3)
    try:
        vectorizer.fit(_tag_docs(P, train_ids))
    except ValueError:
        return None
    return vectorizer


def _profile_features(P: dict[str, dict], ids: list[str], tag_vectorizer: CountVectorizer | None) -> np.ndarray:
    num = numeric_matrix(P, ids)
    cat = np.array(
        [[1.0 if P[i].get(d) == v else 0.0 for d, vals in CATEGORICAL_DIMS.items() for v in vals] for i in ids],
        dtype=float,
    )
    tags = (tag_vectorizer.transform(_tag_docs(P, ids)).toarray().astype(float)
            if tag_vectorizer is not None else np.zeros((len(ids), 0)))
    return np.hstack([num, cat, tags])


def _request_groups(profile: dict) -> set[str]:
    provenance = profile.get("provenance") or {}
    groups = provenance.get("request_groups", [])
    groups = set(g for g in groups if isinstance(g, str) and g) if isinstance(groups, list) else set()
    if isinstance(provenance.get("request_group"), str) and provenance["request_group"]:
        groups.add(provenance["request_group"])
    return groups


def evaluation_groups(
    profile_sets: list[dict[str, dict]], ids: list[str], chapter_of: dict[str, object]
) -> tuple[dict[str, str] | None, str | None]:
    """Union chapters and every model request, so neither can cross a holdout boundary.

    Include profiles outside the evaluation cohort: an omitted verse can connect two
    otherwise separate chapters/requests. The same union is used for every model.
    """
    if any(i not in chapter_of or chapter_of[i] is None for i in ids):
        return None, "Chapter metadata is missing; independent passage holdouts cannot be established."
    if any(not _request_groups(P[i]) for P in profile_sets for i in ids):
        return None, "Request provenance is missing; reprofile with the current blinded profiler before evaluating work recovery."
    if any(P[i]["provenance"].get("prompt_version") != PROMPT_VERSION
           or P[i]["provenance"].get("schema_version") != SCHEMA_VERSION for P in profile_sets for i in ids):
        return None, "Work recovery requires profiles from the current blinded prompt and measurement schema."
    universe = set(ids).union(*(set(P) for P in profile_sets))
    parents = {i: i for i in universe}

    def root(i: str) -> str:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: str, b: str) -> None:
        parents[root(b)] = root(a)

    first_chapter: dict[object, str] = {}
    for i in sorted(universe):
        chapter = chapter_of.get(i)
        if chapter is not None:
            union(first_chapter.setdefault(chapter, i), i)
    for P in profile_sets:
        first_request: dict[str, str] = {}
        for i, p in P.items():
            for request in sorted(_request_groups(p)):
                union(first_request.setdefault(request, i), i)
    return {i: root(i) for i in ids}, None


def _grouped_folds(works: list[str], groups: list[str], seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str | None]:
    """Stratify whole, single-work passage groups, balancing verse counts per work."""
    by_group: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        by_group.setdefault(group, []).append(i)
    grouped: dict[str, list[list[int]]] = {}
    for indexes in by_group.values():
        labels = {works[i] for i in indexes}
        if len(labels) != 1:
            return [], "A passage/request group spans multiple works; this grouping is unsupported."
        grouped.setdefault(next(iter(labels)), []).append(indexes)
    n_splits = min(5, min(len(g) for g in grouped.values()))
    if n_splits < 2:
        return [], "Work recovery requires at least two independent chapter/request groups per work."
    rng = np.random.default_rng(seed)
    tests: list[list[int]] = [[] for _ in range(n_splits)]
    for work in sorted(grouped):
        batches = grouped[work]
        rng.shuffle(batches)
        batches.sort(key=len, reverse=True)
        counts = np.zeros(n_splits, dtype=int)
        for indexes in batches:
            fold = int(np.argmin(counts))
            tests[fold].extend(indexes)
            counts[fold] += len(indexes)
    all_indices = np.arange(len(works))
    return [(np.setdiff1d(all_indices, test), np.array(sorted(test))) for test in tests], None


def discrimination(
    P: dict[str, dict], work_of: dict[str, str], ids: list[str], seed: int = 0,
    *, group_of: dict[str, str] | None = None, chapter_of: dict[str, object] | None = None,
    unavailable_reason: str | None = None,
) -> dict:
    works = [work_of[i] for i in ids]
    n_works = len(set(works))
    X = numeric_matrix(P, ids)
    eta2: dict[str, float] = {}
    for j, d in enumerate(NUMERIC_DIMS):
        col = X[:, j]
        grand = col.mean()
        ss_tot = float(((col - grand) ** 2).sum())
        ss_between = sum(len(g := col[[w == ww for w in works]]) * (g.mean() - grand) ** 2 for ww in set(works))
        eta2[d] = float(ss_between / ss_tot) if ss_tot > 0 else float("nan")
    finite_eta = [v for v in eta2.values() if math.isfinite(v)]
    out = {"n": len(ids), "n_works": n_works, "eta2_by_dim": eta2,
           "eta2_mean": float(np.mean(finite_eta)) if finite_eta else float("nan"),
           "work_cv_available": False, "work_cv_method": "chapter-and-request-grouped holdout"}
    if not unavailable_reason and n_works < 2:
        unavailable_reason = "Work recovery requires at least two works."
    if not unavailable_reason and group_of is None:
        group_of, unavailable_reason = evaluation_groups([P], ids, chapter_of or {})
    if not unavailable_reason and any(i not in group_of for i in ids):
        unavailable_reason = "Holdout group membership is missing for one or more verses."
    folds = []
    if not unavailable_reason:
        folds, unavailable_reason = _grouped_folds(works, [group_of[i] for i in ids], seed)
    out["work_cv_reason"] = unavailable_reason
    if unavailable_reason:
        return out
    scores, predictions, baseline, fold_details = [], {}, {}, []
    for train, test in folds:
        train_ids, test_ids = [ids[i] for i in train], [ids[i] for i in test]
        vocabulary = _fit_tag_vectorizer(P, train_ids)
        F_train = _profile_features(P, train_ids, vocabulary)
        F_test = _profile_features(P, test_ids, vocabulary)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        train_works, test_works = [works[i] for i in train], [works[i] for i in test]
        clf.fit(F_train, train_works)
        pred = clf.predict(F_test)
        scores.append(float(accuracy_score(test_works, pred)))
        majority = Counter(train_works).most_common(1)[0][0]
        predictions.update(zip(test, pred))
        baseline.update((i, majority) for i in test)
        fold_details.append({"train_n": len(train), "test_n": len(test), "accuracy": scores[-1],
                             "train_groups": len({group_of[i] for i in train_ids}),
                             "test_groups": len({group_of[i] for i in test_ids})})
    out.update({"work_cv_available": True, "work_cv_acc": float(accuracy_score(works, [predictions[i] for i in range(len(ids))])),
                "work_cv_sd": float(np.std(scores)), "work_cv_n_splits": len(folds),
                "work_cv_groups": len(set(group_of.values())), "work_cv_folds": fold_details,
                "chance": float(accuracy_score(works, [baseline[i] for i in range(len(ids))]))})
    return out


def cluster_replication(
    sets: dict[str, ModelSet], verses: list[dict], out_root: Path, seed: int = 0
) -> dict[str, dict]:
    """Run the real clustering on the pilot verses with each model's profiles (and with none)."""
    from .cluster import run as cluster_run

    ids = [v["id"] for v in verses]
    n_works = len({v["work"] for v in verses})
    results: dict[str, dict] = {}
    assignments: dict[str, list[str]] = {}

    def one(name: str, profiles: dict | None) -> None:
        d = out_root / name
        free = cluster_run(verses, profiles, d / "free", seed=seed)
        fixed, fixed_reason = None, None
        try:
            fixed = cluster_run(verses, profiles, d / f"k{n_works}", k=n_works, seed=seed)
        except ValueError as exc:
            if "distinct observations" not in str(exc):
                raise
            fixed_reason = "Fewer distinct feature vectors than works; the requested fixed-k partition is unavailable."
        if fixed is not None:
            with (d / f"k{n_works}" / "verse_assignments.csv").open(encoding="utf-8") as fh:
                assignments[name] = [r["author"] for r in csv.DictReader(fh)]
        results[name] = {
            "k_free": free["k_used"], "ari_free": free["validation"]["ari_vs_work"],
            "purity_free": free["validation"]["mean_purity"],
            "k_fixed": n_works, "ari_fixed": fixed["validation"]["ari_vs_work"] if fixed else None,
            "purity_fixed": fixed["validation"]["mean_purity"] if fixed else None,
            "fixed_available": fixed is not None, "fixed_reason": fixed_reason,
        }

    one("lexical-only", None)
    for name, s in sets.items():
        if all(i in s.profiles for i in ids):
            one(name, s.profiles)
    names = list(results)
    for a in names:
        results[a]["ari_vs_others"] = {
            b: round(float(adjusted_rand_score(assignments[a], assignments[b])), 3)
            for b in names if b != a and a in assignments and b in assignments
        }
    return results


# --- report ------------------------------------------------------------------------------------------

def _f(x, nd=2, pct=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def _money(x: float | None, nd=2) -> str:
    return "–" if x is None else f"${x:,.{nd}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(rows_) + " |" for rows_ in rows]
    return "\n".join(out)


def _html_table(headers: list[str], rows: list[list[str]], num_from: int = 1) -> str:
    th = "".join(f'<th{" class=num" if j >= num_from else ""}>{h.escape(x)}</th>' for j, x in enumerate(headers))
    trs = "".join(
        "<tr>" + "".join(f'<td{" class=num" if j >= num_from else ""}>{c}</td>' for j, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def pareto(rows: list[dict]) -> set[str]:
    """Models no other model beats on both consensus score and cost."""
    keep = set()
    for a in rows:
        if not _rankable(a):
            continue
        dominated = any(
            b is not a and _rankable(b)
            and b["score"] >= a["score"] and b["cost"] <= a["cost"]
            and (b["score"] > a["score"] or b["cost"] < a["cost"])
            for b in rows
        )
        if not dominated:
            keep.add(a["name"])
    return keep


def _rankable(row: dict) -> bool:
    return all(row.get(key) is not None and math.isfinite(row[key]) for key in ("score", "cost"))


def recommend(rows: list[dict]) -> dict:
    scored = [r for r in rows if _rankable(r)]
    if not scored:
        return {}
    best = max(scored, key=lambda r: r["score"])
    threshold = best["score"] - (1 - NEAR_BEST) * abs(best["score"])
    near = [r for r in scored if r["score"] >= threshold]
    value = min(near, key=lambda r: r["cost"])
    cheapest = min(scored, key=lambda r: r["cost"])
    return {"best": best["name"], "value": value["name"], "cheapest": cheapest["name"], "threshold": threshold}


def _scatter_svg(rows: list[dict], palette: list[str]) -> str:
    pts = [r for r in rows if _rankable(r) and r["cost"] > 0]
    if len(pts) < 2:
        return ""
    W, H, L, R, T, B = 640, 340, 56, 20, 16, 44
    xs = [math.log10(r["cost"]) for r in pts]
    ys = [r["score"] for r in pts]
    x0, x1 = math.floor(min(xs)) - 0.2, math.ceil(max(xs)) + 0.2
    y0, y1 = max(-0.5, min(ys) - 0.1), min(1.0, max(ys) + 0.1)
    sx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    sy = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;font:12px sans-serif" role="img" '
             'aria-label="Consensus score against cost per verse">']
    for e in range(math.floor(x0), math.ceil(x1) + 1):
        if x0 <= e <= x1:
            parts.append(f'<line x1="{sx(e):.1f}" y1="{T}" x2="{sx(e):.1f}" y2="{H - B}" stroke="var(--border)"/>'
                         f'<text x="{sx(e):.1f}" y="{H - B + 16}" text-anchor="middle" fill="var(--text-2)">${10 ** e:g}</text>')
    yt = math.ceil(y0 * 10) / 10
    while yt <= y1 + 1e-9:
        parts.append(f'<line x1="{L}" y1="{sy(yt):.1f}" x2="{W - R}" y2="{sy(yt):.1f}" stroke="var(--border)"/>'
                     f'<text x="{L - 6}" y="{sy(yt) + 4:.1f}" text-anchor="end" fill="var(--text-2)">{yt:.1f}</text>')
        yt = round(yt + 0.1, 2)
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle" fill="var(--text-2)">cost per verse, list price (log scale)</text>')
    parts.append(f'<text transform="translate(12 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" fill="var(--text-2)">consensus score</text>')
    coords = [(sx(math.log10(r["cost"])), sy(r["score"])) for r in pts]
    for j, r in enumerate(pts):
        cx, cy = coords[j]
        col = f"var(--a{j % len(palette) + 1})"
        ring = ' stroke="var(--text)" stroke-width="2"' if r.get("pareto") else ' stroke="var(--surface)" stroke-width="2"'
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{col}"{ring}>'
                     f'<title>{h.escape(r["name"])}: score {r["score"]:.2f}, {_money(r["cost"], 4)}/verse</title></circle>')
        # label to the right unless another point sits there; then below the point
        lx, ly = cx + 10, cy + 4
        if any(k != j and 0 < px - cx < 90 and abs(py - cy) < 14 for k, (px, py) in enumerate(coords)):
            lx, ly = cx - 4, cy + 20
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="var(--text)">{h.escape(r["name"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(
    sets: dict[str, ModelSet],
    verses: list[dict],
    out_dir: Path,
    reference: str | None = None,
    scopes: dict[str, int] | None = None,
    seed: int = 0,
    do_cluster: bool = True,
    ensembles: bool = True,
) -> dict:
    for model_set in sets.values():
        validate_profiles_for_corpus(verses, model_set.profiles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {v["id"]: k for k, v in enumerate(verses)}
    work_of = {v["id"]: v["work"] for v in verses}
    by_id = {v["id"]: v for v in verses}
    chapter_of = {v["id"]: (v.get("language", ""), v["work"], str(v["chapter"]))
                  for v in verses if v.get("chapter") not in (None, "")}

    retests = {n: base_name(n) for n in sets if base_name(n)}
    main = {n: s for n, s in sets.items() if n not in retests}
    reference = reference or next(iter(main))
    if reference not in main:
        raise SystemExit(f"reference {reference!r} is not one of {list(main)}")
    ref = main[reference]
    voters = list(main)
    if ensembles:
        for bn in list(main):
            members = [main[bn]] + [sets[rn] for rn, b in retests.items() if b == bn]
            if len(members) > 1:
                main[f"{bn}{ENSEMBLE_MARK}{len(members)}"] = ensemble(f"{bn}{ENSEMBLE_MARK}{len(members)}", members)

    shared = sorted(set.intersection(*[s.ids for s in main.values()]) & set(order), key=order.get)
    consensus = consensus_agreement(main, shared, voters) if len(voters) >= 3 and shared else {}
    groups, cv_reason = (evaluation_groups([s.profiles for s in main.values()], shared, chapter_of)
                         if shared else (None, "No shared evaluation cohort exists across all compared models."))
    if len({by_id[i].get("language") for i in shared}) > 1:
        if do_cluster:
            raise ValueError("Mixed-language model comparison with clustering is unsupported; use --language or --no-cluster.")
        cv_reason = "Work recovery across mixed languages is unsupported; compare one language at a time."

    per_model: dict[str, dict] = {}
    for n, s in main.items():
        ids = sorted((s.ids & ref.ids) & set(order), key=order.get)
        row: dict = {"name": n, "model": s.model, "backend": s.backend, "n": s.n,
                     "n_works": len({work_of[i] for i in s.ids if i in work_of}),
                     "cost": s.cost_per_verse, "runs": s.runs}
        row["vs_ref"] = pair_agreement(s.profiles, ref.profiles, ids) if ids and n != reference else None
        works_shared = sorted({work_of[i] for i in ids}, key=lambda w: order[next(i for i in ids if work_of[i] == w)])
        row["by_work"] = (
            {w: pair_agreement(s.profiles, ref.profiles, [i for i in ids if work_of[i] == w]) for w in works_shared}
            if len(works_shared) > 1 and n != reference else None
        )
        row["confusions"] = (
            {d: Counter((ref.profiles[i].get(d), s.profiles[i].get(d)) for i in ids if ref.profiles[i].get(d) != s.profiles[i].get(d)).most_common(3)
             for d in CATEGORICAL_DIMS}
            if ids and n != reference else None
        )
        row["consensus"] = consensus.get(n)
        row["score"] = row["consensus"]["score"] if row["consensus"] else None
        own = sorted(s.ids & set(order), key=order.get)
        evaluation_ids = shared or own
        row["discrimination"] = discrimination(
            s.profiles, work_of, evaluation_ids, seed=seed, group_of=groups, unavailable_reason=cv_reason,
        ) if evaluation_ids else None
        if row["discrimination"] is not None:
            row["discrimination"]["cohort"] = "shared across all compared models" if shared else "model-only descriptive statistics"
        per_model[n] = row
    for rn, bn in retests.items():
        if bn in per_model:
            ids = sorted(sets[rn].ids & sets[bn].ids & set(order), key=order.get)
            per_model[bn]["retest"] = pair_agreement(sets[bn].profiles, sets[rn].profiles, ids) if ids else None

    rows = list(per_model.values())
    for r in rows:
        r["pareto"] = False
    for n in pareto(rows):
        per_model[n]["pareto"] = True
    rec = recommend(rows)

    pilot = [by_id[i] for i in shared]
    clusters = cluster_replication(main, pilot, out_dir / "cluster", seed=seed) if do_cluster and len(pilot) >= 40 else {}

    # pairwise matrix on the shared verses
    names = list(main)
    matrix = {a: {b: pair_agreement(main[a].profiles, main[b].profiles, shared)["r_mean"] if shared else float("nan")
                  for b in names} for a in names}

    result = {
        "reference": reference, "shared_ids": len(shared), "pilot_ids": len(pilot),
        "pilot_works": sorted({v["work"] for v in pilot}, key=lambda w: order[next(i for i in ref.ids if work_of.get(i) == w)]),
        "scopes": scopes or {}, "recommendation": rec, "models": per_model, "clusters": clusters, "matrix": matrix,
    }
    _write_outputs(result, out_dir)
    return result


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_outputs(res: dict, out_dir: Path) -> None:
    models = res["models"]
    names = list(models)
    # models.csv
    with (out_dir / "models.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "model", "backend", "verses", "works", "r_vs_reference", "cat_vs_reference", "tags_vs_reference",
                    "r_vs_consensus", "cat_vs_consensus", "consensus_score", "retest_r", "retest_cat", "eta2_mean",
                    "work_cv_acc", "cluster_k_free", "cluster_ari_free", "cluster_ari_fixed", "usd_per_verse", "pareto",
                    "work_cv_n_splits", "work_cv_reason", "evaluation_cohort"]
                   + [f"usd_{k}" for k in res["scopes"]])
        for n in names:
            m = models[n]
            vr, cs, rt, di = m["vs_ref"], m["consensus"], m.get("retest"), m["discrimination"]
            cl = res["clusters"].get(n, {})
            w.writerow([
                n, m["model"], m["backend"], m["n"], m["n_works"],
                _f(vr["r_mean"]) if vr else "", _f(vr["cat_mean"], 3) if vr else "", _f(vr["tag_jaccard"], 3) if vr else "",
                _f(cs["r_mean"]) if cs else "", _f(cs["cat_mean"], 3) if cs else "", _f(cs["score"], 3) if cs else "",
                _f(rt["r_mean"]) if rt else "", _f(rt["cat_mean"], 3) if rt else "",
                _f(di["eta2_mean"], 3) if di else "", _f(di.get("work_cv_acc"), 3) if di and "work_cv_acc" in di else "",
                cl.get("k_free", ""), cl.get("ari_free", ""), cl.get("ari_fixed", ""),
                f"{m['cost']:.5f}" if m["cost"] is not None else "", int(m["pareto"]),
                di.get("work_cv_n_splits", "") if di else "", di.get("work_cv_reason", "") if di else "",
                di.get("cohort", "") if di else "",
            ] + [f"{m['cost'] * n_:.2f}" if m["cost"] is not None else "" for n_ in res["scopes"].values()])
    # agreement.csv (pairwise mean r on the shared verses)
    with (out_dir / "agreement.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set"] + names)
        for a in names:
            w.writerow([a] + [_f(res["matrix"][a][b]) for b in names])
    (out_dir / "comparison.json").write_text(
        json.dumps(_json_finite(res), ensure_ascii=False, indent=1, default=str, allow_nan=False), encoding="utf-8",
    )
    (out_dir / "model_comparison.md").write_text(render_markdown(res), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(res), encoding="utf-8")


def _sections(res: dict, md: bool) -> list[tuple[str, str, list[str], list[list[str]], str]]:
    """(heading, intro, headers, rows, note) for every table, shared by the markdown and HTML renderers."""
    models, names, ref = res["models"], list(res["models"]), res["reference"]
    rec = res["recommendation"]
    star = lambda n: ("**" if md else "") + n + ("**" if md else "")  # noqa: E731
    tables = []

    # 1. headline
    rows = []
    for n in names:
        m = models[n]
        vr, cs, rt, di = m["vs_ref"], m["consensus"], m.get("retest"), m["discrimination"]
        rows.append([
            (star(n) if n == rec.get("value") else n) + (" ★" if m["pareto"] else ""),
            m["model"] + (" (reference)" if n == ref else ""),
            str(vr["n"]) if vr else str(m["n"]),
            _f(vr["r_mean"]) if vr else "–", _f(vr["cat_mean"], pct=True) if vr else "–",
            _f(cs["r_mean"]) if cs else "–", _f(cs["cat_mean"], pct=True) if cs else "–", _f(cs["score"]) if cs else "–",
            _f(rt["r_mean"]) if rt else "–",
            _f(di.get("work_cv_acc"), pct=True) if di and "work_cv_acc" in di else "–",
            _money(m["cost"], 4) if m["cost"] is not None else "–",
        ])
    tables.append((
        "Headline",
        f"Agreement is measured against {ref} on every verse both models profiled, and against the consensus "
        f"of the other models on the {res['shared_ids']} verses every model profiled. Consensus score = mean of the "
        "numeric r and the categorical agreement against the others. Retest r = correlation between two runs of the "
        "same model on the same verses (repeatability, not an absolute agreement ceiling). Work recovery = grouped "
        "cross-validated accuracy (2–5 folds, depending on independent groups per work) of a "
        "classifier that sees only the model's profile and must name the work "
        f"({', '.join(res['pilot_works'])}). ★ = not beaten on both score and cost by any other model. "
        f"A set named NAME{ENSEMBLE_MARK}2 is the mean of two runs of NAME (scales averaged, labels by majority, up to eight "
        "most frequent tags retained with alphabetical ties) at twice the cost; "
        "it does not vote in the consensus. All models use the same evaluation verses and passage folds; whole chapters "
        "and all shared profiling requests stay together, and feature vocabularies are learned inside each training fold. "
        "Consensus measures agreement, not correctness or authorship accuracy.",
        ["set", "model", "verses", f"r vs {ref}", f"cat vs {ref}", "r vs consensus", "cat vs consensus", "consensus score",
         "retest r", "work recovery", "$/verse"],
        rows,
        "",
    ))

    # 2. per-scale correlation vs reference
    rows = [[d] + [_f(models[n]["vs_ref"]["r_by_dim"][d]) if models[n]["vs_ref"] else "–" for n in names if n != ref]
            for d in NUMERIC_DIMS]
    rows += [[d] + [_f(models[n]["vs_ref"]["cat_by_dim"][d], pct=True) if models[n]["vs_ref"] else "–" for n in names if n != ref]
             for d in CATEGORICAL_DIMS]
    rows.append(["style_tags (Jaccard)"] + [_f(models[n]["vs_ref"]["tag_jaccard"]) if models[n]["vs_ref"] else "–" for n in names if n != ref])
    tables.append((
        f"Per dimension against {ref}",
        "Pearson r for the six 0–1 scales, share of identical labels for the five categorical dimensions, mean "
        "Jaccard overlap of the free-text style tags.",
        ["dimension"] + [n for n in names if n != ref],
        rows,
        "Low r on a scale can mean the two models disagree, or that the scale barely varies within one work "
        "(hypotaxis and register are nearly constant inside Mark, for instance).",
    ))

    # 2b. per work
    bw = [n for n in names if models[n].get("by_work")]
    if bw:
        works = list(models[bw[0]]["by_work"])
        rows = [[w] + [x for n in bw for x in (_f(models[n]["by_work"][w]["r_mean"]) if w in models[n]["by_work"] else "–",
                                                  _f(models[n]["by_work"][w]["cat_mean"], pct=True) if w in models[n]["by_work"] else "–")]
                for w in works]
        tables.append((
            f"Per work against {ref}",
            "Mean r of the six scales and mean categorical agreement, one work at a time. Within a single work the scales "
            "vary less, so r is lower than over the whole pilot set; what matters is that the ranking of the models is stable.",
            ["work"] + [x for n in bw for x in (f"{n} r", f"{n} cat")], rows, "",
        ))
    # 2c. label confusions for the recommended model
    val = rec.get("value")
    if val and models[val].get("confusions"):
        rows = []
        for d, top in models[val]["confusions"].items():
            agree = models[val]["vs_ref"]["cat_by_dim"][d]
            rows.append([d, _f(agree, pct=True), "; ".join(f"{a} → {b} ({c})" for (a, b), c in top) or "–"])
        tables.append((
            f"Where {val} and {ref} label differently",
            f"The three most frequent label pairs ({ref} label → {val} label, verses) for each categorical dimension, on the "
            f"{models[val]['vs_ref']['n']} verses both profiled. A few large systematic swaps mean the two models draw the "
            "category boundary in different places, which is a convention difference rather than noise; many small pairs mean scatter.",
            ["dimension", "agreement", "most frequent disagreements"], rows, "",
        ))

    # 3. consensus per scale
    if any(models[n]["consensus"] for n in names):
        rows = [[d] + [_f(models[n]["consensus"]["r_by_dim"][d]) if models[n]["consensus"] else "–" for n in names] for d in NUMERIC_DIMS]
        rows += [[d] + [_f(models[n]["consensus"]["cat_by_dim"][d], pct=True) if models[n]["consensus"] else "–" for n in names]
                 for d in CATEGORICAL_DIMS]
        tables.append((
            "Per dimension against the consensus of the other models",
            f"On the {res['shared_ids']} verses all models profiled. Each model is compared with the mean scale value "
            "and the majority label of every other model, so no single model is treated as the truth.",
            ["dimension"] + names, rows, "",
        ))

    # 4. stability
    rt_names = [n for n in names if models[n].get("retest")]
    if rt_names:
        rows = [[d] + [_f(models[n]["retest"]["r_by_dim"][d]) for n in rt_names] for d in NUMERIC_DIMS]
        rows += [[d] + [_f(models[n]["retest"]["cat_by_dim"][d], pct=True) for n in rt_names] for d in CATEGORICAL_DIMS]
        rows.append(["style_tags (Jaccard)"] + [_f(models[n]["retest"]["tag_jaccard"]) for n in rt_names])
        rows.append(["verses"] + [str(models[n]["retest"]["n"]) for n in rt_names])
        tables.append((
            "Stability: the same model run twice",
            "Two independent runs on the same verses estimate repeatability. This is not an absolute ceiling on "
            "agreement with another model; shared biases can also produce high agreement.",
            ["dimension"] + rt_names, rows, "",
        ))

    # 5. usefulness
    rows = []
    for n in names:
        di = models[n]["discrimination"]
        cl = res["clusters"].get(n, {})
        rows.append([
            n, str(di["n"]) if di else "–", str(di["n_works"]) if di else "–",
            _f(di["eta2_mean"]) if di else "–",
            _f(di.get("work_cv_acc"), pct=True) if di and "work_cv_acc" in di else "–",
            str(cl.get("k_free", "–")), _f(cl.get("ari_free")) if cl else "–", _f(cl.get("ari_fixed")) if cl else "–",
            _f(cl.get("purity_fixed"), pct=True) if cl else "–",
            ((di.get("work_cv_reason") or f"{di['work_cv_n_splits']} grouped folds; {di['work_cv_groups']} groups")
             if di else "No evaluation verses") + (f"; {cl['fixed_reason']}" if cl.get("fixed_reason") else ""),
        ])
    if "lexical-only" in res["clusters"]:
        cl = res["clusters"]["lexical-only"]
        rows.append(["lexical-only (no AI)", "–", "–", "–", "–", str(cl["k_free"]), _f(cl["ari_free"]), _f(cl["ari_fixed"]),
                     _f(cl["purity_fixed"], pct=True), cl.get("fixed_reason") or "–"])
    tables.append((
        "Work discrimination and exploratory clustering",
        "η² = share of each scale's variance that lies between works (mean over the six scales); higher means the "
        "model's scales separate the works. Work recovery as above. The clustering columns run the project's own "
        "clustering (lexical features + this model's profile) on the shared pilot verses: the number of clusters it picks, "
        "the adjusted Rand index of the clusters against the works with k free and with k fixed to the number of "
        "works, and the mean purity at fixed k.",
        ["set", "verses", "works", "η² between works", "work recovery", "k chosen", "ARI (k free)", "ARI (k fixed)", "purity (k fixed)", "evaluation status"],
        rows,
        "η² is descriptive and includes all evaluation verses. Work recovery measures generalization to held-out passages, "
        "not author identification: genre, topic, and recognizable quotations can separate works. Authorship claims require "
        "independent known-author controls across works with genre/topic controlled. Fold standard deviations describe fold "
        "variation, not confidence intervals. Missing request provenance or fewer than two independent groups per work makes "
        "work recovery unavailable; verse-level splitting is never substituted.",
    ))

    # 6. cost
    if res["scopes"]:
        rows = []
        for n in names:
            c = models[n]["cost"]
            if c is None:
                continue
            spent = sum(float(r.get("cost_usd", 0)) for r in models[n]["runs"])
            done = sum(int(r.get("profiled", 0)) for r in models[n]["runs"])
            rows.append([n, f"{done:,}" if done else "–", _money(spent) if done else "–", _money(c, 4)]
                        + [f"{_money(c * k)} / {_money(c * k * DISCOUNT)}" for k in res["scopes"].values()])
        tables.append((
            "Cost",
            "Measured spend at list prices from the runs so far, and the projection for each scope as list / "
            "discounted (Anthropic and OpenAI batch APIs, DeepSeek off-peak hours: each halves the price). "
            "Measured on Greek verses; Hebrew and Arabic units are of similar length.",
            ["set", "verses run", "spent", "$/verse"] + [f"{k} ({v:,} units)" for k, v in res["scopes"].items()],
            rows,
            "",
        ))
    return tables


def _recommendation_text(res: dict) -> str:
    rec, models = res["recommendation"], res["models"]
    if not rec:
        return "Not enough models with both a consensus score and a measured cost to rank."
    best, value, cheapest = rec["best"], rec["value"], rec["cheapest"]
    b, v = models[best], models[value]
    lines = [
        f"Highest consensus agreement: **{best}** ({b['model']}), score {_f(b['score'])} at {_money(b['cost'], 4)} per verse.",
        f"Best value: **{value}** ({v['model']}), score {_f(v['score'])} at {_money(v['cost'], 4)} per verse — "
        f"the cheapest model whose consensus score is within {100 * (1 - NEAR_BEST):.0f}% of the best "
        f"(threshold {_f(rec['threshold'])}).",
    ]
    if value != best:
        ratio = b["cost"] / v["cost"] if v["cost"] else float("inf")
        lines.append(f"{best} costs {ratio:.0f}× more per verse than {value} for a {_f(b['score'] - v['score'])} higher score.")
    if cheapest != value:
        c = models[cheapest]
        lines.append(f"Cheapest overall: {cheapest} ({c['model']}) at {_money(c['cost'], 4)} per verse, score {_f(c['score'])} — "
                     "below the consensus-agreement threshold.")
    if res["scopes"]:
        first = next(iter(res["scopes"].items()))
        lines.append(f"Whole run on {value}: {first[0]} = {_money(v['cost'] * first[1])} list, "
                     f"{_money(v['cost'] * first[1] * DISCOUNT)} discounted; on {best}: {_money(b['cost'] * first[1])} list, "
                     f"{_money(b['cost'] * first[1] * DISCOUNT)} discounted.")
    return "\n\n".join(lines)


def render_markdown(res: dict) -> str:
    parts = ["# Which model should profile the verses?\n",
             f"{len(res['models'])} model configurations compared on the pilot set ({res['pilot_ids']} verses of "
             f"{', '.join(res['pilot_works'])}); {res['shared_ids']} of those verses were profiled by every model. "
             "There is no ground truth for the style of a verse. These are descriptive proxies: agreement with a "
             "reference model, agreement with the consensus of the other models, repeatability across runs, "
             "and how much the profile helps recover the known work boundaries.\n",
             "## Recommendation\n", _recommendation_text(res), ""]
    for heading, intro, headers, rows, note in _sections(res, md=True):
        parts += [f"## {heading}\n", intro + "\n", _md_table(headers, rows), ""]
        if note:
            parts += [note, ""]
    parts += ["## Files\n",
              "`models.csv` one row per model with every number above; `agreement.csv` pairwise mean r between models; "
              "`comparison.json` everything, including per-dimension values; `cluster/<set>/` the clustering outputs "
              "produced with each model's profiles.", ""]
    return "\n".join(parts)


def render_html(res: dict) -> str:
    from .html import LIGHT, page

    rec = res["recommendation"]
    models = res["models"]
    tiles = ""
    if rec:
        v, b = models[rec["value"]], models[rec["best"]]
        tiles = ('<div class="tiles">'
                 f'<div class="tile"><div class="v">{h.escape(rec["value"])}</div><div class="l">best value · {h.escape(v["model"])} · '
                 f'score {_f(v["score"])} · {_money(v["cost"], 4)}/verse</div></div>'
                 f'<div class="tile"><div class="v">{h.escape(rec["best"])}</div><div class="l">highest consensus agreement · {h.escape(b["model"])} · '
                 f'score {_f(b["score"])} · {_money(b["cost"], 4)}/verse</div></div>'
                 f'<div class="tile"><div class="v">{res["pilot_ids"]}</div><div class="l">pilot verses · {res["shared_ids"]} profiled by every model</div></div>'
                 "</div>")
    rows_for_plot = [{"name": n, "score": m["score"], "cost": m["cost"], "pareto": m["pareto"]} for n, m in models.items()]
    body = [tiles, "<h2>Recommendation</h2>",
            "".join(f"<p>{h.escape(p).replace('**', '')}</p>" for p in _recommendation_text(res).split("\n\n")),
            "<h2>Consensus score against cost</h2>",
            "<p class=\"sub\">Up and to the left is better. Outlined points are not beaten on both axes by any other model.</p>",
            _scatter_svg(rows_for_plot, LIGHT)]
    for heading, intro, headers, rows, note in _sections(res, md=False):
        body += [f"<h2>{h.escape(heading)}</h2>", f"<p class=\"sub\">{h.escape(intro)}</p>",
                 _html_table(headers, [[h.escape(c) for c in r] for r in rows],
                             num_from=2 if heading == "Headline" else (99 if heading.startswith("Where ") else 1))]
        if note:
            body.append(f"<p class=\"sub\">{h.escape(note)}</p>")
    body.append('<p class="sub">Files: <a href="model_comparison.md">model_comparison.md</a>, <a href="models.csv">models.csv</a>, '
                '<a href="agreement.csv">agreement.csv</a>, <a href="comparison.json">comparison.json</a>.</p>')
    nav = '<a href="../index.html">All languages</a><a href="model_comparison.md">Report (markdown)</a>'
    return page("Which model should profile the verses?", "\n".join(body), "",
                f"{len(models)} configurations on {res['pilot_ids']} pilot verses; reference model {h.escape(res['reference'])}", nav=nav)
