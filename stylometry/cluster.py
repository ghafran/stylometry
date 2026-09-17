"""Author discovery: feature matrix -> local smoothing -> choose k -> cluster -> markers per author.

Why smoothing?  A single verse is 10-40 words - far too little text for a stable stylometric
fingerprint.  Instead of clustering isolated verses we cluster each verse's *local style profile*: a
weighted mix of its own feature vector and the mean of its neighbours (same work, +/- ``window``
verses).  Every verse still receives its own label, and blocks of verses whose style breaks from the
surrounding text can still be detected, but noise from individual short verses is damped.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .ai_profile import CATEGORICAL_DIMS, NUMERIC_DIMS, PROMPT_VERSION, validate_profiles_for_corpus
from .features import lexical_features
from .continuity import annotate_continuity, consecutive, contiguous_runs


def _standardize(X: np.ndarray) -> np.ndarray:
    if not np.isfinite(X).all():
        raise ValueError("feature values must be finite")
    return StandardScaler().fit_transform(X).astype(np.float32)


def build_blocks(verses: list[dict], profiles: dict[str, dict] | None, seed: int = 0) -> list[tuple[str, np.ndarray, list[str]]]:
    blocks: list[tuple[str, np.ndarray, list[str]]] = []
    X_lex, names_lex = lexical_features(verses, seed=seed)
    blocks.append(("lex", _standardize(X_lex), names_lex))
    if profiles:
        num = np.array([[float(profiles[v["id"]].get(d, 0.5)) for d in NUMERIC_DIMS] for v in verses], np.float32)
        names = [f"ai:{d}" for d in NUMERIC_DIMS]
        cols = []
        for dim, values in CATEGORICAL_DIMS.items():
            for val in values:
                cols.append([1.0 if profiles[v["id"]].get(dim) == val else 0.0 for v in verses])
                names.append(f"cat:{dim}={val}")
        cat = np.array(cols, np.float32).T
        blocks.append(("ai", _standardize(np.hstack([num, cat])), names))
        docs = [" ".join(str(t).replace(" ", "_") for t in profiles[v["id"]].get("style_tags", [])) for v in verses]
        cv = CountVectorizer(analyzer=str.split, binary=True, min_df=3)
        try:
            T = cv.fit_transform(docs).toarray().astype(np.float32)
            blocks.append(("tags", _standardize(T), [f"tag:{t}" for t in cv.get_feature_names_out()]))
        except ValueError:
            pass
    return blocks


def combine(blocks: list[tuple[str, np.ndarray, list[str]]], weights: dict[str, float], seed: int = 0) -> np.ndarray:
    """Equalise each block's total variance, then apply the per-block weight."""
    parts = []
    for name, Z, _ in blocks:
        if name == "tags" and Z.shape[1] > 30:
            Z = TruncatedSVD(n_components=30, random_state=seed).fit_transform(Z).astype(np.float32)
        weight = weights.get(name, 1.0)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("feature weights must be finite and nonnegative")
        variance = float(np.var(Z, axis=0).sum())
        parts.append(Z / math.sqrt(variance) * weight if variance > 1e-12 else np.zeros_like(Z))
    return np.hstack(parts).astype(np.float32)


def smooth(X: np.ndarray, verses: list[dict], window: int = 5, alpha: float = 0.7) -> np.ndarray:
    if window <= 0 or alpha <= 0:
        return X
    Xs = X.copy()
    for idx in contiguous_runs(verses):
        idx_arr = np.asarray(idx)
        A = X[idx_arr]
        cs = np.vstack([np.zeros((1, A.shape[1]), A.dtype), np.cumsum(A, axis=0)])
        m = len(idx_arr)
        local = np.empty_like(A)
        for j in range(m):
            lo, hi = max(0, j - window), min(m, j + window + 1)
            local[j] = (cs[hi] - cs[lo]) / (hi - lo)
        Xs[idx_arr] = alpha * local + (1 - alpha) * A
    return Xs


CRITERIA = {
    "silhouette": ("silhouette", max),
    "calinski": ("calinski_harabasz", max),
    "davies_bouldin": ("davies_bouldin", min),
    "bic": ("gmm_bic", min),
}


def select_k(
    X: np.ndarray,
    kmin: int,
    kmax: int,
    seed: int = 0,
    sample: int = 4000,
    criterion: str = "silhouette",
    X_eval: np.ndarray | None = None,
    min_bic_gain: float = 10.0,
) -> tuple[int, list[dict]]:
    """Score partitions on unsmoothed observations, always allowing no supported split.

    A candidate must improve BIC over one Gaussian by ``min_bic_gain`` and have
    positive raw silhouette. Mixtures are fitted on one fixed sample of raw
    vectors, rotated into its full PCA basis (no dimensions discarded). This
    makes the diagonal single-Gaussian baseline valid for correlated clouds.
    This is a parametric structure diagnostic, not an authorship significance test.
    """
    X_eval = X if X_eval is None else X_eval
    if criterion not in CRITERIA or kmin < 1 or kmax < kmin or sample < 3:
        raise ValueError("invalid k-selection settings")
    if X.ndim != 2 or X.shape != X_eval.shape or len(X) < 2 or X.shape[1] == 0:
        raise ValueError("k selection needs matching, nonempty feature matrices")
    if not np.isfinite(X).all() or not np.isfinite(X_eval).all():
        raise ValueError("k selection needs finite features")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), sample, replace=False) if len(X) > sample else np.arange(len(X))
    raw = X_eval[idx].astype(np.float64)
    centered = raw - raw.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    varying = singular > max(float(singular[0]), 1.0) * 1e-10
    raw = centered @ axes[varying].T if varying.any() else np.zeros((len(raw), 1))
    baseline = GaussianMixture(n_components=1, covariance_type="diag", random_state=seed).fit(raw)
    base_bic = float(baseline.bic(raw))
    rows = [{"k": 1, "silhouette": None, "calinski_harabasz": None,
             "davies_bouldin": None, "gmm_bic": base_bic, "bic_gain": 0.0,
             "inertia": float(((X - X.mean(axis=0)) ** 2).sum()),
             "split_supported": False, "evaluation_n": len(idx)}]
    distinct = len(np.unique(X, axis=0))
    for k in range(max(2, kmin), min(kmax, len(idx) - 1, distinct) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        lab = km.labels_
        if not 2 <= len(set(lab[idx])) < len(idx):
            continue
        gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=2,
                              random_state=seed).fit(raw)
        bic = float(gmm.bic(raw))
        silhouette = float(silhouette_score(X_eval[idx], lab[idx]))
        gain = base_bic - bic
        rows.append({
            "k": k, "silhouette": silhouette,
            "calinski_harabasz": float(calinski_harabasz_score(X_eval[idx], lab[idx])),
            "davies_bouldin": float(davies_bouldin_score(X_eval[idx], lab[idx])),
            "gmm_bic": bic, "bic_gain": gain, "inertia": float(km.inertia_),
            "split_supported": bool(gmm.converged_ and gain >= min_bic_gain and silhouette > 0),
            "evaluation_n": len(idx),
        })
    column, pick = CRITERIA[criterion]
    supported = [r for r in rows if r["split_supported"]]
    return (pick(supported, key=lambda r: r[column])["k"] if supported else 1), rows


def partition_stability(X: np.ndarray, labels: np.ndarray, verses: list[dict], k: int,
                        window: int = 5, seed: int = 0, repeats: int = 5) -> dict:
    """Perturb fitted partitions by omitting whole passages, with a fixed feature map.

    This checks partition sensitivity only; it is not held-out predictive accuracy
    or a bootstrap confidence interval for the entire feature-extraction pipeline.
    """
    if repeats < 1:
        raise ValueError("stability repeats must be positive")
    if k == 1:
        return {"available": False, "reason": "single_group", "repeats": 0}
    width = max(20, 2 * window + 1)
    blocks = [run[j:j + width] for run in contiguous_runs(verses) for j in range(0, len(run), width)]
    if len(blocks) < 5:
        return {"available": False, "reason": "fewer_than_five_passages", "repeats": 0}
    rng = np.random.default_rng(seed)
    aris = []
    for rep in range(repeats):
        selected = rng.choice(len(blocks), max(2, int(len(blocks) * .8)), replace=False)
        train = np.array([i for b in selected for i in blocks[b]])
        if len(np.unique(X[train], axis=0)) < k:
            aris.append(0.0)
            continue
        fitted = KMeans(n_clusters=k, n_init=10, random_state=seed + rep + 1).fit(X[train])
        aris.append(float(adjusted_rand_score(labels, fitted.predict(X))))
    return {"available": True, "repeats": repeats, "mean_ari": float(np.mean(aris)),
            "min_ari": min(aris), "ari": aris, "passages": len(blocks),
            "method": "80% passage subsamples; fixed feature map"}


def cluster(X: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k < 1 or k > len(np.unique(X, axis=0)):
        raise ValueError("k must be between 1 and the number of distinct observations")
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    if k == 1:
        return np.full(len(X), "A1"), np.zeros(len(X)), km.cluster_centers_
    d = km.transform(X)
    order = np.argsort(d, axis=1)
    rows = np.arange(len(X))
    d1, d2 = d[rows, order[:, 0]], d[rows, order[:, 1]]
    conf = (d2 - d1) / np.maximum(d2, 1e-9)  # margin between best and second-best centroid, 0..1
    ranked = [c for c, _ in Counter(km.labels_.tolist()).most_common()]
    remap = {c: i + 1 for i, c in enumerate(ranked)}
    authors = np.array([f"A{remap[c]}" for c in km.labels_])
    centers = np.vstack([km.cluster_centers_[c] for c in ranked])
    return authors, conf, centers


def hdbscan_summary(X: np.ndarray, min_cluster_size: int, seed: int = 0) -> dict:
    if len(X) < max(min_cluster_size, 11):
        return {"available": False, "reason": "too_few_observations"}
    try:
        import hdbscan  # type: ignore
    except Exception as e:  # optional dependency
        return {"available": False, "error": str(e)}
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10).fit_predict(X)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    return {
        "available": True,
        "min_cluster_size": min_cluster_size,
        "n_clusters": n_clusters,
        "noise_fraction": round(float(np.mean(labels == -1)), 3),
    }


def author_key(a: str) -> int:
    return int(a[1:])


def describe_authors(
    verses: list[dict],
    authors: np.ndarray,
    conf: np.ndarray,
    Zraw: np.ndarray,
    names: list[str],
    Xred: np.ndarray,
    profiles: dict[str, dict] | None,
) -> dict[str, dict]:
    total = len(verses)
    out: dict[str, dict] = {}
    if profiles:
        tags_all = Counter(t for v in verses for t in profiles[v["id"]].get("style_tags", []))
    for a in sorted(set(authors.tolist()), key=author_key):
        mask = authors == a
        n = int(mask.sum())
        eff = Zraw[mask].mean(axis=0)  # Zraw is globally standardised, so this is an effect size in SD units
        order = np.argsort(eff)
        markers_high = [(names[i], round(float(eff[i]), 2)) for i in order[::-1][:15]]
        markers_low = [(names[i], round(float(eff[i]), 2)) for i in order[:10]]
        members = [v for v, f in zip(verses, mask) if f]
        works = Counter(v["work"] for v in members)
        copyists = Counter(v["copyist"] for v in members if v["copyist"])
        centroid = Xred[mask].mean(axis=0)
        dist = np.linalg.norm(Xred[mask] - centroid, axis=1)
        rep_idx = np.where(mask)[0][np.argsort(dist)[:5]]
        entry: dict = {
            "author": a,
            "n_verses": n,
            "share": round(n / total, 4),
            "mean_confidence": round(float(conf[mask].mean()), 3),
            "works": {w: c for w, c in works.most_common(15)},
            "copyists": dict(copyists),
            "markers_high": markers_high,
            "markers_low": markers_low,
            "representative": [
                {"id": verses[i]["id"], "ref": verses[i]["ref"], "text": verses[i]["text"][:200]} for i in rep_idx
            ],
        }
        if profiles:
            entry["ai_profile_means"] = {
                d: round(float(np.mean([profiles[v["id"]].get(d, 0.5) for v in members])), 3) for d in NUMERIC_DIMS
            }
            tags_a = Counter(t for v in members for t in profiles[v["id"]].get("style_tags", []))
            lift = {
                t: (tags_a[t] / n) / (tags_all[t] / total) for t in tags_a if tags_all[t] >= 5 and tags_a[t] >= 3
            }
            entry["tag_lift"] = [(t, round(l, 2), tags_a[t]) for t, l in sorted(lift.items(), key=lambda kv: -kv[1])[:15]]
            entry["phrases"] = Counter(
                p for v in members for p in profiles[v["id"]].get("distinctive_phrases", [])
            ).most_common(15)
            entry["discourse_modes"] = dict(Counter(profiles[v["id"]].get("discourse_mode") for v in members).most_common())
        out[a] = entry
    return out


def segments(verses: list[dict], authors: np.ndarray) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    majority: dict[str, str] = {}
    by_work: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(verses):
        by_work[v["work"]].append(i)
    for work, idx in by_work.items():
        majority[work] = Counter(authors[i] for i in idx).most_common(1)[0][0]
    for idx in contiguous_runs(verses):
        work = verses[idx[0]]["work"]
        maj = majority[work]
        run_start = 0
        for p in range(1, len(idx) + 1):
            if p == len(idx) or authors[idx[p]] != authors[idx[run_start]]:
                s, e = idx[run_start], idx[p - 1]
                rows.append(
                    {
                        "work": work,
                        "author": authors[s],
                        "start": verses[s]["ref"],
                        "end": verses[e]["ref"],
                        "start_id": verses[s]["id"],
                        "end_id": verses[e]["id"],
                        "n": p - run_start,
                        "is_majority": authors[s] == maj,
                    }
                )
                run_start = p
    return rows, majority


def validation(verses: list[dict], authors: np.ndarray, majority: dict[str, str]) -> dict:
    works = [v["work"] for v in verses]
    groups = [v["group"] for v in verses]
    purity = {}
    for w in set(works):
        m = [a for a, ww in zip(authors, works) if ww == w]
        purity[w] = round(sum(1 for a in m if a == majority[w]) / len(m), 3)
    scribe = defaultdict(Counter)
    for v, a in zip(verses, authors):
        if v["copyist"]:
            scribe[a][v["copyist"]] += 1
    return {
        "ari_vs_work": round(float(adjusted_rand_score(works, authors)), 4),
        "ari_vs_group": round(float(adjusted_rand_score(groups, authors)), 4),
        "purity_by_work": purity,
        "mean_purity": round(float(np.mean(list(purity.values()))), 3),
        "author_by_scribe": {a: dict(c) for a, c in scribe.items()},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(
    verses: list[dict],
    profiles: dict[str, dict] | None,
    out_dir: str | Path,
    k: int | None = None,
    kmin: int = 2,
    kmax: int = 20,
    window: int = 5,
    alpha: float = 0.7,
    pca_dims: int = 30,
    weights: dict[str, float] | None = None,
    seed: int = 0,
    criterion: str = "silhouette",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = weights if weights is not None else {"lex": 1.0, "ai": 1.0, "tags": 0.7}
    if window < 0 or not 0 <= alpha <= 1 or pca_dims < 1 or kmin < 1 or kmax < kmin:
        raise ValueError("invalid clustering settings")
    if len({v["id"] for v in verses}) != len(verses):
        raise ValueError("duplicate verse IDs in clustering input")
    verses = annotate_continuity(verses)
    n_input = len(verses)
    if profiles is not None:
        if not profiles:
            raise ValueError("no AI profiles; use no-AI mode explicitly")
        validate_profiles_for_corpus(verses, profiles)
        missing = [v["id"] for v in verses if v["id"] not in profiles]
        if missing:
            print(f"{len(missing)} verses have no AI profile; restricting clustering to the {len(verses) - len(missing)} profiled verses")
            verses = [v for v in verses if v["id"] in profiles]
    if len(verses) < 3:
        raise ValueError("too few verses to cluster (need at least three)")

    blocks = build_blocks(verses, profiles, seed=seed)
    Zraw = np.hstack([Z for _, Z, _ in blocks])
    names = [n for _, _, ns in blocks for n in ns]
    X = combine(blocks, weights, seed=seed)
    Xs = smooth(X, verses, window=window, alpha=alpha)
    if not any(weights.get(name, 1.0) > 0 for name, _, _ in blocks):
        raise ValueError("at least one available feature block must have positive weight")
    if np.var(X, axis=0).sum() <= 1e-12:
        Xred = Xown = np.zeros((len(X), 1), np.float32)
        explained = 0.0
    else:
        pca = PCA(n_components=min(pca_dims, X.shape[1], len(verses) - 1), random_state=seed).fit(X)
        Xred = pca.transform(Xs).astype(np.float32)
        Xown = pca.transform(X).astype(np.float32)
        explained = float(pca.explained_variance_ratio_.sum())

    chosen, ktable = select_k(Xred, kmin, kmax, seed=seed, criterion=criterion, X_eval=Xown)
    forced = k is not None
    k = chosen if k is None else k
    authors, conf, centers = cluster(Xred, k, seed=seed)
    stability = {"tested_k": k, **partition_stability(Xred, authors, verses, k, window=window, seed=seed)}
    bic_candidate = chosen
    if not forced and k > 1 and (not stability["available"] or stability["min_ari"] < 0.8):
        chosen = k = 1
        authors, conf, centers = cluster(Xred, k, seed=seed)
        selection_status = "split_not_stable" if stability["available"] else "insufficient_passages"
    else:
        selection_status = "forced" if forced else ("supported_style_partition" if k > 1 else "no_supported_split")

    # Outliers: verses whose OWN (unsmoothed) style is far from their author's centroid.
    dist_own = np.linalg.norm(Xown - centers[[author_key(a) - 1 for a in authors]], axis=1)
    z = np.zeros_like(dist_own)
    for a in set(authors.tolist()):
        m = authors == a
        z[m] = (dist_own[m] - dist_own[m].mean()) / (dist_own[m].std() + 1e-9)
    outlier_rows = [
        {"id": v["id"], "ref": v["ref"], "work": v["work"], "author": a, "z": round(float(zz), 2), "text": v["text"][:160]}
        for v, a, zz in zip(verses, authors, z)
        if zz > 2.5
    ]
    outlier_rows.sort(key=lambda r: -r["z"])

    desc = describe_authors(verses, authors, conf, Zraw, names, Xred, profiles)
    seg_rows, majority = segments(verses, authors)
    val = validation(verses, authors, majority)
    hdb = hdbscan_summary(Xred, min_cluster_size=max(30, len(verses) // 200), seed=seed)

    assignments = [
        {
            "id": v["id"], "work": v["work"], "chapter": v["chapter"], "verse": v["verse"], "ref": v["ref"],
            "author": a, "confidence": round(float(c), 3), "outlier_z": round(float(zz), 2),
            "copyist": v["copyist"] or "", "group": v["group"], "n_tokens": v["n_tokens"], "text": v["text"],
        }
        for v, a, c, zz in zip(verses, authors, conf, z)
    ]
    _write_csv(out_dir / "verse_assignments.csv", assignments)
    _write_csv(out_dir / "k_selection.csv", ktable)
    _write_csv(out_dir / "segments.csv", seg_rows)
    _write_csv(out_dir / "outliers.csv", outlier_rows)

    author_ids = sorted(set(authors.tolist()), key=author_key)
    crosstab = []
    for w in dict.fromkeys(v["work"] for v in verses):
        row = {"work": w, "group": next(v["group"] for v in verses if v["work"] == w), "n": 0}
        cnt = Counter(a for v, a in zip(verses, authors) if v["work"] == w)
        row["n"] = sum(cnt.values())
        row.update({a: cnt.get(a, 0) for a in author_ids})
        row["majority"] = majority[w]
        row["purity"] = val["purity_by_work"][w]
        crosstab.append(row)
    _write_csv(out_dir / "work_by_author.csv", crosstab)

    summary = {
        "language": verses[0].get("language", "grc"),
        "n_verses": len(verses),
        "n_input_verses": n_input,
        "n_missing_profiles": n_input - len(verses),
        "n_works": len(set(v["work"] for v in verses)),
        "used_ai_profiles": bool(profiles),
        "profile_metadata": {
            "models": sorted({str(profiles[v["id"]].get("model", "unknown")) for v in verses}) if profiles else [],
            "prompt_versions": sorted({str(profiles[v["id"]].get("provenance", {}).get("prompt_version", "legacy")) for v in verses}) if profiles else [],
            "unverified_profiles": sum(profiles[v["id"]].get("provenance", {}).get("prompt_version") != PROMPT_VERSION for v in verses) if profiles else 0,
        },
        "feature_blocks": {name: int(Z.shape[1]) for name, Z, _ in blocks},
        "weights": weights,
        "window": window,
        "alpha": alpha,
        "pca_dims": int(Xred.shape[1]),
        "pca_explained_variance": round(explained, 3),
        "k_criterion": criterion,
        "k_selected": chosen,
        "k_selected_by_silhouette": chosen,  # compatibility with older report readers
        "k_bic_candidate": bic_candidate,
        "k_used": k,
        "k_forced": forced,
        "selection_status": selection_status,
        "selection_thresholds": {"min_bic_gain": 10.0, "min_resample_ari": 0.8},
        "stability": stability,
        "seed": seed,
        "interpretation": "Exploratory style groups, not identified authors. One group means no supported split, not one proven author.",
        "confidence_definition": "centroid distance margin, not a probability; zero for one group",
        "hdbscan": hdb,
        "validation": val,
        "n_outliers": len(outlier_rows),
    }
    (out_dir / "authors.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1, allow_nan=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False))
    return summary
