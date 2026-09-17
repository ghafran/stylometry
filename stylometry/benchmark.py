"""Predeclared, work-disjoint evaluation of the deterministic lexical baseline.

This benchmark measures generalization across works in its reference corpus. It
cannot validate ancient historical identities, AI profiles, or other languages.
All choices below are fixed before looking at held-out reference-text results.
"""
from __future__ import annotations

PROTOCOL = {
    "version": "real-author-pilot-v2-canonical-work-id",
    "partition_basis": "SHA-256(seed:language:canonical work_id), ranked within author and assigned round-robin to three whole-work partitions; legacy fixtures without work_id use language:author:display work",
    "sample_lengths": [100, 500, 1000],
    "max_samples_per_work": 12,
    "seed": 42,
    "folds": 3,
    "model": "cosine nearest author centroid after training-only standardization",
    "ablations": ["closed_class", "full_lexical"],
    "verification_calibration": "threshold permits at most 5% negative calibration matches",
    "rejection_calibration": "threshold accepts at least 95% genuine known-author calibration matches",
    "gates": {
        "balanced_accuracy": {"minimum": 0.80},
        "verification_fpr": {"maximum": 0.05},
        "verification_fnr": {"maximum": 0.20},
        "unknown_false_acceptance": {"maximum": 0.05},
        "known_acceptance": {"minimum": 0.80},
    },
    "minimum_authors_for_pass": 8,
    "minimum_works_for_pass": 60,
    "confidence": "Conditional descriptive 95% intervals from whole-work resampling of fixed predictions; no refitting and no adjustment for shared training/candidate dependencies; zero/all-event rate boundaries guarded by work-level binomial limits",
    "bootstrap_replicates": 1000,
    "scope": "reference-corpus lexical baseline; not historical scripture attribution or AI-profile validation",
}

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .features import LexicalFeatureExtractor


def _work_identity(work: dict) -> str:
    """Canonical works, rather than display titles/editions, define independent units."""
    if "work_id" in work:
        canonical = work["work_id"]
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("work_id must be a nonempty canonical identifier when supplied")
        return f"{work['language']}:{canonical}"
    return ":".join((work["language"], work["author"], work["work"]))


def sample_works(works: list[dict], length: int, max_samples_per_work: int = 12) -> list[dict]:
    """Evenly cover each work with nonoverlapping, exactly ``length`` token spans."""
    if not isinstance(length, int) or length <= 0:
        raise ValueError("sample length must be a positive integer")
    if not isinstance(max_samples_per_work, int) or max_samples_per_work <= 0:
        raise ValueError("max_samples_per_work must be a positive integer")
    samples, identities, fingerprints = [], set(), {}
    for work in sorted(works, key=lambda w: (w["language"], w["author"], w["work"])):
        for name in ("author", "work", "language", "text_bare"):
            if not isinstance(work.get(name), str) or not work[name].strip():
                raise ValueError(f"each work needs a nonempty {name}")
        key = _work_identity(work)
        if key in identities:
            raise ValueError(f"duplicate work identity: {key}")
        identities.add(key)
        tokens = work["text_bare"].split()
        digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
        if digest in fingerprints:
            raise ValueError(f"duplicate reference texts: {fingerprints[digest]} and {key}")
        fingerprints[digest] = key
        blocks = len(tokens) // length
        if not blocks:
            continue
        indices = np.linspace(0, blocks - 1, min(max_samples_per_work, blocks), dtype=int)
        work_id = key
        opaque = hashlib.sha256(work_id.encode()).hexdigest()[:16]
        for index in indices:
            start = int(index) * length
            samples.append({
                "id": f"benchmark:{opaque}:{start}",
                "work_id": work_id, "author": work["author"], "work": work["work"],
                "language": work["language"], "genre": work.get("genre", "unknown"),
                "topic": work.get("topic", "unknown"), "source_id": work.get("source_id"),
                "text_bare": " ".join(tokens[start:start + length]), "n_tokens": length,
                "token_start": start, "token_end": start + length,
            })
    return samples


def work_partitions(samples: list[dict], seed: int = 42) -> tuple[list[dict], dict[str, int], list[dict]]:
    """Keep authors with >=3 works, assigning whole works to three fixed buckets."""
    by_author = defaultdict(set)
    for sample in samples:
        by_author[sample["author"]].add(sample["work_id"])
    eligible = {a for a, works in by_author.items() if len(works) >= 3}
    excluded = [{"author": a, "reason": "fewer than 3 works with complete samples", "works": len(w)}
                for a, w in sorted(by_author.items()) if a not in eligible]
    assignment = {}
    for author in sorted(eligible):
        # Canonical work IDs (language-namespaced) define the fixed hash, so an
        # edition/title change cannot silently become another independent work.
        # Legacy fixture IDs fall back to author plus display work.
        ids = sorted(by_author[author], key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).digest())
        assignment.update({w: i % 3 for i, w in enumerate(ids)})
    return [s for s in samples if s["author"] in eligible], assignment, excluded


def _fit_scores(train: list[dict], calibration: list[dict], test: list[dict], ablation: str, seed: int):
    extractor = LexicalFeatureExtractor(svd_dims=0 if ablation == "closed_class" else 40, seed=seed)
    extractor.fit(train)
    x_train, x_cal, x_test = (extractor.transform(rows) for rows in (train, calibration, test))
    names = extractor.feature_names_
    if ablation == "closed_class":
        mask = np.array([n.startswith("fw:") for n in names])
        x_train, x_cal, x_test = (x[:, mask] for x in (x_train, x_cal, x_test))
    if x_train.shape[1] == 0:
        raise ValueError("benchmark ablation has no feature columns")
    scaler = StandardScaler().fit(x_train)
    x_train, x_cal, x_test = (scaler.transform(x) for x in (x_train, x_cal, x_test))
    authors = sorted({s["author"] for s in train})
    # Each work gets equal mass in its author's centroid, regardless of retained sample count.
    centroids = []
    for author in authors:
        work_ids = sorted({s["work_id"] for s in train if s["author"] == author})
        work_centroids = [x_train[[s["work_id"] == w for s in train]].mean(axis=0) for w in work_ids]
        centroids.append(np.mean(work_centroids, axis=0))
    return authors, cosine_similarity(x_cal, centroids), cosine_similarity(x_test, centroids), x_train.shape[1]


def calibrate_thresholds(calibration: list[dict], scores: np.ndarray, authors: list[str]) -> tuple[float, float]:
    """Use calibration works only: negative-match FPR<=.05, known retention>=.95."""
    labels = np.array([[s["author"] == author for author in authors] for s in calibration])
    if not labels.any() or not (~labels).any():
        raise ValueError("calibration needs at least two known authors")
    # 'higher' plus strict nextafter ensures ties cannot exceed the target false-match rate.
    verification = float(np.nextafter(np.quantile(scores[~labels], .95, method="higher"), np.inf))
    # Genuine scores, rather than winner scores, avoid calibrating rejection on known errors.
    rejection = float(np.quantile(scores[labels], .05, method="lower"))
    return verification, rejection


def _records(rows, scores, authors, verification, rejection, fold, heldout_author=None, references=None):
    result = []
    for sample, values in zip(rows, scores):
        winner = int(np.argmax(values))
        known = sample["author"] in authors
        positive = [float(v) for a, v in zip(authors, values) if a == sample["author"]]
        negative = [float(v) for a, v in zip(authors, values) if a != sample["author"]]
        result.append({
            "id": sample["id"], "work_id": sample["work_id"], "author": sample["author"],
            "candidate_scores": dict(zip(authors, map(float, values))),
            "reference_metadata": {a: {key: sorted({r[key] for r in (references or []) if r["author"] == a}) for key in ("genre", "topic")} for a in authors},
            "genre": sample["genre"], "topic": sample["topic"], "fold": fold,
            "heldout_author": heldout_author, "known": known,
            "predicted_author": authors[winner], "correct": bool(known and authors[winner] == sample["author"]),
            "accepted": bool(values[winner] >= rejection),
            "verification_positive": positive, "verification_negative": negative,
            "verification_fpr": float(np.mean(np.array(negative) >= verification)) if negative else None,
            "verification_fnr": float(np.mean(np.array(positive) < verification)) if positive else None,
            "verification_threshold": verification, "rejection_threshold": rejection,
        })
    return result


def _mean_by_work(records, key):
    grouped = defaultdict(list)
    for row in records:
        value = row.get(key)
        if value is not None:
            grouped[(row["author"], row["work_id"])].append(float(value))
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def _author_work_mean(values):
    by_author = defaultdict(list)
    for (author, _), value in values.items():
        by_author[author].append(value)
    return float(np.mean([np.mean(v) for v in by_author.values()])) if by_author else None


def rate_estimate(records: list[dict], key: str, seed: int = 42, replicates: int = 1000) -> dict:
    """Conditional descriptive work bootstrap of fixed predictions, stratified by author.

    Feature extraction, scaling, and calibration are not refitted. Dependencies
    through shared training works and candidate authors are not modeled, so these
    are not calibrated uncertainty intervals for a new corpus or historical claim.
    """
    values = _mean_by_work(records, key)
    point = _author_work_mean(values)
    if point is None:
        return {"value": None, "ci95": None, "independent_works": 0}
    grouped = defaultdict(list)
    for (author, _), value in values.items():
        grouped[author].append(value)
    rng = np.random.default_rng(seed)
    boot = np.zeros(replicates)
    for group in grouped.values():
        arr = np.asarray(group)
        boot += rng.choice(arr, size=(replicates, len(arr)), replace=True).mean(axis=1) / len(grouped)
    lower, upper = np.quantile(boot, [.025, .975])
    # A percentile bootstrap cannot infer unseen errors. At the boundaries use
    # conservative exact bounds on entire-work events, retaining whole-work n.
    # For nonbinary fractional rates: all-error and any-error work events bound
    # the mean error; the all-zero/all-one cases coincide with those events.
    n = len(values)
    if point == 0.0:
        upper = max(upper, float(beta.ppf(.975, 1, n)))
    if point == 1.0:
        lower = min(lower, float(beta.ppf(.025, n, 1)))
    return {"value": point, "ci95": [float(lower), float(upper)], "independent_works": n,
            "ci_method": "conditional descriptive whole-work bootstrap of fixed predictions, stratified by author; work-event boundary guard; no model refitting"}


def _auc(records):
    if not records:
        return None
    labels, scores, weights = [], [], []
    per_work = Counter(r["work_id"] for r in records)
    for row in records:
        for key, label in (("verification_positive", 1), ("verification_negative", 0)):
            vals = row[key]
            for score in vals:
                labels.append(label)
                scores.append(score)
                weights.append(1 / per_work[row["work_id"]] / max(1, len(vals)))
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def auc_estimate(records: list[dict], seed=42, replicates=1000):
    """Mean within-fold AUROC, with conditional descriptive work-bootstrap intervals."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["fold"], row["author"], row["work_id"])].append(row)
    folds = sorted({r["fold"] for r in records})
    point = np.mean([_auc([r for r in records if r["fold"] == f]) for f in folds]) if folds else None
    if point is None:
        return {"value": None, "ci95": None, "independent_works": 0}
    # Stratify by fold and author; repeated draws are distinct bootstrap copies
    # so each retains its complete sample set and equal work mass.
    strata = defaultdict(list)
    for (fold, author, _), rows in groups.items():
        strata[(fold, author)].append(rows)
    if any(len(blocks) < 2 for blocks in strata.values()):
        return {"value": float(point), "ci95": None, "independent_works": len(groups),
                "ci_method": "unavailable: fewer than two independent works in at least one fold/author stratum",
                "note": "A whole-work bootstrap cannot estimate unseen-work variability from a singleton stratum. The point AUROC is descriptive only."}
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(replicates):
        by_fold = defaultdict(list)
        for (fold, author), blocks in strata.items():
            for copy, index in enumerate(rng.integers(0, len(blocks), len(blocks))):
                by_fold[fold].extend([{**r, "work_id": f"{author}:{copy}"} for r in blocks[index]])
        estimates.append(np.mean([_auc(rows) for rows in by_fold.values()]))
    return {"value": float(point), "ci95": list(map(float, np.quantile(estimates, [.025, .975]))),
            "independent_works": len(groups), "ci_method": "conditional descriptive whole-work bootstrap of fixed predictions, stratified by fold and author; no model refitting",
            "note": "Within-fold AUROC averaged across folds. Intervals are conditional on fixed predictions and these authors; feature fitting, scaling, calibration, and shared training/candidate dependencies are not resampled or modeled. Single-work strata cannot estimate between-work uncertainty."}


def gate_result(estimate: dict, gate: dict) -> str:
    if estimate.get("ci95") is None:
        return "inconclusive"
    lower, upper = estimate["ci95"]
    if "minimum" in gate:
        return "passed" if lower >= gate["minimum"] else "failed" if upper < gate["minimum"] else "inconclusive"
    return "passed" if upper <= gate["maximum"] else "failed" if lower > gate["maximum"] else "inconclusive"


def _subset_checks(records, samples):
    # Require the held-out work AND every contributing reference work for the
    # candidate centroid to share the label. Never infer missing metadata.
    result = {}
    for label in ("genre", "topic"):
        values = sorted({s[label] for s in samples if s[label] not in ("", "unknown", None)})
        entries = []
        for value in values:
            selected = []
            for row in records:
                if row[label] != value:
                    continue
                candidates = {a: score for a, score in row["candidate_scores"].items()
                              if row["reference_metadata"][a][label] == [value]}
                if row["author"] not in candidates or len(candidates) < 2:
                    continue
                negatives = [score for a, score in candidates.items() if a != row["author"]]
                selected.append({**row, "correct": max(candidates, key=candidates.get) == row["author"],
                                 "verification_fpr": float(np.mean(np.array(negatives) >= row["verification_threshold"]))})
            if not selected:
                continue
            entries.append({"value": value, "authors": len({r["author"] for r in selected}),
                            "works": len({r["work_id"] for r in selected}),
                            "balanced_accuracy": rate_estimate(selected, "correct"),
                            "verification_fpr": rate_estimate(selected, "verification_fpr"),
                            "verification_fnr": rate_estimate(selected, "verification_fnr"),
                            "interpretation": "All contributing reference works and test works share the label; comparisons require at least two candidate authors. Thresholds remain calibrated on the original calibration partition."})
        result[f"shared_{label}"] = entries or [{"status": "unsupported", "reason": f"insufficient matching test and reference {label} labels across multiple authors"}]
        cross = [r for r in records if r[label] not in ("", "unknown", None)
                 and r["reference_metadata"].get(r["author"], {}).get(label)
                 and r[label] not in r["reference_metadata"][r["author"]][label]
                 and "unknown" not in r["reference_metadata"][r["author"]][label]]
        result[f"same_author_cross_{label}"] = ({
            "works": len({r["work_id"] for r in cross}), "balanced_accuracy": rate_estimate(cross, "correct"),
            "verification_fnr": rate_estimate(cross, "verification_fnr"),
        } if cross else {"status": "unsupported", "reason": f"no eligible held-out works with {label} different from the author's training works"})
    return result


def _evaluate(samples, assignment, ablation, seed, bootstrap_replicates):
    closed, opened, fold_details = [], [], []
    authors = sorted({s["author"] for s in samples})
    for fold in range(3):
        train = [s for s in samples if assignment[s["work_id"]] == fold]
        cal = [s for s in samples if assignment[s["work_id"]] == (fold + 1) % 3]
        test = [s for s in samples if assignment[s["work_id"]] == (fold + 2) % 3]
        labels, cal_scores, test_scores, feature_count = _fit_scores(train, cal, test, ablation, seed)
        verification, rejection = calibrate_thresholds(cal, cal_scores, labels)
        closed.extend(_records(test, test_scores, labels, verification, rejection, fold, references=train))
        fold_details.append({"fold": fold, "train_works": sorted({s["work_id"] for s in train}),
                             "calibration_works": sorted({s["work_id"] for s in cal}),
                             "test_works": sorted({s["work_id"] for s in test}),
                             "feature_count": feature_count, "verification_threshold": verification,
                             "rejection_threshold": rejection})
        if len(authors) < 3:
            continue
        for unknown in authors:
            known_train = [s for s in train if s["author"] != unknown]
            known_cal = [s for s in cal if s["author"] != unknown]
            # Each unknown work appears once, as its bucket becomes the test
            # partition; no work from this author enters fit OR calibration.
            labels, cal_scores, test_scores, _ = _fit_scores(known_train, known_cal, test, ablation, seed)
            verify, reject = calibrate_thresholds(known_cal, cal_scores, labels)
            opened.extend(_records(test, test_scores, labels, verify, reject, fold, unknown, references=known_train))
    metrics = {
        "balanced_accuracy": rate_estimate(closed, "correct", seed, bootstrap_replicates),
        "verification_auroc": auc_estimate(closed, seed, bootstrap_replicates),
        "verification_fpr": rate_estimate(closed, "verification_fpr", seed, bootstrap_replicates),
        "verification_fnr": rate_estimate(closed, "verification_fnr", seed, bootstrap_replicates),
        "unknown_false_acceptance": rate_estimate([r for r in opened if not r["known"]], "accepted", seed, bootstrap_replicates),
        "known_acceptance": rate_estimate([r for r in opened if r["known"]], "accepted", seed, bootstrap_replicates),
        "accuracy_among_accepted": rate_estimate([r for r in opened if r["known"] and r["accepted"]], "correct", seed, bootstrap_replicates),
    }
    gates = {key: {**gate, "status": gate_result(metrics[key], gate)} for key, gate in PROTOCOL["gates"].items()}
    n_works = len({s["work_id"] for s in samples})
    coverage = len(authors) >= PROTOCOL["minimum_authors_for_pass"] and n_works >= PROTOCOL["minimum_works_for_pass"]
    status = ("failed" if any(g["status"] == "failed" for g in gates.values()) else
              "passed" if coverage and all(g["status"] == "passed" for g in gates.values()) else "inconclusive")
    by_author = []
    for author in authors:
        author_samples = [s for s in samples if s["author"] == author]
        by_author.append({"author": author, "works": len({s["work_id"] for s in author_samples}),
                          "samples": len(author_samples),
                          "balanced_accuracy": rate_estimate([r for r in closed if r["author"] == author], "correct", seed, bootstrap_replicates),
                          "unknown_false_acceptance": rate_estimate([r for r in opened if not r["known"] and r["author"] == author], "accepted", seed, bootstrap_replicates)})
    return {"ablation": ablation, "status": status, "adequate_coverage_for_pass": coverage,
            "authors": len(authors), "works": n_works, "samples": len(samples),
            "metrics": metrics, "gates": gates, "folds": fold_details, "by_author": by_author,
            "subsets": _subset_checks(closed, samples),
            "closed_set_predictions": closed, "open_set_predictions": opened}


def run_benchmark(works: list[dict], output_dir: str | Path | None = None,
                  sample_lengths=(100, 500, 1000), max_samples_per_work=12,
                  seed=42, bootstrap_replicates=1000, progress: Callable[[str], None] | None = None) -> dict:
    """Run the fixed lexical reference-author protocol and optionally write artifacts.

    Overrides are recorded and make results exploratory, never eligible for a
    preregistered pass. No network calls or AI API calls are made here.
    """
    if not works:
        raise ValueError("benchmark needs reference works")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    lengths = list(sample_lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("sample_lengths must contain distinct positive integers")
    runs, exclusions = [], []
    for length in lengths:
        samples = sample_works(works, length, max_samples_per_work)
        for language in sorted({w["language"] for w in works}):
            lang_samples = [s for s in samples if s["language"] == language]
            eligible, assignment, excluded = work_partitions(lang_samples, seed)
            exclusions.extend({"language": language, "sample_length": length, **item} for item in excluded)
            available = {s["work_id"] for s in lang_samples}
            for w in works:
                if w["language"] == language and _work_identity(w) not in available:
                    exclusions.append({"language": language, "sample_length": length, "author": w["author"],
                                       "work": w["work"], "work_id": _work_identity(w), "reason": "fewer tokens than one complete sample"})
            if len({s["author"] for s in eligible}) < 2:
                if progress:
                    progress(f"Skipped {language}, {length} tokens: insufficient independent reference works")
                runs.append({"language": language, "sample_length": length, "status": "inconclusive",
                             "reason": "need at least two authors with at least three distinct works each"})
                continue
            for ablation in PROTOCOL["ablations"]:
                display = "predefined function/common-word features" if ablation == "closed_class" else "full lexical features"
                if progress:
                    progress(f"Evaluating {language}, {length} tokens, {display}")
                result = _evaluate(eligible, assignment, ablation, seed, bootstrap_replicates)
                runs.append({"language": language, "sample_length": length, **result})
                if progress:
                    progress(f"Completed {language}, {length} tokens, {display}: {result['status']}")
    exploratory = (lengths != PROTOCOL["sample_lengths"] or max_samples_per_work != PROTOCOL["max_samples_per_work"]
                   or seed != PROTOCOL["seed"] or bootstrap_replicates != PROTOCOL["bootstrap_replicates"])
    if exploratory:
        for run in runs:
            if run["status"] == "passed":
                run["status"] = "inconclusive"
    overall = ("failed" if any(r["status"] == "failed" for r in runs) else
               "passed" if runs and all(r["status"] == "passed" for r in runs) and not exploratory else "inconclusive")
    report = {
        "protocol": PROTOCOL, "status": overall, "exploratory_override": exploratory,
        "settings": {"sample_lengths": lengths, "max_samples_per_work": max_samples_per_work,
                     "seed": seed, "bootstrap_replicates": bootstrap_replicates},
        "reference_works": [{k: v for k, v in w.items() if k != "text_bare"} for w in works],
        "runs": runs, "exclusions": exclusions,
        "limitations": [
            "Observed labels are reference-corpus attributions, not independently proven ancient identities.",
            "Passing measures this deterministic lexical baseline and corpus only; it does not validate AI profiles, clustering, or scripture attribution.",
            "Intervals are conditional descriptive intervals from resampling fixed held-out predictions by work. They do not refit features, scaling, centroids, or calibration; dependencies from shared training works and candidate authors are unmodeled. They are not calibrated uncertainty for unseen corpora or historical attribution.",
            "The predefined function/common-word ablation (machine key closed_class) includes common content verbs in the Greek and Hebrew lists; it is not topic-free.",
            "Authors, periods, editors, and manuscripts may be confounded in the sampled corpus.",
            "At least 8 reference authors and 60 independent works per language/length are required for a pass; small samples remain pilots.",
            "No canonical or disputed scripture is treated as known authorship ground truth.",
            "Open-set rejection uses known-author calibration only; unseen authors may resemble known authors and be falsely accepted.",
            "Topic and genre metadata subsets are descriptive and not equivalent to a fully matched corpus.",
            "A failure is evidence against the tested reliability claim; an inconclusive result is not evidence of reliability.",
        ],
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (destination / "report.md").write_text(render_report(report))
    return report


def _display(metric):
    if not metric or metric["value"] is None:
        return "unsupported"
    if metric["ci95"] is None:
        return f"{metric['value']:.1%} [interval unavailable]"
    lo, hi = metric["ci95"]
    return f"{metric['value']:.1%} [{lo:.1%}, {hi:.1%}]"


def render_report(report: dict) -> str:
    lines = ["# Real-author reference benchmark", "", f"**Overall result: {report['status'].upper()}.**",
             "", "This is a reference-corpus evaluation of the deterministic lexical baseline. It does not establish scripture authorship or validate AI profiles.",
             "", "## Protocol fixed before evaluating results", "",
             "Entire canonical works rotate through separate training, calibration, and test partitions. Partitions rank SHA-256(seed:language:canonical work_id) within author and assign works round-robin to three buckets. Canonical IDs are required from reference manifests; legacy fixtures without an ID fall back to language:author:display work. Duplicate canonical IDs are rejected even if titles, authors, or editions differ. Protocol v2 corrects the preliminary display-title partition basis; thresholds and modeling choices are unchanged. Samples do not overlap. Features, scaling, and centroids fit only training works. Verification and rejection thresholds use only calibration works. Unknown-author runs exclude that author completely from fitting and calibration.",
             "", "Feature ablations: predefined function/common-word features (machine key `closed_class`), and full lexical features. The predefined Greek and Hebrew lists include common content verbs, so this ablation is not topic-free. Thresholds are not tuned against test results. Samples cover each work uniformly at 100, 500, and 1,000 tokens (up to 12 samples per work).",
             "", "Predeclared gates: balanced accuracy ≥80%, verification false-positive rate ≤5%, verification false-negative rate ≤20%, unknown-author false acceptance ≤5%, known-author acceptance ≥80%. A gate passes only when its entire 95% interval clears the target; intervals overlapping a target are inconclusive. Overall passing also requires ≥8 authors and ≥60 works in every evaluated language and sample length.",
             "", "The intervals are conditional descriptive intervals: they resample fixed held-out predictions by whole work, stratified by author. Features, scaling, centroids, and calibration are not refitted. Dependencies through shared training works and candidate authors are unmodeled, so these intervals are not calibrated uncertainty for unseen corpora or historical claims. Zero/all-event rates additionally use work-count boundary guards. Chunks are not treated as independent evidence. AUROC intervals are unavailable if any fold/author stratum has only one work.",
             "", "## Results", "", "Values show estimate [conditional descriptive 95% interval].", "",
             "| Language | Tokens | Features | Authors / works | Accuracy | Verify AUROC | Verify FPR | Verify FNR | Unknown accepted | Known accepted | Result |",
             "|---|---:|---|---:|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        if "metrics" not in run:
            lines.append(f"| {run['language']} | {run['sample_length']} | — | — | — | — | — | — | — | — | Inconclusive: {run['reason']} |")
            continue
        m = run["metrics"]
        display = "predefined function/common-word features" if run["ablation"] == "closed_class" else "full lexical features"
        lines.append(f"| {run['language']} | {run['sample_length']} | {display} | {run['authors']} / {run['works']} | " + " | ".join(_display(m[k]) for k in ("balanced_accuracy", "verification_auroc", "verification_fpr", "verification_fnr", "unknown_false_acceptance", "known_acceptance")) + f" | {run['status']} |")
    if report["exploratory_override"]:
        lines.extend(["", "**Exploratory settings override:** this run changed predeclared defaults and cannot receive a preregistered pass."])
    lines.extend(["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]])
    lines.extend(["", "Full per-author results, held-out work assignments, calibration thresholds, metadata subsets, predictions, exclusions, and source provenance are in `benchmark.json`.", ""])
    return "\n".join(lines)
