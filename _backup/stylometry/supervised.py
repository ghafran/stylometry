"""Supervised attribution, and the verification that attribution cannot do.

Attribution asks which candidate a text most resembles, and always answers. Verification asks whether
it resembles that candidate more than an arbitrary text would, and can answer no. The project's
accuracy gate fails on exactly that distinction: 85% attribution accuracy alongside 90.5% of unseen
authors wrongly accepted. A classifier with a softmax over known authors cannot fix it, because the
question it is asked has no "none of these" option.

So both are here. The classifiers are evaluated by whole-work holdout, never by random split, because
passages from one work share vocabulary and a random split leaks the answer across the fold boundary.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _model(kind: str, seed: int):
    if kind == "svm":
        return SVC(kernel="linear", C=1.0, random_state=seed)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown classifier {kind!r}; use 'svm' or 'forest'")


def classify(X: np.ndarray, labels: list[str], groups: list[str], kind: str = "svm",
             seed: int = 0) -> dict:
    """Leave-one-work-out supervised attribution.

    Standardisation is fitted on the training fold only. Fitting it on everything would let the test
    fold influence its own scaling, which inflates the result quietly.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if not (len(X) == len(y) == len(g)):
        raise ValueError("features, labels and groups must be the same length")
    if len(set(g)) < 2:
        raise ValueError("leave-one-work-out needs at least two groups")
    predictions = np.empty(len(y), dtype=object)
    for train, test in LeaveOneGroupOut().split(X, y, g):
        if len(set(y[train])) < 2:
            predictions[test] = None
            continue
        scaler = StandardScaler().fit(X[train])
        model = _model(kind, seed).fit(scaler.transform(X[train]), y[train])
        predictions[test] = model.predict(scaler.transform(X[test]))
    scored = [(t, p) for t, p in zip(y, predictions) if p is not None]
    correct = sum(t == p for t, p in scored)
    from collections import Counter
    counts = Counter(t for t, _ in scored)
    return {
        "classifier": kind, "n": len(scored), "n_classes": len(counts),
        "accuracy": correct / len(scored) if scored else 0.0,
        "majority_baseline": max(counts.values()) / len(scored) if scored else 0.0,
        "predictions": [p for p in predictions],
    }


def impostors(target: np.ndarray, candidate_pool: np.ndarray, impostor_pool: np.ndarray,
              rounds: int = 100, feature_fraction: float = 0.4, seed: int = 0) -> float:
    """Koppel & Winter's verification score: how often the candidate survives perturbation.

    A single distance says little - two texts on the same subject are close whoever wrote them. So the
    comparison is repeated on random halves of the feature space against a crowd of impostors, and the
    score is the share of rounds in which the candidate is still the nearest. A genuine match survives
    the perturbation; a topical coincidence does not.

    Returns a proportion in [0, 1]. It is a score, not a probability, and the threshold separating
    accept from reject has to be calibrated on known same-author and different-author pairs.
    """
    if not 0 < feature_fraction <= 1:
        raise ValueError("feature_fraction must be in (0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if len(impostor_pool) < 1:
        raise ValueError("verification needs at least one impostor")
    target = np.asarray(target, dtype=float).ravel()
    candidates = np.atleast_2d(np.asarray(candidate_pool, dtype=float))
    impostors_ = np.atleast_2d(np.asarray(impostor_pool, dtype=float))
    n_features = target.shape[0]
    take = max(1, int(round(n_features * feature_fraction)))
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(rounds):
        cols = rng.choice(n_features, take, replace=False)
        t = target[cols]
        best_candidate = min(np.linalg.norm(candidates[:, cols] - t, axis=1))
        best_impostor = min(np.linalg.norm(impostors_[:, cols] - t, axis=1))
        wins += best_candidate < best_impostor
    return wins / rounds
