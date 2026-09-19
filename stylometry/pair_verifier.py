"""Fixed cross-work same-author verifier for author-disjoint experiments.

This learns a symmetric comparison rule; it does not identify authors or return
probabilities. Whole-gallery calibration and genuinely unseen-author evaluation
belong to the caller. Metadata never enters the predictive feature vectors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations, product
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import LexicalFeatureExtractor


_MAX_PAIRS_PER_WORK_PAIR = 8
_REQUIRED = ("id", "work_id", "author", "language", "text_bare", "genre", "topic")


def _text(row):
    return " ".join(row["text_bare"].split())


def _signature(row):
    return tuple(_text(row) if key == "text_bare" else row[key] for key in _REQUIRED)


def _validate(rows, *, language=None, unique_ids=False, unique_text=False):
    rows = list(rows)
    works, ids, texts, languages = {}, {}, {}, set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in _REQUIRED
        ):
            raise ValueError("passages need nonempty " + ", ".join(_REQUIRED))
        if row["language"] not in ("grc", "hbo", "arb", "eng"):
            raise ValueError("unsupported language")
        languages.add(row["language"])
        author = works.setdefault(row["work_id"], row["author"])
        if author != row["author"]:
            raise ValueError("a canonical work_id cannot have conflicting authors")
        signature = _signature(row)
        if row["id"] in ids and (unique_ids or ids[row["id"]] != signature):
            raise ValueError("duplicate or conflicting passage id")
        ids[row["id"]] = signature
        normalized = _text(row)
        if unique_text and normalized in texts:
            raise ValueError("duplicate normalized training passage text is not independent evidence")
        texts[normalized] = row["id"]
    if len(languages) > 1 or (language is not None and languages - {language}):
        raise ValueError("passages must use the fitted language")
    return rows


def _passage_weights(rows):
    chunks = Counter(row["work_id"] for row in rows)
    works = defaultdict(set)
    for row in rows:
        works[row["author"]].add(row["work_id"])
    total = len(chunks) / len(works)
    return np.array([total / len(works[row["author"]]) / chunks[row["work_id"]]
                     for row in rows])


def _matches(left, right, key):
    return left[key] == right[key] and left[key] not in ("unknown", "unspecified")


class PairVerifier:
    """Train a fixed regularized verifier on cross-canonical-work pairs only."""

    def __init__(self, seed=42):
        self.seed = seed
        self.fitted_ = False

    def fit(self, training_passage_rows):
        self.fitted_ = False
        rows = _validate(training_passage_rows, unique_ids=True, unique_text=True)
        if not rows or len({r["author"] for r in rows}) < 2:
            raise ValueError("training needs passages from at least two authors")
        rows = sorted(rows, key=lambda r: (r["work_id"], r["id"]))
        self.language_ = rows[0]["language"]
        self.training_authors_ = sorted({r["author"] for r in rows})
        self.training_work_ids_ = sorted({r["work_id"] for r in rows})
        self.training_signatures_ = {r["id"]: _signature(r) for r in rows}
        self.training_work_authors_ = {r["work_id"]: r["author"] for r in rows}
        self.training_sample_weights_ = _passage_weights(rows)

        pairs, work_pairs = self._training_pairs(rows)
        labels = np.array([pair["same_author"] for pair in pairs], dtype=int)
        if set(labels) != {0, 1}:
            raise ValueError("training needs different-author pairs and same-author pairs from different canonical works")
        self.training_pairs_ = pairs
        self.pair_weights_ = self._pair_weights(pairs, work_pairs)

        self.extractor_ = LexicalFeatureExtractor(svd_dims=0, seed=self.seed).fit(rows)
        self.static_mask_ = np.array([name != "misc:log_tokens" for name in self.extractor_.feature_names_])
        static_names = [name for name, keep in zip(self.extractor_.feature_names_, self.static_mask_) if keep]
        self.feature_names_ = [f"absdiff:{name}" for name in static_names] + [
            "comparison:static_cosine", "comparison:static_rms_distance", "comparison:character_cosine",
        ]
        static = self.extractor_.transform(rows)[:, self.static_mask_]
        self.static_scaler_ = StandardScaler().fit(static, sample_weight=self.training_sample_weights_)
        static = self.static_scaler_.transform(static)
        self.vectorizer_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000,
                                         min_df=1, sublinear_tf=True, lowercase=False)
        documents = defaultdict(list)
        for row in rows:
            documents[row["work_id"]].append(row["text_bare"])
        self.vectorizer_.fit([" ".join(documents[work]) for work in self.training_work_ids_])
        characters = self.vectorizer_.transform([row["text_bare"] for row in rows])
        left = np.array([pair["left_index"] for pair in pairs])
        right = np.array([pair["right_index"] for pair in pairs])
        features = self._pair_features(static[left], static[right], characters[left], characters[right])
        self.pair_scaler_ = StandardScaler().fit(features, sample_weight=self.pair_weights_)
        self.classifier_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                self.classifier_.fit(self.pair_scaler_.transform(features), labels,
                                     sample_weight=self.pair_weights_)
            except ConvergenceWarning as exc:
                raise ValueError("fixed pair verifier did not converge") from exc

        self.metadata_ = {
            "kind": "cross_work_pair_verifier", "seed": self.seed, "language": self.language_,
            "training_authors": self.training_authors_, "training_work_ids": self.training_work_ids_,
            "training_passages": len(rows), "training_work_pairs": work_pairs,
            "training_pair_counts": {"positive": int(labels.sum()), "negative": int((labels == 0).sum())},
            "training_pair_weight_sums": {"positive": float(self.pair_weights_[labels == 1].sum()),
                                          "negative": float(self.pair_weights_[labels == 0].sum()),
                                          "total": float(self.pair_weights_.sum())},
            "mass_basis": "number of canonical training works, not passages or work-pairs",
            "negative_matching_counts": {
                "same_genre_passage_pairs": sum(not p["same_author"] and p["same_genre"] for p in pairs),
                "same_topic_passage_pairs": sum(not p["same_author"] and p["same_topic"] for p in pairs),
                "same_genre_work_pairs": sum(not p["same_author"] and p["same_genre"] for p in work_pairs),
                "same_topic_work_pairs": sum(not p["same_author"] and p["same_topic"] for p in work_pairs),
            },
            "negative_strata": {
                stratum: {"available": any(p.get("negative_stratum") == stratum for p in work_pairs),
                          "work_pairs": sum(p.get("negative_stratum") == stratum for p in work_pairs),
                          "passage_pairs": sum(p["selected_passage_pairs"] for p in work_pairs if p.get("negative_stratum") == stratum),
                          "weight_sum": float(sum(p["weight_sum"] for p in work_pairs if p.get("negative_stratum") == stratum))}
                for stratum in ("same_genre", "other")
            },
            "missing_negative_strata": [stratum for stratum in ("same_genre", "other")
                                        if not any(p.get("negative_stratum") == stratum for p in work_pairs)],
            "feature_names": self.feature_names_, "feature_count": len(self.feature_names_),
            "character_vocabulary_size": len(self.vectorizer_.vocabulary_),
            "parameters": {"max_passage_pairs_per_work_pair": _MAX_PAIRS_PER_WORK_PAIR,
                           "pair_sampling": "lowest SHA-256(seed:left_id:right_id), with canonical work order",
                           "static_features": "predefined function/common words, word endings, descriptive rates; excludes log_tokens",
                           "character_features": "char_wb 3-5 TF-IDF; max10000; sublinear_tf; canonical training works as IDF documents",
                           "negative_stratum_weighting": "equal same-genre/other mass when both present; available stratum receives full negative mass otherwise",
                           "classifier": "LogisticRegression", "solver": "lbfgs", "C": 1.0, "max_iter": 2000},
            "weighting": "equal positive/negative class mass; positives balance authors then canonical work-pairs then passage-pairs; negatives balance available same-genre/other strata then unordered author-pairs then canonical work-pairs then passage-pairs. Total mass equals canonical training work count, fixing C=1 to available works rather than correlated pairs. Training feature scaling separately balances authors, works, passages.",
            "score_type": "raw same-author decision score, not an author probability",
            "scope": "fixed development candidate; caller must keep calibration and evaluation authors absent from training",
            "limitations": [
                "Different editions and chunks of the same canonical work never form training pairs.",
                "All different-author work-pairs are included; same-genre and other negatives receive equal stratum mass when both exist. Missing strata are reported, not manufactured.",
                "Genre metadata affect fixed training weights, never predictive features. Topic matching is reported without extra weighting.",
                "Genre and topic labels are broad controls; period/editor matching is not established by this metadata.",
                "Word endings and predefined common words are crude proxies, not validated morphology or topic-free syntax.",
                "Whole-gallery calibration is required; a pair score cannot by itself authorize attribution.",
            ],
        }
        self.fitted_ = True
        return self

    def _training_pairs(self, rows):
        by_work = defaultdict(list)
        for index, row in enumerate(rows):
            by_work[row["work_id"]].append(index)
        pairs, work_pairs = [], []
        for work_left, work_right in combinations(sorted(by_work), 2):
            candidates = list(product(by_work[work_left], by_work[work_right]))
            candidates.sort(key=lambda pair: hashlib.sha256(
                f"{self.seed}:{rows[pair[0]]['id']}:{rows[pair[1]]['id']}".encode()).digest())
            chosen = candidates[:_MAX_PAIRS_PER_WORK_PAIR]
            left_author = rows[chosen[0][0]]["author"]
            right_author = rows[chosen[0][1]]["author"]
            same = left_author == right_author
            same_genre = ({rows[i]["genre"] for i in by_work[work_left]}
                          == {rows[i]["genre"] for i in by_work[work_right]})
            same_topic = ({rows[i]["topic"] for i in by_work[work_left]}
                          == {rows[i]["topic"] for i in by_work[work_right]})
            # A mixed-label work pair is not described as a matched control.
            same_genre = same_genre and len({rows[i]["genre"] for i in by_work[work_left]}) == 1 and rows[chosen[0][0]]["genre"] not in ("unknown", "unspecified")
            same_topic = same_topic and len({rows[i]["topic"] for i in by_work[work_left]}) == 1 and rows[chosen[0][0]]["topic"] not in ("unknown", "unspecified")
            pair_id = len(work_pairs)
            work_pairs.append({"left_work_id": work_left, "right_work_id": work_right,
                               "left_author": left_author, "right_author": right_author,
                               "same_author": same, "same_genre": same_genre, "same_topic": same_topic,
                               "available_passage_pairs": len(candidates), "selected_passage_pairs": len(chosen)})
            for left, right in chosen:
                pairs.append({"left_index": left, "right_index": right,
                              "left_id": rows[left]["id"], "right_id": rows[right]["id"],
                              "work_pair_index": pair_id, "same_author": same,
                              "same_genre": _matches(rows[left], rows[right], "genre"),
                              "same_topic": _matches(rows[left], rows[right], "topic")})
        return pairs, work_pairs

    @staticmethod
    def _pair_weights(pairs, work_pairs):
        positive_groups = defaultdict(list)
        negative_strata = defaultdict(lambda: defaultdict(list))
        for index, pair in enumerate(work_pairs):
            if pair["same_author"]:
                positive_groups[pair["left_author"]].append(index)
            else:
                stratum = "same_genre" if pair["same_genre"] else "other"
                pair["negative_stratum"] = stratum
                group = tuple(sorted((pair["left_author"], pair["right_author"])))
                negative_strata[stratum][group].append(index)
        work_count = len({p[key] for p in work_pairs for key in ("left_work_id", "right_work_id")})
        mass = work_count / 2
        for indices in positive_groups.values():
            for index in indices:
                work_pairs[index]["weight_sum"] = mass / len(positive_groups) / len(indices)
        for groups in negative_strata.values():
            for indices in groups.values():
                for index in indices:
                    work_pairs[index]["weight_sum"] = mass / len(negative_strata) / len(groups) / len(indices)
        return np.array([work_pairs[p["work_pair_index"]]["weight_sum"]
                         / work_pairs[p["work_pair_index"]]["selected_passage_pairs"] for p in pairs])

    @staticmethod
    def _pair_features(left_static, right_static, left_characters, right_characters):
        differences = np.abs(left_static - right_static)
        denominator = np.linalg.norm(left_static, axis=1) * np.linalg.norm(right_static, axis=1)
        static_cosine = np.divide(np.einsum("ij,ij->i", left_static, right_static), denominator,
                                  out=np.zeros(len(left_static)), where=denominator > 0)
        rms = np.sqrt(np.mean(differences ** 2, axis=1))
        char_cosine = np.asarray(left_characters.multiply(right_characters).sum(axis=1)).ravel()
        return np.column_stack([differences, static_cosine, rms, char_cosine])

    def score_pairs(self, left_rows, right_rows):
        """Score aligned cross-work pairs without updating vocabulary or parameters."""
        if not self.fitted_:
            raise ValueError("fit the pair verifier before scoring")
        left, right = list(left_rows), list(right_rows)
        if len(left) != len(right):
            raise ValueError("left and right pair batches must have equal lengths")
        _validate(left + right, language=self.language_)
        if not left:
            return np.empty(0, dtype=float)
        for row in left + right:
            if row["id"] in self.training_signatures_ and _signature(row) != self.training_signatures_[row["id"]]:
                raise ValueError("scored passage conflicts with its training identity")
            if (row["work_id"] in self.training_work_authors_
                    and row["author"] != self.training_work_authors_[row["work_id"]]):
                raise ValueError("scored canonical work conflicts with its training author")
        for first, second in zip(left, right):
            if first["work_id"] == second["work_id"]:
                raise ValueError("verification requires different canonical works")
            if _text(first) == _text(second):
                raise ValueError("identical passage text is not independent authorship evidence")
        # A gallery query appears once per reference passage. Transform each
        # unchanged passage signature once, preserving the exact frozen feature
        # map and indexing it back into the requested pair order.
        unique, lookup, indices = [], {}, []
        for row in left + right:
            signature = _signature(row)
            if signature not in lookup:
                lookup[signature] = len(unique)
                unique.append(row)
            indices.append(lookup[signature])
        static = self.static_scaler_.transform(self.extractor_.transform(unique)[:, self.static_mask_])
        characters = self.vectorizer_.transform([row["text_bare"] for row in unique])
        split = len(left)
        left_indices, right_indices = np.asarray(indices[:split]), np.asarray(indices[split:])
        features = self._pair_features(static[left_indices], static[right_indices],
                                       characters[left_indices], characters[right_indices])
        scores = np.asarray(self.classifier_.decision_function(self.pair_scaler_.transform(features)), dtype=float)
        if scores.shape != (split,) or not np.isfinite(scores).all():
            raise ValueError("verifier produced invalid pair scores")
        return scores
