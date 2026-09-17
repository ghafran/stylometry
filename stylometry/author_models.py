"""Fixed candidate models for developmental, work-disjoint author comparisons.

Scores are cosine similarities or linear classifier decision scores, never
author probabilities. No model here establishes historical author identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .features import LexicalFeatureExtractor
from .lang import function_words


MODEL_KINDS = ("function_centroid", "morph_centroid", "char_linear")
_BIGRAM_LIMIT = 256


def _validate_rows(rows, language=None):
    rows = list(rows)
    works = {}
    languages = set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("author", "work_id", "text_bare", "language")
        ):
            raise ValueError("rows need nonempty author, work_id, text_bare, and language strings")
        if row["language"] not in ("grc", "hbo", "arb"):
            raise ValueError("unsupported language")
        languages.add(row["language"])
        if row["work_id"] in works and works[row["work_id"]] != row["author"]:
            raise ValueError("a canonical work_id cannot have multiple author labels")
        works[row["work_id"]] = row["author"]
    if len(languages) > 1 or (language is not None and languages - {language}):
        raise ValueError("rows must use the same language as the fitted model")
    return rows


def _work_weights(rows):
    """Equal author mass, equal works within author, equal chunks within work.

    Total mass is the number of canonical training works, independent of how
    many chunks were sampled. This fixes the regularization scale of C=1.
    """
    chunks = Counter(row["work_id"] for row in rows)
    works = defaultdict(set)
    for row in rows:
        works[row["author"]].add(row["work_id"])
    mass = len(chunks) / len(works)
    return np.array([mass / len(works[row["author"]]) / chunks[row["work_id"]]
                     for row in rows])


def _bigram_rates(text, vocabulary):
    words = text.split()
    counts = Counter((first, second) for first, second in zip(words, words[1:])
                     if first in vocabulary and second in vocabulary)
    return {pair: count * 100 / max(1, len(words) - 1) for pair, count in counts.items()}


class AuthorModel:
    """A fixed feature recipe fitted only on the supplied training passages."""

    def __init__(self, kind, seed=42):
        if kind not in MODEL_KINDS:
            raise ValueError(f"kind must be one of {MODEL_KINDS}")
        self.kind = kind
        self.seed = seed
        self.fitted_ = False

    def fit(self, trainpassagerows):
        self.fitted_ = False
        rows = _validate_rows(trainpassagerows)
        if not rows or len({row["author"] for row in rows}) < 2:
            raise ValueError("training needs at least two authors with nonempty passages")
        self.language_ = rows[0]["language"]
        self.authors_ = sorted({row["author"] for row in rows})
        self.training_work_ids_ = sorted({row["work_id"] for row in rows})
        self.sample_weight_ = _work_weights(rows)
        self.metadata_ = {
            "kind": self.kind, "seed": self.seed, "language": self.language_,
            "candidate_authors": self.authors_, "training_works": self.training_work_ids_,
            "training_samples": len(rows),
            "weighting": "equal authors, then equal canonical works, then equal chunks; total weight equals training work count",
            "scope": "fixed development candidate; no hyperparameter search or validation claim",
        }
        if self.kind == "char_linear":
            self._fit_char(rows)
        else:
            self._fit_centroid(rows)
        self.metadata_["feature_count"] = len(self.feature_names_)
        self.fitted_ = True
        return self

    def _fit_char(self, rows):
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
            min_df=1, sublinear_tf=True, lowercase=False,
        )
        by_work = defaultdict(list)
        for row in rows:
            by_work[row["work_id"]].append(row["text_bare"])
        # IDF documents represent canonical works, not correlated text chunks.
        self.vectorizer_.fit([" ".join(by_work[work]) for work in self.training_work_ids_])
        x = self.vectorizer_.transform([row["text_bare"] for row in rows])
        self.classifier_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                              random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                self.classifier_.fit(x, [row["author"] for row in rows],
                                     sample_weight=self.sample_weight_)
            except ConvergenceWarning as exc:
                raise ValueError("fixed character model did not converge") from exc
        self.authors_ = list(self.classifier_.classes_)
        self.feature_names_ = [f"char:{gram}" for gram in self.vectorizer_.get_feature_names_out()]
        self.metadata_.update({
            "score_type": "logistic regression decision score, not probability",
            "parameters": {"analyzer": "char_wb", "ngram_range": [3, 5],
                           "max_features": 30000, "min_df": 1, "sublinear_tf": True,
                           "lowercase": False, "idf_document_unit": "canonical training work",
                           "classifier": "LogisticRegression", "solver": "lbfgs",
                           "C": 1.0, "max_iter": 2000},
        })

    def _fit_centroid(self, rows):
        self.extractor_ = LexicalFeatureExtractor(svd_dims=0, seed=self.seed).fit(rows)
        names = self.extractor_.feature_names_
        self.static_mask_ = np.array([
            name.startswith("fw:") if self.kind == "function_centroid" else name != "misc:log_tokens"
            for name in names
        ])
        self.feature_names_ = [name for name, keep in zip(names, self.static_mask_) if keep]
        self.function_vocabulary_ = set(function_words(self.language_))
        self.bigram_vocabulary_ = {}
        if self.kind == "morph_centroid":
            counts = defaultdict(float)
            for row, weight in zip(rows, self.sample_weight_):
                for pair, rate in _bigram_rates(row["text_bare"], self.function_vocabulary_).items():
                    counts[pair] += weight * rate
            pairs = sorted(counts, key=lambda pair: (-counts[pair], pair))[:_BIGRAM_LIMIT]
            self.bigram_vocabulary_ = {pair: index for index, pair in enumerate(pairs)}
            self.feature_names_ += [f"fwbigram:{first}|{second}" for first, second in pairs]
        x = self._centroid_features(rows)
        self.scaler_ = StandardScaler().fit(x, sample_weight=self.sample_weight_)
        x = self.scaler_.transform(x)
        self.centroids_ = np.array([
            np.average(x[[row["author"] == author for row in rows]], axis=0,
                       weights=self.sample_weight_[[row["author"] == author for row in rows]])
            for author in self.authors_
        ])
        self.metadata_.update({
            "score_type": "cosine similarity, not probability",
            "parameters": {"svd_dims": 0, "standardization": "training-only weighted mean and variance",
                           "excluded_features": ["misc:log_tokens"],
                           "function_word_bigram_limit": _BIGRAM_LIMIT if self.kind == "morph_centroid" else 0},
            "feature_note": ("Predefined function/common words include content verbs; not topic-free. "
                             "Suffixes are crude word endings; adjacent common-word bigrams are a word-order proxy, "
                             "not validated morphology, syntax, or POS tags."),
        })

    def _centroid_features(self, rows):
        static = self.extractor_.transform(rows)[:, self.static_mask_]
        if not self.bigram_vocabulary_:
            return static
        bigrams = np.zeros((len(rows), len(self.bigram_vocabulary_)))
        for index, row in enumerate(rows):
            for pair, rate in _bigram_rates(row["text_bare"], self.function_vocabulary_).items():
                column = self.bigram_vocabulary_.get(pair)
                if column is not None:
                    bigrams[index, column] = rate
        return np.hstack([static, bigrams])

    def score(self, rows):
        """Return finite rows-by-candidates scores without adapting any fitted state."""
        if not self.fitted_:
            raise ValueError("fit the author model before scoring")
        rows = _validate_rows(rows, self.language_)
        if not rows:
            return np.empty((0, len(self.authors_)))
        if self.kind == "char_linear":
            x = self.vectorizer_.transform([row["text_bare"] for row in rows])
            scores = self.classifier_.decision_function(x)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
        else:
            x = self.scaler_.transform(self._centroid_features(rows))
            scores = cosine_similarity(x, self.centroids_)
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(rows), len(self.authors_)) or not np.isfinite(scores).all():
            raise ValueError("model produced invalid candidate scores")
        return scores
