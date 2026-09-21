"""An independent, whole-work English reference-author evaluation.

This is a *closed-set attribution* test, not evidence that unsupervised discovery
has recovered thirteen authors. Reference labels are used only in this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

EXPECTED_AUTHORS = (
    "Alexander Hamilton", "Arthur Conan Doyle", "Charles Dickens",
    "Charlotte Bronte", "G. K. Chesterton", "George Eliot", "H. G. Wells",
    "James Madison", "Jane Austen", "John Jay", "Mark Twain",
    "Robert Louis Stevenson", "Thomas Hardy",
)
MAX_PASSAGES_PER_BOOK = 20
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_UNCERTAIN = re.compile(
    r"\b(?:disputed|joint|unknown|anonymous|uncertain|unattributed|multiple|various)\b|"
    r"\s(?:and|or)\s|[&|;/]", re.IGNORECASE,
)


def _author(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or _UNCERTAIN.search(value):
        return None
    return value.strip()


def _natural(value: Any) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def _style_features(texts: list[str]) -> csr_matrix:
    """Surface statistics contain no source identifiers or reference labels."""
    rows = []
    for text in texts:
        words = _WORD.findall(text)
        n = max(1, len(words))
        sizes = np.asarray([len(word) for word in words] or [0], dtype=float)
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        rows.append([
            float(sizes.mean()), float(sizes.std()),
            sum(len(word) <= 3 for word in words) / n,
            sum(len(word) >= 8 for word in words) / n,
            sum(word.isupper() for word in words) / n,
            sum(word[:1].isupper() for word in words) / n,
            len(set(word.casefold() for word in words)) / n,
            n / sentences,
            *[text.count(symbol) / n for symbol in (",", ";", ":", "!", "?", "—", "(", '"')],
            len(re.findall(r"['’]", text)) / n,
        ])
    return csr_matrix(np.asarray(rows, dtype=float))


def _make_pipeline(seed: int) -> Pipeline:
    # Vocabulary, inverse document frequencies, and scaling are all fitted only
    # after the work split. Neither filenames, titles nor labels are text inputs.
    features = FeatureUnion([
        ("words", TfidfVectorizer(
            lowercase=True, max_features=800, sublinear_tf=True,
            token_pattern=r"(?u)\b[^\W\d_]+(?:['’][^\W\d_]+)*\b",
        )),
        ("characters", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 4), max_features=8000,
            lowercase=True, sublinear_tf=True,
        )),
        ("surface", Pipeline([
            ("counts", FunctionTransformer(_style_features, validate=False)),
            ("scale", StandardScaler(with_mean=False)),
        ])),
    ], transformer_weights={"words": 1.0, "characters": 0.75, "surface": 0.10})
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(
            class_weight="balanced", random_state=seed, dual="auto", max_iter=10000,
        )),
    ])


def _passages(rows: list[dict], target: int) -> list[dict]:
    """Build disjoint passages within chapters and sample across each work.

    The short final tail is joined to the preceding passage of its chapter.
    Chapters shorter than the minimum are omitted, never borrowed from another
    book. A verse longer than the target may span several passages.
    """
    chapters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        chapters[str(row.get("chapter", "1"))].append(row)
    minimum = min(200, max(1, target // 3))
    passages = []
    for chapter in sorted(chapters, key=_natural):
        chapter_rows = sorted(chapters[chapter], key=lambda row: (
            _natural(row.get("verse", "")), str(row.get("id", "")),
        ))
        text = "\n".join(str(row.get("text") or "") for row in chapter_rows)
        matches = list(_WORD.finditer(text))
        if len(matches) < minimum:
            continue
        starts = list(range(0, len(matches), target))
        if len(starts) > 1 and len(matches) - starts[-1] < minimum:
            starts.pop()
        for index, start in enumerate(starts):
            stop = starts[index + 1] if index + 1 < len(starts) else len(matches)
            char_start = 0 if start == 0 else matches[start].start()
            char_stop = matches[stop].start() if stop < len(matches) else len(text)
            passages.append({
                "chapter": chapter,
                "passage_index": index + 1,
                "text": text[char_start:char_stop].strip(),
                "tokens": stop - start,
            })
    return passages


def _sample_passages(passages: list[dict]) -> list[dict]:
    if len(passages) <= MAX_PASSAGES_PER_BOOK:
        return passages
    indices = np.linspace(0, len(passages) - 1, MAX_PASSAGES_PER_BOOK, dtype=int)
    return [passages[int(index)] for index in indices]


def _book_summary(book: dict) -> dict:
    return {key: book[key] for key in (
        "collection", "book", "book_title", "author", "verse_count",
        "available_passages", "sampled_passages",
    )}


def evaluate_english(verses: list[dict], *, passage_tokens: int = 1200, seed: int = 42) -> dict:
    """Evaluate reference authors on entirely held-out books.

    All labelled English books are eligible regardless of whether their authors
    are on the expected thirteen-author source list. An author needs two usable
    books; otherwise that author is reported but cannot be evaluated. A mixed,
    joint, disputed, or incompletely labelled book is excluded in its entirety.
    """
    if isinstance(passage_tokens, bool) or not isinstance(passage_tokens, int) or passage_tokens < 1:
        raise ValueError("passage_tokens must be a positive integer")
    english = [row for row in verses if str(row.get("language", "")).casefold() in {"eng", "en", "english"}]
    present = sorted({_author(row.get("reference_author")) for row in english} - {None})
    expected = set(EXPECTED_AUTHORS)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in english:
        grouped[(str(row.get("collection", "")), str(row.get("book", "")))].append(row)
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "kind": "held_out_work_closed_set_attribution",
        "expected_author_count": 13,
        "known_author_count": len(present),
        "evaluated_author_count": 0,
        "accuracy": None,
        "balanced_accuracy": None,
        "book_accuracy": None,
        "book_balanced_accuracy": None,
        "source_completeness": {
            "scope": "Reference-author coverage; this does not verify that every source work was loaded.",
            "expected_authors": list(EXPECTED_AUTHORS),
            "present_authors": present,
            "missing_authors": sorted(expected - set(present)),
            "unexpected_authors": sorted(set(present) - expected),
            "all_expected_authors_present": expected <= set(present),
            "all_expected_authors_evaluable": False,
        },
        "english_verse_count": len(english),
        "english_book_count": len(grouped),
        "excluded_books": [],
        "ineligible_authors": [],
        "per_author": [],
        "per_collection": [],
        "held_out_predictions": [],
        "book_predictions": [],
        "split": {
            "unit": "whole_book",
            "strategy": "Seeded per-author book shuffle; 20% of books rounded up, at least one test and one training book per author.",
            "seed": seed,
            "test_fraction": 0.20,
            "train_books": [],
            "test_books": [],
            "train_passages": 0,
            "test_passages": 0,
        },
        "sampling": {
            "target_passage_tokens": passage_tokens,
            "minimum_passage_tokens": min(200, max(1, passage_tokens // 3)),
            "maximum_passages_per_book": MAX_PASSAGES_PER_BOOK,
            "method": "Non-overlapping text passages within chapters; short tails merge with the preceding passage. At most 20 evenly spaced passages per book.",
            "available_passages": 0,
            "sampled_passages": 0,
        },
        "features": {
            "input": "Verse text only; book titles, source identifiers, and reference-author metadata are never features.",
            "pipeline": "Training-fitted word TF-IDF (800 terms), character 3–4-gram TF-IDF (8000 features), and scaled surface statistics; class-balanced linear SVM.",
            "fit_scope": "Training books only, including vocabulary, IDF weights, and surface-statistic scaling.",
        },
        "caveat": "This is closed-set attribution among reference authors with training books. It does not validate how many authors unsupervised discovery finds, prove historical authorship, or measure verse-level accuracy. Held-out books can differ in topic and genre; shared editorial conventions may still affect scores.",
    }
    by_author: dict[str, list[dict]] = defaultdict(list)
    for (collection, book_id), rows in sorted(grouped.items()):
        labels = {_author(row.get("reference_author")) for row in rows}
        summary = {"collection": collection, "book": book_id, "book_title": str(rows[0].get("book_title", book_id)), "verse_count": len(rows)}
        if None in labels or len(labels) != 1:
            result["excluded_books"].append({**summary, "reason": "missing, disputed, joint, or inconsistent reference authorship"})
            continue
        author = next(iter(labels))
        passages = _passages(rows, passage_tokens)
        if not passages:
            result["excluded_books"].append({**summary, "author": author, "reason": "no chapter with enough text for an evaluation passage"})
            continue
        sampled = _sample_passages(passages)
        record = {**summary, "author": author, "available_passages": len(passages), "sampled_passages": len(sampled), "passages": sampled}
        by_author[author].append(record)
    eligible = {author: books for author, books in by_author.items() if len(books) >= 2}
    result["evaluated_author_count"] = len(eligible)
    result["source_completeness"]["all_expected_authors_evaluable"] = expected <= set(eligible)
    result["ineligible_authors"] = [{"author": author, "usable_books": len(by_author.get(author, [])), "reason": "at least two usable books are required"} for author in present if author not in eligible]
    if len(eligible) < 2:
        result["reason"] = "At least two reference authors with at least two usable books each are required."
        return result

    rng = np.random.default_rng(seed)
    train, test = [], []
    for author, books in sorted(eligible.items()):
        order = rng.permutation(len(books))
        n_test = min(len(books) - 1, max(1, math.ceil(len(books) * 0.20)))
        test.extend(books[int(index)] for index in order[:n_test])
        train.extend(books[int(index)] for index in order[n_test:])
    result["split"]["train_books"] = [_book_summary(book) for book in train]
    result["split"]["test_books"] = [_book_summary(book) for book in test]
    train_text = [passage["text"] for book in train for passage in book["passages"]]
    train_labels = [book["author"] for book in train for passage in book["passages"]]
    test_text = [passage["text"] for book in test for passage in book["passages"]]
    test_labels = [book["author"] for book in test for passage in book["passages"]]
    result["split"].update(train_passages=len(train_text), test_passages=len(test_text))
    result["sampling"].update(
        available_passages=sum(book["available_passages"] for book in train + test),
        sampled_passages=len(train_text) + len(test_text),
    )
    model = _make_pipeline(seed)
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text).tolist()
    result.update(
        status="evaluated",
        accuracy=float(accuracy_score(test_labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
    )
    offset = 0
    for book in test:
        book_predictions = predictions[offset:offset + len(book["passages"])]
        offset += len(book_predictions)
        votes = Counter(book_predictions)
        # Ties use a fixed lexicographic rule, never the known author.
        predicted = min(votes, key=lambda author: (-votes[author], author))
        result["book_predictions"].append({
            **_book_summary(book), "predicted_author": predicted,
            "correct": predicted == book["author"], "passage_votes": dict(sorted(votes.items())),
        })
        for passage, prediction in zip(book["passages"], book_predictions):
            result["held_out_predictions"].append({
                "collection": book["collection"], "book": book["book"],
                "chapter": passage["chapter"], "passage_index": passage["passage_index"],
                "tokens": passage["tokens"], "actual_author": book["author"],
                "predicted_author": prediction, "correct": prediction == book["author"],
            })
    book_truth = [item["author"] for item in result["book_predictions"]]
    book_predicted = [item["predicted_author"] for item in result["book_predictions"]]
    result["book_accuracy"] = float(accuracy_score(book_truth, book_predicted))
    result["book_balanced_accuracy"] = float(balanced_accuracy_score(book_truth, book_predicted))
    for author in sorted(eligible):
        rows = [row for row in result["held_out_predictions"] if row["actual_author"] == author]
        correct = sum(row["correct"] for row in rows)
        result["per_author"].append({
            "author": author, "train_books": sum(book["author"] == author for book in train),
            "test_books": sum(book["author"] == author for book in test),
            "test_passages": len(rows), "correct_passages": correct, "accuracy": correct / len(rows),
        })
    for collection in sorted({book["collection"] for book in test}):
        rows = [row for row in result["held_out_predictions"] if row["collection"] == collection]
        correct = sum(row["correct"] for row in rows)
        # Compute within-collection macro recall directly to keep predictions
        # outside that collection in the denominator without metric warnings.
        authors = {row["actual_author"] for row in rows}
        recalls = [sum(row["correct"] for row in rows if row["actual_author"] == author) / sum(row["actual_author"] == author for row in rows) for author in authors]
        result["per_collection"].append({
            "collection": collection, "test_passages": len(rows), "correct_passages": correct,
            "accuracy": correct / len(rows), "balanced_accuracy": float(np.mean(recalls)),
            "train_books": sum(book["collection"] == collection for book in train),
            "test_books": sum(book["collection"] == collection for book in test),
            "reference_authors": len(authors),
        })
    return result
