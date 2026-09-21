from copy import deepcopy

import pytest

from stylometry.benchmark import EXPECTED_AUTHORS, _make_pipeline, evaluate_english


def corpus(authors=("Jane Austen", "Charles Dickens", "Mark Twain"), books=3):
    rows = []
    styles = (
        "She would have thought it quite agreeable, if he had only known; she might not have said so.",
        "The long street was dark! A man rushed past the little door: a strange, cold, miserable night.",
        "Well, I reckon we ain't going there. We took our boat down that river and didn't look back.",
    )
    for author_index, author in enumerate(authors):
        for book_index in range(books):
            for verse_index in range(4):
                rows.append({
                    "id": f"{author_index}:{book_index}:{verse_index}", "language": "eng",
                    "collection": f"Collection {author_index % 2}",
                    "book": f"work-{author_index}-{book_index}",
                    "book_title": f"TITLE_SECRET_{author_index}_{book_index}",
                    "chapter": "1", "verse": str(verse_index + 1),
                    "text": (styles[author_index % 3] + f" Here comes scene {book_index}. ") * 4,
                    "reference_author": author,
                })
    return rows


def test_split_holds_out_entire_books_and_each_author_has_training_and_test():
    verses = corpus()
    original = deepcopy(verses)
    result = evaluate_english(verses, passage_tokens=60)
    assert result["status"] == "evaluated"
    train = {(row["collection"], row["book"]) for row in result["split"]["train_books"]}
    test = {(row["collection"], row["book"]) for row in result["split"]["test_books"]}
    assert not train & test
    assert len(train | test) == 9
    assert all(row["train_books"] >= 1 and row["test_books"] >= 1 for row in result["per_author"])
    assert all((row["collection"], row["book"]) in test for row in result["held_out_predictions"])
    assert result["split"] == evaluate_english(list(reversed(verses)), passage_tokens=60)["split"]
    assert verses == original
    assert sum(row["test_passages"] for row in result["per_author"]) == result["split"]["test_passages"]
    assert sum(row["correct_passages"] for row in result["per_collection"]) / result["split"]["test_passages"] == result["accuracy"]


def test_unknown_disputed_joint_and_mixed_books_are_never_training_truth():
    verses = corpus()
    for index, label in enumerate((None, "disputed", "joint", "Hamilton and Madison", "unknown")):
        verses.append({**verses[0], "id": f"untrusted-{index}", "book": f"untrusted-{index}", "reference_author": label})
    verses.append({**verses[0], "id": "mixed-label", "reference_author": "Mark Twain"})
    result = evaluate_english(verses, passage_tokens=60)
    assert len(result["excluded_books"]) == 6
    assert result["known_author_count"] == 3
    split_books = result["split"]["train_books"] + result["split"]["test_books"]
    assert not any(row["book"].startswith("untrusted") for row in split_books)
    assert not any(row["book"] == "work-0-0" for row in split_books)


def test_requires_distinct_training_and_test_books_for_two_authors():
    result = evaluate_english(corpus(books=1), passage_tokens=60)
    assert result["status"] == "insufficient_data"
    assert result["known_author_count"] == 3
    assert result["evaluated_author_count"] == 0
    assert result["accuracy"] is None
    assert result["split"]["train_books"] == result["split"]["test_books"] == []
    assert len(result["ineligible_authors"]) == 3
    assert evaluate_english([], passage_tokens=60)["status"] == "insufficient_data"


def test_expected_thirteen_is_only_source_coverage_not_forced_class_count():
    result = evaluate_english(corpus(), passage_tokens=60)
    assert result["expected_author_count"] == len(EXPECTED_AUTHORS) == 13
    assert result["known_author_count"] == result["evaluated_author_count"] == 3
    assert not result["source_completeness"]["all_expected_authors_present"]
    assert len(result["source_completeness"]["missing_authors"]) == 10
    assert "does not validate" in result["caveat"]
    foreign = [{**row, "language": "ara"} for row in corpus()]
    assert evaluate_english(foreign, passage_tokens=60)["known_author_count"] == 0


def test_fit_sees_only_training_text_and_features_never_receive_metadata(monkeypatch):
    import stylometry.benchmark as benchmark

    seen = {}
    real_factory = benchmark._make_pipeline

    class AuditedModel:
        def __init__(self, seed):
            self.model = real_factory(seed)

        def fit(self, texts, labels):
            seen["train"] = list(texts)
            seen["labels"] = list(labels)
            self.model.fit(texts, labels)
            seen["vocabulary"] = self.model.named_steps["features"].transformer_list[0][1].vocabulary_
            return self

        def predict(self, texts):
            seen["test"] = list(texts)
            return self.model.predict(texts)

    monkeypatch.setattr(benchmark, "_make_pipeline", AuditedModel)
    verses = corpus()
    # Unique alphabetic book markers make accidental vocabulary leakage visible.
    markers = {f"work-{a}-{b}": f"exclusive{chr(97 + a)}{chr(97 + b)}" for a in range(3) for b in range(3)}
    for row in verses:
        row["text"] = (markers[row["book"]] + " ") * 20 + row["text"]
    result = evaluate_english(verses, passage_tokens=60)
    train = {row["book"] for row in result["split"]["train_books"]}
    test = {row["book"] for row in result["split"]["test_books"]}
    assert all(markers[book] in seen["vocabulary"] for book in train)
    assert all(markers[book] not in seen["vocabulary"] for book in test)
    assert not any("TITLE_SECRET" in text for text in seen["train"] + seen["test"])
    assert not any(author in text for author in EXPECTED_AUTHORS for text in seen["train"] + seen["test"])


def test_surface_scaler_and_vocabulary_are_fitted_by_training_call_only():
    model = _make_pipeline(42)
    model.fit(["Little words, little words.", "Magnificent expression; unconventional imagination!"], ["A", "B"])
    features = model.named_steps["features"]
    vocabulary = dict(features.transformer_list[0][1].vocabulary_)
    scaler = features.transformer_list[2][1].named_steps["scale"]
    observed = scaler.n_samples_seen_
    model.predict(["testonlytoken testonlytoken " * 20])
    assert features.transformer_list[0][1].vocabulary_ == vocabulary
    assert "testonlytoken" not in vocabulary
    assert scaler.n_samples_seen_ == observed == 2


@pytest.mark.parametrize("value", [0, -10, True, 1.5])
def test_rejects_invalid_passage_size(value):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_english([], passage_tokens=value)
