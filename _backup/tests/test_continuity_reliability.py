"""Regressions for passage gaps and lexical feature extraction edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from stylometry.continuity import annotate_continuity, consecutive, contiguous_runs
from stylometry.features import lexical_features


def verse(position: int, **overrides) -> dict:
    return {"id": f"grc:A.1.{position}", "language": "grc", "work": "A", "witness": "S",
            "source": "source", "chapter": "1", "verse": str(position), "order": position,
            "ref": f"A 1:{position}", "has_gap": False, "text_bare": "και ο λογοσ", **overrides}


def test_filtering_cannot_close_a_passage_gap():
    original = [verse(i) for i in range(1, 5)]
    annotated = annotate_continuity(original)
    assert contiguous_runs(annotated) == [[0, 1, 2, 3]]
    retained = [annotated[0], annotated[2], annotated[3]]
    assert contiguous_runs(retained) == [[0], [1, 2]]
    assert contiguous_runs(annotate_continuity(retained)) == [[0], [1, 2]]
    assert all("_continuity_index" not in item for item in original)


def test_annotation_preserves_filter_holes_without_verse_numbers():
    # Position fields may be rebuilt by later code; original indices remain.
    annotated = annotate_continuity([verse(i, verse="?") for i in range(1, 4)])
    retained = [annotated[0], dict(annotated[2], order=2)]
    assert not consecutive(*retained)


@pytest.mark.parametrize("change", [
    {"language": "hbo"}, {"work": "B"}, {"witness": "B"}, {"source": "other"},
    {"document_id": "fragment-2"}, {"fragment": "II"}, {"chapter": "2"},
    {"has_gap": True}, {"verse": "1000"}, {"order": 1000},
])
def test_boundaries_prevent_false_adjacency(change):
    assert not consecutive(verse(1), verse(2, **change))


def test_damaged_verses_are_isolated_on_both_sides():
    assert contiguous_runs([verse(1), verse(2, has_gap=True), verse(3)]) == [[0], [1], [2]]


def test_runs_preserve_order_without_joining_a_repeated_work():
    records = [verse(1), verse(1, work="B"), verse(2)]
    assert contiguous_runs(annotate_continuity(records)) == [[0], [1], [2]]
    assert contiguous_runs([verse(3), verse(2), verse(1)]) == [[0], [1], [2]]


def test_unknown_positions_do_not_imply_continuity():
    assert not consecutive(verse(1, verse="?", order=None), verse(2, verse="?", order=None))
    assert not consecutive(verse(1, verse="¹", order=None), verse(2, verse="²", order=None))
    assert contiguous_runs([]) == []


def test_chapter_runs_match_normal_corpus_units():
    records = [verse(i + 1, chapter=str(i // 10 + 1), verse=str(i % 10 + 1)) for i in range(30)]
    assert contiguous_runs(annotate_continuity(records)) == [list(range(10)), list(range(10, 20)), list(range(20, 30))]


def test_smoothing_does_not_bridge_missing_verses():
    from stylometry.cluster import smooth

    X = np.array([[0.0], [10.0]], dtype=np.float32)
    assert np.array_equal(smooth(X, [verse(1), verse(1000)]), X)
    # A genuine neighbouring pair still receives local averaging.
    assert np.allclose(smooth(X, [verse(1), verse(2)]).ravel(), [3.5, 6.5])


def test_smoothing_and_segment_reports_preserve_filtered_holes():
    from stylometry.cluster import segments, smooth

    annotated = annotate_continuity([verse(i) for i in range(1, 5)])
    retained = [annotated[0], annotated[2], annotated[3]]
    X = np.array([[0.0], [10.0], [12.0]], dtype=np.float32)
    assert np.allclose(smooth(X, retained).ravel(), [0.0, 10.7, 11.3])
    rows, majority = segments(retained, np.array(["A1", "A1", "A1"]))
    assert [(r["start_id"], r["end_id"], r["n"]) for r in rows] == [
        (annotated[0]["id"], annotated[0]["id"], 1), (annotated[2]["id"], annotated[3]["id"], 2),
    ]
    assert majority == {"A": "A1"}


def test_segment_reports_do_not_rejoin_interleaved_work():
    from stylometry.cluster import segments

    records = [verse(1), verse(1, work="B", id="grc:B.1.1"), verse(2)]
    rows, _ = segments(records, np.array(["A1", "A1", "A1"]))
    assert len(rows) == 3
    assert all(row["n"] == 1 for row in rows)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 20])
@pytest.mark.parametrize("dims", [0, 1, 40])
def test_small_constant_corpus_has_finite_correctly_named_features(n, dims):
    X, names = lexical_features([verse(i) for i in range(n)], svd_dims=dims)
    assert X.shape == (n, len(names))
    assert np.isfinite(X).all()
    assert len({name for name in names}) == len(names)
    assert len([name for name in names if name.startswith("cng:")]) <= dims
    assert np.allclose(X, X[0])


def test_all_pruned_character_ngrams_fall_back_to_rare_features():
    records = [verse(i, text_bare=letter * 4) for i, letter in enumerate("αβγδε")]
    X, names = lexical_features(records)
    assert X.shape[1] == len(names)
    assert any(name.startswith("cng:") for name in names)
    assert np.isfinite(X).all()


def test_some_empty_verses_do_not_break_feature_extraction():
    X, names = lexical_features([verse(1, text_bare=""), verse(2)])
    assert X.shape == (2, len(names))
    assert np.isfinite(X).all()


@pytest.mark.parametrize("records", [[], [verse(1, text_bare="")], [verse(1, text_bare=" \n ")]])
def test_empty_corpus_is_rejected_with_clear_error(records):
    with pytest.raises(ValueError, match="at least one verse|nonempty text"):
        lexical_features(records)


def test_svd_dimensions_must_be_nonnegative_integers():
    for dims in (-1, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            lexical_features([verse(1)], svd_dims=dims)
