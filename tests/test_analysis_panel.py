"""The feature panel, the divergence measures, the classifiers and change-point detection.

The controls that matter here are the ones that stop a method producing a confident wrong answer:
vowel points must not be counted as punctuation, a classifier must never see its own test fold, and
a change-point statistic must be read against texts known to have one author rather than against
chance.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from stylometry import measures, panel, rolling, supervised

GREEK = "και ειπεν ο θεοσ· γενηθητω φωσ. και εγενετο φωσ·"
HEBREW = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם"


# --- panel ------------------------------------------------------------------------------------------

def test_hebrew_vowel_points_are_not_counted_as_punctuation():
    """OSHB carries 58 combining marks per verse. Counted as punctuation they would swamp the block."""
    assert sum(panel.punctuation_counts(HEBREW).values()) == 0
    assert sum(panel.punctuation_counts(GREEK).values()) == 3


def test_sentences_split_on_the_scribal_high_point_and_the_editorial_stop():
    assert [len(s) for s in panel.sentences(GREEK)] == [4, 2, 3]
    assert len(panel.sentences("και ο λογοσ")) == 1, "a text with no punctuation is one sentence"
    assert panel.sentences("") == []


def test_word_and_character_ngrams_count_what_they_claim():
    assert panel.word_ngrams(["a", "b", "c"], 2) == {("a", "b"): 1, ("b", "c"): 1}
    assert panel.char_ngrams("ab", 2)["ab"] == 1


def test_collocations_rank_by_association_not_by_raw_frequency():
    """`the of` is frequent because both words are; a collocation is a pair beyond independence."""
    tokens = ("the of " * 30 + "rare pair " * 5).split()
    scored = dict(panel.collocations(tokens, window=1, top=20))
    assert scored[("rare", "pair")] > scored[("the", "of")]


def test_repeated_phrases_need_to_actually_repeat():
    tokens = ("και ειπεν κυριοσ " * 5 + "ουτωσ λεγει ποτε").split()
    phrases = dict(panel.repeated_phrases(tokens, n=3, min_count=3))
    assert ("και", "ειπεν", "κυριοσ") in phrases
    assert ("ουτωσ", "λεγει", "ποτε") not in phrases


def test_rhythm_features_are_finite_for_a_text_with_one_sentence():
    values, names = panel.length_features(GREEK.split(), "και ο λογοσ")
    assert len(values) == len(names) and np.isfinite(values).all()


# --- divergence -------------------------------------------------------------------------------------

def test_jensen_shannon_is_zero_for_identical_and_one_for_disjoint_distributions():
    assert measures.jensen_shannon([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert measures.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)


def test_jensen_shannon_is_symmetric_and_finite_when_a_word_is_missing_on_one_side():
    """Kullback-Leibler would be infinite here; that is the reason to prefer Jensen-Shannon."""
    a, b = [5, 0, 1], [1, 4, 1]
    assert measures.jensen_shannon(a, b) == pytest.approx(measures.jensen_shannon(b, a))
    assert 0 < measures.jensen_shannon(a, b) < 1


def test_mismatched_vocabularies_are_refused():
    with pytest.raises(ValueError, match="share a vocabulary"):
        measures.jensen_shannon([1, 2], [1, 2, 3])


# --- supervised -------------------------------------------------------------------------------------

def _two_classes(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (12, 6)), rng.normal(4, 1, (12, 6))])
    return X, ["A"] * 12 + ["B"] * 12, [f"w{i // 3}" for i in range(24)]


@pytest.mark.parametrize("kind", ["svm", "forest"])
def test_classifiers_separate_two_clear_populations(kind):
    X, y, g = _two_classes()
    out = supervised.classify(X, y, g, kind=kind)
    assert out["accuracy"] > 0.9 and out["n"] == 24


def test_a_work_never_appears_in_its_own_training_fold(monkeypatch):
    """Leakage across the fold boundary is what makes stylometry look better than it is."""
    X, y, g = _two_classes()
    seen = []
    real = supervised.StandardScaler.fit

    def spy(self, data, *args, **kwargs):
        seen.append(len(data))
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(supervised.StandardScaler, "fit", spy)
    supervised.classify(X, y, g)
    assert seen and all(n < len(X) for n in seen), "the scaler saw the whole corpus in some fold"


def test_an_unknown_classifier_is_refused():
    X, y, g = _two_classes()
    with pytest.raises(ValueError, match="unknown classifier"):
        supervised.classify(X, y, g, kind="magic")


def test_impostors_separates_a_genuine_pairing_from_a_false_one():
    X, y, g = _two_classes()
    genuine = supervised.impostors(X[0], X[1:12], X[12:], seed=1)
    false = supervised.impostors(X[0], X[12:], X[1:12], seed=1)
    assert genuine > 0.9 and false < 0.1, "verification must be able to answer no"


# --- change points ----------------------------------------------------------------------------------

def test_a_planted_seam_is_found_at_the_right_place():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(4, 1, (20, 5))])
    found = rolling.change_points(X, permutations=200)
    assert found["available"] and abs(found["split_index"] - 20) <= 1
    assert found["p_value"] < 0.05


def test_the_statistic_is_not_pulled_towards_the_ends_of_the_sequence():
    """Unscaled, the strongest split landed at the earliest permitted position in most real works."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 5))
    edge = rolling.separation(X, 3)
    middle = rolling.separation(X, 20)
    assert edge < middle * 3, "an extreme split must not dominate by having an unstable small side"


def test_too_few_windows_is_reported_rather_than_guessed_at():
    assert rolling.change_points(np.zeros((4, 3)))["available"] is False


def test_the_single_author_baseline_is_what_a_candidate_seam_is_measured_against():
    base = rolling.single_author_baseline([1.6, 1.7, 1.8, 1.88])
    assert base["max"] == pytest.approx(1.88)
    assert rolling.exceeds_baseline(2.5, base)["exceeds_all_single_author_works"] is True
    assert rolling.exceeds_baseline(1.7, base)["exceeds_all_single_author_works"] is False


def test_windows_overlap_so_the_series_is_continuous():
    tokens = [str(i) for i in range(2500)]
    pieces = rolling.windows(tokens, size=1000, step=500)
    assert [start for start, _ in pieces] == [0, 500, 1000, 1500]
    assert all(len(w) == 1000 for _, w in pieces)


def test_a_text_shorter_than_one_window_is_kept_whole_rather_than_dropped():
    assert len(rolling.windows(["a", "b"], size=1000)) == 1


# --- the calibration is per language ----------------------------------------------------------------

def test_a_baseline_from_one_language_is_not_applied_to_another(tmp_path):
    """A threshold taken from Greek prose says nothing about Hebrew, and nothing at all about Arabic."""
    from stylometry.analysis import load_change_point_baseline

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "grc": {"n_works": 13, "mean": 1.75, "p95": 1.86, "max": 1.88},
        "hbo": {"n_works": 10, "mean": 1.81, "p95": 1.91, "max": 1.92},
        "arb": {"available": False, "reason": "no single-author Arabic corpus"},
    }))
    assert load_change_point_baseline("grc", path)["max"] == 1.88
    assert load_change_point_baseline("hbo", path)["max"] == 1.92
    assert load_change_point_baseline("arb", path) is None, "an unavailable language must not borrow one"
    assert load_change_point_baseline("lat", path) is None


def test_the_checked_in_baseline_covers_every_language_the_corpus_holds():
    from stylometry.analysis import BASELINE_PATH, load_change_point_baseline

    stored = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert {"grc", "hbo", "arb"} <= set(stored)
    assert stored["arb"]["available"] is False and stored["arb"]["reason"].strip()
    for language in ("grc", "hbo"):
        base = load_change_point_baseline(language)
        assert base and base["n_works"] >= 8 and 0 < base["max"] < 10
        assert base["note_language"].strip(), "each reference must say what it is drawn from"


def test_a_language_without_a_reference_gets_no_verdict_from_another_language(tmp_path):
    """The loader being language-aware is not enough: the language has to reach it.

    It did not. The Qur'an was scored against the Greek threshold and reported a seam in sura 2 that
    the Arabic evidence cannot support, because the call site dropped the argument and took the
    default. Testing the loader alone missed it, so this drives the whole analysis.
    """
    from stylometry.analysis import analyse

    rng = np.random.default_rng(0)
    vocabulary = "في من علي عن مع حتي ان ما لا لم لن قد هو هي".split()
    docs, texts, labels, works = [], [], [], []
    for work in range(6):
        tokens = [vocabulary[i % len(vocabulary)] for i in range(12_000)]
        for piece in range(6):
            chunk = tokens[piece * 1000:(piece + 1) * 1000]
            docs.append(chunk)
            texts.append(" ".join(chunk))
            labels.append("meccan" if work % 2 else "medinan")
            works.append(f"Q{work}")
    result = analyse(docs, texts, labels, works, language="arb", permutations=20)
    for row in result["change_points"]:
        assert "exceeds_all_single_author_works" not in row, (
            "Arabic has no single-author reference; no row may carry a verdict borrowed from Greek")
    assert "No single-author reference exists for this language" in __import__(
        "stylometry.analysis", fromlist=["render"]).render(result)
