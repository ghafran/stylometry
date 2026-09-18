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


def test_copies_of_one_text_are_held_out_together_whatever_manuscript_they_are_in():
    """With --witnesses the same chapter appears once per manuscript, under JOHN, JOHN@P66, JOHN@P75.

    Holding out by that code would leave P66's John to be judged by Sinaiticus's John - nearly the
    same words - and every method would score superbly while having learnt nothing about scribes.
    """
    from stylometry.cli import _holdout_works

    passages = [
        {"work": "JOHN", "duplicate_of": None},
        {"work": "JOHN@P66", "duplicate_of": "JOHN"},
        {"work": "JOHN@P75", "duplicate_of": "JOHN"},
        {"work": "MARK@05", "duplicate_of": "MARK"},
        {"work": "LUKE", "duplicate_of": None},
    ]
    assert _holdout_works(passages) == ["JOHN", "JOHN", "JOHN", "MARK", "LUKE"]


# --- the strategy registry --------------------------------------------------------------------------

def test_every_strategy_states_what_it_measures_and_where_it_stands():
    """A strategy nobody can interpret is worse than one that is missing."""
    from stylometry.strategies import REGISTRY, MEASURED, APPROXIMATED, PARTIAL, NOT_AUTHORIAL

    keys = [s.key for s in REGISTRY]
    assert len(keys) == len(set(keys)), "strategy keys must be unique"
    valid = {MEASURED, APPROXIMATED, PARTIAL, NOT_AUTHORIAL, "unavailable"}
    for s in REGISTRY:
        assert s.name.strip() and s.measures.strip(), s.key
        assert s.status in valid, f"{s.key} has status {s.status!r}"
        assert s.build is not None, s.key
    # Anything that is not a plain measurement of style must say why, in its own note.
    for s in REGISTRY:
        if s.status != MEASURED:
            assert s.note.strip(), f"{s.key} is {s.status} and must explain itself"


def test_a_strategy_without_data_for_a_language_is_marked_not_silently_empty():
    """Part-of-speech tags exist for Hebrew and nowhere else. A zero-width block must not read as a
    measurement of zero."""
    from stylometry.strategies import BY_KEY, describe

    assert BY_KEY["pos"].available_for("hbo") is True
    assert BY_KEY["pos"].available_for("grc") is False
    greek = {s["key"]: s for s in describe("grc")}
    assert greek["pos"]["available"] is False
    assert "Hebrew only" in BY_KEY["pos"].note


def test_choosing_strategies_returns_only_those_columns():
    from stylometry.strategies import build_matrix

    docs = [("και ο θεοσ ειπεν " * 30).split() for _ in range(4)]
    texts = [" ".join(d) for d in docs]
    one, names_one, spans = build_matrix(["function_words"], docs, texts, "grc")
    two, names_two, _ = build_matrix(["function_words", "richness"], docs, texts, "grc")
    assert all(n.startswith("fw:") for n in names_one)
    assert two.shape[1] > one.shape[1]
    assert spans["function_words"] == (0, one.shape[1])


def test_an_unknown_strategy_is_refused_rather_than_ignored():
    from stylometry.strategies import build_matrix

    docs = [["και"] * 40 for _ in range(4)]
    with pytest.raises(ValueError, match="unknown strategies"):
        build_matrix(["function_words", "telepathy"], docs, [" "] * 4, "grc")


def test_the_sweep_reports_each_strategy_alone_and_the_cost_of_removing_it():
    from stylometry.strategy_report import sweep

    a = [("και ο θεοσ ειπεν και εγενετο ουτωσ " * 20).split() for _ in range(6)]
    b = [("δε γαρ μεν ουν τε αλλα ωστε τοινυν " * 20).split() for _ in range(6)]
    docs = a + b
    result = sweep(docs, [" ".join(d) for d in docs], ["A"] * 6 + ["B"] * 6,
                   [f"w{i}" for i in range(12)], language="grc")
    assert result["combined"]["accuracy"] is not None
    assert result["solo"], "every available strategy is scored on its own"
    for key, row in result["leave_one_out"].items():
        assert "cost_of_removing" in row, key
    assert "pos" in result["unavailable"], "Greek has no tags, and the sweep must say so"


def test_the_explorer_page_carries_its_data_and_every_explanation():
    from stylometry.strategy_report import render_html, sweep

    docs = [("και ο θεοσ " * 40).split() for _ in range(6)]
    result = sweep(docs, [" ".join(d) for d in docs], ["A", "A", "A", "B", "B", "B"],
                   [f"w{i}" for i in range(6)], language="grc")
    html = render_html(result, "Explorer")
    assert "const DATA" in html and '"solo"' in html
    for s in result["strategies"]:
        assert s["name"] in html and s["measures"] in html, s["key"]


# --- the drill-down explorer -------------------------------------------------------------------------

def _explorer_verses(n_works=3, n_chapters=2, per_chapter=6):
    habits = ["και ο θεοσ ειπεν ουτωσ", "δε γαρ μεν ουν τε", "αλλα ωστε τοινυν ειτα"]
    out, order = [], 0
    for w in range(n_works):
        for c in range(1, n_chapters + 1):
            for v in range(1, per_chapter + 1):
                order += 1
                text = (habits[w % len(habits)] + " ") * 6
                out.append({
                    "id": f"grc:W{w}.{c}.{v}", "language": "grc", "work": f"W{w}",
                    "work_title": f"Work {w}", "collection": "NT" if w else "LXX", "canon": "NT",
                    "group": f"W{w}", "source": "x", "witness": "S", "copyist": None,
                    "chapter": str(c), "verse": str(v), "order": order, "ref": f"W{w} {c}:{v}",
                    "text": text.strip(), "text_bare": text.strip(),
                    "n_tokens": len(text.split()), "has_gap": False, "supplied_frac": 0.0,
                    "duplicate_of": None,
                })
    return out


def test_the_explorer_places_every_level_under_every_available_strategy():
    from stylometry.explorer import build

    data = build(_explorer_verses(), "grc", progress=lambda m: None)
    assert data["keys"] and data["strategies"]
    assert {c["label"] for c in data["tree"]} == {"LXX", "NT"}
    for collection in data["tree"]:
        assert len(collection["xy"]) == 2 * len(data["keys"]), "one xy pair per strategy, packed in order"
        for work in collection["children"]:
            assert len(work["xy"]) == 2 * len(data["keys"])
            for chapter in work["children"]:
                assert len(chapter["xy"]) == 2 * len(data["keys"])
                for verse in chapter["v"]:
                    ref, text, tokens, xy = verse
                    assert text and tokens > 0
                    assert len(xy) == 2 * len(data["keys"]), "a verse is placed under every strategy"


def test_a_strategy_without_data_is_absent_from_the_explorer_rather_than_flat():
    """Greek carries no part-of-speech tags. An all-zero block would plot every verse on one spot and
    read as a finding of perfect uniformity."""
    from stylometry.explorer import build

    data = build(_explorer_verses(), "grc", progress=lambda m: None)
    assert "pos" not in data["keys"]
    assert "pos" in data["unavailable"]


def test_verse_counts_add_up_through_the_hierarchy():
    from stylometry.explorer import build

    data = build(_explorer_verses(), "grc", progress=lambda m: None)
    assert sum(c["n_verses"] for c in data["tree"]) == data["n_verses"]
    for collection in data["tree"]:
        assert sum(w["n_verses"] for w in collection["children"]) == collection["n_verses"]
        for work in collection["children"]:
            assert sum(len(ch["v"]) for ch in work["children"]) == work["n_verses"]


def test_the_explorer_writes_an_overview_and_one_page_per_book(tmp_path):
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(), "grc", progress=lambda m: None)
    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "const DATA" in overview and "Koine Greek" in overview
    pages = sorted((out / "works").glob("*.html"))
    assert len(pages) == 3, "one page per book"
    book = pages[0].read_text(encoding="utf-8")
    for s in data["strategies"]:
        assert s["name"] in overview, s["key"]
    assert "back to chapters" in book and "back to collections" in overview
    assert "strategyBar" in book, "the strategy can be changed on a book page too"


def test_the_front_page_offers_the_strategy_before_anything_else(tmp_path):
    """The picker has to be on the page people land on.

    It worked on the language and book pages, but the two pages above them had none, so the control
    was two clicks from the entry point and read as absent. The front page now names every strategy
    and says which language is missing one, rather than falling back in silence.
    """
    from stylometry.cli import _write_explorer_index
    from stylometry.explorer import build

    greek = build(_explorer_verses(), "grc", progress=lambda m: None)
    hebrew = build(_explorer_verses(), "hbo", progress=lambda m: None)
    _write_explorer_index(tmp_path, [("grc", greek), ("hbo", hebrew)])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert "strategyBar" in page and 'id="bar"' in page
    union = {s["key"] for s in greek["strategies"]} | {s["key"] for s in hebrew["strategies"]}
    for key in union:
        assert f'"{key}"' in page, key
    # The chosen strategy travels with the link, so picking one here is not thrown away on arrival.
    assert "?s=${encodeURIComponent(strategy)}" in page
    if union - {s["key"] for s in greek["strategies"]}:
        assert "no ${strategy} data here" in page


def test_a_strategy_missing_for_a_language_is_said_out_loud(tmp_path):
    """Arriving with ?s=pos at a language without it must not look like the choice was honoured."""
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(), "grc", progress=lambda m: None)
    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "const unavailable" in overview
    assert "opened on the first strategy instead" in overview
