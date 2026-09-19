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
                    ref, text, tokens, xy, group, attribution = verse
                    assert text and tokens > 0
                    assert len(xy) == 2 * len(data["keys"]), "a verse is placed under every strategy"
                    assert len(group) == len(data["keys"]), "and grouped under every strategy"


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


def _cards(tmp_path, built):
    """The front page is assembled from what each language wrote beside its own page."""
    from stylometry.explorer_html import write

    for lang, data in built:
        write(data, tmp_path / lang)
    return built


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
    _write_explorer_index(tmp_path, _cards(tmp_path, [("grc", greek), ("hbo", hebrew)]))
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


# --- style-group counts at every level ----------------------------------------------------------------

def test_group_units_refuses_to_partition_too_few_units():
    """Two groups fitted to five documents is arithmetic, not evidence."""
    import numpy as np

    from stylometry.explorer import GROUP_MIN_UNITS, group_units

    rng = np.random.default_rng(0)
    result = group_units(rng.normal(size=(GROUP_MIN_UNITS - 1, 20)))
    assert result["k"] == 1 and result["reason"] == "too_few_units"
    assert result["labels"] == [1] * (GROUP_MIN_UNITS - 1)


def test_group_units_finds_a_real_split_and_declines_a_spurious_one():
    import numpy as np

    from stylometry.explorer import group_units

    rng = np.random.default_rng(0)
    far = np.vstack([rng.normal(0, .3, size=(20, 12)), rng.normal(9, .3, size=(20, 12))])
    split = group_units(far)
    assert split["k"] == 2, "two well-separated clouds are two groups"
    assert set(split["labels"][:20]) != set(split["labels"][20:]), "the split follows the clouds"
    assert split["min_ari"] >= 0.8

    noise = group_units(rng.normal(size=(40, 12)))
    assert noise["k"] == 1, "structureless noise must not be partitioned"
    assert noise["reason"] in {"no_supported_split", "unstable"}


def test_every_level_reports_how_many_groups_its_children_fall_into():
    from stylometry.explorer import build

    data = build(_explorer_verses(n_works=12, n_chapters=3, per_chapter=10),
                 "grc", progress=lambda m: None, workers=1)
    n_keys = len(data["keys"])

    for packed in (data["book_groups"], data["collection_groups"]):
        assert len(packed["gk"]) == n_keys and all(k >= 1 for k in packed["gk"])
    assert data["book_groups"]["gn"] == 12, "the headline counts every book of the language"

    for collection in data["tree"]:
        assert len(collection["gk"]) == n_keys
        assert collection["gn"] == len(collection["children"]), "a collection groups its books"
        for i, k in enumerate(collection["gk"]):
            assert 1 <= collection["g"][i] <= data["collection_groups"]["gk"][i]
            for work in collection["children"]:
                assert 1 <= work["g"][i] <= k, "a book's group is inside its collection's partition"
        for work in collection["children"]:
            assert work["gn"] == len(work["children"]), "a book groups its chapters"
            for chapter in work["children"]:
                assert chapter["gn"] == len(chapter["v"]), "a chapter groups its verses"
                for i, k in enumerate(chapter["gk"]):
                    assert 1 <= chapter["g"][i] <= work["gk"][i]
                    for verse in chapter["v"]:
                        assert 1 <= verse[4][i] <= k, "a verse's group is inside its chapter's"


def test_one_group_always_carries_the_reason_it_is_one():
    """A bare "1" reads as a finding. It is usually the absence of one, and has to say so."""
    from stylometry.explorer import GROUP_REASONS, build

    data = build(_explorer_verses(), "grc", progress=lambda m: None, workers=1)
    seen = set()
    for node in [data["book_groups"], data["collection_groups"]] + data["tree"]:
        for k, reason in zip(node["gk"], node["gr"]):
            assert (reason is None) == (k > 1), "a reason accompanies one group and only one group"
            if reason is not None:
                assert reason in GROUP_REASONS, reason
                seen.add(reason)
    assert seen, "this fixture is small enough that something must decline to split"


def test_grouping_in_parallel_matches_grouping_serially():
    """Fanning out over strategies is a speed change, not a result change."""
    from stylometry.explorer import build

    verses = _explorer_verses(n_works=10, n_chapters=2, per_chapter=8)
    serial = build(verses, "grc", progress=lambda m: None, workers=1)
    parallel = build(verses, "grc", progress=lambda m: None, workers=4)
    assert serial["book_groups"] == parallel["book_groups"]
    assert [c["gk"] for c in serial["tree"]] == [c["gk"] for c in parallel["tree"]]
    assert [w["g"] for c in serial["tree"] for w in c["children"]] == \
           [w["g"] for c in parallel["tree"] for w in c["children"]]


def test_the_pages_show_the_group_counts_and_never_colour_alone(tmp_path):
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=10, n_chapters=2, per_chapter=8),
                 "grc", progress=lambda m: None, workers=1)
    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    book = sorted((out / "works").glob("*.html"))[0].read_text(encoding="utf-8")

    for page in (overview, book):
        assert "groupSummary" in page and "style group" in page
        # The lighter hues fall below 3:1 against the page, so the group name is always written out.
        assert "${A(groupOf(child))}" in page or "${A(vGroupOf(v))}" in page
    assert "book_groups" in overview and "collection_groups" in overview
    assert '"gk"' in overview and '"g"' in overview
    assert "group_reasons" in overview, "the page can explain a count of one"


def test_a_split_of_units_too_short_to_attribute_says_so(tmp_path):
    """A stable split is not automatically an authorial one.

    Vocabulary richness over an eighteen-token verse takes few distinct values, so verses cluster
    cleanly and the stability gate passes. Section 1 of this project measured that units that short
    cannot support attribution, which is why the analysis pools into 1,000-token passages. The page
    has to carry that where it applies, or it asserts what the project has already disproved.
    """
    from stylometry.explorer import GROUP_RELIABLE_TOKENS, build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=10, n_chapters=2, per_chapter=12),
                 "grc", progress=lambda m: None, workers=1)
    assert data["reliable_tokens"] == GROUP_RELIABLE_TOKENS

    for collection in data["tree"]:
        for work in collection["children"]:
            for chapter in work["children"]:
                assert chapter["gtok"] > 0, "a partition records how long its units are"
                assert chapter["gtok"] < GROUP_RELIABLE_TOKENS, "verses in this fixture are short"

    page = (write(data, tmp_path / "grc") / "index.html").read_text(encoding="utf-8")
    assert "shortUnitWarning" in page
    assert "cannot support attribution" in page
    assert "reliable_tokens" in page


def test_groups_are_named_and_sized_the_way_the_rest_of_the_project_names_them(tmp_path):
    """A1, A2 ... largest first, with the split shown as a bar chart at every level.

    The names match what `cluster` produces and what the style-group dashboards print, so a group
    means the same thing in both views. The sizes matter as much as the count: "5 style groups" reads
    very differently once you can see that four of them hold one unit each.
    """
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=12, n_chapters=3, per_chapter=10),
                 "grc", progress=lambda m: None, workers=1)

    for packed in (data["book_groups"], data["collection_groups"]):
        for k, sizes in zip(packed["gk"], packed["gsz"]):
            assert len(sizes) == k, "one size per group"
            assert sum(sizes) == packed["gn"], "every unit is in exactly one group"
            assert sizes == sorted(sizes, reverse=True), "A1 is the largest group"

    for collection in data["tree"]:
        for i, sizes in enumerate(collection["gsz"]):
            assert sum(sizes) == len(collection["children"])
        for work in collection["children"]:
            for i, sizes in enumerate(work["gsz"]):
                assert sum(sizes) == len(work["children"])
            for chapter in work["children"]:
                for i, sizes in enumerate(chapter["gsz"]):
                    assert sum(sizes) == len(chapter["v"])

    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    book = sorted((out / "works").glob("*.html"))[0].read_text(encoding="utf-8")
    for page in (overview, book):
        assert "groupBars" in page, "a bar chart of the group sizes"
        assert "const A = i => 'A' + i" in page, "groups are named A1, A2 ..."
        assert '"gsz"' in page
    # The verse list carries the label on the text itself, not only in a side panel.
    assert "A(vGroupOf(v))" in book
    assert "groupEdge" in book and "groupEdge" in overview


def test_the_front_page_shows_how_each_language_divides(tmp_path):
    from stylometry.cli import _write_explorer_index
    from stylometry.explorer import build

    greek = build(_explorer_verses(n_works=12, n_chapters=2, per_chapter=9),
                  "grc", progress=lambda m: None, workers=1)
    _write_explorer_index(tmp_path, _cards(tmp_path, [("grc", greek)]))
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '"sizes"' in page and 'class="stack"' in page
    assert "A${i + 1}" in page


def test_a_book_carries_its_group_among_all_books_not_only_within_its_collection(tmp_path):
    """The two are different numbers and confusing them answers the wrong question.

    A book's group within its own collection cannot be compared across collections: every
    collection's labels start at A1, so suras would be matched against suras and the comparison
    could not track the collection whatever the data said. The language-wide partition is the one
    that answers it, so every book carries its place in that too, and every collection carries how
    its books spread across it.
    """
    from stylometry.explorer import build

    data = build(_explorer_verses(n_works=12, n_chapters=2, per_chapter=9),
                 "grc", progress=lambda m: None, workers=1)
    n_keys = len(data["keys"])
    seen = {i: [] for i in range(n_keys)}
    for collection in data["tree"]:
        assert len(collection["glsz"]) == n_keys
        for i, spread in enumerate(collection["glsz"]):
            assert len(spread) == data["book_groups"]["gk"][i], "one slot per language-wide group"
            assert sum(spread) == len(collection["children"]), "all of its books are placed"
        for work in collection["children"]:
            assert len(work["gl"]) == n_keys
            for i, g in enumerate(work["gl"]):
                assert 1 <= g <= data["book_groups"]["gk"][i]
                seen[i].append(g)
    for i, groups in seen.items():
        assert len(groups) == data["book_groups"]["gn"], "every book of the language, exactly once"
        for g in range(1, data["book_groups"]["gk"][i] + 1):
            assert groups.count(g) == data["book_groups"]["gsz"][i][g - 1], "and the counts agree"


def test_the_bar_chart_is_shown_at_every_level_and_verses_get_a_group_column(tmp_path):
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=12, n_chapters=2, per_chapter=9),
                 "grc", progress=lambda m: None, workers=1)
    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    book = sorted((out / "works").glob("*.html"))[0].read_text(encoding="utf-8")

    assert "Style groups across the language" in overview
    assert "Style groups within" in overview
    assert "acrossLanguage" in overview and '"glsz"' in overview
    assert "Style groups among the chapters" in book
    assert "Style groups among the verses" in book
    # The verse list is a table: reference, group, text.
    assert 'class="vhead"' in book and 'class="vrow"' in book and 'class="vgrp"' in book
    # A chart with one bar still renders, so the level is never simply blank: the only thing that
    # suppresses it is having nothing at all to draw.
    assert "function barChart" in book
    assert "g.k < 2" not in book.split("function groupBars")[1].split("}")[0]


def test_every_row_carries_its_own_bar_chart(tmp_path):
    """A list should be readable for how each entry divides, without selecting it first."""
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=12, n_chapters=3, per_chapter=10),
                 "grc", progress=lambda m: None, workers=1)
    out = write(data, tmp_path / "grc")
    overview = (out / "index.html").read_text(encoding="utf-8")
    book = sorted((out / "works").glob("*.html"))[0].read_text(encoding="utf-8")

    assert "function rowBars" in overview and "function ownBars" in overview
    # A collection shows both: its books among themselves, and its books across the whole language.
    assert "acrossLanguage(n) + ownBars(n, 'books')" in overview
    assert "ownBars(n, 'chapters')" in overview, "a book row shows how its chapters divide"
    assert "ownBars(c, 'verses')" in book, "a chapter row shows how its verses divide"
    assert ".rbars" in overview and ".rfill" in overview


def test_a_book_shows_its_group_across_the_language_even_when_its_collection_does_not_split(tmp_path):
    """The gap this closes: drill into a collection whose books do not divide among themselves and
    every row was blank, although each book does have a place in the language-wide grouping."""
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=12, n_chapters=2, per_chapter=9),
                 "grc", progress=lambda m: None, workers=1)
    overview = (write(data, tmp_path / "grc") / "index.html").read_text(encoding="utf-8")
    assert "const langOf" in overview and "langBadge(n, LANG)" in overview
    assert "const LANG = " in overview, "the badge names the language it is comparing across"
    assert "acrossChart(here, here.label)" in overview, "and the side panel charts the same thing"
    # The row is tinted by the language-wide group there, so the list reads across collections.
    assert "border-left-color:${G(langOf(n))}" in overview


def test_a_book_page_says_where_the_book_itself_sits(tmp_path):
    """Everything else on a book page is about its insides.

    Without this the page never states which group the book is in — BUKH01 is A3 among the 211
    Arabic books, and its own page had no way to say so, because its single chapter and seven verses
    are both too few to partition and every chart on the page came back with one bar.
    """
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=12, n_chapters=2, per_chapter=9),
                 "grc", progress=lambda m: None, workers=1)
    book = sorted((write(data, tmp_path / "grc") / "works").glob("*.html"))[0].read_text(encoding="utf-8")

    assert '"book_groups"' in book, "the language-wide grouping travels to the book page"
    assert '"collection_books"' in book, "and its collection's grouping"
    assert '"language_name"' in book
    assert "function placement" in book and 'id="head"' in book
    assert "mineAcross()" in book and "mineInCollection()" in book
    # The chart marks which row is this book's rather than leaving it to be counted off.
    assert "Where this book sits" in book
    assert "mark === i + 1 ? ' mine' : ''" in book


def test_the_placement_line_does_not_invent_a_group_where_there_is_none(tmp_path):
    """When nothing divides, it has to say that instead of naming a group."""
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(n_works=3, n_chapters=2, per_chapter=6),
                 "grc", progress=lambda m: None, workers=1)
    book = sorted((write(data, tmp_path / "grc") / "works").glob("*.html"))[0].read_text(encoding="utf-8")
    assert "do not divide under this strategy" in book
    assert "do not divide among themselves" in book


def test_rebuilding_one_language_does_not_drop_the_others_from_the_front_page(tmp_path):
    """The front page is shared, and it is the only route into any language.

    Rebuilding a single language used to rewrite it with that language alone, so the other two
    vanished from the site although their pages were still on disk and perfectly good.
    """
    from stylometry.cli import _write_explorer_index
    from stylometry.explorer import build

    greek = build(_explorer_verses(), "grc", progress=lambda m: None, workers=1)
    hebrew = build(_explorer_verses(), "hbo", progress=lambda m: None, workers=1)
    _write_explorer_index(tmp_path, _cards(tmp_path, [("grc", greek), ("hbo", hebrew)]))
    assert "Koine Greek" in (tmp_path / "index.html").read_text(encoding="utf-8")

    # Now rebuild Greek alone, exactly as `explore --language grc` does.
    _write_explorer_index(tmp_path, _cards(tmp_path, [("grc", greek)]))
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Koine Greek" in page
    assert "Biblical Hebrew" in page, "the language that was not rebuilt keeps its place"


def test_the_front_page_keeps_a_stable_language_order(tmp_path):
    from stylometry.cli import _write_explorer_index
    from stylometry.explorer import build

    built = [("arb", build(_explorer_verses(), "arb", progress=lambda m: None, workers=1)),
             ("grc", build(_explorer_verses(), "grc", progress=lambda m: None, workers=1))]
    _write_explorer_index(tmp_path, _cards(tmp_path, built))
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.index("Koine Greek") < page.index("Quranic Arabic"), "Greek, Hebrew, Arabic"


def test_a_collection_with_every_book_in_one_group_still_draws_its_chart(tmp_path):
    """"All nineteen of them in A2" is an answer, not an absence.

    Requiring two occupied groups left Greek's noncanonical collection with no chart at all under the
    default strategy, because its nineteen books are all in the same language-wide group.
    """
    from stylometry.explorer_html import CSS, JS_COMMON  # noqa: F401
    import stylometry.explorer_html as eh

    source = eh.JS_COMMON
    assert "if (!total) return ''" in source, "the only thing that suppresses a chart is no data"
    assert "sizes.filter(c => c > 0).length < 2" not in source


def test_verses_carry_who_is_speaking_where_the_source_says(tmp_path):
    from stylometry.explorer import build
    from stylometry.explorer_html import write

    data = build(_explorer_verses(), "grc", progress=lambda m: None, workers=1)
    assert data["verse_fields"] == ["ref", "text", "n_tokens", "xy", "group", "attribution"]
    for collection in data["tree"]:
        for work in collection["children"]:
            for chapter in work["children"]:
                for verse in chapter["v"]:
                    assert len(verse) == 6
                    assert verse[5] == "", "Greek carries no attribution"
    book = sorted((write(data, tmp_path / "grc") / "works").glob("*.html"))[0].read_text(encoding="utf-8")
    assert "speakerTag(v[5])" in book
    assert "quote of the Prophet" in book and "speech of God" in book
