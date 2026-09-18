"""Raw-token pooling must not invent continuity or lose source provenance."""
from __future__ import annotations

import copy
import json
import re

import pytest

from stylometry.continuity import annotate_continuity, contiguous_runs
from stylometry.passages import build_passages


def verse(position, count=250, **changes):
    return {"id": f"source:W.1.{position}", "language": "grc", "work": "W",
            "witness": "S", "source": "reference", "chapter": "1", "verse": str(position),
            "order": position, "ref": f"W 1:{position}", "group": "author-a", "copyist": None,
            "has_gap": False, "supplied_frac": 0.0, "duplicate_of": None,
            "text_bare": " ".join(f"v{position}t{i}" for i in range(count)), **changes}


def token_exclusions(result):
    return [r for r in result["exclusions"] if r["reason"] != "insufficient_passages"]


@pytest.mark.parametrize("length", [500, 1000, 2000])
def test_pooling_conserves_tokens_without_overlap_and_maps_source_offsets(length):
    records = [verse(i, length // 2 + 17) for i in range(1, 8)]
    original = copy.deepcopy(records)
    result = build_passages(records, length)
    expected = [word for row in records for word in row["text_bare"].split()]
    kept = [word for row in result["passages"] for word in row["text_bare"].split()]
    assert len(kept) == 3 * length
    assert kept == expected[:len(kept)]
    lookup = {row["id"]: row["text_bare"].split() for row in records}
    all_spans = []
    for row in result["passages"] + token_exclusions(result):
        recovered = [word for span in row["source_spans"]
                     for word in lookup[span["verse_id"]][span["token_start"]:span["token_end"]]]
        assert len(recovered) == row["n_tokens"]
        if "text_bare" in row:
            assert row["text_bare"].split() == recovered
            assert row["text"] == row["text_bare"]
            assert re.fullmatch(r"passage-[0-9a-f]{64}", row["id"])
        all_spans.extend((span["verse_id"], i) for span in row["source_spans"]
                         for i in range(span["token_start"], span["token_end"]))
    assert len(all_spans) == len(set(all_spans)) == len(expected)
    assert result["settings"]["retained_tokens"] + result["settings"]["excluded_tokens"] == len(expected)
    assert token_exclusions(result)[0]["reason"] == "short_tail"
    assert records == original
    assert result == build_passages(records, length)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"work": "other"}, {"chapter": "2"}, {"witness": "other"}, {"source": "other"},
    {"author": "separately-labelled"}, {"group": "other"}, {"copyist": "other"},
    {"duplicate_of": "W"}, {"document_id": "other"}, {"fragment": "other"},
    {"language": "hbo"}, {"order": 99}, {"verse": "99"},
])
def test_boundaries_do_not_pool_independently_observed_or_labelled_text(change):
    result = build_passages([verse(1), verse(2, **change)], 500)
    assert not result["passages"]
    assert [row["n_tokens"] for row in token_exclusions(result)] == [250, 250]
    assert all(row["reason"] == "insufficient_tokens" for row in token_exclusions(result))


@pytest.mark.parametrize("change,reason", [
    ({"has_gap": True}, "gap"), ({"supplied_frac": .01}, "supplied_text"),
    ({"text_bare": ""}, "empty_text"),
])
def test_damaged_supplied_and_empty_verses_are_excluded_and_break_both_sides(change, reason):
    result = build_passages([verse(1), verse(2, 1000, **change), verse(3)], 500)
    assert not result["passages"]
    assert [row["reason"] for row in token_exclusions(result)] == ["insufficient_tokens", reason, "insufficient_tokens"]
    assert result["settings"]["excluded_tokens"] == result["settings"]["input_tokens"]


def test_filtered_holes_survive_rebuilt_positions_and_passage_output():
    records = annotate_continuity([verse(i, 500, verse="?") for i in range(1, 4)])
    retained = [records[0], {**records[2], "order": 2}]
    result = build_passages(retained, 500)
    assert len(result["passages"]) == 2
    assert contiguous_runs(result["passages"]) == [[0], [1]]
    assert contiguous_runs(annotate_continuity(result["passages"])) == [[0], [1]]


def test_huge_verse_splits_exactly_with_offsets_and_short_tail():
    record = verse(1, 2200)
    result = build_passages([record], 1000)
    assert [row["source_spans"] for row in result["passages"]] == [
        [{"verse_id": record["id"], "token_start": 0, "token_end": 1000}],
        [{"verse_id": record["id"], "token_start": 1000, "token_end": 2000}],
    ]
    assert token_exclusions(result)[0]["source_spans"] == [
        {"verse_id": record["id"], "token_start": 2000, "token_end": 2200}]
    assert len({row["id"] for row in result["passages"]}) == 2
    assert contiguous_runs(result["passages"]) == [[0, 1]]
    assert result["exclusions"][-1]["reason"] == "insufficient_passages"
    assert result["exclusions"][-1]["n_passages"] == 2


def test_source_order_and_unknown_positions_do_not_invent_continuity():
    repeated = [verse(1), verse(1, id="other", work="other"), verse(2)]
    unknown = [verse(i, verse="?", order=None) for i in (1, 2)]
    for records in (repeated, unknown, [verse(2), verse(1)]):
        result = build_passages(records, 500)
        assert not result["passages"]
        assert len(token_exclusions(result)) == len(records)


def test_duplicate_ids_are_rejected_before_pooling():
    with pytest.raises(ValueError, match="duplicate source verse id"):
        build_passages([verse(1), verse(2, id=verse(1)["id"])], 500)


@pytest.mark.parametrize("tokens", [True, 0, -1, 250, 500.0, "1000"])
def test_only_declared_integer_window_sizes_are_allowed(tokens):
    with pytest.raises(ValueError, match="500, 1000, or 2000"):
        build_passages([verse(1)], tokens)


@pytest.mark.parametrize("change", [
    {"id": ""}, {"work": ""}, {"text_bare": None}, {"supplied_frac": None},
    {"supplied_frac": float("nan")}, {"supplied_frac": -1}, {"supplied_frac": 2},
])
def test_invalid_source_metadata_is_not_silently_reinterpreted(change):
    with pytest.raises(ValueError):
        build_passages([verse(1, **change)], 500)


def test_empty_input_and_entire_short_runs_are_explicit():
    empty = build_passages([], 500)
    assert empty["passages"] == empty["exclusions"] == []
    assert empty["settings"]["input_tokens"] == 0
    short = build_passages([verse(1, 499)], 500)
    assert short["passages"] == []
    assert short["exclusions"][0]["reason"] == "insufficient_tokens"
    assert short["exclusions"][0]["n_tokens"] == 499
    assert short["exclusions"][1]["reason"] == "insufficient_passages"


def test_passage_records_remain_compatible_with_actual_clustering(tmp_path):
    from stylometry.cluster import run

    record = verse(1, 1500, text_bare="και ο λογοσ " * 500)
    passages = build_passages([record], 500)["passages"]
    summary = run(passages, None, tmp_path, alpha=0)
    assert summary["n_verses"] == 3
    assert summary["alpha"] == 0
    assert summary["k_used"] == 1


# --- bridging chapter divisions ------------------------------------------------------------------

def _chapter_verse(chapter, verse, order, words=("και", "ο", "λογοσ"), **changes):
    text = " ".join(words)
    return {"id": f"grc:W.{chapter}.{verse}", "language": "grc", "work": "W", "work_title": "W",
            "source": "codex", "witness": "S", "collection": "NT", "canon": "NT", "group": "W",
            "copyist": None, "chapter": str(chapter), "verse": str(verse), "order": order,
            "ref": f"W {chapter}:{verse}", "text": text, "text_bare": text, "n_tokens": len(words),
            "has_gap": False, "supplied_frac": 0.0, "duplicate_of": None, **changes}


def test_a_chapter_division_breaks_a_run_unless_bridging_is_asked_for():
    """Chapter numbers are a 13th-century editorial layer, not a feature of any manuscript here."""
    from stylometry.continuity import consecutive

    end, start = _chapter_verse(1, 30, 30), _chapter_verse(2, 1, 31)
    assert consecutive(end, start) is False
    assert consecutive(end, start, bridge_chapters=True) is True


@pytest.mark.parametrize("end,start,why", [
    (_chapter_verse(1, 30, 30), _chapter_verse(3, 1, 31), "a skipped chapter is not adjacency"),
    (_chapter_verse(1, 30, 30), _chapter_verse(2, 4, 31), "the next chapter must start at its first verse"),
    (_chapter_verse(1, 30, 30), _chapter_verse(2, 1, 40), "source order must advance by exactly one"),
    (_chapter_verse(1, 30, 30, has_gap=True), _chapter_verse(2, 1, 31), "a gap is never bridged"),
    (_chapter_verse(1, 30, 30), _chapter_verse(2, 1, 31, work="OTHER"), "a different work is not bridged"),
])
def test_bridging_never_invents_adjacency_the_corpus_does_not_record(end, start, why):
    from stylometry.continuity import consecutive

    assert consecutive(end, start, bridge_chapters=True) is False, why


def test_bridging_recovers_text_that_chapter_length_would_otherwise_discard():
    """Chapters shorter than a passage are dropped whole; that cost 93% of Codex Sinaiticus."""
    verses = []
    order = 0
    for chapter in range(1, 5):
        for verse in range(1, 51):
            order += 1
            verses.append(_chapter_verse(chapter, verse, order, words=("και", "ο", "λογοσ", "εν")))
    # 800 tokens in all, in four 200-token chapters: no single chapter can fill a 500-token passage.
    without = build_passages(verses, tokens=500)
    with_bridge = build_passages(verses, tokens=500, bridge_chapters=True)
    assert without["passages"] == []
    assert len(with_bridge["passages"]) == 1
    assert with_bridge["settings"]["bridge_chapters"] is True
    assert "chapter divisions bridged" in with_bridge["settings"]["boundaries"]


# --- pooling verse profiles into a passage ---------------------------------------------------------

def test_pooled_profile_averages_the_scales_and_takes_the_commonest_category():
    from stylometry.ai.profile import CATEGORICAL_DIMS, NUMERIC_DIMS
    from stylometry.passages import pool_profiles

    def profile(ident, value, category, tag):
        return {"id": ident, **{d: value for d in NUMERIC_DIMS},
                **{d: vs[0] for d, vs in CATEGORICAL_DIMS.items()},
                "discourse_mode": category, "style_tags": [tag],
                "distinctive_phrases": [], "signature": "s"}

    passage = {"id": "p1", "source_verse_ids": ["a", "b", "c", "d"]}
    profiles = {"a": profile("a", 0.2, "narrative", "x"), "b": profile("b", 0.4, "narrative", "x"),
                "c": profile("c", 0.6, "poetry", "y"), "d": profile("d", 0.8, "narrative", "x")}
    pooled = pool_profiles([passage], profiles)["p1"]
    assert pooled["register"] == pytest.approx(0.5)
    assert pooled["discourse_mode"] == "narrative", "three of four verses narrate"
    assert pooled["style_tags"][0] == "x"


def test_a_passage_mostly_lacking_profiles_is_omitted_rather_than_averaged_from_a_fragment():
    from stylometry.ai.profile import CATEGORICAL_DIMS, NUMERIC_DIMS
    from stylometry.passages import pool_profiles

    one = {"id": "a", **{d: 0.5 for d in NUMERIC_DIMS},
           **{d: vs[0] for d, vs in CATEGORICAL_DIMS.items()},
           "style_tags": ["x"], "distinctive_phrases": [], "signature": "s"}
    passage = {"id": "p1", "source_verse_ids": ["a", "b", "c", "d", "e"]}
    assert pool_profiles([passage], {"a": one}) == {}
    assert pool_profiles([passage], {"a": one}, min_coverage=0.2) != {}
