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
