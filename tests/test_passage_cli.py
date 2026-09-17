"""The passage command must preserve provenance and stay lexical and unsmoothed."""
from __future__ import annotations

import json

import pytest

from stylometry import ai_profile, cli, cluster


def source(position, count=250, *, work="W", language="grc", collection="NT", **changes):
    vocabulary = {"grc": "και ο λογοσ εν τω", "hbo": "כי הוא אמר אל בית", "arb": "في من على ان هو"}[language].split()
    text = " ".join(vocabulary[i % len(vocabulary)] for i in range(count))
    return {
        "id": f"{language}:{collection}:{work}.1.{position}", "language": language,
        "work": work, "work_title": work, "source": "reference", "witness": "edition",
        "collection": collection, "canon": "NT", "group": work, "copyist": None,
        "chapter": "1", "verse": str(position), "order": position, "ref": f"{work} 1:{position}",
        "text": text, "text_bare": text, "n_tokens": count,
        "has_gap": False, "supplied_frac": 0.0, "duplicate_of": None, **changes,
    }


def prohibit_profiles(*args, **kwargs):
    pytest.fail("The passage command must never load verse AI profiles")


@pytest.mark.parametrize("tokens", [500, 1000, 2000])
def test_declared_lengths_write_source_mapping_and_run_without_ai_or_smoothing(monkeypatch, tmp_path, tokens):
    records = [source(i, tokens // 2) for i in range(1, 7)]
    monkeypatch.setattr(cli, "_load_corpus", lambda: records)
    monkeypatch.setattr(ai_profile, "load_profiles", prohibit_profiles)
    actual_run = cluster.run
    calls = []

    def capture(verses, profiles, out, **kwargs):
        calls.append((verses, profiles, kwargs))
        return actual_run(verses, profiles, out, **kwargs)

    monkeypatch.setattr(cluster, "run", capture)
    cli.main(["cluster-passages", "--scope", "all", "--tokens", str(tokens), "--out", str(tmp_path),
              "--k", "1", "--kmax", "7", "--seed", "31", "--k-criterion", "bic"])
    assert len(calls) == 1
    passages, profiles, options = calls[0]
    assert profiles is None
    assert options["alpha"] == options["window"] == 0
    assert options["weights"] == {"lex": 1.0}
    assert {key: options[key] for key in ("k", "kmax", "seed", "criterion")} == {
        "k": 1, "kmax": 7, "seed": 31, "criterion": "bic"}
    mapping = json.loads((tmp_path / "passages.json").read_text())
    assert mapping["passages"] == passages
    assert [row["n_tokens"] for row in passages] == [tokens] * 3
    assert {ident for row in passages for ident in row["source_verse_ids"]} == {v["id"] for v in records}
    assert mapping["settings"]["retained_tokens"] == len(records) * tokens // 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["used_ai_profiles"] is False
    assert summary["alpha"] == summary["window"] == 0
    assert summary["input_unit"] == "pooled_token_passage"
    assert summary["passage_tokens"] == tokens
    assert summary["source_mapping_file"] == "passages.json"
    assert "not identified authors" in (tmp_path / "report.md").read_text()


def test_scope_language_and_work_selection_happens_before_pooling(monkeypatch, tmp_path):
    selected = [source(i, work="TARGET") for i in range(1, 7)]
    records = selected + [source(1, 1500, work="OTHER"), source(1, 1500, work="TARGET", language="hbo"),
                          source(1, 1500, work="TARGET", collection="LXX")]
    monkeypatch.setattr(cli, "_load_corpus", lambda: records)
    monkeypatch.setattr(ai_profile, "load_profiles", prohibit_profiles)
    cli.main(["cluster-passages", "--tokens", "500", "--language", "grc", "--scope", "nt",
              "--works", "TARGET", "--k", "1", "--out", str(tmp_path)])
    mapping = json.loads((tmp_path / "passages.json").read_text())
    assert mapping["settings"]["input_verses"] == len(selected)
    assert {ident for row in mapping["passages"] for ident in row["source_verse_ids"]} == {v["id"] for v in selected}


def test_damage_and_supplied_text_are_excluded_with_the_short_tails(monkeypatch, tmp_path):
    records = [source(1, 500), source(2, 250), source(3, 500, has_gap=True),
               source(4, 250), source(5, 500), source(6, 500, supplied_frac=.1), source(7, 500)]
    monkeypatch.setattr(cli, "_load_corpus", lambda: records)
    monkeypatch.setattr(ai_profile, "load_profiles", prohibit_profiles)
    cli.main(["cluster-passages", "--tokens", "500", "--scope", "all", "--k", "1", "--out", str(tmp_path)])
    mapping = json.loads((tmp_path / "passages.json").read_text())
    assert len(mapping["passages"]) == 3
    assert [row["reason"] for row in mapping["exclusions"]] == ["short_tail", "gap", "short_tail", "supplied_text"]
    retained_ids = {ident for row in mapping["passages"] for ident in row["source_verse_ids"]}
    assert not retained_ids & {records[2]["id"], records[5]["id"]}
    assert mapping["settings"]["input_tokens"] == 3000
    assert mapping["settings"]["retained_tokens"] == mapping["settings"]["excluded_tokens"] == 1500


def test_insufficient_passages_save_diagnostics_without_clustering(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_corpus", lambda: [source(1, 1000)])
    monkeypatch.setattr(ai_profile, "load_profiles", prohibit_profiles)
    monkeypatch.setattr(cluster, "run", lambda *args, **kwargs: pytest.fail("insufficient passages must not be clustered"))
    with pytest.raises(SystemExit, match="fewer than three complete passages"):
        cli.main(["cluster-passages", "--tokens", "500", "--scope", "all", "--out", str(tmp_path)])
    mapping = json.loads((tmp_path / "passages.json").read_text())
    assert len(mapping["passages"]) == 2
    assert mapping["exclusions"][-1]["reason"] == "insufficient_passages"
    assert not (tmp_path / "summary.json").exists()


def test_selection_cannot_close_a_gap_between_repeated_work_occurrences(monkeypatch, tmp_path):
    records = [source(1), source(1, work="EXCLUDED"), source(2), source(3, 1000)]
    monkeypatch.setattr(cli, "_load_corpus", lambda: records)
    monkeypatch.setattr(cluster, "run", lambda *args, **kwargs: pytest.fail("filtering must not invent a third passage"))
    with pytest.raises(SystemExit, match="fewer than three complete passages"):
        cli.main(["cluster-passages", "--tokens", "500", "--scope", "all", "--works", "W", "--out", str(tmp_path)])
    mapping = json.loads((tmp_path / "passages.json").read_text())
    assert len(mapping["passages"]) == 2
    assert mapping["settings"]["excluded_tokens"] == 500


@pytest.mark.parametrize("extra", [
    ["--alpha", "0.7"], ["--window", "5"], ["--profiles", "profiles.jsonl"], ["--tokens", "250"],
])
def test_passage_command_rejects_verse_ai_smoothing_and_undeclared_lengths(monkeypatch, extra):
    """Verse smoothing and undeclared lengths stay out, and a profiles file alone is not a way in.

    Passage mode is the AI-free control. Pooling verse profiles into passages is available, but only
    through the explicit --with-ai opt-in, so naming a profiles file on its own must still fail, and
    must fail before any corpus is read.
    """
    monkeypatch.setattr(cli, "_load_corpus", lambda: pytest.fail("invalid settings must fail before corpus loading"))
    with pytest.raises(SystemExit) as error:
        cli.main(["cluster-passages", *extra])
    assert error.value.code == 2


def test_passage_mode_stays_ai_free_unless_the_opt_in_is_given(monkeypatch, tmp_path):
    """The default run must not read profiles at all - that independence is the point of the control."""
    monkeypatch.setattr(cli, "_load_corpus", lambda: [source(i, 500) for i in range(1, 8)])
    monkeypatch.setattr(ai_profile, "load_profiles", prohibit_profiles)
    cli.main(["cluster-passages", "--tokens", "500", "--scope", "all", "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ai_profiles_pooled"] == 0 and summary["used_ai_profiles"] is False
