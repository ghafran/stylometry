"""Offline controls for blind, valid and reproducible AI measurements."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import pytest

from stylometry import ai_profile as ap


def verses(n=4):
    return [{"id": f"grc:SECRET.1.{i}", "language": "grc", "work": "SECRET",
             "work_title": "Identifying title", "witness": "Identifying witness", "chapter": "1", "verse": str(i),
             "ref": f"Identifying reference {i}", "text": f"λόγος καί {i}"} for i in range(1, n + 1)]


def measurement(ident="unit_0001"):
    return {"id": ident, **{d: 0.25 for d in ap.NUMERIC_DIMS},
            **{d: values[0] for d, values in ap.CATEGORICAL_DIMS.items()},
            "style_tags": ["short_clauses"], "distinctive_phrases": [],
            "signature": "Short coordinate clauses."}


def response(chunk):
    return {"profiles": [measurement(f"unit_{i:04d}") for i in range(1, len(chunk) + 1)]}


def records(chunk, backend="cli", settings=None):
    return ap._validate(chunk, response(chunk), "test-model", backend,
                        ap.profile_settings(backend) if settings is None else settings)


def write_records(path, rows):
    path.write_text("".join(json.dumps(p) + "\n" for p in rows))


def test_prompt_invariant_to_identifying_metadata():
    original = verses()
    renamed = copy.deepcopy(original)
    for row in renamed:
        for key in ("id", "work", "work_title", "witness", "ref", "chapter"):
            row[key] = "entirely different metadata"
    assert ap.user_message(original) == ap.user_message(renamed)
    assert "[unit_0001] λόγος καί 1" in ap.user_message(original)
    assert not any(word in ap.user_message(original) for word in ("SECRET", "Identifying"))
    assert '"id": "unit_0001"' in ap.JSON_EXAMPLE


@pytest.mark.parametrize("dimension", ap.NUMERIC_DIMS)
@pytest.mark.parametrize("bad", [None, True, "0.5", -0.01, 1.01, float("nan"), float("inf"), float("-inf")])
def test_invalid_numeric_measurements_are_not_imputed(dimension, bad):
    p = measurement()
    p[dimension] = bad
    assert ap.coerce_profile(p) is None


@pytest.mark.parametrize("field", list(measurement()))
def test_every_measurement_field_is_required(field):
    p = measurement()
    del p[field]
    assert ap.coerce_profile(p) is None


@pytest.mark.parametrize("field,value", [
    ("discourse_mode", "unknown"), ("narrative_tense", "aorist"), ("voice", []),
    ("style_tags", "asyndeton"), ("style_tags", []), ("style_tags", [3]),
    ("style_tags", ["Invented Tag"]), ("style_tags", ["tag"] * 9),
    ("distinctive_phrases", "words"), ("distinctive_phrases", [None]),
    ("distinctive_phrases", ["phrase"] * 5), ("signature", ""), ("signature", 3),
])
def test_invalid_categories_and_lists_rejected(field, value):
    p = measurement()
    p[field] = value
    assert ap.coerce_profile(p) is None


def test_validation_maps_reordered_anonymous_ids_and_attaches_provenance():
    chunk = verses(2)
    data = response(chunk)
    data["profiles"].reverse()
    result = ap._validate(chunk, data, "test-model", "cli", ap.profile_settings("cli"))
    assert [r["id"] for r in result] == [v["id"] for v in chunk]
    assert result[0]["provenance"]["request_group"] == result[1]["provenance"]["request_group"]
    assert result[0]["provenance"]["text_sha256"] != result[1]["provenance"]["text_sha256"]
    assert result[0]["provenance"]["prompt_version"] == ap.PROMPT_VERSION


@pytest.mark.parametrize("failure", ["missing", "duplicate", "identifying_id", "id_only", "not_list"])
def test_invalid_responses_reject_entire_request(failure):
    chunk = verses(2)
    data = response(chunk)
    if failure == "missing":
        data["profiles"].pop()
    elif failure == "duplicate":
        data["profiles"][1]["id"] = "unit_0001"
    elif failure == "identifying_id":
        data["profiles"][0]["id"] = chunk[0]["id"]
    elif failure == "id_only":
        data["profiles"][1] = {"id": "unit_0002"}
    else:
        data["profiles"] = "invalid"
    with pytest.raises(ValueError):
        ap._validate(chunk, data, "test-model", "cli")


def test_legacy_loading_explicitly_warns_and_only_maps_documented_aliases(tmp_path):
    path = tmp_path / "profiles.jsonl"
    p = measurement("grc:SECRET.1.1")
    p["narrative_tense"] = "aorist"
    p["connective_style"] = "kai_parataxis"
    write_records(path, [p])
    with pytest.warns(UserWarning, match="legacy profiles"):
        loaded = ap.load_profiles(path)
    assert loaded[p["id"]]["narrative_tense"] == "past_narrative"
    assert loaded[p["id"]]["connective_style"] == "parataxis"
    with pytest.raises(ValueError, match="legacy or different"):
        ap._resume_chunks(verses(1), loaded, "test-model", "cli", ap.profile_settings("cli"), 25, None)


def test_bad_stored_measurements_report_the_line(tmp_path):
    path = tmp_path / "profiles.jsonl"
    write_records(path, [measurement("a"), {"id": "b"}])
    with pytest.raises(ValueError, match=r":2: incomplete or invalid"):
        ap.load_profiles(path)


@pytest.mark.parametrize("change", ["model", "backend", "prompt", "settings", "legacy"])
def test_mixed_generations_across_distinct_ids_rejected(tmp_path, change):
    path = tmp_path / "profiles.jsonl"
    rows = records(verses(2))
    if change in ("model", "backend"):
        rows[1][change] = "different"
    elif change == "prompt":
        rows[1]["provenance"]["prompt_version"] = "old"
    elif change == "settings":
        rows[1]["provenance"]["generation_settings"]["thinking"] = True
    else:
        del rows[1]["provenance"]
    write_records(path, rows)
    with pytest.raises(ValueError, match="mixed profiling configurations"):
        ap.load_profiles(path)


def test_partial_resume_retains_original_context_and_request_ids(tmp_path, monkeypatch):
    chunk = verses()
    path = tmp_path / "profiles.jsonl"
    write_records(path, records(chunk)[:1])
    seen = []

    def fake_call(model, effort, request):
        seen.append(request)
        return response(request), {}

    monkeypatch.setattr(ap, "cli_call", fake_call)
    totals = ap.run_profile(chunk, path, backend="cli", model="test-model", chunk_size=4, progress=lambda _: None)
    assert totals["profiled"] == 4 and totals["errors"] == 0
    assert seen == [chunk]
    loaded = ap.load_profiles(path)
    assert len(loaded) == 4
    assert len({p["provenance"]["request_group"] for p in loaded.values()}) == 1
    assert ap.run_profile(chunk, path, backend="cli", model="test-model", chunk_size=4)["requests"] == 0


@pytest.mark.parametrize("change", ["text", "context", "model", "settings"])
def test_resume_rejects_stale_or_changed_measurement_instrument(change):
    chunk = verses()
    done = {r["id"]: r for r in records(chunk)}
    size, model, settings = 4, "test-model", ap.profile_settings("cli")
    if change == "text":
        chunk[0]["text"] += " changed"
    elif change == "context":
        size = 2
    elif change == "model":
        model = "different-model"
    else:
        settings["effort"] = "low"
    with pytest.raises(ValueError):
        ap._resume_chunks(chunk, done, model, "cli", settings, size, None)


def test_limit_preserves_whole_original_chunks():
    chunk = verses(8)
    selected = ap._resume_chunks(chunk, {}, "m", "cli", ap.profile_settings("cli"), 4, 5)
    assert list(map(len, selected)) == [4, 4]
    assert ap._request_context(selected[0]) == ap._request_context(chunk[:4])


def test_invalid_run_logs_error_without_saving_fabricated_profiles(tmp_path, monkeypatch):
    calls = []

    def always_invalid(*args):
        calls.append(1)
        return {"profiles": [{"id": "unit_0001"}]}, {"cost_usd": 0.02}

    monkeypatch.setattr(ap, "cli_call", always_invalid)
    monkeypatch.setattr(ap, "time", NS(time=lambda: 0.0, sleep=lambda _: None))
    path = tmp_path / "profiles.jsonl"
    totals = ap.run_profile(verses(1), path, backend="cli", model="test-model", progress=lambda _: None)
    assert totals["errors"] == 1 and totals["profiled"] == 0
    assert len(calls) == ap.VALIDATION_ATTEMPTS, "an unusable answer is re-asked before it is given up on"
    assert totals["cost_usd"] == pytest.approx(0.02 * ap.VALIDATION_ATTEMPTS), "every paid attempt is charged"
    assert not path.exists()
    logged = json.loads((tmp_path / "profiles_errors.jsonl").read_text())["error"]
    assert "invalid profile" in logged and f"attempt {ap.VALIDATION_ATTEMPTS}" in logged
    assert json.loads((tmp_path / "profiles_runs.jsonl").read_text())["cost_usd"] == pytest.approx(
        0.02 * ap.VALIDATION_ATTEMPTS)


def test_a_schema_violation_is_retried_rather_than_costing_the_whole_request(tmp_path, monkeypatch):
    """DeepSeek offers no schema enforcement, so one bad field must not discard 24 good profiles."""
    chunk = verses(4)
    attempts = []

    def flaky(*args):
        attempts.append(1)
        data = response(chunk)
        if len(attempts) < 3:  # an undeclared categorical value, as measured on Quranic Arabic
            data["profiles"][2]["connective_style"] = "fa_chain"
        return data, {"cost_usd": 0.01}

    monkeypatch.setattr(ap, "cli_call", flaky)
    monkeypatch.setattr(ap, "time", NS(time=lambda: 0.0, sleep=lambda _: None))
    path = tmp_path / "profiles.jsonl"
    totals = ap.run_profile(chunk, path, backend="cli", model="test-model", chunk_size=4,
                            progress=lambda _: None)
    assert totals["profiled"] == 4 and totals["errors"] == 0
    assert totals["retries"] == 2 and len(attempts) == 3
    assert totals["cost_usd"] == pytest.approx(0.03), "the two discarded attempts were still paid for"
    assert not (tmp_path / "profiles_errors.jsonl").exists()
    assert set(ap.load_profiles(path)) == {v["id"] for v in chunk}


def test_malformed_json_reports_what_it_cost(tmp_path, monkeypatch):
    """A response that never parses is still billed, so the run's cost must include it."""
    def broken(*args):
        raise ap.ResponseError("invalid JSON in response: boom", {"input_tokens": 100, "output_tokens": 50})

    monkeypatch.setattr(ap, "cli_call", broken)
    monkeypatch.setattr(ap, "time", NS(time=lambda: 0.0, sleep=lambda _: None))
    path = tmp_path / "profiles.jsonl"
    totals = ap.run_profile(verses(1), path, backend="cli", model="test-model", progress=lambda _: None)
    assert totals["errors"] == 1 and totals["profiled"] == 0
    assert totals["input_tokens"] == 100 * ap.VALIDATION_ATTEMPTS
    assert totals["output_tokens"] == 50 * ap.VALIDATION_ATTEMPTS
    assert totals["cost_usd"] > 0, "tokens spent on unparseable answers are not free"


def test_chunking_breaks_at_missing_passages_and_witness_changes():
    chunk = verses(4)
    chunk[2]["verse"] = "100"
    chunk[3]["verse"] = "101"
    assert [len(c) for c in ap.chunk_verses(chunk)] == [2, 2]
    chunk[1]["witness"] = "another witness"
    assert [len(c) for c in ap.chunk_verses(chunk)] == [1, 1, 2]


def test_cost_estimate_uses_exact_preserved_request_boundaries():
    chunk = verses(4)
    estimate = ap.estimate_cost(chunk, "test-model", chunks=[chunk[:1], chunk[1:]])
    assert estimate["requests"] == 2 and estimate["verses"] == 4


def test_analysis_rejects_profiles_of_changed_text():
    chunk = verses()
    profiles = {p["id"]: p for p in records(chunk)}
    ap.validate_profiles_for_corpus(chunk, profiles)
    chunk[2]["text"] += " corrected"
    with pytest.raises(ValueError, match="profile text changed"):
        ap.validate_profiles_for_corpus(chunk, profiles)


@pytest.mark.parametrize("backend", ["sdk", "cli", "openai", "deepseek"])
def test_every_sync_backend_sends_identical_blinded_units(monkeypatch, backend):
    chunk = verses(1)
    captured = {}
    body = json.dumps(response(chunk))

    def create(**kwargs):
        captured.update(kwargs)
        if backend == "sdk":
            return NS(stop_reason="end_turn", content=[NS(type="text", text=body)],
                      usage=NS(input_tokens=1, output_tokens=1))
        return NS(choices=[NS(message=NS(content=body), finish_reason="stop")],
                  usage=NS(prompt_tokens=1, completion_tokens=1))

    if backend == "sdk":
        ap.sdk_call(NS(messages=NS(create=create)), "test-model", "high", chunk)
        prompt = captured["messages"][0]["content"]
        assert "fallbacks" not in captured
    elif backend == "cli":
        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return NS(returncode=0, stdout=json.dumps({"structured_output": response(chunk)}))
        monkeypatch.setattr(ap.subprocess, "run", fake_run)
        ap.cli_call("test-model", "high", chunk)
        prompt = captured["cmd"][2]
    else:
        ap.openai_call(NS(chat=NS(completions=NS(create=create))), "test-model", "high", chunk, api_style=backend)
        prompt = captured["messages"][1]["content"]
    assert prompt == ap.user_message(chunk)
    assert "SECRET" not in prompt


def batch_client(captured, result_data):
    def create(**kwargs):
        captured.update(kwargs)
        return NS(id="batch-1", processing_status="in_progress")
    message = NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(result_data))])
    return NS(messages=NS(batches=NS(
        create=create, retrieve=lambda _: NS(processing_status="ended"),
        results=lambda _: [NS(custom_id="c000000", result=NS(type="succeeded", message=message))],
    )))


def test_batch_roundtrip_blinds_requests_and_verifies_context(tmp_path, monkeypatch):
    chunk, captured = verses(2), {}
    path = tmp_path / "profiles.jsonl"
    monkeypatch.setattr(ap, "_sdk_client", lambda: batch_client(captured, response(chunk)))
    ap.submit_batch([chunk], path, "test-model", "high", ap.profile_settings("batch"))
    prompt = captured["requests"][0]["params"]["messages"][0]["content"]
    assert prompt == ap.user_message(chunk) and "SECRET" not in prompt
    result = ap.collect_batches(path, {v["id"]: v for v in chunk})
    assert result["profiled"] == 2 and result["errored"] == 0
    loaded = ap.load_profiles(path)
    assert loaded[chunk[0]["id"]]["provenance"]["request_group"] == ap._request_context(chunk)["request_group"]


def test_batch_rejects_changed_text_and_duplicate_submission(tmp_path, monkeypatch):
    chunk, captured = verses(2), {}
    path = tmp_path / "profiles.jsonl"
    monkeypatch.setattr(ap, "_sdk_client", lambda: batch_client(captured, response(chunk)))
    ap.submit_batch([chunk], path, "test-model", "high", ap.profile_settings("batch"))
    with pytest.raises(ValueError, match="already have a pending batch"):
        ap.submit_batch([chunk], path, "test-model", "high", ap.profile_settings("batch"))
    chunk[0]["text"] += " changed"
    result = ap.collect_batches(path, {v["id"]: v for v in chunk})
    assert result["profiled"] == 0 and result["errored"] == 1
    assert ap.load_profiles(path) == {}


def test_legacy_batch_is_rejected_before_network_access(tmp_path, monkeypatch):
    path = tmp_path / "profiles.jsonl"
    ap._batch_state_path(path).write_text(json.dumps({"batches": [{"collected": False}]}))
    monkeypatch.setattr(ap, "_sdk_client", lambda: pytest.fail("must not contact API"))
    with pytest.raises(ValueError, match="legacy, unblinded"):
        ap.collect_batches(path, {})


def test_an_empty_account_stops_the_run_instead_of_retrying_every_chunk(tmp_path, monkeypatch):
    """A 402 is an account fact, not a request fact; 1,114 further requests cannot fix it."""
    class Rejected(Exception):
        status_code = 402

    import time as real_time

    calls = []

    def broke(*args):
        calls.append(1)
        real_time.sleep(0.02)  # a real request is not instant; without this the pool drains first
        raise Rejected("Insufficient Balance")

    monkeypatch.setattr(ap, "cli_call", broke)
    monkeypatch.setattr(ap, "time", NS(time=lambda: 0.0, sleep=lambda _: None))
    path = tmp_path / "profiles.jsonl"
    totals = ap.run_profile(verses(40), path, backend="cli", model="test-model", chunk_size=4,
                            workers=1, progress=lambda _: None)
    assert totals["aborted"] == "the account is out of credit"
    assert totals["requests"] == 1, "the run stops at the first account-level refusal"
    assert len(calls) < 10, f"the nine remaining requests were not cancelled: {len(calls)} calls"
    logged = (tmp_path / "profiles_errors.jsonl").read_text().strip().splitlines()
    assert len(logged) == 1, "one account-level refusal is logged once, not once per chunk"
    assert not path.exists()


@pytest.mark.parametrize("status,expected", [
    (401, "the API key was rejected"), (402, "the account is out of credit"),
    (403, "the account is not permitted to use this model"),
    (429, None), (500, None), (None, None),
])
def test_only_account_level_refusals_are_treated_as_fatal(status, expected):
    exc = Exception("boom")
    if status is not None:
        exc.status_code = status
    assert ap.fatal_reason(exc) == expected
