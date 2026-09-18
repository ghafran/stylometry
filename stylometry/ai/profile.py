"""AI style profiling: one structured style profile per verse unit, produced by Claude.

The model sees a whole chapter-sized chunk of consecutive units at a time (so it has context) and
returns one JSON profile per unit. Complete, validated requests are appended to
``data/processed/profiles.jsonl`` as they arrive. Resuming skips complete requests and preserves
the original context for incomplete requests; changed text or generation settings require a new file.

Backends
--------
sdk    synchronous Messages API calls through the ``anthropic`` SDK (needs ANTHROPIC_API_KEY or an
       ``ant auth login`` profile).  Good for pilots; use ``--workers`` for throughput.
batch  Message Batches API - 50% cheaper, asynchronous.  ``profile`` submits, ``profile-collect``
       downloads the finished results.
cli    the local ``claude`` CLI in print mode.  Uses your Claude subscription, no API key needed.
       Slower per call and pays for the CLI's own system prompt, so best for small runs.
deepseek  DeepSeek's OpenAI-compatible API (DEEPSEEK_API_KEY). JSON mode only, so every profile
       is validated here instead of schema-enforced by the server.
openai    any other OpenAI-compatible endpoint (``--base-url``, ``--api-key-env``, ``--model``).

Profiles from different models should not be mixed in one clustering run: keep them in separate
files with ``--profiles``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

MODEL_DEFAULT = "claude-opus-5"
MAX_TOKENS = 16000
PROMPT_VERSION = "blinded-v2"
SCHEMA_VERSION = 2

# JSON mode is not schema-enforced, so a model may answer with an undeclared categorical value,
# a mis-echoed unit id or malformed JSON. Measured on Quranic Arabic, about a quarter of requests
# came back unusable for one of those reasons, and the same request generally succeeded on a retry,
# so a request is re-asked rather than abandoned. Every attempt is billed and is counted.
VALIDATION_ATTEMPTS = 4


class ResponseError(ValueError):
    """A response that was paid for but cannot be used; carries its usage so cost stays honest."""

    def __init__(self, message: str, usage: dict | None = None):
        super().__init__(message)
        self.usage = dict(usage or {})


# An exhausted balance or a rejected key is a property of the account, not of the request: every
# remaining request will fail the same way. A Sinaiticus run spent three minutes pushing 1,114
# requests at an empty account and wrote 1,114 identical errors, burying the one that mattered.
FATAL_STATUS = {401: "the API key was rejected", 402: "the account is out of credit",
                403: "the account is not permitted to use this model"}


def fatal_reason(exc: BaseException) -> str | None:
    """Why no later request in this run can succeed either, or None if it is worth continuing."""
    status = getattr(exc, "status_code", None)
    return FATAL_STATUS.get(status) if isinstance(status, int) else None


# $ per million tokens (input, output).  Batch API halves both; cached input reads are ~10%.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
    # DeepSeek peak-hour list prices (off-peak, 16:00-01:00 and 04:00-06:00 UTC weekdays, is half)
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
    # OpenAI (Batch API halves these)
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
}

# OpenAI-compatible endpoints
OPENAI_PRESETS: dict[str, dict] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-v4-pro"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": None},
}

NUMERIC_DIMS = [
    "register",
    "semitic_interference",
    "hypotaxis",
    "lexical_richness",
    "rhetorical_polish",
    "emotional_intensity",
]
# Language-neutral category values.  Each language's prompt explains what they mean there.
CATEGORICAL_DIMS: dict[str, list[str]] = {
    "discourse_mode": [
        "narrative", "dialogue", "speech", "epistolary", "exhortation", "apocalyptic_vision",
        "prayer_or_hymn", "legal_or_ritual", "genealogy_or_list", "proverb_or_saying", "poetry", "other",
    ],
    "narrative_tense": ["past_narrative", "past_descriptive", "present_vivid", "resultative", "mixed", "not_narrative"],
    "connective_style": ["parataxis", "particle_chain", "asyndeton", "subordinating", "mixed"],
    "voice": ["narrator", "deity", "protagonist", "prophet_or_apostle", "opponent_or_crowd", "author_first_person", "other"],
    "quotation": ["none", "scripture", "other_source"],
}
# Values used by the first (Greek-only) prompt, mapped onto the neutral set when profiles are loaded.
LEGACY_VALUES = {
    "aorist": "past_narrative", "imperfect": "past_descriptive", "historical_present": "present_vivid",
    "perfect": "resultative", "kai_parataxis": "parataxis", "de_chain": "particle_chain",
    "jesus": "protagonist", "apostle_or_disciple": "prophet_or_apostle", "god_or_angel": "deity",
}

BASE_PROMPT = """You are an expert in {language_name} stylometry and authorship attribution.

You receive a numbered list of consecutive verse-sized units from one work. For every unit return a style profile. Judge STYLE ONLY - the unconscious habits of the hand that composed the sentence - never the subject matter, theology, or whether you recognise the passage. {orthography_note}

Scales run 0.0-1.0 with these anchors:
- register: {register_anchor}
- semitic_interference: {interference_anchor}
- hypotaxis: 0 = paratactic chains of short clauses; 1 = deeply subordinated periodic sentences.
- lexical_richness: 0 = only very common words; 1 = rare, poetic or technical vocabulary.
- rhetorical_polish: 0 = artless; 1 = deliberate figures (antithesis, chiasmus, anaphora, alliteration, rhythm, rhyme).
- emotional_intensity: 0 = neutral report; 1 = strong pathos, exclamation, vocatives, superlatives.

Categorical fields use only the listed values:
- discourse_mode: narrative, dialogue, speech, epistolary, exhortation, apocalyptic_vision, prayer_or_hymn, legal_or_ritual, genealogy_or_list, proverb_or_saying, poetry, other.
- narrative_tense: {tense_note}; mixed; not_narrative for units that do not narrate.
- connective_style: parataxis ({parataxis_note}), particle_chain ({particle_note}), asyndeton, subordinating, mixed.
- voice: narrator, deity ({deity_note}), protagonist ({protagonist_note}), prophet_or_apostle, opponent_or_crowd, author_first_person, other.
- quotation: none, scripture, other_source.

style_tags: 1-8 lowercase snake_case tags naming concrete devices or tics visible IN THIS UNIT, e.g. {tag_examples}. Reuse the same tag for the same device so tags are comparable across units.
distinctive_phrases: up to 4 short verbatim {language_name} snippets from the unit that a stylometrist would flag as diagnostic of an author's habits (formulae, favourite connectives, idioms). Empty list if none.
signature: one short sentence characterising the hand behind this unit.

Return exactly one profile per unit, in the given order, echoing each unit's id."""

LANGUAGE_NOTES: dict[str, dict[str, str]] = {
    "grc": {
        "language_name": "Koine Greek",
        "orthography_note": "The text may come from a 4th-century uncial manuscript: ignore itacism, missing accents, nomina sacra and scribal spelling.",
        "register_anchor": "0 = plain vernacular Koine; 0.5 = standard literary Koine; 1 = Atticizing, periodic, sophisticated prose.",
        "interference_anchor": "0 = idiomatic Greek; 1 = translation Greek full of Hebrew/Aramaic calques (kai egeneto, pleonastic pronouns, en to + infinitive, cognate datives, apokritheis eipen).",
        "tense_note": "past_narrative = aorist narrative, past_descriptive = imperfect, present_vivid = historical present, resultative = perfect",
        "parataxis_note": "kai ... kai chains",
        "particle_note": "de / oun / gar chains",
        "deity_note": "God or an angel speaking",
        "protagonist_note": "Jesus or the story's central figure speaking",
        "tag_examples": "genitive_absolute, hina_clause, historical_present, idou, euthys, amen_formula, articular_infinitive, optative, men_de, vocative_address, rhetorical_question, chiasmus, anaphora, asyndeton, polysyndeton, pleonastic_pronoun, periphrastic_tense, litotes, rare_word, compound_verb, superlative, parenthesis, direct_speech, citation_formula, wordplay, hendiadys, inclusio, parallelism, antithesis, long_period, short_clauses",
    },
    "hbo": {
        "language_name": "Biblical Hebrew",
        "orthography_note": "The text may be Masoretic, Qumran or Samaritan: ignore vocalisation, cantillation, plene/defective spelling and scribal orthography; Aramaic passages are judged as Aramaic.",
        "register_anchor": "0 = plain narrative prose (Genesis-Kings style); 0.5 = elevated prose or simple poetry; 1 = dense, archaic or highly wrought poetry (Job, Isaiah 40-66, archaic songs).",
        "interference_anchor": "0 = classical Standard Biblical Hebrew; 1 = Late Biblical Hebrew / Aramaic-influenced language (Aramaisms, Persian loanwords, late syntax and vocabulary as in Chronicles, Esther, Ecclesiastes, Daniel).",
        "tense_note": "past_narrative = wayyiqtol chains, past_descriptive = qatal / participle background, present_vivid = participial or imperative present, resultative = stative qatal",
        "parataxis_note": "waw-consecutive / waw chains",
        "particle_note": "ki / asher / lakhen / hinneh chains",
        "deity_note": "YHWH or an angel speaking",
        "protagonist_note": "the story's central figure speaking (Moses, David, the prophet in his own narrative)",
        "tag_examples": "wayyiqtol_chain, infinitive_absolute, cognate_accusative, casus_pendens, messenger_formula, oracle_formula, hinneh, lemor, vocative_address, rhetorical_question, chiasmus, parallelism, synonymous_parallelism, antithetic_parallelism, inclusio, numerical_saying, list, genealogy_formula, legal_casuistic, legal_apodictic, covenant_formula, divine_name_yhwh, elohim, anaphora, wordplay, rare_word, aramaism, long_sentence, short_clauses, direct_speech, curse_or_blessing, lament, hymn",
    },
    "arb": {
        "language_name": "Quranic Arabic",
        "orthography_note": "The text is in Uthmani orthography: ignore rasm peculiarities, vowel signs and pause marks.",
        "register_anchor": "0 = plain legislative or narrative prose; 0.5 = measured oratory; 1 = highly rhythmic, rhymed, oracular saj' with rare vocabulary.",
        "interference_anchor": "0 = plain native Arabic idiom; 1 = dense foreign loanwords (Syriac, Aramaic, Hebrew, Persian, Ethiopic) and non-native turns of phrase.",
        "tense_note": "past_narrative = perfect (fa'ala) narrative chains, past_descriptive = kana + imperfect background, present_vivid = imperfect / vivid present, resultative = qad + perfect",
        "parataxis_note": "wa- / fa- / thumma chains",
        "particle_note": "inna / anna / bal / lakin chains",
        "deity_note": "God speaking (divine I / We)",
        "protagonist_note": "the Prophet or the central figure of the narrative speaking",
        "tag_examples": "saj_rhyme, oath_formula, qul_formula, ya_ayyuha_alladhina, ya_ayyuha_alnas, divine_we, divine_i, eschatological_threat, eschatological_promise, legal_conditional, parable, rhetorical_question, refrain, repetition, huruf_muqattaat, short_verses, long_verses, epithet_pair, vocative_address, antithesis, parallelism, anaphora, chiasmus, rare_word, loanword, narrative_prophets, dialogue, command_prohibition, exception_clause, oath_response, wa_chain, fa_chain, inna_emphasis",
    },
}


def system_prompt(lang: str = "grc") -> str:
    return BASE_PROMPT.format(**LANGUAGE_NOTES.get(lang, LANGUAGE_NOTES["grc"]))


SYSTEM_PROMPT = system_prompt("grc")


def output_schema() -> dict:
    props: dict = {"id": {"type": "string"}}
    for d in NUMERIC_DIMS:
        props[d] = {"type": "number", "minimum": 0, "maximum": 1}
    for d, values in CATEGORICAL_DIMS.items():
        props[d] = {"type": "string", "enum": values}
    props["style_tags"] = {"type": "array", "items": {"type": "string", "pattern": r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"}, "minItems": 1, "maxItems": 8}
    props["distinctive_phrases"] = {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4}
    props["signature"] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": {
            "profiles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["profiles"],
        "additionalProperties": False,
    }


JSON_EXAMPLE = json.dumps(
    {
        "profiles": [
            {
                "id": "unit_0001",
                "register": 0.2,
                "semitic_interference": 0.7,
                "hypotaxis": 0.1,
                "lexical_richness": 0.2,
                "rhetorical_polish": 0.1,
                "emotional_intensity": 0.2,
                "discourse_mode": "narrative",
                "narrative_tense": "past_narrative",
                "connective_style": "parataxis",
                "voice": "narrator",
                "quotation": "none",
                "style_tags": ["kai_egeneto", "participle_chain", "pleonastic_pronoun"],
                "distinctive_phrases": ["καὶ ἐγένετο"],
                "signature": "Paratactic, Semitizing narrative in the aorist.",
            }
        ]
    },
    ensure_ascii=False,
)

def json_mode_system_prompt(lang: str = "grc") -> str:
    """System prompt for endpoints that offer JSON mode but no schema enforcement."""
    allowed = "; ".join(f"{d}: {', '.join(v)}" for d, v in CATEGORICAL_DIMS.items())
    return (
        system_prompt(lang)
        + "\n\nRespond with a single JSON object and nothing else, in exactly this json shape (one entry per unit, "
        "in order):\n" + JSON_EXAMPLE + "\n\nAllowed categorical values - " + allowed + "."
    )


def coerce_profile(p: object, *, allow_legacy: bool = False) -> dict | None:
    """Validate measurements without inventing, clamping or stringifying missing values.

    The historical function name is retained for callers. The only supported migration is
    mapping documented categorical aliases when explicitly loading a legacy record.
    """
    if not isinstance(p, dict):
        return None
    ident = p.get("id")
    if not isinstance(ident, str) or not ident.strip():
        return None
    out: dict = {"id": ident}
    for d in NUMERIC_DIMS:
        value = p.get(d)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not 0 <= value <= 1 or not math.isfinite(value):
            return None
        out[d] = float(value)
    for d, values in CATEGORICAL_DIMS.items():
        value = p.get(d)
        if not isinstance(value, str):
            return None
        if allow_legacy:
            value = LEGACY_VALUES.get(value, value)
        if value not in values:
            return None
        out[d] = value
    tags = p.get("style_tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= 8:
        return None
    if any(not isinstance(t, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", t) for t in tags):
        return None
    phrases = p.get("distinctive_phrases")
    if not isinstance(phrases, list) or len(phrases) > 4:
        return None
    if any(not isinstance(p, str) or not p.strip() for p in phrases):
        return None
    signature = p.get("signature")
    if not isinstance(signature, str) or not signature.strip():
        return None
    out["style_tags"] = list(tags)
    out["distinctive_phrases"] = list(phrases)
    out["signature"] = signature
    return out


# --- chunking & prompts ------------------------------------------------------------------------------

def chunk_verses(verses: list[dict], size: int = 25) -> list[list[dict]]:
    """Uninterrupted units of one passage, at most ``size`` per chunk."""
    from ..continuity import consecutive

    if size < 1:
        raise ValueError("chunk size must be positive")
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for v in verses:
        if cur:
            if not consecutive(cur[-1], v) or len(cur) >= size:
                chunks.append(cur)
                cur = []
        cur.append(v)
    if cur:
        chunks.append(cur)
    return chunks


def _lang(chunk: list[dict]) -> str:
    return chunk[0].get("language", "grc")


def user_message(chunk: list[dict]) -> str:
    """Only text and anonymous request-local identifiers are exposed to the model."""
    return "\n".join(f"[unit_{i:04d}] {v['text']}" for i, v in enumerate(chunk, 1))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request_context(chunk: list[dict]) -> dict:
    units = [{"id": v["id"], "language": v.get("language", "grc"), "text_sha256": _text_hash(v["text"])} for v in chunk]
    payload = {"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION, "units": units}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {**payload, "request_group": digest}


def estimate_tokens_in(chunk: list[dict]) -> int:
    # Accented Greek is expensive to tokenise: roughly one token per two characters.
    return 40 + sum(len(v["text"]) // 2 + 25 for v in chunk)


def estimate_cost(verses: list[dict], model: str, chunk_size: int = 25, batch: bool = False,
                  *, chunks: list[list[dict]] | None = None) -> dict:
    chunks = chunk_verses(verses, chunk_size) if chunks is None else chunks
    price_in, price_out = PRICES.get(model, PRICES[MODEL_DEFAULT])
    tokens_in = sum(estimate_tokens_in(c) for c in chunks)
    system_tokens = len(SYSTEM_PROMPT) // 4
    tokens_in_cached = system_tokens * len(chunks)
    tokens_out = 230 * len(verses)  # JSON profile (~170) plus adaptive-thinking overhead
    cost = (tokens_in * price_in + tokens_in_cached * price_in * 0.1 + tokens_out * price_out) / 1e6
    if batch:
        cost *= 0.5
    return {
        "verses": len(verses),
        "requests": len(chunks),
        "model": model,
        "batch": batch,
        "est_input_tokens": tokens_in + tokens_in_cached,
        "est_output_tokens": tokens_out,
        "est_cost_usd": round(cost, 2),
    }


# --- storage -----------------------------------------------------------------------------------------

def load_profiles(path: str | Path) -> dict[str, dict]:
    """Load validated measurements; legacy data remain readable but cannot be resumed.

    Bad rows raise a line-specific error instead of silently becoming observations or being
    skipped. Legacy categorical aliases are the sole compatibility migration.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    configuration = None
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid profile JSON: {exc.msg}") from exc
                legacy = isinstance(rec, dict) and "provenance" not in rec
                validated = coerce_profile(rec, allow_legacy=legacy)
                if validated is None:
                    raise ValueError(f"{path}:{line_no}: incomplete or invalid profile; regenerate this request")
                rec.update(validated)
                if "provenance" in rec and not _valid_provenance(rec["provenance"]):
                    raise ValueError(f"{path}:{line_no}: invalid profile provenance; regenerate this request")
                provenance = rec.get("provenance", {})
                record_config = (rec.get("model"), rec.get("backend"), provenance.get("prompt_version"),
                                 provenance.get("schema_version"), provenance.get("generation_settings"))
                if configuration is not None and configuration != record_config:
                    raise ValueError(f"{path}:{line_no}: mixed profiling configurations; use separate profile files")
                configuration = record_config
                previous = out.get(rec["id"])
                if previous is not None and any(previous.get(k) != rec.get(k) for k in ("model", "backend", "provenance")):
                    raise ValueError(f"{path}:{line_no}: conflicting generations for {rec['id']}; use separate profile files")
                out[rec["id"]] = rec
    if any("provenance" not in p for p in out.values()):
        warnings.warn(f"{path}: legacy profiles have no verified blinding, text or request provenance; regenerate for validation", UserWarning, stacklevel=2)
    return out


def _valid_provenance(provenance: object) -> bool:
    if not isinstance(provenance, dict):
        return False
    return (
        isinstance(provenance.get("prompt_version"), str)
        and isinstance(provenance.get("schema_version"), int)
        and all(isinstance(provenance.get(k), str) and re.fullmatch(r"[a-f0-9]{64}", provenance[k])
                for k in ("text_sha256", "request_group"))
        and isinstance(provenance.get("generation_settings"), dict)
    )


def validate_profiles_for_corpus(verses: list[dict], profiles: dict[str, dict]) -> None:
    """Reject stale measurements before an analysis uses them; legacy text is unverified."""
    for verse in verses:
        profile = profiles.get(verse["id"])
        if profile is None:
            continue
        if coerce_profile(profile, allow_legacy="provenance" not in profile) is None:
            raise ValueError(f"incomplete or invalid profile for {verse['id']}; regenerate this request")
        provenance = profile.get("provenance")
        if provenance is not None:
            if not _valid_provenance(provenance):
                raise ValueError(f"invalid profile provenance for {verse['id']}")
            if provenance["text_sha256"] != _text_hash(verse["text"]):
                raise ValueError(f"profile text changed for {verse['id']}; regenerate profiles before analysis")


def _append(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _validate(chunk: list[dict], data: dict, model: str, backend: str, generation_settings: dict | None = None) -> list[dict]:
    """Accept a complete request atomically and map anonymous IDs back to corpus IDs."""
    if not chunk or len({v["id"] for v in chunk}) != len(chunk):
        raise ValueError("profiling requests require unique, nonempty corpus units")
    aliases = {f"unit_{i:04d}": v["id"] for i, v in enumerate(chunk, 1)}
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, list) or len(profiles) != len(chunk):
        raise ValueError(f"response must contain exactly {len(chunk)} profiles; no measurements saved")
    context = _request_context(chunk)
    units = {u["id"]: u for u in context["units"]}
    got: dict[str, dict] = {}
    for raw in profiles:
        p = coerce_profile(raw)
        if p is None:
            raise ValueError("response contains an incomplete or invalid profile; no measurements saved")
        if p["id"] not in aliases:
            raise ValueError(f"response contains unexpected unit id {p['id']!r}; no measurements saved")
        p["id"] = aliases[p["id"]]
        if p["id"] in got:
            raise ValueError("response contains duplicate unit ids; no measurements saved")
        p["model"] = model
        p["backend"] = backend
        p["provenance"] = {
            "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
            "text_sha256": units[p["id"]]["text_sha256"], "request_group": context["request_group"],
            "generation_settings": dict(generation_settings or {}),
        }
        got[p["id"]] = p
    return [got[v["id"]] for v in chunk]


# --- backends ----------------------------------------------------------------------------------------

def _effort_ok(model: str) -> bool:
    return "haiku" not in model


def _sdk_client():
    import anthropic

    return anthropic.Anthropic()


def sdk_call(client, model: str, effort: str, chunk: list[dict]) -> tuple[dict, dict]:
    import anthropic

    kwargs = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_prompt(_lang(chunk)), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(chunk)}],
        output_config={"format": {"type": "json_schema", "schema": output_schema()}},
    )
    if _effort_ok(model):
        kwargs["output_config"]["effort"] = effort

    last: Exception | None = None
    for attempt in range(4):
        try:
            # A fallback model would change the measurement instrument mid-run.
            resp = client.messages.create(**kwargs)
            break
        except anthropic.RateLimitError as e:
            last = e
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                raise
            last = e
        except anthropic.APIConnectionError as e:
            last = e
        time.sleep(min(60, 2 ** attempt * 5))
    else:
        raise RuntimeError(f"gave up after retries: {last}")

    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused chunk starting {chunk[0]['id']}: {getattr(resp, 'stop_details', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"response truncated for chunk starting {chunk[0]['id']}; lower --chunk-size")
    text = next(b.text for b in resp.content if b.type == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
    }
    return json.loads(text), usage


def cli_call(model: str, effort: str, chunk: list[dict]) -> tuple[dict, dict]:
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p", user_message(chunk),
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system_prompt(_lang(chunk)),
        "--json-schema", json.dumps(output_schema()),
        "--no-session-persistence",
        "--tools", "",
    ]
    if _effort_ok(model):
        cmd += ["--effort", effort]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({proc.returncode}): {proc.stderr[-500:]}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI error: {str(data.get('result'))[:300]}")
    parsed = data.get("structured_output")
    if parsed is None:
        text = str(data.get("result", "")).strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        parsed = json.loads(text)
    usage = {"cost_usd": data.get("total_cost_usd", 0.0)}
    return parsed, usage


def _openai_client(preset: str, base_url: str | None, api_key_env: str | None):
    import openai

    cfg = OPENAI_PRESETS[preset]
    key_env = api_key_env or cfg["key_env"]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"{key_env} is not set; export it to use the {preset} backend")
    return openai.OpenAI(api_key=api_key, base_url=base_url or cfg["base_url"])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return text


def _is_openai_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def openai_call(
    client, model: str, effort: str, chunk: list[dict], thinking: bool = False, api_style: str = "deepseek"
) -> tuple[dict, dict]:
    """Chat-completions call in JSON mode (DeepSeek, OpenAI and other compatible endpoints)."""
    import openai

    # DeepSeek V4 reasons by default and bills ~1,000 reasoning tokens per verse (measured: 5,333 for
    # five verses against 1,147 tokens of answer), so thinking is opt-in here and the cap is sized for it.
    cap = 64000 if thinking else MAX_TOKENS
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": json_mode_system_prompt(_lang(chunk))},
            {"role": "user", "content": user_message(chunk)},
        ],
        response_format={"type": "json_object"},
    )
    if api_style == "openai":
        kwargs["max_completion_tokens"] = cap
        if _is_openai_reasoning_model(model):
            # GPT-5 models reason by default at "medium"; keep it low unless --thinking asks for more.
            kwargs["reasoning_effort"] = (effort if effort in ("low", "medium", "high") else "high") if thinking else "low"
    else:
        kwargs["max_tokens"] = cap
        if thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": effort if effort in ("low", "medium", "high") else "high"}
        elif "deepseek" in model:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    last: Exception | None = None
    text = ""
    resp = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            last = e
        except openai.APIStatusError as e:
            if e.status_code < 500:
                raise
            last = e
        except openai.APIConnectionError as e:
            last = e
        else:
            choice = resp.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length":
                raise RuntimeError(f"response truncated for chunk starting {chunk[0]['id']}; lower --chunk-size")
            if text.strip():
                break
            last = RuntimeError("empty content")  # documented DeepSeek JSON-mode quirk: retry
        time.sleep(min(60, 2 ** attempt * 5))
    else:
        raise RuntimeError(f"gave up after retries: {last}")

    u = resp.usage
    cached = getattr(u, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    usage = {
        "input_tokens": (getattr(u, "prompt_tokens", 0) or 0) - (cached or 0),
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_read_input_tokens": cached or 0,
    }
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise ResponseError(f"invalid JSON in response: {exc}", usage) from exc
    return data, usage


# --- drivers -----------------------------------------------------------------------------------------

def profile_settings(backend: str, effort: str = "high", thinking: bool = False, base_url: str | None = None) -> dict:
    """Generation options that must remain fixed across a profile file."""
    return {"effort": effort, "thinking": thinking,
            "endpoint": base_url or OPENAI_PRESETS.get(backend, {}).get("base_url")}

def _resume_chunks(
    verses: list[dict], done: dict[str, dict], model: str, backend: str,
    settings: dict, chunk_size: int, limit: int | None,
) -> list[list[dict]]:
    """Keep original context boundaries, including when an earlier request was partial."""
    if limit is not None and limit < 1:
        raise ValueError("profile limit must be positive")
    if len({v["id"] for v in verses}) != len(verses):
        raise ValueError("corpus contains duplicate unit ids")
    _ensure_compatible_profiles(done, model, backend, settings)
    pending = []
    requested = 0
    for chunk in chunk_verses(verses, chunk_size):
        context = _request_context(chunk)
        for v in chunk:
            p = done.get(v["id"])
            if p is not None and (p["provenance"]["text_sha256"] != _text_hash(v["text"])
                                  or p["provenance"]["request_group"] != context["request_group"]):
                raise ValueError(f"text or request context changed for {v['id']}; choose a new --profiles file")
        if not all(v["id"] in done for v in chunk) and (limit is None or requested < limit):
            pending.append(chunk)
            requested += len(chunk)
    return pending


def _ensure_compatible_profiles(done: dict[str, dict], model: str, backend: str, settings: dict) -> None:
    for ident, p in done.items():
        provenance = p.get("provenance", {})
        compatible = (
            p.get("model") == model and p.get("backend") == backend
            and provenance.get("prompt_version") == PROMPT_VERSION
            and provenance.get("schema_version") == SCHEMA_VERSION
            and provenance.get("generation_settings") == settings
        )
        if not compatible:
            raise ValueError(f"existing profile {ident} uses a legacy or different profiling configuration; choose a new --profiles file")


def run_profile(
    verses: list[dict],
    out_path: str | Path,
    backend: str = "sdk",
    model: str = MODEL_DEFAULT,
    effort: str = "high",
    chunk_size: int = 25,
    workers: int = 4,
    limit: int | None = None,
    progress: Callable[[str], None] = print,
    base_url: str | None = None,
    api_key_env: str | None = None,
    thinking: bool = False,
) -> dict:
    out_path = Path(out_path)
    if backend in OPENAI_PRESETS and model.startswith("claude"):
        model = OPENAI_PRESETS[backend]["model"] or ""
        if not model:
            raise SystemExit(f"--model is required for the {backend} backend")
        progress(f"using {model} for the {backend} backend")
    done = load_profiles(out_path)
    settings = profile_settings(backend, effort, thinking, base_url)
    chunks = _resume_chunks(verses, done, model, backend, settings, chunk_size, limit)
    count = sum(len(c) for c in chunks)
    progress(f"{len(done)} units already profiled; {count} to do in {len(chunks)} requests via {backend} ({model}, effort={effort})")
    if limit is not None and count > limit:
        progress(f"  limit rounded from {limit} to {count} units to preserve complete request context")
    if not chunks:
        return {"profiled": 0, "requests": 0}

    if backend == "batch":
        return submit_batch(chunks, out_path, model, effort, generation_settings=settings)

    call: Callable[[list[dict]], tuple[dict, dict]]
    if backend == "sdk":
        client = _sdk_client()
        call = lambda c: sdk_call(client, model, effort, c)  # noqa: E731
    elif backend == "cli":
        call = lambda c: cli_call(model, effort, c)  # noqa: E731
        workers = min(workers, 2)
    elif backend in OPENAI_PRESETS:
        oclient = _openai_client(backend, base_url, api_key_env)
        call = lambda c: openai_call(oclient, model, effort, c, thinking=thinking, api_style=backend)  # noqa: E731
    else:
        raise ValueError(f"unknown backend {backend!r}")

    def attempt(chunk: list[dict]) -> tuple[list[dict], dict, list[str]]:
        """One request, validated atomically, re-asked while the answer is unusable.

        A response that parses but breaks the declared schema is indistinguishable, to the caller,
        from one that arrives malformed: both are paid for and neither can be stored. Both are
        retried here rather than costing the whole chunk. Usage accumulates across attempts so the
        run's reported cost includes the discarded ones.
        """
        usage_total: dict = {}
        problems: list[str] = []

        def bill(usage: dict) -> None:
            for key, value in usage.items():
                usage_total[key] = usage_total.get(key, 0) + value

        for i in range(VALIDATION_ATTEMPTS):
            try:
                data, usage = call(chunk)
                bill(usage)
                return _validate(chunk, data, model, backend, settings), usage_total, problems
            except ValueError as exc:  # schema violation, or JSON the model malformed
                bill(getattr(exc, "usage", {}))
                problems.append(f"attempt {i + 1}: {str(exc)[:160]}")
                if i + 1 == VALIDATION_ATTEMPTS:
                    raise ResponseError(
                        f"unusable after {VALIDATION_ATTEMPTS} attempts: " + "; ".join(problems), usage_total
                    ) from exc
                time.sleep(min(10, 2 ** i))
        raise AssertionError("unreachable")

    totals = {"profiled": 0, "requests": 0, "errors": 0, "retries": 0,
              "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    err_path = out_path.with_name(out_path.stem + "_errors.jsonl")
    started = time.time()

    def charge(usage: dict) -> None:
        # Rejected measurements still incur API costs; account for them either way.
        totals["input_tokens"] += usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        totals["output_tokens"] += usage.get("output_tokens", 0)
        totals["cost_usd"] += usage.get("cost_usd", 0.0)
        if "input_tokens" in usage:
            pi, po = PRICES.get(model, PRICES[MODEL_DEFAULT])
            totals["cost_usd"] += (
                usage["input_tokens"] * pi
                + usage.get("cache_read_input_tokens", 0) * pi * 0.1
                + usage.get("cache_creation_input_tokens", 0) * pi * 1.25
                + usage["output_tokens"] * po
            ) / 1e6

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(attempt, c): c for c in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            totals["requests"] += 1
            try:
                records, usage, problems = fut.result()
                charge(usage)
                totals["retries"] += len(problems)
            except Exception as e:  # keep going; the run is resumable
                charge(getattr(e, "usage", {}))
                totals["errors"] += 1
                totals["retries"] += VALIDATION_ATTEMPTS - 1
                _append(err_path, [{"ids": [v["id"] for v in chunk], "error": str(e)[:500], "ts": time.time()}])
                progress(f"  [{totals['requests']}/{len(chunks)}] ERROR {chunk[0]['id']}: {str(e)[:120]}")
                reason = fatal_reason(e)
                if reason:
                    for pending in futures:
                        pending.cancel()
                    totals["aborted"] = reason
                    progress(f"  stopping after {totals['requests']} of {len(chunks)} requests: {reason}. "
                             f"Everything profiled so far is saved; rerun the same command to continue.")
                    break
                continue
            _append(out_path, records)
            totals["profiled"] += len(records)
            elapsed = time.time() - started
            retried = f"  ({len(problems)} retried)" if problems else ""
            progress(
                f"  [{totals['requests']}/{len(chunks)}] {chunk[0]['id']}..{chunk[-1]['id']} "
                f"{len(records)} profiles  total ${totals['cost_usd']:.2f}  {elapsed/60:.1f} min{retried}"
            )
    totals["seconds"] = round(time.time() - started, 1)
    if totals["requests"]:  # retain paid failed attempts in the comparison's $/verse
        _append(out_path.with_name(out_path.stem + "_runs.jsonl"), [
            {"ts": time.time(), "model": model, "backend": backend, "effort": effort, "thinking": thinking,
             "chunk_size": chunk_size, **totals}
        ])
    return totals


# --- batch API ---------------------------------------------------------------------------------------

def _batch_state_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_batches.json")


def submit_batch(chunks: list[list[dict]], out_path: Path, model: str, effort: str, generation_settings: dict | None = None) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    state_path = _batch_state_path(out_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    settings = generation_settings or {}
    requested_ids = {v["id"] for c in chunks for v in c}
    for b in state["batches"]:
        if b["collected"]:
            continue
        if b.get("model") != model or b.get("generation_settings") != settings or "contexts" not in b:
            raise ValueError("pending batches use a legacy or different profiling configuration; choose a new --profiles file")
        if requested_ids.intersection(i for ids in b["index"].values() for i in ids):
            raise ValueError("these units already have a pending batch; collect its results before resubmitting")
    _ensure_compatible_profiles(load_profiles(out_path), model, "batch", settings)
    client = _sdk_client()
    schema = output_schema()
    submitted = 0
    for start in range(0, len(chunks), 10000):  # API limit is 100k requests; keep batches modest
        group = chunks[start : start + 10000]
        requests = []
        index: dict[str, list[str]] = {}
        contexts: dict[str, dict] = {}
        for i, chunk in enumerate(group):
            cid = f"c{start + i:06d}"
            index[cid] = [v["id"] for v in chunk]
            contexts[cid] = _request_context(chunk)
            params = dict(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": system_prompt(_lang(chunk)), "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_message(chunk)}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            if _effort_ok(model):
                params["output_config"]["effort"] = effort
            requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)))
        batch = client.messages.batches.create(requests=requests)
        state["batches"].append({"id": batch.id, "model": model, "status": batch.processing_status, "index": index,
                                 "contexts": contexts, "generation_settings": generation_settings or {}, "collected": False})
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        submitted += len(requests)
        print(f"submitted batch {batch.id} with {len(requests)} requests")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    print(f"{submitted} requests submitted; run `stylometry profile-collect` later to fetch results")
    return {"submitted_requests": submitted}


def collect_batches(out_path: str | Path, verses_by_id: dict[str, dict], wait: bool = False) -> dict:
    out_path = Path(out_path)
    state_path = _batch_state_path(out_path)
    if not state_path.exists():
        print("no batches have been submitted")
        return {}
    state = json.loads(state_path.read_text())
    existing = load_profiles(out_path)
    pending_config = None
    for b in state["batches"]:
        if not b["collected"] and "contexts" not in b:
            raise ValueError("pending batch has legacy, unblinded requests; collect it with the previous version into a separate file, or submit a new run")
        if not b["collected"]:
            settings = b.get("generation_settings", {})
            _ensure_compatible_profiles(existing, b["model"], "batch", settings)
            config = (b["model"], settings)
            if pending_config is not None and config != pending_config:
                raise ValueError("pending batches mix profiling configurations; collect into separate profile files")
            pending_config = config
    client = _sdk_client()
    totals = {"profiled": 0, "errored": 0, "pending": 0}
    while True:
        pending = 0
        for b in state["batches"]:
            if b["collected"]:
                continue
            batch = client.messages.batches.retrieve(b["id"])
            b["status"] = batch.processing_status
            if batch.processing_status != "ended":
                pending += 1
                print(f"batch {b['id']}: {batch.processing_status} ({batch.request_counts.processing} processing)")
                continue
            records: list[dict] = []
            for result in client.messages.batches.results(b["id"]):
                ids = b["index"].get(result.custom_id, [])
                chunk = [verses_by_id[i] for i in ids if i in verses_by_id]
                if result.result.type != "succeeded":
                    totals["errored"] += 1
                    continue
                msg = result.result.message
                if msg.stop_reason in ("refusal", "max_tokens"):
                    totals["errored"] += 1
                    continue
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                try:
                    if len(chunk) != len(ids) or _request_context(chunk) != b["contexts"].get(result.custom_id):
                        raise ValueError("corpus text or request context changed since batch submission")
                    records += _validate(chunk, json.loads(text), b["model"], "batch", b.get("generation_settings", {}))
                except (json.JSONDecodeError, ValueError) as exc:
                    totals["errored"] += 1
                    _append(out_path.with_name(out_path.stem + "_errors.jsonl"),
                            [{"ids": ids, "error": str(exc)[:500], "batch": b["id"], "ts": time.time()}])
            _append(out_path, records)
            totals["profiled"] += len(records)
            b["collected"] = True
            print(f"batch {b['id']}: collected {len(records)} profiles")
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        totals["pending"] = pending
        if not pending or not wait:
            break
        time.sleep(60)
    return totals
