"""AI style profiling: one structured style profile per verse unit, produced by Claude.

The model sees a whole chapter-sized chunk of consecutive units at a time (so it has context) and
returns one JSON profile per unit.  Profiles are appended to ``data/processed/profiles.jsonl`` as they
arrive, so every backend is resumable: units that already have a profile are skipped.

Backends
--------
sdk    synchronous Messages API calls through the ``anthropic`` SDK (needs ANTHROPIC_API_KEY or an
       ``ant auth login`` profile).  Good for pilots; use ``--workers`` for throughput.
batch  Message Batches API - 50% cheaper, asynchronous.  ``profile`` submits, ``profile-collect``
       downloads the finished results.
cli    the local ``claude`` CLI in print mode.  Uses your Claude subscription, no API key needed.
       Slower per call and pays for the CLI's own system prompt, so best for small runs.
deepseek  DeepSeek's OpenAI-compatible API (DEEPSEEK_API_KEY).  Much cheaper; JSON mode only, so
       every profile is validated and coerced here instead of schema-enforced by the server.
openai    any other OpenAI-compatible endpoint (``--base-url``, ``--api-key-env``, ``--model``).

Profiles from different models should not be mixed in one clustering run: keep them in separate
files with ``--profiles``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

MODEL_DEFAULT = "claude-opus-5"
MAX_TOKENS = 16000

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

style_tags: 3-8 lowercase snake_case tags naming concrete devices or tics visible IN THIS UNIT, e.g. {tag_examples}. Reuse the same tag for the same device so tags are comparable across units.
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
    props["style_tags"] = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8}
    props["distinctive_phrases"] = {"type": "array", "items": {"type": "string"}, "maxItems": 4}
    props["signature"] = {"type": "string"}
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
                "id": "MARK.1.4",
                "register": 0.2,
                "semitic_interference": 0.7,
                "hypotaxis": 0.1,
                "lexical_richness": 0.2,
                "rhetorical_polish": 0.1,
                "emotional_intensity": 0.2,
                "discourse_mode": "narrative",
                "narrative_tense": "aorist",
                "connective_style": "kai_parataxis",
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

# Where a model returns a value outside the allowed set.
DEFAULT_CATEGORY = {
    "discourse_mode": "other",
    "narrative_tense": "mixed",
    "connective_style": "mixed",
    "voice": "other",
    "quotation": "none",
}


def json_mode_system_prompt(lang: str = "grc") -> str:
    """System prompt for endpoints that offer JSON mode but no schema enforcement."""
    allowed = "; ".join(f"{d}: {', '.join(v)}" for d, v in CATEGORICAL_DIMS.items())
    return (
        system_prompt(lang)
        + "\n\nRespond with a single JSON object and nothing else, in exactly this json shape (one entry per unit, "
        "in order):\n" + JSON_EXAMPLE + "\n\nAllowed categorical values - " + allowed + "."
    )


def coerce_profile(p: object) -> dict | None:
    """Clamp scales, map unknown categories to their default, normalise tag lists."""
    if not isinstance(p, dict):
        return None
    out: dict = {"id": str(p.get("id", "")).strip("[] ")}
    for d in NUMERIC_DIMS:
        try:
            out[d] = min(1.0, max(0.0, float(p.get(d, 0.5))))
        except (TypeError, ValueError):
            out[d] = 0.5
    for d, values in CATEGORICAL_DIMS.items():
        v = str(p.get(d, "")).strip().lower().replace(" ", "_")
        v = LEGACY_VALUES.get(v, v)
        out[d] = v if v in values else DEFAULT_CATEGORY[d]
    tags = p.get("style_tags") if isinstance(p.get("style_tags"), list) else []
    out["style_tags"] = [str(t).strip().lower().replace(" ", "_") for t in tags if str(t).strip()][:8]
    phrases = p.get("distinctive_phrases") if isinstance(p.get("distinctive_phrases"), list) else []
    out["distinctive_phrases"] = [str(x).strip() for x in phrases if str(x).strip()][:4]
    out["signature"] = str(p.get("signature", "")).strip()
    return out


# --- chunking & prompts ------------------------------------------------------------------------------

def chunk_verses(verses: list[dict], size: int = 25) -> list[list[dict]]:
    """Consecutive units of one work, preferring chapter boundaries, at most ``size`` per chunk."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for v in verses:
        if cur:
            new_work = v["work"] != cur[-1]["work"]
            new_chapter = v["chapter"] != cur[-1]["chapter"]
            if new_work or len(cur) >= size or (new_chapter and len(cur) >= size // 2):
                chunks.append(cur)
                cur = []
        cur.append(v)
    if cur:
        chunks.append(cur)
    return chunks


def _lang(chunk: list[dict]) -> str:
    return chunk[0].get("language", "grc")


def user_message(chunk: list[dict]) -> str:
    from .corpus.meta import WITNESSES
    from .lang import LANGUAGE_NAMES

    v0 = chunk[0]
    witness = WITNESSES.get(v0.get("witness", ""), v0.get("witness", ""))
    head = (
        f"Language: {LANGUAGE_NAMES.get(_lang(chunk), _lang(chunk))}. Work: {v0['work_title']} ({v0['work']})"
        + (f", witness: {witness}" if witness else "")
        + f". Units {v0['ref']} to {chunk[-1]['ref']}.\n\n"
    )
    return head + "\n".join(f"[{v['id']}] {v['text']}" for v in chunk)


def estimate_tokens_in(chunk: list[dict]) -> int:
    # Accented Greek is expensive to tokenise: roughly one token per two characters.
    return 40 + sum(len(v["text"]) // 2 + 25 for v in chunk)


def estimate_cost(verses: list[dict], model: str, chunk_size: int = 25, batch: bool = False) -> dict:
    chunks = chunk_verses(verses, chunk_size)
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
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                for d in CATEGORICAL_DIMS:  # profiles written by the Greek-only prompt use older labels
                    if d in rec:
                        rec[d] = LEGACY_VALUES.get(rec[d], rec[d])
                out[rec["id"]] = rec
    return out


def _append(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _validate(chunk: list[dict], data: dict, model: str, backend: str) -> list[dict]:
    wanted = {v["id"] for v in chunk}
    # Some models echo the id without its language prefix ("MARK.1.1" for "grc:MARK.1.1").
    aliases = {v["id"].split(":", 1)[-1]: v["id"] for v in chunk}
    got: dict[str, dict] = {}
    profiles = data.get("profiles", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for raw in profiles:
        p = coerce_profile(raw)
        if not p:
            continue
        p["id"] = aliases.get(p["id"], p["id"])
        if p["id"] in wanted:
            p["model"] = model
            p["backend"] = backend
            got[p["id"]] = p
    missing = wanted - set(got)
    if missing:
        print(f"  warning: {len(missing)} of {len(chunk)} units missing from response ({sorted(missing)[:3]}...)")
    return [got[v["id"]] for v in chunk if v["id"] in got]


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
            try:
                # Server-side refusal fallback (routes a declined request to another model in-call).
                resp = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            except TypeError:  # SDK predates the fallbacks parameter
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

    data = json.loads(_strip_fences(text))
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
    return data, usage


# --- drivers -----------------------------------------------------------------------------------------

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
    todo = [v for v in verses if v["id"] not in done]
    if limit:
        todo = todo[:limit]
    chunks = chunk_verses(todo, chunk_size)
    progress(f"{len(done)} units already profiled; {len(todo)} to do in {len(chunks)} requests via {backend} ({model}, effort={effort})")
    if not chunks:
        return {"profiled": 0, "requests": 0}

    if backend == "batch":
        return submit_batch(chunks, out_path, model, effort)

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

    totals = {"profiled": 0, "requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    err_path = out_path.with_name(out_path.stem + "_errors.jsonl")
    started = time.time()

    def work(chunk: list[dict]) -> tuple[list[dict], list[dict], dict]:
        data, usage = call(chunk)
        return chunk, _validate(chunk, data, model, backend), usage

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, c): c for c in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            totals["requests"] += 1
            try:
                _, records, usage = fut.result()
            except Exception as e:  # keep going; the run is resumable
                totals["errors"] += 1
                _append(err_path, [{"ids": [v["id"] for v in chunk], "error": str(e)[:500], "ts": time.time()}])
                progress(f"  [{totals['requests']}/{len(chunks)}] ERROR {chunk[0]['id']}: {str(e)[:120]}")
                continue
            _append(out_path, records)
            totals["profiled"] += len(records)
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
            elapsed = time.time() - started
            progress(
                f"  [{totals['requests']}/{len(chunks)}] {chunk[0]['id']}..{chunk[-1]['id']} "
                f"{len(records)} profiles  total ${totals['cost_usd']:.2f}  {elapsed/60:.1f} min"
            )
    totals["seconds"] = round(time.time() - started, 1)
    if totals["profiled"]:  # sidecar used by `stylometry compare-models` to measure $/verse
        _append(out_path.with_name(out_path.stem + "_runs.jsonl"), [
            {"ts": time.time(), "model": model, "backend": backend, "effort": effort, "thinking": thinking,
             "chunk_size": chunk_size, **totals}
        ])
    return totals


# --- batch API ---------------------------------------------------------------------------------------

def _batch_state_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_batches.json")


def submit_batch(chunks: list[list[dict]], out_path: Path, model: str, effort: str) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _sdk_client()
    state_path = _batch_state_path(out_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    schema = output_schema()
    submitted = 0
    for start in range(0, len(chunks), 10000):  # API limit is 100k requests; keep batches modest
        group = chunks[start : start + 10000]
        requests = []
        index: dict[str, list[str]] = {}
        for i, chunk in enumerate(group):
            cid = f"c{start + i:06d}"
            index[cid] = [v["id"] for v in chunk]
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
        state["batches"].append({"id": batch.id, "model": model, "status": batch.processing_status, "index": index, "collected": False})
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
    client = _sdk_client()
    state = json.loads(state_path.read_text())
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
                    records += _validate(chunk, json.loads(text), b["model"], "batch")
                except json.JSONDecodeError:
                    totals["errored"] += 1
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
