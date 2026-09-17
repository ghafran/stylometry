"""Pool normalized source tokens before measuring passage-level style.

Passages are disjoint and respect the original observed continuity. Source spans
use zero-based, half-open offsets into each source verse's ``text_bare.split()``.
Damaged or supplied verses are excluded because their token-level boundaries are
not represented in the corpus schema. This representation is experimental; it
does not establish authorship or validate a particular passage length.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from numbers import Real

from .continuity import annotate_continuity, consecutive


PASSAGE_LENGTHS = (500, 1000, 2000)
_LABEL_BOUNDARIES = ("author", "group", "copyist", "duplicate_of")


def build_passages(verses: list[dict], tokens: int = 1000, *, bridge_chapters: bool = False) -> dict:
    """Return exact token passages, explicit exclusions, and source mappings.

    No text is normalized a second time and no source record is reordered. The
    caller must retain the original records to recover diplomatic/display text;
    output ``text`` is the same normalized token sequence as ``text_bare``.
    Existing continuity annotations preserve holes from earlier filtering.
    ``insufficient_passages`` notices retain generated passages but record when
    a language has fewer than the three observations required by clustering.
    """
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens not in PASSAGE_LENGTHS:
        raise ValueError("passage tokens must be one of 500, 1000, or 2000")
    identities = set()
    for verse in verses:
        ident = verse.get("id")
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError("each source verse needs a nonempty id")
        if ident in identities:
            raise ValueError(f"duplicate source verse id: {ident}")
        identities.add(ident)
        if not isinstance(verse.get("work"), str) or not verse["work"].strip():
            raise ValueError("each source verse needs a nonempty work")
        if not isinstance(verse.get("text_bare"), str):
            raise ValueError("each source verse needs normalized text_bare")
        supplied = verse.get("supplied_frac", 0.0)
        if (isinstance(supplied, bool) or not isinstance(supplied, Real)
                or not math.isfinite(supplied) or not 0 <= supplied <= 1):
            raise ValueError("supplied_frac must be a finite number between zero and one")

    annotated = annotate_continuity(verses, bridge_chapters=bridge_chapters)
    passages, exclusions, run = [], [], []
    run_number = 0

    def source_spans(records, start, stop):
        spans, cursor = [], 0
        for record, words in records:
            lo, hi = max(start - cursor, 0), min(stop - cursor, len(words))
            if lo < hi:
                spans.append({"verse_id": record["id"], "token_start": lo, "token_end": hi})
            cursor += len(words)
        return spans

    def exclude(reason, records, start=0, stop=None):
        stop = sum(len(words) for _, words in records) if stop is None else stop
        first = records[0][0]
        spans = source_spans(records, start, stop)
        exclusions.append({
            "reason": reason, "language": first.get("language", "grc"), "work": first["work"],
            "witness": first.get("witness"), "source": first.get("source"), "chapter": first.get("chapter"),
            "source_verse_ids": [s["verse_id"] for s in spans] or [first["id"]],
            "source_spans": spans, "n_tokens": stop - start,
        })

    def flush():
        nonlocal run_number
        if not run:
            return
        first = run[0][0]
        words = [word for _, source_words in run for word in source_words]
        full = len(words) // tokens
        refs = {record["id"]: record.get("ref", record["id"]) for record, _ in run}
        for position in range(full):
            start, stop = position * tokens, (position + 1) * tokens
            spans = source_spans(run, start, stop)
            text = " ".join(words[start:stop])
            signature = json.dumps({"version": 1, "tokens": tokens, "spans": spans, "text": text},
                                   ensure_ascii=False, sort_keys=True).encode()
            ident = "passage-" + hashlib.sha256(signature).hexdigest()
            start_ref, end_ref = refs[spans[0]["verse_id"]], refs[spans[-1]["verse_id"]]
            passage = {**first,
                "id": ident, "language": first.get("language", "grc"),
                "chapter": first.get("chapter", ""), "verse": str(position + 1), "order": position + 1,
                "ref": f"{start_ref} [token {spans[0]['token_start'] + 1}] – {end_ref} [token {spans[-1]['token_end']}]",
                "text": text, "text_bare": text, "n_tokens": tokens,
                "group": first.get("group", ""), "copyist": first.get("copyist"),
                "has_gap": False, "supplied_frac": 0.0, "duplicate_of": first.get("duplicate_of"),
                "source_verse_ids": [span["verse_id"] for span in spans], "source_spans": spans,
                "unit_type": "token_passage", "_continuity_segment": run_number,
                "_continuity_index": len(passages),
            }
            passages.append(passage)
        remainder = len(words) - full * tokens
        if remainder:
            exclude("short_tail" if full else "insufficient_tokens", run, full * tokens, len(words))
        run.clear()
        run_number += 1

    for verse in annotated:
        words = verse["text_bare"].split()
        reason = ("gap" if verse.get("has_gap") else
                  "supplied_text" if verse.get("supplied_frac", 0) > 0 else
                  "empty_text" if not words else None)
        if reason:
            flush()
            exclude(reason, [(verse, words)])
            continue
        if run:
            previous = run[-1][0]
            if (not consecutive(previous, verse, bridge_chapters=bridge_chapters)
                    or any(previous.get(key) != verse.get(key) for key in _LABEL_BOUNDARIES)):
                flush()
        run.append((verse, words))
    flush()

    languages = sorted({v.get("language", "grc") for v in verses})
    counts = Counter(p["language"] for p in passages)
    for language in languages:
        if counts[language] < 3:
            exclusions.append({
                "reason": "insufficient_passages", "language": language,
                "n_passages": counts[language], "minimum_required": 3,
                "n_tokens": 0, "source_verse_ids": [], "source_spans": [],
                "note": "Generated passages are retained; clustering needs at least three observations per language.",
            })
    input_tokens = sum(len(v["text_bare"].split()) for v in verses)
    return {
        "passages": passages, "exclusions": exclusions,
        "settings": {
            "version": "raw-token-passages-v1", "tokens": tokens, "overlap_tokens": 0,
            "bridge_chapters": bridge_chapters,
            "text_field": "text_bare", "display_text": "normalized tokens; use source mappings for original display text",
            "source_offsets": "zero-based, half-open token offsets into source text_bare.split()",
            "boundaries": ("original continuity plus author, group, copyist, and duplicate identity; damaged, "
                           "supplied, and empty verses excluded"
                           + ("; chapter divisions bridged where the corpus records the two as adjacent"
                              if bridge_chapters else "; chapter divisions break a passage")),
            "input_verses": len(verses), "input_tokens": input_tokens,
            "retained_tokens": len(passages) * tokens, "excluded_tokens": sum(r["n_tokens"] for r in exclusions),
            "passages_by_language": {language: counts[language] for language in languages},
            "minimum_passages_per_language": 3, "recommended_smoothing_alpha": 0.0,
            "interpretation": "Experimental passage representation, not validated author identification.",
        },
    }


def pool_profiles(passages: list[dict], profiles: dict[str, dict], *, min_coverage: float = 0.8) -> dict:
    """Average the verse profiles inside each passage into one profile for the passage.

    Passage clustering was built on lexical features alone, which left the AI profiles - the strongest
    signal measured on this corpus, separating translated from composed Greek at d = +2.51 - out of the
    analysis entirely. A passage is a span of verses, each already profiled, so its profile is the mean
    of theirs: the six scales average, each categorical takes the value most of its verses carry, and
    the style tags are the commonest across the span.

    A passage whose verses are mostly unprofiled is omitted rather than averaged from a fragment of
    itself; ``min_coverage`` is that floor. Averaging discards the within-passage distribution of the
    categorical fields, which is real information a later feature map could use.
    """
    from .ai_profile import CATEGORICAL_DIMS, NUMERIC_DIMS

    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    pooled: dict[str, dict] = {}
    for passage in passages:
        ids = passage.get("source_verse_ids") or []
        got = [profiles[i] for i in ids if i in profiles]
        if not ids or len(got) / len(ids) < min_coverage:
            continue
        record: dict = {"id": passage["id"]}
        for dim in NUMERIC_DIMS:
            record[dim] = sum(g[dim] for g in got) / len(got)
        for dim in CATEGORICAL_DIMS:
            record[dim] = Counter(g[dim] for g in got).most_common(1)[0][0]
        tags = Counter(t for g in got for t in g.get("style_tags", []))
        record["style_tags"] = [t for t, _ in tags.most_common(8)]
        record["distinctive_phrases"] = []
        record["signature"] = f"Pooled from {len(got)} profiled verses."
        pooled[passage["id"]] = record
    return pooled
