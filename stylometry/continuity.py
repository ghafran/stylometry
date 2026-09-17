"""Keep local analyses inside observed, uninterrupted passages.

Annotate the original ordered records *before* dropping unavailable profiles.
The private index then preserves holes introduced by filtering, even when a
source has no usable verse numbers.  References are never sorted or grouped:
two separate occurrences of the same work remain separate passages.
"""
from __future__ import annotations

from numbers import Integral


_INDEX = "_continuity_index"
_SEGMENT = "_continuity_segment"
_IDENTITY = ("work", "witness", "source", "document", "document_id", "fragment")


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    return None


def consecutive(previous: dict, following: dict) -> bool:
    """Whether adjacent input records are demonstrably consecutive.

    Both sides must share their language, work, witness and source.  Gapped
    records stand alone.  Chapter/document boundaries are conservative breaks:
    the corpus does not establish that the previous chapter's tail survives.
    Numeric verse and source-order positions must each advance by one when
    available; without either usable position, continuity is not inferred.
    Annotations add a further check, so filtering cannot close an existing gap.
    """
    if not previous.get("work") or not following.get("work"):
        return False
    if previous.get("language", "grc") != following.get("language", "grc"):
        return False
    if any(previous.get(key) != following.get(key) for key in _IDENTITY):
        return False
    if previous.get("has_gap") or following.get("has_gap"):
        return False
    if previous.get("chapter") != following.get("chapter"):
        return False

    annotated = any(key in record for record in (previous, following) for key in (_INDEX, _SEGMENT))
    if annotated:
        if not all(key in record for record in (previous, following) for key in (_INDEX, _SEGMENT)):
            return False
        index_before, index_after = _integer(previous[_INDEX]), _integer(following[_INDEX])
        if index_before is None or index_after != index_before + 1:
            return False
        if previous[_SEGMENT] != following[_SEGMENT]:
            return False

    has_position = False
    for key in ("order", "verse"):
        before, after = _integer(previous.get(key)), _integer(following.get(key))
        if before is None and after is None:
            continue
        if before is None or after != before + 1:
            return False
        has_position = True
    return has_position


def annotate_continuity(verses: list[dict]) -> list[dict]:
    """Copy records and attach stable passage/index metadata before filtering.

    Existing annotations are retained: applying this helper to an already
    filtered annotated corpus must not renumber away its missing records.
    """
    out: list[dict] = []
    segment = -1
    for i, verse in enumerate(verses):
        if not i or not consecutive(verses[i - 1], verse):
            segment += 1
        rec = dict(verse)
        rec.setdefault(_INDEX, i)
        rec.setdefault(_SEGMENT, segment)
        out.append(rec)
    return out


def contiguous_runs(verses: list[dict]) -> list[list[int]]:
    """Return index groups for uninterrupted passages, preserving input order."""
    runs: list[list[int]] = []
    for i, verse in enumerate(verses):
        if not i or not consecutive(verses[i - 1], verse):
            runs.append([])
        runs[-1].append(i)
    return runs
