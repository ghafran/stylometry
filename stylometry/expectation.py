"""Who each work is expected to be by, for comparison with the discovered styles.

Expectations are received opinion, not measurements: a traditional or scholarly
attribution, a known title page, or an explicit statement that the hand is unknown.
Nothing here ever reaches the estimator. It is resolved when the report is written,
long after clustering has finished, so the comparison stays honest.
"""
from __future__ import annotations

from .importers.meta import APOSTOLIC_WORKS, book_meta

#: Suras revealed after the hijra, by Tanzil's classification.
MEDINAN_SURAS = frozenset({2, 3, 4, 5, 8, 9, 13, 22, 24, 33, 47, 48, 49, 55, 57, 58, 59, 60,
                           61, 62, 63, 64, 65, 66, 76, 98, 99, 110})

_APOSTOLIC_GROUPS = {code: group for code, _, group, _ in APOSTOLIC_WORKS.values()}

#: Collections whose every work is expected to come from one hand.
_SINGLE_HAND = {
    "Bukhari": "al-Bukhari's compilation",
    "Hadith Qudsi": "The Prophet's wording",
}
#: Collections of separate artefacts, each with its own unrecoverable scribe.
_ANONYMOUS = {"DSS": "Unknown scribe", "inscriptions": "Unknown hand"}


def expected_writer(verse: dict) -> tuple[str, bool]:
    """Return the expected writer of a verse's work, and whether it is actually known.

    An expectation that is not known — an anonymous scribe, an unattributed fragment —
    is still returned so the work appears in the report, but it cannot be scored.
    """
    collection, book = str(verse.get("collection")), str(verse.get("book"))
    title = str(verse.get("book_title") or book)
    if verse.get("reference_author"):
        return str(verse["reference_author"]), True
    if collection in _ANONYMOUS:
        return f"{_ANONYMOUS[collection]} ({title})", False
    if collection in _SINGLE_HAND:
        return _SINGLE_HAND[collection], True
    if collection == "Quran":
        digits = "".join(c for c in book if c.isdigit())
        if digits:
            return ("Medinan revelation" if int(digits) in MEDINAN_SURAS else "Meccan revelation"), True
        return "Quran", True
    if collection == "noncanonical":
        group = _APOSTOLIC_GROUPS.get(book)
        return (group, True) if group else (f"Anonymous ({title})", False)
    _, _, _, group = book_meta(book, str(verse.get("language")))
    if group and group != book:
        return group, True
    # A named work with no attribution is disputed rather than anonymous where the
    # collection supplies labels for its other works.
    return f"Disputed ({title})", False


def _share(counts: dict[str, int], limit: int | None = None) -> list[dict]:
    """Shares in descending order; beyond ``limit`` the tail is folded into one row."""
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rows = [{"name": name, "words": words, "share": words / total if total else 0.0}
            for name, words in (ranked[:limit] if limit else ranked)]
    if limit and len(ranked) > limit:
        rest = sum(words for _, words in ranked[limit:])
        rows.append({"name": f"{len(ranked) - limit} further", "words": rest,
                     "share": rest / total if total else 0.0, "remainder": True})
    return rows


def compare(verses: list[dict], share: float = 0.8, limit: int = 12) -> dict:
    """Cross-tabulate expected writers against discovered styles, by words.

    Writers are counted inside their own collection, because that is where their works
    are. Styles are counted across the whole language, because that is how wide a style
    is: scoring a style inside one collection would hide every writer it joined outside.
    """
    from collections import defaultdict

    writers: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"styles": defaultdict(int), "books": set(), "known": True})
    styles: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"writers": defaultdict(int), "books": set(), "collections": defaultdict(set)})
    for verse in verses:
        if not verse.get("style_id"):
            continue
        language, collection = str(verse["language"]), str(verse["collection"])
        style, book = str(verse["style_id"]), str(verse["book"])
        name, known = expected_writer(verse)
        words = verse.get("token_count") or 0
        writer = writers[(language, collection, name)]
        writer["styles"][style] += words
        writer["books"].add(book)
        writer["known"] = known
        entry = styles[(language, style)]
        entry["writers"][f"{collection} \u00b7 {name}"] += words
        entry["books"].add(book)
        entry["collections"][collection].add(book)

    languages = []
    for language in sorted({key[0] for key in writers}):
        style_rows = []
        for (owner, style), entry in sorted(styles.items()):
            if owner != language:
                continue
            spread = _share(entry["writers"], limit)
            style_rows.append({
                "style_id": style, "books": len(entry["books"]),
                "words": sum(entry["writers"].values()), "writers": spread,
                "collections": [{"collection": name, "books": len(books)}
                                for name, books in sorted(entry["collections"].items())],
                "verdict": "one writer" if len(spread) == 1
                           else ("dominated" if spread[0]["share"] >= share else "merged"),
            })
        style_rows.sort(key=lambda row: -row["words"])
        purity = {row["style_id"]: row["writers"][0] for row in style_rows}

        scopes = []
        for collection in sorted({key[1] for key in writers if key[0] == language}):
            rows = []
            for (owner, scope, name), writer in writers.items():
                if (owner, scope) != (language, collection):
                    continue
                mine = _share(writer["styles"])
                if not writer["known"]:
                    verdict = "unscored"
                elif mine[0]["share"] < share:
                    verdict = "split"
                elif purity[mine[0]["name"]]["name"] == f"{collection} \u00b7 {name}" \
                        and purity[mine[0]["name"]]["share"] >= share:
                    verdict = "recovered"
                else:
                    verdict = "merged"
                rows.append({"writer": name, "known": writer["known"], "books": len(writer["books"]),
                             "words": sum(writer["styles"].values()), "styles": mine,
                             "verdict": verdict})
            rows.sort(key=lambda row: -row["words"])
            scored = [row for row in rows if row["known"]]
            listed = rows[:limit * 4]
            total = sum(row["words"] for row in rows)
            # Credit each style to the writer it mostly covers here; that is the best case.
            local: dict[str, dict[str, int]] = {}
            for row in rows:
                for item in row["styles"]:
                    local.setdefault(item["name"], {})[row["writer"]] = item["words"]
            agreed = sum(max(counts.values()) for counts in local.values())
            scopes.append({
                "collection": collection, "words": total,
                "expected_count": len(scored), "unscored_count": len(rows) - len(scored),
                "style_count": len(local), "agreement": agreed / total if total else 0.0,
                "recovered": sum(row["verdict"] == "recovered" for row in scored),
                "split": sum(row["verdict"] == "split" for row in scored),
                "merged": sum(row["verdict"] == "merged" for row in scored),
                "writers": listed,
                "hidden": len(rows) - len(listed),
            })
        languages.append({"language": language, "styles": style_rows, "collections": scopes})
    return {"share": share, "languages": languages}
