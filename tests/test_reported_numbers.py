"""Pin the numbers a reader actually looks at.

Coverage was already high for these modules, but mutation testing showed the reported quantities were
unconstrained: flipping the assignment margin to a sum, raising the outlier threshold tenfold, dropping
the per-token normalisation of the lexical rates, measuring word starts instead of endings and
renumbering the style groups all left the suite green.  Each test below fails for exactly one of those.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from stylometry import cluster as cl
from stylometry.ai_profile import CATEGORICAL_DIMS, NUMERIC_DIMS, chunk_verses
from stylometry.features import lexical_features


def _verse(i: int, text: str, work: str = "AAA", language: str = "grc") -> dict:
    return {
        "id": f"{language}:{work}.1.{i}", "language": language, "witness": "S", "work": work,
        "work_title": work, "collection": "NT", "canon": "canonical", "group": "G",
        "chapter": "1", "verse": str(i), "ref": f"{work} 1:{i}",
        "text": text, "text_bare": text, "n_tokens": len(text.split()),
        "copyist": None, "supplied_frac": 0.0, "has_gap": False, "duplicate_of": None, "order": i,
    }


# --- assignment margin -------------------------------------------------------------------------

def test_margin_is_high_at_a_centre_and_zero_between_two_centres() -> None:
    """The margin is the relative gap to the runner-up centre, so a midpoint scores 0 and a centre 1."""
    left = np.tile(np.array([[0.0, 0.0]], dtype=np.float32), (20, 1))
    right = np.tile(np.array([[10.0, 0.0]], dtype=np.float32), (20, 1))
    midpoint = np.array([[5.0, 0.0]], dtype=np.float32)
    X = np.vstack([left, right, midpoint])
    _, conf, _ = cl.cluster(X, 2, seed=0)
    assert conf.min() >= 0.0 and conf.max() <= 1.0, "margins must stay within 0..1"
    assert conf[:40].min() > 0.95, "a point sitting on its own centre is unambiguous"
    # the midpoint joins one group and shifts its centre slightly, so the margin is near zero, not zero
    assert conf[-1] < 0.1, f"a point between the two centres should be barely assigned, got {conf[-1]}"
    assert conf[-1] < conf[:40].min() / 10, "the midpoint must be far less certain than a point on a centre"


def test_margin_falls_as_a_point_moves_towards_the_other_group() -> None:
    """Ordering matters more than the exact value: nearer the boundary must mean a smaller margin."""
    base = [[0.0, 0.0]] * 20 + [[10.0, 0.0]] * 20
    probes = [[1.0, 0.0], [3.0, 0.0], [4.5, 0.0]]
    X = np.array(base + probes, dtype=np.float32)
    _, conf, _ = cl.cluster(X, 2, seed=0)
    near, middling, boundary = conf[-3], conf[-2], conf[-1]
    assert near > middling > boundary, f"margins should decrease towards the boundary: {conf[-3:]}"


# --- style-group numbering ---------------------------------------------------------------------

def test_groups_are_numbered_from_largest_to_smallest() -> None:
    """A1 is documented as the biggest group; reports and the palette rely on it.

    The layout is chosen so that k-means' own label order (22, 15, 26, 13, 5) differs from the size
    order, which a simpler fixture hides because k-means often numbers the biggest cluster first.
    """
    rng = np.random.default_rng(0)
    sizes_in = [26, 22, 13, 15, 5]
    centres = [[0.84, -4.29, 2.89], [10.43, 7.58, -5.63], [-10.12, -4.99, 0.33],
               [-18.6, -1.75, -9.97], [-5.86, -4.35, -2.53]]
    X = np.vstack([np.tile(centres[i], (n, 1)) + rng.normal(0, 0.3, (n, 3))
                   for i, n in enumerate(sizes_in)]).astype(np.float32)
    authors, _, _ = cl.cluster(X, 5, seed=0)
    sizes = [int((authors == f"A{i}").sum()) for i in range(1, 6)]
    assert sizes == sorted(sizes, reverse=True), f"groups are not ordered by size: {sizes}"
    assert sizes == sorted(sizes_in, reverse=True)


# --- outliers ---------------------------------------------------------------------------------

def _uniform_profiles(verses: list[dict], value: float = 0.5) -> dict[str, dict]:
    return {
        v["id"]: {"id": v["id"], **{d: value for d in NUMERIC_DIMS},
                  **{d: vals[0] for d, vals in CATEGORICAL_DIMS.items()},
                  "style_tags": ["short_clauses"], "distinctive_phrases": [], "signature": "flat"}
        for v in verses
    }


def test_a_planted_outlier_is_flagged_and_ordinary_verses_are_not(tmp_path: Path) -> None:
    """outliers.csv must actually single out the verse that does not belong."""
    verses = [_verse(i, "και ο δε και ο") for i in range(1, 60)]
    verses += [_verse(i, "ινα μεν ουν τε γαρ", work="BBB") for i in range(60, 120)]
    odd = _verse(120, "ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω ω", work="BBB")
    verses.append(odd)
    cl.run(verses, None, tmp_path, k=2, window=0, kmin=2, kmax=2)
    rows = list(csv.DictReader((tmp_path / "outliers.csv").open(encoding="utf-8")))
    flagged = {r["id"] for r in rows}
    assert odd["id"] in flagged, f"the planted outlier was not flagged; flagged={sorted(flagged)}"
    assert len(flagged) < len(verses) / 4, "almost everything was flagged, so the threshold means nothing"
    assert all(float(r["z"]) > 2.5 for r in rows), "outliers.csv must only hold verses past the threshold"


def test_outlier_scores_are_recorded_for_every_verse(tmp_path: Path) -> None:
    """The per-verse score backs the flag, so it has to exist and vary."""
    fillers = ["και ο δε και ο", "και ο και δε ο", "δε και ο ο και", "και και ο δε ο"]
    verses = [_verse(i, fillers[i % len(fillers)]) for i in range(1, 40)]
    others = ["ινα μεν ουν τε γαρ", "μεν ινα γαρ ουν τε", "ουν τε ινα μεν γαρ"]
    verses += [_verse(i, others[i % len(others)], work="BBB") for i in range(40, 80)]
    cl.run(verses, _uniform_profiles(verses), tmp_path, k=2, window=0, kmin=2, kmax=2)
    rows = list(csv.DictReader((tmp_path / "verse_assignments.csv").open(encoding="utf-8")))
    zs = [float(r["outlier_z"]) for r in rows]
    assert len(zs) == len(verses)
    assert max(zs) > min(zs), "every verse got the same outlier score"


# --- lexical features --------------------------------------------------------------------------

def test_function_word_features_are_rates_not_counts() -> None:
    """A long verse and a short one with the same proportion of και must measure the same."""
    short = _verse(1, "και ο")
    long = _verse(2, "και ο και ο και ο και ο")
    X, names = lexical_features([short, long], svd_dims=0)
    col = names.index("fw:και")
    assert X[0, col] == pytest.approx(X[1, col]), "function-word features must be per-token rates"
    assert X[0, col] == pytest.approx(50.0), "rates are per 100 tokens"


def test_suffix_features_measure_word_endings() -> None:
    """A word that merely starts with the ending must not count towards it."""
    ending = _verse(1, "λογου λογου")
    starting = _verse(2, "ουτοσ ουτοσ")
    X, names = lexical_features([ending, starting], svd_dims=0)
    if "sfx:-ου" not in names:
        pytest.skip("Greek suffix list does not include -ου")
    col = names.index("sfx:-ου")
    assert X[0, col] > 0, "a word ending in -ου should score on that suffix"
    assert X[1, col] == 0, "a word only starting with ου must not score on the -ου suffix"


# --- request batching ----------------------------------------------------------------------------

def test_chunking_default_size_is_small_enough_to_answer_in_one_response() -> None:
    """The default caps a request; a silently larger default would truncate model output."""
    verses = [_verse(i, "και ο δε") for i in range(1, 121)]
    chunks = chunk_verses(verses)
    assert max(len(c) for c in chunks) <= 25, "default chunk size must stay at or below 25 units"
    assert sum(len(c) for c in chunks) == len(verses), "chunking must not drop or duplicate units"


# --- model comparison ----------------------------------------------------------------------------

def test_a_model_does_not_vote_in_its_own_consensus_score() -> None:
    """Self-voting would inflate every score and flatter whichever model is being judged."""
    from stylometry.compare import ModelSet, consensus_agreement

    ids = [f"grc:X.1.{i}" for i in range(1, 11)]

    def make(offset: float) -> dict[str, dict]:
        return {
            i: {"id": i, **{d: min(1.0, 0.05 * k + offset) for d in NUMERIC_DIMS},
                **{d: vals[0] for d, vals in CATEGORICAL_DIMS.items()},
                "style_tags": ["a"], "distinctive_phrases": [], "signature": "s"}
            for k, i in enumerate(ids)
        }

    def odd_one_out() -> dict[str, dict]:
        return {
            i: {"id": i, **{d: (0.9 if k % 2 else 0.1) for d in NUMERIC_DIMS},
                **{d: vals[-1] for d, vals in CATEGORICAL_DIMS.items()},
                "style_tags": ["b"], "distinctive_phrases": [], "signature": "s"}
            for k, i in enumerate(ids)
        }

    sets = {
        "a": ModelSet("a", None, make(0.0), "m", "b", 0.001),
        "b": ModelSet("b", None, make(0.02), "m", "b", 0.001),
        "c": ModelSet("c", None, make(0.04), "m", "b", 0.001),
        "odd": ModelSet("odd", None, odd_one_out(), "m", "b", 0.001),
    }
    got = consensus_agreement(sets, ids)
    assert got["odd"]["score"] < got["a"]["score"], "a model unlike all the others must score lower"
    assert got["odd"]["n_others"] == 3, "the model itself must be excluded from its own comparison"


# --- dashboard colours ---------------------------------------------------------------------------

def test_each_style_group_gets_its_own_colour() -> None:
    """Identity on the dashboard rides on colour plus label, so the colours must differ."""
    from stylometry.html import LIGHT, color_var

    colours = [color_var(f"A{i}") for i in range(1, len(LIGHT) + 1)]
    assert len(set(colours)) == len(colours), f"style groups share a colour: {colours}"
    assert color_var("A1") != color_var("A2")


def test_groups_beyond_the_palette_fall_back_to_the_neutral_slot() -> None:
    """Running out of hues must not wrap around and reuse A1's colour for a different group."""
    from stylometry.html import LIGHT, color_var

    beyond = color_var(f"A{len(LIGHT) + 1}")
    assert beyond == "var(--a-other)"
    assert beyond != color_var("A1")


# --- summary bookkeeping -------------------------------------------------------------------------

def test_summary_counts_match_the_assignments_written(tmp_path: Path) -> None:
    """A reader compares the headline count with the rows; they must not drift apart."""
    verses = [_verse(i, "και ο δε και ο") for i in range(1, 40)]
    verses += [_verse(i, "ινα μεν ουν τε γαρ", work="BBB") for i in range(40, 80)]
    summary = cl.run(verses, None, tmp_path, k=2, window=0, kmin=2, kmax=2)
    rows = list(csv.DictReader((tmp_path / "verse_assignments.csv").open(encoding="utf-8")))
    assert summary["n_verses"] == len(rows) == len(verses)
    assert summary["n_works"] == len({r["work"] for r in rows}) == 2
    groups = json.loads((tmp_path / "authors.json").read_text())
    assert sum(g["n_verses"] for g in groups.values()) == len(rows)
    assert sum(g["share"] for g in groups.values()) == pytest.approx(1.0)
