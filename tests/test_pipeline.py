"""Smoke tests for the pieces that do not need network access or the raw downloads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stylometry import cluster as cl
from stylometry.ai_profile import CATEGORICAL_DIMS, NUMERIC_DIMS, _validate, chunk_verses, output_schema
from stylometry.corpus import first1k, inscriptions, sinaiticus, vaticanus
from stylometry.corpus.build import resolve_witnesses
from stylometry.corpus.meta import book_code, select_verses
from stylometry.greek import bare, tokenize
from stylometry.lang import bare as lbare, tokenize as ltokenize

TINY_SINAITICUS = """<?xml version="1.0" encoding="utf-8"?>
<TEI><text><body>
<pb xml:id="Q1-1r" n="1-1r" copyist="A"/>
<div xml:id="B-B34-34-MARK" n="34" type="book" title="Κατὰ Μάρκον">
<div xml:id="K-B34K1V0-34-MARK" n="1" type="chapter">
<ab xml:id="V-B34K1V0-34-MARK" n="0"><seg type="margin"><note type="running-title"><w n="1">κατα</w></note></seg></ab>
<ab xml:id="V-B34K1V1-34-MARK" n="1"><w norm="Ἀρχὴ" n="1">αρχη</w> <w norm="τοῦ" n="2">του</w>
<w norm="εὐαγγελίου" n="3">ευαγγελι<note type="hyphen"/>
<lb n="2" break="no"/>ου</w> <app><rdg type="orig" hand="firsthand"><w n="4">ιυ</w></rdg><rdg type="corr" hand="ca"><w n="4">ιησου</w></rdg></app>
<w n="5">χρι<lb n="3" break="no"/>στου</w><pc n="6">·</pc></ab>
<pb xml:id="Q1-1v" n="1-1v" copyist="D"/>
<ab xml:id="V-B34K1V2-34-MARK" n="2"><w norm="καθὼς" n="1">καθωϲ</w> <supplied><w norm="γέγραπται" n="2">γεγραπται</w></supplied> <gap extent="3" unit="chars"/></ab>
</div></div></body></text></TEI>
"""


def test_greek_normalisation():
    assert bare("Ἀρχὴ τοῦ εὐαγγελίου Ἰησοῦ Χριστοῦ") == "αρχη του ευαγγελιου ιησου χριστου"
    assert bare("λόγοϲ ὁ λόγος") == "λογοσ ο λογοσ"
    assert tokenize("καὶ ἐγένετο, ἐν τῷ καιρῷ·") == ["και", "εγενετο", "εν", "τω", "καιρω"]


def test_hebrew_and_arabic_normalisation():
    # vowels and accents stripped, final letters folded to their medial forms
    assert ltokenize("בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃", "hbo") == ["בראשית", "ברא", "אלהימ", "את", "השמימ", "ואת", "הארצ"]
    assert lbare("בֶן־אָמ֔וֹץ", "hbo") == "בנ אמוצ"  # maqaf splits, finals fold
    assert ltokenize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", "arb") == ["بسم", "الله", "الرحمن", "الرحيم"]


def test_book_codes():
    assert book_code("Isa") == book_code("Isaiah") == book_code("Jesaia") == "ISA"
    assert book_code("1Sam") == book_code("1 Samuel") == "1SAM"
    assert book_code("Song") == book_code("Song of Songs") == "CANT"
    assert book_code("nonsense") is None


def test_sinaiticus_parser(tmp_path: Path):
    xml = tmp_path / "s.xml"
    xml.write_text(TINY_SINAITICUS, encoding="utf-8")
    verses = resolve_witnesses(sinaiticus.load(xml))
    assert [v["id"] for v in verses] == ["grc:MARK.1.1", "grc:MARK.1.2"]
    v1, v2 = verses
    assert v1["text"] == "Ἀρχὴ τοῦ εὐαγγελίου ιυ χριστου·"  # first hand, hyphenated words rejoined
    assert v1["copyist"] == "A" and v2["copyist"] == "D"
    assert v2["supplied_frac"] == 0.5 and v2["has_gap"] is True
    assert v1["work_title"] == "Mark" and v1["group"] == "Mark" and v1["witness"] == "S"


def test_vaticanus_mes_cleaning():
    assert vaticanus.clean_mes("\\1186βιβλοσ γενεσεωσ =ιυ =χυ /υιου δαυειδ") == "βιβλοσ γενεσεωσ ιησου χριστου υιου δαυειδ"
    assert vaticanus.clean_mes("ζα/ρε εκ x{ερρε%θη} {ερρηθη} τησ") == "ζαρε εκ ερρεθη τησ"
    assert vaticanus.clean_mes("τα {} a{εργα} αυτου το¯") == "τα αυτου τον"


def test_witness_resolution():
    a = [{"language": "grc", "witness": "S", "work": "ISA", "work_title": "Isaiah", "chapter": "1", "verse": "1", "n_tokens": 5}]
    b = [{"language": "grc", "witness": "Swete", "work": "ISA", "work_title": "Isaiah", "chapter": "1", "verse": str(i), "n_tokens": 5} for i in range(1, 4)]
    out = resolve_witnesses(a + b)
    assert out[0]["work"] == "ISA@S" and out[0]["duplicate_of"] == "ISA" and out[0]["id"] == "grc:ISA@S.1.1"
    assert out[1]["work"] == "ISA" and out[1]["duplicate_of"] is None


def test_inscriptions_parser(tmp_path: Path):
    (tmp_path / "nash.txt").write_text("SOURCE: x\n01  [אנכי י]הוה אלהיך אשר [הוצא]תיך מארץ מ[צרים]\n02  [לוא יהיה ל]ך אלהים אחרים\n", encoding="utf-8")
    recs = inscriptions.load(tmp_path)
    assert len(recs) == 1 and recs[0]["work"] == "NASH" and recs[0]["n_tokens"] == 12
    assert 0.3 < recs[0]["supplied_frac"] < 0.7


def test_first1k_segmentation():
    text = " ".join(["Καὶ εἶπεν ὁ ἀπόστολος ταῦτα πάντα τοῖς ἀδελφοῖς αὐτοῦ."] * 6)
    units = first1k.segment(text, target=25, min_tokens=10)
    assert 2 <= len(units) <= 3
    assert all(len(tokenize(u)) >= 10 for u in units)


def _fake_verses(n: int, work: str, chapter_size: int = 10, language: str = "grc") -> list[dict]:
    words = "και ο θεος ειπεν εγενετο δε αυτου εν τω λογω".split()
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        text = " ".join(rng.choice(words, size=12))
        out.append({
            "id": f"{language}:{work}.{i // chapter_size + 1}.{i % chapter_size + 1}", "source": "x", "language": language, "witness": "T",
            "work": work, "work_title": work, "collection": "NT", "canon": "NT", "group": work,
            "chapter": str(i // chapter_size + 1), "verse": str(i % chapter_size + 1),
            "ref": f"{work} {i}", "text": text, "text_bare": text, "n_tokens": 12, "copyist": "A" if i % 2 else "B",
            "supplied_frac": 0.0, "has_gap": False, "duplicate_of": None, "order": i + 1,
        })
    return out


def test_chunking_respects_work_and_size():
    verses = _fake_verses(40, "AAA") + _fake_verses(7, "BBB")
    chunks = chunk_verses(verses, size=25)
    assert all(len({v["work"] for v in c}) == 1 for c in chunks)
    assert all(len(c) <= 25 for c in chunks)
    assert sum(len(c) for c in chunks) == 47


def test_schema_and_validation():
    schema = output_schema()
    assert schema["properties"]["profiles"]["items"]["required"][0] == "id"
    verses = _fake_verses(2, "AAA")
    data = {"profiles": [
        {"id": f"unit_{i:04d}", **{d: .2 for d in NUMERIC_DIMS},
         **{d: values[0] for d, values in CATEGORICAL_DIMS.items()},
         "style_tags": ["short_clauses"], "distinctive_phrases": [], "signature": "Short clauses."}
        for i in range(1, 3)
    ]}
    recs = _validate(verses, data, "m", "b")
    assert [r["id"] for r in recs] == [v["id"] for v in verses]
    assert recs[0]["model"] == "m"
    assert recs[0]["provenance"]["request_group"] == recs[1]["provenance"]["request_group"]


def test_coerce_profile_and_json_prompt():
    from stylometry.ai_profile import coerce_profile, json_mode_system_prompt, system_prompt

    # Missing measurements must not be replaced with plausible-looking values.
    assert coerce_profile({"id": "unit_0001", "register": "1.7", "hypotaxis": None}) is None
    assert coerce_profile("junk") is None
    prompt = json_mode_system_prompt("hbo")
    assert "json" in prompt and '"profiles"' in prompt and "wayyiqtol" in prompt
    assert "Quranic" in system_prompt("arb") and "Koine" in system_prompt("grc")


def test_select_verses_scope():
    verses = _fake_verses(3, "AAA")
    verses[0]["collection"] = "LXX"
    assert len(select_verses(verses, "christian")) == 2
    assert len(select_verses(verses, "all", works=["aaa"])) == 3
    assert len(select_verses(verses, "all", language="hbo")) == 0
    with pytest.raises(ValueError):
        select_verses(verses, "nope")


def test_cluster_end_to_end(tmp_path: Path):
    verses = _fake_verses(120, "AAA") + _fake_verses(120, "BBB")
    for v in verses[120:]:  # give BBB a different habit so two clusters exist
        v["text"] = v["text_bare"] = v["text_bare"].replace("και", "δε γαρ")
    profiles = {
        v["id"]: {
            "id": v["id"], **{d: (0.2 if v["work"] == "AAA" else 0.8) for d in NUMERIC_DIMS},
            **{d: vals[0] for d, vals in CATEGORICAL_DIMS.items()},
            "style_tags": ["idou", "asyndeton"] if v["work"] == "AAA" else ["hina_clause", "men_de", "long_period"],
            "distinctive_phrases": [], "signature": "Synthetic style control.",
        }
        for v in verses
    }
    summary = cl.run(verses, profiles, tmp_path, kmin=2, kmax=4, window=2)
    assert summary["k_used"] in (2, 3, 4) and summary["language"] == "grc"
    assert (tmp_path / "verse_assignments.csv").exists()
    authors = json.loads((tmp_path / "authors.json").read_text())
    assert set(authors) >= {"A1", "A2"}
    assert summary["validation"]["purity_by_work"]["AAA"] > 0.8


def test_witness_comparison():
    from stylometry.witnesses import compare

    base = {"language": "grc", "work": "MARK", "work_title": "Mark", "chapter": "1", "verse": "1", "ref": "Mark 1:1", "n_tokens": 5, "order": 1}
    prim = dict(base, witness="S", text="αρχη του ευαγγελιου ιησου χριστου", text_bare="αρχη του ευαγγελιου ιησου χριστου", duplicate_of=None)
    sec = dict(base, witness="03", work="MARK@03", text="αρχη του ευαγγελιου ιησου", text_bare="αρχη του ευαγγελιου ιησου", duplicate_of="MARK")
    same = dict(base, witness="P45", work="MARK@P45", text=prim["text"], text_bare=prim["text_bare"], duplicate_of="MARK")
    summary, rows = compare([prim, sec, same])
    by = {r["witness"]: r for r in rows}
    assert by["03"]["similarity"] < 1.0 and "[-χριστου-]" in by["03"]["diff"]
    assert by["P45"]["similarity"] == 1.0
    s = {r["witness"]: r for r in summary}
    assert s["S"]["is_primary_for"] == "MARK" and s["03"]["compared_verses"] == 1
    assert summary[0]["witness"] == "P45"  # oldest first


def test_lexical_features_reject_mixed_languages():
    from stylometry.features import lexical_features

    with pytest.raises(ValueError):
        lexical_features(_fake_verses(5, "A") + _fake_verses(5, "B", language="hbo"))


def test_compare_models_metrics():
    from stylometry.compare import ModelSet, ensemble, pair_agreement, pareto, recommend
    from stylometry.ai_profile import NUMERIC_DIMS, CATEGORICAL_DIMS

    def prof(i, shift=0.0, label="narrative"):
        rec = {"id": i, **{d: min(1.0, 0.1 * i + shift) for d in NUMERIC_DIMS}}
        rec.update({d: label for d in CATEGORICAL_DIMS})
        rec["style_tags"] = ["a", "b"] if i % 2 else ["a"]
        return rec

    ids = [f"grc:X.1.{i}" for i in range(1, 9)]
    A = {i: prof(k) for k, i in enumerate(ids)}
    B = {i: prof(k, shift=0.05) for k, i in enumerate(ids)}  # same ranking, offset scale
    C = {i: prof(k, label="speech" if k % 2 else "narrative") for k, i in enumerate(ids)}
    agree = pair_agreement(A, B, ids)
    assert agree["n"] == 8 and abs(agree["r_mean"] - 1.0) < 1e-9 and agree["cat_mean"] == 1.0
    assert abs(agree["mad_by_dim"]["register"] - 0.05) < 1e-9
    assert pair_agreement(A, C, ids)["cat_mean"] == 0.5
    ens = ensemble("a×2", [ModelSet("a", None, A, "m", "b", 0.001), ModelSet("a_retest", None, B, "m", "b", 0.001)])
    assert ens.n == 8 and abs(ens.profiles[ids[0]]["register"] - 0.025) < 1e-9 and ens.cost_per_verse == 0.002
    rows = [
        {"name": "big", "score": 0.9, "cost": 0.01},
        {"name": "cheap", "score": 0.85, "cost": 0.0003},
        {"name": "bad", "score": 0.6, "cost": 0.005},   # dominated by both
        {"name": "nocost", "score": 0.95, "cost": None},
    ]
    assert pareto(rows) == {"big", "cheap"}
    rec = recommend(rows)
    assert rec["best"] == "big" and rec["value"] == "cheap" and rec["cheapest"] == "cheap"
