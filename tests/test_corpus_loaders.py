"""Parser tests for the manuscript loaders that had none.

These loaders turn the raw downloads into the verse records every later stage measures, and
``build()`` catches loader exceptions so a broken parser removes a source without failing anything.
Each test drives the real loader over a small fixture in the source's own format.

The two Text-Fabric loaders (Samaritan Pentateuch, Dead Sea Scrolls) would otherwise need a real
Text-Fabric dataset, so they run against a hand-built stand-in for the ``F``/``L`` node API.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylometry.corpus import apostolic, build, cntr, dss, mam, oshb, quran, samaritan, vaticanus

# --- fixtures in each source's own format ------------------------------------------------------

OSIS_HEAD = '<osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace"><osisText><div>'
OSIS_FOOT = "</div></osisText></osis>"

OSHB_GENESIS = OSIS_HEAD + """
<chapter osisID="Gen.1">
<verse osisID="Gen.1.1"><w>בְּ/רֵאשִׁ֖ית</w> <w>בָּרָ֣א</w> <w>אֱלֹהִ֑ים</w><seg type="x-sof-pasuq">׃</seg></verse>
<verse osisID="Gen.1.2"><w>וְ/הָ/אָ֗רֶץ</w> <w>הָיְתָ֥ה</w><note>מסורה</note><seg type="x-maqqef">־</seg><w>תֹ֨הוּ֙</w></verse>
</chapter>""" + OSIS_FOOT

MAM_ISAIAH = OSIS_HEAD + """
<verse osisID="Isa.1.1">חֲזוֹן֙ יְשַֽׁעְיָ֣הוּ בֶן־אָמ֔וֹץ</verse>
<verse osisID="Isa.1.2">שִׁמְע֤וּ שָׁמַ֙יִם֙ וְהַאֲזִ֣ינִי אֶ֔רֶץ</verse>
""" + OSIS_FOOT

# ``BBCCCVVV text`` in CNTR's MES notation; {…} marks the original hand's reading.
CNTR_P46 = """45001001 ~παυλος δουλο%σ% =χρυ =ιηυ
45001002 και ειρη^ν^η απο =θυ x{πατροσ} a{πατηρ}
not a verse line
"""

APOSTOLIC_1CLEM = """1.1 Διὰ τὰς αἰφνιδίους συμφοράς
1.2 ἀδελφοὶ βραδεῖον νομίζομεν
SUBSCRIPTION
"""

QURAN_TEXT = """1|1|بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ
1|2|ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ
2|1|الم
"""
QURAN_META = """<quran><suras>
<sura index="1" name="الفاتحة" tname="Al-Faatiha" type="Meccan" order="5"/>
<sura index="2" name="البقرة" tname="Al-Baqara" type="Medinan" order="87"/>
</suras></quran>"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- XML and text loaders ----------------------------------------------------------------------

def test_oshb_reads_verses_and_strips_morpheme_marks(tmp_path: Path) -> None:
    _write(tmp_path / "Gen.xml", OSHB_GENESIS)
    got = oshb.load(tmp_path)
    assert [v["id"] if "id" in v else (v["work"], v["chapter"], v["verse"]) for v in got] == [
        ("GEN", "1", "1"), ("GEN", "1", "2")]
    first = got[0]
    assert first["language"] == "hbo" and first["witness"] == "L" and first["work_title"] == "Genesis"
    assert "/" not in first["text"], "morpheme divider leaked into the text"
    assert first["n_tokens"] == len(first["text_bare"].split()) == 3
    assert "מסורה" not in got[1]["text"], "editorial note leaked into the verse text"
    # the display text keeps the maqaf; the bare form splits on it, so the pair counts as two tokens
    assert "־" in got[1]["text"] and got[1]["n_tokens"] == 3, got[1]["text"]


def test_oshb_orders_books_canonically_not_alphabetically(tmp_path: Path) -> None:
    _write(tmp_path / "Exod.xml", OSHB_GENESIS.replace("Gen.", "Exod."))
    _write(tmp_path / "Gen.xml", OSHB_GENESIS)
    works = [v["work"] for v in oshb.load(tmp_path)]
    assert works.index("GEN") < works.index("EXOD"), "Torah order must beat file-name order"


def test_mam_reads_the_aleppo_text(tmp_path: Path) -> None:
    _write(tmp_path / "mapm.osis.xml", MAM_ISAIAH)
    got = mam.load(tmp_path)
    assert [(v["work"], v["chapter"], v["verse"]) for v in got] == [("ISA", "1", "1"), ("ISA", "1", "2")]
    assert got[0]["witness"] == "A" and got[0]["language"] == "hbo"
    assert got[0]["text_bare"].split()[0] == "חזונ", "final nun should fold in the bare form"
    assert got[0]["n_tokens"] == 4, "maqaf splits בן־אמוץ into two bare tokens"


def test_cntr_decodes_one_manuscript_and_skips_sinaiticus(tmp_path: Path) -> None:
    class1 = tmp_path / "class1"
    _write(class1 / "P46.txt", CNTR_P46)
    _write(class1 / "01.txt", CNTR_P46)  # Sinaiticus comes from the ITSEE transcription instead
    got = cntr.load(tmp_path)
    assert {v["witness"] for v in got} == {"P46"}
    assert [(v["work"], v["chapter"], v["verse"]) for v in got] == [("ROM", "1", "1"), ("ROM", "1", "2")]
    assert "{" not in got[1]["text"] and "}" not in got[1]["text"]
    assert "χριστου" in got[0]["text"] and "ιησου" in got[0]["text"], "marked nomina sacra should expand"
    assert "θεου" in got[1]["text"], "nomen sacrum θυ should expand"
    assert "πατροσ" in got[1]["text"] and "πατηρ" not in got[1]["text"], "corrector reading must be dropped"


def test_apostolic_reads_lake_text_and_drops_editorial_headings(tmp_path: Path) -> None:
    _write(tmp_path / "001-i_clement.txt", APOSTOLIC_1CLEM)
    got = apostolic.load(tmp_path)
    assert [(v["work"], v["chapter"], v["verse"]) for v in got] == [("1CLEM", "1", "1"), ("1CLEM", "1", "2")]
    assert got[0]["witness"] == "Lake" and got[0]["language"] == "grc"
    assert all("SUBSCRIPTION" not in v["text"] for v in got)


def test_quran_reads_suras_with_meccan_medinan_metadata(tmp_path: Path) -> None:
    _write(tmp_path / "quran-uthmani.txt", QURAN_TEXT)
    _write(tmp_path / "quran-data.xml", QURAN_META)
    got = quran.load(tmp_path)
    assert [v["work"] for v in got] == ["Q001", "Q001", "Q002"]
    assert got[0]["language"] == "arb" and got[0]["witness"] == "T"
    assert got[0]["group"] == "Meccan" and got[-1]["group"] == "Medinan"
    assert "Al-Faatiha" in got[0]["work_title"]
    assert "ٱ" not in got[0]["text_bare"], "alef variants should be normalised in the bare text"


def test_swete_joins_per_word_lines_into_verses(tmp_path: Path) -> None:
    swete = tmp_path / "swete"
    _write(swete / "01.Genesis.txt",
           "1.1.1 ΕΝ\n1.1.1 ΑΡΧΗ\n1.1.1 ἐποίησεν\n1.1.2 ἡ\n1.1.2 δὲ\n1.1.2 γῆ\n")
    _write(swete / "readme.txt", "not a book, and has no leading digit")
    got = vaticanus.load_swete(swete)
    assert got, "Swete loader found nothing"
    assert {v["work"] for v in got} == {"GEN"} and got[0]["witness"] == "Swete"
    # one record per verse, with the per-word lines joined back together
    assert [(v["chapter"], v["verse"], v["n_tokens"]) for v in got] == [("1", "1", 3), ("1", "2", 3)]


# --- Text-Fabric stand-in ----------------------------------------------------------------------

class FakeF:
    """Feature accessor: ``F.<feature>.v(node)`` and ``F.otype.s(kind)``, as Text-Fabric exposes them."""

    class _Feature:
        def __init__(self, values: dict, nodes_by_type=None):
            self._values, self._by_type = values, nodes_by_type or {}

        def v(self, node):
            return self._values.get(node)

        def s(self, kind):
            return tuple(self._by_type.get(kind, ()))

    def __init__(self, nodes: dict[int, dict], by_type: dict[str, list[int]]):
        features = {k for attrs in nodes.values() for k in attrs}
        for name in features:
            setattr(self, name, self._Feature({n: a.get(name) for n, a in nodes.items()}, by_type))


class FakeL:
    """Locality: ``L.d(node, otype=...)`` for descendants, ``L.u`` for ancestors."""

    def __init__(self, children: dict[int, list[int]], otype: dict[int, str], parent: dict[int, int]):
        self._children, self._otype, self._parent = children, otype, parent

    def d(self, node, otype=None):
        out, stack = [], list(self._children.get(node, []))
        while stack:
            n = stack.pop(0)
            if otype is None or self._otype.get(n) == otype:
                out.append(n)
            stack = list(self._children.get(n, [])) + stack
        return tuple(out)

    def u(self, node, otype=None):
        n = self._parent.get(node)
        while n is not None:
            if otype is None or self._otype.get(n) == otype:
                return (n,)
            n = self._parent.get(n)
        return ()


class FakeAPI:
    def __init__(self, F, L):
        self.F, self.L = F, L


def _samaritan_api():
    """One verse of Genesis 1:1 with three words."""
    words = {3: {"otype": "word", "g_cons_utf8": "בראשית", "trailer": " "},
             4: {"otype": "word", "g_cons_utf8": "ברא", "trailer": " "},
             5: {"otype": "word", "g_cons_utf8": "אלהים", "trailer": "׃"}}
    nodes = {1: {"otype": "book", "book": "Genesis"}, 2: {"otype": "verse", "chapter": 1, "verse": 1}, **words}
    by_type = {"book": [1], "verse": [2], "word": list(words)}
    children = {1: [2], 2: list(words)}
    parent = {2: 1, **{w: 2 for w in words}}
    otype = {n: a["otype"] for n, a in nodes.items()}
    return FakeAPI(FakeF(nodes, by_type), FakeL(children, otype, parent))


def test_samaritan_builds_verses_from_text_fabric(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tf").mkdir()
    (tmp_path / "tf" / "otype.tf").write_text("", encoding="utf-8")
    monkeypatch.setattr(samaritan, "load_tf", lambda *a, **k: _samaritan_api())
    got = samaritan.load(tmp_path)
    assert len(got) == 1
    v = got[0]
    assert (v["work"], v["chapter"], v["verse"], v["witness"], v["language"]) == ("GEN", "1", "1", "SP", "hbo")
    assert v["text_bare"].split() == ["בראשית", "ברא", "אלהימ"], "final mem folds in the bare form"
    assert v["n_tokens"] == 3


def test_samaritan_returns_nothing_without_text_fabric_files(tmp_path: Path) -> None:
    assert samaritan.load(tmp_path) == []


def _dss_api(all_reconstructed: bool = False):
    """A biblical scroll with one verse of three words; ``rec`` marks reconstructed letters."""
    nodes: dict[int, dict] = {1: {"otype": "scroll", "scroll": "1Qisaa", "biblical": 1},
                              2: {"otype": "line"}}
    children: dict[int, list[int]] = {1: [2], 2: []}
    parent: dict[int, int] = {2: 1}
    node = 10
    for word_text in ("חזון", "ישעיהו", "בן"):
        w = node
        nodes[w] = {"otype": "word", "type": "glyph", "book_etcbc": "Isaiah", "chapter": 1, "verse": 1,
                    "after": " "}
        children[2].append(w)
        parent[w] = 2
        children[w] = []
        node += 1
        for ch in word_text:
            nodes[node] = {"otype": "sign", "type": "cons", "glyph": ch, "rec": all_reconstructed}
            children[w].append(node)
            parent[node] = w
            node += 1
    by_type = {"scroll": [1], "line": [2], "word": [n for n, a in nodes.items() if a["otype"] == "word"]}
    otype = {n: a["otype"] for n, a in nodes.items()}
    return FakeAPI(FakeF(nodes, by_type), FakeL(children, otype, parent))


def test_dss_builds_biblical_verse_units(tmp_path: Path, monkeypatch) -> None:
    tf = tmp_path / "tf" / "2.0.1"
    tf.mkdir(parents=True)
    (tf / "otype.tf").write_text("", encoding="utf-8")
    monkeypatch.setattr(dss, "load_tf", lambda *a, **k: _dss_api())
    got = dss.load(tmp_path)
    assert len(got) == 1
    v = got[0]
    assert (v["work"], v["chapter"], v["verse"], v["witness"]) == ("ISA", "1", "1", "1Qisaa")
    assert v["text_bare"].split() == ["חזונ", "ישעיהו", "בנ"], "final letters fold in the bare form"
    assert v["supplied_frac"] == 0.0


def test_dss_drops_units_that_are_mostly_reconstruction(tmp_path: Path, monkeypatch) -> None:
    tf = tmp_path / "tf" / "2.0.1"
    tf.mkdir(parents=True)
    (tf / "otype.tf").write_text("", encoding="utf-8")
    monkeypatch.setattr(dss, "load_tf", lambda *a, **k: _dss_api(all_reconstructed=True))
    assert dss.load(tmp_path) == [], "a fully reconstructed verse must not become an observation"


# --- build(): a failing source must be visible ---------------------------------------------------

def test_build_reports_a_failing_loader(tmp_path: Path, monkeypatch, capsys) -> None:
    """A parser that raises must be recorded, not silently reduce the corpus."""
    raw = tmp_path / "raw"
    (raw / "good").mkdir(parents=True)
    (raw / "broken").mkdir(parents=True)

    def good(_path):
        return [{"source": "good", "language": "grc", "witness": "S", "work": "MARK", "work_title": "Mark",
                 "collection": "NT", "canon": "canonical", "group": "Gospels", "chapter": "1", "verse": "1",
                 "ref": "Mark 1:1", "text": "αρχη", "text_bare": "αρχη", "n_tokens": 1,
                 "copyist": None, "supplied_frac": 0.0, "has_gap": False, "duplicate_of": None, "order": 1}]

    def broken(_path):
        raise ValueError("unreadable source file")

    monkeypatch.setattr(build, "LOADERS", [("good", good), ("broken", broken)])
    verses = build.build(raw, tmp_path / "verses.jsonl")
    assert len(verses) == 1, "the healthy source should still be parsed"

    report_path = tmp_path / "build_report.json"
    assert report_path.exists(), "build() must record what each source produced"
    report = json.loads(report_path.read_text())
    failed = {s["source"]: s for s in report["sources"] if s.get("error")}
    assert "broken" in failed, f"a raising loader must be reported: {report}"
    assert "unreadable source file" in failed["broken"]["error"]
    assert report["n_failed"] == 1
    assert any(s["source"] == "good" and s["n_units"] == 1 for s in report["sources"])
    assert "broken" in capsys.readouterr().out
