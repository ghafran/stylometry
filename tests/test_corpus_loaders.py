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


def test_a_reference_a_manuscript_reads_twice_keeps_both_with_distinct_ids():
    """Codex Sinaiticus carries a double text of 1 Chronicles 17-18 and Vaticanus two readings at
    Romans 4:4-5. Both copies belong in a witness corpus, and duplicate ids break passage building,
    so the later occurrence is suffixed rather than dropped or silently overwriting the first."""
    from stylometry.corpus.build import _disambiguate_repeated_references

    verses = [
        {"id": "grc:1CHR@S.17.14", "witness": "S", "ref": "1 Chronicles 17:14", "text_bare": "first"},
        {"id": "grc:1CHR@S.17.14", "witness": "S", "ref": "1 Chronicles 17:14", "text_bare": "second"},
        {"id": "grc:JOHN.1.1", "witness": "S", "ref": "John 1:1", "text_bare": "only"},
    ]
    _disambiguate_repeated_references(verses)
    assert [v["id"] for v in verses] == ["grc:1CHR@S.17.14", "grc:1CHR@S.17.14#2", "grc:JOHN.1.1"]
    assert verses[1]["repeated_reference"] == 2
    assert "repeated_reference" not in verses[0] and "repeated_reference" not in verses[2]
    assert verses[0]["text_bare"] == "first" and verses[1]["text_bare"] == "second", "neither is lost"


# --- Sahih al-Bukhari: the matn, without the chain of transmitters --------------------------------

# Real shapes from the edition, shortened. The first is the classic "actions are by intentions",
# whose chain runs five links deep and ends with a Companion; the second ends "... from Abu Hurayra,
# he said: the Messenger of God said", the commonest shape in the collection; the third is a report
# whose body opens with its own speech verb, which must survive; the fourth is a continuation report
# carrying no chain at all.
BUKHARI_EDITION = {
    "metadata": {"name": "Sahih al Bukhari", "sections": {"0": "", "1": "Revelation", "2": "Belief"}},
    "hadiths": [
        {"hadithnumber": 1, "arabicnumber": 1, "reference": {"book": 1, "hadith": 1}, "grades": [],
         "text": "حَدَّثَنَا الْحُمَيْدِيُّ عَبْدُ اللَّهِ بْنُ الزُّبَيْرِ ، قَالَ : حَدَّثَنَا سُفْيَانُ ، قَالَ : "
                 "أَخْبَرَنِي مُحَمَّدُ بْنُ إِبْرَاهِيمَ التَّيْمِيُّ ، أَنَّهُ سَمِعَ عَلْقَمَةَ بْنَ وَقَّاصٍ اللَّيْثِيَّ ، "
                 "يَقُولُ : سَمِعْتُ عُمَرَ بْنَ الْخَطَّابِ رَضِيَ اللَّهُ عَنْهُ عَلَى الْمِنْبَرِ، قَالَ : سَمِعْتُ رَسُولَ اللَّهِ "
                 "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ، يَقُولُ : \" إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ \""},
        {"hadithnumber": 2, "arabicnumber": 2, "reference": {"book": 2, "hadith": 1}, "grades": [],
         "text": "حَدَّثَنِي إِسْحَاقُ، أَخْبَرَنَا عَبْدُ الرَّزَّاقِ، عَنْ هَمَّامٍ، عَنْ أَبِي هُرَيْرَةَ ـ رضى الله عنه ـ "
                 "قَالَ قَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم \" كُلُّ سُلاَمَى عَلَيْهِ صَدَقَةٌ \""},
        {"hadithnumber": 3, "arabicnumber": 3, "reference": {"book": 2, "hadith": 2}, "grades": [],
         "text": "حَدَّثَنَا آدَمُ، قَالَ حَدَّثَنَا شُعْبَةُ، عَنْ أَبِي سَعِيدٍ الْخُدْرِيِّ، "
                 "قَالَتِ النِّسَاءُ لِلنَّبِيِّ صلى الله عليه وسلم غَلَبَنَا عَلَيْكَ الرِّجَالُ"},
        {"hadithnumber": 4, "arabicnumber": 4, "reference": {"book": 2, "hadith": 3}, "grades": [],
         "text": "وَبِإِسْنَادِهِ قَالَ لاَ يَبُولَنَّ أَحَدُكُمْ فِي الْمَاءِ الدَّائِمِ"},
        # The edition lists 311 records under book 0, 307 of them verbatim repeats of a record that
        # does carry a book. Keeping them would count the same report twice.
        {"hadithnumber": 5, "arabicnumber": 5, "reference": {"book": 0, "hadith": 1}, "grades": [],
         "text": "حَدَّثَنَا آدَمُ، قَالَ حَدَّثَنَا شُعْبَةُ، عَنْ أَبِي سَعِيدٍ الْخُدْرِيِّ، "
                 "قَالَتِ النِّسَاءُ لِلنَّبِيِّ صلى الله عليه وسلم غَلَبَنَا عَلَيْكَ الرِّجَالُ"},
    ],
}


def _bukhari_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bukhari"
    d.mkdir()
    (d / "ara-bukhari.json").write_text(json.dumps(BUKHARI_EDITION, ensure_ascii=False), encoding="utf-8")
    return d


def test_bukhari_drops_the_chain_and_keeps_the_report(tmp_path: Path) -> None:
    from stylometry.corpus import bukhari

    rows = bukhari.load(_bukhari_dir(tmp_path))
    assert len(rows) == 4
    first = rows[0]
    # Not one transmitter survives: the chain named five and none of them is the report.
    for name in ("الحميدي", "سفيان", "محمد", "علقمة", "عمر"):
        assert name not in first["text_bare"], f"{name} is a transmitter, not the report"
    assert "الاعمال" in first["text_bare"] and "بالنيات" in first["text_bare"]
    assert first["isnad_tokens"] > 10, "the chain was long and all of it was counted"


def test_bukhari_keeps_a_speech_verb_that_belongs_to_the_report(tmp_path: Path) -> None:
    """"... from Abu Sa'id: *the women said* to the Prophet" - drop that verb and the sentence
    loses its subject. The doubled verb of "he said: the Messenger said" is the chain's and goes."""
    from stylometry.corpus import bukhari

    rows = bukhari.load(_bukhari_dir(tmp_path))
    third = next(r for r in rows if r["verse"] == "3")
    assert third["text_bare"].startswith("قالت النساء"), third["text_bare"]
    second = next(r for r in rows if r["verse"] == "2")
    assert second["text_bare"].startswith("قال رسول الله"), second["text_bare"]
    assert not second["text_bare"].startswith("قال قال")


def test_bukhari_strips_the_honorific_formulae(tmp_path: Path) -> None:
    """Repeated thousands of times, and citation apparatus rather than anyone's style."""
    from stylometry.corpus import bukhari

    for row in bukhari.load(_bukhari_dir(tmp_path)):
        assert "صلي الله عليه وسلم" not in row["text_bare"]
        assert "رضي الله عنه" not in row["text_bare"]


def test_bukhari_leaves_a_report_with_no_chain_alone(tmp_path: Path) -> None:
    """"And with the same chain" carries none of its own; cutting a prefix would eat the report."""
    from stylometry.corpus import bukhari

    fourth = next(r for r in bukhari.load(_bukhari_dir(tmp_path)) if r["verse"] == "4")
    assert "يبولن" in fourth["text_bare"] and "الماء" in fourth["text_bare"]
    assert fourth["isnad_tokens"] <= 2


def test_bukhari_makes_each_book_a_work_and_keeps_the_quoted_speech(tmp_path: Path) -> None:
    """One work per kitab, so whole-work holdout has works to hold out, as the Qur'an has suras."""
    from stylometry.corpus import bukhari

    rows = bukhari.load(_bukhari_dir(tmp_path))
    assert {r["work"] for r in rows} == {"BUKH01", "BUKH02"}
    assert {r["collection"] for r in rows} == {"Bukhari"}
    assert {r["language"] for r in rows} == {"arb"}
    assert next(r for r in rows if r["work"] == "BUKH01")["group"] == "Revelation"
    assert all(r["witness"] == "H" for r in rows)
    # The edition's own marking of direct speech is carried, but is not what set the boundary.
    assert "الْأَعْمَالُ" in rows[0]["text_quoted"]
    assert rows[2]["text_quoted"] == "", "no quotation marks in that report"


def test_bukhari_returns_nothing_without_the_edition(tmp_path: Path) -> None:
    from stylometry.corpus import bukhari

    assert bukhari.load(tmp_path / "absent") == []


def test_bukhari_drops_the_records_the_edition_files_under_no_book(tmp_path: Path) -> None:
    """Almost all of them repeat a report that is already there under its real book."""
    from stylometry.corpus import bukhari

    rows = bukhari.load(_bukhari_dir(tmp_path))
    assert len(rows) == 4, "the book-0 record is not a fifth report"
    assert "BUKH00" not in {r["work"] for r in rows}
    bodies = [r["text_bare"] for r in rows]
    assert len(bodies) == len(set(bodies)), "and nothing is counted twice"


def test_bukhari_labels_the_prophets_own_quoted_words(tmp_path: Path) -> None:
    """Which reports quote him, and which quote somebody else inside a report about him.

    The edition marks direct speech but not whose it is. In a long report the marked spans belong to
    several mouths, so a span counts as his only when he is named as its speaker just before it.
    """
    from stylometry.corpus import bukhari

    rows = bukhari.load(_bukhari_dir(tmp_path))
    by = {r["verse"]: r for r in rows}

    # "I heard the Messenger of God say: ..." - his, although this record never closes its quotation
    # mark, which is why the span finder has to handle that case.
    assert by["1"]["attribution"] == "prophet"
    # text_prophet keeps the vowel points for display; compare on the bare form.
    from stylometry.lang import tokenize
    assert "الاعمال" in tokenize(by["1"]["text_prophet"], "arb")
    assert by["1"]["n_prophet_tokens"] > 0

    # "... the Messenger of God said: 'every joint ...'" - his.
    assert by["2"]["attribution"] == "prophet"
    # "the women said to the Prophet ..." - a report with no quoted speech at all.
    assert by["3"]["attribution"] == "report"
    assert by["3"]["text_prophet"] == ""


def test_bukhari_does_not_call_a_span_his_when_someone_else_is_speaking(tmp_path: Path) -> None:
    """Only 56% of the collection's quoted spans are preceded by him named as the speaker."""
    from stylometry.corpus import bukhari

    # A quoted span whose speaker is a Companion, not the Prophet.
    text = ("حَدَّثَنَا آدَمُ، عَنْ أَبِي سَعِيدٍ، قَالَ عُمَرُ \" لَا أَدْرِي \" ثُمَّ انْصَرَفَ")
    assert bukhari.prophet_spans(text) == []
    assert bukhari.attribution_of(text, ["لَا أَدْرِي"], []) == "other"


# --- Forty Hadith Qudsi ---------------------------------------------------------------------------

QUDSI_EDITION = {
    "metadata": {"name": "Forty Hadith Qudsi", "sections": {"1": "Forty Hadith Qudsi"}},
    "hadiths": [
        {"hadithnumber": 1, "arabicnumber": 1, "reference": {"book": 1, "hadith": 1}, "grades": [],
         "text": "عَنْ أَبِي هُرَيْرَةَ قَالَ: قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: "
                 "إِنَّ رَحْمَتِي تَغْلِبُ غَضَبِي رواه مسلم (وكذلك البخاري والنسائي وابن ماجه)"},
        {"hadithnumber": 2, "arabicnumber": 2, "reference": {"book": 1, "hadith": 2}, "grades": [],
         "text": "عَنْ أَبِي هُرَيْرَةَ رَضِيَ اللَّهُ عَنْهُ، عَنْ النَّبِيِّ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ قَالَ:<br>"
                 "قَالَ اللَّهُ تَعَالَى: كَذَّبَنِي ابْنُ آدَمَ وَلَمْ يَكُنْ لَهُ ذَلِكَ رواه البخاري"},
    ],
}


def _qudsi_dir(tmp_path: Path) -> Path:
    d = tmp_path / "qudsi"
    d.mkdir()
    (d / "ara-qudsi.json").write_text(json.dumps(QUDSI_EDITION, ensure_ascii=False), encoding="utf-8")
    return d


def test_qudsi_drops_the_chain_the_markup_and_the_closing_citation(tmp_path: Path) -> None:
    from stylometry.corpus import qudsi

    rows = qudsi.load(_qudsi_dir(tmp_path))
    assert len(rows) == 2
    for row in rows:
        assert "رواه" not in row["text_bare"], "the 'narrated by X' citation is not the report"
        assert "مسلم" not in row["text_bare"] and "البخاري" not in row["text_bare"]
        assert "<br" not in row["text"], "this edition carries markup"
        assert "هريرة" not in row["text_bare"], "the transmitter is not the report"
    assert "رحمتي" in rows[0]["text_bare"] and "غضبي" in rows[0]["text_bare"]
    assert "كذبني" in rows[1]["text_bare"]


def test_qudsi_is_one_work_labelled_as_the_speech_of_god(tmp_path: Path) -> None:
    """Its whole point is the category: God's speech in the Prophet's wording."""
    from stylometry.corpus import qudsi

    rows = qudsi.load(_qudsi_dir(tmp_path))
    assert {r["collection"] for r in rows} == {"Hadith Qudsi"}
    assert {r["work"] for r in rows} == {"QUDSI"}, "too small to be more than one work"
    assert {r["attribution"] for r in rows} == {"divine"}
    assert {r["language"] for r in rows} == {"arb"}
    assert {r["witness"] for r in rows} == {"Q"}
    assert [r["ref"] for r in rows] == ["Hadith Qudsi 1", "Hadith Qudsi 2"]


def test_qudsi_keeps_a_citation_word_that_is_not_a_closing_note(tmp_path: Path) -> None:
    """A citation is a short list of names. The same word with a paragraph after it is the report."""
    from stylometry.corpus import qudsi

    followed_by_a_report = ["a"] * 10 + ["رواه"] + ["b"] * 40
    _, kept = qudsi.strip_source_note(list(followed_by_a_report), list(followed_by_a_report))
    assert len(kept) == 51, "too much follows it to be a citation"

    followed_by_names = ["a"] * 10 + ["رواه"] + ["b"] * 4
    _, cut = qudsi.strip_source_note(list(followed_by_names), list(followed_by_names))
    assert len(cut) == 10, "a short tail of names is the citation"


def test_qudsi_returns_nothing_without_the_edition(tmp_path: Path) -> None:
    from stylometry.corpus import qudsi

    assert qudsi.load(tmp_path / "absent") == []
