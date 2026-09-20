"""Static metadata for every work in the corpus.

``group`` is the traditional / scholarly authorship grouping.  It is never fed to the clustering;
it exists so the report can show how the discovered authors line up with received opinion.
"""
from __future__ import annotations

# code -> (title, collection, canon, group)
SINAITICUS_BOOKS: dict[str, tuple[str, str, str, str]] = {
    "FRAG": ("Fragments", "LXX", "OT", "LXX:Fragments"),
    "GEN": ("Genesis", "LXX", "OT", "LXX:Pentateuch"),
    "LEV": ("Leviticus", "LXX", "OT", "LXX:Pentateuch"),
    "NUM": ("Numbers", "LXX", "OT", "LXX:Pentateuch"),
    "DEUT": ("Deuteronomy", "LXX", "OT", "LXX:Pentateuch"),
    "JOSH": ("Joshua", "LXX", "OT", "LXX:Histories"),
    "JUDG": ("Judges", "LXX", "OT", "LXX:Histories"),
    "1CHR": ("1 Chronicles", "LXX", "OT", "LXX:Histories"),
    "2ESD": ("2 Esdras (Ezra-Nehemiah)", "LXX", "OT", "LXX:Histories"),
    "EST": ("Esther", "LXX", "OT", "LXX:Histories"),
    "TOB": ("Tobit", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "JDT": ("Judith", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "1MACC": ("1 Maccabees", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "4MACC": ("4 Maccabees", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "ISA": ("Isaiah", "LXX", "OT", "LXX:Prophets"),
    "JER": ("Jeremiah", "LXX", "OT", "LXX:Prophets"),
    "LAM": ("Lamentations", "LXX", "OT", "LXX:Prophets"),
    "JOEL": ("Joel", "LXX", "OT", "LXX:Prophets"),
    "OBAD": ("Obadiah", "LXX", "OT", "LXX:Prophets"),
    "JONAH": ("Jonah", "LXX", "OT", "LXX:Prophets"),
    "NAH": ("Nahum", "LXX", "OT", "LXX:Prophets"),
    "HAB": ("Habakkuk", "LXX", "OT", "LXX:Prophets"),
    "ZEPH": ("Zephaniah", "LXX", "OT", "LXX:Prophets"),
    "HAG": ("Haggai", "LXX", "OT", "LXX:Prophets"),
    "ZECH": ("Zechariah", "LXX", "OT", "LXX:Prophets"),
    "MAL": ("Malachi", "LXX", "OT", "LXX:Prophets"),
    "PS": ("Psalms", "LXX", "OT", "LXX:Poetry"),
    "PROV": ("Proverbs", "LXX", "OT", "LXX:Poetry"),
    "ECCL": ("Ecclesiastes", "LXX", "OT", "LXX:Poetry"),
    "CANT": ("Song of Songs", "LXX", "OT", "LXX:Poetry"),
    "WIS": ("Wisdom of Solomon", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "SIR": ("Sirach", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "JOB": ("Job", "LXX", "OT", "LXX:Poetry"),
    "MATT": ("Matthew", "NT", "NT", "Matthew"),
    "MARK": ("Mark", "NT", "NT", "Mark"),
    "LUKE": ("Luke", "NT", "NT", "Luke-Acts"),
    "JOHN": ("John", "NT", "NT", "Johannine"),
    "ROM": ("Romans", "NT", "NT", "Paul (undisputed)"),
    "1COR": ("1 Corinthians", "NT", "NT", "Paul (undisputed)"),
    "2COR": ("2 Corinthians", "NT", "NT", "Paul (undisputed)"),
    "GAL": ("Galatians", "NT", "NT", "Paul (undisputed)"),
    "EPH": ("Ephesians", "NT", "NT", "Deutero-Pauline"),
    "PHIL": ("Philippians", "NT", "NT", "Paul (undisputed)"),
    "COL": ("Colossians", "NT", "NT", "Deutero-Pauline"),
    "1THESS": ("1 Thessalonians", "NT", "NT", "Paul (undisputed)"),
    "2THESS": ("2 Thessalonians", "NT", "NT", "Deutero-Pauline"),
    "HEB": ("Hebrews", "NT", "NT", "Hebrews"),
    "1TIM": ("1 Timothy", "NT", "NT", "Pastorals"),
    "2TIM": ("2 Timothy", "NT", "NT", "Pastorals"),
    "TITUS": ("Titus", "NT", "NT", "Pastorals"),
    "PHLM": ("Philemon", "NT", "NT", "Paul (undisputed)"),
    "ACTS": ("Acts", "NT", "NT", "Luke-Acts"),
    "JAS": ("James", "NT", "NT", "James"),
    "1PET": ("1 Peter", "NT", "NT", "1 Peter"),
    "2PET": ("2 Peter", "NT", "NT", "2 Peter"),
    "1JOHN": ("1 John", "NT", "NT", "Johannine"),
    "2JOHN": ("2 John", "NT", "NT", "Johannine"),
    "3JOHN": ("3 John", "NT", "NT", "Johannine"),
    "JUDE": ("Jude", "NT", "NT", "Jude"),
    "REV": ("Revelation", "NT", "NT", "Revelation"),
    "BARN": ("Epistle of Barnabas", "noncanonical", "none", "Barnabas"),
    "HERM": ("Shepherd of Hermas", "noncanonical", "none", "Hermas"),
}

# Apostolic Fathers (Lake's text, jtauber/apostolic-fathers): file stem -> (code, title, group, duplicate_of)
APOSTOLIC_WORKS: dict[str, tuple[str, str, str, str | None]] = {
    "001-i_clement": ("1CLEM", "1 Clement", "1 Clement", None),
    "002-ii_clement": ("2CLEM", "2 Clement", "2 Clement", None),
    "003-ignatius-ephesians": ("IGN-EPH", "Ignatius to the Ephesians", "Ignatius", None),
    "004-ignatius-magnesians": ("IGN-MAG", "Ignatius to the Magnesians", "Ignatius", None),
    "005-ignatius-trallians": ("IGN-TRAL", "Ignatius to the Trallians", "Ignatius", None),
    "006-ignatius-romans": ("IGN-ROM", "Ignatius to the Romans", "Ignatius", None),
    "007-ignatius-philadelphians": ("IGN-PHLD", "Ignatius to the Philadelphians", "Ignatius", None),
    "008-ignatius-smyrnaeans": ("IGN-SMYR", "Ignatius to the Smyrnaeans", "Ignatius", None),
    "009-ignatius-polycarp": ("IGN-POL", "Ignatius to Polycarp", "Ignatius", None),
    "010-polycarp-philippians": ("POLYC", "Polycarp to the Philippians", "Polycarp", None),
    "011-didache": ("DID", "Didache", "Didache", None),
    "012-barnabas": ("BARN", "Epistle of Barnabas", "Barnabas", None),
    "013-shepherd": ("HERM", "Shepherd of Hermas", "Hermas", None),
    "014-martyrdom": ("MPOL", "Martyrdom of Polycarp", "Martyrdom of Polycarp", None),
    "015-diognetus": ("DIOG", "Epistle to Diognetus", "Diognetus", None),
}

# Hebrew Bible: code -> (title, group).  Codes match the LXX codes where the work is the same.
HEBREW_BOOKS: dict[str, tuple[str, str]] = {
    "GEN": ("Genesis", "Torah"), "EXOD": ("Exodus", "Torah"), "LEV": ("Leviticus", "Torah"),
    "NUM": ("Numbers", "Torah"), "DEUT": ("Deuteronomy", "Torah"),
    "JOSH": ("Joshua", "Former Prophets"), "JUDG": ("Judges", "Former Prophets"),
    "1SAM": ("1 Samuel", "Former Prophets"), "2SAM": ("2 Samuel", "Former Prophets"),
    "1KGS": ("1 Kings", "Former Prophets"), "2KGS": ("2 Kings", "Former Prophets"),
    "ISA": ("Isaiah", "Isaiah"), "JER": ("Jeremiah", "Jeremiah"), "EZEK": ("Ezekiel", "Ezekiel"),
    "HOS": ("Hosea", "The Twelve"), "JOEL": ("Joel", "The Twelve"), "AMOS": ("Amos", "The Twelve"),
    "OBAD": ("Obadiah", "The Twelve"), "JONAH": ("Jonah", "The Twelve"), "MIC": ("Micah", "The Twelve"),
    "NAH": ("Nahum", "The Twelve"), "HAB": ("Habakkuk", "The Twelve"), "ZEPH": ("Zephaniah", "The Twelve"),
    "HAG": ("Haggai", "The Twelve"), "ZECH": ("Zechariah", "The Twelve"), "MAL": ("Malachi", "The Twelve"),
    "PS": ("Psalms", "Psalms"), "PROV": ("Proverbs", "Wisdom"), "JOB": ("Job", "Wisdom"),
    "CANT": ("Song of Songs", "Megillot"), "RUTH": ("Ruth", "Megillot"), "LAM": ("Lamentations", "Megillot"),
    "ECCL": ("Ecclesiastes", "Wisdom"), "EST": ("Esther", "Megillot"), "DAN": ("Daniel", "Daniel"),
    "EZRA": ("Ezra", "Chronicler"), "NEH": ("Nehemiah", "Chronicler"),
    "1CHR": ("1 Chronicles", "Chronicler"), "2CHR": ("2 Chronicles", "Chronicler"),
}

# Books that exist only in Greek (Septuagint) beyond the Sinaiticus table above.
LXX_EXTRA_BOOKS: dict[str, tuple[str, str, str, str]] = {
    "EXOD": ("Exodus", "LXX", "OT", "LXX:Pentateuch"), "RUTH": ("Ruth", "LXX", "OT", "LXX:Histories"),
    "1SAM": ("1 Kingdoms (1 Samuel)", "LXX", "OT", "LXX:Histories"), "2SAM": ("2 Kingdoms (2 Samuel)", "LXX", "OT", "LXX:Histories"),
    "1KGS": ("3 Kingdoms (1 Kings)", "LXX", "OT", "LXX:Histories"), "2KGS": ("4 Kingdoms (2 Kings)", "LXX", "OT", "LXX:Histories"),
    "2CHR": ("2 Chronicles", "LXX", "OT", "LXX:Histories"), "1ESD": ("1 Esdras", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "EZRA": ("Ezra", "LXX", "OT", "LXX:Histories"), "NEH": ("Nehemiah", "LXX", "OT", "LXX:Histories"),
    "2MACC": ("2 Maccabees", "LXX", "OT-deutero", "LXX:Deuterocanon"), "3MACC": ("3 Maccabees", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "EZEK": ("Ezekiel", "LXX", "OT", "LXX:Prophets"), "DAN": ("Daniel", "LXX", "OT", "LXX:Prophets"),
    "HOS": ("Hosea", "LXX", "OT", "LXX:Prophets"), "AMOS": ("Amos", "LXX", "OT", "LXX:Prophets"), "MIC": ("Micah", "LXX", "OT", "LXX:Prophets"),
    "BAR": ("Baruch", "LXX", "OT-deutero", "LXX:Deuterocanon"), "EPJER": ("Epistle of Jeremiah", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "SUS": ("Susanna", "LXX", "OT-deutero", "LXX:Deuterocanon"), "BEL": ("Bel and the Dragon", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "PSSOL": ("Psalms of Solomon", "LXX", "OT-deutero", "LXX:Deuterocanon"), "ODES": ("Odes", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "PRMAN": ("Prayer of Manasseh", "LXX", "OT-deutero", "LXX:Deuterocanon"), "ENOCH": ("Enoch (Greek)", "LXX", "none", "Pseudepigrapha"),
    # Old Greek and Theodotion are different translations, hence different hands
    "DAN-OG": ("Daniel (Old Greek)", "LXX", "OT", "LXX:Prophets"), "DAN-TH": ("Daniel (Theodotion)", "LXX", "OT", "LXX:Prophets"),
    "SUS-OG": ("Susanna (Old Greek)", "LXX", "OT-deutero", "LXX:Deuterocanon"), "SUS-TH": ("Susanna (Theodotion)", "LXX", "OT-deutero", "LXX:Deuterocanon"),
    "BEL-OG": ("Bel and the Dragon (Old Greek)", "LXX", "OT-deutero", "LXX:Deuterocanon"), "BEL-TH": ("Bel and the Dragon (Theodotion)", "LXX", "OT-deutero", "LXX:Deuterocanon"),
}


def _norm_alias(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# Every spelling of a book name met in the sources -> code (OSIS, English, BHSA/Latin, Swete file names).
_ALIASES = {
    "GEN": "gen genesis", "EXOD": "exod exodus", "LEV": "lev leviticus", "NUM": "num numbers numeri",
    "DEUT": "deut deuteronomy deuteronomium", "JOSH": "josh joshua josua josue", "JUDG": "judg judges judices",
    "RUTH": "ruth", "1SAM": "1sam 1samuel samueli 1kingdoms regnorumi", "2SAM": "2sam 2samuel samuelii 2kingdoms regnorumii",
    "1KGS": "1kgs 1kings regesi 3kingdoms regnorumiii", "2KGS": "2kgs 2kings regesii 4kingdoms regnorumiv",
    "1CHR": "1chr 1chronicles chronicai paralipomenoni", "2CHR": "2chr 2chronicles chronicaii paralipomenonii", "EZRA": "ezra esra",
    "NEH": "neh nehemiah nehemia", "EST": "esth est esther", "JOB": "job iob", "PS": "ps psalms psalm psalmi",
    "PROV": "prov proverbs proverbia", "ECCL": "eccl ecclesiastes qohelet", "CANT": "song songofsongs songofsolomon canticum canticles cant",
    "ISA": "isa isaiah is jesaia isaias", "JER": "jer jeremiah jeremia jeremias", "LAM": "lam lamentations threni threniseulamentationes",
    "EZEK": "ezek ezekiel ezechiel", "DAN": "dan daniel", "HOS": "hos hosea osee", "JOEL": "joel", "AMOS": "amos",
    "OBAD": "obad obadiah obadia abdias", "JONAH": "jonah jona jonas", "MIC": "mic micah micha michaeas", "NAH": "nah nahum",
    "HAB": "hab habakkuk habakuk habacuc", "ZEPH": "zeph zephaniah zephania sophonias", "HAG": "hag haggai aggaeus",
    "ZECH": "zech zechariah sacharia zacharias", "MAL": "mal malachi maleachi malachias",
    "1ESD": "1esd 1esdras esdrasa", "2ESD": "2esd 2esdras esdrasb", "TOB": "tob tobit tobias",
    "JDT": "jdt judith", "1MACC": "1macc 1maccabees machabaeorumi", "2MACC": "2macc 2maccabees machabaeorumii",
    "3MACC": "3macc 3maccabees machabaeorumiii", "4MACC": "4macc 4maccabees machabaeorumiv",
    "WIS": "wis wisdom wisdomofsolomon sapientiasalomonis", "SIR": "sir sirach ecclesiasticus",
    "BAR": "bar baruch", "EPJER": "epjer epistleofjeremiah letterofjeremiah epistulajeremiae", "SUS": "sus susanna",
    "BEL": "bel belandthedragon", "PSSOL": "pssol psalmsofsolomon psalmisalomonis", "ODES": "odes odae",
    "PRMAN": "prman prayerofmanasseh", "ENOCH": "enoch 1enoch",
    "DAN-OG": "danieltranslatiograeca", "DAN-TH": "danieltheodotionisversio",
    "SUS-OG": "susannatranslatiograeca", "SUS-TH": "susannatheodotionisversio",
    "BEL-OG": "beletdracotranslatiograeca", "BEL-TH": "beletdracotheodotionisversio",
}
BOOK_ALIASES: dict[str, str] = {alias: code for code, names in _ALIASES.items() for alias in names.split()}


def book_code(name: str) -> str | None:
    """Map any book-name spelling to the corpus code, or None if unknown."""
    return BOOK_ALIASES.get(_norm_alias(name))


def book_meta(code: str, language: str) -> tuple[str, str, str, str]:
    """(title, collection, canon, group) for a work code in a language."""
    if language == "hbo" and code in HEBREW_BOOKS:
        title, group = HEBREW_BOOKS[code]
        return title, "Tanakh", "OT", group
    if code in SINAITICUS_BOOKS:
        return SINAITICUS_BOOKS[code]
    if code in LXX_EXTRA_BOOKS:
        return LXX_EXTRA_BOOKS[code]
    return code, "other", "none", code


# Witnesses: the manuscript or edition that carries a text.  A work may have several witnesses in one
# language; one is the primary and the others are kept as ``WORK@CODE`` duplicates.
# code -> (name, approximate date, sort year: negative = BC, None = modern edition)
WITNESS_INFO: dict[str, tuple[str, str, int | None]] = {
    # Hebrew
    "KH": ("Ketef Hinnom silver amulets", "c. 650–600 BC", -625),
    "KH1": ("Ketef Hinnom amulet 1", "c. 650–600 BC", -625),
    "KH2": ("Ketef Hinnom amulet 2", "c. 650–600 BC", -625),
    "N": ("Nash Papyrus", "c. 150–100 BC", -125),
    "1Qisaa": ("Great Isaiah Scroll (1QIsaᵃ)", "c. 125 BC", -125),
    "Q": ("Dead Sea Scrolls (non-biblical scrolls)", "c. 250 BC – AD 70", -90),
    "SP": ("Samaritan Pentateuch (MS Dublin CBL 751)", "AD 1225, Samaritan textual tradition", 1225),
    "A": ("Aleppo Codex (Miqra according to the Masorah)", "c. AD 930", 930),
    "L": ("Leningrad Codex (Westminster/OSHB)", "AD 1008", 1008),
    # Greek
    "P52": ("P52, Rylands papyrus (John 18)", "c. AD 125–175", 150),
    "P104": ("P104 (Matthew 21)", "c. AD 150–200", 175),
    "P4": ("P4 (Luke)", "c. AD 150–200", 175),
    "P64": ("P64/P67 (Matthew)", "c. AD 150–200", 175),
    "P66": ("P66, Bodmer II (John)", "c. AD 200", 200),
    "P46": ("P46, Chester Beatty II (Paul, Hebrews)", "c. AD 200", 200),
    "P75": ("P75, Bodmer XIV–XV (Luke, John)", "late 2nd / early 3rd c.", 210),
    "P45": ("P45, Chester Beatty I (Gospels, Acts)", "c. AD 200–250", 225),
    "P47": ("P47, Chester Beatty III (Revelation)", "3rd c.", 250),
    "P72": ("P72, Bodmer VII–VIII (1–2 Peter, Jude)", "3rd/4th c.", 300),
    "03": ("Codex Vaticanus (03)", "c. AD 325–350", 337),
    "B": ("Codex Vaticanus (03)", "c. AD 325–350", 337),
    "S": ("Codex Sinaiticus (01, ITSEE transcription)", "c. AD 330–360", 345),
    "02": ("Codex Alexandrinus (02)", "c. AD 400–440", 420),
    "04": ("Codex Ephraemi Rescriptus (04)", "5th c.", 450),
    "05": ("Codex Bezae (05)", "c. AD 400", 400),
    "032": ("Codex Washingtonianus (032)", "late 4th / 5th c.", 425),
    "Swete": ("Swete's Old Testament in Greek (text of Vaticanus)", "edition 1887–94 of a c. AD 325–350 manuscript", 337),
    "Lake": ("Apostolic Fathers, Lake's text (Tauber & Macdonald)", "edition 1912–13", None),
    "Bonnet": ("Acta Apostolorum Apocrypha, Bonnet (First1KGreek)", "edition 1898–1903", None),
    # Arabic
    "T": ("Quran, Tanzil Uthmani text", "standard text (Cairo 1924 edition)", None),
}


def witness_info(code: str) -> tuple[str, str, int | None]:
    """Name, date and sort year for a witness; unknown CNTR sigla and Qumran scrolls get generic entries."""
    if code in WITNESS_INFO:
        return WITNESS_INFO[code]
    if code.startswith("P") and code[1:].isdigit():
        info = (f"Papyrus {code}", "≤ AD 400 (CNTR class 1)", 350)
    elif code.isdigit():
        info = (f"Majuscule {code}", "≤ AD 400 (CNTR class 1)", 350)
    elif code.startswith("Mur"):
        info = (f"Murabbaʿat scroll {code}", "before AD 135", 100)
    elif code.startswith("Mas"):
        info = (f"Masada scroll {code}", "before AD 73", 50)
    elif code[:1].isdigit() and "Q" in code:
        info = (f"Qumran scroll {code}", "c. 250 BC – AD 70", -90)
    else:
        info = (code, "", None)
    WITNESS_INFO[code] = info
    return info


WITNESSES: dict[str, str] = {code: name for code, (name, _, _) in WITNESS_INFO.items()}

# First1KGreek (Bonnet's Acta Apostolorum Apocrypha): TLG author id -> (code, title, group)
FIRST1K_WORKS: dict[str, tuple[str, str, str]] = {
    "tlg0317": ("AJOHN", "Acts of John", "Acts of John"),
    "tlg2038": ("ATHOM", "Acts of Thomas", "Acts of Thomas"),
    "tlg2948": ("APHIL", "Acts of Philip", "Acts of Philip"),
    "tlg2949": ("ABARN", "Acts of Barnabas", "Acts of Barnabas"),
}
