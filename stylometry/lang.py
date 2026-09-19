"""Language-specific normalisation, tokenisation and closed-class word lists.

Supported languages (ISO 639-3 codes used as keys throughout the corpus):

    grc  Greek (Koine, uncial manuscripts and critical editions)
    hbo  Hebrew (Biblical Hebrew and Aramaic in Hebrew script; Masoretic, Qumran, Samaritan)
    arb  Arabic (Quranic Arabic, Uthmani orthography)
    eng  English (modern prose, held as a validation corpus with known authorship)

Every language exposes the same four operations: ``bare`` (a diacritic-free, case-folded,
letter-variant-neutral form for statistics), ``tokenize`` (bare word tokens), ``split_sentences``
and the two closed-class lists used by the lexical features.  Stylometric features are never
compared across languages; each language is clustered on its own.
"""
from __future__ import annotations

import re
import unicodedata

from . import greek

# --- Hebrew -------------------------------------------------------------------------------------------
# Cantillation (U+0591-05AF), points (U+05B0-05BD, 05BF, 05C1, 05C2, 05C4, 05C5, 05C7) are combining marks.
_HEB_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_HEB_TOKEN_RE = re.compile(r"[א-ת]+")
_HEB_SENT_RE = re.compile(r"(?<=[׃:.])\s+")
# Wide Hebrew letters and other presentation forms used by some editions
_HEB_PRESENTATION = {
    "ﬠ": "ע", "ﬡ": "א", "ﬢ": "ד", "ﬣ": "ה", "ﬤ": "כ", "ﬥ": "ל", "ﬦ": "ם",
    "ﬧ": "ר", "ﬨ": "ת", "שׁ": "ש", "שׂ": "ש", "שּׁ": "ש", "שּׂ": "ש",
}

HEBREW_FUNCTION_WORDS = (
    # particles, prepositions, conjunctions (bare, final letters normalised)
    "את אשר כי על אל לא כל לו לה לכ לי לנו להמ לכמ בו בה במ בהמ בכ בי עמ עד מנ גמ אמ אכ רק פנ כה כה זה זאת "
    "אלה הוא היא המ הנ המה אני אנכי אנחנו אתה אתמ יש אינ הנה עתה אז שמ פה מה מי למה מדוע איכ כנ אחרי לפני "
    "תחת בינ בעד למענ בעבור יענ אולי אכנ אפ אפס בלתי בל מאד מאוד כמו כאשר אחר אחרי ממ ממנו ממנה מהמ מכמ "
    "אליו אליה אליהמ אליכ אלי אלינו עליו עליה עליהמ עליכ עלי עלינו אתו אתה אתמ אותו אותה אותמ אתכמ אתי "
    "בתוכ מתוכ לפניו לפניה לפניהמ אצל נגד סביב מעל מתחת בקרב ולא ואל ועל ואת וכל וכי ואשר ואמ וגמ ואכ והנה "
    "ועתה ואני ואנכי והוא והיא והמ ואתה ואתמ ואלה וזה וזאת ולו ולה ולי ולנו ולהמ ולכמ ובו ובה ובמ ובהמ "
    "ועמ ועד ומנ ועליו ועליהמ ואליו ואליהמ "
    # very high-frequency verbs and formulae
    "ויהי ויאמר ויאמרו ותאמר לאמר אמר אמרה אמרו אמרתי אמרת היה היתה היו יהיה תהיה יהיו והיה והיו ויעש "
    "ויעשו וילכ וילכו ויבא ויבאו ויקמ ויקח ויצא ויצאו ויענ ויקרא וידבר ויגד ויהיו ויראו וירא וישמע וישב "
    "ויתנ ויבנ ויכ נאמ יאמר יהי ויהי־ עשה עשו עשית עשיתי עשיתמ נתנ נתתי בא באו באה יצא הלכ הלכו ראה ראו "
    "ידע ידעתי שמע שמעו דבר דברי דברתי לכ לכו קומ קח בוא צא שב"
).split()

HEBREW_SUFFIXES = [
    "ימ", "ות", "יה", "יו", "יכ", "יהמ", "יכמ", "ינו", "הו", "המ", "הנ", "כמ", "כנ", "נו", "ני", "תי", "תמ", "תנ",
    "ונ", "ינ", "ית", "ה", "י", "כ", "ו", "ת", "מ", "נ", "א",
]

# --- Arabic -------------------------------------------------------------------------------------------
_ARB_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARB_NORMALISE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٰ": "", "ى": "ي", "ؤ": "و", "ئ": "ي"})
_ARB_TOKEN_RE = re.compile(r"[ء-ي]+")
_ARB_SENT_RE = re.compile(r"(?<=[.؟!۝])\s+")

ARABIC_FUNCTION_WORDS = (
    "في من الي علي عن مع حتي ان ما لا لم لن لما اذا اذ الا او ثم بل قد كان كانوا كانت يكون تكون هو هي هم "
    "هن انت انتم نحن انا الذي التي الذين ذلك تلك هذا هذه هولاء اولئك كل بعض غير مثل كذلك هنالك هناك اين "
    "كيف متي لعل ليت كان لكن بلي نعم يا ايها ولا وما وان فان فلا ومن وقد ولقد لقد انما كلا بما عما فيما "
    "مما منهم لهم عليهم اليهم بهم فيهم لكم عليكم اليكم بكم فيكم لنا علينا الينا بنا فينا له لها عليه "
    "عليها اليه اليها به بها فيه فيها منه منها عنه عنها ولو لو اما اذن كي لكي حين حينئذ يوم يومئذ اذ عند "
    "عندهم عنده عندك عندي دون بين بينهم بينكم بيننا قبل بعد فوق تحت وراء امام لدي لديهم ليس ليسوا ليست "
    "لست ذا ذو ذي اي اية سوف سوي بلي هل اليس ولم فلم ولن فلن وهو وهي وهم فهو فهي فهم واذا فاذا واذ فاذ "
    "والذين والذي والتي وكان فكان وكانوا فكانوا وله ولها ولهم فله فلهم وبه وبها وفيه وفيها ومنه ومنها "
    "وعليه وعليهم واليه واليهم وعنه وعنهم ثمة ولئن لئن ولكن فلما ولما فاما واما وكل وبعض وكيف واين"
).split()

ARABIC_SUFFIXES = ["ون", "ين", "ات", "ان", "ها", "هم", "هن", "كم", "كن", "نا", "ني", "وا", "تم", "تن", "ية", "ة", "ه", "ك", "ي", "ت", "ا"]


# --- English ------------------------------------------------------------------------------------------
# Not a scripture language. English is here because the other three have no ground truth: nobody can
# say who wrote Isaiah, so nothing measured on it can be scored. The English corpus is chosen so that
# every text has a known author, which makes it the only place a strategy can be shown to work before
# it is pointed at something that matters.
_ENG_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
_ENG_SENT_RE = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]]*\s+")
_ENG_APOSTROPHE = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})

# Closed-class words plus the highest-frequency auxiliaries and adverbs: the words an author uses
# without choosing them, which is what makes them carry style rather than subject.
ENGLISH_FUNCTION_WORDS = (
    "the a an this that these those such same other another each every either neither any some all "
    "both half no none one two first last next own very much many few little more most less least "
    "i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his "
    "himself she her hers herself it its itself they them their theirs themselves who whom whose "
    "which what whatever whoever whichever someone somebody something anyone anybody anything "
    "everyone everybody everything no-one nobody nothing one's oneself "
    "of in to for with on at by from up down out off over under above below between among through "
    "during before after since until till while within without against toward towards upon into "
    "onto about across behind beyond beside besides around near past along amid amongst despite "
    "except inside outside per than unto via "
    "and or but nor yet so because although though unless whereas whether if lest however therefore "
    "thus hence moreover furthermore nevertheless nonetheless otherwise meanwhile besides also "
    "is are was were be been being am do does did doing done have has had having "
    "will would shall should can could may might must ought need dare used "
    "not no never nothing none neither nor hardly scarcely barely seldom rarely always often "
    "sometimes usually already still yet again ever once twice then now here there where when why "
    "how whence whither thence hither thither "
    "as so too quite rather almost enough just only even indeed perhaps maybe certainly surely "
    "well thus above below likewise accordingly consequently"
).split()

ENGLISH_SUFFIXES = [
    "ation", "ition", "ness", "ment", "ance", "ence", "able", "ible", "ical", "ously", "fully",
    "less", "ship", "hood", "ward", "wise", "ing", "ion", "ity", "ous", "ive", "ful", "ise", "ize",
    "ish", "ism", "ist", "est", "ary", "ory", "ent", "ant", "ial", "ual", "ly", "ed", "er", "or",
    "al", "ic", "es", "ty", "cy", "e", "s", "d", "y", "n", "t",
]


def _strip_marks(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def bare(s: str, lang: str) -> str:
    if lang == "grc":
        return greek.bare(s)
    if lang == "hbo":
        s = "".join(_HEB_PRESENTATION.get(ch, ch) for ch in unicodedata.normalize("NFC", s))
        s = s.replace("־", " ").replace("׃", " ")  # maqaf, sof pasuq
        return _strip_marks(s).translate(_HEB_FINALS)
    if lang == "arb":
        s = _ARB_DIACRITICS_RE.sub("", unicodedata.normalize("NFC", s))
        return s.translate(_ARB_NORMALISE)
    if lang == "eng":
        # Fold accents, fold case, and settle the apostrophe, so "don't" and "don\u2019t" are one word.
        return _strip_marks(s.translate(_ENG_APOSTROPHE)).lower()
    raise ValueError(f"unsupported language {lang!r}")


def tokenize(s: str, lang: str) -> list[str]:
    if lang == "grc":
        return greek.tokenize(s)
    if lang == "hbo":
        return _HEB_TOKEN_RE.findall(bare(s, lang))
    if lang == "arb":
        return _ARB_TOKEN_RE.findall(bare(s, lang))
    if lang == "eng":
        return _ENG_TOKEN_RE.findall(bare(s, lang))
    raise ValueError(f"unsupported language {lang!r}")


def split_sentences(s: str, lang: str) -> list[str]:
    if lang == "grc":
        return greek.split_sentences(s)
    rx = {"hbo": _HEB_SENT_RE, "arb": _ARB_SENT_RE, "eng": _ENG_SENT_RE}[lang]
    return [p.strip() for p in rx.split(s) if p.strip()]


def function_words(lang: str) -> list[str]:
    words = {"grc": greek.FUNCTION_WORDS, "hbo": HEBREW_FUNCTION_WORDS,
             "arb": ARABIC_FUNCTION_WORDS, "eng": ENGLISH_FUNCTION_WORDS}[lang]
    return list(dict.fromkeys(words))  # vectorizer vocabularies must not repeat a term


def suffixes(lang: str) -> list[str]:
    return list(dict.fromkeys({"grc": greek.SUFFIXES, "hbo": HEBREW_SUFFIXES,
                               "arb": ARABIC_SUFFIXES, "eng": ENGLISH_SUFFIXES}[lang]))


def clean_display(s: str, lang: str) -> str:
    """Whitespace-collapsed display text; keeps vowels/accents so humans and the model see the real text."""
    if lang == "grc":
        return greek.clean_display(s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


# Verse-initial connective habits per language: the token(s) that mark parataxis / a second-position particle.
INITIAL_CONNECTIVES = {
    "grc": {"first": {"και"}, "second": {"δε"}, "any": {"και", "δε", "γαρ", "ουν", "αλλα", "τε", "διο", "οθεν"}},
    "hbo": {"first": set(), "second": set(), "any": set()},  # waw is a prefix; handled by the prefix feature below
    "arb": {"first": {"و", "ف", "ثم"}, "second": set(), "any": {"و", "ف", "ثم", "بل", "لكن"}},
    "eng": {"first": {"and", "but", "so"}, "second": set(),
            "any": {"and", "but", "so", "for", "yet", "or", "nor", "then", "however", "therefore",
                    "thus", "still", "now", "besides", "moreover", "nevertheless"}},
}


def initial_waw(tokens: list[str], lang: str) -> float:
    """Hebrew/Arabic conjunction prefixes are written attached to the next word (ו-, و-/ف-)."""
    if not tokens:
        return 0.0
    first = tokens[0]
    if lang == "hbo":
        return 1.0 if first.startswith("ו") and len(first) > 1 else 0.0
    if lang == "arb":
        return 1.0 if first[0] in "وف" and len(first) > 1 else 0.0
    return 0.0


LANGUAGE_NAMES = {"grc": "Greek", "hbo": "Hebrew", "arb": "Arabic"}
