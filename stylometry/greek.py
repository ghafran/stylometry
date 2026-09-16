"""Greek text normalisation, tokenisation and the closed-class word lists used for lexical features.

All feature extraction works on a *bare* form of the text: lower-case, no diacritics, lunate and final
sigma folded to plain sigma.  This makes the 4th-century manuscript orthography (scriptio continua,
itacism, lunate sigma) and the modern critical editions of the apocrypha comparable.
"""
from __future__ import annotations

import re
import unicodedata

_VARIANT_LETTERS = str.maketrans(
    {"ϲ": "σ", "Ϲ": "Σ", "ϐ": "β", "ϑ": "θ", "ϰ": "κ", "ϱ": "ρ", "ϕ": "φ", "ϖ": "π"}
)
_WS_RE = re.compile(r"\s+")
# After diacritics are stripped every Greek letter lives in the basic Greek block.
_TOKEN_RE = re.compile(r"[Ͱ-Ͽ]+")
_SENTENCE_RE = re.compile(r"(?<=[.;·!?])\s+")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def clean_display(s: str) -> str:
    """NFC, standard sigma, collapsed whitespace.  Keeps accents and punctuation (for humans and the LLM)."""
    return _WS_RE.sub(" ", nfc(s).translate(_VARIANT_LETTERS)).strip()


def strip_diacritics(s: str) -> str:
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def bare(s: str) -> str:
    """Lower-case, diacritic-free, sigma-neutral form used for all statistical features."""
    return strip_diacritics(nfc(s).translate(_VARIANT_LETTERS)).lower().replace("ς", "σ")


def tokenize(s: str) -> list[str]:
    """Bare word tokens (Greek letters only)."""
    return _TOKEN_RE.findall(bare(s))


def split_sentences(s: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_RE.split(s) if p.strip()]


# --- Closed-class vocabulary (bare forms). -----------------------------------------------------------
# Function words are the classic stylometric signal: authors use them unconsciously and they are
# nearly independent of subject matter.
_ARTICLES = "ο η το του τησ τω τη τον την οι αι τα των τοισ ταισ τουσ τασ"
_CONJ_PARTICLES = (
    "και δε γαρ ουν αλλα αλλ οτι ινα μεν τε ωσ ει εαν αν ουδε μηδε ουτε μητε ητοι διο διοτι οπωσ "
    "επει επειδη ωστε καθωσ καθαπερ ωσπερ οταν οτε εωσ μη ου ουκ ουχ ουχι ναι αμην ιδου τοτε νυν "
    "νυνι παλιν ετι ηδη ευθυσ ευθεωσ ουτωσ ουτω πωσ που ποτε ποθεν εκει ωδε μαλλον μονον παντοτε "
    "αρα γε δη περ τοινυν μεντοι καιτοι ομωσ πλην ειτα επειτα λοιπον ενθα οπου οθεν αχρι μεχρι "
    "πριν αρτι σημερον αει τοτε ουπω μηπω ουκετι μηκετι πανυ σφοδρα λιαν μαλιστα ολωσ"
)
_PREPOSITIONS = (
    "εν εισ εκ εξ προσ δια δι κατα κατ καθ μετα μετ μεθ υπο υπ υφ περι επι επ εφ απο απ αφ παρα "
    "παρ υπερ αντι ανθ προ συν ανα ενωπιον εμπροσθεν οπισω χωρισ ενεκεν ενεκα χαριν εγγυσ πλησιον "
    "εξω εσω ανευ ατερ"
)
_PRONOUNS = (
    "αυτοσ αυτου αυτω αυτον αυτη αυτησ αυτην αυτο αυτοι αυτων αυτοισ αυτουσ αυται αυτα αυταισ "
    "εγω μου μοι με εμου εμοι εμε συ σου σοι σε ημεισ ημων ημιν ημασ υμεισ υμων υμιν υμασ "
    "ουτοσ τουτο τουτου τουτω τουτον ταυτα τουτων τουτοισ ταυτην ταυτησ ταυτη ουτοι αυται "
    "εκεινοσ εκεινου εκεινω εκεινον εκεινη εκεινο εκεινοι εκεινων εκεινα "
    "οσ ω ον ησ ην οισ ουσ α ων "
    "τισ τι τινοσ τινι τινα τινεσ τινων τισιν τινασ οστισ ητισ οτι "
    "ουδεισ ουδεν ουδενοσ ουδενι ουδενα μηδεισ μηδεν "
    "πασ παντα παντεσ παντων παντι παν πασησ πασιν πασαν πασα πασαι πασασ πολλα πολυ πολλοι πολλων "
    "εαυτου εαυτων εαυτοισ εαυτον εαυτω εαυτησ εαυτην εαυτουσ αλληλων αλληλοισ αλληλουσ "
    "εκαστοσ εκαστου εκαστω εκαστον ετεροσ ετερον αλλοσ αλλο αλλοι αλλα αλλων αλλα μηδενι"
)
_COMMON_VERBS = (
    "εστιν εστι εισιν ην ησαν ειναι εσται εσονται ειμι εσμεν εστε ητε ωσιν ω ωμεν "
    "γινεται εγενετο εγενοντο γεγονεν γενεσθαι γενομενοσ γενομενου γενομενων "
    "λεγει λεγων λεγοντεσ λεγουσιν ελεγεν ελεγον ειπεν ειπον ειπαν εφη λεγω "
    "εχει εχων εχειν εχομεν εχετε εχουσιν ειχεν δει εξεστιν "
    "ηλθεν ηλθον ελθων ερχεται εδωκεν εποιησεν ποιειν ποιησαι οιδα οιδεν οιδατε"
)


def _dedupe(words: str) -> list[str]:
    seen: dict[str, None] = {}
    for w in words.split():
        seen.setdefault(w, None)
    return list(seen)


FUNCTION_WORDS: list[str] = _dedupe(
    " ".join([_ARTICLES, _CONJ_PARTICLES, _PREPOSITIONS, _PRONOUNS, _COMMON_VERBS])
)

# Crude morphology by word ending.  Not a parser, but participle / infinitive / passive-aorist
# density and person-number preferences are strong author-level habits.
SUFFIXES: list[str] = [
    "ουσιν", "ουσι", "εται", "ονται", "ομαι", "ομεθα", "εσθε",
    "θη", "θησαν", "θηναι", "θεισ", "θεντεσ",
    "σαν", "σεν", "σα", "σαμεν", "σατε",
    "μενοσ", "μενοι", "μενον", "μενη", "μενων", "μενοισ", "μενουσ",
    "ντεσ", "ντοσ", "ντων", "ντι", "ντα", "ουσα", "ουσησ",
    "ειν", "σαι", "σθαι", "ναι",
    "ητε", "ωμεν", "ωσιν", "ητω", "ετω", "ατε", "ετε",
    "κεν", "κασιν", "κοτεσ", "κωσ",
    "ων", "ου", "ω", "οισ", "ουσ", "αισ", "ασ", "ησ", "ια", "ιον", "ισκοσ", "τηρ", "τησ", "τωρ", "μα", "σισ",
]
