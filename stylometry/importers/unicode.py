"""Small, dependency-free Unicode helpers used by the source readers."""
import re
import unicodedata

_GREEK_VARIANTS = str.maketrans({'ϲ':'σ','Ϲ':'Σ','ϐ':'β','ϑ':'θ','ϰ':'κ','ϱ':'ρ','ϕ':'φ','ϖ':'π'})
_HEBREW_FINALS = str.maketrans({'ך':'כ','ם':'מ','ן':'נ','ף':'פ','ץ':'צ'})
_ARABIC_VARIANTS = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي','ؤ':'و','ئ':'ي','ـ':''})
_PATTERNS = {'eng': re.compile(r"[a-z]+(?:'[a-z]+)*"), 'grc': re.compile(r'[Ͱ-Ͽ]+'),
             'hbo': re.compile(r'[א-ת]+'), 'arb': re.compile(r'[ء-ي]+')}


def clean_display(text: str, lang: str = 'grc') -> str:
    text = unicodedata.normalize('NFC', text)
    if lang == 'grc':
        text = text.translate(_GREEK_VARIANTS)
    return ' '.join(text.split())


def bare(text: str, lang: str = 'grc') -> str:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    if lang == 'grc':
        return text.translate(_GREEK_VARIANTS).lower().replace('ς','σ')
    if lang == 'hbo':
        return text.translate(_HEBREW_FINALS)
    if lang == 'arb':
        return text.translate(_ARABIC_VARIANTS)
    return text.lower().replace('’', "'").replace('‘', "'")


def tokenize(text: str, lang: str = 'grc') -> list[str]:
    return _PATTERNS[lang].findall(bare(text, lang))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.;·!?])\s+', text) if s.strip()]
