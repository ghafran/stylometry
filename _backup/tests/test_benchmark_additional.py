"""Reference-corpus extraction must not learn publisher or editor identity."""
import pytest

from stylometry.benchmark_additional import extract_additional_text


def extract(value: str, **kwargs) -> str:
    return extract_additional_text(value.encode(), kwargs.get("parser", "benyehuda-html-v1"),
                                   kwargs.get("language", "hbo"))


def test_benyehuda_excludes_author_title_editor_and_publisher():
    source = """<html><head><title>ספר</title><meta name="author" content="מחבר"/></head>
    <body><h1>ספר / מחבר</h1><h2>חלק ראשון</h2>
    <p>הוא <em>הלך</em> בדרך<a class="footnote" href="#fn:1">1</a> ושב.</p>
    <div class="footnotes"><ol><li><p>עורך הספר והערה</p></li></ol></div>
    <hr/>את הטקסט[ים] לעיל הפיקו מתנדבי <a>פרויקט בן־יהודה</a></body></html>"""
    assert extract(source) == "הוא הלך בדרך ושב."


def test_benyehuda_does_not_strip_names_in_actual_prose():
    assert extract("<html><p>דיבר יוסף אל דוד.</p></html>") == "דיבר יוסף אל דוד."


def test_benyehuda_preserves_prose_outside_paragraph_markup():
    assert extract("<body><br>איש הלך.\n\n<p>הוא שב.</p></body>") == "איש הלך.\n\nהוא שב."


def test_benyehuda_keeps_paragraph_boundary_and_inline_tail():
    assert extract("<p>איש<a class='footnote'>1</a>הלך.</p><p>הוא שב.</p>") == "אישהלך.\n\nהוא שב."


def test_benyehuda_rejects_new_footer_layout():
    with pytest.raises(ValueError, match="publisher footer"):
        extract("<p>סיפור</p><p>את הטקסט[ים] לעיל הפיקו מתנדבים.</p>")


@pytest.mark.parametrize("value", ["<html><h1>כותרת בלבד</h1></html>", "<p>  </p>"])
def test_benyehuda_rejects_no_prose(value):
    with pytest.raises(ValueError, match="no prose"):
        extract(value)


def test_benyehuda_requires_explicit_parser_and_normalizer():
    with pytest.raises(ValueError, match="unsupported"):
        extract("<p>סיפור</p>", parser="unknown")
    with pytest.raises(ValueError, match="requires"):
        extract("<p>סיפור</p>", language="arb")
