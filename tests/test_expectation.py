"""Received opinion about who wrote what, compared with the discovered styles."""
import pytest

from stylometry.expectation import compare, expected_writer


def verse(**overrides):
    base = dict(language='grc', collection='NT', book='ROM', book_title='Romans',
                style_id='grc-S001', token_count=100, reference_author=None)
    return {**base, **overrides}


@pytest.mark.parametrize(('row', 'expected', 'known'), [
    (verse(language='eng', collection='Novels', book='AUSTEN-EM', reference_author='Jane Austen'),
     'Jane Austen', True),
    (verse(), 'Paul (undisputed)', True),
    (verse(book='MATT', book_title='Matthew'), 'Matthew', True),
    (verse(language='arb', collection='Quran', book='Q002'), 'Medinan revelation', True),
    (verse(language='arb', collection='Quran', book='Q012'), 'Meccan revelation', True),
    (verse(language='arb', collection='Bukhari', book='BUKH01'), "al-Bukhari's compilation", True),
    (verse(language='hbo', collection='Tanakh', book='GEN'), 'Torah', True),
    (verse(language='hbo', collection='DSS', book='1QS', book_title='Dead Sea Scroll 1QS'),
     'Unknown scribe (Dead Sea Scroll 1QS)', False),
    (verse(language='eng', collection='Federalist', book='FED49', book_title='Federalist No. 49'),
     'Disputed (Federalist No. 49)', False),
])
def test_every_work_resolves_to_an_expectation_or_says_it_cannot(row, expected, known):
    assert expected_writer(row) == (expected, known)


def test_a_reference_author_always_wins_over_a_traditional_grouping():
    # The English control carries real title pages; nothing should override them.
    row = verse(language='eng', collection='Novels', book='GEN', reference_author='Charles Dickens')
    assert expected_writer(row) == ('Charles Dickens', True)


def _scope(result, language, collection):
    block = next(item for item in result['languages'] if item['language'] == language)
    return next(scope for scope in block['collections'] if scope['collection'] == collection)


def test_a_writer_is_recovered_only_when_the_style_is_almost_all_theirs():
    rows = [verse(book='ROM', style_id='grc-S001', token_count=900),
            verse(book='GAL', style_id='grc-S001', token_count=100),
            verse(book='MATT', book_title='Matthew', style_id='grc-S002', token_count=500)]
    scope = _scope(compare(rows), 'grc', 'NT')
    verdicts = {row['writer']: row['verdict'] for row in scope['writers']}
    assert verdicts == {'Paul (undisputed)': 'recovered', 'Matthew': 'recovered'}
    assert scope['recovered'] == 2 and scope['agreement'] == 1


def test_a_writer_spread_over_several_styles_is_split_and_a_shared_style_is_merged():
    rows = [verse(book='ROM', style_id='grc-S001', token_count=500),
            verse(book='GAL', style_id='grc-S002', token_count=500),
            verse(book='MATT', book_title='Matthew', style_id='grc-S002', token_count=500)]
    scope = _scope(compare(rows), 'grc', 'NT')
    verdicts = {row['writer']: row['verdict'] for row in scope['writers']}
    assert verdicts['Paul (undisputed)'] == 'split'
    assert verdicts['Matthew'] == 'merged'
    assert (scope['split'], scope['merged'], scope['recovered']) == (1, 1, 0)


def test_a_style_is_judged_across_its_whole_language_not_inside_one_collection():
    # Alone in its collection the style looks pure; the rest of the language says otherwise.
    rows = [verse(collection='NT', book='ROM', style_id='grc-S001', token_count=100),
            verse(collection='LXX', book='GEN', book_title='Genesis',
                  style_id='grc-S001', token_count=900)]
    result = compare(rows)
    style = next(row for row in result['languages'][0]['styles'] if row['style_id'] == 'grc-S001')
    assert style['verdict'] == 'dominated'
    assert [item['collection'] for item in style['collections']] == ['LXX', 'NT']
    assert _scope(result, 'grc', 'NT')['writers'][0]['verdict'] == 'merged'


def test_unknown_writers_are_listed_but_never_scored():
    rows = [verse(language='hbo', collection='DSS', book='1QS', book_title='Scroll 1QS',
                  style_id='hbo-S001', token_count=100)]
    scope = _scope(compare(rows), 'hbo', 'DSS')
    assert scope['expected_count'] == 0 and scope['unscored_count'] == 1
    assert scope['writers'][0]['verdict'] == 'unscored'
    assert scope['recovered'] == scope['split'] == scope['merged'] == 0


def test_a_long_tail_of_writers_is_folded_into_one_row():
    rows = [verse(language='hbo', collection='DSS', book=f'S{index}', book_title=f'Scroll {index}',
                  style_id='hbo-S001', token_count=10) for index in range(20)]
    style = compare(rows, limit=5)['languages'][0]['styles'][0]
    assert len(style['writers']) == 6
    assert style['writers'][-1] == {'name': '15 further', 'words': 150,
                                    'share': pytest.approx(0.75), 'remainder': True}
