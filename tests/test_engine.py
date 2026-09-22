from copy import deepcopy

import numpy as np
import pytest

from stylometry.engine import Config, _sampled_silhouette, analyze, make_passages, make_rollups


def verse(ident, text, **kwargs):
    return dict(id=ident, language='eng', collection='One', book='B1', book_title='A book',
                chapter='1', verse=ident, text=text, reference_author=None, **kwargs)


def test_book_and_gap_boundaries_keep_every_unit():
    records = [verse('a', 'the word ' * 15), verse('b', 'the word ' * 5, has_gap=True),
               verse('c', 'the word ' * 10), {**verse('d', 'other text ' * 15), 'book': 'B2'},
               verse('e', '')]
    passages, counts = make_passages(records, Config(passage_tokens=60, min_tokens=20))
    assert sorted(i for p in passages for i in p['verse_ids']) == list('abcde')
    assert any(p['verse_ids'] == ['b'] and not p['eligible'] for p in passages)
    assert all(len({next(v['book'] for v in records if v['id'] == i) for i in p['verse_ids']}) == 1
               for p in passages)
    assert counts['e'] == 0


def test_tail_context_merges_only_inside_book():
    records = [verse('a', 'the ' * 60), verse('b', 'end ' * 5)]
    passages, _ = make_passages(records, Config(passage_tokens=60, min_tokens=20))
    assert len(passages) == 1
    assert passages[0]['tokens'] == 65


def test_too_short_still_gets_explicit_tag_and_rollup():
    result = analyze([verse('a', 'tiny text')])
    row = result['verses'][0]
    assert row['style_id'] is None and row['status'] == 'insufficient_text'
    assert len(result['rollups']) == 5
    assert all(r['style_count'] == 0 and r['insufficient_verse_count'] == 1 for r in result['rollups'])


def test_language_ids_shared_across_collections_but_never_languages():
    records = [verse('a', 'the text ' * 30),
               {**verse('b', 'the text ' * 30), 'collection': 'Two', 'book': 'B2'},
               {**verse('c', 'the text ' * 30), 'language': 'grc'}]
    result = analyze(records, Config(passage_tokens=60, min_tokens=20))
    a, b, c = result['verses']
    assert a['style_id'] == b['style_id']
    assert a['style_id'] != c['style_id']
    assert result['styles'][0]['collection_count'] == 2
    language_row = next(r for r in result['rollups'] if r['level'] == 'language' and r['language'] == 'eng')
    assert language_row['style_count'] == 1  # union, not sum of each collection


def test_known_labels_never_enter_discovery():
    texts = ['the and to it was he ' * 10, 'she had her but that very ' * 10] * 6
    records = [{**verse(str(i), t), 'book': str(i)} for i, t in enumerate(texts)]
    changed = deepcopy(records)
    for i, v in enumerate(changed):
        v['reference_author'] = f'Invented {i}'
        v['book_title'] = f'Style {i}'
    config = Config(passage_tokens=60, min_tokens=20, max_styles=2)
    before, after = analyze(records, config), analyze(changed, config)
    assert [v['style_id'] for v in before['verses']] == [v['style_id'] for v in after['verses']]
    assert before['languages'] == after['languages']
    assert records[0].get('style_id') is None  # do not mutate input


def test_constant_text_does_not_force_split():
    records = [{**verse(str(i), 'the same words again ' * 20), 'book': str(i)} for i in range(12)]
    result = analyze(records, Config(passage_tokens=60, min_tokens=20))
    assert result['languages'][0]['estimated_styles'] == 1
    assert result['languages'][0]['selection']['candidate_scores'] == []


def test_repeating_same_style_at_different_lengths_cannot_create_styles():
    records = [{**verse(str(i), 'the and to it was he ' * n), 'book': str(i)}
               for i, n in enumerate(range(200, 260))]
    result = analyze(records)
    assert result['languages'][0]['estimated_styles'] == 1


def test_silhouette_sample_retains_tiny_group():
    x = np.zeros((2000, 2))
    labels = np.zeros(2000, dtype=int)
    x[[0, 1, 3]] = 2
    labels[[0, 1, 3]] = 1
    assert _sampled_silhouette(x, labels, np.random.default_rng(42)) > .99


def test_invalid_configuration_and_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Config(min_tokens=1200, passage_tokens=200)
    with pytest.raises(ValueError, match='unique'):
        analyze([verse('a', 'word'), verse('a', 'word')])


def _spread(text, marker):
    """One body of text placed across two chapters, three books and two collections."""
    places = [('One', 'B1', '1'), ('One', 'B1', '2'), ('One', 'B2', '1'),
              ('Two', 'B3', '1'), ('Two', 'B3', '2'), ('Two', 'B4', '1')]
    return [{**verse(f'{marker}{index}', text), 'collection': collection, 'book': book,
             'chapter': chapter} for index, (collection, book, chapter) in enumerate(places)]


def test_one_style_reaches_across_chapters_books_and_collections():
    result = analyze(_spread('the and to it was he of in that for with as but not they ' * 12, 'a'),
                     Config(passage_tokens=60, min_tokens=20))
    assert {v['style_id'] for v in result['verses']} == {'eng-S001'}
    row = result['styles'][0]
    assert (row['chapter_count'], row['book_count'], row['collection_count']) == (6, 4, 2)
    # Every level reports the same single style rather than one private to its own branch.
    for level in ('language', 'collection', 'book', 'chapter'):
        rows = [r for r in result['rollups'] if r['level'] == level]
        assert rows and all(r['style_ids'] == ['eng-S001'] for r in rows)


def test_rollups_track_each_style_reach_separately_at_every_level():
    """Two styles occupying the same places must each keep their own full reach."""
    tagged = []
    for marker, style in (('a', 'eng-S001'), ('b', 'eng-S002')):
        for row in _spread('shared text', marker):
            tagged.append({**row, 'style_id': style, 'status': 'assigned', 'token_count': 10})
    rollups = make_rollups(tagged)
    language = next(row for row in rollups if row['level'] == 'language')
    assert language['style_ids'] == ['eng-S001', 'eng-S002']
    assert language['style_counts'] == {'eng-S001': 6, 'eng-S002': 6}
    for level, expected in (('collection', 2), ('book', 4), ('chapter', 6)):
        rows = [row for row in rollups if row['level'] == level]
        assert len(rows) == expected
        # Neither style is dropped where they share a chapter, and neither is merged away.
        assert all(row['style_count'] == 2 for row in rows)
        for style in ('eng-S001', 'eng-S002'):
            assert sum(style in row['style_ids'] for row in rows) == expected


def _work(collection, sizes, chapter='1'):
    """One book per entry, sized in tokens, in source order."""
    return [{**verse(f'{collection}-{index}', 'word ' * size), 'collection': collection,
             'book': f'B{index:02d}', 'chapter': chapter} for index, size in enumerate(sizes)]


def test_short_books_share_a_passage_only_in_a_continuous_collection():
    config = Config(passage_tokens=600, min_tokens=100, joinable_collections=('Joined',))
    sizes = [40, 40, 40, 300, 30, 30]
    joined, _ = make_passages(_work('Joined', sizes), config)
    apart, _ = make_passages(_work('Apart', sizes), config)
    assert [p['book_count'] for p in joined] == [3, 1, 2]
    assert [p['eligible'] for p in joined] == [True, True, False]
    # The same books in an ordinary collection stay separate and mostly unusable.
    assert [p['book_count'] for p in apart] == [1] * 6
    assert [p['eligible'] for p in apart] == [False, False, False, True, False, False]


def test_merging_stops_at_the_floor_rather_than_swallowing_a_whole_run():
    config = Config(passage_tokens=600, min_tokens=100, joinable_collections=('Joined',))
    passages, _ = make_passages(_work('Joined', [60] * 8), config)
    # Eight short books become four pairs, not one block of eight.
    assert [p['book_count'] for p in passages] == [2, 2, 2, 2]
    assert all(p['eligible'] for p in passages)


def test_a_short_tail_joins_its_neighbour_and_an_isolated_book_stays_alone():
    config = Config(passage_tokens=600, min_tokens=100, joinable_collections=('Joined',))
    passages, _ = make_passages(_work('Joined', [60, 60, 20, 500, 30, 500]), config)
    assert [p['book_count'] for p in passages] == [3, 1, 1, 1]
    # The isolated short book between two long ones has no neighbour to join.
    assert [p['eligible'] for p in passages] == [True, True, False, True]


def test_a_shared_passage_marks_every_verse_it_tagged():
    config = Config(passage_tokens=600, min_tokens=100, joinable_collections=('Joined',))
    records = _work('Joined', [60, 60, 300])
    result = analyze(records, config)
    shared = [v for v in result['verses'] if v['style_id'] and v['passage_books'] > 1]
    alone = [v for v in result['verses'] if v['style_id'] and v['passage_books'] == 1]
    assert len(shared) == 2 and len(alone) == 1
    assert {v['book'] for v in shared} == {'B00', 'B01'}
