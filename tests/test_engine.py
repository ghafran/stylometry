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
