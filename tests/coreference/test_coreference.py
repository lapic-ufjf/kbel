# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types

import pytest

from kbel.coreference import CoreferenceResolver
from kbel.coreference.fastcoref import FastCorefResolver


def spans(text, *mentions):
    out, position = [], 0
    for mention in mentions:
        start = text.index(mention, position)
        out.append((start, start + len(mention)))
        position = start + len(mention)
    return out


class Scripted(CoreferenceResolver, strategy_name='test-scripted-coref'):
    """Clusters given as lists of mention strings, found in the text in order."""

    def __init__(self, strategy_name, clusters=(), **kwargs):
        super().__init__(**kwargs)
        self.script = [list(c) for c in clusters]
        self.calls = []

    def _clusters(self, texts):
        self.calls.append(list(texts))
        return [[spans(t, *c) for c in self.script] for t in texts]


def resolve(text, *clusters, **options):
    return Scripted('test-scripted-coref', clusters=clusters, **options).resolve(text)


def test_pronouns_are_replaced_by_the_name_of_the_cluster():
    text = 'Marie Curie was born in Warsaw. She won a prize.'
    assert resolve(text, ['Marie Curie', 'She']) == (
        'Marie Curie was born in Warsaw. Marie Curie won a prize.')


@pytest.mark.parametrize('pronoun,expected', [
    ('his', "Alan's work"), ('her', "Alan's work"), ('its', "Alan's work"), ('their', "Alan's work"),
])
def test_possessive_pronouns_become_possessives(pronoun, expected):
    assert resolve(f'Alan met Bob. Bob liked {pronoun} work.', ['Alan', pronoun]) == (
        f'Alan met Bob. Bob liked {expected}.')


def test_possessive_of_a_name_ending_in_s():
    assert resolve('Charles smiled. His work.', ['Charles', 'His']) == "Charles smiled. Charles' work."


def test_her_is_an_object_or_a_possessive_depending_on_what_follows():
    text = 'Marie won. Pierre helped her. Her husband smiled. Pierre thanked her.'
    out = resolve(text, ['Marie', 'her', 'Her', 'her'])
    assert out == "Marie won. Pierre helped Marie. Marie's husband smiled. Pierre thanked Marie."


def test_other_mentions_are_kept_unless_replace_is_all():
    text = 'Marie Curie was a physicist. The scientist won.'
    cluster = ['Marie Curie', 'The scientist']
    assert resolve(text, cluster) == text
    assert resolve(text, cluster, replace='all') == 'Marie Curie was a physicist. Marie Curie won.'


def test_pronouns_before_the_name_and_first_or_second_person_are_untouched():
    assert resolve('When she arrived, Marie smiled. I know.', ['she', 'Marie']) == (
        'When she arrived, Marie smiled. I know.')
    assert resolve('Marie said I like it.', ['Marie', 'I']) == 'Marie said I like it.'


def test_a_cluster_of_pronouns_only_is_left_alone():
    assert resolve('He said it. He left.', ['He', 'He']) == 'He said it. He left.'


def test_longest_representative_and_max_words():
    text = 'Turing studied. Alan Mathison Turing left. He won.'
    cluster = ['Turing', 'Alan Mathison Turing', 'He']
    assert resolve(text, cluster).endswith('Turing won.')
    assert resolve(text, cluster, representative='longest').endswith('Alan Mathison Turing won.')
    assert resolve(text, cluster, representative='longest', max_words=2).endswith('Turing won.')


def test_clause_like_names_are_not_used():
    text = 'The company that employed him closed. It was old.'
    assert resolve(text, ['The company that employed him', 'It']) == text


def test_determiners_follow_the_position_in_the_sentence():
    text = 'The company grew. It hired Bob. Bob thanked it.'
    out = resolve(text, ['The company', 'It', 'it'])
    assert out == 'The company grew. The company hired Bob. Bob thanked the company.'


def test_names_that_contain_replaced_mentions_are_rendered():
    text = 'Alan Turing wrote. The friend of him left. It rained.'
    out = resolve(text, ['Alan Turing', 'him'], ['The friend of him', 'It'])
    assert out == 'Alan Turing wrote. The friend of Alan Turing left. The friend of Alan Turing rained.'


def test_names_that_contain_each_other_do_not_loop():
    text = 'A saw B. B saw A.'
    assert resolve(text, ['A', 'B'], ['B', 'A'], replace='all')


def test_overlapping_edits_keep_the_first():
    text = 'Ann met Bo. Her friend Bo left.'
    out = resolve(text, ['Ann', 'Her'], ['Bo', 'Her friend Bo'])
    assert out.startswith("Ann met Bo. ")


def test_text_without_clusters_is_returned_unchanged():
    assert resolve('Nothing to resolve.') == 'Nothing to resolve.'


def test_batch_and_iterator_keep_order_and_batch_size():
    r = Scripted('test-scripted-coref', clusters=[['Ann', 'She']], batch_size=2)
    texts = [f'Ann {i}. She {i}.' for i in range(5)]
    assert r.resolve_batch(texts) == [f'Ann {i}. Ann {i}.' for i in range(5)]
    r.calls.clear()
    iterator = r.iter_resolve(iter(texts))
    assert r.calls == []
    assert next(iterator) == 'Ann 0. Ann 0.'
    assert [len(c) for c in r.calls] == [2]
    assert list(iterator) == [f'Ann {i}. Ann {i}.' for i in range(1, 5)]
    assert [len(c) for c in r.calls] == [2, 2, 1]
    assert r.resolve_batch([]) == []


def test_async_version_does_not_block_the_loop():
    r = Scripted('test-scripted-coref', clusters=[['Ann', 'She']])

    async def run():
        return await asyncio.gather(r.aresolve('Ann. She.'), r.aresolve_batch(['Ann. She.']))

    assert asyncio.run(run()) == ['Ann. Ann.', ['Ann. Ann.']]


def test_option_validation_and_unknown_plugin():
    with pytest.raises(ValueError, match='replace'):
        Scripted('test-scripted-coref', replace='x')
    with pytest.raises(ValueError, match='representative'):
        Scripted('test-scripted-coref', representative='x')
    with pytest.raises(ValueError, match='batch_size'):
        Scripted('test-scripted-coref', batch_size=0)
    with pytest.raises(ValueError, match='no such coreference'):
        CoreferenceResolver('nope')


def test_a_resolver_may_rewrite_the_text_directly():
    class Upper(CoreferenceResolver, strategy_name='test-upper-coref'):
        def __init__(self, strategy_name):
            super().__init__()

        def _clusters(self, texts):
            raise AssertionError('not used')

        def _resolve_batch(self, texts):
            return [t.upper() for t in texts]

    assert CoreferenceResolver('test-upper-coref').resolve('ann') == 'ANN'


class FakeResult:
    def __init__(self, clusters):
        self.clusters = clusters

    def get_clusters(self, as_strings=True):
        assert as_strings is False
        return self.clusters


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, texts, max_tokens_in_batch):
        self.calls.append((list(texts), max_tokens_in_batch))
        return [FakeResult([[(0, 3), (13, 16)]]) for _ in texts]


def test_fastcoref_plugin_turns_the_model_clusters_into_substitutions():
    model = FakeModel()
    r = CoreferenceResolver('fastcoref', model=model, max_tokens_in_batch=500)
    assert isinstance(r, FastCorefResolver) and r.model is model
    assert r.resolve('Ann arrived. She sat.') == 'Ann arrived. Ann sat.'
    assert model.calls == [(['Ann arrived. She sat.'], 500)]
    assert r.clusters('Ann arrived. She sat.') == [[(0, 3), (13, 16)]]


def test_fastcoref_plugin_is_thread_safe():
    import threading

    class Slow(FakeModel):
        active = peak = 0

        def predict(self, texts, max_tokens_in_batch):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            threading.Event().wait(0.02)
            Slow.active -= 1
            return super().predict(texts, max_tokens_in_batch)

    r = CoreferenceResolver('fastcoref', model=Slow())

    async def run():
        return await asyncio.gather(*(r.aresolve('Ann arrived. She sat.') for _ in range(6)))

    assert set(asyncio.run(run())) == {'Ann arrived. Ann sat.'}
    assert Slow.peak == 1


def test_fastcoref_plugin_validates_options():
    with pytest.raises(ValueError, match='Unknown model'):
        CoreferenceResolver('fastcoref', model_name='x', model=FakeModel())
    with pytest.raises(ValueError, match='max_tokens_in_batch'):
        CoreferenceResolver('fastcoref', max_tokens_in_batch=0, model=FakeModel())


def test_fastcoref_missing_package_names_the_extra(monkeypatch):
    for name in ('fastcoref', 'fastcoref.coref_models.modeling_fcoref', 'spacy'):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r'kbel\[coreference\]'):
        CoreferenceResolver('fastcoref')


def test_fastcoref_patches_models_for_transformers_5():
    from kbel.coreference.fastcoref import _patch_transformers_5

    class Old:
        pass

    class New:
        all_tied_weights_keys = {'x': 1}

    _patch_transformers_5(Old, New)
    assert Old().all_tied_weights_keys == {}
    assert New.all_tied_weights_keys == {'x': 1}
