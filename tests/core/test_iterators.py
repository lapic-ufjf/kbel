# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time

import pytest

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.results import DisambiguationResult
from kbel.core.triple import Triple
from kbel.coreference import CoreferenceResolver
from kbel.disambiguators import Disambiguator, naive  # noqa: F401
from kbel.extractors import TripleExtractor

WD = 'http://www.wikidata.org/entity/'
KB = {'Marie Curie': 'Q7186', 'Warsaw': 'Q270', 'Paris': 'Q90', 'physicist': 'Q169470',
      'place of birth': 'P19', 'occupation': 'P106'}
TEXT = ('Marie Curie was born in Warsaw. She was a physicist.\n\n'
        'Pierre Curie was born in Paris.')
SCRIPT = {
    'Marie Curie was born in Warsaw.': [Triple('Marie Curie', 'place of birth', 'Warsaw')],
    'Marie Curie was a physicist.': [Triple('Marie Curie', 'occupation', 'physicist')],
    'She was a physicist.': [Triple('She', 'occupation', 'physicist')],
    'Pierre Curie was born in Paris.': [Triple('Pierre Curie', 'place of birth', 'Paris')],
}


class Scripted(TripleExtractor, strategy_name='test-iter-scripted'):
    def __init__(self, strategy_name, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.calls = []
        self._guard = threading.Lock()

    def _extract(self, text, prompt=None):
        with self._guard:
            self.calls.append(text)
        time.sleep(self.delay)
        return list(SCRIPT.get(text, []))


class Marie(CoreferenceResolver, strategy_name='test-iter-coref'):
    def __init__(self, strategy_name):
        super().__init__()
        self.calls = []

    def _clusters(self, texts):
        self.calls.append(list(texts))
        return [[[(0, 11), (t.index('She'), t.index('She') + 3)]] if 'She' in t else []
                for t in texts]


class First(Disambiguator, strategy_name='test-iter-first'):
    def __init__(self, strategy_name, fail_on=None):
        self.fail_on = fail_on

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        if mention.term == self.fail_on:
            raise ValueError('boom')
        return [DisambiguationResult(candidate=c, score=1.0 - i / 10)
                for i, c in enumerate(candidates[:limit])]


@pytest.fixture
def kbel(monkeypatch):
    searches = []

    def lookup(self, mention, limit=None):
        searches.append(mention.term)
        qid = KB.get(mention.term)
        if not qid:
            return []
        return [Candidate(iri=WD + qid, id=qid, label=mention.term),
                Candidate(iri=WD + 'Q1', id='Q1', label='other')]

    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)
    k = KBEL('empty', disambiguator=Disambiguator('test-iter-first'),
             extractor=TripleExtractor('test-iter-scripted'))
    k.searches = searches
    return k


def ids(links):
    return [[r.candidate.id for r in part] for part in links]


def test_iter_extract_yields_triples_in_text_order(kbel):
    triples = list(kbel.extract(TEXT))
    assert triples == [SCRIPT['Marie Curie was born in Warsaw.'][0],
                       SCRIPT['She was a physicist.'][0],
                       SCRIPT['Pierre Curie was born in Paris.'][0]]
    assert kbel.searches == []


def test_iter_extract_is_lazy_and_stops_early(kbel):
    kbel.extractor.delay = 0.05
    kbel.extractor.batch_size = 1
    iterator = kbel.extract(TEXT, max_concurrency=1)
    assert next(iterator) == SCRIPT['Marie Curie was born in Warsaw.'][0]
    iterator.close()
    assert len(kbel.extractor.calls) < 3


def test_aextract_batches_sentences_like_construct(kbel):
    kbel.extractor.batch_size = 2

    async def run():
        return [t async for t in kbel.aextract(TEXT)]

    assert len(asyncio.run(run())) == 3


def test_iter_extract_needs_only_an_extractor():
    with pytest.raises(RuntimeError, match='No extractor'):
        KBEL('empty').extract('Some text.')


def test_extract_and_link_yields_every_triple_with_its_ranked_links(kbel):
    results = list(kbel.extract_and_link(TEXT, limit=2))
    assert [t for t, _ in results] == [
        SCRIPT['Marie Curie was born in Warsaw.'][0], SCRIPT['She was a physicist.'][0],
        SCRIPT['Pierre Curie was born in Paris.'][0]]
    assert ids(results[0][1]) == [['Q7186', 'Q1'], ['P19', 'Q1'], ['Q270', 'Q1']]
    assert ids(results[1][1]) == [[], ['P106', 'Q1'], ['Q169470', 'Q1']]
    assert ids(results[2][1]) == [[], ['P19', 'Q1'], ['Q90', 'Q1']]
    assert all(r.entity.iri.content.startswith(WD) for _, parts in results for p in parts for r in p)


def test_extract_and_link_limit_and_failures(kbel):
    kbel.disambiguator = Disambiguator('test-iter-first', fail_on='Warsaw')
    first, *_ = kbel.extract_and_link(TEXT, limit=1)
    assert ids(first[1]) == [['Q7186'], ['P19'], []]


def test_extract_and_link_is_lazy(kbel):
    kbel.extractor.delay = 0.05
    kbel.extractor.batch_size = 1
    iterator = kbel.extract_and_link(TEXT, max_concurrency=1)
    first = next(iterator)
    iterator.close()
    assert first[0] == SCRIPT['Marie Curie was born in Warsaw.'][0]
    calls = len(kbel.extractor.calls)
    time.sleep(0.2)
    assert len(kbel.extractor.calls) == calls


def test_async_iterators_are_async_generators(kbel):
    async def run():
        links = [x async for x in kbel.aextract_and_link(TEXT, limit=1)]
        mentions = [Mention(term='Warsaw', text='Warsaw.'), Mention(term='Paris', text='Paris.')]
        linked = [x async for x in kbel.alink_many(mentions, limit=1)]
        return links, linked

    links, linked = asyncio.run(run())
    assert len(links) == 3
    assert [m.term for m, _ in linked] == ['Warsaw', 'Paris']


def test_link_many_keeps_the_order_and_shares_searches(kbel):
    mentions = [Mention(term='Paris', text='In Paris.'), Mention(term='nothing', text='x'),
                Mention(term='Paris', text='Paris again.'),
                Mention(term='place of birth', text='born', entity_type=EntityType.PROPERTY)]
    results = list(kbel.link_many(mentions, limit=1))
    assert [m for m, _ in results] == mentions
    assert [[r.candidate.id for r in links] for _, links in results] == [
        ['Q90'], [], ['Q90'], ['P19']]
    assert sorted(kbel.searches) == ['Paris', 'nothing', 'place of birth']


def test_link_many_survives_failures_and_stops_early(kbel):
    kbel.disambiguator = Disambiguator('test-iter-first', fail_on='Paris')
    mentions = [Mention(term='Paris', text='x'), Mention(term='Warsaw', text='y')]
    assert [[r.candidate.id for r in links] for _, links in kbel.link_many(mentions)] == [
        [], ['Q270', 'Q1']]
    iterator = kbel.link_many(mentions * 20)
    next(iterator)
    iterator.close()


def test_link_many_validates_its_arguments():
    with pytest.raises(RuntimeError, match='No disambiguator'):
        KBEL('empty').link_many([])
    kb = KBEL('empty', disambiguator='naive')
    with pytest.raises(ValueError, match='max_concurrency'):
        kb.link_many([], max_concurrency=0)
    assert list(kb.link_many([])) == []


def test_coreference_is_resolved_before_anything_else(kbel):
    kbel.coreference = Marie('test-iter-coref')
    statements = list(kbel.construct(TEXT))
    assert [(s.subject.iri.content[len(WD):], s.snak.property.iri.content[len(WD):])
            for s in statements] == [('Q7186', 'P19'), ('Q7186', 'P106')]
    assert 'Marie Curie was a physicist.' in kbel.extractor.calls
    assert 'She was a physicist.' not in kbel.extractor.calls
    assert kbel.coreference.calls == [[TEXT]]


def test_coreference_applies_to_every_text_method(kbel):
    kbel.coreference = 'test-iter-coref'
    assert isinstance(kbel.coreference, Marie)
    assert kbel.resolve_coreference('Marie Curie won. She left.') == 'Marie Curie won. Marie Curie left.'
    assert asyncio.run(kbel.aresolve_coreference('Marie Curie won. She left.')) == (
        'Marie Curie won. Marie Curie left.')
    [subject] = {t.subject for t in kbel.extract(TEXT) if t.predicate == 'occupation'}
    assert subject == 'Marie Curie'
    [(triple, _)] = [x for x in kbel.extract_and_link(TEXT) if x[0].predicate == 'occupation']
    assert triple.subject == 'Marie Curie'
    assert [t.subject for t in kbel.extract('Marie Curie was a physicist.')] == ['Marie Curie']
    kbel.extractor.calls.clear()
    list(kbel.extract('She was a physicist.'))
    assert kbel.extractor.calls == ['She was a physicist.']

    kbel.extractor.calls.clear()
    list(kbel.extract_and_link('Marie Curie won. She was a physicist.'))
    assert sorted(kbel.extractor.calls) == ['Marie Curie was a physicist.', 'Marie Curie won.']


def test_without_a_resolver_the_text_is_untouched(kbel):
    assert kbel.coreference is None
    assert kbel.resolve_coreference(TEXT) == TEXT
    assert asyncio.run(kbel.aresolve_coreference(TEXT)) == TEXT
    list(kbel.extract(TEXT))
    assert 'She was a physicist.' in kbel.extractor.calls


def test_the_resolver_can_be_given_to_the_constructor_by_name():
    k = KBEL('empty', coreference='test-iter-coref')
    assert isinstance(k.coreference, Marie)
    k.coreference = None
    assert k.coreference is None


def test_extractor_iterates_over_a_stream_of_texts_in_batches():
    class Batched(TripleExtractor, strategy_name='test-iter-batched'):
        def __init__(self, strategy_name):
            super().__init__()
            self.batch_size = 2
            self.groups = []

        def _extract(self, text, prompt=None):
            return [Triple(text, 'p', 'o')]

        def _extract_batch(self, texts, prompt=None):
            self.groups.append(list(texts))
            return [self._extract(t) for t in texts]

    e = TripleExtractor('test-iter-batched')
    iterator = e.iter_extract_batch(iter(['a', 'b', 'c', 'd', 'e']))
    assert e.groups == []
    assert next(iterator) == [Triple('a', 'p', 'o')]
    assert e.groups == [['a', 'b']]
    assert [t[0].subject for t in iterator] == ['b', 'c', 'd', 'e']
    assert e.groups == [['a', 'b'], ['c', 'd'], ['e']]

    async def run():
        async def stream():
            for t in 'xyz':
                yield t
        return ([t[0].subject async for t in e.aiter_extract_batch('xyz')],
                [t[0].subject async for t in e.aiter_extract_batch(stream())])

    assert asyncio.run(run()) == (['x', 'y', 'z'], ['x', 'y', 'z'])
    assert list(e.iter_extract_batch([])) == []
