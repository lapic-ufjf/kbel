# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

"""Async hooks of the plugin base classes and their use by KBEL."""

import asyncio
import threading
import time

import pytest

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.results import DisambiguationResult
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

WD = 'http://www.wikidata.org/entity/'
CANDIDATES = [Candidate(iri=WD + 'Q1', label='one', id='Q1'),
              Candidate(iri=WD + 'Q2', label='two', id='Q2')]


# -- plugins used by the tests ------------------------------------------------

class SlowSync(Disambiguator, strategy_name='test-slow-sync'):
    """Blocking disambiguator: records the thread it runs on."""

    def __init__(self, strategy_name, delay=0.2):
        self.delay = delay
        self.threads: list[int] = []

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        self.threads.append(threading.get_ident())
        time.sleep(self.delay)
        return [DisambiguationResult(candidate=c) for c in candidates[:limit]]


class NativeAsync(Disambiguator, strategy_name='test-native-async'):
    """Overrides only the async hook; the sync one must not be used."""

    def __init__(self, strategy_name):
        self.calls = 0

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        raise AssertionError('sync path used')

    async def _adisambiguate(self, mention, candidates, limit, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return [DisambiguationResult(candidate=candidates[-1], score=0.5)]


class SlowExtractor(TripleExtractor, strategy_name='test-slow-extract'):
    def __init__(self, strategy_name, delay=0.2, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def _extract(self, text, prompt=None):
        time.sleep(self.delay)
        return [Triple(text.rstrip('.'), 'is', 'slow')]


class NativeAsyncExtractor(TripleExtractor, strategy_name='test-native-extract'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def _extract(self, text, prompt=None):
        raise AssertionError('sync path used')

    async def _aextract(self, text, prompt=None):
        self.calls.append(text)
        await asyncio.sleep(0)
        return [Triple(text.split()[0], 'mentions', 'something')]


async def ticker(stop: asyncio.Event) -> int:
    """Counts loop iterations until `stop`: stays at ~0 if the loop is blocked."""
    ticks = 0
    while not stop.is_set():
        ticks += 1
        await asyncio.sleep(0.01)
    return ticks


async def run_with_ticker(coro):
    stop = asyncio.Event()
    tick_task = asyncio.ensure_future(ticker(stop))
    result = await coro
    stop.set()
    return result, await tick_task


# -- Disambiguator.adisambiguate -----------------------------------------------

def test_adisambiguate_returns_the_same_list_as_disambiguate():
    d = Disambiguator('test-slow-sync', delay=0)
    mention = Mention(term='one', text='')
    assert asyncio.run(d.adisambiguate(mention, CANDIDATES, 1)) == \
        d.disambiguate(mention, CANDIDATES, 1)


def test_default_adisambiguate_does_not_block_the_event_loop():
    d = Disambiguator('test-slow-sync', delay=0.3)
    results, ticks = asyncio.run(run_with_ticker(
        d.adisambiguate(Mention(term='one', text=''), CANDIDATES)))
    assert len(results) == 2
    assert ticks >= 5                                   # loop kept running
    assert d.threads[0] != threading.main_thread().ident


def test_adisambiguate_rejects_empty_candidates():
    with pytest.raises(AssertionError):
        asyncio.run(Disambiguator('test-slow-sync').adisambiguate(
            Mention(term='x', text=''), []))


# -- TripleExtractor.aextract --------------------------------------------------

def test_aextract_returns_the_same_as_extract():
    e = TripleExtractor('test-slow-extract', delay=0)
    assert asyncio.run(e.aextract('Snails.')) == e.extract('Snails.')


def test_default_aextract_does_not_block_the_event_loop():
    e = TripleExtractor('test-slow-extract', delay=0.3)
    triples, ticks = asyncio.run(run_with_ticker(e.aextract('Snails.')))
    assert triples == [Triple('Snails', 'is', 'slow')]
    assert ticks >= 5


def test_aextract_rejects_empty_text():
    with pytest.raises(AssertionError):
        asyncio.run(TripleExtractor('test-slow-extract').aextract('  '))


def test_aextract_renders_graph(tmp_path):
    path = tmp_path / 'g.dot'
    asyncio.run(TripleExtractor('test-slow-extract', delay=0).aextract(
        'Snails.', graph=True, graph_path=path))
    assert '"Snails" -> "slow"' in path.read_text()


# -- KBEL uses the async hooks ---------------------------------------------------

@pytest.fixture
def scripted_search(monkeypatch):
    calls = []

    def lookup(self, mention, limit=None):
        calls.append(mention.term)
        return [] if mention.term == 'She' else list(CANDIDATES)

    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)
    return calls


def test_alink_uses_native_async_disambiguator(scripted_search):
    d = Disambiguator('test-native-async')
    kbel = KBEL('empty', disambiguator=d)
    [result] = asyncio.run(kbel.alink(Mention(term='x', text='')))
    assert result.candidate.id == 'Q2' and result.score == 0.5
    assert d.calls == 1


def test_construct_uses_native_async_plugins(scripted_search):
    e = TripleExtractor('test-native-extract')
    d = Disambiguator('test-native-async')
    kbel = KBEL('empty', disambiguator=d, extractor=e)

    statements = list(kbel.construct('Alpha one. Beta two.'))
    assert e.calls == ['Alpha one.', 'Beta two.']
    assert len(statements) == 1          # all parts link to Q2: one statement
    assert d.calls > 0


def test_aextract_on_kbel(scripted_search):
    kbel = KBEL('empty', extractor=TripleExtractor('test-native-extract'))
    assert asyncio.run(kbel.aextract('Gamma rays.')) == [
        Triple('Gamma', 'mentions', 'something')]
    with pytest.raises(RuntimeError, match='No extractor'):
        asyncio.run(KBEL('empty').aextract('x'))


# -- aextract_and_link -------------------------------------------------------------

class TwoTriples(TripleExtractor, strategy_name='test-two-triples'):
    def __init__(self, strategy_name, **kwargs):
        super().__init__(**kwargs)

    def _extract(self, text, prompt=None):
        return [Triple('Alpha', 'is', 'one'), Triple('She', 'is', 'one')]


def test_aextract_and_link_matches_extract_and_link(scripted_search):
    kbel = KBEL('empty', disambiguator='test-slow-sync', extractor='test-two-triples')
    kbel.disambiguator = Disambiguator('test-slow-sync', delay=0)

    async_result = asyncio.run(kbel.aextract_and_link('Alpha is one.', limit=1))
    assert async_result == kbel.extract_and_link('Alpha is one.', limit=1)

    (t1, (s1, p1, o1)), (t2, (s2, p2, o2)) = async_result
    assert t1 == Triple('Alpha', 'is', 'one') and len(s1) == len(p1) == len(o1) == 1
    assert t2.subject == 'She' and s2 == []            # nothing to link


def test_aextract_and_link_links_concurrently_and_once(scripted_search):
    kbel = KBEL('empty', extractor='test-two-triples')
    kbel.disambiguator = d = Disambiguator('test-slow-sync', delay=0.2)

    start = time.perf_counter()
    asyncio.run(kbel.aextract_and_link('Alpha is one.'))
    elapsed = time.perf_counter() - start

    # 6 mentions, 4 distinct -> 3 with candidates ("She" has none)
    assert len(d.threads) == 3
    assert elapsed < 3 * 0.2


def test_aextract_and_link_requires_disambiguator(scripted_search):
    kbel = KBEL('empty', extractor='test-two-triples')
    with pytest.raises(RuntimeError, match='No disambiguator'):
        asyncio.run(kbel.aextract_and_link('x'))


def test_extract_and_link_passes_extractor_kwargs(scripted_search, tmp_path):
    kbel = KBEL('empty', extractor='test-two-triples')
    kbel.disambiguator = Disambiguator('test-slow-sync', delay=0)
    kbel.extract_and_link('Alpha is one.', graph=True, graph_path=tmp_path / 'g.dot')
    assert (tmp_path / 'g.dot').exists()
