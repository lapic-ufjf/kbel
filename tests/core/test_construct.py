# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time

import pytest
from kif_lib import Item, Property, Statement, ValueSnak

from kbel.core.candidate import Candidate
from kbel.core.kbel import KBEL
from kbel.core.mention import EntityType, Mention
from kbel.core.results import DisambiguationResult
from kbel.core.text import split_paragraphs, split_sentences
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator, naive  # noqa: F401
from kbel.extractors import TripleExtractor

WD = 'http://www.wikidata.org/entity/'

# Each sentence maps to the triples the fake extractor returns for it.
SCRIPT = {
    'Marie Curie was born in Warsaw.': [Triple('Marie Curie', 'place of birth', 'Warsaw')],
    'She was a physicist.': [Triple('Marie Curie', 'occupation', 'physicist'),
                             Triple('She', 'occupation', 'physicist')],
    'Pierre Curie was born in Paris.': [Triple('Pierre Curie', 'place of birth', 'Paris'),
                                        Triple('Pierre Curie', 'place of birth', 'Paris')],
}
TEXT = ('Marie Curie was born in Warsaw. She was a physicist.\n\n'
        'Pierre Curie was born in Paris.')

KB = {  # term -> entity id; "She" is deliberately missing
    'Marie Curie': 'Q7186', 'Pierre Curie': 'Q37463', 'Warsaw': 'Q270',
    'Paris': 'Q90', 'physicist': 'Q169470',
    'place of birth': 'P19', 'occupation': 'P106',
}


class ScriptedExtractor(TripleExtractor, strategy_name='test-scripted'):
    """Returns SCRIPT[sentence]; records calls and peak concurrency."""

    def __init__(self, strategy_name, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.calls: list[str] = []
        self.active = self.peak = 0
        self._guard = threading.Lock()

    def _extract(self, text, prompt=None):
        with self._guard:
            self.calls.append(text)
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self._guard:
            self.active -= 1
        return list(SCRIPT.get(text, []))


class RecordingDisambiguator(Disambiguator, strategy_name='test-recording'):
    """Picks the first candidate; records mentions; can fail on a term."""

    def __init__(self, strategy_name, fail_on=None):
        self.mentions: list[Mention] = []
        self.fail_on = fail_on

    def _disambiguate(self, mention, candidates, limit, *args, **kwargs):
        self.mentions.append(mention)
        if mention.term == self.fail_on:
            raise ValueError('boom')
        return [DisambiguationResult(candidate=candidates[0], score=1.0)]


@pytest.fixture
def setup(monkeypatch):
    """KBEL on the offline `empty` search, with scripted candidates."""
    searches: list[tuple[str, EntityType]] = []

    def lookup(self, mention, limit=None):
        searches.append((mention.term, mention.entity_type))
        qid = KB.get(mention.term)
        return [Candidate(iri=WD + qid, label=mention.term)] if qid else []

    monkeypatch.setattr(KBEL, 'candidates_lookup', lookup)
    extractor = TripleExtractor('test-scripted')
    disambiguator = Disambiguator('test-recording')
    kbel = KBEL('empty', disambiguator=disambiguator, extractor=extractor)
    return kbel, extractor, disambiguator, searches


def ids(statements):
    """(subject, property, value) ids of KIF statements."""
    return [tuple(e.iri.content.removeprefix(WD) for e in
                  (st.subject, st.snak.property, st.snak.value))
            for st in statements]


def collect(kbel, *args, **kwargs):
    return list(kbel.construct(*args, **kwargs))


async def acollect(kbel, *args, **kwargs):
    return [st async for st in kbel.aconstruct(*args, **kwargs)]


# -- text segmentation ------------------------------------------------------

def test_split_paragraphs():
    assert split_paragraphs('A b.\n  \n\nC\nd.\n\n\n') == ['A b.', 'C d.']


@pytest.mark.parametrize('text, expected', [
    ('One. Two! Three? Four', ['One.', 'Two!', 'Three?', 'Four']),
    ('He said "Hi." Then left.', ['He said "Hi."', 'Then left.']),
    ('It is approx. three km, e.g. near.', ['It is approx. three km, e.g. near.']),
    ('Ends here.\n\nNew paragraph', ['Ends here.', 'New paragraph']),
    ('No break\n\nbetween', ['No break', 'between']),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


# -- construct ----------------------------------------------------------------

def test_construct_yields_kif_statements_in_text_order(setup):
    kbel, *_ = setup
    statements = collect(kbel, TEXT)

    assert ids(statements) == [
        ('Q7186', 'P19', 'Q270'),
        ('Q7186', 'P106', 'Q169470'),
        ('Q37463', 'P19', 'Q90'),     # extracted twice, yielded once
    ]                                 # ("She", ...) is skipped: not linkable

    st = statements[0]
    assert isinstance(st, Statement)
    assert isinstance(st.snak, ValueSnak)
    assert st == Statement(Item(WD + 'Q7186'),
                           ValueSnak(Property(WD + 'P19'), Item(WD + 'Q270')))


def test_construct_returns_a_lazy_iterator(setup):
    kbel, extractor, *_ = setup
    statements = kbel.construct(TEXT)
    assert iter(statements) is statements and not isinstance(statements, list)
    first = next(statements)
    assert ids([first]) == [('Q7186', 'P19', 'Q270')]
    statements.close()


def test_stopping_early_cancels_pending_work(setup):
    kbel, *_ = setup
    kbel.extractor = extractor = TripleExtractor('test-scripted', delay=0.3)

    start = time.perf_counter()
    for _ in kbel.construct(TEXT, max_concurrency=1):
        break                         # only the first statement
    elapsed = time.perf_counter() - start

    # One slot: sentences run one after the other; after the first statement
    # the remaining extraction is cancelled instead of waited for.
    assert len(extractor.calls) < len(SCRIPT)
    assert elapsed < len(SCRIPT) * 0.3


def test_extraction_runs_per_sentence(setup):
    kbel, extractor, *_ = setup
    collect(kbel, TEXT)
    assert sorted(extractor.calls) == sorted(SCRIPT)


def test_mentions_use_sentence_as_text_and_paragraph_as_context(setup):
    kbel, _, disambiguator, _ = setup
    collect(kbel, TEXT, language='en')

    by_term = {m.term: m for m in disambiguator.mentions}
    warsaw, birth, paris = by_term['Warsaw'], by_term['place of birth'], by_term['Paris']

    assert warsaw.text == 'Marie Curie was born in Warsaw.'
    assert warsaw.context == 'Marie Curie was born in Warsaw. She was a physicist.'
    assert warsaw.entity_type is EntityType.ITEM
    assert birth.entity_type is EntityType.PROPERTY
    assert paris.context is None
    assert all(m.language == 'en' for m in disambiguator.mentions)


def test_searches_are_shared_and_mentions_linked_once(setup):
    kbel, _, disambiguator, searches = setup
    collect(kbel, TEXT)

    # "Marie Curie" appears in two sentences: one search, two disambiguations
    # (different sentence => different mention).
    assert searches.count(('Marie Curie', EntityType.ITEM)) == 1
    assert sum(m.term == 'Marie Curie' for m in disambiguator.mentions) == 2
    # The duplicated Pierre Curie triple (same sentence) is linked once.
    assert sum(m.term == 'Pierre Curie' for m in disambiguator.mentions) == 1
    # Nothing is disambiguated without candidates.
    assert all(m.term != 'She' for m in disambiguator.mentions)


def test_a_failing_link_skips_only_its_triples(setup):
    kbel, extractor, *_ = setup
    kbel.disambiguator = Disambiguator('test-recording', fail_on='Paris')
    assert ids(collect(kbel, TEXT)) == [
        ('Q7186', 'P19', 'Q270'), ('Q7186', 'P106', 'Q169470')]


def test_extractor_errors_reach_the_caller(setup):
    kbel, *_ = setup

    class Broken(TripleExtractor, strategy_name='test-broken'):
        def __init__(self, strategy_name, **kwargs):
            super().__init__(**kwargs)

        def _extract(self, text, prompt=None):
            raise ValueError('extractor down')

    kbel.extractor = 'test-broken'
    with pytest.raises(ValueError, match='extractor down'):
        collect(kbel, TEXT)


def test_empty_text_gives_no_statements(setup):
    kbel, extractor, *_ = setup
    assert collect(kbel, '   \n\n ') == []
    assert extractor.calls == []


# -- async behaviour ------------------------------------------------------------

def test_aconstruct_is_an_async_iterator_matching_construct(setup):
    kbel, *_ = setup
    assert ids(asyncio.run(acollect(kbel, TEXT))) == ids(collect(kbel, TEXT))


def test_aconstruct_stops_early(setup):
    kbel, *_ = setup

    async def first():
        async for statement in kbel.aconstruct(TEXT):
            return statement

    assert ids([asyncio.run(first())]) == [('Q7186', 'P19', 'Q270')]


def test_construct_works_inside_a_running_event_loop(setup):
    """As in Jupyter, where a loop is already running."""
    kbel, *_ = setup

    async def caller():
        return collect(kbel, TEXT)

    assert len(asyncio.run(caller())) == 3


def test_sentences_are_extracted_concurrently(setup):
    kbel, *_ = setup
    kbel.extractor = extractor = TripleExtractor('test-scripted', delay=0.2)

    start = time.perf_counter()
    collect(kbel, TEXT)
    elapsed = time.perf_counter() - start

    assert extractor.peak == 3          # all three sentences at once
    assert elapsed < 3 * 0.2


def test_max_concurrency_limits_parallel_calls(setup):
    kbel, *_ = setup
    kbel.extractor = extractor = TripleExtractor('test-scripted', delay=0.05)
    collect(kbel, TEXT, max_concurrency=1)
    assert extractor.peak == 1


def test_alink_matches_link(setup):
    kbel, *_ = setup
    mention = Mention(term='Warsaw', text='Marie Curie was born in Warsaw.')
    assert asyncio.run(kbel.alink(mention)) == kbel.link(mention)
    assert asyncio.run(kbel.alink(Mention(term='She', text=''))) == []


# -- errors -----------------------------------------------------------------------

def test_construct_validates_when_called_not_when_iterated():
    with pytest.raises(RuntimeError, match='No extractor'):
        KBEL('empty', disambiguator='naive').construct(TEXT)
    with pytest.raises(RuntimeError, match='No disambiguator'):
        KBEL('empty', extractor=TripleExtractor('test-scripted')).aconstruct(TEXT)


def test_construct_rejects_bad_concurrency(setup):
    kbel, *_ = setup
    with pytest.raises(ValueError):
        kbel.construct(TEXT, max_concurrency=0)


# -- priority slots (used by aconstruct to stream in text order) ---------------------

from kbel.core.kbel import _PrioritySlots  # noqa: E402


def test_priority_slots_serve_lowest_priority_first():
    async def scenario():
        slots, order = _PrioritySlots(1), []

        async def job(priority):
            async with slots(priority):
                order.append(priority)
                await asyncio.sleep(0.01)

        holder = asyncio.ensure_future(job(-1))          # takes the only slot
        await asyncio.sleep(0)
        await asyncio.gather(holder, *(job(p) for p in (5, 2, 9, 0, 2)))
        return order

    assert asyncio.run(scenario()) == [-1, 0, 2, 2, 5, 9]


def test_priority_slots_survive_cancelled_waiters():
    async def scenario():
        slots = _PrioritySlots(2)
        active = peak = 0

        async def job(priority, delay=0.02):
            nonlocal active, peak
            async with slots(priority):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(delay)
                active -= 1

        jobs = [asyncio.ensure_future(job(p)) for p in range(8)]
        await asyncio.sleep(0.005)
        for j in jobs[3:6]:                               # waiting ones
            j.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        # All slots were given back: two new jobs run at once immediately.
        await asyncio.wait_for(asyncio.gather(job(0), job(0)), timeout=0.2)
        return peak, slots._free

    peak, free = asyncio.run(scenario())
    assert peak == 2 and free == 2


def test_first_statement_does_not_wait_for_the_whole_document(setup, monkeypatch):
    kbel, *_ = setup
    sentences = [f'Marie Curie was born in Warsaw, take {i}.' for i in range(12)]
    for s in sentences:
        monkeypatch.setitem(SCRIPT, s, [Triple('Marie Curie', 'place of birth', 'Warsaw')])
    kbel.extractor = extractor = TripleExtractor('test-scripted', delay=0.05)

    statements = kbel.construct(' '.join(sentences), max_concurrency=2)
    next(statements)
    extracted_before_first = len(extractor.calls)
    statements.close()

    # Work for the first sentence is prioritized over extracting the rest:
    # only a few extractions (those already holding or next in line for a
    # slot) run before it, never most of the document.
    assert extracted_before_first <= len(sentences) // 2
