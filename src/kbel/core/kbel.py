# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import contextlib
import heapq
import itertools
import logging
import queue
import random
import threading
import time
from collections.abc import AsyncIterator, Coroutine, Iterable, Iterator
from typing import Any, TypeVar

from kif_lib import Item, Property, Statement, ValueSnak
from kif_lib.search import Search

from kbel.core.candidate import Candidate
from kbel.core.mention import EntityType, Mention
from kbel.core.results import LinkResult
from kbel.core.text import split_paragraphs, split_sentences
from kbel.core.triple import Triple
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor
from kbel.search import WikidataVDB_Search

LOG = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = ['KBEL', 'WikidataVDB_Search']


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Runs `coro` from synchronous code, also where a loop is running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _PrioritySlots:
    """A semaphore whose waiters are served by priority (lowest first)."""

    def __init__(self, n: int):
        self._free = n
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()

    @contextlib.asynccontextmanager
    async def __call__(self, priority: int):
        if self._free > 0 and not self._waiters:
            self._free -= 1
        else:
            granted = asyncio.get_running_loop().create_future()
            heapq.heappush(self._waiters, (priority, next(self._order), granted))
            try:
                await granted
            except asyncio.CancelledError:
                if granted.done() and not granted.cancelled():
                    self._release()
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            _, _, granted = heapq.heappop(self._waiters)
            if not granted.done():
                granted.set_result(None)
                return
        self._free += 1


def _iterate_sync(agen: AsyncIterator[T]) -> Iterator[T]:
    """Consumes an async iterator lazily from synchronous code.

    The iterator runs on its own event loop in a helper thread (so this
    also works inside Jupyter); closing the returned iterator early cancels
    the async one.
    """
    items: queue.Queue = queue.Queue()
    done = object()
    started = threading.Event()
    state: dict[str, Any] = {}

    async def pump() -> None:
        try:
            async for item in agen:
                items.put((True, item))
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            items.put((False, e))
        finally:
            await agen.aclose()
            items.put((True, done))

    def run() -> None:
        loop = asyncio.new_event_loop()
        state['loop'], state['task'] = loop, loop.create_task(pump())
        started.set()
        try:
            loop.run_until_complete(state['task'])
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    started.wait()
    try:
        while True:
            ok, item = items.get()
            if not ok:
                raise item
            if item is done:
                return
            yield item
    finally:
        if worker.is_alive():
            with contextlib.suppress(RuntimeError):
                state['loop'].call_soon_threadsafe(state['task'].cancel)
        worker.join()


TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _retry_after(error: BaseException) -> float | None:
    """Seconds to wait before retrying `error`, or None if it is not transient.

    Rate limits (429), server errors and network failures are transient;
    the server's `Retry-After` header is honoured when present.
    """
    try:
        import httpx
    except ImportError:
        return None
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code not in TRANSIENT_STATUS:
            return None
        with contextlib.suppress(TypeError, ValueError):
            return max(0.0, float(error.response.headers.get('Retry-After')))
        return 0.0
    if isinstance(error, httpx.TransportError):
        return 0.0
    return None


def _skip_errors(items: Iterable[T]) -> Iterator[T]:
    """Iterates `items`, skipping the ones whose retrieval raises.

    Transient errors (see `_retry_after`) are raised instead, so that the
    whole search can be retried rather than silently losing results.
    """
    it = iter(items)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            if _retry_after(e) is not None:
                raise
            LOG.info(f'Error fetching item: {e}')


class KBEL:
    """Links text to knowledge-base entities.

    Searches always run in worker threads: KIF's async HTTP client is bound
    to the event loop that first used it. Extraction and disambiguation use
    the plugins' async hooks.
    """

    _knowledge_base: Search
    _extractor: TripleExtractor | None = None
    _disambiguator: Disambiguator | None = None
    _search_limit = 10
    search_retries = 6
    search_backoff = 1.0

    def __init__(
        self,
        knowledge_base: Search | str,
        disambiguator: Disambiguator | str | None = None,
        extractor: TripleExtractor | str | None = None,
        search_limit: int = 10,
        **kwargs: Any,
    ):
        assert knowledge_base, 'Knowledge base can not be undefined.'
        self.search_limit = search_limit
        self.knowledge_base = knowledge_base
        if disambiguator is not None:
            self.disambiguator = disambiguator
        if extractor is not None:
            self.extractor = extractor

    @property
    def search_limit(self) -> int:
        return self._search_limit

    @search_limit.setter
    def search_limit(self, value: int):
        if value <= 0:
            raise ValueError('The limit to lookup candidates must be bigger than zero')
        self._search_limit = value

    @property
    def knowledge_base(self) -> Search:
        return self._knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value: Search | str) -> None:
        self._knowledge_base = (
            Search(value, limit=self._search_limit) if isinstance(value, str) else value)

    @property
    def disambiguator(self) -> Disambiguator | None:
        return self._disambiguator

    @disambiguator.setter
    def disambiguator(self, value: Disambiguator | str) -> None:
        self._disambiguator = Disambiguator(value) if isinstance(value, str) else value

    @property
    def extractor(self) -> TripleExtractor | None:
        return self._extractor

    @extractor.setter
    def extractor(self, value: TripleExtractor | str | None) -> None:
        self._extractor = TripleExtractor(value) if isinstance(value, str) else value

    def _require(self, extractor: bool = False, disambiguator: bool = False) -> None:
        if extractor and self._extractor is None:
            raise RuntimeError("No extractor has been configured.")
        if disambiguator and self._disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")

    def _limit(self, limit: int | None) -> int:
        return limit if limit is not None and limit > 0 else self._search_limit

    async def acandidates_lookup(
            self, mention: Mention, limit: int | None = None) -> list[Candidate]:
        """Async version of `candidates_lookup`."""
        return await asyncio.to_thread(self.candidates_lookup, mention, limit)

    async def aextract(self, text: str, *args: Any, **kwargs: Any) -> list[Triple]:
        """Async version of `extract`."""
        self._require(extractor=True)
        return await self._extractor.aextract(text, *args, **kwargs)

    async def alink(
        self,
        mention: Mention,
        search_limit: int | None = None,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:
        """Async version of `link`."""
        self._require(disambiguator=True)
        candidates = await self.acandidates_lookup(mention, search_limit)
        return await self._adisambiguate(mention, candidates, limit, *args, **kwargs)

    async def aextract_and_link(
        self,
        text: str,
        context: str | None = None,
        language: str | None = None,
        search_limit: int | None = None,
        limit: int = 10,
        max_concurrency: int = 8,
        **kwargs: Any,
    ) -> list[tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        """Async version of `extract_and_link`.

        The mentions of all triples are linked concurrently (at most
        `max_concurrency` at a time); identical mentions are linked once.
        """
        self._require(disambiguator=True)
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be a positive integer.')
        triples = await self.aextract(text, **kwargs)
        mentions = TripleExtractor.build_mentions(
            triples, text, context=context, language=language)
        slots = asyncio.Semaphore(max_concurrency)
        links: dict[tuple, asyncio.Future] = {}

        async def link_one(m: Mention) -> list[LinkResult]:
            async with slots:
                return await self.alink(m, search_limit=search_limit, limit=limit)

        def link(m: Mention) -> asyncio.Future:
            key = (m.term, m.entity_type, m.text, m.context, m.language)
            if key not in links:
                links[key] = asyncio.ensure_future(link_one(m))
            return links[key]

        linked = await asyncio.gather(*(
            asyncio.gather(*(link(m) for m in ms)) for ms in mentions))
        return [(triple, tuple(parts)) for triple, parts in zip(triples, linked)]

    async def _adisambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit: int,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:
        if not candidates:
            LOG.info(f'No candidates found for `{mention.term}`.')
            return []
        results = await self._disambiguator.adisambiguate(
            mention, candidates, limit, *args, **kwargs)
        return [LinkResult.from_disambiguation(mention, r) for r in results]

    def aconstruct(
        self,
        text: str,
        language: str | None = None,
        search_limit: int | None = None,
        max_concurrency: int = 8,
    ) -> AsyncIterator[Statement]:
        """Builds KIF statements from a text (async iterator).

        Pipeline:
            1. the text is split into paragraphs, and each paragraph into
               sentences;
            2. triples are extracted from every sentence; sentences go to the
               extractor in groups of its `batch_size`;
            3. each triple becomes three mentions — subject and object as
               items, predicate as a property — whose `text` is the sentence
               and whose `context` is the paragraph (omitted when the
               paragraph is the sentence itself);
            4. every mention is linked (search + disambiguation). Searches
               are shared between mentions with the same term, and identical
               mentions are linked only once;
            5. each fully linked triple becomes
               `Statement(subject, ValueSnak(property, value))`.

        All the work runs concurrently (at most `max_concurrency` calls at a
        time), earlier sentences first. Statements are yielded in text
        order, each as soon as its triple is linked; stopping the iteration
        early cancels the pending work.

        Args:
            text: Input text. Paragraphs are separated by blank lines.
            language: Language code given to the mentions (e.g. "en").
            search_limit: Candidates retrieved per mention. Defaults to
                the instance's `search_limit`.
            max_concurrency: Maximum number of extraction, search and
                disambiguation calls running at the same time.

        Yields:
            Statement: One per distinct linked statement. Triples with a
            part that could not be linked are skipped.

        Raises:
            RuntimeError: If no extractor or no disambiguator is configured
                (raised by the call itself, before iterating).
        """
        self._require(extractor=True, disambiguator=True)
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be a positive integer.')
        return self._aconstruct(text, language, self._limit(search_limit), max_concurrency)

    @staticmethod
    def sentences_with_context(text: str) -> list[tuple[str, str | None]]:
        """(sentence, paragraph) pairs of `text`, as used by `construct`.

        The paragraph is None when it has a single sentence.
        """
        units = []
        for paragraph in split_paragraphs(text):
            sentences = split_sentences(paragraph)
            context = paragraph if len(sentences) > 1 else None
            units.extend((s, context) for s in sentences)
        return units

    async def _aconstruct(
        self,
        text: str,
        language: str | None,
        search_limit: int,
        max_concurrency: int,
    ) -> AsyncIterator[Statement]:
        slots = _PrioritySlots(max_concurrency)
        searches: dict[tuple, asyncio.Future] = {}
        links: dict[tuple, asyncio.Future] = {}

        async def search(m: Mention, rank: int) -> list[Candidate]:
            async with slots(rank):
                return await asyncio.to_thread(self.candidates_lookup, m, search_limit)

        async def link_one(m: Mention, rank: int) -> LinkResult | None:
            try:
                key = (m.term, m.entity_type, m.language)
                if key not in searches:
                    searches[key] = asyncio.ensure_future(search(m, rank))
                candidates = await searches[key]
                if not candidates:
                    return None
                async with slots(rank):
                    results = await self._adisambiguate(m, candidates, 1)
                return results[0] if results else None
            except Exception as e:
                LOG.warning(f'Could not link `{m.term}`: {e}')
                return None

        def link(m: Mention, rank: int) -> asyncio.Future:
            key = (m.term, m.entity_type, m.text, m.context, m.language)
            if key not in links:
                links[key] = asyncio.ensure_future(link_one(m, rank))
            return links[key]

        async def process(first: int, batch: list[tuple[str, str | None]]) -> list[
                tuple[Triple, asyncio.Future]]:
            async with slots(first):
                extracted = await self._extractor.aextract_batch([s for s, _ in batch])
            scheduled = []
            for rank, (sentence, context), triples in zip(
                    itertools.count(first), batch, extracted):
                mentions = TripleExtractor.build_mentions(
                    triples, sentence, context=context, language=language)
                scheduled.extend(
                    (t, asyncio.gather(*(link(m, rank) for m in ms)))
                    for t, ms in zip(triples, mentions))
            return scheduled

        units = self.sentences_with_context(text)
        size = max(1, self._extractor.batch_size)
        tasks = [asyncio.ensure_future(process(i, units[i:i + size]))
                 for i in range(0, len(units), size)]

        seen: set[Statement] = set()
        try:
            for task in tasks:
                for triple, parts in await task:
                    subject, predicate, obj = await parts
                    if subject is None or predicate is None or obj is None:
                        LOG.debug(f'Skipping partially linked triple {triple}')
                        continue
                    statement = self._to_statement(subject, predicate, obj)
                    if statement not in seen:
                        seen.add(statement)
                        yield statement
        finally:
            pending = [*tasks, *searches.values(), *links.values()]
            for future in pending:
                future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _to_statement(
            subject: LinkResult, predicate: LinkResult, obj: LinkResult) -> Statement:
        """`Statement(subject, ValueSnak(property, value))` from linked parts."""
        return Statement(
            Item(subject.candidate.iri),
            ValueSnak(Property(predicate.candidate.iri), Item(obj.candidate.iri)))

    def candidates_lookup(self, mention: Mention, limit: int | None = None) -> list[Candidate]:
        """Fetches candidate entities from the knowledge base for a mention.

        Transient failures of the knowledge base (rate limits, server and
        network errors) are retried up to `search_retries` times with
        exponential backoff; if they persist, the error is raised.

        Args:
            mention (Mention): The mention to look up candidates for.
            limit (int, optional): Maximum number of candidates to retrieve.
                Defaults to the instance's search limit.
        """
        for attempt in range(self.search_retries + 1):
            try:
                return self._candidates(mention, self._limit(limit))
            except Exception as e:
                wait = _retry_after(e)
                if wait is None or attempt == self.search_retries:
                    raise
                delay = max(wait, self.search_backoff * 2 ** attempt) * random.uniform(1, 1.5)
                LOG.info(f'Search for `{mention.term}` failed ({e}); retrying in {delay:.1f}s')
                time.sleep(min(delay, 60.0))
        raise AssertionError('unreachable')

    def _candidates(self, mention: Mention, limit: int) -> list[Candidate]:
        kb = self._knowledge_base
        if mention.entity_type is EntityType.ITEM:
            found = kb.item_descriptor(search=mention.term, limit=limit)
        elif mention.entity_type is EntityType.PROPERTY:
            found = kb.property_descriptor(search=mention.term, limit=limit)
        else:
            found = itertools.chain(
                kb.item_descriptor(search=mention.term, limit=limit),
                kb.property_descriptor(search=mention.term, limit=limit))

        language = mention.language or 'en'

        def text_of(data: dict[str, Any], key: str) -> str:
            value = data.get(key, {}).get(language)
            return value.content if value else ''

        return [Candidate(id=entity.iri.content, iri=entity.iri.content,
                          label=text_of(desc, 'labels'),
                          description=text_of(desc, 'descriptions'))
                for entity, desc in _skip_errors(found)]

    def link(
        self,
        mention: Mention,
        search_limit: int | None = None,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:
        """Links a mention to the most relevant entities in the knowledge base.

        Args:
            mention (Mention): The mention to link.
            search_limit (int, optional): Maximum number of candidates to
                retrieve. Defaults to the instance's search limit.
            limit (int, optional): Maximum number of results after
                disambiguation. Defaults to 10.
        """
        self._require(disambiguator=True)
        candidates = self.candidates_lookup(mention, search_limit)
        if not candidates:
            LOG.info(f'No candidates found for `{mention.term}`.')
            return []
        return [LinkResult.from_disambiguation(mention, result)
                for result in self._disambiguator.disambiguate(
                    mention, candidates, limit, *args, **kwargs)]

    def extract(self, text: str, *args: Any, **kwargs: Any) -> list[Triple]:
        """Extracts subject-predicate-object triples from a text."""
        self._require(extractor=True)
        return self._extractor.extract(text, *args, **kwargs)

    def extract_and_link(
        self,
        text: str,
        context: str | None = None,
        language: str | None = None,
        search_limit: int | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        """Extracts triples from a text and links subject, predicate and object.

        Blocking wrapper of `aextract_and_link` (mentions are linked
        concurrently; also usable inside Jupyter).

        Returns:
            One entry per triple: (triple, (subject_links, predicate_links, object_links)).
        """
        return _run_sync(self.aextract_and_link(
            text, context=context, language=language,
            search_limit=search_limit, limit=limit, **kwargs))

    def construct(
        self,
        text: str,
        language: str | None = None,
        search_limit: int | None = None,
        max_concurrency: int = 8,
    ) -> Iterator[Statement]:
        """Builds KIF statements from a text (iterator).

        Blocking counterpart of `aconstruct`: statements are produced lazily,
        in text order, while the remaining work proceeds in the background.
        Also usable inside Jupyter. Wrap it in `list(...)` to get them all.
        """
        return _iterate_sync(self.aconstruct(
            text, language=language, search_limit=search_limit,
            max_concurrency=max_concurrency))
