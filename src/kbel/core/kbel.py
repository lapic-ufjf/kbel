# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import contextlib
import heapq
import itertools
import logging
import queue
import threading
from collections.abc import AsyncIterator, Coroutine

from kif_lib.typing import Any, Iterator, TypeVar

from kif_lib import Item, Property, Statement, ValueSnak
from kif_lib.search import Search
from kbel.search import WikidataVDB_Search
from kbel.core.mention import Mention, EntityType
from kbel.core.candidate import Candidate
from kbel.core.triple import Triple
from kbel.core.results import LinkResult
from kbel.core.text import split_paragraphs, split_sentences
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor

LOG = logging.getLogger(__name__)

T = TypeVar('T')



def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` to completion from synchronous code.

    Works both in plain scripts and where an event loop is already running
    (e.g. Jupyter), in which case the coroutine runs in a helper thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

class _PrioritySlots:
    """A semaphore whose waiters are served by priority (lowest first).

    `aconstruct` gives every unit of work the index of its sentence, so the
    work for earlier sentences runs first and statements can be yielded in
    text order without waiting for the rest of the document.
    """

    def __init__(self, n: int):
        self._free = n
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()     # FIFO among equal priorities

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
                    self._release()          # got the slot, but won't use it
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            _, _, granted = heapq.heappop(self._waiters)
            if not granted.done():           # skip cancelled waiters
                granted.set_result(None)
                return
        self._free += 1


def _iterate_sync(agen: AsyncIterator[T]) -> Iterator[T]:
    """Consume an async iterator from synchronous code, lazily.

    The async iterator runs on its own event loop in a helper thread (so this
    also works where a loop is already running, e.g. Jupyter); items are
    handed over through a queue as soon as they are produced. Closing the
    returned iterator early cancels the async one.
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
        except BaseException as e:      # forwarded to the consumer
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
            try:
                state['loop'].call_soon_threadsafe(state['task'].cancel)
            except RuntimeError:        # loop already closed: nothing to stop
                pass
        worker.join()


class KBEL:

    _knowledge_base: Search
    _extractor: TripleExtractor | None = None
    _disambiguator: Disambiguator | None = None
    _search_limit = 10

    @property
    def search_limit(self) -> int:
        return self._search_limit

    @search_limit.setter
    def search_limit(self, value: int):
        if value <= 0:
            raise ValueError(
                'The limit to lookup candidates must be bigger than zero'
            )
        self._search_limit = value

    @property
    def knowledge_base(self) -> Search:
        return self._knowledge_base
    

    @knowledge_base.setter
    def knowledge_base(self, value: Search | str) -> None:
        if isinstance(value, str):
            value = Search(value, limit=self._search_limit)

        self._knowledge_base = value

    @property
    def disambiguator(self) -> Disambiguator | None:
        return self._disambiguator

    @disambiguator.setter
    def disambiguator(self, value: Disambiguator | str) -> None:
        if isinstance(value, str):
            value = Disambiguator(value)
        self._disambiguator = value

    @property
    def extractor(self) -> TripleExtractor | None:
        return self._extractor

    @extractor.setter
    def extractor(self, value: TripleExtractor | str | None) -> None:
        if isinstance(value, str):
            value = TripleExtractor(value)
        self._extractor = value


    def __init__(self, 
                 knowledge_base: Search | str, disambiguator: Disambiguator | str | None = None, extractor: TripleExtractor | str | None = None, search_limit=10, **kwargs):
        assert knowledge_base, 'Knowledge base can not be undefined.'
        self._search_limit = search_limit
        self.knowledge_base = knowledge_base
        if disambiguator is not None:
            self.disambiguator = disambiguator
        if extractor is not None:
            self.extractor = extractor


    # -- async API -----------------------------------------------------------
    #
    # Extraction and disambiguation go through the plugins' async hooks
    # (`aextract`, `adisambiguate`): natively async plugins (e.g. LLMs) run
    # on the event loop, the others in worker threads. Searches always run
    # in worker threads: KIF's async HTTP client is bound to the event loop
    # that first used it, so it breaks across `asyncio.run` calls.

    async def acandidates_lookup(
            self, mention: Mention, limit: int | None = None) -> list[Candidate]:
        """Async version of `candidates_lookup`."""
        return await asyncio.to_thread(self.candidates_lookup, mention, limit)

    async def aextract(self, text: str, *args: Any, **kwargs: Any) -> list[Triple]:
        """Async version of `extract`."""
        if self._extractor is None:
            raise RuntimeError("No extractor has been configured.")
        return await self._extractor.aextract(text, *args, **kwargs)

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
        if self._disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")
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
            asyncio.gather(*(link(m) for m in triple_mentions))
            for triple_mentions in mentions))
        return [(triple, tuple(parts)) for triple, parts in zip(triples, linked)]

    async def alink(
        self,
        mention: Mention,
        search_limit: int | None = None,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:
        """Async version of `link`."""
        if self.disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")
        candidates = await self.acandidates_lookup(mention, search_limit)
        return await self._adisambiguate(mention, candidates, limit, *args, **kwargs)

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
            2. triples are extracted from every sentence;
            3. each triple becomes three mentions — subject and object as
               items, predicate as a property — whose `text` is the sentence
               and whose `context` is the paragraph;
            4. every mention is linked (search + disambiguation). Searches
               are shared between mentions with the same term, and identical
               mentions are linked only once;
            5. each fully linked triple becomes
               `Statement(subject, ValueSnak(property, value))`.

        All the work is scheduled at once and runs concurrently (at most
        `max_concurrency` calls at a time). Statements are yielded in text
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
        if self._extractor is None:
            raise RuntimeError("No extractor has been configured.")
        if self._disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be a positive integer.')
        if search_limit is None or search_limit <= 0:
            search_limit = self._search_limit
        return self._aconstruct(text, language, search_limit, max_concurrency)

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

        # `rank` is the index of the sentence the work is for: lower first.
        async def candidates_of(m: Mention, rank: int) -> list[Candidate]:
            key = (m.term, m.entity_type, m.language)
            if key not in searches:
                async def search() -> list[Candidate]:
                    async with slots(rank):
                        return await asyncio.to_thread(
                            self.candidates_lookup, m, search_limit)
                searches[key] = asyncio.ensure_future(search())
            return await searches[key]

        async def link_one(m: Mention, rank: int) -> LinkResult | None:
            try:
                candidates = await candidates_of(m, rank)
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

        async def process(rank: int, sentence: str, paragraph: str) -> list[
                tuple[Triple, asyncio.Future]]:
            """Extracts a sentence's triples and schedules their links."""
            async with slots(rank):
                triples = await self._extractor.aextract(sentence)
            mentions = TripleExtractor.build_mentions(
                triples, sentence, context=paragraph, language=language)
            return [(t, asyncio.gather(*(link(m, rank) for m in ms)))
                    for t, ms in zip(triples, mentions)]

        # 1-4. Schedule everything; each sentence links as soon as extracted.
        units = [(sentence, paragraph)
                 for paragraph in split_paragraphs(text)
                 for sentence in split_sentences(paragraph)]
        tasks = [asyncio.ensure_future(process(rank, *u))
                 for rank, u in enumerate(units)]

        # 5. Yield statements in text order, as they become ready.
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

    def candidates_lookup(self, mention: Mention, limit: int = None) -> list[Candidate]:
        """Fetches candidate entities from the knowledge base for a given mention.
        
        Args:
            mention (Mention): The mention to look up candidates for.
            limit (int, optional): Maximum number of candidates to retrieve.
                Defaults to the instance's search limit.
        """

        if limit is None or limit <= 0:
            limit = self._search_limit

        @staticmethod
        def _safe_next(it: Iterator) -> Iterator:
                """Safely iterate over an iterator, skipping errors."""
                while True:
                    try:
                        yield next(it)
                    except StopIteration:
                        break
                    except Exception as e:
                        LOG.info(f'Error fetching item: {e}')
                        continue

        @staticmethod
        def _extract_text(data: dict[str, Any], key: str) -> str:
            """Extract the text for a given key and language."""
            l = mention.language if mention.language else 'en'
            value = data.get(key, {}).get(l)
            return value.content if value else ''
        
       
        if mention.entity_type is EntityType.ITEM:
            found_candidates =  self._knowledge_base.item_descriptor(search=mention.term, limit=limit)
        elif mention.entity_type is EntityType.PROPERTY:
            found_candidates =  self._knowledge_base.property_descriptor(search=mention.term, limit=limit)
        else:
            from itertools import chain
            found_candidates = chain(
                self._knowledge_base.item_descriptor(search=mention.term, limit=limit),
                self._knowledge_base.property_descriptor(search=mention.term, limit=limit),
            )

        candidates = []
        for entity, desc in _safe_next(iter(found_candidates)):
            candidate = Candidate(
                id = entity.iri.content,
                label = _extract_text(desc, 'labels'),
                description = _extract_text(desc, 'descriptions'),
                iri = entity.iri.content
            )
            candidates.append(candidate)

        return candidates

    def link(
        self,
        mention: Mention,
        search_limit: int = None,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:

        """Links a mention to the most relevant entities in the knowledge base.

        Args:
            mention (Mention): The mention to link.
            search_limit (int, optional): Maximum number of candidates to retrieve from the knowledge base.
                Defaults to the instance's search limit.
            limit (int, optional): Maximum number of top candidates to return after disambiguation.
                Defaults to 10.
        """
        if self.disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")

        if search_limit is None or search_limit <= 0:
            search_limit = self._search_limit

        candidates = self.candidates_lookup(mention, limit=search_limit)
        if not candidates:
            LOG.info(f'No candidates found for `{mention.term}`.')
            return []

        return [
            LinkResult.from_disambiguation(mention, result)
            for result in self._disambiguator.disambiguate(
                mention,
                candidates,
                limit,
                *args,
                **kwargs)
        ]
    
    def extract(
            self, 
            text: str, 
            *args: Any, 
            **kwargs: Any
            ) -> list[Triple]:
        """Extracts subject-predicate-object triples from a text.

        Args:
            text (str): Input text.
        """
        if self._extractor is None:
            raise RuntimeError("No extractor has been configured.")
        return self._extractor.extract(text, *args, **kwargs)

    def extract_and_link(
        self,
        text: str,
        context: str | None = None,
        language: str | None = None,
        search_limit: int = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        """Extracts triples from a text and links subject, predicate and object.

        Blocking wrapper of `aextract_and_link`: the mentions are linked
        concurrently (also usable inside Jupyter).

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
        in text order, while the remaining work proceeds concurrently in the
        background. Also usable inside Jupyter. Wrap it in `list(...)` to
        get all statements at once.
        """
        return _iterate_sync(self.aconstruct(
            text, language=language, search_limit=search_limit,
            max_concurrency=max_concurrency))
