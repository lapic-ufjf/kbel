# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import dataclasses
import heapq
import itertools
import logging
import queue
import random
import threading
import time
from pathlib import Path
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, TypeVar

from kif_lib import (
    Item,
    NoValueSnak,
    Property,
    Quantity,
    QuantityDatatype,
    SomeValueSnak,
    Statement,
    ValueSnak,
)
from kif_lib.search import Search

from kbel.core.candidate import Candidate
from kbel.core.mention import EntityType, Mention
from kbel.core.results import LinkResult
from kbel.core.text import split_paragraphs, split_sentences
from kbel.core.triple import Triple
from kbel.core.values import (
    NO_VALUE,
    SOME_VALUE,
    literal,
    looks_like_literal,
    parse_quantity,
    snak_marker,
    value_class,
)
from kbel.coreference import CoreferenceResolver
from kbel.disambiguators import Disambiguator
from kbel.extractors import TripleExtractor
from kbel.search import WikidataVDB_Search

LOG = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = ['KBEL', 'WikidataVDB_Search']


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


@dataclasses.dataclass
class _Linked:
    """A triple and what its parts were resolved to (see `KBEL._astream`)."""

    triple: Triple
    subject: list[LinkResult] = dataclasses.field(default_factory=list)
    predicate: list[LinkResult] = dataclasses.field(default_factory=list)
    object: list[LinkResult] = dataclasses.field(default_factory=list)
    snak: str | None = None


class KBEL:
    """Links text to knowledge-base entities.

    Searches always run in worker threads: KIF's async HTTP client is bound
    to the event loop that first used it. Extraction and disambiguation use
    the plugins' async hooks.
    """

    _knowledge_base: Search
    _extractor: TripleExtractor | None = None
    _coreference: CoreferenceResolver | None = None
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
        coreference: CoreferenceResolver | str | None = None,
        **kwargs: Any,
    ):
        assert knowledge_base, 'Knowledge base can not be undefined.'
        self.search_limit = search_limit
        self.knowledge_base = knowledge_base
        if disambiguator is not None:
            self.disambiguator = disambiguator
        if extractor is not None:
            self.extractor = extractor
        if coreference is not None:
            self.coreference = coreference

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

    @property
    def coreference(self) -> CoreferenceResolver | None:
        return self._coreference

    @coreference.setter
    def coreference(self, value: CoreferenceResolver | str | None) -> None:
        self._coreference = CoreferenceResolver(value) if isinstance(value, str) else value

    def resolve_coreference(self, text: str) -> str:
        """The text with its coreferences made explicit (see `CoreferenceResolver`).

        Without a configured resolver the text is returned unchanged. Every
        method that takes a text (`extract`, `extract_and_link`, `construct`
        and their iterators) calls this first.
        """
        return text if self._coreference is None else self._coreference.resolve(text)

    async def aresolve_coreference(self, text: str) -> str:
        """Async version of `resolve_coreference`."""
        return text if self._coreference is None else await self._coreference.aresolve(text)

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

    def aextract(
        self,
        text: str,
        language: str | None = None,
        max_concurrency: int = 8,
        prompt: Any = None,
        graph: bool | None = None,
        graph_path: str | Path | None = None,
    ) -> AsyncIterator[Triple]:
        """Extracts the triples of a text, sentence by sentence (async iterator).

        Pipeline: the coreferences are resolved (if a resolver is
        configured), the text is split into paragraphs and sentences, and
        the sentences go to the extractor in groups of its `batch_size`,
        concurrently (at most `max_concurrency` calls at a time). Triples
        are yielded in text order as soon as their sentences are processed;
        stopping early cancels the rest.

        Args:
            text: Input text. Paragraphs are separated by blank lines.
            language: Unused by extraction itself; kept so that every
                iterator takes the same arguments.
            max_concurrency: Maximum number of concurrent extraction calls.
            prompt: Prompt given to the extractor (LLM extractors).
            graph: Render a graph of all the triples when the iteration
                ends (default: the extractor's setting).
            graph_path: Where to save the graph image.

        Raises:
            RuntimeError: If no extractor is configured (raised by the call
                itself, before iterating).
        """
        self._require(extractor=True)
        self._check_concurrency(max_concurrency)
        return self._aextract(text, max_concurrency, prompt, graph, graph_path)

    def extract(
        self,
        text: str,
        language: str | None = None,
        max_concurrency: int = 8,
        prompt: Any = None,
        graph: bool | None = None,
        graph_path: str | Path | None = None,
    ) -> Iterator[Triple]:
        """Extracts the triples of a text, sentence by sentence (iterator).

        Blocking counterpart of `aextract`: triples are produced lazily, in
        text order, while the remaining sentences are processed in the
        background. Also usable inside Jupyter. Wrap it in `list(...)` to
        get them all.
        """
        return _iterate_sync(self.aextract(
            text, language=language, max_concurrency=max_concurrency, prompt=prompt,
            graph=graph, graph_path=graph_path))

    def aextract_and_link(
        self,
        text: str,
        language: str | None = None,
        search_limit: int | None = None,
        limit: int = 10,
        max_concurrency: int = 8,
        prompt: Any = None,
    ) -> AsyncIterator[tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        """Extracts and links the triples of a text, as they are ready (async iterator).

        Same pipeline as `aconstruct`, but every triple is yielded — in text
        order — with the ranked links (up to `limit`) of its subject,
        predicate and object, instead of a statement. The links of the
        object depend on the datatype of the linked predicate: for a
        property whose values are items they are the candidates of the
        search; for one whose values are literals (date, number, string,
        ...) there is one link whose `entity` is the parsed value
        (`LinkResult.is_literal`); an object that says that the value is
        unknown or does not exist has no links. A part that cannot be
        linked has an empty list.
        """
        self._require(extractor=True, disambiguator=True)
        self._check_concurrency(max_concurrency)
        return self._amap_links(self._astream(
            text, language, self._limit(search_limit), max_concurrency, max(1, limit), prompt))

    def extract_and_link(
        self,
        text: str,
        language: str | None = None,
        search_limit: int | None = None,
        limit: int = 10,
        max_concurrency: int = 8,
        prompt: Any = None,
    ) -> Iterator[tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        """Extracts and links the triples of a text, as they are ready (iterator).

        Blocking counterpart of `aextract_and_link`; also usable inside Jupyter.
        """
        return _iterate_sync(self.aextract_and_link(
            text, language=language, search_limit=search_limit, limit=limit,
            max_concurrency=max_concurrency, prompt=prompt))

    def aconstruct(
        self,
        text: str,
        language: str | None = None,
        search_limit: int | None = None,
        max_concurrency: int = 8,
    ) -> AsyncIterator[Statement]:
        """Builds KIF statements from a text (async iterator).

        Pipeline:
            1. the coreferences of the text are resolved, if a coreference
               resolver is configured;
            2. the text is split into paragraphs, and each paragraph into
               sentences;
            3. triples are extracted from every sentence; sentences go to the
               extractor in groups of its `batch_size`;
            4. each triple becomes three mentions — subject and object as
               items, predicate as a property — whose `text` is the sentence
               and whose `context` is the paragraph (omitted when the
               paragraph is the sentence itself);
            5. the subject and the predicate are linked (search +
               disambiguation; searches are shared between mentions with the
               same term and identical mentions are linked only once);
            6. the object depends on the datatype of the predicate: an item
               is linked like the subject; a date, number, string, URL or
               text is parsed from the object (a quantity with a unit also
               links the unit); an object that says that the value is
               unknown or that there is none (see `kbel.core.values`) needs
               no value at all;
            7. each fully resolved triple becomes a KIF statement:
               `Statement(subject, ValueSnak(property, value))`, where the
               value is an `Item`, `Property` or literal (`Time`, `Quantity`,
               `String`, `Text`, `ExternalId`, `IRI`), or
               `Statement(subject, SomeValueSnak(property))` /
               `Statement(subject, NoValueSnak(property))`.

        All the work runs concurrently (at most `max_concurrency` calls at a
        time), earlier sentences first. Statements are yielded in text
        order, each as soon as its triple is resolved; stopping the
        iteration early cancels the pending work.

        Args:
            text: Input text. Paragraphs are separated by blank lines.
            language: Language code given to the mentions (e.g. "en").
            search_limit: Candidates retrieved per mention. Defaults to
                the instance's `search_limit`.
            max_concurrency: Maximum number of extraction, search and
                disambiguation calls running at the same time.

        Yields:
            Statement: One per distinct statement. Triples with a part that
            could not be resolved are skipped.

        Raises:
            RuntimeError: If no extractor or no disambiguator is configured
                (raised by the call itself, before iterating).
        """
        self._require(extractor=True, disambiguator=True)
        self._check_concurrency(max_concurrency)
        return self._aconstruct(text, language, self._limit(search_limit), max_concurrency)

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

    def alink_many(
        self,
        mentions: Iterable[Mention],
        search_limit: int | None = None,
        limit: int = 10,
        max_concurrency: int = 8,
    ) -> AsyncIterator[tuple[Mention, list[LinkResult]]]:
        """Links many mentions concurrently, yielding each as it is ready (async iterator).

        Yields ``(mention, links)`` in the order of `mentions`. Searches are
        shared between mentions with the same term. A mention whose search
        or disambiguation fails gets no links (the error is logged).
        """
        self._require(disambiguator=True)
        self._check_concurrency(max_concurrency)
        return self._alink_many(list(mentions), self._limit(search_limit), limit, max_concurrency)

    def link_many(
        self,
        mentions: Iterable[Mention],
        search_limit: int | None = None,
        limit: int = 10,
        max_concurrency: int = 8,
    ) -> Iterator[tuple[Mention, list[LinkResult]]]:
        """Links many mentions concurrently, yielding each as it is ready (iterator).

        Blocking counterpart of `alink_many`; also usable inside Jupyter.
        """
        return _iterate_sync(self.alink_many(
            mentions, search_limit=search_limit, limit=limit, max_concurrency=max_concurrency))

    @staticmethod
    def _check_concurrency(max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be a positive integer.')

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

    async def _aextract(
        self,
        text: str,
        max_concurrency: int,
        prompt: Any,
        graph: bool | None,
        graph_path: str | Path | None,
    ) -> AsyncIterator[Triple]:
        stream = self._astream(text, None, None, max_concurrency, None, prompt)
        triples: list[Triple] = []
        async with contextlib.aclosing(stream):
            async for linked in stream:
                triples.append(linked.triple)
                yield linked.triple
        if self._extractor._graph if graph is None else graph:
            await asyncio.to_thread(self._extractor._maybe_graph, triples, True, graph_path)

    async def _amap_links(self, stream: AsyncIterator['_Linked']) -> AsyncIterator[
            tuple[Triple, tuple[list[LinkResult], list[LinkResult], list[LinkResult]]]]:
        async with contextlib.aclosing(stream):
            async for linked in stream:
                yield linked.triple, (linked.subject, linked.predicate, linked.object)

    async def _astream(
        self,
        text: str,
        language: str | None,
        search_limit: int | None,
        max_concurrency: int,
        limit: int | None,
        prompt: Any = None,
    ) -> AsyncIterator['_Linked']:
        """Extracts (and, if `limit` is given, links) the triples of a text.

        Yields a `_Linked` per triple, in text order; without `limit`
        nothing is linked.
        """
        text = await self.aresolve_coreference(text)
        slots = _PrioritySlots(max_concurrency)
        searches: dict[tuple, asyncio.Future] = {}
        links: dict[tuple, asyncio.Future] = {}

        async def search(m: Mention, rank: int) -> list[Candidate]:
            async with slots(rank):
                return await asyncio.to_thread(self.candidates_lookup, m, search_limit)

        async def link_one(m: Mention, rank: int) -> list[LinkResult]:
            try:
                key = (m.term, m.entity_type, m.language)
                if key not in searches:
                    searches[key] = asyncio.ensure_future(search(m, rank))
                candidates = await searches[key]
                if not candidates:
                    return []
                async with slots(rank):
                    return await self._adisambiguate(m, candidates, limit)
            except Exception as e:
                LOG.warning(f'Could not link `{m.term}`: {e}')
                return []

        def link(m: Mention, rank: int) -> asyncio.Future:
            key = (m.term, m.entity_type, m.text, m.context, m.language)
            if key not in links:
                links[key] = asyncio.ensure_future(link_one(m, rank))
            return links[key]

        async def object_of(
                triple: Triple, mention: Mention, predicate: asyncio.Future,
                early: asyncio.Future | None, rank: int) -> tuple[list[LinkResult], str | None]:
            marker = snak_marker(triple.object)
            if marker is not None:
                return [], marker
            predicates = await predicate
            datatype = predicates[0].candidate.datatype if predicates else None
            kind = value_class(datatype)
            if kind == 'item':
                return await (early if early is not None else link(mention, rank)), None
            if kind == 'property':
                return await link(
                    dataclasses.replace(mention, entity_type=EntityType.PROPERTY), rank), None
            if kind == 'lexeme':
                return [], None
            if isinstance(datatype, QuantityDatatype):
                parsed = parse_quantity(triple.object)
                if parsed is None:
                    return [], None
                amount, unit_text = parsed
                if unit_text is None:
                    return [LinkResult.from_literal(Quantity(amount))], None
                unit = await link(dataclasses.replace(
                    mention, term=unit_text, entity_type=EntityType.ITEM), rank)
                return ([LinkResult.from_literal(Quantity(amount, unit[0].entity))]
                        if unit else []), None
            value = literal(datatype, triple.object, mention.language)
            return ([LinkResult.from_literal(value)] if value is not None else []), None

        async def resolve(triple: Triple, mentions: tuple[Mention, Mention, Mention],
                          rank: int) -> '_Linked':
            subject, predicate, obj = mentions
            predicate_links = link(predicate, rank)
            early = (None if looks_like_literal(triple.object) else link(obj, rank))
            object_task = asyncio.ensure_future(
                object_of(triple, obj, predicate_links, early, rank))
            try:
                subject_links, predicate_result = await asyncio.gather(
                    link(subject, rank), predicate_links)
                object_links, snak = await object_task
            finally:
                object_task.cancel()
            return _Linked(triple, subject_links, predicate_result, object_links, snak)

        async def process(first: int, batch: list[tuple[str, str | None]]) -> list[
                tuple[Triple, asyncio.Future | None]]:
            async with slots(first):
                extracted = await self._extractor.aextract_batch(
                    [s for s, _ in batch], prompt)
            scheduled = []
            for rank, (sentence, context), triples in zip(
                    itertools.count(first), batch, extracted):
                if limit is None:
                    scheduled.extend((t, None) for t in triples)
                    continue
                mentions = TripleExtractor.build_mentions(
                    triples, sentence, context=context, language=language)
                scheduled.extend(
                    (t, asyncio.ensure_future(resolve(t, ms, rank)))
                    for t, ms in zip(triples, mentions))
            return scheduled

        units = self.sentences_with_context(text)
        size = max(1, self._extractor.batch_size)
        tasks = [asyncio.ensure_future(process(i, units[i:i + size]))
                 for i in range(0, len(units), size)]
        resolved: list[asyncio.Future] = []
        try:
            for task in tasks:
                for triple, pending in await task:
                    if pending is None:
                        yield _Linked(triple)
                    else:
                        resolved.append(pending)
                        yield await pending
        finally:
            pending_all = [*tasks, *resolved, *searches.values(), *links.values()]
            for future in pending_all:
                future.cancel()
            await asyncio.gather(*pending_all, return_exceptions=True)

    async def _aconstruct(
        self,
        text: str,
        language: str | None,
        search_limit: int,
        max_concurrency: int,
    ) -> AsyncIterator[Statement]:
        seen: set[Statement] = set()
        stream = self._astream(text, language, search_limit, max_concurrency, 1)
        async with contextlib.aclosing(stream):
            async for linked in stream:
                statement = self._to_statement(linked)
                if statement is None:
                    LOG.debug(f'Skipping unresolved triple {linked.triple}')
                elif statement not in seen:
                    seen.add(statement)
                    yield statement

    async def _alink_many(
        self,
        mentions: list[Mention],
        search_limit: int,
        limit: int,
        max_concurrency: int,
    ) -> AsyncIterator[tuple[Mention, list[LinkResult]]]:
        slots = _PrioritySlots(max_concurrency)
        searches: dict[tuple, asyncio.Future] = {}

        async def search(m: Mention, rank: int) -> list[Candidate]:
            async with slots(rank):
                return await asyncio.to_thread(self.candidates_lookup, m, search_limit)

        async def link_one(rank: int, m: Mention) -> list[LinkResult]:
            try:
                key = (m.term, m.entity_type, m.language)
                if key not in searches:
                    searches[key] = asyncio.ensure_future(search(m, rank))
                candidates = await searches[key]
                if not candidates:
                    return []
                async with slots(rank):
                    return await self._adisambiguate(m, candidates, limit)
            except Exception as e:
                LOG.warning(f'Could not link `{m.term}`: {e}')
                return []

        tasks = [asyncio.ensure_future(link_one(i, m)) for i, m in enumerate(mentions)]
        try:
            for mention, task in zip(mentions, tasks):
                yield mention, await task
        finally:
            pending = [*tasks, *searches.values()]
            for future in pending:
                future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _to_statement(linked: '_Linked') -> Statement | None:
        """The KIF statement of a resolved triple, or None if it is incomplete.

        A triple whose object says that the value is unknown (or that there
        is none) becomes a `SomeValueSnak` (`NoValueSnak`); otherwise the
        snak is a `ValueSnak` with the linked item or property, or the
        literal parsed from the text.
        """
        if not linked.subject or not linked.predicate:
            return None
        subject, predicate = linked.subject[0].entity, linked.predicate[0].entity
        try:
            if linked.snak == SOME_VALUE:
                snak = SomeValueSnak(predicate)
            elif linked.snak == NO_VALUE:
                snak = NoValueSnak(predicate)
            elif linked.object:
                snak = ValueSnak(predicate, linked.object[0].entity)
            else:
                return None
            return Statement(subject, snak)
        except (TypeError, ValueError) as e:
            LOG.debug(f'The triple {linked.triple} is not a valid statement: {e}')
            return None

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
                          description=text_of(desc, 'descriptions'),
                          datatype=getattr(entity, 'range', None)
                          if isinstance(entity, Property) else None)
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
