# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import logging
import re
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, Final

from kif_lib import KIF_Object

LOG = logging.getLogger(__name__)

Span = tuple[int, int]
Cluster = list[Span]

SUBJECT_PRONOUNS = frozenset({'he', 'she', 'it', 'they', 'him', 'them', 'himself', 'herself',
                              'itself', 'themselves'})
POSSESSIVE_PRONOUNS = frozenset({'his', 'its', 'their'})
STANDALONE_POSSESSIVES = frozenset({'hers', 'theirs'})
AMBIGUOUS_PRONOUNS = frozenset({'her'})
PRONOUNS = (SUBJECT_PRONOUNS | POSSESSIVE_PRONOUNS | STANDALONE_POSSESSIVES
            | AMBIGUOUS_PRONOUNS)
RELATIVE_WORDS = frozenset({'that', 'who', 'whom', 'whose', 'which'})
DETERMINERS = frozenset({'the', 'a', 'an', 'this', 'that', 'these', 'those'})
NOT_A_NOUN = frozenset({
    'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of', 'as', 'than',
    'that', 'who', 'which', 'when', 'while', 'so', 'if', 'because', 'before', 'after', 'into',
    'about', 'over', 'under', 'through', 'during', 'until', 'since', 'is', 'was', 'are', 'were',
    'has', 'had', 'have', 'will', 'would', 'could', 'can', 'did', 'does', 'also', 'then', 'too'})

_NEXT_WORD = re.compile(r"\s*([A-Za-z][\w'-]*)")
_SENTENCE_START = re.compile(r'(?:^|[.!?…]["\'”’)\]]*\s+|\n\s*)$')


class CoreferenceResolver:
    """Base class for coreference resolvers.

    A resolver rewrites a text so that references to an entity become
    explicit: *"Marie Curie was born in Warsaw. She won the Nobel Prize."*
    becomes *"Marie Curie was born in Warsaw. Marie Curie won the Nobel
    Prize."* It runs before any other stage of `KBEL`, so extractors and
    disambiguators see names instead of pronouns.

    Plugins are created by name, ``CoreferenceResolver('fastcoref', ...)``.
    The ones shipped with KBEL (`builtin`) are imported on first use; a
    custom plugin is available once the module that defines it is imported.

    There are two ways to implement one:

    - `_clusters(texts)`: return, for each text, its coreference clusters
      (lists of character spans of the mentions of one entity). The
      substitution is shared: the first nominal mention of a cluster
      replaces its pronouns (`replace='pronouns'`) or all its other
      mentions (`replace='all'`).
    - `_resolve_batch(texts)`: return the rewritten texts directly (e.g. a
      resolver that asks an LLM to rewrite the text).

    Implementations must be thread-safe: `aresolve` runs them in a worker
    thread.

    Args:
        strategy_name: Name of the coreference plugin.
        replace: ``'pronouns'`` replaces only third-person pronouns by the
            cluster's name (*She* → *Marie Curie*, *her* → *Marie Curie's*);
            ``'all'`` replaces every later mention (*the physicist* → *Marie
            Curie*) as well.
        representative: Which mention names a cluster: ``'first'`` or
            ``'longest'`` among its nominal mentions.
        max_words: Mentions with more words are never used as the name of a
            cluster (they are clauses, not names).
        batch_size: Texts resolved per call by `iter_resolve`.
    """

    strategy_name: ClassVar[str]
    registry: Final[dict[str, type['CoreferenceResolver']]] = {}
    builtin: Final[dict[str, str]] = {
        'fastcoref': 'kbel.coreference.fastcoref',
    }

    @classmethod
    def _register(cls, resolver: type['CoreferenceResolver'], strategy_name: str):
        resolver.strategy_name = strategy_name
        cls.registry[strategy_name] = resolver

    @classmethod
    def __init_subclass__(cls, strategy_name: str):
        CoreferenceResolver._register(cls, strategy_name)

    def __new__(cls, strategy_name: str, *args: Any, **kwargs: Any):
        if strategy_name not in cls.registry and strategy_name in cls.builtin:
            importlib.import_module(cls.builtin[strategy_name])
        KIF_Object._check_arg(
            strategy_name,
            strategy_name in cls.registry,
            f'no such coreference plugin "{strategy_name}"',
            CoreferenceResolver,
            'strategy_name',
            1,
            ValueError,
        )
        return super().__new__(cls.registry[strategy_name])

    def __init__(self, replace: str = 'pronouns', representative: str = 'first',
                 max_words: int | None = 8, batch_size: int = 8):
        if replace not in ('pronouns', 'all'):
            raise ValueError("replace must be 'pronouns' or 'all'.")
        if representative not in ('first', 'longest'):
            raise ValueError("representative must be 'first' or 'longest'.")
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        self.replace = replace
        self.representative = representative
        self.max_words = max_words
        self.batch_size = batch_size

    def resolve(self, text: str) -> str:
        """The text with its coreferences made explicit."""
        return self.resolve_batch([text])[0]

    async def aresolve(self, text: str) -> str:
        """Async version of `resolve`; does not block the event loop."""
        return (await self.aresolve_batch([text]))[0]

    def resolve_batch(self, texts: list[str]) -> list[str]:
        """Resolves each text (one result per text, in order)."""
        return self._resolve_batch(list(texts)) if texts else []

    async def aresolve_batch(self, texts: list[str]) -> list[str]:
        """Async version of `resolve_batch`."""
        return await asyncio.to_thread(self.resolve_batch, texts)

    def iter_resolve(self, texts: Iterable[str]) -> Iterator[str]:
        """Resolves many texts lazily: results are yielded, in order, as
        each group of `batch_size` texts is done."""
        group: list[str] = []
        for text in texts:
            group.append(text)
            if len(group) == self.batch_size:
                yield from self.resolve_batch(group)
                group = []
        yield from self.resolve_batch(group)

    def clusters(self, text: str) -> list[Cluster]:
        """The coreference clusters of a text, as lists of character spans."""
        return self._clusters([text])[0]

    @abstractmethod
    def _clusters(self, texts: list[str]) -> list[list[Cluster]]:
        """Coreference clusters of each text (see the class docstring)."""
        ...

    def _resolve_batch(self, texts: list[str]) -> list[str]:
        return [self.substitute(text, clusters)
                for text, clusters in zip(texts, self._clusters(texts))]

    def substitute(self, text: str, clusters: list[Cluster]) -> str:
        """Rewrites `text` replacing the mentions of each cluster."""
        edits: list[tuple[int, int, Span, str]] = []
        for cluster in clusters:
            edits.extend(self._edits(text, sorted(set(cluster))))
        edits.sort()
        return self._render(text, 0, len(text), edits, frozenset())

    def _render(self, text: str, begin: int, end: int,
                edits: list[tuple[int, int, Span, str]], busy: frozenset[Span]) -> str:
        """`text[begin:end]` with the edits inside it applied.

        A replacement is the name of its cluster, which may itself contain
        mentions that are replaced; `busy` stops names that contain each
        other from recursing forever.
        """
        result, position = [], begin
        for start, stop, name_span, kind in edits:
            if start < position or stop > end or start < begin:
                continue
            result.append(text[position:start])
            name = (text[slice(*name_span)] if name_span in busy else self._render(
                text, name_span[0], name_span[1], edits, busy | {name_span}))
            result.append(self._for(kind, ' '.join(name.split()), text, start, stop))
            position = stop
        result.append(text[position:end])
        return ''.join(result)

    def _name_of(self, text: str, cluster: Cluster) -> Span | None:
        names = [(s, e) for s, e in cluster
                 if text[s:e].strip().lower() not in PRONOUNS
                 and (self.max_words is None or len(text[s:e].split()) <= self.max_words)
                 and re.search(r'[A-Za-z]', text[s:e])
                 and not RELATIVE_WORDS.intersection(text[s:e].lower().split())]
        if not names:
            return None
        if self.representative == 'longest':
            return max(names, key=lambda span: (span[1] - span[0], -span[0]))
        return names[0]

    def _edits(self, text: str, cluster: Cluster) -> list[tuple[int, int, Span, str]]:
        name_span = self._name_of(text, cluster)
        if name_span is None:
            return []
        edits = []
        for start, end in cluster:
            if start <= name_span[0] or (start, end) == name_span:
                continue
            word = text[start:end].strip().lower()
            if word in PRONOUNS:
                kind = word if word not in AMBIGUOUS_PRONOUNS else self._her(text, end)
            elif self.replace == 'all' and re.search(r'[A-Za-z]', text[start:end]):
                kind = 'name'
            else:
                continue
            edits.append((start, end, name_span, kind))
        return edits

    @staticmethod
    def _her(text: str, end: int) -> str:
        following = _NEXT_WORD.match(text, end)
        return 'his' if following and following.group(1).lower() not in NOT_A_NOUN else 'him'

    @staticmethod
    def _for(kind: str, name: str, text: str, start: int, stop: int) -> str:
        if kind in POSSESSIVE_PRONOUNS or kind in STANDALONE_POSSESSIVES:
            name = f"{name}'" if name.endswith('s') else f"{name}'s"
        first, _, rest = name.partition(' ')
        if first.lower() in DETERMINERS and rest:
            at_start = bool(_SENTENCE_START.search(text[:start]))
            name = f'{first.capitalize() if at_start else first.lower()} {rest}'
        return name
