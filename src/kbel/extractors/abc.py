# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Final

from kif_lib import KIF_Object

from kbel.core.mention import EntityType, Mention
from kbel.core.triple import Triple

LOG = logging.getLogger(__name__)


class TripleExtractor:
    """Base class for triple extractors.

    Subclasses implement `_extract`. The async API (`aextract`,
    `aextract_batch`) runs `_extract` in worker threads by default, so it
    must be thread-safe; extractors with a natively async backend override
    `_aextract`, and those that process several texts at once more
    efficiently (e.g. a batched model) override `_extract_batch` and set
    `batch_size`.

    Parameters:
        strategy_name (str): Name of the extraction strategy plugin.
        graph (bool): Default for rendering a graph image on `extract`.
        graph_path (str | Path): Default output path of the graph image
            (format taken from the extension: png, svg, pdf or dot).
    """

    strategy_name: ClassVar[str]
    registry: Final[dict[str, type['TripleExtractor']]] = {}
    batch_size: int = 1

    @classmethod
    def _register(cls, extractor: type['TripleExtractor'], strategy_name: str):
        extractor.strategy_name = strategy_name
        cls.registry[strategy_name] = extractor

    @classmethod
    def __init_subclass__(cls, strategy_name: str):
        TripleExtractor._register(cls, strategy_name)

    def __new__(cls, strategy_name: str, *args: Any, **kwargs: Any):
        KIF_Object._check_arg(
            strategy_name,
            strategy_name in cls.registry,
            f'no such triple extractor plugin "{strategy_name}"',
            TripleExtractor,
            'strategy_name',
            1,
            ValueError,
        )
        return super().__new__(cls.registry[strategy_name])

    def __init__(self, graph: bool = False, graph_path: str | Path = 'graph.png'):
        self._graph = graph
        self._graph_path = Path(graph_path)

    def extract(
        self,
        text: str,
        prompt=None,
        graph: bool | None = None,
        graph_path: str | Path | None = None,
    ) -> list[Triple]:
        """Extracts triples from a text.

        Args:
            text (str): Input text to extract triples from.
            prompt: Optional prompt (used by LLM-based extractors).
            graph (bool, optional): Render a graph image of the triples.
                Defaults to the value given to the constructor.
            graph_path (str | Path, optional): Where to save the image.
                Defaults to the constructor's `graph_path`.
        """
        assert text and text.strip(), 'No text to extract triples from'
        triples = self._extract(text, prompt)
        self._maybe_graph(triples, graph, graph_path)
        return triples

    async def aextract(
        self,
        text: str,
        prompt=None,
        graph: bool | None = None,
        graph_path: str | Path | None = None,
    ) -> list[Triple]:
        """Async version of `extract`; does not block the event loop."""
        assert text and text.strip(), 'No text to extract triples from'
        triples = await self._aextract(text, prompt)
        if self._graph if graph is None else graph:
            await asyncio.to_thread(self._maybe_graph, triples, True, graph_path)
        return triples

    def extract_batch(self, texts: list[str], prompt=None) -> list[list[Triple]]:
        """Extracts the triples of each text (one list per text, in order)."""
        assert all(t and t.strip() for t in texts), 'No text to extract triples from'
        return self._extract_batch(texts, prompt)

    async def aextract_batch(self, texts: list[str], prompt=None) -> list[list[Triple]]:
        """Async version of `extract_batch`."""
        assert all(t and t.strip() for t in texts), 'No text to extract triples from'
        return await self._aextract_batch(texts, prompt)

    def _maybe_graph(self, triples: list[Triple], graph: bool | None,
                     graph_path: str | Path | None) -> None:
        if self._graph if graph is None else graph:
            self.to_graphviz(triples, graph_path or self._graph_path)

    @staticmethod
    def to_graphviz(triples: list[Triple], path: str | Path = 'graph.png') -> Path | None:
        """Renders the triples as a directed graph (subject -[predicate]-> object).

        Requires GraphViz (`dot`) in the PATH, except for the `.dot` extension,
        which just writes the DOT source.

        Returns:
            Path | None: The written file, or None if there were no triples.
        """
        if not triples:
            LOG.warning('No triples to draw; graph not generated.')
            return None

        def q(s: str) -> str:
            return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

        dot = '\n'.join([
            'digraph G {', '  rankdir=LR;', '  node [shape=box, style=rounded];',
            *(f'  {q(t.subject)} -> {q(t.object)} [label={q(t.predicate)}];' for t in triples),
            '}'])

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = path.suffix.lstrip('.').lower() or 'png'
        if fmt in ('dot', 'gv'):
            path.write_text(dot, encoding='utf-8')
            return path
        if shutil.which('dot') is None:
            raise RuntimeError(
                'GraphViz not found. Install it (apt-get install graphviz / '
                'brew install graphviz / winget install graphviz) and make sure `dot` is in the PATH.')
        subprocess.run(
            ['dot', f'-T{fmt}', '-o', str(path)],
            input=dot.encode('utf-8'), check=True, capture_output=True)
        return path

    @staticmethod
    def build_mentions(
        triples: list[Triple],
        text: str,
        context: str | None = None,
        language: str | None = None,
    ) -> list[tuple[Mention, Mention, Mention]]:
        """Builds the (subject, predicate, object) mentions of each triple.

        Subject and object are items, the predicate is a property; `text`
        is the text the triples come from and `context` its surroundings.
        """
        def m(term: str, entity_type: EntityType) -> Mention:
            return Mention(term=term, text=text, context=context,
                           entity_type=entity_type, language=language)

        return [(m(t.subject, EntityType.ITEM),
                 m(t.predicate, EntityType.PROPERTY),
                 m(t.object, EntityType.ITEM))
                for t in triples]

    @abstractmethod
    def _extract(self, text: str, prompt=None) -> list[Triple]:
        """Core extraction logic to be implemented by subclasses."""
        ...

    async def _aextract(self, text: str, prompt=None) -> list[Triple]:
        """Runs `_extract` in a worker thread; override for native async."""
        return await asyncio.to_thread(self._extract, text, prompt)

    def _extract_batch(self, texts: list[str], prompt=None) -> list[list[Triple]]:
        """Extracts each text in turn; override to process them together."""
        return [self._extract(t, prompt) for t in texts]

    async def _aextract_batch(self, texts: list[str], prompt=None) -> list[list[Triple]]:
        """Runs the batch at once, or the texts concurrently.

        A subclass that overrides `_extract_batch` gets it run in a worker
        thread; otherwise every text goes through `_aextract` concurrently.
        """
        if type(self)._extract_batch is not TripleExtractor._extract_batch:
            return await asyncio.to_thread(self._extract_batch, texts, prompt)
        return list(await asyncio.gather(*(self._aextract(t, prompt) for t in texts)))
