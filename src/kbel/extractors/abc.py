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

    Subclasses must implement the `_extract` method, which contains the
    logic to extract subject-predicate-object triples from a text.

    Parameters:
        strategy_name (str): Name of the extraction strategy plugin.
        graph (bool): Default for rendering a graph image on `extract`.
        graph_path (str | Path): Default output path of the graph image
            (format taken from the extension: png, svg, pdf or dot).
    """

    strategy_name: ClassVar[str]

    registry: Final[dict[str, type['TripleExtractor']]] = {}

    @classmethod
    def _register(cls, extractor: type['TripleExtractor'], strategy_name: str):
        """Registers a triple extractor plugin class.

        Args:
            extractor (type[TripleExtractor]): Extractor class.
            strategy_name (str): Name to register under.
        """
        extractor.strategy_name = strategy_name
        cls.registry[extractor.strategy_name] = extractor

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

        Returns:
            list[Triple]: List of extracted triples.
        """
        assert text and text.strip(), 'No text to extract triples from'

        triples = self._extract(text, prompt)

        if self._graph if graph is None else graph:
            self.to_graphviz(triples, graph_path or self._graph_path)

        return triples

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

        lines = ['digraph G {', '  rankdir=LR;', '  node [shape=box, style=rounded];']
        lines += [
            f'  {q(t.subject)} -> {q(t.object)} [label={q(t.predicate)}];'
            for t in triples
        ]
        lines.append('}')
        dot = '\n'.join(lines)

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
        """Builds the KBEL mentions of each triple, ready for `KBEL.link`.

        The position in the triple defines the entity type: subject and
        object are items, predicate is a property. `text` is the original
        input text.

        Returns:
            list[tuple[Mention, Mention, Mention]]: (subject, predicate, object)
            mentions, one tuple per triple.
        """
        def m(term: str, entity_type: EntityType) -> Mention:
            return Mention(term=term, text=text, context=context,
                           entity_type=entity_type, language=language)

        return [
            (m(t.subject, EntityType.ITEM),
             m(t.predicate, EntityType.PROPERTY),
             m(t.object, EntityType.ITEM))
            for t in triples
        ]

    @abstractmethod
    def _extract(self, text: str, prompt=None) -> list[Triple]:
        """Core extraction logic to be implemented by subclasses.

        Args:
            text (str): Input text.

        Returns:
            list[Triple]: List of extracted triples.
        """
        ...