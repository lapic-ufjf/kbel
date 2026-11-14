# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from pathlib import Path
from typing import Any, Optional

from kbel.core.triple import Triple

from ..abc import TripleExtractor
from .corenlp import DEFAULT_RESOURCES, CoreNLPServer

LOG = logging.getLogger(__name__)

DEFAULT_PROPERTIES: dict[str, Any] = {
    'annotators': 'tokenize,ssplit,pos,lemma,parse,natlog,openie',
    'tokenize.options': 'splitHyphenated=false',
    'openie.affinity_probability_cap': 2 / 3,
}
CITATION_RE = re.compile(
    r'\[\s*(?:\d+(?:\s*[,–-]\s*\d+)*|[a-z][a-z ]*needed)\s*\]')


class OpenIETripleExtractor(TripleExtractor, strategy_name='openie'):
    """Triple extractor based on Stanford OpenIE (CoreNLP).

    Rule-based and deterministic; English only; requires Java. Relations
    are surface phrases from the text (e.g. "was born in"), not knowledge
    base properties.

    The CoreNLP server is handled by `CoreNLPServer`: a running server is
    reused, otherwise CoreNLP is located (or downloaded) and started, and
    stopped again by `close()` / at interpreter exit.

    Example:
        >>> extractor = TripleExtractor('openie')
        >>> extractor.extract('Barack Obama was born in Hawaii.')
        [Triple(subject='Barack Obama', predicate='was born in', object='Hawaii'), ...]
    """

    def __init__(
        self,
        strategy_name: str,
        server: Optional[CoreNLPServer] = None,
        host: str = 'localhost',
        port: int = 9000,
        corenlp_home: Optional[str | Path] = None,
        auto_download: bool = True,
        memory: str = '4G',
        properties: Optional[dict[str, Any]] = None,
        remove_citations: bool = True,
        remove_subsumed: bool = True,
        min_object_chars: int = 2,
        graph: bool = False,
        graph_path: str | Path = 'graph.png',
    ):
        """Initializes the OpenIE extractor (and makes sure CoreNLP is up).

        Args:
            strategy_name: Name of this strategy (`'openie'`).
            server: A `CoreNLPServer` to use. If omitted, one is built from
                `host`, `port`, `corenlp_home`, `auto_download`, `memory`.
            host: CoreNLP server host.
            port: CoreNLP server port.
            corenlp_home: CoreNLP folder or its `.zip`. Defaults to
                `CORENLP_HOME`, then to `~/.stanfordnlp_resources`.
            auto_download: Download CoreNLP if it is not found.
            memory: Max JVM memory if a server must be started.
            properties: CoreNLP properties, merged over the defaults.
            remove_citations: Strip citation markers such as `[38]` before
                parsing (they break the parse of the sentence).
            remove_subsumed: OpenIE also emits shortened versions of each
                triple ("use" next to "use of significant indentation");
                drop a triple when another with the same subject and
                predicate has an object containing all its words in order.
            min_object_chars: Drop triples whose object is shorter.
            graph: Render a graph image on every `extract` call.
            graph_path: Where to save the graph image.
        """
        assert strategy_name == self.strategy_name
        super().__init__(graph=graph, graph_path=graph_path)

        if min_object_chars < 0:
            raise ValueError('min_object_chars must be >= 0.')

        self._server = server or CoreNLPServer(
            host=host, port=port, home=corenlp_home,
            resources=DEFAULT_RESOURCES, auto_download=auto_download,
            memory=memory)
        self._properties = {**DEFAULT_PROPERTIES, **(properties or {})}
        self._remove_citations = remove_citations
        self._remove_subsumed = remove_subsumed
        self._min_object_chars = min_object_chars

        self._server.start()

    @property
    def server(self) -> CoreNLPServer:
        return self._server

    def close(self) -> None:
        """Stop the CoreNLP server if this extractor started it."""
        self._server.close()

    def __enter__(self) -> 'OpenIETripleExtractor':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


    def _extract(self, text: str, prompt: Any = None) -> list[Triple]:
        """Extracts triples from a text using Stanford OpenIE.

        Args:
            text: Input text.
            prompt: Ignored (kept for interface compatibility).

        Returns:
            Deduplicated triples, in the order CoreNLP produced them.
        """
        if prompt is not None:
            LOG.debug('`prompt` has no effect in the openie extractor.')

        data = self._server.annotate(self._preprocess(text), self._properties)
        triples = self._parse(data)
        if self._remove_subsumed:
            triples = self._drop_subsumed(triples)
        return triples

    def _preprocess(self, text: str) -> str:
        if self._remove_citations:
            text = CITATION_RE.sub('', text)
        return text

    def _parse(self, data: dict) -> list[Triple]:
        """Collect the (valid, unique) OpenIE triples of CoreNLP's output."""
        triples: dict[tuple[str, str, str], Triple] = {}
        for sentence in data.get('sentences', []):
            for t in sentence.get('openie', []):
                s = ' '.join(t.get('subject', '').split())
                p = ' '.join(t.get('relation', '').split())
                o = ' '.join(t.get('object', '').split())
                if not (s and p and o) or len(o) < self._min_object_chars:
                    continue
                triples.setdefault((s, p, o), Triple(subject=s, predicate=p, object=o))
        return list(triples.values())

    @staticmethod
    def _drop_subsumed(triples: list[Triple]) -> list[Triple]:
        """Drop triples whose object is a shortened form of a sibling's."""
        def is_subsequence(short: list[str], long: list[str]) -> bool:
            it = iter(long)
            return all(word in it for word in short)

        def subsumed(t: Triple) -> bool:
            words = t.object.split()
            return any(
                u is not t
                and (u.subject, u.predicate) == (t.subject, t.predicate)
                and len(u.object.split()) > len(words)
                and is_subsequence(words, u.object.split())
                for u in triples)

        return [t for t in triples if not subsumed(t)]
