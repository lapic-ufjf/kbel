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
# Extra annotators for coreference resolution (pronoun -> antecedent).
COREF_PROPERTIES: dict[str, Any] = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,coref,natlog,openie',
    'coref.algorithm': 'statistical',
}

CITATION_RE = re.compile(
    r'\[\s*(?:\d+(?:\s*[,–-]\s*\d+)*|[a-z][a-z ]*needed)\s*\]')

# An innermost parenthetical, e.g. " (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954)".
PARENTHETICAL_RE = re.compile(r'\s*\([^()]*\)')

POSSESSIVE_PRONOUNS = frozenset({'his', 'her', 'its', 'their'})

# Words whose removal would invert a relation ("is not" -> "is").
NEGATIONS = frozenset({'not', "n't", 'no', 'never', 'cannot', 'without', 'nor'})


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
        remove_parentheticals: bool = True,
        resolve_coref: bool = True,
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
            remove_parentheticals: Strip parenthesized asides such as
                "(/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954)" before parsing;
                they often make OpenIE miss the whole sentence.
            resolve_coref: Replace pronouns by the entity they refer to
                ("He" -> "Alan Turing", "his work" -> "Alan Turing's work"),
                using CoreNLP coreference. Only works within the text given
                to one `extract` call. Adds the `ner` and `coref`
                annotators: slower first call, more memory.
            remove_subsumed: OpenIE emits many variants of the same fact;
                keep one triple per fact:
                - same subject and object, predicates differing only by
                  extra words ("contributed to" / "also contributed
                  Turing test to"): keep the shortest predicate, which
                  links best (the extra words usually appear in a sibling
                  triple, e.g. "contributed | Turing test");
                - same subject and predicate, one object a shortened form
                  of another ("use" / "use of significant indentation"):
                  keep the longest object.
                A variant is never dropped if its extra words include a
                negation ("is" vs "is not").
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
        self._properties = {
            **DEFAULT_PROPERTIES,
            **(COREF_PROPERTIES if resolve_coref else {}),
            **(properties or {}),
        }
        self._remove_citations = remove_citations
        self._remove_parentheticals = remove_parentheticals
        self._resolve_coref = resolve_coref
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
        if self._remove_parentheticals:
            previous = None
            while previous != text:             # innermost first (nesting)
                previous, text = text, PARENTHETICAL_RE.sub('', text)
        return text

    def _parse(self, data: dict) -> list[Triple]:
        """Collect the (valid, unique) OpenIE triples of CoreNLP's output."""
        antecedents = self._antecedents(data) if self._resolve_coref else {}
        triples: dict[tuple[str, str, str], Triple] = {}
        for i, sentence in enumerate(data.get('sentences', []), start=1):
            for t in sentence.get('openie', []):
                s = self._resolve(t.get('subject', ''), t.get('subjectSpan'), i, antecedents)
                p = ' '.join(t.get('relation', '').split())
                o = self._resolve(t.get('object', ''), t.get('objectSpan'), i, antecedents)
                if not (s and p and o) or len(o) < self._min_object_chars:
                    continue
                triples.setdefault((s, p, o), Triple(subject=s, predicate=p, object=o))
        return list(triples.values())

    @staticmethod
    def _antecedents(data: dict) -> dict[tuple[int, int, int], tuple[str, str]]:
        """Pronoun mentions -> (pronoun, text of the entity it refers to).

        Keys are `(sentence, start, end)` with CoreNLP's 1-based sentence
        number and 0-based token span (OpenIE's convention). Only pronouns
        whose chain has a non-pronominal representative mention are kept.
        """
        antecedents: dict[tuple[int, int, int], tuple[str, str]] = {}
        for chain in data.get('corefs', {}).values():
            representative = next(
                (m for m in chain if m.get('isRepresentativeMention')), None)
            if representative is None or representative.get('type') == 'PRONOMINAL':
                continue
            name = ' '.join(representative['text'].split())
            for m in chain:
                if m.get('type') == 'PRONOMINAL':
                    key = (m['sentNum'], m['startIndex'] - 1, m['endIndex'] - 1)
                    antecedents[key] = (m['text'].lower(), name)
        return antecedents

    @staticmethod
    def _resolve(
            text: str, span: Optional[list[int]], sentence: int,
            antecedents: dict[tuple[int, int, int], tuple[str, str]]) -> str:
        """Replace a pronoun (or a leading possessive) in a triple part.

        The span locates the pronoun; the text must also be that pronoun,
        so a part is never replaced on position alone.
        """
        text = ' '.join(text.split())
        if not antecedents or not span:
            return text
        start, end = span
        pronoun, name = antecedents.get((sentence, start, end), ('', ''))
        if name and text.lower() == pronoun:                     # "He"
            return name
        first, _, rest = text.partition(' ')
        pronoun, name = antecedents.get((sentence, start, start + 1), ('', ''))
        if (name and rest and first.lower() == pronoun
                and pronoun in POSSESSIVE_PRONOUNS):                # "his work"
            return f"{name}'s {rest}"
        return text

    @staticmethod
    def _drop_subsumed(triples: list[Triple]) -> list[Triple]:
        """Merge OpenIE variants of the same fact (see `remove_subsumed`)."""
        def is_subsequence(short: list[str], long: list[str]) -> bool:
            it = iter(long)
            return all(word in it for word in short)

        def extends(longer: str, shorter: str) -> bool:
            """`longer` = `shorter` plus extra words (none of them negations)."""
            long, short = longer.split(), shorter.split()
            if len(long) <= len(short) or not is_subsequence(short, long):
                return False
            extra = [w.lower() for w in long]
            for w in short:
                extra.remove(w.lower())
            return not NEGATIONS.intersection(extra)

        # 1. Same subject and object: keep the shortest predicate.
        triples = [t for t in triples if not any(
            (u.subject, u.object) == (t.subject, t.object)
            and extends(t.predicate, u.predicate)
            for u in triples)]

        # 2. Same subject and predicate: keep the longest object.
        return [t for t in triples if not any(
            (u.subject, u.predicate) == (t.subject, t.predicate)
            and extends(u.object, t.object)
            for u in triples)]
