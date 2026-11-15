# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

from kbel.core.triple import Triple

from ..abc import TripleExtractor
from .worker_client import ReLiKWorker

LOG = logging.getLogger(__name__)

NME = '--NME--'


class ReLiKTripleExtractor(TripleExtractor, strategy_name='relik'):
    """Relation extraction with ReLiK (Retrieve, Read and LinK).

    ReLiK retrieves the relations that may hold in the text (Wikidata
    properties such as "place of birth") and a reader extracts the
    subject and object spans of each one. Subject and object are the
    spans as written in the text; KBEL links them afterwards.

    ReLiK pins torch 2.3, so it runs in its own Python 3.10/3.11
    interpreter (see `worker_client.resolve_python`), created on first use
    in ``~/.kbel/relik-venv`` unless `python` or ``KBEL_RELIK_PYTHON`` is
    given.

    Note:
        The ReLiK code and some of its checkpoints are distributed under
        CC BY-NC-SA 4.0 (non-commercial).

    Args:
        model_name: ReLiK checkpoint on the Hugging Face Hub.
        python: Interpreter with `relik` installed.
        device: 'cpu', 'cuda' or 'mps'.
        precision: '32', '16' or 'bf16'.
        top_k: Relations retrieved per window and given to the reader.
        window_size: Tokens per window, or 'sentence'.
        window_stride: Stride between windows.
        reader_batch_size: Windows per reader forward pass.
        num_workers: DataLoader processes of the retriever. ReLiK's default
            (4) starts new processes on every call, which costs ~15 s on
            macOS; 0 keeps everything in the worker process.
        min_confidence: Triplets with a lower confidence are dropped.
        batch_size: Texts sent to ReLiK per call (used by `construct`).
        model_options: Extra keyword arguments of `Relik.from_pretrained`.
        auto_install: Create the ReLiK environment if none is found.
        request_timeout: Seconds to wait for one call (None: no limit).
        worker: An already running `ReLiKWorker` (shared between extractors).
    """

    DEFAULT_MODEL = 'relik-ie/relik-relation-extraction-large'

    def __init__(
        self,
        strategy_name: str,
        model_name: str | None = None,
        python: str | Path | None = None,
        device: str = 'cpu',
        precision: str | None = None,
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | None = None,
        reader_batch_size: int | None = None,
        num_workers: int = 0,
        min_confidence: float = 0.0,
        batch_size: int = 8,
        model_options: dict[str, Any] | None = None,
        auto_install: bool = True,
        request_timeout: float | None = None,
        worker: ReLiKWorker | None = None,
        graph: bool = False,
        graph_path: str | Path = 'graph.png',
        **kwargs: Any,
    ):
        assert strategy_name == self.strategy_name
        super().__init__(graph=graph, graph_path=graph_path)
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.request_timeout = request_timeout
        self._options = {k: v for k, v in {
            'top_k': top_k, 'window_size': window_size, 'window_stride': window_stride,
            'reader_batch_size': reader_batch_size, 'num_workers': num_workers,
        }.items() if v is not None}
        self._worker = worker or ReLiKWorker(
            model_name or self.DEFAULT_MODEL, python=python, device=device,
            precision=precision, model_options=model_options, auto_install=auto_install)
        self._configure(**kwargs)

    def _configure(self) -> None:
        pass

    @property
    def worker(self) -> ReLiKWorker:
        return self._worker

    def annotate(self, texts: list[str]) -> list[dict[str, Any]]:
        """Raw ReLiK outputs (spans and triplets) of each text."""
        return self._worker.annotate(texts, timeout=self.request_timeout, **self._options)

    def _part(self, span: dict[str, Any]) -> str | None:
        return ' '.join(span['text'].split()) or None

    def _triples(self, output: dict[str, Any]) -> list[Triple]:
        triples: dict[tuple[str, str, str], Triple] = {}
        for t in output['triplets']:
            confidence = t.get('confidence')
            if confidence is not None and confidence < self.min_confidence:
                continue
            subject, obj = self._part(t['subject']), self._part(t['object'])
            predicate = ' '.join(str(t['label']).split())
            if subject and obj and predicate:
                triples.setdefault((subject, predicate, obj), Triple(subject, predicate, obj))
        return list(triples.values())

    def _extract(self, text: str, prompt: Any = None) -> list[Triple]:
        return self._extract_batch([text], prompt)[0]

    def _extract_batch(self, texts: list[str], prompt: Any = None) -> list[list[Triple]]:
        if prompt is not None:
            LOG.debug('`prompt` has no effect in the ReLiK extractors.')
        return [self._triples(output) for output in self.annotate(texts)]

    def close(self) -> None:
        self._worker.close()


class ReLiKClosedIETripleExtractor(ReLiKTripleExtractor, strategy_name='relik-cie'):
    """Closed information extraction with ReLiK: entity linking and relations.

    The same model links every entity span to a Wikipedia page and extracts
    the relations between them, so subject and object can be the titles of
    the linked pages ("NBA" -> "National Basketball Association"), which
    makes KBEL's search much less ambiguous.

    The entity index of the CIE checkpoints is large: about 8 GB for
    ``relik-ie/relik-cie-large`` and 12.5 GB for ``relik-ie/relik-cie-small``,
    downloaded on first use and loaded in memory.

    Args (besides those of `ReLiKTripleExtractor`):
        use_entity_titles: Use the linked Wikipedia title instead of the
            text span for subject and object.
        keep_unlinked: Keep triplets whose subject or object was not linked
            (their text span is used). If False, they are dropped.
    """

    DEFAULT_MODEL = 'relik-ie/relik-cie-large'

    def _configure(self, use_entity_titles: bool = True, keep_unlinked: bool = True) -> None:
        self.use_entity_titles = use_entity_titles
        self.keep_unlinked = keep_unlinked

    def _part(self, span: dict[str, Any]) -> str | None:
        label = span.get('label')
        linked = bool(label) and label != NME
        if not linked and not self.keep_unlinked:
            return None
        if linked and self.use_entity_titles:
            return ' '.join(label.split())
        return super()._part(span)
