# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from kbel.core.mention import Mention
from kbel.core.results import DisambiguationResult

from .abc import Candidate, Disambiguator

try:
    from sentence_transformers import SentenceTransformer
except ImportError as err:
    raise ImportError(
        "The `sim` disambiguator requires sentence-transformers: "
        "pip install 'kbel[similarity]'") from err

LOG = logging.getLogger(__name__)


class _EmbeddingCache:
    """Normalized embeddings of the most recently used texts (LRU)."""

    def __init__(self, encode, size: int):
        self._encode = encode
        self._size = size
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def __call__(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._items))
        if missing:
            for text, emb in zip(missing, self._encode(missing)):
                self._items[text] = np.asarray(emb, dtype=np.float32)
        for text in texts:
            self._items.move_to_end(text)
        result = np.stack([self._items[t] for t in texts])
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return result


class SimilarityDisambiguator(Disambiguator, strategy_name='sim'):
    """Ranks candidates by embedding similarity, using the mention's context.

    Each candidate is embedded as ``"label: description"`` and scored by

        sentence_weight    * cos(sentence, candidate)
      + context_weight     * cos(context, candidate)
      + term_weight        * cos(term, label)
      + rank_weight        * 1 / (1 + position in the search results)
      + exact_match_weight * (label == term)

    where `sentence` is the mention's `text` and `context` its paragraph
    (`construct` fills both). Queries and candidates are encoded with the
    model's query/document prompts when it defines them. Embeddings are
    cached, so repeated candidates and paragraphs are encoded once.

    The defaults were chosen on 447 T-REx entity mentions with Wikidata
    candidates: bge-small reaches 0.82 accuracy, versus 0.73 for the first
    search result and 0.50 for the previous term+sentence similarity.
    Lighter alternatives: ``Snowflake/snowflake-arctic-embed-xs`` (0.80,
    about half the time) and ``sentence-transformers/static-retrieval-mrl-en-v1``
    (0.77 without context, ~50x faster on CPU; use ``context_weight=0``).

    Args:
        model_name: Sentence-transformers model.
        sentence_weight, context_weight, term_weight, rank_weight,
            exact_match_weight: Weights of the score terms above.
        cache_size: Number of embeddings kept in memory per kind.
        device: Device of the model ('cpu', 'cuda', 'mps'; None: automatic).
        model: An already loaded `SentenceTransformer`.
    """

    DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'

    def __init__(
        self,
        strategy_name: str,
        model_name: str = DEFAULT_MODEL,
        sentence_weight: float = 1.0,
        context_weight: float = 0.75,
        term_weight: float = 0.0,
        rank_weight: float = 0.2,
        exact_match_weight: float = 0.1,
        cache_size: int = 50_000,
        device: str | None = None,
        model: Any = None,
    ):
        assert strategy_name == self.strategy_name
        super().__init__()
        if model is None:
            LOG.info(f'Loading embedding model: {model_name}')
            model = SentenceTransformer(model_name, device=device)
        self._model = model
        self.weights = {'sentence': sentence_weight, 'context': context_weight,
                        'term': term_weight, 'rank': rank_weight, 'exact': exact_match_weight}
        self._lock = threading.Lock()
        encode_query = getattr(model, 'encode_query', model.encode)
        encode_document = getattr(model, 'encode_document', model.encode)
        self._queries = _EmbeddingCache(
            lambda texts: encode_query(texts, normalize_embeddings=True), cache_size)
        self._documents = _EmbeddingCache(
            lambda texts: encode_document(texts, normalize_embeddings=True), cache_size)

    @property
    def model(self) -> Any:
        return self._model

    @staticmethod
    def candidate_text(candidate: Candidate) -> str:
        label = (candidate.label or '').strip()
        description = (candidate.description or '').strip()
        return f'{label}: {description}' if description else label

    def scores(self, mention: Mention, candidates: list[Candidate]) -> np.ndarray:
        """The score of every candidate (see the class docstring)."""
        w = self.weights
        sentence = mention.text or mention.term
        context = mention.context if mention.context and mention.context != mention.text else None
        queries = [sentence] + ([context] if context and w['context'] else []) + (
            [mention.term] if w['term'] else [])
        documents = [self.candidate_text(c) for c in candidates]
        labels = [(c.label or '').strip() or self.candidate_text(c) for c in candidates]
        with self._lock:
            q = self._queries(queries)
            d = self._documents(documents)
            lab = self._documents(labels) if w['term'] else None

        score = w['sentence'] * (d @ q[0])
        if context and w['context']:
            score = score + w['context'] * (d @ q[1])
        if w['term']:
            score = score + w['term'] * (lab @ q[-1])
        term = mention.term.strip().lower()
        score = score + w['rank'] / (1 + np.arange(len(candidates)))
        score = score + w['exact'] * np.array(
            [(c.label or '').strip().lower() == term for c in candidates], dtype=np.float32)
        return score

    def _disambiguate(self, mention: Mention, candidates: list[Candidate], limit: int,
                      *args: Any, **kwargs: Any) -> list[DisambiguationResult]:
        assert candidates, f'No candidates to disambiguate mention `{mention.term}`'
        score = self.scores(mention, candidates)
        order = np.argsort(-score, kind='stable')[:limit]
        return [DisambiguationResult(candidate=candidates[i], score=float(score[i]))
                for i in order]
