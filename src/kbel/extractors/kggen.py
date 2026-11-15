# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kbel.core.triple import Triple

from .abc import TripleExtractor

LOG = logging.getLogger(__name__)


class KGGenTripleExtractor(TripleExtractor, strategy_name='kggen'):
    """Triple extraction with KGGen (https://github.com/stair-lab/kg-gen).

    KGGen prompts an LLM (through DSPy/LiteLLM) twice per text: first for
    its entities, then for subject-predicate-object relations whose subject
    and object must be among those entities. Predicates are free text.

    Models are LiteLLM strings, e.g. ``openrouter/openai/gpt-4o-mini``
    (key from ``OPENROUTER_API_KEY`` or `api_key`), ``openai/gpt-4o`` or
    ``ollama_chat/llama3.1``.

    Args:
        model: LiteLLM model string.
        api_key: API key; LiteLLM reads the provider's variable otherwise.
        api_base: Custom endpoint (OpenAI-compatible servers, Ollama, ...).
        temperature: Sampling temperature (gpt-5 models require 1.0).
        max_tokens: Maximum tokens per LLM answer.
        cache: Reuse DSPy's cached answers for repeated prompts. Off by
            default so that timings and evaluations reflect real calls.
        batch_size: Texts processed per call (used by `construct`); they
            are sent to the LLM concurrently.
        max_workers: Concurrent texts within a batch.
        kg: An existing `kg_gen.KGGen` (its model settings are used as is).
    """

    DEFAULT_MODEL = 'openrouter/openai/gpt-4o-mini'

    def __init__(
        self,
        strategy_name: str,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        cache: bool = False,
        batch_size: int = 8,
        max_workers: int = 8,
        kg: Any = None,
        graph: bool = False,
        graph_path: str | Path = 'graph.png',
        **kwargs: Any,
    ):
        assert strategy_name == self.strategy_name
        super().__init__(graph=graph, graph_path=graph_path)
        if batch_size < 1 or max_workers < 1:
            raise ValueError('batch_size and max_workers must be positive integers.')
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._kg = kg or self._build(model, api_key, api_base, temperature, max_tokens, cache)
        self._configure(**kwargs)

    @staticmethod
    def _build(model: str, api_key: str | None, api_base: str | None,
               temperature: float, max_tokens: int, cache: bool) -> Any:
        try:
            import dspy
            from kg_gen import KGGen
        except ImportError as e:
            raise ImportError(
                "KGGen is not installed. Install it with: pip install 'kbel[kggen]'") from e
        kg = KGGen(model=model, temperature=temperature, max_tokens=max_tokens,
                   api_key=api_key, api_base=api_base)
        kg.lm = dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens,
                        api_key=api_key, api_base=api_base, cache=cache)
        return kg

    def _configure(self) -> None:
        pass

    @property
    def kg(self) -> Any:
        return self._kg

    @staticmethod
    def _clean(value: Any) -> str:
        return ' '.join(str(value).split())

    def _triples(self, relations: Any, entity_of: dict[str, str] | None = None,
                 edge_of: dict[str, str] | None = None) -> list[Triple]:
        entity_of, edge_of = entity_of or {}, edge_of or {}
        triples: dict[tuple[str, str, str], Triple] = {}
        for s, p, o in sorted(relations):
            key = (self._clean(entity_of.get(s, s)), self._clean(edge_of.get(p, p)),
                   self._clean(entity_of.get(o, o)))
            if all(key) and key[0] != key[2]:
                triples.setdefault(key, Triple(*key))
        return list(triples.values())

    def _generate(self, text: str) -> Any:
        return self._kg.generate(input_data=text)

    def _generate_all(self, texts: list[str]) -> list[Any]:
        if len(texts) == 1:
            return [self._generate(texts[0])]
        with ThreadPoolExecutor(min(self.max_workers, len(texts))) as pool:
            return list(pool.map(self._generate, texts))

    def _extract(self, text: str, prompt: Any = None) -> list[Triple]:
        if prompt is not None:
            LOG.debug('`prompt` has no effect in the KGGen extractors.')
        return self._triples(self._generate(text).relations)


class KGGenClusterTripleExtractor(KGGenTripleExtractor, strategy_name='kggen-cluster'):
    """KGGen with clustering of equivalent entities across a batch of texts.

    The texts of a batch (the sentences `construct` sends together) are
    extracted independently, their graphs are merged, and KGGen clusters
    the entities that name the same thing ("Turing", "Alan Turing"); every
    triple then uses its cluster's representative, which helps the search.

    Clustering costs extra LLM calls and can over-merge, especially with
    small models (e.g. different cities in one cluster). Predicates are
    therefore only clustered with `cluster_edges=True`, since merging
    "was born in" with "died in" changes the meaning of a triple.

    Args (besides those of `KGGenTripleExtractor`):
        cluster_edges: Also replace predicates by their cluster's representative.
        context: Description of the data, given to the clustering prompts.
    """

    def _configure(self, cluster_edges: bool = False, context: str = '') -> None:
        self.cluster_edges = cluster_edges
        self.context = context

    @staticmethod
    def _representatives(clusters: dict[str, set[str]] | None) -> dict[str, str]:
        return {variant: rep for rep, variants in (clusters or {}).items()
                for variant in variants if variant != rep}

    def _extract(self, text: str, prompt: Any = None) -> list[Triple]:
        return self._extract_batch([text], prompt)[0]

    def _extract_batch(self, texts: list[str], prompt: Any = None) -> list[list[Triple]]:
        if prompt is not None:
            LOG.debug('`prompt` has no effect in the KGGen extractors.')
        graphs = self._generate_all(texts)
        merged = self._kg.aggregate(graphs)
        if not merged.relations:
            return [[] for _ in texts]
        clustered = self._kg.cluster(merged, context=self.context)
        entity_of = self._representatives(clustered.entity_clusters)
        edge_of = self._representatives(clustered.edge_clusters) if self.cluster_edges else {}
        return [self._triples(g.relations, entity_of, edge_of) for g in graphs]
