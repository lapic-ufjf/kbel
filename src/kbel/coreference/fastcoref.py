# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from typing import Any

from .abc import Cluster, CoreferenceResolver

LOG = logging.getLogger(__name__)


def _patch_transformers_5(*classes: type) -> None:
    """fastcoref predates transformers 5, which expects `all_tied_weights_keys`."""
    for cls in classes:
        if not hasattr(cls, 'all_tied_weights_keys'):
            cls.all_tied_weights_keys = property(lambda self: {})


class FastCorefResolver(CoreferenceResolver, strategy_name='fastcoref'):
    """Coreference resolution with fastcoref (F-COREF, AACL 2022).

    F-COREF is a fast and accurate model (https://github.com/shon-otmazgin/fastcoref,
    "F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"). It
    finds the clusters of mentions that refer to the same entity, and KBEL
    substitutes the pronouns of each cluster by its name (see
    `CoreferenceResolver`).

    Args:
        model_name: ``'fcoref'`` (fast; default) or ``'lingmess'`` (more
            accurate, slower).
        model_path: Hugging Face model to load instead of the default one of
            `model_name`.
        device: ``'cpu'``, ``'cuda'`` or ``'mps'`` (None: automatic).
        max_tokens_in_batch: Tokens per forward pass.
        language: Language of the tokenizer used to split the text.
        nlp: A spaCy pipeline to tokenize with, instead of a blank one.
        model: An already loaded fastcoref model.
        replace, representative, max_words, batch_size: See
            `CoreferenceResolver`.
    """

    MODELS = {'fcoref': ('FCoref', 'biu-nlp/f-coref'),
              'lingmess': ('LingMessCoref', 'biu-nlp/lingmess-coref')}

    def __init__(
        self,
        strategy_name: str,
        model_name: str = 'fcoref',
        model_path: str | None = None,
        device: str | None = None,
        max_tokens_in_batch: int = 10000,
        language: str = 'en',
        nlp: Any = None,
        model: Any = None,
        **kwargs: Any,
    ):
        assert strategy_name == self.strategy_name
        super().__init__(**kwargs)
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model {model_name!r}; use {' or '.join(self.MODELS)}.")
        if max_tokens_in_batch < 1:
            raise ValueError('max_tokens_in_batch must be a positive integer.')
        self.max_tokens_in_batch = max_tokens_in_batch
        self._lock = threading.Lock()
        self._model = model if model is not None else self._load(
            model_name, model_path, device, language, nlp)

    @classmethod
    def _load(cls, model_name: str, model_path: str | None, device: str | None,
              language: str, nlp: Any) -> Any:
        try:
            import spacy
            from fastcoref import FCoref, LingMessCoref
            from fastcoref.coref_models.modeling_fcoref import FCorefModel
            from fastcoref.coref_models.modeling_lingmess import LingMessModel
        except ImportError as e:
            raise ImportError(
                "fastcoref is not installed. Install it with: pip install 'kbel[coreference]'"
            ) from e
        _patch_transformers_5(FCorefModel, LingMessModel)
        class_name, default_path = cls.MODELS[model_name]
        LOG.info('Loading %s (%s)', class_name, model_path or default_path)
        return {'FCoref': FCoref, 'LingMessCoref': LingMessCoref}[class_name](
            model_name_or_path=model_path or default_path, device=device,
            nlp=nlp if nlp is not None else spacy.blank(language),
            enable_progress_bar=False)

    @property
    def model(self) -> Any:
        return self._model

    def _clusters(self, texts: list[str]) -> list[list[Cluster]]:
        with self._lock:
            results = self._model.predict(
                texts=texts, max_tokens_in_batch=self.max_tokens_in_batch)
        return [[[(int(s), int(e)) for s, e in cluster]
                 for cluster in result.get_clusters(as_strings=False)]
                for result in results]
