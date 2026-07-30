# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import abstractmethod
from typing import (Any, AsyncIterator, ClassVar, Final, Iterator,
                    Tuple, TypeVar)

from kif_lib import Entity, KIF_Object, Property, Item

from kbel.knowledge_sources.knowledge_source import KnowledgeSource
from kbel.core.mention import EntityType, Mention
from kbel.core.candidate import Candidate

LOG = logging.getLogger(__name__)

T = TypeVar("T", bound=Entity)


class Disambiguator:
    """Base class for entity disambiguators.

    Subclasses must implement the `_disambiguate` method, which contains the
    logic to select the correct entity among a list of candidates.

    Parameters:
        strategy_name (str): Name of the disambiguation plugin.
    """

    #: Name of the disambiguation plugin.
    strategy_name: ClassVar[str]

    #: Registry of all available disambiguator plugins.
    registry: Final[dict[str, type['Disambiguator']]] = {}

    @classmethod
    def _register(
        cls,
        disambiguator: type['Disambiguator'],
        strategy_name: str,
    ):
        """Registers a disambiguator plugin class.

        Args:
            disambiguator (type[Disambiguator]): Disambiguator class.
            strategy_name (str): Name to register under.
        """
        disambiguator.strategy_name = strategy_name
        cls.registry[disambiguator.strategy_name] = disambiguator

    @classmethod
    def __init_subclass__(cls, strategy_name: str):
        Disambiguator._register(cls, strategy_name)

    def __new__(cls, strategy_name: str, *args: Any, **kwargs: Any):
        KIF_Object._check_arg(
            strategy_name,
            strategy_name in cls.registry,
            f'no such disambiguator plugin "{strategy_name}"',
            Disambiguator,
            'strategy_name',
            1,
            ValueError,
        )
        return super().__new__(
            cls.registry[strategy_name])  # pyright: ignore


    def disambiguate(
        self,
        mention: Mention,
        ks: KnowledgeSource,
        limit = 10,
        *args: Any,
        **kwargs: Any
    ) -> list[Tuple[str, str, Entity]]:
        """Core method to disambiguate a label among candidates from the knowledge base.

        Args:
            mention (Mention): The mention to disambiguate.
            ks (KnowledgeSource): Knowledge source object to retrieve candidates.
            limit (int, optional): Maximum number of candidates to consider. Defaults to 10.
            language (str, optional): Language code for labels/descriptions. Defaults to 'en'.

        Returns:
            list[Tuple[str, str, T]]: List of tuples with label, description, and entity.
        """

        def safe_next(it: Iterator) -> Iterator:
            """Safely iterate over an iterator, skipping errors."""
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    break
                except Exception as e:
                    logging.info(f'Error fetching item: {e}')
                    continue

        def extract_text(data: dict[str, Any], key: str) -> str:
            """Extract the text for a given key and language."""
            l = mention.language if mention.language else 'en'
            value = data.get(key, {}).get(l)
            return value.content if value else ''

        try:
            if mention.entity_type is EntityType.ITEM:
                found_candidates = ks.item_descriptor(search=mention.label)
            elif mention.entity_type is EntityType.PROPERTY:
                found_candidates = ks.property_descriptor(search=mention.label)
            else:
                return []
        except Exception as e:
            raise e

        if not found_candidates:
            return []

        candidates = []
        for entity, desc in safe_next(iter(found_candidates)):
            candidate = Candidate(
                id = entity.iri.content,
                label = extract_text(desc, 'labels'),
                description = extract_text(desc, 'descriptions'),
                iri = entity.iri.content
            )
            candidates.append(candidate)

        return self.disambiguate_candidates(mention, candidates, limit, *args, **kwargs)

    def disambiguate_candidates(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[Tuple[str, str, Entity]]:
        """Synchronously disambiguates a list of candidates.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): Candidate entities.
            cls (Type[T]): Entity type (Item or Property).
            limit (int, optional): Maximum number of candidates to return. Defaults to 10.

        Returns:
            list[Tuple[str, str, T]]: List of tuples with label, description, and entity instance.
        """
        assert len(candidates) > 0, 'No candidates to disambiguate'
        results = self._disambiguate(mention, candidates, limit, *args, **kwargs)
        disamb_entities = []
        if results:
            for result in results:
                _label, description, entity = result
                if mention.entity_type == EntityType.ITEM:
                    entity = Item(iri=entity)
                elif mention.entity_type == EntityType.PROPERTY:
                    entity = Property(iri=entity)
                disamb_entities.append((_label, description, entity)) # type: ignore
        return disamb_entities

    async def adisambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[str, str, T]]:
        """Asynchronously disambiguates a list of candidates.

        Args:
            label (str): Label to disambiguate.
            candidates (list[Candidate]): Candidate entities.
            limit (int, optional): Maximum number of candidates to return. Defaults to 10.

        Yields:
            AsyncIterator[Tuple[str, str, T]]: Tuples with label, description, and entity instance.
        """
        results = self._disambiguate(mention, candidates, limit, *args, **kwargs)
        for label, description, entity in results:
            yield (mention.label, description, cls(iri=entity)) # type: ignore


    @abstractmethod
    def _disambiguate(self, 
        mention: Mention, 
        candidates: list[Candidate], limit: int, *args,
        **kwargs) -> list[Tuple[str, str, str]]:
        """Core disambiguation logic to be implemented by subclasses.

        Args:
            label (str): Label to disambiguate.
            candidates (list[Candidate]): List of candidate entities.
            limit (int): Maximum number of results to return.

        Returns:
            list[Tuple[str, str, str]]: Tuples of label, description, and entity identifier.
        """
        ...
