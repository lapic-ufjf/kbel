# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import abstractmethod
from typing import (Any, AsyncIterator, ClassVar, Final,
                    Tuple)

from kif_lib import Entity, KIF_Object, Property, Item

from kbel.core.mention import EntityType, Mention
from kbel.core.candidate import Candidate
from kbel.core.results import DisambiguationResult

LOG = logging.getLogger(__name__)

class Disambiguator:
    """Base class for entity disambiguators.

    Subclasses must implement the `_disambiguate` method, which contains the
    logic to select the correct entity among a list of candidates.

    Parameters:
        strategy_name (str): Name of the disambiguation strategy plugin.
    """

    #: Name of the disambiguation strategy plugin.
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
        candidates: list[Candidate],
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[DisambiguationResult]:
        """Synchronously disambiguates a list of candidates.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): Candidate entities.
            cls (Type[T]): Entity type (Item or Property).
            limit (int, optional): Maximum number of candidates to return. Defaults to 10.

        Returns:
            list[DisambiguationResult]: List of disambiguation results.
        """
        assert len(candidates) > 0, 'No candidates to disambiguate'
        
        return self._disambiguate(mention, candidates, limit, *args, **kwargs)


    async def adisambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[DisambiguationResult]:
        """Asynchronously disambiguates a list of candidates.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): Candidate entities.
            limit (int, optional): Maximum number of candidates to return. Defaults to 10.

        Yields:
            AsyncIterator[DisambiguationResult]: Disambiguation results.
        """
        results = self.disambiguate(mention, candidates, limit, *args, **kwargs)

        for result in results:
            yield result


    @abstractmethod
    def _disambiguate(
        self, 
        mention: Mention, 
        candidates: list[Candidate], 
        limit: int, 
        *args: Any,
        **kwargs: Any
    ) -> list[DisambiguationResult]:
        """Core disambiguation logic to be implemented by subclasses.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): List of candidate entities.
            limit (int): Maximum number of results to return.

        Returns:
            list[DisambiguationResult]: List of disambiguation results.
        """
        ...
