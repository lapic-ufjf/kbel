# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kbel.core.mention import Mention
from kbel.core.results import DisambiguationResult

from .abc import Candidate, Disambiguator

LOG = logging.getLogger(__name__)


class NaiveDisambiguator(Disambiguator, strategy_name='naive'):
    """Naive disambiguator always returns the first candidate.

    This is the most basic implementation of a disambiguator plugin. It
    ignores the `limit` parameter and selects only the top candidate
    from the provided list.

    Example:
        >>> disamb = NaiveDisambiguator('naive')
        >>> candidates = [
        ...     {"label": "Python", "description": "Programming language", "iri": "https://www.wikidata.org/wiki/Q28865"}
        ... ]
        >>> disamb._disambiguate("Python", candidates, limit=1)
        [DisambiguationResult(iri='https://www.wikidata.org/wiki/Q28865', score=1.0)]
    """

    def _disambiguate(
        self,
        mention: Mention,
        candidates: list[Candidate],
        limit: int,
        *args,
        **kwargs) -> list[DisambiguationResult]:
        """Returns the first candidate from the list.

        Args:
            mention (Mention): The mention to disambiguate.
            candidates (list[Candidate]): List of candidate entities.
            limit (int): Maximum number of results requested (ignored here).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list[DisambiguationResult]: A list containing a single disambiguation result.
        """

        if limit > 1:
            LOG.debug('Limit has no effect here. This method always returns the top 1 candidate.')
        for candidate in candidates:
           if candidate.label.lower() == mention.term.lower():
                return [DisambiguationResult(iri=candidate.iri)]
           
        return [DisambiguationResult(iri=candidates[0].iri)]
