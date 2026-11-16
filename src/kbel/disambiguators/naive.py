# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

from kbel.core.mention import Mention
from kbel.core.results import DisambiguationResult

from .abc import Candidate, Disambiguator

LOG = logging.getLogger(__name__)


class NaiveDisambiguator(Disambiguator, strategy_name='naive'):
    """Picks the candidate whose label is the mention's term, else the first.

    The simplest strategy: it trusts the order of the search results and
    only prefers a candidate with exactly the same label (ignoring case).
    It always returns a single result, whatever the `limit`.

    Example:
        >>> disambiguator = Disambiguator('naive')
        >>> candidates = [
        ...     Candidate(iri='http://www.wikidata.org/entity/Q271218', label='Python (genus)'),
        ...     Candidate(iri='http://www.wikidata.org/entity/Q28865', label='Python'),
        ... ]
        >>> disambiguator.disambiguate(Mention(term='Python', text='Python is a language.'), candidates)
        [DisambiguationResult(candidate=Candidate(iri='http://www.wikidata.org/entity/Q28865', ...), score=None)]
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
        term = mention.term.strip().lower()
        for candidate in candidates:
            if (candidate.label or '').strip().lower() == term:
                return [DisambiguationResult(candidate=candidate)]

        return [DisambiguationResult(candidate=candidates[0])]
