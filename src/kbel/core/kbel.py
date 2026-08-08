import logging

from kif_lib import Entity
from kif_lib.typing import Any, Iterator

from kif_lib.search import Search
from kbel.core.mention import Mention, EntityType
from kbel.core.candidate import Candidate
from kbel.core.results import LinkResult
from kbel.disambiguators import Disambiguator

LOG = logging.getLogger(__name__)

class KBEL:

    _knowledge_base: Search
    _search_limit = 10
    _disambiguator: Disambiguator | None

    @property
    def search_limit(self) -> int:
        return self._search_limit

    @search_limit.setter
    def search_limit(self, value: int):
        if value <= 0:
            raise ValueError(
                'The limit to lookup candidates must be bigger than zero'
            )
        self._search_limit = value

    @property
    def knowledge_base(self) -> Search:
        return self._knowledge_base
    

    @knowledge_base.setter
    def knowledge_base(self, value: str | Search) -> None:
        if isinstance(value, str):
            value = Search(value, limit=self._search_limit)

        self._knowledge_base = value

    @property
    def disambiguator(self) -> Disambiguator | None:
        return self._disambiguator

    @disambiguator.setter
    def disambiguator(self, value: Disambiguator) -> None:
        self._disambiguator = value


    def __init__(self, 
                 knowledge_base: str, disambiguator: Disambiguator | None = None, search_limit=10, **kwargs):
        self._search_limit = search_limit
        self._knowledge_base = Search(
            knowledge_base,
            limit=self._search_limit,
            **kwargs
        )
        self._disambiguator = disambiguator

    def candidates_lookup(self, mention: Mention, limit: int = None) -> list[Candidate]:
        """Fetches candidate entities from the knowledge base for a given mention.
        
        Args:
            mention (Mention): The mention to look up candidates for.
            limit (int, optional): Maximum number of candidates to retrieve.
                Defaults to the instance's search limit.
        """

        if limit is None or limit <= 0:
            limit = self._search_limit

        @staticmethod
        def _safe_next(it: Iterator) -> Iterator:
                """Safely iterate over an iterator, skipping errors."""
                while True:
                    try:
                        yield next(it)
                    except StopIteration:
                        break
                    except Exception as e:
                        LOG.info(f'Error fetching item: {e}')
                        continue

        @staticmethod
        def _extract_text(data: dict[str, Any], key: str) -> str:
            """Extract the text for a given key and language."""
            l = mention.language if mention.language else 'en'
            value = data.get(key, {}).get(l)
            return value.content if value else ''
        
       
        if mention.entity_type is EntityType.ITEM:
            found_candidates =  self._knowledge_base.item_descriptor(search=mention.term, limit=limit)
        elif mention.entity_type is EntityType.PROPERTY:
            found_candidates =  self._knowledge_base.property_descriptor(search=mention.term, limit=limit)
        else:
            from itertools import chain
            found_candidates = chain(
                self._knowledge_base.item_descriptor(search=mention.term, limit=limit),
                self._knowledge_base.property_descriptor(search=mention.term, limit=limit),
            )

        candidates = []
        for entity, desc in _safe_next(iter(found_candidates)):
            candidate = Candidate(
                id = entity.iri.content,
                label = _extract_text(desc, 'labels'),
                description = _extract_text(desc, 'descriptions'),
                iri = entity.iri.content
            )
            candidates.append(candidate)

        return candidates

    def link(
        self,
        mention: Mention,
        search_limit: int = None,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[LinkResult]:

        """Links a mention to the most relevant entities in the knowledge base.

        Args:
            mention (Mention): The mention to link.
            search_limit (int, optional): Maximum number of candidates to retrieve from the knowledge base.
                Defaults to the instance's search limit.
            limit (int, optional): Maximum number of top candidates to return after disambiguation.
                Defaults to 10.
        """
        if self.disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")

        if search_limit is None or search_limit <= 0:
            search_limit = self._search_limit

        candidates = self.candidates_lookup(mention, limit=search_limit)

        return [
            LinkResult.from_disambiguation(mention, result)
            for result in self._disambiguator.disambiguate(mention,
                        candidates,
                        limit,
                        *args,
                        **kwargs,)
        ]
