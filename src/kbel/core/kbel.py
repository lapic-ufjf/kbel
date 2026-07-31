import logging

from kif_lib import Entity
from kif_lib.typing import Any, Iterator

from kbel.knowledge_sources import KnowledgeSource
from kbel.core.mention import Mention, EntityType
from kbel.core.candidate import Candidate
from kbel.disambiguators import Disambiguator

LOG = logging.getLogger(__name__)

class KBEL:

    _knowledge_source: KnowledgeSource
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
    def knowledge_source(self) -> KnowledgeSource:
        return self._knowledge_source
    

    @knowledge_source.setter
    def knowledge_source(self, value: str | KnowledgeSource) -> None:
        if isinstance(value, str):
            value = KnowledgeSource(value, limit= self._search_limit)

        self._knowledge_source = value

    @property
    def disambiguator(self) -> Disambiguator | None:
        return self._disambiguator

    @disambiguator.setter
    def disambiguator(self, value: Disambiguator) -> None:
        self._disambiguator = value


    def __init__(self, knowledge_source: str, search_limit = 10, **kwargs):
        self._search_limit = search_limit
        self._knowledge_source = KnowledgeSource(
            knowledge_source,
            limit=self._search_limit,
            **kwargs
        )
        self._disambiguator = None

    def candidates_lookup(self, mention: Mention) -> list[Candidate]:
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
            found_candidates =  self._knowledge_source.item_descriptor(search=mention.label)
        elif mention.entity_type is EntityType.PROPERTY:
            found_candidates =  self._knowledge_source.property_descriptor(search=mention.label)
        else:
            from itertools import chain
            found_candidates = chain(
                self._knowledge_source.item_descriptor(search=mention.label),
                self._knowledge_source.property_descriptor(search=mention.label),
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

    def disambiguate(
        self,
        mention: Mention,
        limit: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[str, str, Entity]]:
        if self.disambiguator is None:
            raise RuntimeError("No disambiguator has been configured.")

        candidates = self.candidates_lookup(mention)

        return self._disambiguator.disambiguate(
            mention,
            candidates,
            limit,
            *args,
            **kwargs,
        )
