import logging

from kif_lib.typing import Any, Iterator

from kbel.knowledge_sources import KnowledgeSource
from kbel.core.mention import Mention, EntityType
from kbel.core.candidate import Candidate
from kbel.disambiguators import Disambiguator

LOG = logging.getLogger(__name__)

class KBEL:

    _ks: KnowledgeSource
    _disambiguator: Disambiguator | None

    @property
    def ks(self) -> KnowledgeSource:
        return self._ks

    @ks.setter
    def ks(self, value: KnowledgeSource) -> None:
        self._ks = value

    @property
    def disambiguator(self) -> Disambiguator | None:
        return self._disambiguator

    @disambiguator.setter
    def disambiguator(self, value: Disambiguator) -> None:
        self._disambiguator = value

    def __init__(self, ks: KnowledgeSource):
        self._ks = ks
        self._disambiguator = None

    def candidates_lookup(self, mention: Mention) -> list[Candidate]:
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
        
       
        if mention.entity_type is EntityType.ITEM:
            found_candidates =  self._ks.item_descriptor(search=mention.label)
        elif mention.entity_type is EntityType.PROPERTY:
            found_candidates =  self._ks.property_descriptor(search=mention.label)
        else:
            from itertools import chain
            found_candidates = chain(
                self._ks.item_descriptor(search=mention.label),
                self._ks.property_descriptor(search=mention.label),
            )

        candidates = []
        for entity, desc in safe_next(iter(found_candidates)):
            candidate = Candidate(
                id = entity.iri.content,
                label = extract_text(desc, 'labels'),
                description = extract_text(desc, 'descriptions'),
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
    ):
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
