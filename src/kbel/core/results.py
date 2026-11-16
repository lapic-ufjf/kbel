# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from kbel.core.mention import Mention, EntityType
from kbel.core.candidate import Candidate
from dataclasses import dataclass

from kif_lib import Item, Property, Value


@dataclass
class DisambiguationResult:
    candidate: Candidate
    score: float | None = None


@dataclass(frozen=True)
class LinkResult:
    """What a mention was linked to.

    `entity` is a KIF value: an `Item` or `Property` linked from the
    knowledge base (then `candidate` is the search result it came from), or,
    for the object of a property whose values are not entities, the literal
    parsed from the text (`Time`, `Quantity`, `String`, ...; then
    `candidate` is None).
    """

    entity: Value
    candidate: Candidate | None
    score: float | None = None

    @property
    def is_literal(self) -> bool:
        return self.candidate is None

    @classmethod
    def from_disambiguation(
        cls,
        mention: Mention,
        result: DisambiguationResult,
    ) -> "LinkResult":
        """Create a LinkResult from a DisambiguationResult."""

        candidate = result.candidate
        if mention.entity_type == EntityType.PROPERTY:
            entity = Property(candidate.iri, candidate.datatype)
        else:
            entity = Item(candidate.iri)
        return cls(score=result.score, candidate=candidate, entity=entity)

    @classmethod
    def from_literal(cls, value: Value) -> "LinkResult":
        """A link to a literal value written in the text."""
        return cls(entity=value, candidate=None)
