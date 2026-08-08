# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from kbel.core.mention import Mention, EntityType
from dataclasses import dataclass

from kif_lib import Entity, Property, Item


@dataclass
class DisambiguationResult:
    iri: str
    score: float | None = None


@dataclass(frozen=True)
class LinkResult:
    entity: Entity
    score: float | None = None

    @classmethod
    def from_disambiguation(
        cls,
        mention: Mention,
        result: DisambiguationResult,
    ) -> "LinkResult":
        """Create a LinkResult from a DisambiguationResult."""

        if mention.entity_type == EntityType.PROPERTY:
            entity = Property(iri=result.iri)
        else:
            entity = Item(iri=result.iri)
        return cls(
            score=result.score,
            entity=entity
        )