# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    """Supported entity types."""

    ITEM = "item"
    PROPERTY = "property"


@dataclass(slots=True, frozen=True)
class Mention:
    """Represents a textual mention to be linked to a Knowledge Base entity.

    Parameters
    ----------

    term:
        The term of the mention.
    text:
        The surface form of the mention.

    context:
        Optional textual context surrounding the mention.

    entity_type:
        Whether the mention refers to an item or a property.

    language:
        Optional language code (e.g., "en", "pt").

    start:
        Start character offset in the source document.

    end:
        End character offset in the source document.

    metadata:
        Arbitrary user-defined metadata.
    """

    term: str
    text: str
    context: str | None = None

    entity_type: EntityType = EntityType.ITEM

    language: str | None = None

    start: int | None = None
    end: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        """Return whether contextual information is available."""
        return bool(self.context)

    @property
    def span(self) -> tuple[int, int] | None:
        """Return the mention span if available."""
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)

    @property
    def normalized(self) -> str:
        """Return a normalized representation of the mention."""
        return self.text.strip()

    def __str__(self) -> str:
        return self.text