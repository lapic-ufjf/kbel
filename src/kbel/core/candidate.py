# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Candidate:
    """An entity returned by a search.

    `datatype` is, for a property, the KIF datatype of its values (item,
    time, quantity, string, ...); None when unknown or not a property.
    """

    iri: str
    label: str
    id: str | None = None
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    datatype: Any = None