# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Candidate:
    iri: str
    label: str
    id: str | None = None
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)