# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True, frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text
 