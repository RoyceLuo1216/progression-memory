from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """Highest-resolution source-of-truth memory extracted from an interaction."""

    id: str
    summary: str
    dialogue_time: datetime
    semantic_time: datetime | None = None
    embedding: list[float] | None = None
    entities: list[str] = Field(default_factory=list)
    progression_ids: list[str] = Field(default_factory=list)
    source_pointer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
