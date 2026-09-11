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
    state_vector: list[float] | None = None
    entities: list[str] = Field(default_factory=list)

    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0)

    progression_ids: list[str] = Field(default_factory=list)
    source_pointer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def event_time(self) -> datetime:
        """Best available event time, falling back to dialogue time."""
        return self.semantic_time or self.dialogue_time
