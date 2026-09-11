from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TemporalSegment(BaseModel):
    """Finest-level stable temporal interval for a progression."""

    id: str
    progression_id: str
    start_time: datetime
    end_time: datetime | None = None

    summary: str = ""
    episode_ids: list[str] = Field(default_factory=list)

    embedding: list[float] | None = None
    state_vector: list[float] | None = None
    uncertainty: list[float] | None = None
    resolution_level: int = Field(default=0, ge=0)
