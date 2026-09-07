from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TemporalSegment(BaseModel):
    id: str
    progression_id: str
    start_time: datetime
    end_time: datetime | None = None
    summary: str = ""
    episode_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
