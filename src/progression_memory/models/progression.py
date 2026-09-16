from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class StorageTier(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class ProgressionStateCheckpoint(BaseModel):
    """Evidence-backed posterior state of a progression after an update."""

    episode_id: str
    timestamp: datetime
    prior_state: str = ""
    evidence: str
    posterior_state: str
    prior_embedding: list[float] | None = None
    posterior_embedding: list[float] | None = None
    delta_embedding: list[float] | None = None
    confidence: float = 1.0


class Progression(BaseModel):
    """Persistent, recursively conditioned state of an evolving user process."""

    id: str
    title: str
    scope: str

    high_level_summary: str = ""
    current_state: str = ""
    stage: str | None = None

    start_time: datetime | None = None
    last_updated_time: datetime | None = None

    activity: float = 0.0
    volatility: float = 0.0
    continuity: float = 0.0
    persistence: float = 0.0
    importance: float = 0.0
    event_frequency: float = 0.0
    contradiction_rate: float = 0.0
    retrieval_count: int = 0

    episode_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    latest_episode_id: str | None = None

    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)

    embedding: list[float] | None = None
    state_embedding: list[float] | None = None
    state_history: list[ProgressionStateCheckpoint] = Field(default_factory=list)
    storage_tier: StorageTier = StorageTier.HOT

    evidence_version: str | None = None
