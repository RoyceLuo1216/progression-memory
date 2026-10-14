"""Canonical V2 contracts. V1 models and retrievers remain unchanged."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeSource(StrEnum):
    SEMANTIC = "semantic"
    WALL_FALLBACK = "wall_fallback"


class ProgressionLifecycle(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class StorageTier(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class ChangeKind(StrEnum):
    APPEND = "append"
    TRANSITION = "transition"
    SPLIT = "split"
    CLOSE = "close"
    REVIVE = "revive"


class AttributeOperation(StrEnum):
    SET = "set"
    ADD = "add"
    REMOVE = "remove"
    RETRACT = "retract"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    subject_id: str
    source_id: str
    text: str
    recorded_at: datetime
    semantic_at: datetime | None = None
    # Effective time determines the order in which state updates apply.
    # When unknown, ingestion time is only a fallback, not an inferred fact.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_at(self) -> datetime:
        return self.semantic_at or self.recorded_at

    @property
    def time_source(self) -> TimeSource:
        return TimeSource.SEMANTIC if self.semantic_at is not None else TimeSource.WALL_FALLBACK

    @model_validator(mode="after")
    def validate_times(self) -> Observation:
        for value in (self.recorded_at, self.semantic_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("observation times must be timezone-aware")
        return self


class AttributePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(min_length=1)
    operation: AttributeOperation
    value: Any = None

    @model_validator(mode="after")
    def validate_operation(self) -> AttributePatch:
        if self.operation in (AttributeOperation.SET, AttributeOperation.ADD) and self.value is None:
            raise ValueError("set/add requires value")
        return self


class StateDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    progression_id: str
    observation_id: str
    change: ChangeKind = ChangeKind.APPEND
    patches: tuple[AttributePatch, ...] = ()
    reason: str = ""
    # Optional phase label, not a global enum for all progression domains.
    stage: str | None = None


class StateRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    progression_id: str
    version: int
    delta_id: str
    effective_at: datetime
    recorded_at: datetime
    state: dict[str, Any]
    evidence_ids: list[str]
    segment_id: str


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    progression_id: str
    sequence: int
    started_at: datetime
    ended_at: datetime | None = None
    boundary: ChangeKind
    observation_ids: list[str] = Field(default_factory=list)


class Progression(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    subject_id: str
    name: str
    parent_id: str | None = None
    progression_type: str = "general"
    lifecycle: ProgressionLifecycle = ProgressionLifecycle.ACTIVE
    storage_tier: StorageTier = StorageTier.ACTIVE
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 0
    segment_ids: list[str] = Field(default_factory=list)
    last_effective_at: datetime | None = None
