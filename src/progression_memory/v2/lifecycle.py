"""Conservative logical aging policy for persisted progressions.

Lifecycle ("completed" or "active") is separate from physical storage tier.
Never deletes observations, deltas, segments, or progression identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from progression_memory.v2.models import ProgressionLifecycle, StorageTier
from progression_memory.v2.state_engine import StateEngine


@dataclass(frozen=True)
class LifecyclePolicy:
    stale_after_days: int = 90
    archive_after_days: int = 365

    def __post_init__(self) -> None:
        if self.stale_after_days <= 0 or self.archive_after_days <= self.stale_after_days:
            raise ValueError("archive threshold must exceed positive stale threshold")

    def target(self, engine: StateEngine, pid: str, now: datetime) -> StorageTier:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        progression = engine.progressions[pid]
        # A newly created progression has no evidence timestamp; do not age it.
        if progression.last_effective_at is None:
            return progression.storage_tier
        # Use ingestion receipt timestamps to judge inactivity rather than semantic event
        # time, because recently reported historical facts still constitute recent contact.
        last_received = max(
            (engine.observations[d.observation_id].recorded_at
             for d in engine.deltas.values() if d.progression_id == pid),
            default=None,
        )
        if last_received is None:
            return progression.storage_tier
        age = now - last_received
        if age < timedelta(0):
            raise ValueError("maintenance clock predates last ingested update")
        if age >= timedelta(days=self.archive_after_days):
            return StorageTier.ARCHIVED
        if age >= timedelta(days=self.stale_after_days):
            return StorageTier.STALE
        return StorageTier.ACTIVE


def apply_lifecycle_policy(
    engine: StateEngine, policy: LifecyclePolicy, *, now: datetime,
) -> dict[str, str]:
    """Deterministically adjust logical tiers; preserving evidence and user lifecycle."""
    changes: dict[str, str] = {}
    for pid in sorted(engine.progressions):
        progression = engine.progressions[pid]
        target = policy.target(engine, pid, now)
        if target != progression.storage_tier:
            engine.move_to_tier(pid, target)
            changes[pid] = target.value
    return changes
