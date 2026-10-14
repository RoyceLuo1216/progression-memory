"""Deterministic, evidence-preserving V2 state engine.

This deliberately does not invoke an LLM: writer proposals are validated inputs.
The store is in memory for milestone 1; durability and transactions are next.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from progression_memory.v2.models import (
    AttributeOperation, ChangeKind, Observation, Progression,
    ProgressionLifecycle, Segment, StateDelta, StateRevision, StorageTier,
)


class StateEngine:
    def __init__(self) -> None:
        self.progressions: dict[str, Progression] = {}
        self.observations: dict[str, Observation] = {}
        self.deltas: dict[str, StateDelta] = {}
        self.revisions: dict[str, list[StateRevision]] = {}
        self.segments: dict[str, Segment] = {}

    def create_progression(self, progression: Progression) -> None:
        if progression.id in self.progressions:
            raise ValueError("duplicate progression")
        if progression.parent_id is not None:
            parent = self.progressions.get(progression.parent_id)
            if parent is None or parent.subject_id != progression.subject_id:
                raise ValueError("parent missing or subject mismatch")
        if progression.version or progression.segment_ids or progression.state:
            raise ValueError("new progression must have empty materialized state")
        self.progressions[progression.id] = progression.model_copy(deep=True)
        self.revisions[progression.id] = []

    @staticmethod
    def _apply(state: dict[str, Any], delta: StateDelta) -> dict[str, Any]:
        next_state = deepcopy(state)
        keys = [patch.key for patch in delta.patches]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate attribute in delta")
        for patch in delta.patches:
            key, value = patch.key, deepcopy(patch.value)
            if patch.operation == AttributeOperation.SET:
                next_state[key] = value
            elif patch.operation == AttributeOperation.RETRACT:
                next_state.pop(key, None)
            elif patch.operation in (AttributeOperation.ADD, AttributeOperation.REMOVE):
                old = next_state.get(key, [])
                if not isinstance(old, list):
                    raise ValueError(f"{key} is not a list")
                if patch.operation == AttributeOperation.ADD:
                    if value not in old:
                        next_state[key] = old + [value]
                elif value in old:
                    updated = old.copy()
                    updated.remove(value)
                    next_state[key] = updated
            else:
                raise ValueError(f"unsupported operation {patch.operation}")
        if delta.stage is not None:
            next_state["stage"] = delta.stage
        return next_state

    @staticmethod
    def _replay(
        progression: Progression,
        entries: list[tuple[Observation, StateDelta]],
    ) -> tuple[Progression, list[StateRevision], dict[str, Segment]]:
        result = progression.model_copy(deep=True)
        result.state = {}
        result.version = 0
        result.segment_ids = []
        result.last_effective_at = None
        result.lifecycle = ProgressionLifecycle.ACTIVE
        result.storage_tier = progression.storage_tier
        result.metadata = deepcopy(progression.metadata)
        revisions: list[StateRevision] = []
        segments: dict[str, Segment] = {}
        evidence: list[str] = []
        current_segment: Segment | None = None

        # Effective time defines historical order. Observation time resolves ties.
        # Stable ID ties make replay independent of ingestion ordering.
        entries.sort(key=lambda row: (
            row[0].effective_at, row[0].recorded_at, row[0].id,
        ))
        for obs, delta in entries:
            if delta.change == ChangeKind.REVIVE:
                if result.lifecycle not in (
                    ProgressionLifecycle.COMPLETED, ProgressionLifecycle.CANCELLED,
                    ProgressionLifecycle.PAUSED,
                ) and result.storage_tier == StorageTier.ACTIVE:
                    raise ValueError("revive requires an inactive progression")
                result.lifecycle = ProgressionLifecycle.ACTIVE
                result.storage_tier = StorageTier.ACTIVE
            if delta.change == ChangeKind.CLOSE:
                result.lifecycle = ProgressionLifecycle.COMPLETED
            if delta.change in (ChangeKind.TRANSITION, ChangeKind.SPLIT, ChangeKind.REVIVE) or current_segment is None:
                if current_segment is not None:
                    current_segment.ended_at = obs.effective_at
                sid = f"{result.id}:segment:{len(segments) + 1}"
                current_segment = Segment(
                    id=sid, progression_id=result.id, sequence=len(segments) + 1,
                    started_at=obs.effective_at, boundary=delta.change,
                )
                segments[sid] = current_segment
                result.segment_ids.append(sid)
            if delta.change == ChangeKind.CLOSE and current_segment is not None:
                current_segment.ended_at = obs.effective_at
            result.state = StateEngine._apply(result.state, delta)
            current_segment.observation_ids.append(obs.id)
            evidence.append(obs.id)
            result.version += 1
            result.last_effective_at = obs.effective_at
            revisions.append(StateRevision(
                progression_id=result.id, version=result.version,
                delta_id=delta.id, effective_at=obs.effective_at,
                recorded_at=obs.recorded_at, state=deepcopy(result.state),
                evidence_ids=evidence.copy(), segment_id=current_segment.id,
            ))
        return result, revisions, segments

    def ingest(self, observation: Observation, delta: StateDelta) -> bool:
        """Apply an observation and delta atomically; duplicate identical IDs are no-ops.

        Retrospective writes replay the whole progression to ensure the latest
        applicable effective time, rather than arrival order, determines state.
        """
        if delta.observation_id != observation.id:
            raise ValueError("delta observation ID mismatch")
        progression = self.progressions.get(delta.progression_id)
        if progression is None or progression.subject_id != observation.subject_id:
            raise ValueError("missing progression or subject mismatch")
        prior_obs, prior_delta = self.observations.get(observation.id), self.deltas.get(delta.id)
        if prior_obs is not None and prior_obs != observation:
            raise ValueError("conflicting duplicate observation ID")
        if prior_delta is not None:
            if prior_delta == delta and prior_obs == observation:
                return False
            raise ValueError("conflicting duplicate delta ID")
        if any(
            d.observation_id == observation.id and d.progression_id == delta.progression_id
            for d in self.deltas.values()
        ):
            raise ValueError("observation already updates this progression")
        entries = [
            (self.observations[d.observation_id], d)
            for d in self.deltas.values() if d.progression_id == progression.id
        ]
        entries.append((observation, delta))
        base = progression.model_copy(deep=True)
        next_progression, next_revisions, next_segments = self._replay(base, entries)
        # Commit only after full replay succeeds.
        for sid in progression.segment_ids:
            self.segments.pop(sid, None)
        self.progressions[progression.id] = next_progression
        self.revisions[progression.id] = next_revisions
        self.segments.update(next_segments)
        self.observations[observation.id] = observation
        self.deltas[delta.id] = delta
        return True

    def state_at(
        self, progression_id: str, *,
        valid_at: datetime, known_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Reconstruct from events available by knowledge time and effective by valid time."""
        if progression_id not in self.progressions:
            raise KeyError(progression_id)
        subset = [
            (self.observations[d.observation_id], d)
            for d in self.deltas.values()
            if d.progression_id == progression_id
            and self.observations[d.observation_id].effective_at <= valid_at
            and (known_at is None or self.observations[d.observation_id].recorded_at <= known_at)
        ]
        current, _, _ = self._replay(self.progressions[progression_id], subset)
        return deepcopy(current.state)

    def move_to_tier(self, progression_id: str, tier: StorageTier) -> None:
        """Logical archive only. No evidence is removed."""
        progression = self.progressions[progression_id]
        progression.storage_tier = tier

    def evidence_for(self, progression_id: str) -> list[Observation]:
        rows = [
            self.observations[d.observation_id]
            for d in self.deltas.values() if d.progression_id == progression_id
        ]
        return sorted(rows, key=lambda obs: (obs.effective_at, obs.recorded_at, obs.id))
