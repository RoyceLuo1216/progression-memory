"""V2 writer orchestration: explicit proposals, candidate admission and atomic batches.

The model proposes changes; the deterministic engine validates and commits them.
No heuristic fallback on invalid model output.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from progression_memory.v2.models import (
    AttributePatch,
    ChangeKind,
    Observation,
    Progression,
    StateDelta,
)
from progression_memory.v2.state_engine import StateEngine


class ProposedUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str  # create | append | transition | split | close | revive
    progression_id: str | None = None
    name: str | None = None
    parent_id: str | None = None
    progression_type: str = "general"
    patches: tuple[AttributePatch, ...] = ()
    stage: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> ProposedUpdate:
        allowed = {"create", *(x.value for x in ChangeKind)}
        if self.action not in allowed:
            raise ValueError("unsupported action")
        if self.action == "create":
            if self.progression_id is not None or not self.name:
                raise ValueError("create requires a name and no progression ID")
        elif not self.progression_id:
            raise ValueError("updates require an existing progression ID")
        return self


class WriterProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: tuple[ProposedUpdate, ...] = ()


class Writer(Protocol):
    def decide(
        self, observation: Observation, candidates: list[Progression]
    ) -> WriterProposal: ...


class ScriptedWriter:
    """Deterministic oracle for testing the orchestration contract."""

    def __init__(self, decisions: dict[str, WriterProposal]) -> None:
        self.decisions = decisions

    def decide(
        self, observation: Observation, candidates: list[Progression]
    ) -> WriterProposal:
        return self.decisions[observation.id]


def candidates_for(engine: StateEngine, observation: Observation) -> list[Progression]:
    """Conservative same-subject candidate gate; includes archived processes.

    The scoring/reranking model remains replaceable in later milestones.
    Do not force-match solely by topic or similarity.
    """
    return [
        deepcopy(p) for p in sorted(engine.progressions.values(), key=lambda p: p.id)
        if p.subject_id == observation.subject_id
    ]


class WritePipeline:
    def __init__(self, engine: StateEngine, writer: Writer) -> None:
        self.engine = engine
        self.writer = writer

    def process(self, observation: Observation) -> WriterProposal:
        candidates = candidates_for(self.engine, observation)
        proposal = self.writer.decide(observation, candidates)
        if not isinstance(proposal, WriterProposal):
            raise TypeError("writer must return validated WriterProposal")
        # Plan updates on a copy, then swap only when all changes pass.
        staged = deepcopy(self.engine)
        used: set[str] = set()
        for index, update in enumerate(proposal.updates):
            if update.action == "create":
                pid = f"v2:{observation.id}:{index}"
                if any(
                    p.subject_id == observation.subject_id
                    and p.name.casefold() == update.name.casefold()
                    for p in staged.progressions.values()
                ):
                    raise ValueError("duplicate process identity")
                if update.parent_id is not None and update.parent_id not in {
                    p.id for p in candidates
                }:
                    raise ValueError("unknown or inadmissible parent")
                staged.create_progression(Progression(
                    id=pid, subject_id=observation.subject_id,
                    name=update.name, parent_id=update.parent_id,
                    progression_type=update.progression_type,
                ))
                kind = ChangeKind.APPEND
            else:
                pid = update.progression_id
                if pid not in {p.id for p in candidates}:
                    raise ValueError("writer selected unproposed progression")
                kind = ChangeKind(update.action)
            if pid in used:
                raise ValueError("multiple updates to one progression in one observation")
            used.add(pid)
            delta = StateDelta(
                id=f"v2:delta:{observation.id}:{index}", progression_id=pid,
                observation_id=observation.id, change=kind,
                patches=update.patches, stage=update.stage, reason=update.reason,
            )
            staged.ingest(observation, delta)
        # Record raw no-update evidence too, without falsely creating a process.
        existing = staged.observations.get(observation.id)
        if existing is not None and existing != observation:
            raise ValueError("source ID collision")
        staged.observations[observation.id] = observation
        self.engine.progressions = staged.progressions
        self.engine.observations = staged.observations
        self.engine.deltas = staged.deltas
        self.engine.revisions = staged.revisions
        self.engine.segments = staged.segments
        return proposal
