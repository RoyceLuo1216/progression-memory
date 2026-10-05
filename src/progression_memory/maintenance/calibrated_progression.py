"""Auditable progression prototype with a learned attachment proposal.

The small synthetic dataset is for interface validation, not a trained production
writer. The model predicts whether an event belongs to an existing process;
hierarchy and state transitions remain separate, explicit decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np

from progression_memory.evaluation.progressions import hashed_embedding
from progression_memory.evaluation.retrieval import tokenize
from progression_memory.routing.router import cosine_similarity


class Action(StrEnum):
    NO_UPDATE = "no_update"
    ATTACH = "attach"
    CREATE_ROOT = "create_root"
    CREATE_CHILD = "create_child"
    MULTI_ATTACH = "multi_attach"


@dataclass(frozen=True)
class Event:
    id: str
    subject_id: str
    text: str
    time: datetime | None
    source_id: str


@dataclass(frozen=True)
class State:
    text: str
    evidence_ids: frozenset[str]


@dataclass
class Segment:
    id: str
    progression_id: str
    event_ids: list[str]
    before: State
    after: State
    change: str


@dataclass
class Progression:
    id: str
    subject_id: str
    identity: str
    scope: str
    parent_ids: set[str] = field(default_factory=set)
    child_ids: set[str] = field(default_factory=set)
    segment_ids: list[str] = field(default_factory=list)
    current_state: State = field(default_factory=lambda: State("", frozenset()))


@dataclass(frozen=True)
class Decision:
    action: Action
    target_ids: tuple[str, ...] = ()
    parent_id: str | None = None
    identity: str = ""
    scope: str = ""
    state: str = ""
    change: str = ""
    new_segment: bool = False
    reason: str = ""


class ScriptedWriter:
    """Explicit decisions for a small audited trace, independent of model scores."""

    def __init__(self, decisions: dict[str, Decision]) -> None:
        self.decisions = decisions

    def decide(self, event: Event, memory: Memory) -> Decision:
        return self.decisions[event.id]


class Memory:
    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self.progressions: dict[str, Progression] = {}
        self.segments: dict[str, Segment] = {}

    def apply(self, event: Event, decision: Decision) -> None:
        """Validate the complete mutation before changing any store."""
        if not event.id or not event.source_id or not event.subject_id or event.id in self.events:
            raise ValueError("event needs unique id, subject and source")
        if not decision.reason:
            raise ValueError("writer decision needs an auditable reason")
        if len(set(decision.target_ids)) != len(decision.target_ids):
            raise ValueError("duplicate target")
        if decision.action in (Action.ATTACH, Action.MULTI_ATTACH):
            if len(decision.target_ids) < (2 if decision.action == Action.MULTI_ATTACH else 1):
                raise ValueError("attachment targets missing")
            if decision.action == Action.ATTACH and len(decision.target_ids) != 1:
                raise ValueError("use multi_attach for multiple targets")
            targets = [self.progressions[target] for target in decision.target_ids]
            if any(p.subject_id != event.subject_id for p in targets):
                raise ValueError("cross-subject attachment")
        elif decision.action in (Action.CREATE_ROOT, Action.CREATE_CHILD):
            if decision.target_ids or not decision.identity or not decision.scope or not decision.state:
                raise ValueError("new progression requires identity, scope and state")
            if decision.action == Action.CREATE_CHILD:
                parent = self.progressions[decision.parent_id or ""]
                if parent.subject_id != event.subject_id:
                    raise ValueError("cross-subject parent")
            elif decision.parent_id is not None:
                raise ValueError("root cannot have a parent")
            if any(p.subject_id == event.subject_id and p.identity.casefold() == decision.identity.casefold()
                   for p in self.progressions.values()):
                raise ValueError("duplicate progression identity")
            targets = []
        elif decision.action == Action.NO_UPDATE:
            if decision.target_ids or decision.parent_id:
                raise ValueError("no_update cannot target a progression")
            targets = []
        else:
            raise ValueError(decision.action)
        if targets and not decision.state:
            raise ValueError("attachment must supply evidence-backed state")

        self.events[event.id] = event
        if decision.action in (Action.CREATE_ROOT, Action.CREATE_CHILD):
            pid = f"progression:{len(self.progressions)}"
            parent_ids = {decision.parent_id} if decision.parent_id else set()
            created = Progression(pid, event.subject_id, decision.identity, decision.scope,
                                  parent_ids=parent_ids)
            self.progressions[pid] = created
            if decision.parent_id:
                self.progressions[decision.parent_id].child_ids.add(pid)
            targets = [created]
        for progression in targets:
            before = progression.current_state
            after = State(decision.state, before.evidence_ids | {event.id})
            if decision.new_segment or not progression.segment_ids:
                sid = f"segment:{len(self.segments)}"
                segment = Segment(sid, progression.id, [event.id], before, after, decision.change)
                self.segments[sid] = segment
                progression.segment_ids.append(sid)
            else:
                segment = self.segments[progression.segment_ids[-1]]
                segment.event_ids.append(event.id)
                segment.after = after
                segment.change = decision.change
            progression.current_state = after


def _overlap(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    return len(a & b) / len(a | b) if a | b else 0.0


def features(event: Event, progression: Progression, last_time: datetime | None = None) -> np.ndarray:
    """Observable proposal features. No manually weighted composite score."""
    e = hashed_embedding(event.text)
    identity = hashed_embedding(progression.identity + " " + progression.scope)
    state = hashed_embedding(progression.current_state.text)
    days = abs((event.time - last_time).total_seconds()) / 86400 if event.time and last_time else 0.0
    return np.array([
        cosine_similarity(e, identity), cosine_similarity(e, state),
        _overlap(event.text, progression.identity + " " + progression.scope),
        _overlap(event.text, progression.current_state.text),
        float(event.subject_id == progression.subject_id), math.log1p(days),
    ], dtype=float)


@dataclass
class LinearAttachmentModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    threshold: float

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.0  # constant feature contributes zero after centering
        design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        return mean, scale, coefficients

    @staticmethod
    def _cutoff(scores: np.ndarray, labels: np.ndarray) -> float:
        """Empirical balanced-accuracy cutoff, chosen from observed scores."""
        distinct = sorted(set(float(s) for s in scores))
        candidates = [math.nextafter(distinct[0], -math.inf)]
        candidates += [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
        candidates += [math.nextafter(distinct[-1], math.inf)]
        def quality(cut: float) -> float:
            predicted = scores >= cut
            return (np.mean(predicted[labels == 1]) + np.mean(~predicted[labels == 0])) / 2
        return min(candidates, key=lambda cut: (-quality(cut), cut))

    @classmethod
    def train(cls, rows: list[dict]) -> tuple[LinearAttachmentModel, dict]:
        if len(rows) != 100:
            raise ValueError("prototype requires exactly 100 labeled pairs")
        x = np.array([row["features"] for row in rows], dtype=float)
        y = np.array([row["attach"] for row in rows], dtype=float)
        groups = [row["scenario"] for row in rows]
        if not set(y) == {0.0, 1.0} or not np.isfinite(x).all():
            raise ValueError("finite features and both labels are required")
        # Leave each scenario out while calibrating: its four candidate labels
        # cannot leak into its own fitted score.
        out_of_fold = np.zeros(len(rows))
        for group in sorted(set(groups)):
            held = np.array([value == group for value in groups])
            mean, scale, coef = cls._fit(x[~held], y[~held])
            out_of_fold[held] = np.column_stack((np.ones(sum(held)),
                                                (x[held] - mean) / scale)) @ coef
        threshold = cls._cutoff(out_of_fold, y)
        mean, scale, coef = cls._fit(x, y)
        predictions = out_of_fold >= threshold
        report = {"rows": len(rows), "scenarios": len(set(groups)),
                  "out_of_fold_balanced_accuracy": float((np.mean(predictions[y == 1]) +
                                                          np.mean(~predictions[y == 0])) / 2),
                  "out_of_fold_false_attachments": int(sum(predictions & (y == 0))),
                  "out_of_fold_missed_attachments": int(sum(~predictions & (y == 1))),
                  "threshold": threshold,
                  "coefficients": coef.tolist(), "feature_order": [
                      "identity_cosine", "state_cosine", "identity_overlap",
                      "state_overlap", "same_subject", "log_days_since_update",
                  ]}
        return cls(mean, scale, coef, threshold), report

    def score(self, vector: np.ndarray) -> float:
        return float(np.r_[1.0, (vector - self.mean) / self.scale] @ self.coefficients)

    def proposes_attachment(self, vector: np.ndarray) -> bool:
        return self.score(vector) >= self.threshold

    def propose(self, event: Event, memory: Memory) -> list[tuple[str, float]]:
        """Rank learned attach candidates; creating children needs another decision."""
        matches = []
        for progression in memory.progressions.values():
            if progression.subject_id != event.subject_id:
                continue
            latest = memory.segments[progression.segment_ids[-1]] if progression.segment_ids else None
            previous = memory.events[latest.event_ids[-1]].time if latest else None
            value = self.score(features(event, progression, previous))
            if value >= self.threshold:
                matches.append((progression.id, value))
        return sorted(matches, key=lambda item: (-item[1], item[0]))

    def save(self, path: Path, report: dict) -> None:
        path.write_text(json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                                    "coefficients": self.coefficients.tolist(),
                                    "threshold": self.threshold, "report": report}, indent=2) + "\n")
