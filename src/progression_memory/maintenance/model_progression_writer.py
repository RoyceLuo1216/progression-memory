"""Structured model writer for the progression memory contract.

The learned attachment model proposes candidates; the language model decides
identity, hierarchy, per-target state, and phase. Synthetic proposal scores are
advisory until real, subject-held-out attachment labels exist.
"""

from __future__ import annotations

import inspect
import json

from pydantic import BaseModel, ConfigDict, Field

from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, LinearAttachmentModel, Memory, TargetUpdate,
)


class _Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    progression_id: str
    state: str = Field(min_length=1)
    change: str
    new_segment: bool


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Action
    target_ids: list[str]
    parent_id: str | None
    identity: str
    scope: str
    state: str
    change: str
    new_segment: bool
    target_updates: list[_Target]
    reason: str = Field(min_length=1)


class ModelProgressionWriter:
    """Adapter for a model exposing generate_messages or generate."""

    def __init__(self, model, proposal_model: LinearAttachmentModel | None = None):
        self.model = model
        self.proposal_model = proposal_model

    def decide(self, event: Event, memory: Memory) -> Decision:
        candidates = []
        scores = dict(self.proposal_model.propose(event, memory)) if self.proposal_model else {}
        for p in memory.progressions.values():
            if p.subject_id != event.subject_id:
                continue
            last = memory.segments[p.segment_ids[-1]] if p.segment_ids else None
            candidates.append({"id": p.id, "identity": p.identity, "scope": p.scope,
                               "parent_ids": sorted(p.parent_ids), "current_state": p.current_state.text,
                               "latest_change": last.change if last else "",
                               "attachment_proposal_score": scores.get(p.id)})
        system = (
            "Track a single person's ongoing goals, interests and life processes. "
            "The CURRENT event is evidence; existing state and candidate scores are context only. "
            "Return one JSON object with exactly action, target_ids, parent_id, identity, scope, "
            "state, change, new_segment, target_updates, reason. Actions are no_update, attach, "
            "create_root, create_child, multi_attach. A child is a durable narrower process "
            "under an existing parent. For each multiattach target provide its own state, change, "
            "and phase boundary in target_updates. Reuse an identity only for the same evolving "
            "process, even if a different process shares topic words. Start a segment for a "
            "meaningful phase change; otherwise extend it. No-update is valid. Do not infer "
            "unsupported facts, outcomes or parent links. The reason must refer to CURRENT. "
            "For unused fields return empty strings, empty lists and null parent_id."
        )
        user = json.dumps({"current": {"id": event.id, "subject_id": event.subject_id,
                                       "text": event.text, "time": event.time.isoformat()
                                       if event.time else None, "source_id": event.source_id},
                           "candidates": candidates}, ensure_ascii=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.model, "generate_messages"):
            method = self.model.generate_messages
            params = inspect.signature(method).parameters.values()
            if any(p.name == "json_schema" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                raw = method(messages, json_schema=_Response.model_json_schema())
            else:
                raw = method(messages)
        else:
            raw = self.model.generate(json.dumps(messages, ensure_ascii=False))
        parsed = _Response.model_validate_json(raw)
        return Decision(action=parsed.action, target_ids=tuple(parsed.target_ids),
                        parent_id=parsed.parent_id, identity=parsed.identity,
                        scope=parsed.scope, state=parsed.state, change=parsed.change,
                        new_segment=parsed.new_segment, reason=parsed.reason,
                        target_updates=tuple(TargetUpdate(**target.model_dump())
                                             for target in parsed.target_updates))


def construct(documents, subject_id: str, writer: ModelProgressionWriter) -> Memory:
    """Build once per corpus, in source order, retaining every original event."""
    from progression_memory.evaluation.progressions import parse_timestamp

    memory = Memory()
    for ordinal, document in enumerate(documents):
        event = Event(document.id, subject_id, document.text,
                      parse_timestamp(document.timestamp, ordinal), document.id)
        decision = writer.decide(event, memory)
        memory.apply(event, decision)
    return memory
