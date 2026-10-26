"""V2 query-time units: no future evidence and explicit provenance."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.aggregation import effective_state
from progression_memory.v2.state_engine import StateEngine

View = Literal["state", "full", "evidence"]


def build_documents(
    engine: StateEngine, *,
    view: View = "full",
    subject_id: str | None = None,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> tuple[RetrievalDocument, ...]:
    """Materialize read-only units for a single subject and time horizon.

    Evidence IDs are credited only to exact observations backing a unit.
    Parent derived-state provenance recursively collects contributing child evidence.
    Historical queries never use present-day derived summaries.
    """
    if view not in ("state", "full", "evidence"):
        raise ValueError("unknown view")
    from progression_memory.v2.hierarchy import HierarchyIndex

    hierarchy = HierarchyIndex(engine.progressions)
    allowed = {
        obs.id: obs for obs in engine.observations.values()
        if (subject_id is None or obs.subject_id == subject_id)
        and (valid_at is None or obs.effective_at <= valid_at)
        and (known_at is None or obs.recorded_at <= known_at)
    }
    documents: list[RetrievalDocument] = []
    if view == "evidence":
        return tuple(
            RetrievalDocument(id=f"evidence:{obs.id}", text=obs.text,
                              evidence_ids=frozenset({obs.id}),
                              timestamp=obs.effective_at.isoformat())
            for obs in sorted(allowed.values(), key=lambda x: x.id)
        )

    for pid, p in sorted(engine.progressions.items()):
        if subject_id is not None and p.subject_id != subject_id:
            continue
        contributing = {d.observation_id for d in engine.deltas.values()
                        if d.progression_id == pid and d.observation_id in allowed}
        if valid_at is None and known_at is None:
            direct = p.state
        else:
            if valid_at is None:
                # Knowledge-time-only queries still need a valid-time horizon.
                valid_horizon = max((o.effective_at for o in allowed.values()), default=None)
            else:
                valid_horizon = valid_at
            direct = (engine.state_at(pid, valid_at=valid_horizon, known_at=known_at)
                      if valid_horizon is not None else {})
        if not direct and not contributing and not p.derived_state:
            continue
        payload = {"progression": p.name, "type": p.progression_type,
                   "direct_state": direct}
        # Materialized derived state is current only; never leak it to historical snapshots.
        if valid_at is None and known_at is None and view == "full":
            payload["derived_state"] = effective_state(p)["derived"]
            payload["parent"] = p.parent_id
            for descendant in hierarchy.descendants(pid):
                contributing.update(
                    d.observation_id for d in engine.deltas.values()
                    if d.progression_id == descendant and d.observation_id in allowed
                )
        if not contributing and not direct:
            continue
        documents.append(RetrievalDocument(
            id=f"v2:state:{pid}",
            text=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            evidence_ids=frozenset(contributing),
            metadata={"kind": "state", "progression_id": pid},
        ))
    if view == "full":
        for obs in sorted(allowed.values(), key=lambda x: x.id):
            documents.append(RetrievalDocument(
                id=f"v2:evidence:{obs.id}", text=obs.text,
                evidence_ids=frozenset({obs.id}),
                timestamp=obs.effective_at.isoformat(),
                metadata={"kind": "observation"},
            ))
    return tuple(documents)


def retrieve(
    engine: StateEngine, query: str, *,
    k: int = 8, view: View = "full",
    subject_id: str | None = None,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> list[RetrievalDocument]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    units = build_documents(
        engine, view=view, subject_id=subject_id,
        valid_at=valid_at, known_at=known_at,
    )
    return BM25Index(units).retrieve(query, k)
