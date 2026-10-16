"""Rebuildable derived state; never mixes child claims into direct evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from progression_memory.v2.models import Progression


def child_fingerprint(children: list[Progression]) -> str:
    """Changes when immediate child identity, state or derived data changes."""
    snapshot = [
        {
            "id": c.id,
            "version": c.version,
            "parent_id": c.parent_id,
            "lifecycle": c.lifecycle.value,
            "state": c.state,
            "derived": c.derived_state,
            "fingerprint": c.derived_fingerprint,
        }
        for c in sorted(children, key=lambda c: c.id)
    ]
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aggregate_children(children: list[Progression]) -> dict[str, Any]:
    """Generic, truthful direct-child counts; no inferred parent completion."""
    statuses = Counter(child.lifecycle.value for child in children)
    total = len(children)
    completed = statuses["completed"]
    return {
        "child_count": total,
        "completed_children": completed,
        "active_children": statuses["active"],
        "paused_children": statuses["paused"],
        "cancelled_children": statuses["cancelled"],
        "completion_ratio": completed / total if total else None,
        "children": [
            {
                "id": child.id,
                "name": child.name,
                "lifecycle": child.lifecycle.value,
                "state": child.state.copy(),
            }
            for child in sorted(children, key=lambda item: item.id)
        ],
    }


def effective_state(progression: Progression) -> Mapping[str, Any]:
    """Keep independently sourced views separate for query assembly."""
    return {"direct": progression.state.copy(), "derived": progression.derived_state.copy()}
