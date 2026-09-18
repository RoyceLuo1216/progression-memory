from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from progression_memory.maintenance.manager import LanguageModel
from progression_memory.models.progression import Progression
from progression_memory.routing.router import cosine_similarity


@dataclass(frozen=True)
class MaintenanceTrigger:
    reason: str
    score: float


def split_trigger(progression: Progression, *, min_updates: int = 8) -> MaintenanceTrigger | None:
    history = progression.state_history
    if len(history) < min_updates: return None
    deltas = [np.asarray(x.delta_embedding, dtype=float) for x in history if x.delta_embedding]
    if len(deltas) < min_updates // 2: return None
    dispersion = float(np.mean([np.linalg.norm(x) for x in deltas]))
    score = min(1.0, .55 * dispersion + .30 * progression.contradiction_rate + .15 * progression.volatility)
    return MaintenanceTrigger("trajectory_instability", score) if score >= .45 else None


def merge_trigger(left: Progression, right: Progression) -> MaintenanceTrigger | None:
    identity = (cosine_similarity(left.embedding, right.embedding) + 1) / 2
    state = (cosine_similarity(left.state_embedding, right.state_embedding) + 1) / 2
    score = .6 * identity + .4 * state
    return MaintenanceTrigger("trajectory_convergence", score) if score >= .82 else None


def maintenance_prompt(progressions: list[Progression], trigger: MaintenanceTrigger) -> str:
    rows = [{"id": p.id, "identity": p.high_level_summary or p.title, "state": p.current_state,
             "updates": len(p.state_history), "children": p.child_ids} for p in progressions]
    return (
        "You maintain evolving long-term memory. Inspect only the flagged progressions. "
        "Return JSON only with action (keep|split|merge|create_child|archive), ids, reason, "
        "and for split/create_child a concise child_identity. Preserve raw evidence; do not invent facts.\n"
        f"TRIGGER={trigger.reason}:{trigger.score:.3f}\nPROGRESSIONS={rows}"
    )


def inspect(model: LanguageModel, progressions: list[Progression], trigger: MaintenanceTrigger) -> str:
    """Keep execution separate: this returns the model plan for validation/application by the store."""
    return model.generate(maintenance_prompt(progressions, trigger))


def parse_plan(raw: str) -> dict[str, object]:
    import json
    value = json.loads(raw)
    action = str(value.get("action", "")).lower()
    if action not in {"keep", "split", "merge", "create_child", "archive"}:
        raise ValueError("invalid maintenance action")
    value["action"] = action
    return value


def apply_plan(store, plan: dict[str, object]) -> bool:
    """Validate and apply safe structural operations without deleting source evidence."""
    action, ids = str(plan["action"]), [str(x) for x in plan.get("ids", [])]
    if action == "keep": return False
    if not ids or any(x not in store.progressions for x in ids): raise ValueError("unknown progression")
    if action == "archive":
        from progression_memory.models.progression import StorageTier
        store.progressions[ids[0]].storage_tier = StorageTier.COLD
        return True
    if action == "merge":
        if len(ids) != 2: raise ValueError("merge requires two progressions")
        left, right = (store.progressions[x] for x in ids)
        for episode_id in tuple(right.episode_ids): store.attach_episode(episode_id, left.id)
        for child in right.child_ids:
            if child not in left.child_ids: left.child_ids.append(child)
        right.storage_tier = __import__("progression_memory.models.progression", fromlist=["StorageTier"]).StorageTier.COLD
        return True
    # SPLIT/CREATE_CHILD need explicit evidence membership; never let generated text move evidence.
    episode_ids = [str(x) for x in plan.get("episode_ids", [])]
    if not episode_ids or not set(episode_ids) <= set(store.progressions[ids[0]].episode_ids):
        raise ValueError("structural child requires valid source episode_ids")
    from progression_memory.models.progression import Progression
    parent = store.progressions[ids[0]]
    child_id = str(plan.get("child_id") or f"{parent.id}:child:{len(parent.child_ids)}")
    if child_id in store.progressions: raise ValueError("child already exists")
    child = Progression(id=child_id, title=str(plan.get("child_identity") or "subprogression"),
                        scope=parent.scope, parent_ids=[parent.id])
    store.upsert_progression(child)
    parent.child_ids.append(child_id)
    for episode_id in episode_ids: store.attach_episode(episode_id, child_id)
    return True
