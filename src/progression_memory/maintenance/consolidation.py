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
