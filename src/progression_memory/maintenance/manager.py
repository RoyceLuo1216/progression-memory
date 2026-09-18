from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.maintenance.state import MaintenanceAction, MaintenanceDecision, maintenance_decision
from progression_memory.routing.router import progression_route_score


@dataclass(frozen=True)
class WriteDecision:
    action: MaintenanceAction
    progression_id: str | None = None
    updated_state: str | None = None
    state_delta: str | None = None
    confidence: float = 1.0


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class ProgressionManager(Protocol):
    def decide(self, episode: Episode, candidates: list[Progression]) -> WriteDecision: ...


class HeuristicProgressionManager:
    """Fast deterministic ablation/fallback; not the proposed final writer."""

    def decide(self, episode: Episode, candidates: list[Progression]) -> WriteDecision:
        if not candidates:
            return WriteDecision(MaintenanceAction.CREATE)
        progression = max(candidates, key=lambda p: progression_route_score(episode, p))
        d: MaintenanceDecision = maintenance_decision(episode, progression)
        return WriteDecision(d.action, progression.id, confidence=d.compatibility)


class SLMProgressionManager:
    """Selective SLM writer: cheap weighted retrieval proposes candidates; SLM interprets change."""

    def __init__(self, model: LanguageModel, *, candidate_k: int = 6) -> None:
        self.model, self.candidate_k = model, candidate_k

    def decide(self, episode: Episode, candidates: list[Progression]) -> WriteDecision:
        scored = sorted(((progression_route_score(episode, p), p) for p in candidates),
                        key=lambda row: row[0], reverse=True)[:self.candidate_k]
        if not scored:
            return WriteDecision(MaintenanceAction.CREATE)
        ranked = [p for _, p in scored]
        rows = [{
            "id": p.id,
            "proposal_score": round(score, 4),
            "identity": p.high_level_summary or p.title,
            "current_state": p.current_state,
            "recent_delta": p.state_history[-1].state_delta if p.state_history else "",
        } for score, p in scored]
        prompt = (
            "You manage evolving memory progressions. Decide how NEW_EVIDENCE changes one candidate. "
            "Return JSON only: action (extend|transition|create_child|create), progression_id or null, "
            "updated_state (bounded current state, not a transcript), state_delta, confidence 0..1.\n"
            f"NEW_EVIDENCE={json.dumps(episode.summary)}\nCANDIDATES={json.dumps(rows)}"
        )
        try:
            value = json.loads(self.model.generate(prompt))
            action = MaintenanceAction(str(value["action"]).lower())
            pid = value.get("progression_id")
            if action != MaintenanceAction.CREATE and pid not in {p.id for p in ranked}:
                raise ValueError("writer selected an unproposed progression")
            return WriteDecision(action, pid, value.get("updated_state"), value.get("state_delta"),
                                 max(0.0, min(1.0, float(value.get("confidence", 1.0)))))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return HeuristicProgressionManager().decide(episode, ranked)
