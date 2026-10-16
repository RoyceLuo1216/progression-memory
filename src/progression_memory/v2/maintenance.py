"""Version-safe synchronous maintenance worker, schedulable by a future cron runner."""
from __future__ import annotations

from collections.abc import Callable

from progression_memory.v2.aggregation import aggregate_children, child_fingerprint
from progression_memory.v2.hierarchy import HierarchyIndex
from progression_memory.v2.models import Progression
from progression_memory.v2.state_engine import StateEngine

SummaryFn = Callable[[Progression, list[Progression]], str]


class MaintenanceWorker:
    def __init__(self, engine: StateEngine, summarizer: SummaryFn | None = None) -> None:
        self.engine = engine
        self.summarizer = summarizer

    def run_pending(self) -> list[str]:
        """Process dirty parents bottom-up; retain dirty entries on failure."""
        completed: list[str] = []
        index = HierarchyIndex(self.engine.progressions)
        targets = sorted(
            self.engine.dirty_progressions,
            key=lambda pid: (-index.depth(pid), pid),
        )
        for pid in targets:
            parent = self.engine.progressions[pid]
            children = [self.engine.progressions[cid] for cid in index.children(pid)]
            fingerprint = child_fingerprint(children)
            if fingerprint == parent.derived_fingerprint:
                self.engine.dirty_progressions.discard(pid)
                continue
            derived = aggregate_children(children)
            summary = self.summarizer(parent.model_copy(deep=True),
                                      [c.model_copy(deep=True) for c in children]) if self.summarizer else None
            # External summary functions can be slow or have side effects.
            # Detect an intervening child update and leave the item dirty.
            fresh_index = HierarchyIndex(self.engine.progressions)
            latest = [self.engine.progressions[cid] for cid in fresh_index.children(pid)]
            if child_fingerprint(latest) != fingerprint:
                continue
            current = self.engine.progressions[pid]
            current.derived_state = derived
            current.derived_fingerprint = fingerprint
            if summary is not None:
                current.derived_summary = summary
            self.engine.dirty_progressions.discard(pid)
            completed.append(pid)
        return completed
