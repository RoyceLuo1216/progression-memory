"""Validated single-parent hierarchy traversal for V2 progressions."""
from __future__ import annotations

from collections.abc import Mapping

from progression_memory.v2.models import Progression


class HierarchyIndex:
    def __init__(self, progressions: Mapping[str, Progression]) -> None:
        self.nodes = progressions
        self._children: dict[str, list[str]] = {pid: [] for pid in progressions}
        for pid, node in progressions.items():
            if node.parent_id is None:
                continue
            parent = progressions.get(node.parent_id)
            if parent is None or parent.subject_id != node.subject_id:
                raise ValueError(f"invalid parent of {pid}")
            self._children[node.parent_id].append(pid)
        for children in self._children.values():
            children.sort()
        for pid in progressions:
            self.ancestors(pid)  # Detect any preexisting cycle.

    def children(self, pid: str) -> list[str]:
        return list(self._children[pid])

    def ancestors(self, pid: str) -> list[str]:
        if pid not in self.nodes:
            raise KeyError(pid)
        result: list[str] = []
        visited = {pid}
        parent = self.nodes[pid].parent_id
        while parent is not None:
            if parent in visited:
                raise ValueError("progression hierarchy contains a cycle")
            visited.add(parent)
            result.append(parent)
            parent = self.nodes[parent].parent_id
        return result

    def descendants(self, pid: str) -> list[str]:
        if pid not in self.nodes:
            raise KeyError(pid)
        result: list[str] = []
        pending = list(reversed(self.children(pid)))
        while pending:
            current = pending.pop()
            result.append(current)
            pending.extend(reversed(self.children(current)))
        return result

    def depth(self, pid: str) -> int:
        return len(self.ancestors(pid))
