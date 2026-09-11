from __future__ import annotations

from collections import deque
from enum import StrEnum

from pydantic import BaseModel, Field

from progression_memory.models.progression import Progression


class GraphNodeKind(StrEnum):
    PROGRESSION = "progression"
    SEGMENT = "segment"


class GraphRelation(StrEnum):
    CONTAINS = "contains"
    PARENT_CHILD = "parent_child"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class GraphNode(BaseModel):
    id: str
    kind: GraphNodeKind
    progression_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: GraphRelation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ProgressionGraph:
    """Evidence-preserving directed acyclic graph over progression memory structure.

    The graph supports multi-parent nodes. Progression-to-progression PARENT_CHILD
    edges are mirrored into the existing ``Progression.parent_ids`` / ``child_ids``
    fields so the graph can be introduced without replacing the current data model.
    Segment nodes may be attached beneath progressions with CONTAINS edges.
    """

    def __init__(self, progressions: dict[str, Progression] | None = None) -> None:
        self.progressions = progressions if progressions is not None else {}
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[tuple[str, str, GraphRelation], GraphEdge] = {}
        for progression in self.progressions.values():
            self.add_progression(progression)

    def add_progression(self, progression: Progression) -> GraphNode:
        self.progressions[progression.id] = progression
        node = GraphNode(id=progression.id, kind=GraphNodeKind.PROGRESSION)
        self.nodes[progression.id] = node
        return node

    def add_segment(self, segment_id: str, progression_id: str) -> GraphNode:
        if progression_id not in self.nodes:
            raise KeyError(f"unknown progression {progression_id!r}")
        node = GraphNode(
            id=segment_id,
            kind=GraphNodeKind.SEGMENT,
            progression_id=progression_id,
        )
        self.nodes[segment_id] = node
        self.add_edge(progression_id, segment_id, GraphRelation.CONTAINS)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: GraphRelation = GraphRelation.PARENT_CHILD,
        *,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> GraphEdge:
        if source_id not in self.nodes or target_id not in self.nodes:
            missing = [node_id for node_id in (source_id, target_id) if node_id not in self.nodes]
            raise KeyError(f"unknown graph nodes: {missing}")
        if source_id == target_id:
            raise ValueError("self edges are not allowed in a progression DAG")
        key = (source_id, target_id, relation)
        existing = self.edges.get(key)
        if existing is not None:
            return existing
        if self._reachable(target_id, source_id):
            raise ValueError(f"edge {source_id!r}->{target_id!r} would create a cycle")

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.edges[key] = edge

        if relation == GraphRelation.PARENT_CHILD:
            source = self.progressions.get(source_id)
            target = self.progressions.get(target_id)
            if source is not None and target is not None:
                if target_id not in source.child_ids:
                    source.child_ids.append(target_id)
                if source_id not in target.parent_ids:
                    target.parent_ids.append(source_id)
        return edge

    def link_progressions(
        self,
        parent_id: str,
        child_id: str,
        *,
        confidence: float = 1.0,
    ) -> GraphEdge:
        for node_id in (parent_id, child_id):
            node = self.nodes.get(node_id)
            if node is None or node.kind != GraphNodeKind.PROGRESSION:
                raise KeyError(f"unknown progression node {node_id!r}")
        return self.add_edge(
            parent_id,
            child_id,
            GraphRelation.PARENT_CHILD,
            confidence=confidence,
        )

    def children(
        self,
        node_id: str,
        relation: GraphRelation | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            edge.target_id
            for edge in self.edges.values()
            if edge.source_id == node_id and (relation is None or edge.relation == relation)
        )

    def parents(
        self,
        node_id: str,
        relation: GraphRelation | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            edge.source_id
            for edge in self.edges.values()
            if edge.target_id == node_id and (relation is None or edge.relation == relation)
        )

    def descendants(self, node_id: str) -> tuple[str, ...]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        seen: set[str] = set()
        queue = deque(self.children(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.children(current))
        return tuple(seen)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        seen: set[str] = set()
        queue = deque(self.parents(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.parents(current))
        return tuple(seen)

    def topological_order(self) -> tuple[str, ...]:
        indegree = {node_id: 0 for node_id in self.nodes}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            adjacency[edge.source_id].append(edge.target_id)
            indegree[edge.target_id] += 1
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child in sorted(adjacency[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("progression graph contains a cycle")
        return tuple(order)

    def validate(self) -> bool:
        try:
            self.topological_order()
        except ValueError:
            return False
        return all(
            edge.source_id in self.nodes and edge.target_id in self.nodes
            for edge in self.edges.values()
        )

    def _reachable(self, source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return True
        seen: set[str] = set()
        queue = deque([source_id])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for child in self.children(current):
                if child == target_id:
                    return True
                queue.append(child)
        return False
