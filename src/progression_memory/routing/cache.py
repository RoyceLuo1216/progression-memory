from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActiveProgressionCache:
    """Small recency-ordered cache used before global progression retrieval."""

    max_size: int = 8
    progression_ids: list[str] = field(default_factory=list)
    recent_entities: set[str] = field(default_factory=set)
    last_topic_embedding: list[float] | None = None

    def touch(
        self,
        progression_id: str,
        *,
        entities: set[str] | None = None,
        topic_embedding: list[float] | None = None,
    ) -> None:
        if progression_id in self.progression_ids:
            self.progression_ids.remove(progression_id)
        self.progression_ids.insert(0, progression_id)
        del self.progression_ids[self.max_size :]

        if entities:
            self.recent_entities.update(entities)
        if topic_embedding is not None:
            self.last_topic_embedding = topic_embedding

    def candidates(self) -> tuple[str, ...]:
        return tuple(self.progression_ids)

    def clear(self) -> None:
        self.progression_ids.clear()
        self.recent_entities.clear()
        self.last_topic_embedding = None
