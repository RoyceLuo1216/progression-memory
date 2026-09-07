from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActiveProgressionCache:
    progression_ids: list[str] = field(default_factory=list)
    recent_entities: set[str] = field(default_factory=set)
    last_topic_embedding: list[float] | None = None

    def clear(self) -> None:
        self.progression_ids.clear()
        self.recent_entities.clear()
        self.last_topic_embedding = None
