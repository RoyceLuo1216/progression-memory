from datetime import datetime, timezone

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_store import ProgressionStore


def test_episode_ingestion_and_attachment_are_idempotent() -> None:
    store = ProgressionStore()
    progression = Progression(id="career", title="Career", scope="Career trajectory")
    store.upsert_progression(progression)

    episode = Episode(
        id="e1",
        summary="Started applying to new-grad roles.",
        dialogue_time=datetime.now(timezone.utc),
    )

    assert store.ingest_episode(episode) is True
    assert store.ingest_episode(episode) is False

    store.attach_episode("e1", "career")
    first_version = store.progressions["career"].evidence_version
    store.attach_episode("e1", "career")

    assert store.progressions["career"].episode_ids == ["e1"]
    assert store.progressions["career"].evidence_version == first_version
