from datetime import datetime, timedelta, timezone

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.controller import LinearTemporalController, SegmentAction
from progression_memory.temporal.segmenter import AdaptiveTemporalSegmenter


def _episode(index: int, state: float) -> Episode:
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    return Episode(
        id=f"e{index}",
        summary=f"state={state}",
        dialogue_time=when,
        semantic_time=when,
        embedding=[state, 1.0 - state],
        state_vector=[state],
        novelty=0.1,
        importance=0.3,
    )


def test_adaptive_segmenter_splits_on_meaningful_state_change() -> None:
    store = ProgressionStore()
    store.upsert_progression(Progression(id="p", title="P", scope="test"))
    segmenter = AdaptiveTemporalSegmenter(LinearTemporalController())

    actions = []
    for index, state in enumerate([0.10, 0.12, 0.11, 0.80, 0.82]):
        episode = _episode(index, state)
        store.ingest_episode(episode)
        action, _ = segmenter.process(store, episode.id, "p")
        actions.append(action)

    assert actions[0] == SegmentAction.SPLIT
    assert actions[1:3] == [SegmentAction.EXTEND, SegmentAction.EXTEND]
    assert actions[3] == SegmentAction.SPLIT
    assert len(store.progressions["p"].segment_ids) == 2
    assert store.validate_segment_partition("p")
