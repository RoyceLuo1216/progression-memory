from datetime import datetime, timedelta, timezone

import numpy as np

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.controller import (
    HysteresisLinearTemporalController,
    LinearTemporalController,
    SegmentAction,
    SegmentFeatures,
)
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


def _features(**overrides: float | int) -> SegmentFeatures:
    values: dict[str, float | int] = {
        "semantic_distance": 0.30,
        "state_distance": 0.30,
        "novelty": 0.20,
        "contradiction": 0.0,
        "importance": 0.3,
        "delta_days": 1.0,
        "volatility": 0.1,
        "segment_size": 4,
        "information_weight": 1.0,
    }
    values.update(overrides)
    return SegmentFeatures(**values)  # type: ignore[arg-type]


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


def test_v2_preserves_v1_feature_dimension_and_high_confidence_probability() -> None:
    features = _features(information_weight=1.0, segment_size=4)
    v1 = LinearTemporalController()
    v2 = HysteresisLinearTemporalController()
    assert features.as_vector().shape == (8,)
    assert np.isclose(v1.split_probability(features), v2.split_probability(features))
    assert v1.decide(features) == v2.decide(features)


def test_v2_damps_low_confidence_short_event_change() -> None:
    controller = HysteresisLinearTemporalController()
    noisy_short_event = _features(
        semantic_distance=0.55,
        state_distance=0.55,
        novelty=0.45,
        segment_size=1,
        information_weight=0.25,
    )
    assert controller.decide(noisy_short_event) == SegmentAction.EXTEND


def test_v2_strong_contradiction_can_split_before_minimum_segment_size() -> None:
    controller = HysteresisLinearTemporalController()
    contradiction = _features(
        contradiction=0.90,
        segment_size=1,
        information_weight=0.25,
    )
    assert controller.decide(contradiction) == SegmentAction.SPLIT
