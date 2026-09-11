from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_store import ProgressionStore
from progression_memory.temporal.controller import LinearTemporalController, SegmentAction
from progression_memory.temporal.segmenter import (
    AdaptiveTemporalSegmenter,
    FixedWindowTemporalSegmenter,
)


@dataclass(frozen=True)
class SyntheticSequence:
    episodes: list[Episode]
    boundary_indices: list[int]


@dataclass(frozen=True)
class BenchmarkResult:
    system: str
    episode_count: int
    segment_count: int
    predicted_boundaries: list[int]
    true_boundaries: list[int]
    precision: float
    recall: float
    f1: float
    compression_ratio: float


def generate_phase_shift_sequence(seed: int = 7) -> SyntheticSequence:
    """Deterministic irregular state shifts that fixed calendar windows should miss."""
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    phase_starts = [0, 12, 29]
    phase_states = [0.10, 0.58, 0.92]
    phase_embeddings = [
        np.asarray([0.95, 0.20, 0.10]),
        np.asarray([0.72, 0.62, 0.12]),
        np.asarray([0.35, 0.88, 0.18]),
    ]

    episodes: list[Episode] = []
    for index in range(45):
        phase = max(i for i, start in enumerate(phase_starts) if start <= index)
        state = phase_states[phase] + float(rng.normal(0.0, 0.018))
        embedding = phase_embeddings[phase] + rng.normal(0.0, 0.015, size=3)

        is_boundary = index in phase_starts[1:]
        episodes.append(
            Episode(
                id=f"e{index:03d}",
                summary=f"Synthetic progression observation {index}",
                dialogue_time=base_time + timedelta(days=2 * index),
                semantic_time=base_time + timedelta(days=2 * index),
                embedding=embedding.tolist(),
                state_vector=[state],
                entities=["research", "project"],
                importance=0.7 if is_boundary else 0.35,
                novelty=0.9 if is_boundary else 0.15,
                contradiction=0.15 if is_boundary else 0.02,
            )
        )

    return SyntheticSequence(episodes=episodes, boundary_indices=phase_starts[1:])


def _boundary_metrics(
    predicted: list[int],
    expected: list[int],
    *,
    tolerance: int = 1,
) -> tuple[float, float, float]:
    unmatched = set(expected)
    true_positive = 0
    for prediction in predicted:
        eligible = [value for value in unmatched if abs(value - prediction) <= tolerance]
        if not eligible:
            continue
        match = min(eligible, key=lambda value: abs(value - prediction))
        unmatched.remove(match)
        true_positive += 1

    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 1.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def _run_system(
    sequence: SyntheticSequence,
    *,
    system: str,
    segmenter: AdaptiveTemporalSegmenter,
) -> BenchmarkResult:
    store = ProgressionStore()
    store.upsert_progression(
        Progression(
            id="synthetic",
            title="Synthetic progression",
            scope="research project",
            embedding=[0.95, 0.20, 0.10],
        )
    )

    predicted_boundaries: list[int] = []
    for index, episode in enumerate(sequence.episodes):
        store.ingest_episode(episode)
        action, _ = segmenter.process(store, episode.id, "synthetic")
        if index > 0 and action == SegmentAction.SPLIT:
            predicted_boundaries.append(index)

    assert store.validate_segment_partition("synthetic")

    precision, recall, f1 = _boundary_metrics(
        predicted_boundaries,
        sequence.boundary_indices,
    )
    segment_count = len(store.progressions["synthetic"].segment_ids)
    return BenchmarkResult(
        system=system,
        episode_count=len(sequence.episodes),
        segment_count=segment_count,
        predicted_boundaries=predicted_boundaries,
        true_boundaries=sequence.boundary_indices,
        precision=precision,
        recall=recall,
        f1=f1,
        compression_ratio=len(sequence.episodes) / segment_count,
    )


def run_synthetic_benchmark(seed: int = 7) -> list[BenchmarkResult]:
    sequence = generate_phase_shift_sequence(seed=seed)
    return [
        _run_system(
            sequence,
            system="fixed_14d",
            segmenter=FixedWindowTemporalSegmenter(window=timedelta(days=14)),
        ),
        _run_system(
            sequence,
            system="adaptive_linear",
            segmenter=AdaptiveTemporalSegmenter(LinearTemporalController()),
        ),
    ]


def main() -> None:
    payload = [asdict(result) for result in run_synthetic_benchmark()]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
