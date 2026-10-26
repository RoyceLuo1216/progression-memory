"""Build reproducible 12-month, same-domain agent-memory project histories.

Synthetic development fixture. Every statement and writer delta is authored by
this generator; no gold benchmark answer is used to build the memory. This
is an oracle-construction stress test, NOT a real-model writer result.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.v2.models import AttributeOperation, AttributePatch, Observation
from progression_memory.v2.writer_pipeline import ProposedUpdate, WriterProposal

STREAMS = {
    "retrieval": ("Retrieval architecture", "retrieval"),
    "writer": ("Writer and identity matching", "writer"),
    "evaluation": ("Evaluation and benchmarks", "evaluation"),
    "maintenance": ("Hierarchy and maintenance", "maintenance"),
}
# This is one sustained agent-memory research program, not unrelated trivia.
# Milestones change concrete assumptions, targets, constraints, and approaches.
PHASES = (
    ("baseline", "initial prototype", "small lexical baseline", "keep source IDs"),
    ("investigation", "diagnostic analysis", "writer grouping failure", "freeze gold labels"),
    ("revision", "architecture redesign", "structured temporal state", "do not overwrite evidence"),
    ("implementation", "system integration", "hierarchical progressions", "preserve parent direct state"),
    ("verification", "regression validation", "version-aware maintenance", "reject stale jobs"),
    ("pilot", "benchmark pilot", "matched character budgets", "avoid annotation leakage"),
    ("correction", "late-arriving correction", "invalidated prior assumption", "replay event time"),
    ("refinement", "retrieval redesign", "lexical and dense fusion", "retain global rescue"),
    ("scale", "long-horizon scale-up", "archived progression revival", "retain original observations"),
    ("audit", "error analysis", "false progression attachment", "check subject and process"),
    ("retest", "controlled ablation", "hierarchy versus flat retrieval", "freeze writer trace"),
    ("release", "research reporting", "answer-level evaluation", "report uncertainty intervals"),
)
TOPIC_DETAILS = {
    "retrieval": (
        "top-8 evidence recall", "state-only materialization", "query-time BM25",
        "temporal segment index", "global rescue slots", "hybrid ranking",
        "historical as-of filtering", "RRF candidate fusion",
    ),
    "writer": (
        "first-person subject resolution", "candidate attachment", "state delta schema",
        "duplicate process prevention", "late event correction", "transition detection",
        "revive action", "unsupported patch audit",
    ),
    "evaluation": (
        "conversation-level split", "LoCoMo preparation", "LongMemEval preparation",
        "character-budget fairness", "per-question evidence recall", "paired confidence interval",
        "downstream reader consistency", "source citation precision",
    ),
    "maintenance": (
        "single-parent hierarchy", "derived child counts", "dirty ancestor invalidation",
        "bottom-up scheduling", "version fingerprint", "archive threshold",
        "source evidence retention", "semantic consolidation",
    ),
}


def generate(*, events_per_stream: int = 48) -> TraceFixture:
    if events_per_stream < 36:
        raise ValueError("minimum 36 events per workstream for long-horizon coverage")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    subject = "researcher-001"
    observations: list[Observation] = []
    proposals: dict[str, WriterProposal] = {}
    questions: list[dict] = []

    def add(day: int, oid: str, text: str, updates: tuple[ProposedUpdate, ...],
            *, event_day: int | None = None) -> None:
        observations.append(Observation(
            id=oid, source_id=oid, subject_id=subject, text=text,
            recorded_at=start + timedelta(days=day),
            semantic_at=(start + timedelta(days=event_day)
                         if event_day is not None else None),
        ))
        proposals[oid] = WriterProposal(updates=updates)

    add(0, "project-root",
        "I launched a year-long agent-memory research project to track evolving tasks and decisions.",
        (ProposedUpdate(action="create", name="Agent memory research",
                        reason="explicitly launched research project",
                        patches=(AttributePatch(
                            key="goal", operation=AttributeOperation.SET,
                            value="evaluate memory quality across long horizons",
                        ),)),))
    for n, (key, (title, _)) in enumerate(STREAMS.items(), 1):
        add(n, f"start-{key}",
            f"I opened the {title} workstream under agent memory research.",
            (ProposedUpdate(action="create", name=title,
                            parent_id="v2:project-root:0",
                            reason="explicit workstream creation",
                            patches=(AttributePatch(
                                key="phase", operation=AttributeOperation.SET,
                                value="started",
                            ),)),))
    # Interleave all four workstreams over twelve months. Repeated topics are
    # deliberately hard to separate lexically, while status changes accumulate.
    for i in range(events_per_stream):
        phase_number = min(11, i * 12 // events_per_stream)
        phase, activity, decision, constraint = PHASES[phase_number]
        for j, key in enumerate(STREAMS):
            oid = f"{key}-{i:03d}"
            day = 5 + i * 9 + j
            detail = TOPIC_DETAILS[key][i % len(TOPIC_DETAILS[key])]
            revision = i > 0 and i % 9 == 0
            # A delayed correction is ingested after a later event, but has
            # earlier valid time. The assertions still preserve author time.
            effective_day = day - 18 if revision else None
            descriptor = ("Correction to an earlier design note: " if revision else "")
            text = (
                f"{descriptor}Agent memory research / {key} workstream: "
                f"cycle {i + 1}, phase {phase}. "
                f"We reviewed {detail}, recorded {activity}, "
                f"decision: {decision}; constraint: {constraint}. "
                f"Checkpoint {oid} is the authoritative update for this cycle."
            )
            change = "transition" if i and phase_number != min(11, (i - 1) * 12 // events_per_stream) else "append"
            patch = (
                AttributePatch(key="phase", operation=AttributeOperation.SET, value=phase),
                AttributePatch(key="decision", operation=AttributeOperation.SET, value=decision),
                AttributePatch(key="constraint", operation=AttributeOperation.SET, value=constraint),
                AttributePatch(key="checkpoint", operation=AttributeOperation.SET, value=oid),
            )
            # Use distinct keys; this patch set is valid with the real StateEngine.
            add(day, oid, text, (ProposedUpdate(
                action=change,
                progression_id=f"v2:start-{key}:0",
                reason=f"explicit {key} workstream checkpoint",
                patches=patch,
                stage=phase,
            ),), event_day=effective_day)
            # Query every third change with some older historical probes.
            if i % 3 == 0:
                questions.append({
                    "id": f"q-{oid}", "subject_id": subject,
                    "question": (
                        f"For agent memory {key}, which checkpoint recorded "
                        f"{detail} during {phase} cycle {i + 1}?"
                    ),
                    "evidence_ids": [oid],
                    "category": "historical_checkpoint" if i < events_per_stream // 2
                                else "long_horizon_update",
                })
    # A final cross-workstream query needs multiple records many months apart.
    questions.append({
        "id": "q-cross-year-constraints", "subject_id": subject,
        "question": (
            "Which late research-cycle records specify retrieval design and "
            "writer identity-matching constraints?"
        ),
        "evidence_ids": [
            f"retrieval-{events_per_stream - 1:03d}",
            f"writer-{events_per_stream - 1:03d}",
        ],
        "category": "multi_progression",
    })
    order = sorted(range(len(observations)), key=lambda n: (
        observations[n].recorded_at, observations[n].id,
    ))
    ordered = [observations[index] for index in order]
    fixture = TraceFixture.model_validate({
        "observations": [obs.model_dump(mode="json") for obs in ordered],
        "proposals": {key: item.model_dump(mode="json") for key, item in proposals.items()},
        "questions": questions,
    })
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--events-per-stream", type=int, default=48)
    args = parser.parse_args()
    fixture = generate(events_per_stream=args.events_per_stream)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path), "observations": len(fixture.observations),
        "questions": len(fixture.questions), "workstreams": len(STREAMS),
        "label_source": "synthetic oracle",
    }, indent=2))


if __name__ == "__main__":
    main()
