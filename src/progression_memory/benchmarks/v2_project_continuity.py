"""Ten fictionalized long-horizon engineering-workflow stress cases.

Inspired by project types discussed during design (agent memory, capstone
benchmarking, commerce systems, trading infrastructure, recruiting pipelines).
All people, employers, dates, incidents, and results in this dataset are fictional.

Unlike the checkpoint smoke fixture, questions do not repeat evidence IDs.
Gold source IDs are for evaluation only. Oracle writer proposals intentionally
isolate retrieval from writer construction errors.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.v2.models import AttributeOperation, AttributePatch, Observation
from progression_memory.v2.writer_pipeline import ProposedUpdate, WriterProposal

PROJECTS = (
    ("agent-memory", "Long-running agent memory", "temporal evidence graphs", "identity collisions"),
    ("capstone", "Capstone benchmark platform", "isolated experiment runners", "invalid test evidence"),
    ("commerce", "Commerce prediction service", "delivery estimates", "retry storms"),
    ("marketdata", "Market data approvals", "authorization workflows", "stale approvals"),
    ("trading", "Electronic trading gateway", "event ordering", "replayed orders"),
    ("research", "Research evaluation pipeline", "dataset lineage", "answer leakage"),
    ("recruiting", "Recruiting operations tool", "application tracking", "duplicate contacts"),
    ("infrastructure", "Distributed event processing", "idempotency keys", "partial failures"),
    ("analytics", "Product analytics platform", "conversion attribution", "backfilled events"),
    ("knowledge", "Knowledge management agent", "stateful task plans", "obsolete commitments"),
)
# Each phase changes a decision's authority, not merely its wording.
STAGES = (
    ("proposal", "We proposed a lightweight cached index", "cached index", "compatibility required"),
    ("approval", "After review we approved the cached index", "cached index", "audit every update"),
    ("implementation", "We deployed the cached index to staging", "cached index", "legacy API retained"),
    ("incident", "A stress incident exposed inconsistent stale reads", "cached index", "do not trust prior timing results"),
    ("reversal", "We rejected the cached index as the primary design", "append-only log", "cache only ephemeral views"),
    ("migration", "We migrated to an append-only log with snapshot materialization", "append-only log", "preserve source lineage"),
    ("hold", "Deployment was paused by an unresolved dependency", "append-only log", "block production rollout"),
    ("revival", "Dependency cleared; we resumed the migration", "append-only log", "recheck validation"),
    ("verification", "A new validation suite passed without the earlier leak", "append-only log", "rerun held-out tests"),
    ("final", "We finalized the log-backed design and documented the remaining follow-up", "append-only log", "monitor latency regression"),
)
DISTRACTIONS = (
    "Someone suggested reverting to the original index but no decision was made.",
    "A question in the team chat asked whether the old approach was still supported.",
    "An unrelated retrospective quoted the original performance estimate as historical context.",
    "An external draft described the prior design, without approval to restore it.",
    "A developer mentioned the retired plan in a comment; this was not an authoritative update.",
)


def generate(*, cycles: int = 10, distractors_per_stage: int = 4) -> TraceFixture:
    if cycles != 10 or distractors_per_stage < 1:
        raise ValueError("ten defined stages and at least one distractor are required")
    observations: list[Observation] = []
    proposals: dict[str, WriterProposal] = {}
    questions: list[dict] = []
    anchor = datetime(2024, 1, 1, tzinfo=UTC)

    def add(subject: str, oid: str, day: int, message: str,
            update: ProposedUpdate | None = None, *, semantic_offset: int = 0) -> None:
        obs = Observation(
            id=oid, source_id=oid, subject_id=subject, text=message,
            recorded_at=anchor + timedelta(days=day),
            semantic_at=(anchor + timedelta(days=day + semantic_offset)
                         if semantic_offset else None),
        )
        observations.append(obs)
        proposals[oid] = WriterProposal(updates=((update,) if update is not None else ()))

    for p_idx, (slug, title, focus, problem) in enumerate(PROJECTS):
        subject = f"project:{slug}"
        pid = f"v2:{slug}-root:0"
        add(subject, f"{slug}-root", 0,
            f"Our team opened the {title} project, focused on {focus}; we need a durable decision log.",
            ProposedUpdate(action="create", name=title, reason="project founded",
                           patches=(AttributePatch(key="scope", operation=AttributeOperation.SET,
                                                   value=focus),)))
        for stage_idx, (phase, event, decision, constraint) in enumerate(STAGES):
            day = 18 + stage_idx * 42
            oid = f"{slug}-authoritative-{stage_idx}"
            add(subject, oid, day,
                f"Project update for {title}. Topic: {focus}; risk: {problem}. "
                f"{event}. Current authoritative architecture: {decision}. "
                f"Current operating restriction: {constraint}. "
                f"This supersedes earlier conflicting choices when relevant.",
                ProposedUpdate(
                    action="transition" if stage_idx else "append",
                    progression_id=pid, stage=phase, reason="confirmed project decision",
                    patches=(
                        AttributePatch(key="architecture", operation=AttributeOperation.SET,
                                       value=decision),
                        AttributePatch(key="restriction", operation=AttributeOperation.SET,
                                       value=constraint),
                    ),
                ))
            for j in range(distractors_per_stage):
                # All distractors stay in the same project and are genuinely
                # lexically competitive, but are explicitly non-authoritative.
                add(subject, f"{slug}-context-{stage_idx}-{j}",
                    day + 1 + j,
                    f"{title}, {focus}, {problem}: {DISTRACTIONS[j % len(DISTRACTIONS)]} "
                    f"Discussion about cached index, append-only log, and rollout policy. "
                    f"No approved change was recorded.")
            # Two task-oriented questions for every stage. They do not reveal
            # source IDs, but their current state should be evaluated AS OF
            # that stage rather than using final state for all questions.
            questions.extend((
                {"id": f"{slug}-state-{stage_idx}",
                 "subject_id": subject,
                 "question": f"At the {phase} milestone in the {title}, what architecture was authorized?",
                 "evidence_ids": [oid], "category": "historical_authority"},
                {"id": f"{slug}-constraint-{stage_idx}",
                 "subject_id": subject,
                 "question": f"For {focus} at the {phase} stage, what operating restriction governed the next implementation?",
                 "evidence_ids": [oid], "category": "actionable_constraint"},
            ))
        # Additional cross-period questions force evidence from distant months.
        questions.extend((
            {"id": f"{slug}-reversal-proof", "subject_id": subject,
             "question": f"How did the approved architecture for {title} change after the incident?",
             "evidence_ids": [f"{slug}-authoritative-1", f"{slug}-authoritative-4"],
             "category": "supersession"},
            {"id": f"{slug}-resumption-proof", "subject_id": subject,
             "question": f"Why could the {title} deployment resume after it was blocked?",
             "evidence_ids": [f"{slug}-authoritative-6", f"{slug}-authoritative-7"],
             "category": "long_range_dependency"},
        ))
    observations.sort(key=lambda obs: (obs.recorded_at, obs.id))
    return TraceFixture.model_validate({
        "observations": [o.model_dump(mode="json") for o in observations],
        "proposals": {k: v.model_dump(mode="json") for k, v in proposals.items()},
        "questions": questions,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--distractors", type=int, default=4)
    args = parser.parse_args()
    fixture = generate(distractors_per_stage=args.distractors)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"observations": len(fixture.observations),
                      "questions": len(fixture.questions),
                      "projects": len(PROJECTS),
                      "provenance": "fictional synthetic oracle"}, indent=2))


if __name__ == "__main__":
    main()
