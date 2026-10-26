"""Reproducible paired V2-versus-RAG evaluation from an explicit writer trace.

Input is a JSON fixture with observations, writer proposals and held-out questions.
No gold answers are passed to the writer or used in corpus construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from progression_memory.evaluation.retrieval import (
    RetrievalDocument,
    RetrievalQuery,
    evaluate_retrieval_by_char_budget,
)
from progression_memory.v2.models import Observation
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import ScriptedWriter, WritePipeline, WriterProposal


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    subject_id: str
    question: str
    evidence_ids: frozenset[str]
    category: str = "unknown"


class TraceFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation]
    proposals: dict[str, WriterProposal]
    questions: list[GoldQuestion]


def run(fixture: TraceFixture, budgets: list[int]) -> dict[str, Any]:
    engine = StateEngine()
    pipeline = WritePipeline(engine, ScriptedWriter(fixture.proposals))
    evidence: dict[str, list[RetrievalDocument]] = defaultdict(list)
    for observation in fixture.observations:
        pipeline.process(observation)
        evidence[observation.subject_id].append(RetrievalDocument(
            id=observation.id,
            text=observation.text,
            timestamp=observation.effective_at.isoformat(),
            evidence_ids=frozenset({observation.id}),
        ))
    # Derived parent fields are refreshed only after all writes, consistently.
    from progression_memory.v2.maintenance import MaintenanceWorker

    MaintenanceWorker(engine).run_pending()
    systems = ("rag", "v2_state", "v2_full")
    results: dict[str, Any] = {}
    for system in systems:
        queries: list[RetrievalQuery] = []
        cache: dict[str, tuple[RetrievalDocument, ...]] = {}
        for question in fixture.questions:
            sid = question.subject_id
            if sid not in cache:
                if system == "rag":
                    cache[sid] = tuple(evidence[sid])
                else:
                    cache[sid] = build_documents(
                        engine, subject_id=sid,
                        view="state" if system == "v2_state" else "full",
                    )
            queries.append(RetrievalQuery(
                id=question.id, text=question.question,
                category=question.category, answer="",
                evidence_ids=question.evidence_ids,
                documents=cache[sid],
            ))
        results[system] = evaluate_retrieval_by_char_budget(
            queries, budgets, include_per_query=True,
        )
    return {"protocol": "v2-paired-evidence-v1",
            "systems": results,
            "observations": len(fixture.observations),
            "questions": len(fixture.questions),
            "limitations": [
                "Scripted/recorded writer proposals, not live writer-quality evaluation",
                "Evidence recall is not answer accuracy",
                "Writer construction cost excluded; record separately",
                "Whole-document character budgets, no partial-unit credit",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--budgets", default="400,800,1600,3200")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.fixture)
    raw = path.read_bytes()
    fixture = TraceFixture.model_validate_json(raw)
    budgets = sorted({int(v) for v in args.budgets.split(",")})
    if not budgets or any(v < 0 for v in budgets):
        parser.error("budgets must be nonnegative integers")
    report = run(fixture, budgets)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    report["budgets"] = budgets
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(dest), "input_sha256": report["input_sha256"],
        "systems": list(report["systems"]),
    }, indent=2))


if __name__ == "__main__":
    main()
