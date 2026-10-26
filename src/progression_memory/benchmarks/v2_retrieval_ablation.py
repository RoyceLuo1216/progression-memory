"""Matched-budget V2 retrieval ablation runner.

A configured sentence-transformers encoder is required for dense variants.
Unavailable dense variants are reported as skipped, never as synthetic results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.v2.hierarchical_retrieval import MODES, retrieve, select_budget
from progression_memory.v2.maintenance import MaintenanceWorker
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import ScriptedWriter, WritePipeline


def prepare(fixture: TraceFixture) -> StateEngine:
    engine = StateEngine()
    pipeline = WritePipeline(engine, ScriptedWriter(fixture.proposals))
    for obs in fixture.observations:
        pipeline.process(obs)
    MaintenanceWorker(engine).run_pending()
    return engine


def evaluate(
    fixture: TraceFixture, budgets: list[int], *,
    embedder=None, modes: tuple[str, ...] = MODES,
) -> dict:
    engine = prepare(fixture)
    output = {}
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode}")
        if "hybrid" in mode and embedder is None:
            output[mode] = {"status": "skipped", "reason": "embedding model not configured"}
            continue
        outcomes = {}
        for budget in budgets:
            rows = []
            for q in fixture.questions:
                # Regular queries are evaluated on the full frozen corpus.
                # Temporal variant uses a frozen observation horizon; real as-of
                # question timestamps must be added for benchmark-grade results.
                if mode.endswith("temporal"):
                    horizon = max(
                        (o.effective_at for o in fixture.observations
                         if o.subject_id == q.subject_id),
                        default=None,
                    )
                    if horizon is None:
                        selected = []
                    else:
                        ranked = retrieve(
                            engine, q.question, mode=mode, subject_id=q.subject_id,
                            embedder=embedder, k=len(engine.observations) + len(engine.progressions),
                            valid_at=horizon,
                        )
                        selected = select_budget(ranked, budget)
                else:
                    ranked = retrieve(
                        engine, q.question, mode=mode, subject_id=q.subject_id,
                        embedder=embedder, k=len(engine.observations) + len(engine.progressions),
                    )
                    selected = select_budget(ranked, budget)
                relevant = set(q.evidence_ids)
                matched = set().union(*(set(d.evidence_ids) for d in selected)) & relevant if selected else set()
                first = next((i for i, d in enumerate(selected, 1)
                              if set(d.evidence_ids) & relevant), None)
                rows.append({
                    "id": q.id, "category": q.category,
                    "recall": len(matched) / len(relevant) if relevant else None,
                    "mrr": 1 / first if first else 0.0,
                    "retrieved_chars": sum(len(d.text) for d in selected),
                    "retrieved_ids": [d.id for d in selected],
                })
            eligible = [r for r in rows if r["recall"] is not None]
            outcomes[str(budget)] = {
                "query_count": len(rows),
                "eligible_query_count": len(eligible),
                "macro_evidence_recall": (
                    sum(r["recall"] for r in eligible) / len(eligible) if eligible else 0.0
                ),
                "mrr": sum(r["mrr"] for r in eligible) / len(eligible) if eligible else 0.0,
                "avg_retrieved_chars": (
                    sum(r["retrieved_chars"] for r in eligible) / len(eligible)
                    if eligible else 0.0
                ),
                "per_query": rows,
            }
        output[mode] = {"status": "completed", "budgets": outcomes}
    return {"protocol": "v2-hierarchical-ablation-v1",
            "systems": output, "question_count": len(fixture.questions),
            "limitations": [
                "Writer is frozen scripted/recorded, not evaluated here.",
                "Evidence recall is not answer correctness.",
                "Temporal variant currently uses corpus end, not question-specific as-of time.",
                "Flat/hierarchical variants use separate ranking policies; all use identical budgets.",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="400,800,1600,3200")
    parser.add_argument("--embedding-model", help="SentenceTransformer model ID; optional")
    args = parser.parse_args()
    raw = Path(args.fixture).read_bytes()
    fixture = TraceFixture.model_validate_json(raw)
    budgets = sorted({int(v) for v in args.budgets.split(",")})
    if not budgets or budgets[0] < 0:
        parser.error("budgets must be nonnegative")
    embedder = None
    if args.embedding_model:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model)
        embedder = lambda text: model.encode(text).tolist()
    result = evaluate(fixture, budgets, embedder=embedder)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["embedding_model"] = args.embedding_model
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(dest), "systems": {
        k: v["status"] for k, v in result["systems"].items(),
    }}, indent=2))


if __name__ == "__main__":
    main()
