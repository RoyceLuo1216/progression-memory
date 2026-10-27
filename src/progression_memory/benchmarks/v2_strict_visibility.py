"""Strict source-visible and historical-state retrieval stress comparison.

Credit is awarded only when the selected context contains source text or an
explicitly reconstructed snapshot with its value. A progression's historical
provenance set alone cannot earn credit.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from progression_memory.benchmarks.v2_project_continuity import generate
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.hierarchical_retrieval import retrieve, select_budget
from progression_memory.v2.read_path import build_documents


def evaluate(*, distractors: int = 4, budgets=(400, 800, 1600, 3200)):
    fixture = generate(distractors_per_stage=distractors)
    engine = prepare(fixture)
    by_subject = defaultdict(list)
    by_id = {}
    for observation in fixture.observations:
        by_subject[observation.subject_id].append(observation)
        by_id[observation.id] = observation
    # Evaluate actual source-text visibility independently from metadata provenance.
    systems = ("rag", "v2_flat", "v2_hierarchical", "v2_historical")
    results = {}
    for system in systems:
        results[system] = {}
        for budget in budgets:
            rows = []
            for question in fixture.questions:
                sid = question.subject_id
                if system == "rag":
                    corpus = [
                        RetrievalDocument(id=f"evidence:{o.id}", text=o.text,
                                          evidence_ids=frozenset({o.id}))
                        for o in by_subject[sid]
                    ]
                    ranked = [d for _, d in BM25Index(corpus).score(question.question)]
                elif system == "v2_historical":
                    # Historical source units and reconstructed as-of snapshots.
                    # The timestamp is supplied by benchmark gold labels, not
                    # inferred by the retriever; this is an oracle time-router.
                    gold_dates = [by_id[eid].effective_at for eid in question.evidence_ids]
                    horizon = max(gold_dates)
                    corpus = list(build_documents(
                        engine, subject_id=sid, view="full",
                        valid_at=horizon,
                        known_at=max(o.recorded_at for o in by_subject[sid]),
                    ))
                    ranked = [d for _, d in BM25Index(corpus).score(question.question)]
                else:
                    ranked = retrieve(
                        engine, question.question, subject_id=sid,
                        mode=("flat_bm25" if system == "v2_flat"
                              else "hierarchical_bm25"),
                        k=len(engine.observations) + len(engine.progressions),
                    )
                selected = select_budget(ranked, budget)
                # Only original source documents count. A state summary's
                # evidence_ids cannot confer source credit without source text.
                visible = set()
                for doc in selected:
                    if doc.id.startswith(("evidence:", "v2:evidence:")):
                        visible.update(doc.evidence_ids)
                gold = set(question.evidence_ids)
                recall = len(visible & gold) / len(gold) if gold else 0
                rows.append({"id": question.id, "category": question.category,
                             "recall": recall, "fully_supported": gold <= visible})
            results[system][str(budget)] = {
                "visible_evidence_recall": sum(x["recall"] for x in rows) / len(rows),
                "full_support_rate": sum(x["fully_supported"] for x in rows) / len(rows),
                "questions": len(rows),
                "per_query": rows,
            }
    return {
        "protocol": "v2-strict-visible-evidence-v1",
        "dataset": {"observations": len(fixture.observations),
                    "questions": len(fixture.questions),
                    "projects": 10, "distractors_per_stage": distractors},
        "systems": results,
        "limitations": [
            "Original-source visibility is a conservative proxy, not generated QA accuracy.",
            "State snapshots may entail answers but are intentionally NOT credited as original evidence.",
            "Historical variant receives oracle question time and is not deployable temporal routing.",
            "Scripted oracle writer, not measured model-driven construction.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(distractors=args.distractors)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": result["dataset"], "recall_at_1600": {
        key: round(value["1600"]["visible_evidence_recall"], 4)
        for key, value in result["systems"].items(),
    }}, indent=2))


if __name__ == "__main__":
    main()
