from __future__ import annotations

import argparse
import json
from pathlib import Path

from progression_memory.evaluation.dag_retrieval import (
    build_dag_two_stage_queries,
    build_dag_two_stage_v2_queries,
)
from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.hierarchical_retrieval import build_hierarchical_beam_queries
from progression_memory.evaluation.hierarchical_retrieval_v2 import build_hierarchical_beam_v2_queries
from progression_memory.evaluation.hierarchical_retrieval_v3 import build_hierarchical_beam_v3_queries
from progression_memory.evaluation.hierarchical_retrieval_v4 import build_hierarchical_beam_v4_queries
from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import RetrievalQuery, evaluate_retrieval
from progression_memory.evaluation.two_stage import build_two_stage_queries

SYSTEMS = (
    "append", "progression_only", "fixed_progression", "adaptive_progression",
    "adaptive_progression_v2", "two_stage_v2", "dag_two_stage_v1", "dag_two_stage_v2",
    "hierarchical_beam_v1", "hierarchical_beam_v2", "hierarchical_beam_v3", "hierarchical_beam_v4",
)


def _transform_queries(queries: list[RetrievalQuery], system: str) -> tuple[list[RetrievalQuery], dict[str, object]]:
    if system == "append":
        unique: dict[int, tuple] = {}
        for query in queries: unique[id(query.documents)] = query.documents
        raw_counts = [len(docs) for docs in unique.values()]
        raw_chars = [sum(len(doc.text) for doc in docs) for docs in unique.values()]
        return queries, {"unique_corpora": len(unique), "avg_raw_documents": sum(raw_counts) / max(1, len(raw_counts)), "avg_retrieval_units": sum(raw_counts) / max(1, len(raw_counts)), "avg_document_compression_ratio": 1.0, "avg_evidence_duplication_ratio": 1.0, "avg_raw_chars": sum(raw_chars) / max(1, len(raw_chars))}
    if system == "two_stage_v2": return build_two_stage_queries(queries)
    if system == "dag_two_stage_v1": return build_dag_two_stage_queries(queries)
    if system == "dag_two_stage_v2": return build_dag_two_stage_v2_queries(queries)
    if system == "hierarchical_beam_v1": return build_hierarchical_beam_queries(queries)
    if system == "hierarchical_beam_v2": return build_hierarchical_beam_v2_queries(queries)
    if system == "hierarchical_beam_v3": return build_hierarchical_beam_v3_queries(queries)
    if system == "hierarchical_beam_v4": return build_hierarchical_beam_v4_queries(queries)

    if system == "progression_only": mode = "flat"
    elif system == "fixed_progression": mode = "fixed"
    elif system == "adaptive_progression_v2": mode = "adaptive_v2"
    else: mode = "adaptive"
    cache: dict[int, tuple[tuple, dict[str, float | int]]] = {}
    transformed: list[RetrievalQuery] = []
    for query in queries:
        key = id(query.documents)
        if key not in cache: cache[key] = build_progression_documents(query.documents, mode=mode)
        documents, _ = cache[key]
        transformed.append(RetrievalQuery(id=query.id, text=query.text, evidence_ids=query.evidence_ids, documents=documents, category=query.category, answer=query.answer))
    stat_rows = [row for _, row in cache.values()]
    def avg(name: str) -> float: return sum(float(row[name]) for row in stat_rows) / max(1, len(stat_rows))
    result: dict[str, object] = {"unique_corpora": len(cache), "avg_raw_documents": avg("raw_document_count"), "avg_retrieval_units": avg("segment_count"), "avg_progressions": avg("progression_count"), "avg_document_compression_ratio": avg("document_compression_ratio"), "avg_evidence_duplication_ratio": avg("evidence_duplication_ratio"), "avg_raw_chars": avg("raw_chars")}
    if stat_rows and "avg_segment_size" in stat_rows[0]:
        result["avg_segment_size"] = avg("avg_segment_size"); result["avg_single_event_segment_fraction"] = avg("single_event_segment_fraction")
    return transformed, result


def run(dataset: str, data_path: str, cutoffs: list[int], system: str = "append") -> dict[str, object]:
    if dataset == "locomo": queries, base_granularity = load_locomo(data_path), "turn"
    elif dataset == "longmemeval": queries, base_granularity = load_longmemeval(data_path), "session"
    else: raise ValueError(f"unsupported dataset: {dataset}")
    if system not in SYSTEMS: raise ValueError(f"unsupported system: {system}")
    transformed, corpus_stats = _transform_queries(queries, system)
    result = evaluate_retrieval(transformed, cutoffs)
    source_level = {"two_stage_v2", "dag_two_stage_v1", "dag_two_stage_v2", "hierarchical_beam_v1", "hierarchical_beam_v2", "hierarchical_beam_v3", "hierarchical_beam_v4"}
    granularity = base_granularity if system == "append" else (f"{base_granularity}_after_structural_filter" if system in source_level else "progression_segment")
    suffix = "semantic" if system == "hierarchical_beam_v4" else ("hybrid" if system == "hierarchical_beam_v3" else "bm25")
    result.update({"dataset": dataset, "system": "append_bm25" if system == "append" else f"{system}_{suffix}", "granularity": granularity, "corpus_stats": corpus_stats})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--system", choices=SYSTEMS, default="append")
    parser.add_argument("--top-k", default="5,8,10")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.dataset, args.data_path, [int(x) for x in args.top_k.split(",") if x], args.system)
    payload = json.dumps(result, indent=2, sort_keys=True); print(payload)
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__": main()
