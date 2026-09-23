"""Source-level retrieval comparison of temporal, semantic and progression grouping.

Every grouped system ranks structural units with BM25, expands a bounded set of
units to their raw sources, then ranks those same raw sources with BM25. Only
raw source documents count toward evidence recall and reader character budgets.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit
from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import (
    BM25Index, RetrievalDocument, RetrievalQuery, evaluate_retrieval,
    evaluate_retrieval_by_char_budget,
)
from progression_memory.evaluation.semantic import SentenceTransformerEmbedder, TextEmbedder
from progression_memory.evaluation.temporal_semantic_baselines import build_baseline_documents

SYSTEMS = ("append", "temporal_only", "semantic_only", "temporal_semantic", "progression_only")
BUDGETS = {"locomo": (1200, 2400, 4800), "longmemeval": (100000, 150000, 200000)}


def _groups(
    documents: tuple[RetrievalDocument, ...], system: str, embedder: TextEmbedder | None,
    temporal_window_days: float, semantic_threshold: float,
) -> tuple[tuple[RetrievalDocument, ...], dict[str, object]]:
    if system == "progression_only":
        units, stats = build_progression_documents(documents, mode="flat")
    else:
        units, stats = build_baseline_documents(
            documents, system, embedder=embedder, temporal_window_days=temporal_window_days,
            semantic_threshold=semantic_threshold,
        )
    return units, stats


def evaluate_system(
    queries: list[RetrievalQuery], system: str, *, cutoffs: tuple[int, ...],
    budgets: tuple[int, ...], embedder: TextEmbedder | None = None,
    temporal_window_days: float = 14.0, semantic_threshold: float = 0.58,
) -> dict[str, object]:
    """Compare raw evidence exposed to the reader under shared source budgets."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    transformed: list[RetrievalQuery] = []
    cache: dict[int, tuple[tuple[RetrievalDocument, ...], BM25Index | None, dict[str, object]]] = {}
    coverage: list[float] = []
    candidate_counts: list[int] = []
    selected_counts: list[int] = []
    for query in queries:
        key = id(query.documents)
        if key not in cache:
            if system == "append":
                cache[key] = ((), None, {"raw_document_count": len(query.documents)})
            else:
                units, stats = _groups(
                    query.documents, system, embedder, temporal_window_days, semantic_threshold,
                )
                cache[key] = (units, BM25Index(units), stats)
        units, group_index, _ = cache[key]
        if system == "append":
            transformed.append(query)
            candidate_counts.append(len(query.documents))
            if query.evidence_ids:
                coverage.append(len(query.evidence_ids & {
                    source_id for doc in query.documents
                    for source_id in (doc.evidence_ids or {doc.id})
                }) / len(query.evidence_ids))
            continue
        limit = _adaptive_candidate_limit(len(units))
        selected = [doc for _, doc in group_index.score(query.text)[:limit]]
        source_by_id = {source.id: source for source in query.documents}
        candidates: OrderedDict[str, RetrievalDocument] = OrderedDict()
        for unit in selected:
            for evidence_id in unit.evidence_ids:
                if evidence_id in source_by_id:
                    candidates.setdefault(evidence_id, source_by_id[evidence_id])
        # Empty selections must not silently fall back to the full raw corpus.
        ranked_sources = tuple(
            doc for _, doc in BM25Index(tuple(candidates.values())).score(query.text)
        )
        transformed.append(RetrievalQuery(
            id=query.id, text=query.text, evidence_ids=query.evidence_ids,
            documents=ranked_sources, category=query.category, answer=query.answer,
        ))
        selected_counts.append(len(selected))
        candidate_counts.append(len(candidates))
        if query.evidence_ids:
            coverage.append(len(query.evidence_ids & candidates.keys()) / len(query.evidence_ids))

    unit_counts = [len(row[0]) for row in cache.values()]
    result: dict[str, object] = {
        "system": system,
        "protocol": "structural_bm25_to_source_bm25" if system != "append" else "raw_bm25",
        "source_top_k": evaluate_retrieval(transformed, cutoffs),
        "source_char_budget": evaluate_retrieval_by_char_budget(transformed, budgets),
        "corpus_stats": {
            "unique_corpora": len(cache),
            "avg_structural_units": sum(unit_counts) / len(unit_counts) if unit_counts else 0.0,
            "avg_stage1_selected_units": sum(selected_counts) / len(selected_counts) if selected_counts else 0.0,
            "avg_candidate_source_documents": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
            "avg_stage1_evidence_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
            "avg_unit_source_count": 1.0 if system == "append" else (
                sum(sum(len(unit.evidence_ids) for unit in row[0]) for row in cache.values())
                / max(1, sum(unit_counts))
            ),
        },
    }
    if system in {"semantic_only", "temporal_semantic"}:
        result["semantic_backend"] = getattr(embedder, "model_name", "hashed")
    if system in {"temporal_only", "temporal_semantic"}:
        result["temporal_window_days"] = temporal_window_days
    if system in {"semantic_only", "temporal_semantic"}:
        result["semantic_threshold"] = semantic_threshold
    return result


def run(
    dataset: str, data_path: str, *, systems: tuple[str, ...] = SYSTEMS,
    cutoffs: tuple[int, ...] = (5, 8, 10), semantic_model: str | None = None,
    temporal_window_days: float = 14.0, semantic_threshold: float = 0.58,
) -> dict[str, object]:
    if dataset not in BUDGETS:
        raise ValueError(f"unknown dataset: {dataset}")
    queries = load_locomo(data_path) if dataset == "locomo" else load_longmemeval(data_path)
    embedder = SentenceTransformerEmbedder(semantic_model) if semantic_model else None
    return {
        "dataset": dataset,
        "retrieval_level": "raw_source",
        "results": {
            system: evaluate_system(
                queries, system, cutoffs=cutoffs, budgets=BUDGETS[dataset],
                embedder=embedder, temporal_window_days=temporal_window_days,
                semantic_threshold=semantic_threshold,
            ) for system in systems
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--temporal-window-days", type=float, default=14.0)
    parser.add_argument("--semantic-threshold", type=float, default=0.58)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    systems = tuple(value.strip() for value in args.systems.split(",") if value.strip())
    result = run(
        args.dataset, args.data_path, systems=systems, semantic_model=args.semantic_model,
        temporal_window_days=args.temporal_window_days, semantic_threshold=args.semantic_threshold,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for system, row in result["results"].items():
        top8 = row["source_top_k"]["cutoffs"]["8"]
        budget = BUDGETS[args.dataset][1]
        at_budget = row["source_char_budget"]["budgets"][str(budget)]
        print(f"{system}: source recall@8={top8['macro_evidence_recall']:.4f}, "
              f"recall@{budget} chars={at_budget['macro_evidence_recall']:.4f}, "
              f"stage1 coverage={row['corpus_stats']['avg_stage1_evidence_coverage']:.4f}")


if __name__ == "__main__":
    main()
