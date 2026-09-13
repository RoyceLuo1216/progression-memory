from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from progression_memory.benchmarks.real_retrieval import _transform_queries
from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.qa import (
    ExtractiveBM25Reader,
    QARunConfig,
    evaluate_qa,
    write_longmemeval_hypotheses,
)
from progression_memory.evaluation.retrieval import RetrievalQuery

QA_SYSTEMS = (
    "append",
    "adaptive_progression_v2",
    "two_stage_v2",
    "dag_two_stage_v1",
    "dag_two_stage_v2",
)


def _load(dataset: str, path: str) -> list[RetrievalQuery]:
    if dataset == "locomo":
        return load_locomo(path)
    if dataset == "longmemeval":
        return load_longmemeval(path)
    raise ValueError(dataset)


def _stable_sample(
    queries: list[RetrievalQuery],
    sample_size: int | None,
) -> list[RetrievalQuery]:
    if sample_size is None or sample_size >= len(queries):
        return queries
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    by_category: defaultdict[str, list[RetrievalQuery]] = defaultdict(list)
    for query in queries:
        by_category[query.category].append(query)
    categories = sorted(by_category)
    for rows in by_category.values():
        rows.sort(key=lambda query: hashlib.sha256(query.id.encode("utf-8")).hexdigest())

    selected: list[RetrievalQuery] = []
    cursor = {category: 0 for category in categories}
    while len(selected) < sample_size:
        added = False
        for category in categories:
            index = cursor[category]
            rows = by_category[category]
            if index < len(rows):
                selected.append(rows[index])
                cursor[category] += 1
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
    return selected


def run(
    dataset: str,
    data_path: str,
    *,
    systems: tuple[str, ...] = QA_SYSTEMS,
    top_k: int | None = None,
    char_budget: int | None = None,
    sample_size: int | None = None,
    hypotheses_dir: str | None = None,
) -> dict[str, object]:
    unknown = [system for system in systems if system not in QA_SYSTEMS]
    if unknown:
        raise ValueError(f"unsupported QA systems: {unknown}")

    base_queries = _stable_sample(_load(dataset, data_path), sample_size)
    reader = ExtractiveBM25Reader()
    config = QARunConfig(top_k=top_k, char_budget=char_budget)

    results: dict[str, object] = {}
    for system in systems:
        transformed, corpus_stats = _transform_queries(base_queries, system)
        qa = evaluate_qa(transformed, reader, config)
        qa["system"] = system
        qa["corpus_stats"] = corpus_stats
        results[system] = qa
        if hypotheses_dir and dataset == "longmemeval":
            write_longmemeval_hypotheses(
                qa,
                Path(hypotheses_dir) / f"{system}.jsonl",
            )

    return {
        "dataset": dataset,
        "reader": reader.name,
        "sample_size": len(base_queries),
        "systems": list(systems),
        "context_policy": {"top_k": top_k, "char_budget": char_budget},
        "results": results,
        "note": (
            "Deterministic extractive-reader ablation. Not directly comparable to "
            "published LLM-reader / LLM-judge QA accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--systems", default=",".join(QA_SYSTEMS))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--char-budget", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--hypotheses-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run(
        args.dataset,
        args.data_path,
        systems=tuple(value for value in args.systems.split(",") if value),
        top_k=args.top_k,
        char_budget=args.char_budget,
        sample_size=args.sample_size,
        hypotheses_dir=args.hypotheses_dir,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
