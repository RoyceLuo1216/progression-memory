from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from progression_memory.benchmarks.real_retrieval import SYSTEMS, _transform_queries
from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.retrieval import evaluate_retrieval_by_char_budget
from progression_memory.evaluation.statistics import compare_query_outcomes


def _load_queries(dataset: str, data_path: str):
    if dataset == "locomo":
        return load_locomo(data_path)
    if dataset == "longmemeval":
        return load_longmemeval(data_path)
    raise ValueError(f"unsupported dataset: {dataset}")


def run(
    dataset: str,
    data_path: str,
    budgets: Iterable[int],
    *,
    systems: Iterable[str] = SYSTEMS,
    bootstrap_samples: int = 2000,
    seed: int = 17,
) -> dict[str, object]:
    """Compare retrieval systems under identical strict character budgets."""

    queries = _load_queries(dataset, data_path)
    budget_values = sorted({int(value) for value in budgets if int(value) >= 0})
    system_names = tuple(systems)
    unknown = [name for name in system_names if name not in SYSTEMS]
    if unknown:
        raise ValueError(f"unsupported systems: {unknown}")

    corpus_stats: dict[str, object] = {}
    metrics: dict[str, dict[str, dict[str, object]]] = {str(b): {} for b in budget_values}
    rows: dict[str, dict[str, list[dict[str, object]]]] = {}

    for system in system_names:
        transformed, stats = _transform_queries(queries, system)
        evaluated = evaluate_retrieval_by_char_budget(
            transformed,
            budget_values,
            include_per_query=True,
        )
        corpus_stats[system] = stats
        rows[system] = {}
        for budget in budget_values:
            budget_key = str(budget)
            payload = dict(evaluated["budgets"][budget_key])
            rows[system][budget_key] = list(payload.pop("per_query"))
            metrics[budget_key][system] = payload

    comparisons: dict[str, dict[str, object]] = {str(b): {} for b in budget_values}
    if "append" in system_names:
        for budget_index, budget in enumerate(budget_values):
            budget_key = str(budget)
            baseline_rows = rows["append"][budget_key]
            for system_index, system in enumerate(system_names):
                if system == "append":
                    continue
                comparisons[budget_key][system] = compare_query_outcomes(
                    baseline_rows,
                    rows[system][budget_key],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + budget_index * 100 + system_index,
                )

    return {
        "dataset": dataset,
        "budget_policy": "strict_ranked_prefix_whole_units",
        "bootstrap_samples": bootstrap_samples,
        "eligible_query_count": sum(bool(query.evidence_ids) for query in queries),
        "excluded_without_evidence": sum(not query.evidence_ids for query in queries),
        "corpus_stats": corpus_stats,
        "budgets": {
            key: {
                "systems": metrics[key],
                "comparisons_vs_append": comparisons[key],
            }
            for key in metrics
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--char-budgets", required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output")
    args = parser.parse_args()

    budgets = [int(value) for value in args.char_budgets.split(",") if value]
    systems = [value for value in args.systems.split(",") if value]
    result = run(
        args.dataset,
        args.data_path,
        budgets,
        systems=systems,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
