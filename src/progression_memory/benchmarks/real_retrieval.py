from __future__ import annotations

import argparse
import json
from pathlib import Path

from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.retrieval import evaluate_retrieval


def run(dataset: str, data_path: str, cutoffs: list[int]) -> dict[str, object]:
    if dataset == "locomo":
        queries = load_locomo(data_path)
    elif dataset == "longmemeval":
        queries = load_longmemeval(data_path)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    result = evaluate_retrieval(queries, cutoffs)
    result.update({"dataset": dataset, "system": "append_bm25", "granularity": "turn" if dataset == "locomo" else "session"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--top-k", default="5,8,10")
    parser.add_argument("--output")
    args = parser.parse_args()
    cutoffs = [int(x) for x in args.top_k.split(",") if x]
    result = run(args.dataset, args.data_path, cutoffs)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
