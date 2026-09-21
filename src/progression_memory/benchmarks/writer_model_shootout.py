"""Compare local writers on the source-linked construction fixture."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from progression_memory.benchmarks.process_writer_benchmark import run as run_construction
from progression_memory.evaluation.semantic import SentenceTransformerEmbedder
from progression_memory.maintenance.local_model import clear_model_cache

MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-3B-Instruct",
)
QUANTIZED_MODELS = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.casefold()).strip("-")


def _admission_f1(grouping: dict) -> float:
    precision = float(grouping.get("admission_precision", 0.0))
    recall = float(grouping.get("admission_recall", 0.0))
    return 2 * precision * recall / max(1e-12, precision + recall)


def run_model(model: str, output_dir: Path, max_turns: int,
              quantization: str,
              embedder: SentenceTransformerEmbedder) -> dict:
    path = output_dir / f"{_slug(model)}-{_slug(quantization)}.json"
    try:
        result = run_construction(
            dataset="fixture",
            model_name=model,
            quantization=quantization,
            max_turns=max_turns,
            output=str(path),
            embedder=embedder,
        )
    except Exception:  # noqa: BLE001 - each model failure is a recorded benchmark outcome
        result = json.loads(path.read_text())
    finally:
        clear_model_cache()
    corpora = result.get("corpora") or [{}]
    grouping = corpora[0].get("grouping", {})
    writer = result.get("writer", {})
    calls = max(1, int(writer.get("model_calls", 0)))
    return {
        "model": model,
        "quantization": quantization,
        "status": result.get("status", "failed"),
        "pair_f1": float(grouping.get("pair_f1", 0.0)),
        "admission_f1": _admission_f1(grouping),
        "action_accuracy": float(grouping.get("action_accuracy", 0.0)),
        "cluster_purity": float(grouping.get("cluster_purity", 0.0)),
        "invalid_responses": int(writer.get("invalid_responses", 0)),
        "seconds_per_call": float(writer.get("generation_seconds", 0.0)) / calls,
        "result_path": str(path),
    }


def rank_key(row: dict) -> tuple:
    return (
        row["status"] != "completed",
        row["invalid_responses"] > 0,
        -row["pair_f1"],
        -row["admission_f1"],
        -row["action_accuracy"],
        row["seconds_per_call"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--quantized-models", default=",".join(QUANTIZED_MODELS))
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--output-dir", default="evaluation/results/writer-shootout")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder()
    variants = [(model.strip(), "none") for model in args.models.split(",") if model.strip()]
    variants += [
        (model.strip(), "quanto-int8")
        for model in args.quantized_models.split(",") if model.strip()
    ]
    rows = [run_model(model, output_dir, args.max_turns, quantization, embedder)
            for model, quantization in variants]
    rows.sort(key=rank_key)
    payload = {"winner": ({"model": rows[0]["model"],
                            "quantization": rows[0]["quantization"]} if rows else None),
               "max_turns": args.max_turns, "results": rows}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
