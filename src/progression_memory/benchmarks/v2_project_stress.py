"""Project-continuity stress evaluation on frozen oracle writer decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from progression_memory.benchmarks.v2_paired import run as paired_run
from progression_memory.benchmarks.v2_project_continuity import generate
from progression_memory.benchmarks.v2_retrieval_ablation import evaluate as ablate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--budgets", default="400,800,1600,3200")
    args = p.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    fixture = generate(distractors_per_stage=args.distractors)
    results = Path(args.output_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "fixture.json").write_text(fixture.model_dump_json(indent=2) + "\n")
    paired = paired_run(fixture, budgets)
    lexical = ablate(fixture, budgets, modes=("flat_bm25", "hierarchical_bm25"))
    (results / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    (results / "ablation.json").write_text(json.dumps(lexical, indent=2) + "\n")
    summary = {
        "dataset": {
            "projects": 10, "observations": len(fixture.observations),
            "questions": len(fixture.questions), "distractors_per_stage": args.distractors,
        },
        "paired": {
            system: {
                b: {
                    "evidence_recall": round(value["macro_evidence_recall"], 4),
                    "mrr": round(value["mrr"], 4),
                }
                for b, value in result["budgets"].items()
            } for system, result in paired["systems"].items()
        },
        "retrieval_ablation": {
            system: {
                b: {
                    "evidence_recall": round(value["macro_evidence_recall"], 4),
                    "mrr": round(value["mrr"], 4),
                }
                for b, value in result["budgets"].items()
            } for system, result in lexical["systems"].items()
        },
        "caveats": [
            "Synthetic oracle writer; does not measure live model construction.",
            "Historical questions are not evaluated with query-specific time cutoffs.",
            "Evidence-ID recall may credit state units that do not expose the answer.",
            "No LLM QA reader was run; scores are not answer accuracy.",
            "Distinct project subjects are evaluated separately to avoid privacy leakage.",
        ],
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
