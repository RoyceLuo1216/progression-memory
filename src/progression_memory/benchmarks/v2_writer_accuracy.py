"""Writer construction evaluator with frozen human-labeled event decisions.

Gold format:
{"cases":[{"observation_id":"...", "gold_targets":["progression-id"],
           "predicted_targets":["progression-id"], "gold_actions":["transition"],
           "predicted_actions":["append"], "unsupported_patches":0}]}
Empty target/action arrays represent genuine no-update cases. For newly created
identities, use a stable gold cluster key rather than runtime UUID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str
    gold_targets: list[str] = Field(default_factory=list)
    predicted_targets: list[str] = Field(default_factory=list)
    gold_actions: list[str] = Field(default_factory=list)
    predicted_actions: list[str] = Field(default_factory=list)
    unsupported_patches: int = Field(default=0, ge=0)


class LabeledTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[Case]


def evaluate(trace: LabeledTrace) -> dict[str, object]:
    if len({c.observation_id for c in trace.cases}) != len(trace.cases):
        raise ValueError("duplicate observation IDs")
    tp = fp = fn = exact = action_exact = unsupported = 0
    by_case = []
    for case in trace.cases:
        gold = set(case.gold_targets)
        pred = set(case.predicted_targets)
        if len(gold) != len(case.gold_targets) or len(pred) != len(case.predicted_targets):
            raise ValueError("duplicate target IDs in one observation")
        if len(case.gold_actions) != len(case.gold_targets):
            raise ValueError("gold actions/targets must align")
        if len(case.predicted_actions) != len(case.predicted_targets):
            raise ValueError("predicted actions/targets must align")
        gold_pairs = set(zip(case.gold_targets, case.gold_actions, strict=True))
        pred_pairs = set(zip(case.predicted_targets, case.predicted_actions, strict=True))
        same_targets = gold == pred
        same_actions = gold_pairs == pred_pairs
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += same_targets
        action_exact += same_actions
        unsupported += case.unsupported_patches
        by_case.append({
            "observation_id": case.observation_id,
            "target_exact": same_targets,
            "action_exact": same_actions,
            "false_positive": sorted(pred - gold),
            "false_negative": sorted(gold - pred),
            "unsupported_patches": case.unsupported_patches,
        })
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "cases": len(trace.cases),
        "attachment_precision": precision,
        "attachment_recall": recall,
        "attachment_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "target_exact_rate": exact / len(trace.cases) if trace.cases else 0.0,
        "action_exact_rate": action_exact / len(trace.cases) if trace.cases else 0.0,
        "unsupported_patch_count": unsupported,
        "per_observation": by_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = LabeledTrace.model_validate_json(Path(args.labels).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluate(trace), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
