from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

from progression_memory.maintenance.local_model import TransformersLanguageModel
from progression_memory.maintenance.manager import SLMProgressionManager
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression


MODELS = (
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-3B-Instruct",
)


@dataclass(frozen=True)
class Case:
    evidence: str
    state: str
    expected: str


CASES = (
    Case("I have another Google interview next week.", "Interviewing with Google.", "extend"),
    Case("Google sent me a software engineering offer.", "Interviewing with Google.", "transition"),
    Case("I declined Google's offer.", "Received a Google software engineering offer.", "transition"),
    Case("I started preparing specifically for Google's system design round.", "Recruiting with Google.", "create_child"),
    Case("I signed up for a pottery class.", "Recruiting with Google.", "create"),
    Case("My Shopify internship starts next Monday.", "Accepted a Shopify internship.", "transition"),
    Case("I completed another easy run today.", "Running consistently three times per week.", "extend"),
    Case("My ankle injury means I stopped running for now.", "Running consistently three times per week.", "transition"),
)


def run(model_name: str) -> dict[str, object]:
    model = TransformersLanguageModel(model_name)
    manager = SLMProgressionManager(model, candidate_k=1)
    rows, correct, valid, elapsed = [], 0, 0, 0.0
    for i, case in enumerate(CASES):
        p = Progression(id="p", title="progression", scope="", high_level_summary="",
                        current_state=case.state)
        e = Episode(id=str(i), summary=case.evidence, dialogue_time="2026-01-01T00:00:00",
                    embedding=[1.0, 0.0], state_vector=[1.0, 0.0])
        start = time.perf_counter()
        decision = manager.decide(e, [p])
        latency = time.perf_counter() - start
        elapsed += latency
        action = decision.action.value
        is_valid = action in {"extend", "transition", "create_child", "create"}
        valid += int(is_valid); correct += int(action == case.expected)
        rows.append({"expected": case.expected, "predicted": action, "latency_s": latency})
    return {"model": model_name, "cases": len(CASES), "accuracy": correct / len(CASES),
            "valid_rate": valid / len(CASES), "total_seconds": elapsed, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output")
    args = parser.parse_args()
    results = [run(x.strip()) for x in args.models.split(",") if x.strip()]
    results.sort(key=lambda x: (-float(x["accuracy"]), float(x["total_seconds"])))
    payload = {"winner": results[0]["model"], "results": results}
    text = json.dumps(payload, indent=2); print(text)
    if args.output:
        from pathlib import Path
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text + "\n")


if __name__ == "__main__": main()
