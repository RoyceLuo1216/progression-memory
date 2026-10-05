"""Inspect the pinned AlpsBench *development* update task without copying dialogue."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REVISION = "238961f49e4fe6de8b09a4fefd77d76d759d620f"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(inputs: list[dict], references: list[dict], input_sha: str, reference_sha: str) -> dict:
    if len(inputs) != len(references):
        raise ValueError("input/reference row count mismatch")
    if not inputs:
        raise ValueError("empty update development set")
    keys = ("benchmark_id", "canonical_id", "session_id", "task")
    if any(any(left.get(key) != right.get(key) for key in keys)
           for left, right in zip(inputs, references, strict=True)):
        raise ValueError("input/reference IDs or order mismatch")
    if any(not isinstance(row.get("input"), dict) for row in inputs) or any(
            not isinstance(row.get("gold"), dict) for row in references):
        raise ValueError("unexpected nested update schema")
    gold_keys = Counter(k for row in references for k in row["gold"])
    scalar_gold = {}
    for key in gold_keys:
        values = [row["gold"].get(key) for row in references]
        if all(value is None or isinstance(value, (str, int, bool)) for value in values):
            frequencies = Counter(str(value) for value in values)
            if len(frequencies) <= 20 and all(len(value) <= 50 for value in frequencies):
                scalar_gold[key] = dict(sorted(frequencies.items()))
    return {"source": "Cosineyx/Alpsbench", "revision": REVISION,
            "split": "dev/task2", "rows": len(inputs),
            "input_sha256": input_sha, "reference_sha256": reference_sha,
            "input_keys": dict(sorted(Counter(k for row in inputs for k in row).items())),
            "reference_keys": dict(sorted(Counter(k for row in references for k in row).items())),
            "input_nested_keys": dict(sorted(Counter(k for row in inputs for k in row["input"]).items())),
            "gold_nested_keys": dict(sorted(gold_keys.items())),
            "gold_scalar_labels": scalar_gold,
            "strata": dict(sorted(Counter(row.get("stratum") for row in inputs).items())),
            "sessions": len({row["session_id"] for row in inputs}),
            "sample_input_types": {k: type(v).__name__ for k, v in inputs[0].items()},
            "sample_reference_types": {k: type(v).__name__ for k, v in references[0].items()},
            "limitations": "Development set; benchmark update actions do not label progression hierarchy or phase. Test split untouched."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(read_jsonl(args.inputs), read_jsonl(args.references),
                   hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                   hashlib.sha256(args.references.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": report["rows"], "reference_keys": report["reference_keys"]}))


if __name__ == "__main__":
    main()
