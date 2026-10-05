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
    nested_shapes = Counter()
    operation_labels = Counter()
    def walk(value, path="gold", depth=0):
        nested_shapes[f"{path}:{type(value).__name__}"] += 1
        if depth >= 4:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in ("action", "operation", "update_type") and isinstance(item, str):
                    operation_labels[item] += 1
                walk(item, f"{path}.{key}", depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, path + "[]", depth + 1)
    for row in references:
        walk(row["gold"])
    input_shapes = Counter()
    deltas = Counter()
    controls = Counter()
    selected_present = Counter()
    per_case = Counter()
    for input_row, reference_row in zip(inputs, references, strict=True):
        row = input_row
        def profile(value, path="input", depth=0):
            input_shapes[f"{path}:{type(value).__name__}"] += 1
            if depth >= 3:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    profile(item, f"{path}.{key}", depth + 1)
            elif isinstance(value, list):
                for item in value:
                    profile(item, path + "[]", depth + 1)
        profile(row["input"])
        before = row["input"]["memory"]
        after = reference_row["gold"]["answer"]
        if not isinstance(before, list) or not isinstance(after, list):
            raise ValueError("memory update must contain before/after lists")
        old = {item["memory_id"]: item for item in before}
        new = {item["memory_id"]: item for item in after}
        if len(old) != len(before) or len(new) != len(after):
            raise ValueError("duplicate memory IDs")
        added = set(new) - set(old)
        removed = set(old) - set(new)
        changed = {key for key in set(old) & set(new)
                   if old[key]["value"] != new[key]["value"]}
        retained = set(old) & set(new) - changed
        deltas.update(added=len(added), removed=len(removed), changed=len(changed), retained=len(retained))
        per_case[(len(added), len(removed), len(changed))] += 1
        controls.update({str(k) + "=" + str(v): 1 for k, v in
                         row["input"]["metadata"]["controls"].items()})
        selected_present["in_before" if row["input"]["metadata"]["selected_memory_id"] in old
                         else "absent"] += 1
    return {"source": "Cosineyx/Alpsbench", "revision": REVISION,
            "split": "dev/task2", "rows": len(inputs),
            "input_sha256": input_sha, "reference_sha256": reference_sha,
            "input_keys": dict(sorted(Counter(k for row in inputs for k in row).items())),
            "reference_keys": dict(sorted(Counter(k for row in references for k in row).items())),
            "input_nested_keys": dict(sorted(Counter(k for row in inputs for k in row["input"]).items())),
            "gold_nested_keys": dict(sorted(gold_keys.items())),
            "gold_scalar_labels": scalar_gold,
            "gold_shapes": dict(sorted(nested_shapes.items())),
            "input_shapes": dict(sorted(input_shapes.items())),
            "id_based_deltas": dict(sorted(deltas.items())),
            "delta_pattern_counts": {str(k): v for k, v in sorted(per_case.items())},
            "control_flags": dict(sorted(controls.items())),
            "selected_id_presence": dict(sorted(selected_present.items())),
            "operation_labels": dict(sorted(operation_labels.items())),
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
