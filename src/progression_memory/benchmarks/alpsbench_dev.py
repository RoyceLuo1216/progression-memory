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
    return {"source": "Cosineyx/Alpsbench", "revision": REVISION,
            "split": "dev/task2", "rows": len(inputs),
            "input_sha256": input_sha, "reference_sha256": reference_sha,
            "input_keys": dict(sorted(Counter(k for row in inputs for k in row).items())),
            "reference_keys": dict(sorted(Counter(k for row in references for k in row).items())),
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
