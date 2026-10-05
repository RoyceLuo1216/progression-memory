"""Audit external Personal Facts (MSC) labels for writer pretraining.

This source supplies fact extraction/referent/duration supervision, not
progression attachment, hierarchy, phase or full-sequence gold traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCE = "adugeen/personal-facts-msc"
REVISION = "22854562a2934ea07983ecda4f4fd8ff3aef2e85"
FIELDS = ("context", "text", "main_category", "time", "referent", "duration",
          "broken", "broken_reason", "followup")


def normalize(row: dict) -> dict:
    if not all(field in row for field in FIELDS):
        raise ValueError(f"missing fields: {sorted(set(FIELDS) - set(row))}")
    if not row["context"] or not row["text"]:
        raise ValueError("fact and its dialogue context are required")
    if row["referent"] not in ("Self", "Other", "None"):
        raise ValueError("unknown referent")
    if row["broken"] not in ("Yes", "No"):
        raise ValueError("unknown validity annotation")
    result = {field: row[field] for field in FIELDS}
    result["example_id"] = hashlib.sha256(
        (row["context"] + "\n" + row["text"]).encode()).hexdigest()
    # A usable *candidate* is not a gold memory: factual entailment and
    # process relevance need subsequent auditing.
    result["durable_self_candidate"] = (row["referent"] == "Self"
                                        and row["duration"] == "Long-term"
                                        and row["broken"] == "No")
    return result


def summarize(rows: list[dict], split: str, source_sha256: str) -> dict:
    normalized = [normalize(row) for row in rows]
    return {"source": SOURCE, "revision": REVISION, "split": split,
            "source_sha256": source_sha256, "rows": len(normalized),
            "durable_self_candidates": sum(x["durable_self_candidate"] for x in normalized),
            "counts": {field: dict(sorted(Counter(str(x[field]) for x in normalized).items()))
                       for field in ("main_category", "time", "referent", "duration", "broken")},
            "limitations": "Single annotator, MSC-derived, no progression graph or conversation ID; no independent sequence evaluation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(args.train_parquet).to_pylist()
    result = summarize(rows, "train", hashlib.sha256(args.train_parquet.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": result["rows"],
                      "durable_self_candidates": result["durable_self_candidates"]}))


if __name__ == "__main__":
    main()
