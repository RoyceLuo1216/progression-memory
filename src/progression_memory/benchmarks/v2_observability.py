"""Summarize measured V2 component comparisons without inventing missing metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(payload: dict[str, Any], *, family: str) -> list[dict[str, Any]]:
    rows = []
    for name, system in sorted(payload["systems"].items()):
        if system.get("status") == "skipped":
            rows.append({"family": family, "system": name, "status": "skipped",
                         "reason": system["reason"]})
            continue
        for budget, result in sorted(system["budgets"].items(), key=lambda kv: int(kv[0])):
            rows.append({
                "family": family,
                "system": name,
                "status": "measured",
                "budget_chars": int(budget),
                "query_count": result.get("eligible_query_count", result.get("query_count")),
                "macro_evidence_recall": result["macro_evidence_recall"],
                "mrr": result["mrr"],
                "avg_retrieved_chars": result["avg_retrieved_chars"],
            })
    return rows


def compile_report(paired: dict[str, Any], ablation: dict[str, Any]) -> dict[str, Any]:
    if paired.get("input_sha256") != ablation.get("input_sha256"):
        raise ValueError("paired and ablation reports were generated from different fixtures")
    rows = _rows(paired, family="paired") + _rows(ablation, family="ablation")
    benchmark_status = {
        "retrieval_evidence_recall": "measured",
        "writer_attachment_accuracy": "not_measured",
        "writer_live_model_accuracy": "not_measured",
        "state_reconstruction_accuracy": "not_measured",
        "hierarchy_parent_accuracy": "not_measured",
        "maintenance_latency_and_cost": "not_measured",
        "grounded_answer_accuracy": "not_measured",
        "dense_embedding_retrieval": "not_measured" if not ablation.get("embedding_model") else "measured",
        "historical_as_of_query_accuracy": "not_measured",
    }
    return {
        "schema_version": "v2-observability-1",
        "fixture_sha256": paired["input_sha256"],
        "observation_count": paired["observations"],
        "question_count": paired["questions"],
        "measurement_status": benchmark_status,
        "rows": rows,
        "warnings": [
            "Synthetic oracle writer: no measured real-LLM writer performance.",
            "Evidence-ID recall does not demonstrate that retrieved text entails the answer.",
            "Parent/child state units can credit many source IDs even when details were overwritten.",
            "A near-saturated RAG baseline prevents strong ranking conclusions.",
            "Hierarchical and flat BM25 are identical on this fixture; no demonstrated gain.",
            "Hybrid/temporal results must not be imputed when skipped.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Progression V2 — Observability Report",
        "",
        f"Fixture SHA256: \`{report['fixture_sha256']}\`",
        f"Observations: {report['observation_count']}; questions: {report['question_count']}",
        "",
        "## Retrieval results (measured)",
        "",
        "| Family | System | Budget (chars) | Evidence recall | MRR | Mean chars |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in report["rows"]:
        if item["status"] == "skipped":
            continue
        lines.append(
            f"| {item['family']} | {item['system']} | {item['budget_chars']} | "
            f"{item['macro_evidence_recall']:.2%} | {item['mrr']:.3f} | "
            f"{item['avg_retrieved_chars']:.1f} |"
        )
    lines += ["", "## Component measurement coverage", ""]
    for name, status in report["measurement_status"].items():
        lines.append(f"- {name}: **{status}**")
    lines += ["", "## Interpretation limitations", ""]
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = compile_report(
        json.loads(Path(args.paired).read_text(encoding="utf-8")),
        json.loads(Path(args.ablation).read_text(encoding="utf-8")),
    )
    for name, content in [
        (args.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (args.output_md, markdown_report(report)),
    ]:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
