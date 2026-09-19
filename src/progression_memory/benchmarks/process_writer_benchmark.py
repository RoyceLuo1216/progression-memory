"""Real SLM construction + evidence retrieval benchmark. No heuristic fallback."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from progression_memory.evaluation.datasets import load_locomo
from progression_memory.evaluation.process_construction import build_process_documents
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, evaluate_retrieval
from progression_memory.evaluation.semantic import SentenceTransformerEmbedder
from progression_memory.maintenance.local_model import TransformersLanguageModel
from progression_memory.maintenance.process_writer import ProcessWriter


# Development cases, not an independent test set or evidence of model superiority.
CASES = [
    ("Ana: Hi Ben, good to see you!", None, None),
    ("Ana: I applied for a software engineering job at Acme today.", "ana_acme", "create"),
    ("Ben: I applied for a software engineering job at Acme too.", "ben_acme", "create"),
    ("Ana: Acme invited me to an interview next Tuesday.", "ana_acme", "transition"),
    ("Ben: That's great, congratulations!", None, None),
    ("Ana: I started taking pottery classes on Saturdays.", "ana_pottery", "create"),
    ("Ana: I attended another pottery class this Saturday.", "ana_pottery", "extend"),
    ("Ana: Acme offered me the job after my interview.", "ana_acme", "transition"),
    ("Ben: Did you accept their offer?", None, None),
    ("Ana: Yes, I accepted Acme's offer this morning.", "ana_acme", "transition"),
    ("Ben: Acme rejected my application today.", "ben_acme", "transition"),
    ("Ana: I hurt my wrist, so I have paused my pottery classes.", "ana_pottery", "transition"),
]


def grouping_metrics(processes, cases=CASES):
    expected = {f"turn:{i}": label for i, (_, label, _) in enumerate(cases) if label}
    actual: dict[str, set] = {}
    actions = {}
    for p in processes.values():
        for h in p.history:
            actual.setdefault(h["source_id"], set()).add(p.id)
            actions.setdefault(h["source_id"], set()).add(h["action"])
    ids = [f"turn:{i}" for i in range(len(cases))]
    predicted_pairs, gold_pairs = set(), set()
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if actual.get(a, set()) & actual.get(b, set()): predicted_pairs.add((a, b))
            if a in expected and expected.get(a) == expected.get(b): gold_pairs.add((a, b))
    correct = len(predicted_pairs & gold_pairs)
    precision = correct / max(1, len(predicted_pairs))
    recall = correct / max(1, len(gold_pairs))
    expected_actions = {f"turn:{i}": a for i, (_, _, a) in enumerate(cases) if a}
    return {"pair_precision": precision, "pair_recall": recall,
            "pair_f1": 2*precision*recall/max(1e-12, precision+recall),
            "admission_precision": len(set(actual) & set(expected))/max(1, len(actual)),
            "admission_recall": len(set(actual) & set(expected))/max(1, len(expected)),
            "action_accuracy": sum(actions.get(i) == {a} for i, a in expected_actions.items())/max(1, len(expected_actions)),
            "label_source": "hand_authored_development_fixture"}


def run(*, dataset, data_path=None, model_name="Qwen/Qwen2.5-1.5B-Instruct", output,
        corpus_limit=1, max_turns=None, writer=None, embedder=None, cutoffs=(5, 8, 10)):
    if corpus_limit < 1 or (max_turns is not None and max_turns < 1):
        raise ValueError("corpus_limit and max_turns must be positive")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = path.with_suffix(".trace.jsonl")
    trace.write_text("")
    writer = writer or ProcessWriter(TransformersLanguageModel(model_name, max_new_tokens=512), trace_path=str(trace))
    embedder = embedder or SentenceTransformerEmbedder()
    packages = {}
    for package in ("transformers", "torch", "sentence-transformers"):
        try: packages[package] = version(package)
        except PackageNotFoundError: packages[package] = None
    result = {"runtime_packages": packages, "status": "running", "dataset": dataset, "system": "process_slm_bm25",
              "scope": {"corpus_limit": corpus_limit, "max_turns": max_turns},
              "ranking": "BM25 over process updates, then BM25 over original source turns",
              "corpora": []}
    try:
        if dataset == "fixture":
            docs = tuple(RetrievalDocument(id=f"turn:{i}", text=text, timestamp="2026-01-01T00:00:00")
                         for i, (text, _, _) in enumerate(CASES))
            corpora = [("development_fixture", docs, [])]
        else:
            if not data_path: raise ValueError("LoCoMo requires data_path")
            all_queries = load_locomo(data_path)
            groups = {}
            for q in all_queries:
                key = id(q.documents)
                groups.setdefault(key, []).append(q)
            corpora = [(qs[0].id.split(":q")[0], qs[0].documents, qs) for qs in list(groups.values())[:corpus_limit]]
        baseline, transformed = [], []
        for name, docs, queries in corpora:
            if dataset != "fixture" and max_turns is not None: docs = docs[:max_turns]
            units, processes, stats = build_process_documents(docs, writer, embedder)
            row = {"id": name, **stats, "processes": [asdict(p) for p in processes.values()]}
            if dataset == "fixture": row["grouping"] = grouping_metrics(processes)
            available = {eid for d in docs for eid in (d.evidence_ids or {d.id})}
            eligible = [q for q in queries if q.evidence_ids and q.evidence_ids <= available]
            row.update(total_queries=len(queries), eligible_queries=len(eligible), excluded_queries=len(queries)-len(eligible))
            result["corpora"].append(row)
            index = BM25Index(units)
            coverages = []
            for q in eligible:
                selected = index.retrieve(q.text, min(64, max(16, math.ceil(.10*len(units)))))
                sources = {d.id: d for unit in selected for d in unit.metadata["source_documents"]}
                found = {eid for d in sources.values() for eid in (d.evidence_ids or {d.id})}
                coverages.append(len(found & q.evidence_ids)/len(q.evidence_ids))
                baseline.append(replace(q, documents=docs))
                transformed.append(replace(q, documents=tuple(sources.values())))
            row["candidate_evidence_coverage"] = sum(coverages)/max(1, len(coverages))
        if writer.counts["model_calls"] == 0 or writer.counts["valid_responses"] == 0:
            raise RuntimeError("benchmark did not exercise the SLM")
        if not any(c["assigned_source_count"] for c in result["corpora"]):
            raise RuntimeError("SLM produced no process updates; construction smoke check failed")
        result.update(status="completed", retrieval=evaluate_retrieval(transformed, cutoffs),
                      append_baseline=evaluate_retrieval(baseline, cutoffs))
        return result
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        result["writer"] = writer.stats()
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fixture", "locomo"], required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--corpus-limit", type=int, default=1)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(dataset=args.dataset, data_path=args.data_path, model_name=args.model,
                 output=args.output, corpus_limit=args.corpus_limit, max_turns=args.max_turns)
    print(json.dumps({"status": result["status"], "writer": result["writer"]}, indent=2))


if __name__ == "__main__": main()
