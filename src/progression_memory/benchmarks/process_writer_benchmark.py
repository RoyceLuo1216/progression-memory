"""Real SLM construction + evidence retrieval benchmark. No heuristic fallback."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from progression_memory.evaluation.datasets import load_locomo
from progression_memory.evaluation.process_construction import build_process_documents
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, evaluate_retrieval
from progression_memory.evaluation.semantic import SentenceTransformerEmbedder
from progression_memory.maintenance.local_model import TransformersLanguageModel
from progression_memory.maintenance.process_writer import ProcessWriter, obvious_non_update

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
    ("Ben: I signed up for pottery classes at the same studio.", "ben_pottery", "create"),
    ("Ana: Acme promoted me to senior software engineer.", "ana_acme", "transition"),
    ("Ben: I started training for a half marathon.", "ben_running", "create"),
    ("Ana: I started running before work.", "ana_running", "create"),
    ("Ben: My ankle injury forced me to pause half-marathon training.", "ben_running", "transition"),
    ("Ana: My wrist healed and I returned to pottery class.", "ana_pottery", "transition"),
    ("Ben: How is your pottery class going?", None, None),
    ("Ana: I applied for a design role at Beta.", "ana_beta", "create"),
    ("Ana: Beta invited me to a portfolio interview.", "ana_beta", "transition"),
    ("Ana: I declined Beta because I already accepted Acme.", "ana_beta", "transition"),
    ("Ben: I moved into a new apartment this weekend.", "ben_move", "create"),
    ("Ben: I finished unpacking the kitchen.", "ben_move", "extend"),
    ("Ben: The move is complete and I am settled in.", "ben_move", "transition"),
    ("Ana: Nice!", None, None),
]


def _process_record(progression):
    return {
        "id": progression.id,
        "subject": progression.scope,
        "process": progression.title,
        "current_state": progression.current_state,
        "episode_ids": progression.episode_ids,
        "segment_ids": progression.segment_ids,
        "history": [{
            "source_id": checkpoint.episode_id,
            "action": checkpoint.action,
            "fact": checkpoint.evidence,
            "prior_state": checkpoint.prior_state,
            "state": checkpoint.posterior_state,
            "delta": checkpoint.state_delta,
        } for checkpoint in progression.state_history],
    }


def grouping_metrics(processes, cases=CASES):
    expected = {f"turn:{i}": label for i, (_, label, _) in enumerate(cases) if label}
    actual: dict[str, set] = {}
    actions = {}
    members_by_process = {}
    for p in processes.values():
        for checkpoint in p.state_history:
            actual.setdefault(checkpoint.episode_id, set()).add(p.id)
            actions.setdefault(checkpoint.episode_id, set()).add(checkpoint.action)
            members_by_process.setdefault(p.id, set()).add(checkpoint.episode_id)
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
    cross_subject_pairs = sum(
        1 for a, b in predicted_pairs
        if cases[int(a.split(":")[1])][0].split(":", 1)[0]
        != cases[int(b.split(":")[1])][0].split(":", 1)[0]
    )
    skipped = {f"turn:{i}" for i, (text, _, _) in enumerate(cases) if obvious_non_update(text)}
    purity_numerator = sum(
        max(Counter(expected[source] for source in members if source in expected).values(),
            default=0)
        for members in members_by_process.values()
    )
    labelled_memberships = sum(
        sum(source in expected for source in members) for members in members_by_process.values()
    )
    fragments = {
        label: len({pid for source, pids in actual.items() if expected.get(source) == label
                   for pid in pids})
        for label in set(expected.values())
    }
    return {"pair_precision": precision, "pair_recall": recall,
            "pair_f1": 2*precision*recall/max(1e-12, precision+recall),
            "admission_precision": len(set(actual) & set(expected))/max(1, len(actual)),
            "admission_recall": len(set(actual) & set(expected))/max(1, len(expected)),
            "action_accuracy": sum(actions.get(i) == {a} for i, a in expected_actions.items())/max(1, len(expected_actions)),
            "cross_subject_predicted_pairs": cross_subject_pairs,
            "cluster_purity": purity_numerator / max(1, labelled_memberships),
            "avg_fragments_per_gold_process": sum(fragments.values()) / max(1, len(fragments)),
            "gold_process_count": len(set(expected.values())),
            "predicted_process_count": len(processes),
            "gate_event_recall": len(set(expected) - skipped) / max(1, len(expected)),
            "gate_skipped_turns": len(skipped),
            "label_source": "hand_authored_development_fixture"}


def run(*, dataset, data_path=None, model_name="Qwen/Qwen2.5-1.5B-Instruct", output,
        corpus_limit=1, max_turns=None, writer=None, embedder=None, cutoffs=(5, 8, 10),
        enforce_quality=False):
    if corpus_limit < 1 or (max_turns is not None and max_turns < 1):
        raise ValueError("corpus_limit and max_turns must be positive")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = path.with_suffix(".trace.jsonl")
    trace.write_text("")
    writer = writer or ProcessWriter(
        TransformersLanguageModel(model_name, max_new_tokens=256), trace_path=str(trace)
    )
    embedder = embedder or SentenceTransformerEmbedder()
    packages = {}
    for package in ("transformers", "torch", "sentence-transformers", "lm-format-enforcer"):
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
            if max_turns is not None:
                docs = docs[:max_turns]
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
            row = {"id": name, **stats,
                   "processes": [_process_record(p) for p in processes.values()]}
            result["corpora"].append(row)
            if dataset == "fixture":
                row["grouping"] = grouping_metrics(processes, CASES[:len(docs)])
                if enforce_quality:
                    grouping = row["grouping"]
                    if (grouping["admission_recall"] < .80 or grouping["pair_f1"] < .65
                            or grouping["cluster_purity"] < .85
                            or grouping["avg_fragments_per_gold_process"] > 1.5
                            or grouping["cross_subject_predicted_pairs"]):
                        raise RuntimeError(f"construction quality gate failed: {grouping}")
            available = {eid for d in docs for eid in (d.evidence_ids or {d.id})}
            eligible = [q for q in queries if q.evidence_ids and q.evidence_ids <= available]
            row.update(total_queries=len(queries), eligible_queries=len(eligible), excluded_queries=len(queries)-len(eligible))
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
    parser.add_argument("--enforce-quality", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(dataset=args.dataset, data_path=args.data_path, model_name=args.model,
                 output=args.output, corpus_limit=args.corpus_limit, max_turns=args.max_turns,
                 enforce_quality=args.enforce_quality)
    print(json.dumps({"status": result["status"], "writer": result["writer"]}, indent=2))


if __name__ == "__main__": main()
