"""Long-horizon component diagnostics, separate from end-to-end LLM QA.

Oracle-state correctness checks independent patch replay against StateEngine's
bitemporal reconstruction. Retrieval answerability measures literal citation
visibility, NOT natural-language answer accuracy. All observations and patches
are synthetic oracle-labeled data; live LLM writer metrics remain unavailable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from progression_memory.benchmarks.v2_long_horizon_dataset import generate
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.evaluation.retrieval import BM25Index
from progression_memory.v2.hierarchical_retrieval import retrieve, select_budget
from progression_memory.v2.models import AttributeOperation


def oracle_replay(fixture, progression_id: str, valid_at, known_at):
    """Independent SET/RETRACT event replay using source-authored oracle patches."""
    events = []
    for obs in fixture.observations:
        if obs.effective_at > valid_at or obs.recorded_at > known_at:
            continue
        for index, update in enumerate(fixture.proposals[obs.id].updates):
            pid = (f"v2:{obs.id}:{index}" if update.action == "create"
                   else update.progression_id)
            if pid == progression_id:
                events.append((obs.effective_at, obs.recorded_at, obs.id, update))
    events.sort(key=lambda row: row[:3])
    state = {}
    for _, _, _, update in events:
        for patch in update.patches:
            if patch.operation == AttributeOperation.SET:
                state[patch.key] = patch.value
            elif patch.operation == AttributeOperation.RETRACT:
                state.pop(patch.key, None)
            else:
                raise ValueError("oracle check supports SET and RETRACT only")
        if update.stage is not None:
            state["stage"] = update.stage
    return state


def evaluate(fixture, *, budgets=(400, 1600, 3200)):
    engine = prepare(fixture)
    observations = fixture.observations
    last_recorded = max(obs.recorded_at for obs in observations)
    last_valid = max(obs.effective_at for obs in observations)
    # Every 12th observation plus current, evaluated for each progression.
    checkpoints = [observations[i] for i in range(0, len(observations), 12)]
    snapshots = [(o.effective_at, o.recorded_at) for o in checkpoints]
    snapshots.append((last_valid, last_recorded))
    state_correct = state_total = 0
    mismatches = []
    for valid_at, known_at in snapshots:
        for pid in engine.progressions:
            expected = oracle_replay(fixture, pid, valid_at, known_at)
            actual = engine.state_at(pid, valid_at=valid_at, known_at=known_at)
            state_total += 1
            if expected == actual:
                state_correct += 1
            else:
                mismatches.append({"progression": pid, "valid_at": valid_at.isoformat(),
                                   "known_at": known_at.isoformat(),
                                   "expected": expected, "actual": actual})
    # Literal answerability probe: can a reader SEE the exact gold source
    # observation ID in selected text? State metadata alone cannot count.
    corpus = defaultdict(list)
    for obs in observations:
        corpus[obs.subject_id].append(obs)
    output = {}
    for system in ("rag", "v2_flat", "v2_hierarchical"):
        output[system] = {}
        for budget in budgets:
            hits = 0
            rows = []
            for question in fixture.questions:
                if system == "rag":
                    from progression_memory.evaluation.retrieval import RetrievalDocument
                    docs = [
                        RetrievalDocument(id=obs.id, text=obs.text,
                                          evidence_ids=frozenset({obs.id}))
                        for obs in corpus[question.subject_id]
                    ]
                    ranked = [d for _, d in BM25Index(docs).score(question.question)]
                else:
                    ranked = retrieve(engine, question.question,
                                      mode=("flat_bm25" if system == "v2_flat"
                                            else "hierarchical_bm25"),
                                      subject_id=question.subject_id,
                                      k=len(engine.observations) + len(engine.progressions))
                selected = select_budget(ranked, budget)
                source_text = "\n".join(d.text for d in selected)
                gold = set(question.evidence_ids)
                # Require an actual source-text ID, not only opaque credited metadata.
                found = {obsid for obsid in gold if obsid in source_text}
                success = bool(gold) and found == gold
                hits += success
                rows.append({"question_id": question.id, "answerability": success,
                             "gold_count": len(gold), "literal_gold_ids_found": len(found)})
            output[system][str(budget)] = {
                "literal_evidence_visibility_rate": hits / len(fixture.questions),
                "question_count": len(fixture.questions),
                "per_question": rows,
            }
    return {
        "protocol": "v2-oracle-state-and-visibility-v1",
        "dataset": {"observations": len(observations),
                    "questions": len(fixture.questions),
                    "progressions": len(engine.progressions),
                    "horizon_days": (last_recorded - min(o.recorded_at for o in observations)).days},
        "state": {"exact_state_accuracy": state_correct / state_total,
                  "correct": state_correct, "total": state_total,
                  "mismatches": mismatches[:20]},
        "literal_answerability": output,
        "writer": {"status": "not_measured_live_model"},
        "llm_answer_accuracy": {"status": "not_measured"},
        "limitations": [
            "Gold states are computed from scripted oracle patches, not human annotations.",
            "Exact state checks validate replay correctness, not LLM state extraction.",
            "Literal evidence visibility is a diagnostic, not question-answering accuracy.",
            "Benchmark questions often contain checkpoint lexical identifiers.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-stream", type=int, default=96)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="400,1600,3200")
    args = parser.parse_args()
    fixture = generate(events_per_stream=args.events_per_stream)
    result = evaluate(fixture, budgets=tuple(int(x) for x in args.budgets.split(",")))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": result["dataset"], "state": {
        "exact_state_accuracy": result["state"]["exact_state_accuracy"],
        "total": result["state"]["total"],
    }, "literal_visibility": {
        k: {b: round(v["literal_evidence_visibility_rate"], 4)
            for b, v in system.items()}
        for k, system in result["literal_answerability"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
