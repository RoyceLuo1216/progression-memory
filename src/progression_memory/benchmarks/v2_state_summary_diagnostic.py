"""Frozen V2-vs-simple-memory diagnostic: actual V2 engine, no model inference.

This evaluation deliberately separates memory-content accessibility from
generated decision quality. Scripted V2 writes are an oracle upper bound.
The rolling summary is a simple deterministic explicit-field extractor, NOT
a model-generated wiki. Both use only observations available at the cutoff.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from progression_memory.benchmarks.v2_decision_continuity import make_fixture
from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.writer_pipeline import WriterProposal

PROMPT = "What goal, priority and constraint should guide the next career decision?"
BUDGETS = (400, 800, 1600)
SYSTEMS = ("bm25", "recency", "bm25_recency", "flat_summary", "v2_state")
FIELDS = ("long_term_goal", "priority", "constraint")


def make_variant(*, remove_goal_repetition: bool, new_tail: int, distractors: int = 16):
    fixture, checkpoints = make_fixture(distractors=distractors)
    observations = []
    proposals = dict(fixture.proposals)
    for obs in fixture.observations:
        text = obs.text
        if remove_goal_repetition and "-update-" in obs.id:
            text = re.sub(r" Long-term goal remains [^.]+\.", "", text)
        observations.append(obs.model_copy(update={"text": text}))
    if new_tail:
        for checkpoint in checkpoints:
            subject = checkpoint["subject_id"]
            horizon = checkpoint["as_of"]
            from datetime import datetime
            baseline = datetime.fromisoformat(horizon)
            # Future observations after each checkpoint's authoritative update,
            # strictly before the next stage (65 days later).
            for idx in range(new_tail):
                oid = f"{checkpoint['id']}-later-{idx}"
                obs_type = type(fixture.observations[0])
                observations.append(obs_type(
                    id=oid, source_id=oid, subject_id=subject,
                    text="Daily note about the library, weather and lunch. "
                         "No decisions about career goals or priorities.",
                    recorded_at=baseline+timedelta(days=idx+1),
                ))
                proposals[oid] = WriterProposal(updates=())
            checkpoint["as_of"] = (baseline+timedelta(days=new_tail+2)).isoformat()
    observations.sort(key=lambda x: (x.recorded_at, x.id))
    data = TraceFixture(observations=observations, proposals=proposals, questions=[])
    return data, checkpoints


def summarize(observations):
    """Incremental, explicit-field summary, never uses gold labels."""
    state = {}
    for obs in sorted(observations, key=lambda x:(x.recorded_at, x.id)):
        text = obs.text
        rules = (
            ("long_term_goal", r"My enduring goal is ([^.]+)\."),
            ("long_term_goal", r"Long-term goal remains ([^.]+)\."),
            ("priority", r"Current top priority: ([^.]+)\."),
            ("constraint", r"Active constraint: ([^.]+)\."),
        )
        for key, pattern in rules:
            match = re.search(pattern, text)
            if match:
                state[key] = match.group(1)
    return "Current goal: {long_term_goal}. Priority: {priority}. Constraint: {constraint}.".format(
        **{key: state.get(key, "") for key in FIELDS}
    )


def order_bm25(docs, query):
    return BM25Index(docs).retrieve(query, len(docs))


def order_recency(docs):
    return sorted(docs, key=lambda x: (x.timestamp or "", x.id), reverse=True)


def order_fusion(docs, query):
    lex = order_bm25(docs, query)
    rec = order_recency(docs)
    ranks_a = {d.id: i for i,d in enumerate(lex)}
    ranks_b = {d.id: i for i,d in enumerate(rec)}
    return sorted(docs, key=lambda d: (0.5*ranks_a[d.id]+0.5*ranks_b[d.id], d.id))


def context_prefix(docs, budget):
    used, text = 0, []
    for doc in docs:
        if used + len(doc.text) > budget:
            break
        text.append(doc.text)
        used += len(doc.text)
    return "\n".join(text), used


def cluster_ci(rows, system, baseline="flat_summary", metric="field_visibility",
               samples=2000, seed=42):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["persona"],row["checkpoint"])][row["system"]] = row[metric]
    by_persona=defaultdict(list)
    for (persona,_), values in pairs.items():
        if system in values and baseline in values:
            by_persona[persona].append(values[system]-values[baseline])
    people = sorted(by_persona)
    if len(people)<2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selection=[rng.choice(people) for _ in people]
        values=[v for p in selection for v in by_persona[p]]
        draws.append(sum(values)/len(values))
    draws.sort()
    vals=[v for p in people for v in by_persona[p]]
    return {"difference":round(sum(vals)/len(vals),5),
            "ci95":[round(draws[int(.025*samples)],5),
                    round(draws[int(.975*samples)-1],5)],
            "persona_clusters":len(people)}


def evaluate_variant(*, variant, remove_goal_repetition, new_tail, distractors=16):
    fixture, checkpoints = make_variant(
        remove_goal_repetition=remove_goal_repetition,
        new_tail=new_tail, distractors=distractors)
    engine = prepare(fixture)
    rows = []
    for checkpoint in checkpoints:
        from datetime import datetime
        cutoff = datetime.fromisoformat(checkpoint["as_of"])
        subject = checkpoint["subject_id"]
        visible=[o for o in fixture.observations
                 if o.subject_id==subject and o.recorded_at<=cutoff]
        evidence=[RetrievalDocument(id=o.id,text=o.text,
                                   timestamp=o.recorded_at.isoformat()) for o in visible]
        v2docs=list(build_documents(engine, view="state",subject_id=subject,
                                    valid_at=cutoff,known_at=cutoff))
        layouts = {
            "bm25":order_bm25(evidence,PROMPT),
            "recency":order_recency(evidence),
            "bm25_recency":order_fusion(evidence,PROMPT),
            "flat_summary":[RetrievalDocument(id="summary",
                                              text=summarize(visible))],
            "v2_state":order_bm25(v2docs,PROMPT),
        }
        for budget in BUDGETS:
            for system, docs in layouts.items():
                context, used = context_prefix(docs,budget)
                correct = sum(value in context
                              for value in checkpoint["gold"].values())
                rows.append({
                    "variant":variant,"budget":budget,
                    "persona":subject,"checkpoint":checkpoint["id"],
                    "system":system,
                    "field_visibility":correct/3,
                    "complete_context":int(correct==3),
                    "context_chars":used,
                })
    report = {}
    for budget in BUDGETS:
        report[str(budget)]={}
        for system in SYSTEMS:
            values=[r for r in rows if r["budget"]==budget and r["system"]==system]
            report[str(budget)][system]={
                "field_visibility":round(sum(r["field_visibility"] for r in values)/len(values),5),
                "complete_context":round(sum(r["complete_context"] for r in values)/len(values),5),
                "avg_chars":round(sum(r["context_chars"] for r in values)/len(values),1),
                "vs_flat_summary":cluster_ci([r for r in rows if r["budget"]==budget],
                                             system),
            }
    return {"variant":variant,"dataset":{"personas":6,"checkpoints":len(checkpoints),
            "observations":len(fixture.observations),"new_tail":new_tail,
            "goal_repeated":not remove_goal_repetition},
            "budgets":report,"per_checkpoint":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    experiments=[
        evaluate_variant(variant="original",remove_goal_repetition=False,new_tail=0),
        evaluate_variant(variant="no_goal_repetition",remove_goal_repetition=True,new_tail=0),
        evaluate_variant(variant="no_repetition_20_later",remove_goal_repetition=True,new_tail=20),
    ]
    report={"protocol":"v2-state-vs-summary-diagnostic-v1",
            "experiments":experiments,
            "limitations":[
                "Real production StateEngine and read_path, but scripted/oracle writer proposals.",
                "Flat summary uses deterministic regex field extraction, not learned summary writing.",
                "Exact substring visibility, not decisions, grounded plans, or preference inference.",
                "Six highly templated personas, not independent real user histories.",
                "No write/model inference costs; these must be measured in downstream trials.",
                "Tight whole-document budget with stop-on-overflow policy; no truncation."
            ]}
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    for exp in experiments:
        print(exp["variant"],"obs",exp["dataset"]["observations"])
        print(json.dumps({s:exp["budgets"]["400"][s]["field_visibility"]
                          for s in SYSTEMS},sort_keys=True))


if __name__=="__main__":
    main()
