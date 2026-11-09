"""Paired longitudinal planning evaluation; keeps production V2 unchanged.

Live execution requires OPENAI_API_KEY. No synthetic model scores are silently
substituted for actual writer or planner inference.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from progression_memory.benchmarks.v2_decision_continuity import make_fixture
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.model_writer import ModelWriter
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import WritePipeline

FIELDS = ("long_term_goal", "priority", "constraint")
SYSTEMS = ("hybrid_rag", "hybrid_rag_summary", "v2_oracle", "v2_model")
PLANNER_SYSTEM = (
    "You are a decision-support assistant. Make a recommendation from the "
    "provided MEMORY only. Explicitly state the current goal, priority, "
    "constraint, trade-offs, and three feasible next steps. Never assume "
    "historical suggestions override explicit current decisions. "
    "Return a JSON object with keys goal, priority, constraint, recommendation, "
    "tradeoffs (array), next_steps (array), caveats (array)."
)


class OpenAIModel:
    def __init__(self, model: str):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for live model inference")
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.elapsed = 0.0

    def generate_messages(self, messages, json_schema=None):
        start = time.perf_counter()
        kwargs = {"model": self.model, "messages": messages}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        self.elapsed += time.perf_counter() - start
        self.calls += 1
        if response.usage:
            self.input_tokens += response.usage.prompt_tokens or 0
            self.output_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content

    def snapshot(self):
        return {"model": self.model, "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "wall_seconds": round(self.elapsed, 4)}


def lexical_semantic_rank(docs, query):
    """BM25 + TF-IDF cosine via reciprocal rank fusion.

    Lightweight dependency-free hybrid semantic approximation (lexical
    TF-IDF, NOT neural embeddings); the limitation is part of reporting.
    """
    import math
    from collections import Counter

    from progression_memory.evaluation.retrieval import tokenize
    lexical = BM25Index(docs).retrieve(query, len(docs))
    query_tf = Counter(tokenize(query))
    doc_tf = [Counter(tokenize(doc.text)) for doc in docs]
    scores = []
    for doc, tf in zip(docs, doc_tf):
        common = sum(query_tf[t] * freq for t, freq in tf.items())
        denom = math.sqrt(sum(x*x for x in tf.values())) * math.sqrt(
            sum(x*x for x in query_tf.values()))
        scores.append((common / denom if denom else 0.0, doc.id))
    semantic = [s[1] for s in sorted(scores, key=lambda x: (-x[0], x[1]))]
    ranks_lex = {doc.id: i for i, doc in enumerate(lexical)}
    ranks_sem = {id_: i for i, id_ in enumerate(semantic)}
    return sorted(docs, key=lambda d: (-(1/(60+ranks_lex[d.id]) +
                                              1/(60+ranks_sem[d.id])), d.id))


def take_budget(documents, budget):
    pieces, used = [], 0
    for doc in documents:
        if used + len(doc.text) > budget:
            break
        pieces.append(doc.text)
        used += len(doc.text)
    return "\n".join(pieces), used


def personas_and_checkpoints(distractors=2):
    fixture, checkpoints = make_fixture(distractors=distractors)
    # Counterfactual copies alter precisely one late-stage priority *in the
    # observed history*. All other events and hidden constraints remain same.
    # Both variants are separate evaluation trajectories; never mix histories.
    return fixture, checkpoints


def state_trace(fixture, *, live_writer=None):
    if live_writer is None:
        return prepare(fixture), {"model_calls": 0, "writer_failures": 0}
    engine = StateEngine()
    pipeline = WritePipeline(engine, ModelWriter(live_writer))
    failures = []
    for obs in fixture.observations:
        try:
            pipeline.process(obs)
        except (ValueError, TypeError, RuntimeError, KeyError) as error:
            failures.append({"observation": obs.id,
                             "type": type(error).__name__, "message": str(error)[:160]})
    return engine, {"model_calls": live_writer.calls, "writer_failures": len(failures),
                    "writer_failure_examples": failures[:10]}


def score_output(payload, gold):
    """Check factual field restatement and structural output only.

    These are proxies, not grounded decision-quality scores. A field mismatch
    is NOT by itself proof of an actual hard-constraint violation or obsolete
    priority reliance.
    """
    values = {
        key: str(payload.get("goal" if key == "long_term_goal" else key, ""))
        .strip().casefold() == str(gold[key]).strip().casefold()
        for key in FIELDS
    }
    return {
        "goal_fidelity": int(values["long_term_goal"]),
        "preference_fidelity": int(values["priority"]),
        "constraint_fidelity": int(values["constraint"]),
        "constraint_field_mismatch": int(not values["constraint"]),
        "priority_field_mismatch": int(not values["priority"]),
        "two_tradeoff_items": int(
            isinstance(payload.get("tradeoffs"), list)
            and len(payload.get("tradeoffs", [])) >= 2),
        "three_next_steps": int(
            isinstance(payload.get("next_steps"), list)
            and len(payload.get("next_steps", [])) >= 3),
    }


def cluster_ci(rows, a, b, metric, samples=2000, seed=42):
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["persona"], row["id"])][row["system"]] = row["scores"][metric]
    person_diffs = defaultdict(list)
    for (persona, _), values in paired.items():
        if a in values and b in values:
            person_diffs[persona].append(values[a] - values[b])
    personas = sorted(person_diffs)
    if len(personas) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(personas) for _ in personas]
        diffs = [x for p in selected for x in person_diffs[p]]
        estimates.append(sum(diffs)/len(diffs))
    estimates.sort()
    raw = [x for p in personas for x in person_diffs[p]]
    return {"difference": sum(raw)/len(raw),
            "ci95": [estimates[int(0.025*samples)],
                     estimates[int(0.975*samples)-1]],
            "cluster_count": len(personas)}


def run(*, model, writer_model=None, budget=1600, distractors=2, max_cases=None):
    fixture, checkpoints = personas_and_checkpoints(distractors)
    if max_cases:
        checkpoints = checkpoints[:max_cases]
    oracle_engine, _ = state_trace(fixture)
    planner = OpenAIModel(model)
    writer = OpenAIModel(writer_model or model)
    live_engine, diagnostics = state_trace(fixture, live_writer=writer)
    output = []
    for q in checkpoints:
        cutoff = datetime.fromisoformat(q["as_of"])
        subject = q["subject_id"]
        docs = [RetrievalDocument(id=obs.id, text=obs.text)
                for obs in fixture.observations
                if obs.subject_id == subject and obs.recorded_at <= cutoff]
        for system in SYSTEMS:
            if system.startswith("hybrid_rag"):
                ranked = lexical_semantic_rank(docs, q["prompt"])
                context, used = take_budget(ranked, budget)
                if system == "hybrid_rag_summary":
                    # Summary created by a live model from the prefix, not gold.
                    prefix = "\n".join(d.text for d in docs)
                    summary = planner.generate_messages([
                        {"role": "system", "content": "Summarize current goals, explicit changes, commitments and constraints from the chronological history as JSON. Do not use hypothetical discussions as decisions."},
                        {"role": "user", "content": prefix}],
                    )
                    remaining = max(0, budget-len(summary))
                    context = summary[:budget] + ("\n" + take_budget(ranked, remaining)[0]
                                                   if remaining else "")
                    used = min(budget, len(context))
            else:
                engine = oracle_engine if system == "v2_oracle" else live_engine
                memory = list(build_documents(engine, subject_id=subject,
                    valid_at=cutoff, known_at=cutoff, view="state"))
                ranked = lexical_semantic_rank(memory, q["prompt"])
                context, used = take_budget(ranked, budget)
            answer_raw = planner.generate_messages([
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": "MEMORY:\n"+context+"\n\nDECISION:\n"+
                 q["prompt"]+"\nRespond as JSON."}])
            try:
                answer = json.loads(answer_raw)
                scores = score_output(answer, q["gold"])
            except (ValueError, TypeError):
                answer = {"unparseable": answer_raw[:2000]}
                scores = {k: 0 for k in score_output({}, q["gold"])}
            output.append({"system": system, "persona": subject, "id": q["id"],
                           "scores": scores, "context_chars": used,
                           "answer": answer})
    metrics = list(output[0]["scores"]) if output else []
    means = {s:{m:sum(r["scores"][m] for r in output if r["system"]==s) /
                    max(1,sum(r["system"]==s for r in output))
                 for m in metrics} for s in SYSTEMS}
    cis = {s:{m:cluster_ci(output, s, "hybrid_rag_summary", m)
              for m in metrics} for s in SYSTEMS if s!="hybrid_rag_summary"}
    return {"data": {"personas": 6, "checkpoints": len(checkpoints),
                     "distractors": distractors, "budget": budget},
            "means": means, "paired_persona_cluster_ci": cis, "writer": diagnostics,
            "cost_counters": {"planner": planner.snapshot(), "writer": writer.snapshot()},
            "rows": output,
            "limitations": [
                "Hybrid uses BM25+TF-IDF cosine RRF, not a neural embedding.",
                "Running summary is regenerated per checkpoint, not incrementally maintained.",
                "Writer trained/tested on template-like repeated histories.",
                "Gold exact-string scores cannot independently establish trade-off grounding or feasibility.",
                "Counterfactual preference-pair evaluation remains unimplemented.",
                "Token counts are measured; currency costs require provider prices."
            ]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--writer-model")
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=1600)
    p.add_argument("--distractors", type=int, default=2)
    p.add_argument("--max-cases", type=int)
    args = p.parse_args()
    result=run(model=args.model,writer_model=args.writer_model,budget=args.budget,
               distractors=args.distractors,max_cases=args.max_cases)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"means":result["means"],"cost_counters":result["cost_counters"]},
                     indent=2))


if __name__ == "__main__":
    main()
