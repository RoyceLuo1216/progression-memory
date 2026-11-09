"""Tiny, real Qwen3.5-0.8B four-arm longitudinal planning pilot.

One founder persona, one checkpoint, no gold sent to the model. This is a
smoke-sized paired comparison, not a statistically meaningful benchmark.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from scripts.qwen35_cpu_smoke import LocalQwen, MODEL
from progression_memory.benchmarks.v2_decision_continuity import make_fixture
from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.benchmarks.v2_planner_evaluation import (
    lexical_semantic_rank, take_budget, score_output, PLANNER_SYSTEM,
)
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.v2.model_writer import ModelWriter
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import WritePipeline


def json_result(raw):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {"parse_error": raw[:2000]}


def main():
    start = time.monotonic()
    fixture, questions = make_fixture(distractors=0)
    # Phase 4: one real late-stage reassessment after a prior internship,
    # research period and offer. All systems see the same source prefix.
    q = next(x for x in questions if x["id"] == "decision-founder-4")
    horizon = datetime.fromisoformat(q["as_of"])
    obs = [x for x in fixture.observations
           if x.subject_id == q["subject_id"] and x.recorded_at <= horizon]
    trace = TraceFixture(observations=obs,
                         proposals={x.id:fixture.proposals[x.id] for x in obs},
                         questions=[])
    oracle = prepare(trace)
    backend = LocalQwen()
    live = StateEngine()
    pipeline = WritePipeline(live, ModelWriter(backend))
    writer_log = []
    for o in obs:
        try:
            response = pipeline.process(o)
            writer_log.append({"id":o.id,"status":"success",
                               "result":str(response)[:250]})
        except (ValueError,TypeError,RuntimeError,KeyError) as exc:
            writer_log.append({"id":o.id,"status":"failed",
                               "error":type(exc).__name__,
                               "message":str(exc)[:300]})
    source_docs = [RetrievalDocument(id=o.id,text=o.text) for o in obs]
    ranked = lexical_semantic_rank(source_docs,q["prompt"])
    rag,_ = take_budget(ranked, 1600)
    history = "\n".join(o.text for o in obs)
    summary_raw = backend.generate_messages([
        {"role":"system","content":"Summarize this user's CURRENT long-term career goal, current priority and active constraint, respecting explicit revisions, in a compact JSON object. Do not treat hypothetical possibilities as decisions."},
        {"role":"user","content":history}
    ])
    summary = summary_raw[:500]
    contexts = {
        "hybrid_rag":rag,
        "hybrid_rag_summary":(summary+"\n"+rag)[:1600],
    }
    for name, engine in (("v2_oracle",oracle),("v2_model",live)):
        docs = list(build_documents(engine, subject_id=q["subject_id"],
                        view="state",valid_at=horizon,known_at=horizon))
        context,_ = take_budget(lexical_semantic_rank(docs,q["prompt"]),1600) if docs else ("",0)
        contexts[name]=context
    rows=[]
    for name, context in contexts.items():
        raw=backend.generate_messages([
            {"role":"system","content":PLANNER_SYSTEM},
            {"role":"user","content":"MEMORY:\n"+context+"\n\nDECISION:\n"+
             q["prompt"]+" Return only JSON."},
        ])
        answer=json_result(raw)
        try:
            scores=score_output(answer,q["gold"])
        except (TypeError,KeyError,ValueError):
            scores={}
        rows.append({"architecture":name,"context_chars":len(context),
                     "output":answer,"scores":scores})
    report={"model":MODEL,
            "protocol":"tiny-four-arm-qwen-smoke-v1",
            "personas":1,"checkpoints":1,"observations":len(obs),
            "context_char_budget":1600,
            "writer_updates":writer_log,
            "summary_model_output":json_result(summary_raw),
            "arms":rows,"calls":backend.calls,
            "total_elapsed_sec":round(time.monotonic()-start,2),
            "limitations":[
                "One checkpoint, no confidence interval; cannot support general superiority.",
                "Actual Qwen writer errors remain failures, never patched using oracle data.",
                "Topic summary is generated once for this checkpoint, not maintained incrementally.",
                "Hybrid retrieval uses BM25+TF-IDF cosine RRF, not neural embeddings.",
                "Plan-quality score_output contains exact-value and structural proxies, not independent judgment."
            ]}
    out=Path("results/qwen35_four_arm_tiny.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"arms":[{"architecture":r["architecture"],
                               "scores":r["scores"]} for r in rows],
                      "writer_failures":sum(x["status"]=="failed" for x in writer_log),
                      "seconds":report["total_elapsed_sec"]},indent=2))


if __name__=="__main__":
    main()
