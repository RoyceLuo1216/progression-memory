"""Limited real-answer MEME QA: identical Qwen model across four memory views.

30 deterministically selected held-out questions from 100 public episodes;
all outputs are saved. Historical answers scored by chronological gold-value
coverage. Absence/deletion outputs are saved for manual grading, not
automatically assigned a misleading lexical 'correctness' score.

Oracle factual annotations are used for constructing wiki and V2, NOT for
prompts, retrieval routing or the Qwen answering model.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from scripts.qwen35_cpu_smoke import LocalQwen, MODEL
from progression_memory.benchmarks.meme_evidence_retrieval import (
    MODEL_REPO, REVISION, build_episode, ranked_bm25, task_labelled_questions,
)
from progression_memory.benchmarks.meme_v2_oracle_wiki import (
    history_from_sessions, oracle_engine, wiki_docs,
)
from progression_memory.benchmarks.meme_final_hybrid import (
    budget_pack, to_compact, wants_history,
)
from progression_memory.v2.read_path import build_documents

BUDGET=1600
TASK_TARGETS={"Tr":12,"Cas":8,"Abs":5,"Del":5}
ARMS=("raw_bm25","wiki_history","v2_oracle_history","compact_v2_adaptive")


def answer(backend,question,context):
    system=("You answer personal-memory questions only from supplied memory. "
            "Do not invent missing facts. If the answer is unknown, say unknown. "
            "For historical sequence questions, list values in chronological order. "
            "Return the concise answer only, not JSON or an explanation.")
    msgs=[
        {"role":"system","content":system},
        {"role":"user","content":"MEMORY:\n"+context+"\n\nQUESTION:\n"+question},
    ]
    inputs=backend.processor.apply_chat_template(
        msgs,add_generation_prompt=True,tokenize=True,
        return_dict=True,return_tensors="pt",enable_thinking=False)
    started=time.monotonic()
    with torch.inference_mode():
        out=backend.model.generate(
            **inputs,max_new_tokens=128,do_sample=False,
            pad_token_id=backend.processor.tokenizer.eos_token_id)
    raw=backend.processor.decode(
        out[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
    return raw,{"seconds":round(time.monotonic()-started,2),
                "input_tokens":int(inputs["input_ids"].shape[-1]),
                "output_tokens":int(out.shape[-1]-inputs["input_ids"].shape[-1])}


def ordered_values(answer_text,expected_values):
    # Only meaningful for Tr: whether ALL expected historical values
    # appear in the specified chronology, allowing connecting prose.
    if not expected_values:return False
    text=answer_text.casefold()
    position=0
    for value in expected_values:
        idx=text.find(str(value).casefold(),position)
        if idx<0:return False
        position=idx+len(str(value))
    return True


def sample_questions(episodes):
    buckets=defaultdict(list)
    for ep in episodes:
        for timing,cutoff,index,q in task_labelled_questions(ep):
            task=q.get("task_type")
            if task in TASK_TARGETS and q.get("question"):
                buckets[task].append((ep,timing,cutoff,index,q))
    selection=[]
    for task,n in TASK_TARGETS.items():
        bucket=buckets[task]
        # Evenly spaced deterministic sample, distributed over episodes.
        chosen=[bucket[int((i+0.5)*len(bucket)/n)] for i in range(n)]
        selection.extend(chosen)
    return selection


def context_for(ep,cutoff,question):
    raw,_facts,_,_=build_episode(ep)
    facts=history_from_sessions(ep)
    engine=oracle_engine(facts,ep["episode_id"],include_history=True)
    current=oracle_engine(facts,ep["episode_id"],include_history=False)
    corpus=[d for d in raw if int(d.id.split(":")[0][1:])<cutoff]
    when=datetime(2023,1,1,tzinfo=UTC)+timedelta(days=cutoff)
    options={"subject_id":f"meme:{ep['episode_id']}",
             "view":"state","valid_at":when,"known_at":when}
    history=list(build_documents(engine,**options))
    compact_source=list(build_documents(
        engine if wants_history(question) else current,**options))
    docs={
        "raw_bm25":corpus,
        "wiki_history":wiki_docs(facts,cutoff,True),
        "v2_oracle_history":history,
        "compact_v2_adaptive":to_compact(compact_source,
                                     include_history=wants_history(question)),
    }
    output={}
    for arm,group in docs.items():
        ranked=ranked_bm25(group,question)
        packed,used=budget_pack(ranked,BUDGET)
        output[arm]={"context":"\n".join(d.text for d in packed),
                     "chars":used,"ids":[d.id for d in packed]}
    return output


def run():
    source=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
             filename="meme_nofiller.json",revision=REVISION)
    episodes=json.loads(Path(source).read_text())
    selection=sample_questions(episodes)
    # Build context BEFORE loading Qwen to keep memory use down.
    rows=[]
    for ep,timing,pos,i,q in selection:
        cutoff=min(int(pos),len(ep["sessions"]))
        rows.append({
            "episode":ep["episode_id"],"question_id":f"{timing}:{i}",
            "task_type":q["task_type"],"question":q["question"],
            "expected_answer":q.get("expected_answer") or q.get("gold_answer"),
            "reference_values":q.get("entity_values"),
            "contexts":context_for(ep,cutoff,q["question"]),
        })
    backend=LocalQwen()
    started=time.monotonic()
    for idx,row in enumerate(rows,1):
        outputs={}
        for arm in ARMS:
            c=row["contexts"][arm]
            raw,stats=answer(backend,row["question"],c["context"])
            values=list(row["reference_values"].values()) if isinstance(
                row["reference_values"],dict) else []
            expected_order=[]
            for v in values:
                if isinstance(v,list):expected_order.extend(v)
                elif v is not None:expected_order.append(v)
            outputs[arm]={
                "answer":raw,
                "stats":stats,
                "historical_order_correct":(
                    ordered_values(raw,expected_order)
                    if row["task_type"]=="Tr" else None),
                "exact_reference_match":(
                    re.sub(r"\W+","",raw.casefold())==
                    re.sub(r"\W+","",str(row["expected_answer"]).casefold())),
            }
        row["outputs"]=outputs
        if idx%5==0:
            # Persist partial results against workflow interruptions.
            save(rows,started,completed=idx)
            print("Answered",idx,"of",len(rows),flush=True)
    return save(rows,started,completed=len(rows))


def save(rows,started,completed):
    summary={}
    for task in TASK_TARGETS:
        sub=[r for r in rows[:completed] if r["task_type"]==task]
        summary[task]={"n":len(sub)}
        for arm in ARMS:
            if task=="Tr":
                summary[task][arm]={
                    "chronology_accuracy":sum(bool(r["outputs"][arm]["historical_order_correct"])
                                              for r in sub)/len(sub) if sub else None}
            else:
                summary[task][arm]={
                    "literal_exact":sum(bool(r["outputs"][arm]["exact_reference_match"])
                                        for r in sub)/len(sub) if sub else None}
    report={"protocol":"meme-Qwen3.5-0.8B-downstream-qa-30-v1",
            "source":{"hf_repo":MODEL_REPO,"revision":REVISION},
            "model":MODEL,"budget_chars":BUDGET,"completed":completed,
            "total_seconds":round(time.monotonic()-started,2),
            "summary":summary,"rows":rows[:completed],
            "limitations":[
                "30 public, deterministic examples; exploratory, not population-level conclusions.",
                "Oracle annotated facts construct memory; not production model-writer performance.",
                "Chronology exact ordered-value score for Tr only; other exact string scores are conservative.",
                "Abs/Del require manual semantic scoring; no invalid lexical label used as the primary claim.",
                "Four arms use BM25 context ranking; full neural RAG is evaluated separately.",
                "Small Qwen 0.8B reader is limited; no independent LLM judge.",
            ]}
    p=Path("results/meme_qwen_qa_30.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    result=run()
    print(json.dumps({"completed":result["completed"],
                      "summary":result["summary"]},indent=2))
