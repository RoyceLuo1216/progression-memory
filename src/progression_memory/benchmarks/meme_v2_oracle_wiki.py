"""MEME oracle-memory ablation with the unchanged production V2 state engine.

Gold-fact annotations provide *oracle writer input*, not model predictions.
Flat topic-summary and V2 state are given exactly the same annotated event
stream. Scoring measures literal gold value access in context, not answer
accuracy. Deletion and uncertainty questions are excluded from this metric.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from huggingface_hub import hf_hub_download

from progression_memory.benchmarks.meme_evidence_retrieval import (
    REVISION, MODEL_REPO, prefix, ranked_bm25,
    task_labelled_questions, build_episode,
)
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.v2.models import Observation, AttributePatch, AttributeOperation
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import WritePipeline, ScriptedWriter, WriterProposal, ProposedUpdate

BUDGETS=(400,800,1600,3200,6400)
TASKS=("Cas","Tr","Agg","ER")
SYSTEMS=("raw_bm25","wiki_current","wiki_history","v2_current_only","v2_oracle")


def history_from_sessions(ep):
    out=[]
    for si,session in enumerate(ep["sessions"]):
        for fi,fact in enumerate(session.get("gold_facts",[])):
            if not fact.get("conveyed",True) or not fact.get("entity"):
                continue
            ti=fact.get("conveyed_in_turn")
            turns=session.get("conversation",[])
            if not isinstance(ti,int) or ti<0 or ti>=len(turns):
                continue
            content=str(turns[ti].get("content",""))
            if not content.strip():
                continue
            val=fact.get("value")
            if val is None:continue
            out.append({"session":si,"fact_number":fi,
                        "entity":str(fact["entity"]),
                        "value":str(val),"text":content})
    return out


def oracle_engine(facts,episode_id, *, include_history=True):
    engine=StateEngine()
    proposals={}
    observations=[]
    first={}
    histories=defaultdict(list)
    for i,fact in enumerate(facts):
        entity=fact["entity"]
        value=fact["value"]
        if not histories[entity] or histories[entity][-1]!=value:
            histories[entity].append(value)
        oid=f"meme:{episode_id}:fact:{i}"
        timepoint=datetime(2023,1,1,tzinfo=UTC)+timedelta(
            days=fact["session"],microseconds=i)
        observations.append(Observation(id=oid,source_id=oid,
            subject_id=f"meme:{episode_id}",text=fact["text"],recorded_at=timepoint))
        patches=(AttributePatch(key="value",operation=AttributeOperation.SET,value=value),)
        if include_history:
            patches+=(AttributePatch(key="history",operation=AttributeOperation.SET,
                           value=list(histories[entity])),)
        if entity not in first:
            first[entity]=oid
            upd=ProposedUpdate(action="create",name=entity,
                               patches=patches,reason="annotated new state from a source turn")
        else:
            upd=ProposedUpdate(action="transition",
                 progression_id=f"v2:{first[entity]}:0",
                 patches=patches,reason="annotated source change")
        proposals[oid]=WriterProposal(updates=(upd,))
    writer=WritePipeline(engine,ScriptedWriter(proposals))
    for obs in observations:
        writer.process(obs)
    return engine


def wiki_docs(facts, cutoff, with_history):
    state=defaultdict(list)
    for fact in facts:
        if fact["session"]>=cutoff:continue
        ent=fact["entity"]
        val=fact["value"]
        if not state[ent] or state[ent][-1]!=val:
            state[ent].append(val)
    docs=[]
    for ent,history in state.items():
        text=f"Topic: {ent}. Current value: {history[-1]}."
        if with_history:
            text+=f" Historical values in order: {'; '.join(history)}."
        docs.append(RetrievalDocument(id=f"wiki:{ent}",text=text))
    return docs


def target_atoms(q):
    values=q.get("entity_values",{})
    if not isinstance(values,dict):
        return []
    result=[]
    for val in values.values():
        if isinstance(val,list):
            result.extend(str(x) for x in val if x is not None)
        elif val is not None:
            result.append(str(val))
    return list(dict.fromkeys(x for x in result if x and x.casefold()!="none"))


def evaluate(episodes):
    result=[]
    build_seconds=0.0
    build_errors=[]
    for ep in episodes:
        rawdocs,_facts,_matched,_annotated=build_episode(ep)
        facts=history_from_sessions(ep)
        start=time.perf_counter()
        try:
            engine=oracle_engine(facts,ep["episode_id"])
            current_engine=oracle_engine(facts,ep["episode_id"],include_history=False)
        except (ValueError,RuntimeError,KeyError,TypeError) as exc:
            build_errors.append({"episode":ep["episode_id"],"type":type(exc).__name__,
                                 "message":str(exc)[:250]})
            continue
        build_seconds+=time.perf_counter()-start
        for timing,position,i,q in task_labelled_questions(ep):
            task=q.get("task_type")
            if task not in TASKS:continue
            atoms=target_atoms(q)
            if not atoms:continue
            cutoff=min(len(ep["sessions"]),int(position))
            corpus=[doc for doc in rawdocs if int(doc.id.split(":")[0][1:])<cutoff]
            if not corpus:continue
            when=datetime(2023,1,1,tzinfo=UTC)+timedelta(days=cutoff)
            v2docs=list(build_documents(engine,subject_id=f"meme:{ep['episode_id']}",
                                        view="state",valid_at=when,known_at=when))
            current_v2docs=list(build_documents(current_engine,subject_id=f"meme:{ep['episode_id']}",
                                                 view="state",valid_at=when,known_at=when))
            configs={
                "raw_bm25":corpus,
                "wiki_current":wiki_docs(facts,cutoff,False),
                "wiki_history":wiki_docs(facts,cutoff,True),
                "v2_current_only":current_v2docs,
                "v2_oracle":v2docs,
            }
            question=q.get("question","")
            for system,docs in configs.items():
                if not docs:continue
                ranked=ranked_bm25(docs,question)
                for budget in BUDGETS:
                    selected,used=prefix(ranked,budget)
                    text="\n".join(x.text for x in selected).casefold()
                    scores=[int(atom.casefold() in text) for atom in atoms]
                    result.append({
                        "episode":ep["episode_id"],"domain":ep.get("domain"),
                        "question_id":f"{timing}:{i}","task_type":task,
                        "system":system,"budget":budget,
                        "gold_atoms":len(atoms),"gold_value_visibility":sum(scores)/len(scores),
                        "all_gold_values":int(all(scores)),
                        "used_chars":used,
                    })
    summary={}
    for b in BUDGETS:
        summary[str(b)]={}
        for s in SYSTEMS:
            rows=[r for r in result if r["budget"]==b and r["system"]==s]
            summary[str(b)][s]={
                "n":len(rows),
                "mean_value_visibility":round(sum(x["gold_value_visibility"] for x in rows)/len(rows),5) if rows else None,
                "all_gold_values":round(sum(x["all_gold_values"] for x in rows)/len(rows),5) if rows else None,
                "avg_chars":round(sum(x["used_chars"] for x in rows)/len(rows),1) if rows else None,
                "by_task":{t:{
                    "n":len(tr:=[r for r in rows if r["task_type"]==t]),
                    "mean_value_visibility":round(sum(x["gold_value_visibility"] for x in tr)/len(tr),5) if tr else None,
                } for t in TASKS},
            }
    return {"protocol":"meme-100-episode-oracle-state-vs-topic-wiki-v1",
            "dataset":{"episodes":len(episodes),"questions_scored":len({
                (r["episode"],r["question_id"]) for r in result}),
                "oracle_writer_errors":build_errors,"oracle_v2_build_seconds":build_seconds},
            "source":{"hf_repo":MODEL_REPO,"revision":REVISION},
            "summary":summary,"rows":result,
            "limitations":[
                "Oracle annotated gold_facts used to WRITE topic-wiki and V2 state: upper-bound construction, not real model writer performance.",
                "Gold answer atoms drawn from task entity_values are used ONLY for visibility scoring, not memory construction or retrieval.",
                "This is exact value accessibility, not actual question-answer accuracy or semantic correctness.",
                "No model generation, inference cost or hidden decision-grading.",
                "Deletion and absence tasks intentionally excluded because finding a past value is not correct negative inference.",
                "The current-topic wiki and V2 differ in serialization and history format, and hence index overhead.",
                "Production V2 engine and read_path unchanged."
            ]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    src=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
        filename="meme_nofiller.json",revision=REVISION)
    eps=json.loads(Path(src).read_text())
    report=evaluate(eps)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"dataset":report["dataset"],"at_1600":{
        k:{name:v for name,v in x.items() if name in ("n","mean_value_visibility")}
        for k,x in report["summary"]["1600"].items()}},indent=2))


if __name__=="__main__":
    main()
