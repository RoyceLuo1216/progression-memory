"""Factorial memory experiment and shared QA-context export on pinned MEME.

Cross serialization (verbose/compact) with history policy (current/history/
adaptive), and compare against an identically routed wiki. All ranks use
the same neural+BM25 fusion except explicit raw retrieval ablations.
The actual joined context (including separators) is budgeted. Empty retrieval
counts as failure rather than dropping a question. Oracle writers remain
explicitly labeled. No task labels or answers are used to select context.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.benchmarks.meme_evidence_retrieval import (
    MODEL_REPO, REVISION, build_episode, gold_evidence, task_labelled_questions,
    ranked_bm25,
)
from progression_memory.benchmarks.meme_v2_oracle_wiki import (
    history_from_sessions, oracle_engine, target_atoms, TASKS,
)
from progression_memory.benchmarks.meme_final_hybrid import (
    NeuralStore, EMBEDDING_MODEL, EMBEDDING_REVISION, docs_as_of, fusion,
    wants_history,
)

BUDGETS=(400,800,1600,3200,6400)
RAW=("raw_bm25","raw_dense","raw_bm25_dense","raw_bm25_dense_recency")
POLICIES=("current","history","adaptive")
ARMS=RAW+tuple(f"v2_{fmt}_{policy}" for fmt in ("verbose","compact")
              for policy in POLICIES)+tuple(f"wiki_{policy}" for policy in POLICIES)
QA_ARMS=("raw_dense","raw_bm25_dense_recency","wiki_history","wiki_adaptive",
         "v2_verbose_history","v2_compact_history","v2_compact_adaptive")
QA_BUDGETS=BUDGETS


def pack(ranked,budget):
    chosen=[]
    used=0
    for doc in ranked:
        cost=len(doc.text)+bool(chosen)
        if used+cost<=budget:
            chosen.append(doc)
            used+=cost
    context="\n".join(d.text for d in chosen)
    assert len(context)==used<=budget
    return context,chosen


def layouts(groups,query):
    history=wants_history(query)
    out={name:groups["raw"] for name in RAW}
    for fmt,prefix in (("verbose","v2_"),("compact","compact_")):
        for policy in POLICIES:
            view=("history" if history else "current") if policy=="adaptive" else policy
            out[f"v2_{fmt}_{policy}"]=groups[prefix+view]
    for policy in POLICIES:
        view=("history" if history else "current") if policy=="adaptive" else policy
        out[f"wiki_{policy}"]=groups["wiki_"+view]
    return out


def ranking(arm,docs,query,store):
    if arm=="raw_bm25":return ranked_bm25(docs,query)
    if arm=="raw_dense":return store.dense_rank(docs,query)
    return fusion(docs,query,store,recency=arm=="raw_bm25_dense_recency")


def sample_ids(episodes,n):
    """Deterministic stratified sample, across all six task types."""
    groups=defaultdict(list)
    for ep in episodes:
        for timing,pos,i,q in task_labelled_questions(ep):
            if q.get("question"):
                key=(str(ep["episode_id"]),f"{timing}:{int(pos)}:{i}")
                groups[q.get("task_type","?")].append(key)
    if not n:return {key for keys in groups.values() for key in keys}
    selected=set()
    for task,keys in sorted(groups.items()):
        keys.sort(key=lambda k:hashlib.sha256(("meme-factorial-v1:"+repr(k)).encode()).hexdigest())
        selected.update(keys[:n])
    return selected


def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:
        for metric in ("value_visibility","source_recall","all_values"):
            if metric not in r:continue
            for task in ("ALL",r["task"]):
                grouped[(metric,r["budget"],r["arm"],task)].append(r[metric])
    return [{"metric":m,"budget":b,"arm":a,"task":t,"n":len(v),
             "mean":sum(v)/len(v)} for (m,b,a,t),v in sorted(grouped.items())]


def evaluate(episodes,output,qa_per_task):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    sample=sample_ids(episodes,qa_per_task)
    store=NeuralStore(EMBEDDING_MODEL)
    rows=[]
    qa_count=0
    start=time.monotonic()
    with (output/"qa_contexts.jsonl").open("w") as qa:
        for ep in episodes:
            episode=str(ep["episode_id"])
            raw,idx,_,_=build_episode(ep)
            facts=history_from_sessions(ep)
            full=oracle_engine(facts,ep["episode_id"],include_history=True)
            current=oracle_engine(facts,ep["episode_id"],include_history=False)
            snapshots={}
            qs=list(task_labelled_questions(ep))
            texts=[]
            for timing,pos,i,q in qs:
                cutoff=min(len(ep["sessions"]),int(pos))
                if cutoff not in snapshots:
                    snapshots[cutoff]=docs_as_of(ep,raw,facts,full,current,cutoff)
                    texts.extend(d.text for docs in snapshots[cutoff].values() for d in docs)
                texts.append(q.get("question",""))
            store.prepare(texts)
            for timing,pos,i,q in qs:
                query=q.get("question","")
                if not query:continue
                qid=f"{timing}:{int(pos)}:{i}"
                cutoff=min(len(ep["sessions"]),int(pos))
                task=q.get("task_type","?")
                required,_,_=gold_evidence(ep,idx,q,cutoff)
                atoms=target_atoms(q) if task in TASKS else []
                candidates=layouts(snapshots[cutoff],query)
                rankings={arm:ranking(arm,docs,query,store) for arm,docs in candidates.items()}
                for b in BUDGETS:
                    for arm,ranked in rankings.items():
                        context,chosen=pack(ranked,b)
                        row={"episode":episode,"qid":qid,"task":task,"arm":arm,
                             "budget":b,"context_chars":len(context),
                             "history_route":wants_history(query)}
                        if atoms:
                            hits=[v.casefold() in context.casefold() for v in atoms]
                            row["value_visibility"]=sum(hits)/len(hits)
                            row["all_values"]=int(all(hits))
                        if required and arm in RAW:
                            row["source_recall"]=len({d.id for d in chosen}&required)/len(required)
                        rows.append(row)
                        if (episode,qid) in sample and arm in QA_ARMS and b in QA_BUDGETS:
                            # Labels stored separately from the prompt-bearing fields.
                            record={**row,"question":query,"context":context,
                                "source_ids":[d.id for d in chosen],
                                "reference":{"question_annotation":q,"atoms":atoms}}
                            qa.write(json.dumps(record,ensure_ascii=False)+"\n")
                            qa_count+=1
            # Release vectors per episode: no cross-user retrieval or unbounded cache.
            store.vectors.clear()
            print("episode",episode,"rows",len(rows),flush=True)
    result={"protocol":"meme-factorial-v1","dataset":MODEL_REPO,
            "dataset_revision":REVISION,"encoder":EMBEDDING_MODEL,
            "encoder_revision":EMBEDDING_REVISION,"budgets":BUDGETS,
            "arms":ARMS,"qa_arms":QA_ARMS,"qa_contexts":qa_count,
            "episodes":len(episodes),"elapsed_seconds":time.monotonic()-start,
            "embedding_seconds":store.encode_seconds,"summary":summarize(rows),
            "rows":rows,"oracle_memory":True}
    (output/"factorial.json").write_text(json.dumps(result,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/factorial")
    p.add_argument("--qa-per-task",type=int,default=20,
                   help="questions per category; 0 exports all questions")
    p.add_argument("--max-episodes",type=int)
    a=p.parse_args()
    path=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
                         filename="meme_nofiller.json",revision=REVISION)
    episodes=json.loads(Path(path).read_text())
    if a.max_episodes:episodes=episodes[:a.max_episodes]
    evaluate(episodes,a.output,a.qa_per_task)

if __name__=="__main__":
    main()
