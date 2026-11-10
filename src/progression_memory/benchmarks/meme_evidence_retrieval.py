"""MEME v1 released dataset: large-scale source-evidence retrieval benchmark.

Gold fact annotations are used exclusively to identify ground-truth evidence
turn IDs. Indexed retrieval content is raw session conversation text, never
gold_facts/entity maps/expected answers. This is NOT answer accuracy or writer
quality, and it does NOT test a deployed Progression V2 architecture.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument

MODEL_REPO = "meme-benchmark/MEME"
REVISION = "e14c4e685772014deb96f91dc7eddfbeb21d90d8"
BUDGETS = (400, 800, 1600, 3200, 6400)
SYSTEMS = ("bm25", "recency", "bm25_recency")
FOCUS = ("Cas", "Abs", "Del", "Tr", "Agg", "ER")


def ranked_bm25(docs, query):
    return BM25Index(docs).retrieve(query, len(docs))


def ranked_recency(docs):
    return list(reversed(docs))


def ranked_fusion(docs, query):
    lex = ranked_bm25(docs, query)
    rec = ranked_recency(docs)
    lr = {d.id:i for i,d in enumerate(lex)}
    rr = {d.id:i for i,d in enumerate(rec)}
    return sorted(docs, key=lambda d: (-(1/(60+lr[d.id])+1/(60+rr[d.id])), d.id))


def prefix(ranked, budget):
    selected, used = [], 0
    for doc in ranked:
        if used+len(doc.text)>budget:
            break
        selected.append(doc)
        used+=len(doc.text)
    return selected, used


def task_labelled_questions(episode):
    for timing in ("before","after"):
        holder=episode.get(f"{timing}_questions", {})
        if isinstance(holder,dict):
            holders=[holder]
        elif isinstance(holder,list):
            holders=holder
        else:
            continue
        for block in holders:
            if not isinstance(block,dict):
                continue
            position=block.get("position_after_session",len(episode["sessions"]))
            for i,q in enumerate(block.get("questions",[])):
                if isinstance(q,dict):
                    yield timing,position,i,q


def build_episode(episode):
    docs=[]
    # source turn mapping; uses position from recorded annotation, never labels
    facts=defaultdict(list)
    matched_value=0
    annotated=0
    for idx,session in enumerate(episode["sessions"]):
        turns=session.get("conversation") or []
        for ti,turn in enumerate(turns):
            content=str(turn.get("content",""))
            if not content.strip():
                continue
            docs.append(RetrievalDocument(id=f"s{idx}:t{ti}",text=content,
                                          timestamp=f"{idx:03d}:{ti:04d}"))
        for fact in session.get("gold_facts",[]):
            if not fact.get("conveyed",True):
                continue
            entity=fact.get("entity")
            if not entity:
                continue
            annotated+=1
            val=str(fact.get("value",""))
            ti=fact.get("conveyed_in_turn")
            # MEME annotates the turn; checking exact value guards incorrect joins.
            if not isinstance(ti,int) or ti<0 or ti>=len(turns):
                continue
            source=str(turns[ti].get("content",""))
            if val and val.casefold() in source.casefold():
                matched_value+=1
            did=f"s{idx}:t{ti}"
            if source.strip():
                facts[entity].append((idx,did))
    return docs,facts,matched_value,annotated


def gold_evidence(episode, facts, q, cutoff):
    targets=q.get("entity",[])
    if isinstance(targets,str):
        targets=[targets]
    task=q.get("task_type","?")
    if task in ("Cas","Abs"):
        # Cascade/source changes can be required to identify uncertainty.
        for edge in episode.get("dependency_edges_used",[]):
            if edge.get("target") in targets:
                targets=list(targets)+[edge.get("source")]
    needed=set()
    count=0
    for entity in set(targets):
        events=[(idx,did) for idx,did in facts.get(entity,[]) if idx<cutoff]
        if not events:
            continue
        count+=1
        if task=="Tr":
            needed.update(did for _,did in events)
        else:
            newest=max(idx for idx,_ in events)
            needed.update(did for idx,did in events if idx==newest)
    return needed,count,len(set(targets))


def ci(rows, a, b, budget, samples=1500):
    paired=defaultdict(dict)
    for r in rows:
        if r["budget"]==budget:
            paired[(r["episode"],r["question_id"])][r["system"]]=r["source_recall"]
    clusters=defaultdict(list)
    for (ep,_),vals in paired.items():
        if a in vals and b in vals:
            clusters[ep].append(vals[a]-vals[b])
    ps=sorted(clusters)
    if len(ps)<2:
        return None
    rng=random.Random(42)
    values=[]
    for _ in range(samples):
        sample=[rng.choice(ps) for _ in ps]
        d=[v for ep in sample for v in clusters[ep]]
        values.append(sum(d)/len(d))
    values.sort()
    obs=[v for ep in ps for v in clusters[ep]]
    return {"difference":round(sum(obs)/len(obs),5),
            "ci95":[round(values[int(.025*samples)],5),
                    round(values[int(.975*samples)-1],5)],
            "episode_clusters":len(ps)}


def evaluate(episodes, max_episodes=None):
    if max_episodes is not None:
        episodes=episodes[:max_episodes]
    rows=[]
    counts={"episodes":len(episodes),"questions_total":0,
            "questions_eligible":0,"annotation_value_matched":0,
            "annotation_count":0}
    for ep in episodes:
        docs,facts,matched,annotated=build_episode(ep)
        counts["annotation_value_matched"]+=matched
        counts["annotation_count"]+=annotated
        for timing,position,i,q in task_labelled_questions(ep):
            counts["questions_total"]+=1
            cutoff=min(len(ep["sessions"]),int(position))
            corpus=[d for d in docs if int(d.id.split(":")[0][1:])<cutoff]
            required,found,total=gold_evidence(ep,facts,q,cutoff)
            if not corpus or not required:
                continue
            counts["questions_eligible"]+=1
            query=q.get("question","")
            rankings={"bm25":ranked_bm25(corpus,query),
                      "recency":ranked_recency(corpus),
                      "bm25_recency":ranked_fusion(corpus,query)}
            for budget in BUDGETS:
                for system, ranked in rankings.items():
                    chosen,used=prefix(ranked,budget)
                    retrieved={d.id for d in chosen}
                    hits=len(retrieved & required)
                    rows.append({"episode":ep["episode_id"],
                                 "domain":ep.get("domain"),
                                 "question_id":f"{timing}:{i}",
                                 "timing":timing,"task_type":q.get("task_type"),
                                 "budget":budget,"system":system,
                                 "required_evidence":len(required),
                                 "source_recall":hits/len(required),
                                 "any_recall":int(hits>0),
                                 "all_recall":int(hits==len(required)),
                                 "context_chars":used,
                                 "target_entities_found":found,
                                 "target_entities_total":total})
    summary={}
    for budget in BUDGETS:
        summary[str(budget)]={}
        for system in SYSTEMS:
            rs=[r for r in rows if r["budget"]==budget and r["system"]==system]
            by_task={}
            for t in FOCUS:
                tr=[r for r in rs if r["task_type"]==t]
                by_task[t]={"n":len(tr),
                            "mean_source_recall":round(sum(x["source_recall"] for x in tr)/len(tr),5) if tr else None}
            summary[str(budget)][system]={
                "n":len(rs),
                "mean_source_recall":round(sum(x["source_recall"] for x in rs)/len(rs),5) if rs else None,
                "all_required_sources":round(sum(x["all_recall"] for x in rs)/len(rs),5) if rs else None,
                "mean_chars":round(sum(x["context_chars"] for x in rs)/len(rs),1) if rs else None,
                "by_task":by_task,
                "vs_bm25":ci(rows,system,"bm25",budget),
            }
    return {"protocol":"meme-real-conversation-evidence-retrieval-v1",
            "source":{"hf_repo":MODEL_REPO,"revision":REVISION,
                      "filename":"meme_nofiller.json"},
            "counts":counts,"summary":summary,"rows":rows,
            "limitations":[
                "Only raw-conversation evidence retrieval, not LLM-generated answers or V2 performance.",
                "Ground truth evidence IDs use annotated MEME gold_facts conveyed_in_turn and entity names, never indexed into retrieval documents.",
                "Dependency source annotations are used to select required evidence for cascade and absence tasks; this evaluates evidence availability, not causal inference.",
                "Gold-fact labels may have turn alignment errors; annotation value-match fraction is reported.",
                "Question positions are session cutoffs, with no future session evidence.",
                "Entire source-turn documents; stop-on-overflow matching repository retrieval protocol.",
                "Eligible questions require one or more annotated evidence turns.",
                "Before and after question IDs are local within each episode, and paired inference is clustered by episode."
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--max-episodes",type=int)
    args=p.parse_args()
    path=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
                         filename="meme_nofiller.json",revision=REVISION)
    eps=json.loads(Path(path).read_text())
    data=evaluate(eps,args.max_episodes)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({"counts":data["counts"],"summary_at_1600":{
        s:{k:v for k,v in x.items() if k in ("mean_source_recall","all_required_sources")}
        for s,x in data["summary"]["1600"].items()}},indent=2))


if __name__=="__main__":
    main()
