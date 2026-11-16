"""Matched-budget MEME: dense+BM25+recency RAG, wiki, and compact progression.

Real sentence-transformer vectors (no TF-IDF pretending to be dense).
Every method uses the same question-time cutoff and candidate budget.
Wiki/V2 are built from oracle source-fact annotations: upper-bound writers.
No gold question answer, entity labels or task category enters ranking/routing.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.benchmarks.meme_evidence_retrieval import (
    BUDGETS, MODEL_REPO, REVISION, build_episode, gold_evidence,
    task_labelled_questions, ranked_bm25,
)
from progression_memory.benchmarks.meme_v2_oracle_wiki import (
    TASKS, history_from_sessions, oracle_engine, target_atoms, wiki_docs,
)
from progression_memory.v2.read_path import build_documents

# This model provides actual learned semantic embeddings, not sparse SVD.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODES = (
    "raw_bm25", "raw_dense", "raw_bm25_dense",
    "raw_bm25_dense_recency", "wiki_current", "wiki_history",
    "v2_current", "v2_history", "v2_compact_adaptive",
    "wiki_v2_adaptive", "wiki_v2_hybrid_rag",
)
BUDGETS = (400, 800, 1600, 3200, 6400)
HISTORY_PATTERN = re.compile(
    r"\b(before|after|previous|previously|history|historical|over time|"
    r"earliest|latest|first|original|changed|changes|transition|sequence|"
    r"used to|formerly|all vehicles|in order|past)\b",
    flags=re.I,
)


def wants_history(query: str) -> bool:
    """Purely query-text driven; never sees task_type or gold answer."""
    return bool(HISTORY_PATTERN.search(query))


def to_compact(docs, *, include_history):
    """Serialize actual V2 state more economically, with optional history."""
    output = []
    for d in docs:
        data = json.loads(d.text)
        state = data.get("direct_state") or {}
        name = data.get("progression", "")
        current = state.get("value")
        if current is None:
            continue
        text = f"Topic: {name}. Current value: {current}."
        history = state.get("history") or []
        if include_history and history:
            text += " Historical values in order: " + "; ".join(map(str, history)) + "."
        output.append(RetrievalDocument(id=f"compact:{d.id}",
            text=text, evidence_ids=d.evidence_ids,
            timestamp=d.timestamp,
            metadata={"kind":"compact_progression"}))
    return output


def budget_pack(docs, budget):
    """Skip oversized units rather than letting one block the remaining budget."""
    kept, used = [], 0
    for d in docs:
        if used + len(d.text) <= budget:
            kept.append(d)
            used += len(d.text)
    return kept, used


class NeuralStore:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cpu")
        self.vectors = {}
        self.encode_seconds = 0.
        self.sentences_encoded = 0

    def prepare(self, strings):
        pending = list(dict.fromkeys(s for s in strings if s not in self.vectors))
        if not pending: return
        start=time.perf_counter()
        matrix=self.model.encode(pending, batch_size=64,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True,
                                 show_progress_bar=False)
        for s,v in zip(pending,matrix,strict=True):
            self.vectors[s]=v
        self.encode_seconds+=time.perf_counter()-start
        self.sentences_encoded+=len(pending)

    def dense_rank(self, docs, query):
        if not docs:return []
        self.prepare([query,*(d.text for d in docs)])
        q=self.vectors[query]
        return sorted(docs,key=lambda d:(-float(np.dot(q,self.vectors[d.text])),d.id))


def fusion(docs, query, store, *, dense=True, recency=False,
           lexical=True, source_boost=1.0):
    if not docs:return []
    ranks=[]
    if lexical:ranks.append(("lexical",ranked_bm25(docs,query)))
    if dense:ranks.append(("dense",store.dense_rank(docs,query)))
    if recency:
        temporal=sorted(docs,key=lambda d:(d.timestamp or "",d.id),reverse=True)
        ranks.append(("recency",temporal))
    scores=defaultdict(float)
    cutoff=min(len(docs),80)
    for kind,ranked in ranks:
        for i,d in enumerate(ranked[:cutoff]):
            # Keep a reproducible fixed RRF k, equal weights by default.
            weight=0.50 if kind=="recency" else 1.0
            boost=source_boost if d.metadata.get("kind") in (
                "state","compact_progression","wiki") else 1.0
            scores[d.id]+=boost*weight/(60+i+1)
    return sorted(docs,key=lambda d:(-scores[d.id],d.id))


def docs_as_of(episode, raw, facts, full_engine, current_engine, cutoff):
    raw_visible=[d for d in raw if int(d.id.split(":")[0][1:])<cutoff]
    when=datetime(2023,1,1,tzinfo=UTC)+timedelta(days=cutoff)
    key={"subject_id":f"meme:{episode['episode_id']}",
         "view":"state","valid_at":when,"known_at":when}
    hist_docs=list(build_documents(full_engine,**key))
    cur_docs=list(build_documents(current_engine,**key))
    current_wiki=wiki_docs(facts,cutoff,False)
    history_wiki=wiki_docs(facts,cutoff,True)
    current_wiki=[RetrievalDocument(id=d.id,text=d.text,
                 metadata={"kind":"wiki"}) for d in current_wiki]
    history_wiki=[RetrievalDocument(id=d.id,text=d.text,
                 metadata={"kind":"wiki"}) for d in history_wiki]
    return {
        "raw":raw_visible,
        "wiki_current":current_wiki,
        "wiki_history":history_wiki,
        "v2_current":cur_docs,
        "v2_history":hist_docs,
        "compact_current":to_compact(cur_docs,include_history=False),
        "compact_history":to_compact(hist_docs,include_history=True),
    }


def candidates(doc_groups, query, budget):
    temporal=wants_history(query)
    # Compact current state remains default at 400 chars for nonhistorical questions.
    # When asked about temporal changes, history is eligible even at 400 chars.
    wiki=doc_groups["wiki_history" if temporal else "wiki_current"]
    compact=doc_groups["compact_history" if temporal else "compact_current"]
    # Query-aware union; deduplicate identical text, preserving evidence units.
    united=[]
    seen=set()
    for d in [*wiki,*compact]:
        if d.text not in seen:
            united.append(d);seen.add(d.text)
    return {
        "raw_bm25":doc_groups["raw"],
        "raw_dense":doc_groups["raw"],
        "raw_bm25_dense":doc_groups["raw"],
        "raw_bm25_dense_recency":doc_groups["raw"],
        "wiki_current":doc_groups["wiki_current"],
        "wiki_history":doc_groups["wiki_history"],
        "v2_current":doc_groups["v2_current"],
        "v2_history":doc_groups["v2_history"],
        "v2_compact_adaptive":compact,
        "wiki_v2_adaptive":united,
        "wiki_v2_hybrid_rag":[*united,*doc_groups["raw"]],
    }


def rank(method, docs, query, store):
    if method=="raw_bm25":return ranked_bm25(docs,query)
    if method=="raw_dense":return store.dense_rank(docs,query)
    if method in ("raw_bm25_dense","wiki_current","wiki_history",
                  "v2_current","v2_history","v2_compact_adaptive",
                  "wiki_v2_adaptive"):
        return fusion(docs,query,store,recency=False)
    return fusion(docs,query,store,recency=True,
                  source_boost=1.20 if method=="wiki_v2_hybrid_rag" else 1.0)


def evaluate(episodes, *, model, max_episodes=None):
    if max_episodes:episodes=episodes[:max_episodes]
    store=NeuralStore(model)
    rows=[]
    counts={"episodes":len(episodes),"questions_total":0,
            "source_questions":0,"value_questions":0,"build_errors":[]}
    start=time.perf_counter()
    for ei,episode in enumerate(episodes):
        raw,fact_index,_,_=build_episode(episode)
        facts=history_from_sessions(episode)
        try:
            hist_engine=oracle_engine(facts,episode["episode_id"],include_history=True)
            current_engine=oracle_engine(facts,episode["episode_id"],include_history=False)
        except (ValueError,KeyError,RuntimeError,TypeError) as exc:
            counts["build_errors"].append({"id":episode["episode_id"],"error":str(exc)[:200]})
            continue
        questions=list(task_labelled_questions(episode))
        # Two snapshots before/after; encode all docs/queries in batches, not one at a time.
        snapshots={}
        all_texts=[d.text for d in raw]
        for timing,cutoff,i,q in questions:
            cutoff=min(int(cutoff),len(episode["sessions"]))
            if cutoff not in snapshots:
                snapshots[cutoff]=docs_as_of(episode,raw,facts,
                                              hist_engine,current_engine,cutoff)
                all_texts.extend(d.text for group in snapshots[cutoff].values() for d in group)
            all_texts.append(q.get("question",""))
        store.prepare(all_texts)
        for timing,position,i,q in questions:
            counts["questions_total"]+=1
            cutoff=min(int(position),len(episode["sessions"]))
            document_groups=snapshots[cutoff]
            target_sources,_,_=gold_evidence(episode,fact_index,q,cutoff)
            atoms=target_atoms(q) if q.get("task_type") in TASKS else []
            if target_sources:counts["source_questions"]+=1
            if atoms:counts["value_questions"]+=1
            if not target_sources and not atoms:continue
            query=q.get("question","")
            question_id=f"{timing}:{i}"
            # Each ranking is computed exactly once per query, not once per budget.
            layouts=candidates(document_groups,query,max(BUDGETS))
            ranked_layouts={method:rank(method,docs,query,store)
                            for method,docs in layouts.items() if docs}
            for budget in BUDGETS:
                for method,ranked in ranked_layouts.items():
                    selected,used=budget_pack(ranked,budget)
                    texts="\n".join(d.text for d in selected).casefold()
                    retrieved=set()
                    for d in selected:
                        if d.id.startswith("s") and ":t" in d.id:
                            retrieved.add(d.id)
                    row={"episode":episode["episode_id"],"domain":episode["domain"],
                         "question_id":question_id,"timing":timing,"task_type":q.get("task_type"),
                         "budget":budget,"method":method,"used_chars":used,
                         "history_query":wants_history(query)}
                    if atoms:
                        row["value_visibility"]=sum(v.casefold() in texts for v in atoms)/len(atoms)
                    if target_sources and method.startswith("raw"):
                        row["source_recall"]=len(retrieved & target_sources)/len(target_sources)
                    rows.append(row)
        if ei%10==9:print("Processed",ei+1,"episodes",flush=True)
    summary={}
    for metric in ("value_visibility","source_recall"):
        summary[metric]={}
        for budget in BUDGETS:
            summary[metric][str(budget)]={}
            for method in MODES:
                selected=[r for r in rows if r["budget"]==budget and
                          r["method"]==method and metric in r]
                by_task={}
                for task in ("Cas","Tr","Agg","ER","Abs","Del"):
                    task_rows=[r for r in selected if r["task_type"]==task]
                    by_task[task]={"n":len(task_rows),
                        "mean":round(sum(r[metric] for r in task_rows)/len(task_rows),5)
                        if task_rows else None}
                summary[metric][str(budget)][method]={
                    "n":len(selected),
                    "mean":round(sum(r[metric] for r in selected)/len(selected),5) if selected else None,
                    "mean_chars":round(sum(r["used_chars"] for r in selected)/len(selected),1)
                      if selected else None,
                    "by_task":by_task,
                }
    counts.update({"elapsed_seconds":round(time.perf_counter()-start,2),
                   "embedding_seconds":round(store.encode_seconds,2),
                   "sentences_encoded":store.sentences_encoded})
    return {"protocol":"meme-real-neural-hybrid-progressions-v1",
        "source":{"hf_repo":MODEL_REPO,"revision":REVISION,
                  "embedding_model":model},
        "counts":counts,"budgets":BUDGETS,
        "summary":summary,"rows":rows,
        "limitations":[
            "MEME oracle source-fact annotations construct wiki/V2 state; not deployable writer evaluation.",
            "Answer-value substring visibility is NOT answer generation or causal inference.",
            "Query routing uses English question words; no gold task labels in routing.",
            "Semantic encoder uses an actual pretrained sentence-transformer, CPU inference.",
            "RRF weights BM25,dense equally and recency at half strength; no tuned weights.",
            "Memory variants use full candidates while raw RAG uses raw turns, to test representation+retrieval.",
            "Skipping oversized context units differs from previous stop-on-first-overflow diagnostics.",
            "Gold source turn IDs for source_recall are credited only to raw source documents.",
            "One small open-weight embedding model; no cross-model or reranker sweep."
        ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--embedding-model",default=EMBEDDING_MODEL)
    p.add_argument("--max-episodes",type=int)
    args=p.parse_args()
    path=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
                         filename="meme_nofiller.json",revision=REVISION)
    episodes=json.loads(Path(path).read_text())
    report=evaluate(episodes,model=args.embedding_model,
                    max_episodes=args.max_episodes)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"counts":report["counts"],"summary_at_1600":{
        k:{s:{"n":x["n"],"mean":x["mean"]}
            for s,x in v["1600"].items()}
        for k,v in report["summary"].items()}},indent=2))


if __name__=="__main__":
    main()
