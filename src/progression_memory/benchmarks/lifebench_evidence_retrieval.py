"""LifeBench v2: chronological event-source retrieval on real year-long traces.

Scores official required_events_id from a pinned public dataset, never
gold answers. Chinese-aware character 2/3-grams plus alphanumeric tokens.
Queries cannot retrieve future events, and IDF is computed over the
query-time prefix only. No generated answers or production V2 writer.
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO="C1754955896/Lifebenchv2.0"
SHA="3d06052d5ef2d94d9042bd2845787bfd37ba925d"
BUDGETS=(800,1600,3200,6400,12800)
TERMS=re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")

def tokenize(text):
    terms=[]
    for m in TERMS.finditer(text.casefold()):
        s=m.group()
        if '\u3400'<=s[0]<='\u9fff':
            terms.extend(s[i:i+n] for n in (2,3) for i in range(len(s)-n+1))
        else:
            terms.append(s)
    return terms


class PrefixBM25:
    def __init__(self, events):
        self.events=sorted(events,key=lambda e:(e["date"],e["id"]))
        self.dates=[e["date"] for e in self.events]
        self.lengths=[]
        self.prefix_len=[0]
        self.postings=defaultdict(list)
        for i,event in enumerate(self.events):
            tf=Counter(tokenize(event["text"]))
            length=sum(tf.values())
            self.lengths.append(length)
            self.prefix_len.append(self.prefix_len[-1]+length)
            for term,freq in tf.items():
                self.postings[term].append((i,freq))
        self.posting_indices={k:[idx for idx,_ in v] for k,v in self.postings.items()}

    def eligible(self,cutoff):
        return bisect.bisect_right(self.dates,cutoff)

    def rank(self,query,cutoff,limit=60):
        n=self.eligible(cutoff)
        if n==0:return []
        avgdl=self.prefix_len[n]/n
        scores=defaultdict(float)
        for term in set(tokenize(query)):
            postings=self.postings.get(term,[])
            count=bisect.bisect_left(self.posting_indices.get(term,[]),n)
            if count<=0:continue
            idf=math.log(1+(n-count+0.5)/(count+0.5))
            for i,freq in postings[:count]:
                dl=self.lengths[i]
                scores[i]+=idf*(freq*2.5)/(freq+1.5*(0.25+0.75*dl/avgdl))
        inds=heapq.nlargest(limit,range(n),key=lambda i:(scores[i],-i))
        return [self.events[i] for i in inds]

    def recent(self,cutoff,limit=60):
        return list(reversed(self.events[:self.eligible(cutoff)]))[:limit]


def selected(docs,budget):
    used=0;out=[]
    for doc in docs:
        if used+len(doc["text"])>budget:break
        used+=len(doc["text"]);out.append(doc["id"])
    return set(out),used


def list_people():
    api=HfApi()
    files=[x.rfilename for x in api.dataset_info(REPO,files_metadata=False).siblings]
    return sorted(set(x.split("/")[1] for x in files
                      if x.startswith("data/") and x.endswith("/QA_all/QA.json")))


def load(person,filename):
    p=hf_hub_download(repo_id=REPO,repo_type="dataset",
                      filename=f"data/{person}/{filename}",revision=SHA)
    return json.loads(Path(p).read_text())


def evaluate(people,max_questions=None):
    rows=[]
    counts={"people":len(people),"qa_total":0,"qa_with_evidence_ids":0,
            "qa_fully_eligible":0,"daily_events":0,"unmatched_references":0}
    for person in people:
        events=load(person,"daily_event.json")
        questions=load(person,"QA_all/QA.json")
        corpus=[]
        for x in events:
            ids=x.get("event_id")
            if ids is None:continue
            date=x.get("date") or []
            if isinstance(date,list):date=date[0] if date else ""
            date=str(date)[:10]
            if not date:continue
            corpus.append({"id":str(ids),"date":date,
                           "text":f"{x.get('name','')}. {x.get('description','')} "
                                  f"{x.get('type','')}"})
        counts["daily_events"]+=len(corpus)
        index=PrefixBM25(corpus)
        by_id={x["id"]:x for x in corpus}
        qs=questions[:max_questions] if max_questions else questions
        for qi,q in enumerate(qs):
            counts["qa_total"]+=1
            wanted=set(str(v) for v in q.get("required_events_id",[]) if v is not None)
            if not wanted:continue
            counts["qa_with_evidence_ids"]+=1
            cutoff=str(q.get("ask_time",""))[:10]
            present={e for e in wanted if e in by_id and by_id[e]["date"]<=cutoff}
            if present!=wanted:
                counts["unmatched_references"]+=len(wanted-present)
                continue
            counts["qa_fully_eligible"]+=1
            bm25=index.rank(str(q.get("question","")),cutoff)
            recency=index.recent(cutoff)
            for name,ranked in (("cjk_bm25",bm25),("recency",recency)):
                for budget in BUDGETS:
                    retrieved,used=selected(ranked,budget)
                    hits=len(retrieved & wanted)
                    rows.append({"person":person,"question":qi,
                        "question_type":q.get("question_type"),
                        "system":name,"budget":budget,
                        "source_recall":hits/len(wanted),
                        "all_sources":int(hits==len(wanted)),
                        "chars":used,"sources":len(wanted)})
        print(person,"events",len(corpus),"qa",len(qs),flush=True)
    results={}
    for budget in BUDGETS:
        results[str(budget)]={}
        for s in ("cjk_bm25","recency"):
            arr=[r for r in rows if r["budget"]==budget and r["system"]==s]
            results[str(budget)][s]={
                "n":len(arr),
                "mean_source_recall":round(sum(x["source_recall"] for x in arr)/len(arr),5) if arr else None,
                "all_sources":round(sum(x["all_sources"] for x in arr)/len(arr),5) if arr else None,
                "avg_chars":round(sum(x["chars"] for x in arr)/len(arr),1) if arr else None}
    return {"protocol":"lifebenchv2-chronological-daily-event-evidence-v1",
            "source":{"hf_repo":REPO,"revision":SHA},
            "counts":counts,"summary":results,"rows":rows,
            "limitations":[
                "Only official daily_event evidence ID retrieval, no generated answers or V2 writer.",
                "Question cutoff applied to both ranking and corpus statistics; future events excluded.",
                "QA without required_events_id or references unavailable by ask_time are excluded.",
                "Dataset also contains phone and agent-chat modalities, which are NOT indexed here.",
                "Character n-gram BM25 is a suitable lexical Chinese baseline, not semantic dense retrieval.",
                "Whole document budget, stop-before-overflow, maximum 60 ranked events per question.",
                "Confidence intervals should be clustered by person, not by question.",
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--max-people",type=int)
    p.add_argument("--max-questions",type=int)
    args=p.parse_args()
    people=list_people()
    if args.max_people:people=people[:args.max_people]
    report=evaluate(people,args.max_questions)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"counts":report["counts"],"at_3200":report["summary"]["3200"]},indent=2))


if __name__=="__main__":
    main()
