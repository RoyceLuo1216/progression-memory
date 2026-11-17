"""Qwen3.5-2B reader vs existing 0.8B on identical MEME QA samples.

This scores actual generated answers with a fixed, deterministic ordinal
match on chronological questions and a strict normalized exact match for
cascading facts. Oracle source-fact memory construction is held constant.
No gold answer is exposed to the answering model.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMultimodalLM, AutoProcessor

from scripts.meme_downstream_qa import (
    BUDGET, context_for, ordered_values, sample_questions,
)
from progression_memory.benchmarks.meme_evidence_retrieval import MODEL_REPO, REVISION

MODEL="Qwen/Qwen3.5-2B"
ARMS=("wiki_history","compact_v2_adaptive")
TASK_LIMITS={"Tr":12,"Cas":4}
MAX_NEW_TOKENS=128

class Reader:
    def __init__(self):
        torch.set_num_threads(2)
        self.processor=AutoProcessor.from_pretrained(MODEL)
        self.model=AutoModelForMultimodalLM.from_pretrained(
            MODEL,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
        ).eval()
        self.calls=[]

    def ask(self,question,context):
        prompt=[
            {"role":"system","content":(
                "Answer only from MEMORY. For a historical timeline, give all "
                "items in chronological order, earliest first. If evidence is "
                "missing, say 'unknown'. Be concise and do not invent facts.")},
            {"role":"user","content":"MEMORY:\n"+context+"\n\nQUESTION:\n"+question}
        ]
        inputs=self.processor.apply_chat_template(
            prompt,add_generation_prompt=True,tokenize=True,
            return_dict=True,return_tensors="pt",enable_thinking=False)
        start=time.monotonic()
        with torch.inference_mode():
            output=self.model.generate(
                **inputs,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.processor.tokenizer.eos_token_id)
        text=self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()
        stats={"wall_seconds":round(time.monotonic()-start,3),
               "input_tokens":int(inputs["input_ids"].shape[-1]),
               "output_tokens":int(output.shape[-1]-inputs["input_ids"].shape[-1])}
        self.calls.append(stats)
        return text,stats


def ordered_expected(q):
    values=q.get("entity_values") or {}
    arr=[]
    for v in values.values():
        if isinstance(v,list):arr.extend(v)
        elif v is not None:arr.append(v)
    return [str(v) for v in arr]


def normalized(text):
    return re.sub(r"\W+","",str(text).casefold())


def generate():
    source=hf_hub_download(repo_id=MODEL_REPO,repo_type="dataset",
                           filename="meme_nofiller.json",revision=REVISION)
    episodes=json.loads(Path(source).read_text())
    selected=[]
    counts={x:0 for x in TASK_LIMITS}
    for ep,timing,position,index,q in sample_questions(episodes):
        task=q.get("task_type")
        if task not in TASK_LIMITS or counts[task]>=TASK_LIMITS[task]:
            continue
        counts[task]+=1
        cutoff=min(int(position),len(ep["sessions"]))
        contexts=context_for(ep,cutoff,q["question"])
        selected.append({
            "episode":ep["episode_id"],"id":f"{timing}:{index}",
            "task":task,"question":q["question"],
            "reference":q.get("expected_answer") or q.get("gold_answer"),
            "ordered_gold":ordered_expected(q) if task=="Tr" else None,
            "contexts":{arm:contexts[arm] for arm in ARMS},
        })
    backend=Reader()
    started=time.monotonic()
    for i,row in enumerate(selected):
        outputs={}
        for arm in ARMS:
            memory=row["contexts"][arm]["context"]
            answer,stats=backend.ask(row["question"],memory)
            outputs[arm]={"text":answer,"usage":stats,
                          "chronology_correct":(
                              ordered_values(answer,row["ordered_gold"])
                              if row["task"]=="Tr" else None),
                          "literal_exact_correct":(
                              normalized(answer)==normalized(row["reference"])
                              if row["task"]=="Cas" else None)}
        row["outputs"]=outputs
        save(selected[:i+1],started,completed=i+1)
        print("2B QA",i+1,"/",len(selected),flush=True)
    return save(selected,started,completed=len(selected))


def save(rows,started,completed):
    outcomes={}
    for task in TASK_LIMITS:
        subset=[row for row in rows if row["task"]==task]
        outcomes[task]={"count":len(subset)}
        for arm in ARMS:
            key="chronology_correct" if task=="Tr" else "literal_exact_correct"
            outcomes[task][arm]={
                "correct":sum(bool(r["outputs"][arm][key]) for r in subset),
                "total":len(subset)}
    counts={arm:{"calls":0,"input_tokens":0,"output_tokens":0,
                 "generation_seconds":0.0} for arm in ARMS}
    for row in rows:
        for arm in ARMS:
            usage=row["outputs"][arm]["usage"]
            c=counts[arm]
            c["calls"]+=1
            for field in ("input_tokens","output_tokens"):c[field]+=usage[field]
            c["generation_seconds"]+=usage["wall_seconds"]
    report={"model":MODEL,"protocol":"meme-2b-qa-two-memory-arms-v1",
            "budget_chars":BUDGET,
            "source":{"dataset":MODEL_REPO,"revision":REVISION},
            "completed":completed,"elapsed_seconds":round(time.monotonic()-started,2),
            "results":outcomes,"cost":counts,"rows":rows,
            "limitations":[
                "Same deterministic 12 Tr and first 4 Cas samples from prior 0.8B study.",
                "Gold annotated facts construct BOTH memories; not real writer effectiveness.",
                "Historical scorer is strict ordered-value coverage, not semantic judgment.",
                "Cascading scorer is literal exact normalization; not semantic accuracy.",
                "This tests two memory views, not dense RAG; model size differs from 0.8B.",
                "Small sample; persona/episode confidence intervals are wide."
            ]}
    p=Path("results/meme_2b_reader.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    r=generate()
    print(json.dumps({"completed":r["completed"],"results":r["results"],
                      "cost":r["cost"]},indent=2))
