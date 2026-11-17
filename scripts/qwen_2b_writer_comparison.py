"""Fair same-model, same-observation Qwen3.5-2B writer comparison.

Two different serialization contracts: V2's real WriterProposal and a simple
topic-wiki update contract. Both receive the same evidence and prior fact.
No oracle updates are supplied to either model. All model responses, validity,
commit attempts, latency and token accounting are archived.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from progression_memory.v2.models import Observation, Progression
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import (
    ScriptedWriter, WritePipeline, WriterProposal,
)
from scripts.meme_qwen_2b_reader import Reader, MODEL

class TopicUpdate(BaseModel):
    model_config=ConfigDict(extra="forbid")
    topic: str
    current: str
    evidence: str

class WikiProposal(BaseModel):
    model_config=ConfigDict(extra="forbid")
    updates: list[TopicUpdate] = Field(default_factory=list)

CASES=[
    ("founder_goal","My long-term goal is to build an independent AI company.",True,
     "AI",False),
    ("toronto_constraint","I have decided I must remain in Toronto for the next two years.",True,
     "Toronto",False),
    ("masters_deferral","I have decided to postpone my master's applications until 2028.",True,
     "2028",False),
    ("accepted_offer","I accepted a backend engineering offer yesterday.",True,
     "backend",False),
    ("friend_hypothetical","My friend thinks founding an AI company could be interesting.",
     False,"",False),
    ("question","Would graduate school help my career?",False,"",False),
    ("maybe_move","Maybe someday I could relocate if things change.",False,"",False),
    ("casual","I went for a walk and the weather was nice.",False,"",False),
    ("revised_priority","I have decided to stop pursuing graduate school this year. "
     "My new career priority is stable engineering income.",True,"income",True),
    ("revised_location","My Toronto-only relocation restriction is now lifted. "
     "I am willing to relocate for work.",True,"relocate",True),
]
V2_SYS=(
    "You maintain the user's durable, evolving memory. Extract only explicit "
    "first-person plans, commitments, and decisions. Ignore questions, speculation, "
    "casual observations and other people's preferences. Return ONLY JSON, "
    "an object with key \"updates\" (array); for no change return "
    "{\"updates\":[]}. Each update: action=create or transition; for create, "
    "progression_id must be null, name must be descriptive; for transition, "
    "progression_id must exactly match the PRIOR progression id. Include "
    "reason (nonempty), patches (array with key, operation=set or retract, value), "
    "stage (null allowed). For active state set a concise grounded string; "
    "do not invent dates. Example: "
    "{\"updates\":[{\"action\":\"create\",\"progression_id\":null,"
    "\"name\":\"Location constraint\",\"reason\":\"User decided to stay\","
    "\"patches\":[{\"key\":\"constraint\",\"operation\":\"set\","
    "\"value\":\"remain in Toronto\"}]}]}."
)
WIKI_SYS=(
    "You maintain a persistent topic wiki for a user. Extract only explicit "
    "first-person durable plans, commitments or decisions. Ignore questions, "
    "speculation, casual observations, and other people's preferences. "
    "Output ONLY a JSON object with key \"updates\" (array). For no change "
    "return {\"updates\":[]}. Every update must contain exactly keys "
    "topic, current, evidence. current is a concise grounded fact in the "
    "user's current state, evidence is a short literal quote from the current "
    "observation. Preserve revision semantics when prior wiki notes exist. "
    "Example: {\"updates\":[{\"topic\":\"Location constraint\","
    "\"current\":\"must remain in Toronto\","
    "\"evidence\":\"I have decided I must remain in Toronto\"}]}."
)

def make_prior(revision):
    if not revision:return None
    return {"id":"p:career","subject_id":"fictional:user",
            "name":"Career plan","state":{"priority":"graduate school",
                "constraint":"Toronto only"}}

def infer(backend,messages,max_tokens=220):
    inputs=backend.processor.apply_chat_template(
        messages,add_generation_prompt=True,tokenize=True,
        return_dict=True,return_tensors="pt",enable_thinking=False)
    start=time.monotonic()
    with torch.inference_mode():
        out=backend.model.generate(
            **inputs,do_sample=False,max_new_tokens=max_tokens,
            pad_token_id=backend.processor.tokenizer.eos_token_id)
    raw=backend.processor.decode(
        out[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
    return raw,{"seconds":round(time.monotonic()-start,3),
                "input_tokens":int(inputs["input_ids"].shape[-1]),
                "output_tokens":int(out.shape[-1]-inputs["input_ids"].shape[-1])}

def assess(method,raw,case,observation,prior):
    key,message,expected,anchor,revision=case
    cls=WriterProposal if method=="v2" else WikiProposal
    row={"schema_valid":False,"has_update":None,"change_correct":False,
         "anchor_present":False,"action_correct":False,"committed":False,
         "error":None}
    try:
        proposal=cls.model_validate_json(raw)
        row["schema_valid"]=True
        row["has_update"]=bool(proposal.updates)
        row["change_correct"]=bool(proposal.updates)==expected
        payload=json.dumps(proposal.model_dump(mode="json"),ensure_ascii=False)
        row["anchor_present"]=bool(anchor and anchor.casefold() in payload.casefold())
        if method=="v2":
            row["action_correct"]=all(
                u.action==("transition" if revision else "create")
                for u in proposal.updates)
            engine=StateEngine()
            if prior:
                engine.create_progression(Progression(
                    id=prior["id"],subject_id="fictional:user",
                    name=prior["name"],state=prior["state"]))
            try:
                WritePipeline(engine,ScriptedWriter({key:proposal})).process(observation)
                row["committed"]=True
            except (ValueError,TypeError,RuntimeError,KeyError) as e:
                row["error"]="CommitError: "+str(e)[:170]
        else:
            row["action_correct"]=True # Wiki has no explicit transition action.
            row["committed"]=True
        if not expected:
            row["semantic_proxy"]=row["change_correct"]
        else:
            row["semantic_proxy"]=(
                row["change_correct"] and row["anchor_present"] and
                row["action_correct"] and row["committed"]
            )
    except (ValidationError,ValueError,TypeError) as e:
        row["error"]=type(e).__name__+": "+str(e)[:200]
        row["semantic_proxy"]=False
    return row

def save(rows,started):
    summary={}
    for method in ("v2","wiki"):
        sub=[r for r in rows if r["method"]==method]
        summary[method]={
            "n":len(sub),
            "schema_valid":sum(r["quality"]["schema_valid"] for r in sub),
            "correct_update_decision":sum(r["quality"]["change_correct"] for r in sub),
            "positive_grounding_and_commit":sum(r["quality"]["semantic_proxy"] for r in sub
                                                if r["expected_change"]),
            "full_proxy_correct":sum(r["quality"]["semantic_proxy"] for r in sub),
            "input_tokens":sum(r["usage"]["input_tokens"] for r in sub),
            "output_tokens":sum(r["usage"]["output_tokens"] for r in sub),
            "generation_seconds":round(sum(r["usage"]["seconds"] for r in sub),2),
        }
    report={"model":MODEL,"protocol":"qwen-2b-v2-writer-vs-wiki-writer-v1",
            "cases":len(CASES),"completed_pairs":len(rows)//2,
            "elapsed_seconds":round(time.monotonic()-started,2),
            "summary":summary,"rows":rows,
            "limitations":[
                "10 authored cases, four no-change and six explicit changes.",
                "Grounding proxy uses a specified salient anchor and correct "
                "update action, not independent human or semantic QA.",
                "Schema complexity differs, reflecting deployable V2 and wiki contracts.",
                "Neither writer has grammar-constrained JSON decoding.",
                "Independent observations; multi-month memory drift not tested.",
                "Equal model, generation settings, observation text, prior summary, "
                "and number of inference calls; token costs differ."
            ]}
    path=Path("results/qwen_2b_writers.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    backend=Reader()
    rows=[]
    started=time.monotonic()
    for i,case in enumerate(CASES):
        key,message,expected,anchor,revision=case
        prior=make_prior(revision)
        obs=Observation(id=key,source_id=key,subject_id="fictional:user",
                        text=message,recorded_at=datetime(2026,1,1,tzinfo=UTC))
        for method,sys in (("v2",V2_SYS),("wiki",WIKI_SYS)):
            user=json.dumps({"CURRENT":message,"PRIOR":prior},
                            ensure_ascii=False)
            raw,usage=infer(backend,[{"role":"system","content":sys},
                                    {"role":"user","content":user}])
            quality=assess(method,raw,case,obs,prior)
            rows.append({"case":key,"expected_change":expected,
                         "revision":revision,"method":method,"raw":raw,
                         "usage":usage,"quality":quality})
        save(rows,started)
        print("Writer cases",i+1,"/",len(CASES),flush=True)
    report=save(rows,started)
    print(json.dumps(report["summary"],indent=2))

if __name__=="__main__":
    main()
