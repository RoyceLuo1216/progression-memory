"""Schema-constrained 0.8B V2 writer diagnostic: isolate formatting vs semantics.

Uses lm-format-enforcer token constraints on exactly the canonical
WriterProposal JSON Schema. This is actual inference, never a repaired raw
string or heuristic update. Reports separately schema validity, decision
correctness, and state-engine commit success. Experimental model path only.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from progression_memory.v2.models import Observation
from progression_memory.v2.state_engine import StateEngine
from progression_memory.v2.writer_pipeline import (
    ScriptedWriter, WritePipeline, WriterProposal,
)
from scripts.qwen35_cpu_smoke import LocalQwen, MODEL
from scripts.qwen35_writer_diagnostics import CASES, SYS

def run():
    backend=LocalQwen()
    parser=JsonSchemaParser(WriterProposal.model_json_schema())
    prefix=build_transformers_prefix_allowed_tokens_fn(
        backend.processor.tokenizer,parser)
    rows=[]
    started=time.monotonic()
    for identifier,message,expected_change in CASES:
        obs=Observation(id=identifier,source_id=identifier,
            subject_id="fictional:diagnostic",text=message,
            recorded_at=datetime(2026,1,1,tzinfo=UTC))
        inputs=backend.processor.apply_chat_template([
            {"role":"system","content":SYS},
            {"role":"user","content":"CURRENT: "+message+
                "\nCANDIDATES: []\nReturn a valid JSON object."}],
            add_generation_prompt=True,tokenize=True,
            return_dict=True,return_tensors="pt",enable_thinking=False)
        start=time.monotonic()
        with torch.inference_mode():
            out=backend.model.generate(
                **inputs,max_new_tokens=256,do_sample=False,
                prefix_allowed_tokens_fn=prefix,
                pad_token_id=backend.processor.tokenizer.eos_token_id)
        raw=backend.processor.decode(
            out[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
        stats={"seconds":round(time.monotonic()-start,2),
               "input_tokens":int(inputs["input_ids"].shape[-1]),
               "output_tokens":int(out.shape[-1]-inputs["input_ids"].shape[-1])}
        quality={"schema_valid":False,"correct_change":False,
                 "can_commit":False,"error":None}
        try:
            proposal=WriterProposal.model_validate_json(raw)
            quality["schema_valid"]=True
            quality["correct_change"]=bool(proposal.updates)==expected_change
            engine=StateEngine()
            WritePipeline(engine,ScriptedWriter({identifier:proposal})).process(obs)
            quality["can_commit"]=True
        except (ValueError,TypeError,RuntimeError,KeyError) as e:
            quality["error"]=type(e).__name__+": "+str(e)[:200]
        rows.append({"case":identifier,"expected_change":expected_change,
            "raw":raw,"quality":quality,"usage":stats})
        print(identifier,quality,flush=True)
        save(rows,started)
    return save(rows,started)

def save(rows,started):
    result={"model":MODEL,"method":"constrained-json-schema-WriterProposal",
            "cases_completed":len(rows),"summary":{
                "valid":sum(x["quality"]["schema_valid"] for x in rows),
                "correct_change":sum(x["quality"]["correct_change"] for x in rows),
                "committable":sum(x["quality"]["can_commit"] for x in rows)},
            "usage":{"input_tokens":sum(r["usage"]["input_tokens"] for r in rows),
                "output_tokens":sum(r["usage"]["output_tokens"] for r in rows),
                "generation_seconds":round(sum(r["usage"]["seconds"] for r in rows),2)},
            "elapsed_seconds":round(time.monotonic()-started,2),"rows":rows,
            "limitations":[
                "Constrained decoding enforces syntax, not semantic correctness.",
                "Only 8 independent observations, no multi-event drift.",
                "Does not modify the deployed ModelWriter; uses fixed compact prompt.",
                "Model must still select correct updates and grounded patches."
            ]}
    path=Path("results/qwen08_constrained_writer.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    r=run()
    print(json.dumps({"summary":r["summary"],"usage":r["usage"]},indent=2))
