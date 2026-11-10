"""Qwen3.5-0.8B writer diagnostic: original ModelWriter vs compact schema prompt.

Eight hand-written held-out observation types; tests schema compliance and
presence/absence of a memory update, NOT full transition semantics.
Production V2 code unchanged. Runs actual CPU model inference, no mock output.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from pydantic import ValidationError

from qwen35_cpu_smoke import LocalQwen
from progression_memory.v2.model_writer import ModelWriter
from progression_memory.v2.models import Observation
from progression_memory.v2.writer_pipeline import WriterProposal

CASES=(
    ("long_term_goal", "My long-term goal is to build an independent AI company.", True),
    ("location_constraint", "I have decided I must remain in Toronto for the next two years.", True),
    ("school_deferral", "I have decided to postpone my master's applications until 2028.", True),
    ("offer_commitment", "I accepted a backend engineering offer yesterday.", True),
    ("hypothetical_friend", "My friend thinks founding an AI company could be interesting.", False),
    ("question", "Would graduate school help my career?", False),
    ("conditional_idea", "Maybe someday I could relocate if things change.", False),
    ("small_talk", "I went for a walk and the weather was nice.", False),
)

SYS=(
    "You update long-term user memory from first-person statements only. "
    "Output exactly one valid JSON object with the key 'updates' (an array). "
    "Never output a bare array. For explicit stable decisions or plans, create "
    "a progression object with action, name, reason and patches. "
    "For questions, hypotheticals, friends' opinions, and small talk, "
    "return exactly {\"updates\":[]}. "
    "Positive example: {\"updates\":[{\"action\":\"create\","
    "\"name\":\"Location requirement\",\"reason\":\"User explicitly decided\","
    "\"patches\":[{\"key\":\"constraint\",\"operation\":\"set\","
    "\"value\":\"remain in Toronto\"}]}]}. "
    "Return only JSON, with no Markdown fences."
)

class LoggingBackend:
    def __init__(self,backend):
        self.backend=backend
        self.latest_raw=""
    def generate_messages(self,messages,json_schema=None):
        raw=self.backend.generate_messages(messages,json_schema=json_schema)
        self.latest_raw=raw
        return raw


def run():
    backend=LocalQwen()
    log=LoggingBackend(backend)
    rows=[]
    for identifier,message,expected_change in CASES:
        obs=Observation(id=identifier,source_id=identifier,
                        subject_id="fictional:diagnostic",text=message,
                        recorded_at=datetime(2026,1,1,tzinfo=UTC))
        for method in ("production_prompt","compact_prompt"):
            valid=False
            error=None
            updates=None
            try:
                if method=="production_prompt":
                    proposal=ModelWriter(log).decide(obs,[])
                else:
                    raw=log.generate_messages([
                        {"role":"system","content":SYS},
                        {"role":"user","content":"CURRENT: "+message+
                         "\nCANDIDATES: []\nReply with JSON object."}],
                        json_schema=WriterProposal.model_json_schema())
                    proposal=WriterProposal.model_validate_json(raw)
                valid=True
                updates=len(proposal.updates)
            except (ValidationError,ValueError,TypeError,RuntimeError) as exc:
                error={"type":type(exc).__name__,"message":str(exc)[:240]}
            rows.append({"case":identifier,"method":method,"expected_change":expected_change,
                         "schema_valid":valid,"update_count":updates,
                         "change_detection_correct":valid and bool(updates)==expected_change,
                         "raw":log.latest_raw[:3000],"error":error})
    summary={}
    for method in ("production_prompt","compact_prompt"):
        subset=[r for r in rows if r["method"]==method]
        summary[method]={"n":len(subset),
                         "schema_valid":sum(r["schema_valid"] for r in subset),
                         "change_detection_correct":sum(r["change_detection_correct"] for r in subset),
                         "positive_valid":sum(r["change_detection_correct"] for r in subset
                                              if r["expected_change"]),
                         "negative_valid":sum(r["change_detection_correct"] for r in subset
                                              if not r["expected_change"])}
    return {"model":"Qwen/Qwen3.5-0.8B","cases":len(CASES),"summary":summary,
            "inference":backend.calls,"rows":rows,
            "limitations":["8 templated independent observations, not contextual multi-turn state updates",
                            "Update presence/absence and schema validity only, not semantic patch fidelity",
                            "Compact prompt is experimental and does not modify production ModelWriter",
                            "CPU inference without generation-time constrained JSON decoding"]}


if __name__=="__main__":
    results=run()
    output=Path("results/qwen35_writer_diagnostics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"summary":results["summary"],"inference_calls":len(results["inference"])},indent=2))
