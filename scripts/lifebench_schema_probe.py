"""Inspect LifeBench v2 event / QA nested schema; no scoring."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO="C1754955896/Lifebenchv2.0"
SHA="3d06052d5ef2d94d9042bd2845787bfd37ba925d"
FILES=("data/fenghaoran/QA_all/QA.json",
       "data/fenghaoran/daily_event.json",
       "data/fenghaoran/phone_data/agent_chat.json",
       "data/fenghaoran/event_tree.json")

def describe(obj,depth=0):
    if isinstance(obj,dict):
        if depth>=2: return {"type":"dict","keys":list(obj)[:30]}
        return {str(k):describe(v,depth+1) for k,v in list(obj.items())[:15]}
    if isinstance(obj,list):
        return {"type":"list","length":len(obj),
                "first":describe(obj[0],depth+1) if obj else None}
    if isinstance(obj,str):
        return {"type":"str","example":obj[:320]}
    return {"type":type(obj).__name__,"example":obj}

out={}
for filename in FILES:
    raw=Path(hf_hub_download(repo_id=REPO,repo_type="dataset",
                  filename=filename,revision=SHA)).read_text()
    obj=json.loads(raw)
    out[filename]={"bytes":len(raw),"summary":describe(obj)}
q_path=hf_hub_download(repo_id=REPO,repo_type="dataset",
    filename="data/fenghaoran/QA_all/QA.json",revision=SHA)
e_path=hf_hub_download(repo_id=REPO,repo_type="dataset",
    filename="data/fenghaoran/daily_event.json",revision=SHA)
qs=json.loads(Path(q_path).read_text())
es=json.loads(Path(e_path).read_text())
ids={str(x.get("event_id")) for x in es}
allids=[str(y) for q in qs for y in q.get("required_events_id",[])]
out["evidence_id_alignment"]={
    "qa":len(qs),"event_records":len(es),"required_refs":len(allids),
    "ref_matched":sum(x in ids for x in allids),
    "questions_fully_matched":sum(bool(q.get("required_events_id")) and
        all(str(x) in ids for x in q["required_events_id"]) for q in qs),
}
path=Path("results/lifebench_schema.json");path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2)[:13000])
