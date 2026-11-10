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
path=Path("results/lifebench_schema.json");path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2)[:13000])
