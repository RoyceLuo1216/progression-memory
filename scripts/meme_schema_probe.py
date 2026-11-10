"""Inspect pinned MEME benchmark schema with no predictions."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

def summarize(value, depth=0):
    if isinstance(value, dict):
        return {k:summarize(v,depth+1) for k,v in list(value.items())[:15]} if depth<3 else {"type":"dict","keys":list(value)[:18]}
    if isinstance(value, list):
        return {"type":"list","length":len(value),"first":summarize(value[0],depth+1) if value else None}
    if isinstance(value, str):
        return {"type":"str","example":value[:220]}
    return {"type":type(value).__name__,"example":value}

path=hf_hub_download(repo_id="meme-benchmark/MEME",repo_type="dataset",
                     revision="e14c4e685772014deb96f91dc7eddfbeb21d90d8",
                     filename="meme_nofiller.json")
episodes=json.loads(Path(path).read_text())
report={"count":len(episodes),"example_schema":summarize(episodes[0]),
        "domains":{},"tasks":{}}
for ep in episodes:
    domain=str(ep.get("domain"))
    report["domains"][domain]=report["domains"].get(domain,0)+1
    for task in ep.get("tasks",[]):
        t=str(task.get("type") if isinstance(task,dict) else type(task).__name__)
        report["tasks"][t]=report["tasks"].get(t,0)+1
report["first_session"]=summarize(episodes[0]["sessions"][0])
report["first_before_question"]=summarize(episodes[0]["before_questions"][0]) if episodes[0]["before_questions"] else None
report["first_after_question"]=summarize(episodes[0]["after_questions"][0]) if episodes[0]["after_questions"] else None
report["task_samples"]={}
for ep in episodes:
    for t in ep["tasks"]:
        report["task_samples"].setdefault(t["type"],t)
out=Path("results/meme_schema_probe.json");out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2)[:12000])
