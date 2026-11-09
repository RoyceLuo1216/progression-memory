"""Pin released external-memory benchmark dataset revisions and file inventories.

Metadata-only validation: does NOT count as an external performance run.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

DATASETS = {
    "HorizonBench": {"repo":"stellalisy/HorizonBench",
                     "task":"preference evolution; MCQ"},
    "PersonaMem": {"repo":"bowen-upenn/PersonaMem-v1",
                   "task":"personalized reply selection"},
    "MEME": {"repo":"meme-benchmark/MEME",
             "task":"tracking, deletion, dependencies"},
    "LifeBench": {"repo":"C1754955896/Lifebenchv2.0",
                  "task":"year-long multi-source life QA"},
}


def main():
    api=HfApi()
    outputs={}
    for name,item in DATASETS.items():
        try:
            ds=api.dataset_info(item["repo"],files_metadata=False)
            siblings=[f.rfilename for f in (ds.siblings or [])]
            outputs[name]={
                "status":"metadata_accessible",
                "hf_repo":item["repo"],
                "revision_sha":ds.sha,
                "task":item["task"],
                "license":((ds.card_data or {}).get("license")
                           if isinstance(ds.card_data,dict) else
                           getattr(ds.card_data,"license",None)),
                "file_count":len(siblings),
                "sample_files":siblings[:40],
            }
        except (OSError,ValueError,RuntimeError) as e:
            outputs[name]={"status":"metadata_error","hf_repo":item["repo"],
                           "error_type":type(e).__name__,"message":str(e)[:300]}
    report={
        "protocol":"external-dataset-metadata-v1",
        "checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "datasets":outputs,
        "limitations":[
            "Only metadata/revision inventory, no dataset ingestion or model inference.",
            "Dataset viewer availability and sample schema need separate validation.",
            "Model predictions, task scores and external comparisons are NOT produced."
        ],
    }
    out=Path("results/external_dataset_manifest.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({name:{"status":x["status"],"sha":x.get("revision_sha"),
                            "files":x.get("file_count")}
                      for name,x in outputs.items()},indent=2))


if __name__=="__main__":
    main()
