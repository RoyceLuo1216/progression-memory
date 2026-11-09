"""Inspect one real sample file from each of four pinned external benchmarks.

No model inference and no claims of benchmark performance. Records actual
file hashes and schema to support follow-on ingestion adapter correctness.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

SAMPLES = {
    "HorizonBench": ("stellalisy/HorizonBench",
                     "sample/test-00000-of-00001.parquet",
                     "50941f00f90c03a5a60219d76393869b757b835a"),
    "PersonaMem": ("bowen-upenn/PersonaMem-v1",
                   "questions_32k.csv",
                   "a8076d5608c93ba2a28983cd78aa99b01a163ae7"),
    "MEME": ("meme-benchmark/MEME",
             "meme_nofiller.json",
             "e14c4e685772014deb96f91dc7eddfbeb21d90d8"),
    "LifeBench": ("C1754955896/Lifebenchv2.0",
                  "data/fenghaoran/QA_all/QA.json",
                  "3d06052d5ef2d94d9042bd2845787bfd37ba925d"),
}


def main():
    output = {}
    for name,(repo,filename,revision) in SAMPLES.items():
        path = Path(hf_hub_download(repo_id=repo,repo_type="dataset",
                                    filename=filename,revision=revision))
        payload = {"repo":repo,"revision":revision,"file":filename,
                   "bytes":path.stat().st_size,
                   "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        if filename.endswith(".csv"):
            with path.open(newline="",encoding="utf-8") as handle:
                reader=csv.DictReader(handle)
                payload["columns"]=reader.fieldnames
                sample=next(reader,None)
                payload["first_row_field_types"]={
                    k:type(v).__name__ for k,v in (sample or {}).items()}
        elif filename.endswith(".json"):
            with path.open(encoding="utf-8") as handle:
                raw=json.load(handle)
            payload["top_level_type"]=type(raw).__name__
            payload["top_level_keys"]=list(raw)[:25] if isinstance(raw,dict) else None
            sample=(next(iter(raw.values())) if raw else None) if isinstance(raw,dict) else (raw[0] if raw else None)
            payload["sample_type"]=type(sample).__name__
            payload["sample_keys"]=list(sample)[:25] if isinstance(sample,dict) else None
        elif filename.endswith(".parquet"):
            import pyarrow.parquet as pq
            metadata=pq.read_metadata(path)
            payload["columns"]=metadata.schema.names
            payload["num_rows"]=metadata.num_rows
        output[name]=payload
        print(name,payload["bytes"],payload.get("columns",payload.get("sample_keys")))
    result={"protocol":"external-sample-ingestion-v1",
            "samples":output,"limitations":[
                "One file from each release; not complete dataset ingestion.",
                "Only data and schema inspection, no retrieval or model scores."
            ]}
    path=Path("results/external_sample_schema.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
