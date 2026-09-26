from __future__ import annotations

from pathlib import Path
import json

import pytest

from progression_memory.benchmarks import paper_native


def test_inspect_pinned_upstream_and_dataset(tmp_path: Path, monkeypatch):
    expected = paper_native.specification()["papers"]["tsm"]["revision"]
    checkout = tmp_path / "source"
    source = checkout / "data/locomo/locomo10.json"
    source.parent.mkdir(parents=True)
    source.write_text("fixture")

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(paper_native.subprocess, "run", lambda args, **kwargs:
                        Completed(expected if args[1] == "rev-parse" else ""))
    monkeypatch.setattr(paper_native.sys, "version_info", (3, 11))
    monkeypatch.setattr(paper_native.shutil, "which", lambda name: "/bin/docker")
    monkeypatch.setattr(paper_native, "DATASET_SHA256", {"locomo": paper_native._sha256(source)})
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    report = paper_native.inspect("tsm", "locomo", checkout)
    assert report["source_revision"] == expected
    assert report["dataset_sha256"] == paper_native._sha256(source)


def test_rejects_unpinned_paper_before_model_calls(tmp_path: Path, monkeypatch):
    class Completed:
        stdout = "wrong-revision"
    monkeypatch.setattr(paper_native.subprocess, "run", lambda args, **kwargs: Completed())
    with pytest.raises(ValueError, match="expected untouched"):
        paper_native.inspect("timem", "locomo", tmp_path)


def test_rejects_unacknowledged_model_calls(tmp_path: Path):
    with pytest.raises(ValueError, match="acknowledge"):
        paper_native.run("tsm", "locomo", tmp_path, tmp_path / "out", acknowledge=False)


def test_tsm_index_audit_detects_unreachable_snapshot_summaries(tmp_path: Path):
    cache = tmp_path / "sample" / "cache"
    cache.mkdir(parents=True)
    (cache / "snapshots.json").write_text(json.dumps([
        {"persona_summaries": [{"node_id": "summary_0_0"}],
         "topic_summaries": [{"node_id": "entity_summary_0_0"}]},
    ]))
    (cache / "dense_index_mapping.json").write_text(json.dumps({"doc_ids": ["leaf_0"]}))
    audit = paper_native.audit_tsm_indexes(tmp_path)
    assert audit["summary_count"] == 2
    assert audit["indexed_summary_ids"] == 0
    assert audit["cases_missing_index"] == 0
