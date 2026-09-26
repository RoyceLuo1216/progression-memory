from __future__ import annotations

from pathlib import Path

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
