from pathlib import Path

import yaml

from progression_memory.benchmarks import open_weight


def test_prepare_tsm_configs(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    for dataset in ("locomo", "longmemeval"):
        path = source / "configs" / dataset / "tsm.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"llm": {"api_model": "gpt-4o-mini"},
                                       "embedding": {"model": "text-embedding-3-small", "dimension": 1536}}))
    rev = open_weight.specification()["papers"]["tsm"]["revision"]
    monkeypatch.setattr(open_weight.subprocess, "check_output", lambda *args, **kwargs:
                        rev if args[0][1] == "rev-parse" else "")
    destination = tmp_path / "adapted"
    result = open_weight.prepare("tsm", source, destination)
    assert result["track"] == "standardized-open-weight-adaptation"
    for dataset in ("locomo", "longmemeval"):
        original = yaml.safe_load((source / "configs" / dataset / "tsm.yaml").read_text())
        adapted = yaml.safe_load((destination / "configs" / dataset / "tsm.yaml").read_text())
        assert original["llm"]["api_model"] == "gpt-4o-mini"
        assert adapted["llm"]["api_model"] == open_weight.CHAT_MODEL
        assert adapted["embedding"]["dimension"] == open_weight.EMBEDDING_DIMENSION


def test_explicit_timem_model_fields():
    config = {"llm_model": "gpt-4o-mini", "nested": {"stage1_llm_model": "gpt-4o-mini",
              "model": "gpt-4o", "prompt": "gpt-4o-mini is a literal in a prompt"}}
    changed = open_weight._replace_models(config)
    assert changed["llm_model"] == open_weight.CHAT_MODEL
    assert changed["nested"]["stage1_llm_model"] == open_weight.CHAT_MODEL
    assert changed["nested"]["model"] == open_weight.CHAT_MODEL
    assert changed["nested"]["prompt"] == config["nested"]["prompt"]
