"""Prepare isolated, standardized model configurations for paper experiments.

These are adaptations of pinned upstream sources, never native reproductions.
No upstream checkout is edited in place. The model server and embeddings server
must expose OpenAI-compatible chat and embeddings endpoints respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from progression_memory.benchmarks.paper_pairs import specification


CHAT_MODEL = "Qwen/Qwen3-8B"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_models(value, *, chat: bool = True):
    """Replace explicit model selectors while preserving prompts and budgets."""
    if isinstance(value, list):
        return [_replace_models(item, chat=chat) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"llm_model", "eval_model", "stage1_llm_model", "stage2_llm_model",
                   "stage3_llm_model"} and isinstance(item, str):
            result[key] = CHAT_MODEL
        elif key == "model" and isinstance(item, str) and item.startswith(("gpt-", "glm-")):
            result[key] = CHAT_MODEL
        else:
            result[key] = _replace_models(item, chat=chat)
    return result


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare(paper: str, checkout: Path, destination: Path) -> dict:
    """Copy a pinned paper and make only explicit model-configuration edits."""
    if destination.exists():
        raise ValueError(f"destination exists: {destination}")
    spec = specification()["papers"][paper]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != spec["revision"]:
        raise ValueError(f"wrong upstream revision: {revision}")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=checkout, text=True).strip()
    if dirty:
        raise ValueError("upstream tracked files must be clean")
    shutil.copytree(checkout, destination, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "runs", "outputs", ".venv", "*.pyc"))
    modified = []
    if paper == "tsm":
        paths = [destination / "configs" / dataset / "tsm.yaml"
                 for dataset in ("locomo", "longmemeval")]
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg["llm"]["api_model"] = CHAT_MODEL
            cfg["embedding"].update(model=EMBEDDING_MODEL, dimension=EMBEDDING_DIMENSION)
            _write_yaml(path, cfg)
            modified.append(path)
    elif paper == "timem":
        paths = [destination / "config/settings.yaml"]
        paths += list((destination / "config/datasets").glob("*/retrieval_config.yaml"))
        paths += list((destination / "config/datasets").glob("*/qa_prompts.yaml"))
        paths += list((destination / "config/datasets").glob("*/eval_prompt.yaml"))
        paths.append(destination / "config/dataset_profiles.yaml")
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg = _replace_models(cfg)
            if path.name == "settings.yaml":
                cfg["llm"]["providers"]["openai"]["model"] = CHAT_MODEL
                cfg["llm"]["retrieval_planner"]["model"] = CHAT_MODEL
                cfg["llm"]["embedding"]["model"] = "${QWEN_EMBEDDING_PATH:./models/Qwen3-Embedding-0.6B}"
            _write_yaml(path, cfg)
            modified.append(path)
    else:
        raise ValueError(f"unknown paper: {paper}")
    manifest = {
        "track": "standardized-open-weight-adaptation",
        "paper": paper, "upstream_revision": revision,
        "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "configuration_files": {str(path.relative_to(destination)): _sha(path) for path in modified},
        "limitations": ["No progression intervention is implemented by this preparation step.",
                        "Hardcoded model names in upstream Python must be audited before a live run.",
                        "The judge must be held fixed across both sides of each pair."],
    }
    (destination / "standardized_model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=("tsm", "timem"))
    parser.add_argument("checkout", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.paper, args.checkout, args.destination), indent=2))


if __name__ == "__main__":
    main()
