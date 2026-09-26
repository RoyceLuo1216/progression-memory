"""Run pinned, unmodified paper entrypoints with explicit provenance.

This is a native baseline runner, not a reproduction score calculator. It
refuses missing prerequisites and preserves upstream logs and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from progression_memory.benchmarks.paper_pairs import specification


DATASET_SHA256 = {
    "locomo": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "longmemeval": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
}
TSM_DATA = {"locomo": "data/locomo/locomo10.json",
            "longmemeval": "data/longmemeval/longmemeval_s.json"}
TIMEM_DATA = {"locomo": "data/locomo10.json",
              "longmemeval": "data/longmemeval_s_cleaned.json"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(paper: str, dataset: str, checkout: Path) -> dict:
    """Check native source and runtime before any model or database calls."""
    spec = specification()["papers"][paper]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                              text=True, capture_output=True, check=True).stdout.strip()
    if revision != spec["revision"]:
        raise ValueError(f"expected untouched {paper} at {spec['revision']}, got {revision}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("paper baseline checkout contains tracked modifications")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("paper baseline requires Python 3.11")
    if not shutil.which("docker"):
        raise RuntimeError("paper baseline requires Docker for the native database")
    dataset_path = checkout / (TSM_DATA if paper == "tsm" else TIMEM_DATA)[dataset]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"place the pinned upstream dataset at: {dataset_path}")
    actual_hash = _sha256(dataset_path)
    if actual_hash != DATASET_SHA256[dataset]:
        raise ValueError(f"dataset checksum mismatch: {actual_hash}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the pinned model setup")
    return {
        "paper": paper, "dataset": dataset, "source_revision": revision,
        "dataset_sha256": actual_hash, "python": sys.version.split()[0],
        "database": "neo4j" if paper == "tsm" else "postgres+qdrant",
        "native_commands": spec["entrypoints"][dataset],
        "checked_at_utc": _utc(),
    }


def run(paper: str, dataset: str, checkout: Path, output: Path, *, acknowledge: bool,
        isolated_database: bool = False) -> dict:
    if not acknowledge:
        raise ValueError("native runs send dataset text to the configured model; pass --acknowledge-model-calls")
    report = inspect(paper, dataset, checkout)
    if paper == "timem" and not isolated_database:
        # Its native generation test clears tables. Never infer isolation from
        # a connection string or operate on a possibly shared database.
        raise ValueError("TiMem generation clears tables; pass --isolated-database on an isolated deployment")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if paper == "tsm":
        config = checkout / "configs" / dataset / "tsm.yaml"
        commands = [[sys.executable, "scripts/run_experiment.py", "--config", str(config),
                     "--acknowledge-data-transmission", "--force-rebuild"]]
        config_hash = _sha256(config)
    else:
        config_hash = None
        env["TIMEM_DATASET_PROFILE"] = "default" if dataset == "locomo" else "longmemeval_s"
        split_flag = "--split-locomo" if dataset == "locomo" else "--split-longmemeval"
        commands = [[sys.executable, "experiments/dataset_utils/dataset_splitter.py", split_flag]]
        for command in report["native_commands"]:
            parts = shlex.split(command)
            if parts[0] == "python":
                parts[0] = sys.executable
            elif parts[0] == "pytest":
                parts = [sys.executable, "-m", "pytest", *parts[1:]]
            commands.append(parts)
    report.update({"status": "running", "started_at_utc": _utc(),
                   "config_sha256": config_hash, "commands": commands,
                   "isolated_database_acknowledged": isolated_database})
    manifest = output / "native_runner.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for index, command in enumerate(commands):
        with (output / f"step_{index:02d}.stdout.log").open("w") as stdout, \
             (output / f"step_{index:02d}.stderr.log").open("w") as stderr:
            completed = subprocess.run(command, cwd=checkout, env=env, stdout=stdout,
                                       stderr=stderr, check=False)
        report.setdefault("steps", []).append({"command": command, "exit_code": completed.returncode})
        report.update({"status": "running" if completed.returncode == 0 else "failed",
                       "finished_at_utc": _utc() if completed.returncode else None})
        manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if completed.returncode:
            raise RuntimeError(f"native step {index} failed; inspect {output / f'step_{index:02d}.stderr.log'}")
    report.update({"status": "completed", "finished_at_utc": _utc()})
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=tuple(specification()["papers"]))
    parser.add_argument("dataset", choices=tuple(DATASET_SHA256))
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acknowledge-model-calls", action="store_true")
    parser.add_argument("--isolated-database", action="store_true")
    args = parser.parse_args()
    result = (run(args.paper, args.dataset, args.checkout, args.output,
                  acknowledge=args.acknowledge_model_calls,
                  isolated_database=args.isolated_database)
              if args.output else inspect(args.paper, args.dataset, args.checkout))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
