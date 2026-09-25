"""Pin original paper code and compare genuinely paired benchmark artifacts.

This module never substitutes our temporal/semantic BM25 ablations for a paper
implementation. A paper result is accepted only with native-run provenance and
an independently recorded evaluator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SPEC = Path(__file__).resolve().parents[3] / "evaluation" / "paper_pairs.json"
REQUIRED_MATCH = ("paper_revision", "dataset_sha256", "dataset_split", "llm_model",
                  "embedding_model", "reader_model", "retrieval_budget", "evaluator_revision")


def specification() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def prepare(paper: str, destination: Path, *, fetch: bool = False) -> dict:
    """Check out an exact paper revision, refusing an existing or wrong tree."""
    spec = specification()["papers"][paper]
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if fetch:
        subprocess.run(["git", "clone", "--no-checkout", spec["repository"], str(destination)], check=True)
        subprocess.run(["git", "checkout", "--detach", spec["revision"]], cwd=destination, check=True)
    else:
        raise ValueError("preparing a repository requires --fetch; no code was downloaded")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
    if actual != spec["revision"]:
        raise RuntimeError(f"paper revision mismatch: {actual}")
    return {"paper": paper, "repository": spec["repository"], "revision": actual,
            "path": str(destination.resolve())}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(result: dict) -> None:
    required = set(REQUIRED_MATCH) | {"system", "dataset", "implementation_revision",
                                    "native_run_manifest", "official_evaluator_output",
                                    "question_results"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"result missing fields: {sorted(missing)}")
    for field in REQUIRED_MATCH + ("implementation_revision", "native_run_manifest",
                                  "official_evaluator_output"):
        if not result[field]:
            raise ValueError(f"result has empty {field}")
    if not isinstance(result["question_results"], dict) or not result["question_results"]:
        raise ValueError("question_results must map question IDs to outcomes")
    if any(type(outcome) is not bool for outcome in result["question_results"].values()):
        raise ValueError("question outcomes must be booleans from the evaluator")


def compare(baseline: dict, intervention: dict) -> dict:
    """Compare paired *officially evaluated* outcomes, with matched conditions."""
    validate_result(baseline)
    validate_result(intervention)
    names = specification()["pairs"]
    if [baseline["system"], intervention["system"]] not in names:
        raise ValueError("undeclared paper/intervention pair")
    if baseline["system"] in specification()["papers"]:
        pinned = specification()["papers"][baseline["system"]]["revision"]
        if baseline["implementation_revision"] != pinned or baseline["paper_revision"] != pinned:
            raise ValueError("paper baseline is not the pinned original revision")
    if baseline["implementation_revision"] == intervention["implementation_revision"]:
        raise ValueError("intervention has no distinct implementation revision")
    if baseline["dataset"] != intervention["dataset"]:
        raise ValueError("datasets differ")
    differences = [key for key in REQUIRED_MATCH if baseline[key] != intervention[key]]
    if differences:
        raise ValueError(f"comparison conditions differ: {differences}")
    first = baseline["question_results"]
    second = intervention["question_results"]
    if first.keys() != second.keys():
        raise ValueError("question IDs differ; use the same examples and split")
    n = len(first)
    improved = sorted(key for key in first if not first[key] and second[key])
    regressed = sorted(key for key in first if first[key] and not second[key])
    return {
        "baseline": baseline["system"], "intervention": intervention["system"],
        "dataset": baseline["dataset"], "questions": n,
        "baseline_accuracy": sum(first.values()) / n,
        "intervention_accuracy": sum(second.values()) / n,
        "delta_accuracy": (len(improved) - len(regressed)) / n,
        "improved_ids": improved, "regressed_ids": regressed,
        "conditions": {key: baseline[key] for key in REQUIRED_MATCH},
        "native_run_manifests": [baseline["native_run_manifest"], intervention["native_run_manifest"]],
        "official_evaluator_outputs": [baseline["official_evaluator_output"], intervention["official_evaluator_output"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("spec")
    checkout = subs.add_parser("prepare")
    checkout.add_argument("paper", choices=tuple(specification()["papers"]))
    checkout.add_argument("destination", type=Path)
    checkout.add_argument("--fetch", action="store_true")
    paired = subs.add_parser("compare")
    paired.add_argument("baseline", type=Path)
    paired.add_argument("intervention", type=Path)
    paired.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "spec":
        output = specification()
    elif args.action == "prepare":
        output = prepare(args.paper, args.destination, fetch=args.fetch)
    else:
        output = compare(json.loads(args.baseline.read_text()), json.loads(args.intervention.read_text()))
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.action == "compare" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
