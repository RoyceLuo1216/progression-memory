"""Train an exploratory linear attachment proposal on 100 synthetic labeled pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from progression_memory.maintenance.calibrated_progression import (
    Event, LinearAttachmentModel, Progression, State, features,
)


def load_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for scenario_id, (event_text, positive, candidate_ids) in enumerate(data["scenarios"]):
        if len(candidate_ids) != 4 or not set(positive) <= set(candidate_ids):
            raise ValueError(f"invalid candidates at scenario {scenario_id}")
        event = Event(str(scenario_id), data["subject_id"], event_text, None, str(scenario_id))
        for candidate_id in candidate_ids:
            identity, scope, state = data["candidates"][candidate_id]
            progression = Progression(candidate_id, data["subject_id"], identity, scope,
                                      current_state=State(state, frozenset()))
            rows.append({"scenario": scenario_id, "event": event_text,
                         "candidate": candidate_id, "attach": int(candidate_id in positive),
                         "features": features(event, progression).tolist()})
    if len(rows) != 100:
        raise ValueError(f"expected 100 examples, got {len(rows)}")
    return rows


def load_real_pairs(path: Path) -> list[dict]:
    """Load provisional MSC annotations, grouping every fold by conversation."""
    data = json.loads(path.read_text())
    if data.get("annotation_status") != "single-pass provisional; independent adjudication pending":
        raise ValueError("unexpected annotation provenance")
    rows = []
    for pair in data["pairs"]:
        event = Event(pair["event_source_id"], pair["event_speaker_id"],
                      pair["event_text"], None, pair["event_source_id"])
        candidate = Progression(pair["candidate_id"], pair["candidate_subject_id"],
                                pair["candidate_identity"], pair["candidate_identity"],
                                current_state=State(pair["candidate_state"], frozenset()))
        rows.append({"scenario": pair["conversation_id"],
                     "event": pair["event_text"], "candidate": pair["candidate_id"],
                     "attach": pair["attach"], "features": features(event, candidate).tolist()})
    if len(rows) != 100:
        raise ValueError(f"expected 100 examples, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path,
                        default=Path("evaluation/progression_calibration_scenarios.json"))
    parser.add_argument("--real-pairs", type=Path,
                        help="provisional MSC pair JSON; replaces synthetic scenarios")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_real_pairs(args.real_pairs) if args.real_pairs else load_pairs(args.scenarios)
    model, report = LinearAttachmentModel.train(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, {**report, "dataset": str(args.real_pairs or args.scenarios),
                             "limitations": ("Provisional single-pass real labels; fold held out by conversation, "
                                             "but cutoff selected on these folds. Not a held-out score."
                                             if args.real_pairs else
                                             "Synthetic examples; no validation on real writes.")})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
