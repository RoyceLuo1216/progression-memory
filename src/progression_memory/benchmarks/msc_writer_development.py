"""Prepare an independent, speaker-aware MSC training audit for progression writing.

Only the official MSC dialogue train files are accepted. Dialogue is never
automatically labeled as an attachment or a grounded state transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_train(root: Path) -> dict[str, list[dict]]:
    conversations: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    base = root / "msc_dialogue"
    for session in (2, 3, 4):
        path = base / f"session_{session}" / "train.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            identity = str(row["metadata"]["initial_data_id"])
            if session in conversations[identity]:
                raise ValueError(f"duplicate conversation/session: {identity}/{session}")
            turns = []
            for index, utterance in enumerate(row["dialog"]):
                # Alternation is defined by the original row order, before any
                # teacher adds dummy turns or swaps perspective.
                turns.append({"source_id": f"msc:{identity}:s{session}:t{index}",
                              "session": session, "turn": index,
                              "speaker_id": f"msc:{identity}:speaker:{index % 2}",
                              "text": utterance["text"]})
            conversations[identity][session] = turns
    return {key: [turn for session in sorted(sessions)
                  for turn in sessions[session]]
            for key, sessions in conversations.items()}


def sample_audit(conversations: dict[str, list[dict]], count: int) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    ranked = sorted(conversations, key=lambda key: hashlib.sha256(key.encode()).digest())
    selected = ranked[:count]
    return {"dataset": "facebookresearch/ParlAI MSC v0.1", "split": "train",
            "sampling": "SHA256(initial_data_id) ascending; conversation-held-out",
            "selected_conversations": len(selected),
            "total_conversations": len(conversations),
            "annotation_contract": {
                "subject_id": "Speaker ID is a candidate, not a factual subject label. Annotate who each claim describes or abstain.",
                "event": "Extract only assertions about a durable goal, interest or life process with an exact source_id.",
                "routing": "Record no_update, attach, create_root, create_child, multi_attach, or create_and_attach against prior process IDs.",
                "state": "State and change must be entailed by cited source turns; mark ungrounded claims.",
                "review": "Have a separate human review labels before fitting or claiming a gain."},
            "conversations": [{"initial_data_id": key,
                               "turns": conversations[key]} for key in selected]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Extracted official msc_v0.1 directory containing msc_dialogue")
    parser.add_argument("--conversations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = sample_audit(load_train(args.data_root), args.conversations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_conversations": result["selected_conversations"],
                      "total_conversations": result["total_conversations"],
                      "turns": sum(len(c["turns"]) for c in result["conversations"])}))


if __name__ == "__main__":
    main()
