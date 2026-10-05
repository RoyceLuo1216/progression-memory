"""Rebuild the reviewed MSC pair snapshot from the CI audit JSON artifact.

This script intentionally keeps the annotation manifest explicit; it does not
infer labels from lexical overlap or speaker parity.
"""

import argparse
import json
from pathlib import Path


# conversation index in the SHA256-sorted twelve-conversation train audit,
# anchor turn, candidate identity, and state justified by that anchor.
CANDIDATES = {
    "snakes": (0, "s2:t1", "Keeping pet snakes", "The speaker owns and cares for snakes"),
    "dog416": (0, "s2:t4", "Carter the dog", "The speaker has a dog named Carter"),
    "snake_plan": (0, "s3:t2", "Considering a pet snake", "The speaker is reconsidering having a snake"),
    "social416": (0, "s2:t11", "Social connection", "The speaker spends much time alone with snakes"),
    "cat2281": (1, "s2:t7", "Bella the cat", "The speaker has a cat named Bella"),
    "dog2281": (1, "s2:t0", "Dog ownership", "The speaker takes their dog for walks"),
    "dog_plan": (1, "s2:t7", "Considering another pet dog", "The speaker is thinking about getting a dog"),
    "waylon": (1, "s3:t2", "Waylon the dog", "The speaker has a black lab called Waylon"),
    "rabbits": (1, "s3:t6", "Keeping wild rabbits safe from Waylon", "The speaker uses a leash to let wild rabbits flee"),
    "allergy": (1, "s2:t10", "Cat allergy", "The speaker is allergic to cats"),
    "book": (1, "s3:t1", "Writing a book", "The speaker is writing a book"),
    "fishing4419": (2, "s2:t0", "Fishing with children", "The speaker takes their children fishing"),
    "fishing4419b": (2, "s2:t5", "Planning fishing with children", "The speaker plans to buy fishing gear for their family"),
    "move4419": (2, "s2:t3", "Recent relocation", "The speaker moved somewhere warmer and nearer lakes"),
    "premed": (2, "s2:t7", "Pre-med studies", "The speaker is in a difficult pre-med semester"),
    "medschool": (2, "s2:t9", "Future medical degree", "The speaker expects further medical schooling"),
    "family4419": (2, "s2:t2", "Raising children", "The speaker has younger and older children"),
    "music4419": (2, "s3:t8", "Children's music lessons", "The speaker is trying group music lessons with their children"),
    "promotion": (7, "s2:t1", "Career advancement at current job", "The speaker was overlooked for a promotion"),
    "plant": (7, "s2:t2", "Potential plant job", "The other speaker offered work at the plant"),
    "income": (7, "s2:t4", "Current work at the plant", "The speaker values steady income from plant work"),
    "cooking": (7, "s2:t8", "Cooking interests", "The speaker enjoys spaghetti and cupcakes"),
    "italy": (10, "s2:t0", "Trips to boyfriend in Italy", "The speaker has tickets to visit their boyfriend in Italy"),
    "boyfriend": (10, "s2:t2", "Relationship with boyfriend", "The speaker's boyfriend lives in Italy"),
    "housing": (10, "s2:t8", "Living arrangements", "The speaker lives with their mother and brother"),
    "violin": (10, "s2:t11", "Violin playing and teaching", "The speaker practices violin and plans lessons"),
    "cake": (11, "s2:t0", "Baking cakes", "The speaker is finishing icing a cake"),
    "dance": (11, "s2:t3", "Dance group", "The speaker plans to share cake with their dance group"),
    "walking": (11, "s2:t6", "Nature walks", "The speaker likes walking in nature"),
    "reading": (11, "s2:t6", "Reading interest", "The speaker likes to read"),
}

# Event is later than every candidate anchor in its conversation. A few
# explicitly hard negatives concern the same topic or different speaker.
# Positive labels are provisional manual judgments, not ground truth.
SCENARIOS = [
    (0, "s3:t4", ["snake_plan", "dog416", "snakes", "social416"], ["snake_plan", "dog416"]),
    (0, "s3:t5", ["snakes", "snake_plan", "dog416", "social416"], ["snakes"]),
    (0, "s3:t6", ["dog416", "snake_plan", "snakes", "social416"], ["dog416", "snake_plan"]),
    (0, "s3:t10", ["dog416", "snake_plan", "snakes", "social416"], []),
    (1, "s3:t1", ["dog_plan", "cat2281", "allergy", "dog2281"], ["dog_plan", "cat2281"]),
    (1, "s3:t4", ["waylon", "dog_plan", "cat2281", "allergy"], ["waylon"]),
    (1, "s3:t8", ["rabbits", "waylon", "dog_plan", "cat2281"], ["rabbits", "waylon"]),
    (1, "s4:t0", ["waylon", "rabbits", "dog_plan", "cat2281"], ["waylon"]),
    (1, "s4:t4", ["waylon", "rabbits", "dog_plan", "cat2281"], ["waylon"]),
    (1, "s4:t7", ["cat2281", "dog_plan", "waylon", "book"], []),
    (2, "s2:t11", ["medschool", "premed", "move4419", "fishing4419"], ["medschool", "premed"]),
    (2, "s3:t3", ["family4419", "fishing4419", "premed", "move4419"], []),
    (2, "s3:t5", ["fishing4419b", "family4419", "premed", "move4419"], ["fishing4419b"]),
    (2, "s3:t7", ["premed", "medschool", "family4419", "fishing4419"], ["premed"]),
    (2, "s3:t10", ["music4419", "family4419", "fishing4419", "premed"], ["music4419", "family4419"]),
    (7, "s3:t1", ["promotion", "plant", "income", "cooking"], ["promotion", "plant"]),
    (7, "s3:t3", ["promotion", "plant", "income", "cooking"], ["plant", "promotion"]),
    (7, "s3:t6", ["income", "plant", "promotion", "cooking"], ["income"]),
    (7, "s3:t7", ["promotion", "plant", "income", "cooking"], ["promotion", "plant"]),
    (10, "s3:t0", ["boyfriend", "italy", "housing", "violin"], ["boyfriend", "housing"]),
    (10, "s3:t2", ["boyfriend", "italy", "housing", "violin"], ["boyfriend", "housing"]),
    (10, "s3:t6", ["italy", "boyfriend", "housing", "violin"], ["italy"]),
    (10, "s3:t8", ["italy", "boyfriend", "housing", "violin"], ["italy"]),
    (11, "s3:t0", ["cake", "dance", "walking", "reading"], ["cake"]),
    (11, "s4:t0", ["cake", "dance", "walking", "reading"], ["cake"]),
]


def build(audit: dict) -> dict:
    conversations = audit["conversations"]
    turns = [{f"s{t['session']}:t{t['turn']}": t for t in c["turns"]}
             for c in conversations]
    pairs = []
    for number, (conversation, event_ref, candidates, positives) in enumerate(SCENARIOS):
        event = turns[conversation][event_ref]
        for candidate_id in candidates:
            ci, anchor_ref, identity, state = CANDIDATES[candidate_id]
            if ci != conversation:
                raise ValueError("candidate crosses conversation")
            anchor = turns[ci][anchor_ref]
            if (anchor["session"], anchor["turn"]) >= (event["session"], event["turn"]):
                raise ValueError(f"candidate {candidate_id} leaks future turn for scenario {number}")
            # A speaker can make an offer concerning the other participant.
            # The plant-job process belongs to the recipient, not its anchor speaker.
            subject_id = (anchor["speaker_id"].rsplit(":", 1)[0] + ":1"
                          if candidate_id == "plant" else anchor["speaker_id"])
            pairs.append({"scenario_id": number, "conversation_id": conversations[ci]["initial_data_id"],
                          "event_source_id": event["source_id"], "event_speaker_id": event["speaker_id"],
                          "event_text": event["text"], "candidate_id": candidate_id,
                          "candidate_speaker_id": anchor["speaker_id"],
                          "candidate_subject_id": subject_id,
                          "candidate_identity": identity, "candidate_state": state,
                          "candidate_anchor_source_id": anchor["source_id"],
                          "candidate_anchor_text": anchor["text"],
                          "attach": int(candidate_id in positives)})
    assert len(SCENARIOS) == 25 and len(pairs) == 100
    return {"source": "Meta/ParlAI Multi-Session Chat v0.1 train",
            "annotation_status": "single-pass provisional; independent adjudication pending",
            "task": "Same-subject, same-evolving-process event-to-existing-progression attachment",
            "pairs": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(json.loads(args.audit.read_text())), indent=2) + "\n")
