import json

import pytest

from progression_memory.benchmarks.msc_writer_development import load_train, sample_audit


def test_msc_preserves_speaker_and_session_identity(tmp_path):
    base = tmp_path / "msc_dialogue"
    for session in (2, 3, 4):
        path = base / f"session_{session}" / "train.txt"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"metadata": {"initial_data_id": "person-a"},
                                    "dialog": [{"text": "I work at Acme."},
                                               {"text": "I teach."}]}) + "\n")
    rows = load_train(tmp_path)
    assert len(rows["person-a"]) == 6
    assert rows["person-a"][0]["speaker_id"] == rows["person-a"][2]["speaker_id"]
    assert rows["person-a"][0]["speaker_id"] != rows["person-a"][1]["speaker_id"]
    assert rows["person-a"][2]["session"] == 3
    assert sample_audit(rows, 1)["conversations"][0]["initial_data_id"] == "person-a"


def test_reject_duplicate_session(tmp_path):
    base = tmp_path / "msc_dialogue"
    for session in (2, 3, 4):
        path = base / f"session_{session}" / "train.txt"
        path.parent.mkdir(parents=True)
        line = json.dumps({"metadata": {"initial_data_id": "a"}, "dialog": []}) + "\n"
        path.write_text(line * (2 if session == 2 else 1))
    with pytest.raises(ValueError, match="duplicate"):
        load_train(tmp_path)
