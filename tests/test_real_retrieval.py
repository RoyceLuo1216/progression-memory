import json
from pathlib import Path

from progression_memory.benchmarks.real_retrieval import run


def test_locomo_adapter_and_bm25(tmp_path: Path) -> None:
    fixture = [{
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "2026-01-01",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "I adopted a beagle named Scout."},
                {"speaker": "B", "dia_id": "D1:2", "text": "That is great news."}
            ],
            "session_2_date_time": "2026-01-10",
            "session_2": [{"speaker": "A", "dia_id": "D2:1", "text": "Scout loves the dog park."}]
        },
        "qa": [{"question": "Which beagle is mine?", "answer": "Scout", "evidence": ["D1:1"], "category": 1}]
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    result = run("locomo", str(path), [1], "append")
    assert result["cutoffs"]["1"]["any_evidence_recall"] == 1.0


def test_longmemeval_adapter_and_bm25(tmp_path: Path) -> None:
    fixture = [{
        "question_id": "q1", "question_type": "single-session-user",
        "question": "What happened during my Lisbon trip?", "answer": "Lisbon",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026-01-01", "2026-02-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought a chair."}],
            [{"role": "user", "content": "My trip to Lisbon was wonderful.", "has_answer": True}]
        ],
        "answer_session_ids": ["s2"]
    }]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    result = run("longmemeval", str(path), [1], "append")
    assert result["cutoffs"]["1"]["all_evidence_recall"] == 1.0


def test_progression_systems_preserve_evidence(tmp_path: Path) -> None:
    fixture = [{
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1 January, 2026",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "My beagle Scout likes hiking trails."},
                {"speaker": "B", "dia_id": "D1:2", "text": "Scout sounds energetic."}
            ],
            "session_2_date_time": "20 January, 2026",
            "session_2": [
                {"speaker": "A", "dia_id": "D2:1", "text": "I started pottery and made a blue bowl."}
            ]
        },
        "qa": [
            {"question": "Which beagle likes hiking?", "answer": "Scout", "evidence": ["D1:1"], "category": 1},
            {"question": "What pottery item did I make?", "answer": "bowl", "evidence": ["D2:1"], "category": 1}
        ]
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    for system in ("progression_only", "fixed_progression", "adaptive_progression"):
        result = run("locomo", str(path), [5], system)
        assert result["cutoffs"]["5"]["all_evidence_recall"] == 1.0
        assert result["corpus_stats"]["avg_retrieval_units"] >= 1


def test_longmemeval_abstention_excluded_from_retrieval(tmp_path: Path) -> None:
    fixture = [{
        "question_id": "q_abs", "question_type": "single-session-user",
        "question": "What is my missing fact?", "answer": "unknown",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2026-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "unrelated"}]],
        "answer_session_ids": ["s1"]
    }]
    path = tmp_path / "lme_abs.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    result = run("longmemeval", str(path), [1], "append")
    assert result["eligible_query_count"] == 0
    assert result["excluded_without_evidence"] == 1
