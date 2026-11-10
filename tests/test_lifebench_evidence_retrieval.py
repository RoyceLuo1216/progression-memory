from progression_memory.benchmarks.lifebench_evidence_retrieval import (
    PrefixBM25, selected, tokenize
)


def test_chinese_ngrams_and_alphanumeric_tokens():
    tokens=tokenize("预算调整 2025 AI")
    assert "预算" in tokens
    assert "调整" in tokens
    assert "2025" in tokens
    assert "ai" in tokens


def test_query_cutoff_prevents_future_event_retrieval():
    index=PrefixBM25([
        {"id":"past","date":"2025-01-01","text":"预算调整，减少交通开支"},
        {"id":"future","date":"2025-03-01","text":"预算调整，增加交通开支"},
    ])
    early=index.rank("预算调整交通开支","2025-02-01")
    late=index.rank("预算调整交通开支","2025-03-01")
    assert [e["id"] for e in early]==["past"]
    assert {e["id"] for e in late}=={"past","future"}


def test_recency_uses_only_visible_events():
    index=PrefixBM25([
        {"id":"old","date":"2025-01-01","text":"old"},
        {"id":"new","date":"2025-02-01","text":"new"},
    ])
    assert [e["id"] for e in index.recent("2025-01-15")]==["old"]
    assert [e["id"] for e in index.recent("2025-02-01")]==["new","old"]


def test_whole_document_budget():
    docs=[{"id":"first","text":"12345"},{"id":"second","text":"123456"}]
    ids,used=selected(docs,10)
    assert ids=={"first"} and used==5
