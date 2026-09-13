from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def answer_contains(prediction: str, reference: str) -> float:
    gold = normalize_answer(reference)
    pred = normalize_answer(prediction)
    if not gold:
        return float(not pred)
    return float(gold in pred)


class QAReader(Protocol):
    name: str

    def answer(self, question: str, context: tuple[RetrievalDocument, ...]) -> str: ...


class ExtractiveBM25Reader:
    """Deterministic no-API reader for CI and architecture ablations.

    It ranks sentence-sized spans from the retrieved reader context and returns the
    strongest span. This is intentionally a weak reader: it validates the end-to-end
    pipeline and compares memory systems without hiding retrieval effects behind an
    external model dependency.
    """

    name = "extractive_bm25"

    def answer(self, question: str, context: tuple[RetrievalDocument, ...]) -> str:
        spans: list[RetrievalDocument] = []
        for document in context:
            for index, sentence in enumerate(_SENTENCE_SPLIT.split(document.text)):
                sentence = sentence.strip()
                if sentence:
                    spans.append(
                        RetrievalDocument(
                            id=f"{document.id}:span:{index}",
                            text=sentence,
                            evidence_ids=document.evidence_ids,
                            timestamp=document.timestamp,
                        )
                    )
        if not spans:
            return "I don't know."
        ranked = BM25Index(spans).score(question)
        if not ranked or ranked[0][0] <= 0.0:
            return "I don't know."
        return ranked[0][1].text


@dataclass(frozen=True)
class QARunConfig:
    top_k: int | None = None
    char_budget: int | None = None

    def __post_init__(self) -> None:
        if self.top_k is None and self.char_budget is None:
            raise ValueError("top_k or char_budget must be set")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.char_budget is not None and self.char_budget < 0:
            raise ValueError("char_budget must be non-negative")


def select_reader_context(
    query: RetrievalQuery,
    config: QARunConfig,
) -> tuple[RetrievalDocument, ...]:
    scored = BM25Index(query.documents).score(query.text)
    selected: list[RetrievalDocument] = []
    used_chars = 0
    for _, document in scored:
        if config.top_k is not None and len(selected) >= config.top_k:
            break
        if config.char_budget is not None:
            document_chars = len(document.text)
            if used_chars + document_chars > config.char_budget:
                break
            used_chars += document_chars
        selected.append(document)
    return tuple(selected)


def _retrieval_recall(
    query: RetrievalQuery,
    context: tuple[RetrievalDocument, ...],
) -> float:
    if not query.evidence_ids:
        return 0.0
    retrieved: set[str] = set()
    for document in context:
        retrieved.update(document.evidence_ids or {document.id})
    return len(retrieved & query.evidence_ids) / len(query.evidence_ids)


def evaluate_qa(
    queries: list[RetrievalQuery],
    reader: QAReader,
    config: QARunConfig,
    *,
    include_per_query: bool = True,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for query in queries:
        context = select_reader_context(query, config)
        prediction = reader.answer(query.text, context)
        rows.append(
            {
                "id": query.id,
                "category": query.category,
                "question": query.text,
                "reference": query.answer,
                "prediction": prediction,
                "exact_match": exact_match(prediction, query.answer),
                "token_f1": token_f1(prediction, query.answer),
                "answer_contains": answer_contains(prediction, query.answer),
                "retrieval_recall": _retrieval_recall(query, context),
                "context_chars": sum(len(document.text) for document in context),
                "context_units": len(context),
            }
        )

    def aggregate(items: list[dict[str, object]]) -> dict[str, float | int]:
        denominator = max(1, len(items))
        return {
            "query_count": len(items),
            "exact_match": sum(float(row["exact_match"]) for row in items) / denominator,
            "token_f1": sum(float(row["token_f1"]) for row in items) / denominator,
            "answer_contains": (
                sum(float(row["answer_contains"]) for row in items) / denominator
            ),
            "retrieval_recall": (
                sum(float(row["retrieval_recall"]) for row in items) / denominator
            ),
            "avg_context_chars": (
                sum(int(row["context_chars"]) for row in items) / denominator
            ),
            "avg_context_units": (
                sum(int(row["context_units"]) for row in items) / denominator
            ),
        }

    categories = sorted({str(row["category"]) for row in rows})
    result: dict[str, object] = {
        "reader": reader.name,
        "context_policy": {
            "top_k": config.top_k,
            "char_budget": config.char_budget,
        },
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate(
                [row for row in rows if str(row["category"]) == category]
            )
            for category in categories
        },
    }
    if include_per_query:
        result["per_query"] = rows
    return result


def write_longmemeval_hypotheses(
    result: dict[str, object],
    path: str | Path,
) -> None:
    """Export the two-field JSONL format accepted by LongMemEval evaluate_qa.py."""
    rows = list(result.get("per_query", []))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "question_id": str(row["id"]),
                        "hypothesis": str(row["prediction"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
