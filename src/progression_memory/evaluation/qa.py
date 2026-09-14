from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
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


@dataclass(frozen=True)
class QueryPlan:
    kind: str
    needs_multiple_evidence: bool
    temporal: bool
    prefer_recent: bool


def plan_query(question: str) -> QueryPlan:
    """Infer a lightweight retrieval/reader plan without benchmark labels.

    This is intentionally lexical and deterministic. It gives the QA pipeline a
    query-complexity hook that can later be replaced by an LLM/learned planner while
    keeping the same downstream interfaces.
    """
    q = question.casefold()
    aggregation = any(
        cue in q
        for cue in (
            "how many",
            "how much",
            "total",
            "all ",
            "both ",
            "each ",
            "list ",
            "which ",
            "what are",
        )
    )
    temporal = any(
        cue in q
        for cue in (
            "when ",
            "before ",
            "after ",
            "first ",
            "last ",
            "earliest",
            "latest",
            "recent",
            "ago",
            "how long",
        )
    )
    update = any(
        cue in q
        for cue in (
            "currently",
            "current ",
            "now ",
            "latest",
            "most recent",
            "still ",
            "ended up",
        )
    )
    preference = any(
        cue in q for cue in ("prefer", "preference", "favorite", "favourite", "like", "dislike")
    )

    if aggregation:
        kind = "aggregation"
    elif update:
        kind = "update"
    elif temporal:
        kind = "temporal"
    elif preference:
        kind = "preference"
    else:
        kind = "fact"
    return QueryPlan(
        kind=kind,
        needs_multiple_evidence=aggregation or temporal or update,
        temporal=temporal or update,
        prefer_recent=update,
    )


class ExtractiveBM25Reader:
    """Deterministic no-API reader for CI and architecture ablations."""

    name = "extractive_bm25"

    def answer(self, question: str, context: tuple[RetrievalDocument, ...]) -> str:
        spans = _sentence_documents(context)
        if not spans:
            return "I don't know."
        ranked = BM25Index(spans).score(question)
        if not ranked or ranked[0][0] <= 0.0:
            return "I don't know."
        return ranked[0][1].text


class MultiEvidenceExtractiveReader:
    """Return a compact evidence bundle instead of one sentence.

    This is still not a reasoning model. It is a diagnostic reader that tests whether
    QA failures come from forcing a multi-hop/update question through a single-span
    bottleneck. It selects distinct source documents where possible and orders
    temporal/update evidence chronologically for easier downstream synthesis.
    """

    name = "multi_evidence_extractive"

    def __init__(self, max_spans: int = 4) -> None:
        if max_spans < 1:
            raise ValueError("max_spans must be positive")
        self.max_spans = max_spans

    def answer(self, question: str, context: tuple[RetrievalDocument, ...]) -> str:
        spans = _sentence_documents(context)
        if not spans:
            return "I don't know."
        ranked = [(score, span) for score, span in BM25Index(spans).score(question) if score > 0.0]
        if not ranked:
            return "I don't know."

        plan = plan_query(question)
        target = self.max_spans if plan.needs_multiple_evidence else min(2, self.max_spans)
        selected: list[RetrievalDocument] = []
        source_counts: Counter[str] = Counter()
        for _, span in ranked:
            source_id = str(span.metadata.get("source_document_id", span.id))
            per_source_cap = 2 if plan.temporal else 1
            if source_counts[source_id] >= per_source_cap:
                continue
            selected.append(span)
            source_counts[source_id] += 1
            if len(selected) >= target:
                break

        if plan.temporal and all(span.timestamp is not None for span in selected):
            selected.sort(key=lambda span: str(span.timestamp))
        return "\n".join(span.text for span in selected)


def _sentence_documents(
    context: tuple[RetrievalDocument, ...],
) -> tuple[RetrievalDocument, ...]:
    spans: list[RetrievalDocument] = []
    for document in context:
        for index, sentence in enumerate(_SENTENCE_SPLIT.split(document.text)):
            sentence = sentence.strip()
            if not sentence:
                continue
            spans.append(
                RetrievalDocument(
                    id=f"{document.id}:span:{index}",
                    text=sentence,
                    evidence_ids=document.evidence_ids,
                    timestamp=document.timestamp,
                    metadata={
                        **document.metadata,
                        "source_document_id": document.metadata.get(
                            "source_document_id", document.id
                        ),
                        "span_index": index,
                    },
                )
            )
    return tuple(spans)


@dataclass(frozen=True)
class QARunConfig:
    top_k: int | None = None
    char_budget: int | None = None
    context_strategy: str = "flat"
    neighbor_spans: int = 1
    max_packets_per_document: int = 2

    def __post_init__(self) -> None:
        if self.top_k is None and self.char_budget is None:
            raise ValueError("top_k or char_budget must be set")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.char_budget is not None and self.char_budget < 0:
            raise ValueError("char_budget must be non-negative")
        if self.context_strategy not in {"flat", "evidence_pack"}:
            raise ValueError("context_strategy must be flat or evidence_pack")
        if self.neighbor_spans < 0:
            raise ValueError("neighbor_spans must be non-negative")
        if self.max_packets_per_document < 1:
            raise ValueError("max_packets_per_document must be positive")


def _fits_budget(used_chars: int, text: str, char_budget: int | None) -> bool:
    return char_budget is None or used_chars + len(text) <= char_budget


def _select_flat_context(
    query: RetrievalQuery,
    config: QARunConfig,
) -> tuple[RetrievalDocument, ...]:
    scored = BM25Index(query.documents).score(query.text)
    selected: list[RetrievalDocument] = []
    used_chars = 0
    for _, document in scored:
        if config.top_k is not None and len(selected) >= config.top_k:
            break
        if not _fits_budget(used_chars, document.text, config.char_budget):
            # Do not stop the search just because one large document does not fit.
            # A later smaller and still-relevant memory may use the remaining budget.
            continue
        selected.append(document)
        used_chars += len(document.text)
    return tuple(selected)


def _render_packet(
    document: RetrievalDocument,
    sentences: list[str],
    start: int,
    end: int,
) -> RetrievalDocument:
    body = " ".join(sentences[start:end]).strip()
    prefix = f"[{document.timestamp}] " if document.timestamp else ""
    return RetrievalDocument(
        id=f"packet:{document.id}:{start}:{end}",
        text=prefix + body,
        evidence_ids=document.evidence_ids,
        timestamp=document.timestamp,
        metadata={
            **document.metadata,
            "source_document_id": document.id,
            "span_start": start,
            "span_end": end,
        },
    )


def _select_evidence_pack_context(
    query: RetrievalQuery,
    config: QARunConfig,
) -> tuple[RetrievalDocument, ...]:
    """Build compact, dated evidence packets around query-relevant sentence anchors."""
    candidates: list[tuple[float, RetrievalDocument]] = []
    for document in query.documents:
        sentences = [part.strip() for part in _SENTENCE_SPLIT.split(document.text) if part.strip()]
        if not sentences:
            continue
        sentence_docs = tuple(
            RetrievalDocument(
                id=f"{document.id}:anchor:{index}",
                text=sentence,
                evidence_ids=document.evidence_ids,
                timestamp=document.timestamp,
            )
            for index, sentence in enumerate(sentences)
        )
        for score, span in BM25Index(sentence_docs).score(query.text):
            if score <= 0.0:
                continue
            span_index = int(span.id.rsplit(":", 1)[1])
            start = max(0, span_index - config.neighbor_spans)
            end = min(len(sentences), span_index + config.neighbor_spans + 1)
            candidates.append((score, _render_packet(document, sentences, start, end)))

    candidates.sort(key=lambda item: (-item[0], item[1].id))
    plan = plan_query(query.text)
    selected: list[RetrievalDocument] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    used_chars = 0

    # Aggregation questions need breadth across sessions/sources. Temporal/update
    # questions may need two packets from one source to retain local chronology.
    per_source_cap = 1 if plan.kind == "aggregation" else config.max_packets_per_document
    for _, packet in candidates:
        source_id = str(packet.metadata["source_document_id"])
        if source_counts[source_id] >= per_source_cap or packet.id in selected_ids:
            continue
        if config.top_k is not None and len(selected) >= config.top_k:
            break
        if not _fits_budget(used_chars, packet.text, config.char_budget):
            continue
        selected.append(packet)
        selected_ids.add(packet.id)
        source_counts[source_id] += 1
        used_chars += len(packet.text)

    if plan.temporal and all(packet.timestamp is not None for packet in selected):
        selected.sort(key=lambda packet: str(packet.timestamp))
    return tuple(selected)


def select_reader_context(
    query: RetrievalQuery,
    config: QARunConfig,
) -> tuple[RetrievalDocument, ...]:
    if config.context_strategy == "evidence_pack":
        return _select_evidence_pack_context(query, config)
    return _select_flat_context(query, config)


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
        plan = plan_query(query.text)
        rows.append(
            {
                "id": query.id,
                "category": query.category,
                "query_plan": plan.kind,
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
    plans = sorted({str(row["query_plan"]) for row in rows})
    result: dict[str, object] = {
        "reader": reader.name,
        "context_policy": {
            "top_k": config.top_k,
            "char_budget": config.char_budget,
            "context_strategy": config.context_strategy,
            "neighbor_spans": config.neighbor_spans,
            "max_packets_per_document": config.max_packets_per_document,
        },
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate(
                [row for row in rows if str(row["category"]) == category]
            )
            for category in categories
        },
        "by_query_plan": {
            plan: aggregate([row for row in rows if str(row["query_plan"]) == plan])
            for plan in plans
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
