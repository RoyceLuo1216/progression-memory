from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token for m in _TOKEN_RE.finditer(text)
        if (token := m.group(0).casefold()) not in _STOPWORDS
    ]


@dataclass(frozen=True)
class RetrievalDocument:
    id: str
    text: str
    evidence_ids: frozenset[str] = field(default_factory=frozenset)
    timestamp: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    text: str
    evidence_ids: frozenset[str]
    documents: tuple[RetrievalDocument, ...]
    category: str
    answer: str = ""


class BM25Index:
    """Small dependency-free BM25 implementation used as the common retriever."""

    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(doc.text)) for doc in self.documents]
        self.lengths = [sum(tf.values()) for tf in self.term_frequencies]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.term_frequencies:
            df.update(tf.keys())
        n = max(1, len(self.documents))
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> list[tuple[float, RetrievalDocument]]:
        q_terms = tokenize(query)
        scored: list[tuple[float, RetrievalDocument]] = []
        for doc, tf, dl in zip(
            self.documents,
            self.term_frequencies,
            self.lengths,
            strict=True,
        ):
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self.k1 * (
                    1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0)
                )
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0) / norm)
            scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return scored

    def retrieve(self, query: str, k: int) -> list[RetrievalDocument]:
        return [doc for _, doc in self.score(query)[:k]]


def _empty_bucket() -> dict[str, float | int]:
    return {
        "count": 0,
        "any_hits": 0,
        "all_hits": 0,
        "macro_recall_sum": 0.0,
        "micro_hits": 0,
        "micro_total": 0,
        "reciprocal_rank_sum": 0.0,
        "retrieved_chars": 0,
        "retrieved_units": 0,
    }


def _finalize_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    count = int(bucket["count"])
    denom = max(1, count)
    return {
        "query_count": count,
        "any_evidence_recall": float(bucket["any_hits"]) / denom,
        "all_evidence_recall": float(bucket["all_hits"]) / denom,
        "macro_evidence_recall": float(bucket["macro_recall_sum"]) / denom,
        "micro_evidence_recall": float(bucket["micro_hits"]) / max(1, int(bucket["micro_total"])),
        "mrr": float(bucket["reciprocal_rank_sum"]) / denom,
        "avg_retrieved_chars": float(bucket["retrieved_chars"]) / denom,
        "avg_retrieved_units": float(bucket["retrieved_units"]) / denom,
    }


def _query_outcome(
    query: RetrievalQuery,
    selected: list[tuple[int, RetrievalDocument]],
) -> dict[str, object]:
    retrieved_evidence: set[str] = set()
    retrieved_chars = 0
    first_evidence_rank = 0
    for rank, doc in selected:
        doc_evidence = doc.evidence_ids or {doc.id}
        retrieved_evidence.update(doc_evidence)
        retrieved_chars += len(doc.text)
        if not first_evidence_rank and query.evidence_ids & doc_evidence:
            first_evidence_rank = rank
    matched = query.evidence_ids & retrieved_evidence
    return {
        "id": query.id,
        "category": query.category,
        "recall": len(matched) / len(query.evidence_ids),
        "matched_evidence": len(matched),
        "total_evidence": len(query.evidence_ids),
        "any_evidence": bool(matched),
        "all_evidence": query.evidence_ids.issubset(retrieved_evidence),
        "reciprocal_rank": 1.0 / first_evidence_rank if first_evidence_rank else 0.0,
        "retrieved_chars": retrieved_chars,
        "retrieved_units": len(selected),
    }


def _accumulate(bucket: dict[str, float | int], row: dict[str, object]) -> None:
    bucket["count"] = int(bucket["count"]) + 1
    bucket["any_hits"] = int(bucket["any_hits"]) + int(bool(row["any_evidence"]))
    bucket["all_hits"] = int(bucket["all_hits"]) + int(bool(row["all_evidence"]))
    bucket["macro_recall_sum"] = float(bucket["macro_recall_sum"]) + float(row["recall"])
    bucket["micro_hits"] = int(bucket["micro_hits"]) + int(row["matched_evidence"])
    bucket["micro_total"] = int(bucket["micro_total"]) + int(row["total_evidence"])
    bucket["reciprocal_rank_sum"] = float(bucket["reciprocal_rank_sum"]) + float(row["reciprocal_rank"])
    bucket["retrieved_chars"] = int(bucket["retrieved_chars"]) + int(row["retrieved_chars"])
    bucket["retrieved_units"] = int(bucket["retrieved_units"]) + int(row["retrieved_units"])


def _aggregate_outcomes(rows: list[dict[str, object]]) -> dict[str, object]:
    global_bucket = _empty_bucket()
    categories: defaultdict[str, dict[str, float | int]] = defaultdict(_empty_bucket)
    for row in rows:
        _accumulate(global_bucket, row)
        _accumulate(categories[str(row["category"])], row)
    payload: dict[str, object] = _finalize_bucket(global_bucket)
    payload["by_category"] = {
        category: _finalize_bucket(bucket)
        for category, bucket in sorted(categories.items())
    }
    return payload


def evaluate_retrieval(
    queries: Iterable[RetrievalQuery],
    cutoffs: Iterable[int],
) -> dict[str, object]:
    """Evaluate top-k retrieval. MRR is truncated at each requested k."""

    query_list = list(queries)
    eligible = [q for q in query_list if q.evidence_ids]
    cutoff_values = sorted(set(cutoffs))
    max_k = max(cutoff_values, default=0)
    results: dict[str, object] = {
        "query_count": len(query_list),
        "eligible_query_count": len(eligible),
        "excluded_without_evidence": len(query_list) - len(eligible),
        "cutoffs": {},
    }
    index_cache: dict[int, BM25Index] = {}
    rows_by_cutoff: dict[int, list[dict[str, object]]] = {k: [] for k in cutoff_values}

    for query in eligible:
        corpus_key = id(query.documents)
        index = index_cache.get(corpus_key)
        if index is None:
            index = BM25Index(query.documents)
            index_cache[corpus_key] = index
        ranked = index.score(query.text)[:max_k]
        for k in cutoff_values:
            selected = [(rank, doc) for rank, (_, doc) in enumerate(ranked[:k], start=1)]
            rows_by_cutoff[k].append(_query_outcome(query, selected))

    for k in cutoff_values:
        results["cutoffs"][str(k)] = _aggregate_outcomes(rows_by_cutoff[k])
    return results


def evaluate_retrieval_by_char_budget(
    queries: Iterable[RetrievalQuery],
    budgets: Iterable[int],
    *,
    include_per_query: bool = False,
) -> dict[str, object]:
    """Evaluate ranked-prefix retrieval under strict whole-unit character budgets.

    Retrieval stops before the first ranked memory unit that would exceed the budget.
    This intentionally exposes coarse-unit inefficiency instead of truncating a memory
    unit and crediting evidence that may fall outside retained text.
    """

    query_list = list(queries)
    eligible = [q for q in query_list if q.evidence_ids]
    budget_values = sorted({int(b) for b in budgets if int(b) >= 0})
    results: dict[str, object] = {
        "query_count": len(query_list),
        "eligible_query_count": len(eligible),
        "excluded_without_evidence": len(query_list) - len(eligible),
        "budgets": {},
    }
    index_cache: dict[int, BM25Index] = {}
    rows_by_budget: dict[int, list[dict[str, object]]] = {b: [] for b in budget_values}

    for query in eligible:
        corpus_key = id(query.documents)
        index = index_cache.get(corpus_key)
        if index is None:
            index = BM25Index(query.documents)
            index_cache[corpus_key] = index
        ranked = index.score(query.text)
        for budget in budget_values:
            selected: list[tuple[int, RetrievalDocument]] = []
            chars = 0
            for rank, (_, doc) in enumerate(ranked, start=1):
                size = len(doc.text)
                if chars + size > budget:
                    break
                selected.append((rank, doc))
                chars += size
            rows_by_budget[budget].append(_query_outcome(query, selected))

    for budget in budget_values:
        rows = rows_by_budget[budget]
        payload = _aggregate_outcomes(rows)
        payload["char_budget"] = budget
        payload["avg_budget_utilization"] = (
            float(payload["avg_retrieved_chars"]) / budget if budget else 0.0
        )
        if include_per_query:
            payload["per_query"] = rows
        results["budgets"][str(budget)] = payload
    return results
