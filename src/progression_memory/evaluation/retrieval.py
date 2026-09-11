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
    }


def evaluate_retrieval(
    queries: Iterable[RetrievalQuery],
    cutoffs: Iterable[int],
) -> dict[str, object]:
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

    global_buckets = {k: _empty_bucket() for k in cutoff_values}
    category_buckets = {k: defaultdict(_empty_bucket) for k in cutoff_values}
    index_cache: dict[int, BM25Index] = {}

    for query in eligible:
        corpus_key = id(query.documents)
        index = index_cache.get(corpus_key)
        if index is None:
            index = BM25Index(query.documents)
            index_cache[corpus_key] = index
        ranked = index.score(query.text)
        first_evidence_rank = 0
        for rank, (_, doc) in enumerate(ranked, start=1):
            doc_evidence = doc.evidence_ids or {doc.id}
            if query.evidence_ids & doc_evidence:
                first_evidence_rank = rank
                break

        top_docs = [doc for _, doc in ranked[:max_k]]
        evidence_prefix: set[str] = set()
        chars_prefix = 0
        cutoff_set = set(cutoff_values)
        snapshots: dict[int, tuple[set[str], int]] = {}
        for rank, doc in enumerate(top_docs, start=1):
            evidence_prefix.update(doc.evidence_ids or {doc.id})
            chars_prefix += len(doc.text)
            if rank in cutoff_set:
                snapshots[rank] = (set(evidence_prefix), chars_prefix)
        for k in cutoff_values:
            if k not in snapshots:
                snapshots[k] = (set(evidence_prefix), chars_prefix)

        for k in cutoff_values:
            retrieved_evidence, retrieved_chars = snapshots[k]
            matched = query.evidence_ids & retrieved_evidence
            recall = len(matched) / len(query.evidence_ids)
            rr = 1.0 / first_evidence_rank if first_evidence_rank else 0.0
            for bucket in (global_buckets[k], category_buckets[k][query.category]):
                bucket["count"] = int(bucket["count"]) + 1
                bucket["any_hits"] = int(bucket["any_hits"]) + int(bool(matched))
                bucket["all_hits"] = int(bucket["all_hits"]) + int(
                    query.evidence_ids.issubset(retrieved_evidence)
                )
                bucket["macro_recall_sum"] = float(bucket["macro_recall_sum"]) + recall
                bucket["micro_hits"] = int(bucket["micro_hits"]) + len(matched)
                bucket["micro_total"] = int(bucket["micro_total"]) + len(query.evidence_ids)
                bucket["reciprocal_rank_sum"] = float(bucket["reciprocal_rank_sum"]) + rr
                bucket["retrieved_chars"] = int(bucket["retrieved_chars"]) + retrieved_chars

    for k in cutoff_values:
        payload = _finalize_bucket(global_buckets[k])
        payload["by_category"] = {
            category: _finalize_bucket(bucket)
            for category, bucket in sorted(category_buckets[k].items())
        }
        results["cutoffs"][str(k)] = payload
    return results
