from __future__ import annotations

import math
import re
from collections import Counter
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

    def __init__(self, documents: Iterable[RetrievalDocument], *, k1: float = 1.5, b: float = 0.75) -> None:
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
        self.idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str) -> list[tuple[float, RetrievalDocument]]:
        q_terms = tokenize(query)
        scored: list[tuple[float, RetrievalDocument]] = []
        for doc, tf, dl in zip(self.documents, self.term_frequencies, self.lengths, strict=True):
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0) / norm)
            scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return scored

    def retrieve(self, query: str, k: int) -> list[RetrievalDocument]:
        return [doc for _, doc in self.score(query)[:k]]


def evaluate_retrieval(queries: Iterable[RetrievalQuery], cutoffs: Iterable[int]) -> dict[str, object]:
    query_list = list(queries)
    eligible = [q for q in query_list if q.evidence_ids]
    results: dict[str, object] = {
        "query_count": len(query_list),
        "eligible_query_count": len(eligible),
        "excluded_without_evidence": len(query_list) - len(eligible),
        "cutoffs": {},
    }
    for k in sorted(set(cutoffs)):
        any_hits = 0
        all_hits = 0
        macro_recall_sum = 0.0
        micro_hits = 0
        micro_total = 0
        reciprocal_rank_sum = 0.0
        retrieved_chars = 0
        for query in eligible:
            ranked = BM25Index(query.documents).score(query.text)
            top = [doc for _, doc in ranked[:k]]
            retrieved_evidence: set[str] = set()
            for doc in top:
                retrieved_evidence.update(doc.evidence_ids or {doc.id})
                retrieved_chars += len(doc.text)
            matched = query.evidence_ids & retrieved_evidence
            any_hits += bool(matched)
            all_hits += query.evidence_ids.issubset(retrieved_evidence)
            macro_recall_sum += len(matched) / len(query.evidence_ids)
            micro_hits += len(matched)
            micro_total += len(query.evidence_ids)

            rr = 0.0
            for rank, (_, doc) in enumerate(ranked, start=1):
                doc_evidence = doc.evidence_ids or {doc.id}
                if query.evidence_ids & doc_evidence:
                    rr = 1.0 / rank
                    break
            reciprocal_rank_sum += rr

        denom = max(1, len(eligible))
        results["cutoffs"][str(k)] = {
            "any_evidence_recall": any_hits / denom,
            "all_evidence_recall": all_hits / denom,
            "macro_evidence_recall": macro_recall_sum / denom,
            "micro_evidence_recall": micro_hits / max(1, micro_total),
            "mrr": reciprocal_rank_sum / denom,
            "avg_retrieved_chars": retrieved_chars / denom,
        }
    return results
