from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument


class TextEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    """Lazy sentence-transformer adapter with a run-local text embedding cache.

    Retrieval V3 reuses the same corpus text at progression construction, hierarchy,
    branch ranking, global rescue, and final reranking. Caching by exact text makes
    those later stages cheap instead of repeatedly running the transformer for every
    query. The cache lives for the benchmark process and therefore does not change
    retrieval semantics.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)

    @property
    def model(self):
        return _load_sentence_transformer(self.model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Deduplicate misses before invoking the model. This is especially important
        # for candidate subsets, whose documents have already been embedded globally.
        missing: list[str] = []
        seen_missing: set[str] = set()
        for text in texts:
            if text in self._cache:
                self.cache_hits += 1
            elif text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)

        if missing:
            values = self.model.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            values = np.asarray(values, dtype=np.float32)
            for text, value in zip(missing, values, strict=True):
                self._cache[text] = value
            self.cache_misses += len(missing)

        return np.stack([self._cache[text] for text in texts]).astype(np.float32, copy=False)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without semantic extra
        raise RuntimeError(
            "Semantic retrieval requires the 'semantic' extra: pip install -e '.[semantic]'"
        ) from exc
    return SentenceTransformer(model_name)


def cosine_scores(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    if document_embeddings.size == 0:
        return np.empty((0,), dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    docs = np.asarray(document_embeddings, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    doc_norms = np.linalg.norm(docs, axis=1)
    denom = np.maximum(doc_norms * max(query_norm, 1e-12), 1e-12)
    return (docs @ query) / denom


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo <= 1e-12:
        return {key: (1.0 if hi > 0 else 0.0) for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


class SemanticIndex:
    def __init__(self, documents: Sequence[RetrievalDocument], embedder: TextEmbedder):
        self.documents = tuple(documents)
        self.embedder = embedder
        self.embeddings = embedder.encode([document.text for document in self.documents])

    def score(self, query: str) -> list[tuple[float, RetrievalDocument]]:
        query_embedding = self.embedder.encode([query])[0]
        scores = cosine_scores(query_embedding, self.embeddings)
        rows = [(float(score), doc) for score, doc in zip(scores, self.documents, strict=True)]
        return sorted(rows, key=lambda row: (-row[0], row[1].id))


class HybridIndex:
    """Convex combination of normalized BM25 and semantic cosine scores."""

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        embedder: TextEmbedder,
        *,
        semantic_weight: float = 0.7,
    ):
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1]")
        self.documents = tuple(documents)
        self.semantic_weight = semantic_weight
        self.semantic = SemanticIndex(self.documents, embedder)
        self.lexical = BM25Index(self.documents)

    def score(self, query: str) -> list[tuple[float, RetrievalDocument]]:
        semantic = _minmax({doc.id: score for score, doc in self.semantic.score(query)})
        lexical = _minmax({doc.id: score for score, doc in self.lexical.score(query)})
        by_id = {doc.id: doc for doc in self.documents}
        scores = {
            doc_id: self.semantic_weight * semantic.get(doc_id, 0.0)
            + (1.0 - self.semantic_weight) * lexical.get(doc_id, 0.0)
            for doc_id in by_id
        }
        return sorted(((score, by_id[doc_id]) for doc_id, score in scores.items()), key=lambda row: (-row[0], row[1].id))
