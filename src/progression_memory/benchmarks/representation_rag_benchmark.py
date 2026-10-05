"""Source-level dense RAG baseline; progression stores can reuse evaluate()."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from progression_memory.evaluation.datasets import load_locomo, load_longmemeval
from progression_memory.evaluation.qa import MultiEvidenceExtractiveReader, answer_contains, token_f1
from progression_memory.evaluation.representation_rag import records
from progression_memory.evaluation.retrieval import RetrievalQuery
from progression_memory.evaluation.semantic import SemanticIndex, TextEmbedder
from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, Memory, TargetUpdate,
)
from progression_memory.evaluation.progressions import parse_timestamp


class PinnedEncoder:
    """Load one exact sentence-transformer checkpoint for paired experiments."""

    def __init__(self, model: str, revision: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name, self.revision = model, revision
        self.model = SentenceTransformer(model, revision=revision)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def append_memory(query: RetrievalQuery) -> Memory:
    memory = Memory()
    for ordinal, source in enumerate(query.documents):
        if source.id in memory.events:
            raise ValueError(f"duplicate source id: {source.id}")
        memory.events[source.id] = Event(source.id, "corpus", source.text,
                                         parse_timestamp(source.timestamp, ordinal), source.id)
    return memory


def replay_decisions(query: RetrievalQuery, rows: list[dict]) -> Memory:
    """Replay an independently produced writer trace against original sources."""
    if len(rows) != len(query.documents):
        raise ValueError("writer trace must cover every source event in order")
    memory = Memory()
    for ordinal, (source, row) in enumerate(zip(query.documents, rows, strict=True)):
        if row.get("source_id") != source.id:
            raise ValueError("writer trace source order/id mismatch")
        raw = row["decision"]
        decision = Decision(
            action=Action(raw["action"]), target_ids=tuple(raw.get("target_ids", [])),
            parent_id=raw.get("parent_id"), identity=raw.get("identity", ""),
            scope=raw.get("scope", ""), state=raw.get("state", ""),
            change=raw.get("change", ""), new_segment=raw.get("new_segment", False),
            reason=raw["reason"],
            target_updates=tuple(TargetUpdate(**item) for item in raw.get("target_updates", [])),
        )
        memory.apply(Event(source.id, row["subject_id"], source.text,
                           parse_timestamp(source.timestamp, ordinal), source.id), decision)
    return memory


def corpus_key(query: RetrievalQuery, dataset: str) -> str:
    return query.id.rsplit(":q", 1)[0] if dataset == "locomo" else query.id


def evaluate(
    queries: list[RetrievalQuery], embedder: TextEmbedder, *,
    representation: str, search_depth: int | None, context_chars: int,
    memory_by_corpus: dict[int, Memory] | None = None,
) -> dict:
    """Identical ranker, reader, budget and source metric for either store.

    A prepared progression memory is keyed by id(query.documents). If omitted,
    the source observations are stored independently as ordinary RAG chunks.
    """
    if (search_depth is not None and search_depth < 1) or context_chars < 0:
        raise ValueError("search depth and context budget must be explicit")
    reader = MultiEvidenceExtractiveReader()
    indexes: dict[int, tuple[Memory, SemanticIndex]] = {}
    per_query = []
    by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for query in queries:
        key = id(query.documents)
        if key not in indexes:
            memory = (memory_by_corpus[key] if memory_by_corpus is not None
                      else append_memory(query))
            if set(memory.events) != {doc.id for doc in query.documents}:
                raise ValueError("representations must contain identical source events")
            indexes[key] = memory, SemanticIndex(records(memory, representation), embedder)
        memory, index = indexes[key]
        hits = [doc for _, doc in index.score(query.text)[:search_depth]]
        context = []
        delivered = set()
        used = 0
        for hit in hits:
            order = (memory.segments[hit.id].event_ids if hit.id in memory.segments
                     else [hit.id])
            for event_id in order:
                if event_id in delivered:
                    continue
                source = memory.events[event_id]
                if used + len(source.text) > context_chars:
                    continue
                context.append(next(doc for doc in query.documents if doc.id == event_id))
                used += len(source.text)
                delivered.add(event_id)
            if representation == "progression" and "progression_id" in hit.metadata:
                summary = "\n".join(hit.text.split("\n", 3)[:3])
                if used + len(summary) <= context_chars:
                    from progression_memory.evaluation.retrieval import RetrievalDocument
                    context.append(RetrievalDocument(f"summary:{hit.id}", summary))
                    used += len(summary)
        prediction = reader.answer(query.text, tuple(context))
        row = {"id": query.id, "category": query.category,
               "retrieval_recall": len(delivered & query.evidence_ids) / len(query.evidence_ids)
               if query.evidence_ids else None,
               "answer_contains": answer_contains(prediction, query.answer),
               "token_f1": token_f1(prediction, query.answer),
               "context_chars": used, "source_ids": sorted(delivered),
               "indexed_record_ids": [hit.id for hit in hits]}
        per_query.append(row)
        by_category[query.category].append(row)

    def aggregate(rows: list[dict]) -> dict:
        evidence = [row["retrieval_recall"] for row in rows
                    if row["retrieval_recall"] is not None]
        return {"questions": len(rows), "evidence_questions": len(evidence),
                "retrieval_recall": sum(evidence) / len(evidence) if evidence else None,
                "answer_contains": sum(r["answer_contains"] for r in rows) / len(rows) if rows else None,
                "token_f1": sum(r["token_f1"] for r in rows) / len(rows) if rows else None,
                "avg_context_chars": sum(r["context_chars"] for r in rows) / len(rows) if rows else None}

    return {"representation": representation, "reader": reader.name,
            "search_depth": search_depth, "context_chars": context_chars,
            "question_count": len(queries),
            "question_ids_sha256": hashlib.sha256("\n".join(q.id for q in queries).encode()).hexdigest(),
            "overall": aggregate(per_query),
            "by_category": {cat: aggregate(rows) for cat, rows in by_category.items()},
            "per_query": per_query}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--representation", choices=("append", "structure", "progression"),
                        default="append")
    parser.add_argument("--decision-traces", type=Path,
                        help="JSON map of corpus key to ordered per-source writer decisions")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--search-depth", required=True,
                        help="Positive integer or 'all' to rank the full index")
    parser.add_argument("--context-chars", type=int, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search_depth = None if args.search_depth == "all" else int(args.search_depth)
    queries = (load_locomo(args.data_path) if args.dataset == "locomo"
               else load_longmemeval(args.data_path))
    if args.sample_size is not None:
        if args.sample_size < 1:
            raise ValueError("sample-size must be positive")
        queries = sorted(queries, key=lambda q: hashlib.sha256(q.id.encode()).hexdigest())[:args.sample_size]
    memories = None
    if args.representation != "append":
        if args.decision_traces is None:
            raise ValueError("progression representations require audited writer decisions")
        traces = json.loads(args.decision_traces.read_text())
        memories = {}
        for query in queries:
            key = id(query.documents)
            if key not in memories:
                name = corpus_key(query, args.dataset)
                memories[key] = replay_decisions(query, traces[name])
    result = evaluate(queries, PinnedEncoder(args.semantic_model, args.model_revision),
                      representation=args.representation, search_depth=search_depth,
                      context_chars=args.context_chars, memory_by_corpus=memories)
    result.update(dataset=args.dataset, dataset_sha256=hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
                  encoder=args.semantic_model, encoder_revision=args.model_revision,
                  writer_trace=str(args.decision_traces) if args.decision_traces else None,
                  note="Deterministic reader; progression mode replays supplied writer decisions.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("dataset", "question_count", "overall")}, indent=2))


if __name__ == "__main__":
    main()
