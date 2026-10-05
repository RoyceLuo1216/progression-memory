"""Same dense retrieval and evidence budget over raw or progression records."""

from __future__ import annotations

from dataclasses import dataclass

from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.evaluation.semantic import SemanticIndex, TextEmbedder
from progression_memory.maintenance.calibrated_progression import Memory


@dataclass(frozen=True)
class RagResult:
    indexed_record_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    context: tuple[RetrievalDocument, ...]
    used_chars: int


def records(memory: Memory, representation: str) -> tuple[RetrievalDocument, ...]:
    if representation == "append":
        return tuple(RetrievalDocument(event.id, event.text, frozenset({event.id}))
                     for event in memory.events.values())
    if representation not in {"structure", "progression"}:
        raise ValueError(representation)
    output = []
    assigned = set()
    for progression in memory.progressions.values():
        for sid in progression.segment_ids:
            segment = memory.segments[sid]
            assigned.update(segment.event_ids)
            source_text = "\n".join(memory.events[eid].text for eid in segment.event_ids)
            if representation == "structure":
                text = f"{progression.identity}: {progression.scope}\n{source_text}"
            else:
                text = (f"{progression.identity}: {progression.scope}\n"
                        f"State: {segment.after.text}\nChange: {segment.change}\n{source_text}")
            output.append(RetrievalDocument(
                sid, text, frozenset(segment.event_ids),
                metadata={"progression_id": progression.id},
            ))
    for event in memory.events.values():
        if event.id not in assigned:
            output.append(RetrievalDocument(event.id, event.text, frozenset({event.id})))
    return tuple(output)


def retrieve(
    question: str, memory: Memory, embedder: TextEmbedder, *, representation: str,
    search_depth: int, context_chars: int,
) -> RagResult:
    """Index identical underlying events; all retrieved text counts toward budget.

    A progression record can carry several event pointers. It earns source recall
    only when its original event text actually fits in the reader context.
    """
    if search_depth < 1 or context_chars < 0:
        raise ValueError("search depth and context budget must be explicit and valid")
    indexed = records(memory, representation)
    selected = [doc for _, doc in SemanticIndex(indexed, embedder).score(question)[:search_depth]]
    context: list[RetrievalDocument] = []
    source_ids: list[str] = []
    seen = set()
    used = 0
    for unit in selected:
        # Source-level recall requires delivering actual evidence to the reader.
        source_order = (memory.segments[unit.id].event_ids if unit.id in memory.segments
                        else [unit.id])
        for event_id in source_order:
            if event_id in seen:
                continue
            event = memory.events[event_id]
            if used + len(event.text) > context_chars:
                continue
            seen.add(event_id)
            source_ids.append(event_id)
            context.append(RetrievalDocument(event.id, event.text, frozenset({event.id})))
            used += len(event.text)
        if representation == "progression" and "progression_id" in unit.metadata:
            # Charge the reader for the generated representation too. An index
            # hit alone does not count as evidence or grant a free summary.
            state = unit.text.split("\n", 3)
            summary = "\n".join(state[:3])
            if used + len(summary) <= context_chars:
                context.append(RetrievalDocument(f"summary:{unit.id}", summary))
                used += len(summary)
    return RagResult(tuple(doc.id for doc in selected), tuple(source_ids),
                     tuple(context), used)
