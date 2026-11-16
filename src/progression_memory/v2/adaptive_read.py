"""Reusable query-time wiki/progression read policy.

Does not modify writer, stored revisions, or maintenance. The caller can
provide maintained wiki pages and an optional ranking function (e.g. real
semantic+BM25 fusion). The policy chooses one state view per query:
a compact current wiki page for ordinary questions, or the corresponding
compact progression history for temporal questions, with wiki fallback.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import NamedTuple

from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.state_engine import StateEngine

_TEMPORAL = re.compile(
    r"\b(before|after|previous|previously|history|historical|over time|"
    r"earliest|latest|first|original|changed|changes|transition|sequence|"
    r"used to|formerly|in order|past)\b", re.IGNORECASE
)
Ranker = Callable[[str, list[RetrievalDocument]], list[RetrievalDocument]]


class AdaptiveRead(NamedTuple):
    text: str
    documents: tuple[RetrievalDocument, ...]
    used_chars: int
    includes_history: bool


def asks_for_history(query: str) -> bool:
    """Conservative English lexical router; callers may override explicitly."""
    return bool(_TEMPORAL.search(query))


def compact_progression(doc: RetrievalDocument, *, history: bool) -> RetrievalDocument:
    """Compact a materialized V2 state document without altering its provenance."""
    payload = json.loads(doc.text)
    name = str(payload.get("progression", ""))
    state = dict(payload.get("direct_state") or {})
    prior = state.pop("history", None)
    if "value" in state:
        value = state.pop("value")
        text = f"Topic: {name}. Current value: {value}."
        if state:
            text += " " + "; ".join(f"{k}: {v}" for k,v in sorted(state.items())) + "."
    else:
        text = f"Progression: {name}. " + "; ".join(
            f"{k}: {v}" for k,v in sorted(state.items())
        )
    if history and prior:
        if isinstance(prior, list):
            sequence = "; ".join(str(v) for v in prior)
        else:
            sequence = str(prior)
        text += " Historical values in order: " + sequence + "."
    return RetrievalDocument(
        id=f"compact:{doc.id}", text=text, evidence_ids=doc.evidence_ids,
        timestamp=doc.timestamp,
        metadata={**doc.metadata, "kind": "compact_progression"},
    )


def _topic_key(doc: RetrievalDocument) -> str:
    match = re.match(r"^(?:Topic|Progression):\\s*([^.]+)\\.", doc.text)
    return (match.group(1).casefold().strip() if match
            else doc.id.split(":")[-1].casefold())


def _merge_by_topic(
    primary: list[RetrievalDocument], fallback: Sequence[RetrievalDocument],
) -> list[RetrievalDocument]:
    seen = {_topic_key(d) for d in primary}
    output = list(primary)
    for doc in fallback:
        topic = _topic_key(doc)
        if topic not in seen:
            output.append(doc)
            seen.add(topic)
    return output


def pack_skip_oversized(
    ranked: Sequence[RetrievalDocument], budget: int,
) -> tuple[tuple[RetrievalDocument, ...], int]:
    """Whole-document packing that does not block on an oversized top result."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    chosen: list[RetrievalDocument] = []
    used = 0
    for doc in ranked:
        if used + len(doc.text) <= budget:
            chosen.append(doc)
            used += len(doc.text)
    return tuple(chosen), used


def read_adaptive(
    engine: StateEngine,
    query: str,
    *,
    subject_id: str,
    budget_chars: int = 1600,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    wiki_current: Sequence[RetrievalDocument] = (),
    wiki_history: Sequence[RetrievalDocument] = (),
    source_rescue: bool = False,
    include_history: bool | None = None,
    ranker: Ranker | None = None,
) -> AdaptiveRead:
    """Build source-safe context from compact wiki/V2, optionally adding raw turns.

    Wiki pages supplied by the caller MUST already respect the requested time
    horizon; the engine's own progression and raw observations are time-filtered
    here. No gold labels are required or consulted. An injected ranker may use
    semantic/dense retrieval, and should not use future documents.
    """
    wants_history = asks_for_history(query) if include_history is None else include_history
    state_docs = list(build_documents(
        engine, view="state", subject_id=subject_id,
        valid_at=valid_at, known_at=known_at,
    ))
    compact = [compact_progression(doc, history=wants_history)
               for doc in state_docs]

    # One representation per query, not duplicate wiki and progression
    # summaries of identical topics. Temporal questions use compact V2 if
    # it has a history; otherwise fall back to versioned wiki pages.
    if wants_history:
        with_history = [doc for doc in compact
                        if "Historical values in order:" in doc.text]
        no_history = [doc for doc in compact
                      if "Historical values in order:" not in doc.text]
        active = _merge_by_topic(with_history, wiki_history)
        active = _merge_by_topic(active, no_history)
    else:
        active = _merge_by_topic(list(wiki_current), compact)

    if source_rescue:
        source_docs = list(build_documents(
            engine, view="evidence", subject_id=subject_id,
            valid_at=valid_at, known_at=known_at,
        ))
        # Evidence is an optional rescue, not a replacement for maintained state.
        active = [*active, *source_docs]
    if ranker is None:
        ranked = BM25Index(active).retrieve(query, len(active)) if active else []
    else:
        ranked = ranker(query, active)
        if len({doc.id for doc in ranked}) != len(ranked):
            raise ValueError("ranker returned duplicate retrieval units")
        if not {doc.id for doc in ranked}.issubset({doc.id for doc in active}):
            raise ValueError("ranker returned a document outside the permitted corpus")
    selected, used = pack_skip_oversized(ranked, budget_chars)
    return AdaptiveRead(
        text="\n".join(doc.text for doc in selected),
        documents=selected,used_chars=used,includes_history=wants_history,
    )
