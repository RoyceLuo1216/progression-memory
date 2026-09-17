from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import math

from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery, tokenize
from progression_memory.evaluation.semantic import TextEmbedder
from progression_memory.evaluation.two_stage import (
    MAX_STAGE1_CANDIDATES, MIN_STAGE1_CANDIDATES, STAGE1_CORPUS_FRACTION,
    _compact_index_text, _sources_for_segment,
)
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_graph import ProgressionGraph

DEFAULT_PARENT_RRF_WEIGHT = 0.35
DEFAULT_RRF_K = 60.0
V2_PARENT_INDEX_TERMS = 12
V2_PARENT_CORE_TERMS = 4
V2_DIVERSITY_FRACTION = 0.25
V2_MAX_DIVERSITY_PARENTS = 8


def _adaptive_candidate_limit(structural_units: int) -> int:
    proportional = math.ceil(STAGE1_CORPUS_FRACTION * structural_units)
    return min(structural_units, max(MIN_STAGE1_CANDIDATES, min(MAX_STAGE1_CANDIDATES, proportional)))


def _segment_indexes(raw_documents: tuple[RetrievalDocument, ...], *, embedder: TextEmbedder | None = None) -> tuple[tuple[RetrievalDocument, ...], dict[str, list[RetrievalDocument]], dict[str, float | int | str]]:
    segments, stats = build_progression_documents(raw_documents, mode="adaptive_v2", embedder=embedder)
    indexes: list[RetrievalDocument] = []
    by_progression: defaultdict[str, list[RetrievalDocument]] = defaultdict(list)
    for segment in segments:
        progression_id = str(segment.metadata.get("progression_id"))
        index_document = RetrievalDocument(id=f"index:{segment.id}", text=_compact_index_text(segment), evidence_ids=frozenset(), timestamp=segment.timestamp, metadata={"segment_id": segment.id, "progression_id": progression_id, "source_documents": _sources_for_segment(segment, raw_documents)})
        indexes.append(index_document); by_progression[progression_id].append(index_document)
    return tuple(indexes), dict(by_progression), stats


def _graph_for_segments(by_progression: dict[str, list[RetrievalDocument]], segment_indexes: tuple[RetrievalDocument, ...]) -> ProgressionGraph:
    progressions = {pid: Progression(id=pid, title=pid, scope="") for pid in by_progression}
    graph = ProgressionGraph(progressions)
    for segment in segment_indexes: graph.add_segment(segment.id, str(segment.metadata["progression_id"]))
    if not graph.validate(): raise ValueError("constructed progression hierarchy is not acyclic")
    return graph


def _parent_index_text(children: list[RetrievalDocument]) -> str:
    seen: set[str] = set(); terms: list[str] = []
    for child in children:
        for token in tokenize(child.text):
            if token not in seen: seen.add(token); terms.append(token)
    return " ".join(terms)


def _parent_index_text_v2(children: list[RetrievalDocument]) -> str:
    counts: Counter[str] = Counter(); first_seen: dict[str, int] = {}; offset = 0
    for child in children:
        for token in set(tokenize(child.text)):
            counts[token] += 1; first_seen.setdefault(token, offset); offset += 1
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    core = ranked[:V2_PARENT_CORE_TERMS]; remainder = [t for t in ranked[V2_PARENT_CORE_TERMS:] if t not in core]
    return " ".join((core + remainder)[:V2_PARENT_INDEX_TERMS])


def _build_parent_documents(by_progression: dict[str, list[RetrievalDocument]], *, compact: bool = False) -> tuple[RetrievalDocument, ...]:
    docs = []
    for pid, children in sorted(by_progression.items()):
        text = _parent_index_text_v2(children) if compact else _parent_index_text(children)
        docs.append(RetrievalDocument(id=f"parent:{pid}", text=text, evidence_ids=frozenset(), metadata={"progression_id": pid}))
    return tuple(docs)


def _rrf_rank(query: str, segments: tuple[RetrievalDocument, ...], parents: tuple[RetrievalDocument, ...], *, parent_weight: float = DEFAULT_PARENT_RRF_WEIGHT) -> list[RetrievalDocument]:
    seg_rank = BM25Index(segments).score(query); parent_rank = BM25Index(parents).score(query)
    parent_pos = {str(doc.metadata["progression_id"]): i + 1 for i, (_, doc) in enumerate(parent_rank)}
    scored = []
    for i, (_, doc) in enumerate(seg_rank):
        pid = str(doc.metadata["progression_id"]); score = 1.0 / (DEFAULT_RRF_K + i + 1)
        if pid in parent_pos: score += parent_weight / (DEFAULT_RRF_K + parent_pos[pid])
        scored.append((score, doc))
    return [doc for _, doc in sorted(scored, key=lambda row: (-row[0], row[1].id))]


def _diversify(ranked: list[RetrievalDocument], limit: int) -> list[RetrievalDocument]:
    if not ranked: return []
    reserve = min(limit, max(1, math.ceil(V2_DIVERSITY_FRACTION * limit)), V2_MAX_DIVERSITY_PARENTS)
    selected: list[RetrievalDocument] = []; ids: set[str] = set(); seen_parents: set[str] = set()
    for doc in ranked:
        pid = str(doc.metadata["progression_id"])
        if pid not in seen_parents:
            selected.append(doc); ids.add(doc.id); seen_parents.add(pid)
            if len(selected) >= reserve: break
    for doc in ranked:
        if len(selected) >= limit: break
        if doc.id not in ids: selected.append(doc); ids.add(doc.id)
    return selected


def _build_dag_queries(queries: list[RetrievalQuery], *, v2: bool) -> tuple[list[RetrievalQuery], dict[str, object]]:
    cache: dict[int, tuple] = {}; transformed = []; selected_counts=[]; source_counts=[]; coverages=[]
    for query in queries:
        key=id(query.documents)
        if key not in cache:
            segments, by_progression, stats = _segment_indexes(query.documents); graph=_graph_for_segments(by_progression, segments); parents=_build_parent_documents(by_progression, compact=v2)
            cache[key]=(segments,by_progression,stats,graph,parents)
        segments, _, _, _, parents=cache[key]
        limit=_adaptive_candidate_limit(len(segments)); ranked=_rrf_rank(query.text,segments,parents)
        selected=_diversify(ranked,limit) if v2 else ranked[:limit]
        sources: OrderedDict[str, RetrievalDocument]=OrderedDict(); candidate_evidence:set[str]=set()
        for segment in selected:
            for source in segment.metadata.get("source_documents",()):
                if isinstance(source,RetrievalDocument): sources.setdefault(source.id,source); candidate_evidence.update(source.evidence_ids or {source.id})
        candidate_documents=tuple(sources.values()) or query.documents; final_docs=tuple(doc for _,doc in BM25Index(candidate_documents).score(query.text))
        transformed.append(RetrievalQuery(id=query.id,text=query.text,answer=query.answer,evidence_ids=query.evidence_ids,documents=final_docs,category=query.category))
        selected_counts.append(len(selected)); source_counts.append(len(candidate_documents))
        if query.evidence_ids: coverages.append(len(candidate_evidence & query.evidence_ids)/len(query.evidence_ids))
    rows=list(cache.values())
    return transformed,{"unique_corpora":len(cache),"dag_version":2 if v2 else 1,"avg_stage1_selected_units":sum(selected_counts)/max(1,len(selected_counts)),"avg_candidate_source_documents":sum(source_counts)/max(1,len(source_counts)),"avg_stage1_evidence_coverage":sum(coverages)/max(1,len(coverages)),"avg_progressions":sum(len(r[1]) for r in rows)/max(1,len(rows))}


def build_dag_two_stage_queries(queries: list[RetrievalQuery]) -> tuple[list[RetrievalQuery], dict[str, object]]: return _build_dag_queries(queries,v2=False)
def build_dag_two_stage_v2_queries(queries: list[RetrievalQuery]) -> tuple[list[RetrievalQuery], dict[str, object]]: return _build_dag_queries(queries,v2=True)
