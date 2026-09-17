from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import math

from progression_memory.evaluation.progressions import build_progression_documents
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument, RetrievalQuery, tokenize
from progression_memory.evaluation.semantic import TextEmbedder
from progression_memory.evaluation.two_stage import MAX_STAGE1_CANDIDATES, MIN_STAGE1_CANDIDATES, STAGE1_CORPUS_FRACTION, _compact_index_text, _sources_for_segment
from progression_memory.models.progression import Progression
from progression_memory.storage.progression_graph import ProgressionGraph

DEFAULT_PARENT_RRF_WEIGHT=0.35; DEFAULT_RRF_K=60.0
V2_PARENT_INDEX_TERMS=12; V2_PARENT_CORE_TERMS=4; V2_DIVERSITY_FRACTION=0.25; V2_MAX_DIVERSITY_PARENTS=8

def _adaptive_candidate_limit(n:int)->int:
    return min(n,max(MIN_STAGE1_CANDIDATES,min(MAX_STAGE1_CANDIDATES,math.ceil(STAGE1_CORPUS_FRACTION*n))))

def _segment_indexes(raw_documents:tuple[RetrievalDocument,...],*,embedder:TextEmbedder|None=None):
    segments,stats=build_progression_documents(raw_documents,mode="adaptive_v2",embedder=embedder); indexes=[]; by=defaultdict(list)
    for segment in segments:
        pid=str(segment.metadata.get("progression_id")); doc=RetrievalDocument(id=f"index:{segment.id}",text=_compact_index_text(segment),evidence_ids=frozenset(),timestamp=segment.timestamp,metadata={"segment_id":segment.id,"progression_id":pid,"source_documents":_sources_for_segment(segment,raw_documents)}); indexes.append(doc); by[pid].append(doc)
    return tuple(indexes),dict(by),stats

def _graph_for_segments(by,segments):
    graph=ProgressionGraph({pid:Progression(id=pid,title=pid,scope="") for pid in by})
    for segment in segments: graph.add_segment(segment.id,str(segment.metadata["progression_id"]))
    if not graph.validate(): raise ValueError("constructed progression hierarchy is not acyclic")
    return graph

def _parent_index_text(children):
    seen=set(); terms=[]
    for child in children:
        for token in tokenize(child.text):
            if token not in seen: seen.add(token); terms.append(token)
    return " ".join(terms)

def _parent_index_text_v2(children,max_terms=V2_PARENT_INDEX_TERMS):
    counts=Counter(); first={}; offset=0
    for child in children:
        for token in set(tokenize(child.text)): counts[token]+=1; first.setdefault(token,offset); offset+=1
    ranked=sorted(counts,key=lambda t:(-counts[t],first[t],t)); core=ranked[:min(V2_PARENT_CORE_TERMS,max_terms)]; rest=[t for t in ranked if t not in core]
    return " ".join((core+rest)[:max_terms])

def _build_parent_documents(by,*,compact=False,parent_index_terms=V2_PARENT_INDEX_TERMS):
    return tuple(RetrievalDocument(id=f"parent:{pid}",text=_parent_index_text_v2(children,parent_index_terms) if compact else _parent_index_text(children),evidence_ids=frozenset(),metadata={"progression_id":pid,"child_segment_ids":tuple(c.id for c in children)}) for pid,children in sorted(by.items()))

def _build_hierarchy_v2(raw_documents:tuple[RetrievalDocument,...],*,parent_index_terms:int=V2_PARENT_INDEX_TERMS):
    segments,by,stats=_segment_indexes(raw_documents); graph=_graph_for_segments(by,segments); parents=_build_parent_documents(by,compact=True,parent_index_terms=parent_index_terms)
    return parents,segments,graph,stats

def _rrf_rank(query,segments,parents,*,parent_weight=DEFAULT_PARENT_RRF_WEIGHT,rrf_k=DEFAULT_RRF_K):
    if parent_weight<0: raise ValueError("parent weight must be non-negative")
    if rrf_k<=0: raise ValueError("rrf_k must be positive")
    seg=BM25Index(segments).score(query); par=BM25Index(parents).score(query) if parents else []; pos={str(d.metadata["progression_id"]):i+1 for i,(_,d) in enumerate(par)}; scored=[]
    for i,(_,doc) in enumerate(seg):
        pid=str(doc.metadata["progression_id"]); score=1/(rrf_k+i+1)+parent_weight/(rrf_k+pos.get(pid,len(par)+1)); scored.append((score,doc))
    return [d for _,d in sorted(scored,key=lambda x:(-x[0],x[1].id))]

def _hierarchical_segment_ranking_v2(query,parent_indexes,segment_indexes,*,candidate_limit,parent_weight,rrf_k,diversity_fraction=V2_DIVERSITY_FRACTION,max_diversity_parents=V2_MAX_DIVERSITY_PARENTS):
    if candidate_limit<=0 or not segment_indexes:return [],0,0
    if not 0<=diversity_fraction<=1: raise ValueError("diversity_fraction must be between 0 and 1")
    if max_diversity_parents<1: raise ValueError("max_diversity_parents must be positive")
    ranked=_rrf_rank(query,segment_indexes,parent_indexes,parent_weight=parent_weight,rrf_k=rrf_k); parent_scores=BM25Index(parent_indexes).score(query) if parent_indexes else []; positive=[str(d.metadata["progression_id"]) for s,d in parent_scores if s>0]; seed=positive[:min(len(positive),max_diversity_parents,max(1,math.ceil(math.sqrt(candidate_limit))))]
    by=defaultdict(list)
    for d in ranked: by[str(d.metadata["progression_id"])].append(d)
    budget=min(candidate_limit,max(len(seed),math.ceil(diversity_fraction*candidate_limit))); selected=[]; ids=set(); cursor={p:0 for p in seed}
    while len(selected)<budget and seed:
        added=False
        for p in seed:
            if cursor[p]<len(by[p]):
                d=by[p][cursor[p]]; cursor[p]+=1
                if d.id not in ids:selected.append(d);ids.add(d.id);added=True
                if len(selected)>=budget:break
        if not added:break
    for d in ranked:
        if len(selected)>=candidate_limit:break
        if d.id not in ids:selected.append(d);ids.add(d.id)
    return selected,len(seed),len({str(d.metadata["progression_id"]) for d in selected})

def _build_dag_queries(queries,*,v2,parent_rrf_weight=DEFAULT_PARENT_RRF_WEIGHT,rrf_k=DEFAULT_RRF_K):
    if parent_rrf_weight<0: raise ValueError("parent weight must be non-negative")
    if rrf_k<=0: raise ValueError("rrf_k must be positive")
    cache={}; transformed=[]; selected_counts=[]; source_counts=[]; coverages=[]; seeded=[]; unique=[]
    for query in queries:
        key=id(query.documents)
        if key not in cache:
            segments,by,stats=_segment_indexes(query.documents); graph=_graph_for_segments(by,segments); parents=_build_parent_documents(by,compact=v2); cache[key]=(segments,by,stats,graph,parents,query.documents)
        segments,_,_,_,parents,_=cache[key]; limit=_adaptive_candidate_limit(len(segments))
        if v2:selected,nseed,nunique=_hierarchical_segment_ranking_v2(query.text,parents,segments,candidate_limit=limit,parent_weight=parent_rrf_weight,rrf_k=rrf_k);seeded.append(nseed);unique.append(nunique)
        else:selected=_rrf_rank(query.text,segments,parents,parent_weight=parent_rrf_weight,rrf_k=rrf_k)[:limit]
        sources=OrderedDict(); evidence=set()
        for segment in selected:
            for source in segment.metadata.get("source_documents",()):
                if isinstance(source,RetrievalDocument):sources.setdefault(source.id,source);evidence.update(source.evidence_ids or {source.id})
        candidates=tuple(sources.values()) or query.documents; final=tuple(d for _,d in BM25Index(candidates).score(query.text)); transformed.append(RetrievalQuery(id=query.id,text=query.text,answer=query.answer,evidence_ids=query.evidence_ids,documents=final,category=query.category));selected_counts.append(len(selected));source_counts.append(len(candidates))
        if query.evidence_ids:coverages.append(len(evidence&query.evidence_ids)/len(query.evidence_ids))
    rows=list(cache.values()); graph_nodes=[len(r[3].nodes) for r in rows]; graph_edges=[len(r[3].edges) for r in rows]; units=[len(r[0]) for r in rows]
    stats={"unique_corpora":len(cache),"dag_version":2 if v2 else 1,"avg_stage1_selected_units":sum(selected_counts)/max(1,len(selected_counts)),"avg_candidate_source_documents":sum(source_counts)/max(1,len(source_counts)),"avg_stage1_evidence_coverage":sum(coverages)/max(1,len(coverages)),"avg_progressions":sum(len(r[1]) for r in rows)/max(1,len(rows)),"avg_structural_units":sum(units)/max(1,len(units)),"avg_graph_nodes":sum(graph_nodes)/max(1,len(graph_nodes)),"avg_graph_edges":sum(graph_edges)/max(1,len(graph_edges))}
    if v2:stats.update({"parent_index_max_terms":V2_PARENT_INDEX_TERMS,"avg_seeded_parent_count":sum(seeded)/max(1,len(seeded)),"avg_stage1_unique_progressions":sum(unique)/max(1,len(unique))})
    return transformed,stats

def build_dag_two_stage_queries(queries,*,parent_rrf_weight=DEFAULT_PARENT_RRF_WEIGHT,rrf_k=DEFAULT_RRF_K): return _build_dag_queries(queries,v2=False,parent_rrf_weight=parent_rrf_weight,rrf_k=rrf_k)
def build_dag_two_stage_v2_queries(queries,*,parent_rrf_weight=DEFAULT_PARENT_RRF_WEIGHT,rrf_k=DEFAULT_RRF_K): return _build_dag_queries(queries,v2=True,parent_rrf_weight=parent_rrf_weight,rrf_k=rrf_k)
