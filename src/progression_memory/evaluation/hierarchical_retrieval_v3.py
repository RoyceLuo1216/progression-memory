from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from progression_memory.evaluation.dag_retrieval import _adaptive_candidate_limit, _segment_indexes
from progression_memory.evaluation.hierarchical_retrieval import DEFAULT_BEAM_WIDTH, DEFAULT_BRANCH_FACTOR, DEFAULT_MAX_PILLARS, _query_is_temporal
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.evaluation.semantic import HybridIndex, SentenceTransformerEmbedder, TextEmbedder, cosine_scores

DEFAULT_GLOBAL_RESCUE_FRACTION = 0.35
DEFAULT_BRANCH_BUDGET_FRACTION = 0.60
DEFAULT_TEMPORAL_BUDGET_FRACTION = 0.25

@dataclass(frozen=True)
class SemanticHierarchyNode:
    id: str; level: int; text: str; child_ids: tuple[str, ...]; leaf_progression_ids: tuple[str, ...]; embedding: tuple[float, ...]

def _mean_unit(vectors: list[np.ndarray]) -> np.ndarray:
    value=np.mean(np.stack(vectors),axis=0); norm=float(np.linalg.norm(value)); return value/norm if norm else value

def _leaf_nodes(by_progression: dict[str,list[RetrievalDocument]], embedder: TextEmbedder) -> dict[str,SemanticHierarchyNode]:
    pids=sorted(by_progression); texts=["\n".join(doc.text for doc in by_progression[pid]) for pid in pids]; embeddings=embedder.encode(texts)
    return {pid:SemanticHierarchyNode(pid,0,text,tuple(doc.id for doc in by_progression[pid]),(pid,),tuple(float(x) for x in emb)) for pid,text,emb in zip(pids,texts,embeddings,strict=True)}

def _group_level(nodes:list[SemanticHierarchyNode],level:int,branch_factor:int)->list[SemanticHierarchyNode]:
    remaining=sorted(nodes,key=lambda n:n.id); groups=[]
    while remaining:
        seed=remaining.pop(0); se=np.asarray(seed.embedding,dtype=np.float32)
        ranked=sorted(remaining,key=lambda n:(-float(cosine_scores(se,np.asarray([n.embedding],dtype=np.float32))[0]),n.id)); companions=ranked[:branch_factor-1]; ids={n.id for n in companions}; remaining=[n for n in remaining if n.id not in ids]; groups.append([seed,*companions])
    parents=[]
    for i,group in enumerate(groups):
        leaves=tuple(sorted({leaf for n in group for leaf in n.leaf_progression_ids})); emb=_mean_unit([np.asarray(n.embedding,dtype=np.float32) for n in group]); parents.append(SemanticHierarchyNode(f"semantic-hierarchy:l{level}:{i}",level,"\n".join(n.text for n in group),tuple(n.id for n in group),leaves,tuple(float(x) for x in emb)))
    return parents

def build_semantic_progression_hierarchy(by_progression:dict[str,list[RetrievalDocument]],embedder:TextEmbedder,*,branch_factor:int=DEFAULT_BRANCH_FACTOR,max_pillars:int=DEFAULT_MAX_PILLARS)->tuple[dict[str,SemanticHierarchyNode],tuple[str,...],int]:
    leaves=_leaf_nodes(by_progression,embedder); all_nodes=dict(leaves); current=list(leaves.values()); level=1
    while len(current)>max_pillars: current=_group_level(current,level,branch_factor); all_nodes.update({n.id:n for n in current}); level+=1
    return all_nodes,tuple(n.id for n in current),max((n.level for n in all_nodes.values()),default=0)

def _beam_traverse_semantic(query:str,nodes:dict[str,SemanticHierarchyNode],roots:tuple[str,...],embedder:TextEmbedder,beam_width:int)->tuple[set[str],int]:
    qe=embedder.encode([query])[0]; frontier=[nodes[r] for r in roots]; visited=0
    while frontier:
        visited+=len(frontier); matrix=np.asarray([n.embedding for n in frontier],dtype=np.float32); scores=cosine_scores(qe,matrix); chosen=[n for _,n in sorted(zip(scores,frontier,strict=True),key=lambda row:(-float(row[0]),row[1].id))[:beam_width]]; children=[nodes[c] for n in chosen for c in n.child_ids if c in nodes]
        if not children:return {leaf for n in chosen for leaf in n.leaf_progression_ids},visited
        frontier=children
    return set(),visited

def build_hierarchical_beam_v3_queries(queries:list[RetrievalQuery],*,semantic_weight:float=0.7,branch_factor:int=DEFAULT_BRANCH_FACTOR,beam_width:int=DEFAULT_BEAM_WIDTH,max_pillars:int=DEFAULT_MAX_PILLARS,embedder:TextEmbedder|None=None)->tuple[list[RetrievalQuery],dict[str,object]]:
    """End-to-end semantic progression construction, hierarchy routing, rescue and reranking."""
    embedder=embedder or SentenceTransformerEmbedder(); cache={}; transformed=[]; visited_counts=[]; rescued_counts=[]; progression_counts=[]; segment_counts=[]; source_counts=[]; coverages=[]
    for query in queries:
        key=id(query.documents)
        if key not in cache:
            # V3 differs from V2 from the WRITE path onward: semantic embeddings form the
            # progressions themselves, not merely the hierarchy/query-time index.
            segments,by_progression,construction_stats=_segment_indexes(query.documents,embedder=embedder); nodes,roots,depth=build_semantic_progression_hierarchy(by_progression,embedder,branch_factor=branch_factor,max_pillars=max_pillars); cache[key]=(segments,by_progression,nodes,roots,depth,construction_stats)
        segments,by_progression,nodes,roots,_,_=cache[key]; routed,visited=_beam_traverse_semantic(query.text,nodes,roots,embedder,beam_width); candidate_limit=_adaptive_candidate_limit(len(segments)); global_rank=HybridIndex(segments,embedder,semantic_weight=semantic_weight).score(query.text); rescue_count=min(len(global_rank),max(6,math.ceil(DEFAULT_GLOBAL_RESCUE_FRACTION*candidate_limit))); rescued={str(doc.metadata.get("progression_id")) for _,doc in global_rank[:rescue_count]}; active=routed|rescued or set(by_progression); branch_pool=tuple(doc for pid in active for doc in by_progression.get(pid,[])); branch_rank=HybridIndex(branch_pool,embedder,semantic_weight=semantic_weight).score(query.text) if branch_pool else []
        selected=[]; selected_ids=set(); temporal_budget=min(candidate_limit,max(1,math.ceil(DEFAULT_TEMPORAL_BUDGET_FRACTION*candidate_limit))) if _query_is_temporal(query.text) else 0
        for _,anchor in branch_rank:
            if len(selected)>=temporal_budget:break
            pid=str(anchor.metadata.get("progression_id")); siblings=sorted(by_progression.get(pid,[]),key=lambda d:(d.timestamp or "",d.id)); idx=next((i for i,d in enumerate(siblings) if d.id==anchor.id),0)
            for neighbor in siblings[max(0,idx-1):min(len(siblings),idx+2)]:
                if neighbor.id not in selected_ids:selected.append(neighbor);selected_ids.add(neighbor.id)
                if len(selected)>=temporal_budget:break
        branch_target=min(candidate_limit,max(len(selected),math.ceil(DEFAULT_BRANCH_BUDGET_FRACTION*candidate_limit)))
        for _,doc in branch_rank:
            if len(selected)>=branch_target:break
            if doc.id not in selected_ids:selected.append(doc);selected_ids.add(doc.id)
        for _,doc in global_rank:
            if len(selected)>=candidate_limit:break
            if doc.id not in selected_ids:selected.append(doc);selected_ids.add(doc.id)
        sources={}; candidate_evidence=set()
        for segment in selected:
            for source in segment.metadata.get("source_documents",()):
                if isinstance(source,RetrievalDocument):sources.setdefault(source.id,source);candidate_evidence.update(source.evidence_ids or {source.id})
        candidate_documents=tuple(sources.values()) or query.documents; final_docs=tuple(doc for _,doc in HybridIndex(candidate_documents,embedder,semantic_weight=semantic_weight).score(query.text)); transformed.append(RetrievalQuery(id=query.id,text=query.text,answer=query.answer,evidence_ids=query.evidence_ids,documents=final_docs,category=query.category)); visited_counts.append(visited);rescued_counts.append(len(rescued));progression_counts.append(len(active));segment_counts.append(len(selected));source_counts.append(len(candidate_documents))
        if query.evidence_ids:coverages.append(len(candidate_evidence&query.evidence_ids)/len(query.evidence_ids))
    corpora=list(cache.values()); construction_rows=[r[5] for r in corpora]
    return transformed,{"unique_corpora":len(cache),"hierarchy":"semantic_soft_progression_hierarchy_v3","embedding_model":getattr(embedder,"model_name",type(embedder).__name__),"semantic_weight":semantic_weight,"semantic_progression_construction":True,"avg_progressions":sum(float(r.get("progression_count",0)) for r in construction_rows)/max(1,len(construction_rows)),"avg_state_updates_per_progression":sum(float(r.get("avg_state_updates_per_progression",0)) for r in construction_rows)/max(1,len(construction_rows)),"avg_hierarchy_depth":sum(r[4] for r in corpora)/max(1,len(corpora)),"avg_hierarchy_nodes":sum(len(r[2]) for r in corpora)/max(1,len(corpora)),"avg_nodes_visited_per_query":sum(visited_counts)/max(1,len(visited_counts)),"avg_selected_progressions":sum(progression_counts)/max(1,len(progression_counts)),"avg_rescued_progressions":sum(rescued_counts)/max(1,len(rescued_counts)),"avg_stage1_selected_units":sum(segment_counts)/max(1,len(segment_counts)),"avg_candidate_source_documents":sum(source_counts)/max(1,len(source_counts)),"avg_stage1_evidence_coverage":sum(coverages)/max(1,len(coverages)),"hard_branch_pruning":False}
