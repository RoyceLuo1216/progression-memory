"""Flat process construction experiment; all source text remains available for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from progression_memory.evaluation.progressions import parse_timestamp
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.evaluation.semantic import TextEmbedder, cosine_scores
from progression_memory.maintenance.process_writer import ProcessWriter


@dataclass
class Process:
    id: str
    subject: str
    identity: str
    state: str
    history: list[dict] = field(default_factory=list)


def build_process_documents(documents: tuple[RetrievalDocument, ...], writer: ProcessWriter,
                            embedder: TextEmbedder, *, candidate_k: int = 6):
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    processes: dict[str, Process] = {}
    units: list[RetrievalDocument] = []
    previous: list[dict] = []
    for ordinal, doc in enumerate(documents):
        candidates = []
        now = parse_timestamp(doc.timestamp, ordinal)
        if processes:
            rows = list(processes.values())
            q = embedder.encode([doc.text])[0]
            identity = cosine_scores(q, embedder.encode([p.subject + ': ' + p.identity for p in rows]))
            recent = cosine_scores(q, embedder.encode([p.history[-1]["fact"] for p in rows]))
            for p, semantic, episode in zip(rows, identity, recent, strict=True):
                age = abs((now - parse_timestamp(p.history[-1]["timestamp"], p.history[-1]["ordinal"])).total_seconds()) / 86400
                # Proposals only: this score never authorizes an attachment.
                score = .60 * float(semantic) + .25 * float(episode) + .15 * math.exp(-math.log(2) * age / 90)
                candidates.append({"id": p.id, "subject": p.subject, "process": p.identity,
                                   "state": p.state, "recent_fact": p.history[-1]["fact"], "score": round(score, 4)})
            candidates = sorted(candidates, key=lambda c: (-c["score"], c["id"]))[:candidate_k]
        source = {"id": doc.id, "text": doc.text, "timestamp": doc.timestamp}
        updates = writer.decide(source, previous[-3:], candidates)
        for u in updates:
            if u.action == "create":
                if any((p.subject.casefold(), p.identity.casefold()) ==
                       (u.subject.casefold(), u.process.casefold()) for p in processes.values()):
                    raise ValueError("writer created duplicate identity outside candidate shortlist")
                pid = f"process:{len(processes)}"
                processes[pid] = Process(pid, u.subject, u.process, u.state)
            else:
                pid = u.progression_id
            p = processes[pid]
            prior = p.state if p.history else ""
            p.state = u.state
            p.history.append({"source_id": doc.id, "timestamp": doc.timestamp, "ordinal": ordinal,
                              "action": u.action, "fact": u.fact, "prior_state": prior,
                              "state": u.state, "delta": u.delta})
            units.append(RetrievalDocument(
                id=f"{pid}:update:{len(p.history)}",
                text=f"{p.subject}: {p.identity}\n{u.fact}\nState: {u.state}\nChange: {u.delta}",
                timestamp=doc.timestamp,
                metadata={"progression_id": pid, "source_documents": (doc,)},
            ))
        if not updates:
            # No process update does not imply no useful evidence (e.g. standalone facts).
            units.append(RetrievalDocument(id=f"unassigned:{doc.id}", text=doc.text,
                                          metadata={"source_documents": (doc,)}))
        previous.append(source)
    sizes = [len({h["source_id"] for h in p.history}) for p in processes.values()]
    assignments = {h["source_id"] for p in processes.values() for h in p.history}
    stats = {"raw_document_count": len(documents), "progression_count": len(processes),
             "assigned_source_count": len(assignments), "unassigned_source_count": len(documents)-len(assignments),
             "largest_progression_fraction": max(sizes, default=0) / max(1, len(documents)),
             "singleton_progression_fraction": sum(n == 1 for n in sizes) / max(1, len(sizes)),
             "source_membership_count": sum(sizes), "structural_units": len(units)}
    return tuple(units), processes, stats
