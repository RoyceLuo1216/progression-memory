from __future__ import annotations

import json
import re
from pathlib import Path

from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def _normalize_locomo_evidence(values: list[str]) -> frozenset[str]:
    ids: set[str] = set()
    for value in values:
        ids.update(part.strip() for part in value.split(";") if part.strip())
    return frozenset(ids)


def load_locomo(path: str | Path) -> list[RetrievalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries: list[RetrievalQuery] = []
    for sample in data:
        conversation = sample["conversation"]
        sample_id = str(sample["sample_id"])
        docs: list[RetrievalDocument] = []
        for key in sorted(
            (k for k in conversation if _SESSION_KEY.match(k)),
            key=lambda k: int(_SESSION_KEY.match(k).group(1)),
        ):
            session_num = int(_SESSION_KEY.match(key).group(1))
            timestamp = conversation.get(f"session_{session_num}_date_time")
            for turn in conversation[key]:
                dia_id = str(turn["dia_id"])
                text = str(turn.get("text", ""))
                if turn.get("blip_caption"):
                    text += " " + str(turn["blip_caption"])
                speaker = str(turn.get("speaker", ""))
                docs.append(
                    RetrievalDocument(
                        id=dia_id,
                        text=f"{speaker}: {text}",
                        evidence_ids=frozenset({dia_id}),
                        timestamp=str(timestamp) if timestamp is not None else None,
                        metadata={"sample_id": sample_id, "session": session_num},
                    )
                )
        frozen_docs = tuple(docs)
        for idx, qa in enumerate(sample["qa"]):
            queries.append(
                RetrievalQuery(
                    id=f"{sample_id}:q{idx}",
                    text=str(qa["question"]),
                    answer=str(qa.get("answer", "")),
                    evidence_ids=_normalize_locomo_evidence(list(qa.get("evidence", []))),
                    documents=frozen_docs,
                    category=str(qa.get("category", "unknown")),
                )
            )
    return queries


def load_longmemeval(path: str | Path) -> list[RetrievalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries: list[RetrievalQuery] = []
    for item in data:
        session_ids = item["haystack_session_ids"]
        dates = item["haystack_dates"]
        sessions = item["haystack_sessions"]
        docs: list[RetrievalDocument] = []
        for session_id, date, turns in zip(session_ids, dates, sessions, strict=True):
            rendered_turns: list[str] = []
            for turn in turns:
                rendered_turns.append(f"{turn.get('role', '')}: {turn.get('content', '')}")
            sid = str(session_id)
            docs.append(
                RetrievalDocument(
                    id=sid,
                    text="\n".join(rendered_turns),
                    evidence_ids=frozenset({sid}),
                    timestamp=str(date),
                    metadata={"question_id": str(item["question_id"])},
                )
            )
        question_id = str(item["question_id"])
        evidence_ids = (
            frozenset()
            if question_id.endswith("_abs")
            else frozenset(str(x) for x in item.get("answer_session_ids", []))
        )
        queries.append(
            RetrievalQuery(
                id=question_id,
                text=str(item["question"]),
                answer=str(item.get("answer", "")),
                evidence_ids=evidence_ids,
                documents=tuple(docs),
                category=str(item.get("question_type", "unknown")),
            )
        )
    return queries
