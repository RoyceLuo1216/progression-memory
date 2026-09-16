from datetime import datetime

import numpy as np

from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.evaluation.semantic import HybridIndex, SemanticIndex
from progression_memory.maintenance.state import MaintenanceAction, condition_progression_state, maintenance_decision
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression


class FakeEmbedder:
    dimension = 2

    def encode(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            if "career" in lower or "job" in lower or "internship" in lower:
                rows.append([1.0, 0.0])
            elif "health" in lower or "run" in lower:
                rows.append([0.0, 1.0])
            else:
                rows.append([0.5, 0.5])
        values = np.asarray(rows, dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def test_semantic_index_recovers_paraphrase():
    docs = (
        RetrievalDocument(id="a", text="career internship", evidence_ids=frozenset()),
        RetrievalDocument(id="b", text="health running", evidence_ids=frozenset()),
    )
    assert SemanticIndex(docs, FakeEmbedder()).score("job")[0][1].id == "a"
    assert HybridIndex(docs, FakeEmbedder(), semantic_weight=1.0).score("job")[0][1].id == "a"


def test_conditioned_state_preserves_prior_and_delta():
    progression = Progression(id="p", title="career", scope="", embedding=[1.0, 0.0], current_state="applied")
    episode = Episode(id="e", summary="received internship", dialogue_time=datetime(2026, 1, 1), embedding=[0.8, 0.2], state_vector=[0.8, 0.2])
    checkpoint = condition_progression_state(progression, episode)
    assert checkpoint.prior_state == "applied"
    assert checkpoint.posterior_state.endswith("received internship")
    assert progression.state_history[-1].episode_id == "e"
    assert progression.state_embedding is not None


def test_maintenance_distinguishes_extension_from_new_progression():
    progression = Progression(id="p", title="career", scope="", embedding=[1.0, 0.0], state_embedding=[1.0, 0.0], continuity=1.0)
    related = Episode(id="e1", summary="job", dialogue_time=datetime(2026, 1, 1), embedding=[1.0, 0.0], state_vector=[1.0, 0.0])
    unrelated = Episode(id="e2", summary="run", dialogue_time=datetime(2026, 1, 2), embedding=[0.0, 1.0], state_vector=[0.0, 1.0])
    assert maintenance_decision(related, progression).action == MaintenanceAction.EXTEND
    assert maintenance_decision(unrelated, progression).action == MaintenanceAction.CREATE
