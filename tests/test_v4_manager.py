from datetime import datetime, timedelta

from progression_memory.maintenance.consolidation import run_maintenance
from progression_memory.maintenance.manager import SLMProgressionManager
from progression_memory.models.episode import Episode
from progression_memory.models.progression import Progression, ProgressionStateCheckpoint
from progression_memory.storage.progression_store import ProgressionStore


class FakeLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def test_slm_writer_returns_bounded_transition_and_delta():
    model = FakeLM('{"action":"transition","progression_id":"p","updated_state":"Offer received.",'
                   '"state_delta":"Interviewing -> offer","confidence":0.9}')
    manager = SLMProgressionManager(model, candidate_k=1)
    progression = Progression(id="p", title="Google recruiting", scope="", embedding=[1.0, 0.0],
                              state_embedding=[1.0, 0.0], current_state="Interviewing.")
    episode = Episode(id="e", summary="Google sent an offer.", dialogue_time=datetime(2026, 1, 2),
                      embedding=[1.0, 0.0], state_vector=[1.0, 0.0])
    decision = manager.decide(episode, [progression])
    assert decision.action.value == "transition"
    assert decision.updated_state == "Offer received."
    assert decision.state_delta == "Interviewing -> offer"
    assert model.calls == 1


def test_trigger_first_maintenance_creates_evidence_backed_child():
    store = ProgressionStore()
    progression = Progression(id="p", title="career", scope="", embedding=[1.0, 0.0],
                              state_embedding=[1.0, 0.0])
    store.upsert_progression(progression)
    start = datetime(2026, 1, 1)
    ids = []
    for i in range(8):
        episode_id = f"e{i}"
        ids.append(episode_id)
        episode = Episode(id=episode_id, summary=f"evidence {i}", dialogue_time=start + timedelta(days=i),
                          embedding=[1.0, 0.0], state_vector=[1.0, 0.0])
        store.ingest_episode(episode)
        store.attach_episode(episode_id, "p")
        progression.state_history.append(ProgressionStateCheckpoint(
            episode_id=episode_id, timestamp=episode.dialogue_time, evidence=episode.summary,
            posterior_state=episode.summary, state_delta=episode.summary, delta_embedding=[1.0, 0.0],
        ))
    model = FakeLM('{"action":"create_child","ids":["p"],"child_id":"p:child",'
                   '"child_identity":"interviews","episode_ids":' + str(ids).replace("'", '"') + '}')
    stats = run_maintenance(store, model, max_actions=1)
    assert stats == {"triggered": 1, "inspected": 1, "applied": 1, "invalid": 0}
    assert store.progressions["p:child"].episode_ids == ids
    assert model.calls == 1
