import json
import sys
import types
from collections import deque

import numpy as np
import pytest

from progression_memory.benchmarks.process_writer_benchmark import run
from progression_memory.evaluation.process_construction import build_process_documents
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.maintenance import local_model
from progression_memory.maintenance.local_model import TransformersLanguageModel
from progression_memory.maintenance.process_writer import (
    ProcessWriter,
    explicit_first_person_subject,
    obvious_lifecycle_transition,
    obvious_non_update,
)


class Model:
    model_name = 'test-double'
    def __init__(self, replies):
        self.replies = deque(replies)
        self.prompts = []
    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.popleft()


class Embedder:
    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts])


def update(action='create', pid=None, subject='Ana', process='Acme application', state='Applied'):
    return {'action': action, 'progression_id': pid, 'subject': subject, 'process': process,
            'fact': f'{subject}: {state}', 'state': state, 'delta': state}


def response(*updates): return json.dumps({'updates': list(updates)})


def test_process_identity_state_and_no_update_provenance():
    model = Model([response(update()),
                   response(update(subject='Ben')),
                   response(update('transition', 'process:0', state='Offer accepted'))])
    writer = ProcessWriter(model)
    docs = tuple(RetrievalDocument(id=str(i), text=text) for i, text in enumerate(
        ['Ana: Hi!', 'Ana: I applied.', 'Ben: I applied.', 'Ana: I accepted the offer.']))
    units, processes, stats = build_process_documents(docs, writer, Embedder())
    assert len(processes) == 2  # identical proposal embeddings cannot force an attachment
    assert [h.episode_id for h in processes['process:0'].state_history] == ['1', '3']
    assert processes['process:0'].state_history[-1].prior_state == 'Applied'
    assert processes['process:0'].current_state == 'Offer accepted'
    assert len(processes['process:0'].segment_ids) == 2
    assert {d.id for u in units for d in u.metadata['source_documents']} == {'0','1','2','3'}
    assert all(not u.evidence_ids for u in units)  # index summaries earn no evidence credit
    assert stats['unassigned_source_count'] == 1
    assert writer.stats()['model_calls'] == 3
    assert writer.stats()['gate_skips'] == 1
    assert 'Ana accepted the offer.' not in model.prompts[0]  # no future evidence
    ben_prompt = json.loads(model.prompts[1])[-1]['content']
    assert 'CANDIDATES (existing processes):\n[]' in ben_prompt


@pytest.mark.parametrize('raw', ['not JSON', response(update('extend', 'unknown')),
    response(update('extend', 'p', subject='Ben')), response(update(pid='p'))])
def test_invalid_model_output_fails_without_fallback(tmp_path, raw):
    writer = ProcessWriter(Model([raw]), trace_path=str(tmp_path/'trace.jsonl'))
    with pytest.raises(ValueError):
        writer.decide({'id':'x','text':'Ana applied'}, [],
                      [{'id':'p','subject':'Ana','process':'Acme application'}])
    assert writer.stats()['invalid_responses'] == 1
    assert writer.stats()['valid_responses'] == writer.stats()['fallback_calls'] == 0
    assert json.loads((tmp_path/'trace.jsonl').read_text())['raw_response'] == raw


def test_multiple_processes_in_one_turn():
    writer = ProcessWriter(Model([response(update(), update(process='Pottery practice'))]))
    doc = RetrievalDocument(id='a', text='Ana applied and started pottery.')
    units, processes, stats = build_process_documents((doc,), writer, Embedder())
    assert len(processes) == 2
    assert stats['assigned_source_count'] == 1
    assert stats['source_membership_count'] == 2
    assert all(u.metadata['source_documents'] == (doc,) for u in units)


def test_benchmark_uses_model_and_saves_failure(tmp_path):
    writer = ProcessWriter(Model(['broken']))
    path = tmp_path/'result.json'
    with pytest.raises(ValueError):
        run(dataset='fixture', output=str(path), writer=writer, embedder=Embedder())
    result = json.loads(path.read_text())
    assert result['status'] == 'failed'
    assert result['writer']['model_calls'] == 1
    assert result['writer']['fallback_calls'] == 0


def test_benchmark_no_updates_is_not_success(tmp_path):
    from progression_memory.benchmarks.process_writer_benchmark import CASES
    writer = ProcessWriter(Model([response()] * len(CASES)))
    with pytest.raises(RuntimeError, match='no process updates'):
        run(dataset='fixture', output=str(tmp_path/'result.json'), writer=writer, embedder=Embedder())
    assert writer.stats()['model_calls'] == len(CASES) - writer.stats()['gate_skips']
    assert writer.stats()['gate_skips'] == 5


def test_retrieval_entrypoint_routes_to_real_writer(monkeypatch, tmp_path):
    from progression_memory.benchmarks import process_writer_benchmark, real_retrieval
    called = {}
    def fake_run(**kwargs):
        called.update(kwargs)
        return {'status': 'completed'}
    monkeypatch.setattr(process_writer_benchmark, 'run', fake_run)
    assert real_retrieval.run('locomo', 'data.json', [8], 'process_slm',
        output=str(tmp_path/'result.json'), max_turns=32) == {'status': 'completed'}
    assert called['max_turns'] == 32
    assert called['cutoffs'] == [8]
    assert called['model_name'] == 'Qwen/Qwen2.5-1.5B-Instruct'


def test_locomo_scope_filters_whole_evidence_sets(tmp_path):
    from progression_memory.benchmarks.process_writer_benchmark import run
    data = [{'sample_id': 'sample', 'conversation': {'session_1': [
        {'dia_id': 'D1:1', 'speaker': 'Ana', 'text': 'I applied to Acme.'},
        {'dia_id': 'D1:2', 'speaker': 'Ana', 'text': 'I received an offer.'}],
        'session_1_date_time': '2026-01-01'}, 'qa': [
        {'question': 'Where did Ana apply?', 'evidence': ['D1:1'], 'category': 1},
        {'question': 'What happened next?', 'evidence': ['D1:1','D1:2'], 'category': 1}]}]
    source = tmp_path/'data.json'
    source.write_text(json.dumps(data))
    result = run(dataset='locomo', data_path=str(source), max_turns=1,
                 output=str(tmp_path/'result.json'), writer=ProcessWriter(Model([response(update())])),
                 embedder=Embedder())
    assert result['corpora'][0]['eligible_queries'] == 1
    assert result['corpora'][0]['excluded_queries'] == 1
    assert result['retrieval']['eligible_query_count'] == result['append_baseline']['eligible_query_count'] == 1
    assert result['retrieval']['cutoffs']['8']['avg_retrieved_chars'] == len('Ana: I applied to Acme.')
    assert 'embedding' not in result['corpora'][0]['processes'][0]
    assert result['corpora'][0]['diagnostics']['transition_segment_mismatch_count'] == 0


def test_fenced_json_is_visible_not_fallback():
    writer = ProcessWriter(Model(['```json\n{"updates":[]}\n```']), selective=False)
    assert writer.decide({'id':'greeting','text':'Hi'}, [], []) == []
    assert writer.stats()['fenced_responses'] == 1
    assert writer.stats()['model_calls'] == writer.stats()['valid_responses'] == 1
    assert writer.stats()['fallback_calls'] == 0


def test_slm_cannot_silently_enter_heuristic_transform():
    from progression_memory.benchmarks.real_retrieval import _transform_queries
    with pytest.raises(ValueError, match='explicit model tracing'):
        _transform_queries([], 'process_slm')


def test_chat_backend_receives_separate_instructions_and_evidence():
    class ChatModel:
        def generate_messages(self, messages):
            assert messages[0]['role'] == 'system'
            assert messages[-1]['role'] == 'user'
            assert 'new message to classify:\nAna: hello' in messages[-1]['content']
            assert any(m['role'] == 'assistant' for m in messages)
            return response()
    writer = ProcessWriter(ChatModel(), selective=False)
    assert writer.decide({'id':'a','text':'Ana: hello'}, [], []) == []
    assert writer.stats()['model_calls'] == 1


def test_constrained_backend_receives_response_schema():
    class ConstrainedModel:
        def generate_messages(self, messages, *, json_schema=None):
            assert json_schema['properties']['updates']['type'] == 'array'
            return response()
    writer = ProcessWriter(ConstrainedModel(), selective=False)
    assert writer.decide({'id':'a','text':'Ana: hello'}, [], []) == []
    assert writer.stats()['constrained_calls'] == 1


def test_selective_gate_only_skips_obvious_non_updates():
    assert obvious_non_update('Ana: Hi Ben, good to see you!')
    assert obvious_non_update("Ben: That's great, congratulations!")
    assert obvious_non_update('Ben: Did you accept their offer?')
    assert not obvious_non_update('Ana: I accepted their offer this morning.')
    assert not obvious_non_update('Ana: My interview is tomorrow? I just received the schedule.')


def test_first_person_subject_is_explicit_and_enforced():
    assert explicit_first_person_subject('Ben: I applied to Acme.') == 'Ben'
    assert explicit_first_person_subject('Ana: Ben applied to Acme.') is None
    writer = ProcessWriter(Model([response(update(subject='Ana'))]), selective=False)
    with pytest.raises(ValueError, match="belongs to speaker 'Ben'"):
        writer.decide({'id': 'b', 'text': 'Ben: I applied to Acme.'}, [], [])


def test_obvious_lifecycle_change_upgrades_extension_to_transition():
    assert obvious_lifecycle_transition('Ana was invited to an interview')
    writer = ProcessWriter(Model([response(update(
        action='extend', pid='process:0', state='Interview invitation received'
    ))]), selective=False)
    updates = writer.decide(
        {'id': 'i', 'text': 'Ana: Acme invited me to an interview.'},
        [],
        [{'id': 'process:0', 'subject': 'Ana', 'process': 'Acme application'}],
    )
    assert updates[0].action == 'transition'
    assert writer.stats()['transition_overrides'] == 1


def test_reassigns_clear_same_subject_candidate_when_model_picks_weaker_match():
    writer = ProcessWriter(Model([response(update(
        action='transition', pid='process:1', process='Pottery', state='Offer accepted'
    ))]), selective=False)
    updates = writer.decide(
        {'id': 'offer', 'text': "Ana: Acme offered me the job."},
        [],
        [
            {'id': 'process:0', 'subject': 'Ana', 'process': 'Acme application', 'score': 0.72},
            {'id': 'process:1', 'subject': 'Ana', 'process': 'Pottery', 'score': 0.41},
        ],
    )
    assert updates[0].progression_id == 'process:0'
    assert updates[0].process == 'Acme application'
    assert writer.stats()['candidate_reassignments'] == 1


def test_distilled_prompt_contract_and_quanto_loading(monkeypatch):
    messages = [
        {'role': 'system', 'content': 'instructions'},
        {'role': 'user', 'content': 'evidence'},
    ]
    prepared = local_model._prepare_messages(
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', messages
    )
    assert [message['role'] for message in prepared] == ['user']
    assert prepared[0]['content'] == 'instructions\n\nevidence'

    observed = {}
    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name: f'tokenizer:{name}'
    )
    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            observed.update(model=name, **kwargs)
            return types.SimpleNamespace(eval=lambda: None)
    transformers.AutoModelForCausalLM = AutoModel
    transformers.QuantoConfig = lambda *, weights: {'weights': weights}
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    local_model._load.cache_clear()
    tokenizer, _ = local_model._load('distilled-test', 'quanto-int8')
    local_model._load.cache_clear()
    assert tokenizer == 'tokenizer:distilled-test'
    assert observed['quantization_config'] == {'weights': 'int8'}


def test_transformers_backend_applies_json_constraint(monkeypatch):
    class Inputs(dict):
        def to(self, _device): return self
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs): return 'prompt'
        def __call__(self, text, return_tensors):
            return Inputs(input_ids=np.array([[1, 2]]))
        def decode(self, tokens, skip_special_tokens): return '{"updates":[]}'
    class ModelBackend:
        device = 'cpu'
        def __init__(self): self.kwargs = {}
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return np.array([[1, 2, 3]])
    backend = ModelBackend()
    monkeypatch.setattr(
        TransformersLanguageModel, 'runtime', property(lambda self: (Tokenizer(), backend))
    )
    root = types.ModuleType('lmformatenforcer')
    root.JsonSchemaParser = lambda schema: schema
