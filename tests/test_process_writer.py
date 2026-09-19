import json
from collections import deque

import numpy as np
import pytest

from progression_memory.evaluation.process_construction import build_process_documents
from progression_memory.evaluation.retrieval import RetrievalDocument
from progression_memory.maintenance.process_writer import ProcessWriter
from progression_memory.benchmarks.process_writer_benchmark import run


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
    return dict(action=action, progression_id=pid, subject=subject, process=process,
                fact=f'{subject}: {state}', state=state, delta=state)


def response(*updates): return json.dumps({'updates': list(updates)})


def test_process_identity_state_and_no_update_provenance():
    model = Model([response(), response(update()),
                   response(update(subject='Ben')),
                   response(update('transition', 'process:0', state='Offer accepted'))])
    writer = ProcessWriter(model)
    docs = tuple(RetrievalDocument(id=str(i), text=text) for i, text in enumerate(
        ['Ana: Hi!', 'Ana applied.', 'Ben applied.', 'Ana accepted the offer.']))
    units, processes, stats = build_process_documents(docs, writer, Embedder())
    assert len(processes) == 2  # identical proposal embeddings cannot force an attachment
    assert [h['source_id'] for h in processes['process:0'].history] == ['1', '3']
    assert processes['process:0'].history[-1]['prior_state'] == 'Applied'
    assert processes['process:0'].state == 'Offer accepted'
    assert {d.id for u in units for d in u.metadata['source_documents']} == {'0','1','2','3'}
    assert all(not u.evidence_ids for u in units)  # index summaries earn no evidence credit
    assert stats['unassigned_source_count'] == 1
    assert writer.stats()['model_calls'] == 4
    assert 'Ana accepted the offer.' not in model.prompts[0]  # no future evidence


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
    writer = ProcessWriter(Model([response()]*12))
    with pytest.raises(RuntimeError, match='no process updates'):
        run(dataset='fixture', output=str(tmp_path/'result.json'), writer=writer, embedder=Embedder())
    assert writer.stats()['model_calls'] == 12


def test_retrieval_entrypoint_routes_to_real_writer(monkeypatch, tmp_path):
    from progression_memory.benchmarks import real_retrieval, process_writer_benchmark
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


def test_fenced_json_is_visible_not_fallback():
    writer = ProcessWriter(Model(['```json\n{"updates":[]}\n```']))
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
    writer = ProcessWriter(ChatModel())
    assert writer.decide({'id':'a','text':'Ana: hello'}, [], []) == []
    assert writer.stats()['model_calls'] == 1
