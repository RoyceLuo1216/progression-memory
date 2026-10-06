"""A future turn must never be indexed in its own continuation query."""

from progression_memory.benchmarks.msc_progression_pair import cases, evaluate, evaluate_budget_sweep


class ExactEncoder:
    def encode(self, texts):
        import numpy as np
        return np.array([[float('school' in t.lower()), float('career' in t.lower())]
                         for t in texts])


def test_heldout_continuation_uses_only_earlier_sources():
    cid = 'train:example'
    source = {'conversations': [{'initial_data_id': cid, 'turns': [
        {'session': 2, 'turn': 0, 'source_id': 'a', 'text': 'Started school'},
        {'session': 2, 'turn': 1, 'source_id': 'b', 'text': 'A separate career'},
        {'session': 2, 'turn': 2, 'source_id': 'c', 'text': 'School continues'},
    ]}]}
    trace = {'traces': [{'conversation_id': cid, 'split': 'holdout_review', 'rows': [
        {'ref': 's2:t0', 'subject': 0, 'action': 'create_root', 'identity': 'School',
         'scope': 'School', 'state': 'Started school', 'evidence': 'Started school'},
        {'ref': 's2:t1', 'subject': 1, 'action': 'create_root', 'identity': 'Career',
         'scope': 'Career', 'state': 'Career', 'evidence': 'Career'},
        {'ref': 's2:t2', 'subject': 0, 'action': 'attach', 'targets': [0],
         'state': 'School continues', 'evidence': 'School continues'},
    ]}]}
    sample = cases(trace, source)
    assert len(sample) == 1
    assert sample[0]['evidence'] == {'a'}
    assert set(sample[0]['memory'].events) == {'a', 'b'}
    result = evaluate(sample, ExactEncoder(), depths=(1,), context_chars=1000)
    assert result['aggregates']['1']['append']['recall'] == 1
    assert result['aggregates']['1']['progression']['recall'] == 1
    sweep = evaluate_budget_sweep(sample, ExactEncoder(), (5, 1000))
    assert all(sweep['results']['5']['aggregates'][name]['recall'] == 0
               for name in ('append', 'structure', 'progression'))
    assert all(sweep['results']['1000']['aggregates'][name]['recall'] == 1
               for name in ('append', 'structure', 'progression'))
