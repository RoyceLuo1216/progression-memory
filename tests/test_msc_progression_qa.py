"""Source-backed QA replay and identical store coverage for paired arms."""

import json
from pathlib import Path

from progression_memory.benchmarks.msc_progression_qa import prepare, run


class Encoder:
    def encode(self, texts):
        import numpy as np
        return np.array([[float('dog' in text.casefold()),
                          float('cat' in text.casefold())] for text in texts])


def test_paired_qa_uses_original_evidence_and_same_reader():
    cid = 'train:test'
    audit = {'conversations': [{'initial_data_id': cid, 'turns': [
        {'session': 2, 'turn': 0, 'source_id': 'a', 'text': 'I adopted a dog.'},
        {'session': 2, 'turn': 1, 'source_id': 'b', 'text': 'The dog is named Max.'},
    ]}]}
    traces = {'traces': [{'conversation_id': cid, 'rows': [
        {'ref': 's2:t0', 'subject': 0, 'action': 'create_root', 'identity': 'Dog',
         'scope': 'Dog ownership', 'state': 'Adopted a dog', 'evidence': 'Adopted a dog'},
        {'ref': 's2:t1', 'subject': 0, 'action': 'attach', 'targets': [0],
         'state': 'Dog Max', 'evidence': 'Named Max'},
    ]}]}
    qa = {'questions': [{'conversation_id': cid, 'question': 'What is the dog named?',
                        'answer': 'Max', 'evidence': ['s2:t1'], 'category': 'fact'}]}
    queries, memories = prepare(audit, traces, qa)
    assert queries[0].evidence_ids == {'b'}
    assert set(next(iter(memories.values())).events) == {'a', 'b'}
    result = run(queries, memories, Encoder(), (200,))
    assert set(result['results']['200']) == {'append', 'structure', 'progression'}
    assert all(item['reader'] == 'multi_evidence_extractive'
               for item in result['results']['200'].values())


def test_checked_in_questions_are_distinct_and_sourced():
    path = Path('evaluation/data/progression_routing/msc_qa_sanity_v1.json')
    questions = json.loads(path.read_text())['questions']
    assert len(questions) == 10
    assert len({(q['conversation_id'], q['question']) for q in questions}) == len(questions)
    assert all(q['answer'] and q['evidence'] for q in questions)
