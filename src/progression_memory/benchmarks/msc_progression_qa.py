"""Diagnostic paired QA on source-backed, supplied MSC progression traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from progression_memory.benchmarks.msc_progression_pair import _decision
from progression_memory.benchmarks.representation_rag_benchmark import PinnedEncoder, evaluate
from progression_memory.evaluation.retrieval import RetrievalDocument, RetrievalQuery
from progression_memory.maintenance.calibrated_progression import Event, Memory


def prepare(audit: dict, traces: dict, qa: dict) -> tuple[list[RetrievalQuery], dict[int, Memory]]:
    source = {c['initial_data_id']: c['turns'] for c in audit['conversations']}
    by_id = {trace['conversation_id']: trace for trace in traces['traces']}
    docs_by_id = {}
    memories = {}
    for cid in {row['conversation_id'] for row in qa['questions']}:
        turns = source[cid]
        trace = by_id[cid]
        lookup = {f"s{t['session']}:t{t['turn']}": t for t in turns}
        if list(lookup) != [r['ref'] for r in trace['rows']]:
            raise ValueError(f'source mismatch: {cid}')
        docs = tuple(RetrievalDocument(t['source_id'], t['text']) for t in turns)
        memory = Memory()
        for row in trace['rows']:
            turn = lookup[row['ref']]
            memory.apply(Event(turn['source_id'], f"{cid}:{row['subject']}",
                               turn['text'], None, turn['source_id']), _decision(row))
        docs_by_id[cid] = (docs, lookup)
        memories[id(docs)] = memory
    queries = []
    for number, row in enumerate(qa['questions']):
        docs, lookup = docs_by_id[row['conversation_id']]
        evidence = frozenset(lookup[ref]['source_id'] for ref in row['evidence'])
        if not evidence <= {doc.id for doc in docs} or not row['answer']:
            raise ValueError('invalid QA label')
        if not any(row['answer'].casefold() in lookup[ref]['text'].casefold()
                   for ref in row['evidence']):
            raise ValueError(f'answer is not in cited source: {number}')
        queries.append(RetrievalQuery(f"msc-qa:{number}", row['question'], evidence,
                                      docs, row['category'], row['answer']))
    return queries, memories


def run(queries, memories, encoder, budgets: tuple[int, ...]) -> dict:
    if not budgets or any(b < 1 for b in budgets):
        raise ValueError('positive budgets required')
    results = {}
    for budget in budgets:
        results[str(budget)] = {
            arm: evaluate(queries, encoder, representation=arm, search_depth=None,
                          context_chars=budget,
                          memory_by_corpus=memories if arm != 'append' else None)
            for arm in ('append', 'structure', 'progression')
        }
    return {'question_count': len(queries), 'reader': 'multi_evidence_extractive',
            'search_depth': 'all', 'budgets': budgets, 'results': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--semantic-model', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--budgets', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit, traces, qa = (json.loads(p.read_text()) for p in (args.audit, args.traces, args.qa))
    queries, memories = prepare(audit, traces, qa)
    result = run(queries, memories, PinnedEncoder(args.semantic_model, args.model_revision),
                 tuple(args.budgets))
    result.update(encoder=args.semantic_model, encoder_revision=args.model_revision,
                  source_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                  trace_sha256=hashlib.sha256(args.traces.read_bytes()).hexdigest(),
                  qa_sha256=hashlib.sha256(args.qa.read_bytes()).hexdigest(),
                  caveat='Author-written diagnostic QA, silver writer labels, extractive reader; no official QA gain claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({b: {arm: output['overall'] for arm, output in arms.items()}
                      for b, arms in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
