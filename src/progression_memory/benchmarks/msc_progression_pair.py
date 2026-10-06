"""Paired, prequential continuation retrieval on held-out MSC graph traces.

The future turn is the query. Only earlier source events are indexed. Labels
identify the earlier process events to recover, never text used in the query.
This tests a representation with supplied silver decisions, not writer accuracy
or downstream QA on LoCoMo/LongMemEval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from progression_memory.evaluation.representation_rag import records, retrieve
from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, Memory, TargetUpdate,
)
from progression_memory.benchmarks.representation_rag_benchmark import PinnedEncoder


def _decision(row: dict) -> Decision:
    return Decision(
        action=Action(row['action']),
        target_ids=tuple(f"progression:{n}" for n in row.get('targets', [])),
        parent_id=f"progression:{row['parent']}" if 'parent' in row else None,
        identity=row.get('identity', ''), scope=row.get('scope', ''),
        state=row.get('state', ''), change=row['evidence'],
        new_segment=row.get('segment', False), reason=row['evidence'],
        target_updates=tuple(TargetUpdate(f"progression:{u['target']}", u['state'],
                                          row['evidence'], u.get('segment', False))
                             for u in row.get('updates', [])),
    )


def cases(traces: dict, audit: dict, *, split: str = 'holdout_review') -> list[dict]:
    sources = {c['initial_data_id']: c for c in audit['conversations']}
    output = []
    for trace in traces['traces']:
        if trace.get('split') != split:
            continue
        cid = trace['conversation_id']
        turns = sources[cid]['turns']
        lookup = {f"s{t['session']}:t{t['turn']}": t for t in turns}
        if list(lookup) != [r['ref'] for r in trace['rows']]:
            raise ValueError(f"source order mismatch for {cid}")
        memory = Memory()
        for row in trace['rows']:
            turn = lookup[row['ref']]
            decision = _decision(row)
            if decision.target_ids:
                # The most recent earlier event on each target is the evidence
                # needed to recover its state when this new turn arrives.
                evidence = frozenset(
                    memory.segments[memory.progressions[pid].segment_ids[-1]].event_ids[-1]
                    for pid in decision.target_ids
                )
                output.append({'id': f"{cid}:{row['ref']}", 'query': turn['text'],
                               'evidence': evidence, 'memory': deepcopy(memory),
                               'action': row['action'], 'source_id': turn['source_id']})
            memory.apply(Event(turn['source_id'], f"{cid}:{row['subject']}",
                               turn['text'], None, turn['source_id']), decision)
    return output


def evaluate(cases_: list[dict], embedder, *, depths: tuple[int, ...],
             context_chars: int) -> dict:
    if not depths or any(depth < 1 for depth in depths) or context_chars < 1:
        raise ValueError('positive depths and context budget required')
    rows = []
    for case in cases_:
        for depth in depths:
            per_representation = {}
            for name in ('append', 'structure', 'progression'):
                result = retrieve(case['query'], case['memory'], embedder,
                                  representation=name, search_depth=depth,
                                  context_chars=context_chars)
                found = set(result.source_event_ids) & case['evidence']
                per_representation[name] = {
                    'recall': len(found) / len(case['evidence']),
                    'complete': found == case['evidence'],
                    'context_chars': result.used_chars,
                    'source_ids': result.source_event_ids,
                }
            rows.append({'id': case['id'], 'action': case['action'],
                         'depth': depth, 'evidence_ids': sorted(case['evidence']),
                         'results': per_representation})
    aggregates = {}
    for depth in depths:
        group = [row for row in rows if row['depth'] == depth]
        aggregates[str(depth)] = {
            name: {
                'recall': sum(row['results'][name]['recall'] for row in group) / len(group),
                'complete_rate': sum(row['results'][name]['complete'] for row in group) / len(group),
                'avg_context_chars': sum(row['results'][name]['context_chars'] for row in group) / len(group),
            } for name in ('append', 'structure', 'progression')
        }
    return {'case_count': len(cases_), 'context_chars': context_chars,
            'depths': depths, 'aggregates': aggregates, 'per_case': rows}


def evaluate_budget_sweep(cases_: list[dict], embedder, budgets: tuple[int, ...]) -> dict:
    """Rank the full index; vary only the same hard context cap in each arm."""
    if not budgets or any(b < 1 for b in budgets):
        raise ValueError('positive context budgets required')
    output = {}
    for budget in budgets:
        paired = []
        for case in cases_:
            result = {}
            for name in ('append', 'structure', 'progression'):
                depth = len(records(case['memory'], name))
                hit = retrieve(case['query'], case['memory'], embedder,
                               representation=name, search_depth=depth,
                               context_chars=budget)
                found = set(hit.source_event_ids) & case['evidence']
                result[name] = {'recall': len(found) / len(case['evidence']),
                                'complete': found == case['evidence'],
                                'context_chars': hit.used_chars,
                                'source_ids': hit.source_event_ids}
            paired.append({'id': case['id'], 'evidence_ids': sorted(case['evidence']),
                           'results': result})
        output[str(budget)] = {
            'aggregates': {name: {
                'recall': sum(x['results'][name]['recall'] for x in paired) / len(paired),
                'complete_rate': sum(x['results'][name]['complete'] for x in paired) / len(paired),
                'avg_context_chars': sum(x['results'][name]['context_chars'] for x in paired) / len(paired),
            } for name in ('append', 'structure', 'progression')},
            'per_case': paired,
        }
    return {'search_depth': 'all', 'budgets': budgets, 'results': output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--split', default='holdout_review')
    parser.add_argument('--semantic-model', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--depths', type=int, nargs='+', required=True)
    parser.add_argument('--context-chars', type=int, required=True)
    parser.add_argument('--budget-sweep', type=int, nargs='+',
                        help='Also compare full-index retrieval at equal hard context caps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    traces = json.loads(args.traces.read_text())
    sample = cases(traces, audit, split=args.split)
    if not sample:
        raise ValueError(f'no eligible attachment cases for {args.split}')
    encoder = PinnedEncoder(args.semantic_model, args.model_revision)
    result = evaluate(sample, encoder,
                      depths=tuple(args.depths), context_chars=args.context_chars)
    if args.budget_sweep:
        result['budget_sweep'] = evaluate_budget_sweep(sample, encoder, tuple(args.budget_sweep))
    result.update(split=args.split, encoder=args.semantic_model,
                  encoder_revision=args.model_revision,
                  audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                  traces_sha256=hashlib.sha256(args.traces.read_bytes()).hexdigest(),
                  interpretation='Silver supplied writer labels; process-continuation retrieval only; no QA or writer accuracy.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': result['case_count'], 'aggregates': result['aggregates'],
                      'budget_sweep': {k: v['aggregates'] for k, v in
                                       result.get('budget_sweep', {}).get('results', {}).items()}}, indent=2))


if __name__ == '__main__':
    main()
