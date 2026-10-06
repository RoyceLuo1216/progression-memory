"""Replay silver MSC traces against the pinned source and export a review CSV.

The CSV contains raw source utterances and belongs in a short-lived CI artifact,
not in version control. Decisions in JSON are provisional model annotations.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from progression_memory.maintenance.calibrated_progression import (
    Action, Decision, Event, Memory, TargetUpdate,
)


def review_rows(traces, audit):
    source = {c['initial_data_id']: c for c in audit['conversations']}
    assert set(source) == {t['conversation_id'] for t in traces['traces']}
    output = []
    for trace in traces['traces']:
        conversation_id = trace['conversation_id']
        memory = Memory()
        turns = source[conversation_id]['turns']
        lookup = {f"s{t['session']}:t{t['turn']}": t for t in turns}
        assert list(lookup) == [r['ref'] for r in trace['rows']], conversation_id
        for row in trace['rows']:
            turn = lookup[row['ref']]
            slot = int(turn['speaker_id'].rsplit(':', 1)[1])
            if row['subject'] != slot and row['action'] != 'no_update':
                assert (conversation_id, row['ref']) in {
                    ('train:ordered_4998', 's2:t2'),
                    ('train:ordered_6340', 's3:t10'),
                }
            subject = f"{conversation_id}:{row['subject']}"
            targets = tuple(f"progression:{n}" for n in row.get('targets', []))
            updates = tuple(TargetUpdate(f"progression:{u['target']}", u['state'],
                                         row['evidence'], u.get('segment', False))
                            for u in row.get('updates', []))
            decision = Decision(action=Action(row['action']), target_ids=targets,
                                parent_id=f"progression:{row['parent']}" if 'parent' in row else None,
                                identity=row.get('identity', ''), scope=row.get('scope', ''),
                                state=row.get('state', ''), change=row['evidence'],
                                new_segment=row.get('segment', False), reason=row['evidence'],
                                target_updates=updates)
            memory.apply(Event(turn['source_id'], subject, turn['text'], None,
                               turn['source_id']), decision)
            output.append({
                'conversation_id': conversation_id, 'split': trace['split'],
                'source_id': turn['source_id'],
                'speaker_slot': slot, 'subject_slot': row['subject'],
                'source_text': turn['text'], 'action': row['action'],
                'progression': row.get('identity', ''),
                'parent': row.get('parent', ''), 'targets': ','.join(targets),
                'proposed_state': row.get('state') or json.dumps(row.get('updates', []), ensure_ascii=False),
                'new_segment': row.get('segment', False), 'evidence_paraphrase': row['evidence'],
                'label_origin': row['label_origin'], 'review_priority': row['review_priority'],
                'review_issue': row.get('review_issue', ''),
                'reviewer_action': '', 'reviewer_subject': '', 'reviewer_state': '',
                'reviewer_notes': '',
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('traces', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = review_rows(json.loads(args.traces.read_text()), json.loads(args.audit.read_text()))
    with args.output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'turns': len(rows), 'actions': Counter(row['action'] for row in rows),
                      'high_priority': sum(row['review_priority'] == 'high' for row in rows)}))


if __name__ == '__main__':
    main()
