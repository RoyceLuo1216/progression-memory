"""Complete real-dialogue validation traces for multiple process changes."""

import json
import os
import runpy
from collections import Counter
from pathlib import Path


HERE = Path('evaluation/data/progression_routing')


def test_expansion_rebuild_and_split_coverage():
    rebuilt = runpy.run_path(str(HERE / 'build_msc_graph_expansion.py'))['build']()
    stored = json.loads((HERE / 'msc_graph_expansion_v1.json').read_text())
    assert rebuilt == stored
    assert len(stored['traces']) == 3
    assert {t['conversation_id'] for t in stored['traces']}.isdisjoint({
        t['conversation_id'] for t in json.loads((HERE / 'msc_graph_traces_v2.json').read_text())['traces']})
    assert sum(len(t['rows']) for t in stored['traces']) == 84
    action_by_split = {split: Counter(row['action'] for trace in stored['traces']
                                     if trace['split'] == split for row in trace['rows'])
                       for split in ('development', 'holdout_review')}
    assert action_by_split['development']['multi_attach'] == 2
    assert action_by_split['holdout_review']['multi_attach'] == 2
    assert action_by_split['holdout_review']['create_and_attach'] == 1


def test_expansion_replays_source_when_available():
    audit_path = os.getenv('MSC_EXPANSION_AUDIT_PATH')
    if not audit_path:
        return
    rows = runpy.run_path(str(HERE / 'review_msc_graph_traces.py'))['review_rows'](
        json.loads((HERE / 'msc_graph_expansion_v1.json').read_text()),
        json.loads(Path(audit_path).read_text()))
    assert len(rows) == 84
    assert all(row['source_text'] for row in rows)
    assert sum(row['action'] == 'multi_attach' for row in rows) == 4
