"""Silver traces are complete, deterministic and replay against original MSC."""

import json
import os
import runpy
from pathlib import Path

from progression_memory.maintenance.calibrated_progression import Action


HERE = Path('evaluation/data/progression_routing')


def test_silver_trace_rebuilds_and_exposes_defaults():
    code = runpy.run_path(str(HERE / 'build_msc_graph_traces_v2.py'))
    rebuilt = code['build']()
    stored = json.loads((HERE / 'msc_graph_traces_v2.json').read_text())
    assert rebuilt == stored
    rows = [r for trace in stored['traces'] for r in trace['rows']]
    assert len(stored['traces']) == 12 and len(rows) == 309
    assert {r['action'] for r in rows} == {action.value for action in Action}
    assert all(r['review_priority'] == 'high' for r in rows
               if r['label_origin'] == 'default_no_update')
    assert all(r['evidence'] for r in rows)


def test_silver_trace_replays_pinned_audit_if_available():
    audit_path = os.getenv('MSC_DEV_AUDIT_PATH')
    if not audit_path:
        return
    code = runpy.run_path(str(HERE / 'review_msc_graph_traces.py'))
    audit = json.loads(Path(audit_path).read_text())
    stored = json.loads((HERE / 'msc_graph_traces_v2.json').read_text())
    rows = code['review_rows'](stored, audit)
    assert len(rows) == 309
    assert sum(r['review_priority'] == 'high' for r in rows) >= 116
    assert len({r['source_id'] for r in rows}) == 309
