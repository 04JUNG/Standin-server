import json
from types import SimpleNamespace

import pytest

from converter.body_trace import BodyTrace, PREFIX, relay_lines, validated_event


def test_nested_calls_preserve_failures_and_never_log_arguments():
    rows = []
    trace = BodyTrace(enabled=True, emit=rows.append)
    original = lambda value: (_ for _ in ()).throw(ValueError(value))
    owner = SimpleNamespace(call=original)
    trace.wrap(owner, 'call', 'neutral_restore')
    with pytest.raises(ValueError, match='private path'):
        with trace.span('prepare_total'):
            owner.call('private path and token')
    assert [r['succeeded'] for r in rows] == [False, False]
    assert rows[0]['parent_span_id'] == rows[1]['span_id']
    assert rows[0]['duration_ms'] <= rows[1]['duration_ms']
    assert 'private' not in json.dumps(rows)
    trace.__exit__()
    assert owner.call is original


def test_stdout_relay_drops_private_fields_forged_correlation_and_bad_metrics():
    good = dict(event='converter_body_stage', stage='neutral_restore', span_id=2,
        parent_span_id=1, start_ms=0., duration_ms=1., succeeded=True)
    rows = [good | dict(token='private', request_id='forged'),
            good | dict(stage='/private/path'), good | dict(duration_ms=float('nan')),
            dict(event='converter_body_reuse', status='prepared', hit=True, controls='private'),
            dict(event='converter_body_reuse', status='prepared', hit=False)]
    raw = 'Blender private stdout\n' + '\n'.join(PREFIX + json.dumps(r) for r in rows)
    emitted = []
    relay_lines(raw, emitted.append, 'req_owned')
    assert len(emitted) == 2
    assert all(json.loads(r)['request_id'] == 'req_owned' for r in emitted)
    assert 'private' not in '\n'.join(emitted)
    assert 'forged' not in '\n'.join(emitted)


def test_disabled_trace_and_failed_sink_never_change_the_return_value():
    rows = []
    with BodyTrace(enabled=False, emit=rows.append) as trace:
        with trace.span('prepare_total'):
            pass
    assert rows == []
    trace = BodyTrace(enabled=True, emit=lambda row: (_ for _ in ()).throw(OSError('full')))
    owner = SimpleNamespace(call=lambda: 'unchanged')
    trace.wrap(owner, 'call', 'neutral_restore')
    assert owner.call() == 'unchanged'
    trace.__exit__()


@pytest.mark.parametrize('result', ['prepared', 'disabled', 'custom_shape', 'not_prepared', 'invalid_prepared'])
def test_reuse_event_reports_every_safe_fallback(result):
    rows = []
    trace = BodyTrace(enabled=True, emit=rows.append)
    trace.reused(({} if result == 'prepared' else None, result))
    assert validated_event(rows[0]) == dict(event='converter_body_reuse',
        status=result, hit=result == 'prepared')


def test_solver_summary_counts_search_without_exposing_numeric_pose_data():
    rows = []
    trace = BodyTrace(enabled=True, emit=rows.append)
    trace.solver_report({'foot_plant_selection': {'candidates': {'L': [{'private': 1}], 'R': [{}, {}]},
        'fallback_to_exact_v321': True}})
    assert validated_event(rows[0]) == dict(event='converter_body_solver', stage='foot_plant',
        candidate_count=3, fallback=True)
    assert 'private' not in json.dumps(rows)
    owner = SimpleNamespace(call=lambda: 'unchanged')
    trace.wrap(owner, 'call', 'body_worker', after=lambda value: 1/0)
    assert owner.call() == 'unchanged'
    trace.__exit__()
