"""Post-retrieval mesh gating without model/cloud/Blender calls."""
import hashlib
import json
from urllib.error import HTTPError, URLError

import pytest

from converter.hand_contact import SCREEN_VERSION, validate_screen
from converter.hand_library import load_library
from converter.protocol import SOLVER_VERSION
from src.hand_skin_screen import HandSkinGate, HandSkinClient, unknown
from tests.test_fullbody_routing import engine, decision, evidence, Camera, Semantic


def receipt(status='clear_screen', **extra):
    return dict(status=status, method=SCREEN_VERSION, scope='opposite_hands_only',
        complete_collision_certificate=False, triangle_counts=dict(left=12, right=12),
        intersection_pairs=int(status == 'intersection'), **extra)


class Checker:
    def __init__(self, statuses=()):
        self.statuses, self.calls = iter(statuses), []

    def check(self, text, hands, timeout):
        self.calls.append((text, hands, timeout))
        status = next(self.statuses, 'clear_screen')
        return unknown('offline') if status == 'unavailable' else receipt(status)


def gate(tmp_path, statuses=(), ids=('a', 'b', 'c')):
    hashes = {}
    for pid in ids:
        data = 'BVH '+pid
        (tmp_path/(pid+'.bvh')).write_text(data)
        hashes[pid] = hashlib.sha256(data.encode()).hexdigest()
    return HandSkinGate(Checker(statuses), tmp_path, hashes, budget_seconds=30.)


def test_crossing_removed_and_next_clear_pose_used(tmp_path):
    e = engine(); e.top_k = 1; e.hand_skin_gate = gate(tmp_path, ['intersection', 'clear_screen'])
    out = e.search(decision(), evidence())
    assert out['status'] == 'ok' and len(out['selected']) == 1
    assert out['selected'][0]['hand_skin_screen']['status'] == 'clear_screen'
    assert out['selected'][0]['pose_id'] != out['hand_skin_screen']['rejected'][0]['pose_id']
    assert len(e.hand_skin_gate.client.calls) == 2 and e.generator.calls == 0
    assert not out['selected'][0]['actual_mesh_validated']


def test_unavailable_is_error_not_empty_generation_and_fast_fails(tmp_path):
    e = engine(); e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = e.search(decision(), evidence())
    assert out['status'] == 'error' and out['selected'] == []
    assert out['hand_skin_screen']['unverified'] and e.generator.calls == 0
    assert len(e.hand_skin_gate.client.calls) == 1


def test_partial_error_keeps_only_clear_results(tmp_path):
    e = engine(); e.hand_skin_gate = gate(tmp_path, ['clear_screen', 'unavailable'])
    out = e.search(decision(), evidence())
    assert out['status'] == 'partial_error' and len(out['selected']) == 1
    assert out['hand_skin_screen']['unverified'] and e.generator.calls == 0


@pytest.mark.parametrize('status', ['intersection', 'unavailable'])
def test_generated_candidate_also_screened_no_unsafe_bvh_in_diagnostics(tmp_path, status):
    e = engine(Camera(()), Semantic(())); e.hand_skin_gate = gate(tmp_path, [status])
    out = e.search(decision(), evidence())
    assert e.generator.calls == 1 and out['generation']['status'] == 'rejected'
    assert not out['generation']['candidates']
    records = out['hand_skin_screen']['rejected' if status == 'intersection' else 'unverified']
    assert records[0]['generated'] and 'bvh' not in records[0]
    if status == 'unavailable': assert out['status'] == 'error'


def test_all_intersections_can_trigger_existing_generation_and_screen_it(tmp_path):
    e = engine(); e.hand_skin_gate = gate(tmp_path, ['intersection']*3 + ['clear_screen'])
    out = e.search(decision(), evidence())
    assert not out['selected'] and e.generator.calls == 1
    assert out['generation']['status'] == 'preview_ready'
    assert out['generation']['candidates'][0]['hand_skin_screen']['status'] == 'clear_screen'
    assert out['generation']['release_allowed'] is False


def test_changed_or_unbound_bvh_is_unknown_not_safe(tmp_path):
    g = gate(tmp_path); (tmp_path/'a.bvh').write_text('changed')
    for pid in ('a', '../escape', 'not-in-library'):
        session = g.session()
        assert session.filter([{'pose_id': pid}]) == [] and session.unverified
    assert not g.client.calls


def test_shared_budget_and_presets_library_and_generated(tmp_path):
    g = gate(tmp_path); g.max_checks = 1
    hands = {'left': 'relaxed', 'right': 'fist'}
    session = g.session(hands)
    assert session.filter([{'pose_id': 'a'}])
    assert not session.filter([{'bvh': 'new BVH'}], generated=True)
    assert session.unverified and len(g.client.calls) == 1 and g.client.calls[0][1] == hands
    assert session.filter([{'pose_id': 'a'}]) and len(g.client.calls) == 1


def test_exhausted_time_budget_skips_network(tmp_path):
    g = gate(tmp_path); session = g.session(); session.spent_seconds = 31
    assert session.filter([{'pose_id': 'a'}]) == [] and not g.client.calls


class Response:
    status = 200
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self, limit): return json.dumps(self.payload).encode()


@pytest.mark.parametrize('changed', [None, 'source_bvh_sha256', 'character_sha256',
    'character_id', 'solver_version', 'hands', 'hand_library_sha256', 'screened_artifact_sha256', 'method'])
def test_http_receipt_binds_pose_character_solver_and_hands(changed):
    text = 'BVH input'
    payload = receipt(source_bvh_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_id='test-character', character_sha256='1'*64, solver_version=SOLVER_VERSION,
        hands={'left': 'fist', 'right': 'relaxed'}, hand_library_sha256=load_library().sha256,
        screened_artifact_sha256='2'*64)
    if changed: payload[changed] = 'wrong'
    def opener(request, timeout):
        assert request.full_url == 'http://converter/screen-hand-contact'
        assert b'hand_left' in request.data and b'fist' in request.data and timeout == 3
        return Response(payload)
    client = HandSkinClient('http://converter', 'test-character', '1'*64, opener=opener)
    result = client.check(text, {'left': 'fist', 'right': 'relaxed'}, 3)
    assert result['status'] == ('unavailable' if changed else 'clear_screen')


@pytest.mark.parametrize('field,value', [('intersection_pairs', True), ('intersection_pairs', 2),
    ('triangle_counts', {'left': 0, 'right': 1}), ('complete_collision_certificate', True),
    ('scope', 'whole_body'), ('status', 'passed')])
def test_false_clear_receipts_rejected(field, value):
    result = receipt(); result[field] = value
    with pytest.raises(ValueError): validate_screen(result)


def test_new_fields_survive_public_api_serialization(tmp_path):
    from api.models import DynamicSearchOut
    e = engine(); e.hand_skin_gate = gate(tmp_path)
    result = DynamicSearchOut(**e.search(decision(), evidence())).model_dump()
    assert result['hand_skin_screen']['checks'] > 0
    assert result['selected'][0]['hand_skin_screen']['status'] == 'clear_screen'


@pytest.mark.parametrize('status', ['clear_screen', 'intersection', 'unavailable'])
def test_geometry_review_fallback_cannot_bypass_skin_gate(tmp_path, status):
    from tests.test_integrated_recovery import constrained_scene
    from api.models import DynamicSearchOut
    e, d, ev, _ = constrained_scene()
    e.generator = None
    e.hand_skin_gate = gate(tmp_path, [status], ids=('a',))
    result = e.search(d, ev, review_geometry=True)
    assert len(e.hand_skin_gate.client.calls) == 1
    wire = DynamicSearchOut.model_validate(result).model_dump()
    if status == 'clear_screen':
        [row] = wire['selected']
        assert row['hand_skin_screen']['status'] == 'clear_screen'
        assert row['match_level'] == 'geometry_fallback'
        assert row['review_required'] and row['not_exact_match']
        assert row['constraints_satisfied'] is False
        assert not row['release_allowed'] and not row['refine_allowed']
    else:
        assert not wire['selected']
        assert wire['status'] == ('error' if status == 'unavailable' else 'no_result')


def test_generated_screen_error_is_not_hidden_by_geometry_fallback(tmp_path, monkeypatch):
    from tests.test_integrated_recovery import constrained_scene
    import src.pose_requirements as conditions
    monkeypatch.setattr(conditions, 'audit_generated', lambda *a: {'status': 'passed'})
    e, d, ev, _ = constrained_scene()
    e.hand_skin_gate = gate(tmp_path, ['unavailable'], ids=('a',))
    result = e.search(d, ev, review_geometry=True)
    assert result['status'] == 'error' and not result['selected']
    assert 'geometry_review_fallback' not in result


def test_factory_reads_real_config_attribute_names(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.config import CFG
    from src.hand_skin_screen import build_hand_skin_gate
    monkeypatch.setattr(CFG, 'thumbnail_character_id', 'standin-master-v2')
    monkeypatch.setattr(CFG, 'thumbnail_converter_url', 'http://converter')
    result = build_hand_skin_gate(SimpleNamespace(hashes={}), tmp_path)
    assert result.client.character_id == 'standin-master-v2'
    assert result.client.base_url == 'http://converter'
    assert len(result.client.character_sha256) == 64
    assert result.budget_seconds == CFG.hand_skin_screen_budget_seconds


def test_staging_seventh_screen_trace_preserves_gate_and_fits_bounded_budget(monkeypatch, tmp_path):
    import src.hand_skin_screen as module
    # Actual Converter durations: two intersections before the fifth safe candidate.
    durations = [4.271759, 4.052420, 4.024773, 7.226617, 4.030341, 4.011594, 7.467987]
    statuses = ['intersection', 'clear_screen', 'clear_screen', 'clear_screen',
                'intersection', 'clear_screen', 'clear_screen']

    def run(budget):
        now = [0.]
        monkeypatch.setattr(module, 'monotonic', lambda: now[0])
        class TimedChecker:
            def __init__(self): self.calls = 0
            def check(self, text, hands, timeout):
                index = self.calls
                self.calls += 1
                now[0] += min(durations[index], timeout)
                if durations[index] > timeout:
                    return unknown('hand_screen_request_timeout')
                return receipt(statuses[index])
        client = TimedChecker()
        g = HandSkinGate(client, tmp_path, {}, budget_seconds=budget)
        session = g.session()
        selected = []
        for i in range(7):
            result = session.check('actual-candidate-' + str(i))
            if result['status'] == 'clear_screen': selected.append(i)
        return selected, session, client

    selected, old, _ = run(30.)
    assert selected == [1, 2, 3, 5] and old.failed
    selected, new, client = run(45.)
    assert selected == [1, 2, 3, 5, 6] and not new.failed
    assert client.calls == 7 and new.spent_seconds < 45.
    assert 0 not in selected and 4 not in selected  # Both true intersections still rejected.


@pytest.mark.parametrize('budget', [0., -1., float('inf'), float('nan'), 60.01])
def test_invalid_skin_budget_fails_before_network(tmp_path, budget):
    with pytest.raises(ValueError):
        HandSkinGate(Checker(), tmp_path, {}, budget_seconds=budget)


@pytest.mark.parametrize('failure,reason', [
    (TimeoutError('private URL'), 'hand_screen_request_timeout'),
    (URLError(TimeoutError('private URL')),
     'hand_screen_request_timeout'),
    (URLError('private hostname'),
     'hand_screen_network_error'),
])
def test_client_diagnostics_distinguish_timeout_without_private_values(failure, reason):
    def opener(*args, **kwargs): raise failure
    client = HandSkinClient('http://private-host', 'character', '1'*64, opener=opener)
    result = client.check('private BVH', None, 3.)
    assert result['status'] == 'unavailable' and result['reason'] == reason
    assert 'private' not in json.dumps(result)


def test_client_lineage_diagnostic_names_fields_without_values():
    text = 'BVH input'
    payload = receipt(source_bvh_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_id='wrong-private-character', character_sha256='1'*64,
        solver_version=SOLVER_VERSION, hands=None,
        hand_library_sha256=load_library().sha256, screened_artifact_sha256='2'*64)
    client = HandSkinClient('http://private-host', 'expected', '1'*64,
                           opener=lambda *a, **k: Response(payload))
    result = client.check(text, None, 3.)
    assert result['reason'] == 'hand_screen_receipt_lineage_mismatch'
    assert result['mismatched_fields'] == ['character_id']
    assert 'wrong-private-character' not in json.dumps(result)


def test_http_error_reports_only_status_code():
    from io import BytesIO
    error = HTTPError('http://private-host', 503, 'private detail', {}, BytesIO(b'private body'))
    def opener(*args, **kwargs): raise error
    client = HandSkinClient('http://private-host', 'character', '1'*64, opener=opener)
    result = client.check('private BVH', None, 3.)
    assert result['reason'] == 'hand_screen_http_error'
    assert result['upstream_status'] == 503 and error.closed
    assert 'private' not in json.dumps(result)


def test_budget_and_check_limit_have_distinct_diagnostics(tmp_path):
    g = gate(tmp_path)
    session = g.session()
    session.spent_seconds = 30.
    assert session.check('candidate')['reason'] == 'hand_screen_budget_exhausted'
    session = g.session()
    session.checks = 10
    assert session.check('candidate')['reason'] == 'hand_screen_check_limit_exhausted'
    assert not g.client.calls


def test_client_sends_bounded_work_budget_and_optional_priority_is_session_local(tmp_path):
    requests = []
    def opener(request, timeout):
        requests.append((dict(request.header_items()), timeout))
        raise TimeoutError('private data')
    client = HandSkinClient('http://private-host', 'character', '1'*64, opener=opener)
    session = HandSkinGate(client, tmp_path, {}).session()
    with session.optional_phase():
        session.client.check('BVH', None, 3.)
    session.client.check('BVH', None, .05)
    client.check('BVH', None, 15.)
    assert requests[0][0]['X-standin-work-budget-ms'] == '2850'
    assert requests[0][0]['X-standin-screen-optional'] == '1'
    assert requests[1][0]['X-standin-work-budget-ms'] == '1'
    assert 'X-standin-screen-optional' not in requests[1][0]
    assert requests[2][0]['X-standin-work-budget-ms'] == '14850'
    assert not hasattr(client, '_screen_optional')


def test_http_queue_deadline_reports_safe_timing_and_cause_without_body_values():
    from io import BytesIO
    error = HTTPError('http://private-host', 504, 'private detail', {
        'X-Standin-Queue-Wait-Ms':'1234.5', 'X-Standin-Execution-Ms':'nan',
        'X-Standin-Work-Phase':'queue', 'X-Standin-Work-Reason':'WORK_BUDGET_EXHAUSTED'},
        BytesIO(b'{"error":{"code":"WORK_BUDGET_EXHAUSTED","message":"private raw text"}}'))
    def opener(*args, **kwargs): raise error
    result = HandSkinClient('http://private-host', 'character', '1'*64, opener=opener).check('BVH', None, 3.)
    assert result['reason'] == 'hand_screen_work_budget_exhausted'
    assert result['availability'] == {'status':'unavailable', 'queue_wait_ms':1234.5,
        'phase':'queue', 'cause':'WORK_BUDGET_EXHAUSTED'}
    assert 'private' not in json.dumps(result)


def test_per_check_and_summary_logs_explain_fail_closed_failure_without_payload(tmp_path, monkeypatch):
    events=[]
    monkeypatch.setattr('src.logging_setup.log_info',lambda name,**fields:events.append((name,fields)))
    class QueueFailure:
        def check(self,text,hands,timeout):
            return unknown('hand_screen_work_budget_exhausted',availability={
                'status':'unavailable','phase':'queue','queue_wait_ms':100.,
                'execution_ms':0.,'cause':'WORK_BUDGET_EXHAUSTED'})
    e=engine(); e.hand_skin_gate=gate(tmp_path); e.hand_skin_gate.client=QueueFailure()
    result=e.search(decision(),evidence())
    assert result['status']=='error' and not result['selected']
    check=[fields for name,fields in events if name=='hand_skin_screen_check'][0]
    summary=[fields for name,fields in events if name=='hand_skin_screen_summary'][0]
    assert check['check']==1 and check['status']=='unavailable' and check['phase']=='queue'
    assert check['call_budget_ms']==15000 and check['optional'] is False
    assert check['queue_wait_ms']==100 and check['cause']=='WORK_BUDGET_EXHAUSTED'
    assert summary['unverified_count']>=1 and summary['status']=='error'
    assert all('bvh' not in key.lower() and 'url' not in key.lower() for key in check)


def test_unavailable_receipt_does_not_log_arbitrary_upstream_reason():
    text='BVH input'
    payload=receipt('unavailable',reason='private raw upstream failure',
        source_bvh_sha256=hashlib.sha256(text.encode()).hexdigest(),
        character_id='expected',character_sha256='1'*64,solver_version=SOLVER_VERSION,
        hands=None,hand_library_sha256=load_library().sha256,screened_artifact_sha256='2'*64)
    client=HandSkinClient('http://private-host','expected','1'*64,
                          opener=lambda *a,**k:Response(payload))
    result=client.check(text,None,3.)
    assert result['status']=='unavailable' and result['reason']=='hand_screen_mesh_unavailable'
    assert 'private' not in json.dumps(result)


def test_malformed_http_failure_code_is_not_reflected_or_raised():
    from io import BytesIO
    error=HTTPError('http://private-host',503,'private message',{},
        BytesIO(b'{"error":{"code":["private response"]}}'))
    def opener(*args,**kwargs):raise error
    result=HandSkinClient('http://private-host','character','1'*64,opener=opener).check('BVH',None,3.)
    assert result['status']=='unavailable' and result['reason']=='hand_screen_http_error'
    assert 'private' not in json.dumps(result)
