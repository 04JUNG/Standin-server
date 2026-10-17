"""Post-retrieval mesh gating without model/cloud/Blender calls."""
import hashlib
import json

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
    return HandSkinGate(Checker(statuses), tmp_path, hashes)


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
