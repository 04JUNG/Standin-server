"""Definite head-only omission is not a clear-screen or a permission token."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib

import pytest

from src.body_scope import BodyScope, ScopeDetection
from src.config import CFG
from src.hand_scope import is_head_only_output, head_hand_omission, head_hand_summary
from src.hand_shapes import apply_hand_shapes, detect_person_hands
from src.pipeline import Pipeline
from src.vlm.client import _coerce
from tests.test_hand_shapes import _side, _analysis
from tests.test_hand_skin_gate import gate
from tests.test_fullbody_routing import engine, decision, evidence, Camera, Semantic
from tests.test_mirror_reference import fixture as mirror_fixture
from tests.test_pose_presentation import hit, report
from src.mirror_reference import mirror_reference
from api.pose_presentation import PreviewAssetOut, build_pose_presentation
from converter.hand_contact import validate_screen


@pytest.mark.parametrize('scope', ['full', 'half', 'bust', 'shoulder_head', None, 'HEAD'])
def test_other_or_unknown_scopes_never_claim_head_only(scope):
    assert not is_head_only_output(scope, 'vlm_person')


@pytest.mark.parametrize('source', ['unknown', 'observed_legs', None, 'user', '', {}, []])
def test_head_label_without_trusted_ownership_does_not_skip_work(source):
    assert not is_head_only_output('head', source)


@pytest.mark.parametrize('source', ['vlm_person', 'vlm_description', 'legacy_shot'])
def test_explicit_head_sources_omit_work(source):
    assert is_head_only_output('head', source)


def test_subset_response_preserves_original_slots_and_unrequested_heads():
    vlm = _analysis(4)
    apply_hand_shapes(vlm, {'people': [
        {'left': _side('fist')}, {'left': _side('grip')},
    ]}, person_indices=(1, 3))
    assert [h.status for h in vlm.person_hands] == ['not_requested', 'vlm', 'not_requested', 'vlm']
    assert detect_person_hands(vlm, 1).left.preset == 'fist'
    assert detect_person_hands(vlm, 3).left.preset == 'grip'
    assert detect_person_hands(vlm, 0).left.source == 'default'


@pytest.mark.parametrize('payload', [None, {'people': []}, {'people': [{}, {}, {}]}])
def test_failed_subset_marks_only_requested_slots_unavailable(payload):
    vlm = _analysis(3)
    apply_hand_shapes(vlm, payload, person_indices=(1,))
    assert [h.status for h in vlm.person_hands] == ['not_requested', 'unavailable', 'not_requested']


@pytest.mark.parametrize('indices', [(0, 0), (-1,), (3,), (True,), ('0',)])
def test_invalid_subset_mapping_fails_before_mutation(indices):
    vlm = _analysis(3)
    with pytest.raises(ValueError):
        apply_hand_shapes(vlm, {'people': [{}]}, person_indices=indices)
    assert 'hand_shapes' not in vlm.raw


def test_legacy_all_head_starts_no_hand_thread_and_mixed_boxes_keep_original_order(monkeypatch):
    monkeypatch.setattr(CFG, 'vlm_hand_shapes', True)
    calls = []
    p = Pipeline([], vlm_client=SimpleNamespace(read_hands=lambda image, boxes:
        calls.append(boxes) or {'people': [{'left': _side('fist')} for _ in boxes]}))
    vlm = _analysis(3)
    vlm.body_scopes = [BodyScope.HEAD] * 3
    p._start_hand_shapes(None, 100, 200, vlm)
    assert vlm._hand_shapes_future is None and calls == []
    p._join_hand_shapes(vlm)
    vlm.body_scopes = [BodyScope.HEAD, BodyScope.FULL, BodyScope.HEAD]
    p._start_hand_shapes(None, 100, 200, vlm)
    p._join_hand_shapes(vlm)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert calls[0][0] == (.4, .1, .7, .9)
    assert [detect_person_hands(vlm, i).status for i in range(3)] == ['not_requested', 'vlm', 'not_requested']


@pytest.mark.parametrize('scopes', [('head',), ('head', 'head'), ('head', 'half')])
def test_fullbody_waits_for_final_description_before_hand_request(monkeypatch, tmp_path, scopes):
    from src.library import build_synthetic_index
    from src.search_routing import PoseDescription, DescribeOutcome
    from tests.test_search_routing import _CountingVLM, _RoughPose, _Img
    monkeypatch.setattr(CFG, 'vlm_hand_shapes', True)
    monkeypatch.setattr(CFG, 'vlm_person_tags', False)
    monkeypatch.setattr(CFG, 'body_matching_mode', 'off')
    events = []
    class VLM(_CountingVLM):
        def analyze(self, image, img_w, img_h):
            value = _analysis(len(scopes))
            value.body_scopes = [BodyScope.HALF] * len(scopes)
            return value
        def describe_people(self, image, boxes):
            events.append('describe')
            return DescribeOutcome('ok', descriptions={
                i: PoseDescription(**{**decision(False).semantic.description,
                    'visible_scope': scopes[i]}) for i in range(len(boxes))})
        def read_hands(self, image, boxes):
            events.append(('hands', boxes))
            return {'people': [{'left': _side('fist')} for _ in boxes]}
    entries = build_synthetic_index()
    pid = entries[0].pose_id
    e = engine(Camera((pid,)), Semantic((pid,)))
    e.meta = {entry.pose_id: entry.meta for entry in entries}
    e.allowed = set(e.meta)
    e.hand_skin_gate = gate(tmp_path, ids=(pid,))
    p = Pipeline(entries, vlm_client=VLM(), pose_model=_RoughPose(all_rough=True),
                 fullbody_search=e, skeleton_repairer=False)
    result = p.process_cut(_Img(f'full_half standing front {len(scopes)}p rough'), 100, 200)
    assert events[0] == 'describe'
    hand_events = [event for event in events if isinstance(event, tuple)]
    body_indices = [i for i, scope in enumerate(scopes) if scope != 'head']
    assert len(hand_events) == int(bool(body_indices))
    if body_indices:
        assert len(hand_events[0][1]) == len(body_indices)
        assert [box[0] for box in hand_events[0][1]] == pytest.approx([i * .4 for i in body_indices])
    by_slot = {d.quality_trace['vlm_person_index']: d for d in result.descriptors}
    for index, scope in enumerate(scopes):
        assert by_slot[index].output_scope == ScopeDetection(BodyScope(scope), 'vlm_description')
        assert by_slot[index].hands.status == ('not_requested' if scope == 'head' else 'vlm')
        if scope == 'head':
            assert result.person_routing[index]['fullbody_search']['hand_skin_screen'] == head_hand_summary()
        else:
            assert by_slot[index].hands.left.preset == 'fist'
    assert len(e.hand_skin_gate.client.calls) == len(body_indices)


def head_evidence():
    return {**evidence(), 'output_scope': 'head', 'output_scope_source': 'vlm_description'}


def gesture_decision(sufficient=True):
    d = decision(sufficient)
    # Keep the real semantic-pose path rather than the neutral closeup shortcut.
    d.semantic.description['visible_scope'] = 'full'
    return d


def test_head_skin_gate_is_not_started_and_search_permissions_stay_closed(tmp_path):
    e = engine()
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = e.search(gesture_decision(), head_evidence())
    assert out['status'] == 'ok' and out['selected']
    assert not e.hand_skin_gate.client.calls
    assert out['hand_skin_screen'] == head_hand_summary()
    for row in out['selected']:
        assert row['hand_skin_screen'] == head_hand_omission()
        assert row['release_allowed'] is False
        assert row['refine_allowed'] is False
        assert row['actual_mesh_validated'] is False
    with pytest.raises(ValueError):
        validate_screen(out['selected'][0]['hand_skin_screen'])


@pytest.mark.parametrize('scope,source', [('full','vlm_person'), ('half','vlm_person'),
    ('bust','vlm_person'), ('shoulder_head','vlm_person'), ('head','unknown'),
    ('head', None), (None, 'vlm_person')])
def test_non_head_or_untrusted_hand_gates_still_fail_closed(tmp_path, scope, source):
    e = engine()
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = e.search(gesture_decision(), {**evidence(), 'output_scope': scope, 'output_scope_source': source})
    assert len(e.hand_skin_gate.client.calls) == 1
    assert out['status'] == 'error' and out['selected'] == []
    assert e.generator.calls == 0


def test_head_and_body_requests_do_not_mutate_a_shared_gate(tmp_path):
    e = engine()
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    assert e.search(gesture_decision(), head_evidence())['status'] == 'ok'
    body = e.search(decision(), evidence())
    assert body['status'] == 'error' and not body['selected']
    assert len(e.hand_skin_gate.client.calls) == 1


def head_pair():
    e, state, d, required, skin = mirror_fixture()
    state.unsafe = True
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin, head_only=True)
    assert not hasattr(state, 'screened')
    data = report()
    data.update(selected=[hit('left', hand_skin_screen=head_hand_omission())],
        mirror_references=[row], mirror_comparison=comparison, hand_skin_screen=head_hand_summary())
    return data


def head_present(data, scope='head'):
    return build_pose_presentation(data, output_scope=scope, resolve_library=lambda pid:
        PreviewAssetOut(transport='library_bvh', bvh_url='/pose/'+pid+'/bvh',
            bvh_sha256=hashlib.sha256(pid.encode()).hexdigest(), library_lookup_supported=True))


def test_head_mirror_keeps_asset_and_condition_checks_without_claiming_skin_clear():
    data = head_pair()
    before = deepcopy(data)
    out = head_present(data)
    assert data == before
    [mirror] = out.review_candidates
    assert mirror.kind == 'mirror_reference'
    assert mirror.validation['hand_skin_screen'] == head_hand_omission()
    assert mirror.mirror_comparison.original_candidate_id == out.library_candidates[0].candidate_id
    assert not mirror.export_allowed and not mirror.refine_allowed and not mirror.auto_apply_allowed
    assert not out.export_allowed and not out.refine_allowed


@pytest.mark.parametrize('change', ['scope_full', 'scope_half', 'scope_bust', 'scope_shoulder_head',
    'summary_missing', 'summary_fake_clear', 'main_fake_clear', 'mirror_fake_clear', 'both_fake_clear',
    'source_hash', 'camera', 'source_not_exact', 'release', 'refine', 'actual_mesh'])
def test_head_marker_never_bypasses_other_scope_binding_or_permission_checks(change):
    data = head_pair()
    scope = change[6:] if change.startswith('scope_') else 'head'
    row = data['mirror_references'][0]
    if change == 'summary_missing': data.pop('hand_skin_screen')
    if change == 'summary_fake_clear': data['hand_skin_screen']['status'] = 'clear_screen'
    if change == 'main_fake_clear': data['selected'][0]['hand_skin_screen']['status'] = 'clear_screen'
    if change == 'mirror_fake_clear': row['hand_skin_screen']['status'] = 'clear_screen'
    if change == 'both_fake_clear':
        data['selected'][0]['hand_skin_screen']['status'] = row['hand_skin_screen']['status'] = 'clear_screen'
    if change == 'source_hash': row['source_bvh_sha256'] = 'bad'
    if change == 'camera': row['camera'] = {'yaw': 45.}
    if change == 'source_not_exact': data['selected'][0]['not_exact_match'] = True
    if change in ('release', 'refine', 'actual_mesh'):
        row[{'release':'release_allowed', 'refine':'refine_allowed', 'actual_mesh':'actual_mesh_validated'}[change]] = True
    out = head_present(data, scope)
    assert not out.review_candidates
    assert out.library_candidates and not out.export_allowed


def test_head_mirror_still_rejects_changed_family_hash_and_conditions():
    for changed in ('hash', 'family', 'conditions'):
        e, state, d, required, _ = mirror_fixture()
        if changed == 'hash': e.camera.hashes['right'] = 'bad'
        if changed == 'family': e.semantic.core_records['right']['meta']['pose_family_id'] = 'other'
        if changed == 'conditions': state.bad_conditions = True
        row, _ = mirror_reference(e, [hit('left')], d, required, None, head_only=True)
        assert row is None


def test_head_generation_omits_skin_without_granting_generation_permissions(tmp_path):
    e = engine(Camera(()), Semantic(()))
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = e.search(gesture_decision(), head_evidence())
    assert not e.hand_skin_gate.client.calls
    assert not out['selected']
    assert out['generation']['status'] == 'preview_ready'
    assert out['generation']['release_allowed'] is False
    assert out['generation']['refine_allowed'] is False
    assert out['generation']['actual_mesh_validated'] is False
    assert out['generation']['candidates'][0]['hand_skin_screen'] == head_hand_omission()


def test_head_geometry_fallback_stays_an_unqualified_review_reference(tmp_path):
    from tests.test_integrated_recovery import constrained_scene
    e, d, ev, _ = constrained_scene()
    d.semantic.description['visible_scope'] = 'full'
    e.generator = None
    e.hand_skin_gate = gate(tmp_path, ['unavailable'], ids=('a',))
    out = e.search(d, {**ev, 'output_scope': 'head', 'output_scope_source': 'vlm_person'}, review_geometry=True)
    assert not e.hand_skin_gate.client.calls
    assert out['status'] == 'ok'
    [row] = out['selected']
    assert row['match_level'] == 'geometry_fallback'
    assert row['review_required'] and row['not_exact_match']
    assert row['constraints_satisfied'] is False
    assert row['release_allowed'] is False and row['refine_allowed'] is False
    assert row['actual_mesh_validated'] is False


def test_na_and_false_permissions_survive_api_serialization(tmp_path):
    from api.models import DynamicSearchOut
    e = engine()
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = DynamicSearchOut.model_validate(e.search(gesture_decision(), head_evidence())).model_dump()
    assert out['hand_skin_screen'] == head_hand_summary()
    for row in out['selected']:
        assert row['hand_skin_screen'] == head_hand_omission()
        assert not row['actual_mesh_validated'] and not row['refine_allowed'] and not row['release_allowed']
