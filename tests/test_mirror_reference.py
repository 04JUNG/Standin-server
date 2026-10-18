"""Real-library reference selection is bounded and never ignores safety checks."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from src.mirror_reference import mirror_reference, one_observed_side
from src.search_routing import PoseDescription
from tests.test_pose_presentation import hit, report, present
from api.pose_presentation import PreviewAssetOut, build_pose_presentation


def fixture():
    ids = ['left', 'right', 'other']
    meta = {pid: {'bvh_sha256': hashlib.sha256(pid.encode()).hexdigest(), 'pose_family_id': 'pair' if pid != 'other' else 'other',
                  'is_mirror': pid == 'right'} for pid in ids}
    state = SimpleNamespace(unsafe=False, bad_conditions=False, calls=[])
    class Validator:
        def check(self, required, pid):
            state.calls.append((deepcopy(required), pid))
            wanted = required['arms']
            actual = {'left': {'left': 'raised'}, 'right': {'right': 'raised'}, 'other': {}}[pid]
            return dict(status='passed' if wanted == actual else 'conflict', checks={}, conflicts=[])
    class Skin:
        unverified = []
        def filter(self, rows, limit):
            state.screened = rows[0]['pose_id']
            return [] if state.unsafe else [{**r, 'hand_skin_screen': {'status': 'clear_screen'}} for r in rows]
    semantic = SimpleNamespace(core_records={pid: {'meta': m} for pid, m in meta.items()},
        body={pid: {'source_sha256': m['bvh_sha256']} for pid, m in meta.items()},
        service=SimpleNamespace(index=SimpleNamespace(lexicon=SimpleNamespace(labels={}))),
        conditions=lambda *args: {'conflicts': ['action'] if state.bad_conditions else [], 'relations': {}, 'object_state': 'unknown'})
    engine = SimpleNamespace(top_k=5, validator=Validator(), semantic=semantic,
        camera=SimpleNamespace(hashes={pid: m['bvh_sha256'] for pid, m in meta.items()}),
        allowed=set(ids), include_stored_compositions=True, meta=meta)
    desc = PoseDescription(arm_states={'left': 'raised', 'right': 'unknown'},
        search_text='왼팔을 들어 올림', body_details='왼팔이 올라가고 반대팔은 보이지 않음')
    required = dict(arms={'left': 'raised'}, object_hands={}, posture=None, body_states={}, hand_counts={}, active=True)
    return engine, state, desc, required, Skin()


def test_one_known_side_does_not_invent_the_hidden_arm():
    e, state, d, required, skin = fixture()
    before = deepcopy(d)
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    assert row['pose_id'] == 'right' and row['rank_score'] is None
    assert comparison['status'] == 'ready' and comparison['observed_side'] == 'left'
    assert row['mirror_comparison']['reference_side'] == 'right'
    assert row['review_required'] and row['not_exact_match'] and not row['constraints_satisfied']
    assert not row['release_allowed'] and not row['refine_allowed']
    assert d == before and required['arms'] == {'left': 'raised'}
    assert any(r['arms'] == {'right': 'raised'} for r, _ in state.calls)
    assert state.screened == 'right'


@pytest.mark.parametrize('failure', ['quarantine', 'hash', 'family', 'same_side', 'condition', 'skin', 'no_gate'])
def test_missing_or_failed_mirror_is_not_faked(monkeypatch, failure):
    e, state, d, required, skin = fixture()
    if failure == 'quarantine': monkeypatch.setattr('src.semantic_adapter_285.excluded_pose_ids', lambda: {'right'})
    if failure == 'hash': e.camera.hashes['right'] = 'stale'
    if failure == 'family': e.semantic.core_records['right']['meta']['pose_family_id'] = 'unrelated'
    if failure == 'same_side': e.semantic.core_records['right']['meta']['is_mirror'] = False
    if failure == 'condition': state.bad_conditions = True
    if failure == 'skin': state.unsafe = True
    if failure == 'no_gate': skin = None
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    assert row is None and comparison['status'] == 'unavailable'


def test_bilateral_and_unknown_descriptions_do_not_claim_one_arm():
    assert one_observed_side(PoseDescription(arm_states={'left': 'raised', 'right': 'lowered'})) is None
    assert one_observed_side(PoseDescription(arm_states={'left': 'unknown', 'right': 'unknown'})) is None


def test_presentation_pair_uses_real_assets_and_five_total_without_export():
    e, _, d, required, skin = fixture()
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    data = report()
    data['selected'] = [hit(pid, hand_skin_screen={'status': 'clear_screen'}) for pid in ('left', 'a', 'b', 'c')]
    data['mirror_references'] = [row]
    data['mirror_comparison'] = comparison
    data['near_candidates'] = [hit('ignored')]
    p = build_pose_presentation(data, output_scope='half', resolve_library=lambda pid: PreviewAssetOut(transport='library_bvh', bvh_url='/pose/'+pid+'/bvh', bvh_sha256=hashlib.sha256(pid.encode()).hexdigest(), library_lookup_supported=True))
    assert len(p.library_candidates) + len(p.review_candidates) == 5
    reference = p.review_candidates[0]
    assert reference.kind == 'mirror_reference' and reference.pose_id == 'right'
    assert reference.mirror_comparison.original_candidate_id == p.library_candidates[0].candidate_id
    assert reference.asset.bvh_url == '/pose/right/bvh'
    assert reference.rank_score is None and reference.score_kind is None
    assert not reference.render_camera.projection_matches_search
    assert not reference.release_allowed and not reference.export_allowed and not reference.refine_allowed


def test_presentation_drops_orphaned_reference():
    e, _, d, required, skin = fixture()
    row, _ = mirror_reference(e, [hit('left')], d, required, skin)
    p = present(report(mirror_references=[row]))
    assert not p.review_candidates
    assert 'mirror_reference_preview_unavailable' in p.issues


def test_known_second_hand_prevents_one_hand_claim():
    assert one_observed_side(PoseDescription(arm_states={'left':'raised'},held_object_hands={'cup':'right'})) is None


def test_source_change_between_search_and_asset_projection_drops_reference():
    e, _, d, required, skin = fixture()
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    data = report(); data.update(selected=[hit('left')],mirror_references=[row],mirror_comparison=comparison)
    p = present(data)  # resolver now returns different bytes than the checked library hashes
    assert not p.review_candidates
    assert p.mirror_comparison['status']=='unavailable'


def test_engine_pair_occupies_one_of_five_slots_and_keeps_main_rank(monkeypatch):
    from tests.test_fullbody_routing import Camera, Semantic, decision, evidence
    from src.fullbody_routing import FullBodySearch
    from types import SimpleNamespace
    rows = ['a','b','c','d','e']
    engine=FullBodySearch([SimpleNamespace(pose_id=p,meta={}) for p in rows], Semantic(rows), Camera(rows))
    reference=hit('actual_mirror',review_required=True,not_exact_match=True)
    monkeypatch.setattr('src.mirror_reference.mirror_reference',lambda *a:(reference,{'status':'ready','reason':'opposite_hand_reference'}))
    out=engine.search(decision(False),evidence())
    assert len(out['selected'])==4 and len(out['mirror_references'])==1
    assert [r['pose_id'] for r in out['selected']]==rows[:4]
    assert out['selected'][0]['pose_id']=='a' and out['generation'] is None


@pytest.mark.parametrize('extra', [
    {'arms': {'left': 'raised', 'right': 'lowered'}},
    {'object_hands': {'cup': 'right'}},
    {'object_hands': {'cup': 'both'}},
    {'hand_counts': {'cup': 'two'}},
])
def test_required_conditions_cannot_hide_second_observed_hand(extra):
    e, _, d, required, skin = fixture()
    required.update(extra)
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    assert row is None and comparison['status'] == 'not_requested'


@pytest.mark.parametrize('flag,value', [
    ('review_required', True), ('not_exact_match', True), ('constraints_satisfied', False),
])
def test_review_or_unsatisfied_original_never_becomes_reference_source(flag, value):
    e, _, d, required, skin = fixture()
    row, comparison = mirror_reference(e, [hit('left', **{flag: value})], d, required, skin)
    assert row is None and comparison['status'] == 'unavailable'


def pair_report():
    e, _, d, required, skin = fixture()
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    data = report(mirror_references=[row], mirror_comparison=comparison)
    data['selected'] = [hit('left', hand_skin_screen={'status': 'clear_screen'})]
    return data


def pair_presentation(data):
    return build_pose_presentation(data, output_scope='half', resolve_library=lambda pid:
        PreviewAssetOut(transport='library_bvh', bvh_url='/pose/'+pid+'/bvh',
            bvh_sha256=hashlib.sha256(pid.encode()).hexdigest(), library_lookup_supported=True))


@pytest.mark.parametrize('change', ['same_pose', 'bool_string', 'rank_score', 'score', 'score_kind',
    'exact', 'satisfied', 'review_missing', 'camera', 'source_not_exact', 'source_fifth', 'mirror_skin', 'main_skin'])
def test_api_rejects_malformed_reference_without_failing_verified_main(change):
    data = pair_report()
    row = data['mirror_references'][0]
    if change == 'same_pose':
        row['pose_id'] = 'left'
        row['source_bvh_sha256'] = hashlib.sha256(b'left').hexdigest()
    elif change == 'bool_string': row['mirror_comparison']['is_mirror'] = 'true'
    elif change in ('rank_score', 'score'): row[change] = 0.9
    elif change == 'score_kind': row['score_kind'] = 'weighted_rrf'
    elif change == 'exact': row['not_exact_match'] = False
    elif change == 'satisfied': row['constraints_satisfied'] = True
    elif change == 'review_missing': row.pop('review_required')
    elif change == 'camera': row['camera'] = {'yaw': 45.}
    elif change == 'mirror_skin': row['hand_skin_screen']['status'] = 'unavailable'
    elif change == 'main_skin': data['selected'][0]['hand_skin_screen']['status'] = 'unavailable'
    elif change == 'source_not_exact': data['selected'][0]['not_exact_match'] = True
    elif change == 'source_fifth': data['selected'] = [hit(p) for p in ('a', 'b', 'c', 'd', 'left')]
    out = pair_presentation(data)
    assert not out.review_candidates and out.library_candidates
    assert out.status == 'ready' and out.mirror_comparison['status'] == 'unavailable'
    assert 'mirror_reference_preview_unavailable' in out.issues


def test_api_reserves_one_slot_even_with_extra_review_near_and_framing_rows():
    data = pair_report()
    data['selected'] += [hit('a'), hit('b'), hit('c', review_required=True), hit('d')]
    data['near_candidates'] = [hit('near'+str(i)) for i in range(5)]
    data['framing_references'] = [hit('framing'+str(i)) for i in range(5)]
    out = pair_presentation(data)
    assert len(out.library_candidates) + len(out.review_candidates) == 5
    assert [c.pose_id for c in out.library_candidates] == ['left', 'a', 'b']
    assert [c.kind for c in out.review_candidates] == ['geometry_review', 'mirror_reference']


def test_optional_skin_unavailable_preserves_main_receipt_and_mandatory_failure(tmp_path, monkeypatch):
    from tests.test_fullbody_routing import engine, decision, evidence
    from tests.test_hand_skin_gate import gate
    e = engine()
    e.hand_skin_gate = gate(tmp_path)
    def unavailable(engine, selected, description, required, session):
        session.unverified.append({'pose_id': 'optional', 'hand_skin_screen': {'status': 'unavailable'}})
        return None, {'status': 'unavailable', 'reason': 'mirror_hand_skin_not_clear'}
    monkeypatch.setattr('src.mirror_reference.mirror_reference', unavailable)
    out = e.search(decision(), evidence())
    assert out['status'] == 'ok' and len(out['selected']) == 3
    assert not out['hand_skin_screen']['unverified']
    assert out['mirror_hand_skin_screen']['unverified'][0]['pose_id'] == 'optional'
    e.hand_skin_gate = gate(tmp_path, ['unavailable'])
    out = e.search(decision(), evidence())
    assert out['status'] == 'error' and not out['selected']
    assert out['reason'] == 'hand_skin_screen_unavailable_no_generation'
    assert 'mirror_hand_skin_screen' not in out


def test_already_selected_opposite_asset_is_not_demoted_from_main_results():
    e, _, d, required, skin = fixture()
    row, comparison = mirror_reference(e, [hit('left'), hit('right')], d, required, skin)
    assert row is None and comparison['status'] == 'unavailable'


def test_large_internal_top_k_still_uses_first_four_sources_only():
    e, _, d, required, skin = fixture()
    e.top_k = 20
    row, comparison = mirror_reference(e, [hit('other')]*4 + [hit('left')], d, required, skin)
    assert row is None and comparison['status'] == 'unavailable'


@pytest.mark.parametrize('core', [None,
    {'conflicts': [], 'relations': {'held_hand:cup': {'status': 'unknown'}}, 'object_state': 'match'},
    {'conflicts': [], 'relations': {}, 'object_state': 'unknown'}])
def test_missing_semantic_condition_evidence_never_passes_reference(core):
    e, _, d, required, skin = fixture()
    d.objects = ['cup']
    d.held_object_hands = {'cup': 'left'}
    e.semantic.service.index.lexicon.labels[('object', 'cup')] = '컵'
    e.semantic.conditions = lambda *args: core
    row, comparison = mirror_reference(e, [hit('left')], d, required, skin)
    assert row is None and comparison['status'] == 'unavailable'
