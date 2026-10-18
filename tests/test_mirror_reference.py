"""Real-library reference selection is bounded and never ignores safety checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.mirror_reference import mirror_reference, one_observed_side
from src.search_routing import PoseDescription
from tests.test_pose_presentation import hit, report, present


def fixture():
    ids = ['left', 'right', 'other']
    meta = {pid: {'bvh_sha256': pid * 10, 'pose_family_id': 'pair' if pid != 'other' else 'other',
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
    data['selected'] = [hit(pid) for pid in ('left', 'a', 'b', 'c')]
    data['mirror_references'] = [row]
    data['mirror_comparison'] = comparison
    data['near_candidates'] = [hit('ignored')]
    p = present(data)
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
