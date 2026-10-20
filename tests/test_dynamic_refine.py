from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from api.models import CandidateOut
from src.camera_search import camera_grid
from src.config import CFG
from src.descriptor import build_slot_descriptors
from src.dynamic_refine import dynamic_refine_candidates
from src.pipeline import Pipeline
from src.refine_projection import query_sha256
from src.schema import Action, Relationship, Shot, VLMAnalysis, View
from src.skeleton_extraction import PersonSlot, analyze_skeleton
from tests.test_skeleton_extraction import _skeleton


def fixture(monkeypatch):
    monkeypatch.setattr(CFG, 'refine_enabled', True)
    monkeypatch.setattr(CFG, 'refine_v2_enabled', True)
    sk = _skeleton()
    ev = analyze_skeleton(sk)
    slot = PersonSlot(0, 'vlm', skeleton=sk, evidence=ev, state=ev.state,
                      skeleton_source='full_image', lower_body_observed=True)
    vlm = VLMAnalysis(1, Shot.FULL_HALF, Action.STANDING, View.FRONT, Relationship.SOLO)
    desc = build_slot_descriptors(vlm, [slot])[0]
    cameras = camera_grid()
    camera = asdict(cameras[0])
    index = SimpleNamespace(cameras=cameras, hashes={'pose': 'a' * 64})
    hit = dict(kind='library_original', pose_id='pose', camera=camera,
        evidence={'geometry': dict(pose_id='pose', camera=camera, camera_index=0, distance=.2)})
    return desc, slot, dict(status='ok', selected=[hit]), index


def test_qualified_dynamic_hit_keeps_original_search_projection_and_masked_scores(monkeypatch):
    desc, slot, report, index = fixture(monkeypatch)
    before = deepcopy(report)
    bindings = dynamic_refine_candidates(desc, slot, report, index)
    assert report == before
    binding = CandidateOut(**bindings[0])
    assert binding.pose_id == 'pose'
    assert binding.camera.version == 'candidate-camera-v2'
    assert binding.camera.source_bvh_sha256 == 'a' * 64
    assert binding.camera.query_sha256 == query_sha256(desc.skeleton.keypoints)
    assert binding.camera.observed_joint_mask == slot.evidence.valid_joint_mask.tolist()
    Pipeline._apply_refine_policy(None, desc, slot, 'low', has_candidates=True)
    assert desc.refine_allowed and desc.refinable_limbs
    np.testing.assert_array_equal(desc.skeleton.scores, slot.evidence.refine_scores)


@pytest.mark.parametrize('patch', [
    {'kind': 'framing_reference'}, {'kind': 'generated_review'},
    {'review_required': True}, {'constraints_satisfied': False},
    {'match_level': 'geometry_fallback'}, {'match_level': 'near_match'},
    {'camera': {'yaw': 99}}, {'pose_id': 'missing'}, {'evidence': {}},
])
def test_review_or_unbound_hits_never_get_refine_bindings(monkeypatch, patch):
    desc, slot, report, index = fixture(monkeypatch)
    report['selected'][0].update(patch)
    assert dynamic_refine_candidates(desc, slot, report, index) == []


@pytest.mark.parametrize('source', ['vlm_repair', 'crop_retry', 'fallback_full_image', 'none'])
def test_existing_structural_source_policy_still_blocks_refine(monkeypatch, source):
    desc, slot, report, index = fixture(monkeypatch)
    slot.skeleton_source = source
    Pipeline._apply_refine_policy(None, desc, slot, 'low', has_candidates=True)
    assert not desc.refine_allowed
    assert not desc.skeleton.scores.any()


def test_unobserved_lower_body_stays_frozen(monkeypatch):
    desc, slot, report, index = fixture(monkeypatch)
    desc.lower_body_observed = slot.lower_body_observed = False
    Pipeline._apply_refine_policy(None, desc, slot, 'low', has_candidates=True)
    assert desc.refine_allowed
    assert set(desc.refinable_limbs) == {'left_arm', 'right_arm'}


@pytest.mark.parametrize('flag', ['refine_enabled', 'refine_v2_enabled'])
def test_explicit_kill_switches_keep_dynamic_refine_disabled(monkeypatch, flag):
    desc, slot, report, index = fixture(monkeypatch)
    monkeypatch.setattr(CFG, flag, False)
    assert dynamic_refine_candidates(desc, slot, report, index) == []


def test_missing_anchors_and_semantic_only_hits_stay_disabled(monkeypatch):
    desc, slot, report, index = fixture(monkeypatch)
    slot.evidence.valid_joint_mask[11] = False
    assert dynamic_refine_candidates(desc, slot, report, index) == []
    slot.evidence.valid_joint_mask[11] = True
    report['selected'][0]['evidence'] = {'semantic': {'score': 1.}}
    assert dynamic_refine_candidates(desc, slot, report, index) == []


def test_real_pipeline_and_http_preserve_refine_context_without_promoting_viewer_permissions(monkeypatch):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    import api.app as api
    from api.pose_presentation import PreviewAssetOut
    from src.fullbody_routing import FullBodySearch
    from src.library import build_synthetic_index
    from src.search_routing import DescribeOutcome, PoseDescription
    from tests.test_fullbody_routing import Semantic, decision
    from tests.test_search_routing import _Img, _RoughPose, _CountingVLM
    fixture(monkeypatch)
    monkeypatch.setattr(CFG, 'vlm_hand_shapes', False)
    entries = build_synthetic_index(); pid = entries[0].pose_id
    class Index:
        cameras = camera_grid()
        hashes = {pid: 'a' * 64}
        def search(self, *args):
            return [dict(pose_id=pid, distance=.2, camera=asdict(self.cameras[0]), camera_index=0)]
    vlm = _CountingVLM()
    vlm.describe_people = lambda image, boxes: DescribeOutcome('ok', descriptions={
        i: PoseDescription(**decision().semantic.description) for i in range(len(boxes))})
    pipe = Pipeline(entries, vlm_client=vlm, pose_model=_RoughPose(),
                    fullbody_search=FullBodySearch(entries, Semantic((pid,)), Index()))
    result = pipe.process_cut(_Img('full_half standing front'))
    assert result.descriptors[0].refine_allowed
    assert result.descriptors[0].skeleton.scores.any()
    monkeypatch.setitem(api.STATE, 'pipeline', SimpleNamespace(process_cut=lambda *a, **k: result))
    monkeypatch.setattr(api, '_library_preview_asset', lambda pose: PreviewAssetOut(
        transport='library_bvh', bvh_url=f'/pose/{pose}/bvh', bvh_sha256='a'*64, library_lookup_supported=True))
    stream = io.BytesIO(); Image.new('RGB', (128,128)).save(stream, format='PNG')
    response = TestClient(api.app).post('/analyze', files={'file': ('rough.png', stream.getvalue(), 'image/png')})
    assert response.status_code == 200, response.text
    person = response.json()['people'][0]
    assert person['refine_allowed'] and person['scores'] == result.descriptors[0].skeleton.scores.tolist()
    assert person['candidates'] == []
    assert person['refine_candidates'][0]['pose_id'] == pid
    assert person['refine_candidates'][0]['camera']['version'] == 'candidate-camera-v2'
    assert not person['pose_presentation']['library_candidates'][0]['refine_allowed']
