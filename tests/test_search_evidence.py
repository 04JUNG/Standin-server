from io import BytesIO
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api.app import STATE, app
from src.camera_search import Camera, CameraIndex, project
from src.fullbody_routing import FullBodySearch
from src.pipeline import CFG, Pipeline
from src.schema import Action, BBox, Relationship, Shot, Skeleton, View, VLMAnalysis
from src.search_evidence import SearchEvidence
from src.search_routing import DescribeOutcome, PoseDescription


class SearchEvidenceTest(TestCase):
    def test_fresh_analyze_reranks_using_contact_and_calls_description_once(self):
        joints = np.asarray([
            [0, 2.18, 0], [.10, 2.20, 0], [-.10, 2.20, 0], [.15, 2.17, 0], [-.15, 2.17, 0],
            [.35, 1.80, 0], [-.35, 1.80, 0], [.67, 1.37, 0], [-.67, 1.37, 0],
            [.31, 1.20, 0], [-.31, 1.20, 0], [.23, 1.10, 0], [-.23, 1.10, 0],
            [.27, .61, 0], [-.27, .61, 0], [.27, .12, 0], [-.27, .12, 0],
        ], dtype=np.float32)
        ids = [f'new_pose_{i}' for i in range(6)]
        # All have the same COCO joints. Only the last original BVH has the
        # measured contact specified by the newly analyzed description.
        camera = CameraIndex(ids, np.repeat(joints[None], len(ids), axis=0),
                             np.ones((len(ids), 17)), hand_contacts={ids[-1]: {'both_waist': 1.}})
        points = project(joints, Camera(0))[0] * 170 + [300, 400]
        entries = [SimpleNamespace(pose_id=pid, meta={}) for pid in ids]
        semantic = SimpleNamespace(service=SimpleNamespace(index=SimpleNamespace(docs=[])))
        search_evidence = SearchEvidence(camera, semantic)
        screen = Mock(side_effect=lambda eligible, *, limit: eligible[:limit])
        session = SimpleNamespace(filter=screen, report=lambda: {}, optional_phase=nullcontext, unverified=[])
        gate = SimpleNamespace(session=lambda _: session)
        engine = FullBodySearch(entries, semantic, camera, search_evidence=search_evidence, hand_skin_gate=gate)
        engine.semantic_search = Mock(return_value=dict(status='ok', hits=[
            dict(pose_id=pid, score=.7, matched={}) for pid in ids]))
        pose = SimpleNamespace(self_detecting=True,
            estimate=Mock(return_value=[Skeleton(points.astype(np.float32), np.full(17, .9, dtype=np.float32))]),
            estimate_crop_candidates=Mock(return_value=[]))
        vlm = SimpleNamespace(
            analyze=Mock(return_value=VLMAnalysis(
                1, Shot.FULL_HALF, Action.STANDING, View.FRONT, Relationship.SOLO,
                approx_boxes=[BBox(105, 105, 495, 665)], lower_body_visible=[True])),
            describe_people=Mock(return_value=DescribeOutcome('ok', {0: PoseDescription(
                actions=['pose_for_display'], body_details='양손 허리 · 팔꿈치 벌리기',
                search_text='양손을 허리에 대고 서 있는 자세다.', certainty='medium',
                visible_scope='full')})),
        )
        pipeline = Pipeline([], vlm_client=vlm, pose_model=pose, fullbody_search=engine,
                            skeleton_repairer=False)
        image = BytesIO()
        Image.new('RGB', (600, 700)).save(image, format='PNG')
        with patch.dict(STATE, {'pipeline': pipeline}), patch.dict(
            'os.environ', {'PARALLEL_POSE_DESCRIPTION': '0'},
        ), patch.multiple(CFG, vlm_hand_shapes=False, vlm_person_tags=False, body_matching_mode='off'):
            with patch('src.fullbody_routing.excluded_pose_ids', return_value=set()), patch(
                'api.app._library_preview_asset', side_effect=lambda pid: dict(
                    transport='library_bvh', bvh_url=f'/pose/{pid}/bvh', bvh_sha256='a' * 64,
                    library_lookup_supported=True),
            ):
                response = TestClient(app).post('/analyze', files={'file': ('fresh.png', image.getvalue(), 'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()['people'][0]['dynamic_search']
        self.assertEqual(report['search_evidence_rerank'], 'both_waist')
        self.assertEqual([row['pose_id'] for row in report['selected']], ids[:4] + [ids[-1]])
        self.assertEqual(screen.call_args.args[0][4]['pose_id'], ids[-1])
        self.assertEqual(response.json()['people'][0]['pose_presentation']['library_candidates'][-1]['pose_id'], ids[-1])
        vlm.describe_people.assert_called_once()

    def test_partial_geometry_reserves_candidates_without_exceeding_80(self):
        search = object.__new__(SearchEvidence)
        hits = [{'pose_id': f'pose_{i}'} for i in range(160)]
        search.projection_costs = Mock(return_value={'pose_99': .08, 'pose_4': .09})
        result = search.geometry_shortlist(hits, {'lower_body_observed': False})
        self.assertEqual(len(result), 80)
        self.assertEqual(result[-1]['pose_id'], 'pose_99')
        self.assertEqual(result[0], hits[0])
