"""Runtime wiring, frozen-policy parity and fail-safe search contracts."""
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from src.config import CFG, Config
from src.library import _synthetic_pose, build_entries_from_pose, build_synthetic_index
from src.pipeline import Pipeline
from src.perspective_search import PerspectiveSearchIndex, _Pose
from src.schema import (Action, BBox, PersonDescriptor, Relationship, Shot,
                        Skeleton, VLMAnalysis, View)
from src import search_geometry as runtime


def gallery(names=('a', 'b', 'c')):
    poses = []
    for i, name in enumerate(names):
        points = _synthetic_pose(('standing', 'sitting', 'reaching')[i % 3])
        entries = build_entries_from_pose(name, points, {}, name+'.bvh')
        poses.append(_Pose(entries[0], points, tuple(entries)))
    return PerspectiveSearchIndex(poses)


def observations():
    points = _synthetic_pose('standing')[:, :2] * [120, -120] + [256, 330]
    return points.astype(np.float32), np.ones(17, bool), np.full((768, 512, 3), 255, np.uint8)


def descriptor():
    xy, mask, _ = observations()
    from src.features import normalize_skeleton
    skel = Skeleton(xy, np.full(17, .9, np.float32))
    return PersonDescriptor(Shot.FULL_HALF, Action.OTHER, View.FRONT,
                            Relationship.SOLO, skel, normalize_skeleton(xy),
                            valid_joint_mask=mask)


def test_runtime_geometry_equals_frozen_research_policy():
    from standin_eval import perspective_search as frozen
    from standin_eval import camera_refinement as fitting
    index = gallery()
    points = np.stack([p.joints for p in index.poses])
    cameras = frozen.camera_grid()['yaw_pitch_perspective_117'] + [
        frozen.Camera(c.yaw, c.pitch, d)
        for c in frozen.camera_grid()['yaw_pitch_39'] for d in (2., 3.)]
    assert [asdict(c) for c in index.cameras] == [asdict(c) for c in cameras]
    expected = frozen.build_camera_index(index.poses, cameras)
    for actual, target in zip(index.index, expected):
        assert np.array_equal(actual, target) and not actual.flags.writeable
    xy, mask, _ = observations()
    q = frozen.normalize_batch(xy)
    for query in (q, q[runtime.LEFT_RIGHT]):
        a, ap = runtime.refine_costs(index.points, index.index, index.cameras, query, mask)
        b, bp = fitting.refine_costs(fitting.normalized_joints(points), expected, cameras, query, mask)
        assert np.array_equal(a, b) and np.array_equal(ap, bp)


def test_runtime_surface_and_contour_equal_frozen_research():
    from src.search_outline import render_pose
    from src.search_contour import prepare_letterboxed_query, candidate_evidence
    from scripts.experiment_z_rendered import render_pose as frozen
    from standin_eval.contour_letterbox import prepare_letterboxed_query as old_query
    from standin_eval.contour_rerank import candidate_evidence as old_evidence
    points = _synthetic_pose('sitting')
    for pitch, distance in ((0, None), (30, 4.), (-30, 8.)):
        a, xy = render_pose(points, 45, size=256, style='rough', seed=0, pitch=pitch, distance=distance)
        b, by = frozen(points, 45, size=256, style='rough', seed=0, pitch=pitch, distance=distance)
        assert np.array_equal(a, b) and np.array_equal(xy, by)
        query, _ = prepare_letterboxed_query(np.asarray(a), xy)
        oq, _ = old_query(np.asarray(b), by)
        params = [45, pitch, 0 if distance is None else 1/distance]
        value, affine, mask = candidate_evidence(query, points, params)
        ov, oa, om = old_evidence(oq, points, params)
        assert value == ov and np.array_equal(mask, om) and np.array_equal(affine, oa)


def controlled_search(index, penalties, failure=None, quarantine=()):
    xy, mask, image = observations()
    n = len(index.poses)-len(quarantine)
    values = np.arange(n)*.1+.1
    calls = iter(penalties)
    def evidence(*args):
        value = next(calls)
        if value == failure:
            raise ValueError('surface intersects camera')
        return value, None, None
    with patch('src.perspective_search.score_camera_index', return_value=values), \
         patch('src.perspective_search.refine_costs', return_value=(values, np.zeros((n,3)))), \
         patch('src.search_contour.prepare_letterboxed_query', return_value=({}, {})), \
         patch('src.search_contour.candidate_evidence', side_effect=evidence):
        return index.search(xy, mask, image, top_k=3, quarantined_pose_ids=quarantine)


def test_first_candidate_can_move_by_fair_contour_cost():
    result, trace = controlled_search(gallery(), [.5, 0., 0.])
    assert [c.pose_id for c in result] == ['b', 'c', 'a']
    assert trace['contour'] == 'applied' and trace['coarse_combinations'] == 117
    assert trace['refinement_combinations'] == 195


def test_one_failed_surface_discards_all_partial_contours():
    result, trace = controlled_search(gallery(), [.5, 0., -1.], failure=-1.)
    assert [c.pose_id for c in result] == ['a', 'b', 'c']
    assert trace['contour'] == 'camera_only_fallback'
    assert np.allclose([c.distance for c in result], [.1, .2, .3])


def test_quarantine_before_shortlisting_and_family_dedup_backfill():
    result, trace = controlled_search(gallery(('bad','a','a_mirror','b')), [0., 0., 0.], quarantine=('bad',))
    assert [c.pose_id for c in result] == ['a', 'b']
    assert trace['gallery_poses'] == 3


def test_query_and_index_remain_unchanged_on_real_numeric_search():
    index = gallery()
    xy, mask, image = observations()
    before = [a.copy() for a in index.index]
    old_xy, old_mask = xy.copy(), mask.copy()
    result, trace = index.search(xy, mask, image, top_k=2)
    assert len(result) == 2 and trace['contour'] == 'camera_only_fallback'  # blank image
    assert np.array_equal(xy, old_xy) and np.array_equal(mask, old_mask)
    assert all(np.array_equal(a,b) for a,b in zip(index.index,before))


def test_partial_torso_is_not_invented_for_runtime_search():
    xy, mask, image = observations(); mask[11] = False
    with pytest.raises(ValueError, match='observed torso'):
        gallery().search(xy, mask, image)


def test_bvh_snapshot_validation_is_all_or_nothing():
    points = _synthetic_pose('standing')
    entries = build_entries_from_pose('a', points, {}, 'a.bvh')
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('src.perspective_search.load_coco17', return_value=(points, np.ones(17))):
        assert len(PerspectiveSearchIndex.build(entries).poses) == 1
        entries[1].feature = entries[1].feature + 1
        with pytest.raises(ValueError, match='disagree'):
            PerspectiveSearchIndex.build(entries)


def test_legacy_mode_never_loads_bvh_index():
    with patch.object(CFG, 'pose_search_mode', 'legacy'), \
         patch.object(PerspectiveSearchIndex, 'build', side_effect=AssertionError('must not load')):
        pipe = Pipeline(build_synthetic_index())
    assert pipe.perspective_index is None and pipe.perspective_index_status == 'disabled'


def test_unavailable_index_preserves_original_candidates():
    with patch.object(CFG, 'pose_search_mode', 'perspective'):
        pipe = Pipeline(build_synthetic_index())
        desc = descriptor()
        result = pipe._search_one(desc, desc.skeleton, search_image=observations()[2])
        assert not desc.quality_trace['perspective_search']['applied']
        with patch.object(CFG, 'pose_search_mode', 'legacy'):
            old = pipe._search_one(desc, desc.skeleton, search_image=observations()[2])
    assert [(c.pose_id,c.distance) for c in result[0]] == [(c.pose_id,c.distance) for c in old[0]]


def test_new_search_routes_through_pipeline_and_remains_low_confidence():
    index = gallery()
    with patch.object(CFG, 'pose_search_mode', 'perspective'), \
         patch.object(PerspectiveSearchIndex, 'build', return_value=index):
        pipe = Pipeline(build_synthetic_index())
        desc = descriptor()
        candidates, confidence, _ = pipe._search_one(desc, desc.skeleton, search_image=observations()[2])
        assert candidates and confidence == 'low'
        assert desc.quality_trace['perspective_search']['applied']


def test_nonpos_partial_and_no_image_keep_legacy_path():
    index = gallery()
    with patch.object(CFG, 'pose_search_mode', 'perspective'), \
         patch.object(PerspectiveSearchIndex, 'build', return_value=index):
        pipe = Pipeline(build_synthetic_index())
        for coverage, metric, image in (('reduced','pos',observations()[2]),
                                        ('full','angle',observations()[2]), ('full','pos',None)):
            desc = descriptor(); desc.coverage_class = coverage
            with patch.object(index, 'search', side_effect=AssertionError('must not enter')):
                pipe._search_one(desc, desc.skeleton, distance_metric=metric, search_image=image)
            assert not desc.quality_trace['perspective_search']['applied']


def test_camera_failure_falls_back_to_complete_legacy_search():
    index = gallery()
    with patch.object(CFG, 'pose_search_mode', 'perspective'), \
         patch.object(PerspectiveSearchIndex, 'build', return_value=index), \
         patch.object(index, 'search', side_effect=ValueError('invalid camera')):
        pipe = Pipeline(build_synthetic_index()); desc = descriptor()
        candidates, _, _ = pipe._search_one(desc, desc.skeleton, search_image=observations()[2])
        assert candidates and desc.quality_trace['perspective_search'] == {
            'applied': False, 'reason': 'camera_search_failed'}


def test_full_cut_reaches_camera_search_but_refine_is_blocked_even_v2():
    xy, _, image = observations()
    class VLM:
        def analyze(self, *args):
            return VLMAnalysis(1, Shot.FULL_HALF, Action.OTHER, View.FRONT, Relationship.SOLO,
                               approx_boxes=[BBox(0,0,512,768)], lower_body_visible=[True])
    class Detector:
        def detect(self, *args): return [BBox(0,0,512,768)]
    class Pose:
        def estimate(self, *args): return [Skeleton(xy.copy(), np.full(17,.9))]
    with patch.object(CFG, 'pose_search_mode', 'perspective'), \
         patch.object(CFG, 'refine_v2_enabled', True), \
         patch.object(PerspectiveSearchIndex, 'build', return_value=gallery()):
        pipe = Pipeline(build_synthetic_index(), vlm_client=VLM(), detector=Detector(), pose_model=Pose())
        result = pipe.process_cut(image, 512, 768)
    assert result.candidates and result.person_confidence == ['low']
    desc = result.descriptors[0]
    assert desc.quality_trace['perspective_search']['applied']
    assert desc.confidence_threshold is None and not desc.refine_allowed
    assert desc.quality_trace['refine_policy_block'] == 'perspective_camera_unverified'
    assert not desc.skeleton.scores.any()


def test_search_mode_rejects_typo():
    with pytest.raises(ValueError, match='POSE_SEARCH_MODE'):
        Config(pose_search_mode='perspectve')


def test_runtime_has_no_dependency_on_research_scripts_or_z_models():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ('perspective_search','search_geometry','search_contour','search_outline'):
        tree = ast.parse((root/'src'/(name+'.py')).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('standin_eval','scripts','rtmlib'))


def test_two_requests_do_not_change_shared_index():
    from concurrent.futures import ThreadPoolExecutor
    index=gallery(); xy,mask,image=observations()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(lambda _: index.search(xy,mask,image)[0], range(2)))
    assert [(c.pose_id,c.distance) for c in results[0]] == [(c.pose_id,c.distance) for c in results[1]]
    assert all(not a.flags.writeable for a in index.index)


def test_multi_person_cut_never_uses_whole_cut_contour():
    xy, _, image=observations()
    class VLM:
        def analyze(self,*args):
            return VLMAnalysis(2,Shot.FULL_HALF,Action.OTHER,View.FRONT,Relationship.SOLO,
                               approx_boxes=[BBox(0,0,256,768),BBox(256,0,512,768)])
    class Detector:
        def detect(self,*args): return [BBox(0,0,256,768),BBox(256,0,512,768)]
    class Pose:
        def estimate(self,*args): return [Skeleton(xy*.5,np.full(17,.9)),Skeleton(xy*.5+[256,0],np.full(17,.9))]
        def estimate_crop_candidates(self,*args): return []
    index=gallery()
    with patch.object(CFG,'pose_search_mode','perspective'), \
         patch.object(PerspectiveSearchIndex,'build',return_value=index), \
         patch.object(index,'search',side_effect=AssertionError('multi-person contour must not run')):
        result=Pipeline(build_synthetic_index(),vlm_client=VLM(),detector=Detector(),pose_model=Pose()).process_cut(image,512,768)
    assert len(result.descriptors)==2
    assert all(not d.quality_trace.get('perspective_search',{}).get('applied') for d in result.descriptors)
