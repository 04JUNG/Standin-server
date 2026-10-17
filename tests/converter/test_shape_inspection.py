"""Regression for neutral/posed false negatives and false release approvals."""
import copy
import numpy as np
import pytest
from scripts.shape_inspection_metrics import (
    INSPECTION_SCHEMA, complete_pairs, inspect_geometry, require_inspection, pose_digest, pose_frame_agrees,
)


def fixture():
    # Six closed tetrahedra: pose-dependent deformation does not preserve the
    # neutral displacement ordering, even when the authored output is exact.
    tetra = np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    base = np.concatenate([tetra + [i*.2,0,0] for i in range(6)])
    faces = np.concatenate([np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])+i*4 for i in range(6)])
    neutral_change = np.linspace(.001,.012,len(base))
    posed_change = neutral_change[::-1] * 2
    reference = {'preset': base, 'slider': base + np.column_stack([posed_change,np.zeros((len(base),2))])}
    return reference, faces, neutral_change


def test_correct_posed_reference_passes_despite_neutral_correlation_failure():
    reference, faces, neutral = fixture()
    posed = np.linalg.norm(reference['slider']-reference['preset'], axis=1)
    assert np.corrcoef(neutral, posed)[0,1] < .8
    score = inspect_geometry(reference, reference, faces, .001)
    assert score['passed'] and score['magnitude_correlation'] > .9999


def test_dropped_slider_or_half_strength_fails_even_with_matching_direction():
    reference, faces, _ = fixture()
    for strength in [0, .5]:
        actual = {'preset': reference['preset'], 'slider': reference['preset']+strength*(reference['slider']-reference['preset'])}
        assert not inspect_geometry(actual, reference, faces, .001)['passed']


def test_local_geometry_corruption_fails_despite_near_perfect_global_correlation():
    reference, faces, _ = fixture();actual=copy.deepcopy(reference)
    actual['slider'][0,2]+=.0001  # 0.1 mm corruption at one vertex
    score=inspect_geometry(actual,reference,faces,.001)
    assert score['magnitude_correlation'] > .99
    assert not score['passed'] and score['max_surface_error_mm'] > .01


def test_joint_length_error_still_fails():
    ref,faces,_=fixture()
    assert not inspect_geometry(ref,ref,faces,.501)['passed']


@pytest.mark.parametrize('change',['nan','missing_vertex','bad_face','empty','bad_joint'])
def test_invalid_or_unmeasurable_geometry_is_rejected(change):
    ref,faces,_=fixture();actual=copy.deepcopy(ref);joint=.001
    if change=='nan':actual['slider'][0,0]=np.nan
    if change=='missing_vertex':actual['slider']=actual['slider'][:-1]
    if change=='bad_face':faces[0,0]=1000
    if change=='empty':actual['slider']=np.empty((0,3))
    if change=='bad_joint':joint=np.inf
    with pytest.raises(ValueError):inspect_geometry(actual,ref,faces,joint)


def reports():
    runs=[{'case':'female-base','bvh':'pose.bvh','label':label,'id':label,'ok':True,'fbx_sha256':digest} for label,digest in [('preset','a'*64),('slider','b'*64)]]
    pose={'schema':'shape-authoring-pose.v1','space':'armature-local','bones':{'Hips':np.eye(4).tolist()},'channels':{'Hips':{'rotation_mode':'QUATERNION','location':[0,0,0],'scale':[1,1,1],'rotation_quaternion':[1,0,0,0],'rotation_euler':[0,0,0],'rotation_axis_angle':[0,0,1,0]}}}
    for r in runs:r['authoring_pose']=copy.deepcopy(pose)
    inspection=[{'reference':'pinned-controller-at-verified-native-pose.v1','pose_sha256':{'preset':pose_digest(pose),'slider':pose_digest(pose)},'schema':INSPECTION_SCHEMA,'case':'female-base','bvh':'pose.bvh','passed':True,
        'artifact_sha256':{'preset':'a'*64,'slider':'b'*64},'joints':{'max_error_mm':.001},
        'shape_effect':{'max_surface_error_mm':.002,'magnitude_correlation':1.}}]
    return runs,inspection


def test_report_binds_current_complete_pair_and_exact_fbx_hashes():
    runs,inspection=reports();require_inspection(runs,inspection)
    inspection[0]['artifact_sha256']['slider']='c'*64
    with pytest.raises(ValueError,match='another FBX'):require_inspection(runs,inspection)


@pytest.mark.parametrize('change',['empty','old','failed','partial','bad_surface','bad_joint','bad_corr','nan','duplicate'])
def test_release_gate_rejects_invalid_or_stale_inspections(change):
    runs,inspection=reports()
    if change=='empty':inspection=[]
    if change=='old':inspection[0]['schema']='shape-inspection.v1'
    if change=='failed':inspection[0]['passed']=False
    if change=='partial':runs+=[dict(r,case='female-other') for r in runs]
    if change=='bad_surface':inspection[0]['shape_effect']['max_surface_error_mm']=.011
    if change=='bad_joint':inspection[0]['joints']['max_error_mm']=.501
    if change=='bad_corr':inspection[0]['shape_effect']['magnitude_correlation']=.79
    if change=='nan':inspection[0]['shape_effect']['max_surface_error_mm']=float('nan')
    if change=='duplicate':inspection*=2
    with pytest.raises(ValueError):require_inspection(runs,inspection)


@pytest.mark.parametrize('change',['empty','missing','failed','duplicate'])
def test_incomplete_or_failed_bake_pairs_cannot_disappear_from_inspection(change):
    runs,_=reports()
    if change=='empty':runs=[]
    if change=='missing':runs.pop()
    if change=='failed':runs[0]['ok']=False
    if change=='duplicate':runs.append(runs[0])
    with pytest.raises(ValueError):complete_pairs(runs)


def test_pose_replay_must_be_finite_and_bound_to_the_output_pose():
    runs, report = reports()
    runs[1]['authoring_pose']['bones']['Hips'][0][3]=.1
    with pytest.raises(ValueError, match='native pose'):require_inspection(runs,report)
    runs[1]['authoring_pose']['bones']['Hips'][0][0]=float('nan')
    with pytest.raises(ValueError):pose_digest(runs[1]['authoring_pose'])


def test_pose_frame_comparison_handles_float32_quaternion_rounding_but_rejects_other_poses():
    q=np.array([.9238795,0,.3826834,0],np.float32)
    assert pose_frame_agrees([0,0,0],[0,0,0],q,q)
    assert not pose_frame_agrees([0,0,0],[.00002,0,0],q,q)
    assert not pose_frame_agrees([0,0,0],[0,0,0],q,[1,0,0,0])
    assert not pose_frame_agrees([0,0,0],[0,0,0],q,[np.nan,0,0,0])


@pytest.mark.parametrize('change', ['missing', 'mode', 'nan', 'shape'])
def test_invalid_native_transform_channels_are_rejected(change):
    runs, _ = reports(); pose = runs[0]['authoring_pose']
    if change == 'missing': pose.pop('channels')
    if change == 'mode': pose['channels']['Hips']['rotation_mode'] = 'UNKNOWN'
    if change == 'nan': pose['channels']['Hips']['location'][0] = float('nan')
    if change == 'shape': pose['channels']['Hips']['scale'] = [1, 1]
    with pytest.raises(ValueError): pose_digest(pose)
