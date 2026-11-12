import math
import numpy as np
import pytest
from converter.bone_map import MIXAMO
from converter.interactive_camera import camera_for_model
from src.refine_projection import search_rotation


def base_model():
    bones=[]
    for name,key in [(v,k) for k,v in MIXAMO.items()]:
        m=np.eye(4)
        if key=='upperarm.L':m[:3,3]=[-.2,.5,0]
        if key=='upperarm.R':m[:3,3]=[.2,.5,0]
        if key=='upleg.L':m[:3,3]=[-.1,0,0]
        if key=='upleg.R':m[:3,3]=[.1,0,0]
        bones.append(dict(name=name,parent=-1,restLocal=m.ravel().tolist()))
    return dict(bones=bones)


@pytest.mark.parametrize('yaw,pitch',[(y,p) for y in range(0,360,45) for p in (-38.25,0,26.75)]+[(17.713,-22.981)])
@pytest.mark.parametrize('projection',['orthographic','perspective'])
def test_continuous_search_projection_exact(yaw,pitch,projection):
    r=search_rotation(yaw,pitch);distance=4.7 if projection=='perspective' else None
    descriptor=dict(version='candidate-camera-v2',reference='search-fit',rotation=r.tolist(),projection=projection,distance_torso=distance)
    camera=camera_for_model(descriptor,base_model(),'c'*64,'full')
    view=np.array(camera['viewMatrix']).reshape(4,4);p=np.array(camera['projectionMatrix']).reshape(4,4)
    assert np.linalg.det(view[:3,:3])==pytest.approx(1.)
    for xyz in [[.15,.3,.1],[-.2,-.2,-.1],[0,.6,.2]]:
        point=np.array([*xyz,1.]);actual=p@view@point;actual=actual[:2]/actual[3]
        aligned=r@np.array(xyz);expected=aligned[:2]*.8
        if distance:expected=aligned[:2]/(aligned[2]+distance*.5)*distance*.5*.8
        expected[1]-=.1
        assert actual==pytest.approx(expected,abs=1e-12)


def presentation_descriptor(projection='perspective'):
    return dict(version='pose-preview-camera-v1', projection=projection,
                rotation=search_rotation(45., -12.5).tolist(),
                yaw_degrees=45., pitch_degrees=-12.5, roll_degrees=0.,
                distance_torso=4.7 if projection=='perspective' else None,
                basis='right-handed-Y-up', rotation_order='Rz@Rx@Ry_column_vectors',
                origin='hip_midpoint', scale='hip_to_shoulder_midpoint_length', screen_y='down',
                projection_rule='xy=(x,-y)/(z+distance_torso)' if projection=='perspective' else 'xy=(x,-y)',
                projection_matches_search=False, fit_to_rough_pixels=False,
                bvh_camera_rotation_applied=False, source_bvh_sha256='b'*64)


@pytest.mark.parametrize('projection', ['orthographic', 'perspective'])
def test_explicit_presentation_camera_reuses_exact_projection_without_relabeling(projection):
    descriptor=presentation_descriptor(projection)
    search=dict(version='candidate-camera-v2', reference='search-fit',
                rotation=descriptor['rotation'], projection=projection,
                distance_torso=descriptor['distance_torso'])
    before=dict(descriptor)
    assert camera_for_model(descriptor,base_model(),'c'*64,'full') == camera_for_model(search,base_model(),'c'*64,'full')
    assert descriptor == before
    assert descriptor['version']=='pose-preview-camera-v1'
    assert descriptor['projection_matches_search'] is False
    assert 'reference' not in descriptor


@pytest.mark.parametrize('change', [
    dict(version='candidate-camera-v1'), dict(version='candidate-camera-v2'),
    dict(basis='left-handed-Y-up'), dict(rotation_order='Ry@Rx@Rz'),
    dict(origin='bounds'), dict(scale='bounds'), dict(screen_y='up'),
    dict(fit_to_rough_pixels=True), dict(bvh_camera_rotation_applied=True),
    dict(projection_matches_search=None), dict(yaw_degrees=float('nan')),
    dict(projection_rule='xy=(x,-y)'), dict(rotation=None),
    dict(rotation=[[-1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
    dict(distance_torso=True), dict(distance_torso=0.), dict(distance_torso=float('inf')),
])
def test_presentation_camera_rejects_incomplete_or_changed_conventions(change):
    with pytest.raises(ValueError):
        camera_for_model({**presentation_descriptor(), **change},base_model(),'c'*64,'full')
