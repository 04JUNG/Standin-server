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
