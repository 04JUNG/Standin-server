import numpy as np
from experiments.refine_v5.learning_data import _render,image_features,geometry
from experiments.refine_v5.predict_v3 import predict

def test_predict_rejects_gt_before_any_file_is_opened():
    try:predict('missing',{'gt':'secret'})
    except ValueError:pass
    else:raise AssertionError('GT input accepted')

def test_synthetic_is_mesh_projection_and_records_occlusion():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],float);f=np.array([[0,1,2]])
    h=np.tile(np.array([[0.,0.,0.]]),(12,1))
    image,points,mask=_render(v,f,h,np.eye(3),np.random.default_rng(2),False)
    assert image.shape==(96,96) and image.min()==0 and image.max()==255
    assert np.isfinite(image_features(image,points)).all() and mask.sum()==12

def test_occluded_observation_cannot_leak_coordinates():
    base=np.ones((13,2,3));q=np.ones((12,2));mask=np.zeros(12)
    a=geometry(base,q,mask,'stick',np.eye(3));b=geometry(base,q*100,mask,'stick',np.eye(3))
    assert np.array_equal(a,b)

def test_reflected_camera_rejected():
    # Reflection cannot be silently used as a rotation.
    from experiments.refine_v5.learning_data import frame
    b=[dict(name=n,head=p,matrix=np.eye(4).tolist()) for n,p in [('Hips',[0,0,0]),('Spine2',[0,1,0]),('LeftArm',[1,1,0]),('RightArm',[-1,1,0])]]
    _,_,_,f=frame(b);assert np.linalg.det(f)>.999

def test_native_camera_matches_shared_library_projection():
    from experiments.refine_v5.learning_data import source_camera,frame
    from src.library import project_3d_to_2d,view_angle
    bones=[dict(name=n,head=p,matrix=np.eye(4).tolist()) for n,p in [('Hips',[0,0,0]),('Spine2',[0,0,1]),('LeftArm',[1,0,1]),('RightArm',[-1,0,1])]]
    _,_,_,body=frame(bones);native=np.array([[.2,.3,.4],[-.6,.1,.8]])
    search=native[:,[0,2,1]]*np.array([1,1,-1])
    for view in ['front','three_quarter','side','back']:
        actual=(native@body@source_camera(bones,view).T)[:,:2]
        expected=project_3d_to_2d(search,view_angle(view))*np.array([1,-1])
        assert np.allclose(actual,expected,atol=1e-6)


def test_occluded_forearm_does_not_mask_visible_elbow():
    from experiments.refine_v5.learning_data import point_visibility
    m=np.zeros(13,bool);m[0]=True;m[6]=True
    assert point_visibility(m).tolist()==[True,True,False,False,False,False,False,False,False,True,True,False]
