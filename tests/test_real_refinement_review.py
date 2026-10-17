import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.compare_refinement_real_roughs as review
from standin_eval.perspective_search import Camera,build_camera_index,score_camera_index
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def test_view_costs_match_existing_scoring_and_camera_conversion():
    poses=[Pose3D('run','run',_synthetic_pose('running'),np.ones(17))]
    cams=[Camera(0),Camera(75,20,4.)];idx=build_camera_index(poses,cams)
    q=idx[0][1,0];mask=np.ones(17,dtype=bool)
    costs,views=review.best_views(idx,q,mask)
    assert np.array_equal(costs,score_camera_index(idx,q,mask)) and views.tolist()==[1]
    assert review.params_camera([0,0,0])==Camera(0)
    assert review.params_camera([75,20,.25])==Camera(75,20,4)


def test_preview_fallback_keeps_camera_instead_of_changing_view():
    original=review.render_pose
    def reject(*args,**kwargs):raise ValueError('test volume collision')
    review.render_pose=reject
    try:
        im,kind,error=review.projection_preview(_synthetic_pose('running'),Camera(75,20,4),size=120)
        assert im.size==(120,120) and np.asarray(im).min()<100
        assert kind=='투영 관절만' and error=='test volume collision'
    finally:review.render_pose=original


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn();print('PASS',name)
