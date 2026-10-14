"""Scoring contracts for the image-only Z retrieval pilot."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_z_rendered import render_pose, retrieve, score_records
from standin_eval.z_axis_search import SearchIndex3D


def test_views_collapse_but_mirrors_are_not_accepted_as_exact_id():
    xy = np.zeros((4,17,2),dtype=np.float32)
    xy[1] += .1
    xy[2] += .2
    xy[3] += .3
    index = SearchIndex3D(("A_mirror","A_mirror","B","A"),
                          ("A","A","B","A"),np.array([0,45,0,0]),
                          xy,np.zeros((4,17)))
    candidates = retrieve(index,np.zeros((17,2)),np.zeros(17),np.ones(17,bool),0)
    assert [c["pose_id"] for c in candidates] == ["A_mirror","B","A"]


def test_failed_extractions_stay_in_denominator():
    records = [{"xy_rank":1,"xyz_rank":6},
               {"xy_rank":7,"xyz_rank":2},
               {"xy_rank":None,"xyz_rank":None,"error":"failed"}]
    metrics = score_records(records)
    assert metrics["xy"]["top1"] == 1/3
    assert metrics["xyz"]["top1"] == 0
    assert metrics["xy"]["top5"] == metrics["xyz"]["top5"] == 1/3


def test_rough_strokes_do_not_change_source_joint_projection():
    from src.library import _synthetic_pose
    joints = _synthetic_pose("running")
    for yaw in (0,45,90,180):
        _, original = render_pose(joints,yaw,size=128)
        rough, projected = render_pose(joints,yaw,size=128,style="rough",seed=7)
        assert np.array_equal(original,projected)
        pixels = np.asarray(rough)
        assert pixels.min()<150 and pixels.max()==255
        assert np.array_equal(pixels[:,:,0],pixels[:,:,1])


if __name__ == "__main__":
    test_views_collapse_but_mirrors_are_not_accepted_as_exact_id()
    test_failed_extractions_stay_in_denominator()
    test_rough_strokes_do_not_change_source_joint_projection()
    print("PASS 3 scoring/rendering contracts")
