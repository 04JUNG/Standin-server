import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bvh import coco17_from_fk, fk
from standin_eval.z_diagnostics import infer_pose
from scripts.audit_pose_handedness import asymmetric_front_pose
from scripts.experiment_z_rendered import render_pose


def test_bvh_named_joint_to_coco_mapping_preserves_sides():
    expected = {5:'LeftArm', 6:'RightArm', 7:'LeftForeArm', 8:'RightForeArm',
                9:'LeftHand', 10:'RightHand', 11:'LeftUpLeg', 12:'RightUpLeg',
                13:'LeftLeg', 14:'RightLeg', 15:'LeftFoot', 16:'RightFoot'}
    for prefix in ('', 'mixamorig:'):
        joints = [[prefix+'Hips', -1, np.zeros(3), [], False]]
        for ci, name in expected.items():
            joints.append([prefix+name, 0, np.array([ci, ci*2, -ci], dtype=float), [], False])
        world = fk(joints, np.array([]))
        result, scores = coco17_from_fk(joints, world)
        hip = (world[7]+world[8])/2
        for index, ci in enumerate(expected, 1):
            assert np.allclose(result[ci], world[index]-hip)
            assert scores[ci] == 1


def test_every_joint_channel_survives_all_four_rotations():
    image = np.zeros((91, 137, 3), dtype=np.uint8)
    points = np.array([[5+i*7, 8+i*4] for i in range(17)])
    for i, (x, y) in enumerate(points): image[y, x, 0] = i+1
    class MarkerModel:
        def __call__(self, im, **kw):
            coords = []
            for i in range(17):
                y, x = np.where(im[:, :, 0] == i+1)
                coords.append([int(x[0]), int(y[0])])
            return np.array([coords]), np.linspace(.4, .9, 17)[None, :]
    for turn in range(4):
        output = infer_pose(image, MarkerModel(), crop=True, turns=(turn,), is_3d=False)
        assert np.array_equal(output['keypoints2d'], points)
        assert np.allclose(output['scores'], np.linspace(.4, .9, 17))


def test_rtm_simcc_decode_keeps_joint_channel_order():
    from rtmlib import RTMPose
    model = RTMPose.__new__(RTMPose)
    model.model_input_size = (192, 256)
    sx, sy = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
    expected = []
    for j in range(17):
        x, y = 20+j*7, 30+j*9
        sx[0, j, x] = sy[0, j, y] = .5+j*.01
        expected.append([x/2, y/2])
    coords, scores = model.postprocess([sx, sy], np.array([96, 128]), np.array([192, 256]))
    assert np.allclose(coords[0], expected)
    assert np.allclose(scores[0], .5+np.arange(17)*.01)


def test_frontal_renderer_has_left_on_screen_right_and_visible_eyes():
    p = asymmetric_front_pose()
    front, qf = render_pose(p, 0, style='solid')
    back, qb = render_pose(p, 180, style='solid')
    assert qf[5, 0] > qf[6, 0] and qb[5, 0] < qb[6, 0]
    def dark_head_pixels(image, q):
        pixels = np.asarray(image)
        x, y = np.round(q[0]).astype(int)
        patch = pixels[y-13:y+13, x-17:x+17]
        return int((patch.max(axis=-1) < 70).sum())
    assert dark_head_pixels(front, qf) > 0
    assert dark_head_pixels(back, qb) == 0


if __name__ == '__main__':
    for name, fun in sorted(list(globals().items())):
        if name.startswith('test_'): fun(); print('PASS', name)
