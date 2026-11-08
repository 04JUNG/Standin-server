import unittest

import numpy as np

from experiments.refine_v5.ab_common import project
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.part_head import head_axes, propose as propose_head, unit
from experiments.refine_v5.part_upper import propose as propose_upper


def fixture():
    points = [
        ('Hips', None, (0, 0, 0)),
        ('Spine', 'Hips', (0, .15, 0)),
        ('Spine1', 'Spine', (0, .30, 0)),
        ('Spine2', 'Spine1', (0, .45, 0)),
        ('Neck', 'Spine2', (0, .55, 0)),
        ('Head', 'Neck', (0, .64, 0)),
        ('LeftShoulder', 'Spine2', (.06, .49, 0)),
        ('LeftArm', 'LeftShoulder', (.19, .49, 0)),
        ('LeftHand', 'LeftArm', (.47, .29, 0)),
        ('RightShoulder', 'Spine2', (-.06, .49, 0)),
        ('RightArm', 'RightShoulder', (-.19, .49, 0)),
        ('RightHand', 'RightArm', (-.47, .29, 0)),
    ]
    bones = []
    for name, parent, xyz in points:
        matrix = np.eye(4)
        matrix[:3, 3] = xyz
        bones.append(dict(name=name, parent=parent, matrix=matrix.tolist()))
    camera = dict(origin=[0, 0, 0], torso_world=.45, rotation=np.diag([1, -1, -1]).tolist(),
                  inverse_distance_torso=0., pixel_scale=450., pixel_offset=[500, 600],
                  rough_unit_px=450., status='fitted')
    return dict(case_id='synthetic', source_sha256='b0', body_asset_sha256='body',
                body_forward=[0, 0, 1], torso_world=.45, camera=camera,
                ownership_authority='offline_single_target_review', bones=bones,
                part_goal={'head': None})


class PartTargetsTest(unittest.TestCase):
    def test_head_can_move_independently_and_neck_requires_observation(self):
        case = fixture()
        case['part_goal']['head'] = dict(source='reviewed', up_screen=[.1, -np.sqrt(.99)])
        plan = propose_head(case)
        self.assertEqual(set(plan['edits_deg']), {'Head'})
        self.assertLess(plan['after_deg']['screen_up'], .05)
        names, index, _, _, world = prepare(case)
        for name in ['LeftHand', 'RightHand', 'Neck', 'Hips']:
            self.assertEqual(plan['expected_heads'][name], world[index[name], :3, 3].tolist())

    def test_known_3d_head_and_neck_goals(self):
        case = fixture()
        names, index, parents, local, base = prepare(case)
        target = fk(local, parents, [index['Neck'], index['Head']], np.deg2rad([[3., 0., 0.], [0., 7., -4.]]))
        case['part_goal']['head'] = dict(source='synthetic',
                                      axes_body={name: axis.tolist() for name, axis in head_axes(target, index).items()})
        plan = propose_head(case)
        self.assertEqual(set(plan['edits_deg']), {'Head', 'Neck'})
        self.assertLess(max(plan['after_deg'].values()), .3)

    def test_spine_and_clavicle_open_separately(self):
        case = fixture()
        _, ix, parents, local, _ = prepare(case)
        target = fk(local, parents, [ix[n] for n in ('Spine', 'Spine1', 'Spine2', 'LeftShoulder')],
                    np.deg2rad([[0, 0, 1.8], [0, 0, 1.8], [0, 0, 1.8], [0, 0, 3.]]))
        xy = project(target[[ix[n] for n in ('Hips', 'Spine2', 'LeftArm')], :3, 3], case['camera'])
        case['part_goal']['upper'] = dict(source='synthetic', independent_camera=True,
                                         chest_up_screen=unit(xy[1] - xy[0]).tolist(),
                                         shoulders_px={'Left': xy[2].tolist()})
        spine = propose_upper(case, spine=True, clavicle=False)
        clavicle = propose_upper(case, spine=False, clavicle=True)
        both = propose_upper(case, spine=True, clavicle=True)
        self.assertEqual(set(spine['edits_deg']), {'Spine', 'Spine1', 'Spine2'})
        self.assertEqual(set(clavicle['edits_deg']), {'LeftShoulder'})
        self.assertLess(both['after_residual'], clavicle['after_residual'])
        case['camera']['status'] = 'fallback'
        self.assertIsNone(propose_upper(case, spine=True, clavicle=True))
