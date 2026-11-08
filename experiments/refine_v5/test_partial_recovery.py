import unittest

from experiments.refine_v5.target_safety import independent_edits


class PartialRecoveryTest(unittest.TestCase):
    def test_both_arm_failures_leave_the_head_edit(self):
        edits = dict(LeftArm=[1, 0, 0], LeftForeArm=[2, 0, 0],
                     RightArm=[3, 0, 0], Head=[4, 0, 0])
        attempts = [
            {'reasons': ['new_skin_penetration:RightUpperArm:Torso']},
            {'reasons': ['new_skin_penetration:LeftHand:RightThigh']},
        ]
        groups, blocked = independent_edits(edits, attempts)
        self.assertEqual(groups, {'head': {'Head': [4, 0, 0]}})
        self.assertEqual(blocked, ['left_arm', 'right_arm'])

    def test_two_moving_colliders_are_both_excluded(self):
        edits = dict(LeftLeg=[1, 0, 0], RightArm=[1, 0, 0], Head=[1, 0, 0])
        attempts = [{'reasons': ['deeper_skin_penetration:LeftShin:RightHand']}]
        groups, blocked = independent_edits(edits, attempts)
        self.assertEqual(list(groups), ['head'])
        self.assertEqual(blocked, ['left_leg', 'right_arm'])

    def test_coupled_spine_edit_has_no_partial_fallback(self):
        self.assertEqual(independent_edits({'Spine2': [1, 0, 0], 'LeftArm': [1, 0, 0]}, []), ({}, []))
