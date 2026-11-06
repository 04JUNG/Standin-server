import unittest
from .labels import build_label

class LabelsTests(unittest.TestCase):

    def record(self):

        def b(name, head):
            return dict(name=name, head=head, tail=[head[0], head[1], head[2] + 0.1], matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return dict(status='imported', case_id='x', sha256='s', bones=[b('Hips', [0, 0, 0]), b('Spine2', [0, 0, 1])])

    def test_world_baked_rest_not_zero_euler(self):
        self.assertEqual(build_label(self.record())['torso_scale_m'], 1)

    def test_nonfinite_and_ambiguous(self):
        r = self.record()
        r['bones'][0]['head'][0] = float('nan')
        with self.assertRaises(ValueError):
            build_label(r)
        r = self.record()
        r['bones'].append(dict(r['bones'][0]))
        with self.assertRaises(ValueError):
            build_label(r)
if __name__ == '__main__':
    unittest.main()
