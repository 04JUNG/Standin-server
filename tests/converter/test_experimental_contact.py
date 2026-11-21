import unittest
import numpy as np
from scripts.local_converter.experimental.contact import baseline_contact_pins, contact_guard


class ContactTests(unittest.TestCase):
    def test_crossing_between_clear_endpoints_is_rejected(self):
        tri = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        start = np.vstack([tri,tri+[0,0,-1]])
        end = np.vstack([tri,tri+[0,0,1]])
        self.assertFalse(contact_guard(start,end,np.arange(6).reshape(2,3))["passCheck"])

    def test_baseline_contact_pins_both_faces_and_leaves_distant_face_free(self):
        points = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
                           [.5,.5,-1.],[.5,.5,1.],[1.,.5,0.],
                           [4.,4.,0.],[5.,4.,0.],[4.,5.,0.]])
        faces = np.arange(9).reshape(3,3)
        free = np.ones(9,bool)
        pinned, report = baseline_contact_pins(points,faces,free)
        self.assertTrue(report["complete"])
        self.assertEqual(report["pairs"],1)
        np.testing.assert_array_equal(pinned,[True]*6+[False]*3)
        pinned, report = baseline_contact_pins(points,faces,free,pair_budget=0)
        self.assertFalse(report["complete"])
        np.testing.assert_array_equal(pinned,free)

    def test_coplanar_separated_motion_passes(self):
        start = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
                          [1.5,1.5,0.],[3.,1.5,0.],[1.5,3.,0.]])
        end = start.copy()
        end[3:] += [.1,0,0]
        self.assertTrue(contact_guard(start,end,np.arange(6).reshape(2,3))["passCheck"])
