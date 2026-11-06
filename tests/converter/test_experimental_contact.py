import unittest
import numpy as np
from scripts.local_converter.experimental.contact import contact_guard


class ContactTests(unittest.TestCase):
    def test_crossing_between_clear_endpoints_is_rejected(self):
        tri = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        start = np.vstack([tri,tri+[0,0,-1]])
        end = np.vstack([tri,tri+[0,0,1]])
        self.assertFalse(contact_guard(start,end,np.arange(6).reshape(2,3))["passCheck"])

    def test_coplanar_separated_motion_passes(self):
        start = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
                          [1.5,1.5,0.],[3.,1.5,0.],[1.5,3.,0.]])
        end = start.copy()
        end[3:] += [.1,0,0]
        self.assertTrue(contact_guard(start,end,np.arange(6).reshape(2,3))["passCheck"])
