import unittest
import numpy as np
from scripts.local_converter.experimental.assembly import distinct_object_contact


class AssemblyContactTests(unittest.TestCase):
    def test_separate_objects_may_share_a_boundary_without_welding(self):
        start=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,0,0],[0,0,0],[0,-1,0]])
        faces=np.array([[0,1,2],[3,4,5]])
        end=start.copy()
        end[5,2]=.5
        result=distinct_object_contact(start,end,faces,np.array([0,1]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["pairs"],1)

    def test_other_mesh_crossing_during_step_fails(self):
        start=np.array([[0.,0,0],[1,0,0],[0,1,0],[.1,.1,1],[.2,.1,1],[.1,.2,1]])
        end=start.copy()
        end[3:,2]=-1
        r=distinct_object_contact(start,end,np.arange(6).reshape(2,3),np.array([0,1]))
        self.assertFalse(r["passed"])

    def test_shared_coordinate_does_not_exempt_overlapping_interiors(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,0],[.1,.2,0],[.2,.1,0]])
        q=p.copy()
        q[4,2]=.1
        r=distinct_object_contact(p,q,np.arange(6).reshape(2,3),np.array([0,1]))
        self.assertFalse(r["passed"])
