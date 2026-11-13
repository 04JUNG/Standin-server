import unittest
import numpy as np
from scripts.local_converter.experimental.automatic_deformation import correct
from scripts.local_converter.experimental.seams import split_boundaries,gap_changes,guard_gap_step


class SeamTests(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[x,y,0.] for lo in (-1,0) for y in (0,1,2) for x in (lo,lo+1)])
        faces=[]
        for i in (0,2,6,8):
            faces.extend(((i,i+1,i+3),(i,i+3,i+2)))
        self.faces=np.array(faces)
        self.ids=np.zeros((12,1),int)
        self.weights=np.ones((12,1))

    def test_gap_at_each_matched_vertex_is_measured(self):
        p,_=split_boundaries(self.xyz,self.faces,self.ids,self.weights)
        np.testing.assert_array_equal(p,[[1,6],[3,8],[5,10]])
        moved=self.xyz.copy()
        moved[6:,0]+=.01
        stats=gap_changes(self.xyz,moved,p)
        self.assertEqual(stats["increased"],3)
        self.assertAlmostEqual(stats["maxAfter"],.01)
        recovered,check=guard_gap_step(self.xyz,moved,p)
        self.assertFalse(check["accepted"])
        np.testing.assert_array_equal(recovered,self.xyz)
        # A prior opening may close; evaluation always references the input pose.
        recovered,check=guard_gap_step(moved,self.xyz,p)
        self.assertTrue(check["accepted"])
        np.testing.assert_array_equal(recovered,self.xyz)
        q,report=correct(self.xyz,moved,self.faces,np.zeros(12,bool),
                         bone_ids=self.ids,bone_weights=self.weights)
        np.testing.assert_array_equal(q,moved)
        self.assertEqual(report["seams"]["input"]["increased"],3)

    def test_invalid_candidate_rolls_back(self):
        p,_=split_boundaries(self.xyz,self.faces,self.ids,self.weights)
        trial=self.xyz.copy()
        trial[6,0]=np.nan
        recovered,check=guard_gap_step(self.xyz,trial,p)
        self.assertFalse(check["accepted"])
        np.testing.assert_array_equal(recovered,self.xyz)

    def test_ambiguous_weights_and_stacked_faces_do_not_pair(self):
        self.ids[8]=1
        pairs,_=split_boundaries(self.xyz,self.faces,self.ids,self.weights)
        self.assertFalse(len(pairs))
        self.ids[8]=0
        self.xyz[6:,0]*=-1
        pairs,_=split_boundaries(self.xyz,self.faces,self.ids,self.weights)
        self.assertFalse(len(pairs))
