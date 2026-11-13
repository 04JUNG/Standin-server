import unittest
import numpy as np
from scripts.local_converter.experimental.vertex_contact import shared_vertex_guard


class NeighborContactTests(unittest.TestCase):
    def test_faces_cross_at_midstep_beyond_shared_vertex(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[.1,.3,1],[.3,.1,1]])
        q=p.copy()
        q[3:,2]=-1
        result=shared_vertex_guard(p,q,np.array([[0,1,2],[0,3,4]]))
        self.assertFalse(result["passCheck"])
        self.assertTrue(result["initialSeparated"] and result["finalSeparated"])

    def test_folding_through_neighbor_with_shared_edge_is_rejected(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,.5,1]])
        q=p.copy()
        q[3,2]=-1
        self.assertFalse(shared_vertex_guard(p,q,np.array([[0,1,2],[1,0,3]]))["passCheck"])

    def test_flat_neighbors_can_move_without_overlapping(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,0]])
        q=p.copy()
        q[3]+=[0,.25,.5]
        result=shared_vertex_guard(p,q,np.array([[0,1,2],[1,0,3]]))
        self.assertTrue(result["passCheck"])
        self.assertEqual(result["edgePairs"],1)
