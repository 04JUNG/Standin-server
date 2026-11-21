import unittest
import numpy as np
from scripts.local_converter.experimental.automatic_deformation import correct


class IsolatedInfluenceTests(unittest.TestCase):
    def setUp(self):
        grid=np.arange(63).reshape(7,9)
        y,x=np.indices(grid.shape)
        self.rest=np.column_stack((x.ravel(),y.ravel(),(.005*x*x).ravel())).astype(float)
        faces=[]
        for a in grid[:-1,:-1].ravel():
            faces.extend([[a,a+1,a+10],[a,a+10,a+9]])
        self.faces=np.asarray(faces,int)
        self.distal=x.ravel()>=7
        self.seed=grid[3,3]
        bones=self.distal.astype(int)
        bones[self.seed]=1
        self.ids=bones[:,None]
        self.posed=self.rest.copy()
        self.posed[bones==1,2]+=4.
        self.fixed=np.zeros(len(self.rest),bool)

    def run_repair(self,rest,posed,ids):
        return correct(rest,posed,self.faces,self.fixed,bone_ids=ids,
                       bone_weights=np.ones_like(ids,float),regions=0)

    def test_repairs_island_preserving_driver_motion_and_units(self):
        for scale in (.1,100.):
            points,report=self.run_repair(scale*self.rest,scale*self.posed,self.ids)
            expected=scale*self.posed
            expected[self.seed]=scale*self.rest[self.seed]
            self.assertTrue(report["isolatedInfluence"]["applied"])
            np.testing.assert_allclose(points,expected,atol=scale*1e-7)

    def test_vertex_order_and_bone_numbers_do_not_select_the_repair(self):
        order=np.random.default_rng(7601).permutation(len(self.rest))
        inverse=np.argsort(order)
        renamed=np.where(self.ids[order]==0,73,11)
        result, report=correct(self.rest[order],self.posed[order],inverse[self.faces],
            self.fixed[order],bone_ids=renamed,bone_weights=np.ones_like(renamed,float),regions=0)
        self.assertTrue(report["applied"])
        expected=self.posed.copy()
        expected[self.seed]=self.rest[self.seed]
        np.testing.assert_allclose(result[inverse],expected,atol=1e-7)

    def test_driver_without_other_coherent_support_is_preserved(self):
        ids=self.ids.copy()
        ids[self.seed]=2
        result,report=self.run_repair(self.rest,self.posed,ids)
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result,self.posed)

    def test_driver_on_other_disconnected_piece_cannot_authorize_repair(self):
        ids=self.ids.copy()
        ids[self.seed]=2
        count=len(self.rest)
        rest=np.vstack((self.rest,self.rest+[20.,0.,0.]))
        posed=np.vstack((self.posed,self.rest+[20.,0.,4.]))
        faces=np.vstack((self.faces,self.faces+count))
        ids=np.vstack((ids,np.full_like(ids,2)))
        result,report=correct(rest,posed,faces,np.zeros(len(rest),bool),
                              bone_ids=ids,bone_weights=np.ones_like(ids,float),regions=0)
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result,posed)

    def test_invalid_influences_return_original(self):
        result,report=self.run_repair(self.rest,self.posed,self.ids.astype(float)+.25)
        self.assertEqual(report["reason"],"invalid_weights")
        np.testing.assert_array_equal(result,self.posed)
