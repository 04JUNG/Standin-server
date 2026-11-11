import unittest,numpy as np
import run  # initializes shared geometry module paths
from local_repair import face_score
from scipy.spatial.transform import Rotation
class SurfaceObjectiveTests(unittest.TestCase):
 def params(self):
  v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]]);e=f[:,[[0,1],[1,2],[2,0]]];length=np.linalg.norm(v[e[:,:,0]]-v[e[:,:,1]],axis=2)
  return v,f,length,np.eye(4)[None],np.ones((1,1)),np.array([[0.,0,1]]),np.array([.5]),1.
 def test_inversion_without_edge_stretch_is_penalized(self):
  v,f,length,t,fw,n,a,scale=self.params();v[[1,2]]=v[[2,1]];signed,exc=face_score(v,f,length,t,fw,n,a,scale)
  self.assertLess(signed[0],0);self.assertEqual(exc.max(),0)
 def test_rigid_rotation_does_not_create_surface_damage(self):
  v,f,length,t,fw,n,a,scale=self.params();r=Rotation.from_euler('y',160,degrees=True).as_matrix();t[0,:3,:3]=r;signed,exc=face_score(v@r.T,f,length,t,fw,n,a,scale)
  self.assertAlmostEqual(signed[0],1);self.assertLess(exc.max(),1e-12)
 def test_collapse_without_inversion_is_penalized(self):
  v,f,length,t,fw,n,a,scale=self.params();v[2]*=.01;signed,exc=face_score(v,f,length,t,fw,n,a,scale);self.assertGreater(signed[0],0);self.assertLess(signed[0],.15)
if __name__=='__main__':unittest.main()
