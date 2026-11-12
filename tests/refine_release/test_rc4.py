"""RC4 contracts use synthetic geometry, never evaluation case identifiers."""
import unittest
import numpy as np
from policy import MeshPolicy, articulation, LIMITS
import test_policy as baseline

class LocalSurfaceTests(unittest.TestCase):
 def test_local_fold_is_not_diluted_by_long_limb(self):
  # One physical patch with reversed winding, plus large undamaged distal skin.
  v=np.array([[0.,0,0],[.12,0,0],[0,.12,0],[.12,.12,0],[10,0,0],[20,0,0],[10,10,0]])
  f=np.array([[0,1,2],[1,3,2],[4,5,6]]);weights=np.ones((7,1))
  p=MeshPolicy(v,f,weights,['RightForeArm'],1,joint_positions=np.zeros((1,3)))
  changed=v.copy();changed[3]=[-.06,-.06,0]
  d=p.measure(changed,np.eye(4)[None])
  self.assertTrue(any(i['kind']=='joint_surface_fold' for i in d['issues']))
 def test_submillimetre_sliver_is_not_local_collapse(self):
  v=np.array([[0.,0,0],[1e-5,0,0],[0,1e-5,0],[10,0,0],[20,0,0],[10,10,0]])
  f=np.array([[0,1,2],[3,4,5]])
  p=MeshPolicy(v,f,np.ones((6,1)),['RightForeArm'],1,joint_positions=np.zeros((1,3)))
  changed=v.copy();changed[[1,2]]=changed[[2,1]]
  self.assertFalse(any(i['kind']=='joint_surface_fold' for i in p.measure(changed,np.eye(4)[None])['issues']))
 def test_rigid_pose_preserves_local_surface(self):
  from scipy.spatial.transform import Rotation
  v=np.array([[0.,0,0],[.12,0,0],[0,.12,0]])
  p=MeshPolicy(v,np.array([[0,1,2]]),np.ones((3,1)),['RightForeArm'],1,joint_positions=np.zeros((1,3)))
  t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',[100,45,-80],degrees=True).as_matrix()
  self.assertFalse(p.measure(v@t[:3,:3].T,t[None])['issues'])
 def test_cervical_chain_checked_even_if_local_delta_small(self):
  _,ix,w,parents=baseline.SafetyTests().skeleton();base=w.copy();w[ix['Head'],:3,3]=w[ix['Neck'],:3,3]+[.3,0,0]
  self.assertTrue(any(i['kind']=='cervical_overbend' for i in articulation(w,base,parents,ix,neutral=base)['issues']))
 def test_endpoint_near_straight_is_not_ten_degree_escape(self):
  self.assertLessEqual(LIMITS.hinge_near_straight_deg,3)

if __name__=='__main__':unittest.main()
