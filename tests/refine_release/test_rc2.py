"""Regression checks for measured RC2 failure modes, without cut-specific rules."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_policy
ROOT=Path(__file__).resolve().parents[2]/'services/refine_release/runtime/experiments'
spec=importlib.util.spec_from_file_location('rc2_validity',ROOT/'refine_unified/validity.py');validity=importlib.util.module_from_spec(spec);spec.loader.exec_module(validity)
from policy import articulation
class RC2Tests(unittest.TestCase):
 def test_nearly_straight_reverse_knee_not_hidden_at_ten_degrees(self):
  _,ix,w,p=test_policy.SafetyTests().skeleton()
  for degrees in [9.9,10.1]:
   v=w.copy();v[ix['RightFoot'],:3,3]=v[ix['RightLeg'],:3,3]+[0,-.5*np.cos(np.radians(degrees)),.5*np.sin(np.radians(degrees))]
   issues=articulation(v,w,p,ix,neutral=w)['issues']
   self.assertTrue(any(x['kind']=='hinge_plane_reversal' and x['bone']=='RightLeg' for x in issues))
 def test_fixed_source_anisotropy_allows_rotations_not_offsets(self):
  base=np.repeat(np.eye(4)[None],3,0);base[0,:3,:3]=np.diag([.999988,1,1]);local=np.eye(4);local[1,3]=.2;base[1]=base[0]@local;base[2]=base[1]@local
  edited=local.copy();edited[:3,:3]=Rotation.from_euler('z',30,degrees=True).as_matrix();w=base.copy();w[1]=base[0]@edited;w[2]=w[1]@local
  validity.check_world(w,base,[-1,0,1])
  w[2,1,3]+=.001
  with self.assertRaises(ValueError):validity.check_world(w,base,[-1,0,1])
