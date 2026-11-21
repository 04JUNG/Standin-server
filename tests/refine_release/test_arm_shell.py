import unittest
import numpy as np
import run
from arm_shell import propose
from shell_policy import rasterize_silhouette
class ArmShellContracts(unittest.TestCase):
 def test_manual_input_cannot_be_relabelled_automatic(self):
  for field in ['manual_targets','manual_conditions_used','ground_truth_used','manual_experiment']:
   with self.assertRaises(ValueError):propose(None,dict(pr={field:True}))
 def test_unaffected_geometry_is_never_read(self):
  class NoGeometry:
   def __getattr__(self,name):raise AssertionError('Unexpected access '+name)
  self.assertEqual(propose(NoGeometry(),dict(pr={},diagnostic=dict(collision=dict(intersections={})))),[])
 def test_previously_passed_branch_is_not_modified(self):
  class PassedOnly:pool=[dict(id='shadow',diagnostic=dict(passed=True)),dict(id='preserve',diagnostic=dict(passed=True))]
  c=PassedOnly();before=repr(c.pool);run.Case.generate_repairs(c);self.assertEqual(repr(c.pool),before)
 def test_overlapping_projected_faces_do_not_make_silhouette_holes(self):
  xy=np.array([[1,1],[8,1],[4,8]],float);faces=np.array([[0,1,2],[0,1,2]])
  mask=rasterize_silhouette(xy,faces,(10,10));self.assertTrue(mask[3,4]);self.assertGreater(mask.sum(),20);self.assertFalse(mask[9,9])
if __name__=='__main__':unittest.main()
