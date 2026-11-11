import unittest
import numpy as np
from release_policy import largest_component_area,contact_issue,unresolved_hand_disagreement
class Tests(unittest.TestCase):
 def test_connected_and_disconnected(self):
  f=np.array([[0,1,2],[1,2,3],[4,5,6]]);a=np.array([1.,2.,4.])
  self.assertEqual(largest_component_area(f,np.ones(3,bool),a),4.)
  self.assertEqual(largest_component_area(f,np.array([1,1,0],bool),a),3.)
 def test_missing_evidence_blocks(self):self.assertIsNotNone(contact_issue('a/b',1,{},1.))
 def test_shallow_small_contact(self):
  c={'extent':{'a/b':dict(complete=True,inward_samples=10,max_depth=.001,max_area_fraction=.0005)}}
  self.assertIsNone(contact_issue('a/b',1,c,1.));c['extent']['a/b']['max_depth']=.1
  self.assertIsNotNone(contact_issue('a/b',1,c,1.))
 def test_unresolved_hand_crossing_blocks(self):
  from types import SimpleNamespace
  a=np.repeat(np.eye(4)[None],3,axis=0);b=a.copy();a[0,2,3]=-.2;b[0,2,3]=.2;a[1,0,3]=b[1,0,3]=2
  c=SimpleNamespace(pool=[dict(world=a,cam={'camera':np.eye(3)}),dict(world=b,cam={'camera':np.eye(3)})],ix={'LeftHand':0,'RightHand':1,'Head':2},scale=1.,relations=[])
  self.assertEqual(len(unresolved_hand_disagreement(c)),1)
  c.relations=[dict(near='Head',far='LeftHand')];self.assertEqual(unresolved_hand_disagreement(c),[])
if __name__=='__main__':unittest.main()
