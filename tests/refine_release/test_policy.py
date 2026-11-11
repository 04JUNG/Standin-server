import unittest,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from policy import *
from relations import validate,PROMPT
class SafetyTests(unittest.TestCase):
 def skeleton(self):
  names=['Hips','Spine2','Neck','Head','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot'];ix={n:i for i,n in enumerate(names)};w=np.repeat(np.eye(4)[None],len(names),axis=0);w[:,:3,3]=[[0,0,0],[0,1,0],[0,1.2,0],[0,1.5,0],[.3,1,0],[.6,1,0],[.9,1,0],[-.3,1,0],[-.6,1,0],[-.9,1,0],[.15,0,0],[.15,-.5,0],[.15,-1,0],[-.15,0,0],[-.15,-.5,0],[-.15,-1,0]];parents=np.array([-1,0,1,2,1,4,5,1,7,8,0,10,11,0,13,14]);return names,ix,w,parents
 def test_knee_twist_without_endpoint_motion_is_blocked(self):
  names,ix,w,p=self.skeleton();v=w.copy();v[ix['RightLeg'],:3,:3]=Rotation.from_euler('y',180,degrees=True).as_matrix();d=articulation(v,w,p,ix,neutral=w);self.assertTrue(any(x['kind']=='joint_axial_twist' and x['bone']=='RightLeg' for x in d['issues']))
 def test_reverse_knee_plane_is_blocked(self):
  names,ix,w,p=self.skeleton();v=w.copy();v[ix['RightFoot'],2,3]=.5;d=articulation(v,w,p,ix,neutral=w);self.assertTrue(any(x['kind']=='hinge_plane_reversal' and x['bone']=='RightLeg' for x in d['issues']))
 def test_head_flip_blocked(self):
  _,ix,w,p=self.skeleton();v=w.copy();v[ix['Head'],:3,:3]=Rotation.from_euler('y',180,degrees=True).as_matrix();self.assertTrue(any(x['kind']=='neck_head_rotation' for x in articulation(v,w,p,ix,neutral=w)['issues']))
 def test_bone_translation_is_not_a_collision_repair(self):
  _,ix,w,p=self.skeleton();v=w.copy();v[ix['RightHand'],2,3]+=.01;self.assertIn('bone_offset_changed',structural(v,w,p))
 def mesh(self,scale=1):
  v=np.array([[0.,0,0],[1.,0,0],[0.,1,0]])*scale;f=np.array([[0,1,2]]);return v,MeshPolicy(v,f,np.ones((3,1)),['Head'],1)
 def test_small_high_ratio_is_not_large_damage(self):
  v,g=self.mesh(1e-5);self.assertFalse(g.measure(v*9,np.eye(4)[None])['issues'])
 def test_collapsed_surface_blocked(self):
  v,g=self.mesh();self.assertTrue(g.measure(v*.01,np.eye(4)[None])['issues'])
 def test_inverted_face_blocked(self):
  v,g=self.mesh();v[[1,2]]=v[[2,1]];self.assertTrue(g.measure(v,np.eye(4)[None])['issues'])
 def test_rigid_mesh_rotation_is_safe(self):
  v,g=self.mesh();t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',130,degrees=True).as_matrix();self.assertFalse(g.measure(v@t[:3,:3].T,t[None])['issues'])
 def test_behind_head_relation_cannot_escape_forward(self):
  _,ix,w,p=self.skeleton();r=[dict(near='Head',far='RightHand',confidence='clear')];w[ix['RightHand'],2,3]=-.2;self.assertFalse(relationship_errors(w,ix,np.eye(3),1,r));w[ix['RightHand'],2,3]=.2;self.assertTrue(relationship_errors(w,ix,np.eye(3),1,r))
 def test_relations_are_image_bound_and_cycles_rejected(self):
  d=dict(image_sha256='x',prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),model='gpt-6-luna',reasoning_effort='low',manual_conditions_used=False,raw=dict(person_count=1,unresolved=[dict(a='LeftHand',b='Head',reason='unclear')],relations=[dict(near='Head',far='RightHand',confidence='clear',evidence='overlap')]))
  self.assertEqual(len(validate(d,'x')),1)
  with self.assertRaises(ValueError):validate(d,'different')
  d['raw']['relations'].append(dict(near='RightHand',far='Head',confidence='clear',evidence='contradiction'))
  with self.assertRaises(ValueError):validate(d,'x')
if __name__=='__main__':unittest.main()
