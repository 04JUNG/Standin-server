"""Final measured safety policy. Case IDs never participate in decisions."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class Limits:
 knee_hinge_deg:float=60.
 hinge_near_straight_deg:float=3.
 burial_depth_torso:float=.015
 burial_area_fraction:float=.20
 maximum_projection_rmse_viewport:float=.08
 export_position_torso:float=1e-4
 knee_repair_target_deg:float=55.
 absolute_peak_excess_torso:float=.15
 connected_inverted_area:float=.01
 total_inverted_area:float=.02
 contact_depth_torso:float=.06
 contact_area_fraction:float=.06
 contact_severity:float=.003
 repair_rotation_deg:float=6.
 repair_soft_motion_torso:float=.038
 repair_backtrack_motion_torso:float=.039
 repair_motion_torso:float=.04
 shell_iou:float=.94
 selection_shift_weight:float=.1
 profile_min_yaw_deg:float=55.
 landmark_error_increase_px:float=3.
 export_rotation_deg:float=.05
 export_relative_scale:float=1e-4
 export_position:float=1e-5
 export_skin_torso:float=1e-4
 knee_twist_deg:float=35.
 elbow_twist_deg:float=100.
 hinge_reversal_deg:float=60.
 cervical_bend_deg:float=55.
 local_patch_radius_torso:float=.18
 local_inverted_component_fraction:float=.01
 local_inverted_total_fraction:float=.05
 local_damage_area_torso2:float=.001
 expanded_repair_rotation_deg:float=24.
 expanded_repair_motion_torso:float=.08
 sharp_crease_length_torso:float=.75
 neck_delta_deg:float=65.
 head_delta_deg:float=100.
 regional_damaged_area:float=.01
 edge_excess_torso:float=.005
 peak_edge_excess_torso:float=.04
 connected_stretch_area:float=.01
 connected_stretch_peak_torso:float=.02
 collapsed_face_ratio:float=.15
 landmark_budget_px:float=12.
 depth_margin_torso:float=.03

LIMITS=Limits()
def unit(x):return x/max(np.linalg.norm(x),1e-12)
def rotation(m):
 u,_,v=np.linalg.svd(m);r=u@v
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
 return r

def angle(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(unit(a),unit(b)),-1,1))))
def bend(p):return angle(p[1]-p[0],p[2]-p[1])
def twist_degrees(matrix,axis):
 from scipy.spatial.transform import Rotation
 q=Rotation.from_matrix(rotation(matrix)).as_quat();axis=unit(axis);along=np.dot(q[:3],axis);n=np.hypot(along,q[3])
 if n<1e-10:return 180.
 return float(abs((np.degrees(2*np.arctan2(along,q[3]))+180)%360-180))
def region(n):
 for s in ['Left','Right']:
  if n.startswith(s):return s+('_leg' if any(t in n for t in ['UpLeg','Leg','Foot','Toe']) else '_arm')
 return 'head' if n.startswith(('Head','Neck')) else 'torso'

def structural(world,base,parents):
 if world.shape!=base.shape or not np.isfinite(world).all():return ['nonfinite_or_shape']
 errors=[]
 if not np.allclose(world[:,3,:],[0,0,0,1],atol=1e-7,rtol=0):errors.append('nonhomogeneous')
 for i,p in enumerate(parents):
  a=np.linalg.solve(base[p],base[i]) if p>=0 else base[i];b=np.linalg.solve(world[p],world[i]) if p>=0 else world[i]
  if p>=0 and not np.allclose(a[:3,3],b[:3,3],atol=1e-8,rtol=0):errors.append('bone_offset_changed')
  d=np.linalg.solve(a[:3,:3],b[:3,:3])
  if not np.allclose(d.T@d,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(d)-1)>1e-5:errors.append('bone_scale_or_shear')
 return sorted(set(errors))

def articulation(world,base,parents,ix,limits=LIMITS,neutral=None):
 from scipy.spatial.transform import Rotation
 issues=[];details={}
 # FBX can bake a selected library pose into its bind/rest matrices. Such a
 # rest pose is not anatomical neutral; torso-facing does not define knee axes.
 baked=neutral is not None and np.allclose(neutral,base,atol=1e-5,rtol=1e-5) and any(bend(base[[ix[s+'UpLeg'],ix[s+'Leg'],ix[s+'Foot']],:3,3])>5 for s in ['Left','Right'])
 anatomical_neutral=neutral is not None and not baked
 reference=base if not anatomical_neutral else neutral
 for side in ['Left','Right']:
  for upper,joint,end,kind,limit in [(side+'UpLeg',side+'Leg',side+'Foot','knee',limits.knee_twist_deg),(side+'Arm',side+'ForeArm',side+'Hand','elbow',limits.elbow_twist_deg)]:
   if not all(n in ix for n in [upper,joint,end]):issues.append(dict(kind='missing_joint_chain',region=side));continue
   i,j,k=map(ix.get,[upper,joint,end]);p=parents[j]
   oldlocal=np.linalg.solve(reference[p],reference[j]);newlocal=np.linalg.solve(world[p],world[j]);delta=np.linalg.solve(oldlocal[:3,:3],newlocal[:3,:3]);axis=np.linalg.solve(reference[j,:3,:3],reference[k,:3,3]-reference[j,:3,3]);twist=twist_degrees(delta,axis)
   oldp=base[[i,j,k],:3,3];newp=world[[i,j,k],:3,3];flex=bend(newp);oldflex=bend(oldp)
   reversal=None
   expected=None;actual=None
   if oldflex>5 and flex>1e-4:
    oldnormal=unit(np.linalg.solve(base[i,:3,:3],np.cross(oldp[1]-oldp[0],oldp[2]-oldp[1])))
    newnormal=unit(np.linalg.solve(world[i,:3,:3],np.cross(newp[1]-newp[0],newp[2]-newp[1])))
    reversal=angle(oldnormal,newnormal);expected=oldnormal;actual=newnormal
   if anatomical_neutral:
    up=unit(neutral[ix['Spine2'],:3,3]-neutral[ix['Hips'],:3,3]);right=unit(neutral[ix['LeftArm'],:3,3]-neutral[ix['RightArm'],:3,3]);forward=unit(np.cross(right,up));u=unit(neutral[j,:3,3]-neutral[i,:3,3]);intended=-forward if kind=='knee' else forward
    hinge=np.cross(u,intended)
    if np.linalg.norm(hinge)>1e-6 and flex>1e-4:
     expected=unit(np.linalg.solve(neutral[i,:3,:3],hinge));actual=unit(np.linalg.solve(world[i,:3,:3],np.cross(newp[1]-newp[0],newp[2]-newp[1])));reversal=angle(expected,actual)
   hinge_limit=limits.knee_hinge_deg if kind=='knee' else (limits.hinge_reversal_deg if flex>15 and (anatomical_neutral or oldflex>=15) else 120.)
   details[joint]=dict(hinge_limit_deg=hinge_limit,twist_deg=twist,bend_deg=flex,reference_bend_deg=oldflex,hinge_plane_change_deg=reversal,axis_calibrated=expected is not None,expected_normal_local=None if expected is None else expected.tolist(),actual_normal_local=None if actual is None else actual.tolist())
   if twist>limit:issues.append(dict(kind='joint_axial_twist',bone=joint,region=region(joint),value=twist,limit=limit))
   if flex>165:issues.append(dict(kind='joint_overfold',bone=joint,region=region(joint),value=flex))
   if flex>limits.hinge_near_straight_deg and reversal is not None and reversal>hinge_limit:issues.append(dict(kind='hinge_plane_reversal',bone=joint,region=region(joint),value=reversal))
 for n,limit in [('Neck',limits.neck_delta_deg),('Head',limits.head_delta_deg)]:
  if n not in ix:continue
  i=ix[n];p=parents[i];a=np.linalg.solve(base[p,:3,:3],base[i,:3,:3]);b=np.linalg.solve(world[p,:3,:3],world[i,:3,:3]);change=float(np.degrees(Rotation.from_matrix(rotation(np.linalg.solve(a,b))).magnitude()))
  details[n]=dict(delta_deg=change)
  if change>limit:issues.append(dict(kind='neck_head_rotation',bone=n,region='head',value=change,limit=limit))
 if all(n in ix for n in ['Spine2','Neck','Head']):
  cervical=bend(world[[ix['Spine2'],ix['Neck'],ix['Head']],:3,3]);details['cervical_chain']=dict(bend_deg=cervical)
  if cervical>limits.cervical_bend_deg:issues.append(dict(kind='cervical_overbend',region='head',value=cervical,limit=limits.cervical_bend_deg))
 return dict(issues=issues,joints=details,reference='anatomical neutral axes' if anatomical_neutral else 'source-local bend planes',calibrated=anatomical_neutral,source_pose_baked_into_rest=bool(baked))

class MeshPolicy:
 def __init__(self,vertices,faces,weights,names,torso,limits=LIMITS,joint_positions=None):
  self.v=vertices;self.f=faces;self.weights=weights;self.scale=torso;self.limits=limits
  self.edges=faces[:,[[0,1],[1,2],[2,0]]];e=self.edges;self.length=np.linalg.norm(vertices[e[:,:,0]]-vertices[e[:,:,1]],axis=2)
  self.normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]);self.area=np.linalg.norm(self.normals,axis=1)/2
  self.regions=sorted(set(map(region,names)));mass=np.stack([weights[:,[i for i,n in enumerate(names) if region(n)==r]].sum(1) for r in self.regions],axis=1)
  self.owner=np.array(self.regions)[mass[faces].mean(1).argmax(1)];self.faceweights=weights[faces].mean(1)
  edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);fi=np.tile(np.arange(len(faces)),3)
  order=np.lexsort((edges[:,1],edges[:,0]));ee=edges[order];same=np.flatnonzero((ee[1:]==ee[:-1]).all(1))
  self.adj_a=fi[order[same]];self.adj_b=fi[order[same+1]];self.adj_edge=ee[same]
  self.patches={} 
  if joint_positions is not None:
   centers=vertices[faces].mean(1)
   for i,n in enumerate(names):
    if n not in ['LeftArm','RightArm','LeftForeArm','RightForeArm','LeftLeg','RightLeg','LeftHand','RightHand']:continue
    mask=(np.linalg.norm(centers-joint_positions[i],axis=1)<limits.local_patch_radius_torso*torso)&(self.owner==region(n))&(self.area>1e-12)
    if mask.any():self.patches[n]=mask
 def measure(self,vertices,transforms):
  if not np.isfinite(vertices).all():return dict(issues=[dict(kind='nonfinite_mesh')])
  f=self.f;e=self.edges;lim=self.limits
  length=np.linalg.norm(vertices[e[:,:,0]]-vertices[e[:,:,1]],axis=2)
  excess=np.maximum(np.maximum(length-2*self.length,.5*self.length-length),0).max(1)/self.scale
  normals=np.cross(vertices[f[:,1]]-vertices[f[:,0]],vertices[f[:,2]]-vertices[f[:,0]]);area=np.linalg.norm(normals,axis=1)/2
  expected=np.einsum('fn,nij,fj->fi',self.faceweights,transforms[:,:3,:3],self.normals,optimize=True)
  inverted=(np.einsum('fi,fi->f',expected,normals)<0)&(self.area>1e-12)
  collapsed=(area<lim.collapsed_face_ratio*self.area)&(self.area>1e-12)
  rows={};issues=[]
  from release_policy import largest_component_area
  for r in self.regions:
   mask=self.owner==r;total=float(max(self.area[mask].sum(),1e-12))
   damaged=mask&(excess>lim.edge_excess_torso)
   row=dict(affected_area_fraction=float(self.area[damaged].sum()/total),peak_excess_torso=float(excess[mask].max(initial=0)),collapsed_area_fraction=float(self.area[mask&collapsed].sum()/total),inverted_area_fraction=float(self.area[mask&inverted].sum()/total));row.update(largest_inverted_component_fraction=largest_component_area(f,mask&inverted,self.area)/total,largest_collapsed_component_fraction=largest_component_area(f,mask&collapsed,self.area)/total,largest_stretched_component_fraction=largest_component_area(f,damaged,self.area)/total);rows[r]=row
   severe_stretch=(row['peak_excess_torso']>lim.peak_edge_excess_torso and row['affected_area_fraction']>lim.regional_damaged_area) or row['peak_excess_torso']>lim.absolute_peak_excess_torso
   severe_fold=row['largest_inverted_component_fraction']>lim.connected_inverted_area or row['inverted_area_fraction']>lim.total_inverted_area
   if severe_stretch or severe_fold or (row['largest_stretched_component_fraction']>lim.connected_stretch_area and row['peak_excess_torso']>lim.connected_stretch_peak_torso) or row['collapsed_area_fraction']>lim.regional_damaged_area:issues.append(dict(kind='surface_damage',region=r,metrics=row))
  patches={}
  for n,mask in self.patches.items():
   total=float(self.area[mask].sum())
   cosine=np.einsum('fi,fi->f',expected,normals)/np.maximum(np.linalg.norm(expected,axis=1)*np.linalg.norm(normals,axis=1),1e-12)
   inv=mask&(cosine<-.25)
   normal_unit=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
   sharp=(np.einsum('ij,ij->i',normal_unit[self.adj_a],normal_unit[self.adj_b])<np.cos(np.radians(100)))&(mask[self.adj_a]|mask[self.adj_b])
   crease_length=float(np.linalg.norm(vertices[self.adj_edge[sharp,0]]-vertices[self.adj_edge[sharp,1]],axis=1).sum()/self.scale)
   connected=largest_component_area(f,inv,self.area);inverted_area=float(self.area[inv].sum())
   patch=dict(sharp_crease_length_torso=crease_length,region=region(n),area_torso2=total/self.scale**2,inverted_fraction=inverted_area/total,connected_inverted_fraction=connected/total,connected_inverted_area_torso2=connected/self.scale**2)
   patches[n]=patch
   # Both concentration AND physical area are required: a seam/sliver is not collapse.
   concentrated=(connected/total>lim.local_inverted_component_fraction and connected/self.scale**2>lim.local_damage_area_torso2)
   widespread=(inverted_area/total>lim.local_inverted_total_fraction and inverted_area/self.scale**2>lim.local_damage_area_torso2)
   if ((concentrated or widespread) and crease_length>.05) or crease_length>lim.sharp_crease_length_torso:issues.append(dict(kind='joint_surface_fold',bone=n,region=region(n),metrics=patch))
  return dict(issues=issues,regions=rows,joint_patches=patches)

def relationship_errors(world,ix,cam,torso,relations):
 errors=[]
 for r in relations:
  if r.get('confidence')!='clear':continue
  near,far=r['near'],r['far'];depth=float(((world[ix[near],:3,3]-world[ix[far],:3,3])@cam)[2]/torso)
  if depth<LIMITS.depth_margin_torso:errors.append(dict(kind='depth_order',near=near,far=far,value=depth,required=LIMITS.depth_margin_torso))
 return errors

def order_satisfied(candidate_depth,required_sign,margin=.03):
 return required_sign in (-1,1) and candidate_depth*required_sign>=margin

def policy_fingerprint():
 import hashlib,json
 return hashlib.sha256(json.dumps(asdict(LIMITS),sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def safety_margin_penalty(articulation_diagnostic):
 """Prefer a comfortable cervical margin when fit errors are close, without
 rejecting an otherwise safe sole candidate for a ranking preference."""
 cervical=articulation_diagnostic.get('joints',{}).get('cervical_chain',{}).get('bend_deg',0.)
 return max(0.,float(cervical)-45.)*1.2
