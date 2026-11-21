"""Small, seed-relative pose corrections. No case labels, desired answers or vertex edits."""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from common import skin,write
from policy import region,articulation,LIMITS

def face_score(v,faces,reference,transforms,faceweights,normal0,area0,scale):
 tri=v[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);expected=np.einsum('fn,nij,fj->fi',faceweights,transforms[:,:3,:3],normal0,optimize=True)
 expected/=np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-12)
 signed=np.einsum('fi,fi->f',expected,cross)/np.maximum(2*area0,1e-12)
 e=faces[:,[[0,1],[1,2],[2,0]]];length=np.linalg.norm(v[e[:,:,0]]-v[e[:,:,1]],axis=2)
 excess=np.maximum(np.maximum(length-2*reference,.5*reference-length),0)/scale
 return signed,excess

def hinge_seed(c,state,moving):
 """Rotate the bend around the root/end axis, keeping the limb endpoint fixed.
 This supplies a continuous IK start when a nearly straight hinge flips sign.
 """
 ed=copy.deepcopy(state['pr']['edits']);world=c.world(ed)
 from policy import unit
 def orient(name,desired):
  w=c.world(ed);i=c.ix[name];old=Rotation.from_rotvec(np.radians(ed.get(name,[0,0,0]))).as_matrix()
  ed[name]=np.degrees(Rotation.from_matrix(old@np.linalg.solve(w[i,:3,:3],desired)).as_rotvec()).tolist()
 def swing(a,b):
  a=unit(a);b=unit(b);axis=np.cross(a,b);s=np.linalg.norm(axis)
  return np.eye(3) if s<1e-10 else Rotation.from_rotvec(axis/s*np.arctan2(s,np.dot(a,b))).as_matrix()
 art=articulation(world,c.base,c.parents,c.ix,neutral=c.calibration)['joints']
 for side in ['Left','Right']:
  for upper,joint,end in [(side+'UpLeg',side+'Leg',side+'Foot'),(side+'Arm',side+'ForeArm',side+'Hand')]:
   if upper not in moving or joint not in moving:continue
   d=art[joint];expected=d.get('expected_normal_local');limit=LIMITS.knee_repair_target_deg if joint.endswith('Leg') else d.get('hinge_limit_deg',LIMITS.hinge_reversal_deg)-10
   if expected is None or (d['hinge_plane_change_deg'] or 0)<=limit:continue
   w=c.world(ed);i,j,k=[c.ix[n] for n in [upper,joint,end]];a,b,z=w[[i,j,k],:3,3];axis=unit(z-a);actual=unit(np.cross(b-a,z-b));target=w[i,:3,:3]@expected;target=unit(target-axis*np.dot(target,axis))
   theta=np.arctan2(np.dot(axis,np.cross(actual,target)),np.dot(actual,target));theta=np.sign(theta)*max(0,abs(theta)-np.radians(limit)*.8)
   bnew=a+Rotation.from_rotvec(axis*theta).apply(b-a);endrotation=w[k,:3,:3].copy()
   orient(upper,swing(b-a,bnew-a)@w[i,:3,:3]);w=c.world(ed)
   orient(joint,swing(w[k,:3,3]-w[j,:3,3],z-w[j,:3,3])@w[j,:3,:3])
   if end in moving:orient(end,endrotation)
 return ed

def propose(c,seed,startrow=None,suffix="",rotation_budget=None):
 rotation_budget=LIMITS.repair_rotation_deg if rotation_budget is None else rotation_budget
 motion_budget=LIMITS.expanded_repair_motion_torso if rotation_budget>LIMITS.repair_rotation_deg else LIMITS.repair_motion_torso
 state=startrow or seed
 issues=state['diagnostic']['issues'];bad={x.get('region') for x in issues if x['kind'] in ('surface_damage','joint_axial_twist','hinge_plane_reversal','joint_overfold','joint_surface_fold','cervical_overbend')}
 collisions=state['diagnostic']['collision'];witnesses=collisions.get('contact_witnesses',[])
 for pair in collisions['intersections']:
  tokens=pair.split('/')
  arms=[t for t in tokens if t.startswith(('Left_','Right_')) and t.split('_')[1] in ('hand','forearm','upperarm')]
  # Hand/body and hand/leg contacts repair the arm, not the opposing torso/leg.
  for token in arms or tokens:
   if token.startswith(('Left_','Right_')):bad.add(token.split('_')[0]+('_leg' if token.split('_')[1] in ('thigh','shin','foot') else '_arm'))
 moving=[side+n for side in ['Left','Right'] for n in ['Arm','ForeArm','Hand','UpLeg','Leg','Foot'] if side+n in c.ix and region(side+n) in bad]
 if 'head' in bad:moving += [n for n in ['Neck','Head'] if n in c.ix]
 if not moving:return
 # Do not guess the forward/back exit around the head.
 filtered=[]
 for x in witnesses:
  if 'head' in x['pair'] and '_hand' in x['pair']:
   side='Left' if 'Left_hand' in x['pair'] else 'Right'
   if not any({r['near'],r['far']}=={side+'Hand','Head'} for r in c.relations):continue
  filtered.append(x)
 witnesses=filtered
 posed=skin(c.v,c.weights,state['world'],c.base);mp=c.meshpolicy;transforms=state['world']@np.linalg.inv(c.base)
 signed,excess=face_score(posed,c.f,mp.length,transforms,mp.faceweights,mp.normals,mp.area,c.scale)
 score=excess.max(1)+np.maximum(0,.15-signed)*.01
 faceids=[]
 for r in sorted(x for x in bad if x):
  ids=np.flatnonzero(mp.owner==r)
  if len(ids):faceids.extend(ids[np.argsort(score[ids])[-64:]].tolist())
 faceids=np.array(sorted(set(faceids)),int)
 if not len(faceids):return
 allids=set(c.f[faceids].ravel().tolist())
 for x in witnesses:allids.add(x['vertex']);allids.update(x['face'])
 verts=np.array(sorted(allids));lookup={int(v):i for i,v in enumerate(verts)};faces=np.vectorize(lookup.get)(c.f[faceids]);vv=c.v[verts];ww=c.weights[verts]
 vi=np.array([lookup[x['vertex']] for x in witnesses],int);tf=np.array([[lookup[i] for i in x['face']] for x in witnesses],int).reshape(-1,3);bary=np.array([x['bary'] for x in witnesses]).reshape(-1,3)
 initial=copy.deepcopy(state['pr']['edits']);start=np.radians([seed['pr']['edits'].get(n,[0,0,0]) for n in moving]).ravel();seed_ed=hinge_seed(c,state,moving);x0=np.radians([seed_ed.get(n,[0,0,0]) for n in moving]).ravel();cam=seed['cam'];ids=[c.ix[n] for n in c.goal];refxy=c.xy(seed['world'],cam)[ids];refworld=seed['world'][:,:3,3];faceweight=np.sqrt(np.clip(mp.area[faceids]/max(mp.area[faceids].mean(),1e-12),.01,10))
 def calc(x):
  ed=copy.deepcopy(initial);ed.update({n:np.degrees(x[i*3:i*3+3]).tolist() for i,n in enumerate(moving)});return ed,c.world(ed)
 def loss(x):
  ed,w=calc(x);v=skin(vv,ww,w,c.base);signed,exc=face_score(v,faces,mp.length[faceids],w@np.linalg.inv(c.base),mp.faceweights[faceids],mp.normals[faceids],mp.area[faceids],c.scale)
  contact=[]
  if len(vi):
   tri=v[tf];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);p=np.einsum('fi,fij->fj',bary,tri);gap=np.einsum('fi,fi->f',v[vi]-p,normal)/c.scale;contact=np.maximum(0,.003-gap)*1500
  art=articulation(w,c.base,c.parents,c.ix,neutral=c.calibration)['joints'];joint=[]
  for side in ['Left','Right']:
   for name,limit in [(side+'Leg',LIMITS.knee_twist_deg),(side+'ForeArm',LIMITS.elbow_twist_deg)]:
    d=art[name];joint.extend([max(0,d['twist_deg']-(limit-2))/4,max(0,d['bend_deg']-165)/20])
    # The angle of an almost straight chain is singular. Optimize the signed
    # bending vector instead so the solver can pass through a straight pose.
    expected=d.get('expected_normal_local');actual=d.get('actual_normal_local')
    if expected is not None and actual is not None:
     sine=np.sin(np.radians(d['bend_deg']));target=LIMITS.knee_repair_target_deg if name.endswith('Leg') else d.get('hinge_limit_deg',LIMITS.hinge_reversal_deg)-10
     joint.append(max(0.,np.cos(np.radians(target))-float(np.dot(expected,actual)))*sine*32)
    else:joint.append(0.)
  depth=[max(0,LIMITS.depth_margin_torso-((w[c.ix[r['near']],:3,3]-w[c.ix[r['far']],:3,3])@cam['camera'])[2]/c.scale) for r in c.relations]
  shift=(w[:,:3,3]-refworld)/c.scale
  barrier=np.maximum(0,np.linalg.norm(shift,axis=1)-(motion_budget-.002))*1500
  projection=c.xy(w,cam)[ids]-refxy
  projection_barrier=np.maximum(0,np.linalg.norm(projection,axis=1)-2.8)*20 if rotation_budget>LIMITS.repair_rotation_deg else np.zeros(len(ids))
  return np.r_[projection.ravel()/3,projection_barrier,(shift[ids]/motion_budget).ravel(),barrier,(x-start)*2,np.maximum(0,.15-signed)*faceweight*7,np.maximum(0,exc-.004).ravel()*150,np.array(contact),np.array(joint)*4,np.array(depth)*30,max(0,art.get('cervical_chain',{}).get('bend_deg',0)-(LIMITS.cervical_bend_deg-10))/3]
 solved=least_squares(loss,np.clip(x0,start-np.radians(rotation_budget),start+np.radians(rotation_budget)),bounds=(start-np.radians(rotation_budget),start+np.radians(rotation_budget)),max_nfev=60,ftol=1e-5,xtol=1e-5,gtol=1e-5)
 x=solved.x.copy();ed,w=calc(x);clipped=0
 # Enforce the trust region before exact validation, including all descendants.
 while np.linalg.norm(w[:,:3,3]-refworld,axis=1).max()/c.scale>(motion_budget-.001) and clipped<30:
  x=start+(x-start)*.8;ed,w=calc(x);clipped+=1
 pr=copy.deepcopy(seed['pr']);pr['local_repair_origin']=seed['id'];pr['bounded_repair']=rotation_budget>LIMITS.repair_rotation_deg;row=c.add('local-'+seed['id']+suffix,ed,pr,cam)
 write(row['path']/'repair.json',dict(origin=seed['id'],trust_region_backtracks=clipped,converged=bool(solved.success),evaluations=solved.nfev,modified_bones=moving,contact_witness_count=len(witnesses),sampled_faces=len(faceids),surface_inversion_in_objective=True,camera_unchanged=True,torso_unchanged=True,world_joint_motion_torso=float(np.linalg.norm(w[:,:3,3]-refworld,axis=1).max()/c.scale)))
