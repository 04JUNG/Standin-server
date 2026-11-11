"""Offline 004/006/010 integration. No reference pose or case-specific constants."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow

import sys,copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_manual_004'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_manual_006'))
from geometry import prepare,fk,frame,render_projected
from perspective import project

JOINTS={'LeftArm':5,'RightArm':6,'LeftForeArm':7,'RightForeArm':8,'LeftHand':9,'RightHand':10,'LeftUpLeg':11,'RightUpLeg':12,'LeftLeg':13,'RightLeg':14,'LeftFoot':15,'RightFoot':16}

def targets(observation,size):
    kp=np.asarray(observation['keypoints'],float);scores=np.asarray(observation['scores'],float)
    if kp.shape!=(17,2) or scores.shape!=(17,) or not np.isfinite(kp).all() or not np.isfinite(scores).all():raise ValueError('Invalid COCO17')
    width,height=size;ratio=min(535/width,687/height);pad=np.array([(535-width*ratio)/2,(687-height*ratio)/2])
    mask=np.asarray(observation.get('joint_mask',scores>=.3),bool)
    if mask.shape!=(17,):raise ValueError('Invalid joint mask')
    valid=mask&(kp[:,0]>width*.03)&(kp[:,0]<width*.97)&(kp[:,1]>height*.03)&(kp[:,1]<height*.95)
    result={n:kp[i]*ratio+pad for n,i in JOINTS.items() if valid[i]}
    if valid[5] and valid[6]:result['Neck']=(kp[5]+kp[6])/2*ratio+pad
    return result,kp*ratio+pad,valid

def validate_relations(doc,image_hash,names):
    if not doc:return []
    if doc.get('image_sha256')!=image_hash or doc.get('frame')!='camera_toward_viewer_positive':raise ValueError('Unbound depth document')
    if doc.get('source') not in ['user_review','image_model']:raise ValueError('Missing depth provenance')
    rels=doc.get('relations',[]);graph={}
    for r in rels:
        if r['near'] not in names or r['far'] not in names or r['near']==r['far'] or r['strength'] not in ['clear','weak']:raise ValueError('Invalid depth relation')
        graph.setdefault(r['near'],set()).add(r['far'])
    def visit(n,path):
        if n in path:raise ValueError('Depth cycle')
        for nxt in graph.get(n,[]):visit(nxt,path|{n})
    for n in graph:visit(n,set())
    return rels

def depth_errors(world,cam,ix,torso,rels):
    # Ordinal lower bound, not exact case-specific world distance.
    return np.array([max(0,(.12 if r['strength']=='clear' else .05)-((world[ix[r['near']],:3,3]-world[ix[r['far']],:3,3])@cam)[2]/torso) for r in rels])

def camera_stage(world,pr,basis,origin,ix,goal,cue,image_hash):
    output=copy.deepcopy(pr);diag={'changed':False,'reason':'no_camera_evidence'}
    if not cue:return output,diag
    if cue.get('image_sha256')!=image_hash:raise ValueError('Camera image hash mismatch')
    if cue['raw'].get('confidence')=='weak':return output,dict(diag,reason='weak_view_preserve_orientation',framing_already_fitted=True)
    c=cue['raw'];ranges={'frontal':(0,10),'slight_oblique':(15,35),'three_quarter':(30,60),'profile':(65,85)}
    if c.get('person_count')!=1 or c.get('torso_view') not in ranges or c.get('confidence') not in ['clear','weak'] or c.get('nearer_shoulder') not in ['equal','screen_left','screen_right']:
        return output,dict(diag,reason='uncertain_camera')
    if not all(n in goal for n in ['LeftArm','RightArm']):return output,dict(diag,reason='missing_shoulders')
    if c['nearer_shoulder']=='equal' and c['torso_view']!='frontal':return output,dict(diag,reason='ambiguous_sign')
    left_on_right=goal['LeftArm'][0]>goal['RightArm'][0];left_near=(c['nearer_shoulder']=='screen_right')==left_on_right
    low,high=ranges[c['torso_view']];interval=np.radians([low,high] if left_near else [-high,-low])
    def angle(cam):
        s=(world[ix['LeftArm'],:3,3]-world[ix['RightArm'],:3,3])@cam
        return np.arctan2(s[2],s[0])
    oldangle=angle(np.array(pr['camera']))
    if interval[0]-np.radians(15)<=oldangle<=interval[1]+np.radians(15):return output,dict(diag,reason='existing_view_compatible',shoulder_yaw_deg=float(np.degrees(oldangle)))
    ids=[ix[n] for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg'] if n in goal];tar=np.array([goal[n] for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg'] if n in goal])
    pitch=np.radians({'above':[-30,-5],'level':[-8,8],'below':[5,30],'uncertain':[-40,40]}.get(c.get('elevation'),[-40,40]))
    def err(v,ab):return max(ab[0]-v,0)+min(ab[1]-v,0)
    def calc(x):
        cam=basis@Rotation.from_rotvec(x[:3]).as_matrix();s=np.exp(x[3]);off=x[4:]
        xy=((world[:,:3,3]-origin)@cam)[:,:2]*[s,-s]+off
        return cam,s,off,xy
    strength=25 if c['confidence']=='clear' else 8
    def loss(x):
        cam,s,off,xy=calc(x);an=angle(cam)
        return np.r_[(xy[ids]-tar).ravel()/25,err(an,interval)*strength,err(x[0],pitch)*strength,(an-np.mean(interval))*2,x[:3]*.5]
    x=np.r_[Rotation.from_matrix(basis.T@np.array(pr['camera'])).as_rotvec(),np.log(pr['pixel_scale']),pr['offset']]
    lo=np.r_[[-.7,-1.5,-.8],np.log(100),[-200,-200]];hi=np.r_[[.7,1.5,.8],np.log(1600),[800,1200]]
    r=least_squares(loss,np.clip(x,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=200)
    if not r.success and not shadow():return output,dict(diag,reason='camera_solver_failed',converged=False,evaluations=r.nfev)
    cam,s,off,xy=calc(r.x);output.update(camera=cam.tolist(),pixel_scale=float(s),offset=off.tolist())
    return output,dict(changed=True,converged=bool(r.success),evaluations=r.nfev,reason='image_view_constraint',cue=c,shoulder_yaw_deg=float(np.degrees(angle(cam))),camera_delta_deg=float(np.degrees(Rotation.from_matrix(np.array(pr['camera']).T@cam).magnitude())))

def pose_stage(local,parents,base,ix,world,pr,goal,kp,valid,torso,rels,camera_changed):
    cam=np.array(pr['camera']);origin=np.array(pr['origin']);scale=pr['pixel_scale'];offset=np.array(pr['offset']);edits=copy.deepcopy(pr['edits'])
    moving=[]
    for side in ['Left','Right']:
        if side+'Arm' in goal and side+'Hand' in goal:moving += [side+'Arm',side+'ForeArm']
        elif side+'Arm' in goal and side+'ForeArm' in goal:moving += [side+'Arm']
        if side+'Foot' in goal and any(n in goal for n in ['LeftUpLeg','RightUpLeg']):moving += [side+'UpLeg',side+'Leg']
        elif side+'UpLeg' in goal and side+'Leg' in goal:moving += [side+'UpLeg']
    head_valid=bool(camera_changed and np.min(valid[:5]) and kp[1,0]-kp[2,0]>5 and kp[3,0]-kp[4,0]>10)
    if head_valid:moving+=['Neck','Head']
    # Relations involving two torso anchors belong to the camera stage only.
    limb_rels=[r for r in rels if not all(n in ['LeftArm','RightArm','Hips','Spine','Spine1','Spine2'] for n in [r['near'],r['far']])]
    if not moving:return world,edits,{'solved':False,'reason':'no_valid_chains'}
    start=np.radians([edits.get(n,[0,0,0]) for n in moving]).ravel();fixed={n:v for n,v in edits.items() if n not in moving}
    ids=[ix[n] for n in goal];tar=np.array(list(goal.values()))
    if head_valid:
        eyes=kp[1]-kp[2];ears=kp[3]-kp[4];center=(kp[3]+kp[4])/2
        yaw=np.arcsin(np.clip(2*(kp[0,0]-center[0])/abs(ears[0]),-.8,.8));roll=np.arctan2(-eyes[1],eyes[0]);head_goal=Rotation.from_euler('zy',[roll,yaw]).as_matrix()
    def calc(x):return fk(local,parents,[ix[n] for n in moving+list(fixed)],np.r_[x,np.radians(list(fixed.values())).ravel()])
    bend_priors=[]
    for side in ['Left','Right']:
        for chain in [('Arm','ForeArm','Hand'),('UpLeg','Leg','Foot')]:
            a,b,c=[ix[side+n] for n in chain]
            if side+chain[0] not in moving:continue
            u=base[b,:3,3]-base[a,:3,3];v=base[c,:3,3]-base[b,:3,3];cross=np.cross(u,v);sine=np.linalg.norm(cross)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12)
            if sine>np.sin(np.radians(5)):
                normal=np.linalg.solve(base[a,:3,:3],cross);normal/=np.linalg.norm(normal);bend_priors.append((a,b,c,normal))
    def loss(x):
        w=calc(x);xy=((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+offset
        hr=Rotation.from_matrix(head_goal.T@cam.T@w[ix['Head'],:3,:3]).as_rotvec()*4 if head_valid else []
        hinge=[]
        for a,b,c,normal in bend_priors:
            u=w[b,:3,3]-w[a,:3,3];v=w[c,:3,3]-w[b,:3,3];cross=np.linalg.solve(w[a,:3,:3],np.cross(u,v));hinge.append(max(0.,-float(np.dot(cross,normal)))/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12)*8)
        return np.r_[(xy[ids]-tar).ravel()/25,(x-start)*.9,depth_errors(w,cam,ix,torso,limb_rels)*12,hr,hinge]
    # Increment bounds preserve the accepted local twist rather than force a reset.
    limit=np.radians(60);r=least_squares(loss,start,bounds=(start-limit,start+limit),max_nfev=250)
    if not r.success and not shadow():return world,edits,{'solved':False,'reason':'pose_solver_failed','converged':False,'evaluations':r.nfev}
    w=calc(r.x);edits.update({n:np.degrees(r.x[i*3:i*3+3]).tolist() for i,n in enumerate(moving)})
    return w,edits,dict(solved=bool(r.success),converged=bool(r.success),evaluations=r.nfev,active=moving,camera_frozen=True,torso_frozen=True,head_goal_used=head_valid,bound_hit_bones=[n for i,n in enumerate(moving) if np.any(abs(r.x[i*3:i*3+3]-start[i*3:i*3+3])>limit-1e-4)],depth_error_before=depth_errors(world,cam,ix,torso,limb_rels).tolist(),depth_error_after=depth_errors(w,cam,ix,torso,limb_rels).tolist())

def shoulder_stage(local,parents,ix,world,edits):
    # 006: reset clavicle roll, minimal swing initialization, preserve distal targets.
    from copy import deepcopy
    e=deepcopy(edits)
    def calc(ed):return fk(local,parents,[ix[n] for n in ed],np.radians(list(ed.values())).ravel())
    def orient(ed,n,desired):
        w=calc(ed);old=Rotation.from_rotvec(np.radians(ed.get(n,[0,0,0]))).as_matrix();delta=np.linalg.solve(w[ix[n],:3,:3],desired)
        ed[n]=np.degrees(Rotation.from_matrix(old@delta).as_rotvec()).tolist()
    for side in ['Left','Right']:
        e[side+'Shoulder']=[0,0,0]
        for n,child in [(side+'Arm',side+'ForeArm'),(side+'ForeArm',side+'Hand')]:
            e[n]=[0,0,0];w=calc(e);v=w[ix[child],:3,3]-w[ix[n],:3,3];v/=np.linalg.norm(v);t=world[ix[child],:3,3]-w[ix[n],:3,3];t/=max(np.linalg.norm(t),1e-12)
            axis=np.cross(v,t);norm=np.linalg.norm(axis)
            if norm<1e-8:
                if np.dot(v,t)>0:rot=np.eye(3)
                else:
                    axis=np.cross(v,np.eye(3)[np.argmin(abs(v))]);axis/=np.linalg.norm(axis);rot=Rotation.from_rotvec(axis*np.pi).as_matrix()
            else:rot=Rotation.from_rotvec(axis/norm*np.arctan2(norm,np.dot(v,t))).as_matrix()
            orient(e,n,rot@w[ix[n],:3,:3])
    active=[s+n for s in ['Left','Right'] for n in ['Shoulder','Arm','ForeArm']];start=np.radians([e[n] for n in active]).ravel();ids=[ix[s+n] for s in ['Left','Right'] for n in ['ForeArm','Hand']];target=world[ids,:3,3]
    def candidate(x):
        ed=deepcopy(e);ed.update({n:np.degrees(x[3*i:3*i+3]).tolist() for i,n in enumerate(active)});return ed,calc(ed)
    def loss(x):
        _,w=candidate(x);return np.r_[(w[ids,:3,3]-target).ravel()*35,(x-start),x[[0,1,2,9,10,11]]*4]
    limits=np.repeat(np.radians([12,180,180,12,180,180]),3);r=least_squares(loss,np.clip(start,-limits+1e-8,limits-1e-8),bounds=(-limits,limits),max_nfev=200)
    if not r.success and not shadow():return world,edits,{'applied':False,'converged':False,'evaluations':r.nfev}
    e,w=candidate(r.x)
    for n in ['LeftHand','RightHand']:orient(e,n,world[ix[n],:3,:3])
    w=calc(e)
    protected=[ix[n] for n in ['Hips','Spine','Spine1','Spine2','Neck','Head','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']]
    assert np.allclose(w[protected],world[protected],rtol=0,atol=1e-10)
    return w,e,dict(applied=True,converged=bool(r.success),evaluations=r.nfev,protected_world_max_error=float(np.max(abs(w[protected]-world[protected]))),endpoint_max_shift=float(np.linalg.norm(w[ids,:3,3]-target,axis=1).max()))

from skin_shared import skin

def deformation(vertices,original,faces):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    before=np.linalg.norm(original[edges[:,0]]-original[edges[:,1]],axis=1);after=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1);r=after[before>1e-7]/before[before>1e-7]
    return dict(extreme_edges=int(np.sum((r<.5)|(r>2))),ratio_min=float(r.min()),ratio_max=float(r.max()))

def geometry_accept(before,after,oldrmse,newrmse,depth_before,depth_after,camera_changed):
    reasons=[]
    if after['extreme_edges']>before['extreme_edges']:reasons.append('edge_stretch_regression')
    if after['ratio_max']>max(2,before['ratio_max'])*1.05:reasons.append('max_edge_ratio_regression')
    if not np.isfinite(newrmse):reasons.append('nonfinite_error')
    if newrmse>oldrmse+1:reasons.append('target_error_regression')
    if depth_after>depth_before+1e-5:reasons.append('depth_regression')
    if not camera_changed and newrmse>=oldrmse-.5 and depth_before-depth_after<.02:reasons.append('no_material_gain')
    return not reasons,reasons

def partial_pose(local,parents,base,ix,original_world,original_pr,proposal_pr,goal,torso,rels,v,f,weights):
    """Independent limb rollback: a bad arm must not discard a good leg."""
    accepted=copy.deepcopy(original_pr);w=original_world.copy();ids=[ix[n] for n in goal];target=np.array(list(goal.values()));cam=np.array(accepted['camera']);o=accepted['origin'];s=accepted['pixel_scale'];off=accepted['offset']
    def metrics(world):
        verts=skin(v,weights,world,base);deform=deformation(verts,v,f);xy=((world[:,:3,3]-o)@cam)[:,:2]*[s,-s]+off
        return verts,deform,float(np.sqrt(np.mean((xy[ids]-target)**2))),float(np.linalg.norm(depth_errors(world,cam,ix,torso,rels)))
    vertices,d,rmse,dep=metrics(w);groups={};history=[]
    for side in ['Left','Right']:
        groups[side+'leg']=[side+'UpLeg',side+'Leg'];groups[side+'arm']=[side+'Shoulder',side+'Arm',side+'ForeArm',side+'Hand']
    for group,joints in groups.items():
        start=copy.deepcopy(accepted['edits']);chosen=None
        for fraction in [1.,.5,.25]:
            ed=copy.deepcopy(start)
            for n in joints:
                old=np.array(start.get(n,[0,0,0]));new=np.array(proposal_pr['edits'].get(n,old));ed[n]=(old+(new-old)*fraction).tolist()
            nw=fk(local,parents,[ix[n] for n in ed],np.radians(list(ed.values())).ravel());nv,nd,nr,ndep=metrics(nw)
            ok,reasons=geometry_accept(d,nd,rmse,nr,dep,ndep,False)
            if ok:
                chosen=(ed,nw,nv,nd,nr,ndep,fraction);break
        if chosen:
            ed,w,vertices,d,rmse,dep,fraction=chosen;accepted['edits']=ed;history.append(dict(group=group,fraction=fraction))
    return w,accepted,vertices,d,history

def orientation_stage(local,parents,ix,world,pr,goal,cue):
    """006 intuition: independent head turn and visible toe direction, no per-cut angles."""
    evidence=(cue or {}).get('orientation',{});e=copy.deepcopy(pr['edits']);cam=np.array(pr['camera']);changes=[]
    def calc():return fk(local,parents,[ix[n] for n in e],np.radians(list(e.values())).ravel())
    def orient(n,desired):
        w=calc();old=Rotation.from_rotvec(np.radians(e.get(n,[0,0,0]))).as_matrix();delta=np.linalg.solve(w[ix[n],:3,:3],desired);e[n]=np.degrees(Rotation.from_matrix(old@delta).as_rotvec()).tolist()
    if evidence.get('person_count')!=1:return world,pr,{'applied':False}
    head=evidence.get('head',{})
    if head.get('confidence')=='clear' and head.get('direction') in ['screen_left','screen_right','front'] and head.get('turn') in ['profile','oblique','front']:
        yaw={'profile':75,'oblique':30,'front':0}[head['turn']] * (-1 if head['direction']=='screen_left' else 1)
        desired=cam@Rotation.from_euler('y',yaw,degrees=True).as_matrix();w=calc();delta=Rotation.from_matrix(desired@w[ix['Head'],:3,:3].T).as_rotvec()
        orient('Neck',Rotation.from_rotvec(delta*.6).as_matrix()@w[ix['Neck'],:3,:3]);orient('Head',desired);changes.append('head_neck')
    if all(n in goal for n in ['LeftFoot','RightFoot']):
        order=sorted(['LeftFoot','RightFoot'],key=lambda n:goal[n][0]);mapping=dict(zip(['screen_left','screen_right'],order))
        feet=evidence.get('feet',[]);positions=[x.get('position') for x in feet]
        if len(positions)!=len(set(positions)):feet=[] # Ambiguous ownership: abstain, never silently swap feet.
        for foot in feet:
            if foot.get('confidence')!='clear' or foot.get('position') not in mapping or foot.get('tip_direction') not in ['down','down_away']:continue
            n=mapping[foot['position']];child=n.replace('Foot','ToeBase');w=calc();v=w[ix[child],:3,3]-w[ix[n],:3,3];v/=np.linalg.norm(v);t=cam@np.array([0,-.8,-.6] if foot['tip_direction']=='down_away' else [0,-1,0]);axis=np.cross(v,t);norm=np.linalg.norm(axis)
            if norm<1e-8:continue
            rotation=Rotation.from_rotvec(axis/norm*min(np.arctan2(norm,np.dot(v,t)),np.pi/2)).as_matrix();orient(n,rotation@w[ix[n],:3,:3]);changes.append(n)
    result=copy.deepcopy(pr);result['edits']=e;return calc(),result,dict(applied=bool(changes),changed=changes,source='image_model_categorical_orientation',numeric_angles='shared category priors, not model predictions')
