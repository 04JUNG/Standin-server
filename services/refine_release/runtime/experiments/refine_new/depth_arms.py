"""Depth-first arm IK: bounded chain depth, minimum-swing initialization, fixed legs."""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from core import fk

def fit_depth_arms(local,parents,ix,world,pr,goal,relations):
    e=copy.deepcopy(pr['edits']);cam=np.array(pr['camera']);origin=np.array(pr['origin']);scale=pr['pixel_scale'];offset=np.array(pr['offset']);diagnostics=[]
    def calc(ed):return fk(local,parents,[ix[n] for n in ed],np.radians(list(ed.values())).ravel())
    def orient(ed,n,rotation):
        w=calc(ed);old=Rotation.from_rotvec(np.radians(ed.get(n,[0,0,0]))).as_matrix();ed[n]=np.degrees(Rotation.from_matrix(old@np.linalg.solve(w[ix[n],:3,:3],rotation)).as_rotvec()).tolist()
    for side in ['Left','Right']:
        arm,elbow,hand=[side+n for n in ['Arm','ForeArm','Hand']]
        pair=[r for r in relations if {r['near'],r['far']}=={arm,hand}]
        if not pair or not all(n in goal for n in [arm,elbow,hand]):continue
        sign=1 if pair[0]['near']==arm else -1
        before=calc(e);length=sum(np.linalg.norm(before[ix[b],:3,3]-before[ix[a],:3,3]) for a,b in [(arm,elbow),(elbow,hand)])
        active=[side+'Shoulder',arm,elbow]
        for n in active:e[n]=[0,0,0]
        w=calc(e);zshoulder=((w[ix[arm],:3,3]-origin)@cam)[2]
        for n,child,fraction in [(arm,elbow,.15),(elbow,hand,.3)]:
            w=calc(e);target_cam=np.r_[(goal[child]-offset)/[scale,-scale],zshoulder-sign*fraction*length];target=origin+cam@target_cam
            v=w[ix[child],:3,3]-w[ix[n],:3,3];v/=np.linalg.norm(v);t=target-w[ix[n],:3,3];t/=np.linalg.norm(t);axis=np.cross(v,t);norm=np.linalg.norm(axis)
            if norm<1e-8:
                axis=np.cross(v,np.eye(3)[np.argmin(abs(v))]);axis/=np.linalg.norm(axis)
            else:axis/=norm
            rot=Rotation.from_rotvec(axis*np.arctan2(norm,np.dot(v,t))).as_matrix();orient(e,n,rot@w[ix[n],:3,:3])
        start=np.radians([e[n] for n in active]).ravel();ids=[ix[n] for n in [arm,elbow,hand]];target=np.array([goal[n] for n in [arm,elbow,hand]])
        def candidate(x):
            ed=copy.deepcopy(e);ed.update({n:np.degrees(x[3*i:3*i+3]).tolist() for i,n in enumerate(active)});return ed,calc(ed)
        def depths(w):
            z=((w[ids,:3,3]-origin)@cam)[:,2];return sign*(z[0]-z[1:])/length
        def loss(x):
            ed,w=candidate(x);xy=((w[ids,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+offset;de,dw=depths(w)
            upper=w[ix[elbow],:3,3]-w[ix[arm],:3,3];lower=w[ix[hand],:3,3]-w[ix[elbow],:3,3];cos=np.dot(upper,lower)/np.linalg.norm(upper)/np.linalg.norm(lower)
            # Both minimum and maximum: do not satisfy "away" by folding farther indefinitely.
            bands=np.r_[max(0,.05-de),max(0,de-.35),max(0,.15-dw),max(0,dw-.55),max(0,de-dw+.02)]
            return np.r_[(xy-target).ravel()/15,bands*35,(x-start)*.35,x[:3]*4,max(0,-cos)*10]
        limits=np.repeat(np.radians([12,180,180]),3);r=least_squares(loss,np.clip(start,-limits+1e-8,limits-1e-8),bounds=(-limits,limits),max_nfev=350)
        if not r.success:raise RuntimeError(r.message)
        e,w=candidate(r.x);orient(e,hand,before[ix[hand],:3,:3]);w=calc(e);de,dw=depths(w)
        diagnostics.append(dict(side=side,direction='away' if sign==1 else 'toward',elbow_signed_depth_arm_lengths=float(de),wrist_signed_depth_arm_lengths=float(dw),depth_satisfied=bool(de>=.04 and dw>=.14 and de<=.37 and dw<=.57),source='explicit_relation',initialized='minimum_swing',camera_frozen=True))
    result=calc(e)
    protected=[ix[n] for n in ['Hips','Spine','Spine1','Spine2','Neck','Head','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']]
    assert np.allclose(result[protected],world[protected],atol=1e-10,rtol=0)
    output=copy.deepcopy(pr);output['edits']=e
    return result,output,diagnostics
