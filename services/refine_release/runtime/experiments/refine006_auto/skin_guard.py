"""Measured shoulder skin-strain refinement with fixed torso/legs and endpoint anchors."""
import copy
import numpy as np
from scipy.optimize import least_squares
from core import fk,skin

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from solve_budget import solve as budget_solve, eligible
def least_squares(fun,x0,*args,**kwargs):
 return budget_solve(fun,x0,*args,stage='skin',**kwargs)

def refine(local,parents,base,ix,names,world,edits,vertices,faces,weights,torso):
    active=[s+n for s in ['Left','Right'] for n in ['Shoulder','Arm','ForeArm']];ids=[ix[s+n] for s in ['Left','Right'] for n in ['ForeArm','Hand']];goal=world[ids,:3,3]
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    shoulder_mass=weights[:,[ix[s+n] for s in ['Left','Right'] for n in ['Shoulder','Arm']]].sum(1)
    mask=(shoulder_mass[edges].max(1)>.2)&(shoulder_mass[edges].min(1)<.99);edges=edges[mask];lengths=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1);edges=edges[lengths>1e-7];lengths=lengths[lengths>1e-7]
    indices,inverse=np.unique(edges,return_inverse=True);small_edges=inverse.reshape(-1,2);v=vertices[indices];wt=weights[indices]
    start=np.radians([edits.get(n,[0,0,0]) for n in active]).ravel()
    def candidate(x):
        ed=copy.deepcopy(edits);ed.update({n:np.degrees(x[3*i:3*i+3]).tolist() for i,n in enumerate(active)});return ed,fk(local,parents,[ix[n] for n in ed],np.radians(list(ed.values())).ravel())
    def loss(x):
        ed,w=candidate(x);vv=skin(v,wt,w,base);ratios=np.linalg.norm(vv[small_edges[:,0]]-vv[small_edges[:,1]],axis=1)/lengths
        strain=np.r_[np.maximum(ratios-2,0),np.maximum(.5-ratios,0)]
        return np.r_[(w[ids,:3,3]-goal).ravel()/torso*35,(x-start)*.4,strain*3]
    absolute=np.repeat(np.radians([12,180,180]*2),3);lo=np.maximum(start-np.radians(25),-absolute);hi=np.minimum(start+np.radians(25),absolute);start=np.clip(start,lo+1e-9,hi-1e-9)
    r=least_squares(loss,start,bounds=(lo,hi),max_nfev=120)
    if not np.isfinite(r.x).all() or np.sum(loss(r.x)**2)>=np.sum(loss(start)**2):return world,edits,dict(applied=False,reason='no_measured_skin_loss_gain')
    ed,w=candidate(r.x);return w,ed,dict(applied=True,converged=bool(r.success),endpoint_max_shift=float(np.linalg.norm(w[ids,:3,3]-goal,axis=1).max()),sampled_edges=len(edges),nfev=r.nfev)
