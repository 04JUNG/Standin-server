"""Compare perspective against equally refitted orthographic framing using held-out joints."""
import numpy as np
from scipy.optimize import least_squares
from core import project

def select_lens(cj,cv,anchor,ids,goal,torso,scale,offset):
    ids=np.array(ids);goal=np.array(goal);offset=np.array(offset);maxq=min(1/3,.8*torso/max(float((cv[:,2]-anchor[2]).max()),1e-6))
    def fit(mask,perspective):
        def predict(x,points):return project(points,anchor,x[0]/torso,np.exp(x[1]),x[2:])
        def residual(x):return (predict(x,cj[ids[mask]])-goal[mask]).ravel()
        if perspective:
            lo=np.r_[0,np.log(scale*.8),offset-60];hi=np.r_[maxq,np.log(scale*1.2),offset+60];x=np.r_[maxq*.1,np.log(scale),offset]
            r=least_squares(residual,x,bounds=(lo,hi),max_nfev=150)
        else:
            def res(y):return residual(np.r_[0,y])
            r=least_squares(res,np.r_[np.log(scale),offset],bounds=(np.r_[np.log(scale*.8),offset-60],np.r_[np.log(scale*1.2),offset+60]),max_nfev=150);r.x=np.r_[0,r.x]
        if not r.success:raise ValueError('Lens solver failed')
        return r.x,predict
    original=project(cj[ids],anchor,0,scale,offset);original_rmse=float(np.sqrt(np.mean((original-goal)**2)))
    if len(ids)<6:return dict(selected='orthographic',reason='insufficient_targets',inverse_distance=0,pixel_scale=float(scale),offset=offset.tolist(),base_rmse=original_rmse)
    mask=np.ones(len(ids),bool);xo,pred=fit(mask,False);xp,_=fit(mask,True);eo=[];ep=[]
    for i in range(len(ids)):
        mask=np.arange(len(ids))!=i
        a,_=fit(mask,False);b,_=fit(mask,True);eo.extend((pred(a,cj[ids[i:i+1]])-goal[i]).ravel());ep.extend((pred(b,cj[ids[i:i+1]])-goal[i]).ravel())
    co=float(np.sqrt(np.mean(np.square(eo))));cp=float(np.sqrt(np.mean(np.square(ep))));trainp=float(np.sqrt(np.mean((pred(xp,cj[ids])-goal)**2)))
    gain=co-cp;accept=gain>1 and cp<co*.92 and xp[0]<maxq*.95 and xp[0]>.005 and trainp<=original_rmse and np.ptp(cj[ids,2])/torso>.15
    # Orthographic fallback is exact, not subtly reframed by the diagnostic fit.
    chosen=xp if accept else np.r_[0,np.log(scale),offset]
    return dict(selected='perspective' if accept else 'orthographic',reason='heldout_gain' if accept else 'no_reliable_heldout_gain',inverse_distance=float(chosen[0]/torso),pixel_scale=float(np.exp(chosen[1])),offset=chosen[2:].tolist(),base_rmse=original_rmse,cv_ortho_rmse=co,cv_perspective_rmse=cp,cv_gain=gain,candidate_q=float(xp[0]),candidate_rmse=trainp,candidate_inverse_distance=float(xp[0]/torso),candidate_scale=float(np.exp(xp[1])),candidate_offset=xp[2:].tolist(),refit_ortho_rmse=float(np.sqrt(np.mean((pred(xo,cj[ids])-goal)**2))),pose_frozen=True,camera_orientation_frozen=True)
