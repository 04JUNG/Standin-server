"""P4/P5 bounded source + selected-target solve. No GT or dataset access.

All rounds use the original B0 local transforms, fixed evidence and camera.
Native/skin verification is a later mandatory boundary; this is a proposal.
"""
import hashlib,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiments.refine_v5.goal_probe_v2 import prepare,fk
from experiments.refine_v5.learning_data import BONES,POINTS,frame

PROFILES={
 'conservative12':dict(limb=12.,spine=0.,clavicle=0.,head=8.,wrist=12.,foot=8.),
 'experimental30':dict(limb=30.,spine=6.,clavicle=4.,head=8.,wrist=12.,foot=8.)}
CHAINS=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]

def hierarchy_digest(bones):
    return hashlib.sha256(json.dumps([(b['name'],b['parent']) for b in bones],separators=(',',':')).encode()).hexdigest()

def validate_selected(context,profile,rig):
    if context is None:return False
    if set(context)!={'kind','source_sha256','body_asset_sha256'} or context['kind']!='offline_fixture_selection':raise ValueError('explicit fixture selection required; user confirmation is never synthesized')
    if context['source_sha256']!=rig['source_sha256'] or context['body_asset_sha256']!=profile['body_asset_sha256'] or profile['hierarchy_sha256']!=hierarchy_digest(rig['bones']):raise ValueError('stale target/profile binding')
    return True

def solve(rig,evidence,prediction,*,profile,selection=None,rounds=1,max_nfev=35,seconds=30.,redetect=None):
    if set(evidence)-{'points','point_mask','part_mask','camera','ownership_verified','orientation_observed','support_verified','offline_ownership_assumption'}:raise ValueError('unknown evidence fields, GT forbidden')
    if rounds not in (1,2,3):raise ValueError('iteration cap exceeded')
    names,ix,parents,local,base=prepare(rig)
    if profile['name'] not in PROFILES or profile['hierarchy_sha256']!=hierarchy_digest(rig['bones']):raise ValueError('unsupported range profile')
    selected=validate_selected(selection,profile,rig);limits=PROFILES[profile['name']]
    fusion=profile.get('fusion','observation')
    if fusion not in ('observation','learned_balanced'):raise ValueError('unknown fusion policy')
    learned=fusion=='learned_balanced'
    mask=np.asarray(evidence['part_mask'],bool).copy()
    if not selected:mask[8:]=False
    pmask=np.asarray(evidence['point_mask'],bool)
    q=np.asarray(evidence['points'],float);cam=np.asarray(evidence['camera'],float)
    if (mask.shape!=(13,) or pmask.shape!=(12,) or q.shape!=(12,2) or cam.shape!=(3,3) or not np.isfinite(q).all() or not np.isfinite(cam).all() or not np.allclose(cam@cam.T,np.eye(3),atol=1e-5) or np.linalg.det(cam)<.99):raise ValueError('invalid fixed evidence')
    target=np.array(prediction['axes'],float);unc=np.array(prediction['uncertainty_rad'],float)
    if target.shape!=(13,2,3) or unc.shape!=(13,) or not np.isfinite(target).all() or not np.isfinite(unc).all() or np.any(unc<=0):raise ValueError('invalid learned proposal')
    for axis in range(2):
        if not np.allclose(np.linalg.norm(target[:,axis],axis=1),1,atol=.01):raise ValueError('non-unit predicted frame')
    # No ownership authorization means exact B0; evidence is not silently trusted.
    if not (evidence['ownership_verified'] or (evidence.get('offline_ownership_assumption',False) and profile.get('offline_only') is True)):return dict(status='fallback_ownership',profile=profile['name'],fusion=fusion,delta_rotvec=[],active_bones=[],planned_heads={n:base[i,:3,3].tolist() for i,n in enumerate(names)},trace=[],source_sha256=rig['source_sha256'],target_applied=False)
    by,center,scale,body=frame(rig['bones']);pts=[ix[n] for n in POINTS]
    def project(world):
        # Fixed original body frame and camera, never re-fit to the candidate.
        c=(world[pts,:3,3]-center)@body@cam.T
        return c[:,:2]
    p0=project(base);width=np.linalg.norm(p0[0]-p0[3]);shoulder=p0[[0,3]].mean(0)
    if width<1e-6:raise ValueError('degenerate projection')
    desired=q*width+shoulder
    active=[];degrees=[]
    def add(n,deg):
        if n not in active:active.append(n);degrees.append(deg)
    for i,n in enumerate(BONES[:4]):
        if mask[i]:add(n,limits['limb'])
    if mask[:4].all() and limits['spine']:
        for n in ['Spine','Spine1','Spine2']:add(n,limits['spine']/3)
        for n in ['LeftShoulder','RightShoulder']:add(n,limits['clavicle'])
    # Ground/support is not inferred from a low detector score or a hidden leg.
    if evidence.get('support_verified',False):
        for i in range(4,8):
            if mask[i]:add(BONES[i],min(limits['limb'],12.))
    orientation=np.asarray(evidence.get('orientation_observed',[False]*5),bool)
    if orientation.shape!=(5,):raise ValueError('invalid orientation mask')
    if selected:
        for j,kind in enumerate(['head','wrist','wrist','foot','foot']):
            # A learned prior is allowed as a diagnostic proposal, never called observed.
            if mask[j+8] and unc[j+8]<np.radians(75):add(BONES[j+8],limits[kind])
    activeix=[ix[n] for n in active];bound=np.repeat(np.radians(degrees)/np.sqrt(3),3)
    initial_axes=np.array([base[ix[n],:3,:2].T@body for n in BONES]);initial_axes/=np.linalg.norm(initial_axes,axis=-1,keepdims=True)
    fixed_target=target.copy();weight=np.minimum(1.,1./np.maximum(unc,.3))*(.8 if learned else .2)
    # One frozen selection metric prevents changing proposals from gaming acceptance.
    def state(world,goal):
        projected=project(world);a=np.array([world[ix[n],:3,:2].T@body for n in BONES]);a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-9)
        endpoint=np.linalg.norm(projected-desired,axis=1)/scale
        direction=np.linalg.norm(a[:,1]-goal[:,1],axis=1)*weight
        detail=np.linalg.norm(a[8:,0]-goal[8:,0],axis=1)*weight[8:]*.15
        return projected,a,endpoint,direction,detail
    def metric(world):
        _,_,end,direction,detail=state(world,fixed_target)
        # Endpoints + direction priors are measured per chain; detail separately.
        chain=[]
        for k,ids in enumerate(CHAINS):
            mids=[2*k,2*k+1];obs=pmask[list(ids)];vals=end[list(ids)][obs]
            observation=float(np.mean(vals)) if len(vals) else 0.
            prior_values=direction[mids][mask[mids]]
            chain.append((.5*observation+.8*float(prior_values.mean()) if len(prior_values) else .5*observation) if learned else observation)
        detail_loss=float(np.mean(detail[mask[8:]])) if selected and mask[8:].any() else 0.
        primary=float(np.mean(end[pmask])) if pmask.any() else 0.
        prior=float(np.mean(direction[mask])) if mask.any() else 0.
        return ((.5*primary+.8*prior+.1*detail_loss) if learned else primary+.1*prior+.1*detail_loss),np.array(chain),primary
    def safety(world,x):
        violations=[]
        if not np.isfinite(world).all() or np.any(abs(x)>bound+1e-7):violations.append('cumulative_original_B0_bound')
        # Root stays fixed and inactive lower/support heads cannot drift.
        if not np.allclose(world[ix['Hips']],base[ix['Hips']],atol=1e-8):violations.append('root_changed')
        if not evidence.get('support_verified',False):
            low=[ix[n] for n in POINTS[6:]]
            if np.max(abs(world[low,:3,3]-base[low,:3,3]))>1e-7:violations.append('unsupported_lower_changed')
        # Preserve bend branch and avoid folding a chain through itself.
        for ids in CHAINS:
            ids=[pts[k] for k in ids];old=base[ids,:3,3];new=world[ids,:3,3]
            u,v=old[1]-old[0],old[2]-old[1];un,vn=new[1]-new[0],new[2]-new[1]
            cross0,cross1=np.cross(u,v),np.cross(un,vn)
            if np.linalg.norm(cross0)*np.linalg.norm(cross1)>1e-6 and np.dot(cross0,cross1)<0:violations.append('bend_branch')
            if np.dot(un,vn)/(np.linalg.norm(un)*np.linalg.norm(vn))<-.94:violations.append('overfolded_chain')
        return violations
    x=np.zeros(len(bound));best=base.copy();score,chains,primary=metric(best);initial_primary=primary;trace=[];started=time.monotonic()
    if not len(bound):return dict(status='fallback_unobservable',profile=profile['name'],fusion=fusion,active_bones=[],delta_rotvec=[],planned_heads={n:base[i,:3,3].tolist() for i,n in enumerate(names)},trace=[],source_sha256=rig['source_sha256'],target_applied=False)
    for step in range(rounds):
        if time.monotonic()-started>=seconds:trace.append(dict(round=step+1,status='timeout_keep_best'));break
        if redetect is not None:
            try:proposed=redetect(best.copy());candidate_target=np.asarray(proposed['axes'],float)
            except Exception as e:
                trace.append(dict(round=step+1,status='rediagnosis_failure_keep_best',error=type(e).__name__));break
            if candidate_target.shape!=target.shape or not np.isfinite(candidate_target).all():trace.append(dict(round=step+1,status='invalid_rediagnosis'));break
            target=candidate_target
        # Re-diagnosis emphasizes the current largest visible residual, using fixed evidence.
        _,_,ends,_,_=state(best,target);emphasis=np.ones(12)
        visible=np.flatnonzero(pmask)
        if len(visible):emphasis[visible[np.argmax(ends[visible])]]=1.4
        def residual(delta):
            if time.monotonic()-started>seconds:raise TimeoutError('solve budget')
            w=fk(local,parents,activeix,delta);proj,a,_,_,_=state(w,target)
            obs=((proj-desired)/scale*emphasis[:,None])[pmask].ravel()
            dirs=((a[:,1]-target[:,1])*weight[:,None])[mask].ravel()
            detail=((a[8:,0]-target[8:,0])*weight[8:,None]*np.where(orientation,1.,.15)[:,None])[mask[8:]].ravel() if selected else np.array([])
            return np.r_[obs*(.5 if learned else 1.),dirs*(.8 if learned else .2),.15*detail,.035*delta]
        try:
            fit=least_squares(residual,x,bounds=(-bound,bound),max_nfev=max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5)
            world=fk(local,parents,activeix,fit.x);newscore,newchains,newprimary=metric(world);violations=safety(world,fit.x)
            # Reject regressions on any observed limb; inactive limbs are included.
            regression=bool(np.any(newchains>chains+1e-5))
            accepted=not violations and not regression and newscore<score-1e-7 and newprimary<=(initial_primary+.025 if learned else primary+1e-6)
            trace.append(dict(round=step+1,status='accepted' if accepted else 'rejected_keep_best',score=newscore,primary=newprimary,per_chain=newchains.tolist(),violations=violations,part_regression=regression,nfev=fit.nfev,max_cumulative_degrees=float(max(np.linalg.norm(fit.x.reshape(-1,3),axis=1))*180/np.pi)))
            if accepted:best=world;x=fit.x.copy();score,chains,primary=newscore,newchains,newprimary
            else:break
        except (TimeoutError,ValueError,np.linalg.LinAlgError) as e:trace.append(dict(round=step+1,status='solver_failure_keep_best',error=type(e).__name__));break
    return dict(status='proposal' if np.any(x) else 'fallback_exact_B0',profile=profile['name'],fusion=fusion,profile_approved=False,active_bones=active,delta_rotvec=x.reshape(-1,3).tolist(),planned_heads={n:best[i,:3,3].tolist() for i,n in enumerate(names)},planned_matrices={n:best[i].tolist() for i,n in enumerate(names)},trace=trace,source_sha256=rig['source_sha256'],target_applied=selected and any(n in BONES[8:] and np.linalg.norm(v)>1e-7 for n,v in zip(active,x.reshape(-1,3))),orientation_authority='learned_weak_prior_unless_explicit_observation',unresolved=['finger_articulation_no_observed_targets','palm_sole_surface_normals_not_certified']+([] if evidence.get('support_verified',False) else ['lower_body_support_unverified']),elapsed_ms=(time.monotonic()-started)*1000,final_objective=score,production_ready=False)
