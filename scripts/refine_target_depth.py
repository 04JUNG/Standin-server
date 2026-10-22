#!/usr/bin/env python3
"""Offline target-rig perspective fit with hinge, depth and full-mesh guards.

Consumes actual Blender planning inputs and a previously validated skin plan.
Produces plans only; exported FBX QA must independently validate and adopt bytes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from converter.pose_safety import (VERSION, CHAINS, audit_pose, index_map,
        descendants, rotate_subtree, camera_points, rotation)
    from converter.skin_profile_repair import correct_rest_surface
    from converter.deformation_witness import _skin_maps, _normals
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--skin-plan',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--candidate-comparison',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-step',type=float,default=1.,help='Bounded line-search starting step, 0 < step <= 1')
    p.add_argument('--uniform-only',action='store_true',help='Use uniform bounded backtracking when rig realization rejects a proposal')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    if not 0<a.max_step<=1:raise ValueError('invalid_depth_step')
    with np.load(a.inputs,allow_pickle=False) as z: x={k:z[k] for k in z.files}
    with np.load(a.skin_plan,allow_pickle=False) as z: skin={k:z[k] for k in z.files}
    for key in ('source_sha256','asset_sha256'):
        if str(x[key])!=str(skin[key]): raise ValueError('plan_source_mismatch')
    candidates=json.loads(a.candidate_comparison.read_text())
    chosen=next(c for c in candidates if c['source_sha256']==str(x['source_sha256']))
    camera=chosen['camera']
    if camera.get('labels_swapped') is not False:
        raise ValueError('target_depth_requires_unambiguous_side_labels')
    record=json.loads(a.analysis.read_text());person=record['analyze']['people'][0]
    from src.refine_projection import query_sha256
    if len(record['analyze']['people'])!=1 or query_sha256(person['keypoints'])!=camera['query_sha256']:
        raise ValueError('target_query_binding_mismatch')
    names=list(x['names']);parents=json.loads(str(x['parents']));lookup=index_map(names)
    original=x['body_bones'];baseline=original.copy()
    idx=lookup['RightArm'];active=descendants(names,parents,names[idx])
    baseline=rotate_subtree(baseline,active,idx,Rotation.from_euler('y',float(skin['right_arm_clearance_degrees']),degrees=True).as_matrix())
    query=np.asarray(person['keypoints'],float);center=query[[11,12]].mean(0)
    qs=np.linalg.norm(query[[5,6]].mean(0)-center);target=(query-center)/qs
    aligned=camera_points(baseline,names,camera);origin=aligned[[11,12]].mean(0)
    torso=np.linalg.norm(aligned[[5,6]].mean(0)-origin)
    distance=None if camera['distance_torso'] is None else camera['distance_torso']*torso
    def project(bones):
        pt=camera_points(bones,names,camera)-origin
        xy=pt[:,:2]*np.array([1,-1])
        if distance is not None:
            z=pt[:,2]+distance
            if np.any(z<=.1*torso): raise ValueError('camera_near_plane')
            xy=xy/z[:,None]
        return xy
    xy0=project(baseline);scale=np.linalg.norm(xy0[[5,6]].mean(0)-xy0[[11,12]].mean(0))
    center0=xy0[[11,12]].mean(0)
    def fit(bones):return (project(bones)-center0)/scale
    def nme(bones):return float(np.linalg.norm(fit(bones)[5:]-target[5:],axis=1).mean())
    specs=[];bounds=[]
    # Root swing has 2 axes perpendicular to the actual bone. Elbow/knee use
    # one transported bend-plane hinge; no unconstrained axial twist variables.
    for label,chain in CHAINS.items():
        r,m,e=[lookup[n] for n in chain];direction=baseline[m,:3,3]-baseline[r,:3,3];direction/=np.linalg.norm(direction)
        seed=np.eye(3)[np.argmin(abs(direction))];axis=np.cross(direction,seed);axis/=np.linalg.norm(axis)
        specs.append(('swing',r,m,e,np.stack([axis,np.cross(direction,axis)]),descendants(names,parents,names[r])))
        bounds.extend([6. if 'Leg' in label else 8.]*2)
        specs.append(('hinge',m,r,e,None,descendants(names,parents,names[m])))
        bounds.append(6. if 'Leg' in label else 8.)
    bounds=np.array(bounds)
    def pose(params):
        out=baseline.copy();cursor=0
        for kind,r,m,e,axes,sub in specs:
            if kind=='swing':vec=np.deg2rad(params[cursor:cursor+2])@axes;cursor+=2
            else:
                u=out[r,:3,3]-out[m,:3,3];v=out[e,:3,3]-out[r,:3,3]
                pole=np.cross(u,v)
                if np.linalg.norm(pole)<1e-8:
                    pole=rotation(out[r])[:,0]
                pole/=np.linalg.norm(pole);vec=np.deg2rad(params[cursor])*pole;cursor+=1
            out=rotate_subtree(out,sub,r,Rotation.from_rotvec(vec).as_matrix())
        # Counter-rotate feet so leg fitting cannot roll soles freely. This
        # still has to satisfy the independent 12-degree ankle bound.
        for side in ('Left','Right'):
            i=lookup[side+'Foot'];r=rotation(baseline[i])@rotation(out[i]).T
            out=rotate_subtree(out,descendants(names,parents,names[i]),i,r)
        # FBX bone frames can carry ~2e-5 numerical scale. Author the new
        # pose with proper rigid rotations, rather than transporting that
        # scale into another bake. Keep the original strict export comparison.
        for i in range(len(out)):
            out[i,:3,:3]=rotation(out[i])
        return out
    before_depth=aligned[:,2]
    def residual(params):
        bones=pose(params);current=camera_points(bones,names,camera)
        xy=(fit(bones)[5:]-target[5:]).ravel()
        # A source prior, NOT observed Z. Retain depth and use fixed perspective
        # foreshortening to resolve only supported joint swing/hinge motion.
        z=.22*(current[5:,2]-before_depth[5:])/torso
        guard=audit_pose(original,bones,names,parents,camera,depth_reference=baseline)
        penalties=[]
        for row in guard['joints']:
            penalties.append(3*max(0.,row['delta_degrees']-row['limit_degrees'])/12)
        for row in guard['depth_pairs']:
            old,new=row['base_depth_torso'],row['depth_torso']
            penalties.append(3*max(0.,.25*abs(old)-np.sign(old)*new) if row['prior_constrained'] else 0.)
        penalties.extend(3*np.maximum(abs(current[5:,2]-camera_points(original,names,camera)[5:,2])/torso-.15,0))
        return np.r_[xy,z,.025*params/8,penalties]
    result=least_squares(residual,np.zeros(len(bounds)),bounds=(-bounds,bounds),max_nfev=100,
        diff_step=.002,ftol=1e-9,xtol=1e-9,gtol=1e-7)
    attempts=[];normals=_normals(x['neutral'],x['faces']);f=x['faces'];weights=skin['weights']
    print('FIT',nme(baseline),nme(pose(result.x)),flush=True)
    # Reconstruct final hands from their actual local transforms after the
    # changed body. Non-finger world motion is inherited, not double-skinned.
    old_local={}
    for i,n in enumerate(names):
        par=parents[n]
        old_local[n]=x['hand_bones'][i] if par is None else np.linalg.inv(x['hand_bones'][names.index(par)])@x['hand_bones'][i]
    def hands(body):
        out=body.copy();pending=set(names);done=set()
        while pending:
            for n in list(pending):
                par=parents[n]
                if par is not None and par not in done:continue
                i=names.index(n)
                # Presets modify fingers only; keep all other posed bones.
                if 'Hand' in n and n.split(':')[-1] not in ('LeftHand','RightHand'):
                    out[i]=out[names.index(par)]@old_local[n]
                pending.remove(n);done.add(n)
        return out
    trials=[dict(label='uniform',steps=np.full(len(bounds),alpha))
            for alpha in np.array((1.,.75,.5,.25,.125))*a.max_step]
    feedback_used=False
    while trials:
        trial=trials.pop(0);steps=trial['steps'];alpha=float(steps.min())
        body=pose(result.x*steps);guard=audit_pose(original,body,names,parents,camera,depth_reference=baseline)
        entry=dict(alpha=alpha,step_kind=trial['label'],steps=steps.tolist(),nme=nme(body),pose_audit=guard)
        if guard['status']!='pose_validated' or nme(body)>=nme(baseline)-1e-5:
            entry['accepted']=False;attempts.append(entry);continue
        try:
            hand=hands(body)
            bm,bc,_=_skin_maps(weights,body@np.linalg.inv(x['bones']),len(x['neutral']),'lbs')
            hm,hc,_=_skin_maps(weights,hand@np.linalg.inv(body),len(x['neutral']),'lbs')
            bd=np.einsum('fvij,fj->fvi',bc[f],normals);hd=np.einsum('fvij,fvj->fvi',hc[f],bd)
            corrected,report=correct_rest_surface(x['neutral'],x['rest'],f,[('body',bm,bd),('hands',hm@bm,hd)])
        except ValueError as error:
            entry.update(accepted=False,reason=str(error),failed_planning=getattr(error,'report',None));attempts.append(entry)
            failed=getattr(error,'report',None)
            if failed and not feedback_used and not a.uniform_only:
                affected=set();pairs=[]
                for audit in failed['planning_audits'].values():
                    for first,second in audit['triangle_intersections']:
                        regions=[names[int(np.argmax(weights[f[t]].mean(0)))].split(':')[-1] for t in (first,second)]
                        pairs.append(dict(triangles=[first,second],regions=regions))
                        # Distal arm/thigh contact: back off the hand-bearing
                        # chain first. The rest of the body keeps its proposal.
                        arms={r[:4]+'Arm' if r.startswith('Left') else 'RightArm' for r in regions
                              if any(part in r for part in ('Hand','ForeArm','Shoulder','Arm'))}
                        legs={r[:4]+'Leg' if r.startswith('Left') else 'RightLeg' for r in regions
                              if any(part in r for part in ('UpLeg','Leg','Foot','Toe'))}
                        affected.update(arms or legs)
                if affected:
                    local=steps.copy()
                    for chain_index,label in enumerate(CHAINS):
                        if label in affected:local[chain_index*3:chain_index*3+3]=min(.2*a.max_step,float(local[chain_index*3]))
                    entry['collision_feedback']=dict(affected_chains=sorted(affected),triangle_regions=pairs)
                    trials.insert(0,dict(label='collision_local_backtrack',steps=local))
                    feedback_used=True
            (a.output_dir/'attempts.json').write_text(json.dumps(attempts,indent=2));print('REJECT',alpha,str(error),flush=True);continue
        entry.update(accepted=True,corrective=report);attempts.append(entry)
        skin['corrected_rest']=corrected
        # The generic pose plan contains the complete correction, including
        # the earlier 6-degree clearance; do not apply that twice.
        np.savez(a.output_dir/'skin-plan.npz',**skin)
        np.savez(a.output_dir/'pose-plan.npz',version=VERSION,source_sha256=x['source_sha256'],asset_sha256=x['asset_sha256'],
            skin_plan_sha256=hashlib.sha256((a.output_dir/'skin-plan.npz').read_bytes()).hexdigest(),
            names=x['names'],parents=x['parents'],base_bones=original,depth_reference=baseline,body_bones=body,hand_bones=hand,
            camera_json=json.dumps(camera,sort_keys=True),analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest())
        summary=dict(version=VERSION,base_nme=nme(baseline),planned_nme=nme(body),
            solver_evaluations=result.nfev,parameters_degrees=(result.x*steps).tolist(),attempts=attempts,
            depth_ground_truth=False,final_fbx_verified=False,deployed=False)
        (a.output_dir/'plan-report.json').write_text(json.dumps(summary,indent=2));print('PLANNED',alpha,nme(body),flush=True);return
    (a.output_dir/'attempts.json').write_text(json.dumps(attempts,indent=2))
    raise ValueError('no_depth_and_mesh_safe_improvement')

if __name__=='__main__':main()
