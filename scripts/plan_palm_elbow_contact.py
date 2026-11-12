#!/usr/bin/env python3
"""Plan broad palm/side-waist contact with rearward elbow and hard non-regression constraints.

A pose plan is only a research candidate. A failed palm contract is never adopted.
"""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from converter.pose_safety import audit_pose,rotation,descendants,rotate_subtree,camera_points
    from converter.surface_contact import nearest_surface
    from converter.deformation_witness import _apply
    from converter.palm_waist_contact import define_palm_contract,measure_palm,anatomical_axes
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','skin-plan','pose-plan','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--side',choices=('Left','Right'),required=True)
    p.add_argument('--objective',choices=('palm','elbow'),default='palm')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    def load(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    x,skin,plan=load(a.inputs),load(a.skin_plan),load(a.pose_plan)
    for key in ('source_sha256','asset_sha256'):
        if str(x[key])!=str(skin[key]) or str(x[key])!=str(plan[key]):raise ValueError('contact_source_mismatch')
    if str(plan['skin_plan_sha256'])!=hashlib.sha256(a.skin_plan.read_bytes()).hexdigest():raise ValueError('contact_skin_binding_mismatch')
    names=list(x['names']);parents=json.loads(str(x['parents']));camera=json.loads(str(plan['camera_json']))
    if hashlib.sha256(a.analysis.read_bytes()).hexdigest()!=str(plan['analysis_sha256']):raise ValueError('contact_analysis_binding')
    record=json.loads(a.analysis.read_text());query=np.asarray(record['analyze']['people'][0]['keypoints'],float)
    from src.refine_projection import FixedProjection
    target=(query-query[[11,12]].mean(0))/np.linalg.norm(query[[5,6]].mean(0)-query[[11,12]].mean(0))
    aligned=camera_points(plan['body_bones'],names,camera);projection=FixedProjection.from_aligned_base(aligned,camera['distance_torso'])
    def projected(body):
        pt=camera_points(body,names,camera)-np.array(projection.origin);xy=pt[:,:2]*[1,-1]
        if projection.distance_world is not None:xy=xy/(pt[:,2,None]+projection.distance_world)
        return xy
    origin=projected(plan['body_bones'])[[11,12]].mean(0);unit=np.linalg.norm(projected(plan['body_bones'])[[5,6]].mean(0)-origin)
    def pose_error(body):return (projected(body)-origin)/unit-target
    base=plan['body_bones'];inv=np.linalg.inv(x['bones']);w=skin['weights'];w=w/w.sum(1)[:,None]
    r,m,e=[names.index('mixamorig:'+a.side+n) for n in ('Arm','ForeArm','Hand')]
    local={i:np.linalg.inv(base[i])@plan['hand_bones'][i] for i in range(len(names))}
    def hands(body):return np.array([body[i]@local[i] for i in range(len(names))])
    def skin_vertices(body,ids=None):
        weights=w if ids is None else w[ids];rest=skin['corrected_rest'] if ids is None else skin['corrected_rest'][ids]
        bm=np.einsum('nb,bij->nij',weights,body@inv)
        hm=np.einsum('nb,bij->nij',weights,hands(body)@np.linalg.inv(body))
        return _apply(hm@bm,rest)
    initial=skin_vertices(base);contract=define_palm_contract(initial,x['faces'],w,names,base,a.side)
    contract.update(source_sha256=str(x['source_sha256']),asset_sha256=str(x['asset_sha256']),
        baseline_pose_plan_sha256=hashlib.sha256(a.pose_plan.read_bytes()).hexdigest())
    (a.output_dir/'contact-contract.json').write_text(json.dumps(contract,indent=2))
    ids=np.unique(x['faces'][contract['palm_faces']+contract['waist_faces']])
    def vertices(body):
        out=initial.copy();out[ids]=skin_vertices(body,ids);return out
    direction=base[m,:3,3]-base[r,:3,3];direction/=np.linalg.norm(direction)
    seed=np.eye(3)[np.argmin(abs(direction))];ax=np.cross(direction,seed);ax/=np.linalg.norm(ax)
    axes=np.stack([ax,np.cross(direction,ax)])
    def pose(params):
        out=rotate_subtree(base,descendants(names,parents,names[r]),r,Rotation.from_rotvec(np.deg2rad(params[:2])@axes).as_matrix())
        pole=np.cross(out[m,:3,3]-out[r,:3,3],out[e,:3,3]-out[m,:3,3]);pole/=np.linalg.norm(pole)
        out=rotate_subtree(out,descendants(names,parents,names[m]),m,Rotation.from_rotvec(np.deg2rad(params[2])*pole).as_matrix())
        for joint,child,angle in ((r,m,params[3]),(m,e,params[4])):
            axis=out[child,:3,3]-out[joint,:3,3];axis/=np.linalg.norm(axis)
            out=rotate_subtree(out,descendants(names,parents,names[joint]),joint,Rotation.from_rotvec(np.deg2rad(angle)*axis).as_matrix())
        for i in range(len(out)):out[i,:3,:3]=rotation(out[i])
        return out
    before=measure_palm(initial,x['faces'],contract)
    hip,lateral,up,front,torso=anatomical_axes(plan['base_bones'],names);back=-front
    baseline_errors=np.linalg.norm(pose_error(base),axis=1)
    elbow_index=7 if a.side=='Left' else 8;wrist_index=9 if a.side=='Left' else 10
    limb=[elbow_index,wrist_index]
    def constraints(params):
        body=pose(params);guard=audit_pose(plan['base_bones'],body,names,parents,camera,depth_reference=plan['depth_reference'])
        errors=np.linalg.norm(pose_error(body),axis=1)
        vals=[baseline_errors[5:].mean()-errors[5:].mean()-1e-5,baseline_errors[limb].mean()-errors[limb].mean()-1e-5,
            baseline_errors[elbow_index]-errors[elbow_index]-1e-5,baseline_errors[wrist_index]-errors[wrist_index]-1e-5,
            (body[m,:3,3]-base[m,:3,3])@back-.001]
        for row in guard['joints']:
            if row['joint'] in (a.side+'Arm',a.side+'ForeArm'):
                vals.extend([(row['limit_degrees']-.01-row['delta_degrees'])/12,(3.99-row['axial_twist_degrees'])/4])
        vals.append(.149-guard['max_depth_drift_torso'])
        vals.extend(np.sign(row['base_depth_torso'])*row['depth_torso']-.25*abs(row['base_depth_torso'])-.0001 for row in guard['depth_pairs'] if row['prior_constrained'])
        return np.array(vals)
    pf=np.array(contract['palm_faces']);tf=np.array(contract['waist_faces'])
    def objective(params):
        body=pose(params);errors=np.linalg.norm(pose_error(body),axis=1)
        if a.objective=='elbow':return -.8*(body[m,:3,3]-base[m,:3,3])@back/torso+errors[5:].mean()+errors[elbow_index]*.3
        v=vertices(body);tri=v[x['faces'][pf]];centers=tri.mean(1)
        q,_,d,sign,n=nearest_surface(centers,v[x['faces'][tf]])
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
        cosine=-np.einsum('ij,ij->i',normals,n)
        return float(np.mean(((d-.002)/.05)**2)+np.mean(np.minimum(sign-.001,0)**2)*5000+.25*np.mean((1-cosine)**2)+errors[5:].mean())
    trials=[];best=None
    for seed in ([0.,0.,0.,0.,0.],[3.,-4.,4.,0.,0.],[-3.,3.,-4.,0.,0.]):
        result=minimize(objective,seed,method='SLSQP',bounds=[(-14,14)]*3+[(-3.8,3.8)]*2,constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=100,ftol=1e-9,eps=1e-4))
        body=pose(result.x);guard=audit_pose(plan['base_bones'],body,names,parents,camera,depth_reference=plan['depth_reference'])
        feasible=constraints(result.x).min()>=-1e-7 and guard['status']=='pose_validated'
        trials.append(dict(parameters=result.x.tolist(),objective=float(result.fun),minimum_constraint=float(constraints(result.x).min()),feasible=bool(feasible),solver_message=str(result.message)))
        if feasible and (best is None or result.fun<best.fun):best=result
    if best is None:
        (a.output_dir/'planning.json').write_text(json.dumps(dict(trials=trials,adopted=False),indent=2));raise ValueError('no_nonregressing_safe_elbow_candidate')
    result=best;body=pose(result.x);guard=audit_pose(plan['base_bones'],body,names,parents,camera,depth_reference=plan['depth_reference'])
    current_errors=np.linalg.norm(pose_error(body),axis=1)
    original=plan['base_bones'];arm=original[m,:3,3]-original[r,:3,3];length=np.linalg.norm(arm);angle=np.arccos(np.clip(arm@back/length,-1,1))
    report=dict(version='palm-elbow-planner-v1',objective=a.objective,trials=trials,baseline_joint_nme=float(baseline_errors[5:].mean()),planned_joint_nme=float(current_errors[5:].mean()),
        baseline_elbow_nme=float(baseline_errors[elbow_index]),planned_elbow_nme=float(current_errors[elbow_index]),baseline_wrist_nme=float(baseline_errors[wrist_index]),planned_wrist_nme=float(current_errors[wrist_index]),
        elbow_rearward_mm=float((body[m,:3,3]-base[m,:3,3])@back*1000),before=before,planned=measure_palm(vertices(body),x['faces'],contract),pose_audit=guard,
        parameters_degrees=result.x.tolist(),final_fbx_verified=False,
        shoulder_plane_bound=dict(original_forward_elbow_mm=float(-arm@back*1000),minimum_turn_to_plane_degrees=float(np.rad2deg(angle)-90),allowed_degrees=12.,maximum_elbow_back_coordinate_mm=float(length*np.cos(max(0,angle-np.deg2rad(12)))*1000),plane_interpretation='anatomical shoulder plane; not observed image depth'))
    (a.output_dir/'planning.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('planned_joint_nme','elbow_rearward_mm','planned','shoulder_plane_bound')},indent=2),flush=True)
    plan.update(body_bones=body,hand_bones=hands(body))
    for key in ('realized_body_sha256','proposal_sha256'):plan.pop(key,None)
    np.savez(a.output_dir/'pose-plan.npz',**plan)
if __name__=='__main__':main()
