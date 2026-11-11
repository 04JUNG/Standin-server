#!/usr/bin/env python3
"""Plan the separate surface-contact experiment; never issues an FBX receipt."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from converter.pose_safety import audit_pose,rotation,descendants,rotate_subtree,camera_points
    from converter.surface_contact import define_contact,contact_samples,audit_contact,TARGET_GAP_M,nearest_surface
    from converter.deformation_witness import _apply
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','skin-plan','pose-plan','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--side',choices=('Left','Right'),required=True)
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
    xy=projection.project(aligned);origin=xy[[11,12]].mean(0);unit=np.linalg.norm(xy[[5,6]].mean(0)-origin)
    def pose_error(body):return (projection.project(camera_points(body,names,camera))-origin)/unit-target
    base=plan['body_bones'];inv=np.linalg.inv(x['bones']);w=skin['weights'];w=w/w.sum(1)[:,None]
    r,m,e=[names.index('mixamorig:'+a.side+n) for n in ('Arm','ForeArm','Hand')]
    local={i:np.linalg.inv(base[i])@plan['hand_bones'][i] for i in range(len(names))}
    def hands(body):return np.array([body[i]@local[i] for i in range(len(names))])
    def skin_vertices(body,ids=None):
        weights=w if ids is None else w[ids];rest=skin['corrected_rest'] if ids is None else skin['corrected_rest'][ids]
        bm=np.einsum('nb,bij->nij',weights,body@inv)
        hm=np.einsum('nb,bij->nij',weights,hands(body)@np.linalg.inv(body))
        return _apply(hm@bm,rest)
    initial=skin_vertices(base);contract=define_contact(initial,x['faces'],w,names,a.side,base)
    contract.update(source_sha256=str(x['source_sha256']),asset_sha256=str(x['asset_sha256']),
        baseline_pose_plan_sha256=hashlib.sha256(a.pose_plan.read_bytes()).hexdigest())
    (a.output_dir/'contact-contract.json').write_text(json.dumps(contract,indent=2))
    ids=np.unique(x['faces'][contract['hand_faces']+contract['target_faces']])
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
    before=audit_contact(initial,x['faces'],contract,mesh_validated=False)
    def residual(params):
        body=pose(params);v=vertices(body);_,_,d,s,_,_=contact_samples(v,x['faces'],contract)
        patch_ids=np.unique(x['faces'][contract['hand_faces']])
        _,_,_,signed_vertices,_=nearest_surface(v[patch_ids],v[x['faces'][contract['target_faces']]])
        guard=audit_pose(plan['base_bones'],body,names,parents,camera,depth_reference=plan['depth_reference'])
        penalties=[max(0,row['delta_degrees']-row['limit_degrees']+.02)*30 for row in guard['joints']]
        penalties += [max(0,row['axial_twist_degrees']-row['axial_twist_limit_degrees']+.02)*30 for row in guard['joints']]
        penalties += [max(0,guard['max_depth_drift_torso']-.149)*1000]
        penalties += [max(0,.25*abs(row['base_depth_torso'])-np.sign(row['base_depth_torso'])*row['depth_torso']+.001)*1000 if row['prior_constrained'] else 0. for row in guard['depth_pairs']]
        # Surface gap is the new objective. All final constraints are also
        # audited independently, so a small soft loss never bypasses a guard.
        return np.r_[pose_error(body)[5:].ravel()*4.,(d-TARGET_GAP_M)*100,np.minimum(s-.0015,0)*500,np.minimum(signed_vertices-.0012,0)*5000,.003*params,penalties]
    result=least_squares(residual,[0.,0.,0.,0.,0.],bounds=(-np.array([14.,14.,14.,3.8,3.8]),np.array([14.,14.,14.,3.8,3.8])),max_nfev=150,diff_step=.001)
    body=pose(result.x);guard=audit_pose(plan['base_bones'],body,names,parents,camera,depth_reference=plan['depth_reference'])
    report=dict(baseline_joint_nme=float(np.linalg.norm(pose_error(base)[5:],axis=1).mean()),planned_joint_nme=float(np.linalg.norm(pose_error(body)[5:],axis=1).mean()),before=before,planned=audit_contact(vertices(body),x['faces'],contract),pose_audit=guard,parameters_degrees=result.x.tolist(),evaluations=result.nfev,final_fbx_verified=False)
    (a.output_dir/'planning.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('before','planned','parameters_degrees','evaluations')},indent=2),flush=True)
    if guard['status']!='pose_validated':raise ValueError('contact_pose_rejected:'+str(guard['violations']))
    plan.update(body_bones=body,hand_bones=hands(body))
    for key in ('realized_body_sha256','proposal_sha256'):plan.pop(key,None)
    np.savez(a.output_dir/'pose-plan.npz',**plan)
if __name__=='__main__':main()
