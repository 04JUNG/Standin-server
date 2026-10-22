#!/usr/bin/env python3
"""Recompute bounded rest correction using actual Blender rig frames."""
from pathlib import Path
import argparse,sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from converter.pose_safety import audit_pose
    from converter.skin_profile_repair import correct_rest_surface
    from converter.deformation_witness import _skin_maps,_normals
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--skin-plan',type=Path,required=True)
    p.add_argument('--realized-pose',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    def load(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    x,skin,pose=load(a.inputs),load(a.skin_plan),load(a.realized_pose)
    for key in ('source_sha256','asset_sha256'):
        if str(x[key])!=str(pose[key]) or str(x[key])!=str(skin[key]):raise ValueError('realized_source_mismatch')
    audit=audit_pose(x['body_bones'],pose['body_bones'],list(x['names']),json.loads(str(x['parents'])),json.loads(str(pose['camera_json'])),depth_reference=pose['depth_reference'])
    if audit['status']!='pose_validated':raise ValueError('realized_pose_rejected')
    n,f=x['neutral'],x['faces'];w=skin['weights'];normal=_normals(n,f)
    bm,bc,_=_skin_maps(w,pose['body_bones']@np.linalg.inv(x['bones']),len(n),'lbs')
    hm,hc,_=_skin_maps(w,pose['hand_bones']@np.linalg.inv(pose['body_bones']),len(n),'lbs')
    bd=np.einsum('fvij,fj->fvi',bc[f],normal);hd=np.einsum('fvij,fvj->fvi',hc[f],bd)
    rest,report=correct_rest_surface(n,x['rest'],f,[('body',bm,bd),('hands',hm@bm,hd)])
    skin['corrected_rest']=rest;np.savez(a.output_dir/'skin-plan.npz',**skin)
    pose['skin_plan_sha256']=hashlib.sha256((a.output_dir/'skin-plan.npz').read_bytes()).hexdigest()
    np.savez(a.output_dir/'pose-plan.npz',**pose)
    (a.output_dir/'plan-report.json').write_text(json.dumps(dict(pose_audit=audit,corrective=report,final_fbx_certified=False),indent=2))
    print('REALIZED_PLAN',a.output_dir,flush=True)
if __name__=='__main__':main()
