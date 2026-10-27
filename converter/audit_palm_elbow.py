"""Actual-byte full-palm, rearward-elbow and non-regression audit."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def measure_quality(before,after,names,camera,keypoints,side):
    import numpy as np
    from converter.pose_safety import camera_points
    from converter.palm_waist_contact import anatomical_axes
    from src.refine_projection import FixedProjection
    aligned=camera_points(before,names,camera);p=FixedProjection.from_aligned_base(aligned,camera['distance_torso'])
    bxy=p.project(aligned).astype(float);origin=bxy[[11,12]].mean(0);unit=np.linalg.norm(bxy[[5,6]].mean(0)-origin)
    q=np.asarray(keypoints,float);target=(q-q[[11,12]].mean(0))/np.linalg.norm(q[[5,6]].mean(0)-q[[11,12]].mean(0))
    errors=[]
    for bones in (before,after):errors.append(np.linalg.norm((p.project(camera_points(bones,names,camera))-origin)/unit-target,axis=1))
    elbow,wrist=(7,9) if side=='Left' else (8,10)
    _,_,_,front,_=anatomical_axes(before,names);idx=list(names).index('mixamorig:'+side+'ForeArm')
    rearward=float((after[idx,:3,3]-before[idx,:3,3])@(-front))
    rows={name:dict(before=float(errors[0][indices].mean()),after=float(errors[1][indices].mean())) for name,indices in
        [('body',list(range(5,17))),('selected_arm',[elbow,wrist]),('selected_elbow',[elbow]),('selected_wrist',[wrist])]}
    failures=[name+'_regressed' for name,row in rows.items() if row['after']>row['before']]
    if rearward<.001:failures.append('elbow_not_rearward')
    return dict(status='pose_nonregression_validated' if not failures else 'pose_nonregression_blocked',reasons=failures,metrics=rows,elbow_rearward_mm=rearward*1000,
        normalization='same baseline-FBX camera/torso/hip for both; no refitting',observed_3d_ground_truth=False)


def main():
    import hashlib,numpy as np
    from converter.audit_surface_contact import verified_snapshot
    from converter.refine_v3_safe_snapshot_qa import sha
    from converter.palm_waist_contact import define_palm_contract,measure_palm
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','baseline-qa','fbx','qa','contract','pose-plan','analysis','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);c=json.loads(a.contract.read_text())
    before,bq=verified_snapshot(a.baseline,a.baseline_qa);after,q=verified_snapshot(a.fbx,a.qa)
    with np.load(a.pose_plan,allow_pickle=False) as z:plan={k:z[k] for k in z.files}
    if sha(a.analysis)!=str(plan['analysis_sha256']) or bq['source_sha256']!=q['source_sha256'] or q['source_sha256']!=c['source_sha256'] or q['original_asset_sha256']!=c['asset_sha256']:raise ValueError('palm_input_binding')
    regenerated=define_palm_contract(before['vertices'],before['faces'],before['weights'],before['names'],before['bones'],c['side'])
    for key in regenerated:
        if regenerated[key]!=c[key]:raise ValueError('palm_region_binding:'+key)
    record=json.loads(a.analysis.read_text());quality=measure_quality(before['bones'],after['bones'],before['names'],json.loads(str(plan['camera_json'])),record['analyze']['people'][0]['keypoints'],c['side'])
    report=dict(version='actual-palm-elbow-audit-v1',baseline_sha256=sha(a.baseline),artifact_sha256=sha(a.fbx),receipt_sha256=q['receipt_sha256'],contract_sha256=sha(a.contract),analysis_sha256=sha(a.analysis),
        before=measure_palm(before['vertices'],before['faces'],c,True),after=measure_palm(after['vertices'],after['faces'],c,True),quality=quality,
        mesh_validated=True,pose_safety_validated=True,deployed=False)
    report['full_request_passed']=quality['status']=='pose_nonregression_validated' and report['after']['status']=='palm_contact_validated'
    report['safe_pose_candidate']=quality['status']=='pose_nonregression_validated'
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['safe_pose_candidate']:raise ValueError('actual_pose_regression')
if __name__=='__main__':main()
