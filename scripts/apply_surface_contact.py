#!/usr/bin/env python3
"""Separate opt-in surface-contact version; no production endpoint or deployment."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,os,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('analysis','baseline-run','character','shape-registry','blender','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--character-id',required=True);p.add_argument('--side',choices=('Left','Right'),required=True)
    a=p.parse_args();base=a.baseline_run.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    a.blender=a.blender.resolve();a.character=a.character.resolve();a.shape_registry=a.shape_registry.resolve()
    applied=json.loads((base/'applied.json').read_text());qa=base/'depth-final-qa';plan=base/'depth-plan'
    if applied.get('applied') is not True or applied.get('depth_safe') is not True:raise ValueError('contact_requires_validated_depth_baseline')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(qa/'final.fbx')!=applied['artifact_sha256']:raise ValueError('contact_baseline_sha_mismatch')
    if sha(a.character)!=json.loads((qa/'qa-report.json').read_text())['original_asset_sha256']:raise ValueError('contact_character_mismatch')
    import numpy as np
    with np.load(plan/'pose-plan.npz',allow_pickle=False) as z:left,right=str(z['hand_left']),str(z['hand_right'])
    stages=[]
    def run(label,script,args,blender=False):
        start=time.monotonic();cmd=([str(a.blender),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/script),'--'] if blender else [sys.executable,str(ROOT/script)])+list(map(str,args))
        with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        stages.append(dict(stage=label,seconds=time.monotonic()-start,returncode=r.returncode));(out/'stages.json').write_text(json.dumps(stages,indent=2))
        if r.returncode:raise ValueError('contact_stage_blocked:'+label)
    result=dict(version='surface-contact-experiment-v1',contact_applied=False,baseline_sha256=sha(qa/'final.fbx'),deployed=False,production_registered=False,side=a.side)
    try:
        run('proposal','scripts/plan_surface_contact.py',['--inputs',base/'planning-inputs/inputs.npz','--skin-plan',plan/'skin-plan.npz','--pose-plan',plan/'pose-plan.npz','--side',a.side,'--analysis',a.analysis.resolve(),'--output-dir',out/'proposal'])
        run('carrier','converter/realize_contact_rig.py',['--baseline-body',qa/'body.fbx','--proposal',out/'proposal/pose-plan.npz','--side',a.side,'--output',out/'unapproved-carrier.fbx'],True)
        run('realization','converter/realize_pose_plan.py',['--proposal',out/'proposal/pose-plan.npz','--realized-body',out/'unapproved-carrier.fbx','--hand-left',left,'--hand-right',right,'--output-dir',out/'realization'],True)
        run('skin','scripts/plan_realized_pose_skin.py',['--inputs',base/'planning-inputs/inputs.npz','--skin-plan',plan/'skin-plan.npz','--realized-pose',out/'realization/realized-pose.npz','--output-dir',out/'skin'])
        run('qa','converter/refine_v3_safe_snapshot_qa.py',['--character',a.character,'--character-id',a.character_id,'--shape-registry',a.shape_registry,'--source-bvh',qa/'source.bvh','--skin-plan',out/'skin/skin-plan.npz','--pose-plan',out/'skin/pose-plan.npz','--pose-rig',out/'realization/unapproved-body-rig.fbx','--hand-left',left,'--hand-right',right,'--output-dir',out/'qa'],True)
        run('contact','converter/audit_surface_contact.py',['--baseline',qa/'final.fbx','--baseline-qa',qa/'qa-report.json','--fbx',out/'qa/final.fbx','--qa',out/'qa/qa-report.json','--contract',out/'proposal/contact-contract.json','--output',out/'contact-audit.json'],True)
        contact=json.loads((out/'contact-audit.json').read_text())
        if contact.get('contact_applied') is not True:raise ValueError('contact_not_verified')
        result.update(contact_applied=True,artifact=str(out/'qa/final.fbx'),artifact_sha256=sha(out/'qa/final.fbx'),contact_audit_sha256=sha(out/'contact-audit.json'),mesh_receipt_sha256=contact['mesh_receipt_sha256'])
    except Exception as exc:
        result['blocked_reason']=str(exc);raise
    finally:(out/'applied.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
