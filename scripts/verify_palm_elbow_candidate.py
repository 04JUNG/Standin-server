#!/usr/bin/env python3
"""Verify a planned full-palm candidate; full_request_passed is never inferred from safe pose alone."""
from pathlib import Path
import argparse,json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    for k in ('baseline-run','proposal','analysis','character','shape-registry','blender','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--character-id',required=True);p.add_argument('--side',choices=('Left','Right'),required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);base=a.baseline_run.resolve();qa=base/'depth-final-qa';proposal=a.proposal.resolve()
    import numpy as np
    with np.load(base/'depth-plan/pose-plan.npz',allow_pickle=False) as z:left,right=str(z['hand_left']),str(z['hand_right'])
    stages=[];summary=dict(full_request_passed=False,safe_pose_candidate=False,deployed=False)
    def run(label,script,args,blender=False):
        cmd=([str(a.blender),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/script),'--'] if blender else [sys.executable,str(ROOT/script)])+list(map(str,args));start=time.monotonic()
        with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        stages.append(dict(stage=label,seconds=time.monotonic()-start,returncode=r.returncode));(out/'stages.json').write_text(json.dumps(stages,indent=2))
        if r.returncode:raise ValueError('candidate_blocked:'+label)
    try:
        run('carrier','converter/realize_contact_rig.py',['--baseline-body',qa/'body.fbx','--proposal',proposal/'pose-plan.npz','--side',a.side,'--output',out/'unapproved-carrier.fbx'],True)
        run('realization','converter/realize_pose_plan.py',['--proposal',proposal/'pose-plan.npz','--realized-body',out/'unapproved-carrier.fbx','--hand-left',left,'--hand-right',right,'--output-dir',out/'realization'],True)
        run('skin','scripts/plan_realized_pose_skin.py',['--inputs',base/'planning-inputs/inputs.npz','--skin-plan',base/'depth-plan/skin-plan.npz','--realized-pose',out/'realization/realized-pose.npz','--output-dir',out/'skin'])
        run('qa','converter/refine_v3_safe_snapshot_qa.py',['--character',a.character.resolve(),'--character-id',a.character_id,'--shape-registry',a.shape_registry.resolve(),'--source-bvh',qa/'source.bvh','--skin-plan',out/'skin/skin-plan.npz','--pose-plan',out/'skin/pose-plan.npz','--pose-rig',out/'realization/unapproved-body-rig.fbx','--hand-left',left,'--hand-right',right,'--output-dir',out/'qa'],True)
        run('palm-elbow','converter/audit_palm_elbow.py',['--baseline',qa/'final.fbx','--baseline-qa',qa/'qa-report.json','--fbx',out/'qa/final.fbx','--qa',out/'qa/qa-report.json','--contract',proposal/'contact-contract.json','--pose-plan',out/'skin/pose-plan.npz','--analysis',a.analysis.resolve(),'--output',out/'palm-elbow-audit.json'],True)
        report=json.loads((out/'palm-elbow-audit.json').read_text());summary.update(full_request_passed=report['full_request_passed'],safe_pose_candidate=report['safe_pose_candidate'],artifact_sha256=report['artifact_sha256'],remaining_reasons=report['after']['reasons'])
    except Exception as exc:summary['blocked_reason']=str(exc);raise
    finally:(out/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
