"""Coverage diagnosis for manual cues without relaxing cached whole-pose rank."""
import argparse
import fcntl
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import Camera
from standin_eval.manual_pose_cues import describe,satisfies
from standin_eval.folded_arm_contact import folded_arm_contact
from standin_eval.cue_shortlist_guard import guarded_shortlist

CASES={'rough_315':{'folded_arm_contact':True},
       'rough_301':{'elbow_degrees':{'right':[155,180]},'left_hand_hip_max':.6},
       'rough_291':{'elbow_degrees':{'left':[70,110],'right':[70,110]},'projected_leg_crossing':True}}


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    paths=(a.gallery,*[a.source/'costs'/(name+'.npz') for name in CASES])
    hashes={str(p):sha256(p) for p in paths}
    sources=('scripts/diagnose_cue_shortlist_guard.py','standin_eval/cue_shortlist_guard.py','standin_eval/manual_pose_cues.py','standin_eval/folded_arm_contact.py')
    policy={'stage':'manual-cue coverage diagnosis on previously reviewed real data, not fresh validation',
            'hypothesis':'search whole pose first, then permit manual cue selection only within its existing Top20 budget; otherwise abstain from cue-success claim',
            'rank_budget':20,'budget_origin':'same existing contour candidate count, not tuned to these outcomes',
            'whole_pose_score':'.25 frozen coarse XY cost+.75 frozen refined XY cost, same1981 candidates/cameras/XY/masks',
            'conditions':CASES,'excluded':['288 uses different camera-constraint fitting and uncalibrated intrinsic lean','178 has changed input and occlusion'],
            'limits':['Top20 XY match not verified whole-pose correctness','manual cue geometry not semantic GT','abstention does not prove whole gallery lacks suitable pose',
                      'no new inference, contour recomputation, score-weight tuning or existing ranking/screen replacement'],
            'inputs':hashes,'code':{n:sha256(ROOT/n) for n in sources}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    points=normalized_joints(joints);records=[]
    for name,cues in CASES.items():
        with np.load(a.source/'costs'/(name+'.npz'),allow_pickle=False) as d:base,fit,params=[d[k].copy() for k in ('base','fit','parameters')]
        assert len(base)==len(ids)==1981
        xy,valid=project_parameters(points,params);assert valid.all()
        matched=[]
        for j,p in enumerate(joints):
            yaw,pitch,inv=map(float,params[j]);camera=Camera(yaw,pitch,None if inv==0 else 1/inv)
            yes=folded_arm_contact(p)[0] if 'folded_arm_contact' in cues else satisfies(describe(p,camera,xy[j]),cues)[0]
            matched.append(yes)
        scores=.25*base.astype(float)+.75*fit.astype(float);guard=guarded_shortlist(scores,matched,20)
        # Independently enumerate gallery-sorted pairs instead of argsort.
        reference=sorted(range(len(ids)),key=lambda j:(float(scores[j]),j))[:20]
        allowed=[j for j in reference if matched[j]]
        assert reference==guard['whole_pose_shortlist'] and allowed==guard['cue_matches']
        all_allowed=[j for j in range(len(ids)) if matched[j]]
        record={'query':name,'gallery_cue_match_count':len(all_allowed),'whole_pose_top20_cue_match_count':len(allowed),
                'cue_abstention':guard['cue_abstention'],'top20_pose_ids':[ids[j] for j in reference],
                'guarded_candidate_ids':[ids[j] for j in allowed],
                'whole_gallery_cue_first_by_xy':ids[min(all_allowed,key=lambda j:float(scores[j]))] if all_allowed else None,
                'unconstrained_xy_first':ids[guard['unconstrained_first']],
                'scope':'counterfactual candidate restriction; no saved original ranks overwritten'}
        records.append(record)
        np.savez_compressed(a.out/(name+'-guard.npz'),scores=scores,eligible=np.asarray(matched),short=np.asarray(reference),matches=np.asarray(allowed,dtype=int))
    save_json(a.out/'records.json',records)
    result={'queries':len(records),'cue_abstentions':sum(r['cue_abstention'] for r in records),
            'accuracy':None,'rankings_changed':False,'next':'review constrained candidates or explicitly report no joint match; do not expand budget post hoc to force success'}
    save_json(a.out/'results.json',result)
    for p,h in hashes.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print([{k:v for k,v in r.items() if k!='top20_pose_ids'} for r in records],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('gallery','source','out'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
