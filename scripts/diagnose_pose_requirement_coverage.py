"""Does a pinned gallery jointly cover arm proximity and deep-crouch geometry?"""
import argparse
import fcntl
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from standin_eval.pose_requirement_coverage import requirement_metrics


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    policy={'stage':'gallery-only development diagnosis prompted by reviewed315; no new ranking, query depth, GT or image inference',
        'purpose':'distinguish single-feature coverage from simultaneous whole-pose requirement coverage',
        'arm_definition':'unchanged bilateral wrists crossing torso midline and within.5torso of opposite upper-arm midpoint; heights.35..1.1',
        'crouch_definition':'both intrinsic knee angles10..120; worldY shoulder-hip>=.3torso,hip-knee<=.5torso,hip-ankle>=.15torso,knee-ankle>=.15torso',
        'recorded_before_measurement':True,'threshold_sweep':False,
        'snapshots':'pinned historical1981 versus pinned1995 from same audited fresh study; no live shared DB access',
        'limits':['geometry predicates not verified human posture labels','arms may be crossed behind torso; hand contact and facing direction not resolved',
                  'crouch requires consistent worldY-up and is not furniture contact','a zero count would not prove no semantically suitable pose exists',
                  'no real-rough accuracy or model improvement claim'],
        'inputs':{str(p):sha256(p) for p in (a.historical,a.extended)},
        'code':{n:sha256(ROOT/n) for n in ('scripts/diagnose_pose_requirement_coverage.py','standin_eval/pose_requirement_coverage.py','standin_eval/folded_arm_contact.py')}}
    save_json(a.out/'protocol.json',policy);snapshots={};results={}
    for name,path in (('historical1981',a.historical),('extended1995',a.extended)):
        with np.load(path,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy();families=d['family_ids'].tolist()
        assert len(ids)==(1981 if name=='historical1981' else 1995)
        snapshots[name]={pid:p for pid,p in zip(ids,joints)};rows=[]
        for pid,family,p in zip(ids,families,joints):rows.append({'pose_id':pid,'family_id':family,**requirement_metrics(p)})
        save_json(a.out/(name+'-metrics.json'),rows)
        fields=('deep_crouch_geometry','bilateral_arm_proximity_geometry','joint_requirements_geometry')
        results[name]={'gallery_count':len(ids),'counts':{field:sum(r[field] for r in rows) for field in fields},
                       'joint_pose_ids':[r['pose_id'] for r in rows if r['joint_requirements_geometry']],
                       'joint_family_count':len({r['family_id'] for r in rows if r['joint_requirements_geometry']})}
    shared=set(snapshots['historical1981'])&set(snapshots['extended1995'])
    assert len(shared)==1980 and all(np.array_equal(snapshots['historical1981'][pid],snapshots['extended1995'][pid]) for pid in shared)
    result={'snapshots':results,'shared_geometry_unchanged':len(shared),'original_rankings_changed':False,
            'accuracy':None,'scope':'predicate coverage only, not semantic or retrieval correctness'}
    save_json(a.out/'results.json',result)
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('historical','extended','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
