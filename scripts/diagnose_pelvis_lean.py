"""Separate pelvis-relative pose tilt from camera-relative tilt on reviewed288."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bvh import parse_bvh,fk,coco17_from_fk,find_joint
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.bvh_orientation_diagnostic import world_transforms
from standin_eval.pelvis_lean_diagnostic import measure
from standin_eval.perspective_search import Camera,camera_points


def run(a):
    rows=json.loads(a.rankings.read_text());query=next(r for r in rows if r['id']=='rough_288')
    conn=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
    try:lookup=dict(conn.execute('SELECT pose_id,bvh_path FROM poses').fetchall())
    finally:conn.close()
    paths={}
    for c in query['candidates']:
        stored=Path(lookup[c['pose_id']])
        paths[c['pose_id']]=next(p for p in (stored,a.library/stored,a.library/'bvh'/stored.name) if p.is_file())
    a.out.mkdir(parents=True,exist_ok=False)
    inputs={str(p):sha256(p) for p in (a.rankings,a.database,a.gallery,*paths.values())}
    source=('scripts/diagnose_pelvis_lean.py','standin_eval/pelvis_lean_diagnostic.py','standin_eval/bvh_orientation_diagnostic.py','src/bvh.py')
    policy={'stage':'development geometry diagnosis on all existing manual288 Top5, no ranking changes',
            'predeclared':'zero all channels to define rest shoulder/hip up and left; cross(up,left) is rest-front proxy; transport axes using native pelvis rotation',
            'root_joint':'Hips preferred, otherwise Hip; no guessed fallback',
            'output':'signed rest-frame forward/side angle and unsigned inclination; no fitted decision threshold',
            'limits':['rest-front proxy is not mesh-calibrated anatomical forward','angle is aggregate shoulder/hip tilt, not lumbar spine flexion',
                      'manual camera lean and actual pelvis-relative tilt are different quantities','Top5 coverage is not whole-gallery coverage',
                      'no query depth or new inference; no actual rough correctness claim or unseen holdout'],
            'inputs':inputs,'code':{n:sha256(ROOT/n) for n in source}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:gallery=dict(zip(d['pose_ids'].tolist(),d['joints']))
    results=[]
    for rank,c in enumerate(query['candidates'],1):
        pid=c['pose_id'];joints,data=parse_bvh(str(paths[pid]));frames=world_transforms(joints,data[0])
        zero=world_transforms(joints,np.zeros(data.shape[1]));idx=find_joint(joints,'Hips')
        if idx<0:idx=find_joint(joints,'Hip')
        p,sc=coco17_from_fk(joints,fk(joints,data[0]))
        r,rs=coco17_from_fk(joints,{i:f[:3,3] for i,f in enumerate(zero)})
        assert np.array_equal(p,gallery[pid])
        row={'pose_id':pid,'original_rank':rank,'camera':c['camera']}
        if idx<0 or not np.all(sc[[5,6,11,12]]>0) or not np.all(rs[[5,6,11,12]]>0):
            row['unsupported']='missing required pelvis/shoulder/hip anchors';results.append(row);continue
        try:metric=measure(p,frames[idx,:3,:3],r,zero[idx,:3,:3])
        except ValueError as error:row['unsupported']=str(error);results.append(row);continue
        # Independent vector-dot angle calculation from native frames.
        up=r[[5,6]].mean(0)-r[[11,12]].mean(0);up/=np.linalg.norm(up)
        left=r[5]-r[6];left-=left.dot(up)*up;left/=np.linalg.norm(left)
        transform=frames[idx,:3,:3]@zero[idx,:3,:3].T
        front=transform@np.cross(up,left);native_up=transform@up
        torso=p[[5,6]].mean(0)-p[[11,12]].mean(0)
        alternate=np.degrees(np.arctan2(torso.dot(front),torso.dot(native_up)))
        assert abs(alternate-metric['forward_rest_axis_degrees'])<1e-5
        view=camera_points(p,Camera(**c['camera']))
        row.update(metric);row['camera_shoulders_nearer_than_hips']=float(view[[11,12],2].mean()-view[[5,6],2].mean())
        results.append(row)
    save_json(a.out/'records.json',results)
    save_json(a.out/'results.json',{'evaluated':len(results),'supported':sum('unsupported' not in r for r in results),
                                  'accuracy':None,'rankings_and_review_unchanged':True,
                                  'next':'validate rest-frame front calibration before any feature gate or ranking experiment'})
    for p,h in inputs.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('rankings','database','library','gallery','out'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
