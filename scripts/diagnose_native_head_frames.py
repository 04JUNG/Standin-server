"""Check retained BVH frames on all cached reviewed Top5 candidates, not GT."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bvh import parse_bvh,fk,find_joint,coco17_from_fk,rotation_channel_indices
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.bvh_orientation_diagnostic import world_transforms,relative_rotation_angle


def run(a):
    ranks=json.loads(a.rankings.read_text())
    selected=sorted({c['pose_id'] for row in ranks for c in row.get('candidates',[])})
    conn=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
    try: lookup=dict(conn.execute('SELECT pose_id,bvh_path FROM poses').fetchall())
    finally: conn.close()
    paths={}; missing=[]
    for pid in selected:
        if pid not in lookup: missing.append(pid);continue
        stored=Path(lookup[pid]); choices=(stored,a.library/stored,a.library/'bvh'/stored.name)
        path=next((p for p in choices if p.is_file()),None)
        if path is None: missing.append(pid)
        else: paths[pid]=path
    a.out.mkdir(parents=True,exist_ok=False)
    code=('scripts/diagnose_native_head_frames.py','standin_eval/bvh_orientation_diagnostic.py','src/bvh.py')
    inputs={str(p):sha256(p) for p in (a.rankings,a.database,a.gallery,*paths.values())}
    policy={'stage':'native frame retention diagnostic, not semantic facing or search experiment',
            'selection':'all unique Top5 candidate IDs of frozen current manual review; no success-based pose selection',
            'sampling_after_review':True,'holdout':False,'bvh_frame':0,
            'fixed_counterfactual':'add30deg to first Head rotation channel; other channels unchanged',
            'checks':'native world positions equal existing FK and pinned COCO; orthonormal rotations; Head-parent angle independent Frobenius check',
            'limitations':['no canonical native facial forward axis is assumed','Head-parent angle includes bind/rest axis differences',
                           'counterfactual proves information loss, not that native heads match rough or that previews are corrected',
                           'no Human-Art inference, query Z, reranking, frozen renderer change or review replacement'],
            'selected_pose_ids':selected,'missing_files':missing,'inputs':inputs,'code':{n:sha256(ROOT/n) for n in code}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:gallery=dict(zip(d['pose_ids'].tolist(),d['joints']))
    records=[]; max_error=0.
    for pid,path in paths.items():
        skeleton,data=parse_bvh(str(path));values=data[0].copy();frames=world_transforms(skeleton,values)
        pos=fk(skeleton,values)
        err=max(np.linalg.norm(frames[i,:3,3]-pos[i]) for i in range(len(skeleton)))
        assert err<1e-10;max_error=max(max_error,float(err))
        kp,scores=coco17_from_fk(skeleton,pos)
        assert np.array_equal(kp,gallery[pid]),'pinned geometry mismatch: '+pid
        for f in frames:
            assert np.allclose(f[:3,:3].T@f[:3,:3],np.eye(3),atol=1e-10)
            assert abs(np.linalg.det(f[:3,:3])-1)<1e-10
        head=find_joint(skeleton,'Head');row={'pose_id':pid,'joint_count':len(skeleton),'native_fk_position_max_error':float(err)}
        if head<0:row['head_present']=False;records.append(row);continue
        parent=skeleton[head][1]; channels=rotation_channel_indices(skeleton,head)
        row.update(head_present=True,head_rotation_channel_count=len(channels))
        if parent>=0:
            angle=relative_rotation_angle(frames[parent],frames[head])
            relative=frames[parent,:3,:3].T@frames[head,:3,:3]
            alternate=float(np.rad2deg(np.arccos(np.clip(1-np.sum((relative-np.eye(3))**2)/4,-1,1))))
            assert abs(angle-alternate)<1e-5
            row['head_parent_relative_angle_degrees_uncalibrated']=angle
        if channels:
            changed=values.copy();changed[channels[0]]+=30
            other=world_transforms(skeleton,changed)
            delta=relative_rotation_angle(frames[head],other[head])
            same=np.array_equal(coco17_from_fk(skeleton,fk(skeleton,changed))[0],kp)
            assert same and abs(delta-30)<1e-5
            row.update(counterfactual_coco17_unchanged=same,counterfactual_head_frame_angle=delta)
        records.append(row)
    save_json(a.out/'records.json',records)
    result={'selected_candidates':len(selected),'files_checked':len(records),'missing_files':missing,
            'heads_with_rotation_channels':sum(r.get('head_rotation_channel_count',0)>0 for r in records),
            'counterfactual_information_loss_cases':sum(r.get('counterfactual_coco17_unchanged',False) for r in records),
            'max_fk_position_error':max_error,'search_accuracy':None,'preview_changed':False,
            'next':'rig-local forward-axis calibration is required before using native head frames for facial previews; keep scoring renderer frozen'}
    save_json(a.out/'results.json',result)
    for p,h in inputs.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('rankings','database','library','gallery','out'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
