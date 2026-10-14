"""Independent vector geometry verification for joint-requirement coverage."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256


def read(p):return json.loads(p.read_text())


def audit(a):
    policy=read(a.study/'protocol.json')
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    total=0
    for name,path in (('historical1981',a.historical),('extended1995',a.extended)):
        with np.load(path,allow_pickle=False) as d:p=d['joints'].astype(float);ids=d['pose_ids'].tolist()
        hip=p[:,[11,12]].mean(1);shoulder=p[:,[5,6]].mean(1);torso=np.linalg.norm(shoulder-hip,axis=1)
        up=(shoulder-hip)/torso[:,None];left=p[:,5]-p[:,6];left-=(left*up).sum(1)[:,None]*up;left/=np.linalg.norm(left,axis=1)[:,None]
        a1=p[:,[11,12]]-p[:,[13,14]];b1=p[:,[15,16]]-p[:,[13,14]]
        bends=np.degrees(np.arctan2(np.linalg.norm(np.cross(a1,b1),axis=-1),(a1*b1).sum(-1)))
        knee=p[:,[13,14]].mean(1);ankle=p[:,[15,16]].mean(1)
        heights=np.stack(((shoulder[:,1]-hip[:,1])/torso,(hip[:,1]-knee[:,1])/torso,
                          (hip[:,1]-ankle[:,1])/torso,(knee[:,1]-ankle[:,1])/torso),axis=1)
        crouch=np.all((bends>=10)&(bends<=120),axis=1)&(heights[:,0]>=.3)&(heights[:,1]<=.5)&(heights[:,2]>=.15)&(heights[:,3]>=.15)
        wrists=p[:,[9,10]]-hip[:,None,:];side=(wrists*left[:,None,:]).sum(-1)/torso[:,None];height=(wrists*up[:,None,:]).sum(-1)/torso[:,None]
        opposite=np.stack((p[:,[6,8]].mean(1),p[:,[5,7]].mean(1)),axis=1)
        distance=np.linalg.norm(p[:,[9,10]]-opposite,axis=-1)/torso[:,None]
        folded=(side[:,0]<-.02)&(side[:,1]>.02)&np.all(distance<=.5,axis=1)&np.all((height>=.35)&(height<=1.1),axis=1)
        records=read(a.study/(name+'-metrics.json'))
        assert ids==[r['pose_id'] for r in records]
        for i,r in enumerate(records):
            assert np.allclose(r['knee_angles_degrees'],bends[i],atol=1e-5)
            assert np.allclose(r['wrist_opposite_upper_arm_distances'],distance[i],atol=1e-10)
            assert r['deep_crouch_geometry']==bool(crouch[i]) and r['bilateral_arm_proximity_geometry']==bool(folded[i])
            assert r['joint_requirements_geometry']==bool(crouch[i]&folded[i])
        counts=read(a.study/'results.json')['snapshots'][name]['counts']
        assert counts=={'deep_crouch_geometry':int(crouch.sum()),'bilateral_arm_proximity_geometry':int(folded.sum()),'joint_requirements_geometry':int((crouch&folded).sum())}
        total+=len(ids)
    save_json(a.study/'audit.json',{'result_sha256':sha256(a.study/'results.json'),'individual_geometry_replays':total,
        'independent_vector_angle_method':'atan2 cross-product/dot rather than scalar arccos',
        'counts_verified':True,'quality_or_semantic_accuracy_claim':None,'shared_library_modified':False})
    print('PASS',total,'independent geometry and predicate-count replays',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('study','historical','extended'):p.add_argument('--'+n,type=Path,required=True)
    audit(p.parse_args())
