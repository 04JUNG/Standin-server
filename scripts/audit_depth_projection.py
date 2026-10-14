"""Independent projection and scalar paired-score audit."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_depth_projection import read,verify
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.diagnose_depth_style import fixed_infer
from standin_eval.z_diagnostics import infer_pose,foreground_bbox
from standin_eval.mixed_z import corrected_rtmw

SWAP=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
PAIRS=[(7,5),(9,5),(8,6),(10,6),(13,11),(15,11),(14,12),(16,12)]


def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def sign(v):return int(v>0)-int(v<0)


def check(args):
    verify(args,('prepared.json','human-frozen.json','predicted.json'))
    manifest=read(args.out/'manifest.json');human=read(args.out/'human.json');pred=read(args.out/'predictions.json')
    answers=read(args.out/'answers.json');result=read(args.out/'results.json')
    assert len(manifest)==len(human)==len(pred)==80 and len(answers)==len(result['records'])==40
    hs={(r['sample'],r['projection']):r for r in human}
    ps={(r['sample'],r['projection']):r for r in pred}
    poses={p.pose_id:p for p in load_library(args.library_root)}
    max_z=max_projection=0.; ratios=[]; descriptions={}
    for arm in result['description']:
        descriptions[arm]={'images':40,'failed':0,'body_score_ge03':0,'errors':[]}
    for a,row in zip(answers,result['records']):
        assert a['sample']==row['sample']
        joints=poses[a['pose_id']].joints.astype(float);hip=(joints[11]+joints[12])/2
        points=(joints-hip)/distance((joints[5]+joints[6])/2,hip)
        yaw,pitch=map(math.radians,(a['camera']['yaw'],a['camera']['pitch']))
        ry=np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
        rx=np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
        camera=points@ry.T@rx.T
        zerr=max(abs(camera[:,2]-np.asarray(a['camera_z'])));max_z=max(max_z,float(zerr));assert zerr<1e-6
        depths=camera[5:,2]+a['camera']['distance'];ratio=float(max(depths)/min(depths))
        assert ratio>=1.75-1e-6 and abs(ratio-a['body_depth_ratio'])<1e-6;ratios.append(ratio)
        truth=[a['camera_z'][j]-a['camera_z'][k] for j,k in PAIRS]
        assert row['truth_signs']==[sign(v) for v in truth] and row['truth_valid']==[abs(v)>=.1 for v in truth]
        for projection in ('ortho','perspective'):
            plane=camera[:,:2].copy();plane[:,1]*=-1
            if projection=='perspective':plane/=camera[:,2,None]+a['camera']['distance']
            xy=np.asarray(a['projections'][projection]);pc=plane-plane.mean(0);xc=xy-xy.mean(0)
            scale=float((pc*xc).sum()/(pc*pc).sum());assert scale>0
            err=float(abs(pc*scale-xc).max());max_projection=max(max_projection,err);assert err<1e-4
            h=hs[(a['sample'],projection)];pr=ps[(a['sample'],projection)]
            if 'human' in h:
                hx=np.asarray(h['human']['keypoints2d'],np.float32);scores=np.asarray(h['human']['scores'],np.float32)
                mask=scores>=.3;assert h['mask']==mask.tolist()
                center=(hx[11]+hx[12])/2;normalized=hx-center;torso=np.linalg.norm((hx[5]+hx[6])/2-center)
                normalized/=torso if torso>=1e-6 else 1.
                # Existing normalizer zeroes invalid joint coordinates.
                normalized[~mask]=0
                assert np.allclose(normalized,h['xy'],atol=2e-6,rtol=0)
                assert bool(h.get('error'))==bool(mask[5:].sum()<6 or not mask[[5,6,11,12]].all())
            for style in ('rough','solid'):
                arm=projection+'_'+style;out=pr['styles'][style];observed=row['arms'][arm];desc=descriptions[arm]
                if out.get('error'):desc['failed']+=1
                if 'raw' in out:desc['body_score_ge03']+=sum(s>=.3 for s in out['raw']['scores'][5:])
                if 'mapped' in out:desc['errors'].append(min(out['mapped']['mean_direct'],out['mapped']['mean_swapped']))
                if h.get('error') or out.get('error'):
                    assert observed['all']==observed['xy_gate']==[False]*8 and observed['signs']==[0]*8
                    continue
                hx=h['human']['keypoints2d'];r=out['raw'];body=[j for j in range(5,17) if h['mask'][j]]
                torso=distance([(hx[5][j]+hx[6][j])/2 for j in range(2)],[(hx[11][j]+hx[12][j])/2 for j in range(2)])
                model_swap=sum(distance(hx[j],r['keypoints2d'][SWAP[j]]) for j in body)<sum(distance(hx[j],r['keypoints2d'][j]) for j in body)
                assert model_swap==out['mapped']['swapped']
                gt_swap=sum(distance(hx[j],xy[SWAP[j]]) for j in body)<sum(distance(hx[j],xy[j]) for j in body)
                assert gt_swap==observed['oracle_swap']
                hm=SWAP if model_swap else list(range(17));hg=SWAP if gt_swap else list(range(17))
                rawz=np.asarray([r['z_meters'][j] for j in hm],np.float32)
                nz=(rawz-(rawz[11]+rawz[12])/2)/.5;nz[~np.asarray(h['mask'],bool)]=0
                canonical=nz[hg];signs=[];active=[];gated=[]
                for j,k in PAIRS:
                    delta=canonical[j]-canonical[k];signs.append(sign(float(delta)))
                    valid=bool(h['mask'][hg[j]] and h['mask'][hg[k]] and abs(delta)>=.1)
                    active.append(valid)
                    errs=[distance(hx[hg[q]],r['keypoints2d'][hm[hg[q]]])/torso for q in (j,k)]
                    gated.append(bool(valid and max(errs)<=.5))
                assert (signs,active,gated)==(observed['signs'],observed['all'],observed['xy_gate'])
    for arm,desc in descriptions.items():
        values=desc.pop('errors');desc['median_xy_disagreement']=float(np.median(values)) if values else None
        assert desc==result['description'][arm]
    for comp in result['comparisons']:
        left,right,gate=comp['left'],comp['right'],comp['gate']
        expected={'left':left,'right':right,'gate':gate,'gt_n':0,'gains':0,'losses':0,
                  'images_better':0,'images_worse':0,'images_tied':0}
        for scope in ('own','common'):
            expected[scope]={arm:dict(n=0,correct=0,always_front=0,always_back=0) for arm in (left,right)}
        for row in result['records']:
            gains=losses=0
            for k,t in enumerate(row['truth_signs']):
                valid=row['truth_valid'][k];expected['gt_n']+=valid
                use={a:valid and row['arms'][a][gate][k] for a in (left,right)}
                correct={a:row['arms'][a]['signs'][k]==t for a in (left,right)}
                common=all(use.values())
                gains+=bool(common and not correct[left] and correct[right]);losses+=bool(common and correct[left] and not correct[right])
                for scope in ('own','common'):
                    for arm in (left,right):
                        if (use[arm] if scope=='own' else common):
                            cell=expected[scope][arm];cell['n']+=1;cell['correct']+=correct[arm]
                            cell['always_front']+=t<0;cell['always_back']+=t>0
            expected['gains']+=gains;expected['losses']+=losses
            expected['images_better']+=gains>losses;expected['images_worse']+=gains<losses;expected['images_tied']+=gains==losses
        assert expected==comp
    for item in manifest:
        for entry in item['styles'].values():assert sha256(Path(entry['path']))==entry['sha256']
    from rtmlib import RTMPose
    hm=RTMPose(str(args.human_model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    zm=corrected_rtmw(args.z_model)
    for item in manifest[:2]:
        h=hs[(item['sample'],item['projection'])];pr=ps[(item['sample'],item['projection'])]
        rgb=read_rgb(item['styles']['rough']['path'])
        assert infer_pose(rgb,hm,crop=True,turns=(0,1,2,3),is_3d=False)==h['human']
        assert foreground_bbox(np.ascontiguousarray(np.rot90(rgb,h['turn'])))==h['bbox']
        for style in ('rough','solid'):
            assert fixed_infer(read_rgb(item['styles'][style]['path']),zm,h['turn'],h['bbox'])==pr['styles'][style]['raw']
    report={'samples':40,'images':160,'human_baselines':80,'scored_depth_observations':160,
            'scalar_pair_comparisons':8*40*8,'human_replay_rotations':8,'depth_replays':4,
            'max_camera_z_error':max_z,'max_projection_pixel_error':max_projection,
            'depth_ratio_min_median_max':np.quantile(ratios,[0,.5,1]).tolist(),
            'human_baseline_unchanged':True,'retrieval_accuracy':None,
            'hashes':{p:sha256(args.out/p) for p in ('protocol.json','human.json','predictions.json','results.json')},
            'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('out','library-root','human-model','z-model'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
