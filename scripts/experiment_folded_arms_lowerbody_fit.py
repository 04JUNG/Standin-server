"""Manual folded-arm gate plus torso/leg-only camera fit, separate input arm."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import best_views
from standin_eval.folded_arm_contact import folded_arm_contact
from standin_eval.z_axis_search import Pose3D
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,project,normalize_batch
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.dual_label_search import alternate_labels
from standin_eval.body_part_evidence import body_part_errors
from standin_eval.pose_requirement_coverage import requirement_metrics
from src.features import normalize_skeleton


def lowerbody_mask(original):
    original=np.asarray(original,bool)
    if original.shape!=(17,):raise ValueError('COCO17 mask required')
    mask=np.zeros(17,bool);mask[[5,6,11,12,13,14,15,16]]=original[[5,6,11,12,13,14,15,16]]
    if not mask[[5,6,11,12]].all() or mask[5:].sum()<6:raise ValueError('observed torso anchors and at least6 selected joints required')
    return mask


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    files=(a.gallery,a.source/'rankings.json',a.source/'predictions/rough_315.json',a.source/'costs/rough_315.npz')
    hashes={str(p):sha256(p) for p in files}
    code=('scripts/experiment_folded_arms_lowerbody_fit.py','standin_eval/folded_arm_contact.py','standin_eval/camera_refinement.py')
    policy={'stage':'reviewed315 manual-assisted development experiment, not automatic image search or fresh validation',
            'hypothesis':'manual arm cue supplies arm requirement; camera fits only original observed shoulders/hips/knees/ankles to preserve crouch geometry',
            'selection':'all pinned1981 poses satisfying unchanged bilateral wrist-opposite-upperarm proximity gate; no preferred IDs/names',
            'control':'same gate, cached original body-mask blend .25base+.75fit, no contour',
            'experiment':'same gate and original XY, intentionally narrower scoring mask5,6,11..16; refit195 cameras with same3starts/steps/sweeps/global label hypotheses; same .25/.75 blend',
            'primary_baseline':'original Human-Art XY and masks unchanged in original study; new scoring mask is a separate ablation, not XY/Z isolation',
            'limits':['different objectives mean raw cost reduction is not an accuracy gain','native full-pose relevance and arm contact need human verification',
                      'no source3D GT','manual gate can be satisfied behind torso','camera fit can hide or foreshorten limbs','no default ranking or screen replacement'],
            'inputs':hashes,'code':{n:sha256(ROOT/n) for n in code}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:
        ids=d['pose_ids'].tolist();joints=d['joints'].copy();families=d['family_ids'].tolist();scores=d['scores'].copy()
    selected=np.asarray([i for i,p in enumerate(joints) if folded_arm_contact(p)[0]],int)
    assert len(selected)>0
    poses=[Pose3D(ids[i],families[i],joints[i],scores[i]) for i in selected]
    pred=json.loads((a.source/'predictions/rough_315.json').read_text());xy=np.asarray(pred['keypoints2d'],np.float32);confidence=np.asarray(pred['scores'],np.float32);original_mask=confidence>=.3
    mask=lowerbody_mask(original_mask);q=normalize_skeleton(xy,confidence,valid_mask=original_mask).reshape(17,2)
    old=next(r for r in json.loads((a.source/'rankings.json').read_text()) if r['id']=='rough_315')
    assert hashlib.sha256(q.tobytes()).hexdigest()==old['shared_xy_sha256']
    with np.load(a.source/'costs/rough_315.npz',allow_pickle=False) as d:
        original_cost=.25*d['base'][selected].astype(float)+.75*d['fit'][selected].astype(float);original_params=d['parameters'][selected].copy()
    cameras117=camera_grid()['yaw_pitch_perspective_117'];cameras=cameras117+[Camera(c.yaw,c.pitch,dist) for c in camera_grid()['yaw_pitch_39'] for dist in (2.,3.)]
    index=build_camera_index(poses,cameras);coarse=(index[0][:117],index[1][:117]);alt,am=alternate_labels(q,mask)
    first,_=best_views(coarse,q,mask);other,_=best_views(coarse,alt,am);base=np.minimum(first,other)
    fit,params=refine_costs(normalized_joints(joints[selected]),index,cameras,q,mask)
    cost=.25*base.astype(float)+.75*fit.astype(float);records={}
    for method,values,parameters,used_mask in (('original_body_control',original_cost,original_params,original_mask),('torso_legs_experiment',cost,params,mask)):
        rows=[]
        for rank,j in enumerate(np.argsort(values,kind='stable'),1):
            pid=poses[j].pose_id;yaw,pitch,inv=map(float,parameters[j]);camera=Camera(yaw,pitch,None if inv==0 else 1/inv)
            projected,valid=project(poses[j].joints,camera);assert valid
            normalized=normalize_batch(projected);aquery,amask=alternate_labels(q,used_mask)
            direct=np.flatnonzero(used_mask[5:])+5;swapped=np.flatnonzero(amask[5:])+5
            refit_cost=min(np.linalg.norm(normalized[direct]-q[direct],axis=-1).mean(),np.linalg.norm(normalized[swapped]-aquery[swapped],axis=-1).mean())
            if method=='torso_legs_experiment':assert abs(float(refit_cost)-float(fit[j]))<2e-5
            rows.append({'rank_in_gate_subset':rank,'pose_id':pid,'cost_in_this_objective':float(values[j]),
                         'camera':{'yaw':yaw,'pitch':pitch,'distance':camera.distance},
                         'full_original_query_deviation':body_part_errors(q,original_mask,normalized),
                         'geometry_predicates':requirement_metrics(poses[j].joints)})
        records[method]=rows
    save_json(a.out/'rankings-diagnostic.json',records)
    np.savez_compressed(a.out/'costs.npz',pose_ids=[p.pose_id for p in poses],query=q,original_mask=original_mask,experiment_mask=mask,
                        original_cost=original_cost,original_params=original_params,base=base,fit=fit,cost=cost,parameters=params)
    result={'candidate_subset':len(selected),'original_observed_body':int(original_mask[5:].sum()),'experiment_observed_body':int(mask[5:].sum()),
            'control_first':records['original_body_control'][0]['pose_id'],'experiment_first':records['torso_legs_experiment'][0]['pose_id'],
            'accuracy':None,'original_results_unchanged':True,'next':'verify whole crouch and folded-arm semantics together; do not adopt from subset cost alone'}
    save_json(a.out/'results.json',result)
    for p,h in hashes.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('source','gallery','out'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
