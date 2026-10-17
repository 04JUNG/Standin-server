"""Frozen Human-Art XY vs identical XY + corrected RTMW3D Z, local only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256,retrieve
from standin_eval.z_axis_search import build_search_index
from standin_eval.z_diagnostics import infer_pose,foreground_bbox
from standin_eval.mixed_z import corrected_rtmw,compose_query,model_agreement


def select_real(root):
    rows=[json.loads(p.read_text()) for p in sorted((root/'러프_분류/메타데이터/개별 분류').glob('*.json'))]
    rng=np.random.default_rng(20261012)
    selected=[]
    for style in ('단순 러프','상세 러프'):
        pool=[r for r in rows if r['person_count_label']=='1명' and r['visible_range_labels']==['전신'] and r['expression_subtype']==style]
        for ix in rng.choice(len(pool),5,replace=False):
            r=pool[int(ix)]
            path=root/r['source_image']
            assert sha256(path)==r['source_sha256']
            selected.append({'image':r['file_name'],'path':str(path.resolve()),
                             'sha256':r['source_sha256'],'style':style,
                             'metadata_human_verified':r['human_verified'],
                             'target_pose_id':None})
    return selected


def read_rgb(path):
    with Image.open(path) as source:
        rgba=source.convert('RGBA')
        white=Image.new('RGBA',rgba.size,'white')
        white.alpha_composite(rgba)
        return np.asarray(white.convert('RGB'))


def aggregate(records,has_answers):
    n=len(records)
    success=[r for r in records if not r.get('error')]
    summary={'samples':n,'xy_extraction_failures':n-len(success),
        'z_inference_fallbacks':sum(r.get('z_fallback',False) for r in records),
        'top1_changed':sum(r['xy'][0]['pose_id']!=r['xyz'][0]['pose_id'] for r in success),
        'top5_set_changed':sum({c['pose_id'] for c in r['xy']}!={c['pose_id'] for c in r['xyz']} for r in success),
        'top5_order_changed':sum([c['pose_id'] for c in r['xy']]!=[c['pose_id'] for c in r['xyz']] for r in success),
        'mean_top5_overlap':float(np.mean([len({c['pose_id'] for c in r['xy']}&{c['pose_id'] for c in r['xyz']}) for r in success])) if success else None}
    if has_answers:
        summary['accuracy']={arm:{f'top{k}':sum(r.get(arm+'_rank') is not None and r[arm+'_rank']<=k for r in records)/n for k in (1,5)} for arm in ('xy','xyz')}
        hit=lambda r,arm:r.get(arm+'_rank') is not None and r[arm+'_rank']<=5
        summary['top5_gains']=sum(not hit(r,'xy') and hit(r,'xyz') for r in records)
        summary['top5_losses']=sum(hit(r,'xy') and not hit(r,'xyz') for r in records)
    else:
        summary['accuracy']=None
        summary['reason']='No original library pose ID or verified 3D ground truth. Candidate change is not improvement.'
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--synthetic',type=Path,required=True)
    p.add_argument('--human-cache',type=Path,required=True)
    p.add_argument('--rough-root',type=Path,required=True)
    p.add_argument('--human-model',type=Path,required=True)
    p.add_argument('--z-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cached=json.loads(a.human_cache.read_text())
    answers=json.loads((a.synthetic/'answers.json').read_text())
    assert len(cached)==len(answers)==40
    synthetic=[]
    for pred,answer in zip(cached,answers):
        assert pred['image']==answer['image']
        path=a.synthetic/'images'/answer['image']
        assert sha256(path)==answer['sha256']
        synthetic.append({'image':answer['image'],'path':str(path.resolve()),
                          'sha256':answer['sha256'],'target_pose_id':answer['pose_id']})
    real=select_real(a.rough_root)
    policy={'arms':['Human-Art XY','identical Human-Art XY + RTMW3D Z'],
        'xy_source':'Human-Art M, RGB 192x256, pixel foreground crop, 4 rotations; median body score selects rotation',
        'synthetic_xy':'reuse frozen cache byte-for-byte; no new model selection',
        'real_xy':'same local Human-Art recipe, not full deployed pipeline',
        'z_source':'RTMW3D X RGB 288x384; exact same pixel crop and Human-Art selected rotation',
        'z_decoder':'Z bin count / 2 = 288, NOT image height 384. Corrected from raw SimCC coordinates.',
        'joint_mapping':'anatomical COCO17 first 17, identity, no oracle left/right swaps',
        'depth_units':'camera relative depth; hip midpoint subtracted; divide by fixed 0.5m torso prior',
        'z_weight':.25,'z_sign':'+Z away from camera; in-plane rotation does not flip depth',
        'threshold':.3,'valid_mask':'Human-Art only; identical in A and B',
        'failure':'<6 body joints or any missing shoulder/hip: both arms fail, keep denominator',
        'z_failure':'exception or nonfinite Z: B returns A unchanged, record fallback',
        'rtmw_scores':'diagnostic XY response values only, NOT calibrated Z confidence; no score filtering of Z',
        'gallery_views':[0,45,90,180],'target':'exact pose_id, collapse views only, no mirror equivalence',
        'sampling':'real 5 simple + 5 detailed, single-person full-body metadata; seed20261012, before inference',
        'reused_test_warning':'synthetic 40 previously examined, not a new unseen holdout',
        'rough_answers':'none; existing 2D annotations are unverified drafts, not 3D pose labels',
        'hashes':{'human_cache':sha256(a.human_cache),'human_model':sha256(a.human_model),
                  'z_model':sha256(a.z_model),'library_db':sha256(a.library_root/'poses.db'),
                  'script':sha256(Path(__file__)),'fusion_module':sha256(ROOT/'standin_eval/mixed_z.py')},
        'synthetic':synthetic,'real':real}
    save_json(a.out/'protocol.json',policy)
    from rtmlib import RTMPose
    hm=RTMPose(str(a.human_model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    zm=corrected_rtmw(a.z_model)
    save_json(a.out/'z_runtime.json',{'outputs':[{'name':v.name,'shape':v.shape} for v in zm.session.get_outputs()]})
    index=build_search_index(load_library(a.library_root))
    all_records={}
    for group,items in (('synthetic',synthetic),('real',real)):
        folder=a.out/group
        folder.mkdir()
        records=[]
        for i,item in enumerate(items):
            rgb=read_rgb(item['path'])
            rec={**item,'width':rgb.shape[1],'height':rgb.shape[0]}
            try:
                h=cached[i] if group=='synthetic' else infer_pose(rgb,hm,crop=True,turns=(0,1,2,3),is_3d=False)
                rec['human']=h
                turn=h['turn']
                rec['crop_in_rotated_image']=foreground_bbox(np.ascontiguousarray(np.rot90(rgb,turn)))
                zpred=None
                try:
                    zpred=infer_pose(rgb,zm,crop=True,turns=(turn,),is_3d=True)
                    assert np.isfinite(zpred['z_meters']).all()
                    rec['rtmw']=zpred
                    rec['z_fallback']=False
                    rec['agreement']=model_agreement(h['keypoints2d'],zpred['keypoints2d'])
                except Exception as exc:
                    rec['z_fallback']=True
                    rec['z_error']=type(exc).__name__+': '+str(exc)
                xy,z,mask=compose_query(h['keypoints2d'],h['scores'],zpred['z_meters'] if zpred and not rec['z_fallback'] else np.zeros(17))
                rec['shared_xy']=xy.tolist(); rec['shared_mask']=mask.tolist(); rec['normalized_z']=z.tolist()
                rec['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
                rec['valid_body_joints']=int(mask[5:].sum())
                if mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                    rec['error']='insufficient Human-Art body joints/anchors'
                else:
                    for arm,w in (('xy',0.),('xyz',0. if rec['z_fallback'] else .25)):
                        candidates=retrieve(index,xy,z,mask,w)
                        rec[arm]=candidates[:5]
                        if item['target_pose_id'] is not None:
                            rec[arm+'_rank']=next(j+1 for j,c in enumerate(candidates) if c['pose_id']==item['target_pose_id'])
            except Exception as exc:
                rec['error']=type(exc).__name__+': '+str(exc)
            records.append(rec)
            save_json(folder/(Path(item['image']).stem+'.json'),rec)
            print(group,i+1,len(items),rec.get('error','ok'),flush=True)
        all_records[group]={'summary':aggregate(records,group=='synthetic'),'records':records}
        save_json(folder/'results.json',all_records[group])
        print(group,all_records[group]['summary'],flush=True)
    save_json(a.out/'results.json',all_records)


if __name__=='__main__':
    main()
