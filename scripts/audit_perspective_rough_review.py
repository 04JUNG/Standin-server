"""Replay costs/ranks/cameras and verify local review media before upload."""
import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.review_perspective_roughs import select_cases,preview
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_z_axis_search import load_library
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index
from standin_eval.dual_label_search import alternate_labels
from standin_eval.camera_refinement import normalized_joints,refine_costs
from src.features import normalize_skeleton


def main(a):
    policy=json.loads((a.study/'protocol.json').read_text()); rows=json.loads((a.study/'rankings.json').read_text())
    summary=json.loads((a.study/'summary.json').read_text()); public=json.loads((a.study/'review-data-private.json').read_text())
    assert sha256(a.study/'rankings.json')==summary['rankings_sha256']
    assert select_cases(a.dataset_root)==policy['cases']
    for name,digest in policy['files'].items():
        if name=='scripts/templates/perspective_rough_review.html': continue
        assert sha256(ROOT/name)==digest,name
    assert sha256(a.library_root/'poses.db')==policy['library_db_sha256']
    poses=load_library(a.library_root);points=normalized_joints(np.stack([p.joints for p in poses]))
    cameras117=camera_grid()['yaw_pitch_perspective_117']; cameras=cameras117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cameras);index117=(index[0][:117],index[1][:117]); cams4=camera_grid()['baseline_4'];index4=build_camera_index(poses,cams4)
    assert len(rows)==len(public['queries'])==len({r['id'] for r in rows})==10
    checked=images=0
    for row,item in zip(rows,public['queries']):
        assert row['id']==item['id'] and sha256(Path(row['path']))==row['sha256']
        pred=json.loads((a.study/'predictions'/(row['id']+'.json')).read_text())
        xy=np.asarray(pred['keypoints2d'],np.float32); scores=np.asarray(pred['scores'],np.float32);mask=scores>=.3
        assert row['valid_mask']==mask.tolist()
        if row.get('error'):
            assert mask[5:].sum()<6 or not mask[[5,6,11,12]].all()
            assert item['error']==row['error'] and item['before']==item['after']==[]
        else:
            q=normalize_skeleton(xy,scores,valid_mask=mask).reshape(17,2)
            assert hashlib.sha256(q.tobytes()).hexdigest()==row['shared_xy_sha256']
            alt,am=alternate_labels(q,mask)
            original=score_camera_index(index4,q,mask)
            base=np.minimum(score_camera_index(index117,q,mask),score_camera_index(index117,alt,am))
            fit,parameters=refine_costs(points,index,cameras,q,mask)
            costpath=a.study/'costs'/(row['id']+'.npz');assert sha256(costpath)==row['costs_sha256']
            with np.load(costpath,allow_pickle=False) as saved:
                assert np.array_equal(original,saved['original']) and np.array_equal(base,saved['base']) and np.array_equal(fit,saved['fit'])
                assert np.array_equal(parameters,saved['parameters'])
                first=int(np.argmin(base));score=.25*base.astype(float)+.75*fit.astype(float)
                blend=sorted(range(len(poses)),key=lambda j:(score[j],j));blend=[first]+[j for j in blend if j!=first]
                after=blend[:]
                if not row['contour_fallback']:
                    head=sorted(range(1,20),key=lambda k:(score[blend[k]]+saved['contour'][k],k))
                    after=[first]+[blend[k] for k in head]+blend[20:]
                assert saved['blend_order'].tolist()==blend and saved['after_order'].tolist()==after
                assert after[0]==first and after[20:]==blend[20:]
                for arm,indices in (('before',np.argsort(original,kind='stable')),('after',after)):
                    assert [c['pose_id'] for c in row[arm]]==[poses[j].pose_id for j in indices[:5]]
                    assert [c['pose_id'] for c in item[arm]]==[c['pose_id'] for c in row[arm]]
                    for c in row[arm]:
                        assert poses[c['pose_index']].pose_id==c['pose_id']
                        expected,kind,error=preview(poses[c['pose_index']].joints,Camera(**c['camera']))
                        assert kind==c['kind'] and error==c['render_error']
            checked+=1
        for encoded in [item['input'],item['overlay']]+[c[key] for arm in ('before','after') for c in item[arm] for key in ('matched','canonical')]:
            assert encoded.startswith('data:image/webp;base64,')
            with Image.open(io.BytesIO(base64.b64decode(encoded.split(',')[1]))) as im: im.verify()
            images+=1
        with Image.open(a.study/'sheets'/(row['id']+'.png')) as im: assert im.size==(1500,840);im.verify()
    valid=[r for r in rows if not r.get('error')]
    assert summary['searchable']==checked==len(valid) and summary['failed']==10-checked
    assert summary['accuracy'] is None and summary['human_ratings_received']==0
    assert summary['top1_changed']==sum(r['before'][0]['pose_id']!=r['after'][0]['pose_id'] for r in valid)
    assert summary['top5_set_changed']==sum({c['pose_id'] for c in r['before']}!={c['pose_id'] for c in r['after']} for r in valid)
    save_json(a.study/'audit.json',{'queries':10,'full_cost_camera_rank_replays':checked,'media_verified':images,
        'failed':10-checked,'ranking_sha256':sha256(a.study/'rankings.json'),'summary_sha256':sha256(a.study/'summary.json'),
        'auditor_sha256':sha256(Path(__file__)),'human_ratings':0})
    print('PASS',checked,'cost/camera/rank replays,',images,'review images',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','dataset-root','library-root'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
