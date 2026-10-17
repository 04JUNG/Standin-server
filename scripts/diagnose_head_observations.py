"""Post-hoc face-position diagnostic on previously used synthetic development.

No retrieval changes. Source geometry is diagnostic truth, never query input.
Nose mesh center is a renderer landmark, not verified anatomical ground truth.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_near_preview import capture,surface,plane
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_near_camera import read
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera,normalize_batch


def to_pixels(points,state,distance):
    return (plane(np.asarray(points),distance)-[state['cx'],state['cy']])*[state['scale'],-state['scale']]+256


def facing(center,basis,distance):
    view=np.array([0.,0.,-1.]) if distance is None else np.array([0.,0.,-distance])-center
    return float(np.dot(basis[:,2],view)/np.linalg.norm(view))


def summary(rows):
    selected=[r for r in rows if 'nose_score' in r]
    stats={'n':len(rows),'predictions':len(selected),'body_searchable':sum(r['body_searchable'] for r in rows)}
    for label,group in [('nose_ge_03',[r for r in selected if r['nose_score']>=.3]),
                        ('nose_ge_07',[r for r in selected if r['nose_score']>=.7]),
                        ('back_nose_not_visible',[r for r in selected if r['head_front_dot']<0 and r['visible_nose_pixels']==0])]:
        item={'n':len(group),'high_score_ge_07':sum(r['nose_score']>=.7 for r in group)}
        if group:
            for key in ('error_to_head_center_gt_torso','error_to_nose_marker_gt_torso','head_to_nose_marker_gt_torso'):
                item[key+'_median_p90']=np.quantile([r[key] for r in group],[.5,.9]).tolist()
            item['within_projected_head']=sum(r['inside_projected_head'] for r in group)
            item['head_center_error_le_025']=sum(r['error_to_head_center_gt_torso']<=.25 for r in group)
        stats[label]=item
    return stats


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[args.source/n for n in ('answers.json','manifest.json','rankings.json','audit.json')]
    paths+=sorted((args.source/'predictions').glob('*.json'))
    paths+=[ROOT/'scripts/experiment_z_rendered.py',ROOT/'src/bvh.py',ROOT/'scripts/diagnose_near_preview.py']
    hashes={str(p):sha256(p) for p in paths}
    save_json(args.out/'protocol.json',{
        'kind':'post-hoc development geometry diagnosis, not fresh holdout or search accuracy',
        'source':str(args.source),'sample':'all360 existing renders/120 families, retain body extraction failures',
        'comparisons':'Human-Art nose vs BVH Head center and rendered nose marker center; errors divided by GT projected torso length',
        'confidence':'describe score>=.3 and>=.7; not probabilities or visibility labels',
        'back':'head body-frame forward dot toward camera<0 AND no visible nose-object pixels; does not prove inferred hidden landmark is wrong',
        'head_region':'projected unoccluded head ellipsoid silhouette; not visible semantic face region',
        'body_fixed':'no XY,mask,search rankings or camera policies changed',
        'illustrations':'three highest-score back/no-nose cases and three highest head-center-error cases among score>=.7, unique images',
        'hashes':hashes,'script_sha256':sha256(Path(__file__))})
    poses={p.pose_id:p for p in load_library(args.library_root)}
    manifest={m['image']:m for m in read(args.source/'manifest.json')}
    rows=[]
    for target in read(args.source/'answers.json'):
        name=target['image'];path=args.source/'images'/name
        assert sha256(path)==manifest[name]['sha256']
        pose=poses[target['pose_id']];cam=Camera(**target['camera'])
        _,xy,state=capture(pose.joints,cam)
        gt=np.asarray(target['ground_truth_xy']);assert np.allclose(xy,gt,atol=1e-4)
        assert np.array_equal(pose.joints[:5],np.repeat(pose.joints[:1],5,axis=0))
        center,radii,basis,_=state['objects'][23]
        nose=state['objects'][24][0];nosepx=to_pixels(nose[None],state,cam.distance)[0]
        hull=cv2.convexHull(to_pixels(surface(center,radii,basis,64),state,cam.distance).astype(np.float32)).reshape(-1,2)
        pred=read(args.source/'predictions'/(Path(name).stem+'.json'))
        row={'image':name,'group':target['group'],'body_searchable':valid_query(pred) is not None,
             'head_front_dot':facing(center,basis,cam.distance),
             'visible_nose_pixels':int((state['surface_ids']==24).sum()),
             'head_center_pixels':xy[0].tolist(),'nose_marker_pixels':nosepx.tolist()}
        if not pred.get('error'):
            observed=np.asarray(pred['keypoints2d'],float);score=np.asarray(pred['scores'],float)
            if np.isfinite(observed[0]).all() and np.isfinite(score[0]):
                torso=np.linalg.norm((gt[5]+gt[6])/2-(gt[11]+gt[12])/2)
                assert torso>1e-6
                row.update(nose_score=float(score[0]),nose_pixels=observed[0].tolist(),
                    error_to_head_center_gt_torso=float(np.linalg.norm(observed[0]-xy[0])/torso),
                    error_to_nose_marker_gt_torso=float(np.linalg.norm(observed[0]-nosepx)/torso),
                    head_to_nose_marker_gt_torso=float(np.linalg.norm(xy[0]-nosepx)/torso),
                    inside_projected_head=bool(cv2.pointPolygonTest(hull,tuple(observed[0]),False)>=0))
                if row['body_searchable']:
                    row['retrieval_normalized_head_error']=float(np.linalg.norm(normalize_batch(observed)[0]-normalize_batch(gt)[0]))
        rows.append(row)
        if len(rows)%60==0:print('diagnosed',len(rows),'/360',flush=True)
    assert len(rows)==360
    stats={g:summary([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    stats['all']=summary(rows)
    back=sorted([r for r in rows if 'nose_score' in r and r['head_front_dot']<0 and r['visible_nose_pixels']==0],key=lambda r:(-r['nose_score'],r['image']))[:3]
    errors=sorted([r for r in rows if r.get('nose_score',0)>=.7],key=lambda r:(-r['error_to_head_center_gt_torso'],r['image']))
    examples=list(back)
    for row in errors:
        if row['image'] not in {r['image'] for r in examples}:examples.append(row)
        if len(examples)==6:break
    sheet=Image.new('RGB',(1536,1160),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(examples):
        im=Image.open(args.source/'images'/row['image']).convert('RGB');d=ImageDraw.Draw(im)
        for key,color in [('head_center_pixels','#2864bd'),('nose_marker_pixels','#29914a'),('nose_pixels','#d83c41')]:
            x,y=row[key];d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        x,y=(i%3)*512,(i//3)*580;sheet.paste(im,(x,y+45))
        draw.text((x+12,y+5),row['image'],fill='black')
        draw.text((x+12,y+23),f'nose score {row["nose_score"]:.3f}; head error {row["error_to_head_center_gt_torso"]:.3f} torso',fill='black')
        draw.text((x+12,y+558),'blue: Head | green: nose marker | red: detected nose',fill='black')
    sheet.save(args.out/'head-observation-examples.png')
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    save_json(args.out/'results.json',{'summary':stats,'records':rows,'example_images':[r['image'] for r in examples],
              'source_hashes_unchanged':True,'accuracy':None,'source_gt_projection_reproduced':360})
    print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','out','library-root'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
