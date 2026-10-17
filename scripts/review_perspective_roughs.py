"""Local, fixed-policy before/after review on ten existing rough test cases.

Outputs stay ignored: copyrighted input artwork and library files are not pushed.
"""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from standin_eval.z_diagnostics import infer_pose
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,normalize_batch,project
from standin_eval.dual_label_search import alternate_labels
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.contour_rerank import candidate_evidence,rerank
from standin_eval.contour_letterbox import prepare_letterboxed_query
from src.features import normalize_skeleton

SEED=20261013
EDGES=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def rgb(path):
    with Image.open(path) as im:
        rgba=im.convert('RGBA'); white=Image.new('RGBA',rgba.size,'white'); white.alpha_composite(rgba)
        return np.asarray(white.convert('RGB'))


def select_cases(dataset_root):
    """All5 single-person/full-body set01 cases, plus fixed5 from other sets."""
    primary=[]; pool=[]
    for manifest in sorted(dataset_root.glob('set-*/dataset.json')):
        data=json.loads(manifest.read_text())
        for cut in data['cuts']:
            if cut['person_count']!=1 or cut['visible_ranges']!=['전신']: continue
            image=Path(cut['image_path'])
            if not image.is_file(): image=manifest.parent/'images'/Path(cut['image_path']).name
            record={'id':cut['source_id'],'set':manifest.parent.name,'path':str(image.resolve()),
                    'sha256':cut['image_sha256'],'dataset_sha256':sha256(manifest)}
            (primary if manifest.parent.name=='set-01' else pool).append(record)
    assert len(primary)==5 and len(pool)>=5
    chosen=np.random.default_rng(SEED).choice(len(pool),5,replace=False)
    selected=primary+[pool[int(i)] for i in chosen]
    assert len({r['sha256'] for r in selected})==10
    for r in selected: assert sha256(Path(r['path']))==r['sha256']
    return selected


def best_views(index,xy,mask):
    features,valid=index; selected=np.flatnonzero(mask[5:])+5
    costs=np.linalg.norm(features[:,:,selected]-xy[selected],axis=-1).mean(axis=-1)
    costs=np.where(valid,costs,np.inf); views=costs.argmin(axis=0)
    return costs[views,np.arange(costs.shape[1])],views


def keep_first_order(base,score):
    first=int(np.argmin(base)); order=np.argsort(score,kind='stable')
    return np.concatenate(([first],order[order!=first]))


def preview(joints,camera,size=320):
    try:
        im,xy=render_pose(joints,camera.yaw,size=size,pitch=camera.pitch,distance=camera.distance)
        expected,valid=project(joints,camera)
        assert valid and np.allclose(normalize_batch(xy),normalize_batch(expected),atol=1e-5)
        return im,'3D 포즈',None
    except ValueError as exc:
        xy,valid=project(joints,camera)
        if not valid: raise
        low=xy[5:].min(0); span=max(float(np.ptp(xy[5:],axis=0).max()),1e-6)
        pixels=(xy-low)*(size-48)/span+24
        im=Image.new('RGB',(size,size),'white'); d=ImageDraw.Draw(im)
        for a,b in EDGES: d.line([tuple(pixels[a]),tuple(pixels[b])],fill='#247b9b',width=3)
        return im,'관절 투영 (표면 렌더 실패)',str(exc)


def data_url(im,size=360):
    im=im.copy(); im.thumbnail((size,size)); buf=io.BytesIO(); im.save(buf,format='WEBP',quality=75)
    return 'data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()


def run(args):
    from rtmlib import RTMPose
    cases=select_cases(args.dataset_root)
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('predictions','costs','sheets'): (args.out/name).mkdir()
    code=(['scripts/review_perspective_roughs.py','scripts/templates/perspective_rough_review.html',
           'scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py']+
          ['standin_eval/'+n+'.py' for n in ('z_axis_search','z_diagnostics','perspective_search',
           'dual_label_search','camera_refinement','contour_rerank','contour_letterbox')]+['src/features.py','src/bvh.py'])
    policy={'study_id':'rough-test-perspective-20261013-v1','seed':SEED,'cases':cases,
            'selection':'set01 all5 single/full + uniform5 remaining sets; fixed before inference; no replacements',
            'before':'four views0/45/90/180, direct labels, same Human-Art XY/mask, raw exact pose IDs',
            'after':'dual labels;117+78 near camera fit, score .25*117+.75*fit; keep117 first; contour1 rerank2..20',
            'xy':'Human-Art RGB192x256, foreground crop,4 rotations; >=6 body plus shoulders/hips>=.3',
            'contour':'256px white square padding, RGB area resize thenmin<200, cap.5; any error keeps camera blend',
            'render':'matched camera per method; canonical45deg available; fitted view is fit component, not a single blend camera; face/hands approximate',
            'ground_truth':'no verified source3D IDs; draft2D GT not used; human review pending; not deployed pipeline replay',
            'files':{p:sha256(ROOT/p) for p in code},'model_sha256':sha256(args.model),
            'library_db_sha256':sha256(args.library_root/'poses.db')}
    save_json(args.out/'protocol.json',policy)
    model=RTMPose(str(args.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    poses=load_library(args.library_root); cameras117=camera_grid()['yaw_pitch_perspective_117']
    cameras=cameras117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cameras); index117=(index[0][:117],index[1][:117])
    cameras4=camera_grid()['baseline_4']; index4=build_camera_index(poses,cameras4)
    points=normalized_joints(np.stack([p.joints for p in poses])); rows=[]; public=[]
    font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',20)
    for case in cases:
        image=rgb(Path(case['path'])); row=dict(case); item={'id':case['id'],'set':case['set'],'input':data_url(Image.fromarray(image),480)}
        pred=infer_pose(image,model,crop=True,turns=(0,1,2,3),is_3d=False)
        save_json(args.out/'predictions'/(case['id']+'.json'),pred)
        xy=np.asarray(pred['keypoints2d'],np.float32); scores=np.asarray(pred['scores'],np.float32); mask=scores>=.3
        overlay=Image.fromarray(image); draw=ImageDraw.Draw(overlay); width=max(2,int(max(image.shape[:2])/200))
        for a,b in EDGES:
            if mask[a] and mask[b]: draw.line([tuple(xy[a]),tuple(xy[b])],fill='#008aa6',width=width)
        for j,p in enumerate(xy):
            if mask[j]: draw.ellipse((p[0]-width,p[1]-width,p[0]+width,p[1]+width),fill='#df8034')
        item['overlay']=data_url(overlay,480); row['valid_mask']=mask.tolist()
        if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            row['error']='Human-Art 관절 기준 미달'; item['error']=row['error']; item['before']=[]; item['after']=[]
        else:
            q=normalize_skeleton(xy,scores,valid_mask=mask).reshape(17,2)
            row['shared_xy_sha256']=hashlib.sha256(q.tobytes()).hexdigest()
            alternate,am=alternate_labels(q,mask)
            original,views4=best_views(index4,q,mask)
            direct,dv=best_views(index117,q,mask); other,av=best_views(index117,alternate,am)
            base=np.minimum(direct,other); views=np.where(other<direct,av,dv)
            fit,params=refine_costs(points,index,cameras,q,mask)
            score=.25*base.astype(float)+.75*fit.astype(float); blend=keep_first_order(base,score)
            penalties=np.zeros(20); errors={}
            try:
                evidence,_=prepare_letterboxed_query(image,xy)
                for k,j in enumerate(blend[1:20],1):
                    try: penalties[k]=candidate_evidence(evidence,poses[j].joints,params[j])[0]
                    except ValueError as exc: errors[str(k)]=str(exc)
            except ValueError as exc: errors['query']=str(exc)
            after=blend if errors else rerank(blend,score,penalties,1.)
            assert after[0]==blend[0] and np.array_equal(after[20:],blend[20:])
            row['contour_fallback']=errors; item['contour_fallback']=bool(errors)
            path=args.out/'costs'/(case['id']+'.npz')
            np.savez_compressed(path,original=original,base=base,fit=fit,parameters=params,contour=penalties,
                                before_order=np.argsort(original,kind='stable'),blend_order=blend,after_order=after)
            row['costs_sha256']=sha256(path)
            for arm,indices in (('before',np.argsort(original,kind='stable')),('after',after)):
                row[arm]=[]; item[arm]=[]
                for rank,j in enumerate(indices[:5],1):
                    if arm=='before': cam=cameras4[int(views4[j])]
                    elif rank==1: cam=cameras[int(views[j])]
                    else:
                        yaw,pitch,inverse=map(float,params[j]); cam=Camera(yaw,pitch,None if inverse==0 else 1/inverse)
                    rendered,kind,error=preview(poses[j].joints,cam)
                    canonical,ckind,cerror=preview(poses[j].joints,Camera(45))
                    c={'pose_id':poses[j].pose_id,'pose_index':int(j),'camera':asdict(cam),'kind':kind,'render_error':error}
                    row[arm].append(c); item[arm].append(dict(c,matched=data_url(rendered),canonical=data_url(canonical),canonical_kind=ckind))
        rows.append(row); public.append(item)
        sheet=Image.new('RGB',(1500,840),'#f4f6f8'); d=ImageDraw.Draw(sheet)
        d.text((15,12),case['id']+' · '+case['set']+' · 검수 전 · 얼굴/손 방향은 근사',font=font,fill='#24364c')
        im=Image.fromarray(image); im.thumbnail((230,650)); sheet.paste(im,(10,80))
        if row.get('error'): d.text((290,150),row['error']+' · 두 방법 검색 실패',font=font,fill='#a53a25')
        else:
            for arm,y,label in (('before',60,'Before · 기존 네 시점'),('after',440,'After · 좌우 두 해석 + 원근·카메라 + 윤곽')):
                d.text((270,y),label,font=font,fill='#24364c')
                for k,c in enumerate(item[arm]):
                    rendered=Image.open(io.BytesIO(base64.b64decode(c['matched'].split(',')[1]))); rendered.thumbnail((235,260))
                    x=270+k*244; sheet.paste(rendered,(x,y+40))
                    d.text((x,y+312),f'{k+1}위 · #{c["pose_index"]}',font=font,fill='#24364c')
        sheet.save(args.out/'sheets'/(case['id']+'.png'))
        print('reviewed',case['id'],'failed' if row.get('error') else '10 candidates',flush=True)
    save_json(args.out/'rankings.json',rows)
    data={'study_id':policy['study_id'],'queries':public}
    save_json(args.out/'review-data-private.json',data)
    template=(ROOT/'scripts/templates/perspective_rough_review.html').read_text()
    html=template.replace('__REVIEW_DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    assert len(html.encode())<1024*1024,'inline review too large'
    (args.out/'review-inline.html').write_text(html,encoding='utf-8')
    valid=[r for r in rows if not r.get('error')]
    summary={'queries':10,'searchable':len(valid),'failed':10-len(valid),'accuracy':None,'human_ratings_received':0,
             'top1_changed':sum(r['before'][0]['pose_id']!=r['after'][0]['pose_id'] for r in valid),
             'top5_set_changed':sum({c['pose_id'] for c in r['before']}!={c['pose_id'] for c in r['after']} for r in valid),
             'contour_fallbacks':sum(bool(r['contour_fallback']) for r in valid),'gallery_count':len(poses),
             'rankings_sha256':sha256(args.out/'rankings.json'),'inline_bytes':len(html.encode())}
    for p,h in policy['files'].items(): assert sha256(ROOT/p)==h
    save_json(args.out/'summary.json',summary); print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dataset-root','library-root','model','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    (ROOT/'out').mkdir(parents=True,exist_ok=True)
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); run(a)
