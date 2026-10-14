"""Frozen fit-blend on ten existing real roughs, with scientific pose sheets.

No source3D labels are supplied or inferred. Candidate changes are not accuracy.
"""
import argparse
import fcntl
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_refinement_blend import blend_cost
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,normalize_batch,project
from standin_eval.dual_label_search import alternate_labels,merged_order

CONFIRM=ROOT/'out/refinement-blend-confirm-20261013'
CACHE=ROOT/'out/humanart-plus-z-20261012/results.json'
OLD_REVIEW=ROOT/'out/real-rough-blind-review-20261012/rankings-private.json'
ARMS=('anchor_4','pooled_117','keepfirst_blend75')
EDGES=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def best_views(index,q,mask):
    features,valid=index;selected=np.flatnonzero(mask[5:])+5
    cost=np.linalg.norm(features[:,:,selected]-q[selected],axis=-1).mean(axis=-1)
    cost=np.where(valid,cost,np.inf);views=cost.argmin(axis=0)
    return cost[views,np.arange(cost.shape[1])],views


def params_camera(values):
    yaw,pitch,inv=map(float,values)
    return Camera(yaw,pitch,None if inv==0 else 1./inv)


def projection_preview(joints,camera,size=240):
    try:
        im,xy=render_pose(joints,camera.yaw,size=size,pitch=camera.pitch,distance=camera.distance)
        expected,valid=project(joints,camera)
        assert valid and np.allclose(normalize_batch(xy),normalize_batch(expected),atol=1e-5)
        return im,'3D 타원체',None
    except ValueError as error:
        # Never silently move a camera when the volume renderer rejects it.
        xy,valid=project(joints,camera)
        if not valid:raise
        p=xy[5:];low=p.min(axis=0);span=np.maximum(p.max(axis=0)-low,1e-6)
        pixels=(xy-low)*(size-40)/max(span)+20
        im=Image.new('RGB',(size,size),'white');draw=ImageDraw.Draw(im)
        for a,b in EDGES:draw.line([tuple(pixels[a]),tuple(pixels[b])],fill='#3c647a',width=3)
        for x,y in pixels[5:]:draw.ellipse((x-3,y-3,x+3,y+3),fill='#263f4c')
        return im,'투영 관절만',str(error)


def paste_fit(canvas,source,box):
    x,y,w,h=box
    im=source.convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
    im=bg.convert('RGB');im.thumbnail((w,h))
    canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))


def run(args):
    confirmed=read(CONFIRM/'protocol.json');result=read(CONFIRM/'results.json');audit=read(CONFIRM/'audit.json')
    assert result['gate_passed'] and confirmed['selected']=='keepfirst_blend75' and confirmed['alpha']==.75
    assert audit['ranking_sha256']==sha256(CONFIRM/'rankings.json')
    for p,digest in confirmed['frozen_sources'].items():assert sha256(ROOT/p)==digest
    assert read(ROOT/'out/humanart-plus-z-20261012/protocol.json')['hashes']['human_model']==confirmed['hashes']['model']
    poses=load_library(args.library_root);assert geometry_hash(poses)==confirmed['hashes']['geometry']
    assert sha256(args.library_root/'poses.db')==confirmed['hashes']['library_db']
    records=read(CACHE)['real']['records'];assert len(records)==10
    old={r['image']:r for r in read(OLD_REVIEW)}
    args.out.mkdir(parents=True,exist_ok=False)
    for d in ('assets','sheets','costs'):(args.out/d).mkdir()
    sources=[CACHE,OLD_REVIEW,CONFIRM/'protocol.json',CONFIRM/'results.json',CONFIRM/'audit.json']
    hashes={str(p):sha256(p) for p in sources}
    policy={'stage':'fixed policy application to existing10 roughs; not an independent labeled accuracy test',
        'arms':ARMS,'selected':'keepfirst_blend75','alpha':.75,
        'input':'same cached normalized HumanArt XY and mask, no new inference',
        'unknown_ground_truth':'no verified source3D IDs, accuracy remains null; no assistant relevance ratings',
        'display':'rank1 retained from pooled117 and shown with its117 camera; other blend candidates use fitted-camera component view, not a single blend camera; small reference is same45degree orthographic pose',
        'preview_failure':'if fitted view intersects renderer volume, show projected skeleton at exact camera and label it, never silently change camera',
        'limits':['convenience sample already examined','face/hands approximate','rough309 related gallery metadata, not unseen'],
        'input_hashes':hashes,'script_sha256':sha256(Path(__file__)),'confirmed_sources':confirmed['frozen_sources']}
    save_json(args.out/'protocol.json',policy)
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cameras);idx117=(index[0][:117],index[1][:117])
    oldcams=camera_grid()['baseline_4'];idx4=build_camera_index(poses,oldcams)
    points=normalized_joints(np.stack([p.joints for p in poses]))
    rows=[]
    for item in records:
        path=Path(item['path']);assert sha256(path)==item['sha256']
        row={k:item[k] for k in ('image','path','sha256')}
        if item.get('error'):
            row['error']=item['error'];rows.append(row);continue
        q=np.asarray(item['shared_xy'],dtype=np.float32);mask=np.asarray(item['shared_mask'],bool)
        assert hashlib.sha256(q.tobytes()).hexdigest()==item['shared_xy_sha256']
        assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
        row['shared_xy_sha256']=item['shared_xy_sha256'];row['shared_mask']=mask.tolist()
        alt,am=alternate_labels(q,mask)
        original,ov=best_views(idx4,q,mask)
        direct,dv=best_views(idx117,q,mask);other,av=best_views(idx117,alt,am)
        base=np.minimum(direct,other);views=np.where(other<direct,av,dv)
        fit,params=refine_costs(points,index,cameras,q,mask)
        score=blend_cost(base,fit,.75);first=int(np.argmin(base))
        neworder=merged_order(score,score);neworder=np.concatenate(([first],neworder[neworder!=first]))
        orders={'anchor_4':merged_order(original,original),'pooled_117':merged_order(base,base),'keepfirst_blend75':neworder}
        for arm,order in orders.items():
            candidates=[]
            for rank,j in enumerate(order[:5],1):
                j=int(j);cam=oldcams[int(ov[j])] if arm=='anchor_4' else cameras[int(views[j])]
                mode='기존 카메라'
                if arm=='keepfirst_blend75' and rank>1:cam=params_camera(params[j]);mode='미세 조정 성분의 카메라'
                candidates.append({'pose_id':poses[j].pose_id,'pose_index':j,'camera':asdict(cam),'view_note':mode,
                    'baseline117_camera':asdict(cameras[int(views[j])]),'fitted_camera':asdict(params_camera(params[j])),
                    'baseline117_cost':float(base[j]),'fitted_cost':float(fit[j]),'blend_cost':float(score[j])})
            row[arm]=candidates
        assert [c['pose_id'] for c in row['anchor_4']]==[c['pose_id'] for c in item['xy'][:5]]
        assert [c['pose_id'] for c in row['pooled_117']]==[c['pose_id'] for c in old[item['image']]['pooled_117']]
        assert row['keepfirst_blend75'][0]['pose_id']==row['pooled_117'][0]['pose_id']
        np.savez_compressed(args.out/'costs'/(Path(item['image']).stem+'.npz'),original=original,baseline=base,refined=fit,parameters=params)
        rows.append(row);print('ranked',item['image'],flush=True)
    fontpath='/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,15);title=ImageFont.truetype(fontpath,26)
    assets={};fallbacks=[]
    def preview(j,cam):
        key=json.dumps([j,asdict(cam)],sort_keys=True)
        filename=hashlib.sha256(key.encode()).hexdigest()[:20]+'.png'
        if filename not in assets:
            im,kind,error=projection_preview(poses[j].joints,cam)
            im.save(args.out/'assets'/filename);assets[filename]={'kind':kind,'sha256':sha256(args.out/'assets'/filename)}
            if error:fallbacks.append({'pose_id':poses[j].pose_id,'camera':asdict(cam),'reason':error,'image':filename})
        return filename,assets[filename]['kind']
    for row,item in zip(rows,records):
        sheet=Image.new('RGB',(1500,1120),'#f3f5f7');draw=ImageDraw.Draw(sheet)
        draw.text((22,14),row['image']+' · 동일 관절로 검색 후보 비교',font=title,fill='#203342')
        draw.text((22,52),'정답 ID 없음 · 후보 변경 ≠ 정확도 개선 · 얼굴/손가락 방향은 근사',font=font,fill='#715525')
        with Image.open(row['path']) as im:
            paste_fit(sheet,im,(15,112,260,420))
            rgba=im.convert('RGBA');white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
            overlay=white.convert('RGB');od=ImageDraw.Draw(overlay);raw=np.asarray(item['human']['keypoints2d']);scores=np.asarray(item['human']['scores'])
            width=max(2,int(max(im.size)/220))
            for a,b in EDGES:
                if scores[a]>=.3 and scores[b]>=.3:od.line([tuple(raw[a]),tuple(raw[b])],fill='#0086a4',width=width)
            paste_fit(sheet,overlay,(15,610,260,420))
        draw.text((25,88),'입력 러프',font=font,fill='#203342');draw.text((25,575),'공통 Human-Art 관절',font=font,fill='#203342')
        if row.get('error'):
            draw.text((340,350),'관절 추출 기준 미달 · 모든 방법 검색 실패',font=title,fill='#a64532')
        else:
            labels=('A · 기존 XY 네 시점','B · 두 해석과 117개 시점','C · 카메라 미세 조정 혼합 (선택 설정)')
            baseids={c['pose_id'] for c in row['pooled_117']}
            for m,arm in enumerate(ARMS):
                y=92+m*330;draw.text((300,y),labels[m],font=font,fill='#203342')
                for rank,c in enumerate(row[arm],1):
                    x=300+(rank-1)*237
                    draw.rectangle((x,y+32,x+225,y+312),fill='white')
                    name,kind=preview(c['pose_index'],Camera(**c['camera']));c['preview']='assets/'+name;c['preview_kind']=kind
                    canonical,_=preview(c['pose_index'],Camera(45));c['canonical_preview']='assets/'+canonical
                    with Image.open(args.out/c['preview']) as im:paste_fit(sheet,im,(x+3,y+55,216,215))
                    with Image.open(args.out/c['canonical_preview']) as im:paste_fit(sheet,im,(x+150,y+229,70,70))
                    new=arm=='keepfirst_blend75' and c['pose_id'] not in baseids
                    draw.text((x+8,y+35),f'{rank}위'+(' · B에 없던 후보' if new else ''),font=small,fill='#8a5421' if new else '#203342')
                    draw.text((x+8,y+275),f'포즈 #{c["pose_index"]} · {kind}',font=small,fill='#3b4b55')
            draw.text((300,1090),'큰 그림: 해당 방법의 참고 시점 / 작은 그림: 같은 포즈의 공통 45° 직교 시점',font=small,fill='#4f5d68')
        sheetpath=args.out/'sheets'/(Path(row['image']).stem+'.png');sheet.save(sheetpath)
        row['sheet']='sheets/'+sheetpath.name;row['sheet_sha256']=sha256(sheetpath)
    for p,digest in hashes.items():assert sha256(Path(p))==digest
    save_json(args.out/'rankings.json',rows);save_json(args.out/'asset-manifest.json',assets)
    save_json(args.out/'render-fallbacks.json',fallbacks)
    valid=[r for r in rows if not r.get('error')]
    pairs={}
    for arm,base in ((ARMS[2],ARMS[1]),(ARMS[2],ARMS[0])):
        pairs[arm+'_vs_'+base]={'top1_changed':sum(r[arm][0]['pose_id']!=r[base][0]['pose_id'] for r in valid),
            'top5_set_changed':sum({c['pose_id'] for c in r[arm]}!={c['pose_id'] for c in r[base]} for r in valid),
            'new_candidates_per_image':{r['image']:len({c['pose_id'] for c in r[arm]}-{c['pose_id'] for c in r[base]}) for r in valid}}
    summary={'queries':10,'searchable':len(valid),'failed':10-len(valid),'accuracy':None,'human_ratings_received':0,
        'candidate_changes':pairs,'render_fallbacks':len(fallbacks),'warning':'candidate changes only, no source ID accuracy or independent human relevance judgment'}
    save_json(args.out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--library-root',type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.real-refinement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
