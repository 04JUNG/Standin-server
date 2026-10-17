"""Apply fixed confirmed contour policy to unlabeled real roughs; no accuracy."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_near_camera import read
from scripts.experiment_z_rendered import save_json, sha256
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.contour_rerank import candidate_evidence, rerank
from standin_eval.contour_letterbox import prepare_letterboxed_query
from src.features import normalize_skeleton

SOURCE = ROOT/'out/refinement-real-roughs-20261013'
CACHE = ROOT/'out/humanart-plus-z-20261012/results.json'
CONFIRM = ROOT/'out/contour-rerank-confirm-20261013'
BASE = 'keepfirst_blend75'
NEW = 'contour100'
CODE = ('scripts/compare_contour_real_roughs.py', 'standin_eval/contour_letterbox.py',
        'standin_eval/contour_rerank.py', 'scripts/experiment_z_rendered.py',
        'scripts/experiment_head_hint.py', 'scripts/experiment_refinement_blend.py',
        'scripts/experiment_humanart_z.py', 'src/features.py')


def verify(out):
    p = read(out/'protocol.json')
    for name, digest in p['inputs'].items(): assert sha256(Path(name)) == digest, name
    for name, digest in p['code'].items(): assert sha256(ROOT/name) == digest, name


def make_sheet(out, row, masks):
    sheet = Image.new('RGB', (1540, 720), '#f3f5f7')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 18)
    draw.text((16,12), row['image']+' · 실제 러프 윤곽 비교 · 정답과 사람 평가 없음', font=font, fill='#172c40')
    with Image.open(row['path']) as im:
        im = im.convert('RGBA'); bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im)
        im = bg.convert('RGB'); im.thumbnail((245,560)); sheet.paste(im,(10,90))
    if row.get('error'):
        draw.text((290,160), '기존 관절 추출 기준 미달 · 두 방법 모두 검색 실패', font=font, fill='#8b3926')
    else:
        draw.text((270,42), '검정: 입력 선 / 청록: 후보 선 / 겹침: 주황 · 1위는 고정',font=font,fill='#172c40')
        lookup = {pose:i for i,pose in enumerate(row['shortlist'])}
        for arm, y in ((BASE,100),(NEW,400)):
            label = '기존 카메라 검색' if arm == BASE else '윤곽선 추가'
            if row['fallback'] and arm == NEW: label += ' · 렌더 실패로 기존 순위 유지'
            draw.text((270,y-30),label,font=font,fill='#172c40')
            for rank,pose in enumerate(row[arm]):
                x=270+rank*250; idx=lookup[pose]
                draw.text((x,y+245),f'{rank+1}위 · 후보 #{row["shortlist_indices"][idx]}',font=font,fill='#172c40')
                if idx == 0:
                    draw.text((x+12,y+70),'동일한 1위 유지',font=font,fill='#172c40'); continue
                key=f'candidate_{idx}'
                if key not in masks:
                    draw.text((x+12,y+70),'해당 시점 렌더 실패',font=font,fill='#8b3926'); continue
                pixels=np.full((256,256,3),255,np.uint8); q=masks['query']; c=masks[key]
                pixels[q]=[40,40,40]; pixels[c]=[0,150,165]; pixels[q&c]=[225,115,20]
                sheet.paste(Image.fromarray(pixels).resize((235,235)),(x,y))
    path=out/'sheets'/(Path(row['image']).stem+'.png'); sheet.save(path)
    row['sheet']='sheets/'+path.name; row['sheet_sha256']=sha256(path)


def run(args):
    assert read(CONFIRM/'results.json')['gate_passed']
    assert read(CONFIRM/'audit.json')['ranking_sha256'] == sha256(CONFIRM/'rankings.json')
    assert read(SOURCE/'audit.json')['ranking_sha256'] == sha256(SOURCE/'rankings.json')
    frozen=read(CONFIRM/'protocol.json')
    for name,digest in frozen['code'].items(): assert sha256(ROOT/name)==digest
    originals=read(SOURCE/'rankings.json'); cached={r['image']:r for r in read(CACHE)['real']['records']}
    assert len(originals)==len(cached)==10
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('masks','sheets'): (args.out/name).mkdir()
    inputs=[CACHE,SOURCE/'rankings.json',SOURCE/'audit.json',SOURCE/'protocol.json',
            CONFIRM/'protocol.json',CONFIRM/'results.json',CONFIRM/'audit.json',args.library_root/'poses.db']
    inputs+=sorted((SOURCE/'costs').glob('*.npz'))+[Path(r['path']) for r in originals]
    save_json(args.out/'protocol.json',{
        'stage':'fixed-policy application to existing10 unlabeled roughs; not holdout accuracy',
        'policy':{'baseline':BASE,'selected':NEW,'weight':1.,'limit':20,'size':256,
                  'threshold':200,'distance_cap':.5,'first_and_tail':'unchanged'},
        'preprocessing':'white symmetric integer padding to max(height,width); no crop/stretch/rotation; shift original pixel XY by left/top; frozen pixel-center resize256',
        'fixed':'same Human-Art XY/mask and all-gallery cached costs/cameras; no inference or Z or camera fit; no GT',
        'fallback':'query failure or ANY of19 candidate failures retains entire baseline order',
        'claims':'candidate changes only; accuracy null; no human preference labels; no threshold retuning',
        'limits':['whole-image contours include clothes/background/other people/text',
                  'same-renderer synthetic benefit may not transfer','convenience sample previously examined',
                  'first result fixed cannot improve; source-ID GT absent; approximate surface geometry'],
        'inputs':{str(p):sha256(p) for p in inputs},'code':{p:sha256(ROOT/p) for p in CODE}})
    poses=load_library(args.library_root)
    oldconfirmed=read(ROOT/'out/refinement-blend-confirm-20261013/protocol.json')
    assert geometry_hash(poses)==oldconfirmed['hashes']['geometry']
    assert sha256(args.library_root/'poses.db')==oldconfirmed['hashes']['library_db']
    rows=[]
    for old in originals:
        item=cached[old['image']]; row={k:old[k] for k in ('image','path','sha256')}; masks={}
        assert sha256(Path(row['path']))==row['sha256']==item['sha256']
        if old.get('error'):
            assert item.get('error'); row['error']=old['error']
        else:
            xy=np.asarray(item['human']['keypoints2d'],np.float32)
            confidence=np.asarray(item['human']['scores'],np.float32)
            mask=np.asarray(item['shared_mask'],bool)
            assert np.array_equal(mask,confidence>=.3) and mask[[5,6,11,12]].all()
            q=normalize_skeleton(xy,confidence,valid_mask=mask).reshape(17,2)
            assert hashlib.sha256(q.tobytes()).hexdigest()==old['shared_xy_sha256']==item['shared_xy_sha256']
            row['shared_xy_sha256']=item['shared_xy_sha256']; row['shared_mask']=mask.tolist()
            costpath=SOURCE/'costs'/(Path(row['image']).stem+'.npz')
            row['costs_sha256']=sha256(costpath)
            with np.load(costpath,allow_pickle=False) as d: base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
            scores=blend_cost(base,fit,.75); original=order(base,scores); short=original[:20]
            row[BASE]=[poses[i].pose_id for i in original[:5]]
            assert row[BASE]==[c['pose_id'] for c in old[BASE]]
            row['shortlist']=[poses[i].pose_id for i in short]; row['shortlist_indices']=short.tolist()
            row['contour_costs']=[0.]*20; row['candidate_errors']={}; row['affines']={}
            try:
                evidence,pad=prepare_letterboxed_query(read_rgb(Path(row['path'])),xy)
                row['padding']=pad; row['query_torso']=evidence['torso']; row['query_line_pixels']=int(evidence['mask'].sum())
                masks['query']=evidence['mask']
                normalized=normalize_skeleton(evidence['xy'].astype(np.float32),confidence,valid_mask=mask).reshape(17,2)
                assert np.allclose(normalized,q,atol=2e-6)
                for pos,index in enumerate(short[1:],1):
                    try:
                        value,affine,pixels=candidate_evidence(evidence,poses[index].joints,params[index])
                        row['contour_costs'][pos]=value; row['affines'][str(pos)]=affine.tolist(); masks[f'candidate_{pos}']=pixels
                    except ValueError as exc: row['candidate_errors'][str(pos)]=str(exc)
            except ValueError as exc: row['query_error']=str(exc)
            row['fallback']=bool(row['candidate_errors'] or row.get('query_error'))
            new=original if row['fallback'] else rerank(original,scores,row['contour_costs'],1.)
            assert np.array_equal(new[20:],original[20:]) and new[0]==original[0]
            row[NEW]=[poses[i].pose_id for i in new[:5]]; row['order20_before']=original[:20].tolist(); row['order20_after']=new[:20].tolist()
            maskpath=args.out/'masks'/(Path(row['image']).stem+'.npz'); np.savez_compressed(maskpath,**masks)
            row['masks_sha256']=sha256(maskpath)
        make_sheet(args.out,row,masks); rows.append(row); print('compared',row['image'],flush=True)
    verify(args.out); save_json(args.out/'rankings.json',rows)
    valid=[r for r in rows if not r.get('error')]
    summary={'queries':len(rows),'searchable':len(valid),'extraction_failed':len(rows)-len(valid),
             'render_or_query_fallback':sum(r['fallback'] for r in valid),
             'top1_changed':sum(r[BASE][0]!=r[NEW][0] for r in valid),
             'top5_set_changed':sum(set(r[BASE])!=set(r[NEW]) for r in valid),
             'top5_order_changed':sum(r[BASE]!=r[NEW] for r in valid),
             'new_candidates_per_image':{r['image']:len(set(r[NEW])-set(r[BASE])) for r in valid},
             'accuracy':None,'human_ratings_received':0,'warning':'candidate change is not accuracy improvement'}
    save_json(args.out/'summary.json',summary); print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--library-root',type=Path,required=True)
    args=p.parse_args()
    with (ROOT/'out/.contour-real.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); run(args)
