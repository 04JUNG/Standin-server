#!/usr/bin/env python3
"""Build local arm-only BVH + JSON parts + honest geometry previews. No DB writes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.arm_refine.parts import extract_part, write_part, standalone_bvh
from src.bvh import parse_bvh, fk

# This is a small authored starter set, not a measured semantic classifier.
STARTER = [
 ('relaxed_hip', 'gen_st_neutral__ball_hip__hang__h0', False),
 ('reach_forward', 'gen_kn_seiza__reach_fwd__hang__h0_r2', False),
 ('reach_up', 'gen_st_neutral__reach_up__hip__h0', False),
 ('raise_hand', 'gen_kn_one__raise_hand__hang__h0', False),
 ('offer_hand', 'gen_st_lean_fwd__rm_offer_hand__hang__h1', False),
 ('stop_palm', 'gen_st_lunge__stop_palm__hang__h0', False),
 ('head_touch', 'gen_st_twist__scratch_head__hang__h2', False),
 ('chin_think', 'gen_ch_slouch__chin_think__x__h2', False),
 ('cheeks_pair', 'gen_ch_slouch__cheeks__x__h0', True),
 ('fold_tight_pair', 'gen_st_narrow__fold_arms_tight', True),
 ('fold_seated_pair', 'ext_seated_folded_arms', True),
 ('guard_pair', 'gen_sq_half__block_arms__x__h5', True),
]


def preview(part, output):
    from PIL import Image, ImageDraw
    # Same torso coordinate view and scale for every asset. Actual chain positions,
    # never a synthetic image masquerading as the mesh/skin validation.
    im = Image.new('RGB',(540,300),'#f5f4f0')
    d = ImageDraw.Draw(im)
    d.text((12,10),part['label']+' / '+part['selection'],fill='#172e42')
    for col,(name,axes) in enumerate((('front',(0,1)),('side',(2,1)))):
        d.text((col*270+12,35),name+' - skeleton',fill='#687683')
        def xy(v): return (col*270+135-v[axes[0]]*75,180-v[axes[1]]*75)
        d.line([xy((0,-1,0)),xy((0,0,0))],fill='#d1d6d8',width=5)
        for side,arm in part['arms'].items():
            pts=[arm[k] for k in ('anchor_position_torso','shoulder_position_torso','elbow_position_torso','wrist_position_torso')]
            color='#e5774b' if side=='left' else '#338b9b'
            d.line([xy(p) for p in pts],fill=color,width=6)
            for p in pts:
                x,y=xy(p);d.ellipse((x-4,y-4,x+4,y+4),fill='white',outline=color,width=2)
        d.text((col*270+12,275),'clavicle / shoulder / elbow / wrist',fill='#687683')
    im.save(output)


def build(library, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'parts').mkdir();(output/'previews').mkdir()
    records=[]; failures=[];roundtrips=[]
    for label,pose,paired in STARTER:
        source=library/(pose+'.bvh')
        if not source.exists():
            failures.append({'pose_id':pose,'reason':'source_missing'});continue
        for selection in ('left','right','both'):
            # Both single and paired variants exist; paired requires atomic apply.
            relation='paired' if paired and selection=='both' else 'independent'
            part=extract_part(source,selection,label=label,relation=relation)
            target=output/'parts'/(label+'--'+selection)
            write_part(part,target.with_suffix('.json'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning) # gimbal equivalent, matrices verified below
                target.with_suffix('.bvh').write_text(standalone_bvh(part))
            preview(part,output/'previews'/(target.name+'.png'))
            joints,frames=parse_bvh(str(target.with_suffix('.bvh')))
            pos=fk(joints,frames[0]); byname={j[0]:pos[i] for i,j in enumerate(joints)}
            max_error=0.
            for side,arm in part['arms'].items():
                for suffix,field in (('Shoulder','anchor_position_torso'),('Arm','shoulder_position_torso'),('ForeArm','elbow_position_torso'),('Hand','wrist_position_torso')):
                    max_error=max(max_error,float(np.linalg.norm(byname[side.title()+suffix]-arm[field])))
            if max_error>1e-7:raise RuntimeError(f'part BVH position roundtrip failed: {part["id"]}: {max_error}')
            roundtrips.append({'part_id':part['id'],'max_position_error_torso':max_error})
            records.append({'id':part['id'],'label':label,'selection':selection,'relation':relation,
                'json':str(target.relative_to(output))+'.json','bvh':str(target.relative_to(output))+'.bvh',
                'preview':'previews/'+target.name+'.png','source':part['source'],
                'sha256':hashlib.sha256(target.with_suffix('.json').read_bytes()).hexdigest()})
    result={'schema':'standin.arm-parts-catalog.v1','run_id':output.name,
        'parts_count':len(records),'source_library':str(library.resolve()),
        'source_type':'existing_library_curated_starter_not_automatic_detection',
        'items':records,'failures':failures,'release_allowed':False}
    (output/'catalog.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'validation.json').write_text(json.dumps({'roundtrips':roundtrips,'passed':not failures,'source_files_unchanged':all(hashlib.sha256((library/(r['source']['pose_id']+'.bvh')).read_bytes()).hexdigest()==r['source']['sha256'] for r in records)},indent=2)+'\n')
    from html import escape
    cards=[]
    for item in records:
        cards.append(f'<article><img src="{item["preview"]}"><p>{escape(item["label"])} · {item["selection"]} · {item["relation"]}</p><a href="{item["json"]}">JSON part</a> / <a href="{item["bvh"]}">arm-only BVH</a></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Standin arm pose parts</title><style>body{font:15px system-ui;margin:30px;background:#faf9f6;color:#172e42}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:white;border:1px solid #ddd;padding:12px}img{width:100%}a{color:#167b8b}</style><h1>팔 포즈 파츠 · starter v1</h1><p>자기 기준 왼팔 / 오른팔 / 양팔. 기존 BVH에서 추출한 실제 관절 파츠. 미리보기는 정면·측면 스켈레톤이며 메시 검증이 아닙니다.</p><p>쇄골 앵커·어깨·팔꿈치·손목을 포함합니다. 손가락·손 메시 생성, 자동 부위 탐지는 포함하지 않습니다. 양팔 관계가 중요한 paired는 양팔 원자 적용 대상입니다. 출처 이름은 수동 분류이며 정확도 측정값이 아닙니다.</p><main>'+''.join(cards)+'</main>')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--native-thumbnails',type=Path,help='verified native renderer output; require matching source hashes')
    args=parser.parse_args();r=build(args.library,args.output)
    if args.native_thumbnails:
        from publish_arm_thumbnails import publish
        publish(args.output,args.native_thumbnails)
    print(json.dumps({'parts_count':r['parts_count'],'failures':r['failures']}))
