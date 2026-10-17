"""Build a local saved-results review: input, fixed XY, and both top-five lists."""
import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose

EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
NAMES=['코','왼눈','오른눈','왼귀','오른귀','왼어깨','오른어깨','왼팔꿈치','오른팔꿈치','왼손목','오른손목','왼골반','오른골반','왼무릎','오른무릎','왼발목','오른발목']


def overlay(record,src):
    kp=record['human']['keypoints2d']; sc=record['human']['scores']
    width,height=record['width'],record['height']
    radius=max(width,height)/100
    marks=[]
    for a,b in EDGES:
        dash=f'stroke-dasharray="{radius} {radius}"' if min(sc[a],sc[b])<.3 else ''
        color='#007d99' if b%2 else '#c55300'
        marks.append(f'<line x1="{kp[a][0]}" y1="{kp[a][1]}" x2="{kp[b][0]}" y2="{kp[b][1]}" stroke="{color}" stroke-width="{radius/2}" {dash}/>')
    for j in range(5,17):
        color='#007d99' if j%2 else '#c55300'
        fill=color if sc[j]>=.3 else 'none'
        marks.append(f'<circle cx="{kp[j][0]}" cy="{kp[j][1]}" r="{radius/1.5}" fill="{fill}" stroke="{color}" stroke-width="{radius/3}"/>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="공통 Human-Art XY 관절"><image href="{src}" width="{width}" height="{height}"/>'+''.join(marks)+'</svg>'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--library-root',type=Path,required=True)
    a=p.parse_args()
    results=json.loads((a.run/'results.json').read_text())
    poses={p.pose_id:p for p in load_library(a.library_root)}
    thumbs=a.run/'thumbs'; thumbs.mkdir(exist_ok=True)
    inputs=a.run/'inputs'; inputs.mkdir(exist_ok=True)
    def thumb(candidate):
        key=candidate['pose_id']+str(candidate['view_degrees'])
        name=hashlib.sha256(key.encode()).hexdigest()[:20]+'.png'
        file=thumbs/name
        if not file.exists():
            im,_=render_pose(poses[candidate['pose_id']].joints,candidate['view_degrees'],size=256)
            im.save(file)
        return 'thumbs/'+name
    css='body{font:15px system-ui;margin:24px auto;max-width:1200px;padding:0 20px;color:#17242b;background:#f6f7f8}h1{font-size:25px}h2{font-size:20px}h3{font-size:16px}article{background:white;padding:20px;margin:24px 0;border-radius:12px}a{color:#146b97}p{line-height:1.6}.inputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));max-width:700px;gap:16px}.inputs img,.inputs svg{width:100%;height:310px;object-fit:contain}.candidates{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.candidate img{width:100%;height:auto}.candidate{overflow-wrap:anywhere;font-size:12px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.bad{color:#a02f26}@media(max-width:600px){.inputs{grid-template-columns:1fr}.candidates{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    for group in ('real','synthetic'):
        records=results[group]['records']
        cards=[]
        for i,r in enumerate(records):
            src=inputs/(group+'_'+r['image'])
            if not src.exists(): shutil.copyfile(r['path'],src)
            rel='inputs/'+src.name
            pieces=[f'<article id="{html.escape(Path(r["image"]).stem)}"><h2>{i+1:02}. {html.escape(r["image"])}</h2>',
                f'<div class="inputs"><div><h3>입력 러프</h3><img loading="lazy" src="{rel}" alt="입력 러프"></div>',
                '<div><h3>두 검색에 공통으로 사용한 XY</h3>'+overlay(r,rel)+'</div></div>']
            if r.get('error'):
                pieces.append('<p class="bad">양쪽 검색 제외: '+html.escape(r['error'])+'</p>')
            else:
                for arm,label in (('xy','A · Human-Art XY만'),('xyz','B · 같은 XY + RTMW3D Z')):
                    extra=f' · 원본 {r[arm+"_rank"]}위' if arm+'_rank' in r else ''
                    pieces.append('<h3>'+label+extra+'</h3><div class="candidates">')
                    for rank,c in enumerate(r[arm],1):
                        image=thumb(c)
                        pieces.append(f'<div class="candidate"><img loading="lazy" src="{image}" alt="{rank}위 후보"><div>{rank}위 · {html.escape(c["pose_id"])}</div><div>카메라 {c["view_degrees"]:g}° · 거리 {c["distance"]:.3f}</div></div>')
                    pieces.append('</div>')
            if 'rtmw' in r:
                z=np.asarray(r['rtmw']['z_meters']); z-=z[[11,12]].mean()
                pieces.append('<details><summary>추정 Z와 모델 간 차이</summary><p>Z는 골반 중심 기준 모델 추정값(m)이며 실측값이 아닙니다. +는 뒤쪽, −는 앞쪽입니다. 관절 점수는 Z 정확도를 보증하지 않습니다.</p><table><tr><th>관절</th><th>Human-Art 점수</th><th>RTMW 점수</th><th>상대 Z</th></tr>')
                for j in range(5,17):
                    pieces.append(f'<tr><td>{NAMES[j]}</td><td>{r["human"]["scores"][j]:.3f}</td><td>{r["rtmw"]["scores"][j]:.3f}</td><td>{z[j]:+.3f}</td></tr>')
                pieces.append('</table><p>두 모델 XY의 평균 차이 / Human-Art 몸통 길이: '+f'{r["agreement"]["mean_xy_disagreement_torso"]:.3f}'+'</p></details>')
            pieces.append('</article>'); cards.append(''.join(pieces))
        name='보유 러프 10장' if group=='real' else '합성 선화 40장'
        other='synthetic-review.html' if group=='real' else 'review.html'
        intro='정답 포즈 ID 없음: 후보 변경은 정확도 개선을 의미하지 않습니다.' if group=='real' else '이전에 사용한 40장 재사용. 원본 ID는 채점에만 사용했습니다.'
        body=f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} · XY와 Z 비교</title><style>{css}</style><h1>{name} · XY 고정 / Z 추가</h1><p>{intro}<br>XY·유효 관절·검색 대상은 동일. Z 가중치 0.25, 몸통 길이 가정 0.5m.<br>후보 그림은 BVH 기반 3D 타원체 미리보기입니다. 머리·얼굴 방향은 근사이며 실제 캐릭터 메시가 아닙니다.</p><p><a href="{other}">다른 표본 보기</a> · <a href="REPORT.md">결과 요약</a> · <a href="results.json">원시 결과 JSON</a></p>'+''.join(cards)+'</html>'
        (a.run/('review.html' if group=='real' else 'synthetic-review.html')).write_text(body)
    # Scientific contact sheet of observed inputs and computed top-1 geometry.
    records=results['real']['records']
    sheet=Image.new('RGB',(1000,1700),'white'); draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
    small=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',14)
    for i,r in enumerate(records):
        x=(i%2)*500; y=(i//2)*340
        draw.text((x+10,y+8),r['image'],font=font,fill='#17242b')
        sources=[Path(r['path'])]
        if not r.get('error'): sources += [a.run/thumb(r[arm][0]) for arm in ('xy','xyz')]
        for j,path in enumerate(sources):
            with Image.open(path) as im:
                rgba=im.convert('RGBA'); bg=Image.new('RGBA',rgba.size,'white'); bg.alpha_composite(rgba); im=bg.convert('RGB')
                im.thumbnail((154,238)); sheet.paste(im,(x+j*164+5+(154-im.width)//2,y+42+(238-im.height)//2))
            draw.text((x+j*164+8,y+285),['입력 러프','XY · 1위','XY+Z · 1위'][j],font=small,fill='#17242b')
        text='XY 추출 기준 미달 → 양쪽 제외' if r.get('error') else ('1위 후보 변경' if r['xy'][0]['pose_id']!=r['xyz'][0]['pose_id'] else '1위 후보 동일')
        draw.text((x+8,y+310),text,font=small,fill='#724c25')
    sheet.save(a.run/'real-top1-contact-sheet.png')
    print('Saved review.html, synthetic-review.html, real-top1-contact-sheet.png',flush=True)


if __name__=='__main__':
    main()
