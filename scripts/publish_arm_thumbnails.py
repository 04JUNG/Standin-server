#!/usr/bin/env python3
"""Attach verified native arm/library thumbnails to a local part gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from html import escape

STYLE='approved-anatomy-detail-dark-5scope-8view-v1'
VIEWS=('front','three_quarter','side','back')
NAMES={'relaxed_hip':'기본·허리 근처','reach_forward':'앞으로 뻗기','reach_up':'위로 뻗기',
 'raise_hand':'손 들기','offer_hand':'손 내밀기','stop_palm':'멈춤 제스처','head_touch':'머리 근처',
 'chin_think':'턱 근처·생각','cheeks_pair':'양손 뺨 근처','fold_tight_pair':'손을 편 팔 포개기',
 'fold_seated_pair':'아래팔 교차','guard_pair':'방어 자세','fold_contact_pair':'위팔을 감싸는 팔짱 후보',
 'hip_both':'양손 허리 짚기','hip_side':'한 손 허리 짚기','hip_offer_pair':'허리 짚고 손 내밀기',
 'chin_supported_pair':'반대팔로 받치는 턱 근처 손','chin_low':'낮은 팔꿈치·턱 근처',
 'head_back':'뒤통수 짚기','head_back_high':'높은 팔꿈치·뒷목 손',
 'reach_full_forward':'팔 길게 앞으로 뻗기','reach_full_up':'팔 높이 끝까지 뻗기',
 'asymmetric_reach_pair':'앞뒤 비대칭 팔','phone_hold':'한 손 물건 들기','phone_pair':'양손 물건 들기',
 'hands_behind_pair':'등 뒤 낮은 팔'}
SIDES={'left':'왼팔','right':'오른팔','both':'양팔'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(item,record,package,native):
    if record['part_id']!=item['id'] or record['source_bvh_sha256']!=item['source']['sha256']:
        raise ValueError('thumbnail/part source mismatch')
    if record['selection']!=item['selection'] or record['style']!=STYLE:
        raise ValueError('thumbnail side/style mismatch')
    if sha(package/item['json'])!=item['sha256']:raise ValueError('part payload changed')
    for group in ('images','library_references'):
        if set(record[group])!=set(VIEWS):raise ValueError('missing thumbnail views')
        for image in record[group].values():
            p=(native/image['path']).resolve()
            if not p.is_relative_to(native.resolve()):raise ValueError('invalid thumbnail path')
            if sha(p)!=image['sha256']:raise ValueError('thumbnail hash mismatch')
            if group=='images' and (image['width']!=512 or image['height']!=512 or image['border_pixels']<12):
                raise ValueError('thumbnail size/framing mismatch')


def gallery(items):
    payload=[];cards=[]
    for i,item in enumerate(items):
        title=NAMES.get(item['label'],item['label']);side=SIDES[item['selection']]
        payload.append({'views':item['native_preview']['images'],'library':item['native_preview']['library_references']})
        paired='<span class="tag">양팔 함께 적용</span>' if item['relation']=='paired' else ''
        semantics=item.get('semantics',{})
        quality=semantics.get('pair_quality')
        state=('접촉 검사 보류' if quality and not quality['eligible_strict_fold'] else '시각 검토 필요')
        description=escape(semantics.get('description_ko',''))
        semantic_info=f'<p class="semantic-note">{description}</p><span class="tag">{state}</span>' if semantics else ''
        cards.append(f'''<article data-side="{item['selection']}" data-part="{escape(item['id'])}">
<img class="native" data-i="{i}" width="512" height="512" loading="lazy" src="{item['preview']}" alt="{escape(title)} {side} 실제 메시 썸네일">
<div class="cardbody"><div class="cardtitle"><h2>{escape(title)}</h2><span class="side">{side}</span></div>{paired}{semantic_info}
<div class="links"><a href="{item['json']}">JSON</a><a href="{item['bvh']}">BVH</a></div>
<details><summary>관절 구조</summary><img width="540" height="300" loading="lazy" src="{item['skeleton_preview']}" alt="{escape(title)} {side} 관절 구조"></details></div></article>''')
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    result= '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>팔 파츠 · Standin</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#203044;font:15px/1.6 system-ui,-apple-system,sans-serif}header,main,footer{max-width:1440px;margin:auto;padding:28px 32px}header{padding-bottom:20px}h1{margin:4px 0;font-size:30px;letter-spacing:-1px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:1.5px;color:#797e82}header p{margin:8px 0;color:#64707d}.controls{display:flex;gap:16px;flex-wrap:wrap;margin-top:22px;align-items:center}label{font-size:13px;color:#566271}select{font:inherit;background:white;border:1px solid #d4d8dd;border-radius:8px;padding:8px 30px 8px 12px;margin-left:6px;color:#203044}#count{margin-left:auto;font-size:13px;color:#6d7985}main{padding-top:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}article{border:1px solid #e0e3e5;background:white;border-radius:12px;overflow:hidden}article[hidden]{display:none}.native{width:100%;height:auto;aspect-ratio:1;background:#383f4c;display:block}.cardbody{padding:17px 20px}.cardtitle{display:flex;align-items:center;justify-content:space-between;gap:10px}h2{font-size:17px;margin:0;font-weight:650;letter-spacing:-.4px}.side{font-size:12px;background:#eef3f4;padding:3px 9px;border-radius:20px;white-space:nowrap}.tag{font-size:11px;color:#b66743}.links{display:flex;gap:16px;margin-top:10px}a{color:#286d7b;text-decoration:none;font-size:13px}a:hover{text-decoration:underline}details{border-top:1px solid #edf0f2;margin-top:13px;padding-top:9px}summary{cursor:pointer;color:#7b858e;font-size:12px}details img{width:100%;height:auto}footer{font-size:12px;color:#77818a;padding-top:0}select:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #e48257;outline-offset:3px}@media(max-width:950px){main{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){header,main,footer{padding-left:18px;padding-right:18px}main{grid-template-columns:1fr}.controls{gap:9px}#count{margin-left:0}h1{font-size:25px}}</style>
<header><div class="eyebrow">STANDIN · ARM PARTS</div><h1>팔 파츠 라이브러리</h1><p>12개 자세 · 왼팔 12 / 오른팔 12 / 양팔 12</p><p>라이브러리와 같은 인체 메시·재질·조명으로 확인하세요.</p>
<div class="controls"><label>부위<select id="side"><option value="all">전체</option><option value="left">왼팔</option><option value="right">오른팔</option><option value="both">양팔</option></select></label><label>각도<select id="view"><option value="front">정면</option><option value="three_quarter">사선</option><option value="side">측면</option><option value="back">후면</option></select></label><label>미리보기<select id="kind"><option value="views">선택 팔만 보기</option><option value="library">인형에서 보기</option></select></label><span id="count">36개 파츠</span></div></header><main>'''+''.join(cards)+'''</main><footer>자기 기준 좌우입니다. 손 모양은 원본 포즈를 참고하기 위한 이미지입니다. 실제 교체에서는 기존 손 모양을 유지합니다.</footer>
<script>const items='''+data+''';const side=document.getElementById('side'),view=document.getElementById('view'),kind=document.getElementById('kind');function update(){let n=0;document.querySelectorAll('article').forEach(card=>{card.hidden=side.value!=='all'&&card.dataset.side!==side.value;if(!card.hidden)n++;});document.getElementById('count').textContent=n+'개 파츠';document.querySelectorAll('img.native').forEach(img=>{img.src=items[Number(img.dataset.i)][kind.value][view.value];});}for(const element of [side,view,kind])element.addEventListener('change',update);</script></html>'''
    counts={s:sum(i['selection']==s for i in items) for s in SIDES}
    return result.replace('12개 자세 · 왼팔 12 / 오른팔 12 / 양팔 12',f"{len({i['source']['pose_id'] for i in items})}개 원본 자세 · 왼팔 {counts['left']} / 오른팔 {counts['right']} / 양팔 {counts['both']}").replace('36개 파츠',str(len(items))+'개 파츠')


def publish(package,native):
    package=Path(package);native=Path(native)
    manifest=json.loads((native/'manifest.json').read_text());catalog=json.loads((package/'catalog.json').read_text())
    if manifest['status']!='complete':raise ValueError('native render incomplete')
    records={r['part_id']:r for r in manifest['items']}
    if len(records)!=len(catalog['items']):raise ValueError('part count mismatch')
    # Validate every asset before replacing the gallery.
    for item in catalog['items']:verify(item,records[item['id']],package,native)
    for item in catalog['items']:
        record=records[item['id']];groups={}
        for group in ('images','library_references'):
            groups[group]={}
            for view,asset in record[group].items():
                src=native/asset['path'];target=package/'native-thumbnails'/asset['path']
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
                groups[group][view]=str(target.relative_to(package))
        item['skeleton_preview']=item.get('skeleton_preview',item['preview'])
        item['preview']=groups['images']['front']
        item['native_preview']={'style':STYLE,**groups,'provenance':'native-thumbnail-manifest.json'}
    shutil.copy2(native/'manifest.json',package/'native-thumbnail-manifest.json')
    index=package/'index.html';backup=package/'index-skeleton-v1.html'
    if index.exists() and not backup.exists():shutil.copy2(index,backup)
    index.write_text(gallery(catalog['items']))
    catalog['thumbnail_style']=STYLE;catalog['thumbnail_views']=list(VIEWS)
    (package/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    return {'parts':len(catalog['items']),'images':len(catalog['items'])*8,'style':STYLE}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path);p.add_argument('native',type=Path)
    args=p.parse_args();print(json.dumps(publish(args.package,args.native)))
