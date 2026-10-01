#!/usr/bin/env python3
"""Show the actual returned Top-K mesh previews next to each rough person."""
from pathlib import Path
import argparse,hashlib,json,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_review(run,out,*,build_manifest):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (run/'results.jsonl').read_text().splitlines() if x.strip()]
    plans=json.loads((run/'plans.json').read_text());by_cut={x['cut_id']:x for x in plans}
    manifest_path=Path('data/thumbs/thumbnail_manifest.json');manifest=json.loads(manifest_path.read_text())
    # Service previews were rendered from the same geometry DB snapshot.
    build=json.loads(Path(build_manifest).read_text())
    expected_db=build['identity']['geometry_db_sha256'];sources=[]
    def verify_chain(path,expected=None):
        if expected and sha(path)!=expected:raise ValueError('thumbnail provenance hash mismatch')
        doc=json.loads(Path(path).read_text())
        children=list(doc.get('source_manifests',[]))
        if doc.get('source_manifest'):
            children.append(dict(path=doc['source_manifest'],sha256=doc['source_manifest_sha256']))
        if children:
            for child in children:verify_chain(Path(child['path']),child['sha256'])
        else:
            if doc.get('db_sha256')!=expected_db:raise ValueError('thumbnail/search DB snapshot mismatch')
            sources.append(str(path))
    verify_chain(manifest_path)
    notes_path=out/'visual-review-notes.json'
    notes=json.loads(notes_path.read_text()) if notes_path.exists() else {}
    inventory={(r['pose_id'],r['view']):r for r in manifest['results']}
    ids={c['pose_id'] for r in rows for q in r['execution']['semantic_requests'] for c in q['results']}
    previews={};copied=[];assetdir=out/'model_previews';assetdir.mkdir(exist_ok=True)
    for pid in sorted(ids):
        previews[pid]={}
        for view in ('front','three_quarter'):
            record=inventory[(pid,view)];src=manifest_path.parent/record['filename']
            if record['status']!='ok' or sha(src)!=record['sha256']:raise ValueError('invalid thumbnail: '+pid)
            name=hashlib.sha256((pid+'|'+view).encode()).hexdigest()[:24]+'.jpg'
            shutil.copyfile(src,assetdir/name);previews[pid][view]='model_previews/'+name
            copied.append(dict(pose_id=pid,view=view,source=str(src.resolve()),sha256=record['sha256'],output=name))
    images=out/'images';images.mkdir(exist_ok=True);cases=[]
    for r in rows:
        cut=by_cut[r['cut_id']];person=next(p for p in cut['people'] if p['person_index']==r['person_index'])
        src=Path(cut['image']);dest=images/(r['cut_id']+src.suffix);shutil.copyfile(src,dest)
        with Image.open(src) as im:w,h=im.size
        box=person['box'];valid=box and 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
        cases.append(dict(cut_id=r['cut_id'],person_index=r['person_index'],route=r['plan']['route'],
            image='images/'+dest.name,width=w,height=h,box=box if valid else None,
            queries=r['execution']['semantic_requests'],reason=r['observation_mapping'],note=notes.get(r['cut_id'],'')))
    payload=dict(cases=cases,previews=previews)
    (out/'visual-results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    audit=dict(pose_count=len(ids),preview_count=len(copied),results_sha256=sha(run/'results.jsonl'),
        thumbnail_manifest_sha256=sha(manifest_path),geometry_db_sha256=expected_db,verified_source_manifests=sources,
        source_character=manifest['source_character'],preview_kind='existing_library_mesh',
        camera_matched_to_rough=False,ranking_changed=False,files=copied)
    (out/'preview-provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    html=HTML.replace('__DATA__',data)
    (out/'index.html').write_text(html)
    return dict(pose_count=len(ids),preview_count=len(copied),people=len(cases),html=str(out/'index.html'))


HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>검색된 실제 포즈 · 러프 비교</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f3f5;color:#202b37;font:15px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:26px}header,article{background:white;border:1px solid #dde1e5;border-radius:14px;padding:24px;margin-bottom:24px}h1{font-size:30px;margin:0 0 12px}h2{font-size:21px;margin:0}h3{font-size:17px;margin:10px 0}.hint{color:#596571;font-size:13px}.warning{background:#fff3d6;padding:12px;border-radius:8px}.layout{display:grid;grid-template-columns:minmax(200px,25%) 1fr;gap:22px}.rough{position:relative;align-self:start}.rough img{width:100%;display:block;max-height:580px;object-fit:contain}.rough svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.card{border:1px solid #dbe0e5;border-radius:10px;overflow:hidden;min-width:0}.card img{width:100%;display:block;cursor:zoom-in}.caption{padding:8px;font-size:12px;overflow-wrap:anywhere}.rank{font-size:15px;font-weight:700}.viewbar{display:flex;gap:8px;margin:12px 0}button,select{font:inherit;border:1px solid #cbd2db;border-radius:7px;background:white;padding:7px 10px}button{cursor:pointer}button.active{background:#203b57;color:white}select{width:100%;margin:12px 0}nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}a{color:#196aa8}dialog{max-width:90vw;border:0;border-radius:14px;padding:18px}dialog::backdrop{background:#0008}dialog img{width:512px;max-width:80vw;image-rendering:auto}details{margin-top:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}@media(max-width:1000px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:700px){main{padding:12px}article,header{padding:16px}.layout{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.rough img{max-height:400px}}</style>
<main><header><h1>어떤 포즈를 찾았는지 확인하기</h1><p><b>러프 → 인물별 의미 질의 → 실제 검색 Top-5 모델</b></p><p class="warning">표시된 것은 질의별 검색 결과입니다. 통합 최종 순위·러프에 맞춘 카메라·상하체 조합·refine 결과는 아닙니다. 모델은 라이브러리 정면/3·4 시점 미리보기입니다.</p><p>질의를 바꾸면 해당 질의의 실제 Top-5를 순서 그대로 표시합니다. 기본은 전신 질의이며, 이미지 클릭으로 확대할 수 있습니다.</p><nav id="nav"></nav></header><div id="results"></div></main>
<dialog id="zoom"><button id="close">닫기</button><p id="zoom-title"></p><img id="zoom-image" alt="확대한 검색 포즈"></dialog>
<script>
const data=__DATA__; const host=document.querySelector('#results');const nav=document.querySelector('#nav');
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
const region={full:'전신',upper:'상체',lower:'하체'};
for(const [idx,c] of data.cases.entries()){
 const article=el('article');article.id='case-'+idx;host.append(article);
 const anchor=el('a',c.cut_id+' P'+c.person_index);anchor.href='#'+article.id;nav.append(anchor);
 article.append(el('h2',c.cut_id+' · P'+c.person_index));const layout=el('div',undefined,'layout');article.append(layout);
 const rough=el('div',undefined,'rough');layout.append(rough);const input=el('img');input.src=c.image;input.alt='러프 '+c.cut_id;input.loading='lazy';rough.append(input);
 if(c.box){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${c.width} ${c.height}`);const rect=document.createElementNS(svg.namespaceURI,'rect');for(const [k,v] of Object.entries({x:c.box[0],y:c.box[1],width:c.box[2]-c.box[0],height:c.box[3]-c.box[1],fill:'none',stroke:'#df3153','stroke-width':3}))rect.setAttribute(k,v);svg.append(rect);rough.append(svg)}
 const body=el('div');layout.append(body);
 if(!c.queries.length){body.append(el('p','검색 보류: 인물 박스 좌표를 신뢰할 수 없어 후보를 반환하지 않았습니다.','warning'));continue}
 if(c.note)body.append(el('p',c.note,'warning'));
 const select=el('select');select.setAttribute('aria-label',c.cut_id+' P'+c.person_index+' 의미 질의');
 c.queries.forEach((q,i)=>{const option=el('option',(region[q.region]||q.region)+' · '+q.query);option.value=i;select.append(option)});
 const full=c.queries.map((q,i)=>q.region==='full'?i:-1).filter(i=>i>=0);select.value=full.length?full[full.length-1]:0;body.append(select);
 const bar=el('div',undefined,'viewbar');body.append(bar);let view='front';const buttons={};
 for(const [v,label] of [['front','정면'],['three_quarter','3/4 시점']]){const b=el('button',label);bar.append(b);buttons[v]=b;b.onclick=()=>{view=v;draw()}}
 const cards=el('div',undefined,'cards');body.append(cards);body.append(el('p','원래 검색 순위를 유지했습니다. 유사도 점수는 정답 확률이 아닙니다.','hint'));
 const details=el('details');details.append(el('summary','질의와 후보 출처 보기'));const pre=el('pre');details.append(pre);body.append(details);
 function draw(){cards.replaceChildren();const q=c.queries[Number(select.value)];for(const [v,b]of Object.entries(buttons))b.classList.toggle('active',v===view);
 q.results.forEach((p,i)=>{const card=el('div',undefined,'card');cards.append(card);const image=el('img');image.src=data.previews[p.pose_id][view];image.alt=`${i+1}위 ${p.pose_id} ${view}`;image.loading='lazy';card.append(image);const caption=el('div',undefined,'caption');caption.append(el('div',`${i+1}위 · 유사도 ${p.score.toFixed(3)}`,'rank'),el('div',p.pose_id));card.append(caption);image.onclick=()=>{document.querySelector('#zoom-image').src=image.src;document.querySelector('#zoom-title').textContent=p.pose_id+' · '+view;document.querySelector('#zoom').showModal()}});pre.textContent=JSON.stringify(q,null,2)}
 select.onchange=draw;draw();
}
document.querySelector('#close').onclick=()=>document.querySelector('#zoom').close();
</script></html>'''

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--build-manifest',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(build_review(a.run,a.out,build_manifest=a.build_manifest),ensure_ascii=False))
