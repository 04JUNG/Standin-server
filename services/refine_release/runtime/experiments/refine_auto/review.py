"""Automatic final review: execution, quality decisions, stage fallbacks and images."""
import argparse,json,html,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_new'))
from core import prepare,skin,render_projected
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'review';out.mkdir(exist_ok=True);rows=[]
for row in json.loads((r/'final/results.json').read_text()):
 c=row['case_id'];fx=json.loads((r/'final'/c/'fixture.json').read_text());names,ix,_,_,base=prepare(json.loads((a.assets/fx['files']['rig']['path']).read_text()));m=np.load(a.assets/fx['files']['mesh']['path']);ims=[ImageOps.pad(Image.open(a.assets/fx['files']['rough']['path']).convert('RGB'),(535,687),color='white')];labels=['ROUGH','INPUT','UPPER PREFLIGHT','FINAL'];worlds=[]
 folders=[a.baseline/c/'fit',r/'cases'/c/'upper'/c/'fit',r/'final'/c/'fit'];folders=[f if f.exists() else folders[0] for f in folders]
 for f in folders:
  pr=json.loads((f/'adjustment.json').read_text());w=np.load(f/'expected-world.npz')['world'];worlds.append(w);cv=(skin(m['vertices'],m['weights'],w,base)-pr['origin'])@np.array(pr['camera']);xy=cv[:,:2]*[pr['pixel_scale'],-pr['pixel_scale']]+pr['offset'];ims.append(render_projected(cv,m['faces'],xy,(535,687)))
 sheet=Image.new('RGB',(2140,730),'white');d=ImageDraw.Draw(sheet)
 for i,(im,label) in enumerate(zip(ims,labels)):sheet.paste(im,(535*i,35));d.text((535*i+8,8),c+' '+label,fill='black')
 sheet.save(out/f'{c}.png');meta=json.loads((r/'final'/c/'fit/metrics.json').read_text());trace=json.loads((r/'cases'/c/'trace.json').read_text());rows.append(dict(case=c,trace=trace,last_stage_geometry_accepted=meta['geometry_accepted'],pose_changed_from_input=bool(np.max(abs(worlds[0]-worlds[-1]))>1e-7),anchor_policy=meta.get('anchor_policy'),reasons=meta['rejection_reasons']))
(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));cards=''.join('<h2>'+x['case']+'</h2><img src="'+x['case']+'.png"><pre>'+html.escape(json.dumps(x['trace']['stages'],ensure_ascii=False,indent=2))+'</pre>' for x in rows)
(out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Automatic refine pipeline</title><style>body{font:16px/1.6 system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:20px}</style><h1>자동 상체 검사 → 다리 보정 → 검사·복구</h1><p>입력은 선택된004/006/010 체크포인트와 러프·스켈레톤이다. 신규 포즈 검색이나 최초 카메라 추론을 이번에 다시 실행한 것은 아니다. 조건은 해시 결합된 캐시 또는 gpt-6-luna/low로 자동 확보한다. 실행 완료와 러프 품질 통과는 별개다. 미해결 상체는 검증된 기준으로 간주하지 않고 관측이 있으면 제한된 공동 보정을 시도한다. 실패한 다리 단계가 통과한 어깨 보정을 지우지 않는다.</p>'+cards+'<details><summary>전체 진단</summary><pre>'+html.escape(json.dumps(rows,ensure_ascii=False,indent=2))+'</pre></details>')
