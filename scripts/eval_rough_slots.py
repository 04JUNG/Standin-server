#!/usr/bin/env python3
"""Real image-only VLM slot pilot, immutable per-cut attempts and conservative replay."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG
from src.vlm import prompts
from src.vlm.rough_slots import parse_slots, user_prompt, SCHEMA_VERSION, valid_normalized_box
from src.vlm.client import _coerce, _extract_json, GeminiVLMClient
from src.experimental.rough_router import Observation, plan_search, POLICY_VERSION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    return x*y/max(area(a)+area(b)-x*y,1e-9)


def match_observation(box, fresh_boxes, old_cases):
    """Mutual best overlap + margins; rejected mappings never acquire old joints."""
    if box is None or not old_cases:
        return None,'no_cached_observation'
    overlaps=[iou(box,c['box']) for c in old_cases]
    order=sorted(range(len(overlaps)),key=lambda j:-overlaps[j])
    best=order[0]
    if overlaps[best]<.30 or (len(order)>1 and overlaps[best]-overlaps[order[1]]<.10):
        return None,'ambiguous_or_low_overlap'
    other=sorted([iou(b,old_cases[best]['box']) for b in fresh_boxes if b is not None],reverse=True)
    if len(other)>1 and other[0]-other[1]<.10:
        return None,'ambiguous_reverse_overlap'
    if overlaps[best] < other[0]-1e-9:
        return None,'not_mutual_best'
    if old_cases[best]['owner_verified'] is not True:
        return None,'cached_owner_unverified'
    return old_cases[best],'mutual_iou_with_verified_cached_owner'


def replay(raw, width, height, old_cases):
    analysis=_coerce(raw,width,height)
    parsed=parse_slots(raw,analysis.num_people)
    boxes=[b.as_list() if b is not None else None for b in analysis.approx_boxes]
    rows=[]
    for p in parsed.people:
        box=boxes[p.person_index] if p.person_index<len(boxes) else None
        raw_boxes=raw.get('approx_boxes',[])
        valid_box=bool(box and p.person_index<len(raw_boxes) and valid_normalized_box(raw_boxes[p.person_index]))
        cached,reason=match_observation(box,boxes,old_cases) if valid_box else (None,'invalid_vlm_box')
        if cached:
            obs=Observation(cached['keypoints'],cached['mask'],True,'valid')
        else:
            obs=Observation(ownership_valid=valid_box)
        plan=plan_search(p,obs)
        rows.append({'person_index':p.person_index,'box':box,'slots':p.to_dict(),
                     'observation_mapping':reason,'cached_case_id':cached['id'] if cached else None,
                     'plan':plan.to_dict(),'plan_id':plan.plan_id})
    return {'parsed_people':len(parsed.people),'slot_errors':list(parsed.issues),'people':rows}


def render_report(out, inputs):
    records=[]
    for case in inputs:
        attempts=sorted((out/'responses').glob(case['cut_id']+'.attempt-*.json'))
        if attempts:records.append(json.loads(attempts[-1].read_text()))
    summary={'cuts_attempted':len(records),'success':sum(r['status']=='ok' for r in records),
        'errors':sum(r['status']!='ok' for r in records),'people_parsed':sum(r.get('parsed_people',0) for r in records),
        'routes':dict(Counter(p['plan']['route'] for r in records for p in r.get('people',[]))),
        'slot_errors':dict(Counter(e for r in records for e in r.get('slot_errors',[]))),
        'field_issues':dict(Counter(e for r in records for p in r.get('people',[]) for e in p['slots']['issues'])),
        'model_versions':sorted({r.get('model_version','unknown') for r in records if r['status']=='ok'}),
        'actual_semantic_retrievals':0,'actual_compositions':0,'human_accuracy_evaluated':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    parts=['<!doctype html><meta charset="utf-8"><title>러프 의미 슬롯 검토</title><style>body{font:15px system-ui;margin:28px;background:#f3f4f6;color:#17202a}article{background:white;padding:20px;margin:20px 0;border-radius:12px;display:grid;grid-template-columns:minmax(200px,38%) 1fr;gap:24px}img{width:100%;max-height:700px;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd}h3{margin-top:26px}.warning{color:#944400}pre{white-space:pre-wrap}</style><h1>러프 → 의미 슬롯 → 경로 검토</h1><p>실제 VLM 출력. 포즈 검색·조합 결과 또는 작가 정확도 판정이 아닙니다.</p>']
    for r in records:
        parts.append('<article><div><h2>'+html.escape(r['cut_id'])+'</h2><img src="images/'+html.escape(r['cut_id'])+'.png"></div><div>')
        if r['status']!='ok':
            parts.append('<p class="warning">호출 실패: '+html.escape(r['error_type'])+'</p>')
        for p in r.get('people',[]):
            parts.append('<h3>인물 '+str(p['person_index'])+' · '+html.escape(p['plan']['route'])+'</h3><p>관절 연결: '+html.escape(p['observation_mapping'])+'</p><table><tr><th>슬롯</th><th>해석 / 근거</th></tr>')
            for name,values in p['slots']['meanings']:
                text=' / '.join(v['value']+' ['+v['status']+', '+v['evidence_kind']+'] '+v['evidence_note'] for v in values) or 'unknown'
                parts.append('<tr><td>'+html.escape(name)+'</td><td>'+html.escape(text)+'</td></tr>')
            parts.append('</table><p>가시성: '+html.escape(str(dict(p['slots']['visibility'])))+'</p>')
            parts.append('<p>계획: '+html.escape(', '.join(c['kind']+'_'+c['region'] for c in p['plan']['channels']))+'</p>')
            parts.append('<p class="warning">'+html.escape('; '.join(p['plan']['reasons']))+'</p>')
        parts.append('</div></article>')
    (out/'index.html').write_text(''.join(parts))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=17);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args()
    if args.limit<1:raise ValueError('limit must be positive')
    protocol=json.loads(args.protocol.read_text());by_cut={}
    for case in protocol['cases']:
        if sha(case['image'])!=case['image_sha256'] or sha(case['response'])!=case['response_sha256']:
            raise ValueError('protocol input changed')
        by_cut.setdefault(case['cut_id'],[]).append(case)
    inputs=[dict(cut_id=k,image=v[0]['image'],image_sha256=v[0]['image_sha256']) for k,v in sorted(by_cut.items())]
    prompt=user_prompt(prompts.USER_TEMPLATE,True)
    identity={'schema_version':SCHEMA_VERSION,'policy_version':POLICY_VERSION,
        'protocol_sha256':sha(args.protocol),'prompt_sha256':hashlib.sha256((prompts.SYSTEM+prompt).encode()).hexdigest(),
        'requested_model':CFG.gemini_model,'inputs':inputs,'provider':'gemini','automatic_retries':0}
    if args.out.exists():
        if not (args.resume or args.report_only):raise ValueError('use --resume for existing run')
        if json.loads((args.out/'manifest.json').read_text())!=identity:raise ValueError('run identity changed')
    else:
        args.out.mkdir(parents=True);(args.out/'responses').mkdir();(args.out/'images').mkdir()
        (args.out/'manifest.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
        for c in inputs:(args.out/'images'/(c['cut_id']+'.png')).write_bytes(Path(c['image']).read_bytes())
    if args.report_only:
        print(json.dumps(render_report(args.out,inputs),ensure_ascii=False));return
    # No factory fallback: failure to initialize or call the actual provider is an error.
    from google import genai
    from google.genai import types
    from PIL import Image
    client=genai.Client(api_key=os.environ['GEMINI_API_KEY'],http_options=types.HttpOptions(
        timeout=60000,retry_options=types.HttpRetryOptions(attempts=1)))
    attempted=0
    for case in inputs:
        attempts=sorted((args.out/'responses').glob(case['cut_id']+'.attempt-*.json'))
        if attempts:
            latest=json.loads(attempts[-1].read_text())
            if latest['status']=='ok' or not args.retry_errors:continue
        if attempted>=args.limit:break
        attempted+=1;started=time.perf_counter()
        row={'cut_id':case['cut_id'],'image_sha256':case['image_sha256'],
             'created_at':datetime.now(timezone.utc).isoformat(),'mock':False}
        try:
            with Image.open(case['image']) as im:
                im=im.convert('RGB');width,height=im.size
                response=client.models.generate_content(model=CFG.gemini_model,
                    contents=[prompt,GeminiVLMClient._to_part(im)],config=types.GenerateContentConfig(
                        system_instruction=prompts.SYSTEM,response_mime_type='application/json',temperature=0))
            raw=_extract_json(response.text)
            row.update(status='ok',raw=raw,model_version=getattr(response,'model_version',None) or 'unknown',
                       usage=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None,
                       **replay(raw,width,height,by_cut[case['cut_id']]))
        except Exception as exc:
            message=str(getattr(exc,'message',''))
            for name,value in os.environ.items():
                if value and ('KEY' in name or 'TOKEN' in name or 'SECRET' in name):
                    message=message.replace(value,'[redacted]')
            row.update(status='error',error_type=type(exc).__name__,
                error_code=getattr(exc,'code',None),error_status=getattr(exc,'status',None),error_message=message[:800])
        row['elapsed_seconds']=round(time.perf_counter()-started,3)
        path=args.out/'responses'/(case['cut_id']+f'.attempt-{len(attempts)+1:02}.json')
        with path.open('x') as f:json.dump(row,f,ensure_ascii=False,indent=2)
        print(case['cut_id'],row['status'],row.get('error_type',''),row.get('error_code',''),row.get('error_status',''),row.get('parsed_people',0),flush=True)
        render_report(args.out,inputs)
    client.close()
    print(json.dumps(render_report(args.out,inputs),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
