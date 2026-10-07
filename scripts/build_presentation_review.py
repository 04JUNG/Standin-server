#!/usr/bin/env python3
"""Presentation correction report. Author labels are evaluation-only, never model input."""
import argparse,copy,html,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experimental.body_matching.selection import compare_body_shapes
H=html.escape


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    raw=json.loads((out/'results.json').read_text());old=json.loads(args.baseline.read_text());catalog=json.loads((out/'shape-catalog.json').read_text());labels=json.loads((out/'author-labels.json').read_text())['labels']
    assets={a['body_id']:a for a in catalog['assets']};before={p['case_id']:p for c in old['cuts'] for p in c['people']};rows=[]
    extra=json.loads((out/'context-results.json').read_text())['people'] if (out/'context-results.json').exists() else {}
    for c in raw['cuts']:
        for original in c['people']:
            p=extra.get(original['case_id'],original);o=p['observation'];ablated=copy.deepcopy(o);ablated.pop('presentation',None)
            no_style=compare_body_shapes(catalog['assets'],ablated,default_body_id=catalog['default_body_id'])
            key=p['case_id'];a=assets[p['decision']['auto_body_id']];b=assets[before[key]['decision']['auto_body_id']];expected=labels.get(key)
            rows.append(dict(case_id=key,expected=expected,before=b['body_id'],after=a['body_id'],without_presentation=no_style['auto_body_id'],
                before_match=(b['metadata']['presentation_style']==expected) if expected else None,
                after_match=(a['metadata']['presentation_style']==expected) if expected else None,
                without_presentation_match=(assets[no_style['auto_body_id']]['metadata']['presentation_style']==expected) if expected else None,
                presentation=o.get('presentation'),presentation_source=o.get('presentation_source','person_crop'),
                shape_source=p['decision']['selection_source'],observation=o,decision=p['decision'],crop=p['crop']))
    annotated=[r for r in rows if r['expected']]
    summary=dict(total_people=len(rows),author_labeled=len(annotated),before_correct=sum(r['before_match'] for r in annotated),
        after_correct=sum(r['after_match'] for r in annotated),same_observation_without_style_correct=sum(r['without_presentation_match'] for r in annotated),
        uncorrected=[r['case_id'] for r in annotated if not r['after_match']],
        scope='8 user-reported development errors; presentation match only, not whole-body accuracy',labels_sent_to_model=False)
    (out/'comparison.json').write_text(json.dumps(dict(summary=summary,people=rows),ensure_ascii=False,indent=2))
    tab='\n'.join('| '+r['case_id']+' | '+('여성형' if r['expected']=='feminine' else '남성형')+' | '+assets[r['before']]['label']+' | '+assets[r['after']]['label']+' | '+('일치' if r['after_match'] else '미해결')+' |' for r in annotated)
    report=f'''# 남성형·여성형 디자인 구분 개선 — 2026-10-07

[비교 화면](review.html). 브랜치 `codex/body-matching-improvements`.

## 사용자 지정 8건

사용자가 직접 지적한 여성→남성 7건, 남성→여성 1건을 개발용 정답으로 저장했다. 정답 파일이나 기대 성별을 Gemini 입력에 넣지 않았다. 같은 17컷·28인물을 실제 Gemini Flash-Lite로 다시 관측했다.

- 지적한 8건의 디자인 계열 일치: **{summary['before_correct']}/8 → {summary['after_correct']}/8**.
- 동일한 새 관측에서 presentation 신호를 제거한 비교: **{summary['same_observation_without_style_correct']}/8**. 이는 프롬프트 변경에 따른 다른 속성 변화를 줄이기 위한 선택기 ablation이다.
- 미해결: {', '.join(summary['uncorrected']) or '없음'}.
- 이 8건은 오류로 선별하고 개선에 사용한 개발 사례다. 전체 정확도, 성별별 일반화 성능, 선택한 체형 전체의 정답률로 확대 해석하지 않는다. 나머지 20인물은 정답이 없어 회귀 여부를 확정하지 않았다.

| 인물 | 사용자 기준 | 기존 선택 | 개선 선택 | 결과 |
|---|---|---|---|---|
{tab}

## 변경

1. 체격·근육량·등신과 별도로 fictional character presentation을 관측한다: feminine/masculine/androgynous/unknown. 머리카락·의상만 있으면 uncertain으로 제한하며, 근육량·날씬함을 성별 대용으로 쓰지 않는다.
2. 실제 모델의 기존 제작 분류를 metadata.presentation_style로 등록한다. 명확한 단서가 있으면 QA/포즈 호환 자산 중 같은 계열 및 공용 모델을 비교한다. 약한 단서는 점수 동점이나 정보 부족 기본값에만 반영한다.
3. feminine 기본값은 female-base, masculine 기본값은 male-base. 관측이 없으면 기존 기본값을 유지하되 성별 탐지 성공으로 기록하지 않는다. 스타일 관측만 있고 체형이 가렸으면 auto_default 상태를 유지한다.
4. 같은 계열의 실행 가능 모델이 없으면 matching_style_unavailable을 기록한다. 검수·인물 소유권·기존 포즈 순위 계약은 유지한다.

## 증거와 한계

[관측 원자료](results.json), [선택 비교·ablation](comparison.json), [사용자 정답](author-labels.json), [모델 metadata](shape-catalog.json).

포즈 추출·Top-5는 기존 실제 결과를 재사용했다. 체형을 입힌 새 포즈 렌더는 실행하지 않았다. 실제 키·근육량이나 실제 인물의 생물학적 성별을 추정하는 기능이 아니다. 실제 관측·기본값·약한 단서를 별도 기록한다.
'''
    if not extra:report+='\n원본 전체 패널 문맥을 추가 전송하는 작업은 자동 승인 검사가 범위 확대를 이유로 차단했다. 로컬 [전송 범위 미리보기](context-proposal/preview.html)만 준비했으며 별도 승인을 기다린다. 기존 crop 기반 테스트는 완료했다.\n'
    (out/'REPORT.md').write_text(report)
    cards=[]
    for r in sorted(rows,key=lambda x:(x['expected'] is None,x['case_id'])):
        expected='사용자 정답 없음' if not r['expected'] else ('여성형' if r['expected']=='feminine' else '남성형')
        state='unlabeled' if not r['expected'] else 'fixed' if r['after_match'] else 'remaining'
        figures=''.join('<figure><img loading="lazy" src="'+H(src)+'" alt="'+H(label)+'"><figcaption>'+H(label)+'</figcaption></figure>' for label,src in [('실제 러프',r['crop']),('기존 '+assets[r['before']]['label'],assets[r['before']]['preview']),('개선 '+assets[r['after']]['label'],assets[r['after']]['preview'])])
        pres=r['presentation'] or {};cards.append('<article data-state="'+state+'"><h2>'+H(r['case_id'])+' · '+H(expected)+' · '+state+'</h2><div class="compare">'+figures+'</div><p>디자인 관측: '+H(str(pres.get('value')))+' / '+H(str(pres.get('visibility')))+' · 체격 선택: '+H(r['shape_source'])+'</p><p>'+H(pres.get('evidence','미관측'))+'</p><details><summary>관측·선택 근거</summary><pre>'+H(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></article>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>남성형·여성형 구분 개선</title><style>*{box-sizing:border-box}body{font:15px/1.7 system-ui;background:#f3f5f8;color:#192738;max-width:1160px;margin:auto;padding:24px}h1,h2{word-break:keep-all}h1{font-size:28px}h2{font-size:20px}article,.box{background:white;border-radius:14px;padding:20px;margin:20px 0}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:300px;object-fit:contain;background:#e8ebef}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}p{overflow-wrap:anywhere}a{color:#27569c}button{padding:10px 14px;margin:3px;border:1px solid #97a5b4;border-radius:6px;background:white;cursor:pointer}[hidden]{display:none}@media(max-width:650px){body{padding:12px}.compare{grid-template-columns:1fr 1fr}.compare figure:first-child{grid-column:1/-1}img{height:240px}}</style>'''
    page+='<h1>남성형·여성형 디자인 구분 개선</h1><p>실제 Gemini Flash-Lite · 17컷 / 28인물 · 체형 9종 · 2026-10-07</p><div class="box"><h2>지적한 8건 중 '+str(summary['after_correct'])+'건 개선</h2><p>사용자 기준과 일치: 기존 '+str(summary['before_correct'])+'/8 → 개선 '+str(summary['after_correct'])+'/8. 체격 속성을 그대로 두고 디자인 신호만 제거하면 '+str(summary['same_observation_without_style_correct'])+'/8입니다.</p><p>알려주신 오류 사례만의 개발셋 결과이며 전체 정확도가 아닙니다. 나머지 20인물은 정답 주석이 없습니다. 여성형·남성형 기본값도 체격 인식 성공으로 세지 않습니다.</p><a href="REPORT.md">상세 보고서</a> · <a href="comparison.json">비교 원자료</a> · <a href="author-labels.json">사용자 지정 정답</a></div>'
    page+='<nav>'+''.join('<button data-filter="'+k+'">'+v+'</button>' for k,v in [('annotated','사용자 지정 8건'),('fixed','개선'),('remaining','미해결'),('all','전체 28')])+'</nav>'+''.join(cards)
    page+='''<script>function filter(f){document.querySelectorAll('article').forEach(a=>a.hidden=!(f==='all'||f===a.dataset.state||f==='annotated'&&a.dataset.state!=='unlabeled'))}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>filter(b.dataset.filter));filter('annotated');</script></html>'''
    (out/'review.html').write_text(page);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
