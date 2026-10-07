#!/usr/bin/env python3
"""Build a local evidence review from completed evaluations; no inference calls."""
import argparse,html,json,shutil
from collections import Counter
from pathlib import Path
H=html.escape

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();before=json.loads(args.baseline.read_text());after=json.loads((out/'results.json').read_text());visual=json.loads((out/'visual-final/results.json').read_text())
    shutil.copy2(args.baseline,out/'baseline-results.json')
    old={p['case_id']:p for c in before['cuts'] for p in c['people']};new={p['case_id']:p for c in after['cuts'] for p in c['people']}
    assets=json.loads((out/'shape-catalog.json').read_text())['assets'];labels={a['body_id']:a['label'] for a in assets};previews={a['body_id']:a['preview'] for a in assets}
    def summarize(people):
        ds=[p['decision'] for p in people]
        return dict(unique=sum(len(d.get('tied_body_ids',[]))==1 for d in ds),ties=sum(len(d.get('tied_body_ids',[]))>1 for d in ds),default=sum(d['selection_source']=='auto_default' for d in ds))
    counts=[summarize(old.values()),summarize(new.values()),summarize(visual['people'])]
    reasons=Counter();attempted=0;dual=0;same_set=0;head_rejected=0
    for p in visual['people']:
        t=p['decision']['visual_comparison'];reasons[t['reason']]+=1
        if 'forward' in t:
            dual+=1;same_set+=set(t['forward']['best_ids'])==set(t['reverse']['best_ids'])
        key=p['case_id'].replace(':','-')
        errs=list((out/'visual-final/provider'/key).glob('*.error.json'))
        head_rejected+=any(json.loads(x.read_text()).get('validation_error')=='visual_forbidden_head_proportion' for x in errs)
    measurements=json.loads((out/'measurements.json').read_text())['profiles']
    matrix='\n'.join(f'| {name} | {c["unique"]} | {c["ties"]} | {c["default"]} |' for name,c in zip(['기존','부위별 관측 개선','후보 이미지 비교 후'],counts))
    distribution=Counter(p['decision']['auto_body_id'] for p in visual['people'])
    per=[];cards=[]
    for p in visual['people']:
        key=p['case_id'];old_id=old[key]['decision']['auto_body_id'];d=p['decision'];chosen=d['auto_body_id'];obs=p['observation'];t=d['visual_comparison']
        state='default' if d['selection_source']=='auto_default' else 'tie' if len(d.get('tied_body_ids',[]))>1 else 'unique'
        per.append(f'| {key} | {labels[old_id]} | {labels[chosen]} | {state} | {t["reason"]} |')
        rows=''.join(f'<figure><img loading="lazy" src="{H(src)}" alt="{H(label)}"><figcaption>{H(label)}</figcaption></figure>' for label,src in [('실제 러프',p['crop']),('기존: '+labels[old_id],previews[old_id]),('개선 후: '+labels[chosen],previews[chosen])])
        cards.append(f'<article data-state="{state}"><h2>{H(key)} · {H(state)}</h2><div class="compare">{rows}</div><p>전신 관측: {obs.get("full_body_visible")} · 비교: {H(t["reason"])}</p><p>{H(json.dumps(obs.get("coverage",{}),ensure_ascii=False))}</p><details><summary>관측·선택·양방향 비교 원자료</summary><pre>{H(json.dumps(p,ensure_ascii=False,indent=2))}</pre></details></article>')
    report=f'''# 체형 매칭 개선·실제 재평가 — 2026-10-07

브랜치: `codex/body-matching-improvements`. [비교 화면](review.html).

## 결과

같은 실제 러프 17컷·28인물, 실제 FBX 9종(통통형 제외), 응답 모델 `gemini-3.5-flash-lite`로 평가했다. 인물 박스·관절·기존 포즈 Top-5는 앞선 실제 추론 결과를 해시 검증 후 재사용했다.

| 단계 | 단일 최저점/채택 | 동점 | 기본값 |
|---|---:|---:|---:|
{matrix}

**단일 후보 수는 정확도가 아니다.** 전신이 보이지 않는 러프의 등신을 제외하면서 기존 아동형 3건(131211:p0/p1, 4.56.21:p0)은 아동형으로 선택되지 않았다. 이 3건의 실제 정답이 확정됐다는 의미는 아니다. 동점 감소 중 일부는 근거 제거 후 기본값으로 이동한 결과다.

최종 선택 분포: {', '.join(labels[k]+' '+str(v) for k,v in distribution.items())}.

## 실제로 바꾼 것

1. 부위별 가시성: head/torso/arms/legs와 전신 관측 여부. 가려진 몸통의 폭·볼륨만 제외하고 보이는 팔의 근육 단서는 유지한다. 잘림/단축 또는 머리·다리 미관측이면 전체 등신 축을 제외한다.
2. 실제 FBX 프로필: 9종에서 높이로 정규화한 관절 길이·몸통 단면 등 12개 지표를 추출하고 측면 렌더를 추가했다. 머리 extent와 단면은 skin-weight/band 기반 proxy로, 사람이 검수한 정밀 해부학 치수가 아니다. 서로 다른 pose의 3D 치수를 러프 2D 길이에 직접 점수화하지 않는다. 기존 의미 catalog는 고정하고 새 지표는 후보 이미지 비교의 참조 자료로 제공했다.
3. 후보 이미지 비교: 정면·측면 실제 자산 이미지, 익명 후보 ID, 역순+라벨 회전 재비교. 동일 단독 후보로 합의할 때만 채택하며 오류/동점/불일치 시 이전 결정을 보존한다. 실측 수치와 이미지 비교의 효과를 따로 분리한 ablation은 아직 없다.
4. 안전 계약: 기존 포즈 순서·QA 자산 필터·오류 복구 유지. 서비스에 명시적 selector 주입 지점만 추가했고 기본 API에는 추가 호출을 켜지 않았다.

## 후보 직접 비교 결과

- 두 순서에서 단독 1위 합의로 채택: **{visual['summary']['accepted_visual']}명**.
- 두 응답 모두 파싱·근거 검증을 통과한 비교: {dual}명. 같은 최선 후보 집합: {same_set}명. 같은 집합이어도 여러 체형이면 단독 채택하지 않는다.
- 판정 사유: `{json.dumps(dict(reasons),ensure_ascii=False)}`.
- 직접 비교 첫 실험에서 잘린 러프를 머리 비율로 아동형으로 고른 사례를 발견했다. 최종 구현은 허용 근거 축/부위를 제한하고 명시적인 머리 비율 근거를 추가 차단한다. 프롬프트만으로 충분하지 않았다.
- 최종 단계에서 명시적인 머리 비율 위반으로 거부한 응답 기록: {head_rejected}인물. 이는 provider 장애와 별도의 검증 거부다.
- 순서 일치가 정답을 보장하지 않으므로 이 비교를 운영 정확도 개선으로 승격하지 않았다.

## 오류와 재현

관측 첫 실행의 429 두 컷은 원기록을 보존하고 순차 재시도해 최종 provider 오류 0명이 됐다. 후보 비교 최초 실험(`visual/`)은 고해상도 입력과 429 오류를 포함한다. 근거 제한·저해상도 입력을 적용한 `visual-v2/` 이후, 설명 길이에 대한 파서를 보완해 저장 응답을 `visual-final/`에서 재파싱하고 미응답 건만 재호출했다. 이를 독립 표본이나 여러 번의 정확도 측정으로 세지 않는다. 원문과 usage·모델 버전·입력/코드 해시·재사용 출처를 각 단계에 보존했다.

- [기존 결과](baseline-results.json), [개선 관측 결과](results.json), [최종 비교 결과](visual-final/results.json).
- [12개 지표](measurements.json), [FBX manifest](models.json), [초기 코드 snapshot hash](baseline-code.json).
- 원본 checkout·FBX·배포 catalog는 변경하지 않았다. Top-5를 새 체형으로 리타게팅하거나 렌더하지 않았다.
- 코드 검증은 `verification.json`에 기록한다. 실제 러프 정확도와 구분한다.

## 다음 판정 기준

같은 자료로 개선한 개발셋 결과다. 작가의 허용 체형 집합 주석이 없어 BodyHit@1/@3 및 실제 사용자 수정률은 미측정이다. 기준선의 오류 감소와 기본값 증가를 함께 본다. 운영 승격에는 새 작가·캐릭터의 주석 데이터와 동일 pose/camera의 투영 비교가 필요하다. 2.0 캐릭터 기억은 이번 구현 범위 밖이다.

## 인물별 결과

| 인물 | 기존 | 최종 | 상태 | 후보 비교 |
|---|---|---|---|---|
{chr(10).join(per)}
'''
    (out/'REPORT.md').write_text(report)
    table='<table><thead><tr><th>단계</th><th>단일 후보</th><th>동점</th><th>기본값</th></tr></thead><tbody>'+''.join(f'<tr><td>{name}</td><td>{c["unique"]}</td><td>{c["ties"]}</td><td>{c["default"]}</td></tr>' for name,c in zip(['기존','관측 개선','이미지 비교 후'],counts))+'</tbody></table>'
    model_cards=''.join(f'<figure><img loading="lazy" src="{H(a["preview"])}" alt="{H(a["label"])}"><figcaption>{H(a["label"])}</figcaption></figure>' for a in assets)
    metric_keys=list(measurements[0]['metrics']);measurement_table='<table><tr><th>체형</th>'+''.join('<th>'+H(k)+'</th>' for k in metric_keys)+'</tr>'+''.join('<tr><td>'+H(labels[m['body_id']])+'</td>'+''.join(f'<td>{m["metrics"].get(k,0):.3f}</td>' for k in metric_keys)+'</tr>' for m in measurements)+'</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>체형 매칭 개선 · 실제 테스트</title><style>
*{box-sizing:border-box}body{font:15px/1.7 system-ui,sans-serif;background:#f2f4f7;color:#1d2939;max-width:1180px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:20px}article,.box{background:white;padding:22px;border-radius:14px;margin:20px 0}p{overflow-wrap:anywhere}a{color:#2456a6}.compare,.models{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;height:270px;object-fit:contain;background:#eaecef}figcaption{font-size:13px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #d9dfe6}.scroll{overflow:auto}button{padding:9px 14px;margin:4px;background:white;border:1px solid #8090a5;border-radius:6px;cursor:pointer}[hidden]{display:none}.warn{border-left:4px solid #b47716;padding-left:16px}@media(max-width:650px){body{padding:14px}.compare{grid-template-columns:1fr 1fr}.compare figure:first-child{grid-column:1/-1}article{padding:14px}.models{grid-template-columns:repeat(2,minmax(0,1fr))}img{height:220px}}
</style><h1>체형 매칭 개선 · 실제 러프 재평가</h1><p>2026-10-07 · 별도 워크트리 · Gemini 3.5 Flash-Lite · 실제 17컷 / 28인물 / 체형 9종</p>'''
    page+='<div class="box"><h2>잘림 처리 개선, 9종 구분은 아직 미완성</h2>'+table+f'<p>직접 이미지 비교의 단독 후보 합의 채택: {visual["summary"]["accepted_visual"]}명. 단일 후보 수와 정답률은 다릅니다.</p><p class="warn">정답 주석이 없는 개발셋 평가입니다. 인물 검출·포즈 Top-5는 기존 실제 결과를 재사용했고, 선택 FBX는 rest 렌더로 표시합니다. 새 체형의 포즈 렌더는 실행하지 않았습니다.</p><a href="REPORT.md">상세 보고서</a> · <a href="visual-final/results.json">최종 원자료</a> · <a href="measurements.json">실측 프로필</a></div>'
    page+='<details class="box"><summary>실제 FBX 9종 · 측정 지표</summary><div class="models">'+model_cards+'</div><p>정규화한 rest geometry 지표. 머리와 단면은 proxy이며 사람 검수가 필요합니다.</p><div class="scroll">'+measurement_table+'</div></details>'
    page+='<nav><button data-filter="all">전체 28</button><button data-filter="unique">단일 후보</button><button data-filter="tie">동점</button><button data-filter="default">기본값</button></nav>'+''.join(cards)
    page+='''<script>document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>document.querySelectorAll('article').forEach(a=>a.hidden=!(b.dataset.filter==='all'||b.dataset.filter===a.dataset.state)));</script></html>'''
    (out/'review.html').write_text(page)
    print('REPORT.md and review.html written',counts)
if __name__=='__main__':main()
