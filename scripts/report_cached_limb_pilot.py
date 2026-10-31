"""Publish an offline diagnosis pilot; verification and visual audit stay distinct."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--previous-run',type=Path)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    load=lambda p:json.loads(p.read_text())
    esc=lambda value:html.escape(str(value),quote=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert out.parent==run.parent,'keep report beside the immutable run'
    cases=load(run/'cases.json');summary=load(run/'summary.json')
    verified={r['id']:r for r in load(run/'verified-summary.json')}
    audit=load(run/'visual-audit.json')
    assert audit['source']=='same_agent_native_render_review_not_independent_scoring'
    assert set(audit['cases'])=={c['id'] for c in cases}
    cards=[];rows=[]
    side_names={'left_arm':'인물 왼팔','right_arm':'인물 오른팔','left_leg':'인물 왼다리','right_leg':'인물 오른다리'}
    status_names={'held':'판단 보류','match':'지원하는 조건만 일치','mismatch':'불일치 발견',
                  'shape_matched_contact_held':'형태 조건 일치 · 접촉 보류'}
    reasons={
        'unobservable_or_unknown':'기존 판독에 목표 없음',
        'uncertain_target_or_anatomical_side':'자기 기준 좌우 근거 부족',
        'unresolved_anatomical_side_evidence':'명시 좌우 또는 관절 연결 근거가 없음/보류됨',
        'uncertain_target_observation':'목표 관측의 확신이 부족함',
        'no_supported_geometric_comparator':'목표는 보존했지만 해당 동작의 3D 비교 검사가 미연결',
        'chin_support_requires_whole_hand_region_and_orientation_comparator':'턱 접촉만으로 손 전체 배치를 판단할 수 없음',
        'coupled_both_chin_support_requires_atomic_recipe_and_pair_checker':'양손 턱 받침의 공동 검사·보정 미연결',
        'correction_recipe_or_measurement_unavailable':'안전한 파츠 선택에 필요한 목표·측정 부족',
    }
    leg_goals=0
    for case,s in zip(cases,summary):
        tid=case['id'];assert s['id']==tid
        folder=run/tid;v=verified[tid];a=audit['cases'][tid]
        assert sha(folder/'input.png')==case['input_sha256']
        assert sha(folder/'before.fbx')==case['source_sha256']
        assert sha(folder/'after.fbx')==v['final_fbx_sha256']
        # Verification may roll back an export. Use the original diagnostic in
        # that case, never stale measurements of the rejected candidate.
        after_name='diagnosis-after.json' if sha(folder/'after.fbx')==s['final_fbx_sha256'] else 'diagnosis-before.json'
        before=load(folder/'diagnosis-before.json');after=load(folder/after_name)
        # The separate leg adapter has richer bilateral/support observations.
        # Use its actual saved-file diagnosis instead of the arm placeholder.
        if (folder/'leg-diagnosis-after.json').is_file():
            for diagnostic,filename in ((before,'leg-diagnosis-before.json'),(after,'leg-diagnosis-after.json')):
                for leg in load(folder/filename)['legs']:
                    name=leg['side']+'_leg';wanted=leg['target']
                    normalized={'action':wanted['action'],'evidence':' · '.join(c['text'] for c in wanted['evidence']),
                        'confidence':wanted['confidence'],'chain_visible':wanted['chain_visible'],
                        'laterality_confidence':'high' if wanted['confidence']=='high' and wanted['laterality_source']!='unknown' else 'unknown',
                        'laterality_source':wanted['laterality_source'],'contact_side':'unknown','orientation_source':'unknown'}
                    replacement={**leg,'limb':name,'target':normalized,'clause':None}
                    diagnostic['members']=[replacement if r['limb']==name else r for r in diagnostic['members']]
        leg_goals+=sum(r['limb'].endswith('_leg') and r['target']['action']!='unknown' for r in before['members'])
        plan=load(folder/'plan.json');members={m['side']:m for m in plan['members'] if m.get('active')}
        limb_rows=[]
        for row,final in zip(before['members'],after['members']):
            name=row['limb'];wanted=row['target'];side=name.split('_')[0]
            part=members.get(side) if name.endswith('_arm') and v['final_applied'] else None
            evidence=wanted['evidence'] or '기존 기록에 부위별 관측 없음'
            reason=' · '.join(reasons.get(r,r) for r in row['reasons'])
            checks=' · '.join(f"{c['metric']}={c['value']:.3f} ({c['status']})" for c in row['checks'] if isinstance(c['value'],(float,int)))
            side_source='자연어에 좌우/양팔 명시' if wanted.get('laterality_source') in ('explicit_natural_language','explicit_bilateral_natural_language') else ('관절 연결 근거 있음' if wanted['chain_visible'] else '명시 근거 없음/보류')
            contact=' · 접촉 무릎: '+{'left':'인물 왼쪽','right':'인물 오른쪽'}[wanted['contact_side']] if wanted['contact_side'] in ('left','right') else ''
            limb_rows.append(f'''<tr><th>{side_names[name]}</th><td><b>{esc(wanted['action'])}</b><br>{esc(evidence)}
              <small>{side_source} · 체인 가시성: {'확인됨' if wanted['chain_visible'] else '별도 확인 없음'}{contact}</small></td>
              <td>{status_names[row['status']]}<small>{esc(reason)}</small><small>{esc(checks)}</small></td>
              <td>{esc(part['action']) if part else '적용 없음'}<small>{esc(row.get('clause') or '') if part else ''}</small></td>
              <td>{status_names[final['status']]}</td></tr>''')
        prefix=run.name+'/'+tid
        images=''.join(f'''<figure><figcaption>{title}</figcaption><img {'class="native" data-prefix="'+prefix+'/'+kind+'_"' if kind!='input' else ''}
                    src="{prefix}/{kind+'_0.png' if kind!='input' else 'input.png'}" alt="{tid} {title}" loading="lazy"></figure>'''
                    for title,kind in (('평가 원본','input'),('동결된 선택 3D','before'),('로컬 자동 진단 후','after')))
        timing=s['timing'];is_applied=v['final_applied']
        badge='기하 검사 통과 · 부분 적용' if is_applied else '보류 · 원본 그대로'
        if not is_applied:assert v['byte_exact_fallback']
        delta=''
        if args.previous_run:
            old=load(args.previous_run/tid/'target.json')['limbs'];changed=[]
            for row in before['members']:
                name=row['limb'];new=row['target'];prior=old[name]
                if any(new[k]!=prior[k] for k in ('action','laterality_confidence','contact_side')):
                    changed.append(f"<tr><th>{side_names[name]}</th><td>{esc(prior['action'])} · 좌우 {esc(prior['laterality_confidence'])}</td><td>{esc(new['action'])} · 좌우 {esc(new['laterality_confidence'])}</td><td>{esc(new['evidence'])}</td></tr>")
            if changed:delta='<h3>같은 VLM 원문의 목표 해석 수정</h3><div class="table"><table><thead><tr><th>부위</th><th>이전 연결</th><th>수정 후 연결</th><th>기존 원문 근거</th></tr></thead><tbody>'+''.join(changed)+'</tbody></table></div>'
        for angle in (0,45,90):
            for kind in ('before','after'):assert (folder/f'{kind}_{angle}.png').is_file()
        cards.append(f'''<article id="{esc(tid)}" data-applied="{str(is_applied).lower()}"><h2>{tid} · {esc(case['case_id'])}<span>{badge}</span></h2>
            <p class="audit">{esc(a['observation'])}</p>{delta}<div class="images">{images}</div>
            <div class="table"><table><thead><tr><th>부위</th><th>기존 VLM 목표</th><th>로컬 진단</th><th>붙인 파츠</th><th>저장 후 재진단</th></tr></thead>
            <tbody>{''.join(limb_rows)}</tbody></table></div>
            <p class="meta">로컬 실행 {timing['case_total_seconds']:.3f}초 · 상태 읽기 {timing['base_geometry_p50_ms']:.3f}ms · 접촉 포함 읽기 {timing['target_contact_read_seconds']*1000:.1f}ms
            · 비교 {timing['comparison_ms']:.3f}ms · 파츠 보정 {timing['correction_seconds']:.3f}초</p>
            <details><summary>판단·검증 파일과 실제 FBX</summary><p>
            {' · '.join(f'<a href="{prefix}/{filename}">{label}</a>' for label,filename in [('기존 VLM 원문','cached-vlm-evidence.json'),('목표','target.json'),('원본 상태','native-before.json'),('진단','diagnosis-before.json'),('파츠 계획','plan.json'),('재임포트 검사','export-verification.json'),('최종 FBX','after.fbx')])}</p>
            <p>변경하지 않은 관절의 최대 행렬 차이 {v['protected_matrix_delta']:.8f} · 새/악화 피부 경고 {len(v['new_skin'])}.
            기하 검사는 동작 인식 정확도나 손가락 내부 충돌을 보증하지 않는다.</p></details></article>''')
        rows.append({'id':tid,'geometric_application':is_applied,'visual_audit':a,
            'decision_before':{m['limb']:m['status'] for m in before['members']},'timing':timing})
    totals=load(run/'native-timing.json')
    all_times=[s['timing']['case_total_seconds'] for s in summary]
    applied_times=[s['timing']['case_total_seconds'] for s in summary if verified[s['id']]['final_applied']]
    applied_count=sum(v['final_applied'] for v in verified.values())
    measured={'new_api_calls':0,'new_api_tokens':0,'cases':len(cases),
        'geometric_applications':applied_count,'preserved_explicit_leg_goals':leg_goals,
        'fresh_image_recognition_tested':False,'accuracy_estimated':False,
        'base_geometry_median_ms':statistics.median(s['timing']['base_geometry_p50_ms'] for s in summary),
        'case_total_seconds_range':[min(all_times),max(all_times)],
        'applied_case_seconds_range':[min(applied_times),max(applied_times)],
        'batch_wall_seconds':totals['batch_wall_seconds'],'cases_detail':rows}
    (run/'report-results.json').write_text(json.dumps(measured,ensure_ascii=False,indent=2)+'\n')
    (out.parent/'results.json').write_text(json.dumps(measured,ensure_ascii=False,indent=2)+'\n')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>로컬 자동 진단 · CASE_COUNT명 실험</title><style>
    *{box-sizing:border-box}body{margin:0;background:#101923;color:#e3edf5;font:16px/1.65 system-ui,sans-serif}main{max-width:1460px;margin:auto;padding:28px}
    h1{font-size:30px;line-height:1.35}h2{font-size:23px;margin-top:0}p{max-width:1120px}a{color:#9ed8ff}small{display:block;color:#a9bdcb;font-size:13px;margin-top:8px}
    .lead{font-size:20px;color:#bce6d3}.note,.audit{background:#253544;padding:16px;border-radius:10px}.audit{border-left:4px solid #e7c58d}
    article{border:1px solid #354b5e;border-radius:16px;padding:24px;margin:28px 0;background:#192733}h2 span{float:right;font-size:14px;color:#e7c58d}
    .images{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0;border-radius:12px;overflow:hidden;background:#35404e}figcaption{padding:12px;background:#293d4e}img{display:block;width:100%;aspect-ratio:1;object-fit:contain}
    .table{overflow:auto}table{width:100%;border-collapse:collapse;margin-top:22px;font-size:14px}td,th{padding:14px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #354b5e}th{white-space:nowrap}td:nth-child(2){width:31%}td:nth-child(3){width:32%}
    button{background:#263c4d;border:1px solid #587287;color:#dfebf4;padding:10px 16px;margin:3px;border-radius:8px;cursor:pointer}button.active{background:#bce6d3;color:#102c30}nav{position:sticky;top:0;background:#101923ef;padding:10px 0;z-index:1}.meta{font-size:14px;color:#abc0ce}details{margin-top:16px}article[hidden]{display:none}
    @media(max-width:800px){main{padding:14px}article{padding:14px}.images{grid-template-columns:1fr}h2 span{float:none;display:block}}
    </style><main><h1>추가 API 0회 · 로컬 자동 진단 CASE_COUNT명</h1>
    <p class="lead">기존 VLM 목표 → 실제 3D 관절·피부 측정 → 좌우별 불일치 판정 → 파츠 선택 → 저장 후 검사</p>
    <p class="note">같은 이미지인지 SHA-256으로 확인한 기존 판독 기록을 재사용했다. 새 이미지를 로컬 VLM으로 읽은 실험은 아니다.
    추가 API 호출·토큰 사용은 0이다. 평가 정답·점수·사용자 수정 지시는 진단 입력에서 제외했다.</p>
    <p><b>CASE_COUNT명 중 APPLIED_COUNT명에 팔 보정이 부분 적용됐고 HELD_COUNT명은 원본을 유지했다.</b> 이것은 동작 인식 정확도가 아니다.
    기존 자연어에 명시된 다리 목표 LEG_GOALS개도 보존했다. 다리 보정은 이 팔 실험에 연결하지 않았으며, 동작별 3D 비교 미지원과 좌우 미인식을 구분해서 표시한다. 양팔 관계·가려진 손·시선까지 완성한 결과가 아니며 앱에는 연결하지 않았다.</p>
    <p>실제 보정이 들어간 컷은 <b>APPLIED_TIMES초</b>, 네 팔다리의 기본 상태 읽기는 중앙값 <b>BASE_MS밀리초</b>였다.
    Blender를 켠 뒤 순차 실행한 시간이며 이미지 판독·프로세스 시작·미리보기 렌더·큐 대기는 제외한다. 피부 접촉 읽기는 별도 약 0.1~0.3초가 걸린다.</p>
    <p>이번 수정: 별도 좌우 검증 필드가 비어 있어도 자연어에 명시된 인물 좌우를 목표로 받는다. 관절 연결이 실제로 보이는지는 별도 값으로 보존한다.
    한 문장의 오른팔·왼팔 설명을 끊어서 다른 팔의 무릎 동작이 섞이지 않도록 했다. 명시적 보류·모순·화면 기준 좌우는 계속 보류한다.</p>
    <nav><button data-filter="all" class="active">전체 CASE_COUNT</button><button data-filter="true">부분 적용 APPLIED_COUNT</button><button data-filter="false">보류 HELD_COUNT</button>
    <button data-angle="0" class="active">정면</button><button data-angle="45">45°</button><button data-angle="90">측면</button></nav>
    CARDS
    <p class="meta">검사 범위: 접촉·방향·비선택 관절·기존 대비 새/악화 피부 경고. 자연스러움 검토는 같은 에이전트가 네이티브 렌더를 본 정성 기록이며 독립 재채점이 아니다.
    <a href="RUN/report-results.json">결과·시간 JSON</a> · <a href="RUN/visual-audit.json">렌더 검토</a> · <a href="RUN/protocol.json">실험 조건</a></p>
    </main><script>
    document.querySelectorAll('[data-angle]').forEach(b=>b.onclick=()=>{
      document.querySelectorAll('[data-angle]').forEach(x=>x.classList.toggle('active',x===b));
      document.querySelectorAll('img.native').forEach(i=>i.src=i.dataset.prefix+b.dataset.angle+'.png');
    });
    document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{
      document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));
      document.querySelectorAll('article').forEach(a=>a.hidden=b.dataset.filter!=='all'&&a.dataset.applied!==b.dataset.filter);
    });</script></html>'''
    page=page.replace('APPLIED_TIMES',f'{min(applied_times):.2f}~{max(applied_times):.2f}').replace('BASE_MS',f"{measured['base_geometry_median_ms']:.2f}")
    for key,value in {'CASE_COUNT':len(cases),'APPLIED_COUNT':applied_count,'HELD_COUNT':len(cases)-applied_count,'LEG_GOALS':leg_goals}.items():page=page.replace(key,str(value))
    page=page.replace('CARDS',''.join(cards)).replace('RUN/',run.name+'/')
    out.write_text(page)
    print(json.dumps({k:v for k,v in measured.items() if k!='cases_detail'},ensure_ascii=False))


if __name__=='__main__':main()
