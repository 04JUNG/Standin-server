"""Full roster report and compact, non-image experiment evidence."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();run=args.run.resolve();out=args.output.resolve();assert out.parent==run.parent
    load=lambda p:json.loads(p.read_text());esc=lambda v:html.escape(str(v),quote=True)
    protocol=load(run/'protocol.json');inventory=load(run/'inventory.json')
    verified={r['id']:r for r in load(run/'verified-summary.json')}
    summary={r['id']:r for r in load(run/'summary.json')}
    rows=[];decisions=Counter();people=[];matched_arms=0
    for entry in inventory:
        ident=entry['id'];folder=run/ident;native=ident in verified;v=verified.get(ident)
        goals=[]
        if entry.get('target_extracted'):
            target=load(folder/'target.json')
            goals += [side+': '+target['limbs'][side+'_arm']['action'] for side in ('left','right')]
        else:goals=['목표 추출 보류']
        reason=' · '.join(entry['reasons']) or '실제 FBX 대조 완료'
        status='적용 없음 · 대조 불가'
        if native:
            assert hashlib.sha256((folder/'after.fbx').read_bytes()).hexdigest()==v['final_fbx_sha256']
            before=load(folder/'diagnosis-before.json');after=load(folder/'diagnosis-after.json')
            selected=summary[ident]['automatically_selected_sides']
            for member in before['members'][:2]:
                decisions[member['status']]+=1
                if member['status']=='match':
                    matched_arms+=1;assert member['limb'].split('_')[0] not in selected
            if v['final_applied']:
                assert not after['selection'],'already-satisfied goals must not request another correction'
                assert all(m['status']=='match' for m in after['members'] if m['limb'].split('_')[0] in selected and m['limb'].endswith('_arm'))
            else:assert v['byte_exact_fallback']
            assert not v['new_skin']
            status='부분 적용' if v['final_applied'] else '원본 유지'
            reason=' · '.join(v['reasons']) if not v['final_applied'] else '접촉·비선택 관절·새/악화 피부 검사 통과'
            people.append({'id':ident,'final_applied':v['final_applied'],'selected_sides':selected,
                'baseline_sha256':hashlib.sha256((folder/'before.fbx').read_bytes()).hexdigest(),
                'final_sha256':v['final_fbx_sha256'],'protected_matrix_delta':v['protected_matrix_delta'],
                'new_skin_warnings':len(v['new_skin']),'timing':summary[ident]['timing']})
        link=f'<a href="native-gallery.html#{esc(ident)}">{esc(ident)}</a>' if native else esc(ident)
        target_link=f'<a href="{run.name}/{ident}/target.json">목표 JSON</a>' if entry.get('target_extracted') else ''
        rows.append(f'<tr><th>{link}</th><td>{esc(entry["reference_person_id"])}</td><td>{"<br>".join(esc(g) for g in goals)}</td><td>{status}</td><td>{esc(reason)} {target_link}</td></tr>')
    applied=[p for p in people if p['final_applied']];times=[p['timing']['case_total_seconds'] for p in applied]
    compact={'schema':'standin.cached-limb-expanded-evidence.v1','cuts':protocol['cuts'],
        'expected_people':len(inventory),'targets_extracted_people':sum(r.get('target_extracted',False) for r in inventory),
        'native_compared_people':len(people),'partially_applied_people':len(applied),'byte_exact_retained_people':len(people)-len(applied),
        'native_unavailable_people':len(inventory)-len(people),'inventory_hold_reasons':dict(Counter(r for e in inventory for r in e['reasons'])),
        'arm_supported_constraint_decisions_before':dict(decisions),'already_matching_arms_never_selected':matched_arms,
        'applied_files_repeat_diagnosis_proposals':0,'applied_case_seconds_range':[min(times),max(times)],
        'basic_state_median_ms':statistics.median(p['timing']['base_geometry_p50_ms'] for p in people),
        'new_vlm_calls':0,'new_vlm_tokens':0,'fresh_recognition_measured':False,'independent_accuracy_measured':False,
        'independent_score_delta':None,'default_enabled':False,'release_allowed':False,'people':people}
    (run/'expanded-summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('report_cached_limb_pilot.py')),str(run),str(out.parent/'native-gallery.html')],check=True)
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>자동 진단 · 동결 50컷 확대 실험</title>
    <style>body{{background:#111d28;color:#e1edf4;font:16px/1.7 system-ui;margin:30px auto;padding:0 24px;max-width:1400px}}a{{color:#9ed8ff}}h1{{line-height:1.3}}.note{{padding:20px;background:#253847;border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:13px;border-bottom:1px solid #3c5264;text-align:left;vertical-align:top}}th{{white-space:nowrap}}.table{{overflow:auto}}small{{color:#abc3d3}}code{{color:#bce6d3}}</style>
    <h1>자동 진단 · 동결 50컷 / 60명 확대 실험</h1>
    <p class="note"><b>추가 VLM 호출·토큰 0 · 기본 실행 비활성</b><br>기존 판독과 이미지 해시를 확인해 재사용했다. 새 이미지 인식이나 전체 정확도/점수 상승을 측정한 실험은 아니다.</p>
    <p>60명 중 <b>{compact['targets_extracted_people']}명</b>은 캐시 원문을 대응시켜 구조화했고, <b>{len(people)}명</b>은 동결된 실제 FBX와 비교했다. 구조화 건수는 확실하거나 올바른 목표의 수가 아니다.
    그중 <b>{len(applied)}명 부분 적용 · {len(people)-len(applied)}명 바이트 그대로 유지</b>했다. 나머지 {len(inventory)-len(people)}명은 3D 누락 또는 인물 대응의 불확실성으로 실행하지 않았다.</p>
    <p>다인 컷의 인물 순서가 이전 캐시와 다를 수 있어 원본 박스의 고유 대응을 사용했다. {compact['inventory_hold_reasons'].get('cached_person_mapping_missing_or_ambiguous',0)}명의 대응을 보류했고, 선택 FBX 기록이 없는 인물은{compact['inventory_hold_reasons'].get('no_frozen_selected_fbx_record',0)}명이다. 두 사유는 일부 겹치므로 합산하지 않는다.</p>
    <p>새로 확인한 허리 손 3명(T015·T029·T034)을 포함해, 기존 VLM의 명시 좌우·양팔 목표를 보존한다. ‘허리로 올림’과 ‘팔을 높이 듦’, 머리 옆/뒤, 소품 잡기를 구분한다.
    기존 비교 조건이 맞던 팔 {matched_arms}개는 보정 대상으로 선택하지 않았고, 보정한{len(applied)}명의 같은 목표를 다시 진단하면 추가 보정 요청은0개였다. 이 확인은 <b>지원 조건의 일관성 검사이며 이미지 정답에 대한 오보정률은 아니다.</b></p>
    <p>부분 적용 소요시간 <b>{min(times):.2f}~{max(times):.2f}초</b>. 이미지 판독·Blender 시작·렌더·큐 시간은 제외했다.
    저장본 재임포트에서 비선택 관절·피부·접촉을 검사했고 새/악화 피부 경고는0개였다. 기존 경고나 손가락 내부 충돌을 모두 해결했다는 뜻은 아니다.</p>
    <p><a href="native-gallery.html">{len(people)}명 원본·전후 3방향 렌더·팔/다리 목표와 판정 보기</a> · <a href="{run.name}/expanded-summary.json">확대 실험 결과 JSON</a> · <a href="{run.name}/protocol.json">프로토콜</a> · <a href="../standin-leg-auto-diagnosis-local-20261030/index.html">다리 담당의 별도 확대·후보 검토</a></p>
    <p class="note">미해결: 우산 컷의 휴대폰 손, 무릎 컷의 반대 주먹, 양손/팔짱 공동 보정, 가려진 손, 머리 뒤·기타 소품 접촉은 완성되지 않았다.
    턱 받침은 손 전체 배치를 검사할 수 없어 보류한다. 다리 목표와 형태 비교는 보존하지만 접지/책상 표면 근거가 없으면 자동 적용하지 않는다. VLM 자체가 틀리게 읽은 목표는 이번 연결 수정만으로 정정할 수 없다.</p>
    <h2>전체 60명 실행 범위</h2><div class="table"><table><thead><tr><th>실험 ID</th><th>동결 인물</th><th>팔 목표</th><th>최종 파일</th><th>실행/보류 근거</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p><small>렌더 정성 검토는 같은 에이전트가 적용{len(applied)}명을 확인한 기록이다. 유지{len(people)-len(applied)}명은 파일 바이트 동일성을 확인했으며 올바른 자세로 인증하지 않는다. 앱/운영/스테이징에 자동 진단을 연결하거나 배포하지 않았다.</small></p></html>'''
    out.write_text(page)
    print(json.dumps({k:v for k,v in compact.items() if k!='people'},ensure_ascii=False))


if __name__=='__main__':main()
