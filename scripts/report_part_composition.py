"""Report an offline composition probe with authentic input + diagnostic images."""
import argparse
import base64
import json
from pathlib import Path
import shutil


def encoded(path):
    path=Path(path);mime='image/png' if path.suffix.lower()=='.png' else 'image/jpeg'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','images','thumbs','attribution'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();data=json.loads(args.experiment.read_text());out=args.experiment.parent;s=data['summary']
    lines=['# T013 팔짱 상체 + 앉은 하체 분리 검색·합성 실험','',
        '## 결론','',
        f"- 앉기 베이스 {s['attempted']}개에 같은 팔짱 donor를 이식: 구조 검사 {s['kinematic_probe_pass']}개 통과, 1개 제외.",
        f"- 리그 간 Euler 회전값 단순 복사는 팔짱 관계 {s['naive_fold_geometry_pass']}/{s['attempted']}개 통과.",
        '- 방향 기반 retarget는 앉은 상태를 유지하면서 가슴 앞 손 교차를 만들었다. 이 결과는 단일 러프의 구조 실험이지 11/12 정확도나 제품 성공률이 아니다.',
        '- 전신 통째 검색(v3)은 스트레칭을 반환했고, 핵심 조건 검색(v4)은 검증 후보 0개였다. 이번에는 별도의 synthetic identity/BVH를 만들었으며 라이브러리에 삽입하지 않았다.','',
        '## 입력과 분리 검색','',
        '- 같은 T013 원본 러프, 인물 슬롯 2(0-based index 1). 추가 Gemini 호출·러프 전송 0회.',
        f"- 설명 출처: `{data['description_source']}`의 이전 실제 Gemini 3.8 Flash/d2 설명을 고정.",
        '- T013의 새 Gemini 실호출 실패를 해결한 실험은 아니다. 검색/합성 효과만 분리했다.',
        f"- 실제 설명: {data['description']['search_text']}",
        f"- 상체 쿼리: {data['upper_query']['text']}",
        f"- 상체 donor: `{data['donor']}`. reviewed fold_arms + 실제 BVH 가슴 높이 손 교차 검증. 원본/미러는 중복 제외.",
        f"- 하체 쿼리: {data['lower_query']['text']}",
        f"- sitting 태그 {data['lower_pool']}개에 하체 조건으로 별도 검색. arms 행동/관찰은 하체 검색에서 제거.",
        '- chair-like 기하 검증: 양 무릎 70~115도, 허벅지 수평 대비 기울기 30도 이하, 양 정강이 아래 방향 분율 >=0.65.',
        '- 범위는 미리 고정한 실험 기준이며 실제 의자 접촉/물리적 지지를 보증하지 않는다.',
        '- 이번 분해는 팔짱+앉기 사례에 한정된 관찰 문장의 명시적 분리다. 일반 자연어 자동 분해 성능을 검증한 것은 아니다.','',
        '## 합성 방법','',
        '- 첫 단계는 팔 체인 이식이다. 목/머리/쇄골/척추/골반/다리는 앉기 베이스를 유지하며 torso 전체를 donor로 교체하지 않는다.',
        '- 두 포즈의 몸통 축을 맞춰 donor의 상완·전완·매핑 가능한 손가락 방향을 target 리그로 옮긴다.',
        '- target의 원래 OFFSET·부모 계층·position 채널과 뼈 길이를 보존하고 target의 Euler 채널 순서로 새 회전값을 계산한다.',
        '- 손바닥은 여러 손가락 방향의 강체 최소제곱 근사다. 단일 bone의 twist는 방향만으로 유일하게 결정되지 않는다.',
        '- spine/hips 경계는 바꾸지 않으므로 허리 seam을 새로 붙이는 방식이 아니다. 러프에서 관절 좌표를 생성하거나 스켈레톤 없는 컷에 기존 refine을 실행하지 않았다.','',
        '## 구조 검증','',
        '- 메모리 합성에서 모든 보호 관절 및 COCO 하체 6관절 위치 변화 = 0.',
        f"- 전체 뼈 길이 오차 최대 {s['max_bone_length_delta_torso']:.3g} torso(수치 오차).",
        f"- BVH 저장/재읽기 성공 {s['reloaded_export_pass']}개. HIERARCHY 원문 보존·팔짱 유지 확인.",
        f"- 재읽기 후 보호 관절 변화 최대 {s['max_export_protected_delta_torso']:.3g} torso(6자리 직렬화 오차, 검사 상한 1e-6).",
        '- 팔-몸통 및 팔-다리 capsule proxy에서 신규 깊은 관통을 검사했다. proxy의 접촉 반경/허용치(0.02 torso)는 실험용이며 피부 메시 검사는 아니다.',
        f"- 입력 BVH SHA 불변: {s['source_bvhs_unchanged']}. 원본 DB/태그/기하 검색/API/refine은 수정하지 않았다.",'',
        '## 전체 12개 조합','', '| 조합 | 하체 베이스 | 단순 복사 팔짱 | 방향 이식 | 사유 |', '|---|---|---|---|---|']
    for i,r in enumerate(data['trials'],1):
        d=r.get('diagnostics',{});lines.append(f"| {i} | {r['base']} | {'통과' if r.get('naive_fold_geometry') else '실패'} | {'구조 통과' if d.get('kinematic_probe_pass') else '제외'} | {', '.join(d.get('reasons',[])) or r.get('error','-')} |")
    lines += ['', '## 실패 분석','',
        '- sd_cmu_114_05_f01284는 donor의 팔 방향 자체는 맞췄지만 target의 어깨 폭/팔 길이 비율 때문에 가슴 앞 손 교차 조건을 충족하지 못했다.',
        '- 팔 방향 일치와 팔짱 관계 일치는 다르다. 뼈를 늘리거나 조건을 낮추지 않고 해당 조합을 제외했다.',
        '- 새 합성은 손·팔 pose ambiguity와 손가락 roll·관절 제한·arm-arm 접촉을 완전히 해결한 것이 아니다. 서로 다른 체형의 위치/contact-aware IK가 후속 과제다.','',
        '## 렌더링과 적용 제한','',
        '- 결과 PNG는 실제 합성 BVH에서 계산한 COCO17 관절 마네킹이다. AI가 그려낸 포즈 그림이나 실제 인체 FBX 메시 렌더가 아니다.',
        '- 같은 베이스 몸통 축/카메라로 base·naive·composed를 정면·사선·측면·뒤에서 확인했다.',
        '- 로컬에 Blender/bpy/mathutils 및 원본 캐릭터 FBX가 없어 실제 skinning/손목 twist/피부 접촉/CSP import까지는 미검증이다.',
        '- 성공 BVH는 재사용 가능한 연구 산출물이지만 release_allowed=false이다. 서버 동적 라우팅에 자동 합성으로 연결하지 않았다.',
        '- 합성 전 라이브러리 후보 0개를 정식 검색 성공으로 바꿔 기록하지 않았다. 합성 후보는 별도 출처와 ID를 가진다.',
        '- 원본 ATTRIBUTION 표기를 이 폴더에 보존한다. 외부 배포 전 donor/base 양쪽의 출처·파생 이용 조건을 확인해야 한다.',
        '- PR/push/운영 배포 없음. 단일 T013 결과이며 다른 러프 일반화나 작가 평가 개선률은 미측정.','',
        '## 검증','', '- 단위 테스트 21개 통과. 서버 회귀 893 passed/8 skipped/22 subtests; 스모크 54/54.',
        '- Blender 전용 tests/converter는 해당 런타임 부재로 제외. 실제 메시 검증을 대신한 수치가 아니다.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(args.attribution,out/'ATTRIBUTION.md')
    selected=[data['trials'][i] for i in (0,4) if i<len(data['trials'])]
    visual={'rough':encoded(args.images/data['rough']),'old_image':encoded(args.thumbs/'mass_daily_seated_stretch__front.jpg'),
        'donor_image':encoded(args.thumbs/(data['donor']+'__front.jpg')),'donor':data['donor'],
        'description':data['description']['search_text'],'upper_query':data['upper_query']['text'],
        'lower_query':data['lower_query']['text'],'summary':s,'cases':[]}
    for r in selected:
        visual['cases'].append({'id':r['synthetic_id'],'base':r['base'],'diagnostics':r['diagnostics'],
            'roundtrip':r.get('roundtrip'),'renders':{k:{view:encoded(file) for view,file in images.items() if view!='back' and (k!='naive' or view=='three_quarter')} for k,images in r['renders'].items()}})
    (out/'visual-data.json').write_text(json.dumps(visual,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'report':str(out/'REPORT.md'),'visual_bytes':(out/'visual-data.json').stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
