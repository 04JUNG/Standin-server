"""Render the final, evidence-backed local staging handoff."""
import json,html,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'artifacts/shape-staging'
rows=sorted([json.loads(p.read_text()) for p in (R/'api-qa').glob('*.json') if not p.name.startswith('summary')],key=lambda x:x['id']);assert len(rows)==103 and all(q['ok'] for q in rows)
runtime=json.loads((R/'runtime-qa/summary.json').read_text());upload=json.loads((R/'upload-verification.json').read_text());release=json.loads((R/'release.json').read_text())
source=json.loads((R/'source-hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in source.items())
(R/'source-preservation.json').write_text(json.dumps({'all_originals_unchanged':True,'files':source},indent=2))
selected=[79,80,75,76,99,100,101,102];canvas=Image.new('RGB',(1024,568),'#171b22');draw=ImageDraw.Draw(canvas)
for i,num in enumerate(selected):
 q=next(q for q in rows if int(q['id'].split('-',1)[0])==num);p=R/'api-qa'/(q['id']+'.png');canvas.paste(Image.open(p).convert('RGB').resize((256,256)),((i%4)*256,(i//4)*284));draw.text(((i%4)*256+4,(i//4)*284+261),q['id'].replace('-v0-6-7','').replace('-v0-2-2',''),fill='white')
canvas.save(R/'POSE_REVIEW.png')
models=[]
for cid in json.loads((R/'release/body-models.local.json').read_text())['assets']:
 ident=cid['characterId'];tests=[q for q in rows if q['character_id']==ident]
 models.append(f'<tr><td>{html.escape(cid["displayName"])}</td><td>{len(tests)}</td><td>{max(q["shape_bake"]["roundtrip_max_error_mm"] for q in tests):.6f} mm</td><td><a href="release/png/{ident}.png">T 포즈</a> · <a href="release/fbx/{ident}.fbx">중립 FBX</a></td></tr>')
pose_rows=[]
for q in rows:
 ext='glb' if q['view']=='model' else 'png';name=html.escape(q['id']);pose_rows.append(f'<tr><td>{name}</td><td>{q["seconds"]:.1f}s</td><td><a href="api-qa/{name}.{ext}">미리보기</a> · <a href="api-qa/{name}.fbx">같은 최종 FBX</a> · <a href="api-qa/{name}.json">검증</a></td></tr>')
max_surface=max(q['shape_bake']['roundtrip_max_error_mm'] for q in rows);max_joint=max(q['shape_bake']['roundtrip_joint_error_mm'] for q in rows);max_transfer=max(q['shape_bake']['pose_transfer_max_joint_error_mm'] for q in rows)
summary={'api_cases':103,'api_passed':103,'presets':19,'live_runtime_max_error_mm':runtime['max_error_mm'],'max_fbx_roundtrip_surface_error_mm':max_surface,'max_fbx_roundtrip_joint_error_mm':max_joint,'max_pose_transfer_joint_error_mm':max_transfer,'all_original_sources_unchanged':True,'s3':upload,'ecs_service_changed':False,'runtime_platform':'macOS arm64 / Blender 5.2.0 fbe6228777e7; Linux container not executed locally','converter_contracts_passed':177,'bff_tests_passed':363,'bff_tests_skipped':2,'client_tests_passed':503,'neutral_preview_pose':'t_pose'}
(R/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>체형 수치 굽기 · S3 스테이징 검증</title><style>body{{font:16px/1.65 system-ui;background:#10151d;color:#e8eef6;max-width:1160px;margin:40px auto;padding:0 24px}}a{{color:#9fdcc6}}h1{{font-size:30px}}h2{{margin-top:40px}}.stats{{display:flex;gap:14px;flex-wrap:wrap}}.stat{{background:#1d2836;padding:18px;border-radius:12px;min-width:180px}}strong{{color:#b6e2ce}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #33404e}}img{{width:100%;border-radius:12px}}code,pre{{background:#1d2836;overflow-wrap:anywhere;white-space:pre-wrap}}small{{color:#a9b8c9}}</style>
<h1>체형 수치 굽기 · S3 스테이징 검증</h1><p>2026-10-15 · <strong>남성 0.6.7 / 여성 0.2.2 원본 유지</strong></p>
<div class="stats"><div class="stat">실제 API 검증<br><strong>103 / 103 통과</strong></div><div class="stat">앱 ↔ 중립 FBX<br><strong>19개 · 최대 {runtime['max_error_mm']:.4f} mm</strong></div><div class="stat">S3 재다운로드 검증<br><strong>{upload['objects_verified']}개 일치</strong></div></div>
<h2>기본 포즈 확정</h2><p>사용자 결정에 따라 기본 사진·체형 비교·실시간 체형 모델은 T 포즈를 사용한다. 차렷 이미지 릴리스는 보존하고 새 T 포즈 릴리스를 게시했다.</p><h2>완료한 연결</h2><p>체형 수치 → 확정 컨트롤러의 중립 메시·골격 굽기 → 기존 V3.2.5 BVH 변환 → 확정 자세·접촉 보정 → 최종 FBX → 그 FBX의 PNG / GLB.</p><p>실제 BFF 연동에서 서로 다른 수치의 PNG·FBX 4쌍, GLB·FBX 2쌍과 중복 요청 병합을 확인했다. 수치가 바뀐 Top-K는 정적 라이브러리 GLB를 재사용하지 않고 PNG로 폴백하며 최종 검토 GLB는 실제 결과 FBX에서 생성한다.</p>
<h2>실제 앉기·깊은 굽힘</h2><p>BVH 관절각으로 확인: Lawyer_02_00008은 무릎 약 90도, Finding_01은 몸통 약 78도 기울기. 모든 19개 프리셋에 두 실제 포즈를 적용했다.</p><img src="POSE_REVIEW.png" alt="남녀 극단 체형의 실제 앉기와 굽힘 검수"/>
<h2>S3 게시</h2><pre>s3://{release['bucket']}/{release['manifestKey']}</pre><p>파일 크기·SHA-256을 업로드 후 전체 재다운로드로 확인했다. 기존 prefix·프로덕션·ECS 서비스는 변경하지 않았다. <a href="release.json">릴리스 정보</a> · <a href="upload-verification.json">업로드 검증</a> · <a href="source-preservation.json">원본 보존</a> · <a href="bff-real-pair.json">실제 BFF 검증</a></p>
<h2>연결 시 필요한 코드</h2><p>converter: <code>{ROOT}</code><br>BFF: <code>/Users/dowon/dev/Standin-app-server-shape-bake-staging</code><br>인프라: <code>/Users/dowon/dev/Standin-infra-shape-bake-staging</code><br>클라이언트: <code>/Users/dowon/dev/Standin-client-shape-bake-staging</code></p><p>일치하는 converter·BFF·클라이언트 코드를 배포한 뒤 인프라의 <code>stagingShapeAssetsEnabled=true</code>로 새 릴리스를 선택한다. 기본은 기존 릴리스이며 자동 체형 검색용 카탈로그는 draft다. 지원 BVH 5종은 명시적으로 제한했고 전체 라이브러리 검수나 무관통 판정을 뜻하지 않는다.</p>
<h2>검증 범위</h2><p>converter 계약 테스트 177개, BFF 363개(2개 skip), 클라이언트 503개 통과. 클라이언트 테스트는 Node 26 내장 webstorage를 끈 환경에서 실행했다. 세 TypeScript 프로젝트 타입 검사와 스테이징 CDK 합성을 통과했다.</p><p>실제 BVH 5종 × 19프리셋 95개와 중간값·최댓값·GLB·후면 8개. FBX 표면 왕복 최대 {max_surface:.6f} mm, 관절 왕복 {max_joint:.6f} mm, solver→원본 관절 전달 {max_transfer:.6f} mm. 컨트롤러·원본 blend 해시는 모두 보존했다.</p><p><small>실측 환경: macOS arm64, Blender 5.2.0 fbe6228777e7. 서버 호출은 테스트 병렬 실행 시간이며 운영 SLA가 아니다. Linux 컨테이너와 CSP 앱 실행 검증은 이번 로컬 검사에 포함되지 않는다. 기존 확정 모델의 자세 접촉 한계는 그대로이며 조형·웨이트·비율은 변경하지 않았다.</small></p>
<h2>프리셋</h2><table><tr><th>프리셋</th><th>검사 수</th><th>FBX 표면 왕복 오차</th><th>자산</th></tr>{''.join(models)}</table>
<details><summary>103개 결과 펼치기</summary><table>{''.join(pose_rows)}</table></details></html>'''
(R/'REPORT.html').write_text(page)
print(json.dumps(summary,ensure_ascii=False,indent=2))
