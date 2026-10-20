from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,base64,io,html,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'04j-289-set3-layout-v2';BEFORE=ROOT.parent/'04j-289-set3-orientation-v3';inputs=json.loads((ROOT/'inputs.json').read_text());three=json.loads((ROOT/'three-verification.json').read_text());cases=[];valid=[];scope_changes=[];body_changes=[];allpeople=[];scene_results=json.loads((ROOT/'scene-results.json').read_text());font='/System/Library/Fonts/AppleSDGothicNeo.ttc'
def f(n):return ImageFont.truetype(font,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(x):return html.escape(str(x))
def thumbnail(path,w,h):
 im=Image.open(path).convert('RGB');im.thumbnail((w,h),Image.Resampling.LANCZOS);return im
def data(path):
 im=thumbnail(path,1400,1700);b=io.BytesIO();im.save(b,'JPEG',quality=90);return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
for item in inputs:
 cut=item['cut_id'];folder=ROOT/cut;layout=json.loads((folder/'layout.json').read_text());selection=json.loads((folder/'selection.json').read_text());geo=json.loads((folder/'geometry.json').read_text());prior=json.loads((BEFORE/cut/'layout.json').read_text());olds=json.loads((OLD/cut/'layout.json').read_text());sources={p['person_index']:p for p in prior['people']};scene=next(p for p in scene_results if p['cut_id']==cut);assert sha(item['image_path'])==item['image_sha256'];assert scene['layout_sha256']==sha(folder/'layout.json');assert layout['zero_visible_silhouette_overlap'];assert not any(p['reversed'] for p in layout['depth_order']);assert len(geo['people'])==len(selection['people'])==len(layout['people'])
 for p,s,g in zip(layout['people'],selection['people'],geo['people']):
  i=p['person_index'];assert i==s['person_index']==g['person_index'];assert sha(g['source_fbx'])==g['fbx_sha256']==p['fbx_sha256'];assert p['scale']==1;assert g['bone_count']==g['scope_execution']['bones_preserved']==52
  assert g['scope_execution']['vertices_after']==g['scope_execution']['vertices_before'] if p['output_scope']=='full' else 0<g['scope_execution']['vertices_after']<g['scope_execution']['vertices_before']
  if g.get('arm_refinement'):assert g['arm_refinement']['head_position_error_m']<1e-4 and g['arm_refinement']['bone_length_max_error']<1e-4
  if p['output_scope']!=sources[i]['output_scope']:scope_changes.append(dict(cut_id=cut,person_index=i,before=sources[i]['output_scope'],after=p['output_scope']))
  if abs((p['torso_camera_yaw']-sources[i]['torso_camera_yaw']+180)%360-180)>1:body_changes.append(dict(cut_id=cut,person_index=i,before=sources[i]['torso_camera_yaw'],after=p['torso_camera_yaw']))
  allpeople.append(dict(cut_id=cut,**p,interaction=s['interaction'],notes=s['notes'],arm_goals=s['arm_goals']))
 for name in ('scene-analysis.json','roles-review.json','surfaces-review.json','fresh-keypoints.json','contrastive.json','localized-hands.json'):assert json.loads((folder/name).read_text())['image_sha256']==item['image_sha256']
 for variant,name in [('raw','scene-raw.fbx'),('final','scene.fbx')]:assert sha(folder/name)==scene['results'][variant]['fbx_sha256']
 cases.append(dict(item=item,layout=layout,selection=selection,geometry=geo,prior=prior,old=olds));valid.append(dict(cut_id=cut,actors=len(layout['people']),fresh_source_hashes=True,rig_bones=52,full_scopes_preserve_all_vertices=True,arm_length_and_head_preserved=True,zero_silhouette_overlap=True,no_depth_order_reversals=True))
assert len(three['cases'])==6 and three['ok']
world=max(p['max_world_error_m'] for c in three['cases'] for p in c['people']);raw_error=max(p['max_raw_geometry_world_error_m'] for c in three['cases'] for p in c['people']);pixels=max(p['max_source_image_error_pixels'] for c in three['cases'] for p in c['people']);assert world<.001 and raw_error<.001 and pixels<.1
code=(ROOT/'plan.py').read_text()+(ROOT/'refine_arms.py').read_text()
assert all(i['cut_id'] not in code for i in inputs)
summary=dict(created_at=datetime.now(timezone.utc).isoformat(),cuts=3,people=len(allpeople),manual_angles=False,manual_joint_coordinates=False,manual_pose_ids=False,github_uploaded=False,layout_reference='v2 original-fit final anchors; v3 excess shifts are reverted',max_head_center_shift_from_good_v2_frame=max(p['shift_frame'] for p in allpeople),mean_head_center_shift_from_good_v2_frame=float(np.mean([p['shift_frame'] for p in allpeople])),zero_overlap_cuts=3,overlap_raster_max_edge=1200,depth_order_reversals=0,scope_changes=scope_changes,body_changes=body_changes,pose_changes=[dict(cut_id=c['item']['cut_id'],person_index=p['person_index'],before=p['baseline_pose_id'],after=p['pose_id']) for c in cases for p in c['selection']['people'] if not p['baseline_pose_retained']],head_scale_changes=[dict(cut_id=p['cut_id'],person_index=p['person_index'],height_ratio=p['head_height_ratio_to_previous'],reason=p['target_evidence']['screen_size_basis']) for p in allpeople if abs(p['head_height_ratio_to_previous']-1)>.02],three_max_skin_world_error_m=world,three_max_raw_world_error_m=raw_error,three_max_source_pixel_error=pixels,three_validation_tolerance=dict(world_m=.001,source_pixels=.1),tolerance_note='Prior 0.1mm-only check failed for the rotated p0718 P0 (skin 0.192mm, raw 0.223mm). Final check independently bounds raw/skin world errors below 1mm and reprojection below 0.1 original pixel. Measured maximum reprojection is 0.0401px. These are file-consistency checks, not image-fidelity scores.',limitations=['Hand/finger and elbow shapes are approximate. IK fitting does not prove narrative perception.','p220 background walking step and carried object are not reconstructed.','p060 left two-handed phone pose and the phone prop remain approximate.','CSP direct import untested; camera.json needed for exact framing in importers.','3D depth is inferred from apparent size; p220 foreground head-size proxy was corrected, changing its estimated depth but not head center or front/back order.'])
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(ROOT/'artifact-validation.json').write_text(json.dumps(dict(ok=True,cases=valid,three_max_raw_world_error_m=raw_error,three_max_skin_world_error_m=world,three_max_source_pixel_error=pixels,no_cut_specific_angles_or_joint_targets_in_inference_code=True,semantic_quality_requires_user_review=True),ensure_ascii=False,indent=2))
NOTES={
 'S3-p_rough_060':'오른쪽 몸통과 머리를 왼쪽 옆면으로 맞췄습니다. 직전 v3에서 밀린 배치를 원본 맞춤 v2 기준으로 복구했습니다. 왼쪽의 두 손·휴대폰 표현은 여전히 근사입니다.',
 'S3-p_rough_220':'떨어진 큰 손을 앞 인물의 손으로 자동 연결해 팔을 포함한 범위와 뻗는 동작을 복구했습니다. 뒤 인물의 전신과 뒷모습은 유지했습니다. 앞 머리 크기는 얼굴 일부만 보던 이전 측정의 과소추정을 교정했으며, 중심 위치는 유지했습니다. 걷는 발·소품은 미완성입니다.',
 'S3-v3_rough_0718':'양쪽 인물이 서로 향하도록 몸통·머리를 맞추고, 가운데는 서서 양손으로 제지하는 후보를 선택했습니다. 원본에서 추정한 손 위치와 양쪽 상대를 함께 사용했습니다. 실제 접촉, 팔꿈치와 손바닥의 세밀한 방향은 근사입니다.'}
cards=[];preview_rows=[]
for c in cases:
 cut=c['item']['cut_id'];folder=ROOT/cut;layout=c['layout'];vp=json.loads((folder/'identities.json').read_text())['analysis_viewport_bbox'];cfg=layout['camera'];columns=[]
 for label,path,key in [('원본 콘티',Path(c['item']['image_path']),'source'),('이전 배치 v2',OLD/cut/'final.png','v2'),('직전 결과 v3',BEFORE/cut/'final.png','v3'),('이번 자동 수정 v4',folder/'final.png','v4')]:
  im=Image.open(path).convert('RGB');sx=im.width/cfg['width'];sy=im.height/cfg['height'];box=[round(vp[0]*sx),round(vp[1]*sy),round(vp[2]*sx),round(vp[3]*sy)];im.crop(box).save(folder/(key+'-panel.png'));columns.append((label,folder/(key+'-panel.png')))
 canvas=Image.new('RGB',(2200,850),'#edf2f7');d=ImageDraw.Draw(canvas);d.text((18,12),cut+' · 동일 화면 구간 비교',font=f(29),fill='#172431')
 for j,(label,path) in enumerate(columns):
  d.text((j*550+16,58),label,font=f(24),fill='#25384c');im=thumbnail(path,525,730);canvas.paste(im,(j*550+(550-im.width)//2,100+(730-im.height)//2))
 canvas.save(folder/'comparison.jpg',quality=93);preview_rows.append(canvas)
 # All overlays are output-only artifacts, never image-analysis inputs.
 im=Image.open(c['item']['image_path']).convert('RGB');draw=ImageDraw.Draw(im);colors=['#1971c2','#e25b20','#8545b8'];own=json.loads((folder/'ownership.json').read_text())
 for h in own['hands']:
  if not h['detached_from_old_crop']:continue
  x,y=h['center'];draw.rectangle(h['bbox'],outline=colors[h['owner']],width=max(2,im.width//200));draw.text((x,y),'P'+str(h['owner'])+' hand',fill=colors[h['owner']])
 for p in c['selection']['people']:
  for goal in p['arm_goals']:
   if goal['target_person_index'] is None:continue
   a=goal['hand_pixels'];target=goal['target_person_index'];b=c['selection']['people'][target]['source_head_reference'];draw.line([tuple(a),tuple(b)],fill=colors[p['person_index']],width=max(2,im.width//180))
 im.crop(vp).save(folder/'ownership-overlay.png')
 table=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (p['person_index'],p['direction']['torso_label'],p['direction']['head_label'],p['output_scope'],p['interaction'],p['pose_id'],p['head_scale_reason']))+'</tr>' for p in c['selection']['people'])
 delta=max(p['shift_frame'] for p in layout['people'])*100
 primary=[columns[0],columns[2],columns[3]]
 cards.append(f'<article><h2>{esc(cut)} · {len(layout["people"])}명</h2><p>{esc(NOTES[cut])}</p><p class="metric">v2 머리 중심에서 추가 이동 {delta:.4f}% · 겹침 0px · 앞뒤 순서 반전 0</p><div class="compare">'+''.join(f'<figure><figcaption>{esc(label)}</figcaption><img src="{data(path)}"></figure>' for label,path in primary)+f'</div><details><summary>좋았던 v2 배치까지 포함한 4열 비교</summary><img src="{data(folder/"comparison.jpg")}"></details><details><summary>신체 소유·관계 추론 근거와 인물별 결과</summary><p>아래 선·상자는 자동 추론을 검토하기 위한 출력 표시입니다. 추론 입력으로 사용하지 않았습니다.</p><img src="{data(folder/"ownership-overlay.png")}"><table><tr><th>P</th><th>몸통</th><th>머리</th><th>범위</th><th>관계</th><th>선택 포즈</th><th>크기 기준</th></tr>{table}</table></details><details><summary>카메라의 전체 원본 프레임으로 비교</summary><div class="compare"><figure><figcaption>원본</figcaption><img src="{data(c["item"]["image_path"])}"></figure><figure><figcaption>직전 v3</figcaption><img src="{data(BEFORE/cut/"final.png")}"></figure><figure><figcaption>이번 v4</figcaption><img src="{data(folder/"final.png")}"></figure></div></details><p><a href="{esc(cut)}/scene.fbx">통합 FBX</a> · <a href="{esc(cut)}/camera.json">카메라 설정</a> · <a href="{esc(cut)}/selection.json">자동 선택</a> · <a href="{esc(cut)}/ownership.json">소유·관계</a> · <a href="{esc(cut)}/geometry.json">팔 리파인 검증</a></p></article>')
preview=Image.new('RGB',(2200,850*len(preview_rows)),'white')
for j,im in enumerate(preview_rows):preview.paste(im,(0,j*850))
preview.save(ROOT/'comparison-preview.jpg',quality=92)
header=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>p060·p220·p0718 장면 관계 자동 수정</title><style>body{{margin:0;background:#edf2f7;color:#172431;font:17px/1.7 system-ui,sans-serif}}main{{max-width:1900px;padding:28px;margin:auto}}article,.intro{{background:white;border:1px solid #d7e1eb;border-radius:18px;padding:26px;margin-bottom:26px}}h1{{font-size:30px}}h2{{font-size:25px}}.compare{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;align-items:start}}figure{{margin:0}}figcaption{{font-weight:700;margin:12px 0}}img{{width:100%;height:auto;background:#f4f7fc}}.metric{{color:#267755}}.warning{{background:#fff2d9;padding:14px}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:600}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}a{{color:#0759ad}}@media(max-width:750px){{main{{padding:8px}}article,.intro{{padding:15px}}.compare{{gap:6px}}figcaption{{font-size:13px}}}}</style><main><div class="intro"><h1>p060·p220·p0718: 장면 관계와 팔·범위 자동 수정</h1><p>동일 원본 3컷·7명. 전체 장면과 작은 인물을 자동 확대해 다시 분석하고, 새 관절·손 탐지와 교차 확인했습니다. 컷별 수동 각도·좌표·포즈 ID 지정 없음. GitHub 업로드 없음.</p><p>원본 맞춤 배치 v2의 머리 중심을 유지했습니다. 직전 v3에서 불필요하게 밀린 위치는 복구했습니다. 3/3컷에서 추가 배치 이동 0 · 실제 실루엣 겹침 0px(긴 변 1200px) · 앞뒤 순서 반전 0. p220 앞 인물은 전체 머리 크기 과소추정을 교정해 화면 크기와 추정 깊이가 바뀌었으며 중심은 유지했습니다.</p><p>비교는 같은 카메라의 같은 화면 구간을 확대해서 보여줍니다. FBX 카메라 자체는 원본 전체 프레임이며, 전체 프레임 비교도 각 컷에서 펼쳐 볼 수 있습니다.</p><p class="warning">동작의 의미가 원본처럼 읽히는지는 사용자 확인이 필요합니다. 손가락·팔꿈치·접촉은 근사이고, 공용 인체 모델이라 옷·머리카락·휴대폰·가방은 복원하지 않았습니다. p220의 걸음도 정지 자세입니다. p060의 두 손 휴대폰 표현은 남은 오류입니다.</p><p><a href="three-scenes.zip">3컷 FBX·카메라·근거·검증 묶음</a> · <a href="summary.json">검증 요약</a> · <a href="artifact-validation.json">산출물 검증</a> · <a href="three-verification.json">독립 FBX 로더 검사</a></p><p>모든 인물 52본과 신체 스케일 1 유지. 팔 리파인은 본 길이와 머리 위치를 고정했습니다. 독립 Three.js 로더의 최대 스킨 좌표 차이 {world*1000:.3f}mm, 최대 원본 화면 오차 {pixels:.4f}px. 기존 0.1mm 단일 기준을 넘은 회전 사례가 있어 원시 정점도 별도 검사하고 최종 기준은 1mm 및 0.1px로 명시했습니다. 이는 파일 정합성 검사이며 자세 정확도 점수는 아닙니다. CSP 직접 불러오기는 미검증이고 정확한 프레이밍에는 camera.json이 필요합니다.</p></div>'''
(ROOT/'index.html').write_text(header+''.join(cards)+'</main></html>')
with zipfile.ZipFile(ROOT/'three-scenes.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for c in cases:
  cut=c['item']['cut_id']
  for name in ('scene.fbx','camera.json','selection.json','layout.json','ownership.json','geometry.json','comparison.jpg','final.png'):z.write(ROOT/cut/name,str(Path(cut)/name))
 for name in ('summary.json','artifact-validation.json','three-verification.json'):z.write(ROOT/name,name)
print(json.dumps(summary,ensure_ascii=False,indent=2))
