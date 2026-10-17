from pathlib import Path
from datetime import datetime,timezone
import json,base64,io,html,zipfile,hashlib,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'04j-289-set3-layout-v2';BASE=ROOT.parent/'04j-289-set34-refine-scenes'
inputs=json.loads((ROOT/'inputs.json').read_text());cases=[json.loads((ROOT/i['cut_id']/'layout.json').read_text()) for i in inputs];rows=[p for c in cases for p in c['people']];scenes=json.loads((ROOT/'scene-results.json').read_text());verification=json.loads((ROOT/'three-verification.json').read_text());font='/System/Library/Fonts/AppleSDGothicNeo.ttc'
def f(n):return ImageFont.truetype(font,n)
def img(path,w,h):
 im=Image.open(path).convert('RGB');im.thumbnail((w,h),Image.Resampling.LANCZOS);return im
def data(path):
 im=img(path,1100,1600);b=io.BytesIO();im.save(b,'JPEG',quality=88);return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def e(x):return html.escape(str(x))
scope_changes=[];posture_changes=[];items=[]
for c in cases:
 old={p['person_index']:p for p in json.loads((OLD/c['cut_id']/'layout.json').read_text())['people']};sel=json.loads((ROOT/c['cut_id']/'selection.json').read_text())['people'];prior={p['person_index']:p for p in json.loads((BASE/c['cut_id']/'selection.json').read_text())['people']}
 for p,s in zip(c['people'],sel):
  i=p['person_index']
  if p['output_scope']!=old[i]['output_scope']:scope_changes.append(dict(cut_id=c['cut_id'],person_index=i,before=old[i]['output_scope'],after=p['output_scope']))
  if s['pose_id']!=prior[i]['pose_id']:posture_changes.append(dict(cut_id=c['cut_id'],person_index=i,before=prior[i]['pose_id'],after=s['pose_id']))
  items.append(dict(cut_id=c['cut_id'],person_index=i,torso=s['direction']['torso_label'],head=s['direction']['head_label'],scope=s['output_scope'],posture=s['posture'],notes=s['direction']['notes']+s['notes']))
summary=dict(created_at=datetime.now(timezone.utc).isoformat(),cuts=len(cases),people=len(rows),fresh_local_analysis=True,manual_angles=False,manual_coordinates=False,github_uploaded=False,scope_changes=scope_changes,pose_changes=posture_changes,final_zero_overlap_cuts=sum(c['zero_visible_silhouette_overlap'] for c in cases),mean_anchor_shift_frame=float(np.mean([p['shift_frame'] for p in rows])),max_anchor_shift_frame=max(p['shift_frame'] for p in rows),depth_order_reversals=sum(p['reversed'] for c in cases for p in c['depth_order']),max_head_height_change_unchanged_scope=max(abs(p['head_height_ratio_to_previous']-1) for c in cases for p in c['people'] if not any(s['cut_id']==c['cut_id'] and s['person_index']==p['person_index'] for s in scope_changes)),three_max_world_error_m=max(p['max_world_error_m'] for c in verification['cases'] for p in c['people']),camera_max_projection_error_pixels=max(v['verification']['camera_projection_max_error_source_pixels'] for c in scenes for v in c['results'].values()),common_body_model=True,limitations=['Frozen 34-person census retains pre-existing missing foreground hand/arm actor in p_rough_128.','Eight-bin camera-relative orientation remains estimated; unknown head follows torso as a flagged reference.','Full output is extrapolated from library legs when source feet are clipped.','Metric depth inferred from apparent size; common body proportions and clothing are not reconstructed.','Synthetic skeleton/image loss is not a human-rated accuracy score.'])
size_changes=[]
for c in cases:
 old={p['person_index']:p for p in json.loads((OLD/c['cut_id']/'layout.json').read_text())['people']}
 for p in c['people']:
  prior=old[p['person_index']]
  if p['output_scope']!=prior['output_scope']:continue
  ratio=(p['final_body_bbox'][3]-p['final_body_bbox'][1])/(prior['final_body_bbox'][3]-prior['final_body_bbox'][1]) if p['output_scope']=='full' else p['head_height_ratio_to_previous'];size_changes.append(abs(ratio-1))
summary['max_screen_size_proxy_change_unchanged_scope']=max(size_changes);summary['cuts_without_additional_anchor_move']=sum(max(p['shift_frame'] for p in c['people'])<1e-8 for c in cases);summary['layout_exception']=max(((c['cut_id'],max(p['shift_frame'] for p in c['people'])) for c in cases),key=lambda x:x[1]);summary['remaining_pose_errors']='p_rough_038 right palm still has a visible cheek gap; fine hand contact is not solved. Reclassified output ranges outside p_rough_066 still require visual review.'
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(ROOT/'per-person-results.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
priority=['S3-p_rough_038','S3-p_rough_139','S3-p_rough_066','S3-p_rough_047'];ordered=sorted(cases,key=lambda c:(priority.index(c['cut_id']) if c['cut_id'] in priority else 100,c['cut_id']));cards=[]
for c in ordered:
 cut=c['cut_id'];folder=ROOT/cut;columns=[('원본 콘티',Path(c['image_path'])),('이전 자동 결과',OLD/cut/'final.png'),('이번 자동 결과',folder/'final.png')];canvas=Image.new('RGB',(1650,860),'#edf2f7');d=ImageDraw.Draw(canvas);d.text((18,12),cut+' · 원본 / 이전 / 수정 후',font=f(27),fill='#172431')
 for j,(label,path) in enumerate(columns):
  d.text((j*550+18,55),label,font=f(24),fill='#263648');im=img(path,520,740);canvas.paste(im,(j*550+(550-im.width)//2,100+(740-im.height)//2))
 canvas.save(folder/'comparison.jpg',quality=92)
 table=''.join('<tr>'+''.join('<td>'+e(v)+'</td>' for v in (p['person_index'],p['torso'],p['head'],p['scope'],p['posture'],'; '.join(p['notes'])))+'</tr>' for p in items if p['cut_id']==cut)
 maxmove=max(p['shift_frame'] for p in c['people'])*100;over=sum(p['pixels'] for p in c['final_overlap_pairs']);extra={'S3-p_rough_128':'기존 인물 목록의 전경 손·팔 1명 누락이 남아 있습니다.','S3-p_rough_038':'손을 볼 가까이 고르는 데는 개선이 있으나 오른손과 볼의 접촉 간격이 남습니다.','S3-p_rough_066':'배경 전신 뒷모습은 출력됐습니다. 머리 방향은 미상이라 몸통을 따르는 참고 방향이고, 여성 체형·원본 기울기는 충분히 재현되지 않았습니다.'}.get(cut,'')
 cards.append(f'<article id="{e(cut)}"><h2>{e(cut)} · {len(c["people"])}명</h2><p>기존 머리 위치에서 최대 이동 {maxmove:.2f}% · 최종 실루엣 겹침 {over:,}px (긴 변 1200px 검사). {e(extra)}</p><div class="compare">'+''.join(f'<figure><figcaption>{e(label)}</figcaption><img src="{data(path)}"></figure>' for label,path in columns)+f'</div><details><summary>인물별 자동 판단·불확실성</summary><table><tr><th>P</th><th>몸통</th><th>머리</th><th>범위</th><th>기본 자세</th><th>참고</th></tr>{table}</table></details><p><a href="{e(cut)}/scene.fbx">통합 FBX</a> · <a href="{e(cut)}/camera.json">카메라 설정</a> · <a href="{e(cut)}/selection.json">자동 선택 근거</a> · <a href="{e(cut)}/layout.json">배치 변화</a></p></article>')
preview=Image.new('RGB',(1650,860*4),'white')
for j,cut in enumerate(priority):preview.paste(Image.open(ROOT/cut/'comparison.jpg'),(0,j*860))
preview.save(ROOT/'comparison-preview.jpg',quality=90)
with zipfile.ZipFile(ROOT/'set3-automatic-scenes.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for c in cases:
  folder=ROOT/c['cut_id']
  for name in ('scene.fbx','camera.json','selection.json','layout.json','final.png','comparison.jpg'):z.write(folder/name,str(Path(c['cut_id'])/name))
 for name in ('summary.json','per-person-results.json','three-verification.json'):z.write(ROOT/name,name)
head=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>set3 몸통·머리·범위·포즈 자동 수정 비교</title><style>body{{margin:0;background:#edf2f7;color:#172431;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:28px}}article,.intro{{background:white;border:1px solid #d7e1eb;border-radius:18px;padding:26px;margin:24px 0}}h1{{font-size:30px}}h2{{font-size:24px}}.compare{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}figure{{margin:0}}figcaption{{font-weight:700;margin-bottom:12px}}img{{width:100%;height:auto;background:#f5f7fb}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}.warning{{background:#fff3dc;padding:14px}}a{{color:#0759ad}}@media(max-width:750px){{main{{padding:10px}}article{{padding:14px}}.compare{{gap:6px}}figcaption{{font-size:13px}}}}</style><main><div class="intro"><h1>set3: 몸통·머리 방향, 출력 범위와 기본 포즈 자동 수정</h1><p>같은 {len(cases)}컷·{len(rows)}명. 원본을 로컬 모델로 다시 분석하고 몸통과 머리를 따로 추정했습니다. 새 관측에서 기본 구조가 맞는 기존 자동 포즈는 유지하고, 보이는 무릎·골반·몸통 구조가 틀린 경우에만 교체했습니다. 이전 방향·범위·분석 문장은 재사용하지 않았습니다. 기존 배치를 기준으로 자동 재배치했습니다. 컷별 수동 각도·좌표·포즈 지정 없음. GitHub 업로드 없음.</p><p>출력 범위 변경 {len(scope_changes)}명 · 기본 구조를 통과해 기존 자동 포즈 유지 {len(rows)-len(posture_changes)}명 · 포즈 교체 {len(posture_changes)}명 · 겹침 0인 컷 {summary['final_zero_overlap_cuts']}/13 · 머리 위치 평균 이동 {summary['mean_anchor_shift_frame']*100:.2f}% / 최대 {summary['max_anchor_shift_frame']*100:.2f}% · 앞뒤 순서 반전 {summary['depth_order_reversals']}쌍.</p><p>추가 위치 이동이 없는 컷 {summary['cuts_without_additional_anchor_move']}/13. 범위가 같은 인물의 화면 크기 기준값 변화는 최대 {summary['max_screen_size_proxy_change_unchanged_scope']*100:.2f}%입니다. 가장 많이 이동한 컷은 {e(summary['layout_exception'][0])}입니다.</p><p>이동률은 이전 머리 중심과 비교한 화면 가로·세로 정규화 거리입니다. 화면 크기는 전신이면 몸 높이, 부분 인체면 머리 높이를 기준으로 유지합니다. 전신 범위가 바로잡힌 인물은 원본의 보이는 몸 높이로 다시 맞췄습니다.</p><p class="warning">자동 추정이므로 방향·자세가 모두 원본과 일치한다고 판정하지 않았습니다. 미상인 머리는 몸통을 따라가는 참고 방향이며 근거에 표시했습니다. 공용 인체 모델을 사용해 여성 체형·옷·머리카락은 재현하지 않았습니다. p_rough_128의 기존 전경 손·팔 인물 누락도 남아 있습니다.</p><p><a href="set3-automatic-scenes.zip">13컷 FBX·카메라·검증 묶음</a> · <a href="summary.json">검증 요약</a> · <a href="three-verification.json">실제 FBX 재불러오기 검사</a></p><p>52본 유지 및 모든 스킨 정점의 원본→통합 FBX 좌표를 확인했습니다. Three.js 최대 오차 {summary['three_max_world_error_m']:.8f}m. Blender FBX 카메라는 재불러올 때 HORIZONTAL sensor fit을 camera.json대로 복원해야 하며, Three.js도 정확한 프레이밍에는 camera.json이 필요합니다.</p></div>'''
(ROOT/'index.html').write_text(head+''.join(cards)+'</main></html>')
print(json.dumps(summary,ensure_ascii=False,indent=2))
