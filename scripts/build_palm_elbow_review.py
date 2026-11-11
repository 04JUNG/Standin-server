#!/usr/bin/env python3
"""Evidence view for a safe elbow candidate that failed broad palm adherence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib,html,shutil
import numpy as np
from converter.palm_waist_contact import anatomical_axes


def main():
    p=argparse.ArgumentParser()
    for k in ('run','proposal','review-dir','rough','analysis','baseline-plan','observation-image'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=a.run;o=a.review_dir;o.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit=load(r/'palm-elbow-audit.json');qa=load(r/'qa/qa-report.json');plan=load(a.proposal/'planning.json');vlm=load(o/'gemini-review.json');joints=load(o/'actual-joints.json');metrics=load(o/'metrics.json')
    if audit['full_request_passed']:raise ValueError('review_template_requires_blocked_full_palm')
    from PIL import Image
    same_pixels=np.array_equal(np.array(Image.open(a.rough).convert('RGB')),np.array(Image.open(a.observation_image).convert('RGB')))
    if not same_pixels:raise ValueError('review_rough_pixels_differ_from_observations')
    if sha(r/'qa/final.fbx')!=audit['artifact_sha256']:raise ValueError('review_artifact_mismatch')
    for row,key in zip(joints,('baseline_sha256','artifact_sha256')):
        if row['artifact_sha256']!=audit[key]:raise ValueError('review_render_mismatch')
    for row in vlm['images']:
        path=a.rough if row['name']==a.rough.name else o/row['name']
        if sha(path)!=row['sha256']:raise ValueError('review_vlm_image_mismatch')
    for src,name in ((r/'qa/final.fbx','safe-elbow-candidate.fbx'),(r/'qa/qa-report.json','qa-report.json'),(r/'palm-elbow-audit.json','palm-elbow-audit.json'),(a.proposal/'contact-contract.json','palm-contract.json'),(a.rough,'rough.png')):shutil.copyfile(src,o/name)
    alternatives=[]
    for label in ('hand-on-hip','saxophone'):
        path=Path('/private/tmp')/('refine-'+label+'-source-check-20261024')
        alternatives.append(dict(diagnostic=load(path/'candidate-diagnostic.json'),failure=load(path/'qa-failure.json')))
    # Anatomical sagittal plot: same axes and shoulder origin for both meshes.
    before=np.array(joints[0]['joints_world']);after=np.array(joints[1]['joints_world'])
    with np.load(a.baseline_plan,allow_pickle=False) as z:_,lateral,up,front,_=anatomical_axes(z['body_bones'],list(z['names']))
    back=-front
    origin=before[5]
    def xy(v):return 220+float((v-origin)@back)*650,70-float((v-origin)@up)*650
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 410"><rect width="420" height="410" fill="#162338"/><path d="M220 35V370" stroke="#526780" stroke-dasharray="5 5"/><text x="228" y="28" fill="#afbfdb" font-size="12">어깨 기준면</text><text x="45" y="392" fill="#afbfdb">앞 ←</text><text x="305" y="392" fill="#afbfdb">→ 뒤</text>'
    for values,color,label in ((before,'#96abc8','기존'),(after,'#70e1b6','후보')):
        pts=[xy(values[i]) for i in (5,7,9)];svg+='<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="4"/>'
        for x,y in pts:svg+=f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>'
        x,y=pts[1];svg+=f'<text x="{x+9:.2f}" y="{y-9:.2f}" fill="{color}" font-size="13">{label} 팔꿈치</text>'
    svg+='</svg>';(o/'elbow-depth.svg').write_text(svg)
    z_before=float((before[7]-origin)@back*1000);z_after=float((after[7]-origin)@back*1000)
    trace=dict(schema='palm-elbow-review-v1',full_request_passed=audit['full_request_passed'],safe_pose_candidate=audit['safe_pose_candidate'],
        audit=audit,qa_mesh=qa['mesh_audit'],pose_safety=qa['pose_safety'],planning=plan,metrics=metrics,vlm=vlm,
        vlm_attempts=[dict(outcome='read_timeout',timeout_seconds=45),dict(outcome='response',timeout_seconds=90)],
        alternatives=alternatives,anatomical_elbow_coordinate_mm=dict(before=z_before,after=z_after),
        source_image_sha256=sha(a.rough),analysis_sha256=sha(a.analysis),attached_rough_pixels_identical_to_analysis_image=True,
        branch='codex/refine-palm-elbow-20261024',baseline_code='814ea9730',previous_experiment_code='64d18960b',deployed=False)
    (o/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))
    q=audit['quality'];c=audit['after'];vr=vlm['response']
    quality_rows=''.join(f"<tr><td>{name}</td><td>{row['before']:.8f}</td><td>{row['after']:.8f}</td><td>비악화 통과</td></tr>" for name,row in q['metrics'].items())
    zone_rows=''.join(f"<tr><td>{z['zone']+1}</td><td>{z['area_mm2']:.1f} mm²</td><td>{z['median_gap_mm']:.1f} mm</td><td>{z['contact_fraction']*100:.0f}%</td></tr>" for z in c['zones'])
    reasons=''.join('<li>'+html.escape(x)+'</li>' for x in vr['reasons'])
    alt=''.join(f"<tr><td>{html.escape(x['diagnostic']['candidate'])}</td><td>{x['diagnostic']['quality']['metrics']['body']['after']:.6f}</td><td>{html.escape(x['failure']['reason'])}</td></tr>" for x in alternatives)
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손바닥·팔꿈치 재검증</title><style>*{box-sizing:border-box}body{margin:0;background:#0d1422;color:#e5edf9;font:16px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px 22px 65px}h1{font-size:34px;margin:12px 0}h2{font-size:22px;margin-top:0}p,small{color:#aebfd7}a{color:#a3c7fa}nav{display:flex;flex-wrap:wrap;gap:18px}.notice{padding:18px 22px;border-left:4px solid #eea576;background:#30251f;margin:24px 0}.bad{color:#ffbd9a}.good{color:#80e0ba}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card,section{background:#162135;border:1px solid #2b3b55;border-radius:12px;padding:21px}.card b{display:block;font-size:25px}.card small{font-size:12px}section{margin:25px 0}button{background:#263b5b;border:1px solid #536b8e;border-radius:7px;padding:9px 14px;color:white;cursor:pointer}button[aria-pressed=true]{background:#315f91}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:15px}.images{display:grid;grid-template-columns:150px 1fr 1fr;gap:14px}.images img{width:100%;display:block;border-radius:7px}.label{margin:8px 0;font-size:14px}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #33455e;text-align:left;padding:10px}code{font-size:12px;word-break:break-all;color:#b5d2f9}li{margin:9px 0;color:#bdcde4}.plot{width:100%;max-height:370px}.scroll{overflow:auto}@media(max-width:750px){main{padding:18px 13px}.stats{grid-template-columns:1fr}.images{grid-template-columns:1fr 1fr}.rough{grid-column:1/-1}.rough img{height:155px;object-fit:contain;background:white}.two{grid-template-columns:1fr}h1{font-size:28px}}
</style><main><nav><a href="../#depth">기존 안전 버전</a><a href="../surface-contact/">이전 엄지 실험</a><a href="safe-elbow-candidate.fbx" download>팔꿈치 후보 FBX</a><a href="trace.json">전체 실행 근거</a></nav><small>FULL PALM / ELBOW · 실제 러프 재검증</small><h1>팔꿈치는 뒤로 이동했지만,<br>손바닥 전체 밀착은 미달성</h1><div class="notice"><b class="bad">요청 전체 통과: false</b><br>기존 안전 기준과 2D 포즈 비악화 조건을 지킨 후보입니다. 손바닥 밀착을 통과한 완성품으로 채택하지 않았습니다. Gemini도 눈에 띄는 팔꿈치 개선과 옆허리 밀착을 확인하지 못했습니다.</div><div class="stats"><div class="card"><small>실제 FBX · 몸통 기준 팔꿈치 후방 이동</small><b class="good">@@REAR@@ mm</b></div><div class="card"><small>동일 카메라 2D 관절 NME · 낮을수록 좋음</small><b>@@NME0@@ → @@NME1@@</b><small>약 @@GAIN@@% 감소. 실질적 유사도 향상을 주장하기에는 작음.</small></div><div class="card"><small>손바닥 전체의 접촉 면적 비율</small><b class="bad">0% · 실패</b><small>엄지의 작은 면으로 대신 통과시키지 않음.</small></div></div>
<section><h2>01 · 첨부 러프와 실제 FBX 비교</h2><div class="toolbar"><button data-view="side" aria-pressed="true">전신 측면</button><button data-view="front" aria-pressed="false">전신 정면</button><button data-view="three_quarter" aria-pressed="false">전신 사선</button><button data-view="contact-three_quarter" aria-pressed="false">손바닥 확대</button></div><div class="images"><div class="rough"><div class="label">사용자가 첨부한 러프</div><img src="rough.png" alt="첨부한 러프"></div><div><div class="label">기존 깊이·안전 버전</div><img id="before" src="before-side.png" alt="기존 실제 FBX"></div><div><div class="label">팔꿈치 후방 이동 후보</div><img id="after" src="after-side.png" alt="팔꿈치 개선 후보 실제 FBX"></div></div><p id="view-label">전신 측면</p><small>동일 시점으로 렌더했습니다. 확대 이미지의 주황색은 고정한 손바닥 검사 영역이며, 안쪽 면은 가려질 수 있습니다. 첨부 러프의 픽셀은 기존 관측 데이터의 원본 이미지와 동일함을 확인했습니다.</small></section>
<section><h2>02 · 팔꿈치 관절점을 직접 사용한 제약</h2><div class="two"><div><img class="plot" src="elbow-depth.svg" alt="실제 관절 좌표로 그린 팔꿈치 앞뒤 변화"></div><div><p>왼쪽은 앞, 오른쪽은 뒤입니다. 실제 FBX 관절 좌표를 동일한 몸통 좌표계의 옆면에 투영했습니다. 높이 Z와 카메라 깊이, 몸통의 뒤쪽 방향을 구분했습니다.</p><p>팔꿈치의 어깨 기준면 좌표:<br><b>@@Z0@@ → @@Z1@@ mm</b><br>@@REAR@@mm 뒤로 이동했어도 어깨 기준면보다 아직 앞쪽입니다.</p><p>원본 후보에서 팔꿈치를 이 기준면까지 보내는 최소 상완 회전은 약 13.67°입니다. 기존 상완 수정 한도 12°는 변경하지 않았습니다. 이 계산은 현재 리그에 대한 기하학적 범위이며, 러프의 실제 3D 깊이 정답은 아닙니다.</p></div></div><div class="scroll"><table><tr><th>오차 대상</th><th>기존</th><th>실제 후보</th><th>판정</th></tr>@@QUALITY@@</table></div><p>팔꿈치·손목 점과 전체 포즈의 고정 카메라 오차가 악화되면 채택을 차단합니다. 카메라나 정규화 기준을 후보에 맞춰 다시 맞추지 않았습니다.</p></section>
<section><h2>03 · 엄지를 제외한 손바닥 전체 / 옆허리 검사</h2><p>손목·손가락 뿌리로 손바닥 면 방향을 정하고, 넓은 손바닥을 고정된 네 구역으로 나눴습니다. 손등과 손가락을 제외합니다. 대상도 하복부 전체가 아니라 해당 쪽 허리 표면으로 제한했습니다.</p><p>전체 손바닥 면적 약 @@PALM@@mm². 0.5–5mm 간격과 서로 마주 보는 면 방향을 검사하며, 전체 80% 이상 및 네 구역 각각 60% 이상이 충족되어야 합니다. 현재 네 구역 모두 실패입니다.</p><table><tr><th>손바닥 구역</th><th>면적</th><th>허리 옆까지 중앙 간격</th><th>접촉 비율</th></tr>@@ZONES@@</table><p class="bad">전체 중앙 간격 약 @@GAP@@mm. 자연스러운 손바닥 밀착에 도달하지 않았습니다.</p></section>
<section><h2>04 · 실제 안전 검증과 다른 후보 점검</h2><p class="good">현재 팔꿈치 후보: 전신 메시·관절·깊이·독립 스킨 재생·최종 바이트 검증 통과.</p><p>기존 3초 전신 검사, 10µm 정점 재생 오차, 1e−5 본 행렬 오차 한계를 유지했습니다. 팔꿈치가 뒤로 이동한다는 이유로 관통이나 비틀림 검사를 생략하지 않았습니다.</p><div class="scroll"><table><tr><th>별도 실제 검색 후보</th><th>고정 카메라 NME</th><th>거부 사유</th></tr>@@ALTERNATIVES@@</table></div><p>다른 원본 후보도 실제 변환해 확인했으나 현재 버전보다 오차가 크고 스킨 방향 검증이 실패했습니다. 이 파일들은 다운로드 대상으로 채택하지 않았습니다.</p><a href="qa-report.json">전신 QA</a> · <a href="palm-elbow-audit.json">손바닥·팔꿈치 최종 검사</a></section>
<section><h2>05 · 실제 Gemini 재평가</h2><p>gemini-3.8-flash · 첫 요청 읽기 시간 초과 후 재시도 1회 응답.<br>선호 uncertain · 팔꿈치 변화 same · 손바닥 전체 밀착 not_achieved.</p><ul>@@VLM@@</ul><p>VLM은 안전 판정 권한이 없습니다. 수치상 작은 이동과 육안으로 충분하지 않은 개선을 모두 기록했습니다.</p><a href="gemini-review.json">Gemini 원문</a></section>
<section><h2>06 · 보존된 별도 후보</h2><p>브랜치 <code>codex/refine-palm-elbow-20261024</code><br>기존 안전 버전 및 엄지 실험을 보존했습니다. 배포·원본 자산 수정 없음.</p><p>실제 검증된 팔꿈치 후보 FBX SHA-256<br><code>@@SHA@@</code></p><a href="safe-elbow-candidate.fbx" download>팔꿈치 후보 FBX 다운로드 — 밀착 완성품 아님</a></section></main><script>document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-view]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));for(const s of ['before','after'])document.getElementById(s).src=s+'-'+b.dataset.view+'.png';document.getElementById('view-label').textContent=b.textContent;}));</script></html>'''
    vals=dict(NME0=f"{q['metrics']['body']['before']:.8f}",NME1=f"{q['metrics']['body']['after']:.8f}",GAIN=f"{(1-q['metrics']['body']['after']/q['metrics']['body']['before'])*100:.4f}",REAR=f"{q['elbow_rearward_mm']:.2f}",Z0=f'{z_before:.2f}',Z1=f'{z_after:.2f}',QUALITY=quality_rows,PALM=f"{c['palm_area_mm2']:.1f}",GAP=f"{c['median_gap_mm']:.1f}",ZONES=zone_rows,ALTERNATIVES=alt,VLM=reasons,SHA=audit['artifact_sha256'])
    for key,value in vals.items():body=body.replace('@@'+key+'@@',value)
    (o/'index.html').write_text(body)
if __name__=='__main__':main()
