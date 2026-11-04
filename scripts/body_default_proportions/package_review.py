"""Package local-only screenshots and exact float32 geometry for review."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--release', type=Path, required=True)
ap.add_argument('--viewer-js', type=Path, required=True)
args = ap.parse_args()
out = args.release / 'review'
out.mkdir(exist_ok=True)
font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
font = ImageFont.truetype(font_path, 28)
small = ImageFont.truetype(font_path, 20)
validation = json.loads((args.release / 'validation.json').read_text())
for sex in ('male', 'female'):
    candidate = next(r for r in validation['rows'] if r['sex'] == sex and r['label'] == 'candidate')
    target = candidate['measurement']['measuredHeads']
    for view in ('front', 'three_quarter', 'side', 'back'):
        sheet = Image.new('RGB', (1800, 1160), '#202832')
        draw = ImageDraw.Draw(sheet)
        for i, state in enumerate(('current_editor', 'candidate')):
            im = Image.open(args.release / 'images' / f'{sex}-{state}-{view}.png')
            sheet.paste(im, (i * 900, 80))
            title = ('기존 기본형' if i == 0 else f'변경형 · {target:.1f}등신')
            draw.text((i * 900 + 30, 18), ('남성' if sex == 'male' else '여성') + ' · ' + title, font=font, fill='white')
        draw.text((30, 1128), f'{view} · 실제 T 포즈 · 동일 직교 카메라/조명/재질/척도 · 키 유지 · 확대/축소 없음', font=small, fill='#ced7e2')
        sheet.save(out / f'{sex}-comparison-{view}.png')
    contact = Image.new('RGB', (1800, 820), '#202832')
    d = ImageDraw.Draw(contact)
    d.text((20, 12), ('남성 8.3등신' if sex == 'male' else '여성 8.0등신') + ' · 기존 기본형 → 변경형', font=font, fill='white')
    for j, view in enumerate(('front', 'three_quarter', 'side')):
        pair = Image.open(out / f'{sex}-comparison-{view}.png').resize((600, 387))
        contact.paste(pair, (j * 600, 65))
        # Additional latest source baseline comparison is separate and labelled.
        for i, state in enumerate(('latest_base', 'candidate')):
            im = Image.open(args.release / 'images' / f'{sex}-{state}-{view}.png').resize((300, 350))
            contact.paste(im, (j * 600 + i * 300, 470))
    d.text((20, 445), '아래: 최신 손·어깨 소스 BASE → 변경형 (위: 실제 기존 편집 기본형)', font=small, fill='#ced7e2')
    contact.save(out / f'{sex}-three-view-sheet.png')

surfaces = dict(np.load(args.release / 'review-surfaces.npz'))
pose_file = args.release / 'pose-qa/surfaces.npz'
if pose_file.exists():
    surfaces.update(dict(np.load(pose_file)))
payload = {'geometry': {}, 'measurements': {r['sex']: r['measurement'] for r in validation['rows'] if r['label'] == 'candidate'}}
for key, q in surfaces.items():
    if key.endswith('_faces'):
        continue
    sex = key.split('_')[0]
    faces = surfaces.get(key + '_faces', surfaces[sex + '_faces'])
    payload['geometry'][key] = {
        'positions': base64.b64encode(q.astype('<f4').tobytes()).decode(),
        'indices': base64.b64encode(faces.astype('<u4').tobytes()).decode(),
    }
(out / 'geometry.json.gz').write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
shutil.copy2(args.viewer_js, out / 'viewer.js')
(out / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>기본 체형 실제 모델 검수</title>
<style>body{margin:0;background:#202832;color:#eef2f7;font:15px system-ui}header{padding:16px 24px}h1{font-size:22px;margin:0 0 12px}select,button{font:inherit;background:#344352;color:white;padding:7px 12px;border:1px solid #66778b;border-radius:6px;margin-right:8px}main{display:flex;position:relative;height:76vh}.panel{width:50%;position:relative}.panel h2{font-size:17px;position:absolute;top:0;left:24px;z-index:1}canvas{width:100%;height:100%}footer{padding:10px 24px;line-height:1.6;color:#bdccda}#status{padding-left:12px}</style>
<header><h1>기본 체형 · 실제 모델 비교</h1><select id="sex"><option value="male">남성 · 8.3등신</option><option value="female">여성 · 8등신</option></select>
<select id="baseline"><option value="current_editor">기존 편집 기본형</option><option value="latest_base">최신 손·어깨 BASE</option><option value="legacy_fallback">공통 fallback FBX</option></select>
<select id="pose"><option>T</option><option>ARMS_DOWN</option><option>RAISE</option><option>COMBINED</option><option>TWIST</option><option>FIST</option></select>
<button data-angle="0">정면</button><button data-angle="45">3/4</button><button data-angle="90">측면</button><button data-angle="180">후면</button><span id="status">실제 표면 불러오는 중…</span></header>
<main><section class="panel"><h2 id="left-label">기존 기본형</h2><canvas id="left"></canvas></section><section class="panel"><h2 id="right-label">변경형</h2><canvas id="right"></canvas></section></main>
<footer>드래그 회전 · 휠 확대 · 양쪽 카메라 동기화 · 동일 바닥과 실제 미터 단위 · 중립 키 유지(정규화 확대 없음)<br>포즈 비교는 최신 소스 BASE와 변경형을 표시합니다. 원본 FBX에는 작성기 보정/실시간 관절 제한이 자동 포함되지 않습니다. 로컬 검수용, 제품 기본값 비활성.</footer>
<script type="module" src="viewer.js"></script></html>''')
print('REVIEW_READY', out.resolve())
