"""Build a local, side-by-side review of the ten saved Blender cases."""

import argparse
from collections import Counter
from html import escape
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.ab_evaluate import error_deg, point_map
from experiments.refine_v5.learning_data import BONES, axes, digest, frame
from experiments.refine_v5.render_review import render, render_projected

LABELS = {
    'B0': '원본 B0',
    'ortho_fixed': '기존 시점 정사영',
    'ortho_calibrated': '시점 보정 정사영',
    'perspective': '원근 보정',
    'GT': '정답 FBX',
}
SHORT = {'B0': 'B0', 'ortho_fixed': 'Fixed ortho', 'ortho_calibrated': 'Fitted ortho',
         'perspective': 'Perspective', 'GT': 'Ground truth'}
STATUS = {'reference': '원본', 'depth_pass': '깊이 조건 충족', 'partial_depth': '일부 깊이 조건 미충족',
          'waist_contact_unresolved': '허리 근접 미충족', 'fallback_exact_B0': 'B0 복귀'}
READABLE_CUE = {'body_behind': '몸통 기준 뒤', 'body_in_front': '몸통 기준 앞',
                'camera_farther': '카메라에서 멀어짐', 'sole_clearance': '지지발 대비 높이',
                'waist_side_contact': '허리 옆면 기준 뒤', 'head_behind': '머리 뒤쪽 기준 뒤',
                'head_front': '머리 앞쪽 기준 앞'}


def measure(root, out, cases):
    frozen = load_json(root / 'frozen.json')
    if digest(root / 'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('frozen manifest differs')
    plans = {(r['case_id'], r['method']): r for r in frozen['results']}
    native = {(r['case_id'], r['method']): r for r in load_json(root / 'native/audit.json')}
    measured = {(r['case_id'], r['method']): r for r in load_json(root / 'depth-ten-readback.json')}
    selection = {(r['case_id'], r['method']): r for r in load_json(root / 'selection.json')}
    records = []
    for case in cases:
        cid = case['case_id']
        gt = native[cid, 'GT']
        if digest(gt['path']) != gt['sha256']:
            raise ValueError('ground truth FBX changed')
        gt_axes, gt_points = axes(gt['bones']), point_map(gt['bones'])
        chain = case['depth_evidence']['movable']
        files = {}
        for method in LABELS:
            actual = native[cid, method]
            original = Path(actual['path'])
            if digest(original) != actual['sha256']:
                raise ValueError('audited bytes changed')
            key = 'B0' if actual.get('status') == 'fallback_exact_B0' else method
            destination = out / 'fbx' / f'{cid}-{key}.fbx'
            if not destination.exists():
                shutil.copyfile(original, destination)
            if digest(destination) != actual['sha256']:
                raise ValueError('published artifact differs')
            files[method] = str(destination.relative_to(out))
            if method == 'GT':
                continue
            actual_axes, actual_points = axes(actual['bones']), point_map(actual['bones'])
            angles = {}
            for name in chain:
                index = BONES.index(name)
                angles[name] = dict(direction_deg=error_deg(actual_axes[index, 1], gt_axes[index, 1]),
                                    joint_nme=float(np.linalg.norm(actual_points[name] - gt_points[name])))
            plan = plans[cid, method]
            metrics = measured[cid, method]
            records.append(dict(
                case_id=cid, split=case['split'], depth_authority=case['depth_authority'], method=method,
                status=actual['status'], camera_status=plan.get('camera_status'), selected_profile=plan.get('selected_profile'),
                perspective_fitted=case['cameras']['perspective']['status'] == 'fitted',
                chain_error_deg=float(np.mean([angles[n]['direction_deg'] for n in chain[:2]])),
                endpoint_3d_nme=float(np.mean([angles[n]['joint_nme'] for n in chain[1:]])),
                parts=angles, depth=metrics, contact=actual['contact'], gt_contact=gt['contact'],
                skin_regressions=actual['skin_nonregression_failures'], strict_skin_status=actual['full_skin']['status'],
                full_skin_certified=actual['full_skin']['complete_collision_certificate'],
                native_errors=dict(cloud_m=actual['max_cloud_error_m'], frame_element=actual['max_rotation_element_error']),
                rejection_reasons=dict(Counter(reason for item in selection[cid, method].get('rejected', [])
                                               for reason in item['reasons'])),
                fbx=files[method], sha256=actual['sha256'],
            ))
        case['review_files'] = files
    summary = {}
    for method in list(LABELS)[:-1]:
        group = [row for row in records if row['method'] == method]
        eligible = [row for row in group if row['perspective_fitted']]
        summary[method] = dict(
            cases=len(group), applied=sum(row['status'] not in ('reference', 'fallback_exact_B0') for row in group),
            fallback=sum(row['status'] == 'fallback_exact_B0' for row in group),
            depth_all_pass=sum(all(cue['passes'] for cue in row['depth']['cues']) for row in group),
            chain_error_deg=float(np.mean([row['chain_error_deg'] for row in group])),
            endpoint_3d_nme=float(np.mean([row['endpoint_3d_nme'] for row in group])),
            eligible_cases=len(eligible),
            eligible_chain_error_deg=float(np.mean([row['chain_error_deg'] for row in eligible])),
            eligible_endpoint_3d_nme=float(np.mean([row['endpoint_3d_nme'] for row in eligible])),
            eligible_depth_all_pass=sum(all(cue['passes'] for cue in row['depth']['cues']) for row in eligible),
            common_screen_nme=float(np.mean([row['depth']['common_screen_nme'] for row in group])),
            full_skin_certified=sum(row['full_skin_certified'] for row in group),
            statuses=dict(Counter(row['status'] for row in group)),
        )
    save_json(out / 'comparison.json', dict(
        summary=summary, records=records, camera_options={c['case_id']: c['cameras'] for c in cases},
        protocol='paired B0, two orthographic cameras, one perspective camera, same depth cues/rotation bounds/native gate',
        provider_calls=0, ground_truth_used_for_selection=False,
    ))
    return summary, records, native


def review_images(root, out, cases, rows, native):
    lookup = {(r['case_id'], r['method']): r for r in rows}
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
        small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    width, height = 245, 500
    size = (width, 224)
    strips = []
    for case in cases:
        cid = case['case_id']
        strip = Image.new('RGB', (width * 6, height), 'white')
        draw = ImageDraw.Draw(strip)
        draw.text((9, 8), cid + ' | Rough', fill='#202020', font=font)
        with Image.open(case['image']) as source:
            image = source.convert('RGB')
        image.thumbnail((width - 16, height - 45))
        strip.paste(image, ((width - image.width) // 2, 36))
        screen = Image.new('RGB', (width * 5, 224), 'white')
        camera_draw = ImageDraw.Draw(screen)
        with Image.open(case['image']) as source:
            rough = source.convert('RGB')
        scale = min((width - 12) / rough.width, 184 / rough.height)
        rough = rough.resize((round(rough.width * scale), round(rough.height * scale)), Image.Resampling.LANCZOS)
        margin = ((width - rough.width) // 2, (184 - rough.height) // 2)
        screen.paste(rough, (margin[0], 30 + margin[1]))
        camera_draw.text((9, 8), cid + ' | Rough', fill='#202020', font=font)
        views = []
        for col, method in enumerate(LABELS, 1):
            _, center, torso, basis = frame(native[cid, method]['bones'])
            with np.load(root / 'native' / f'{cid}-{method}.npz') as mesh:
                vertices, faces = mesh['vertices'], mesh['faces']
            views.append(((vertices - center) @ basis / torso, faces))
            if method == 'GT':
                continue
            camera = case['cameras']['ortho_fixed' if method == 'B0' else method]
            camera_space = (vertices - camera['origin']) @ np.asarray(camera['rotation']).T
            # The fitted camera's positive depth is farther from the viewer.
            camera_space[:, 2] *= -1
            try:
                pixels = project(vertices, camera) * scale
            except ValueError as exc:
                if str(exc) != 'near plane':
                    raise
                camera_draw.text((col * width + 8, 60), 'Near plane clipped', fill='#202020', font=small)
            else:
                screen.paste(render_projected(camera_space, faces, pixels, rough.size),
                             (col * width + margin[0], 30 + margin[1]))
            camera_draw.text((col * width + 8, 8), SHORT[method], fill='#202020', font=font)
        screen.save(out / 'images' / f'{cid}-rough-camera.png')
        for index, yaw in enumerate((0., 90.)):
            rotation = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
            xy = np.concatenate([(v @ rotation.T)[:, :2] * [1, -1] for v, _ in views])
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            scale = min((size[0] - 18) / max(hi[0] - lo[0], 1e-5), (size[1] - 12) / max(hi[1] - lo[1], 1e-5))
            origin = np.asarray(size) / 2 - (lo + hi) * scale / 2
            for column, (vertices, faces) in enumerate(views, 1):
                strip.paste(render(vertices, faces, size=size, yaw=yaw, framing=(scale, origin)),
                            (column * width, 43 + index * size[1]))
        for col, method in enumerate(LABELS, 1):
            draw.text((col * width + 8, 8), SHORT[method], fill='#202020', font=font)
            if method != 'GT':
                row = lookup[cid, method]
                label = f"{row['chain_error_deg']:.1f}°"
                if row['status'] == 'fallback_exact_B0':
                    label += ' | B0 fallback'
                elif row['camera_status'] and row['camera_status'].startswith('fallback'):
                    label += ' | ortho fallback'
                draw.text((col * width + 8, 25), label, fill='#333333', font=small)
        draw.line((7, height - 2, strip.width - 7, height - 2), fill='#dddddd')
        strip.save(out / 'images' / f'{cid}.png')
        strips.append(strip)
    overview = Image.new('RGB', (strips[0].width, height * len(strips)), 'white')
    for index, strip in enumerate(strips):
        overview.paste(strip, (0, index * height))
    overview.save(out / 'overview.png')


def html_report(out, cases, summary, records):
    data = {(r['case_id'], r['method']): r for r in records}
    links = ' '.join(f'<a href="#case-{c["case_id"]}">{c["case_id"]}</a>' for c in cases)
    lines = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Refine v5 깊이 보정 10컷 비교</title>',
             '<style>body{font:15px system-ui,sans-serif;background:#f4f6f9;color:#182433;margin:0}main{max-width:1330px;margin:auto;padding:24px}section,header{background:white;border:1px solid #d6dde5;border-radius:10px;margin-bottom:18px;padding:16px}img{display:block;width:100%;height:auto}table{width:100%;border-collapse:collapse;text-align:left}td,th{border-bottom:1px solid #dde3ea;padding:8px}a{color:#085f9e}nav{display:flex;gap:12px;flex-wrap:wrap}small{color:#455569}h1,h2{margin:0 0 12px}.wide{overflow-x:auto}.wide img{min-width:860px}</style>',
             '<main><header><h1>Refine v5 깊이 보정 10컷 비교</h1>',
             '<p>같은 러프·B0·에셋에서 세 카메라에 같은 깊이 조건, 회전 한도, 메시 검사를 적용했습니다. 위 행은 몸통 기준 정면, 아래 행은 측면입니다. 측면에서 몸의 앞쪽은 오른쪽입니다.</p>',
             '<p>컷별 펼치기에서 러프 좌표계 투영을 볼 수 있습니다. B0는 기존 시점으로 표시했습니다. 정답 FBX의 원본 카메라 정보는 확보하지 못했습니다.</p>',
             '<p>사용자 검수 4컷과 러프 육안 관측 6컷을 구분했습니다. 화상마다 표시된 각도는 정답 FBX 대비 수정 대상 두 뼈의 평균 방향 오차입니다. 3D 위치 오차는 몸통 길이로 정규화했습니다.</p>',
             '<nav>' + links + '<a href="comparison.json">전체 수치 JSON</a><a href="overview.png">전체 이미지</a></nav>',
             '<table><thead><tr><th>방식</th><th>체인 각도 ↓</th><th>3D 끝점 오차 ↓</th><th>깊이 조건 충족</th><th>B0 복귀</th><th>전신 피부 인증</th></tr></thead><tbody>']
    for method, value in summary.items():
        lines.append(f'<tr><td>{LABELS[method]}</td><td>{value["chain_error_deg"]:.2f}°</td>'
                     f'<td>{value["endpoint_3d_nme"]:.3f}</td><td>{value["depth_all_pass"]}/10</td>'
                     f'<td>{value["fallback"]}</td><td>{value["full_skin_certified"]}/10</td></tr>')
    lines.append('</tbody></table><p>050·074·081은 골반 기준점이 부족해 시점 보정과 원근 보정 모두 기존 정사영으로 계산했습니다.</p>')
    lines.append('<details><summary>카메라를 실제로 보정한 7컷</summary><table><thead><tr><th>방식</th>'
                 '<th>체인 각도 ↓</th><th>3D 끝점 오차 ↓</th><th>깊이 조건 충족</th></tr></thead><tbody>')
    for method, value in summary.items():
        lines.append(f'<tr><td>{LABELS[method]}</td><td>{value["eligible_chain_error_deg"]:.2f}°</td>'
                     f'<td>{value["eligible_endpoint_3d_nme"]:.3f}</td><td>{value["eligible_depth_all_pass"]}/'
                     f'{value["eligible_cases"]}</td></tr>')
    lines.append('</tbody></table></details></header>')
    for case in cases:
        cid = case['case_id']
        origin = '육안 관측' if case['depth_authority'] == 'analyst_rough_only' else '사용자 검수'
        lines.append(f'<section id="case-{cid}"><h2>{cid} · {origin} · {case["split"]}</h2>'
                     f'<div class="wide"><a href="images/{cid}.png"><img loading="lazy" src="images/{cid}.png" alt="{cid} 정면과 측면 비교"></a></div>'
                     f'<details><summary>러프 좌표계에서 카메라 비교</summary><div class="wide"><a href="images/{cid}-rough-camera.png"><img loading="lazy" src="images/{cid}-rough-camera.png" alt="{cid} 러프와 카메라 투영 비교"></a></div></details>'
                     '<div class="wide"><table><thead><tr><th>방식</th><th>선택 결과</th><th>3D 각도</th>'
                     '<th>3D 끝점</th><th>깊이 관계와 기준</th><th>FBX</th></tr></thead><tbody>')
        for method in list(LABELS)[:-1]:
            row = data[cid, method]
            status = STATUS[row['status']]
            if row['status'] == 'fallback_exact_B0':
                status += {'039': ' · 머리 관통 또는 새 피부 관통', '050': ' · 팔–몸통 관통 악화 또는 머리 관통',
                           '081': ' · FBX 재로드 후 뼈 프레임 오차'}.get(cid, ' · 후보 검증 실패')
            if row['camera_status'] and row['camera_status'].startswith('fallback'):
                status += ' · 카메라 정사영 복귀'
            measurements = '<br>'.join(
                f'{READABLE_CUE[c["kind"]]} {c["value_mm"]:+.1f}mm '
                f'({"충족" if c["passes"] else "미충족"})'
                for c in row['depth']['cues'])
            if 'side_waist_palm_gap_mm' in row['depth']:
                measurements += f'<br>허리 옆면 최단 거리 {row["depth"]["side_waist_palm_gap_mm"]:.1f}mm'
            if 'forearm_hand_head_crossings' in row['depth']:
                measurements += f'<br>팔·손–머리 교차 {row["depth"]["forearm_hand_head_crossings"]}쌍'
            lines.append(f'<tr><td>{LABELS[method]}</td><td>{escape(status)}</td>'
                         f'<td>{row["chain_error_deg"]:.1f}°</td><td>{row["endpoint_3d_nme"]:.3f}</td>'
                         f'<td>{measurements}</td><td><a href="{row["fbx"]}">열기</a></td></tr>')
        lines.append('</tbody></table></div>')
        contact = data[cid, 'B0']['contact']
        if contact:
            lines.append('<details><summary>피부 접촉과 관통 수치</summary><table><thead><tr><th>방식</th>'
                         '<th>손바닥–표면 거리 중앙값</th><th>관통으로 판정된 손바닥 표본</th>'
                         '</tr></thead><tbody>')
            for method in LABELS:
                patches = data[cid, 'B0']['gt_contact'] if method == 'GT' else data[cid, method]['contact']
                for patch in patches:
                    if patch['status'] == 'measured':
                        gap = f"{patch['palm_median_gap_mm']:.1f} mm"
                        samples = f"{patch['palm_negative_signed_samples']}/{patch['palm_samples']}"
                    else:
                        gap = samples = '측정 불가'
                    lines.append(f'<tr><td>{LABELS[method]}</td><td>{gap}</td><td>{samples}</td></tr>')
            lines.append('</tbody></table></details>')
        lines.append(f'<p><a href="{case["review_files"]["GT"]}">정답 FBX</a> · '
                     '<small>저장된 후보는 검수용입니다. 전신 피부 판정은 상단과 JSON에서 확인할 수 있습니다.</small></p></section>')
    lines.append('</main></html>')
    (out / 'review.html').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    cases = load_json(args.root / 'manifest.json')['cases']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'fbx').mkdir()
    (args.out / 'images').mkdir()
    summary, rows, native = measure(args.root, args.out, cases)
    review_images(args.root, args.out, cases, rows, native)
    html_report(args.out, cases, summary, rows)
    print(summary)


if __name__ == '__main__':
    main()
