"""Summarize P0 bindings and P1 solver, FBX readback, and regional fallback."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.original_seed_oracle import PROFILES, metrics, problem


def summarize(rows):
    names = ('arm_direction_deg', 'leg_direction_deg', 'elbow_bend_error_deg',
             'knee_bend_error_deg', 'head_up_forward_deg', 'endpoint_error_torso')
    result = {}
    for name in names:
        values = [value for row in rows for value in (row[name] or [])]
        result[name] = dict(mean=float(np.mean(values)) if values else None, observations=len(values))
    return result


def direction(case):
    return float(np.mean(case['arm_direction_deg'] + case['leg_direction_deg']))


def render_example(root, row):
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation
    from experiments.refine_v5.render_review import canonical, render

    captured = root / 'review-native'
    methods = [('B0', 'Recorded BVH / B0'), ('oracle', 'GT goal / proposal'),
               ('regional', 'Saved after skin gate'), ('GT', 'Ground truth FBX')]
    shapes = []
    width, height = 245, 224
    image = Image.new('RGB', (width * 5, height * 2 + 35), 'white')
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), row['case_id'] + ' | Rough', fill='black', font=font)
    with Image.open(root / row['rough']) as src:
        rough = src.convert('RGB')
        rough.thumbnail((width - 14, height * 2 - 10))
        image.paste(rough, ((width - rough.width) // 2, 32))
    for column, (name, label) in enumerate(methods, 1):
        draw.text((column * width + 8, 8), label, fill='black', font=font)
        with np.load(captured / f'{name}.npz') as mesh:
            shape = canonical(mesh, load_json(captured / f'{name}.json')['bones'])
        shapes.append(shape)
    for index, angle in enumerate((0., 90.)):
        rotation = Rotation.from_euler('y', angle, degrees=True).as_matrix()
        points = np.concatenate([(v @ rotation.T)[:, :2] * [1, -1] for v, _ in shapes])
        low, high = points.min(axis=0), points.max(axis=0)
        scale = min((width - 15) / max(high[0] - low[0], 1e-8), (height - 12) / max(high[1] - low[1], 1e-8))
        framing = scale, np.array([width, height]) / 2 - (low + high) * scale / 2
        for column, (vertices, faces) in enumerate(shapes, 1):
            image.paste(render(vertices, faces, (width, height), yaw=angle, framing=framing),
                        (column * width, 31 + index * height))
    image.save(root / 'comparison.png')


def run(root, gt_audit):
    manifest = load_json(root / 'manifest.json')
    plans = load_json(root / 'p1-plans.json')
    cohort = load_json(root / 'p1-cohort.json')
    truth = {row['case_id']: row for row in load_json(gt_audit)}
    cases = {row['case_id']: row for row in manifest['rows']}
    if (digest(root / 'manifest.json') != plans['manifest_sha256']
            or [row['case_id'] for row in plans['rows']] != cohort['case_ids']):
        raise ValueError('P1 plan differs from frozen cohort')
    rows = []
    for row in plans['rows']:
        if row['status'] != 'computed':
            rows.append(row)
            continue
        cid = row['case_id']
        native = root / 'p1-native'
        audit = load_json(native / 'results' / f'{cid}.json')
        source = load_json(native / f'{cid}-source.json')
        names, lookup, _, _, _, target, head, torso, spec = problem(cases[cid], source, truth[cid])
        candidate = native / 'results' / f'{cid}-oracle.fbx'
        regional = native / 'results' / f'{cid}-regional.fbx'
        if digest(regional) != audit['regional_sha256']:
            raise ValueError('regional output changed')
        if not audit['accepted'] and audit['regional_sha256'] != row['source_sha256']:
            raise ValueError('rejected output was not exact B0')
        if audit['candidate_exists'] and digest(candidate) != audit['candidate_sha256']:
            raise ValueError('diagnostic output changed')
        matrices = np.asarray([audit['matrices'][n] for n in names])
        readback = metrics(matrices, target, spec, lookup, head, torso)
        rows.append(dict(case_id=cid, split=row['split'], representation=cases[cid]['representation'],
                         source_bvh=cases[cid]['selection']['verified_selected_bvh'],
                         source_pose=cases[cid]['selection']['selected_pose_id'],
                         files=dict(source=row['source'], regional=str(regional.relative_to(root)),
                                    candidate=str(candidate.relative_to(root)) if audit['candidate_exists'] else None),
                         status=row['status'], B0=row['B0'],
                         **{name: row['profiles'][name]['metrics'] for name in PROFILES},
                         regional=readback if audit['accepted'] else row['B0'],
                         audit={k: v for k, v in audit.items() if k != 'matrices'}))
    kept = [row for row in rows if row['status'] == 'computed']
    methods = ('B0', *PROFILES, 'regional')
    summary = {method: summarize([row[method] for row in kept]) for method in methods}
    counts = dict(Counter(row['status'] for row in rows))
    accepted = sum(row['audit']['accepted'] for row in kept)
    full_skin = sum(row['audit']['full_skin_certified'] for row in kept)
    direction_changes = {}
    for method in methods[1:]:
        changes = [direction(row[method]) - direction(row['B0']) for row in kept
                   if row['B0']['arm_direction_deg'] or row['B0']['leg_direction_deg']]
        direction_changes[method] = dict(improved=sum(d < -.1 for d in changes),
                                         worsened=sum(d > .1 for d in changes),
                                         unchanged=sum(-.1 <= d <= .1 for d in changes))
    report = dict(
        p0={k: v for k, v in manifest.items() if k != 'rows'}, p1_counts=counts,
        summary=summary, regionally_accepted=accepted,
        full_skin_certified=full_skin, direction_changes=direction_changes,
        audit_reasons=dict(Counter(reason for row in kept for reason in row['audit']['reasons'])),
        skin_status=dict(Counter(str(row['audit']['full_skin_status']) for row in kept)),
        full_skin_reasons=dict(Counter(reason for row in kept for reason in row['audit']['full_skin_reasons'])),
        rows=rows,
    )
    save_json(root / 'p1-comparison.json', report)
    full_reference = all(c.get('target_scope') for c in manifest['rows'])
    scope = ('029의 실제 GT 메시에서 대응 부위를 확인하고 러프에서 가려진 부위까지 정답 목표로 사용했다.'
             if full_reference else '목표 마스크는 기존 러프 가시성 검수를 따랐다.')
    headers = ['# Refine v5 · 원본 BVH P0/P1', '',
               '선택 로그와 제작 시작 기록이 같은 BVH를 기본 체형으로 변환했다. 정답 목표는 FBX의 관절·머리 축에서 얻었으며, 카메라·VLM·검색은 사용하지 않았다.',
               '기존 봉인 평가셋은 P1에서 제외했다. 자세는 몸통 좌표계로 정렬했다. ' + scope, '',
               f'- 원본 BVH 확인: {manifest["status_counts"]["original_library_bvh_confirmed"]}/{manifest["final_samples"]}',
               f'- P1 계산: {len(kept)}/{len(rows)} · 지역 피부 검사 채택 {accepted}/{len(kept)} · 전신 피부 인증 {full_skin}/{len(kept)}',
               '- regional은 limbs_head를 FBX로 저장한 뒤 지역 검사에서 거절된 포즈를 시작 FBX로 되돌린 결과다.',
               '- 방향과 굽힘은 각도 오차, 끝점은 몸통 길이로 정규화한 위치 오차다.', '',
               '| 방식 | 팔 방향 | 다리 방향 | 팔꿈치 | 무릎 | 머리 축 | 끝점 |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for method, columns in summary.items():
        formatted = ['—' if x['mean'] is None else f"{x['mean']:.3f}" for x in columns.values()]
        headers.append('| ' + ' | '.join([method, *formatted]) + ' |')
    headers += ['', '[데이터 원장](manifest.json) · [P1 컷별 수치와 파일 경로](p1-comparison.json)']
    if full_reference and (root / 'review-native' / 'oracle.npz').exists():
        render_example(root, manifest['rows'][0])
        headers += ['', '[정면·측면 비교](comparison.png)']
    (root / 'REPORT.md').write_text('\n'.join(headers) + '\n')
    print({k: v for k, v in report.items() if k not in ('rows', 'audit_reasons')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gt-audit', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.gt_audit)
