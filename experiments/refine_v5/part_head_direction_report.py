"""Render a close-up of the actual composed FBXs and save their measurements."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.learning_data import digest, frame
from experiments.refine_v5.part_head import unit
from experiments.refine_v5.render_review import render, render_projected


SIZE = (276, 225)


def mesh(path):
    with np.load(path) as data:
        return data['vertices'], data['faces']


def camera_closeup(case, vertices, faces, center):
    camera = case['camera']
    space = (vertices - camera['origin']) @ np.asarray(camera['rotation']).T
    space[:, 2] *= -1.
    origin = project(center[None], camera)[0]
    pixels = (project(vertices, camera) - origin) * (SIZE[1] / (1.05 * camera['pixel_scale'])) + np.array(SIZE) / 2.
    return render_projected(space, faces, pixels, SIZE)


def axis_in_body(bones, head=None):
    by, _, _, basis = frame(bones)
    head = np.asarray(by['Head']['matrix'] if head is None else head)
    return unit(basis.T @ head[:3, 2])


def body_angles(axis):
    x, up, forward = axis
    return dict(yaw_deg=float(np.degrees(np.arctan2(x, forward))),
                pitch_deg=float(np.degrees(np.arctan2(-up, np.hypot(x, forward)))))


def side_closeup(vertices, faces, bones):
    by, _, scale, basis = frame(bones)
    center = np.asarray(by['Head']['matrix'])[:3, 3] + .25 * scale * basis[:, 1]
    return render((vertices - center) @ basis / scale, faces, SIZE, yaw=90.,
                  framing=(SIZE[1] / 1.02, np.array(SIZE) / 2.))


def report(root, native, prior, out):
    cases = {c['case_id']: c for c in load_json(root / 'manifest.json')['cases']}
    previous = {(r['case_id'], r['method']): r for r in load_json(prior / 'native/audit.json')}
    results = load_json(native / 'audit.json')
    out.mkdir(parents=True, exist_ok=False)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    row_height = SIZE[1] + 35
    grid = Image.new('RGB', (SIZE[0] * 3, len(results) * row_height * 2), 'white')
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Part goals 머리·목 보정</title>',
            '<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}img{max-width:100%}td,th{padding:5px 10px;text-align:left}</style>',
            '<h1>Part goals 머리·목 보정</h1>',
            '<p>위는 러프와 보정에 사용한 카메라입니다. 아래는 같은 몸통 기준 측면이며 정답의 원래 카메라는 제공되지 않았습니다.</p>',
            '<p>좌우는 화면 기준이고 숙임은 아래 방향이 양수입니다. 각도 범위는 러프를 보고 붙인 검수값입니다.</p>',
            '<p><a href="overview.png">전체 이미지</a> · <a href="comparison.json">수치 JSON</a></p>']
    for i, row in enumerate(results):
        cid = row['case_id']
        case = cases[cid]
        source, gt = previous[cid, 'fused'], previous[cid, 'GT']
        if digest(gt['path']) != gt['sha256'] or digest(source['path']) != row['source_sha256']:
            raise ValueError('reference FBX changed')
        before_mesh = mesh(prior / 'native' / f'{cid}-fused.npz')
        gt_mesh = mesh(prior / 'native' / f'{cid}-GT.npz')
        after_mesh = mesh(native / f'{cid}.npz')
        before_axis = axis_in_body(source['bones'])
        after_axis = axis_in_body(source['bones'], row['saved_head_matrix'])
        gt_axis = axis_in_body(gt['bones'])
        row['body_frame'] = {name: body_angles(axis) for name, axis in (
            ('before', before_axis), ('after', after_axis), ('gt', gt_axis))}
        row['head_forward_error_to_gt_deg'] = {
            name: float(np.degrees(np.arccos(np.clip(axis @ gt_axis, -1., 1.))))
            for name, axis in (('before', before_axis), ('after', after_axis))}
        by, _, scale, basis = frame(source['bones'])
        center = np.asarray(by['Head']['matrix'])[:3, 3] + .25 * scale * basis[:, 1]
        rough = Image.open(case['image']).convert('RGB').crop(row['goal']['rough_crop'])
        rough.thumbnail(SIZE)
        canvas = Image.new('RGB', (SIZE[0] * 3, row_height * 2), 'white')
        draw = ImageDraw.Draw(canvas)
        canvas.paste(rough, ((SIZE[0] - rough.width) // 2, 23))
        for col, geometry in enumerate((before_mesh, after_mesh), 1):
            canvas.paste(camera_closeup(case, *geometry, center), (col * SIZE[0], 23))
        for col, (geometry, bones) in enumerate(((before_mesh, source['bones']), (after_mesh, source['bones']), (gt_mesh, gt['bones']))):
            canvas.paste(side_closeup(*geometry, bones), (col * SIZE[0], row_height + 23))
        for line, labels in enumerate(((cid + ' Rough', 'Part goals / Rough view', 'Part goals + head / Rough view'),
                                       ('Part goals / Body side', 'Part goals + head / Body side', 'Ground truth / Body side'))):
            for col, label in enumerate(labels):
                draw.text((col * SIZE[0] + 7, row_height * line + 4), label, fill='black', font=font)
        canvas.save(out / f'{cid}.png')
        grid.paste(canvas, (0, i * row_height * 2))
        artifact = out / f'{cid}-part-goals-head.fbx'
        shutil.copyfile(row['path'], artifact)
        if digest(artifact) != row['sha256']:
            raise ValueError('head FBX changed')
        row['fbx'] = artifact.name
        page.append(f'<h2>{cid}</h2><img src="{cid}.png" alt="{cid} 고개 비교">')
        page.append(f'<p><a href="{artifact.name}">저장 FBX</a></p>'
                    '<table><tr><th>항목</th><th>Part goals</th><th>머리 보정 후</th><th>러프 검수 범위</th></tr>')
        for key, label in [('yaw_deg', '좌우 돌림'), ('pitch_deg', '숙임'), ('neck_tilt_deg', '화면 목 축')]:
            if key in row['goal']:
                low, high = row['goal'][key]
                page.append(f'<tr><td>{label}</td><td>{row["before"][key]:.1f}°</td>'
                            f'<td>{row["after"][key]:.1f}°</td><td>{low}° ~ {high}°</td></tr>')
        page.append('</table>')
        errors = row['head_forward_error_to_gt_deg']
        page.append(f'<p>정답 FBX 대비 머리 방향 오차 {errors["before"]:.1f}° → {errors["after"]:.1f}° '
                    '(몸통 기준)</p>')
        if row['after_range_error'].get('neck_tilt_deg', 0.) > .1:
            page.append('<p>목 축은 실험 회전 한도 안에서 개선됐으며 검수 범위에는 아직 도달하지 않았습니다.</p>')
    grid.save(out / 'overview.png')
    save_json(out / 'comparison.json', results)
    (out / 'review.html').write_text('\n'.join(page), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'native', 'prior', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    report(**vars(parser.parse_args()))
