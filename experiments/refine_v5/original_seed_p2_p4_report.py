"""Compare P2–P4 on the same original BVH and draw the saved FBX poses."""

import argparse
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.goal_probe_v2 import fk
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.original_seed_oracle import metrics, problem
from experiments.refine_v5.original_seed_report import summarize
from experiments.refine_v5.render_review import canonical, render


LABELS = {'B0': 'Original BVH / B0', 'oracle': 'GT goals', 'automatic': 'Auto observations',
          'learned': 'Auto + learned', 'GT': 'Ground truth'}
METHODS = ('oracle', 'automatic', 'learned')


def score(world, data):
    _, lookup, _, _, _, target, head, torso, spec = data
    return metrics(world, target, spec, lookup, head, torso)


def planned_pose(data, plan):
    _, lookup, parents, local, base, *_ = data
    edits = plan.get('edits_deg', {}) if plan else {}
    if not edits:
        return base
    return fk(local, parents, [lookup[n] for n in edits], np.radians(list(edits.values())))


def mean_direction(row):
    parts = row['arm_direction_deg'] + row['leg_direction_deg']
    return float(np.mean(parts)) if parts else None


def kept(audit):
    return audit['fraction'] is not None or bool(audit.get('group_fractions'))


def draw_case(p1, run, case, row):
    cid = row['case_id']
    width, height = 245, 220
    size = (width, height)
    image = Image.new('RGB', (width * 6, height * 2 + 46), 'white')
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 12)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), cid + ' | Rough', fill='#111', font=font)
    with Image.open(p1 / case['rough']) as opened:
        rough = opened.convert('RGB')
        rough.thumbnail((width - 16, height * 2))
        image.paste(rough, ((width - rough.width) // 2, 35))
    gallery = run / 'gallery-native'
    shapes = []
    for column, (name, label) in enumerate(LABELS.items(), 1):
        draw.text((column * width + 8, 8), label, fill='#111', font=font)
        if name != 'GT':
            number = mean_direction(row['saved'][name])
            caption = f'{number:.1f}°' if number is not None else 'No visible limb score'
            if name != 'B0':
                audit = row['audit'][name]
                fraction = audit['fraction']
                if audit.get('group_fractions'):
                    caption += ' | Head only' if set(audit['group_fractions']) == {'head'} else ' | Partial edit'
                elif fraction is not None:
                    caption += f' | {fraction:.0%} of edit'
                else:
                    caption += ' | B0 retained'
            draw.text((column * width + 8, 25), caption, fill='#333', font=small)
        with np.load(gallery / f'{cid}-{name}.npz') as mesh:
            shape = canonical(mesh, load_json(gallery / f'{cid}-{name}.json')['bones'])
        shapes.append(shape)
    for index, angle in enumerate((0., 90.)):
        rotation = Rotation.from_euler('y', angle, degrees=True).as_matrix()
        points = np.concatenate([(v @ rotation.T)[:, :2] * [1, -1] for v, _ in shapes])
        low, high = points.min(axis=0), points.max(axis=0)
        scale = min((width - 16) / max(high[0] - low[0], 1e-8), (height - 12) / max(high[1] - low[1], 1e-8))
        framing = scale, np.asarray(size) / 2 - (low + high) * scale / 2
        for column, (vertices, faces) in enumerate(shapes, 1):
            image.paste(render(vertices, faces, size=size, yaw=angle, framing=framing),
                        (column * width, 42 + index * height))
    draw.line((7, image.height - 1, image.width - 7, image.height - 1), fill='#ddd')
    image.save(run / 'images' / f'{cid}.png')
    return image


def write_review(run, report):
    model = report['training']
    page = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Refine v5 P2–P4 원본 BVH 비교</title>',
        '<style>body{max-width:1480px;margin:25px auto;padding:0 16px;font:16px system-ui;color:#1c2630}'
        'img{display:block;width:100%}a{color:#135da0}td,th{padding:7px 12px;text-align:left;border-bottom:1px solid #ddd}'
        'table{border-collapse:collapse}section{border-top:1px solid #ddd;margin-top:22px}</style>',
        '<h1>Refine v5 P2–P4 · 같은 원본 BVH에서 17컷 비교</h1>',
        f'<p>개발 학습 {len(model["development_used"])}쌍 · 검증 17쌍. 이미지는 저장·재로드 후의 정면과 측면입니다. '
        '정답 목표 열은 GT를 직접 알려 준 상한 비교입니다. 각도는 보이는 팔다리의 평균 방향 오차입니다.</p>',
        '<p><a href="overview.png">전체 비교 이미지</a> · <a href="comparison.json">컷별 수치와 감사 기록</a></p>',
        '<table><thead><tr><th>단계</th><th>방법</th><th>팔 방향</th><th>다리 방향</th><th>머리 축</th>'
        '<th>정규화 끝점</th><th>국소 검사 채택</th></tr></thead><tbody>',
    ]
    for stage, block in report['summary'].items():
        for method, values in block.items():
            if stage == 'proposed' and method == 'B0':
                continue
            numbers = [values[key]['mean'] for key in ('arm_direction_deg', 'leg_direction_deg',
                        'head_up_forward_deg', 'endpoint_error_torso')]
            text = ''.join(f'<td>{x:.3f}</td>' if x is not None else '<td>—</td>' for x in numbers)
            accepted = report['gate'][method]['accepted'] if stage == 'saved' and method != 'B0' else '—'
            stage_label = '검사 전' if stage == 'proposed' else '저장 후'
            page.append(f'<tr><td>{stage_label}</td><td>{LABELS[method]}</td>{text}<td>{accepted}</td></tr>')
    page.append('</tbody></table><p>국소 검사를 통과한 결과와 전신 피부 인증은 별도입니다.</p>')
    for row in report['rows']:
        cid = row['case_id']
        links = ' · '.join(f'<a href="native/{name}/{cid}.fbx">{LABELS[name]} FBX</a>' for name in METHODS)
        page.append(f'<section><h2>{cid} · {escape(row["representation"])}</h2><a href="images/{cid}.png">'
                    f'<img loading="lazy" src="images/{cid}.png" alt="{cid} 비교"></a>'
                    f'<p>{links} · <a href="observations/{cid}.json">VLM 응답</a></p></section>')
    (run / 'review.html').write_text('\n'.join(page) + '\n')


def run_report(p1, run, gt_audit):
    cohort = load_json(run / 'cohort.json')
    source_manifest = load_json(p1 / 'manifest.json')
    if digest(p1 / 'manifest.json') != cohort['p1_manifest_sha256']:
        raise ValueError('P0 manifest changed')
    cases = {row['case_id']: row for row in source_manifest['rows']}
    oracle = {row['case_id']: row for row in load_json(p1 / 'p1-plans.json')['rows']}
    auto = load_json(run / 'plans.json')['cases']
    target = {row['case_id']: row for row in load_json(gt_audit)}
    model = load_json(run / 'model.json')
    rows, receipt = [], []
    for cid in cohort['validation']:
        case = cases[cid]
        source = load_json(p1 / 'p1-native' / f'{cid}-source.json')
        gt = target[cid]
        if case['split'] != 'validation' or gt['sha256'] != case['ground_truth_sha256']:
            raise ValueError('target does not belong to the frozen validation cohort')
        source_sha = source['source_sha256']
        if (auto[cid]['source_sha256'] != source_sha or oracle[cid]['source_sha256'] != source_sha
                or digest(p1 / source['source']) != source_sha):
            raise ValueError('methods did not start from the same FBX')
        data = problem(case, source, gt)
        names, _, _, _, base, *_ = data
        start = score(base, data)
        row = dict(case_id=cid, representation=case['representation'], source_sha256=source_sha,
                   observation_status=auto[cid]['status'], leg_roles=auto[cid].get('leg_roles'),
                   head_goal=auto[cid].get('head_goal'), audit={}, proposed={'B0': start}, saved={'B0': start})
        observation = load_json(run / 'observations' / f'{cid}.json')
        receipt.append(dict(case_id=cid, model=observation['model'], response_id=observation['response_id']))
        for method in METHODS:
            plan = oracle[cid]['profiles']['limbs_head'] if method == 'oracle' else auto[cid][method]
            audit = load_json(run / 'native' / method / f'{cid}.json')
            path = run / 'native' / method / f'{cid}.fbx'
            if (audit['source_sha256'] != source_sha or digest(path) != audit['output_sha256']
                    or (not kept(audit) and audit['output_sha256'] != source_sha)):
                raise ValueError('saved pose binding differs')
            row['proposed'][method] = score(planned_pose(data, plan), data)
            row['saved'][method] = score(np.asarray([audit['matrices'][n] for n in names]), data)
            row['audit'][method] = {key: value for key, value in audit.items() if key != 'matrices'}
        rows.append(row)
    summary = {stage: {method: summarize([row[stage][method] for row in rows])
                       for method in ('B0', *METHODS)} for stage in ('proposed', 'saved')}
    gate = {}
    for method in METHODS:
        audits = [row['audit'][method] for row in rows]
        gate[method] = dict(
            accepted=sum(kept(a) for a in audits),
            partial_accepted=sum(bool(a.get('group_fractions')) for a in audits),
            fractions=dict(Counter(str(a['fraction']) for a in audits)),
            no_supported_edit=sum(not a['attempts'] for a in audits),
            rejected_after_attempts=sum(not kept(a) and bool(a['attempts']) for a in audits),
            full_skin_certified=sum(a['full_skin_certified'] for a in audits),
            failure_reasons=dict(Counter(reason for a in audits if not kept(a) and a['attempts']
                                         for reason in a['attempts'][-1]['reasons'])),
        )
    report = dict(training={key: model[key] for key in ('development_used', 'development_skipped', 'method')},
                  validation_case_ids=cohort['validation'], evaluation_note='Development/validation diagnostic, not blind P6.',
                  actual_vlm_responses=receipt, summary=summary, gate=gate, rows=rows)
    save_json(run / 'comparison.json', report)
    (run / 'images').mkdir(exist_ok=True)
    strips = [draw_case(p1, run, cases[row['case_id']], row) for row in rows]
    overview = Image.new('RGB', (strips[0].width, sum(image.height for image in strips)), 'white')
    y = 0
    for strip in strips:
        overview.paste(strip, (0, y))
        y += strip.height
    overview.save(run / 'overview.png')
    write_review(run, report)
    print('P2_P4', len(rows), 'cases', {key: value['accepted'] for key, value in gate.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'run', 'gt-audit'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run_report(args.p1, args.run, args.gt_audit)
