"""Compare the frozen part-goal FBXs and render the original rough beside them."""

import argparse
from html import escape
from pathlib import Path
import shutil

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.ab_evaluate import error_deg, point_map
from experiments.refine_v5.depth_ten_report import review_images
from experiments.refine_v5.learning_data import BONES, axes, digest
from experiments.refine_v5.part_quality import segment_cos


LABELS = {'B0': '원본 B0', 'ordinal': '깊이 순서 + 기존 피팅', 'fused': '굽힘·단축·접촉 추가',
          'learned': '추가 관측 + 학습 시작값', 'GT': '정답 FBX'}
SHORT = {'B0': 'B0', 'ordinal': 'Ordinal fit', 'fused': 'Part goals', 'learned': 'Part goals + learned start',
         'GT': 'Ground truth'}


def bundle(root, out, head_root):
    manifest = root / 'manifest.json'
    data = load_json(manifest)
    cases = data['cases']
    frozen = load_json(root / 'frozen.json')
    if digest(manifest) != frozen['manifest_sha256']:
        raise ValueError('manifest changed after freeze')
    if 'learned_prior' not in data:
        labels = {name: label for name, label in LABELS.items() if name != 'learned'}
    else:
        labels = LABELS
    methods = [name for name in labels if name != 'GT']
    native = {(r['case_id'], r['method']): r for r in load_json(root / 'native/audit.json')}
    readback = {(r['case_id'], r['method']): r for r in load_json(root / 'depth-ten-readback.json')}
    selected = {(r['case_id'], r['method']): r for r in load_json(root / 'selection.json')}
    if len(native) != len(cases) * len(labels) or len(readback) != len(cases) * len(methods):
        raise ValueError('incomplete audit')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'fbx').mkdir()
    (out / 'images').mkdir()
    rows, files = [], {}
    for case in cases:
        cid = case['case_id']
        gt = native[cid, 'GT']
        chain = case['depth_evidence']['movable']
        goal = case['part_goal']
        gt_axes, gt_points = axes(gt['bones']), point_map(gt['bones'])
        gt_bend = float(np.degrees(np.arccos(segment_cos([gt_points[name] for name in chain]))))
        case['cameras'].update({method: case['camera'] for method in methods})
        case['cameras']['ortho_fixed'] = case['camera']
        for method in labels:
            actual = native[cid, method]
            source = Path(actual['path'])
            key = 'B0' if actual.get('status') == 'fallback_exact_B0' else method
            target = out / 'fbx' / f'{cid}-{key}.fbx'
            if not target.exists():
                shutil.copyfile(source, target)
            if digest(target) != actual['sha256']:
                raise ValueError('audited FBX changed')
            files[cid, method] = str(target.relative_to(out))
            if method == 'GT':
                continue
            if key == 'B0' and actual['sha256'] != case['source_sha256']:
                raise ValueError('fallback bytes differ from B0')
            if key != 'B0' and actual['skin_nonregression_failures']:
                raise ValueError('published regional skin regression')
            actual_axes, actual_points = axes(actual['bones']), point_map(actual['bones'])
            direction = [error_deg(actual_axes[BONES.index(n), 1], gt_axes[BONES.index(n), 1]) for n in chain[:2]]
            end = [np.linalg.norm(actual_points[n] - gt_points[n]) for n in chain[1:]]
            detail = readback[cid, method]
            rows.append(dict(case_id=cid, method=method, status=actual['status'], camera_status=case['camera']['status'],
                             chain_error_deg=float(np.mean(direction)), endpoint_3d_nme=float(np.mean(end)),
                             gt_bend_deg=gt_bend, observation_abstention=goal['reason'],
                             local_skin_failures=actual['skin_nonregression_failures'],
                             full_skin_certified=actual['full_skin']['complete_collision_certificate'],
                             depth=detail, contact=actual['contact'], fbx=files[cid, method],
                             rejected=selected[cid, method].get('rejected', [])))
    summary = {}
    for method in methods:
        group = [row for row in rows if row['method'] == method]
        summary[method] = dict(
            direction_deg=float(np.mean([r['chain_error_deg'] for r in group])),
            endpoint_3d_nme=float(np.mean([r['endpoint_3d_nme'] for r in group])),
            depth_pass=sum(all(c['passes'] for c in r['depth']['cues']) for r in group),
            fallback=sum(r['status'] == 'fallback_exact_B0' for r in group),
            certified=sum(r['full_skin_certified'] for r in group),
        )
    head_rows = []
    if head_root:
        head_rows = load_json(head_root / 'head-audit.json')
        for row in head_rows:
            if row['status'] == 'diagnostic':
                target = out / 'fbx' / row['fbx']
                shutil.copyfile(head_root / row['fbx'], target)
                if digest(target) != row['sha256']:
                    raise ValueError('head FBX changed')
                row['fbx'] = str(target.relative_to(out))
    save_json(out / 'comparison.json', dict(
        summary=summary, rows=rows, head=head_rows, learned_prior=data.get('learned_prior'),
        scope='Ten previously exposed cases, rough reviewer cues, offline comparison, held-out test unused',
        ground_truth_opened_after_freeze=True,
        observation_note='Previously exposed 043 anatomical correspondence was blocked before fitting.',
    ))
    review_images(root, out, cases, rows, native, labels=labels, short=SHORT)
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Refine v5 파츠 목표 검증</title>',
            '<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}img{max-width:100%}td,th{padding:6px 12px;text-align:left}section{border-top:1px solid #ddd;margin-top:1rem}</style>',
            '<h1>Refine v5 파츠 목표 검증</h1><p>같은 러프·B0·카메라에서 비교했습니다. 정면은 위, 측면은 아래입니다. 관측은 검수 입력을 사용했고, 모든 결과는 개발 검수용입니다.</p>',
            '<p><a href="comparison.json">원시 지표</a> · <a href="overview.png">전체 비교 이미지</a></p>',
            '<table><tr><th>방식</th><th>방향 오차</th><th>끝점 NME</th><th>깊이 조건 충족</th><th>B0 복귀</th></tr>']
    for method, value in summary.items():
        page.append(f'<tr><td>{labels[method]}</td><td>{value["direction_deg"]:.2f}°</td>'
                    f'<td>{value["endpoint_3d_nme"]:.3f}</td><td>{value["depth_pass"]}/{len(cases)}</td>'
                    f'<td>{value["fallback"]}</td></tr>')
    page.append('</table>')
    for row in head_rows:
        if row['status'] == 'diagnostic' and 'screen_up' in row['before_deg']:
            page.append(f'<p>{row["case_id"]}의 고개 단독 보정 '
                        f'{row["before_deg"]["screen_up"]:.2f}° → {row["after_deg"]["screen_up"]:.3f}° '
                        f'<a href="{row["fbx"]}">저장 FBX</a></p>')
    for case in cases:
        cid = case['case_id']
        links = ' · '.join(f'<a href="{files[cid, method]}">{labels[method]}</a>' for method in labels)
        page.append(f'<section><h2>{escape(cid)}</h2><a href="images/{cid}.png"><img src="images/{cid}.png" alt="{cid} 비교"></a>'
                    f'<p>{links}</p></section>')
    page.append('</html>')
    (out / 'review.html').write_text('\n'.join(page))
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--head', type=Path)
    args = parser.parse_args()
    bundle(args.root, args.out, args.head)


if __name__ == '__main__':
    main()
