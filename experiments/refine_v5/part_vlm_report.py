"""Compare saved VLM-goal FBX, the detector replay, and original B0 against GT."""

import argparse
from collections import Counter
from html import escape
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.ab_evaluate import error_deg, point_map
from experiments.refine_v5.depth_ten_report import review_images
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.part_quality import segment_cos
from experiments.refine_v5.part_vlm_observe import CHAINS


LABELS = {'B0': 'B0', 'detector': 'Detector / one arm', 'proposal': 'VLM proposal', 'vlm': 'VLM saved', 'GT': 'Ground truth'}
METHODS = tuple(LABELS)[:-1]


def score(pose, truth, mask):
    actual, gt = point_map(pose), point_map(truth)
    parts = {}
    for index, (limb, (chain, _)) in enumerate(CHAINS.items()):
        actual_chain = np.asarray([actual[name] for name in chain])
        gt_chain = np.asarray([gt[name] for name in chain])
        actual_axes, gt_axes = np.diff(actual_chain, axis=0), np.diff(gt_chain, axis=0)
        angles = [error_deg(a / np.linalg.norm(a), g / np.linalg.norm(g))
                  if mask[2 * index + part] else None
                  for part, (a, g) in enumerate(zip(actual_axes, gt_axes, strict=True))]
        a_bend, gt_bend = [float(np.degrees(np.arccos(segment_cos(points))))
                           for points in (actual_chain, gt_chain)]
        parts[limb] = dict(segment_error_deg=angles, bend_deg=a_bend, gt_bend_deg=gt_bend,
                           bend_error_deg=abs(a_bend - gt_bend) if all(mask[2*index:2*index+2]) else None,
                           endpoint_error_torso=float(np.linalg.norm(actual_chain[-1] - gt_chain[-1]))
                           if all(mask[2*index:2*index+2]) else None)
    return parts


def mean(values):
    numbers = [value for value in values if value is not None]
    return float(np.mean(numbers)) if numbers else None


def angle(parts, category=None):
    return mean(value for key, row in parts.items() if category is None or key.endswith(category)
                for value in row['segment_error_deg'])


def compare(root, control, out):
    cases = load_json(root / 'manifest.json')['cases']
    for folder in ('images', 'fbx', 'observations'):
        (out / folder).mkdir(parents=True, exist_ok=False)
    rows, lookup, image_rows = [], {}, []
    with TemporaryDirectory() as temporary:
        stage = Path(temporary) / 'native'
        stage.mkdir()
        for case in cases:
            cid = case['case_id']
            folder, previous = root / cid, control / cid
            baseline = load_json(previous / 'detector/audit.json')
            vlm = load_json(folder / 'vlm-audit.json')
            gt = load_json(previous / 'gt.json')
            response = load_json(folder / 'vlm-response.json')
            plan = load_json(folder / 'vlm-plan.json')
            if load_json(previous / 'input.json') != case:
                raise ValueError('detector control differs from the frozen B0')
            paths = dict(B0=(case['source'], case['source_sha256']),
                         detector=(previous / 'detector' / f'{cid}-part-goals-head-limbs.fbx', baseline['output_sha256']),
                         vlm=(folder / 'vlm.fbx', vlm['sha256']))
            for method, (path, sha) in paths.items():
                if digest(path) != sha:
                    raise ValueError('FBX changed after auditing')
                if method == 'vlm' and not vlm['accepted'] and sha != case['source_sha256']:
                    raise ValueError('fallback changed B0 bytes')
                shutil.copyfile(path, out / 'fbx' / f'{cid}-{method}.fbx')
            shutil.copyfile(folder / 'vlm-response.json', out / 'observations' / f'{cid}.json')
            meshes = dict(B0=root / f'{cid}-B0.npz', detector=previous / 'detector/final.npz',
                          vlm=folder / 'vlm.npz', GT=previous / 'gt.npz')
            proposed = folder / 'vlm-proposal.npz'
            meshes['proposal'] = proposed if proposed.exists() else meshes['B0']
            native = dict(B0=case, detector=baseline, vlm=vlm, GT=gt,
                          proposal={'bones': vlm['proposed_bones']} if vlm['proposed_bones'] else case)
            for method, path in meshes.items():
                (stage / f'{cid}-{method}.npz').symlink_to(path.resolve())
                lookup[cid, method] = native[method]
            metrics = {method: score(native[method]['bones'], gt['bones'], case['part_mask']) for method in METHODS}
            for method, parts in metrics.items():
                image_rows.append(dict(case_id=cid, method=method, chain_error_deg=angle(parts),
                                       status='fallback_exact_B0' if ((method == 'vlm' and not vlm['accepted']) or
                                                                     (method == 'proposal' and not vlm['proposed_bones']))
                                       else 'reference',
                                       camera_status=None))
            case['cameras'] = dict.fromkeys(('ortho_fixed', *METHODS), case['camera'])
            rows.append(dict(case_id=cid, split=case['split'], representation=case['representation'],
                             metrics=metrics, detector_applied=baseline['accepted'], vlm_applied=vlm['accepted'],
                             model=response['model'], usage=response['usage'], response_id=response['response_id'],
                             statuses=plan['statuses'], cues=response['goals'], goal_fit=plan['result'],
                             native={key: value for key, value in vlm.items() if key not in ('bones', 'proposed_bones')},
                             reference_contact=load_json(folder / 'reference-contact.json'), gt_sha256=gt['sha256']))
        review_images(stage.parent, out, cases, image_rows, lookup, labels=LABELS, short=LABELS)
    summary = {}
    for method in METHODS:
        parts = [r['metrics'][method] for r in rows]
        pooled = {f'{index}/{key}': part for index, record in enumerate(parts) for key, part in record.items()}
        summary[method] = dict(
            segment_deg=angle(pooled), arms_deg=angle(pooled, 'arm'), legs_deg=angle(pooled, 'leg'),
            elbow_bend_mae=mean(part['bend_error_deg'] for key, part in pooled.items() if key.endswith('arm')),
            knee_bend_mae=mean(part['bend_error_deg'] for key, part in pooled.items() if key.endswith('leg')),
            endpoint_error_torso=mean(part['endpoint_error_torso'] for part in pooled.values()),
        )
    receipt = dict(model=sorted({r['model'] for r in rows}), actual_responses=sum(bool(r['response_id']) for r in rows),
                   total_tokens=sum(r['usage']['total_tokens'] for r in rows),
                   proposals=sum(bool(r['goal_fit'] and r['goal_fit']['status'] == 'proposal') for r in rows),
                   accepted=sum(r['vlm_applied'] for r in rows),
                   control_accepted=sum(r['detector_applied'] for r in rows),
                   reasons=dict(Counter(reason for r in rows for reason in r['native']['reasons'])),
                   full_skin=dict(Counter(r['native']['full_skin_status'] for r in rows)),
                   full_skin_certified=sum(r['native']['full_skin_certified'] for r in rows))
    save_json(out / 'comparison.json', dict(summary=summary, receipt=receipt, rows=rows,
              metrics_scope='Pooled segments visible in the prior rough review. Same B0 rig for both methods.',
              selection='Nine earlier replay cuts with visible lower-body evidence and one crop control. Development cases.',
              ground_truth_used_for_selection=False))
    html(out, rows, summary, receipt)


def html(out, rows, summary, receipt):
    lines = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Refine v5 · VLM 팔다리 목표 10컷</title>',
             '<style>body{max-width:1260px;margin:24px auto;padding:0 14px;font:16px system-ui;color:#182431}'
             'table{border-collapse:collapse}td,th{padding:7px 12px;border-bottom:1px solid #ccc;text-align:left}'
             'img{display:block;width:100%}section{border-top:1px solid #ccc;margin-top:18px}a{color:#17558c}</style>',
             '<h1>Refine v5 · VLM 팔다리 목표 10컷</h1>',
             '<p>위는 몸통 기준 정면, 아래는 측면입니다. 각도는 독립 검수에서 보인 뼈 방향의 정답 대비 오차입니다.</p>',
              '<p>VLM proposal은 피부 검사를 통과하지 못한 보정안도 보여줍니다. VLM saved는 검사 후 저장한 포즈입니다.</p>',
             f'<p>실제 VLM 응답 {receipt["actual_responses"]}회 · 보정 제안 {receipt["proposals"]}컷 · 저장 적용 {receipt["accepted"]}컷</p>',
             '<p><a href="overview.png">전체 비교 이미지</a> · <a href="comparison.json">수치와 근거 JSON</a></p>',
             '<table><tr><th>방법</th><th>전체</th><th>팔</th><th>다리</th><th>팔꿈치 굽힘</th><th>무릎 굽힘</th></tr>']
    for method, data in summary.items():
        cells = ''.join(f'<td>{data[key]:.1f}°</td>' for key in
                        ('segment_deg', 'arms_deg', 'legs_deg', 'elbow_bend_mae', 'knee_bend_mae'))
        lines.append(f'<tr><th>{LABELS[method]}</th>{cells}</tr>')
    lines.append('</table>')
    for row in rows:
        cid = row['case_id']
        status = '보정 적용' if row['vlm_applied'] else 'B0 유지'
        lines.append(f'<section><h2>{cid} · {status}</h2><a href="images/{cid}.png"><img src="images/{cid}.png" '
                     f'loading="lazy" alt="{cid} 비교"></a>')
        lines.append(f'<p><a href="images/{cid}-rough-camera.png">러프 시점 투영</a> · '
                     f'<a href="observations/{cid}.json">VLM 관측</a> · <a href="fbx/{cid}-vlm.fbx">저장 FBX</a></p>')
        if row['native']['reasons']:
            lines.append('<p>' + escape(', '.join(row['native']['reasons'])) + '</p>')
        lines.append('</section>')
    (out / 'review.html').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'control', 'out'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    args = parser.parse_args()
    compare(args.root, args.control, args.out)
