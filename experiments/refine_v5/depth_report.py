"""Compare sealed depth candidates against re-read native FBXs and held-out GT geometry."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.ab_evaluate import error_deg, palm_normal, point_map
from experiments.refine_v5.depth_evidence import has_cue
from experiments.refine_v5.learning_data import BONES, axes, digest, frame
from experiments.refine_v5.render_review import render


def compare(root, cases):
    native = {(r['case_id'], r['method']): r for r in load_json(root / 'native/audit.json')}
    measured = {(r['case_id'], r['method']): r for r in load_json(root / 'depth-readback.json')}
    frozen = load_json(root / 'frozen.json')
    if digest(root / 'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('sealed depth inputs changed')
    records = []
    for plan in frozen['results']:
        case_id, method = plan['case_id'], plan['method']
        case = cases[case_id]
        original = native[case_id, 'B0']
        actual = native[case_id, method]
        gt = native[case_id, 'GT']
        if digest(actual['path']) != actual['sha256'] or digest(gt['path']) != gt['sha256']:
            raise ValueError('audited FBX changed')
        current_axes, gt_axes = axes(actual['bones']), axes(gt['bones'])
        current_points, gt_points = point_map(actual['bones']), point_map(gt['bones'])
        changed = case['depth_evidence']['movable']
        errors = {}
        for name in changed:
            i = BONES.index(name)
            errors[name] = dict(direction_deg=error_deg(current_axes[i, 1], gt_axes[i, 1]),
                                position_torso=float(np.linalg.norm(current_points[name] - gt_points[name])))
        if has_cue(case, 'waist_side_contact') or has_cue(case, 'head_behind'):
            errors['RightHand']['palm_normal_deg'] = error_deg(palm_normal(actual['bones'], 'Right'),
                                                              palm_normal(gt['bones'], 'Right'))
        by_name = {b['name']: np.asarray(b['matrix']) for b in actual['bones']}
        support_drift = None
        if 'locked_support' in case['depth_evidence']:
            support_drift = 1000 * max(
                np.linalg.norm(np.asarray(b['matrix'])[:3, 3] - by_name[b['name']][:3, 3])
                for b in original['bones'] if b['name'].startswith(('RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe'))
            )
        metric = measured[case_id, method]
        records.append(dict(case_id=case_id, method=method, status=plan['status'],
                            selected_profile=plan.get('selected_profile'), depth=metric,
                            part_errors_vs_gt=errors,
                            chain_mean_direction_deg=float(np.mean([errors[n]['direction_deg'] for n in changed[:2]])),
                            ground_truth_contacts=gt['contact'], contact=actual['contact'],
                            native=actual['native'], max_cloud_error_m=actual['max_cloud_error_m'],
                            right_support_readback_max_joint_drift_mm=float(support_drift) if support_drift is not None else None,
                            skin_regressions=actual['skin_nonregression_failures'],
                            full_skin_status=actual['full_skin']['status'],
                            full_skin_certified=actual['full_skin']['complete_collision_certificate'],
                            artifact=actual['path'], sha256=actual['sha256']))
    selected = [r for r in records if r['method'] == 'depth_selected']
    summary = dict(cases=len(selected), selected_statuses=dict(Counter(r['status'] for r in selected)),
                   all_depth_cues_pass=sum(all(c['passes'] for c in r['depth']['cues']) for r in selected),
                   full_skin_certified=sum(r['full_skin_certified'] for r in selected))
    save_json(root / 'depth-comparison.json', dict(summary=summary, records=records))
    print(summary, flush=True)
    return native, records


def review(root, cases, native, metrics):
    methods = ('B0', 'perspective', 'depth_selected', 'GT')
    titles = ('Input rough', 'Original B0', 'Previous perspective', 'Selected depth fit', 'GT FBX')
    cell_w, cell_h = 222, 420
    canvas = Image.new('RGB', (len(titles) * cell_w, 55 + cell_h * len(cases)), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    for col, title in enumerate(titles):
        draw.text((col * cell_w + 8, 9), title, fill='#202020', font=font)
    draw.text((8, 31), 'Top: body-frame front. Bottom: side view, with the front of the body toward the right.',
              fill='#454545', font=font)
    by_metric = {(row['case_id'], row['method']): row for row in metrics}
    for index, (case_id, case) in enumerate(cases.items()):
        y = 55 + cell_h * index
        with Image.open(case['image']) as image:
            rough = image.convert('RGB')
        rough.thumbnail((cell_w - 12, cell_h - 54))
        canvas.paste(rough, ((cell_w - rough.width) // 2, y + 29))
        draw.text((8, y + 5), case_id, fill='#202020', font=font)
        views = []
        for method in methods:
            bones = native[case_id, method]['bones']
            _, center, scale, basis = frame(bones)
            with np.load(root / 'native' / f'{case_id}-{method}.npz') as mesh:
                vertices = (mesh['vertices'] - center) @ basis / scale
                views.append((vertices, mesh['faces']))
        size = (cell_w, (cell_h - 30) // 2)
        for view_index, yaw in enumerate((0., 90.)):
            rotation = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
            projected = np.concatenate([(vertices @ rotation.T)[:, :2] * [1, -1] for vertices, _ in views])
            lo, hi = projected.min(axis=0), projected.max(axis=0)
            scale = min((size[0] - 14) / max(1e-5, hi[0] - lo[0]), (size[1] - 10) / max(1e-5, hi[1] - lo[1]))
            offset = np.asarray(size) / 2 - (lo + hi) * scale / 2
            for col, (vertices, faces) in enumerate(views, 1):
                image = render(vertices, faces, size=size, yaw=yaw, framing=(scale, offset))
                canvas.paste(image, (col * cell_w, y + 26 + view_index * size[1]))
        for col, method in enumerate(methods[:-1], 1):
            row = by_metric[case_id, method]
            cues = row['depth']['cues']
            caption = f"{sum(c['passes'] for c in cues)}/{len(cues)} depth"
            if row['status'] == 'fallback_exact_B0':
                caption += '  |  B0 fallback'
            draw.text((col * cell_w + 8, y + 5), caption, fill='#202020', font=font)
        draw.line((8, y + cell_h - 2, canvas.width - 8, y + cell_h - 2), fill='#dddddd')
    canvas.save(root / 'depth-review.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    cases = {r['case_id']: r for r in load_json(args.root / 'manifest.json')['cases']}
    native, records = compare(args.root, cases)
    review(args.root, cases, native, records)


if __name__ == '__main__':
    main()
