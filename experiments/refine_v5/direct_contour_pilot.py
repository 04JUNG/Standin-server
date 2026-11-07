"""Direct rough-to-projected-skin comparison on reviewed foreground windows.

Ranks a frozen P5 proposal pool with rough-only registration. Masks cover
four reviewed roughs.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from experiments.refine_v5.learning_data import digest, frame, source_camera

VARIANTS = {
    'geometric': 'iterative-geometric',
    'observation': 'iterative',
    'balanced': 'iterative-balanced',
}
CURRENT = 'balanced:experimental30'
COCO_ANCHORS = [5, 6, 11, 12]
ANCHORS = ['LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg']


def read(path):
    return json.loads(path.read_text())


def build_pool(run, previous, config):
    seeds = {row['case_id']: row for row in read(previous / 'all-seeds/seeds.json')}
    rigs = {row['case_id']: row for row in read(run / 'native-data/records.json') if row['status'] == 'ready'}
    variant_rows = {name: read(run / directory / 'results.json') for name, directory in VARIANTS.items()}
    cases = []
    for case_id, observations in config['cases'].items():
        rig, seed = rigs[case_id], seeds[case_id]
        if rig['split'] not in ('development', 'validation'):
            raise ValueError('sealed test case in contour pilot')
        source_sha = digest(rig['base_fbx'])
        image_sha = digest(seed['image'])
        if image_sha != seed['image_sha256']:
            raise ValueError('rough bytes changed')
        by_name, _, _, body = frame(rig['bones'])
        # Same orthographic view convention as P5. Freeze this for every candidate.
        camera = source_camera(rig['bones'], seed['view']) @ body.T
        world_anchors = np.array([np.asarray(by_name[name]['matrix'])[:3, 3] for name in ANCHORS])
        base_heads = {name: np.asarray(bone['matrix'])[:3, 3].tolist() for name, bone in by_name.items()}
        candidates = [dict(name='B0', active_bones=[], delta_rotvec=[], expected_heads=base_heads)]
        current = 'B0'
        for variant, rows in variant_rows.items():
            for row in rows:
                if row['case_id'] != case_id or row['stage'] != 'selected_target' or row['rounds'] != 1:
                    continue
                if row['source_sha256'] != source_sha:
                    raise ValueError('stale proposal')
                if row['status'] != 'proposal':
                    continue
                name = f"{variant}:{row['profile']}"
                candidates.append(dict(name=name, active_bones=row['active_bones'], delta_rotvec=row['delta_rotvec'],
                                       expected_heads=row['planned_heads']))
                if name == CURRENT:
                    current = name
        for index, candidate in enumerate(candidates):
            candidate['mesh'] = f'{case_id}-{index}.npz'
        cases.append(dict(case_id=case_id, split=rig['split'], source=rig['base_fbx'], source_sha256=source_sha,
                          image=seed['image'], image_sha256=image_sha, view=seed['view'], camera=camera.tolist(),
                          rough_anchors=np.asarray(seed['keypoints'])[COCO_ANCHORS].tolist(),
                          anchor_scores=np.asarray(seed['scores'])[COCO_ANCHORS].tolist(),
                          source_anchors=world_anchors.tolist(), current=current, candidates=candidates, **observations))
    return cases


def foreground(image, mode):
    if mode == 'orange':
        blue, _, red = cv2.split(image.astype(np.int16))
        mask = ((red > 190) & (red - blue > 45)).astype(np.uint8)
    elif mode in ('gray_on_white', 'outlined_on_white'):
        mask = (image.min(axis=2) < 235).astype(np.uint8)
    else:
        raise ValueError('unsupported foreground extraction')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return np.zeros(mask.shape, np.uint8)
    largest = (labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    if mode != 'outlined_on_white':
        return largest
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(np.zeros_like(largest), contours, -1, 1, cv2.FILLED)


def align(case, resize):
    source = (np.asarray(case['source_anchors']) @ np.asarray(case['camera']).T)[:, :2]
    rough = np.asarray(case['rough_anchors']) * resize
    if min(case['anchor_scores']) < .3:
        return None, 'unreliable_torso_anchors'
    # Two torso centers establish scale and offset. The fit has no roll or yaw freedom.
    src_vector = source[2:].mean(axis=0) - source[:2].mean(axis=0)
    rough_vector = rough[2:].mean(axis=0) - rough[:2].mean(axis=0)
    if min(np.linalg.norm(src_vector), np.linalg.norm(rough_vector)) < 1e-5:
        return None, 'degenerate_torso_anchors'
    scale = float(src_vector @ rough_vector / (src_vector @ src_vector))
    mismatch = float(np.linalg.norm(src_vector * scale - rough_vector) / np.linalg.norm(rough_vector))
    if scale <= 0 or mismatch > .25:
        return None, 'torso_projection_mismatch'
    return (scale, rough.mean(axis=0) - source.mean(axis=0) * scale, mismatch), None


def rasterize(path, camera, scale, offset, shape):
    with np.load(path) as mesh:
        xy = (mesh['vertices'] @ camera.T)[:, :2] * scale + offset
        triangles = np.round(xy[mesh['faces']] * 16).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 1, shift=4)
    return mask


def score(rough, candidate, windows):
    return [float(1 - np.count_nonzero((rough & candidate)[window]) /
                  np.count_nonzero((rough | candidate)[window])) for window in windows]


def overlay(image, rough, candidate, boxes):
    view = (image.astype(float) * .45 + 255 * .55).astype(np.uint8)
    for mask, color in [(rough, (215, 78, 28)), (candidate, (34, 151, 31))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, color, 1, cv2.LINE_AA)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(view, (x0, y0), (x1, y1), (124, 124, 124), 1)
    return view


def evaluate(case, out):
    if digest(case['image']) != case['image_sha256']:
        raise ValueError('rough bytes changed')
    image = cv2.imread(case['image'])
    if image is None:
        raise ValueError('unable to decode rough image')
    resize = 420 / max(image.shape[:2])
    image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
    rough = foreground(image, case['foreground'])
    h, w = rough.shape
    boxes = [(round(x0*w), round(y0*h), round(x1*w), round(y1*h)) for x0, y0, x1, y1 in case['windows']]
    windows = [np.s_[y0:y1, x0:x1] for x0,y0,x1,y1 in boxes]
    alignment, reason = align(case, resize)
    if reason is None and any(not .04 < rough[window].mean() < .92 for window in windows):
        reason = 'insufficient_foreground_boundary_in_window'
    result = dict(case_id=case['case_id'], split=case['split'], view=case['view'], current=case['current'])
    if reason:
        return dict(result, status='abstained', reason=reason)
    scale, offset, mismatch = alignment
    masks = {}
    losses = {}
    for candidate in case['candidates']:
        name = candidate['name']
        masks[name] = rasterize(out / candidate['mesh'], np.asarray(case['camera']), scale, offset, rough.shape)
        losses[name] = score(rough, masks[name], windows)
    current = case['current']
    best = min(losses, key=lambda name: np.mean(losses[name]))
    # Apply direct evidence only when it improves the current candidate with no visible window regression.
    eligible = [name for name, values in losses.items()
                if np.mean(values) <= np.mean(losses[current]) - .02
                and np.all(np.asarray(values) <= np.asarray(losses[current]) + .02)]
    selected = min(eligible, key=lambda name: np.mean(losses[name])) if eligible else current
    for label, name in [('b0', 'B0'), ('current', current), ('selected', selected)]:
        cv2.imwrite(str(out / f"{case['case_id']}-{label}.png"), overlay(image, rough, masks[name], boxes))
    return dict(result, status='scored', anchor_mismatch=mismatch, best_image=best, selected=selected, losses=losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['run', 'previous', 'out', 'blender']:
        parser.add_argument(f'--{arg}', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = read(repo / 'evaluation/refine_v5/direct_contour_pilot.json')
    cases = build_pool(args.run, args.previous, config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'pool.json').write_text(json.dumps(cases, indent=2))
    script = Path(__file__).with_name('export_contour_meshes.py')
    with (args.out / 'blender.log').open('w') as log:
        subprocess.run([str(args.blender), '-b', '--python-exit-code', '1', '--python', str(script), '--', str(args.out)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = [evaluate(case, args.out) for case in cases]
    (args.out / 'image-scores.json').write_text(json.dumps(dict(metric='per_window_1_minus_silhouette_IoU', rows=rows), indent=2))
    for row in rows:
        print(row['case_id'], row['status'], row.get('selected', row.get('reason')))


if __name__ == '__main__':
    main()
