"""Freeze common rough-only inputs and apply bounded rotations from original B0."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest, frame, source_camera, visibility


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare_manifest(run, previous, body, output):
    repo = Path(__file__).resolve().parents[2]
    fixture = load_json(repo / 'evaluation/refine_v5/camera_vlm_ab.json')
    reviews = {r['case_id']: r for r in load_json(repo / 'evaluation/refine_v5/visual_review_v2.json')['records']}
    seeds = {r['case_id']: r for r in load_json(previous / 'all-seeds/seeds.json')}
    rigs = {r['case_id']: r for r in load_json(run / 'native-data/records.json') if r['status'] == 'ready'}
    body_sha = digest(body)
    prior_bindings = {
        r['case_id']: r for r in load_json(run / 'iterative-balanced/manifest.json')['records']
    }
    cases = []
    for case_id, notes in fixture['cases'].items():
        rig, seed, review = rigs[case_id], seeds[case_id], reviews[case_id]
        source_sha = digest(rig['base_fbx'])
        binding = prior_bindings[case_id]
        if source_sha != binding['source_sha256'] or body_sha != binding['profile']['body_asset_sha256']:
            raise ValueError(f'{case_id} B0 or asset differs from previous v5')
        if digest(seed['image']) != seed['image_sha256']:
            raise ValueError(f'{case_id} stale rough')
        if rig['split'] not in ('development', 'validation') or review['target_identity'] != 'single_depicted_target':
            raise ValueError(f'{case_id} unsupported trial target')
        with Image.open(seed['image']) as image:
            dimensions = list(image.size)
        _, _, _, basis = frame(rig['bones'])
        camera = source_camera(rig['bones'], seed['view']) @ basis.T
        limits = {}
        for side, points in notes['arms'].items():
            limits[side + 'Arm'] = 30.
            if len(points) == 3:
                limits[side + 'ForeArm'] = 30.
                limits[side + 'Hand'] = 12.
        cases.append({
            **notes, 'case_id': case_id, 'split': rig['split'], 'representation': review['representation'],
            'image': seed['image'], 'image_sha256': seed['image_sha256'], 'dimensions': dimensions,
            'source': rig['base_fbx'], 'source_sha256': source_sha, 'body_asset_sha256': body_sha,
            'bones': rig['bones'], 'seed_view': seed['view'], 'base_camera': camera.tolist(),
            'torso_anchors': np.asarray(seed['keypoints'])[[5, 6, 11, 12]].tolist(),
            'torso_scores': np.asarray(seed['scores'])[[5, 6, 11, 12]].tolist(),
            'part_mask': visibility(review).tolist(), 'limits_deg': limits,
            'ownership_authority': 'offline_single_target_review',
        })
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'manifest.json', dict(schema='v5-camera-vlm-ab-v1', cases=cases))


def named_pose(case, edits, *, historical=False):
    """Apply post-multiplied B0-local rotvecs. The original hierarchy stays fixed."""
    names, lookup, parents, local, base = prepare(case)
    active = list(edits)
    angles = np.array(list(edits.values()), dtype=float).reshape(-1, 3)
    if not np.isfinite(angles).all() or set(active) - set(lookup):
        raise ValueError('invalid rotation proposal')
    if not historical:
        if set(active) - set(case['limits_deg']):
            raise ValueError('rotation on a locked joint')
        if any(np.linalg.norm(v) > case['limits_deg'][n] + 1e-6 for n, v in zip(active, angles)):
            raise ValueError('rotation outside B0 bounds')
    world = fk(local, parents, [lookup[n] for n in active], np.deg2rad(angles)) if active else base.copy()
    if not historical:
        protected = [i for i, n in enumerate(names) if n == 'Hips' or n.startswith(('LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot', 'LeftToe', 'RightToe'))]
        if not np.allclose(world[protected], base[protected], atol=1e-8):
            raise ValueError('lower/support pose changed')
        for side in case['arms']:
            joints = world[[lookup[side + n] for n in ('Arm', 'ForeArm', 'Hand')], :3, 3]
            first, second = np.diff(joints, axis=0)
            if first @ second / (np.linalg.norm(first) * np.linalg.norm(second)) < -.94:
                raise ValueError('overfolded arm')
    return world


def result_row(case, method, edits, **metadata):
    world = named_pose(case, edits, historical=(method == 'p5_legacy'))
    heads = {b['name']: w[:3, 3].tolist() for b, w in zip(case['bones'], world, strict=True)}
    return dict(case_id=case['case_id'], method=method, source_sha256=case['source_sha256'],
                body_asset_sha256=case['body_asset_sha256'], edits_deg=edits, expected_heads=heads, **metadata)


def project(points, camera):
    points = np.asarray(points)
    relative = (points - camera['origin']) / camera['torso_world']
    camera_space = relative @ np.asarray(camera['rotation']).T
    depth = 1 + camera_space[:, 2] * camera['inverse_distance_torso']
    if np.any(depth <= .25):
        raise ValueError('near plane')
    return camera_space[:, :2] / depth[:, None] * camera['pixel_scale'] + camera['pixel_offset']
