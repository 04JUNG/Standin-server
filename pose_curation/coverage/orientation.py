"""Compose reviewed 3D poses for rough camera angles without changing limbs.

These are explicitly labelled composition variants, not new captured motions.
The BVH root carries the orientation so geometry, thumbnails and downloads agree.
Queries remain the real estimator output. This is calibration on supplied cases,
not a held-out accuracy benchmark.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.bvh import load_coco17, parse_bvh, write_single_frame_bvh
from src.features import normalize_skeleton
from ..candidates import build_candidates
from ..review.catalog import Catalog
from ..review.selection import is_publishable
from ..review.store import ReviewStore
from ..storage import read_json, sha256, utc_now, write_json


def orient_bvh(source: Path, destination: Path, matrix: np.ndarray) -> None:
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1:
        raise ValueError('composition variants require one frame')
    frame = frames[0].copy()
    channels = joints[0][3]
    indices = [i for i, name in enumerate(channels) if name.endswith('rotation')]
    axes = ''.join(channels[i][0] for i in indices)
    original = Rotation.from_euler(axes, frame[indices], degrees=True).as_matrix()
    frame[indices] = Rotation.from_matrix(matrix @ original).as_euler(axes, degrees=True)
    write_single_frame_bvh(str(source), frame, str(destination))
    before, _ = load_coco17(str(source)); after, _ = load_coco17(str(destination))
    if not np.allclose(before @ matrix.T, after, atol=.0002, rtol=0):
        raise ValueError('orientation changed body geometry')


def run(root: Path, batch: Path, queries: list[tuple[str, int]]) -> dict:
    if (batch / 'manifest.json').exists():
        raise ValueError('choose a new batch')
    catalog = Catalog(Path('data'), Path('data/curation'))
    reviews = ReviewStore(Path('data/curation/reviews.sqlite')).all()
    bases = [p for p in catalog.all() if p.batch.startswith('combat-') and is_publishable(p, reviews)
             and not p.metadata.get('composition_variant')]
    # Uniform orientation grid, shared by every query. Camera composition only.
    angles = np.array(list(product(range(-60, 61, 20), range(-180, 180, 15), range(-40, 41, 20))))
    rotations = Rotation.from_euler('YXZ', angles[:, [1, 0, 2]], degrees=True).as_matrix()
    features = []
    for base in bases:
        points, _ = load_coco17(str(base.bvh))
        projected = np.einsum('aij,kj->aki', rotations, points)[:, :, :2].copy()
        projected[:, :, 1] *= -1
        projected -= projected[:, [11, 12]].mean(axis=1)[:, None, :]
        scale = np.linalg.norm(projected[:, [5, 6]].mean(axis=1), axis=-1)
        projected /= np.maximum(scale[:, None, None], 1e-6)
        features.append(projected)
    features = np.stack(features).astype(np.float32)
    records = []
    for identity, person_index in queries:
        extracted = read_json(root / 'extraction' / f'{identity}.json')
        person = extracted['people'][person_index]
        scores = np.array(person['scores']); valid = np.flatnonzero(scores[5:] >= .3) + 5
        if len(valid) < 8 or not person['torso_visible']:
            continue
        query = normalize_skeleton(np.array(person['keypoints']), scores).reshape(17, 2)
        distances = np.linalg.norm(features[:, :, valid] - query[None, None, valid], axis=-1).mean(axis=-1)
        base_index, angle_index = np.unravel_index(np.argmin(distances), distances.shape)
        base = bases[base_index]; rotation = rotations[angle_index]
        pid = f'combat_composition_{identity}_p{person_index}'
        path = batch / 'bvh' / (pid + '.bvh')
        orient_bvh(base.bvh, path, rotation)
        record = dict(base.metadata)
        for key in ('preview', 'thumbnail_versions', 'nearest_existing', 'nearest_existing_key'):
            record.pop(key, None)
        record.update(pose_id=pid, style=base.metadata['style'] + ' · 구도 변형', bvh=path.relative_to(batch).as_posix(),
            bvh_sha256=sha256(path), thumbnails={}, preview_kind='pending', retarget_status='not_validated',
            pose_family_id=base.metadata.get('pose_family_id', base.pose_id),
            composition_variant={'base_pose_id': base.pose_id, 'base_sha256': base.content_hash,
                                 'root_rotation_matrix': rotation.tolist(), 'pitch_yaw_roll_degrees': angles[angle_index].tolist(),
                                 'calibration_image_id': identity, 'calibration_image_sha256': extracted['image_sha256'],
                                 'person_index': person_index, 'measured_distance': float(distances[base_index, angle_index]),
                                 'limb_rotations_preserved': True, 'manual_query_coordinates': False},
            transform='Composition variant: only BVH root orientation changed; all limb/finger channels and bone lengths preserved. Calibration on supplied rough, not held-out evaluation.')
        records.append(record)
    projections = build_candidates(records, batch, batch.name)
    manifest = {'schema_version': 1, 'batch_id': batch.name, 'created_at': utc_now(), 'status': 'complete', 'poses': records,
                'failures': [], 'summary': {'poses': len(records), 'projections': projections}}
    write_json(batch / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--query', action='append', required=True, help='image_id:person_index')
    args = parser.parse_args()
    queries = [(value.rsplit(':', 1)[0], int(value.rsplit(':', 1)[1])) for value in args.query]
    print(run(args.root, args.batch, queries)['summary'])
