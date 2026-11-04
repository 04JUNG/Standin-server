"""Actual source skin/hand/pose regression and fixed FBX round trips.

This supplements, and does not replace, the production solver/same-pose checks.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion
from mathutils.bvhtree import BVHTree
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build_shape_release import load_controller
from scripts.body_default_proportions.build import invariants, write

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--release', type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = args.release / 'pose-qa'
out.mkdir(exist_ok=True)
validation = json.loads((args.release / 'validation.json').read_text())
rows, surfaces = [], {}
poses = ('T', 'ARMS_DOWN', 'RAISE', 'COMBINED', 'TWIST', 'FIST')


def independent_pairs(q, faces):
    # The latest shoulder controller deliberately forbids its old intersection
    # correction path. Inspect externally without enabling or altering it.
    tree = BVHTree.FromPolygons(q.tolist(), faces.tolist(), all_triangles=True)
    return sum(1 for i, j in tree.overlap(tree)
               if i < j and not set(faces[i]).intersection(faces[j]))

for model in validation['rows']:
    sex, label = model['sex'], model['label']
    source = model['source']
    c = load_controller(source['blend'], source['controller'], 'pose_' + sex + label)
    m, a = c.objects()
    c.set_controls(model['controls'], correct=False)
    locked = invariants(c)
    neutral = c.compute(model['controls'])[0]
    faces = c.source()['faces']
    for pose in poses:
        c.apply_pose('T' if pose == 'FIST' else pose)
        if pose == 'FIST':
            c.BUSY = True
            try:
                # Deliberately request beyond the MCP envelope on both hands;
                # inspect actual constrained channels, not the requested angle.
                for name in c.hl_names():
                    pb = a.pose.bones[name]
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = Quaternion((1, 0, 0), math.radians(90))
                a.update_tag()
                bpy.context.view_layer.update()
                c.correct_pose()
            finally:
                c.BUSY = False
        q = c.evaluated(m)
        normals = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        if not np.isfinite(q).all() or np.any(np.linalg.norm(normals, axis=1) < 2e-12):
            raise ValueError('nonfinite or collapsed pose surface')
        if invariants(c) != locked:
            raise ValueError('authoring invariants changed during posing')
        limits = c.hl_calibrate(a)
        violations = []
        for name, spec in limits.items():
            effective = c.hl_effective_basis(a, a.pose.bones[name]).to_euler('XYZ')
            for axis, value, (lo, hi) in zip('xyz', effective, spec['limits_deg']):
                d = math.degrees(value)
                # Blender's float32 constrained matrices are measured here; no
                # authoring or solver tolerance is changed by this diagnostic.
                if d < lo - .001 or d > hi + .001:
                    violations.append([name, axis, d, lo, hi])
        if violations:
            raise ValueError('actual finger constraints violated: ' + str(violations[:2]))
        pair_count = independent_pairs(q, faces)
        neutral_pairs = independent_pairs(neutral, faces)
        row = {'sex': sex, 'state': label, 'pose': pose, 'finite': True, 'collapsedFaces': 0,
               'fingerBonesChecked': len(limits), 'fingerLimitViolations': violations,
               'intersectionPairs': pair_count, 'neutralIntersectionPairs': neutral_pairs,
               'invariantsPreserved': True,
               'surfaceQa': {k: json.loads(m[k]) for k in m.keys() if k.endswith('_qa') and isinstance(m[k], str)}}
        if label == 'candidate':
            bundle = c.export_fixed(str(out / 'roundtrips'), posed=True)
            manifest = json.loads((bundle / 'manifest.json').read_text())
            row['fbx'] = str(bundle / 'model.fbx')
            row['roundtrip'] = {k: manifest[k] for k in ('roundtrip_max_error_mm', 'roundtrip_joint_error_mm', 'roundtrip_weight_error')}
        surfaces[f'{sex}_{label}_{pose}'] = q.astype('<f4')
        rows.append(row)
        write(out / 'results.json', {'status': 'running', 'rows': rows})
        print('POSE_QA', sex, label, pose, 'pairs', pair_count, 'roundtrip', row.get('roundtrip'), flush=True)
    c.unregister()
np.savez_compressed(out / 'surfaces.npz', **surfaces)
comparisons = []
for sex in ('male', 'female'):
    for pose in poses:
        old, new = [next(r for r in rows if r['sex'] == sex and r['pose'] == pose and r['state'] == state)
                    for state in ('latest_base', 'candidate')]
        comparisons.append({'sex': sex, 'pose': pose, 'baselinePairs': old['intersectionPairs'],
                            'candidatePairs': new['intersectionPairs'],
                            'increase': new['intersectionPairs'] - old['intersectionPairs']})
write(out / 'results.json', {'status': 'complete', 'rows': rows, 'pairComparisons': comparisons,
    'intersectionMeaning': 'Broad-phase triangle pairs, including inherited overlaps; not a collision-free or artist verdict.'})
