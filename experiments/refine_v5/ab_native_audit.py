"""Native contact measurements and exported FBX skin checks after inference is frozen."""

import json
from pathlib import Path

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from converter import retarget
from converter.mesh_validation import audit_mesh
from experiments.refine_v5.ab_blender import apply_rotations, bone_world, cloud_distance, sha, verify_plan, weighted_mesh
from experiments.refine_v5.ab_geometry import contact_faces
from experiments.refine_v5.audit_iterative_native import snapshot
from experiments.refine_v5.export_native_v2 import export_preserving_bones
from experiments.refine_v5.native_contract_v2 import compare_native
from experiments.refine_v5.target_safety import skin_nonregression
from src.refine_v4.skin_collision import inspect_self


def frozen_contacts(vertices, faces, weights, names, world, notes):
    patches = []
    for side, region in notes:
        try:
            hand, palm, target = contact_faces(vertices, faces, weights, names, world, side, region)
        except (ValueError, KeyError) as exc:
            patches.append((side, region, str(exc)))
            continue
        if not len(hand) or not len(palm) or not len(target):
            patches.append((side, region, 'missing_weighted_patch'))
        else:
            patches.append((side, region, (hand, palm, target)))
    return patches


def measure_contact(vertices, faces, patches):
    output = []
    for side, region, patch in patches:
        row = dict(side=side, region=region)
        if isinstance(patch, str):
            output.append(dict(row, status='unavailable', reason=patch))
            continue
        hand, palm, target = patch
        triangles = faces[target]
        tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
        hand_distances = [tree.find_nearest(Vector(p))[3] for p in vertices[np.unique(faces[hand])]]
        palm_tri = vertices[faces[palm]]
        centers = palm_tri.mean(axis=1)
        normals = np.cross(palm_tri[:, 1] - palm_tri[:, 0], palm_tri[:, 2] - palm_tri[:, 0])
        area = np.linalg.norm(normals, axis=1) * .5
        normals /= np.maximum(2 * area[:, None], 1e-12)
        gaps, signed, opposed = [], [], []
        for center, normal in zip(centers, normals, strict=True):
            location, outward, _, gap = tree.find_nearest(Vector(center))
            gaps.append(gap)
            signed.append((Vector(center) - location).dot(outward))
            opposed.append(-Vector(normal).dot(outward))
        gaps, signed, opposed = map(np.asarray, (gaps, signed, opposed))
        near = (gaps <= .005) & (signed >= -.0005) & (opposed >= .45)
        output.append(dict(row, status='measured', hand_min_gap_mm=1000 * min(hand_distances),
                           palm_median_gap_mm=1000 * float(np.median(gaps)),
                           palm_min_gap_mm=1000 * float(gaps.min()),
                           palm_contact_area_fraction=float(area[near].sum() / area.sum()),
                           palm_negative_signed_samples=int((signed < -.002).sum()),
                           palm_samples=len(gaps)))
    return output


def world_bones(armature):
    return [dict(name=b.name.split(':')[-1], matrix=np.array(armature.matrix_world @ b.matrix).tolist())
            for b in armature.pose.bones]


def strict_skin(vertices, faces, body_hash, *, full_body):
    return audit_mesh(vertices, faces, profile_sha256=body_hash, full_body=full_body, seconds=2.)


def triage(armature):
    try:
        return inspect_self(armature)
    except ValueError as exc:
        return {'unavailable': str(exc)}


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


class RoundtripError(ValueError):
    """Saved FBX failed a structural or mesh verification gate."""


def export_checked(path, armature, meshes, names, base, load):
    planned = snapshot(armature)
    vertices, faces, _ = weighted_mesh(meshes, names)
    retarget.apply_output_mode(armature, meshes, 'rigged_rest')
    export_preserving_bones(path)
    armature, meshes = load(path)
    actual = snapshot(armature)
    try:
        native = compare_native(planned['heads'], actual['heads'], planned['parents'],
                                actual['parents'], base['lengths'], actual['lengths'])
    except ValueError as exc:
        raise RoundtripError(str(exc)) from exc
    av, af, _ = weighted_mesh(meshes, names)
    if len(av) != len(vertices) or len(af) != len(faces):
        raise RoundtripError('saved mesh inventory changed')
    cloud = max(cloud_distance(vertices, av), cloud_distance(av, vertices))
    frame = max(float(np.max(abs(np.asarray(planned['matrices'][name])[:3, :3]
                                 - np.asarray(actual['matrices'][name])[:3, :3]))) for name in names)
    if cloud > 1e-5 or frame > 2e-4:
        raise RoundtripError(f'saved skin or bone frame differs (cloud={cloud:.8g}m, frame={frame:.8g})')
    return armature, av, af, native, cloud, frame


def audit(root, cases, gt_root, load):
    frozen = json.loads((root / 'frozen.json').read_text())
    if sha(root / 'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('frozen input mismatch')
    rows = frozen['results']
    out = root / 'native'
    out.mkdir(exist_ok=False)
    ledger = []
    for case in cases:
        case_id = case['case_id']
        source = Path(case['source'])
        if sha(source) != case['source_sha256']:
            raise ValueError('source digest changed')
        names = [b['name'] for b in case['bones']]
        with np.load(root / (case_id + '-B0.npz')) as skin:
            b0_vertices, b0_faces, b0_weights = [skin[key] for key in ('vertices', 'faces', 'weights')]
        base_world = np.array([bone['matrix'] for bone in case['bones']])
        patches = frozen_contacts(b0_vertices, b0_faces, b0_weights, names, base_world, case['contact'])
        base_skin = None
        selected = [row for row in rows if row['case_id'] == case_id]
        for row in sorted(selected, key=lambda row: row['method'] != 'B0'):
            method = row['method']
            if row['source_sha256'] != case['source_sha256'] or row['body_asset_sha256'] != case['body_asset_sha256']:
                raise ValueError('candidate source or asset differs')
            armature, meshes = load(source)
            base = snapshot(armature)
            apply_rotations(armature, row['edits_deg'])
            plan_error = verify_plan(armature, row['expected_heads'])
            v, f, _ = weighted_mesh(meshes, names)
            if not np.array_equal(f, b0_faces):
                raise ValueError('posed mesh changed B0 triangle identities')
            contact = measure_contact(v, f, patches)
            if method == 'B0' or row['status'] == 'fallback_exact_B0':
                if row['edits_deg']:
                    raise ValueError('B0 fallback contains edits')
                saved = source
                native_error = {'max_head_error_m': 0., 'max_length_error_m': 0.}
                cloud_error = 0.
                frame_error = 0.
            else:
                saved = out / f'{case_id}-{method}.diagnostic.fbx'
                armature, v, f, native_error, cloud_error, frame_error = export_checked(
                    saved, armature, meshes, names, base, load)
            regional = triage(armature)
            if method == 'B0':
                base_skin = regional
            if 'unavailable' in regional or 'unavailable' in base_skin:
                worsening = ['skin_triage_unavailable']
            else:
                worsening = skin_nonregression(base_skin, regional)
            strict = strict_skin(v, f, case['body_asset_sha256'], full_body=True)
            np.savez_compressed(out / f'{case_id}-{method}.npz', vertices=v, faces=f)
            ledger.append(dict(case_id=case_id, method=method, status=row['status'], path=str(saved),
                               sha256=sha(saved), source_sha256=case['source_sha256'],
                               bones=world_bones(armature), native=native_error, max_fk_error_m=plan_error,
                               max_cloud_error_m=cloud_error, max_rotation_element_error=frame_error,
                               skin_triage=regional, skin_nonregression_failures=worsening,
                               full_skin=strict, contact=contact))
            write(out / 'audit.json', ledger)
            print('AB_NATIVE', case_id, method, strict['status'], len(worsening), flush=True)

        gt_path = gt_root / (case_id + '.fbx')
        armature, meshes = load(gt_path)
        gt_bones = world_bones(armature)
        gt_names = [bone['name'] for bone in gt_bones]
        gt_world = np.array([bone['matrix'] for bone in gt_bones])
        vertices, faces, weights = weighted_mesh(meshes, gt_names)
        gt_patches = frozen_contacts(vertices, faces, weights, gt_names, gt_world, case['contact'])
        gt_contact = measure_contact(vertices, faces, gt_patches)
        np.savez_compressed(out / f'{case_id}-GT.npz', vertices=vertices, faces=faces)
        ledger.append(dict(case_id=case_id, method='GT', path=str(gt_path), sha256=sha(gt_path), bones=gt_bones,
                           contact=gt_contact, skin_triage=triage(armature),
                           full_skin=strict_skin(vertices, faces, case['body_asset_sha256'], full_body=len(faces) >= len(b0_faces))))
        write(out / 'audit.json', ledger)
        print('AB_NATIVE', case_id, 'GT', len(gt_contact), flush=True)
