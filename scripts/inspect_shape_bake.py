"""Independent evidence that slider values reached the final posed FBX. Run with Blender (or pip bpy).

    CONVERTER_SHAPE_REGISTRY=<release>/shape-bakes.local.json \
    python scripts/inspect_shape_bake.py --runs <bake-qa>/runs.json --out <bake-qa>/inspection.json

For each preset/slider pair produced by verify_shape_bake.py (same BVH, same body):

* joints: parent→child joint distances in the final FBX against the controller's warped
  rest skeleton for the requested values. Posing is rigid per bone, so these distances are
  the body proportions (height, heads, limb lengths) and must match within 0.5 mm;
* shape_effect: the per-vertex displacement between preset and slider bakes, compared
  with the same displacement in the controller's neutral meshes. The bake is posed and
  contact-corrected, so the vectors rotate, but their magnitudes must follow the neutral
  change (correlation >= 0.8) and the signed volume change must keep its direction and at
  least half its size (contact correction in deep poses legitimately removes some volume).

This uses only the FBX files and the approved controller; it does not trust the bake's
own lineage report.
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_spec import load_registry  # noqa: E402


def mesh_volume(co, faces):
    a, b, c = co[faces[:, 0]], co[faces[:, 1]], co[faces[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def import_fbx(path):
    import bpy
    from mathutils import Matrix
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=str(path), ignore_leaf_bones=False, automatic_bone_orientation=False)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    mesh = next(o for o in bpy.data.objects if o.type == 'MESH')
    co = np.array([tuple(mesh.matrix_world @ v.co) for v in mesh.data.vertices])
    faces = np.array([p.vertices[:] for p in mesh.data.polygons])
    heads = {b.name: np.array(tuple(arm.matrix_world @ b.head_local)) for b in arm.data.bones}
    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    return co, faces, heads, parents


def controller(recipe):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=recipe['blend'])
    spec = importlib.util.spec_from_file_location('inspect_controller', recipe['controller'])
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    return sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    folder = Path(args.runs).parent
    registry = load_registry()
    by_key = {}
    for r in runs:
        if r.get('ok'):
            by_key.setdefault((r['case'], r['bvh']), {})[r['label']] = r
    results = []
    cache = {}
    for (cid, bvh), pair in sorted(by_key.items()):
        if set(pair) != {'preset', 'slider'}:
            continue
        recipe = {**registry['characters'][cid], **registry['sources'][registry['characters'][cid]['rules']]}
        if recipe['rules'] not in cache:
            cache.clear()
            cache[recipe['rules']] = controller(recipe)
        sb = cache[recipe['rules']]
        mesh, _ = sb.objects()
        s = sb.source()
        f = np.array([p.vertices[:] for p in mesh.data.polygons])
        neutral = {}
        rest = {}
        for label in ('preset', 'slider'):
            values = sb.normalize(pair[label]['values'])
            neutral[label] = sb.compute(values)[0]
            rest[label] = {name: np.asarray(sb.warp([row['head']], values, mesh)[0]) for name, row in s['rest'].items()}
        fbx = {label: import_fbx(folder / (pair[label]['id'] + '.fbx')) for label in ('preset', 'slider')}
        # Re-open the controller scene for the next case (import cleared the objects).
        cache.clear()
        joint_err = []
        joint_change = []
        co_s, faces, heads_s, parents = fbx['slider']
        co_p, _, heads_p, _ = fbx['preset']
        for name, parent in parents.items():
            if parent is None or name not in rest['slider'] or parent not in rest['slider']:
                continue
            want = np.linalg.norm(rest['slider'][name] - rest['slider'][parent])
            got = np.linalg.norm(heads_s[name] - heads_s[parent])
            before = np.linalg.norm(heads_p[name] - heads_p[parent])
            joint_err.append(abs(got - want) * 1000)
            joint_change.append(abs(got - before) * 1000)
        d_neutral = np.linalg.norm(neutral['slider'] - neutral['preset'], axis=1)
        d_posed = np.linalg.norm(co_s - co_p, axis=1)
        moved = d_neutral > 1e-4
        corr = float(np.corrcoef(d_neutral, d_posed)[0, 1]) if moved.sum() > 10 else None
        vol = {'neutral_ratio': mesh_volume(neutral['slider'], f) / mesh_volume(neutral['preset'], f),
               'posed_ratio': mesh_volume(co_s, faces) / mesh_volume(co_p, faces)}
        row = {'case': cid, 'bvh': bvh,
               'joints': {'max_error_mm': round(max(joint_err), 3), 'max_change_mm': round(max(joint_change), 2), 'pairs': len(joint_err)},
               'shape_effect': {'neutral_mean_mm': round(float(d_neutral.mean() * 1000), 3),
                                'posed_mean_mm': round(float(d_posed.mean() * 1000), 3),
                                'magnitude_correlation': None if corr is None else round(corr, 4),
                                'volume_ratio_neutral': round(vol['neutral_ratio'], 5),
                                'volume_ratio_posed': round(vol['posed_ratio'], 5)}}
        # Posed meshes in deep-contact poses (kneeling, sitting) are contact-corrected and may
        # self-overlap, so signed volume is only a coarse signal: require the same direction and
        # at least half of the neutral change, not equality.
        dn, dp = vol['neutral_ratio'] - 1, vol['posed_ratio'] - 1
        row['passed'] = bool(row['joints']['max_error_mm'] <= 0.5 and (corr is None or corr >= 0.8)
                             and (abs(dn) < 0.002 or (np.sign(dn) == np.sign(dp) and abs(dp) >= 0.5 * abs(dn))))
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print('INSPECTED', len(results), 'passed', sum(r['passed'] for r in results), flush=True)


if __name__ == '__main__':
    main()
