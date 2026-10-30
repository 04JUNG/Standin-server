"""Independently compare final FBXs with the pinned controller at the same pose.

Run with Blender: --background --python-exit-code 1 --python-expr
"import sys,runpy; sys.argv=[...]; runpy.run_path('scripts/inspect_shape_bake.py',run_name='__main__')".
The old neutral-vs-posed correlation is retained as a diagnostic, not a pose-
invariant acceptance test. Correlation stays >= .8; the same-pose source surface
must additionally agree pointwise within the existing .01 mm export bound.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_spec import load_registry
from scripts.shape_inspection_metrics import INSPECTION_SCHEMA, complete_pairs, inspect_geometry, pose_digest, pose_frame_agrees


def import_fbx(path):
    import bpy
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False, ignore_leaf_bones=False, automatic_bone_orientation=False)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    mesh = next(o for o in bpy.data.objects if o.type == 'MESH')
    return (np.array([tuple(mesh.matrix_world @ v.co) for v in mesh.data.vertices]),
            np.array([p.vertices[:] for p in mesh.data.polygons]),
            {b.name: (arm.matrix_world @ b.matrix_local).copy() for b in arm.data.bones},
            {b.name: b.parent.name if b.parent else None for b in arm.data.bones})


def controller(recipe):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=recipe['blend'])
    spec = importlib.util.spec_from_file_location('inspect_controller', recipe['controller'])
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    return sb


def posed_reference(recipe, values, actual_path, temp, pose):
    """Read observed pose frames, then author expected geometry from requested values.

    The output's surface and weights never enter the reference calculation.
    Native pose metadata is checked against the FBX bones; it supplies no surface.
    A neutral source FBX supplies only the FBX bone-axis convention. This does
    not call convert_shaped, the retarget solver or the converter lineage report.
    Pose correctness remains covered by the solver's independent contracts.
    """
    import bpy
    from mathutils import Vector
    sb = controller(recipe)
    sb.set_controls(values, correct=False)
    source_mesh, _ = sb.objects()
    source_faces = np.array([p.vertices[:] for p in source_mesh.data.polygons])
    neutral = sb.compute(values)[0].copy()
    rest_source = sb.source()['rest']
    wanted_heads = {name: np.asarray(sb.warp([bone['head']], values, source_mesh)[0])
                    for name, bone in rest_source.items()}
    neutral_bundle = sb.export_fixed(str(temp / 'neutral'))
    _, _, rest, rest_parents = import_fbx(neutral_bundle / 'model.fbx')
    actual, faces, observed, parents = import_fbx(actual_path)
    if set(observed) != set(rest) or parents != rest_parents or not np.array_equal(faces, source_faces):
        raise ValueError('output topology or rig differs from approved source')
    joint_errors = []
    for name, parent in parents.items():
        if parent in wanted_heads and name in wanted_heads:
            wanted_length = float(np.linalg.norm(wanted_heads[name] - wanted_heads[parent]))
            measured = (observed[name].translation - observed[parent].translation).length
            joint_errors.append(abs(measured - wanted_length) * 1000)
    if not joint_errors:
        raise ValueError('no measured source joints')
    from mathutils import Matrix
    sb = controller(recipe)
    sb.set_controls(values, correct=False)
    mesh, rig = sb.objects()
    pose_digest(pose)
    if set(pose['bones']) != set(rest):
        raise ValueError('native pose bone set mismatch')
    native = pose['bones']
    channels = pose.get('channels')
    if not isinstance(channels, dict) or set(channels) != set(native):
        raise ValueError('exact authoring pose channels required')
    sb.BUSY = True
    try:
        # Verify recorded native frames against actual FBX bones. The native
        # frames are not a surface oracle; geometry is recomputed from source.
        for bone in rig.data.bones:
            target = Matrix(native[bone.name])
            expected_fbx = (rig.matrix_world @ target) @ (rig.matrix_world @ bone.matrix_local).inverted() @ rest[bone.name]
            if not pose_frame_agrees(expected_fbx.translation, observed[bone.name].translation,
                                     expected_fbx.to_quaternion(), observed[bone.name].to_quaternion()):
                raise ValueError('authoring pose does not match output FBX bone frames')
            pb = rig.pose.bones[bone.name]
            channel = channels[bone.name]
            pb.rotation_mode = channel['rotation_mode']
            for key in ('location', 'scale', 'rotation_quaternion', 'rotation_euler', 'rotation_axis_angle'):
                setattr(pb, key, channel[key])
        bpy.context.view_layer.update()
        for bone in rig.pose.bones:
            if max(abs(x-y) for a,b in zip(bone.matrix, Matrix(native[bone.name])) for x,y in zip(a,b)) > 1e-7:
                raise ValueError('authoring channels do not reproduce native pose')
        # The recorded pose is already projected by the authoring joint limits.
        # Rebuild its cache signature so correct_pose recomputes the surface at
        # that verified pose instead of projecting the shoulder a second time.
        if callable(getattr(sb, 'sl_signature', None)):
            rig['standin_shoulder_signature'] = sb.sl_signature(rig)
        sb.correct_pose(check_intersections=True, contacts=True)
        for bone in rig.pose.bones:
            if max(abs(x-y) for a,b in zip(bone.matrix, Matrix(native[bone.name])) for x,y in zip(a,b)) > 1e-7:
                raise ValueError('surface inspection changed verified native pose')
        expected = np.array([mesh.matrix_world @ Vector(co) for co in sb.evaluated(mesh)])
    finally:
        sb.BUSY = False
    return actual, expected, neutral, faces, max(joint_errors), len(joint_errors)


def inspect_pair(recipe, pair, folder, temp):
    actual, reference, neutral, artifacts = {}, {}, {}, {}
    joint_error = 0.; joint_count = 0; faces = None
    for label in ('preset', 'slider'):
        path = folder / (pair[label]['id'] + '.fbx')
        artifacts[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        if artifacts[label] != pair[label]['fbx_sha256']:
            raise ValueError('bake run artifact checksum mismatch')
        actual[label], reference[label], neutral[label], current_faces, error, count = posed_reference(
            recipe, pair[label]['values'], path, temp, pair[label].get('authoring_pose'))
        if faces is not None and not np.array_equal(faces, current_faces):
            raise ValueError('preset/slider topology mismatch')
        faces = current_faces; joint_error = max(joint_error, error); joint_count = count
    score = inspect_geometry(actual, reference, faces, joint_error)
    passed = score.pop('passed')
    nd = np.linalg.norm(neutral['slider'] - neutral['preset'], axis=1)
    pd = np.linalg.norm(actual['slider'] - actual['preset'], axis=1)
    legacy = float(np.corrcoef(nd, pd)[0, 1]) if min(nd.std(), pd.std()) > 1e-12 else None
    return {'schema': INSPECTION_SCHEMA, 'case': pair['preset']['case'], 'bvh': pair['preset']['bvh'],
            'source': {k: recipe[k] for k in ('rules', 'blend_sha256', 'controller_sha256')},
            'artifact_sha256': artifacts,
            'pose_sha256': {label: pose_digest(row.get('authoring_pose')) for label, row in pair.items()}, 'reference': 'pinned-controller-at-verified-native-pose.v1',
            'joints': {'max_error_mm': joint_error, 'pairs': joint_count},
            'shape_effect': score, 'legacy_neutral_correlation': legacy, 'passed': passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).unlink(missing_ok=True)
    runs = json.loads(Path(args.runs).read_text()); pairs = complete_pairs(runs)
    registry = load_registry()
    if registry is None:
        raise ValueError('approved shape registry required')
    results = []
    for (cid, _), pair in sorted(pairs.items()):
        recipe = registry['characters'][cid] | registry['sources'][registry['characters'][cid]['rules']]
        with tempfile.TemporaryDirectory(prefix='shape-inspection-') as temp:
            row = inspect_pair(recipe, pair, Path(args.runs).parent, Path(temp))
        results.append(row)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print('INSPECTED', len(results), 'passed', sum(r['passed'] for r in results), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
