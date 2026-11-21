"""Measure the influence of bounded single-joint rotations on a damaged edge."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def capture_pose(runtime, model, pose, source):
    sys.path.insert(0, str(runtime))
    from converter import retarget
    from converter.local import inspection

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm, meshes = inspection.import_fbx(model)
    if len(meshes) != 1:
        raise ValueError("This diagnostic requires a single mesh")
    mesh = meshes[0]
    profile, _ = inspection.inspect(arm, meshes)
    retarget._apply_object_transform(arm)
    motion = retarget.import_bvh(str(pose), frame=0)
    if not retarget.retarget(motion, arm, dst_profile=profile["profile"], apply_root_translation=False).ok:
        raise ValueError("Retarget failed")
    raw = np.load(source)
    rest, baseline, faces = raw["rest"], raw["solver"], raw["triangles"]
    guard = DeformationGuard(rest, baseline, faces, raw["anchors"])
    guard.upper = np.maximum(1., guard.reference["stretch"])
    guard.edge_upper = np.maximum(1., guard.reference["edge"])
    guard.crease_limit = np.maximum(1e-7, abs(guard.angle_difference(guard.reference["bend"], guard.rest_bend)))
    bones = list(arm.pose.bones)
    index_by_name = {bone.name: i for i, bone in enumerate(bones)}
    transforms = []
    world = np.asarray(arm.matrix_world)
    for bone in bones:
        transforms.append(world @ np.asarray(bone.matrix) @ np.linalg.inv(world @ np.asarray(bone.bone.matrix_local)))
    transforms = np.asarray(transforms)
    weights = np.zeros((len(rest), len(bones)))
    for vertex in mesh.data.vertices:
        for group in vertex.groups:
            weights[vertex.index, index_by_name[mesh.vertex_groups[group.group].name]] = group.weight
    weights /= weights.sum(axis=1, keepdims=True)
    maps = np.einsum("vb,bij->vij", weights, transforms, optimize=True)
    homogeneous = np.column_stack([rest, np.ones(len(rest))])
    reconstruction = np.einsum("vij,vj->vi", maps[:, :3], homogeneous)
    reconstruction_error = float(np.linalg.norm(reconstruction - baseline, axis=1).max())
    if reconstruction_error >= 2e-5:
        raise ValueError("The captured skin matrices do not reproduce the source pose")
    return arm, meshes, guard, bones, transforms, weights, homogeneous, reconstruction_error


def probe(runtime, model, pose, source, output):
    from converter.local import inspection
    arm, meshes, guard, bones, transforms, weights, homogeneous, reconstruction_error = capture_pose(
        runtime, model, pose, source)
    baseline = guard.baseline
    index = guard.reference["edge"].argmax()
    a, b = guard.graph.a[index], guard.graph.b[index]
    edge = baseline[b] - baseline[a]
    unit = edge / np.linalg.norm(edge)
    index_by_name = {bone.name: i for i, bone in enumerate(bones)}
    rows = []
    joint_bounds = {}
    partial_subtrees = []
    for bone in bones:
        ids = [index_by_name[x.name] for x in [bone, *bone.children_recursive]]
        pair_weights = weights[[a, b]][:, ids]
        if not np.any(pair_weights):
            continue
        pivot = np.asarray(arm.matrix_world @ bone.head)
        subgroup = np.einsum("vb,bij,vj->vi", pair_weights, transforms[ids, :3], homogeneous[[a, b]])
        lever = subgroup[1] - subgroup[0] - (pair_weights[1].sum() - pair_weights[0].sum()) * pivot
        # Rotation of this subtree changes this edge by (R - I) times lever.
        # Bound the best possible result over every axis within two degrees.
        fixed_part = edge - lever
        length = np.linalg.norm(lever)
        base_length = np.linalg.norm(fixed_part)
        bound = np.linalg.norm(edge)
        if length > 1e-12 and base_length > 1e-12:
            cosine = np.clip(-np.dot(lever, fixed_part) / (length * base_length), -1., 1.)
            angle = max(0., np.arccos(cosine) - np.deg2rad(2.))
            bound = np.sqrt((length - base_length)**2 + 4 * length * base_length * np.sin(angle / 2)**2)
        joint_bounds[bone.name] = float(bound / guard.lengths[index])
        if not np.all(weights[[a, b]][:, np.setdiff1d(np.arange(len(bones)), ids)] == 0):
            partial_subtrees.append((set(ids), length))
        gradient = np.cross(lever, unit)
        rows.append((float(np.linalg.norm(gradient)), bone.name, ids, pivot, gradient))
    sensitivity, name, ids, pivot, gradient = max(rows, key=lambda row: row[0])
    if sensitivity < 1e-12:
        raise ValueError("Peak edge has no measurable local rotation sensitivity")
    weighted_map = np.einsum("vb,bij->vij", weights[:, ids], transforms[ids, :3], optimize=True)
    offsets = np.einsum("vij,vj->vi", weighted_map, homogeneous) - weights[:, ids].sum(axis=1)[:, None] * pivot
    selected = arm.pose.bones[name]
    original = selected.matrix.copy()
    center = Matrix.Translation(Vector(pivot))
    samples = []
    for degrees in (.25, 1., 2.):
        rotate = np.asarray(Matrix.Rotation(np.deg2rad(degrees), 3, Vector(-gradient / sensitivity)))
        delta = offsets @ (rotate - np.eye(3)).T
        rotation = Matrix(rotate.tolist()).to_4x4()
        transform = center @ rotation @ center.inverted()
        selected.matrix = arm.matrix_world.inverted() @ transform @ arm.matrix_world @ original
        bpy.context.view_layer.update()
        evaluated = np.asarray(inspection.surface(meshes)[meshes[0].name])
        proposed = baseline + delta
        pose_error = float(np.linalg.norm(proposed - evaluated, axis=1).max())
        if pose_error >= 2e-5:
            raise ValueError(f"Joint rotation differs from evaluated mesh by {pose_error} m")
        ratios = guard.measure(proposed)["edge"]
        samples.append(dict(degrees=degrees, poseErrorM=pose_error, target=float(ratios[index]), peak=float(ratios.max()),
                            maxMoveMm=float(1000*np.linalg.norm(delta, axis=1).max()),
                            newDouble=int(((guard.reference["edge"] <= 2) & (ratios > 2+1e-7)).sum()),
                            violations=guard.violations(proposed)))
    # A common ancestor moves both endpoints rigidly. When the remaining
    # weighted subtrees are disjoint, their edge changes add independently.
    combined_bound = None
    if all(left.isdisjoint(right) for i, (left, _) in enumerate(partial_subtrees)
           for right, _ in partial_subtrees[i+1:]):
        reach = 2 * np.sin(np.deg2rad(2.) / 2) * sum(length for _, length in partial_subtrees)
        combined_bound = float(max(0., np.linalg.norm(edge) - reach) / guard.lengths[index])
    report = dict(combinedLowerBoundAtTwoDegreesEach=combined_bound, stage="raw linear skinning", peak=float(guard.reference["edge"][index]),
                  reconstructionErrorM=reconstruction_error, joint=name, samples=samples,
                  minimumRatioAtTwoDegrees=joint_bounds, rankedMmPerDegree={n: float(1000*np.deg2rad(s)) for s,n,*_ in sorted(rows,key=lambda row:-row[0])})
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("runtime", "model", "pose", "source", "output"):
        parser.add_argument(arg, type=Path)
    probe(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
