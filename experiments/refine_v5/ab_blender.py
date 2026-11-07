"""Read and audit evaluated Blender meshes for the fixed A/B comparison."""

import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector
from mathutils.kdtree import KDTree
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_mesh(meshes, names):
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, faces, weights = [], [], []
    bone_indices = {name: index for index, name in enumerate(names)}
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            group_indices = {g.index: bone_indices.get(g.name.split(':')[-1]) for g in obj.vertex_groups}
            weight = np.zeros((len(mesh.vertices), len(names)), np.float32)
            offset = sum(len(chunk) for chunk in vertices)
            vertices.append(np.array([evaluated.matrix_world @ v.co for v in mesh.vertices]))
            for vertex in mesh.vertices:
                for influence in vertex.groups:
                    index = group_indices.get(influence.group)
                    if index is not None:
                        weight[vertex.index, index] += influence.weight
            weights.append(weight)
            mesh.calc_loop_triangles()
            faces.extend([[offset + i for i in triangle.vertices] for triangle in mesh.loop_triangles])
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not faces:
        raise ValueError('missing evaluated skin')
    return np.concatenate(vertices), np.asarray(faces, int), np.concatenate(weights)


def bone_world(armature):
    return {b.name.split(':')[-1]: np.array(armature.matrix_world @ b.matrix)
            for b in armature.pose.bones}


def verify_plan(armature, expected):
    world = bone_world(armature)
    if world.keys() != expected.keys():
        raise ValueError('bone inventory changed')
    error = max(np.linalg.norm(world[name][:3, 3] - point) for name, point in expected.items())
    if error > 1e-5:
        raise ValueError(f'FK and Blender disagree: {error:.6g}m')
    return float(error)


def apply_rotations(armature, edits):
    by_name = {b.name.split(':')[-1]: b for b in armature.pose.bones}
    for name, components in edits.items():
        vector = Vector(np.deg2rad(components))
        if vector.length > 1e-12:
            rotation = Quaternion(vector.normalized(), vector.length).to_matrix().to_4x4()
            by_name[name].matrix_basis = by_name[name].matrix_basis @ rotation
    bpy.context.view_layer.update()


def prepare(root, cases, load):
    for case in cases:
        if sha(case['source']) != case['source_sha256']:
            raise ValueError('stale B0')
        armature, meshes = load(case['source'])
        names = [b['name'] for b in case['bones']]
        heads = {b['name']: np.array(b['matrix'])[:3, 3] for b in case['bones']}
        verify_plan(armature, heads)
        v, f, w = weighted_mesh(meshes, names)
        np.savez_compressed(root / (case['case_id'] + '-B0.npz'), vertices=v, faces=f, weights=w)
        print('AB_B0', case['case_id'], len(v), len(f), flush=True)


def cloud_distance(first, second):
    tree = KDTree(len(second))
    for i, point in enumerate(second):
        tree.insert(Vector(point), i)
    tree.balance()
    return max(tree.find(Vector(point))[2] for point in first)


def main():
    command, root_arg, *args = sys.argv[sys.argv.index('--') + 1:]
    root = Path(root_arg)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.refine_v5.audit_iterative_native import load
    cases = json.loads((root / 'manifest.json').read_text())['cases']
    if command == 'prepare':
        prepare(root, cases, load)
    elif command == 'audit':
        from experiments.refine_v5.ab_native_audit import audit
        audit(root, cases, Path(args[0]), load)
    else:
        raise ValueError('unknown action')


if __name__ == '__main__':
    main()
