"""Blender child: crop reviewed FBX with the frozen anatomical framing code.

Writes temporary NumPy surfaces instead of another set of FBX/GLB copies.
Each job is one source pose; the caller removes its temporary files after rendering.
"""
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def surface(meshes, pivot):
    graph = bpy.context.evaluated_depsgraph_get()
    points, normals, indices = [], [], []
    offset = 0
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            co = np.empty(len(mesh.vertices) * 3, dtype='f4')
            mesh.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            loops = np.empty(len(mesh.loops), dtype='i4')
            mesh.loops.foreach_get('vertex_index', loops)
            ns = np.empty(len(mesh.corner_normals) * 3, dtype='f4')
            mesh.corner_normals.foreach_get('vector', ns)
            ns = ns.reshape(-1, 3)
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype='i4')
            mesh.loop_triangles.foreach_get('loops', triangles)
            matrix = np.array(evaluated.matrix_world)
            p = co[loops] @ matrix[:3, :3].T + matrix[:3, 3] - np.array(pivot)
            n = ns @ np.linalg.inv(matrix[:3, :3])
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            points.append(p[:, [0,2,1]] * [1,1,-1])
            normals.append(n[:, [0,2,1]] * [1,1,-1])
            indices.append(triangles + offset)
            offset += len(p)
        finally:
            evaluated.to_mesh_clear()
    return dict(positions=np.concatenate(points).astype('f4'), normals=np.concatenate(normals).astype('f4'),
                indices=np.concatenate(indices).astype('u4'))


def prepare(job):
    if job['converter_root'] not in sys.path:
        sys.path.insert(0, job['converter_root'])
    from converter import retarget as rt
    from converter.framing import crop_meshes
    from converter.bone_map import PROFILES, resolve_profile
    started = time.monotonic()
    rt.reset_scene()
    arm, originals = rt.import_character(job['fbx'])
    mapping = PROFILES[resolve_profile(list(arm.data.bones.keys()))]
    pivot = arm.matrix_world @ arm.pose.bones[mapping['hips']].head
    reports = {}
    for scope in ('half', 'bust', 'shoulder_head'):
        meshes = []
        for original in originals:
            duplicate = original.copy(); duplicate.data = original.data.copy()
            bpy.context.collection.objects.link(duplicate)
            meshes.append(duplicate)
        reports[scope] = crop_meshes(arm, meshes, scope)
        live = [m for m in meshes if m.name in bpy.data.objects]
        np.savez(Path(job['output']) / (scope + '.npz'), **surface(live, pivot))
        for mesh in live:
            data = mesh.data
            bpy.data.objects.remove(mesh, do_unlink=True)
            bpy.data.meshes.remove(data)
    reports['seconds'] = time.monotonic() - started
    receipt = Path(job['output'], 'framing.tmp')
    receipt.write_text(json.dumps(reports))
    receipt.replace(Path(job['output'], 'framing.json'))


if __name__ == '__main__':
    argument = sys.argv[sys.argv.index('--') + 1]
    if argument == '--worker':
        for line in sys.stdin:
            job = json.loads(line)
            try:
                prepare(job)
            except Exception as error:
                Path(job['output'], 'error.json').write_text(json.dumps({'error': repr(error)}))
    else:
        prepare(json.loads(Path(argument).read_text()))
