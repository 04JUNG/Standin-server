"""Blender-only snapshot of rigged-rest skin, in the display coordinate frame.

No animation, scripts, external resources, dropped weights or shader payloads.
All pose/shape/hand surface corrections are already baked by approved adapters.
"""
from __future__ import annotations

import math
from converter.solved_state import MODEL_SCHEMA, validate_model

# Existing scene round-trip geometry tolerance, never a release threshold change.
POSITION_TOLERANCE_M = 0.0001
MATRIX_TOLERANCE = 1e-5
WEIGHT_TOLERANCE = 1e-6


def _flat(matrix):
    return [float(matrix[r][c]) for r in range(4) for c in range(4)]


def _checked_scene():
    """Scene invariants every capture enforces; returns (objects, armature, profile)."""
    import bpy
    from converter.bone_map import resolve_profile
    bpy.context.view_layer.update()
    objects = list(bpy.context.scene.objects)
    arms = [o for o in objects if o.type == 'ARMATURE']
    if len(arms) != 1 or any(o.type not in {'ARMATURE', 'MESH', 'EMPTY'} for o in objects):
        raise ValueError('snapshot requires a single private character scene')
    arm = arms[0]
    if any(o.animation_data and (o.animation_data.action or o.animation_data.nla_tracks) for o in objects):
        raise ValueError('snapshot contains animation')
    if any(abs(float(pb.matrix_basis[r][c]) - (r == c)) > MATRIX_TOLERANCE
           for pb in arm.pose.bones for r in range(4) for c in range(4)):
        raise ValueError('snapshot must have all deformation baked to rigged_rest')
    return objects, arm, resolve_profile(list(arm.data.bones.keys()))


def _hip_pivot(arm, profile):
    from converter.bone_map import PROFILES
    return arm.matrix_world @ arm.data.bones[PROFILES[profile]['hips']].head_local


def scene_origin():
    """The originBlender capture() would report, without traversing the surface.

    Same scene invariants and the same mathutils expression as capture(), so the
    value is bit-identical; the solve stage only needs this before freezing the FBX.
    """
    _, arm, profile = _checked_scene()
    return list(_hip_pivot(arm, profile))


def capture(origin=None):
    import bpy
    from mathutils import Matrix, Vector
    objects, arm, profile = _checked_scene()
    pivot = Vector(origin) if origin is not None else _hip_pivot(arm, profile)
    conversion = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1))) @ Matrix.Translation(-pivot)
    order = []
    def visit(bone):
        order.append(bone)
        for child in sorted(bone.children, key=lambda b: b.name):
            visit(child)
    for root in sorted((b for b in arm.data.bones if b.parent is None), key=lambda b: b.name):
        visit(root)
    indices = {b.name: i for i, b in enumerate(order)}
    world = {b.name: conversion @ arm.matrix_world @ b.matrix_local for b in order}
    bones = []
    for b in order:
        local = world[b.parent.name].inverted() @ world[b.name] if b.parent else world[b.name]
        bones.append(dict(name=b.name, parent=indices[b.parent.name] if b.parent else -1,
            restLocal=_flat(local), poseLocal=_flat(local), inverseBind=_flat(world[b.name].inverted())))
    meshes = []
    for obj in sorted((o for o in objects if o.type == 'MESH'), key=lambda o: o.name):
        if not obj.data.vertices or not obj.data.polygons:
            continue
        mods = list(obj.modifiers)
        if len(mods) != 1 or mods[0].type != 'ARMATURE' or mods[0].object != arm or obj.data.shape_keys:
            raise ValueError('snapshot requires one baked skin modifier per mesh')
        # Read the same evaluated skin that use_mesh_modifiers=True exports.
        # Even rigged-rest can have float-level identity-modifier normal changes.
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.data
        mesh.calc_loop_triangles()
        matrix = conversion @ evaluated.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()
        weights = []
        for vertex in mesh.vertices:
            entries = []
            for group in vertex.groups:
                name = obj.vertex_groups[group.group].name
                if group.weight > 0:
                    if name not in indices:
                        raise ValueError('non-bone weighted group cannot be exported')
                    entries.append([indices[name], float(group.weight)])
            weights.append(sorted(entries))
        # Preserve FBX custom split normals. Vertex.normal silently discards
        # these and can disagree after an otherwise identical FBX round trip.
        # Index identical split normals without rounding. A smooth vertex often
        # repeats its exact normal in six corners; preserve real hard edges
        # while avoiding six copies in canonical JSON and browser validation.
        normals = []; normal_indices = {}; loops = []
        for normal in mesh.corner_normals:
            value = tuple((normal_matrix @ normal.vector).normalized())
            index = normal_indices.get(value)
            if index is None:
                index = len(normals); normal_indices[value] = index
                normals.append(list(value))
            loops.append(index)
        meshes.append(dict(name=f'mesh_{len(meshes):03d}',
            positions=[list(matrix @ v.co) for v in mesh.vertices],
            normals=normals,
            normalTriangles=[[loops[i] for i in t.loops] for t in mesh.loop_triangles],
            triangles=[list(t.vertices) for t in mesh.loop_triangles], weights=weights))
    result = dict(schema=MODEL_SCHEMA, coordinates=dict(space='right-handed-y-up-metres-hip-origin',
        originBlender=list(pivot), matrixLayout='row-major', baked='rigged_rest'),
        bones=bones, meshes=meshes, capabilities={'localHands': False})
    validate_model(result)
    return result


def compare(expected, actual):
    """Compare serialized output to the exact packet, including bind/skin identity."""
    validate_model(expected); validate_model(actual)
    if expected['coordinates'] != actual['coordinates'] or len(expected['bones']) != len(actual['bones']) or len(expected['meshes']) != len(actual['meshes']):
        raise ValueError('snapshot coordinate/rig/mesh identity changed')
    matrix_error = position_error = weight_error = normal_error = 0.0
    for a, b in zip(expected['bones'], actual['bones']):
        if (a['name'], a['parent']) != (b['name'], b['parent']):
            raise ValueError('snapshot bone identity changed')
        for field in ('restLocal', 'poseLocal', 'inverseBind'):
            for i, (x, y) in enumerate(zip(a[field], b[field])):
                error = abs(x-y)
                tolerance = POSITION_TOLERANCE_M if i in (3, 7, 11) else MATRIX_TOLERANCE
                if not math.isfinite(error) or error > tolerance:
                    raise ValueError('snapshot bind/pose matrix changed')
                matrix_error = max(matrix_error, error)
    for a, b in zip(expected['meshes'], actual['meshes']):
        if (a['name'] != b['name'] or a['triangles'] != b['triangles'] or a['normalTriangles'] != b['normalTriangles']
                or len(a['positions']) != len(b['positions']) or len(a['normals']) != len(b['normals'])):
            raise ValueError('snapshot topology changed')
        for x, y in zip(a['positions'], b['positions']):
            error = math.dist(x, y)
            if not math.isfinite(error) or error > POSITION_TOLERANCE_M:
                raise ValueError('snapshot vertex changed')
            position_error = max(position_error, error)
        for x, y in zip(a['normals'], b['normals']):
            error = math.dist(x, y)
            if not math.isfinite(error) or error > 1e-4:
                raise ValueError('snapshot normal changed')
            normal_error = max(normal_error, error)
        for x, y in zip(a['weights'], b['weights']):
            if [i for i, _ in x] != [i for i, _ in y]:
                raise ValueError('snapshot skin joint identity changed')
            for (_, wx), (_, wy) in zip(x, y):
                error = abs(wx-wy)
                if not math.isfinite(error) or error > WEIGHT_TOLERANCE:
                    raise ValueError('snapshot skin weight changed')
                weight_error = max(weight_error, error)
    return dict(maxPositionErrorM=position_error, maxMatrixError=matrix_error,
                maxWeightError=weight_error, maxNormalError=normal_error)
