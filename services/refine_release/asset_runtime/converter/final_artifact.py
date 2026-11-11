"""Final FBX import/skin integrity, independent of any display render.

``skin_verified`` confirms skin binding and finite geometry only. It never
certifies absence of penetration, containment, folding or safe joint limits.
The separate mesh_validation gate is required for refine-v3 certification.
"""
import hashlib
import math
from pathlib import Path

VERSION = 'final-fbx-v1'


def validate(path: Path, scope: str, expected_bones=None):
    from converter import retarget as rt
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    import bpy
    if len([o for o in bpy.context.scene.objects if o.type == 'ARMATURE']) != 1:
        raise ValueError('expected one final armature')
    bones = len(arm.data.bones)
    if not bones or (expected_bones is not None and bones != expected_bones):
        raise ValueError('final FBX skeleton changed')
    vertices = polygons = 0
    for mesh in meshes:
        if not mesh.data.vertices or not mesh.data.polygons:
            continue
        modifiers = [m for m in mesh.modifiers if m.type == 'ARMATURE']
        if len(modifiers) != 1 or modifiers[0].object != arm:
            raise ValueError('final FBX lost skin binding')
        groups = {g.index for g in mesh.vertex_groups if g.name in arm.data.bones}
        for vertex in mesh.data.vertices:
            if not all(math.isfinite(v) for v in vertex.co):
                raise ValueError('nonfinite final mesh')
            weights = [g.weight for g in vertex.groups if g.group in groups]
            if not weights or not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
                raise ValueError('final FBX has unbound vertices')
        vertices += len(mesh.data.vertices)
        polygons += len(mesh.data.polygons)
    if not vertices or not polygons or not all(math.isfinite(v) for b in arm.data.bones for row in b.matrix_local for v in row):
        raise ValueError('invalid final FBX geometry')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('validation modified final FBX')
    return {'version': VERSION, 'artifact_sha256': digest, 'scope': scope,
            'bones': bones, 'meshes': len(meshes), 'vertices': vertices, 'polygons': polygons,
            'skin_verified': True}


def valid_receipt(value, digest, scope):
    return (isinstance(value, dict) and value.get('version') == VERSION
            and value.get('artifact_sha256') == digest and value.get('scope') == scope
            and value.get('skin_verified') is True
            and all(type(value.get(k)) is int and value[k] > 0 for k in ('bones', 'meshes', 'vertices', 'polygons')))
