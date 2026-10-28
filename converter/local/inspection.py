"""Conservative personal-rig inspection, executed only inside Blender children."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from contextlib import contextmanager

import bpy
from mathutils import Matrix
from mathutils.kdtree import KDTree

from converter import bone_map
from converter.local.contract import LIMITS, LocalError


def require(condition, code, message):
    if not condition:
        raise LocalError(code, message)


@contextmanager
def bounded_fbx_parser():
    """Bound declared and actual compressed arrays before Blender allocates them.

    This adapter intentionally targets Blender 5.2.0's pinned parser API. It is
    installed only inside one isolated job and covers both preflight and import.
    """
    from io_scene_fbx import parse_fbx
    unpack_array = parse_fbx.unpack_array
    decompress_array = parse_fbx._decompress_and_insert_array
    expanded = 0
    def bounded_unpack(read, array_type, stride, byteswap):
        nonlocal expanded
        header = read(12)
        count, encoding, compressed = struct.unpack("<III", header)
        expanded += count * stride
        require(expanded <= LIMITS["expandedBytes"] and compressed <= LIMITS["sourceBytes"]
                and encoding in {0, 1}, "LIMIT_EXCEEDED", "FBX arrays exceed the expanded-data limit")
        first = True
        def replay(n):
            nonlocal first
            if first:
                first = False
                return header
            return read(n)
        return unpack_array(replay, array_type, stride, byteswap)
    def bounded_decompress(props, index, args):
        data, count, array_type, stride, byteswap = args
        expected = count * stride
        inflater = zlib.decompressobj()
        unpacked = inflater.decompress(data, expected + 1)
        require(len(unpacked) == expected and inflater.eof and not inflater.unused_data,
                "LIMIT_EXCEEDED", "FBX array compressed length does not match its declaration")
        props[index] = parse_fbx._create_array(unpacked, count, array_type, stride, byteswap)
    parse_fbx.unpack_array = bounded_unpack
    parse_fbx._decompress_and_insert_array = bounded_decompress
    try:
        yield parse_fbx
    finally:
        parse_fbx.unpack_array = unpack_array
        parse_fbx._decompress_and_insert_array = decompress_array


def import_fbx(path):
    # Do not use the server importer: it drops terminal bones and bakes object transforms.
    # Preflight before Blender can resolve an absolute/relative external image.
    # Texture delivery needs an explicit selected-asset scope; v1 accepts solid
    # materials only, including refusing embedded images before extraction.
    from io_scene_fbx import parse_fbx
    try:
        with bounded_fbx_parser():
            tree, _ = parse_fbx.parse(str(path))
        pending = [tree]
        while pending:
            element = pending.pop()
            require(element.id not in {b"Texture", b"Video"}, "UNSUPPORTED_MATERIAL",
                    "Texture assets are not certified in this local runtime")
            pending.extend(element.elems)
        del tree, pending
    except LocalError:
        raise
    except Exception:
        raise LocalError("UNSUPPORTED_FORMAT", "Unsupported or malformed binary FBX") from None
    with bounded_fbx_parser():
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False, use_image_search=False,
                                 ignore_leaf_bones=False, automatic_bone_orientation=False)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    require(len(arms) == 1 and bool(meshes), "UNSUPPORTED_RIG", "Exactly one skinned character is required")
    arm = arms[0]
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data_clear()
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    return arm, meshes


def matrix(m):
    return [[float(x) for x in row] for row in m]


def solid_materials(meshes):
    materials = {}
    for mesh in meshes:
        for slot in mesh.material_slots:
            mat = slot.material
            if mat is None:
                continue
            color = list(mat.diffuse_color)
            scalar = {"Metallic": mat.metallic, "Roughness": mat.roughness}
            if mat.use_nodes:
                nodes = mat.node_tree.nodes
                principled = [n for n in nodes if n.type == "BSDF_PRINCIPLED"]
                inert_normals = [n for n in nodes if n.type == "NORMAL_MAP" and
                                not any(s.is_linked for s in n.inputs) and n.inputs["Strength"].default_value == 0]
                require(len(principled) == 1 and all(n.type in {"BSDF_PRINCIPLED", "OUTPUT_MATERIAL"}
                                                   or n in inert_normals for n in nodes),
                        "UNSUPPORTED_MATERIAL", "Only solid Principled materials are certified")
                shader = principled[0]
                require(not any(s.is_linked and not (s.name == "Normal" and
                            all(link.from_node in inert_normals for link in s.links)) for s in shader.inputs), "UNSUPPORTED_MATERIAL",
                        "Texture and procedural materials require a separate compatibility adapter")
                require(abs(shader.inputs["Alpha"].default_value - 1) < 1e-6,
                        "UNSUPPORTED_MATERIAL", "Transparent materials are not yet certified")
                color = list(shader.inputs["Base Color"].default_value)
                scalar = {key: float(shader.inputs[key].default_value)
                          for key in ("Metallic", "Roughness", "Specular IOR Level")}
                require(all(abs(shader.inputs[key].default_value) < 1e-6 for key in
                            ("Subsurface Weight", "Transmission Weight", "Coat Weight", "Sheen Weight", "Emission Strength")),
                        "UNSUPPORTED_MATERIAL", "Layered and emissive materials are not certified")
            require(all(math.isfinite(x) for x in color), "UNSUPPORTED_MATERIAL", "Invalid material values")
            materials[mat.name] = {"baseColor": color, "scalars": scalar, "support": "solid_principled"}
    return materials


def inspect(arm, meshes):
    from converter.local.rig_profiles import register
    register()
    bones = arm.data.bones
    require(len(meshes) <= LIMITS["meshes"] and len(bones) <= LIMITS["bones"]
            and sum(len(m.data.vertices) for m in meshes) <= LIMITS["vertices"],
            "LIMIT_EXCEEDED", "Character exceeds the geometry limits")
    try:
        profile = bone_map.resolve_profile(list(bones.keys()))
    except ValueError:
        raise LocalError("UNSUPPORTED_RIG", "Unknown humanoid skeleton") from None
    table = {c: n for c, n in bone_map.PROFILES[profile].items() if n in bones}
    required = bone_map.REQUIRED_BONES | {"hand.L", "hand.R", "foot.L", "foot.R"}
    require(required <= table.keys(), "UNSUPPORTED_RIG", "Required body joints are missing")
    inverse = {n: c for c, n in table.items()}
    for canon, name in table.items():
        expected = bone_map.CANONICAL_PARENT[canon]
        while expected and expected not in table:
            expected = bone_map.CANONICAL_PARENT[expected]
        parent = bones[name].parent
        while parent and parent.name not in inverse:
            parent = parent.parent
        require((inverse[parent.name] if parent else None) == expected,
                "INVALID_HIERARCHY", "Body joint ancestry does not match the known skeleton")
        require(bones[name].length > 1e-7, "UNSUPPORTED_RIG", "Degenerate body joint")
    for obj in [arm, *meshes]:
        require(all(math.isfinite(x) for row in obj.matrix_world for x in row),
                "UNSUPPORTED_TRANSFORM", "Non-finite object transform")
        scale = obj.matrix_world.to_scale()
        require(obj.matrix_world.determinant() > 0 and min(scale) > 1e-6
                and max(scale) / min(scale) < 1.0001,
                "UNSUPPORTED_TRANSFORM", "Reflected or nonuniform object transforms are not certified")
        require(not obj.constraints, "UNSUPPORTED_FEATURE", "Object constraints are not supported")
    for pb in arm.pose.bones:
        require(not pb.constraints, "UNSUPPORTED_FEATURE", "Bone constraints are not supported")
        require(all(math.isfinite(x) for row in pb.bone.matrix_local for x in row),
                "UNSUPPORTED_TRANSFORM", "Non-finite bind transform")
    def point(c):
        return arm.matrix_world @ bones[table[c]].head_local
    # Known T-pose, Blender Z-up, left is +X. No guess based on names alone.
    for side, sign in (("L", 1), ("R", -1)):
        arm_axis = point("hand." + side) - point("upperarm." + side)
        leg_axis = point("foot." + side) - point("upleg." + side)
        require(arm_axis.length > 1e-6 and arm_axis.x * sign / arm_axis.length > .95
                and leg_axis.length > 1e-6 and -leg_axis.z / leg_axis.length > .9,
                "UNSUPPORTED_REST_POSE", "Only validated upright T-pose rigs are supported")
    mesh_info = []
    for mesh in meshes:
        require(not mesh.data.shape_keys, "UNSUPPORTED_FEATURE", "Morph preservation is not certified")
        mods = list(mesh.modifiers)
        require(len(mods) == 1 and mods[0].type == "ARMATURE" and mods[0].object == arm,
                "INVALID_SKIN", "Each mesh must use the character armature only")
        require(not mods[0].use_deform_preserve_volume, "UNSUPPORTED_FEATURE",
                "Dual quaternion skinning is not certified for FBX export")
        groups = {g.index: g.name for g in mesh.vertex_groups}
        weights = hashlib.sha256()
        for vertex in mesh.data.vertices:
            require(all(math.isfinite(x) for x in vertex.co), "INVALID_SKIN", "Non-finite vertex")
            entries = sorted((groups[g.group], g.weight) for g in vertex.groups if g.weight > 0)
            require(entries and all(name in bones and math.isfinite(w) and 0 < w <= 1 for name, w in entries)
                    and abs(sum(w for _, w in entries) - 1) <= 1e-4,
                    "INVALID_SKIN", "Missing or invalid original skin weights")
            for name, weight in entries:
                weights.update(name.encode() + b"\0" + struct.pack("<f", weight))
        mesh_info.append({"name": mesh.name, "vertices": len(mesh.data.vertices),
                          "polygons": len(mesh.data.polygons), "worldTransform": matrix(mesh.matrix_world),
                          "weightsHash": weights.hexdigest(),
                          "materials": [s.material.name if s.material else None for s in mesh.material_slots]})
    hands = {}
    for side, letter in (("left", "L"), ("right", "R")):
        wrist = bones[table["hand." + letter]]
        descendants = list(wrist.children_recursive)
        fingers = [b for b in descendants if any(t in b.name.lower() for t in ("thumb", "index", "middle", "ring", "pinky"))]
        hands[side] = {"supported": False, "fingerBones": len(fingers),
                       "reason": "UNVALIDATED_FINGER_RIG" if fingers else "NO_FINGER_RIG"}
    character = {"profile": profile, "mapping": table,
                 "units": {"axisUp": "Z", "scaleLength": bpy.context.scene.unit_settings.scale_length,
                           "physicalUnitConfirmed": False},
                 "armatureWorldTransform": matrix(arm.matrix_world),
                 "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None,
                            "restMatrix": matrix(b.matrix_local),
                            "inverseBindMatrix": matrix((arm.matrix_world @ b.matrix_local).inverted())}
                           for b in bones],
                 "meshes": mesh_info, "materials": solid_materials(meshes)}
    if "standin_cs3c_provenance" in arm:
        character["sourceFormatDetails"] = json.loads(arm["standin_cs3c_provenance"])
    return character, {"supportLevel": "body_only", "bodyPose": True, "hands": hands,
                       "collisionPrevention": False, "shapeSliders": False}


def surface(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in meshes:
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        result[obj.name] = [tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        evaluated.to_mesh_clear()
    return result


def surface_error(a, b):
    require(set(a) == set(b), "ROUNDTRIP_FAILED", "Mesh identities changed during FBX roundtrip")
    error = 0.
    for key in a:
        require(len(a[key]) == len(b[key]), "ROUNDTRIP_FAILED", "Vertex count changed during FBX roundtrip")
        for left, right in ((a[key], b[key]), (b[key], a[key])):
            tree = KDTree(len(right))
            for i, p in enumerate(right):
                tree.insert(p, i)
            tree.balance()
            error = max(error, max((tree.find(p)[2] for p in left), default=0))
    return error


def deformation_probe(arm, meshes, table, joint="forearm.L"):
    before = surface(meshes)
    bone = arm.pose.bones[table[joint]]
    saved = bone.matrix_basis.copy()
    bone.matrix_basis = saved @ Matrix.Rotation(.25, 4, "X")
    bpy.context.view_layer.update()
    posed = surface(meshes)
    movement = max((math.dist(a, b) for key in before for a, b in zip(before[key], posed[key])), default=0)
    bone.matrix_basis = saved
    bpy.context.view_layer.update()
    require(movement > 1e-6, "INVALID_SKIN", "Body FK probe did not deform the character")
    return {"joint": joint, "rotationRadians": .25, "maxVertexDisplacement": movement,
            "surface": posed}


def material_regions(meshes, points):
    """Per-material face centers: independent of FBX face/slot reordering."""
    regions = {}
    for mesh in meshes:
        for polygon in mesh.data.polygons:
            material = (mesh.material_slots[polygon.material_index].material
                        if polygon.material_index < len(mesh.material_slots) else None)
            key = (mesh.name, material.name if material else None)
            center = tuple(sum(points[mesh.name][v][axis] for v in polygon.vertices) / len(polygon.vertices)
                           for axis in range(3))
            regions.setdefault(str(key), []).append(center)
    return regions


def roundtrip(arm, meshes, path, table):
    from converter import retarget as rt
    before = surface(meshes)
    original_bones = {b.name: (b.parent.name if b.parent else None,
                              tuple(arm.matrix_world @ b.head_local)) for b in arm.data.bones}
    material_before = solid_materials(meshes)
    regions_before = material_regions(meshes, before)
    polygons_before = {m.name: len(m.data.polygons) for m in meshes}
    # Probe the *output* rest pose too: this checks that exported skin still follows bones.
    probe_before = {joint: deformation_probe(arm, meshes, table, joint)
                    for joint in ("forearm.L", "forearm.R", "leg.L", "leg.R", "spine")}
    rt.export_fbx(str(path), embed_textures=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    after_arm, after_meshes = import_fbx(path)
    after_points = surface(after_meshes)
    error = surface_error(before, after_points)
    extent = max((math.dist(p, (0, 0, 0)) for points in before.values() for p in points), default=1)
    tolerance = max(1e-5, extent * 1e-5)
    require(error <= tolerance, "ROUNDTRIP_FAILED", "Posed surface changed during FBX roundtrip")
    material_region_error = surface_error(regions_before, material_regions(after_meshes, after_points))
    require(material_region_error <= tolerance, "ROUNDTRIP_FAILED", "Face material regions changed during FBX roundtrip")
    for name, (parent, head) in original_bones.items():
        bone = after_arm.data.bones.get(name)
        require(bone is not None and (bone.parent.name if bone.parent else None) == parent,
                "ROUNDTRIP_FAILED", "Original bone hierarchy changed during FBX roundtrip")
        require(math.dist(head, after_arm.matrix_world @ bone.head_local) <= tolerance,
                "ROUNDTRIP_FAILED", "Bone position changed during FBX roundtrip")
    require(polygons_before == {m.name: len(m.data.polygons) for m in after_meshes},
            "ROUNDTRIP_FAILED", "Mesh topology changed during FBX roundtrip")
    after_materials = solid_materials(after_meshes)
    require(material_before.keys() == after_materials.keys(), "ROUNDTRIP_FAILED", "Material identities changed")
    for name, mat in material_before.items():
        require(max(abs(x-y) for x, y in zip(mat["baseColor"], after_materials[name]["baseColor"])) < 1e-4,
                "ROUNDTRIP_FAILED", "Solid material color changed")
        require(mat["scalars"].keys() == after_materials[name]["scalars"].keys()
                and all(abs(v-after_materials[name]["scalars"][k]) < 1e-4 for k, v in mat["scalars"].items()),
                "ROUNDTRIP_FAILED", "Solid material properties changed")
    probe_after = {joint: deformation_probe(after_arm, after_meshes, table, joint) for joint in probe_before}
    probe_error = max(surface_error(probe_before[j].pop("surface"), probe_after[j].pop("surface"))
                      for j in probe_before)
    require(probe_error <= tolerance, "ROUNDTRIP_FAILED", "Skin deformation changed during FBX roundtrip")
    return {"passed": True, "meshCount": len(after_meshes), "originalBoneCount": len(original_bones),
            "vertexCount": sum(len(p) for p in before.values()), "maxSurfaceError": error,
            "maxSkinProbeError": probe_error, "maxMaterialRegionError": material_region_error,
            "materialCount": len(material_before), "tolerance": tolerance, "fkProbes": probe_after,
            "blenderVersion": bpy.app.version_string}
