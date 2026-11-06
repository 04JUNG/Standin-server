"""Canonical target evidence from final FBX bytes, outside the frozen solver.

The GLB is an exact float32 encoding of the evaluated surface, with no camera
baked into the FBX. Structural skin binding is distinct from collision safety.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

VERSION = "selected-target-snapshot-v1"
TOLERANCE_M = 0.0001  # Existing scene roundtrip tolerance, never relaxed here.


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def revision():
    from converter.framed_model import revision as model_revision
    from converter.protocol import SOLVER_MANIFEST_SHA256
    root = Path(__file__).parent
    sources = b"".join((root / name).read_bytes() for name in (
        "selected_snapshot.py", "selected_snapshot_worker.py", "selected_convert_worker.py", "interactive_adapter.py", "worker.py", "protocol.py", "hands.py", "hand_geometry.py", "hand_rig.py", "hand_library.py", "hand_library.v1.json", "shape_spec.py", "final_artifact.py", "mesh_validation.py",
        "hand_contact.py", "shape_bake.py"))
    sources += (root.parent / "converter_api/selected_target.py").read_bytes()
    return sha(sources + model_revision().encode() + SOLVER_MANIFEST_SHA256.encode())


def empty_fingerprint():
    """Reset audit independent of solver reports; includes all resident datablocks."""
    import bpy
    collections = ("objects", "meshes", "armatures", "materials", "actions",
                   "shape_keys", "images", "textures", "node_groups", "collections",
                   "cameras", "lights", "worlds", "scenes")
    scene = bpy.context.scene
    return {"datablocks": {name: sorted(item.name for item in getattr(bpy.data, name))
                            for name in collections},
            "frame": [scene.frame_start, scene.frame_end, scene.frame_current],
            "unit": [scene.unit_settings.system, scene.unit_settings.scale_length],
            "render": [scene.render.engine, scene.render.fps, scene.render.resolution_x,
                       scene.render.resolution_y, scene.render.resolution_percentage]}


def _surface(arm, meshes):
    import bpy
    from converter.bone_map import PROFILES, resolve_profile
    profile = resolve_profile(list(arm.data.bones.keys()))
    pivot = arm.matrix_world @ arm.pose.bones[PROFILES[profile]["hips"]].head
    graph = bpy.context.evaluated_depsgraph_get()
    positions, normals, indices = [], [], []
    # Preserve every corner, including split normals. Appearance adapters may
    # replace the presentation but must retain this canonical geometry digest.
    for obj in sorted(meshes, key=lambda item: item.name):
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
            vertex_indices = {}
            for triangle in mesh.loop_triangles:
                for loop_index in triangle.loops:
                    vertex = mesh.loops[loop_index].vertex_index
                    p = evaluated.matrix_world @ mesh.vertices[vertex].co - pivot
                    n = (normal_matrix @ mesh.corner_normals[loop_index].vector).normalized()
                    key = (vertex, *n)
                    if key not in vertex_indices:
                        vertex_indices[key] = len(positions) // 3
                        positions.extend((p.x, p.z, -p.y))
                        normals.extend((n.x, n.z, -n.y))
                    indices.append(vertex_indices[key])
        finally:
            evaluated.to_mesh_clear()
    if not indices or not all(math.isfinite(v) for v in (*positions, *normals)):
        raise ValueError("invalid canonical surface")
    joints = {bone.name: {"parent": bone.parent.name if bone.parent else None,
                         "head": list(arm.matrix_world @ arm.pose.bones[bone.name].head),
                         "tail": list(arm.matrix_world @ arm.pose.bones[bone.name].tail)}
              for bone in arm.data.bones}
    encode(joints)  # Reject nonfinite FK, even when the mesh itself is finite.
    return positions, normals, indices, joints, list(pivot), profile


def _triangles(positions, indices):
    # Float32 is the preview wire precision. Triangle order may change during
    # UInt16 partitioning; winding and duplicate multiplicities must not change.
    result = Counter()
    for offset in range(0, len(indices), 3):
        corners = [struct.pack("<3f", *positions[3*i:3*i+3])
                   for i in indices[offset:offset+3]]
        result[min(tuple(corners[i:] + corners[:i]) for i in range(3))] += 1
    return result


def verify_glb(data, positions, indices):
    """Independently decode the serialized preview; metadata alone is insufficient."""
    from converter.framed_model import metadata
    metadata(data)
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20+length])
    binary = memoryview(data)[28+length:]
    all_positions, all_indices = [], []

    def accessor(index, component, shape):
        row = document["accessors"][index]
        if row["componentType"] not in component or row["type"] != shape or row.get("normalized"):
            raise ValueError("unsupported canonical accessor")
        view = document["bufferViews"][row["bufferView"]]
        if view.get("buffer") != 0 or view.get("byteStride") or row.get("sparse"):
            raise ValueError("unsupported canonical buffer")
        fmt, width = component[row["componentType"]]
        start = view.get("byteOffset", 0) + row.get("byteOffset", 0)
        size = row["count"] * width * struct.calcsize(fmt)
        if start < 0 or start + size > len(binary):
            raise ValueError("canonical accessor overflow")
        return struct.unpack_from(f"<{row['count'] * width}{fmt}", binary, start)

    if document["nodes"] != [{"mesh": 0}] or len(document["meshes"]) != 1:
        raise ValueError("canonical preview transforms changed")
    for part in document["meshes"][0]["primitives"]:
        values = accessor(part["attributes"]["POSITION"], {5126: ("f", 3)}, "VEC3")
        faces = accessor(part["indices"], {5123: ("H", 1), 5125: ("I", 1)}, "SCALAR")
        if len(faces) % 3 or any(i >= len(values)//3 for i in faces):
            raise ValueError("invalid canonical triangles")
        base = len(all_positions)//3
        all_positions.extend(values)
        all_indices.extend(base+i for i in faces)
    if _triangles(all_positions, all_indices) != _triangles(positions, indices):
        raise ValueError("preview surface differs from final FBX")
    maximum = max(abs(value - struct.unpack("<f", struct.pack("<f", value))[0])
                  for value in positions)
    if maximum > TOLERANCE_M:
        raise ValueError("preview float precision exceeds correspondence tolerance")
    return {"surface_verified": True, "max_coordinate_error_m": maximum,
            "tolerance_m": TOLERANCE_M, "triangles": len(indices)//3}


def inspect_final(path, destination, binding, *, full_body=True, mesh_seconds=3.0):
    """Reimport final bytes, inspect actual skin/FK/contact, then encode preview.

    No target/source solver, render, or FBX exporter is called. Factory-empty
    restoration is checked even on failure. Warm posed-target reuse is NOT
    enabled by this check; only sequential independent inspections may reuse a
    process after this exact empty baseline has been recovered.
    """
    from converter import retarget as rt
    from converter.final_artifact import validate
    from converter.shape_bake import import_full_character
    from converter.mesh_validation import audit_mesh, extract_evaluated_skin
    from converter.hand_contact import screen_hands, SCREEN_VERSION
    from converter.preview_model import encode_surface
    path, destination = Path(path), Path(destination)
    digest = sha(path.read_bytes())
    started = time.monotonic()
    rt.reset_scene()
    baseline = empty_fingerprint()
    try:
        previous_importer = rt.import_character
        rt.import_character = import_full_character
        try:
            structure = validate(path, binding["output_scope"])
        finally:
            rt.import_character = previous_importer
        rt.reset_scene()
        arm, meshes = import_full_character(str(path))
        positions, normals, indices, joints, pivot, profile = _surface(arm, meshes)
        vertices, faces = extract_evaluated_skin(meshes)
        mesh = audit_mesh(vertices, faces, full_body=full_body, seconds=mesh_seconds)
        hands = (screen_hands(meshes) if full_body else
                 {"status": "unavailable", "reason": "full_body_required_before_framing"})
        hands.update(method=SCREEN_VERSION, scope="opposite_hands_only",
                     complete_collision_certificate=False, screened_artifact_sha256=digest)
        snapshot = {"version": VERSION, "binding": binding, "artifact_sha256": digest,
                    "coordinates": "Y-up-hips-origin", "pivot_blender_m": pivot,
                    "joints_blender_m": joints, "rig_profile": profile,
                    "geometry_sha256": sha(struct.pack(f"<{len(positions)}f", *positions)),
                    "structure": structure, "mesh": mesh, "hands": hands}
        metadata = {"version": VERSION, "binding": binding, "artifact_sha256": digest,
                    "coordinates": "Y-up-hips-origin", "pivot_blender_m": pivot}
        data = encode_surface(positions, normals, indices, metadata)
        correspondence = verify_glb(data, positions, indices)
        destination.write_bytes(data)
        snapshot.update(preview_sha256=sha(data), preview_size=len(data),
                        correspondence=correspondence)
        if sha(path.read_bytes()) != digest:
            raise ValueError("inspection mutated final FBX")
    finally:
        rt.reset_scene()
        restored = empty_fingerprint()
        if restored != baseline:
            raise ValueError("candidate inspection did not restore empty baseline")
    snapshot.update(baseline_sha256=sha(encode(baseline)),
                    restored_sha256=sha(encode(restored)), baseline_restored=True,
                    elapsed_ms=(time.monotonic()-started)*1000)
    return snapshot
