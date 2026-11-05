"""Offline, static GLB from the evaluated surface of a validated FBX.

No new retargeting: the frozen converter must succeed first. Vertices are Y-up,
relative to Hips, so the candidate-camera matrix applies directly at the origin.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.protocol import SOLVER_MANIFEST_SHA256, EXPECTED_BLENDER_BUILD_HASH

MODEL_VERSION = "posed-mesh-v1"
MAX_MODEL_BYTES = 8 * 1024 * 1024


def model_revision() -> str:
    # Source edits and frozen runtime changes invalidate precomputed assets.
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    fingers = Path(__file__).with_name("source_fingers.py").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source + fingers + SOLVER_MANIFEST_SHA256.encode()
                          + EXPECTED_BLENDER_BUILD_HASH.encode()).hexdigest()


def model_key(source_sha: str, character_sha: str) -> str:
    for digest in (source_sha, character_sha):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("expected lowercase SHA256")
    digest = hashlib.sha256(f"{model_revision()}:{character_sha}:{source_sha}".encode()).hexdigest()
    return f"{MODEL_VERSION}/{digest}"


MAX_PRIMITIVES = 8
MAX_PRIMITIVE_INDICES = 499_998


def partition_surface(positions, normals, indices, *, max_vertices=65_535):
    """Remap complete triangles into UInt16 parts, preserving every corner value.

    Neither positions, split normals nor triangle winding are approximated.
    Boundary vertices can repeat across parts; original FBX/BVH remain untouched.
    """
    parts, mapping, points, directions, triangles = [], {}, [], [], []
    if not 3 <= max_vertices <= 65_535 or len(indices) % 3:
        raise ValueError("invalid indexed triangle surface")
    def flush():
        nonlocal mapping, points, directions, triangles
        if triangles:
            parts.append((points, directions, triangles, 5123))
        if len(parts) > MAX_PRIMITIVES:
            raise ValueError("preview model has too many parts")
        mapping, points, directions, triangles = {}, [], [], []
    # FBX polygon order can jump around the body and repeat most vertices in each
    # part. Walk adjacent faces so only boundary vertices need to repeat. The
    # single opaque material makes face order immaterial to the exact surface.
    adjacency = [[] for _ in range(len(positions) // 3)]
    for offset in range(0, len(indices), 3):
        for source in indices[offset:offset + 3]:
            adjacency[source].append(offset // 3)
    seen = bytearray(len(indices) // 3)
    def faces():
        for seed in range(len(seen)):
            if seen[seed]:
                continue
            pending = deque([seed])
            seen[seed] = 1
            while pending:
                face = pending.popleft()
                yield face * 3
                for source in indices[face * 3:face * 3 + 3]:
                    for neighbor in adjacency[source]:
                        if not seen[neighbor]:
                            seen[neighbor] = 1
                            pending.append(neighbor)
                    adjacency[source].clear()
    for offset in faces():
        face = indices[offset:offset + 3]
        if (len(mapping) + sum(source not in mapping for source in set(face)) > max_vertices
                or len(triangles) + 3 > MAX_PRIMITIVE_INDICES):
            flush()
        for source in face:
            if source not in mapping:
                mapping[source] = len(mapping)
                points.extend(positions[3 * source:3 * source + 3])
                directions.extend(normals[3 * source:3 * source + 3])
            triangles.append(mapping[source])
    flush()
    return parts


def _encode_parts(parts, metadata):
    chunks, views, accessors, primitives = [], [], [], []
    offset = 0
    for positions, normals, indices, component in parts:
        base = len(accessors)
        values = ((positions, "f", 5126, "VEC3", 3),
                  (normals, "f", 5126, "VEC3", 3),
                  (indices, "H" if component == 5123 else "I", component, "SCALAR", 1))
        for values_array, packing, gltf_type, shape, width in values:
            chunk = struct.pack(f"<{len(values_array)}{packing}", *values_array)
            view = len(views)
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(chunk)})
            accessor = {"bufferView": view, "componentType": gltf_type,
                        "count": len(values_array) // width, "type": shape}
            if view % 3 == 0:
                accessor.update(min=[min(positions[i::3]) for i in range(3)],
                                max=[max(positions[i::3]) for i in range(3)])
            accessors.append(accessor)
            chunk += b"\0" * (-len(chunk) % 4)
            chunks.append(chunk)
            offset += len(chunk)
        primitives.append({"attributes": {"POSITION": base, "NORMAL": base + 1},
                           "indices": base + 2, "material": 0})
    binary = b"".join(chunks)
    document = {
        "asset": {"version": "2.0", "generator": MODEL_VERSION, "extras": metadata},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.72, .74, .76, 1],
                                               "metallicFactor": 0, "roughnessFactor": .8}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views, "accessors": accessors,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def encode_surface(positions, normals, indices, metadata):
    """Use bounded UInt16 parts when the exact UInt32 surface exceeds the budget."""
    if not indices or len(indices) % 3 or len(positions) != len(normals):
        raise ValueError("invalid indexed triangle surface")
    data = _encode_parts([(positions, normals, indices, 5125)], metadata)
    # The desktop also bounds each accessor; dense bodies can exceed the index
    # count before reaching the byte limit even with fewer than 500k vertices.
    if len(data) > MAX_MODEL_BYTES or len(indices) > MAX_PRIMITIVE_INDICES:
        data = _encode_parts(partition_surface(positions, normals, indices), metadata)
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("preview model exceeds size limit")
    return data


def bake_fbx(fbx: Path, destination: Path, identity: dict) -> dict:
    import bpy
    from converter import retarget as rt
    from converter.bone_map import PROFILES, resolve_profile

    rt.reset_scene()
    arm, meshes = rt.import_character(str(fbx))
    profile = resolve_profile(list(arm.data.bones.keys()))
    pivot = arm.matrix_world @ arm.pose.bones[PROFILES[profile]["hips"]].head
    depsgraph = bpy.context.evaluated_depsgraph_get()
    positions, normals, indices = [], [], []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
            vertex_indices = {}
            # Per-corner vertices preserve split normals (eyes/skin/cut edges).
            for triangle in mesh.loop_triangles:
                for loop_index in triangle.loops:
                    loop = mesh.loops[loop_index]
                    p = evaluated.matrix_world @ mesh.vertices[loop.vertex_index].co - pivot
                    n = (normal_matrix @ mesh.corner_normals[loop_index].vector).normalized()
                    key = (loop.vertex_index, *n)
                    if key not in vertex_indices:
                        vertex_indices[key] = len(positions) // 3
                        positions.extend((p.x, p.z, -p.y))
                        normals.extend((n.x, n.z, -n.y))
                    indices.append(vertex_indices[key])
        finally:
            evaluated.to_mesh_clear()
    if not indices:
        raise ValueError("empty evaluated character")
    metadata = {**identity, "version": MODEL_VERSION, "revision": model_revision(),
                "coordinates": "Y-up-hips-origin", "scope": "full",
                "solver_manifest_sha256": SOLVER_MANIFEST_SHA256,
                "base_fbx_sha256": hashlib.sha256(fbx.read_bytes()).hexdigest()}
    data = encode_surface(positions, normals, indices, metadata)
    destination.write_bytes(data)
    return {**metadata, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:]
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    manifest = bake_fbx(Path(job["fbx"]), Path(job["glb"]), job["identity"])
    Path(job["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
