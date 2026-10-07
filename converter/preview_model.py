"""Offline, static GLB from the evaluated surface of a validated FBX.

No new retargeting: the frozen converter must succeed first. Vertices are Y-up,
relative to Hips, so the candidate-camera matrix applies directly at the origin.
"""
from __future__ import annotations

import hashlib
import json
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
    return hashlib.sha256(source + SOLVER_MANIFEST_SHA256.encode()
                          + EXPECTED_BLENDER_BUILD_HASH.encode()).hexdigest()


def model_key(source_sha: str, character_sha: str) -> str:
    for digest in (source_sha, character_sha):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("expected lowercase SHA256")
    digest = hashlib.sha256(f"{model_revision()}:{character_sha}:{source_sha}".encode()).hexdigest()
    return f"{MODEL_VERSION}/{digest}"


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
    chunks = [struct.pack(f"<{len(positions)}f", *positions),
              struct.pack(f"<{len(normals)}f", *normals),
              struct.pack(f"<{len(indices)}I", *indices)]
    binary = b"".join(chunks)
    metadata = {**identity, "version": MODEL_VERSION, "revision": model_revision(),
                "coordinates": "Y-up-hips-origin", "scope": "full",
                "solver_manifest_sha256": SOLVER_MANIFEST_SHA256,
                "base_fbx_sha256": hashlib.sha256(fbx.read_bytes()).hexdigest()}
    document = {
        "asset": {"version": "2.0", "generator": MODEL_VERSION, "extras": metadata},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.72, .74, .76, 1],
                                               "metallicFactor": 0, "roughnessFactor": .8}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": sum(map(len, chunks[:i])),
                         "byteLength": len(chunk)} for i, chunk in enumerate(chunks)],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions) // 3, "type": "VEC3",
             "min": [min(positions[i::3]) for i in range(3)],
             "max": [max(positions[i::3]) for i in range(3)]},
            {"bufferView": 1, "componentType": 5126, "count": len(normals) // 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    data = (struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("preview model exceeds size limit")
    destination.write_bytes(data)
    return {**metadata, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:]
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    manifest = bake_fbx(Path(job["fbx"]), Path(job["glb"]), job["identity"])
    Path(job["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
