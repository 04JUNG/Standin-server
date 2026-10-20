#!/usr/bin/env python3
"""Offline reference-only skin template. Run in the pinned Blender child runtime.

Never used by final export. Exact approved character; canonical reference skinning
is intentionally distinct from the frozen final retarget solver.
"""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import bpy
    from converter import retarget as rt
    from converter.bone_map import PROFILES, resolve_profile, CANONICAL_BONES
    from converter.framing import OUTPUT_SCOPES, crop_meshes
    from converter.protocol import EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH
    from src.pose_posters import CHARACTER, character_sha256

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if (
        not bpy.app.version_string.startswith(EXPECTED_BLENDER_VERSION)
        or bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH
    ):
        raise ValueError("unapproved Blender runtime")
    sha = hashlib.sha256(args.character.read_bytes()).hexdigest()
    if sha != character_sha256():
        raise ValueError("unapproved default body")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "default-body-template-library-v1",
        "character_id": CHARACTER,
        "character_sha256": sha,
        "scopes": {},
    }

    def vec(v):
        return [round(float(v.x), 6), round(float(v.z), 6), round(float(-v.y), 6)]

    for scope in sorted(OUTPUT_SCOPES):
        rt.reset_scene()
        arm, meshes = rt.import_character(str(args.character))
        mapping = PROFILES[resolve_profile([b.name for b in arm.data.bones])]
        roles = [r for r in CANONICAL_BONES if mapping.get(r) in arm.data.bones]
        inverse = {mapping[r]: i for i, r in enumerate(roles)}
        joints = []
        for role in roles:
            bone = arm.data.bones[mapping[role]]
            parent = bone.parent
            while parent and parent.name not in inverse:
                parent = parent.parent
            joints.append(
                {
                    "role": role,
                    "parent": inverse[parent.name] if parent else -1,
                    "head": vec(arm.matrix_world @ bone.head_local),
                    "tail": vec(arm.matrix_world @ bone.tail_local),
                }
            )
        crop_meshes(arm, meshes, scope)
        geometry = []
        max_discard = 0.0
        for mesh in meshes:
            if mesh.name not in bpy.data.objects:
                continue
            data = mesh.data
            data.calc_loop_triangles()
            positions = []
            normals = []
            weights = []
            indices = []
            normal_matrix = mesh.matrix_world.to_3x3().inverted().transposed()
            groupmap = {}
            for g in mesh.vertex_groups:
                bone = arm.data.bones.get(g.name)
                while bone and bone.name not in inverse:
                    bone = bone.parent
                if bone:
                    groupmap[g.index] = inverse[bone.name]
            for vertex in data.vertices:
                positions.extend(vec(mesh.matrix_world @ vertex.co))
                normals.extend(vec((normal_matrix @ vertex.normal).normalized()))
                w = {}
                for g in vertex.groups:
                    if g.group in groupmap:
                        w[groupmap[g.group]] = w.get(groupmap[g.group], 0) + g.weight
                total = sum(w.values())
                if total <= 1e-8:
                    raise ValueError("unweighted mesh vertex")
                top = sorted(w.items(), key=lambda item: -item[1])[:8]
                kept = sum(v for _, v in top)
                max_discard = max(max_discard, 1 - kept / total)
                if 1 - kept / total > 0.001:
                    raise ValueError("reference skin loses excessive weights")
                top += [(0, 0)] * (8 - len(top))
                indices.extend(i for i, _ in top)
                weights.extend(round(v / kept, 7) for _, v in top)
            geometry.append(
                {
                    "positions": positions,
                    "normals": normals,
                    "skinIndices": indices,
                    "skinWeights": weights,
                    "triangles": [i for t in data.loop_triangles for i in t.vertices],
                }
            )
        payload = {
            "version": "default-body-template-v1",
            "character_id": CHARACTER,
            "character_sha256": sha,
            "scope": scope,
            "coordinates": "Y-up",
            "preview_only": True,
            "pose_method": "reference-skinning-v1",
            "influences": 8,
            "joints": joints,
            "profiles": PROFILES,
            "meshes": geometry,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode()
        packed = gzip.compress(raw, mtime=0)
        filename = scope + ".json.gz"
        (args.out / filename).write_bytes(packed)
        manifest["scopes"][scope] = {
            "file": filename,
            "sha256": hashlib.sha256(packed).hexdigest(),
            "json_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(packed),
            "json_bytes": len(raw),
            "vertices": sum(len(m["positions"]) // 3 for m in geometry),
            "max_discarded_weight": max_discard,
        }
        print(json.dumps({"scope": scope, **manifest["scopes"][scope]}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
