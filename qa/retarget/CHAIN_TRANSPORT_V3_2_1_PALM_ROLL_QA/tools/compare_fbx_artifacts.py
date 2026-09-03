"""두 rigged-rest FBX의 bone rest matrix와 baked mesh를 독립 비교한다."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def snapshot(path: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=path,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    arms = sorted((o for o in bpy.data.objects if o.type == "ARMATURE"), key=lambda o: o.name)
    if len(arms) != 1:
        raise RuntimeError(f"expected one armature in {path}, got {len(arms)}")
    arm = arms[0]
    bones = {
        b.name: [[float(x) for x in row] for row in b.matrix_local]
        for b in arm.data.bones
    }
    meshes = {}
    for obj in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
        meshes[obj.name] = [
            [float(x) for x in (obj.matrix_world @ vertex.co)]
            for vertex in obj.data.vertices
        ]
    return {"bones": bones, "meshes": meshes}


def max_matrix_delta(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[r][c] - b[r][c]) for r in range(4) for c in range(4))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    a, b = snapshot(args.a), snapshot(args.b)

    common_bones = sorted(set(a["bones"]) & set(b["bones"]))
    bone_deltas = {
        name: max_matrix_delta(a["bones"][name], b["bones"][name])
        for name in common_bones
    }
    mesh_compatible = set(a["meshes"]) == set(b["meshes"]) and all(
        len(a["meshes"][name]) == len(b["meshes"][name]) for name in a["meshes"]
    )
    mesh_max = None
    mesh_worst = None
    if mesh_compatible:
        mesh_max = 0.0
        for name in sorted(a["meshes"]):
            for index, (va, vb) in enumerate(zip(a["meshes"][name], b["meshes"][name])):
                distance = math.sqrt(sum((va[i] - vb[i]) ** 2 for i in range(3)))
                if distance > mesh_max:
                    mesh_max = distance
                    mesh_worst = {"object": name, "vertex": index}
    payload = {
        "a": str(Path(args.a).resolve()),
        "b": str(Path(args.b).resolve()),
        "bone_sets_equal": set(a["bones"]) == set(b["bones"]),
        "common_bones": len(common_bones),
        "bone_matrix_max_abs": max(bone_deltas.values(), default=0.0),
        "bone_matrix_worst": max(bone_deltas, key=bone_deltas.get) if bone_deltas else None,
        "bone_matrix_delta_by_name": bone_deltas,
        "mesh_compatible": mesh_compatible,
        "mesh_vertex_max_distance": mesh_max,
        "mesh_vertex_worst": mesh_worst,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PALM_ARTIFACT_COMPARE=" + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
