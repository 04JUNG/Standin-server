"""Audit every preset on a real character with Blender, including posed-rest input.

blender -b --python-exit-code 1 --python scripts/audit_hand_library.py -- \
  --character character.fbx --out /tmp/hand-audit [--base posed.fbx] [--render]

Capsule contacts and nonadjacent triangle crossings are separate observations;
neither is a complete collision certificate. Source files are never modified.
Preview crops use the neutral character's spatial hand region, not skin-weight
thresholds (which can cut holes in fingertips). All presets start from rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def crossings(vertices, triangles, hand_length, labels=None):
    """Count nonadjacent, noncoplanar crossings on this hand's actual skin."""
    import numpy as np
    from mathutils.bvhtree import BVHTree

    if not triangles:
        raise RuntimeError("No hand triangles")
    p = np.asarray(vertices, float) / hand_length
    faces = np.asarray(triangles, int)
    tree = BVHTree.FromPolygons(p.tolist(), triangles, all_triangles=True)
    pairs = [
        (a, b)
        for a, b in tree.overlap(tree)
        if a < b and not set(triangles[a]).intersection(triangles[b])
    ]
    if not pairs:
        return {"pairs": 0, "examples": []}
    pairs = np.asarray(pairs, int)
    aa, bb = p[faces[pairs[:, 0]]], p[faces[pairs[:, 1]]]
    hit = np.zeros(len(pairs), bool)
    for edges, face in ((aa, bb), (bb, aa)):
        e1, e2 = face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            origin, direction = edges[:, i], edges[:, j] - edges[:, i]
            h = np.cross(direction, e2)
            det = np.einsum("ij,ij->i", e1, h)
            valid = abs(det) > 1e-12
            inv = np.zeros_like(det)
            inv[valid] = 1 / det[valid]
            v = origin - face[:, 0]
            u = np.einsum("ij,ij->i", v, h) * inv
            q = np.cross(v, e1)
            w = np.einsum("ij,ij->i", direction, q) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            hit |= (
                valid
                & (u > 1e-7)
                & (w > 1e-7)
                & (u + w < 1 - 1e-7)
                & (t > 1e-7)
                & (t < 1 - 1e-7)
            )
    found = pairs[hit]
    regions = {}
    if labels:
        for a, b in found:
            key = "/".join(sorted((labels[a], labels[b])))
            regions[key] = regions.get(key, 0) + 1
    return {"pairs": len(found), "examples": found[:8].tolist(), "regions": regions}


def main():
    import bpy
    import numpy as np
    from mathutils import Matrix, Quaternion, Vector
    from converter import retarget as rt
    from converter.hand_collision import contacts
    from converter.hand_geometry import (
        finger_rotations,
        hand_frame,
        matrix_to_quaternion,
    )
    from converter.hand_library import FINGERS, SIDES, bone_name, load_library
    from converter.hands import _prefix, _side_geometry, _hand_length, _posed_chains
    from converter.preview_style import configure_workbench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", required=True, type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--presets", nargs="+")
    parser.add_argument(
        "--exclude-articulated",
        action="store_true",
        help="Audit generic curl/spread presets; specialized profiles are verified on baked FBX",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repair-roots", action="store_true")
    parser.add_argument("--preserve-volume", action="store_true")
    parser.add_argument(
        "--as-is",
        action="store_true",
        help="Inspect a delivered --base without applying another preset",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.out.mkdir(parents=True, exist_ok=True)
    library = load_library()
    ids = args.presets or list(library.preset_ids)
    if args.exclude_articulated:
        ids = [pid for pid in ids if library.preset(pid).articulation is None]
    for pid in ids:
        library.preset(pid)
    if args.as_is and (args.base is None or len(ids) != 1):
        parser.error("--as-is requires --base and one --presets label")

    # Locate the entire hand on the neutral character before any deformation.
    # Reuse indices to crop the evaluated skin, including low-weight fingertips.
    rt.reset_scene()
    arm, meshes = rt.import_character(str(args.character))
    prefix = _prefix(arm)
    regions = {}
    neutral_lengths = {}
    for side in SIDES:
        wrist, bases, rests = _side_geometry(arm, side, prefix)
        frame = hand_frame(side, wrist, bases)
        length = _hand_length(arm, side, prefix)
        neutral_lengths[side] = length
        regions[side] = []
        for mesh in meshes:
            positions = np.array(
                [
                    arm.matrix_world.inverted() @ mesh.matrix_world @ v.co
                    for v in mesh.data.vertices
                ]
            )
            keep = (positions - wrist) @ frame.length_axis >= -length * 0.14
            mesh.data.calc_loop_triangles()
            faces = [
                tuple(t.vertices)
                for t in mesh.data.loop_triangles
                if all(keep[i] for i in t.vertices)
            ]
            regions[side].append((len(mesh.data.vertices), faces))

    if args.base:
        rt.reset_scene()
        arm, meshes = rt.import_character(str(args.base))
        prefix = _prefix(arm)
    if args.repair_roots:
        from converter.hand_rig import repair_digit_roots

        for side in SIDES:
            print(
                "REPAIRED",
                side,
                repair_digit_roots(arm, meshes, side, prefix),
                flush=True,
            )
    scene = bpy.context.scene
    if args.preserve_volume:
        for obj in meshes:
            for modifier in obj.modifiers:
                if modifier.type == "ARMATURE":
                    modifier.use_deform_preserve_volume = True
    configure_workbench(scene)
    shade = scene.display.shading
    shade.single_color = (0.72, 0.76, 0.82)
    shade.show_cavity = True
    shade.cavity_type = "BOTH"
    shade.show_specular_highlight = False
    shade.background_type = "VIEWPORT"
    shade.background_color = (0.045, 0.055, 0.075)
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    camera_data = bpy.data.cameras.new("Hand audit")
    camera = bpy.data.objects.new("Hand audit", camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.001
    camera_data.clip_end = 1000
    scene.camera = camera
    result = {
        "character": args.character.name,
        "character_sha256": hashlib.sha256(args.character.read_bytes()).hexdigest(),
        "base": str(args.base) if args.base else None,
        "library_sha256": library.sha256,
        "scope": "same-hand capsules and nonadjacent skin triangle crossings; no containment certificate",
        "sides": {},
    }
    for side in SIDES:
        wrist, bases, rests = _side_geometry(arm, side, prefix)
        frame = hand_frame(side, wrist, bases)
        length = neutral_lengths[side]
        geometry = {
            "wrist": wrist.tolist(),
            "bases": {k: v.tolist() for k, v in bases.items()},
            "length": length,
            "rests": {
                f"{f}:{j}": {
                    "head": r.head.tolist(),
                    "direction": r.direction.tolist(),
                    "basis": r.basis.tolist(),
                    "tail": list(
                        arm.data.bones[bone_name(side, f, j, prefix)].tail_local
                    ),
                }
                for (f, j), r in rests.items()
            },
        }
        result["sides"][side] = {"geometry": geometry, "presets": {}}
        normal = arm.matrix_world.to_3x3() @ Vector(frame.palm_normal)
        across = arm.matrix_world.to_3x3() @ Vector(frame.pinky_side)
        axis = arm.matrix_world.to_3x3() @ Vector(frame.length_axis)
        for pid in ids:
            for pb in arm.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)
            rotations = (
                {}
                if args.as_is
                else finger_rotations(side, library.preset(pid), frame, rests, bases)
            )
            for (finger, joint), local in rotations.items():
                pb = arm.pose.bones[bone_name(side, finger, joint, prefix)]
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = Quaternion(matrix_to_quaternion(local))
            bpy.context.view_layer.update()
            found = contacts(
                _posed_chains(arm, side, prefix),
                palm_normal=frame.palm_normal,
                pinky_side=frame.pinky_side,
                length_axis=frame.length_axis,
                wrist=wrist,
                hand_length=length,
            )
            entry = {
                "capsule_contacts": [
                    {"a": c.a, "b": c.b, "depth_m": c.depth} for c in found
                ]
            }
            vertices, triangles, labels = [], [], []
            dg = bpy.context.evaluated_depsgraph_get()
            for obj, (count, faces) in zip(meshes, regions[side]):
                evaluated = obj.evaluated_get(dg)
                weights = [
                    {obj.vertex_groups[g.group].name: g.weight for g in v.groups}
                    for v in obj.data.vertices
                ]
                for face in faces:
                    names = {n for i in face for n in weights[i]}
                    label = max(
                        names, key=lambda n: sum(weights[i].get(n, 0.0) for i in face)
                    )
                    labels.append(label.split("Hand")[-1] or "palm")
                mesh = evaluated.to_mesh()
                try:
                    if len(mesh.vertices) != count:
                        raise RuntimeError(
                            "Topology changed; cannot reuse spatial hand crop"
                        )
                    offset = len(vertices)
                    vertices.extend(
                        evaluated.matrix_world @ v.co for v in mesh.vertices
                    )
                    triangles.extend(tuple(i + offset for i in face) for face in faces)
                finally:
                    evaluated.to_mesh_clear()
            entry["skin_crossings"] = crossings(vertices, triangles, length, labels)
            if args.render:
                data = bpy.data.meshes.new("Audit skin")
                # from_pydata retains unused vertices. Fit only surface vertices.
                data.from_pydata(vertices, [], triangles)
                data.update()
                for face in data.polygons:
                    face.use_smooth = True
                preview = bpy.data.objects.new("Audit skin", data)
                scene.collection.objects.link(preview)
                for obj in meshes:
                    obj.hide_render = True
                points = [
                    Vector(vertices[i]) for i in {v for f in triangles for v in f}
                ]
                for view, direction in [
                    ("oblique", normal - across * 0.75 + axis * 0.28),
                    ("side", -across + normal * 0.05),
                    ("back", -normal - across * 0.35 + axis * 0.2),
                ]:
                    direction.normalize()
                    up = (axis - direction * axis.dot(direction)).normalized()
                    right = up.cross(direction).normalized()
                    camera.rotation_euler = (
                        Matrix((right, up, direction)).transposed().to_euler()
                    )
                    coords = np.array(
                        [[p.dot(right), p.dot(up), p.dot(direction)] for p in points]
                    )
                    low, high = coords.min(axis=0), coords.max(axis=0)
                    center = (low + high) / 2
                    camera.location = (
                        right * float(center[0])
                        + up * float(center[1])
                        + direction * float(high[2] + length * 5)
                    )
                    camera_data.ortho_scale = float(
                        max(high[0] - low[0], high[1] - low[1]) * 1.16
                    )
                    scene.render.filepath = str(args.out / f"{pid}-{side}-{view}.png")
                    bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(preview, do_unlink=True)
                bpy.data.meshes.remove(data)
                for obj in meshes:
                    obj.hide_render = False
            result["sides"][side]["presets"][pid] = entry
            print(
                f'AUDIT {side} {pid}: capsules={len(found)} triangles={entry["skin_crossings"]["pairs"]}',
                flush=True,
            )
    (args.out / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
