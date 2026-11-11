"""Transfer source finger articulation after the frozen body conversion.

The body solver deliberately maps only 22 bones. This separate stage transports
finger rotations in rest palm frames, then bakes the result before cropping,
previewing or export. Explicit hand presets take precedence, independently per
side. Missing source finger data is reported, never replaced with an invented
gesture. This module is safe to import outside Blender.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

VERSION = "source-fingers-v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")


def bone_names(names, side):
    """Resolve Mixamo, 100STYLE and Unreal names in the shipped library."""
    names = set(names)
    label = side.title()
    suffix = side[0]
    wrist = next((n for n in (f"mixamorig:{label}Hand", f"{label}Hand",
                             f"{label}Wrist", f"hand_{suffix}") if n in names), None)
    fingers = {}
    for finger in FINGERS:
        for joint in range(1, 4):
            candidates = (f"mixamorig:{label}Hand{finger.title()}{joint}",
                          f"{label}Hand{finger.title()}{joint}",
                          f"{finger}_{joint:02d}_{suffix}")
            found = next((n for n in candidates if n in names), None)
            if found:
                fingers[finger, joint] = found
    return wrist, fingers


def _rotation(matrix):
    return matrix.to_quaternion().to_matrix()


def _palm_frame(arm, wrist, fingers):
    from mathutils import Matrix
    def head(name):
        return arm.matrix_world @ arm.data.bones[name].head_local
    origin = head(wrist)
    length = head(fingers["middle", 1]) - origin
    across = head(fingers["index", 1]) - head(fingers["pinky", 1])
    if length.length < 1e-8:
        raise ValueError("middle finger base coincides with wrist")
    length.normalize()
    across -= length * across.dot(length)
    if across.length < 1e-8:
        raise ValueError("source palm has no measurable width")
    across.normalize()
    normal = across.cross(length).normalized()
    return Matrix((across, length, normal)).transposed()


def bake_source_fingers(path, bvh_path, *, frame=0, mirror=False, hands=None,
                        force_exact_v324=False):
    """Update a full, rigged-rest FBX using only source finger rotations.

    Rotation deltas are cumulative relative to the source wrist, expressed in
    an anatomical rest palm frame. This removes wrist motion already owned by
    the frozen solver and handles different BVH/FBX bone axes. Child heads keep
    the target rig's lengths/offsets. No source translations or scale are copied.
    """
    receipt = dict(version=VERSION, applied_bones=0, body_bones_preserved=True,
                   sides={}, source_frame=frame, mirrored=mirror)
    if force_exact_v324:
        receipt["status"] = "exact_v324_bypass"
        return receipt
    wanted = [s for s in SIDES if not (hands or {}).get(s)]
    for side in SIDES:
        if side not in wanted:
            receipt["sides"][side] = dict(status="preset_override")
    if not wanted:
        receipt["status"] = "preset_override"
        return receipt

    # A body-only BVH has nothing to transfer and needs no extra Blender import.
    header = Path(bvh_path).read_text(encoding="utf-8-sig", errors="replace").partition("MOTION")[0]
    joint_names = re.findall(r"\bJOINT\s+(\S+)", header)
    source_sides = {}
    keys = [(finger, joint) for finger in FINGERS for joint in range(1, 4)]
    for side in wanted:
        source_side = ("right" if side == "left" else "left") if mirror else side
        wrist, fingers = bone_names(joint_names, source_side)
        if wrist and len(fingers) == len(keys):
            source_sides[side] = source_side
        else:
            receipt["sides"][side] = dict(status="missing_finger_chain", source_bones=len(fingers))
    if not source_sides:
        receipt["status"] = "source_unavailable"
        return receipt

    import bpy
    from mathutils import Matrix
    from converter import retarget as rt

    rt.reset_scene()
    dst, meshes = rt.import_character(str(path))
    # Freeze the baked body's transforms before importing an animated source.
    body_before = {b.name: b.matrix.copy() for b in dst.pose.bones}
    src = rt.import_bvh(str(bvh_path), frame=frame)
    reflect = Matrix.Diagonal((1.0, 1.0, -1.0))
    assigned = set()
    for side, source_side in source_sides.items():
        sw, sf = bone_names(src.data.bones.keys(), source_side)
        dw, df = bone_names(dst.data.bones.keys(), side)
        if not sw or not dw or len(sf) != len(keys) or len(df) != len(keys):
            receipt["sides"][side] = dict(status="missing_finger_chain",
                source_bones=len(sf), target_bones=len(df))
            continue
        if all(src.pose.bones[name].matrix_basis.to_quaternion().angle <= 1e-4
               for name in sf.values()):
            receipt["sides"][side] = dict(status="no_finger_rotation")
            continue
        try:
            source_rest = _palm_frame(src, sw, sf)
            target_rest = _palm_frame(dst, dw, df)
        except ValueError as exc:
            receipt["sides"][side] = dict(status="invalid_palm_frame", reason=str(exc))
            continue
        wrist_delta = (_rotation(rt._pose_world(src, sw))
                       @ _rotation(rt._rest_world(src, sw)).transposed())
        source_posed = wrist_delta @ source_rest
        maximum = 0.0
        # Parent before child: each new head follows its target parent.
        for key in keys:
            sname, dname = sf[key], df[key]
            delta = (source_posed.transposed()
                     @ _rotation(rt._pose_world(src, sname))
                     @ _rotation(rt._rest_world(src, sname)).transposed()
                     @ source_rest)
            if mirror:
                delta = reflect @ delta @ reflect
            desired = (target_rest @ delta @ target_rest.transposed()
                       @ _rotation(rt._rest_world(dst, dname)))
            if not all(math.isfinite(v) for row in desired for v in row):
                raise ValueError("non-finite source finger rotation: " + sname)
            pb = dst.pose.bones[dname]
            world = desired.to_4x4()
            world.translation = dst.matrix_world @ pb.head
            pb.matrix = dst.matrix_world.inverted() @ world
            bpy.context.view_layer.update()
            assigned.add(dname)
            maximum = max(maximum, abs(delta.to_quaternion().angle))
        receipt["sides"][side] = dict(status="transferred", applied_bones=15,
            source_side=source_side, max_cumulative_rotation_deg=math.degrees(maximum))

    bpy.data.objects.remove(src, do_unlink=True)
    for name, before in body_before.items():
        if name in assigned:
            continue
        # Leaf marker children follow fingers; they are not body bones.
        if any(p.name in assigned for p in dst.pose.bones[name].parent_recursive):
            continue
        after = dst.pose.bones[name].matrix
        if max(abs(before[r][c] - after[r][c]) for r in range(4) for c in range(4)) > 1e-5:
            raise RuntimeError("source fingers moved body bone: " + name)
    receipt["applied_bones"] = len(assigned)
    receipt["status"] = "transferred" if assigned else "source_unavailable"
    if assigned:
        rt.apply_output_mode(dst, meshes, "rigged_rest")
        # Never replace the artifact with an incomplete export.
        temporary = Path(path).with_name(Path(path).stem + ".source-fingers.fbx")
        rt.export_fbx(str(temporary), embed_textures=False)
        temporary.replace(path)
    return receipt
