"""Retarget evaluated body landmarks from library poses encoded in BVH offsets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from converter.bone_map import PROFILES, mirror_name, resolve_profile

SPINE = ("spine", "spine1", "spine2", "neck", "head")
SHOULDERS = ("shoulder.L", "shoulder.R")
ARMS = tuple(tuple(f"{joint}.{side}" for joint in ("upperarm", "forearm", "hand")) for side in "LR")
UPPER_ARMS = tuple(arm[0] for arm in ARMS)
LEGS = tuple(tuple(f"{joint}.{side}" for joint in ("upleg", "leg", "foot", "toe")) for side in "LR")


def reviewed_frames(source_sha256, character_sha256, *, mirror=False):
    """Look up reviewed Blender world-space quaternions for the exact pose and rig."""
    if mirror or not character_sha256:
        return None
    catalog = json.loads(Path(__file__).with_name("reviewed_pose_frames.json").read_text())
    row = catalog.get(source_sha256)
    return row if row and row["character_sha256"] == character_sha256 else None


def smooth_reviewed_waist(mesh, rig, neutral, posed):
    """Blend the corrected waist surface while keeping the rig and weights intact."""
    import numpy as np

    core = {"Hips", "Spine", "Spine1", "LeftUpLeg", "RightUpLeg"}
    groups = {g.index for g in mesh.vertex_groups if g.name.split(":")[-1] in core}
    weights = np.array([sum(g.weight for g in v.groups if g.group in groups) for v in mesh.data.vertices])
    spine = rig.data.bones["mixamorig:Spine"]
    center = (mesh.matrix_world.inverted() @ rig.matrix_world @ spine.head_local).z
    alpha = np.clip((weights - .68) / .32, 0, 1) * np.clip(1 - np.abs(neutral[:, 2] - center) / .16, 0, 1) ** .9
    edges = np.empty((len(mesh.data.edges), 2), dtype=np.int32)
    mesh.data.edges.foreach_get("vertices", edges.ravel())
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    affected = np.flatnonzero(alpha)
    # Only solve for vertices in the waist band. Neighboring vertices are fixed.
    slots = np.full(len(alpha), -1, dtype=np.int32)
    slots[affected] = np.arange(len(affected))
    inside = slots[dst] >= 0
    src, dst = src[inside], slots[dst[inside]]
    degree = np.maximum(1, np.bincount(dst, minlength=len(affected)))
    coords = np.array(posed, dtype=float)
    for _ in range(160):
        mean = np.column_stack([np.bincount(dst, weights=coords[src, i], minlength=len(affected)) / degree
                                for i in range(3)])
        coords[affected] += (mean - coords[affected]) * (.68 * alpha[affected, None])
    delta = coords[affected] - posed[affected]
    # The original thumbnail also blends this seam. Limit movement to 25 mm.
    delta *= np.minimum(1, .025 / np.maximum(np.linalg.norm(delta, axis=1), 1e-9))[:, None]
    coords[affected] = posed[affected] + delta
    return coords


def apply_source_body(src, dst, *, mirror=False, authored=None):
    """Match source joint directions using the target's own bone lengths and weights."""
    import bpy
    from mathutils import Matrix, Quaternion
    from converter import retarget as rt
    from converter.reviewed_source_fingers import bone_names, _palm_frame

    source_names = PROFILES["mixamo"]
    target_names = PROFILES[resolve_profile(list(dst.pose.bones.keys()))]
    limb_roles = tuple(role for limb in (*ARMS, *LEGS) for role in limb)
    roles = ("hips", *SPINE, *SHOULDERS, *limb_roles)
    if any(source_names[role] not in src.pose.bones or target_names.get(role) not in dst.pose.bones
           for role in roles):
        return {"status": "missing_body_chain"}

    source = {role: rt._source_pose_point(src, source_names, role, mirror) for role in roles}
    target = {role: dst.pose.bones[target_names[role]] for role in roles}
    world = dst.matrix_world
    inverse = world.inverted()
    skipped = {}

    def head(role):
        return world @ target[role].head

    def rotation(bone):
        return (world @ bone.matrix).to_quaternion()

    def set_rotation(bone, orientation):
        transform = orientation.to_matrix().to_4x4()
        transform.translation = world @ bone.head
        bone.matrix = inverse @ transform
        bpy.context.view_layer.update()

    def align(role, current, desired):
        correction, reason = rt._min_rotation(current, desired)
        if correction is None:
            skipped[role] = reason
        else:
            bone = target[role]
            set_rotation(bone, correction.to_quaternion() @ rotation(bone))

    def align_edge(parent, child):
        align(parent, head(child) - head(parent), source[child] - source[parent])

    def align_end(role):
        source_role = mirror_name(role) if mirror else role
        bone = src.pose.bones[source_names[source_role]]
        desired = src.matrix_world.to_3x3() @ (bone.tail - bone.head)
        if mirror:
            desired = rt._reflect_x(desired)
        align(role, world @ target[role].tail - head(role), desired)

    def pelvis_frame(point):
        left, right = point("upleg.L"), point("upleg.R")
        lateral = right - left
        up = point("spine") - (left + right) / 2
        return rt._frame_from_direction_normal(up, lateral.cross(up))

    def match_frame(bone, current, desired):
        set_rotation(bone, (desired @ current.transposed()).to_quaternion() @ rotation(bone))

    retained = {role: rotation(target[role]) for role in ("head", *limb_roles)}
    authored_rotations = {
        role: Quaternion(q).normalized() for role, q in (authored or {}).get("rotations", {}).items()
    }
    edges = list(zip(SPINE, SPINE[1:]))

    def spine_error():
        return max(rt._angle_deg(head(child) - head(parent), source[child] - source[parent])
                   for parent, child in edges)

    before = spine_error()
    src_pelvis, src_reason = pelvis_frame(source.__getitem__)
    dst_pelvis, dst_reason = pelvis_frame(head)
    if src_pelvis is not None and dst_pelvis is not None:
        match_frame(target["hips"], dst_pelvis, src_pelvis)
    else:
        skipped["hips"] = src_reason or dst_reason

    for parent, child in edges[:3]:
        if parent in authored_rotations:
            set_rotation(target[parent], authored_rotations[parent])
        align_edge(parent, child)

    # The spine axis leaves roll free. Use the shoulder line to recover it.
    axis = (head("neck") - head("spine2")).normalized()
    current = head(UPPER_ARMS[1]) - head(UPPER_ARMS[0])
    desired = source[UPPER_ARMS[1]] - source[UPPER_ARMS[0]]
    align("spine2", current - axis * current.dot(axis), desired - axis * desired.dot(axis))
    align_edge(*edges[-1])
    set_rotation(target["head"], retained["head"])
    align_end("head")

    for shoulder, arm in zip(SHOULDERS, UPPER_ARMS):
        align_edge(shoulder, arm)

    # Restore each segment's solved twist before steering its child into place.
    for limb in (*ARMS, *LEGS):
        for parent, child in zip(limb, limb[1:]):
            set_rotation(target[parent], authored_rotations.get(parent, retained[parent]))
            align_edge(parent, child)
        end = limb[-1]
        set_rotation(target[end], authored_rotations.get(end, retained[end]))
        if end.startswith("toe."):
            align_end(end)

    for side, other_side in (("left", "right"), ("right", "left")):
        sw, sf = bone_names(src.pose.bones.keys(), other_side if mirror else side)
        dw, df = bone_names(dst.pose.bones.keys(), side)
        if not sw or not dw or not {("middle", 1), ("index", 1), ("pinky", 1)} <= sf.keys() & df.keys():
            skipped[f"wrist.{side}"] = "missing palm landmarks"
            continue
        try:
            desired = _palm_frame(src, sw, sf, posed=True)
            current = _palm_frame(dst, dw, df, posed=True)
        except ValueError as exc:
            skipped[f"wrist.{side}"] = str(exc)
            continue
        if mirror:
            # Reflect the landmark axes and rebuild their right-handed normal.
            desired = Matrix.Diagonal((-1.0, 1.0, 1.0)) @ desired @ Matrix.Diagonal((1.0, 1.0, -1.0))
        match_frame(dst.pose.bones[dw], current, desired)

    result = {"status": "transferred", "max_direction_error_before_deg": before,
              "max_direction_error_after_deg": spine_error(), "skipped": skipped}
    if authored:
        result["reviewed_pose_frames"] = {key: authored[key] for key in
                                          ("pose_id", "reference_fbx_sha256", "smooth_waist")}
    return result


def bake_source_body(path, bvh_path, *, frame=0, mirror=False, character_sha256=None):
    """Apply to the single-frame baked-offset format recognized by the importer."""
    source_bytes = Path(bvh_path).read_bytes()
    source = source_bytes.decode("utf-8-sig", errors="replace")
    if "ROOT mixamorig:Hips" not in source:
        return None
    from converter import retarget as rt
    try:
        normalized = rt._offset_delta_pose(source)
    except ValueError:
        return None
    if normalized is None:
        return None

    import bpy
    from converter.shape_bake import import_full_character
    rt.reset_scene()
    dst, meshes = import_full_character(str(path))
    src = rt.import_bvh(str(bvh_path), frame=frame)
    authored = reviewed_frames(hashlib.sha256(source_bytes).hexdigest(), character_sha256, mirror=mirror)
    report = apply_source_body(src, dst, mirror=mirror, authored=authored)
    if report["status"] == "transferred":
        bpy.data.objects.remove(src, do_unlink=True)
        rt.apply_output_mode(dst, meshes, "rigged_rest")
        output = Path(path)
        temporary = output.with_suffix(".source-body.fbx")
        rt.export_fbx(str(temporary), embed_textures=False)
        temporary.replace(output)
    return report
