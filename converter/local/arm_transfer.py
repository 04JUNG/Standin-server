"""Transfer axial arm motion when arm directions match but wrist frames differ."""
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

from converter import bone_map
from converter import retarget as rt
from converter.local.surface import lengths, triangle_geometry, within_budget


def palm_frame_pair(src, dst, source_name, target_name):
    so, sm, _ = rt._rest_palm_landmarks(src, source_name)
    to, tm, _ = rt._rest_palm_landmarks(dst, target_name)
    roles = rt._common_palm_roles(sm, tm)
    if not roles:
        return None
    sf = rt._palm_frame_from_rest_landmarks(so, sm, roles)[0]
    tf = rt._palm_frame_from_rest_landmarks(to, tm, roles)[0]
    if sf is None or tf is None:
        return None
    return sf["frame"], tf["frame"]


def rotation(arm, name, rest=False):
    frame = rt._rest_world if rest else rt._pose_world
    return rt._rot_only(frame(arm, name))


def evaluated_points(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        flat = np.empty(3*len(mesh.vertices), dtype=np.float32)
        mesh.vertices.foreach_get("co", flat)
        world = np.asarray(evaluated.matrix_world)
        return flat.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    finally:
        evaluated.to_mesh_clear()


def arm_surface(meshes, neutral, names):
    samples = []
    for obj in meshes:
        mesh = obj.data
        groups = {g.index for g in obj.vertex_groups if g.name in names}
        affected = np.array([any(g.group in groups and g.weight > 0 for g in v.groups)
                             for v in mesh.vertices])
        mesh.calc_loop_triangles()
        faces = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=int).reshape(-1, 3)
        faces = faces[affected[faces].any(axis=1)]
        if len(faces):
            samples.append((obj, faces, np.asarray(neutral[obj.name]), evaluated_points(obj)))
    return samples


def skin_safe(samples):
    for obj, faces, rest, old in samples:
        current = evaluated_points(obj)
        edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]))
        a, b = edges.T
        length = lengths(rest[a]-rest[b])
        valid = length > max(length.max()*1e-7, 1e-12)
        if not valid.any():
            continue
        before = lengths(old[a]-old[b])[valid] / length[valid]
        after = lengths(current[a]-current[b])[valid] / length[valid]
        # Ordinary bending may change, but do not trade twist fidelity for a
        # newly collapsed triangle or a worse extreme stretch on this arm.
        old_area, _ = triangle_geometry(old, faces)
        new_area, _ = triangle_geometry(current, faces)
        reference, _ = triangle_geometry(rest, faces)
        if (not np.isfinite(after).all() or after.max() > max(2., before.max()) + 1e-5
                or np.any((new_area < .01*reference) & (old_area >= .01*reference))):
            return False
    return True


def torso_frame(arm, table, rest):
    position = rt._rest_world if rest else rt._pose_world
    def point(role):
        return position(arm, table[role]).translation
    up = point("neck") - point("spine")
    lateral = point("shoulder.L") - point("shoulder.R")
    frame, _ = rt._frame_from_direction_normal(up, lateral.cross(up))
    return frame


def align_direction(rotation_candidate, neutral_edge, posed_edge):
    predicted = rotation_candidate.to_3x3() @ neutral_edge
    valid, _ = rt._min_rotation(predicted, posed_edge)
    if valid is None:
        return None
    return predicted.rotation_difference(posed_edge).to_matrix().to_4x4() @ rotation_candidate


def transfer_side(src, dst, meshes, neutral, report, side):
    chain = report.chain_diagnostics.get("arm."+side, {})
    if (not chain.get("activated") or chain.get("fallback")
            or chain.get("rest_direction_max_deg", 180.) >= 10.
            or chain.get("wrist_rest_compatibility_deg", 0.) <= 10.):
        return {"applied": False, "reason": "not_wrist_only_mismatch"}
    roles = [name+"."+side for name in ("upperarm","forearm","hand")]
    source = [bone_map.PROFILES[report.src_profile][key] for key in roles]
    target = [bone_map.PROFILES[report.dst_profile][key] for key in roles]
    palms = palm_frame_pair(src, dst, source[-1], target[-1])
    if palms is None:
        return {"applied": False, "reason": "palm_basis_unavailable"}
    source.insert(0, bone_map.PROFILES[report.src_profile]["shoulder."+side])
    target.insert(0, bone_map.PROFILES[report.dst_profile]["shoulder."+side])
    source_frames = [torso_frame(src,bone_map.PROFILES[report.src_profile],rest) for rest in (True,False)]
    target_frames = [torso_frame(dst,bone_map.PROFILES[report.dst_profile],rest) for rest in (True,False)]
    if any(frame is None for frame in source_frames+target_frames):
        return {"applied": False, "reason": "ambiguous_torso"}
    torso_map = target_frames[1] @ source_frames[1].inverted()
    bones = [dst.pose.bones[name] for name in target]
    backup = [b.matrix_basis.copy() for b in bones]
    world = [rt._pose_world(dst, name).copy() for name in target]
    target_rests = [rt._rest_world(dst, name) for name in target]
    source_rest = [rt._rest_world(src,name) for name in source]
    source_pose = [rt._pose_world(src,name) for name in source]
    source_delta = [rt._rot_only(pose) @ rt._rot_only(rest).inverted()
                    for pose,rest in zip(source_pose,source_rest)]
    if all(rt._rotation_error_deg(source_delta[0], delta) < 1. for delta in source_delta[1:]):
        return {"applied": False, "reason": "rigid_arm"}
    desired = []
    directions = []
    for k in range(3):
        edge = target_rests[k+1].translation - target_rests[k].translation
        source_edge = source_pose[k+1].translation - source_pose[k].translation
        goal = torso_map.to_3x3() @ source_edge
        motion = torso_map @ source_delta[k]
        if k==0:
            # Transfer the CHANGE of clavicle direction in the torso frame.
            # Absolute source shoulder height would erase the target's build.
            rest_local = source_frames[0].to_3x3().transposed() @ (
                source_rest[1].translation-source_rest[0].translation)
            pose_local = source_frames[1].to_3x3().transposed() @ source_edge
            valid, _ = rt._min_rotation(rest_local, pose_local)
            if valid is None:
                return {"applied": False, "reason": "ambiguous_clavicle"}
            target_local = target_frames[0].to_3x3().transposed() @ edge
            goal = target_frames[1].to_3x3() @ (rest_local.rotation_difference(pose_local) @ target_local)
            motion = rt._rot_only(world[0]) @ rt._rot_only(target_rests[0]).inverted()
        aligned = align_direction(motion, edge, goal)
        if aligned is None:
            return {"applied": False, "reason": "ambiguous_frame"}
        desired.append(aligned @ rt._rot_only(target_rests[k]))
        directions.append(goal)
    sf, tf = palms
    desired.append(torso_map @ source_delta[-1] @ sf @ tf.inverted() @ rt._rot_only(target_rests[-1]))
    samples = arm_surface(meshes, neutral, set(target))
    def errors(points):
        return [rt._angle_deg(points[k+1]-points[k], directions[k]) for k in range(3)]
    old_points = [m.translation for m in world]
    before_errors = errors(old_points)
    report_row = {"palmFrameErrorBeforeDeg": rt._rotation_error_deg(rt._rot_only(world[-1]),desired[-1]),
                  "directionErrorsBeforeDeg": before_errors}
    reach = sum((b-a).length for a,b in zip(old_points,old_points[1:]))
    wrist_before = backup[-1].to_quaternion().angle
    wrist_source = src.pose.bones[source[-1]].matrix_basis.to_quaternion().angle
    wrist_limit = max(wrist_source, wrist_before) + math.radians(35.)
    for fraction in (1., .75, .5, .25):
        for bone, old, new in zip(bones, world, desired):
            amount = 1. if bone == bones[-1] else fraction
            matrix = old.to_quaternion().slerp(new.to_quaternion(),amount).to_matrix().to_4x4()
            matrix.translation = (dst.matrix_world @ bone.head)
            bone.matrix = dst.matrix_world.inverted() @ matrix
            bpy.context.view_layer.update()
        location, quaternion, scale = bones[-1].matrix_basis.decompose()
        if quaternion.w < 0:
            quaternion.negate()
        if quaternion.angle > wrist_limit:
            quaternion = Quaternion().slerp(quaternion, wrist_limit / quaternion.angle)
            bones[-1].matrix_basis = Matrix.LocRotScale(location, quaternion, scale)
            bpy.context.view_layer.update()
        points = [dst.matrix_world@bone.head for bone in bones]
        drift = max((a-b).length for a,b in zip(points,old_points))
        direction_errors = errors(points)
        wrist_angle = bones[-1].matrix_basis.to_quaternion().angle
        direction_safe = all(new <= old+.05 for new,old in zip(direction_errors,before_errors))
        palm_error = rt._rotation_error_deg(rotation(dst,target[-1]),desired[-1])
        if (samples and palm_error <= report_row["palmFrameErrorBeforeDeg"]+.05 and drift <= .12*reach and direction_safe
                and wrist_angle <= wrist_limit+1e-5 and skin_safe(samples)):
            return dict(report_row,applied=True,strength=fraction,maxHeadDisplacement=drift,
                        directionErrorsAfterDeg=direction_errors,
                        palmFrameErrorAfterDeg=palm_error,
                        wristLocalAngleDeg=math.degrees(wrist_angle))
        for bone, basis in zip(bones, backup):
            bone.matrix_basis = basis
        bpy.context.view_layer.update()
    return dict(report_row,applied=False,reason="shape_or_joint_guard")


def apply(src, dst, meshes, neutral, report):
    if not within_budget(meshes):
        return {"applied": False, "reason": "mesh_budget"}
    return {side:transfer_side(src,dst,meshes,neutral,report,side) for side in ("L","R")}
