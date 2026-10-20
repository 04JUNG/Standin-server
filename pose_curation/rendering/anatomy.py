"""Blender-only checks on the actual converted character, including its skin.

The registered Master V2 T-pose faces -Y. Its elbow flexion reference is the
forward vector transported by the upper-arm's *actual* rotation. This exposes
reverse/sideways skin bending that three joint positions alone cannot detect.
Triangle intersections exclude shoulder attachment seams using skin weights.
"""

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from pose_curation.anatomy import hinge_alignment
from pose_curation.qa.policy import SPINE_ROTATION, ELBOW_HINGE, fingerprint


def rest_frames(arm):
    bones = {
        "Left": "LeftArm",
        "Right": "RightArm",
        **{name: name for name in ("Hips", "Spine", "Spine1", "Spine2")},
    }
    return {
        key: (arm.matrix_world @ arm.data.bones[f"mixamorig:{name}"].matrix_local)
        .to_3x3()
        .normalized()
        for key, name in bones.items()
    }


def inspect(arm, rest):
    hinge, flags = {}, []
    # Relative transport removes the character's own rest curvature. These
    # conservative per-segment guards catch helper rotations and opposing
    # twists even when the final chest orientation happens to look correct.
    import math

    transports, spine_angles = {}, {}
    for name in ("Hips", "Spine", "Spine1", "Spine2"):
        pose = (
            (arm.matrix_world @ arm.pose.bones["mixamorig:" + name].matrix)
            .to_3x3()
            .normalized()
        )
        transports[name] = pose @ rest[name].transposed()
    for parent, child in [("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2")]:
        angle = math.degrees(
            (transports[parent].transposed() @ transports[child]).to_quaternion().angle
        )
        angle = min(angle, 360 - angle)
        spine_angles[child] = round(angle, 3)
        if angle > SPINE_ROTATION:
            flags.append(
                f"{child} concentrated torso rotation: {angle:.1f} degrees > {SPINE_ROTATION} review guard"
            )
    for side in ("Left", "Right"):
        upper, lower, hand = [
            arm.pose.bones[f"mixamorig:{side}{name}"]
            for name in ("Arm", "ForeArm", "Hand")
        ]
        shoulder, elbow, wrist = [
            arm.matrix_world @ bone.head for bone in (upper, lower, hand)
        ]
        rotation = (arm.matrix_world @ upper.matrix).to_3x3().normalized() @ rest[
            side
        ].transposed()
        mismatch = hinge_alignment(
            elbow - shoulder, wrist - elbow, rotation @ Vector((0, -1, 0))
        )
        hinge[side] = mismatch
        if mismatch is not None and mismatch > ELBOW_HINGE:
            flags.append(f"{side} elbow hinge-plane mismatch: {mismatch:.1f} degrees")

    intersections = {side: 0 for side in ("Left", "Right")}
    attachments = {side: 0 for side in ("Left", "Right")}
    segments = {
        side: tuple(
            arm.matrix_world @ arm.pose.bones[f"mixamorig:{side}{name}"].head
            for name in ("Arm", "ForeArm")
        )
        for side in ("Left", "Right")
    }
    graph = bpy.context.evaluated_depsgraph_get()
    # Accumulate meshes in one coordinate system so split clothing/body meshes
    # can intersect too. Only weighted torso and distal arm regions participate.
    vertices, regions = [], {name: [] for name in ("torso", "Left", "Right")}
    torso_names = {"Hips", "Spine", "Spine1", "Spine2", "Neck"}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            start = len(vertices)
            vertices.extend(
                evaluated.matrix_world @ vertex.co for vertex in mesh.vertices
            )
            group_names = {
                group.index: group.name.removeprefix("mixamorig:")
                for group in obj.vertex_groups
            }
            labels = []
            for vertex in mesh.vertices:
                weights = {
                    group_names.get(group.group, ""): group.weight
                    for group in vertex.groups
                }
                label = None
                if sum(v for k, v in weights.items() if k in torso_names) > 0.8:
                    label = "torso"
                for side in ("Left", "Right"):
                    if (
                        sum(
                            v
                            for k, v in weights.items()
                            if k == side + "ForeArm" or k.startswith(side + "Hand")
                        )
                        > 0.65
                    ):
                        label = side
                    if weights.get(side + "Arm", 0) > 0.65:
                        shoulder, elbow = segments[side]
                        axis = elbow - shoulder
                        position = evaluated.matrix_world @ vertex.co
                        if (position - shoulder).dot(axis) / axis.length_squared > 0.30:
                            label = side
                labels.append(label)
            mesh.calc_loop_triangles()
            for triangle in mesh.loop_triangles:
                label = labels[triangle.vertices[0]]
                if label and all(labels[i] == label for i in triangle.vertices):
                    regions[label].append(tuple(start + i for i in triangle.vertices))
        finally:
            evaluated.to_mesh_clear()
    torso = BVHTree.FromPolygons(vertices, regions["torso"], all_triangles=True)
    for side in ("Left", "Right"):
        tree = BVHTree.FromPolygons(vertices, regions[side], all_triangles=True)
        pairs = torso.overlap(tree)
        shoulder, elbow = segments[side]
        axis = elbow - shoulder
        for _, index in pairs:
            center = sum((vertices[i] for i in regions[side][index]), Vector()) / 3
            # The proximal upper arm shares the axilla surface with the torso.
            # Keep those contacts visible separately; do not call them deep
            # penetration or silently count them as a clear non-contact mesh.
            fraction = (center - shoulder).dot(axis) / axis.length_squared
            if (center - shoulder).length < axis.length * 0.60 and fraction < 0.60:
                attachments[side] += 1
            else:
                intersections[side] += 1
        if intersections[side]:
            flags.append(
                f"{side} arm/hand skin intersects torso: {intersections[side]} triangle pairs"
            )
    return {
        "method": "Master V2 transported hinge and weighted skin triangle intersections",
        "torso_segment_rotation_degrees": spine_angles,
        "torso_review_guard_degrees": SPINE_ROTATION,
        "policy_fingerprint": fingerprint(),
        "hinge_mismatch_degrees": hinge,
        "skin_intersections": intersections,
        "shoulder_attachment_intersections": attachments,
        "checked_triangles": {k: len(v) for k, v in regions.items()},
        "flags": flags,
    }
