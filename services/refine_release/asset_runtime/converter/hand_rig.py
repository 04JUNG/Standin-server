"""Repair digit roots placed in the middle of a finger, before hand posing.

The neutral mesh stays fixed. Only outlying middle/ring rest chains and their
local hand skin weights change; the wrist/body rest frames are untouched.
Geometry is measured in the hand's frame, so posed-rest FBX works as well.
"""

from __future__ import annotations

import numpy as np

from converter.hand_library import FINGERS, bone_name, hand_bone_name


def joint_weights(distance, length):
    """Compact linear skin transitions at MCP/PIP/DIP, in metres."""
    width = min(length * 0.075, 0.006)
    pivots = (0.0, length * 0.44, length * 0.77)
    weights = np.zeros(4)  # hand, first, second, third
    for i, pivot in enumerate(pivots):
        if distance < pivot - width:
            weights[i] = 1.0
            return weights
        if distance <= pivot + width:
            t = (distance - pivot + width) / (2 * width)
            weights[i], weights[i + 1] = 1 - t, t
            return weights
    weights[3] = 1.0
    return weights


def repair_digit_roots(arm, meshes, side, prefix):
    import bpy
    from mathutils import Vector
    from converter.hands import _side_geometry, _hand_length
    from converter.hand_geometry import hand_frame

    wrist, bases, rests = _side_geometry(arm, side, prefix)
    frame = hand_frame(side, wrist, bases)
    length = _hand_length(arm, side, prefix)
    u = frame.pinky_side
    across = np.asarray(bases["pinky"]) - bases["index"]
    span = float(across @ u)
    if abs(span) < 1e-6:
        return []
    plans = {}
    outer = max(float(np.linalg.norm(bases[f] - wrist)) for f in ("index", "pinky"))
    for finger in ("middle", "ring"):
        base = bases[finger]
        if np.linalg.norm(base - wrist) <= outer * 1.18:
            continue
        t = float(np.clip((base - bases["index"]) @ u / span, 0.0, 1.0))
        head = (1 - t) * bases["index"] + t * bases["pinky"]
        head = head + frame.length_axis * length * 0.035 * (1 - abs(2 * t - 1))
        tip = np.array(arm.data.bones[bone_name(side, finger, 2, prefix)].tail_local)
        vector = tip - head
        extent = float(np.linalg.norm(vector))
        if extent < length * 0.25 or np.dot(vector, frame.length_axis) <= 0:
            raise ValueError("Invalid reconstructed finger chain")
        plans[finger] = (head, tip, extent)
    if not plans:
        return []

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for finger, (head, tip, extent) in plans.items():
        edges = [head + (tip - head) * t for t in (0.0, 0.44, 0.77, 1.0)]
        for j in range(3):
            bone = arm.data.edit_bones[bone_name(side, finger, j, prefix)]
            bone.head = Vector(edges[j])
            bone.tail = Vector(edges[j + 1])
    bpy.ops.object.mode_set(mode="OBJECT")

    for obj in meshes:
        names = {g.name: g for g in obj.vertex_groups}
        hand_group = names[hand_bone_name(side, prefix)]
        groups = {
            f: [names[bone_name(side, f, j, prefix)] for j in range(3)] for f in FINGERS
        }
        world_to_arm = arm.matrix_world.inverted() @ obj.matrix_world
        updates = []
        original_mass = {}
        for vertex in obj.data.vertices:
            old = {g.group: g.weight for g in vertex.groups}
            point = np.array(world_to_arm @ vertex.co)
            weights_by_f = {
                f: sum(old.get(g.index, 0.0) for g in gs) for f, gs in groups.items()
            }
            candidates = []
            for finger, (head, tip, extent) in plans.items():
                direction = (tip - head) / extent
                distance = float((point - head) @ direction)
                radial = float(np.linalg.norm(point - head - direction * distance))
                other = max(weights_by_f[f] for f in FINGERS if f != finger)
                mass = old.get(hand_group.index, 0.0) + weights_by_f[finger]
                if (
                    mass < 0.5
                    or other > 0.25
                    or radial > length * 0.06
                    or not -extent * 0.1 < distance < extent * 1.15
                ):
                    continue
                candidates.append(
                    (radial, finger, mass * joint_weights(distance, extent))
                )
            if candidates:
                _, finger, weights = min(candidates, key=lambda candidate: candidate[0])
                updates.append((vertex.index, finger, weights))
                original_mass[vertex.index] = sum(old.values())
        for index, finger, weights in updates:
            gs = [hand_group, *groups[finger]]
            for group in gs:
                group.remove([index])
            for group, weight in zip(gs, weights):
                if weight > 1e-8:
                    group.add([index], float(weight), "REPLACE")
            total = sum(g.weight for g in obj.data.vertices[index].groups)
            if abs(total - original_mass[index]) > 1e-5:
                raise RuntimeError("hand root repair changed total skin weight")
    bpy.context.view_layer.update()
    return list(plans)


def add_finger_smoothing(mesh, sides, prefix, iterations):
    """Attach the same local smoothing to an export or an evaluated preview."""
    names = {
        bone_name(side, finger, joint, prefix)
        for side in sides
        for finger in FINGERS
        for joint in range(3)
    }
    indices = {g.index for g in mesh.vertex_groups if g.name in names}
    weights = [
        (v.index, sum(g.weight for g in v.groups if g.group in indices))
        for v in mesh.data.vertices
    ]
    group = mesh.vertex_groups.new(name="_fist_skin_smoothing")
    for index, weight in weights:
        if weight > 0.01:
            group.add([index], min(weight, 1.0), "REPLACE")
    modifier = mesh.modifiers.new(name="Fist skin folds", type="SMOOTH")
    modifier.factor = 0.5
    modifier.iterations = iterations
    modifier.vertex_group = group.name
    return modifier.name, group.name


def smooth_baked_fingers(meshes, sides, prefix, iterations):
    """Round the skin folds on a baked fist, preserving body and topology."""
    import bpy

    if not iterations:
        return
    for mesh in meshes:
        modifier_name, group_name = add_finger_smoothing(
            mesh, sides, prefix, iterations
        )
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.modifier_apply(modifier=modifier_name)
        remaining_group = mesh.vertex_groups.get(group_name)
        if remaining_group is not None:
            mesh.vertex_groups.remove(remaining_group)
