"""Fixed-length shoulder/elbow reach; selected wrist orientation stays fixed."""

from pathlib import Path
import json, math, hashlib
import bpy, numpy as np
from mathutils import Matrix, Vector
from converter import retarget as rt
from .rig import validate_rig
from converter.framing import crop_meshes


def project(v, cfg):
    d = v.y - cfg["position_blender"][1]
    return np.array(
        [
            cfg["principal_pixels"][0] + cfg["focal_pixels"] * v.x / d,
            cfg["principal_pixels"][1]
            - cfg["focal_pixels"] * (v.z - cfg["position_blender"][2]) / d,
        ]
    )


def raypoint(p, depth, cfg):
    return Vector(
        (
            (p[0] - cfg["principal_pixels"][0]) * depth / cfg["focal_pixels"],
            cfg["position_blender"][1] + depth,
            cfg["position_blender"][2]
            + (cfg["principal_pixels"][1] - p[1]) * depth / cfg["focal_pixels"],
        )
    )


def joint(arm, name):
    return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head


def solve(arm, side, target, elbow_target, cfg, headheight, palm_offset=None):
    offset = Vector((0, 0, 0)) if palm_offset is None else palm_offset
    S = joint(arm, side + "Arm")
    E = joint(arm, side + "ForeArm")
    W = joint(arm, side + "Hand")
    a = (E - S).length
    b = (W - E).length
    reach = a + b
    d0 = W.y - cfg["position_blender"][1]
    ds = np.linspace(
        max(0.12, S.y - cfg["position_blender"][1] - reach),
        S.y - cfg["position_blender"][1] + reach,
        81,
    )
    solutions = []
    for depth in ds:
        T = raypoint(target, depth, cfg) - offset
        length = (T - S).length
        if not abs(a - b) + 0.005 < length < reach - 0.005:
            continue
        direction = (T - S).normalized()
        distance = (a * a - b * b + length * length) / (2 * length)
        radius = math.sqrt(max(0, a * a - distance * distance))
        center = S + direction * distance
        pole = E - center
        pole -= direction * pole.dot(direction)
        if pole.length < 1e-5:
            pole = direction.cross(Vector((0, 0, 1)))
        if pole.length < 1e-5:
            pole = direction.cross(Vector((1, 0, 0)))
        pole.normalize()
        other = direction.cross(pole).normalized()
        for theta in np.linspace(-math.pi, math.pi, 25):
            candidate = center + radius * (
                pole * math.cos(theta) + other * math.sin(theta)
            )
            score = 0.03 * (candidate - E).length / max(reach, 1e-6) + 0.015 * abs(
                depth - d0
            ) / max(reach, 1e-6)
            if elbow_target is not None:
                score += np.linalg.norm(project(candidate, cfg) - elbow_target) / max(
                    headheight, 1
                )
            solutions.append((float(score), T, candidate))
    if not solutions:
        return dict(
            accepted=False,
            reason="image_goal_outside_fixed_length_arm_reach",
            before_hand_pixels=project(W + offset, cfg).tolist(),
            target_hand_pixels=list(target),
        )
    score, T, wanted_elbow = min(solutions, key=lambda x: x[0])
    before = project(W + offset, cfg)
    hand = arm.pose.bones["mixamorig:" + side + "Hand"]
    hand_world = (arm.matrix_world @ hand.matrix).copy()
    bases = {
        p.name: p.matrix_basis.copy()
        for p in (
            arm.pose.bones["mixamorig:" + side + "Arm"],
            arm.pose.bones["mixamorig:" + side + "ForeArm"],
            hand,
        )
    }

    def rotate(name, current, wanted, pivot):
        pb = arm.pose.bones["mixamorig:" + name]
        world = arm.matrix_world @ pb.matrix
        delta = current.rotation_difference(wanted).to_matrix().to_4x4()
        pb.matrix = (
            arm.matrix_world.inverted()
            @ Matrix.Translation(pivot)
            @ delta
            @ Matrix.Translation(-pivot)
            @ world
        )
        bpy.context.view_layer.update()

    rotate(side + "Arm", (E - S).normalized(), (wanted_elbow - S).normalized(), S)
    actual_elbow = joint(arm, side + "ForeArm")
    actual_hand = joint(arm, side + "Hand")
    rotate(
        side + "ForeArm",
        (actual_hand - actual_elbow).normalized(),
        (T - actual_elbow).normalized(),
        actual_elbow,
    )
    # Keep the selected gesture's wrist orientation; only shoulder/elbow reach changes.
    new_hand = joint(arm, side + "Hand")
    hand_world.translation = new_hand
    hand.matrix = arm.matrix_world.inverted() @ hand_world
    bpy.context.view_layer.update()
    after = project(joint(arm, side + "Hand") + offset, cfg)
    accepted = np.linalg.norm(after - target) + 0.01 < np.linalg.norm(before - target)
    if not accepted:
        for name, mat in bases.items():
            arm.pose.bones[name].matrix_basis = mat
        bpy.context.view_layer.update()
    return dict(
        accepted=bool(accepted),
        side=side,
        effector="palm_skin_centroid" if palm_offset is not None else "wrist",
        before_hand_pixels=before.tolist(),
        after_hand_pixels=after.tolist(),
        target_hand_pixels=list(target),
        pixel_error_before=float(np.linalg.norm(before - target)),
        pixel_error_after=float(np.linalg.norm(after - target)),
        elbow_target=None if elbow_target is None else list(elbow_target),
        elbow_pixels=project(joint(arm, side + "ForeArm"), cfg).tolist(),
        score=score,
        arm_length_m=reach,
        hand_orientation_preserved=True,
    )


def palm_offset(arm, meshes, side):
    points = []
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        group = obj.vertex_groups.get("mixamorig:" + side + "Hand")
        if group is None:
            continue
        ev = obj.evaluated_get(graph)
        mesh = ev.to_mesh()
        for v in obj.data.vertices:
            if any(g.group == group.index and g.weight > 0.6 for g in v.groups):
                points.append(ev.matrix_world @ mesh.vertices[v.index].co)
        ev.to_mesh_clear()
    return (
        sum(points, Vector()) / len(points) - joint(arm, side + "Hand")
        if points
        else Vector((0, 0, 0))
    )


def export(path):
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        object_types={"ARMATURE", "MESH", "EMPTY"},
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        global_scale=1,
        add_leaf_bones=False,
        bake_anim=False,
        use_mesh_modifiers=True,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        embed_textures=False,
    )


def geometry(path):
    points = []
    triangles = []
    heads = []
    offset = 0
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted(bpy.context.scene.objects, key=lambda o: o.name):
        if obj.type != "MESH":
            continue
        ev = obj.evaluated_get(graph)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        positions = np.array([tuple(ev.matrix_world @ v.co) for v in mesh.vertices])
        groups = {g.index: g.name for g in obj.vertex_groups}
        mask = [
            sum(
                g.weight
                for g in v.groups
                if groups.get(g.group, "").split(":")[-1] == "Head"
            )
            > 0.5
            for v in obj.data.vertices
        ]
        heads.extend(positions[np.array(mask, bool)].tolist())
        points.extend(positions.tolist())
        triangles.extend(
            [[offset + j for j in t.vertices] for t in mesh.loop_triangles]
        )
        offset += len(positions)
        ev.to_mesh_clear()
    np.savez_compressed(
        path,
        points=np.array(points),
        triangles=np.array(triangles, np.int32),
        head_points=np.array(heads),
    )
    return len(points), len(triangles), len(heads)


def run(ctx):
    ROOT = ctx.work
    for item in json.loads((ROOT / "inputs.json").read_text()):
        folder = ROOT / item["cut_id"]
        layout = json.loads((folder / "layout.json").read_text())
        cfg = layout["camera"]
        selections = {
            p["person_index"]: p
            for p in json.loads((folder / "selection.json").read_text())["people"]
        }
        geo = json.loads((folder / "geometry.json").read_text())
        for meta in geo["people"]:
            i = meta["person_index"]
            person = selections[i]
            goals = person["arm_goals"]
            if not goals:
                continue
            p = next(p for p in layout["people"] if p["person_index"] == i)
            rt.reset_scene()
            bpy.ops.import_scene.fbx(
                filepath=str(folder / f"person_{i}-full.fbx"),
                ignore_leaf_bones=False,
                automatic_bone_orientation=False,
                use_anim=False,
            )
            arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
            meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            validate_rig(arm, meshes)
            lengths = {b.name: b.length for b in arm.data.bones}
            transform = Matrix.Translation(
                Vector(p["translation_blender"])
            ) @ Matrix.Rotation(math.radians(p["rotation_y_degrees"]), 4, "Z")
            roots = [o for o in bpy.context.scene.objects if o.parent is None]
            for obj in roots:
                obj.matrix_world = transform @ obj.matrix_world
            bpy.context.view_layer.update()
            head_before = joint(arm, "Head")
            headbox = np.array(p["raw_head_bbox"])
            headcenter = (headbox[:2] + headbox[2:]) / 2
            offset = headcenter - np.array(person["source_head_reference"])
            reports = []
            used = set()
            for goal in goals:
                target = np.array(goal["hand_pixels"]) + offset
                elbow = (
                    None
                    if goal["elbow_pixels"] is None
                    else np.array(goal["elbow_pixels"]) + offset
                )
                side = goal["side"]
                offsets = (
                    {s: palm_offset(arm, meshes, s) for s in ("Left", "Right")}
                    if "localized" in goal["source"]
                    else {}
                )
                if side is None:
                    # Evaluate actual fixed-length reach for both arms; closest screen shoulder can be wrong.
                    original = {
                        pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones
                    }
                    trials = []
                    for candidate_side in ("Left", "Right"):
                        if candidate_side in used:
                            continue
                        result = solve(
                            arm,
                            candidate_side,
                            target,
                            elbow,
                            cfg,
                            headbox[3] - headbox[1],
                            offsets.get(candidate_side),
                        )
                        trials.append((candidate_side, result))
                        for name, mat in original.items():
                            arm.pose.bones[name].matrix_basis = mat
                        bpy.context.view_layer.update()
                    feasible = [x for x in trials if x[1].get("accepted")]
                    side = (
                        min(feasible, key=lambda x: x[1].get("score", 1e6))[0]
                        if feasible
                        else min(
                            (s for s in ("Left", "Right") if s not in used),
                            key=lambda s: np.linalg.norm(
                                project(joint(arm, s + "Arm"), cfg) - target
                            ),
                        )
                    )
                used.add(side)
                reports.append(
                    solve(
                        arm,
                        side,
                        target,
                        elbow,
                        cfg,
                        headbox[3] - headbox[1],
                        offsets.get(side),
                    )
                )
            rt.apply_output_mode(arm, meshes, "rigged_rest")
            bpy.context.view_layer.update()
            head_error = (joint(arm, "Head") - head_before).length
            length_error = max(abs(b.length - lengths[b.name]) for b in arm.data.bones)
            assert head_error < 1e-4 and length_error < 1e-4
            for obj in roots:
                obj.matrix_world = transform.inverted() @ obj.matrix_world
            bpy.context.view_layer.update()
            export(folder / f"person_{i}-full.fbx")
            framing = crop_meshes(arm, meshes, person["output_scope"])
            export(folder / f"person_{i}.fbx")
            rt.reset_scene()
            bpy.ops.import_scene.fbx(
                filepath=str(folder / f"person_{i}.fbx"),
                ignore_leaf_bones=False,
                automatic_bone_orientation=False,
                use_anim=False,
            )
            counts = geometry(folder / f"person_{i}-geometry.npz")
            meta.update(
                fbx_sha256=hashlib.sha256(
                    (folder / f"person_{i}.fbx").read_bytes()
                ).hexdigest(),
                vertices=counts[0],
                triangles=counts[1],
                head_vertices=counts[2],
                scope_execution=framing,
                arm_refinement=dict(
                    goals=reports,
                    head_position_error_m=head_error,
                    bone_length_max_error=length_error,
                    manual_targets=False,
                ),
            )
            print("ARMS", item["cut_id"], i, reports, flush=True)
        (folder / "geometry.json").write_text(
            json.dumps(geo, ensure_ascii=False, indent=2)
        )
