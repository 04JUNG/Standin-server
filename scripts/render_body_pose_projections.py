#!/usr/bin/env python3
"""Blender-only, hash-pinned diagnostic body meshes at fixed BVH pose and view.
No candidate-specific camera search, 2D pose warp, mesh fitting, or runtime approval.
Usage: blender -b --python this.py -- --jobs jobs.json --out new_directory
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
VERSION = "body-pose-parts-projection.v1"
ANGLES = {
    "front": 0,
    "three_quarter": 45,
    "side": 90,
    "back": 180,
    "side_opposite": -90,
    "three_quarter_opposite": -45,
}
PARTS = (
    "torso",
    "left_upper_arm",
    "left_forearm",
    "right_upper_arm",
    "right_forearm",
    "left_thigh",
    "left_calf",
    "right_thigh",
    "right_calf",
)
BONES = (
    ("Hips", "Spine", "Spine1", "Spine2", "Neck"),
    ("LeftArm",),
    ("LeftForeArm",),
    ("RightArm",),
    ("RightForeArm",),
    ("LeftUpLeg",),
    ("LeftLeg",),
    ("RightUpLeg",),
    ("RightLeg",),
)
KP = (
    "Head",
    "Head",
    "Head",
    "Head",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
)
COLORS = (
    (0.15, 0.15, 0.15),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0.2, 0.5, 1),
    (0.5, 0.2, 1),
    (1, 0.2, 0.5),
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    from converter import retarget as rt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    data = json.loads(args.jobs.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    solver_sha = sha(Path(rt.__file__))
    renderer_sha = sha(__file__)
    # Verify EVERY input before rendering any candidate.
    for a in data["assets"]:
        if sha(a["fbx_path"]) != a["asset_sha256"]:
            raise ValueError("body_fbx_hash_mismatch")
    for j in data["jobs"]:
        if j["view"] not in ANGLES or sha(j["bvh_path"]) != j["pose_sha256"]:
            raise ValueError("pose_or_view_identity_mismatch")
    records = []
    for asset in data["assets"]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(
            filepath=asset["fbx_path"],
            ignore_leaf_bones=False,
            automatic_bone_orientation=False,
        )
        arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if len(arms) != 1 or len(meshes) != 1:
            raise ValueError("unsupported_target_structure")
        arm, mesh = arms[0], meshes[0]
        arm.animation_data_clear()
        mesh.animation_data_clear()
        rt._apply_object_transform(arm)
        for b in arm.pose.bones:
            b.matrix_basis.identity()
        groups = {g.index: g.name.split(":")[-1] for g in mesh.vertex_groups}
        vw = np.zeros((len(mesh.data.vertices), 10))
        for v in mesh.data.vertices:
            for g in v.groups:
                name = groups[g.group]
                label = next((i for i, bs in enumerate(BONES, 1) if name in bs), 0)
                vw[v.index, label] += g.weight
        mesh.data.materials.clear()
        for i, col in enumerate(COLORS):
            mat = bpy.data.materials.new("part-" + str(i))
            mat.diffuse_color = (*col, 1)
            mesh.data.materials.append(mat)
        for f in mesh.data.polygons:
            f.material_index = int(vw[list(f.vertices)].mean(0).argmax())
        sc = bpy.context.scene
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.render.resolution_x = 512
        sc.render.resolution_y = 512
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.film_transparent = True
        sc.display.shading.light = "FLAT"
        sc.display.shading.color_type = "MATERIAL"
        sc.display.shading.show_shadows = False
        sc.display.shading.show_cavity = False
        sc.display.shading.show_specular_highlight = False
        sc.display.render_aa = "OFF"
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
        camera = bpy.data.objects.new(
            "fixed-body-camera", bpy.data.cameras.new("fixed-body-camera")
        )
        sc.collection.objects.link(camera)
        sc.camera = camera
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 3.2
        target = Vector((0, 0, 1))
        for job in data["jobs"]:
            dest = args.out / asset["body_id"] / job["key"]
            dest.mkdir(parents=True, exist_ok=True)
            meta = dest / "projection.json"
            if meta.exists():
                previous = json.loads(meta.read_text())
                if (
                    previous["asset_sha256"] != asset["asset_sha256"]
                    or previous["pose_sha256"] != job["pose_sha256"]
                    or previous["renderer_sha256"] != renderer_sha
                    or previous["retarget_sha256"] != solver_sha
                    or previous["key"] != job["key"]
                    or previous["view"] != job["view"]
                    or previous["body_id"] != asset["body_id"]
                    or sha(previous["image"]) != previous["image_sha256"]
                ):
                    raise ValueError("stale_projection_cache")
                records.append(previous)
                continue
            for b in arm.pose.bones:
                b.matrix_basis.identity()
            bpy.context.view_layer.update()
            source = rt.import_bvh(job["bvh_path"], frame=0)
            report = rt.retarget(source, arm, palm_roll_mu=0.5)
            if not report.ok:
                raise ValueError("retarget_failed")
            bpy.data.objects.remove(source, do_unlink=True)
            bpy.context.view_layer.update()
            angle = math.radians(ANGLES[job["view"]])
            camera.location = target + Vector(
                (-5 * math.sin(angle), -5 * math.cos(angle), 0)
            )
            camera.rotation_euler = (
                (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            )
            bpy.context.view_layer.update()
            names = {b.name.split(":")[-1]: b for b in arm.pose.bones}
            points = []
            for name in KP:
                v = world_to_camera_view(
                    sc, camera, arm.matrix_world @ names[name].head
                )
                points.append([float(v.x * 512), float((1 - v.y) * 512)])
            evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
            ev = evaluated.to_mesh()
            co = np.array([evaluated.matrix_world @ v.co for v in ev.vertices])
            evaluated.to_mesh_clear()
            if not np.isfinite(co).all() or np.ptp(co, axis=0).max() < 0.1:
                raise ValueError("invalid_posed_mesh")
            sc.render.filepath = str(dest / "parts.png")
            bpy.ops.render.render(write_still=True)
            row = dict(
                body_id=asset["body_id"],
                asset_sha256=asset["asset_sha256"],
                pose_id=job["pose_id"],
                pose_sha256=job["pose_sha256"],
                view=job["view"],
                key=job["key"],
                camera=dict(
                    version="library-orthographic-yaw.v1",
                    yaw_degrees=ANGLES[job["view"]],
                    ortho_scale=3.2,
                    target=list(target),
                ),
                keypoints=points,
                valid_joint_mask=[False] * 5 + [True] * 12,
                parts=list(PARTS),
                colors_linear=COLORS,
                renderer_version=VERSION,
                renderer_sha256=renderer_sha,
                retarget_sha256=solver_sha,
                production_pose_qa=False,
                mesh_finite=True,
                image=str(dest / "parts.png"),
                image_sha256=sha(dest / "parts.png"),
            )
            meta.write_text(json.dumps(row, indent=2))
            records.append(row)
            print("PROJECTED", asset["body_id"], job["key"], flush=True)
    (args.out / "index.json").write_text(
        json.dumps(
            dict(schema_version=VERSION, jobs_sha256=sha(args.jobs), records=records),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
