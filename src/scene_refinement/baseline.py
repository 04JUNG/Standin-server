"""Derive the preservation baseline from the exact reviewed input artifacts."""

from pathlib import Path
import json, math, hashlib
import bpy, numpy as np
from converter.scene import _import, compose
from converter.rough_layout import camera_for, geometry_for, fit_translation
from .people import geometry
from .metrics import measures, posture_cost


def run(ctx):
    request = json.loads((ctx.work / "request.json").read_text())
    cut = request["cut_id"]
    target = ctx.baseline / cut
    target.mkdir(parents=True, exist_ok=True)
    camera = camera_for(request["image"]["width"], request["image"]["height"])
    camera["look_direction_blender"] = [0, 1, 0]
    layout = []
    people = []
    geometries = []
    selected = []
    paths = []
    rows = []
    catalog = {p["pose_id"]: p for p in json.loads(ctx.catalog.read_text())["poses"]}
    for person in request["persons"]:
        i = person["person_index"]
        source = ctx.work / cut / f"person_{i}-reviewed.fbx"
        paths.append(source)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        objects = _import(source)
        bpy.context.view_layer.update()
        rig = next(o for o in objects if o.type == "ARMATURE")
        meshes = [o for o in objects if o.type == "MESH"]
        points, joints = geometry_for(rig, meshes)
        translation, fit = fit_translation(
            points, joints, person["observation"], camera
        )
        geometry(target / f"person_{i}-geometry.npz")
        g = np.load(target / f"person_{i}-geometry.npz")
        p = g["head_points"] + translation
        d = p[:, 1] - camera["position_blender"][1]
        xy = np.column_stack(
            (
                camera["principal_pixels"][0] + camera["focal_pixels"] * p[:, 0] / d,
                camera["principal_pixels"][1]
                - camera["focal_pixels"]
                * (p[:, 2] - camera["position_blender"][2])
                / d,
            )
        )
        headbox = np.r_[xy.min(0), xy.max(0)]

        def bone(suffix):
            return next(b for b in rig.data.bones if b.name.split(":")[-1] == suffix)

        left = rig.matrix_world @ bone("LeftArm").head_local
        right = rig.matrix_world @ bone("RightArm").head_local
        direction = (left - right).cross(__import__("mathutils").Vector((0, 0, 1)))
        yaw = math.degrees(math.atan2(direction.x, -direction.y))
        center = (headbox[:2] + headbox[2:]) / 2
        ray = -math.degrees(
            math.atan(
                (center[0] - camera["principal_pixels"][0]) / camera["focal_pixels"]
            )
        )
        scope = person["output_scope"]
        metric = measures(catalog[person["pose_id"]]["points"])
        posture = min(
            ("standing", "sitting"),
            key=lambda p: posture_cost(metric, p, "both_grounded"),
        )
        layout.append(
            dict(
                person_index=i,
                final_head_bbox=headbox.tolist(),
                target_head_bbox=headbox.tolist(),
                depth_m=float(p[:, 1].mean() - camera["position_blender"][1]),
                output_scope=scope,
            )
        )
        people.append(
            dict(
                person_index=i,
                bbox=person["observation"]["box"],
                head_center=center.tolist(),
                head_reference_reliable=sum(
                    s >= 0.45 for s in person["observation"]["scores"][:5]
                )
                >= 2,
            )
        )
        geometries.append(
            dict(
                person_index=i,
                torso_camera_yaw=(yaw - ray + 180) % 360 - 180,
                output_scope=scope,
            )
        )
        selected.append(
            dict(
                person_index=i,
                pose_id=person["pose_id"],
                output_scope=scope,
                posture=posture,
            )
        )
        rows.append(
            dict(
                person_index=i,
                fbx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                observation=person["observation"],
            )
        )
    for name, value in [
        ("people", dict(people=people)),
        ("selection", dict(people=selected)),
        ("geometry", dict(people=geometries)),
        ("layout", dict(camera=camera, people=layout)),
    ]:
        (target / f"{name}.json").write_text(json.dumps(value, indent=2))
    manifest = dict(
        schema_version=1,
        cut_id=cut,
        coordinate_system="threejs-y-up-m",
        rough_layout=dict(
            version="rough-layout-v1", width=camera["width"], height=camera["height"]
        ),
        persons=rows,
    )
    report = compose(manifest, paths, target / "scene.fbx")
    from .scenes import add_camera, render

    cam = add_camera(camera)
    render(cam, camera, target / "final.png")
    (target / "report.json").write_text(json.dumps(report, indent=2))
