#!/usr/bin/env python3
"""Render searched BVH candidates on the real Standin skinned character.

Run inside Blender. The input manifest is produced by
``scripts/build_hybrid_3d_review.py``. Each unique ``pose_id + view`` is
retargeted through the frozen CHAIN_TRANSPORT_V3 QA converter and rendered as
an orthographic PNG. This is an offline review path, not a runtime dependency.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


REPO = Path(__file__).resolve().parent.parent
CONVERTER_ROOT = REPO / "qa/retarget/CHAIN_TRANSPORT_V3"
sys.path.insert(0, str(CONVERTER_ROOT))

from converter import retarget as rt


def parse_args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(raw)


def evaluated_bounds(mesh) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        points = [mesh.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
        lower = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
        upper = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
        return lower, upper
    finally:
        evaluated.to_mesh_clear()


def point_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location: Vector, target: Vector,
                   energy: float, size: float) -> None:
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    point_at(light, target)


def body_horizontal_axes(armature) -> tuple[Vector, Vector]:
    """Return stable character-relative right/forward axes on the ground plane.

    BVH files can carry different root yaw values. Deriving the camera from the
    retargeted shoulder line makes ``front``/``three_quarter`` mean the same
    body-relative direction for every candidate while preserving pose pitch,
    roll, leaning, and falling.
    """
    by_suffix = {
        bone.name.rsplit(":", 1)[-1].lower(): bone
        for bone in armature.pose.bones
    }

    def world_head(suffix: str) -> Vector:
        bone = by_suffix[suffix.lower()]
        return armature.matrix_world @ bone.head

    for left_name, right_name in (
        ("LeftShoulder", "RightShoulder"),
        ("LeftUpLeg", "RightUpLeg"),
    ):
        try:
            right = world_head(right_name) - world_head(left_name)
        except KeyError:
            continue
        right.y = 0.0
        if right.length > 1e-5:
            right.normalize()
            # Rest-facing convention matches the existing +Z front camera
            # when the Mixamo shoulder line points along +X.
            forward = Vector((-right.z, 0.0, right.x)).normalized()
            return right, forward
    return Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))


def setup_scene(mesh, armature, view: str, output: Path, size: int) -> None:
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(output)

    world = scene.world or bpy.data.worlds.new("HybridReviewWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.035, 0.042, 0.052, 1.0)
        background.inputs["Strength"].default_value = 0.45

    lower, upper = evaluated_bounds(mesh)
    centre = (lower + upper) * 0.5
    extent = max(*(upper - lower), 1e-3)

    camera_data = bpy.data.cameras.new("HybridReviewCamera")
    camera = bpy.data.objects.new("HybridReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    right, forward = body_horizontal_axes(armature)
    offsets = {
        "front": forward * (extent * 3.0),
        "three_quarter": (right + forward).normalized() * (extent * 3.0),
        "side": right * (extent * 3.0),
        "back": -forward * (extent * 3.0),
    }
    if view not in offsets:
        raise ValueError(f"unsupported review view: {view}")
    camera.location = centre + offsets[view]
    point_at(camera, centre)
    size_xyz = upper - lower
    camera.data.ortho_scale = max(
        size_xyz.y * 1.18, max(size_xyz.x, size_xyz.z) * 1.32, 1.0
    )

    add_area_light(
        "HybridReviewKey",
        centre + Vector((extent * 1.8, extent * 1.8, extent * 2.2)),
        centre,
        900.0,
        extent * 2.0,
    )
    add_area_light(
        "HybridReviewFill",
        centre + Vector((-extent * 2.0, extent * 0.8, extent * 0.8)),
        centre,
        520.0,
        extent * 1.8,
    )
    add_area_light(
        "HybridReviewRim",
        centre + Vector((0.0, extent * 1.2, -extent * 2.0)),
        centre,
        700.0,
        extent * 1.4,
    )

    # Y-up 캐릭터를 위한 XZ ground plane. Mesh와 겹치지 않게 약간 아래에 둔다.
    bpy.ops.mesh.primitive_plane_add(
        size=extent * 5.0,
        location=(centre.x, lower.y - extent * 0.015, centre.z),
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    ground = bpy.context.object
    ground.name = "HybridReviewGround"
    material = bpy.data.materials.new("HybridReviewGroundMaterial")
    material.diffuse_color = (0.075, 0.085, 0.105, 1.0)
    material.roughness = 0.92
    ground.data.materials.append(material)


def render_job(job: dict, character: Path, size: int) -> dict:
    rt.reset_scene()
    destination = Path(job["output"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)

    destination_armature, meshes = rt.import_character(str(character))
    source_armature = rt.import_bvh(str(Path(job["bvh_path"]).resolve()), frame=0)
    report = rt.ConvertReport(output_mode="review_render", frame=0)
    report = rt.retarget(source_armature, destination_armature, report=report)
    if not report.ok:
        raise RuntimeError(
            f"retarget rejected: missing={report.missing_required}; "
            f"warnings={report.warnings}"
        )
    source_armature.hide_render = True
    mesh = max(meshes, key=lambda item: len(item.data.vertices))
    setup_scene(mesh, destination_armature, job["view"], destination, size)
    bpy.ops.render.render(write_still=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Blender did not create the render")
    return {
        "status": "rendered",
        "output": str(destination),
        "retarget": report.as_dict(),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    character = args.character.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    jobs = manifest["jobs"]
    for number, job in enumerate(jobs, 1):
        destination = Path(job["output"]).resolve()
        if destination.is_file() and not args.force:
            results.append({"status": "cached", "output": str(destination)})
            print(f"HYBRID_3D={number}/{len(jobs)}:cached:{job['pose_id']}:{job['view']}")
            continue
        try:
            result = render_job(job, character, args.size)
        except Exception as exc:
            result = {
                "status": "failed",
                "output": str(destination),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        results.append(result)
        print(
            f"HYBRID_3D={number}/{len(jobs)}:{result['status']}:"
            f"{job['pose_id']}:{job['view']}"
        )

    manifest["character"] = str(character)
    manifest["render_size"] = args.size
    manifest["results"] = results
    manifest["status"] = (
        "complete" if all(row["status"] in {"rendered", "cached"} for row in results)
        else "failed"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = sum(row["status"] == "failed" for row in results)
    print(f"rendered/cached={len(results) - failed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
