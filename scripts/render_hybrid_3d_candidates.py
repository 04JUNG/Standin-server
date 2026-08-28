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
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Matrix, Vector


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


def evaluated_points(mesh) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        return [
            (evaluated.matrix_world @ vertex.co).copy()
            for vertex in evaluated_mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()


def point_at(obj, target: Vector,
             world_up: Vector = Vector((0.0, 1.0, 0.0))) -> None:
    direction = (target - obj.location).normalized()
    world_up = world_up.normalized()
    right = direction.cross(world_up)
    if right.length < 1e-6:
        world_up = Vector((0.0, 0.0, 1.0))
        right = direction.cross(world_up)
    right.normalize()
    up = right.cross(direction).normalized()
    back = -direction
    # Matrix의 열을 카메라 local X(right), Y(up), Z(back)로 둔다.
    rotation = Matrix((right, up, back)).transposed()
    obj.rotation_euler = rotation.to_euler()


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


def body_axes(armature) -> tuple[Vector, Vector, Vector]:
    """Return character-relative right/forward/up axes.

    BVH files can carry arbitrary root yaw/roll. The shoulder line defines
    screen-right and pelvis-to-shoulder defines screen-up, so the review image
    remains anatomically readable while preserving all joint relationships.
    """
    by_suffix = {
        bone.name.rsplit(":", 1)[-1].lower(): bone
        for bone in armature.pose.bones
    }

    def world_head(suffix: str) -> Vector:
        bone = by_suffix[suffix.lower()]
        return armature.matrix_world @ bone.head

    try:
        left = world_head("LeftShoulder")
        right_point = world_head("RightShoulder")
        pelvis = world_head("Hips")
        shoulder_centre = (left + right_point) * 0.5
        right = right_point - left
        up = shoulder_centre - pelvis
        if right.length > 1e-5 and up.length > 1e-5:
            right.normalize()
            up = up - right * up.dot(right)
            up.normalize()
            forward = right.cross(up).normalized()
            up = forward.cross(right).normalized()
            return right, forward, up
    except KeyError:
        pass
    return (
        Vector((1.0, 0.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
        Vector((0.0, 1.0, 0.0)),
    )


def setup_scene(mesh, pose_axes: tuple[Vector, Vector, Vector], view: str,
                output: Path, size: int) -> None:
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

    points = evaluated_points(mesh)
    lower = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    centre = (lower + upper) * 0.5
    extent = max(*(upper - lower), 1e-3)

    camera_data = bpy.data.cameras.new("HybridReviewCamera")
    camera = bpy.data.objects.new("HybridReviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    right, forward, up = pose_axes
    offsets = {
        "front": forward * (extent * 3.0),
        "three_quarter": (right + forward).normalized() * (extent * 3.0),
        "side": right * (extent * 3.0),
        "back": -forward * (extent * 3.0),
    }
    if view not in offsets:
        raise ValueError(f"unsupported review view: {view}")
    camera.location = centre + offsets[view]
    point_at(camera, centre, up)
    bpy.context.view_layer.update()
    world_to_camera = camera.matrix_world.inverted()
    camera_points = [world_to_camera @ point for point in points]
    projected_width = (
        max(point.x for point in camera_points)
        - min(point.x for point in camera_points)
    )
    projected_height = (
        max(point.y for point in camera_points)
        - min(point.y for point in camera_points)
    )
    # 정사영 화면에 실제로 보이는 폭/높이만 사용한다. 카메라 깊이를
    # 포함하면 누움·측면 포즈가 카드 안에서 지나치게 작아진다.
    camera.data.ortho_scale = max(
        projected_height * 1.16, projected_width * 1.16, 1.0
    )

    add_area_light(
        "HybridReviewKey",
        centre + (right * 1.8 + up * 1.8 + forward * 2.2) * extent,
        centre,
        900.0,
        extent * 2.0,
    )
    add_area_light(
        "HybridReviewFill",
        centre + (-right * 2.0 + up * 0.8 + forward * 0.8) * extent,
        centre,
        520.0,
        extent * 1.8,
    )
    add_area_light(
        "HybridReviewRim",
        centre + (up * 1.2 - forward * 2.0) * extent,
        centre,
        700.0,
        extent * 1.4,
    )

def prepare_pose(job: dict, character: Path):
    rt.reset_scene()
    destination_armature, meshes = rt.import_character(str(character))
    # 검수용 FBX에 남아 있는 Action/NLA가 render dependency-graph 평가 때
    # 수동 리타기팅 포즈를 덮어쓰지 못하게 한다. 이 상태를 지우지 않으면
    # 같은 BVH도 첫 번째와 두 번째 카메라 렌더가 서로 다른 포즈가 된다.
    destination_armature.animation_data_clear()
    for mesh in meshes:
        mesh.animation_data_clear()
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
    pose_axes = body_axes(destination_armature)
    # 검수 이미지는 애니메이션이 필요 없다. 현재 포즈를 메시 정점에 굽고
    # 리그를 제거해 이후 카메라/렌더 평가가 포즈를 바꿀 여지를 없앤다.
    rt.apply_output_mode(destination_armature, meshes, "static_mesh")
    return mesh, pose_axes, report


def cleanup_render_setup() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("HybridReview"):
            bpy.data.objects.remove(obj, do_unlink=True)


def render_prepared(job: dict, mesh, pose_axes, report, size: int) -> dict:
    cleanup_render_setup()
    destination = Path(job["output"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    setup_scene(mesh, pose_axes, job["view"], destination, size)
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
    jobs = manifest["jobs"]
    results = [None] * len(jobs)
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for index, job in enumerate(jobs):
        key = (job["pose_id"], str(Path(job["bvh_path"]).resolve()))
        groups.setdefault(key, []).append((index, job))

    for grouped_jobs in groups.values():
        pending = [
            (index, job) for index, job in grouped_jobs
            if args.force or not Path(job["output"]).resolve().is_file()
        ]
        for index, job in grouped_jobs:
            if (index, job) not in pending:
                results[index] = {
                    "status": "cached",
                    "output": str(Path(job["output"]).resolve()),
                }
        if not pending:
            continue
        try:
            mesh, pose_axes, report = prepare_pose(pending[0][1], character)
        except Exception as exc:
            for index, job in pending:
                results[index] = {
                    "status": "failed",
                    "output": str(Path(job["output"]).resolve()),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            continue
        for index, job in pending:
            try:
                results[index] = render_prepared(
                    job, mesh, pose_axes, report, args.size
                )
            except Exception as exc:
                results[index] = {
                    "status": "failed",
                    "output": str(Path(job["output"]).resolve()),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

    for number, (job, result) in enumerate(zip(jobs, results), 1):
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
