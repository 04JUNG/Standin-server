"""변환된 FBX 산출물을 라이브러리 썸네일과 같은 anatomical 카메라로 렌더한다.

Blender 자식 프로세스 안에서만 실행된다. ``bpy``/``mathutils``는 함수 안에서
import하므로 HTTP 프로세스가 이 모듈을 import해도 Blender에 의존하지 않는다.

카메라·조명·재질·배경은 ``qa/retarget/CHAIN_TRANSPORT_V3_2_RELATIVE_MESH_QA/tools/
render_manifest_fronts.py::_render_view``와 같다 — 2026-09-03 라이브러리 번들이 그
코드로 만들어졌으므로, 여기서 값을 바꾸면 refine preview와 후보 썸네일의 그림이
달라진다. 바꿔야 한다면 라이브러리 썸네일도 함께 다시 굽는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from converter.protocol import (
    THUMBNAIL_CAMERA_CONVENTION,
    THUMBNAIL_ENGINES,
    THUMBNAIL_VIEWS,
)


class ThumbnailRenderError(RuntimeError):
    """요청한 어떤 엔진으로도 썸네일을 쓰지 못했다."""


def _point_at(obj, target, world_up) -> None:
    from mathutils import Matrix, Vector

    direction = (target - obj.location).normalized()
    up = world_up.normalized()
    right = direction.cross(up)
    if right.length < 1.0e-6:
        up = Vector((0.0, 1.0, 0.0))
        right = direction.cross(up)
    right.normalize()
    up = right.cross(direction).normalized()
    obj.rotation_euler = Matrix((right, up, -direction)).transposed().to_euler()


def _body_axes(armature):
    """어깨 두 점과 Hips에서 화면 오른쪽/정면/위 축을 유도한다."""
    by_suffix = {
        bone.name.rsplit(":", 1)[-1].lower(): bone
        for bone in armature.pose.bones
    }

    def head(suffix: str):
        try:
            bone = by_suffix[suffix.lower()]
        except KeyError as exc:
            raise ThumbnailRenderError(
                f"target rig lacks {suffix}; cannot derive anatomical camera"
            ) from exc
        return armature.matrix_world @ bone.head

    left = head("LeftShoulder")
    right_point = head("RightShoulder")
    pelvis = head("Hips")
    shoulder_center = (left + right_point) * 0.5
    # 이 부호가 라이브러리 ``front`` 썸네일의 정면 방향과 일치한다.
    right = left - right_point
    up = shoulder_center - pelvis
    if right.length <= 1.0e-5 or up.length <= 1.0e-5:
        raise ThumbnailRenderError("cannot derive anatomical front from shoulders/Hips")
    right.normalize()
    up = up - right * up.dot(right)
    if up.length <= 1.0e-5:
        raise ThumbnailRenderError("degenerate anatomical up axis")
    up.normalize()
    forward = right.cross(up)
    if forward.length <= 1.0e-5:
        raise ThumbnailRenderError("degenerate anatomical front axis")
    forward.normalize()
    up = forward.cross(right).normalized()
    return right, forward, up


def available_engines() -> set[str]:
    import bpy

    prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
    return {item.identifier for item in prop.enum_items}


def _configure_engine(scene, engine: str, samples: int) -> None:
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = False
    elif engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = samples


def _stage_scene(view: str, resolution: int) -> dict[str, Any]:
    """임포트된 캐릭터에 카메라·조명·재질을 얹고 렌더 설정을 준비한다."""
    import bpy
    from mathutils import Vector

    for obj in list(bpy.context.scene.objects):
        if obj.name.startswith(("QA_ANATOMICAL_", "QA_VIEW_")):
            bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ThumbnailRenderError(
            f"expected one target armature after import, got {len(armatures)}"
        )
    right, forward, up = _body_axes(armatures[0])
    points = [
        obj.matrix_world @ vertex.co
        for obj in meshes
        for vertex in obj.data.vertices
    ]
    if not points:
        raise ThumbnailRenderError("no mesh vertices after import")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    extent = max(*size, 1.0e-4)

    view_axis = {
        "front": forward,
        "three_quarter": (right + forward).normalized(),
        "side": right,
        "back": -forward,
    }[view]
    screen_right = (-view_axis).cross(up)
    if screen_right.length <= 1.0e-5:
        raise ThumbnailRenderError(f"degenerate screen-right axis for {view}")
    screen_right.normalize()

    bpy.ops.object.camera_add(location=center + view_axis * (extent * 3.0))
    camera = bpy.context.object
    camera.name = f"QA_ANATOMICAL_{view.upper()}"
    camera.data.type = "ORTHO"
    _point_at(camera, center, up)
    bpy.context.view_layer.update()
    world_to_camera = camera.matrix_world.inverted()
    projected = [world_to_camera @ point for point in points]
    projected_width = (
        max(point.x for point in projected) - min(point.x for point in projected)
    )
    projected_height = (
        max(point.y for point in projected) - min(point.y for point in projected)
    )
    camera.data.ortho_scale = max(
        projected_height * 1.14, projected_width * 1.14, 1.0
    )
    bpy.context.scene.camera = camera

    material = bpy.data.materials.get("QA_NEUTRAL_MATERIAL")
    if material is None:
        material = bpy.data.materials.new("QA_NEUTRAL_MATERIAL")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise ThumbnailRenderError("Principled BSDF node is unavailable")
    principled.inputs["Base Color"].default_value = (0.20, 0.22, 0.26, 1.0)
    principled.inputs["Roughness"].default_value = 0.78
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)

    radius = max(size.length, 1.0)
    bpy.ops.object.light_add(
        type="AREA",
        location=center + (screen_right * 0.8 + up * 1.0 + view_axis * 1.2) * radius,
    )
    key = bpy.context.object
    key.name = "QA_VIEW_KEY"
    key.data.energy = 500.0
    key.data.shape = "DISK"
    key.data.size = radius * 0.8
    _point_at(key, center, up)
    bpy.ops.object.light_add(
        type="AREA",
        location=center + (-screen_right * 0.9 + up * 0.35 + view_axis * 0.5) * radius,
    )
    fill = bpy.context.object
    fill.name = "QA_VIEW_FILL"
    fill.data.energy = 110.0
    fill.data.size = radius
    _point_at(fill, center, up)

    world = bpy.context.scene.world or bpy.data.worlds.new("QA_FRONT_WORLD")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise ThumbnailRenderError("world Background node is unavailable")
    background.inputs["Color"].default_value = (0.62, 0.62, 0.62, 1.0)
    background.inputs["Strength"].default_value = 0.55

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    return {
        "bbox_min": [float(value) for value in minimum],
        "bbox_max": [float(value) for value in maximum],
    }


def render_artifact_view(
    *,
    artifact_fbx: Path,
    output_png: Path,
    view: str,
    resolution: int,
    samples: int,
    engines: tuple[str, ...] = THUMBNAIL_ENGINES,
) -> dict[str, Any]:
    """빈 씬에 산출물 FBX를 다시 임포트해 ``view`` 카메라로 PNG를 쓴다.

    변환기가 씬에 남긴 상태를 믿지 않고 **내보낸 바이트 그대로**를 임포트한다 —
    작가가 받는 FBX와 preview가 같은 것을 보게 하기 위해서다(라이브러리 빌드도 같은
    방식이다). 엔진은 ``engines`` 순서대로 시도하고 첫 성공을 쓴다.
    """
    import bpy

    if view not in THUMBNAIL_VIEWS:
        raise ThumbnailRenderError(f"unsupported thumbnail view: {view}")
    if not engines:
        raise ThumbnailRenderError("no render engine requested")
    artifact = Path(artifact_fbx)
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise ThumbnailRenderError("artifact FBX is missing or empty")
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(artifact.resolve()))
    staged = _stage_scene(view, resolution)
    scene = bpy.context.scene
    scene.render.filepath = str(output.resolve())

    supported = available_engines()
    attempts: list[dict[str, str]] = []
    used_engine: str | None = None
    for engine in engines:
        if engine not in supported:
            attempts.append({"engine": engine, "error": "engine unavailable"})
            continue
        output.unlink(missing_ok=True)
        try:
            _configure_engine(scene, engine, samples)
            bpy.ops.render.render(write_still=True)
            if not output.is_file() or output.stat().st_size <= 0:
                raise ThumbnailRenderError("render finished without writing a PNG")
        except Exception as exc:  # noqa: BLE001 - 다음 엔진으로 넘어간다
            attempts.append({"engine": engine, "error": f"{type(exc).__name__}: {exc}"[:300]})
            output.unlink(missing_ok=True)
            continue
        used_engine = engine
        break
    if used_engine is None:
        raise ThumbnailRenderError(
            "thumbnail render failed for every engine: "
            + "; ".join(f"{item['engine']}={item['error']}" for item in attempts)
        )
    return {
        "view": view,
        "camera_convention": THUMBNAIL_CAMERA_CONVENTION.format(view=view),
        "resolution": resolution,
        "samples": samples,
        "engine": used_engine,
        "engine_attempts": attempts,
        **staged,
    }


__all__ = ["ThumbnailRenderError", "available_engines", "render_artifact_view"]
