"""Versioned mesh-only output framing after the frozen retarget solver.

Skin weights define anatomical regions, so raising a knee or lowering a hand
does not change which body part survives. Every bone and bind transform stays
intact. Full output is a byte-preserving no-op. Blender imports are child-only.
"""
from __future__ import annotations

from pathlib import Path

FRAMING_VERSION = "skin-regions-v1"
OUTPUT_SCOPES = frozenset({"full", "half", "bust", "head"})
PREVIEW_VIEWS = frozenset({"front", "three_quarter", "side", "back"})


def validate_scope(value: str) -> str:
    if not isinstance(value, str) or value not in OUTPUT_SCOPES:
        raise ValueError("output_scope must be full, half, bust or head")
    return value


def retained_roles(scope: str) -> set[str]:
    validate_scope(scope)
    if scope == "head":
        return {"head"}
    if scope == "bust":
        return {"spine2", "neck", "head", "shoulder.L", "shoulder.R"}
    if scope == "half":
        return {"spine", "spine1", "spine2", "neck", "head"} | {
            f"{part}.{side}" for part in ("shoulder", "upperarm", "forearm", "hand")
            for side in ("L", "R")
        }
    from converter.bone_map import CANONICAL_BONES
    return set(CANONICAL_BONES)


def crop_meshes(arm, meshes, scope: str) -> dict:
    """Clip interpolated skin-region membership at 0.5 and seal only new cuts.

    BMesh interpolates position/UV/deform layers at the cut. A temporary scalar
    coordinate drives clipping; the saved posed coordinates are restored before
    capping, so no global-height plane can remove a raised leg or retained hand.
    """
    import bpy
    import bmesh
    from converter.bone_map import PROFILES, resolve_profile

    validate_scope(scope)
    before = sum(len(m.data.vertices) for m in meshes)
    bones = {b.name: (b.parent.name if b.parent else None, b.matrix_local.copy())
             for b in arm.data.bones}
    if scope == "full":
        return {"version": FRAMING_VERSION, "scope": scope, "vertices_before": before,
                "vertices_after": before, "cap_faces": 0, "bones_preserved": len(bones)}
    profile = resolve_profile(list(bones))
    mapping = PROFILES[profile]
    if any(mapping.get(role) not in bones for role in ("hips", "spine", "spine2", "neck", "head")):
        raise ValueError("character lacks the anatomical anchors for framing")
    inverse = {value: key for key, value in mapping.items()}
    roles = retained_roles(scope)

    def keep_bone(name):
        bone = arm.data.bones.get(name)
        while bone is not None:
            if bone.name in inverse:
                return inverse[bone.name] in roles
            bone = bone.parent
        return False

    caps = 0
    kept_meshes = []
    for mesh in meshes:
        if mesh.data.shape_keys:
            raise ValueError("shape-key character framing is not supported")
        groups = {group.index: keep_bone(group.name) for group in mesh.vertex_groups}
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh.data)
            deform = bm.verts.layers.deform.active
            if deform is None:
                raise ValueError("framing requires skin weights on every mesh")
            position = bm.verts.layers.float_vector.new("_framing_position")
            scalar = bm.verts.layers.float.new("_framing_membership")
            for vertex in bm.verts:
                weights = vertex[deform]
                total = sum(weights.values())
                if total <= 1e-8:
                    raise ValueError("framing cannot classify unweighted vertices")
                membership = sum(weight for group, weight in weights.items()
                                 if groups.get(group, False)) / total - .5
                vertex[position] = vertex.co.copy()
                vertex[scalar] = membership
                vertex.co.z = membership
            bmesh.ops.bisect_plane(
                bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                plane_co=(0, 0, 0), plane_no=(0, 0, 1), dist=1e-7,
                clear_inner=True, clear_outer=False,
            )
            for vertex in bm.verts:
                vertex.co = vertex[position]
            cut_edges = [edge for edge in bm.edges if edge.is_boundary
                         and all(abs(v[scalar]) < 1e-5 for v in edge.verts)]
            if cut_edges:
                cap = bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)["faces"]
                caps += len(cap)
                for face in cap:
                    face.smooth = False
                bmesh.ops.triangulate(bm, faces=cap)
                if any(edge.is_boundary for edge in cut_edges if edge.is_valid):
                    raise ValueError("framing left an open cut boundary")
            bm.verts.layers.float_vector.remove(position)
            bm.verts.layers.float.remove(scalar)
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            bm.to_mesh(mesh.data)
            mesh.data.update()
        finally:
            bm.free()
        if mesh.data.polygons:
            kept_meshes.append(mesh)
        else:
            bpy.data.objects.remove(mesh, do_unlink=True)
    after = sum(len(mesh.data.vertices) for mesh in kept_meshes)
    if after <= 0 or after >= before:
        raise ValueError("framing must retain a nonempty strict subset of the character")
    for name, (parent, matrix) in bones.items():
        bone = arm.data.bones[name]
        if (bone.parent.name if bone.parent else None) != parent or bone.matrix_local != matrix:
            raise ValueError("framing changed the skeleton")
    return {"version": FRAMING_VERSION, "scope": scope, "vertices_before": before,
            "vertices_after": after, "cap_faces": caps, "bones_preserved": len(bones)}


def render_preview(meshes, output: Path, view="front", *, aligned=False) -> None:
    """Frame the visible cropped mesh, not the retained full skeleton."""
    import math
    import bpy
    from mathutils import Vector

    if view not in PREVIEW_VIEWS:
        raise ValueError("unsupported preview view")
    points = [mesh.matrix_world @ v.co for mesh in meshes for v in mesh.data.vertices]
    if not points:
        raise ValueError("empty preview mesh")
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    center = (low + high) * .5
    radius = max((p-center).length for p in points)
    angle = {"front": 0, "three_quarter": 45, "side": 90, "back": 180}[view]
    yaw = math.radians(angle)
    direction = Vector((math.sin(yaw), -math.cos(yaw), 0 if aligned else .07))
    camera_data = bpy.data.cameras.new("scope-preview")
    camera = bpy.data.objects.new("scope-preview", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + direction * max(radius*5, 1)
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = max(radius*2.25, .01)
    camera_data.clip_end = max(radius*20, 100)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'SINGLE'
    scene.display.shading.single_color = (.72, .74, .76)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = 'WORLD'
    if aligned:
        scene.display.render_aa = '8'
    if scene.world is None:
        scene.world = bpy.data.worlds.new("scope-preview-world")
    scene.world.color = (.14, .15, .17)
    scene.render.resolution_x = scene.render.resolution_y = 256 if aligned else 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def process_output(path, scope="full", *, preview_path=None, view="front", camera_rotation=None) -> dict:
    from converter import retarget as rt
    from converter.camera import validate_rotation, rotate_scene

    validate_scope(scope)
    validate_rotation(camera_rotation)
    if camera_rotation is not None and view != "front":
        raise ValueError("aligned output requires a fixed front camera")
    if scope == "full" and preview_path is None and camera_rotation is None:
        return {"version": FRAMING_VERSION, "scope": scope}
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    report = crop_meshes(arm, meshes, scope)
    if camera_rotation is not None:
        rotate_scene(arm, camera_rotation)
    if scope != "full" or camera_rotation is not None:
        rt.export_fbx(str(path), embed_textures=False)
    if preview_path:
        # Reimport the actual delivered FBX; preview/export cannot silently diverge.
        rt.reset_scene()
        _, meshes = rt.import_character(str(path))
        render_preview(meshes, Path(preview_path), view, aligned=camera_rotation is not None)
    return report
