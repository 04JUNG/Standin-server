"""Disposable actual-model FBX ablations for a separate CSP import owner.

These are diagnostic controls, never registration inputs or qualified exports.
No CSP acceptance or crash cause follows from a successful Blender reimport.
Run inside pinned Blender with --runtime, --source and a new --output folder.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False, use_image_search=False,
                            ignore_leaf_bones=False, automatic_bone_orientation=False)
    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    meshes = sorted((o for o in bpy.context.scene.objects if o.type == "MESH"), key=lambda o: o.name)
    return arm, meshes


def snapshot(arm, meshes):
    return {
        "points": {o.name: np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in meshes},
        "polygons": {o.name: [tuple(p.vertices) for p in o.data.polygons] for o in meshes},
        "weights": {o.name: [sorted((o.vertex_groups[g.group].name, g.weight) for g in v.groups)
                              for v in o.data.vertices] for o in meshes},
        "normals": {o.name: np.array([n.vector[:] for n in o.data.corner_normals]) for o in meshes},
        "bones": {} if arm is None else {b.name: b.parent.name if b.parent else None for b in arm.data.bones},
        "materials": sorted({s.material.name for o in meshes for s in o.material_slots if s.material}),
    }


def minimal_rig(arm, meshes):
    from mathutils import Matrix
    # Bake no new deformation: the input is already a posed-rest FBX.
    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        obj.modifiers.clear()
        obj.vertex_groups.clear()
    bpy.data.objects.remove(arm, do_unlink=True)
    data = bpy.data.armatures.new("QA minimal rig")
    arm = bpy.data.objects.new("QA minimal rig", data)
    bpy.context.collection.objects.link(arm)
    arm.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = data.edit_bones.new("QA root")
    root.head, root.tail = (0, 0, 0), (0, 0, .1)
    bpy.ops.object.mode_set(mode="OBJECT")
    for obj in meshes:
        group = obj.vertex_groups.new(name="QA root")
        group.add(list(range(len(obj.data.vertices))), 1., "REPLACE")
        obj.modifiers.new("QA identity skin", "ARMATURE").object = arm
    return arm


def minimal_material(meshes):
    mat = bpy.data.materials.new("QA single solid")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.65, .65, .65, 1)
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for face in obj.data.polygons:
            face.material_index = 0


def make(runtime, source, output):
    sys.path.insert(0, str(runtime))
    from converter.local.camera import apply_camera
    from converter.local.contract import versions
    output.mkdir(parents=True, exist_ok=False)
    angle = math.radians(35)
    rotation = [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]]
    variants = [("source-orientation", False, False, False), ("camera-yaw35", True, False, False),
                ("minimal-rig", False, True, False), ("single-material", False, False, True),
                ("minimal-rig-single-material", False, True, True)]
    report = {"schemaVersion": 1, "sourceSha256": sha(source), "versions": versions(),
              "runtimeManifestSha256": sha(runtime / "runtime-manifest.json"),
              "cspImport": "unverified; client owner must record acceptance/crash separately",
              "cameraCausality": "not established", "diagnosticOnly": True, "variants": []}
    for name, camera, simple_rig, simple_material in variants:
        arm, meshes = load(source)
        original = snapshot(arm, meshes)
        camera_result = None
        if camera:
            camera_result = apply_camera(arm, meshes, {"version": "personal-camera/1",
                "sourceBvhSha256": "0" * 64, "rotation": rotation})
        if simple_rig:
            arm = minimal_rig(arm, meshes)
        if simple_material:
            minimal_material(meshes)
        bpy.context.view_layer.update()
        before = snapshot(arm, meshes)
        path = output / (name + ".fbx")
        if name == "source-orientation":
            shutil.copyfile(source, path)
        else:
            # The production file already contains its export leaf bones. A
            # second automatic leaf generation would confound the camera test
            # with a hierarchy change. Keep every existing leaf on import.
            bpy.ops.export_scene.fbx(filepath=str(path), use_selection=False,
                apply_unit_scale=True, global_scale=1., axis_forward="-Z", axis_up="Y",
                object_types={"ARMATURE", "MESH"}, use_mesh_modifiers=True,
                add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
                bake_anim=False, path_mode="AUTO", embed_textures=False, mesh_smooth_type="FACE")
        after = snapshot(*load(path))
        position_error = max(float(np.linalg.norm(after["points"][key] - value, axis=1).max())
                             for key, value in before["points"].items())
        normal_error = max(float(np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", value, after["normals"][key]), -1, 1))).max())
            for key, value in before["normals"].items())
        assert position_error < 2e-5
        assert before["polygons"] == after["polygons"]
        assert before["weights"] == after["weights"]
        assert all(after["bones"].get(name) == parent for name, parent in before["bones"].items())
        added_bones = sorted(after["bones"].keys() - before["bones"].keys())
        assert all(name.endswith("_end") for name in added_bones)
        assert before["materials"] == after["materials"]
        control_position_delta = max(float(np.linalg.norm(before["points"][key] - value, axis=1).max())
                                     for key, value in original["points"].items())
        if not camera:
            assert control_position_delta < 2e-5
        report["variants"].append({"file": path.name, "sha256": sha(path),
            "camera": {"rotation": rotation, "validation": camera_result} if camera else None,
            "rig": "one identity root; original weights intentionally replaced" if simple_rig else "original",
            "materials": "one QA solid; original assignments intentionally replaced" if simple_material else "original",
            "maxControlPositionDeltaM": control_position_delta,
            "reimport": {"maxPositionErrorM": position_error, "maxLocalCornerNormalErrorDeg": normal_error,
                "exactPolygonsWeightsOriginalHierarchy": True, "addedExportLeafBones": added_bones,
                "boneCountIncludingExportLeaves": len(after["bones"]),
                "materials": after["materials"]}})
        (output / "controls.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    make(args.runtime.resolve(), args.source.resolve(), args.output.resolve())
