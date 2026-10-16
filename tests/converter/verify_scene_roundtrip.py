"""Blender integration: source geometry -> independent placement -> serialized FBX.

Run after make_fixtures.py and test_convert.py, or pass real --source FBXs.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
from mathutils import Vector

from converter.scene import SceneInputError, _import, _snapshot, _verify, compose


def render_preview(path):
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for obj in meshes:
        color = (0.18, 0.43, 0.67, 1) if obj.name.startswith("person_0000_") else (0.8, 0.35, 0.18, 1)
        material = bpy.data.materials.new(obj.name + "_qa")
        material.diffuse_color = color
        obj.data.materials.clear()
        obj.data.materials.append(material)
    points = [o.matrix_world @ Vector(corner) for o in meshes for corner in o.bound_box]
    lo, hi = Vector([min(p[i] for p in points) for i in range(3)]), Vector([max(p[i] for p in points) for i in range(3)])
    center = (lo + hi) / 2
    bpy.ops.object.camera_add(location=center + Vector((0, -7, 2.3)))
    camera = bpy.context.object
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(hi.x-lo.x, (hi.z-lo.z)*1.5) * 1.35
    scene = bpy.context.scene
    scene.camera = camera
    scene.world = bpy.data.worlds.new("QA world")
    scene.world.color = (0.5, 0.5, 0.5)
    bpy.ops.object.light_add(type="AREA", location=center + Vector((-3, -4, 6)))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.render.resolution_x, scene.render.resolution_y = 960, 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = Path(os.getenv("CONVERTER_TEST_ARTIFACT_ROOT", "tests/converter"))
    sources = args.source or [root / "out/pose_rigged_rest.fbx", root / "assets/character_mixamo.fbx"]
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for source in sources:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _import(source)
        bpy.context.view_layer.update()
        snapshots.append(_snapshot())
    checks = []
    for name, indices in [("single", [0]), ("two_people", [0, 1]), ("three_repeated", [0, 0, 1])]:
        rows, expected = [], {}
        paths = [sources[i] for i in indices]
        for i, source in enumerate(indices):
            person_id = [0, 2, 7][i]
            position, yaw, scale = [(-1, 0, 0), (1.2, .3, -.5), (0, .8, 1.5)][i], [0, 35, -65][i], [1, .75, 1.2][i]
            rows.append(dict(person_index=person_id, position=list(position), rotation_y_degrees=yaw,
                             scale=scale, fbx_sha256=hashlib.sha256(paths[i].read_bytes()).hexdigest()))
            prefix = f"person_{person_id:04d}_"
            sine, cosine = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
            for key, item in snapshots[source].items():
                item = copy.deepcopy(item)
                # Independent formula for Three.js Y-up -> Blender Z-up placement.
                item["points"] = [(scale*(cosine*x-sine*y)+position[0],
                                   scale*(sine*x+cosine*y)-position[2],
                                   scale*z+position[1]) for x, y, z in item["points"]]
                if "rigs" in item:
                    item["rigs"] = [prefix + rig for rig in item["rigs"]]
                expected[prefix + key] = item
        manifest = dict(schema_version=1, cut_id=name, coordinate_system="threejs-y-up-m", persons=rows)
        report = compose(manifest, paths, args.out / f"{name}.fbx")
        actual = _snapshot()
        independent_error = _verify(expected, actual)
        assert len([v for v in actual.values() if v["type"] == "ARMATURE"]) == len(indices)
        checks.append(dict(case=name, independent_world_error_m=independent_error, **report))
        (args.out / f"{name}.json").write_text(json.dumps(manifest, indent=2))
        if args.preview and name == "two_people":
            render_preview(args.out / "two_people.png")
    rejected = []
    invalid_sources = [args.out / "two_people.fbx"]
    if not args.source:
        invalid_sources += [root / "out/pose_rigged_anim.fbx", root / "out/pose_static_mesh.fbx"]
    for source in invalid_sources:
        manifest = dict(schema_version=1, cut_id="invalid", coordinate_system="threejs-y-up-m",
                        persons=[dict(person_index=0, position=[0, 0, 0],
                                      fbx_sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
        try:
            compose(manifest, [source], args.out / "must_not_publish.fbx")
        except SceneInputError:
            rejected.append(source.name)
        else:
            raise AssertionError(f"unsupported source accepted: {source.name}")
        assert not (args.out / "must_not_publish.fbx").exists()
    (args.out / "verification.json").write_text(json.dumps({"ok": True, "cases": checks,
                                                           "rejected_sources": rejected}, indent=2))
    print("PASS: one, two and three-person FBX; independent XYZ/yaw/scale; skin and skeleton round trip")


if __name__ == "__main__":
    main()
