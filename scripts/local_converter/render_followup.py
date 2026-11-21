"""Matched actual-material views of baseline, pre-surface and final FBXs.

No diagnostic material replacement. These are Blender renders, not CSP photos.
Camera, lights and framing are fixed across each comparison row.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_csp_controls import load


def points(path):
    _, meshes = load(path)
    return np.array([obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices])


def render(path, destination, center, scale, back):
    load(path)
    center = Vector(center)
    offset = Vector((.65, 2.8 if back else -2.8, .3)) * scale
    data = bpy.data.cameras.new("QA matched camera")
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + offset
    camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
    data.type, data.ortho_scale = "ORTHO", scale * 1.14
    scene = bpy.context.scene
    scene.camera = camera
    direction = -1 if back else 1
    for name, delta, energy in [("Key", (-1.5, -2*direction, 2.5), 180),
                                ("Fill", (2, -direction, 1), 80), ("Rim", (0, 2*direction, 2), 130)]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size = energy*scale**2, "DISK", scale*1.5
        lamp = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(lamp)
        lamp.location = center + Vector(delta)*scale
        lamp.rotation_euler = (center-lamp.location).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("QA background")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.24, .27, .31, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .5
    scene.world = world
    scene.render.engine, scene.cycles.device = "CYCLES", "CPU"
    scene.cycles.samples, scene.cycles.use_denoising = 16, True
    scene.render.resolution_x, scene.render.resolution_y = 520, 680
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(destination)
    bpy.ops.render.render(write_still=True)


def main(directory, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    run = json.loads((directory / "report.json").read_text())
    for mi, model in enumerate(run["models"]):
        for pi, pose in enumerate(model["poses"]):
            files = {"pre-surface": directory / f"stage-{mi}-{pi}-before-surface.fbx",
                     "final": directory / pose["file"]}
            old = baseline / f"model-{mi}-pose-{pi}.fbx"
            if mi < 2 and old.is_file():
                files = {"baseline-v9": old, **files}
            array = np.concatenate([points(path) for path in files.values()])
            center = (array.min(0)+array.max(0))/2
            scale = float(np.ptp(array, axis=0).max())
            for back in (False, True):
                images = []
                for label, path in files.items():
                    destination = output / f"{mi}-{pi}-{'back' if back else 'front'}-{label}.png"
                    render(path, destination, center, scale, back)
                    images.append({"label": label, "file": destination.name})
                rows.append({"model": mi, "pose": pi, "backView": back, "images": images})
                (output / "renders.json").write_text(json.dumps({"renderer": "Blender Cycles CPU",
                    "cspScreenshot": False, "rows": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    main(args.run_dir.resolve(), args.baseline.resolve(), args.output.resolve())
