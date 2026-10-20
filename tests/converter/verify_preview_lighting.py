"""Blender smoke: a front-facing normal must receive light, not turn into a dark rim.

blender -b --factory-startup --python-exit-code 1 --python \
    tests/converter/verify_preview_lighting.py -- <output.png>
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import bpy
    from mathutils import Vector
    from converter.preview_style import configure_surface
    from converter.thumbnail_render import _configure_engine

    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1)
    sphere = bpy.context.object
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    bpy.ops.object.camera_add(location=(0, -3, 0))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type, camera.data.ortho_scale = 'ORTHO', 2.4
    scene = bpy.context.scene
    scene.camera = camera
    bpy.context.view_layer.update()
    configure_surface(scene, [sphere])
    _configure_engine(scene, 'CYCLES', 8)
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(output), check_existing=False)
    def luminance(x, y):
        offset = (y * 128 + x) * 4
        return sum(image.pixels[offset:offset + 3]) / 3
    center, background = luminance(64, 64), luminance(0, 0)
    assert center > background + 0.1, (center, background)
    assert center < 0.95, 'preview highlights are clipped'
    print(f'PREVIEW_LIGHTING_OK center={center:.4f} background={background:.4f}', flush=True)


if __name__ == '__main__':
    main()
