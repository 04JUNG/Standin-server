"""Render measured surfaces and original published FBXs with one studio setup."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build_shape_release import load_controller
from scripts.body_default_proportions.build import measure, sha, write


ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--release', type=Path, required=True)
ap.add_argument('--old-package', type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = args.release / 'images'
out.mkdir(exist_ok=True)
data = dict(np.load(args.release / 'review-surfaces.npz'))
registry = json.loads((args.release / 'shape-bakes.local.json').read_text())
previous = []

for sex, ver in [('male', '0-7-2'), ('female', '0-3-4')]:
    cid = f'{sex}-base-v{ver}'
    r = registry['characters'][cid]
    s = registry['sources'][r['rules']]
    c = load_controller(s['blend'], s['controller'], 'old_' + sex)
    c.set_controls(r['defaults'], correct=False)
    data[sex + '_current_editor'] = c.compute(r['defaults'])[0].astype('<f4')
    data[sex + '_current_editor_faces'] = c.source()['faces']
    measurement = measure(c, r['defaults'], data[sex + '_current_editor'])
    previous.append({'characterId': cid, 'sex': sex, 'source': s,
                     'fbxSha256': r['asset_sha256'], 'controls': r['defaults'],
                     'measurement': measurement, 'rigSha256': c.digest(c.source()['rest']),
                     'assetLock': json.loads(c.objects()[0]['asset_lock'])})
    c.unregister()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # This path comes from the local immutable old release, not a guessed URL.
    for label, fbx in [('current_editor', cid + '.fbx'), ('legacy_fallback', sex + '-base.fbx')]:
        if label == 'current_editor' and sha(args.old_package / 'fbx' / fbx) != r['asset_sha256']:
            raise ValueError('published existing FBX hash mismatch')
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(args.old_package / 'fbx' / fbx), use_anim=False)
        m = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
        q = np.array([m.matrix_world @ v.co for v in m.data.vertices], dtype='<f4')
        faces = np.array([list(p.vertices) for p in m.data.polygons], dtype='<i4')
        if label == 'current_editor':
            expected = data[sex + '_current_editor']
            if q.shape != expected.shape or np.linalg.norm(q - expected, axis=1).max() > 1e-5:
                raise ValueError('published existing default differs from authoring source')
        data[sex + '_' + label] = q
        data[sex + '_' + label + '_faces'] = faces

write(args.release / 'previous-defaults.json', previous)

np.savez_compressed(args.release / 'review-surfaces.npz', **data)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
sh = scene.display.shading
sh.light = 'STUDIO'
sh.studiolight_rotate_z = .3
sh.color_type = 'SINGLE'
sh.single_color = (.66, .69, .73)
sh.show_shadows = True
sh.show_cavity = True
sh.cavity_type = 'BOTH'
sh.curvature_ridge_factor = 1.1
sh.curvature_valley_factor = 1.
sh.show_specular_highlight = True
sh.background_type = 'WORLD'
scene.world = bpy.data.worlds.new('comparison-world')
scene.world.color = (.065, .075, .095)
scene.view_settings.view_transform = 'Standard'
scene.render.resolution_x = 900
scene.render.resolution_y = 1050
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
cam = bpy.data.objects.new('fixed-orthographic-camera', bpy.data.cameras.new('camera'))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 2.5
views = [('front', 0), ('three_quarter', 45), ('side', 90), ('back', 180)]
records = []
for sex in ('male', 'female'):
    for label in ('current_editor', 'latest_base', 'candidate', 'legacy_fallback'):
        q = data[sex + '_' + label]
        faces = data.get(sex + '_' + label + '_faces', data[sex + '_faces'])
        me = bpy.data.meshes.new('measured-surface')
        me.from_pydata(q.tolist(), [], faces.tolist())
        me.update()
        obj = bpy.data.objects.new('exact-surface', me)
        scene.collection.objects.link(obj)
        for p in me.polygons:
            p.use_smooth = True
        for view, angle in views:
            rad = np.deg2rad(angle)
            target = Vector((0, 0, .89))
            cam.location = target + Vector((4 * np.sin(rad), -4 * np.cos(rad), 0))
            cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = str(out / f'{sex}-{label}-{view}.png')
            bpy.ops.render.render(write_still=True)
            records.append({'sex': sex, 'state': label, 'view': view, 'camera': list(cam.location),
                            'target': list(target), 'orthoScale': cam.data.ortho_scale,
                            'heightMm': float(np.ptp(q[:, 2]) * 1000),
                            'meshScaleApplied': 1, 'file': scene.render.filepath})
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(me)
(args.release / 'render-protocol.json').write_text(json.dumps({'pose': 'neutral T',
    'sameCameraLightMaterial': True, 'floor': 'original world Z=0; no per-state translation or scale',
    'heightNormalizedAuxiliary': False, 'renderer': 'Blender Workbench studio, orthographic', 'renders': records}, indent=2))
