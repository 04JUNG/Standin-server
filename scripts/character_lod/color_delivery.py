"""Restore visible skin to local FBX/authoring deliveries from a verified backup.

Geometry is never re-exported to add a skin: use the typed material/UV writer,
then compare imported geometry/rig/weights and preserve every original object.
No production registry, original artifact or certification is changed.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np
from mathutils import Vector

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from converter.outline_worker import tree
from converter.outline_fbx import one, write, fingerprint
from converter.skin_fbx import augment, UV_NAME
from converter.skin_worker import verify_import
from converter.preview_style import configure_workbench

RGB = (250 / 255, 223 / 255, 202 / 255)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def linear(x):
    return x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4


def render_loaded(dest):
    """Render the imported file's own material; never inject a repair at render time."""
    scene = bpy.context.scene
    configure_workbench(scene)
    shade = scene.display.shading
    shade.color_type = 'TEXTURE'
    shade.light = 'STUDIO'
    shade.show_cavity = True
    shade.show_shadows = True
    shade.show_specular_highlight = False
    target = Vector((0, 0, .95))
    camera = bpy.data.objects.new('ColorCheckCamera', bpy.data.cameras.new('ColorCheckCamera'))
    scene.collection.objects.link(camera)
    camera.location = target + Vector((0, -5, .15))
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.35
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(dest)
    bpy.ops.render.render(write_still=True)


def color_fbx(source, target, asset, uv, png_path, skin, picture=None):
    root, version = tree(source)
    original = {e.props[0]: fingerprint(e) for e in one(root, b'Objects').elems}
    geometry = next(e for e in one(root, b'Objects').elems if e.id == b'Geometry' and e.props[2] == b'Mesh')
    original_geometry = fingerprint(geometry)
    old_layers = len(geometry.elems)
    png = png_path.read_bytes()
    augment(root, asset, uv, png, skin)
    # A missing/disabled texture must still fall back to skin color, not white.
    material = next(e for e in one(root, b'Objects').elems if e.id == b'Material' and e.props[0] not in original)
    props = one(material, b'Properties70')
    next(e for e in props.elems if e.props[0] == b'DiffuseColor').props[-3:] = RGB
    for e in one(root, b'Objects').elems:
        if e.props[0] in original and e != geometry:
            assert fingerprint(e) == original[e.props[0]], 'non-geometry source object changed'
    restored = copy.deepcopy(geometry)
    restored.elems[:] = [e for e in restored.elems if e.id != b'LayerElementUV']
    layer = one(restored, b'Layer')
    layer.elems[:] = [e for e in layer.elems if not (e.id == b'LayerElement' and one(e, b'Type').props == [b'LayerElementUV'])]
    assert fingerprint(restored) == original_geometry, 'body data changed beyond UV'
    assert len(geometry.elems) == old_layers + 1
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target, root, version)
    expected = [fingerprint(e) for e in root.elems if e.id not in {b'FBXHeaderExtension', b'FileId', b'CreationTime', b'Creator'}]
    actual, actual_version = tree(target)
    assert actual_version == version
    assert expected == [fingerprint(e) for e in actual.elems if e.id not in {b'FBXHeaderExtension', b'FileId', b'CreationTime', b'Creator'}]
    video = next(e for e in one(actual, b'Objects').elems if e.id == b'Video' and e.props[0] not in original)
    filename = one(video, b'Filename').props[0].decode()
    sidecar = target.parent / filename
    if sidecar.exists():
        assert sidecar.read_bytes() == png
    else:
        sidecar.write_bytes(png)
    errors, final = verify_import(source, target, asset, uv, png)
    mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    mat = mesh.material_slots[0].material
    image = next(n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE')
    result = dict(file=str(target), source_sha256=sha(source), sha256=sha(target),
                  vertices=len(mesh.data.vertices), bones=asset['bones'], skin=skin,
                  geometry_and_rig_typed_data_exact=True, uv_verified=True,
                  image_sha256=hashlib.sha256(png).hexdigest(), embedded_image=True,
                  relative_texture=filename, nonwhite_fallback_rgb=RGB,
                  imported_texture_loaded=bool(image.size[0] and image.size[1]),
                  import_errors=errors, final_validation=final, csp_certified=False)
    if picture:
        render_loaded(picture)
        result['render'] = str(picture)
    return result


def authoring_geometry_digest():
    """Hash editable geometry, shape keys, weights, rig, drivers and control graph."""
    h = hashlib.sha256()
    for obj in sorted(bpy.context.scene.objects, key=lambda o: o.name):
        h.update(obj.name.encode())
        h.update(np.array(obj.matrix_world, dtype='<f8').tobytes())
        h.update(json.dumps({k: str(v) for k, v in obj.items()}, sort_keys=True).encode())
        if obj.type == 'MESH':
            a = np.empty((len(obj.data.vertices), 3), dtype='<f4')
            obj.data.vertices.foreach_get('co', a.ravel())
            h.update(a.tobytes())
            h.update(json.dumps([list(p.vertices) for p in obj.data.polygons]).encode())
            h.update(json.dumps([[(g.group, g.weight) for g in v.groups] for v in obj.data.vertices]).encode())
            if obj.data.shape_keys:
                for key in obj.data.shape_keys.key_blocks:
                    key.data.foreach_get('co', a.ravel())
                    h.update(key.name.encode() + a.tobytes())
                    h.update(str((key.value, key.mute, key.slider_min, key.slider_max)).encode())
        elif obj.type == 'ARMATURE':
            for bone in obj.data.bones:
                h.update(bone.name.encode() + str(bone.parent.name if bone.parent else None).encode())
                h.update(np.array(bone.matrix_local, dtype='<f8').tobytes())
                h.update(np.array(obj.pose.bones[bone.name].matrix_basis, dtype='<f8').tobytes())
        for data in (obj, obj.data.shape_keys if obj.type == 'MESH' else None):
            if data is not None and data.animation_data:
                for curve in data.animation_data.drivers:
                    h.update(str((curve.data_path, curve.array_index, curve.driver.expression)).encode())
                    for var in curve.driver.variables:
                        h.update(str((var.name, var.type, [(getattr(t.id, 'name', None), t.data_path, t.bone_target, t.transform_type, t.transform_space) for t in var.targets])).encode())
    return h.hexdigest()


def color_authoring(backup, dest, png_path):
    source = backup / 'authoring'
    dest.mkdir(parents=True, exist_ok=False)
    for p in source.iterdir():
        if p.is_file() and p.name not in ('model.blend', 'manifest.json'):
            shutil.copy2(p, dest / p.name)
    bpy.ops.wm.open_mainfile(filepath=str(source / 'model.blend'))
    before = authoring_geometry_digest()
    main = next(o for o in bpy.context.scene.objects if o.get('standin_studio_mesh'))
    uv = np.load(source / 'skin-uv.npy')
    assert uv.shape == (len(main.data.polygons), 3, 2)
    assert not main.data.uv_layers
    layer = main.data.uv_layers.new(name=UV_NAME.decode())
    layer.data.foreach_set('uv', uv.astype(np.float32).ravel())
    image = bpy.data.images.load(str(png_path), check_existing=False)
    image.colorspace_settings.name = 'sRGB'
    image.pack()
    color = tuple(linear(x) for x in RGB) + (1.,)
    mat = bpy.data.materials.new('Standin_Packed_Flesh_20261116')
    mat.diffuse_color = color
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Roughness'].default_value = .85
    shader.inputs['Metallic'].default_value = 0.
    tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = image
    mat.node_tree.links.new(tex.outputs['Color'], shader.inputs['Base Color'])
    main.data.materials.clear()
    main.data.materials.append(mat)
    for p in main.data.polygons:
        p.material_index = 0
    preview_mat = bpy.data.materials.new('Standin_Preview_Flesh_20261116')
    preview_mat.diffuse_color = color
    preview_mat.use_nodes = True
    ps = next(n for n in preview_mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    ps.inputs['Base Color'].default_value = color
    ps.inputs['Roughness'].default_value = .85
    for obj in bpy.context.scene.objects:
        if obj.get('standin_preview_lod'):
            obj.data.materials.clear()
            obj.data.materials.append(preview_mat)
            for p in obj.data.polygons:
                p.material_index = 0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.color_type = 'MATERIAL'
    assert authoring_geometry_digest() == before
    target = dest / 'model.blend'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(target), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(target))
    assert authoring_geometry_digest() == before, 'editable geometry or control graph changed'
    main = next(o for o in bpy.context.scene.objects if o.get('standin_studio_mesh'))
    image = next(n.image for n in main.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE')
    assert image.packed_file and hashlib.sha256(bytes(image.packed_file.data)).hexdigest() == sha(png_path)
    record = dict(file=str(target), sha256=sha(target), source_blend_sha256=sha(source / 'model.blend'),
                  geometry_controls_rig_drivers_sha256=before, editable_geometry_exact=True,
                  vertices=len(main.data.vertices), packed_image_sha256=sha(png_path),
                  source_geometry_manifest=str(source / 'manifest.json'),
                  controller_sha256=sha(dest / 'controller.py'))
    save_json(dest / 'color-manifest.json', record)
    return record


def main(root):
    root = Path(root)
    backup_manifest = json.loads((root / 'backup-manifest.json').read_text())
    assert backup_manifest['before_edit'] and backup_manifest['all_copies_sha256_verified']
    for row in backup_manifest['files']:
        assert sha(root / row['backup']) == row['sha256'], 'backup changed'
    out = root / 'models'
    out.mkdir(exist_ok=False)
    qa = root / 'qa'
    qa.mkdir(exist_ok=False)
    report = dict(schema='local-colored-character-delivery.v1', reason='geometry-only default deliveries had no skin texture/UV',
                  backup_sha256=sha(root / 'backup-manifest.json'), fbx=[], authoring={}, production_activated=False)
    for sex, spec in backup_manifest['models'].items():
        backup = root / 'backup' / sex
        assets = json.loads((backup / 'skin-assets.json').read_text())['models']
        for job in spec['jobs']:
            cid, case = job['character'], job['case']
            asset = assets[cid]
            uv = json.loads(gzip.decompress((backup / 'skins' / asset['uv']['path']).read_bytes()))
            source = backup / 'fbx' / (cid + '.fbx')
            source_tree, _ = tree(source)
            geometry = next(e for e in one(source_tree, b'Objects').elems if e.id == b'Geometry' and e.props[2] == b'Mesh')
            assert not any(e.id == b'LayerElementUV' for e in geometry.elems)
            for skin in ('flesh', 'flesh-guide'):
                suffix = '' if skin == 'flesh' else '-guide'
                target = out / (sex + suffix + '.fbx') if case == 'base' else out / 'variants' / (sex + '-' + case + suffix + '.fbx')
                png = backup / 'skins' / asset['textures'][skin]['path']
                picture = qa / (sex + suffix + '.png') if case == 'base' else None
                record = color_fbx(source, target, asset, uv, png, skin, picture)
                record.update(gender=sex, case=case)
                report['fbx'].append(record)
                save_json(root / 'color-report.json', report)
                print('COLOR_FBX_PASS', sex, case, skin, record['vertices'], flush=True)
        base = next(j for j in spec['jobs'] if j['case'] == 'base')
        png = backup / 'skins' / assets[base['character']]['textures']['flesh']['path']
        report['authoring'][sex] = color_authoring(backup, root / 'authoring' / sex, png)
        save_json(root / 'color-report.json', report)
        print('COLOR_AUTHORING_PASS', sex, flush=True)
    assert len(report['fbx']) == 12 and len(report['authoring']) == 2
    print('COLOR_DELIVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args.root)
