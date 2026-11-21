#!/usr/bin/env python3
"""Rebuild one observed bad frozen raster offline; never uploads or deploys.

The supplied response headers bind the exact old JPEG/view/library. Blender
uses the approved character and BVH, then the production finger adapter. The
small immutable archive contains only manifest.json and one corrected JPEG.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bake(args):
    import bpy
    import numpy as np
    from converter import retarget as rt
    from converter.bone_map import PROFILES, resolve_profile
    from pose_curation.rendering.fingers import rest_bases, apply_channels
    from pose_curation.thumbnail_style.prepare_blender import surface
    bases = rest_bases(str(args.character))
    rt.reset_scene()
    source = rt.import_bvh(str(args.bvh))
    arm, meshes = rt.import_character(str(args.character))
    rt.retarget(source, arm)
    fingers = apply_channels(arm, str(args.bvh), bases)
    if fingers['applied_joints'] != 30:
        raise ValueError('this correction requires all 30 source finger joints')
    bpy.context.view_layer.update()
    mapping = PROFILES[resolve_profile(list(arm.data.bones.keys()))]
    pivot = arm.matrix_world @ arm.pose.bones[mapping['hips']].head
    np.savez(args.output / 'surface.npz', **surface(meshes, pivot))
    (args.output / 'fingers.json').write_text(json.dumps(fingers, sort_keys=True))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--bvh', type=Path, required=True)
    p.add_argument('--character', type=Path, required=True)
    p.add_argument('--headers', type=Path, required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blender', default='/Applications/Blender.app/Contents/MacOS/Blender')
    p.add_argument('--bake', action='store_true')
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bake:
        return bake(args)
    h = {k.lower(): v for k, v in json.loads(args.headers.read_text()).items()}
    from src.directional_posters import CHARACTER_SHA, VIEWS
    from src.thumbnail_patch import STAGING_LIBRARY_URI
    if (sha(args.bvh) != h['x-standin-source-bvh-sha256']
            or sha(args.character) != CHARACTER_SHA
            or h['x-standin-character-sha256'] != CHARACTER_SHA
            or h['x-standin-output-scope'] != 'full'
            or h['x-standin-preview-view'] not in VIEWS
            or h['x-standin-native-render-method'] != 'reviewed-surface-outline-v1'):
        raise ValueError('source or camera lineage mismatch')
    deps = ['converter/retarget.py', 'converter/bone_map.py', 'src/bvh.py',
            'pose_curation/hands/presets.py',
            'pose_curation/rendering/fingers.py', 'pose_curation/thumbnail_style/prepare_blender.py',
            'pose_curation/thumbnail_style/renderer.py', 'scripts/build_staging_poster_patch.py']
    recipe = {name: sha(ROOT / name) for name in deps}
    revision = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    command = [args.blender, '-b', '-t', '1', '--python-exit-code', '1', '--python',
               str(Path(__file__).resolve()), '--', '--bake', '--bvh', str(args.bvh.resolve()),
               '--character', str(args.character.resolve()), '--headers', str(args.headers.resolve()),
               '--pose', args.pose, '--output', str(args.output.resolve())]
    subprocess.run(command, check=True)
    import numpy as np
    from pose_curation.thumbnail_style.renderer import Renderer
    surface = np.load(args.output / 'surface.npz')
    renderer = Renderer()
    try:
        renderer.load([(surface['positions'], surface['normals'], surface['indices'])])
        image = renderer.render(int(h['x-standin-preview-view'][4:]))
        image.save(args.output / 'corrected.png')
    finally:
        renderer.ctx.release()
    pack = args.output / 'pack'; pack.mkdir(exist_ok=True)
    name = 'full__' + h['x-standin-preview-view'] + '.jpg'
    image.save(pack / name, quality=90)
    manifest = dict(schema_version=1, environment='staging', pose_id=args.pose,
        base_library_uri=STAGING_LIBRARY_URI,
        source_bvh_sha256=sha(args.bvh), character_sha256=CHARACTER_SHA,
        source_library_version=h['x-standin-asset-library-version'],
        base_asset_version=h['x-standin-poster-asset-version'],
        base_renderer_revision=h['x-standin-preview-revision'],
        renderer_revision=revision, recipe_sha256=revision,
        recipe=recipe, finger_joints_applied=30, surface_sha256=sha(args.output / 'surface.npz'),
        images={name: dict(old_sha256=h['x-standin-native-artifact-sha256'],
                           sha256=sha(pack / name), bytes=(pack / name).stat().st_size)})
    manifest['asset_version'] = 'staging-raster-patch-' + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    (pack / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    archive = args.output / 'patch.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(pack.iterdir()):
            stream.add(path, arcname=path.name, recursive=False)
    (args.output / 'archive.sha256').write_text(sha(archive) + '\n')
    print(json.dumps(dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
