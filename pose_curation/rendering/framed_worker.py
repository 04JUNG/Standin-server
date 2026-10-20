"""Render anatomical scopes with the production cropper in a Blender child."""
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
from converter import retarget
from converter.convert import convert
from converter.framing import crop_meshes
from pose_curation.rendering.fingers import apply_channels, rest_bases
from pose_curation.rendering.profiles import register
from pose_curation.rendering.scene import render_views
from pose_curation.storage import read_json, sha256, write_json


def run(job):
    if bpy.app.version[:3] != (5, 2, 0):
        raise ValueError("부분 미리보기에는 Blender 5.2.0이 필요합니다.")
    register()
    if sha256(Path(job['bvh'])) != job['bvh_sha256']:
        raise ValueError("생성 전에 원본 BVH가 변경되었습니다.")
    bases = rest_bases(job['character'])
    directory = Path(job['directory'])
    output = directory / 'character.fbx'
    report = convert(bvh_path=job['bvh'], character_fbx=job['character'],
                     out_path=str(output), src_profile=job['rig_profile'],
                     dst_profile='mixamo', embed_textures=False)
    if not report.ok:
        raise ValueError(f"캐릭터 변환 실패: {report.as_dict()}")
    retarget.reset_scene()
    arm, meshes = retarget.import_character(str(output))
    fingers = apply_channels(arm, job['bvh'], bases)
    # Bake the finger pose into the same rigged-rest representation as the body.
    retarget.apply_output_mode(arm, meshes, 'rigged_rest')
    crop = crop_meshes(arm, meshes, job['scope'])
    retarget.export_fbx(str(output), embed_textures=False)
    retarget.reset_scene()
    retarget.import_character(str(output))
    images = render_views(directory, 'preview', resolution=512)
    if sha256(Path(job['bvh'])) != job['bvh_sha256']:
        raise ValueError("생성 도중 원본 BVH가 변경되었습니다.")
    return {'ok': True, 'fingerprint': job['fingerprint'], 'crop': crop,
            'fingers': fingers, 'fbx_sha256': sha256(output),
            'thumbnails': {view: {'file': path.name, 'sha256': sha256(path)}
                           for view, path in images.items()}}


if __name__ == '__main__':
    job = read_json(Path(sys.argv[sys.argv.index('--') + 1]))
    try:
        result = run(job)
    except Exception as exc:
        traceback.print_exc()
        result = {'ok': False, 'error': str(exc)}
    write_json(Path(job['directory']) / 'result.json', result)
